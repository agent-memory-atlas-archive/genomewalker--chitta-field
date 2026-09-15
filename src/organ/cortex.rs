use super::pq::{ProductQuantizer, PQ_BYTES};
use super::prototype::{ProtoId, PrototypeIndex};
use crate::error::{FieldError, Result};
use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{BufReader, BufWriter, Read as IoRead, Write};
use std::path::Path;

/// V1 cortex snapshot: PostingEntry without affect_q field.
const CORTEX_SNAPSHOT_MAGIC_V1: u64 = 0xC417745F3A7_0001;
/// Current cortex snapshot (V4): EMBED_DIM=768, HALF_DIM=384 (nomic-embed-text v1.5).
const CORTEX_SNAPSHOT_MAGIC: u64 = 0xC417745F3A7_0004;

// ── Sparse Code ──────────────────────────────────────────────────────────────

/// Sparse representation of a memory: K=64 active features out of N=16,384.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseCode {
    pub feature_ids: Vec<u32>, // exactly K entries, sorted ascending
    pub activations: Vec<f32>, // parallel to feature_ids, normalized
}

impl SparseCode {
    pub fn is_empty(&self) -> bool {
        self.feature_ids.is_empty()
    }
}

// ── Sparse Encoder (product-key) ─────────────────────────────────────────────

pub use crate::ops::EMBED_DIM;
pub const HALF_DIM: usize = EMBED_DIM / 2;
pub const N_LEFT: usize = 128; // centroids for left half
pub const N_RIGHT: usize = 128; // centroids for right half
pub const N_ATOMS: usize = N_LEFT * N_RIGHT; // 16,384 total
pub const K_ACTIVE: usize = 64; // active features per memory
pub const ENCODER_LR: f32 = 5e-4;
pub const SHORTLIST_PER_HALF: usize = 16; // top-16 from each half → 256 candidates

/// Online product-key sparse encoder.
/// Dictionary: N_LEFT × N_RIGHT = 16,384 atoms, each EMBED_DIM-dimensional.
/// Represented as two half-dictionaries for O(√N) top-K selection.
#[derive(Serialize, Deserialize)]
pub struct SparseEncoder {
    left_atoms: Vec<Vec<f32>>,  // N_LEFT × HALF_DIM
    right_atoms: Vec<Vec<f32>>, // N_RIGHT × HALF_DIM
}

impl Default for SparseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseEncoder {
    /// Initialize with random unit vectors from LCG seeded by index.
    pub fn new() -> Self {
        let mut left_atoms = Vec::with_capacity(N_LEFT);
        let mut right_atoms = Vec::with_capacity(N_RIGHT);
        for i in 0..N_LEFT {
            left_atoms.push(random_unit_vec(HALF_DIM, i as u64));
        }
        for i in 0..N_RIGHT {
            right_atoms.push(random_unit_vec(HALF_DIM, (N_LEFT + i) as u64));
        }
        Self {
            left_atoms,
            right_atoms,
        }
    }

    /// Encode a 768-dim embedding into a sparse code of K=64 active features.
    /// Product-key top-K: O(128×384 + 128×384 + 256) instead of O(16384×768).
    pub fn encode(&self, embedding: &[f32]) -> SparseCode {
        if embedding.len() != EMBED_DIM {
            return SparseCode::default();
        }
        let (e_left, e_right) = embedding.split_at(HALF_DIM);

        // Score each half-dictionary
        let mut left_scores: Vec<(f32, usize)> = self
            .left_atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (dot(e_left, a), i))
            .collect();
        let mut right_scores: Vec<(f32, usize)> = self
            .right_atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (dot(e_right, a), i))
            .collect();

        // Partial sort: top SHORTLIST_PER_HALF from each half
        left_scores.select_nth_unstable_by(SHORTLIST_PER_HALF, |a, b| {
            b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
        });
        right_scores.select_nth_unstable_by(SHORTLIST_PER_HALF, |a, b| {
            b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
        });

        // Build shortlist of SHORTLIST_PER_HALF² = 256 candidates
        let mut candidates: Vec<(f32, u32)> =
            Vec::with_capacity(SHORTLIST_PER_HALF * SHORTLIST_PER_HALF);
        for &(ls, li) in &left_scores[..SHORTLIST_PER_HALF] {
            for &(rs, ri) in &right_scores[..SHORTLIST_PER_HALF] {
                let atom_id = (li * N_RIGHT + ri) as u32;
                candidates.push((ls + rs, atom_id));
            }
        }

        // Top-K from candidates
        let k = K_ACTIVE.min(candidates.len());
        candidates.select_nth_unstable_by(k, |a, b| {
            b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
        });
        candidates.truncate(k);
        candidates.sort_by(|a, b| a.1.cmp(&b.1)); // sort by feature_id ascending

        // Normalize activations (relu + L1 norm)
        let sum: f32 = candidates.iter().map(|(s, _)| s.max(0.0)).sum();
        if sum < 1e-9 {
            return SparseCode::default();
        }

        let feature_ids: Vec<u32> = candidates.iter().map(|(_, id)| *id).collect();
        let activations: Vec<f32> = candidates.iter().map(|(s, _)| s.max(0.0) / sum).collect();

        SparseCode {
            feature_ids,
            activations,
        }
    }

    /// Reconstruct a dense 768-dim embedding from a sparse code.
    /// atom_id = left_idx * N_RIGHT + right_idx
    /// reconstructed[0..HALF_DIM] += activation * left_atoms[left_idx]
    /// reconstructed[HALF_DIM..EMBED_DIM] += activation * right_atoms[right_idx]
    pub fn decode(&self, code: &SparseCode) -> Vec<f32> {
        let mut out = vec![0.0f32; EMBED_DIM];
        for (&feat_id, &act) in code.feature_ids.iter().zip(code.activations.iter()) {
            let left_idx = feat_id as usize / N_RIGHT;
            let right_idx = feat_id as usize % N_RIGHT;
            for (o, &a) in out[..HALF_DIM]
                .iter_mut()
                .zip(self.left_atoms[left_idx].iter())
            {
                *o += act * a;
            }
            for (o, &a) in out[HALF_DIM..]
                .iter_mut()
                .zip(self.right_atoms[right_idx].iter())
            {
                *o += act * a;
            }
        }
        out
    }

    /// Compute reconstruction error: ||original - decode(encode(original))||².
    /// Normalized to [0, 1] range. Used as surprise signal for FEP plasticity.
    pub fn reconstruction_error(&self, embedding: &[f32], code: &SparseCode) -> f32 {
        if code.is_empty() || embedding.len() != EMBED_DIM {
            return 1.0; // maximally surprising if we can't encode
        }
        let reconstructed = self.decode(code);
        let mut sse = 0.0f32;
        let mut norm = 0.0f32;
        for (i, (&orig, &recon)) in embedding.iter().zip(reconstructed.iter()).enumerate() {
            let _ = i;
            let diff = orig - recon;
            sse += diff * diff;
            norm += orig * orig;
        }
        if norm < 1e-9 { return 0.0; }
        (sse / norm).min(1.0)
    }

    /// Online FEP-derived update: accuracy (Hebbian) + complexity penalty (orthogonalization).
    /// The complexity term pushes underused atom weights toward zero,
    /// while the Gram-Schmidt step decorrelates active atoms. FEP §4.2.
    pub fn update(&mut self, embedding: &[f32], code: &SparseCode) {
        if code.is_empty() {
            return;
        }
        let (e_left, e_right) = embedding.split_at(HALF_DIM);

        // Complexity penalty weight (λ in F = accuracy - λ·complexity)
        const COMPLEXITY_LAMBDA: f32 = 1e-4;

        for (&fid, &act) in code.feature_ids.iter().zip(code.activations.iter()) {
            let li = (fid as usize) / N_RIGHT;
            let ri = (fid as usize) % N_RIGHT;

            // Left atom: prediction error + complexity penalty
            let la = &mut self.left_atoms[li];
            for (w, &x) in la.iter_mut().zip(e_left) {
                let pred_error = x - act * *w;
                let complexity = -COMPLEXITY_LAMBDA * *w;
                *w += ENCODER_LR * act * (pred_error + complexity);
            }
            normalize(la);

            // Right atom: prediction error + complexity penalty
            let ra = &mut self.right_atoms[ri];
            for (w, &x) in ra.iter_mut().zip(e_right) {
                let pred_error = x - act * *w;
                let complexity = -COMPLEXITY_LAMBDA * *w;
                *w += ENCODER_LR * act * (pred_error + complexity);
            }
            normalize(ra);
        }

        // Gram-Schmidt orthogonalization pass on active atom pairs.
        // Decorrelates representations to maximize mutual information. FEP §5.1.
        self.orthogonalize_active(code);
    }

    /// Partial Gram-Schmidt: for each pair of active atoms in the same half,
    /// subtract the projection of one onto the other, then renormalize.
    fn orthogonalize_active(&mut self, code: &SparseCode) {
        const ORTHO_RATE: f32 = 0.01; // gentle: 1% of projection removed per step

        // Collect unique left and right indices from active features
        let mut left_indices: Vec<usize> = Vec::new();
        let mut right_indices: Vec<usize> = Vec::new();
        for &fid in &code.feature_ids {
            let li = fid as usize / N_RIGHT;
            let ri = fid as usize % N_RIGHT;
            if !left_indices.contains(&li) { left_indices.push(li); }
            if !right_indices.contains(&ri) { right_indices.push(ri); }
        }

        // Orthogonalize left atoms
        for i in 0..left_indices.len() {
            for j in (i + 1)..left_indices.len() {
                let (li, lj) = (left_indices[i], left_indices[j]);
                let proj = dot(&self.left_atoms[li], &self.left_atoms[lj]);
                if proj.abs() > 0.01 {
                    for d in 0..HALF_DIM {
                        self.left_atoms[lj][d] -= ORTHO_RATE * proj * self.left_atoms[li][d];
                    }
                    normalize(&mut self.left_atoms[lj]);
                }
            }
        }

        // Orthogonalize right atoms
        for i in 0..right_indices.len() {
            for j in (i + 1)..right_indices.len() {
                let (ri, rj) = (right_indices[i], right_indices[j]);
                let proj = dot(&self.right_atoms[ri], &self.right_atoms[rj]);
                if proj.abs() > 0.01 {
                    for d in 0..HALF_DIM {
                        self.right_atoms[rj][d] -= ORTHO_RATE * proj * self.right_atoms[ri][d];
                    }
                    normalize(&mut self.right_atoms[rj]);
                }
            }
        }
    }
}

// ── Compile-time guard: PostingEntry size must match cortex snapshot magic ────
// If you add/remove fields to PostingEntry, this assert will fail.
// You MUST: (1) bump CORTEX_SNAPSHOT_MAGIC, (2) add a LegacyPostingEntryVN,
// (3) add migration in CorticalIndex::load_snapshot(), (4) update this constant.
const _POSTING_ENTRY_SIZE_V2: usize = 16;
const _: () = assert!(
    std::mem::size_of::<PostingEntry>() == _POSTING_ENTRY_SIZE_V2,
);

// ── Cortical Posting Index ────────────────────────────────────────────────────

/// V1 PostingEntry: no affect_q field.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyPostingEntryV1 {
    pub mem_id: MemoryId,
    pub activation_q: u8,
    pub strength_q: u8,
    pub proto_id: ProtoId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingEntry {
    pub mem_id: MemoryId,
    pub activation_q: u8,  // activation quantized to u8 (0.0-1.0 → 0-255)
    pub strength_q: u8,    // strength quantized to u8
    pub proto_id: ProtoId, // prototype cluster assignment
    /// Affect arousal quantized: 0=calm, 255=intense. High arousal boosts retrieval
    /// (flashbulb memory effect, inspired by Anthropic emotion vectors 2026).
    pub affect_q: u8,
}

/// V1 CorticalIndex: uses LegacyPostingEntryV1 (no affect_q).
#[derive(Serialize, Deserialize)]
struct LegacyCorticalIndexV1 {
    postings: HashMap<u32, Vec<LegacyPostingEntryV1>>,
    df: HashMap<u32, u64>,
    pub n_memories: u64,
    pub(crate) mem_codes: HashMap<MemoryId, SparseCode>,
    mem_ts: HashMap<MemoryId, i64>,
    mem_kind: HashMap<MemoryId, String>,
    prototype_idx: PrototypeIndex,
    pub pq: Option<ProductQuantizer>,
    pub mem_pq: HashMap<MemoryId, [u8; PQ_BYTES]>,
}

/// Inverted posting index over sparse codes. O(K) posting lookups per query.
#[derive(Serialize, Deserialize)]
pub struct CorticalIndex {
    /// feature_id → posting list (sorted by mem_id for merge efficiency)
    postings: HashMap<u32, Vec<PostingEntry>>,
    /// document frequency: how many memories have each feature active
    df: HashMap<u32, u64>,
    /// total indexed memories
    pub n_memories: u64,
    /// mem_id → its sparse code (for removal)
    pub(crate) mem_codes: HashMap<MemoryId, SparseCode>,
    /// mem_id → ts_ms (for recency scoring)
    mem_ts: HashMap<MemoryId, i64>,
    /// mem_id → kind (for recency weighting: only "episode" kind gets decay)
    mem_kind: HashMap<MemoryId, String>,
    /// ART-like online prototype clustering
    prototype_idx: PrototypeIndex,
    /// Trained product quantizer for residual compression (optional until trained)
    pub pq: Option<ProductQuantizer>,
    /// mem_id → PQ codes for its residual
    pub mem_pq: HashMap<MemoryId, [u8; PQ_BYTES]>,
}

impl Default for CorticalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CorticalIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            df: HashMap::new(),
            n_memories: 0,
            mem_codes: HashMap::new(),
            mem_ts: HashMap::new(),
            mem_kind: HashMap::new(),
            prototype_idx: PrototypeIndex::new(),
            pq: None,
            mem_pq: HashMap::new(),
        }
    }

    pub fn index(
        &mut self,
        mem_id: MemoryId,
        code: &SparseCode,
        strength: f32,
        ts_ms: i64,
        kind: &str,
    ) {
        self.index_with_affect(mem_id, code, strength, ts_ms, kind, 0.0);
    }

    pub fn index_with_affect(
        &mut self,
        mem_id: MemoryId,
        code: &SparseCode,
        strength: f32,
        ts_ms: i64,
        kind: &str,
        affect_arousal: f32,
    ) {
        if code.is_empty() {
            return;
        }
        // Remove old code if re-indexing
        if self.mem_codes.contains_key(&mem_id) {
            self.remove(mem_id);
        }

        let strength_q = (strength.clamp(0.0, 1.0) * 255.0) as u8;
        let affect_q = (affect_arousal.clamp(0.0, 1.0) * 255.0) as u8;
        let proto_id = self.prototype_idx.assign(mem_id, code);

        for (&fid, &act) in code.feature_ids.iter().zip(code.activations.iter()) {
            let entry = PostingEntry {
                mem_id,
                activation_q: (act * 255.0) as u8,
                strength_q,
                proto_id,
                affect_q,
            };
            self.postings.entry(fid).or_default().push(entry);
            *self.df.entry(fid).or_insert(0) += 1;
        }

        self.mem_codes.insert(mem_id, code.clone());
        self.mem_ts.insert(mem_id, ts_ms);
        self.mem_kind.insert(mem_id, kind.to_string());
        self.n_memories += 1;
    }

    pub fn remove(&mut self, mem_id: MemoryId) {
        if let Some(code) = self.mem_codes.remove(&mem_id) {
            for fid in &code.feature_ids {
                if let Some(list) = self.postings.get_mut(fid) {
                    list.retain(|e| e.mem_id != mem_id);
                    if let Some(df) = self.df.get_mut(fid) {
                        *df = df.saturating_sub(1);
                    }
                }
            }
            self.n_memories = self.n_memories.saturating_sub(1);
        }
        self.mem_ts.remove(&mem_id);
        self.mem_kind.remove(&mem_id);
        self.prototype_idx.remove_memory(mem_id);
    }

    /// Update strength for a memory (called on reconsolidation).
    pub fn update_strength(&mut self, mem_id: MemoryId, strength: f32) {
        if let Some(code) = self.mem_codes.get(&mem_id) {
            let sq = (strength.clamp(0.0, 1.0) * 255.0) as u8;
            for fid in &code.feature_ids.clone() {
                if let Some(list) = self.postings.get_mut(fid) {
                    for e in list.iter_mut() {
                        if e.mem_id == mem_id {
                            e.strength_q = sq;
                        }
                    }
                }
            }
        }
    }

    /// Search: IDF-weighted sparse overlap scoring with prototype bonus.
    /// Returns up to k results as (mem_id, score), sorted by score descending.
    pub fn search(
        &self,
        query_code: &SparseCode,
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
    ) -> Vec<(MemoryId, f32)> {
        if query_code.is_empty() || self.n_memories == 0 {
            return Vec::new();
        }

        let n = self.n_memories as f32;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Find query's nearest prototype
        let query_proto: Option<ProtoId> = self.prototype_idx.nearest_proto(query_code);

        // Accumulate scores per candidate memory
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        // Also track proto_id per candidate for proto_bonus computation
        let mut candidate_protos: HashMap<MemoryId, ProtoId> = HashMap::new();
        let mut query_norm = 0.0f32;

        for (&fid, &q_act) in query_code
            .feature_ids
            .iter()
            .zip(query_code.activations.iter())
        {
            let df = self.df.get(&fid).copied().unwrap_or(1) as f32;
            let idf = ((n + 1.0) / (df + 1.0)).ln() + 1.0;
            query_norm += idf * q_act;

            let Some(list) = self.postings.get(&fid) else { continue };

            for entry in list {
                if let Some(allowed_set) = allowed {
                    if !allowed_set.contains(&entry.mem_id) {
                        continue;
                    }
                }
                let c_act = entry.activation_q as f32 / 255.0;
                let term = idf * q_act.min(c_act);
                *scores.entry(entry.mem_id).or_insert(0.0) += term;
                candidate_protos
                    .entry(entry.mem_id)
                    .or_insert(entry.proto_id);
            }
        }

        if query_norm < 1e-9 {
            query_norm = 1.0;
        }

        // Finalize scores with strength + recency + proto_bonus
        let mut results: Vec<(MemoryId, f32)> = scores
            .into_iter()
            .map(|(mem_id, sparse_raw)| {
                let sparse = sparse_raw / query_norm;

                // Get strength from a posting entry
                let strength = self
                    .mem_codes
                    .get(&mem_id)
                    .and_then(|code| code.feature_ids.first())
                    .and_then(|&fid| self.postings.get(&fid))
                    .and_then(|list| list.iter().find(|e| e.mem_id == mem_id))
                    .map(|e| e.strength_q as f32 / 255.0)
                    .unwrap_or(0.5);

                // Recency: only for episodic memories
                let is_episodic = self
                    .mem_kind
                    .get(&mem_id)
                    .map(|k| k == "episode" || k == "observation")
                    .unwrap_or(false);
                let recency = if is_episodic {
                    let age_days = (now_ms - self.mem_ts.get(&mem_id).copied().unwrap_or(0)).max(0)
                        as f32
                        / (86400.0 * 1000.0);
                    (-age_days / 30.0).exp()
                } else {
                    0.0
                };

                // Proto bonus
                let proto_bonus = match (query_proto, candidate_protos.get(&mem_id).copied()) {
                    (Some(qp), Some(cp)) => {
                        if qp == cp {
                            1.0f32
                        } else {
                            // transitions scale 0→0.5 for non-same-proto candidates
                            let t = self.prototype_idx.transition(qp, cp);
                            (t / 0.5).min(1.0) * 0.5
                        }
                    }
                    _ => 0.0,
                };

                // Affect arousal boost: high-arousal memories get flashbulb retrieval bonus
                // (Anthropic emotion vectors 2026: arousal correlates with behavioral salience)
                let affect_boost = self
                    .mem_codes
                    .get(&mem_id)
                    .and_then(|code| code.feature_ids.first())
                    .and_then(|&fid| self.postings.get(&fid))
                    .and_then(|list| list.iter().find(|e| e.mem_id == mem_id))
                    .map(|e| e.affect_q as f32 / 255.0)
                    .unwrap_or(0.0);

                let score = 0.65 * sparse + 0.13 * proto_bonus + 0.10 * strength + 0.05 * recency + 0.07 * affect_boost;
                (mem_id, score)
            })
            .collect();

        // Top-K
        results.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        results.truncate(k);
        results
    }

    /// Attractor-based pattern completion: iteratively settle a query code
    /// by blending with prototype centroids and following asymmetric transitions.
    /// Partial cues converge to stored attractor basins. FEP §3.1.
    pub fn attractor_settle(&self, query_code: &SparseCode, max_steps: usize) -> SparseCode {
        if query_code.is_empty() {
            return query_code.clone();
        }

        let mut current = query_code.clone();
        let blend_rate = 0.3f32; // how much to pull toward prototype centroid

        for _step in 0..max_steps {
            let Some((proto_id, sim)) = self.prototype_idx.nearest_proto_with_sim(&current) else {
                break;
            };

            // Already well-matched — settled
            if sim > 0.95 {
                break;
            }

            // Blend current code with prototype centroid (pattern completion)
            if let Some(centroid) = self.prototype_idx.get_centroid(proto_id) {
                current = blend_sparse_codes(&current, centroid, blend_rate);
            }

            // Follow strongest outgoing transition (asymmetric flow)
            let transitions = self.prototype_idx.top_transitions(proto_id, 3);
            if let Some(&(next_proto, tw)) = transitions.first() {
                if tw > 0.1 {
                    if let Some(next_centroid) = self.prototype_idx.get_centroid(next_proto) {
                        // Lightly blend toward the transition target
                        current = blend_sparse_codes(&current, next_centroid, blend_rate * tw * 0.5);
                    }
                }
            }
        }
        current
    }

    /// Search with attractor settling: settle the query first, then search.
    pub fn search_attractor(
        &self,
        query_code: &SparseCode,
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
        settle_steps: usize,
    ) -> Vec<(MemoryId, f32)> {
        let settled = self.attractor_settle(query_code, settle_steps);
        self.search(&settled, k, allowed)
    }

    pub fn len(&self) -> usize {
        self.n_memories as usize
    }
    pub fn is_empty(&self) -> bool {
        self.n_memories == 0
    }

    pub fn prototype_count(&self) -> usize {
        self.prototype_idx.count()
    }

    /// Backfill affect_q for all indexed memories using a kind-based proxy.
    /// Called once after loading a snapshot built with index() (affect_q=0).
    pub fn backfill_affect_from_kind(&mut self) {
        let kind_arousal = |k: &str| -> u8 {
            let a: f32 = if k.to_ascii_lowercase().contains("correction") {
                0.85
            } else {
                match k {
                    "wisdom" => 0.70,
                    "insight" => 0.50,
                    "signal" => 0.30,
                    "episode" | "observation" => 0.15,
                    _ => 0.25,
                }
            };
            (a * 255.0) as u8
        };

        let updates: Vec<(MemoryId, u8)> = self
            .mem_kind
            .iter()
            .map(|(&mid, k)| (mid, kind_arousal(k)))
            .collect();

        for (mid, aq) in updates {
            if let Some(code) = self.mem_codes.get(&mid) {
                for &fid in &code.feature_ids {
                    if let Some(list) = self.postings.get_mut(&fid) {
                        if let Some(entry) = list.iter_mut().find(|e| e.mem_id == mid) {
                            if entry.affect_q == 0 {
                                entry.affect_q = aq;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn set_pq(&mut self, pq: ProductQuantizer) {
        self.pq = Some(pq);
    }

    pub fn index_pq(&mut self, memory_id: MemoryId, codes: [u8; PQ_BYTES]) {
        self.mem_pq.insert(memory_id, codes);
    }

    pub fn get_pq(&self, memory_id: MemoryId) -> Option<&[u8; PQ_BYTES]> {
        self.mem_pq.get(&memory_id)
    }

    pub fn pq_count(&self) -> usize {
        self.mem_pq.len()
    }

    pub fn is_pq_trained(&self) -> bool {
        self.pq.is_some()
    }

    /// Adapt vigilance threshold based on aggregate prediction error.
    /// High error → lower vigilance (create more prototypes).
    /// Low error → higher vigilance (fewer, coarser prototypes). FEP §4.1.
    pub fn adapt_vigilance(&mut self, avg_reconstruction_error: f32) {
        // Map error [0,1] to vigilance [0.001, 0.01]
        // High error → low vigilance (more prototypes needed)
        let v = 0.01 - avg_reconstruction_error * 0.009;
        self.prototype_idx.set_vigilance(v);
    }

    /// Strengthen prototype transitions for a set of co-retrieved memory IDs.
    /// Called after recall reconsolidation.
    pub fn strengthen_proto_transitions(&mut self, ids: &[MemoryId]) {
        // Collect proto_ids for all provided memory IDs
        let proto_ids: Vec<(usize, ProtoId)> = ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| self.prototype_idx.get_proto(id).map(|pid| (i, pid)))
            .collect();

        for i in 0..proto_ids.len() {
            for j in (i + 1)..proto_ids.len() {
                let (_, pa) = proto_ids[i];
                let (_, pb) = proto_ids[j];
                self.prototype_idx.strengthen_transition(pa, pb, 0.02);
            }
        }
    }

    /// Save the full cortical index to a binary snapshot file.
    /// Writes atomically: first to a `.tmp` file, then renames into place.
    /// The `snapshot_seqno` marks the last log op included in this snapshot.
    pub fn save_snapshot(&self, path: &Path, snapshot_seqno: u64) -> Result<()> {
        let tmp_path = path.with_extension("tmp");

        {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            let mut writer = BufWriter::new(file);

            writer.write_all(&CORTEX_SNAPSHOT_MAGIC.to_be_bytes())?;
            writer.write_all(&snapshot_seqno.to_be_bytes())?;

            let encoded =
                bincode::serialize(self).map_err(|e| FieldError::Serialization(e.to_string()))?;
            writer.write_all(&encoded)?;
            writer.flush()?;
        }

        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Read only the magic and snapshot_seqno without deserializing the full cortical index.
    pub fn peek_snapshot_seqno(path: &Path) -> Result<u64> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buf = [0u8; 16];
        reader.read_exact(&mut buf).map_err(FieldError::Io)?;
        let magic = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        if magic != CORTEX_SNAPSHOT_MAGIC && magic != CORTEX_SNAPSHOT_MAGIC_V1 {
            return Err(FieldError::Manifest(format!(
                "cortex snapshot magic mismatch: expected {:#x} or {:#x}, got {:#x}",
                CORTEX_SNAPSHOT_MAGIC, CORTEX_SNAPSHOT_MAGIC_V1, magic
            )));
        }
        Ok(u64::from_be_bytes(buf[8..16].try_into().unwrap()))
    }

    /// Load a cortical snapshot from disk.
    /// Returns `(CorticalIndex, snapshot_seqno)`.
    pub fn load_snapshot(path: &Path) -> Result<(Self, u64)> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic_buf = [0u8; 8];
        reader
            .read_exact(&mut magic_buf)
            .map_err(|e| FieldError::Io(e))?;
        let magic = u64::from_be_bytes(magic_buf);
        if magic != CORTEX_SNAPSHOT_MAGIC && magic != CORTEX_SNAPSHOT_MAGIC_V1 {
            return Err(FieldError::Manifest(format!(
                "cortex snapshot magic mismatch: expected {:#x} or {:#x}, got {:#x}",
                CORTEX_SNAPSHOT_MAGIC, CORTEX_SNAPSHOT_MAGIC_V1, magic
            )));
        }

        let mut seqno_buf = [0u8; 8];
        reader
            .read_exact(&mut seqno_buf)
            .map_err(|e| FieldError::Io(e))?;
        let snapshot_seqno = u64::from_be_bytes(seqno_buf);

        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| FieldError::Io(e))?;

        if magic == CORTEX_SNAPSHOT_MAGIC {
            let index: CorticalIndex = bincode::deserialize(&data)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            return Ok((index, snapshot_seqno));
        }

        // V1 migration: PostingEntry without affect_q
        eprintln!("[chitta-field] migrating cortex v1 snapshot → v2 (adding affect_q)");
        let v1: LegacyCorticalIndexV1 = bincode::deserialize(&data)
            .map_err(|e| FieldError::Serialization(e.to_string()))?;
        let postings = v1.postings.into_iter().map(|(fid, entries)| {
            let upgraded: Vec<PostingEntry> = entries.into_iter().map(|e| PostingEntry {
                mem_id: e.mem_id,
                activation_q: e.activation_q,
                strength_q: e.strength_q,
                proto_id: e.proto_id,
                affect_q: 0,
            }).collect();
            (fid, upgraded)
        }).collect();
        let index = CorticalIndex {
            postings,
            df: v1.df,
            n_memories: v1.n_memories,
            mem_codes: v1.mem_codes,
            mem_ts: v1.mem_ts,
            mem_kind: v1.mem_kind,
            prototype_idx: v1.prototype_idx,
            pq: v1.pq,
            mem_pq: v1.mem_pq,
        };
        Ok((index, snapshot_seqno))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Blend two sparse codes: result = (1-alpha)*a + alpha*b, keeping top-K features.
fn blend_sparse_codes(a: &SparseCode, b: &SparseCode, alpha: f32) -> SparseCode {
    let mut merged: HashMap<u32, f32> = HashMap::new();
    for (&fid, &act) in a.feature_ids.iter().zip(a.activations.iter()) {
        *merged.entry(fid).or_insert(0.0) += (1.0 - alpha) * act;
    }
    for (&fid, &act) in b.feature_ids.iter().zip(b.activations.iter()) {
        *merged.entry(fid).or_insert(0.0) += alpha * act;
    }
    let mut features: Vec<(u32, f32)> = merged.into_iter().filter(|(_, a)| *a > 0.0).collect();
    if features.len() > K_ACTIVE {
        features.select_nth_unstable_by(K_ACTIVE, |a, b| {
            b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
        });
        features.truncate(K_ACTIVE);
    }
    features.sort_by_key(|(fid, _)| *fid);
    let sum: f32 = features.iter().map(|(_, a)| *a).sum();
    if sum < 1e-9 { return SparseCode::default(); }
    SparseCode {
        feature_ids: features.iter().map(|(fid, _)| *fid).collect(),
        activations: features.iter().map(|(_, a)| a / sum).collect(),
    }
}

/// Generate a random unit vector using a simple LCG seeded by index.
fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Box-Muller for normal distribution
            let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;
            let r = (-2.0 * (u1 + 1e-10).ln()).sqrt();
            r * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    normalize(&mut v);
    v
}
