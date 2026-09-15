use crate::error::{FieldError, Result};
use crate::field::{AssocEdge, ChittaField};
use crate::ids::{compute_chunk_hash, ArtifactId, ChunkHash, MemoryId};
use crate::learner::route::{Route, RouteLearner};
use crate::ops::{EMBED_DIM, EMBED_MODEL_ID};
use crate::ops::{
    AddAssocEdgeOp, AddSymCallEdgeOp, AddTripletOp, ArtifactRef, DeleteMemoryOp, DemoteMemoryOp,
    EdgeType, InvalidateTripletOp, Op, PutPayloadOp, RemoveSymbolOp, StateDeltaOp, TrainPQOp,
    UpdateResidualPQOp, UpdateSparseCodeOp, UpsertArtifactOp, UpsertCodeFileOp, UpsertSymbolOp,
};
use crate::organ::memory_kind::{edge_legal, MemoryKind};
use crate::organ::provenance::WitnessKind;
use crate::organ::pq::ProductQuantizer;
use crate::organ::query_router::{DispatchKind, QueryRouter, RecallRequest};
use crate::organ::reconciler::Reconciler;
use crate::organ::symbol::SymbolEntry;
use crate::organ::triplet::TripletEntry;
use crate::payload::MemoryPayload;
use crate::recall::{RecallHit, SpreadingRecallHit};
use crate::scoring::{RecallMode, ScoringContext};
use crate::state::MemoryState;
use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

const RESERVOIR_SIZE: usize = 500;

// ── Utility posteriors ───────────────────────────────────────────────────────
// Access is not value: `strength`/`access_count` say a memory was retrieved,
// not that retrieving it helped. `MemoryState::{utility_alpha, utility_beta}`
// hold a Beta posterior fed by `record_outcome`, and recall can Thompson-sample
// it as a score multiplier. Gated OFF by default — with the flag unset every
// candidate's multiplier is exactly 1.0, so ranking is unchanged.

/// Total pseudo-observations (α+β) below which θ is pinned to the neutral 0.5.
/// Beta(1,1) is uniform, so sampling an untested memory injects pure noise into
/// the ranking; 5 = three real observations past the (1,1) prior.
const UTILITY_MIN_OBSERVATIONS: f32 = 5.0;

pub(crate) struct UtilityRecallConfig {
    enabled: bool,
    weight: f32,
    seed: Option<u64>,
}

/// Read once per process — this sits inside the recall scoring loop.
fn utility_recall_config() -> &'static UtilityRecallConfig {
    static CFG: std::sync::OnceLock<UtilityRecallConfig> = std::sync::OnceLock::new();
    CFG.get_or_init(|| UtilityRecallConfig {
        enabled: std::env::var("CHITTA_UTILITY_RECALL").ok().as_deref() == Some("1"),
        weight: std::env::var("CHITTA_UTILITY_WEIGHT")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|w| w.is_finite() && (0.0..=1.0).contains(w))
            .unwrap_or(0.3),
        seed: std::env::var("CHITTA_UTILITY_SEED").ok().and_then(|s| s.parse().ok()),
    })
}

/// splitmix64 — deterministic from its seed, so tests assert exact draws.
pub(crate) struct UtilityRng(u64);

impl UtilityRng {
    pub(crate) fn new(seed: u64) -> Self {
        UtilityRng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1) — open at both ends so `ln()` is always finite.
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal, Box-Muller (one of the pair; the second is discarded).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Gamma(shape, 1) by Marsaglia-Tsang. Valid for shape >= 1, which is all
    /// this needs: posterior counts never drop below the (1, 1) prior.
    fn next_gamma(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.next_normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

/// Thompson draw from Beta(alpha, beta) as the ratio of two Gammas, or the
/// neutral 0.5 when the posterior is too thin to be evidence.
pub(crate) fn thompson_theta(alpha: f32, beta: f32, rng: &mut UtilityRng) -> f32 {
    if alpha + beta < UTILITY_MIN_OBSERVATIONS {
        return 0.5;
    }
    let x = rng.next_gamma(alpha as f64);
    let y = rng.next_gamma(beta as f64);
    (x / (x + y)) as f32
}

/// One RNG per recall call. `CHITTA_UTILITY_SEED` pins it so an eval run is
/// reproducible; otherwise it is seeded from the recall's own clock read plus a
/// per-call counter, so two recalls in the same millisecond still differ.
fn utility_rng(now_ms: i64) -> UtilityRng {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    match utility_recall_config().seed {
        Some(s) => UtilityRng::new(s),
        None => {
            let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            UtilityRng::new((now_ms as u64) ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        }
    }
}

/// Score multiplier `(1 - w) + w·θ`. Every under-observed memory lands on the
/// same neutral `1 - w/2`, so enabling the flag never reorders memories that
/// have no outcome evidence — it only moves proven and disproven ones apart.
#[inline]
fn utility_multiplier_with(weight: f32, state: &MemoryState, rng: &mut UtilityRng) -> f32 {
    (1.0 - weight) + weight * thompson_theta(state.utility_alpha, state.utility_beta, rng)
}

/// Exactly 1.0 when the flag is off, so `score * utility_multiplier(..)` is
/// bit-identical to the pre-change score by default.
#[inline]
fn utility_multiplier(state: &MemoryState, rng: &mut UtilityRng) -> f32 {
    let cfg = utility_recall_config();
    if !cfg.enabled {
        return 1.0;
    }
    utility_multiplier_with(cfg.weight, state, rng)
}

/// In-flight state for the deferred-batched-insert backfill (stage → plan → apply).
/// Held in `ChittaField::backfill_plan_stage`. `staged_ids` = every memory durably
/// staged (embed_pending cleared at apply); `plan_ids` ⊆ staged_ids = those needing a
/// global-HNSW plan; `plan` is filled off the write lock by `backfill_plan`.
pub struct BackfillStage {
    pub(crate) staged_ids: Vec<MemoryId>,
    pub(crate) plan_ids:   Vec<MemoryId>,
    pub(crate) plan:       Option<crate::hnsw::DeltaBatchPlan>,
}

pub(crate) struct GroupStats {
    sum:            Vec<f64>,
    sum_sq:         Vec<f64>,
    count:          u64,
    reservoir:      Vec<Vec<f32>>,
    reservoir_seen: u64,
}

impl GroupStats {
    fn new() -> Self {
        Self {
            sum:            vec![0.0f64; EMBED_DIM],
            sum_sq:         vec![0.0f64; EMBED_DIM],
            count:          0,
            reservoir:      Vec::new(),
            reservoir_seen: 0,
        }
    }

    fn add(&mut self, emb: &[f32]) {
        if emb.len() != EMBED_DIM { return; }
        self.count += 1;
        for (i, &v) in emb.iter().enumerate() {
            let v64 = v as f64;
            self.sum[i]    += v64;
            self.sum_sq[i] += v64 * v64;
        }
        self.reservoir_seen += 1;
        if self.reservoir.len() < RESERVOIR_SIZE {
            self.reservoir.push(emb.to_vec());
        } else {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            self.reservoir_seen.hash(&mut h);
            let r = (h.finish() as usize) % self.reservoir_seen as usize;
            if r < RESERVOIR_SIZE {
                self.reservoir[r] = emb.to_vec();
            }
        }
    }

    fn remove(&mut self, emb: &[f32]) {
        if emb.len() != EMBED_DIM || self.count == 0 { return; }
        self.count -= 1;
        for (i, &v) in emb.iter().enumerate() {
            let v64 = v as f64;
            self.sum[i]    -= v64;
            self.sum_sq[i] -= v64 * v64;
        }
    }

    fn geometry(&self, group_name: &str) -> Option<serde_json::Value> {
        let n = self.count as usize;
        if n < 2 { return None; }
        let n_f = self.count as f64;

        let mut variance = vec![0.0f64; EMBED_DIM];
        for d in 0..EMBED_DIM {
            let mean_d    = self.sum[d] / n_f;
            let mean_sq_d = self.sum_sq[d] / n_f;
            variance[d]   = (mean_sq_d - mean_d * mean_d).max(0.0);
        }

        let sum_var: f64    = variance.iter().sum();
        let sum_var_sq: f64 = variance.iter().map(|v| v * v).sum();
        let effective_dim = if sum_var_sq > 1e-30 {
            (sum_var * sum_var) / sum_var_sq
        } else { 0.0 };
        let isotropy = effective_dim / EMBED_DIM as f64;

        let res = &self.reservoir;
        let max_pairs = 500usize;
        let mut cos_sum = 0.0f64;
        let mut pair_count = 0u64;
        if res.len() <= 32 {
            for i in 0..res.len() {
                for j in (i + 1)..res.len() {
                    let dot: f64 = res[i].iter().zip(res[j].iter())
                        .map(|(&a, &b)| a as f64 * b as f64).sum();
                    cos_sum += dot;
                    pair_count += 1;
                }
            }
        } else {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            group_name.hash(&mut h);
            let mut seed = h.finish();
            let rn = res.len();
            for _ in 0..max_pairs {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let i = (seed >> 32) as usize % rn;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (seed >> 32) as usize % rn;
                if i == j { continue; }
                let dot: f64 = res[i].iter().zip(res[j].iter())
                    .map(|(&a, &b)| a as f64 * b as f64).sum();
                cos_sum += dot;
                pair_count += 1;
            }
        }
        let mean_cosine = if pair_count > 0 { cos_sum / pair_count as f64 } else { 0.0 };

        Some(serde_json::json!({
            "group":           group_name,
            "count":           n,
            "effective_dim":   (effective_dim * 10.0).round() / 10.0,
            "isotropy":        (isotropy * 1000.0).round() / 1000.0,
            "mean_cosine_sim": (mean_cosine * 1000.0).round() / 1000.0,
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterLevel {
    #[default]
    None,
    Signatures,
    MinimalContext,
}

pub fn extract_bm25_text(content: &str, level: FilterLevel) -> String {
    match level {
        FilterLevel::None => content.to_string(),
        FilterLevel::Signatures => extract_signatures(content),
        FilterLevel::MinimalContext => extract_signatures_with_docs(content),
    }
}

fn is_signature_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("pub fn ") || t.starts_with("fn ") ||
    t.starts_with("pub struct ") || t.starts_with("struct ") ||
    t.starts_with("pub enum ") || t.starts_with("enum ") ||
    t.starts_with("pub trait ") || t.starts_with("trait ") ||
    t.starts_with("impl ") || t.starts_with("pub impl ") ||
    t.starts_with("pub type ") || t.starts_with("type ") ||
    t.starts_with("pub const ") || t.starts_with("const ") ||
    t.starts_with("def ") || t.starts_with("class ") ||
    t.starts_with("function ") || t.starts_with("async fn ") ||
    t.starts_with("pub async fn ")
}

fn extract_signatures(content: &str) -> String {
    content.lines().filter(|l| is_signature_line(l)).collect::<Vec<_>>().join("\n")
}

fn extract_signatures_with_docs(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if is_signature_line(line) {
            if i > 0 {
                let prev = lines[i - 1].trim();
                if prev.starts_with("///") || prev.starts_with("//") || prev.starts_with('#') {
                    result.push(lines[i - 1]);
                }
            }
            result.push(line);
        }
    }
    result.join("\n")
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Stratified recall post-processor: cap any single realm to ceil(k/divisor)
/// hits so a dominant realm can't flood unscoped results. `divisor` 0 disables.
/// Input must already be sorted desc by score (true for every recall lane), so
/// the first `per_realm_cap` survivors of each realm are its highest-scoring
/// ones — preserving global score order while truncating to `k`.
/// Cap any single realm's share of an unscoped recall so a dominant realm can't
/// flood results. When `reliability` is `Some`, each realm's cap is Thompson-
/// sampled from its Beta posterior via `sample_arm` (reliable realms earn more
/// slots; unreliable/unknown realms fall back to the anti-flooding floor of 1).
/// `None` disables capping entirely (scoped queries are never reshaped).
/// Merge semantic + keyword result lists via Reciprocal Rank Fusion:
/// `RRF(doc) = Σ_i 1/(rrf_k + rank_i(doc))` with 1-based ranks. The fused
/// score is written into `RecallHit::score` and the list is sorted desc and
/// truncated to `k`. Each memory appears once even if present in both lanes.
fn rrf_merge(
    semantic: Vec<RecallHit>,
    keyword: Vec<RecallHit>,
    k: usize,
    rrf_k: f32,
) -> Vec<RecallHit> {
    use std::collections::HashMap;
    let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();
    for (rank, hit) in semantic.iter().enumerate() {
        *rrf_scores.entry(hit.memory_id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
    }
    for (rank, hit) in keyword.iter().enumerate() {
        *rrf_scores.entry(hit.memory_id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
    }
    let mut seen: HashMap<MemoryId, RecallHit> = HashMap::new();
    for hit in semantic.into_iter().chain(keyword.into_iter()) {
        seen.entry(hit.memory_id).or_insert(hit);
    }
    let mut ranked: Vec<RecallHit> = rrf_scores
        .into_iter()
        .filter_map(|(id, score)| {
            seen.remove(&id).map(|mut h| {
                h.score = score;
                h
            })
        })
        .collect();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
    ranked.truncate(k);
    ranked
}

fn stratify_recall_hits(
    mut hits: Vec<RecallHit>,
    k: usize,
    reliability: Option<&crate::learner::DomainReliability>,
) -> Vec<RecallHit> {
    let reliability = match reliability {
        Some(r) if k > 0 && hits.len() > 1 => r,
        _ => {
            hits.truncate(k);
            return hits;
        }
    };
    // One seed per stratify call; per-realm draws decorrelate inside sample_arm.
    let seed = now_ms() as u64;
    let mut caps: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut keep = Vec::with_capacity(hits.len().min(k));
    for hit in hits.drain(..) {
        if keep.len() == k {
            break;
        }
        let cap = *caps.entry(hit.realm.clone()).or_insert_with(|| {
            let h = hit
                .realm
                .bytes()
                .fold(seed, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            reliability.sample_arm(&hit.realm, k, h)
        });
        let n = counts.entry(hit.realm.clone()).or_insert(0);
        if *n < cap {
            *n += 1;
            keep.push(hit);
        }
    }
    keep
}

/// Delete old snapshot families from `data_dir`, keeping the `keep` most recent.
/// Identifies families by `chitta.*.snapshot` mtime order; removes all sidecar
/// extensions for each stale stem.
/// Delete WAL segments fully dominated by the coverage vector (THEORY.md §4).
/// Segment file names are `{instance:08x}_{first_seqno:012}.seg`.
///
/// An interior segment (one with a later sibling from the same instance) is
/// prunable iff covered[instance] >= its end (the next sibling's first_seqno-1).
///
/// The final segment of an instance is open-ended — its end can't be read from
/// filenames — so it is pruned only for a DEAD instance (`inst != live`): that
/// segment is closed forever and covered[inst] is its exact fold watermark
/// (replay walks the whole segment and records its max seqno; log.rs coverage),
/// so presence in `covered` proves full domination. The LIVE instance's tail
/// may still grow and is never pruned. Absence from `covered` (e.g. a
/// lineage-fenced foreign-vsid segment, never folded) also keeps the segment.
/// Without this, a daemon whose every lifetime writes a single segment (one
/// instance-id each) accumulates them unboundedly — windows(2) is empty so the
/// interior rule alone never fires. Returns deleted count.
fn prune_covered_segments(
    seg_dir: &std::path::Path,
    covered: &std::collections::BTreeMap<crate::ids::InstanceId, u64>,
    live_instance: crate::ids::InstanceId,
) -> usize {
    let mut per_instance: std::collections::BTreeMap<u32, Vec<(u64, std::path::PathBuf)>> =
        std::collections::BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir(seg_dir) {
        for path in entries.filter_map(|e| e.ok().map(|e| e.path())) {
            if path.extension().map(|e| e == "seg") != Some(true) {
                continue;
            }
            let stem = path.file_stem().and_then(|f| f.to_str()).unwrap_or("");
            let (inst, first) = match stem.split_once('_') {
                Some((i, f)) => (
                    u32::from_str_radix(i, 16).ok(),
                    f.parse::<u64>().ok(),
                ),
                None => (None, None),
            };
            if let (Some(inst), Some(first)) = (inst, first) {
                per_instance.entry(inst).or_default().push((first, path));
            }
        }
    }
    let mut deleted = 0usize;
    for (inst, mut segs) in per_instance {
        segs.sort();
        // Empty segments (size <= header, zero ops) of a DEAD instance hold no
        // memories by construction — a daemon lifetime that opened a segment and
        // appended nothing before exiting. Coverage is op-derived (log.rs records
        // it per replayed op) so it can NEVER prove an empty segment; prune them
        // directly by size. Excludes the live instance, whose freshly-opened
        // segment is also header-sized (V3_HEADER_SIZE) until its first op.
        if inst != live_instance {
            segs.retain(|(_, path)| {
                let empty = std::fs::metadata(path)
                    .map(|m| m.len() <= crate::log::V3_HEADER_SIZE as u64)
                    .unwrap_or(false);
                if empty && std::fs::remove_file(path).is_ok() {
                    deleted += 1;
                    false
                } else {
                    true
                }
            });
        }
        let max_covered = covered.get(&inst).copied().unwrap_or(0);
        for w in segs.windows(2) {
            let (_, ref path) = w[0];
            let (next_first, _) = w[1];
            let seg_end = next_first.saturating_sub(1);
            if max_covered >= seg_end && std::fs::remove_file(path).is_ok() {
                deleted += 1;
            }
        }
        // The instance's final (open-ended) segment: prune only for a dead
        // instance present in `covered` (fully folded). Never the live tail.
        if inst != live_instance {
            if let Some((first, path)) = segs.last() {
                if covered.get(&inst).is_some_and(|&c| c >= *first)
                    && std::fs::remove_file(path).is_ok()
                {
                    deleted += 1;
                }
            }
        }
    }
    deleted
}

fn prune_old_snapshots(data_dir: &std::path::Path, keep: usize) {
    const SIDECAR_EXTS: &[&str] = &[
        "snapshot", "hdc", "emb", "bin", "mu", "shdr", "hnsw", "realm_hnsw", "pld", "sup.json", "rsf",
    ];
    let delta_ext = "delta.hnsw";

    // Collect (seqno, stem) for all chitta.*.snapshot files. Keep by SEQNO, not mtime:
    // the just-written snapshot always has the highest seqno, whereas mtime can be misleading
    // (sidecar rewrites / NFS resurrection can make an older family appear newer, which would
    // wrongly prune the snapshot we just saved — fatal for the re-embed migration's output).
    let mut families: Vec<(u64, String)> = match std::fs::read_dir(data_dir) {
        Ok(rd) => rd.filter_map(|e| {
            let entry = e.ok()?;
            let name = entry.file_name().into_string().ok()?;
            if !name.starts_with("chitta.") || !name.ends_with(".snapshot") { return None; }
            let stem = name.strip_suffix(".snapshot")?.to_string();
            let seqno = crate::snapshot::FullSnapshot::peek_seqno(&entry.path()).unwrap_or(0);
            Some((seqno, stem))
        }).collect(),
        Err(_) => return,
    };

    families.sort_by(|a, b| b.0.cmp(&a.0)); // highest seqno first
    let mut keep_stems: std::collections::HashSet<String> =
        families.iter().take(keep).map(|(_, s)| s.clone()).collect();
    // Lineage guard: the seqno ranking above is vsid-blind, so a higher-seqno snapshot from
    // a FOREIGN vector space (e.g. a stale pre-migration lineage that keeps consolidating)
    // can outrank — and evict — the only family this binary can actually load, leaving the
    // store unopenable on the next boot (cf_open fails: no candidate passes the load fence).
    // Always retain the best (highest-seqno) family whose .shdr matches the compiled vector
    // space, on top of the keep-N window. `families` is sorted seqno-desc, so the first match
    // is the best loadable family.
    if let Some((_, best_compiled)) = families.iter().find(|(_, stem)| {
        let shdr = data_dir.join(format!("{}.shdr", stem));
        crate::snapshot::StoreHeader::load(&shdr)
            .map(|h| h.matches_compiled())
            .unwrap_or(false)
    }) {
        keep_stems.insert(best_compiled.clone());
    }

    let mut removed = 0usize;

    // Delete stale snapshot families (paired sidecars). NOTE: use a plain unlink here —
    // do NOT truncate-before-unlink. On a slow/replicating NFS (Isilon) a multi-GB ftruncate
    // can block for tens of seconds while this runs under the snapshot-save lock, starving
    // recall and every other op (observed: 72-deep pool stall). unlink is metadata-only/fast.
    for (_, stem) in families.iter().skip(keep) {
        for ext in SIDECAR_EXTS {
            let p = data_dir.join(format!("{}.{}", stem, ext));
            if std::fs::remove_file(&p).is_ok() { removed += 1; }
        }
        let p = data_dir.join(format!("{}.{}", stem, delta_ext));
        if std::fs::remove_file(&p).is_ok() { removed += 1; }
    }

    // Delete orphaned sidecars (chitta.* files with no corresponding .snapshot).
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        for entry in rd.filter_map(|e| e.ok()) {
            let name = match entry.file_name().into_string() { Ok(n) => n, Err(_) => continue };
            if !name.starts_with("chitta.") { continue; }
            let stem = name.split('.').take(2).collect::<Vec<_>>().join(".");
            if keep_stems.contains(&stem) { continue; }
            // Not a kept family — remove if it's a known sidecar extension.
            let is_sidecar = SIDECAR_EXTS.iter().any(|e| name.ends_with(&format!(".{}", e)))
                || name.ends_with(&format!(".{}", delta_ext));
            if is_sidecar {
                let p = data_dir.join(&name);
                if std::fs::remove_file(&p).is_ok() { removed += 1; }
            }
        }
    }

    // Prune old cortex.*.snapshot files (keep same 2 most recent stems).
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        let mut cortex: Vec<(std::time::SystemTime, std::path::PathBuf)> = rd.filter_map(|e| {
            let entry = e.ok()?;
            let name = entry.file_name().into_string().ok()?;
            if !name.starts_with("cortex.") || !name.ends_with(".snapshot") { return None; }
            let mtime = entry.metadata().ok()?.modified().ok()?;
            Some((mtime, entry.path()))
        }).collect();
        cortex.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, p) in cortex.iter().skip(keep) {
            if std::fs::remove_file(p).is_ok() { removed += 1; }
        }
    }

    // Delete stale .emb.tmp files (re-embed leftovers; safe once reembedding is done).
    if let Ok(rd) = std::fs::read_dir(data_dir) {
        let threshold = std::time::SystemTime::now()
            .checked_sub(std::time::Duration::from_secs(3600))
            .unwrap_or(std::time::UNIX_EPOCH);
        for entry in rd.filter_map(|e| e.ok()) {
            let name = match entry.file_name().into_string() { Ok(n) => n, Err(_) => continue };
            if !name.ends_with(".emb.tmp") { continue; }
            let mtime = entry.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            if mtime < threshold {
                if std::fs::remove_file(entry.path()).is_ok() { removed += 1; }
            }
        }
    }

    if removed > 0 {
        eprintln!("[chitta-field] prune_old_snapshots: removed {} files (kept {} families)", removed, keep_stems.len());
    }
}

/// NFS ghost janitor — the residue classes prune_old_snapshots doesn't cover:
///   - seen_offsets.<inst>.json of dead reader instances (thousands accumulate;
///     one file per daemon lifetime). Age-gated: a LIVE peer rewrites its file
///     every sync cycle, so anything older than `max_age_secs` is a corpse.
///     Deleting a live peer's file would force it to re-ingest foreign
///     segments from offset 0 — the age gate is the safety, not a nicety.
///   - cortex.<inst>.* and orphan chitta.<inst>.<sidecar> whose instance has
///     no chitta.<inst>.snapshot family (instances that died before a full
///     save), same age gate.
/// Deletions are recorded in .janitor.json; on every pass, previously-deleted
/// names that EXIST again are counted as resurrections (this volume restores
/// deleted files via replication) — measurement first, escalation only with
/// data. Runs after prune_old_snapshots on every snapshot save.
fn janitor_sweep(data_dir: &std::path::Path, own_instance: crate::ids::InstanceId, max_age_secs: u64) {
    let now = std::time::SystemTime::now();
    let own_hex = format!("{:08x}", own_instance);
    let ledger_path = data_dir.join(".janitor.json");

    let old_enough = |p: &std::path::Path| -> bool {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .map(|d| d.as_secs() >= max_age_secs)
            .unwrap_or(false)
    };

    // Family stems that exist (post-prune) — their sidecars are protected.
    let mut family_stems: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Stems referenced by any manifest entry stay protected even without a
    // local .snapshot (mid-save peers).
    if let Ok(Some(m)) = crate::manifest::Manifest::load(data_dir) {
        for cp in m.families.values().chain(m.checkpoints.iter()) {
            if let Some(stem) = cp.snapshot.name.strip_suffix(".snapshot") {
                family_stems.insert(stem.to_string());
            }
        }
    }
    let entries: Vec<std::path::PathBuf> = match std::fs::read_dir(data_dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return,
    };
    for p in &entries {
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("chitta.") && name.ends_with(".snapshot") {
                if let Some(stem) = name.strip_suffix(".snapshot") {
                    family_stems.insert(stem.to_string());
                }
            }
        }
    }

    // Previous ledger → resurrection accounting.
    #[derive(serde::Serialize, serde::Deserialize, Default)]
    struct Ledger {
        deleted: Vec<String>,
    }
    let prev: Ledger = std::fs::read_to_string(&ledger_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let resurrected = prev
        .deleted
        .iter()
        .filter(|n| data_dir.join(n).exists())
        .count();

    let mut deleted: Vec<String> = Vec::new();
    let mut freed: u64 = 0;
    for p in &entries {
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        let kill = if let Some(rest) = name.strip_prefix("seen_offsets.") {
            rest.strip_suffix(".json")
                .map(|hex| hex != own_hex && old_enough(p))
                .unwrap_or(false)
        } else if let Some(rest) = name.strip_prefix("cortex.") {
            // cortex.<hex>.snapshot (and cortex sidecars) for instances with
            // no full family — keep our own and anything family-protected.
            rest.split('.').next()
                .map(|hex| {
                    hex != own_hex
                        && !family_stems.contains(&format!("chitta.{hex}"))
                        && old_enough(p)
                })
                .unwrap_or(false)
        } else if let Some(rest) = name.strip_prefix("chitta.") {
            // Orphan sidecars (no .snapshot for their stem). Never touch the
            // snapshot itself here — prune_old_snapshots owns family removal.
            if name.ends_with(".snapshot") {
                false
            } else {
                rest.split('.').next()
                    .map(|hex| {
                        !family_stems.contains(&format!("chitta.{hex}")) && old_enough(p)
                    })
                    .unwrap_or(false)
            }
        } else {
            false
        };
        if kill {
            freed += std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            if std::fs::remove_file(p).is_ok() {
                deleted.push(name.to_string());
            }
        }
    }

    if !deleted.is_empty() || resurrected > 0 {
        eprintln!(
            "[chitta-field] janitor: deleted {} ghosts ({:.1} MB); {} previously-deleted resurrected",
            deleted.len(),
            freed as f64 / 1e6,
            resurrected
        );
    }
    // Carry forward names still relevant for resurrection tracking (cap 10k).
    let mut ledger = Ledger { deleted };
    for n in prev.deleted {
        if ledger.deleted.len() >= 10_000 {
            break;
        }
        if !ledger.deleted.contains(&n) {
            ledger.deleted.push(n);
        }
    }
    if let Ok(json) = serde_json::to_string(&ledger) {
        let tmp = ledger_path.with_extension("json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, &ledger_path);
        }
    }
}

// Score multipliers are now driven by the ScoringPipeline (see scoring/mod.rs).
// Status, kind, and epistemic multipliers live in scoring/config.rs and are
// configurable via scoring.json at runtime.

/// Compute embedding geometry stats for a group of embeddings.
/// Returns JSON value with group name, count, effective_dim, isotropy, mean_cosine_sim.
#[allow(dead_code)]
fn compute_geometry(embeddings: &[&[f32]], group_name: &str) -> Option<serde_json::Value> {
    let n = embeddings.len();
    if n < 2 {
        return None;
    }
    let dim = EMBED_DIM;
    let n_f = n as f64;

    // Per-dimension mean
    let mut mean = vec![0.0f64; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n_f;
    }

    // Per-dimension variance
    let mut variance = vec![0.0f64; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            let d = v as f64 - mean[i];
            variance[i] += d * d;
        }
    }
    for v in &mut variance {
        *v /= n_f;
    }

    // Participation ratio: effective dimensionality
    let sum_var: f64 = variance.iter().sum();
    let sum_var_sq: f64 = variance.iter().map(|v| v * v).sum();
    let effective_dim = if sum_var_sq > 1e-30 {
        (sum_var * sum_var) / sum_var_sq
    } else {
        0.0
    };
    let isotropy = effective_dim / dim as f64;

    // Mean pairwise cosine similarity (sample if large)
    let max_pairs = 500usize;
    let mut cos_sum = 0.0f64;
    let mut pair_count = 0u64;
    if n <= 32 {
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f64 = embeddings[i]
                    .iter()
                    .zip(embeddings[j].iter())
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                cos_sum += dot;
                pair_count += 1;
            }
        }
    } else {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        group_name.hash(&mut h);
        let mut seed = h.finish();
        for _ in 0..max_pairs {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = (seed >> 32) as usize % n;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (seed >> 32) as usize % n;
            if i == j {
                continue;
            }
            let dot: f64 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            cos_sum += dot;
            pair_count += 1;
        }
    }
    let mean_cosine = if pair_count > 0 {
        cos_sum / pair_count as f64
    } else {
        0.0
    };

    Some(serde_json::json!({
        "group": group_name,
        "count": n,
        "effective_dim": (effective_dim * 10.0).round() / 10.0,
        "isotropy": (isotropy * 1000.0).round() / 1000.0,
        "mean_cosine_sim": (mean_cosine * 1000.0).round() / 1000.0,
    }))
}

/// Extract deterministic provenance keys from a `[done]` record body.
///
/// Grammar (from the file-provenance ritual):
///   `[done] input:<path> sha:<first-8-of-sha256> task:<what> output:<path> status:...`
///
/// Returns normalized exact-match keys: `sha:<hex>` (content identity — the
/// primary key, since NFS resurrects deleted paths so path != identity) and
/// `input:<path>` (the human handle). Both map to the same record so a lookup
/// by either the file's content-hash or its path resolves deterministically —
/// this is the keyed lane that replaces fuzzy semantic recall for the
/// anti-reprocessing question "have I already done this?".
pub(crate) fn parse_prov_keys(content: &[u8]) -> Vec<String> {
    let text = match std::str::from_utf8(content) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    if !text.starts_with("[done]") {
        return Vec::new();
    }
    let mut keys = Vec::new();
    for tok in text.split_whitespace() {
        if let Some(v) = tok.strip_prefix("sha:") {
            let v = v.trim_matches(|c: char| !c.is_ascii_alphanumeric());
            if v.len() >= 6 && v.chars().all(|c| c.is_ascii_hexdigit()) {
                keys.push(format!("sha:{}", v.to_ascii_lowercase()));
            }
        } else if let Some(v) = tok.strip_prefix("input:") {
            let v = v.trim_matches(|c: char| c == '"' || c == ',');
            if !v.is_empty() {
                keys.push(format!("input:{v}"));
            }
        }
    }
    keys
}

/// SSL structural markers + English function words + conversational-frame verbs
/// that leak across unrelated turns ("can you run", "how should I check ...").
/// The precision sweep found these frame words are the top false-fire drivers:
/// they pair into generic bigrams shared by many stored corrections, so a
/// normal turn injects an irrelevant one. Distinctive short commands ("cp",
/// "rm", "ls", "db") and content prepositions ("over") are NOT stopped — they
/// are what make a mistake phrase specific, and over-stopping collapses a phrase
/// past the 2-token bigram floor (the original bug: "cp over the running binary"
/// reduced to a single token and never keyed).
const CORRECTION_STOPWORDS: &[&str] = &[
    "use", "not", "the", "and", "for", "with", "was", "are", "you", "your",
    "that", "this", "from", "into", "but", "its", "to", "of", "in", "on", "at",
    "by", "is", "it", "as", "or", "be", "do", "so", "an", "if", "we", "my",
    // conversational-frame leakers surfaced by the false-fire sweep
    "can", "how", "what", "should", "have", "need", "want", "about", "check",
    "run", "dont", "get", "make", "did", "does", "could", "would", "will",
    "our", "no", "yes",
];

/// Lowercase, split on non-alphanumerics, drop function-word / single-char
/// tokens. Floor is 2 chars so meaningful short commands ("cp", "rm", "ls",
/// "db") survive. Shared by the store and query sides so a stored trigger and a
/// recurring turn normalize identically.
fn norm_correction_tokens(s: &str) -> Vec<String> {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|t| t.len() >= 2 && !CORRECTION_STOPWORDS.contains(t))
        .map(|t| t.to_string())
        .collect()
}

/// Strip memory-metadata footer lines (`kind:`/`tags:`/`visibility:`/`realm:`/
/// `source_session:`/`confidence:`) and filesystem-path / URL whitespace-tokens
/// from a correction body before bigram extraction. Without this the ~81% of
/// corrections that carry no explicit `NOT:`/`WRONG:` line fall back to the
/// whole body, whose path segments (`/maps/projects/caeg/...`) and metadata
/// footer produce generic bigrams (`projects_caeg`, `kind_correction`) shared
/// across nearly every stored correction — the dominant false-fire source in
/// the sweep. Paths are single whitespace-tokens, so dropping any token
/// containing `/` removes them cleanly.
/// ceiling: also drops legitimate "and/or"-style tokens; upgrade: tokenize
/// slashes only inside path-shaped runs.
fn clean_correction_text(s: &str) -> String {
    s.lines()
        .filter(|l| {
            let t = l.trim().to_ascii_lowercase();
            !(t.starts_with("kind:")
                || t.starts_with("tags:")
                || t.starts_with("visibility:")
                || t.starts_with("realm:")
                || t.starts_with("source_session:")
                || t.starts_with("confidence:"))
        })
        .flat_map(|l| l.split_whitespace())
        .filter(|w| !w.contains('/'))
        .collect::<Vec<_>>()
        .join(" ")
}

/// First double-quoted span (straight or unicode curly) with >= 2 whitespace
/// tokens. In the dominant free-form correction header `[CORRECTION to memory
/// #… ] … "<mistake>" was WRONG …`, the quoted claim IS the distinctive trigger
/// the user restates. Keying the whole body instead floods the latest-wins index
/// with dozens of low-value bigrams that newer corrections overwrite, so the
/// mistake restatement stops firing (only incidental cause-clause phrasing
/// survives) — observed live at 34k keyed triggers.
fn first_quoted_span(text: &str) -> Option<String> {
    let is_q = |c: char| c == '"' || c == '\u{201c}' || c == '\u{201d}';
    let chars: Vec<char> = text.chars().collect();
    let start = chars.iter().position(|&c| is_q(c))?;
    let end = chars[start + 1..].iter().position(|&c| is_q(c))? + start + 1;
    let span: String = chars[start + 1..end].iter().collect();
    (span.split_whitespace().count() >= 2).then_some(span)
}

/// Order-preserving adjacent bigrams of normalized tokens, namespaced
/// `correction:bg:<a>_<b>`. The bigram — not the unigram — is the trigger unit:
/// a single word over-fires on unrelated turns, a two-word phrase from the
/// actual mistake rarely collides. Returns `[]` when there are < 2 significant
/// tokens (that correction stays on the fuzzy lane — no regression).
fn correction_bigram_keys(tokens: &[String]) -> Vec<String> {
    tokens
        .windows(2)
        .map(|w| format!("correction:bg:{}_{}", w[0], w[1]))
        .collect()
}

/// Trigger keys for the correction keyed lane (capability #2, durable
/// corrections with override semantics). A `[correction]` record is stored as
/// `USE: <right>\nNOT: <wrong>`; the *wrong* phrase is the trigger — when it
/// recurs in a turn the correction must FIRE (not compete on cosine and lose).
/// Extracts the `NOT:`/`WRONG:` mistake line (falls back to the whole body for
/// the free-form "User corrected me" form), normalizes it, and emits its
/// distinctive bigrams as exact keys. Returns `[]` for non-correction content
/// or a mistake too short to bigram.
pub(crate) fn parse_correction_keys(content: &[u8]) -> Vec<String> {
    let text = match std::str::from_utf8(content) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    // Accept every correction marker, case-insensitively: the canonical
    // `[correction]` AND the free-form header `[CORRECTION to memory #… — topic]`.
    // The old lowercase-exact gate keyed only ~20% of stored corrections (most
    // lead with the uppercase header), silently disabling the deterministic lane
    // for them. Match a leading `[correction` prefix in any case.
    let lead = text.chars().take(11).collect::<String>().to_ascii_lowercase();
    if !lead.starts_with("[correction") {
        return Vec::new();
    }
    let wrong = text
        .lines()
        .find_map(|l| {
            let t = l.trim();
            t.strip_prefix("NOT:")
                .or_else(|| t.strip_prefix("WRONG:"))
                .map(|s| s.to_string())
        })
        // Free-form header form has no NOT:/WRONG: line; the quoted claim is the
        // distinctive mistake trigger. Prefer it over the whole body so the
        // latest-wins index isn't diluted by generic body bigrams.
        .or_else(|| first_quoted_span(text))
        .unwrap_or_else(|| {
            text.trim_start_matches("[correction]").to_string()
        });
    let cleaned = clean_correction_text(&wrong);
    // Distinct-key gate: only index a correction that yields >= 2 DISTINCT
    // bigrams. A 1-bigram correction would sit on a single (often generic) key
    // and, under the AND firing rule in `correction_check`, could never reach the
    // 2-key threshold anyway — so keying it only lets it steal a shared key from
    // a fully-keyable correction (latest-wins overwrite). Leaving it unindexed
    // keeps it on the fuzzy lane (no regression).
    let mut keys = correction_bigram_keys(&norm_correction_tokens(&cleaned));
    keys.sort();
    keys.dedup();
    if keys.len() < 2 {
        return Vec::new();
    }
    keys
}

/// Normalize a task slug to the canonical index key `task:<slug>`. Lowercase,
/// keep `[a-z0-9._-]`, collapse everything else out. A task's slug is its one
/// stable handle (unlike a path, which NFS resurrects) so a single normalized
/// key is the whole identity — no need for the multi-key fan-out the provenance
/// (sha+path) and correction (bigram) lanes carry.
fn norm_task_key(slug: &str) -> Option<String> {
    let s: String = slug
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .collect::<String>()
        .to_ascii_lowercase();
    if s.is_empty() { None } else { Some(format!("task:{s}")) }
}

/// Extract the deterministic task-state key from a `[task]` record body.
///
/// Grammar (from the task-state ritual):
///   `[task] id:<slug> status:<in-progress|blocked|done> next:<what> ...`
///
/// The indexable key is `task:<slug>` derived from the first `id:<slug>` (or
/// `task:<slug>`) token — a task has ONE identity, so this returns at most one
/// key. Self-gated on the `[task]` prefix (kind-independent) so the live path
/// and the load-time rebuild share one gate, mirroring `parse_correction_keys`.
/// Returns `[]` for non-`[task]` content or a record with no usable slug (that
/// record stays on the fuzzy lane — no regression).
pub(crate) fn parse_task_keys(content: &[u8]) -> Vec<String> {
    let text = match std::str::from_utf8(content) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    if !text.starts_with("[task]") {
        return Vec::new();
    }
    for tok in text.split_whitespace() {
        let slug = tok
            .strip_prefix("id:")
            .or_else(|| tok.strip_prefix("task:"));
        if let Some(slug) = slug {
            if let Some(key) = norm_task_key(slug) {
                return vec![key];
            }
        }
    }
    Vec::new()
}

impl ChittaField {
    /// Deterministic provenance lookup — the keyed lane for capability #1
    /// (anti-reprocessing). Given a content-hash and/or an input path, returns
    /// the earliest live `[done]` record that registered a matching key, or
    /// `None`. This is an O(1) exact-key lookup that BYPASSES the fuzzy
    /// BM25+HNSW retriever entirely: no embedding, no ranking, no false
    /// neighbours. `sha` is tried first (content identity), then `input` path.
    /// Both args accept a raw value or an already-prefixed key.
    pub fn provenance_lookup(&self, sha: &str, input: &str) -> Option<(MemoryId, String)> {
        let idx = self.prov_key_idx.read();
        let mut candidates: Vec<String> = Vec::new();
        let sha = sha.trim();
        if !sha.is_empty() {
            let s = sha.strip_prefix("sha:").unwrap_or(sha).to_ascii_lowercase();
            candidates.push(format!("sha:{s}"));
        }
        let input = input.trim();
        if !input.is_empty() {
            let p = input.strip_prefix("input:").unwrap_or(input);
            candidates.push(format!("input:{p}"));
        }
        for key in candidates {
            if let Some(&id) = idx.get(&key) {
                let alive = self
                    .states
                    .read()
                    .get(&id)
                    .map(|s| !s.deleted)
                    .unwrap_or(false);
                if alive {
                    let content = self
                        .payloads
                        .read()
                        .get(&id)
                        .map(|p| String::from_utf8_lossy(&p.content).into_owned())
                        .unwrap_or_default();
                    return Some((id, content));
                }
            }
        }
        None
    }

    /// Deterministic correction lookup — the keyed lane for capability #2
    /// (durable corrections with override semantics). Given free turn/context
    /// text, generates the same normalized bigram keys and probes the
    /// correction index. Returns up to `MAX_FIRED` distinct *live* corrections
    /// whose corrected-mistake trigger recurs in the text, NEWEST FIRST
    /// (latest-wins), joined by `\n---\n`. This is an O(turn_tokens) set of
    /// exact HashMap reads that BYPASSES the fuzzy BM25+HNSW retriever entirely
    /// — the correction can no longer lose on cosine similarity and be dropped
    /// from the top-k. The first element of the tuple is the newest fired
    /// correction's id (for exposed-correction / M-metric tracking).
    pub fn correction_check(&self, text: &str) -> Option<(MemoryId, String)> {
        use std::collections::HashMap;
        const MAX_FIRED: usize = 3;
        // AND firing rule: a correction fires only when >= 2 of ITS distinct
        // bigrams recur in this turn. A single shared/generic bigram is NOT
        // enough — that OR rule made ~44% of normal turns inject an irrelevant
        // stored correction (precision sweep). Requiring 2 distinct bigrams of
        // the same correction cut the false-fire rate to single digits while
        // keeping the vast majority of corrections fireable; the rest fall
        // through to the fuzzy lane (no regression).
        let mut keys = correction_bigram_keys(&norm_correction_tokens(text));
        keys.sort();
        keys.dedup();
        if keys.len() < 2 {
            return None;
        }
        // Count how many of the turn's DISTINCT bigrams map to each correction.
        // The index is multi-valued: each key resolves to every correction that
        // carries that trigger, so a record is credited for EACH of its bigrams
        // present in the turn even when a newer correction shares one. Turn keys
        // are deduped and each id appears once per key, so a per-id count of 2
        // means two distinct trigger bigrams of that correction co-occur.
        let mut per_id: HashMap<MemoryId, usize> = HashMap::new();
        {
            let idx = self.correction_key_idx.read();
            for k in &keys {
                if let Some(ids) = idx.get(k) {
                    for &id in ids {
                        *per_id.entry(id).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut ids: Vec<MemoryId> = per_id
            .into_iter()
            .filter(|&(_, n)| n >= 2)
            .map(|(id, _)| id)
            .collect();
        if ids.is_empty() {
            return None;
        }
        // Monotonic ids => descending == newest correction first.
        ids.sort_unstable_by(|a, b| b.cmp(a));
        let payloads = self.payloads.read();
        let states = self.states.read();
        let mut out: Vec<String> = Vec::new();
        let mut newest: Option<MemoryId> = None;
        for id in ids {
            if states.get(&id).map(|s| s.deleted).unwrap_or(true) {
                continue;
            }
            if let Some(p) = payloads.get(&id) {
                if newest.is_none() {
                    newest = Some(id);
                }
                out.push(String::from_utf8_lossy(&p.content).into_owned());
                if out.len() >= MAX_FIRED {
                    break;
                }
            }
        }
        newest.map(|id| (id, out.join("\n---\n")))
    }

    /// Deterministic task-state lookup — the keyed lane for capability #3 (task
    /// hand-off across discontinuous sessions). Given a task slug, returns the
    /// LATEST live `[task]` record for that id, or `None`. An O(1) exact-key
    /// HashMap read that BYPASSES the fuzzy BM25+HNSW retriever entirely: no
    /// embedding, no ranking. Unlike provenance (write-once, earliest wins), a
    /// task's status EVOLVES, so the index is latest-wins (SUPERSEDE) — the same
    /// discipline as the correction lane. `id` accepts a raw slug or an
    /// already-prefixed `task:<slug>` key.
    pub fn task_state_lookup(&self, id: &str) -> Option<(MemoryId, String)> {
        let raw = id.trim().strip_prefix("task:").unwrap_or(id.trim());
        let key = norm_task_key(raw)?;
        let mem_id = *self.task_key_idx.read().get(&key)?;
        let alive = self
            .states
            .read()
            .get(&mem_id)
            .map(|s| !s.deleted)
            .unwrap_or(false);
        if !alive {
            return None;
        }
        let content = self
            .payloads
            .read()
            .get(&mem_id)
            .map(|p| String::from_utf8_lossy(&p.content).into_owned())
            .unwrap_or_default();
        Some((mem_id, content))
    }

    /// Store a new memory. Returns `(MemoryId, ChunkHash)`.
    pub fn put_memory(
        &self,
        kind: &str,
        realm: &str,
        content: &[u8],
        embedding: &[f32],
        confidence: f32,
        decay_rate: f32,
        authored_at_ms: i64,
        artifact_refs: Vec<ArtifactRef>,
        source_session: Option<String>,
        source_tool: Option<String>,
    ) -> Result<(MemoryId, ChunkHash)> {
        // Memories shorter than this can't produce a useful BGE embedding; store as keyword-only.
        const MIN_EMBED_CHARS: usize = 20;
        let embed_pending = embedding.is_empty() && content.len() >= MIN_EMBED_CHARS;
        if !embedding.is_empty() && embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim {
                expected: EMBED_DIM,
                actual: embedding.len(),
            });
        }

        let chunk_hash = compute_chunk_hash(kind, realm, content, embedding);

        // Provenance key: content hash for [done] signal dedup (cross-realm, O(1)).
        let prov_key: Option<u64> = if kind == "signal" && content.starts_with(b"[done]") {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            content.hash(&mut h);
            Some(h.finish())
        } else {
            None
        };

        // Provenance gate: if same [done] content already stored in any realm → reinforce.
        if let Some(key) = prov_key {
            let idx = self.content_prov_idx.read();
            if let Some(&existing_id) = idx.get(&key) {
                drop(idx);
                let is_alive = self.states.read()
                    .get(&existing_id)
                    .map(|s| !s.deleted)
                    .unwrap_or(false);
                if is_alive {
                    let _ = self.update_state(existing_id, Some(0.0), Some(0.03), None, true, None);
                    return Ok((existing_id, chunk_hash));
                }
            }
        }

        {
            let idx = self.chunk_hash_idx.read();
            if let Some(&existing_id) = idx.get(&chunk_hash) {
                drop(idx);
                // Skip if the matched memory was deleted (ghost in chunk_hash_idx)
                let is_alive = self.states.read()
                    .get(&existing_id)
                    .map(|s| !s.deleted)
                    .unwrap_or(false);
                if is_alive {
                    // Recurrence: same observation seen again → boost confidence (+0.05)
                    // After 6+ recurrences, provisional (0.50) reaches durable tier (0.80)
                    let _ = self.update_state(existing_id, Some(0.0), Some(0.05), None, true, None);
                    // PoE: recurrence is a weak positive signal for the realm.
                    // 0.3 weight is low enough that repeated recurrences cannot
                    // runaway-inflate reliability.
                    self.learners
                        .write()
                        .domain_reliability
                        .record_partial_success(realm, 0.3);
                    return Ok((existing_id, chunk_hash));
                }
            }
        }

        // Semantic novelty gate (Omni-SimpleMem selective ingestion):
        // If a near-duplicate already exists (cosine_sim ≥ dedup_cosine_threshold,
        // default 0.88), skip storage and lightly reinforce the existing memory
        // instead of creating a new node. Only deduplicates within the same realm —
        // cross-realm near-matches must produce independent nodes to prevent silent
        // cross-realm reinforcement.
        //
        // CONSTRAINT — when this can fire. `embed_pending` is
        // `embedding.is_empty() && content.len() >= MIN_EMBED_CHARS`, so this guard
        // is FALSE (and the gate runs) in exactly two cases:
        //   (a) the caller supplied a vector — tests, and any `cf_put_memory` caller
        //       that passes one. This is the live path; see
        //       `put_memory_write_time_dedup_collapses_supplied_near_duplicates`.
        //   (b) content is shorter than MIN_EMBED_CHARS with no vector — then
        //       `search` is handed an empty query, returns `vec![]` on the
        //       `query.len() != EMBED_DIM` check (hnsw.rs), and nothing is collapsed.
        // The production C++ path is neither: it passes no vector and long content,
        // so it is embed_pending and skips this entirely. Near-duplicates from that
        // path are collapsed later by the supersede pass in `backfill_embedding`,
        // which is the first moment a vector exists. Do not delete this branch as
        // "dead" — case (a) is exercised by the test named above.
        if !embed_pending {
            let (dedup_thresh, dedup_upper) = {
                let cfg = &self.scoring_pipeline.read().config;
                (cfg.dedup_cosine_threshold, cfg.dedup_cosine_upper)
            };
            let neighbors = self.semantic_idx.read().search(embedding, 1, None, None);
            if let Some(top) = neighbors.first() {
                if top.cosine_similarity >= dedup_thresh && top.cosine_similarity < dedup_upper {
                    let candidate_realm = self.payloads.read()
                        .get(&top.memory_id)
                        .map(|p| p.realm.clone())
                        .unwrap_or_default();
                    let candidate_deleted = self.states.read()
                        .get(&top.memory_id)
                        .map(|s| s.deleted)
                        .unwrap_or(true);
                    if candidate_realm == realm && !candidate_deleted {
                        let _ = self.update_state(top.memory_id, Some(0.0), Some(0.02), None, true, None);
                        return Ok((top.memory_id, chunk_hash));
                    }
                }
            }
        }

        let memory_id = self.id_alloc.next_id();
        let ts = now_ms();

        let authored_at_ms = if authored_at_ms == 0 {
            ts
        } else {
            authored_at_ms
        };

        // Write-time identity-atom population: the FFI always passes an empty
        // artifact_refs, so derive them from content (paths/DOIs/run-ids) to keep
        // the bridge-lane postings index populated prospectively. Callers that do
        // supply refs (tests) are left untouched. df-gated at recall, so common
        // paths cost nothing there.
        let artifact_refs = if artifact_refs.is_empty() {
            self.derive_artifact_refs(content)
        } else {
            artifact_refs
        };

        // Captured before `source_session` is moved into the op below; used by
        // the write-time densification hook (SameSession chain) at the tail.
        let densify_session = source_session.clone();

        let op = PutPayloadOp {
            memory_id,
            version: 0,
            chunk_hash,
            created_at_ms: ts,
            authored_at_ms,
            kind: kind.to_string(),
            realm: realm.to_string(),
            content: content.to_vec(),
            embedding_model: if embed_pending { "none".to_string() } else { EMBED_MODEL_ID.to_string() },
            embedding_model_id: if embed_pending { String::new() } else { EMBED_MODEL_ID.to_string() },
            embedding_dim: if embed_pending { 0 } else { EMBED_DIM as u32 },
            embedding: embedding.to_vec(),
            artifact_refs: artifact_refs.clone(),
            harness: source_tool.as_deref().map(|t| {
                if t.starts_with("codex") { "codex".to_string() }
                else { "claude-code".to_string() }
            }),
            source_session,
            source_tool,
        };

        let op_enum = Op::PutPayload(op.clone());
        let _seqno = self.log.write().append(&op_enum)?;
        // Flush the append to the OS (cheap, microseconds) under the lock; the durable
        // fdatasync runs OFF the C++ rpc_mutex via cf_sync after the caller releases it,
        // so recall is no longer blocked by the per-write fsync (~200-330ms on NFS /home).
        let _ = self.log.write().flush_buf();

        let mut payload = MemoryPayload::from(op);
        if !embed_pending {
            // The semantic index (upserted below) is the embedding's single
            // in-RAM home; a payload copy would duplicate ~600MB across the
            // store. The WAL op above keeps the full vector for replay.
            payload.embedding = Vec::new();
        }
        let mut state = MemoryState::new(memory_id, chunk_hash, ts);
        state.confidence = confidence;
        state.decay_rate = decay_rate;

        state.embed_pending = embed_pending;
        if embed_pending {
            self.pending_embed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.payloads.write().insert(memory_id, payload);
        self.pld_mutations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.memory_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.states.write().insert(memory_id, state);
        self.chunk_hash_idx
            .write()
            .entry(chunk_hash)
            .or_insert(memory_id);
        self.realm_members
            .write()
            .entry(realm.to_string())
            .or_default()
            .insert(memory_id);
        self.kind_members
            .write()
            .entry(kind.to_string())
            .or_default()
            .insert(memory_id);
        if let Some(key) = prov_key {
            self.content_prov_idx.write().entry(key).or_insert(memory_id);
        }
        // Keyed provenance lane: index parsed (sha / input-path) keys so the
        // deterministic anti-reprocessing lookup can hit this record by exact
        // key. Gated to kind=="signal" to match the load-time rebuild exactly.
        // or_insert => earliest live record wins (chronological on the live path;
        // the rebuild sorts by created_at to match).
        if kind == "signal" {
            for pk in parse_prov_keys(content) {
                self.prov_key_idx.write().entry(pk).or_insert(memory_id);
            }
        }
        // Correction keyed lane (capability #2, durable corrections with
        // override). Index the mistake's distinctive bigrams -> this correction.
        // Multi-valued append (not overwrite): a bigram accumulates every
        // correction that carries it, so no record loses a distinctive trigger to
        // a newer correction sharing a domain-common word. Supersession is
        // applied at query time (deleted-filter + newest-first). Self-gated on the
        // case-insensitive `[correction` prefix inside parse_correction_keys, so
        // no kind check — matches the load-time rebuild exactly.
        for ck in parse_correction_keys(content) {
            let mut idx = self.correction_key_idx.write();
            let v = idx.entry(ck).or_default();
            if !v.contains(&memory_id) {
                v.push(memory_id);
            }
        }
        // Task-state keyed lane (capability #3, task hand-off). Index the task's
        // slug -> this record. `insert` (NOT `or_insert`): a newer status for the
        // same task SUPERSEDES the older one (LATEST-WINS), since status evolves
        // in-progress -> done. Self-gated on the `[task]` prefix inside
        // parse_task_keys, so no kind check — matches the load-time rebuild.
        for tk in parse_task_keys(content) {
            self.task_key_idx.write().insert(tk, memory_id);
        }
        if !embed_pending {
            self.semantic_idx
                .write()
                .upsert(memory_id, embedding.to_vec(), Some(realm));

            // Write-path: compute interference density (competitive_weight + lure_risk).
            // Query k=8 nearest neighbors to measure local crowding.
            // Exclude self and near-exact matches (above dedup threshold) since
            // those represent the same information, not competitors.
            let dedup_upper = self.scoring_pipeline.read().config.dedup_cosine_upper;
            let neighbors = self.semantic_idx.read().search(embedding, 9, None, None);
            if neighbors.len() > 1 {
                let payloads_r = self.payloads.read();
                let mut cos_sum = 0.0f32;
                let mut same_kind_count = 0u32;
                let mut neighbor_count = 0u32;
                for n in &neighbors {
                    if n.memory_id == memory_id { continue; }
                    if n.cosine_similarity >= dedup_upper { continue; }
                    cos_sum += n.cosine_similarity;
                    neighbor_count += 1;
                    if let Some(p) = payloads_r.get(&n.memory_id) {
                        if p.kind == kind { same_kind_count += 1; }
                    }
                }
                drop(payloads_r);
                if neighbor_count > 0 {
                    let cw = cos_sum / neighbor_count as f32;
                    let same_kind_ratio = same_kind_count as f32 / neighbor_count as f32;
                    let lure = cw * same_kind_ratio;
                    let mut states_w = self.states.write();
                    if let Some(st) = states_w.get_mut(&memory_id) {
                        st.competitive_weight = cw;
                        st.lure_risk = lure;
                    }
                }
            }
        }
        let content_str = std::str::from_utf8(content).unwrap_or("").to_string();
        let observer_canonicals = self.observer.extract(
            &content_str, memory_id, authored_at_ms, &mut self.observer_state.write(),
        );
        let index_text = if observer_canonicals.is_empty() {
            extract_bm25_text(&content_str, self.filter_level())
        } else {
            let canonical_text = observer_canonicals.join(". ");
            format!(
                "{} {}",
                extract_bm25_text(&content_str, self.filter_level()),
                extract_bm25_text(&canonical_text, self.filter_level()),
            )
        };
        // Genome 'process' memories are JSON config, not prose: keep them in the
        // semantic index (searchable) but out of the BM25 keyword index.
        if kind != "process" {
            self.keyword_idx.write().index(memory_id, &index_text);
        }
        self.hdc_idx.write().insert(memory_id, &content_str, realm);

        // Log structured event to CEC tape; compute surprisal for strength gating.
        {
            let (sym, turn, last_n, surprisal) = {
                let mut tape = self.event_tape.write();
                let context = tape.last_n_syms(8);
                let preview = tape.symbol_of("remember", realm, 0);
                let surprisal = self.cdawg.read().surprisal(&context, preview);
                let s = tape.log("remember", realm, 0, 0, authored_at_ms);
                let t = tape.events.len() as u32 - 1;
                let n = tape.last_n_syms(16);
                (s, t, n, surprisal)
            };
            let mut cdawg = self.cdawg.write();
            cdawg.extend(sym, turn);
            // Phase 15: update FEP model and blend surprisal signal.
            let fep_free_energy = self.fep_prior.write().observe_packed(sym, &cdawg).free_energy;
            // Surprisal-gated burn-in: blend PPM surprisal with FEP free energy.
            // High free_energy (>2.0 nats) OR high PPM surprisal → burn in memory.
            const SURPRISAL_THRESHOLD: f32 = 2.0;
            const SURPRISAL_DECAY_FACTOR: f32 = 0.5;
            let combined_surprisal = surprisal.unwrap_or(0.0) * 0.5 + fep_free_energy * 0.5;
            if combined_surprisal > SURPRISAL_THRESHOLD {
                if let Some(st) = self.states.write().get_mut(&memory_id) {
                    st.decay_rate = (st.decay_rate * SURPRISAL_DECAY_FACTOR).max(1e-6);
                }
            }
            // Positive TD credit for successful memory formation.
            cdawg.push_td_credit(&last_n, 0.05, 0.9);
        }

        // Update temporal index.
        {
            use crate::organ::temporal::TemporalEntry;
            self.time_idx.write().upsert(TemporalEntry {
                memory_id,
                ts_ms: authored_at_ms,
                kind: kind.to_string(),
                realm: realm.to_string(),
                strength: 1.0,
            });
        }

        // Update artifact index for each artifact ref.
        {
            let artifact_paths = self.artifact_paths.read();
            let mut artifact_idx = self.artifact_idx.write();
            for art_ref in &artifact_refs {
                if let Some(path) = artifact_paths.get(&art_ref.artifact_id) {
                    artifact_idx.associate(memory_id, art_ref.artifact_id, path, 1.0);
                }
            }
        }

        // Auto-encode into cortical sparse index (non-fatal if fails)
        let _ = self.encode_memory(memory_id);

        if !embedding.is_empty() {
            self.realm_stats.write().entry(realm.to_string()).or_insert_with(GroupStats::new).add(embedding);
            self.kind_stats.write().entry(kind.to_string()).or_insert_with(GroupStats::new).add(embedding);
        }

        // PoE: corrections penalise the realm they target.
        // A correction stored in realm X signals that X produced an error.
        if kind == "correction" {
            self.learners
                .write()
                .domain_reliability
                .record_correction(realm);
        }

        // G6: register process-genome into the QD archive under its (realm, task_type) niche.
        if kind == "process" {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(content) {
                let genome_realm = v["sampler_config"]["realm"].as_str().unwrap_or("unknown").to_string();
                let task = v["task_type"].as_str().unwrap_or("unknown").to_string();
                let desc = crate::learner::archive::BehaviorDescriptor { realm: genome_realm, task_type: task };
                // fitness = 0.5 default (G7/G11 will update)
                self.archive.write().unwrap().update(desc, memory_id, 0.5);
            }
        }

        // Span-lane live link: extract this memory's verbatim atoms and form the
        // memory↔span edge immediately (idempotent by content hash). In-RAM only;
        // the periodic span flush persists — keeps the sidecar serialize off the
        // write hot path. No other locks are held here (span_store is last).
        self.span_link_memory(memory_id, &String::from_utf8_lossy(content), realm);

        // Write-time causal densification (#13): link this memory to its recent
        // same-session, same-realm siblings with SameSession edges. Default off;
        // gated by CHITTA_DENSIFY. edge_legal() drops rationalization pairs.
        self.densify_write_edges(memory_id, realm, densify_session.as_deref());

        Ok((memory_id, chunk_hash))
    }

    /// Write-time SameSession chain (#13). When enabled, links `new_id` to the
    /// last K same-session, same-realm, non-deleted memories with bidirectional
    /// SameSession edges of decaying weight, then records `new_id` in the ring.
    ///
    /// Bidirectional so the new memory is reachable from its siblings and vice
    /// versa in graph recall (assoc_edges is keyed by src). edge_legal() inside
    /// add_assoc_edge silently drops any pair that would launder speculation as
    /// evidence, so the surviving chain respects the Phase-16 kind lattice.
    ///
    /// Rollback: `remove_assoc_edges_by_type(EdgeType::SameSession)` — no other
    /// code path creates SameSession edges, so removal is surgical.
    fn densify_write_edges(&self, new_id: MemoryId, realm: &str, session: Option<&str>) {
        use std::sync::atomic::Ordering;
        if !self.densify_enabled.load(Ordering::Relaxed) {
            return;
        }
        let session = match session {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return,
        };
        const DENSIFY_K: usize = 3;
        const BASE_WEIGHT: f32 = 0.6; // EdgeType::SameSession nominal weight.
        const DECAY: f32 = 0.7; // Older siblings link more weakly.

        // Snapshot the recent sibling ids under session_recent alone, then drop
        // that lock BEFORE taking payloads/states. payloads is ordered earlier
        // than session_recent, so holding session_recent while acquiring payloads
        // is a lock-order inversion (deadlock risk) — collect ids first, filter
        // after. Already in the ring → set_source_session fired after put_memory
        // already carried the session; skip the double-fire.
        let candidates: Vec<MemoryId> = {
            let ring = self.session_recent.read();
            match ring.get(&session) {
                Some(q) if q.contains(&new_id) => return,
                Some(q) => q.iter().rev().filter(|id| **id != new_id).copied().collect(),
                None => Vec::new(),
            }
        };
        let recent: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            candidates
                .into_iter()
                .filter(|id| payloads.get(id).map(|p| p.realm == realm).unwrap_or(false))
                .filter(|id| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .take(DENSIFY_K)
                .collect()
        };

        let mut weight = BASE_WEIGHT;
        for prev in recent {
            // Bidirectional; edge_legal drops illegal directions internally.
            let _ = self.add_assoc_edge(new_id, prev, EdgeType::SameSession, weight);
            let _ = self.add_assoc_edge(prev, new_id, EdgeType::SameSession, weight);
            weight *= DECAY;
        }

        // Record this memory as the newest in the session ring (bounded to K).
        let mut ring = self.session_recent.write();
        let q = ring.entry(session).or_default();
        q.push_back(new_id);
        while q.len() > DENSIFY_K {
            q.pop_front();
        }
    }

    /// Remove all assoc edges of a given type across the whole graph. Returns
    /// the number removed. Surgical rollback for #13 densification
    /// (EdgeType::SameSession — no other path creates that type).
    ///
    /// In-RAM mutation persisted at the next snapshot, matching the store's
    /// existing convention for edge mutation (see set_source_session).
    /// ceiling: a crash between this call and the next snapshot save replays the
    /// original AddAssocEdge ops from the WAL and resurrects the edges; upgrade:
    /// call save_full_snapshot() right after to truncate the WAL.
    pub fn remove_assoc_edges_by_type(&self, edge_type: EdgeType) -> usize {
        let mut removed = 0usize;
        let mut edges = self.assoc_edges.write();
        for list in edges.values_mut() {
            let before = list.len();
            list.retain(|e| e.edge_type != edge_type);
            removed += before - list.len();
        }
        edges.retain(|_, list| !list.is_empty());
        removed
    }

    /// Assoc-graph census: per-EdgeType directed-edge count plus a weight
    /// histogram (buckets: <0.05, 0.05-0.2, 0.2-0.5, 0.5-0.8, >=0.8). Read-only;
    /// the measure-first gate for plasticity levers (e.g. Gate B co-retrieval
    /// decay needs to know whether CoRetrieved edges exist at all before a
    /// decay rule is worth building).
    pub fn assoc_census(&self) -> [(u64, [u64; 5]); 7] {
        let mut out = [(0u64, [0u64; 5]); 7];
        let edges = self.assoc_edges.read();
        for list in edges.values() {
            for e in list {
                // Same wire numbering as ffi.rs et_u8 (Supports=4, Contradicts=5),
                // NOT declaration order — keeps tool output consistent with
                // cooccurrence_graph edge_type values.
                let t = match e.edge_type {
                    EdgeType::DerivedFrom => 0usize,
                    EdgeType::SameSession => 1,
                    EdgeType::SameArtifact => 2,
                    EdgeType::CoRetrieved => 3,
                    EdgeType::Supports => 4,
                    EdgeType::Contradicts => 5,
                    EdgeType::SemanticNeighbor => 6,
                };
                out[t].0 += 1;
                let b = match e.weight {
                    w if w < 0.05 => 0,
                    w if w < 0.2 => 1,
                    w if w < 0.5 => 2,
                    w if w < 0.8 => 3,
                    _ => 4,
                };
                out[t].1[b] += 1;
            }
        }
        out
    }

    /// Gate B: multiplicative decay + floor-prune for one assoc EdgeType.
    /// Every edge of `edge_type` has its weight multiplied by `factor`; edges
    /// ending below `prune_below` are removed. `apply=false` is a DRY RUN
    /// (counts only, no mutation). Serves both the hourly plasticity pass
    /// (0.98 / 0.05, gated CHITTA_PLASTICITY_DECAY via run_demotion_pass) and
    /// the one-shot saturation migration (factor 1.0, prune_below 0.2) through
    /// the assoc_decay tool. Returns (survivors, pruned).
    ///
    /// In-RAM mutation persisted at the next snapshot (same convention and
    /// WAL-replay ceiling as remove_assoc_edges_by_type above).
    pub fn assoc_decay(
        &self,
        edge_type: EdgeType,
        factor: f32,
        prune_below: f32,
        apply: bool,
    ) -> (u64, u64) {
        let mut survivors = 0u64;
        let mut pruned = 0u64;
        if apply {
            let mut edges = self.assoc_edges.write();
            for list in edges.values_mut() {
                list.retain_mut(|e| {
                    if e.edge_type != edge_type {
                        return true;
                    }
                    e.weight *= factor;
                    if e.weight < prune_below {
                        pruned += 1;
                        false
                    } else {
                        survivors += 1;
                        true
                    }
                });
            }
            edges.retain(|_, list| !list.is_empty());
        } else {
            let edges = self.assoc_edges.read();
            for list in edges.values() {
                for e in list {
                    if e.edge_type != edge_type {
                        continue;
                    }
                    if e.weight * factor < prune_below {
                        pruned += 1;
                    } else {
                        survivors += 1;
                    }
                }
            }
        }
        (survivors, pruned)
    }

    /// #14 fine-tune: read-only dump of the memory graph for offline training-
    /// pair mining. Writes two files into `out_dir`:
    ///   nodes.jsonl — one live (non-deleted) memory per line: id, kind, realm,
    ///                 status, source_session, provenance, created_at_ms,
    ///                 confidence, strength, access_count, content (utf8-lossy)
    ///   edges.jsonl — every assoc edge: src, dst, t (wire u8), w
    /// Vectors are NOT dumped — payload.embedding is empty post-V23; read the
    /// snapshot family's .emb sidecar directly ([magic u64][count u64] then
    /// (id u64 + EMBED_DIM f32) records, all LE).
    /// All policy (gold exclusion, realm filters, splits) lives in the Python
    /// miner — this stays a dumb, deterministic snapshot of substrate state.
    /// Returns (nodes_written, edges_written).
    pub fn dump_training_graph(&self, out_dir: &std::path::Path) -> std::io::Result<(u64, u64)> {
        use std::io::Write;
        std::fs::create_dir_all(out_dir)?;

        let payloads = self.payloads.read();
        let states = self.states.read();

        let mut nodes = std::io::BufWriter::new(std::fs::File::create(out_dir.join("nodes.jsonl"))?);
        let mut n_nodes = 0u64;
        for (id, p) in payloads.iter() {
            let Some(s) = states.get(id) else { continue };
            if s.deleted {
                continue;
            }
            let line = serde_json::json!({
                "id": id,
                "kind": p.kind,
                "realm": p.realm,
                "status": format!("{:?}", s.status),
                "source_session": p.source_session,
                "provenance": p.provenance,
                "created_at_ms": s.created_at_ms,
                "confidence": s.confidence,
                "strength": s.strength,
                "access_count": s.access_count,
                "content": String::from_utf8_lossy(&p.content),
            });
            writeln!(nodes, "{line}")?;
            n_nodes += 1;
        }
        drop(payloads);
        drop(states);

        let mut edges_f = std::io::BufWriter::new(std::fs::File::create(out_dir.join("edges.jsonl"))?);
        let mut n_edges = 0u64;
        let edges = self.assoc_edges.read();
        for (src, list) in edges.iter() {
            for e in list {
                // Wire numbering (matches ffi edge_type_from_u8).
                let t: u8 = match e.edge_type {
                    EdgeType::DerivedFrom => 0,
                    EdgeType::SameSession => 1,
                    EdgeType::SameArtifact => 2,
                    EdgeType::CoRetrieved => 3,
                    EdgeType::Supports => 4,
                    EdgeType::Contradicts => 5,
                    EdgeType::SemanticNeighbor => 6,
                };
                writeln!(edges_f, "{{\"src\":{},\"dst\":{},\"t\":{},\"w\":{}}}", src, e.dst, t, e.weight)?;
                n_edges += 1;
            }
        }
        Ok((n_nodes, n_edges))
    }

    /// One-shot retro-backfill of #13 SameSession edges over EXISTING memories.
    /// Groups non-deleted, session-tagged memories by (source_session, realm),
    /// orders each group by created_at_ms, and links each memory to its up-to-K
    /// most-recent prior siblings with decaying-weight SameSession edges — the
    /// same rule densify_write_edges applies at write time, but retroactively so
    /// historical golds that predate the write hook get their edges.
    ///
    /// `apply=false` is a DRY RUN: counts the sibling pairs the rule would create
    /// plus a group-size histogram, and writes nothing. Idempotent: add_assoc_edge
    /// merges by max weight and SameSession is created by no other path, so
    /// re-running (or running after write-time densify) is safe.
    ///
    /// Returns (sessions, memories_in_sessions, pairs, histogram) where histogram
    /// buckets group sizes as [1, 2, 3-5, 6-10, 11-50, 51+]. Applied directed
    /// edges = pairs * 2 (bidirectional).
    pub fn densify_backfill(&self, apply: bool) -> (u64, u64, u64, [u64; 6]) {
        const DENSIFY_K: usize = 3;
        const BASE_WEIGHT: f32 = 0.6;
        const DECAY: f32 = 0.7;
        // Snapshot (id, realm, session, created_at_ms) under payloads->states
        // (lock-order: payloads before states), then release before mutating.
        let mut rows: Vec<(MemoryId, String, String, i64)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .filter_map(|(id, p)| {
                    p.source_session
                        .as_ref()
                        .filter(|s| !s.is_empty())
                        .map(|s| (*id, p.realm.clone(), s.clone(), p.created_at_ms))
                })
                .collect()
        };
        // Contiguous (session, realm) groups, time-ordered within each.
        rows.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then(a.1.cmp(&b.1))
                .then(a.3.cmp(&b.3))
                .then(a.0.cmp(&b.0))
        });
        let mut sessions = 0u64;
        let mut mem_in_sessions = 0u64;
        let mut pairs = 0u64;
        let mut hist = [0u64; 6];
        let mut i = 0;
        while i < rows.len() {
            let mut j = i + 1;
            while j < rows.len() && rows[j].2 == rows[i].2 && rows[j].1 == rows[i].1 {
                j += 1;
            }
            let group = &rows[i..j];
            let n = group.len();
            sessions += 1;
            mem_in_sessions += n as u64;
            let bucket = match n {
                1 => 0,
                2 => 1,
                3..=5 => 2,
                6..=10 => 3,
                11..=50 => 4,
                _ => 5,
            };
            hist[bucket] += 1;
            // Each memory links back to its up-to-K most-recent prior siblings,
            // decaying weight per step (0.6, 0.42, 0.294) — matches the write hook.
            for idx in 1..n {
                let k = idx.min(DENSIFY_K);
                for back in 1..=k {
                    pairs += 1;
                    if apply {
                        let cur = group[idx].0;
                        let prev = group[idx - back].0;
                        let weight = BASE_WEIGHT * DECAY.powi((back - 1) as i32);
                        let _ = self.add_assoc_edge(cur, prev, EdgeType::SameSession, weight);
                        let _ = self.add_assoc_edge(prev, cur, EdgeType::SameSession, weight);
                    }
                }
            }
            i = j;
        }
        (sessions, mem_in_sessions, pairs, hist)
    }

    /// One-shot retro-backfill of SemanticNeighbor edges — the first genuine
    /// memory↔memory *knowledge* relation in the assoc graph. For each
    /// non-deleted memory with an embedding, query the realm-scoped HNSW index
    /// for its top-K dense-cosine neighbors and add bidirectional
    /// SemanticNeighbor edges weighted by cosine. Unlike CoRetrieved (a circular
    /// retrieval-history popularity prior), these encode actual semantic
    /// relatedness, so PPR bridges over real relations, not co-retrieval
    /// popularity.
    ///
    /// Realm-scoped (search passes each memory's own realm) so edges never cross
    /// project boundaries. `min_cos` floors edge quality; `k` caps out-degree.
    /// SemanticNeighbor is exempt from the Phase16/17 kind lattice in
    /// add_assoc_edge, so backfill emits no laundering triplets.
    ///
    /// `apply=false` is a DRY RUN: runs the same searches and counts candidate
    /// directed edges, writing nothing. Idempotent: add_assoc_edge merges by max
    /// weight and SemanticNeighbor is created by no other path, so re-running is
    /// safe. Returns (memories_scanned, memories_with_neighbors, directed_edges).
    pub fn semantic_backfill(&self, apply: bool, k: usize, min_cos: f32) -> (u64, u64, u64) {
        let k = k.max(1);
        let rows: Vec<(MemoryId, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .map(|(id, p)| (*id, p.realm.clone()))
                .collect()
        };
        let mut scanned = 0u64;
        let mut with_neighbors = 0u64;
        let mut edges = 0u64;
        for (id, realm) in &rows {
            scanned += 1;
            let Some(emb) = self.embedding_of(*id) else { continue };
            let realm_opt = if realm.is_empty() { None } else { Some(realm.as_str()) };
            // k+1: the query memory returns itself at rank 0; skip it below.
            let neighbors = self.semantic_idx.read().search(&emb, k + 1, None, realm_opt);
            let mut linked = false;
            for nb in neighbors {
                if nb.memory_id == *id { continue; }
                if nb.cosine_similarity < min_cos { continue; }
                // Bidirectional: kNN is not symmetric, so add both directions
                // explicitly to guarantee PPR can traverse either way. Cosine is
                // symmetric, so both edges carry the same weight.
                edges += 2;
                if apply {
                    let _ = self.add_assoc_edge(*id, nb.memory_id, EdgeType::SemanticNeighbor, nb.cosine_similarity);
                    let _ = self.add_assoc_edge(nb.memory_id, *id, EdgeType::SemanticNeighbor, nb.cosine_similarity);
                }
                linked = true;
            }
            if linked { with_neighbors += 1; }
        }
        (scanned, with_neighbors, edges)
    }

    /// Return the active cognitive-process genome: the most recently authored,
    /// non-deleted `process` memory in the `brahman` realm, parsed as JSON.
    /// Read-only — uses the existing kind/realm/payload indices, no QD archive.
    pub fn active_genome(&self) -> Option<serde_json::Value> {
        let ids: Vec<MemoryId> = {
            let kind_members = self.kind_members.read();
            kind_members.get("process")?.iter().copied().collect()
        };
        let payloads = self.payloads.read();
        let states = self.states.read();
        let latest = ids
            .iter()
            .filter_map(|id| payloads.get(id).map(|p| (*id, p)))
            .filter(|(_, p)| p.realm == "brahman")
            .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
            .max_by_key(|(_, p)| p.authored_at_ms)?;
        serde_json::from_slice(&latest.1.content).ok()
    }

    /// Retrieve the payload for a memory. Also records a touch access.
    /// Stage B: set (or clear, if empty) the natural-language retrieval surface for a
    /// memory. When set, this is what gets embedded instead of the telegraphic content.
    pub fn set_retrieval_surface(&self, memory_id: MemoryId, surface: &str) {
        if surface.is_empty() {
            self.retrieval_surfaces.write().remove(&memory_id);
        } else {
            self.retrieval_surfaces.write().insert(memory_id, surface.as_bytes().to_vec());
        }
    }

    /// Stage B: the retrieval surface for a memory, if one was stored.
    pub fn get_retrieval_surface(&self, memory_id: MemoryId) -> Option<String> {
        self.retrieval_surfaces
            .read()
            .get(&memory_id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }

    pub fn get_memory(&self, memory_id: MemoryId) -> Result<MemoryPayload> {
        let payload = self.peek_memory(memory_id)?;
        self.pending_touches.lock().push((memory_id, now_ms()));
        Ok(payload)
    }

    /// Hydrate recall results without scheduling access updates. Scoring and
    /// explicit reads own learning; metadata reads must not race the touch timer.
    pub(crate) fn peek_memory(&self, memory_id: MemoryId) -> Result<MemoryPayload> {
        {
            let states = self.states.read();
            let state = states
                .get(&memory_id)
                .ok_or(FieldError::NotFound(memory_id))?;
            if state.deleted {
                return Err(FieldError::Deleted(memory_id));
            }
        }

        let payload = self.payloads.read().get(&memory_id).cloned()
            .ok_or(FieldError::NotFound(memory_id))?;
        Ok(payload)
    }

    pub(crate) fn drain_pending_touches(&self) -> Result<()> {
        let _drain = self.touch_drain.lock();
        self.drain_pending_touches_locked()
    }

    pub(crate) fn drain_pending_touches_locked(&self) -> Result<()> {
        let mut touches = std::mem::take(&mut *self.pending_touches.lock());
        if touches.is_empty() { return Ok(()); }
        touches.sort_unstable_by_key(|&(_, ts)| ts);
        let mut preview = self.learners.read().plasticity.access_preview(touches.iter().map(|&(id, _)| id));
        let deltas: Vec<_> = {
            touches.iter().map(|&(memory_id, ts)| StateDeltaOp {
                memory_id,
                strength_delta: None, confidence_delta: None,
                decay_rate: Some(preview.record_access(memory_id, ts)),
                touch: true, pin: None, op_ts_ms: ts,
                status: None, epistemic_status: None, staged: None, invalidated_by: None,
            }).collect()
        };
        // WAL before apply, with no learner/state/pending guard held across I/O.
        if let Err(e) = self.log.write().append(&Op::UpdateStateBatch(deltas.clone())) {
            self.pending_touches.lock().extend(touches);
            return Err(e);
        }
        {
            let mut learners = self.learners.write();
            for &(id, ts) in &touches { learners.plasticity.record_access(id, ts); }
        }
        let mut states = self.states.write();
        for delta in &deltas {
            if let Some(state) = states.get_mut(&delta.memory_id) {
                crate::field::apply_access_delta(state, delta);
            }
        }
        drop(states);
        // A completed drain bounds crash loss to the accumulation interval,
        // rather than adding another WAL timer interval to the access window.
        let mut log = self.log.write();
        if log.pending_sync_count() > 0 { log.sync()?; }
        Ok(())
    }

    /// Return current mutable state for a memory.
    pub fn get_state(&self, memory_id: MemoryId) -> Result<MemoryState> {
        self.states
            .read()
            .get(&memory_id)
            .cloned()
            .ok_or(FieldError::NotFound(memory_id))
    }

    /// Apply a delta to a memory's mutable state.
    pub fn update_state(
        &self,
        memory_id: MemoryId,
        strength_delta: Option<f32>,
        confidence_delta: Option<f32>,
        decay_rate: Option<f32>,
        touch: bool,
        pin: Option<bool>,
    ) -> Result<()> {
        {
            let states = self.states.read();
            let state = states
                .get(&memory_id)
                .ok_or(FieldError::NotFound(memory_id))?;
            if state.deleted {
                return Err(FieldError::Deleted(memory_id));
            }
        }

        let ts = now_ms();
        let delta = StateDeltaOp {
            memory_id,
            strength_delta,
            confidence_delta,
            decay_rate,
            touch,
            pin,
            op_ts_ms: ts,
            status: None,
            epistemic_status: None,
            staged: None,
            invalidated_by: None,
        };
        let _seqno = self.log.write().append(&Op::UpdateState(delta.clone()))?;

        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.apply_delta(&delta, ts);
            }
        }

        Ok(())
    }

    /// Increment ack_score by 1 for the given memory (signals proven useful).
    pub fn ack_memory(&self, memory_id: MemoryId) -> Result<()> {
        if !self.states.read().contains_key(&memory_id) {
            return Err(FieldError::NotFound(memory_id));
        }
        let mut scores = self.ack_scores.write();
        let score = scores.entry(memory_id).or_insert(0);
        *score = score.saturating_add(1);
        Ok(())
    }

    /// Decrement ack_score by 1 for the given memory (signals stale or wrong).
    pub fn nack_memory(&self, memory_id: MemoryId) -> Result<()> {
        if !self.states.read().contains_key(&memory_id) {
            return Err(FieldError::NotFound(memory_id));
        }
        let mut scores = self.ack_scores.write();
        let score = scores.entry(memory_id).or_insert(0);
        *score = score.saturating_sub(1);
        Ok(())
    }

    /// Record an outcome observation against a memory's utility posterior.
    /// Orthogonal to ack/nack: ack counts *reported* usefulness into a scalar,
    /// this counts *observed* success/failure into a Beta posterior that recall
    /// can sample. Returns the updated (alpha, beta).
    ///
    /// Durable: writes a `RecordOutcome` op to the WAL before mutating, so an
    /// observation made between two periodic snapshots survives a restart
    /// (it used to live only in RAM until the next `utility_posteriors`
    /// section was written — a crash reverted α/β to the last snapshot).
    ///
    /// Existence is checked under a short read lock before the append so a
    /// bad id never reaches the log; the accrual then happens under one
    /// `states` write lock. Never called while iterating recall results
    /// (parking_lot is writer-preferring; a queued writer stalls every later
    /// reader).
    pub fn record_outcome(&self, memory_id: MemoryId, success: bool, weight: f32) -> Result<(f32, f32)> {
        if !self.states.read().contains_key(&memory_id) {
            return Err(FieldError::NotFound(memory_id));
        }
        // A non-observation (weight <= 0 or NaN) is dropped by
        // MemoryState::record_outcome — don't spend a WAL record on it.
        if weight > 0.0 {
            let op = Op::RecordOutcome(crate::ops::RecordOutcomeOp {
                memory_id,
                success,
                weight,
                ts_ms: now_ms(),
            });
            self.log.write().append(&op)?;
        }
        let mut states = self.states.write();
        let state = states.get_mut(&memory_id).ok_or(FieldError::NotFound(memory_id))?;
        state.record_outcome(success, weight);
        Ok((state.utility_alpha, state.utility_beta))
    }

    // ── Analogy Lane (VSA over the triplet store) ──────────────────────────────

    /// Copy the live triplet lane out for `crate::analogy`. Both guards are
    /// dropped before the caller does any hypervector work — scoring under the
    /// triplet read lock starves every queued writer (parking_lot is
    /// writer-preferring; see the lock-order note in field.rs).
    ///
    /// Returns (facts, source memory → realm, live triplet count). The count is
    /// the analogy index's staleness key.
    ///
    /// Realm only, deliberately: realm is what the caller filters on, and it is
    /// a short, highly-repeated string. Payload *text* is fetched separately by
    /// `analogy_texts` for the ranked subset alone — carrying every source
    /// memory's full content through here cost ~17 MB per call at 33k memories,
    /// all of it discarded except the handful of rows that survive ranking.
    pub fn analogy_snapshot(
        &self,
        max_facts: usize,
    ) -> (Vec<crate::analogy::Fact>, std::collections::HashMap<MemoryId, String>, usize) {
        let now = now_ms();
        let (facts, count) = {
            let ts = self.triplet_store.read();
            let facts = ts.analogy_facts(now, max_facts);
            (facts, ts.triplet_count())
        };
        let mut facts = facts;
        let meta = {
            let payloads = self.payloads.read();
            // Most production triplets reach the lane through add_triplet's
            // 3-argument form, which leaves source_memory_id unset (0 -> None):
            // native_distiller.cpp and ingester.cpp instead encode the source as
            // the SUBJECT, `add_triplet(std::to_string(mem_id), "has_flag", ...)`.
            // Without recovering that, the structural index is empty on a live
            // store even though the fact table is full. Only subjects that
            // resolve to a real payload are accepted, so an entity that merely
            // looks numeric is never mistaken for a memory.
            for f in facts.iter_mut() {
                if f.memory_id.is_none() {
                    if let Ok(id) = f.subject.parse::<MemoryId>() {
                        if payloads.contains_key(&id) {
                            f.memory_id = Some(id);
                        }
                    }
                }
            }
            let wanted: HashSet<MemoryId> = facts.iter().filter_map(|f| f.memory_id).collect();
            wanted
                .iter()
                .filter_map(|id| payloads.get(id).map(|p| (*id, p.realm.clone())))
                .collect()
        };
        (facts, meta, count)
    }

    /// Payload text for the ranked subset of an analogy result, keyed by id.
    /// Companion to `analogy_snapshot`, which carries realms only.
    ///
    /// Takes `payloads` alone and holds it only for the copy — a different lock
    /// from the triplet guard `analogy_snapshot` uses, and never held with it.
    /// Call this AFTER ranking and realm-filtering, so the copy is bounded by
    /// the result limit rather than by the size of the store.
    pub fn analogy_texts(&self, ids: &[MemoryId]) -> std::collections::HashMap<MemoryId, String> {
        if ids.is_empty() {
            return std::collections::HashMap::new();
        }
        let payloads = self.payloads.read();
        ids.iter()
            .filter_map(|id| {
                payloads
                    .get(id)
                    .map(|p| (*id, String::from_utf8_lossy(&p.content).into_owned()))
            })
            .collect()
    }

    // ── Span Lane (verbatim transcript atoms) ──────────────────────────────────

    /// Query the span lane. No embedding, no GPU, no LLM. `realm=None` is
    /// unscoped; a realm that has no atoms returns empty (no cross-project leak).
    /// Returns (text, class, count, last_ms, realm, session, line, score,
    /// memory_ids) tuples. `memory_ids` is the reverse edge — beliefs referencing
    /// this atom — so a matched span can jump to the memories that mention it.
    pub fn span_query(
        &self,
        query: &str,
        realm: Option<&str>,
        k: usize,
    ) -> Vec<(String, u8, u32, i64, String, String, u32, f32, Vec<u64>)> {
        self.span_store
            .write()
            .query(query, realm, k)
            .into_iter()
            .map(|h| {
                (h.text, h.class, h.count, h.last_ms, h.realm, h.session, h.line, h.score, h.memory_ids)
            })
            .collect()
    }

    /// Forward edge: the verbatim atoms a recalled memory's text references.
    /// Returns (text, class, count, realm) tuples, most-distinctive first.
    pub fn span_for_memory(&self, memory_id: u64, k: usize) -> Vec<(String, u8, u32, String)> {
        self.span_store
            .read()
            .spans_for_memory(memory_id, k)
            .into_iter()
            .map(|h| (h.text, h.class, h.count, h.realm))
            .collect()
    }

    /// Link one memory's text into the span store (idempotent by content hash),
    /// persisting immediately. For write hot paths use span_link_memory instead.
    pub fn span_ingest_memory(&self, memory_id: u64, text: &str, realm: &str) -> u64 {
        let mut s = self.span_store.write();
        let stats = s.ingest_memory(memory_id, text, realm);
        s.save_if_dirty();
        stats.new_spans
    }

    /// Deferred-persistence memory link for write hot paths (put_memory /
    /// content update): links in RAM only; span_flush persists periodically.
    pub fn span_link_memory(&self, memory_id: u64, text: &str, realm: &str) {
        self.span_store.write().ingest_memory(memory_id, text, realm);
    }

    /// Persist the span store iff it has unsaved changes. Called periodically
    /// by the queue processor and on daemon shutdown. Returns true iff saved.
    pub fn span_flush(&self) -> bool {
        self.span_store.write().save_if_dirty()
    }

    /// Backfill the memory→span edge over every live memory. Idempotent: a
    /// memory whose text is unchanged since last link is skipped. Returns
    /// (memories_linked, new_spans).
    pub fn span_backfill_memories(&self) -> (u64, u64) {
        // Snapshot (id, realm, text) under the payloads read-lock, then release it
        // before taking the span_store write-lock to avoid holding both at once.
        let snapshot: Vec<(u64, String, String)> = {
            // payloads is ordered before states — acquire in that order (lock-order audit).
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .map(|(id, p)| {
                    (*id, p.realm.clone(), String::from_utf8_lossy(&p.content).into_owned())
                })
                .collect()
        };
        let mut linked = 0u64;
        let mut new_spans = 0u64;
        {
            let mut s = self.span_store.write();
            for (id, realm, text) in &snapshot {
                let stats = s.ingest_memory(*id, text, realm);
                new_spans += stats.new_spans;
                if s.has_memory_link(*id) {
                    linked += 1;
                }
            }
            s.save();
        }
        (linked, new_spans)
    }

    /// Incrementally ingest one transcript from its watermark. Idempotent.
    /// In-RAM only (called from the queue thread on register/distill); the
    /// periodic span_flush persists spans and watermark together.
    pub fn span_ingest_transcript(&self, path: &std::path::Path) -> u64 {
        self.span_store.write().ingest_transcript(path).new_spans
    }

    /// Full backfill over a projects dir. Returns (unique_total, new, redacted).
    pub fn span_backfill(&self, projects_dir: &std::path::Path) -> (usize, u64, u64) {
        let mut s = self.span_store.write();
        let stats = s.ingest_dir(projects_dir);
        (s.len(), stats.new_spans, stats.redacted)
    }

    /// (unique_total, on_disk_bytes, redacted_total).
    pub fn span_stats(&self) -> (usize, u64, u64) {
        let s = self.span_store.read();
        (s.len(), s.on_disk_bytes(), s.redacted_total())
    }

    // ── Soul REPL session persistence ──────────────────────────────────────────

    pub fn repl_session_get(&self, id: &str) -> Option<String> {
        self.repl_sessions.read().get(id).map(|s| s.namespace_json.clone())
    }

    pub fn repl_session_set(&self, id: &str, namespace_json: &str, updated_ms: i64) {
        self.repl_sessions.write().set(id.to_string(), namespace_json.to_string(), updated_ms);
    }

    pub fn repl_session_delete(&self, id: &str) -> bool {
        self.repl_sessions.write().delete(id)
    }

    /// Execute Python code in the REPL sandbox. Atomically: get namespace →
    /// execute → persist namespace. Returns JSON result.
    pub fn repl_execute(
        &self,
        session_id: &str,
        code: &str,
        reset: bool,
        socket_path: &str,
        max_output: usize,
    ) -> String {
        let initial_ns = if reset {
            None
        } else {
            self.repl_sessions.read().get(session_id).map(|s| s.namespace_json.clone())
        };

        let result = crate::repl_executor::repl_execute(
            code,
            initial_ns.as_deref(),
            socket_path,
            max_output,
        );

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        self.repl_sessions.write().set(
            session_id.to_string(),
            result.namespace_json.clone(),
            now_ms,
        );

        serde_json::json!({
            "success":   result.success,
            "output":    result.output,
            "error":     result.error,
            "session_id": session_id,
            "trajectory": serde_json::from_str::<serde_json::Value>(&result.trajectory_json)
                .unwrap_or(serde_json::json!([])),
        }).to_string()
    }

    pub fn repl_session_list(&self) -> String {
        let store = self.repl_sessions.read();
        let entries: Vec<serde_json::Value> = store.list().iter().map(|s| serde_json::json!({
            "id": s.id,
            "updated_ms": s.updated_ms,
            "namespace_size": s.namespace_json.len(),
        })).collect();
        serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// Find memories whose content contains any of `patterns` (case-insensitive substring),
    /// for the `prune-memories` maintenance command. Matching runs under read locks that are
    /// dropped before any mutation. When `apply` is true: `action` 0 deletes (forget), 1
    /// archives (down-weight via MemoryStatus::Archived). Returns (id, kind, ≤80-char snippet)
    /// per live match. Skips already-deleted memories.
    pub fn prune_by_content(&self, patterns: &[String], apply: bool, action: u8)
        -> Result<Vec<(u64, String, String)>>
    {
        let pats: Vec<String> = patterns.iter()
            .map(|p| p.trim().to_lowercase())
            .filter(|p| !p.is_empty())
            .collect();
        if pats.is_empty() { return Ok(Vec::new()); }
        let matches: Vec<(u64, String, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads.iter().filter_map(|(&id, p)| {
                if states.get(&id).map(|s| s.deleted).unwrap_or(false) { return None; }
                let content = String::from_utf8_lossy(&p.content);
                let lc = content.to_lowercase();
                if pats.iter().any(|pat| lc.contains(pat.as_str())) {
                    Some((id, p.kind.clone(), content.chars().take(80).collect()))
                } else {
                    None
                }
            }).collect()
        };
        if apply {
            for (id, _, _) in &matches {
                if action == 1 {
                    let _ = self.set_memory_status(*id, crate::state::MemoryStatus::Archived);
                } else {
                    let _ = self.forget(*id);
                }
            }
        }
        Ok(matches)
    }

    /// Soft-delete a memory.
    pub fn forget(&self, memory_id: MemoryId) -> Result<()> {
        {
            let states = self.states.read();
            states
                .get(&memory_id)
                .ok_or(FieldError::NotFound(memory_id))?;
        }

        let ts = now_ms();
        let op = Op::DeleteMemory(DeleteMemoryOp {
            memory_id,
            deleted_at_ms: ts,
        });
        let _seqno = self.log.write().append(&op)?;
        self.log.write().sync()?; // forget is irreversible — propagate durability failures

        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.deleted = true;
            }
        }
        // Remove from temporal index (need authored_at_ms from payload).
        // Also subtract from spectral accumulators before removing from semantic_idx.
        {
            let payloads = self.payloads.read();
            if let Some(payload) = payloads.get(&memory_id) {
                if let Some(emb) = self.semantic_idx.read().get_embedding(memory_id) {
                    let emb_owned: Vec<f32> = emb.to_vec();
                    if let Some(s) = self.realm_stats.write().get_mut(&payload.realm) {
                        s.remove(&emb_owned);
                    }
                    if let Some(s) = self.kind_stats.write().get_mut(&payload.kind) {
                        s.remove(&emb_owned);
                    }
                }

                self.time_idx
                    .write()
                    .remove(memory_id, payload.authored_at_ms);
                let mut realm_members = self.realm_members.write();
                let remove_realm = if let Some(ids) = realm_members.get_mut(&payload.realm) {
                    ids.remove(&memory_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_realm {
                    realm_members.remove(&payload.realm);
                }
                let mut kind_members = self.kind_members.write();
                let remove_kind = if let Some(ids) = kind_members.get_mut(&payload.kind) {
                    ids.remove(&memory_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_kind {
                    kind_members.remove(&payload.kind);
                }
                self.hdc_idx.write().remove(memory_id, &payload.realm);
            }
        }

        self.semantic_idx.write().remove(memory_id);
        self.keyword_idx.write().remove(memory_id);
        self.cortical_idx.write().remove(memory_id);
        self.artifact_idx.write().remove_memory(memory_id);

        // Transitive forgetting: invalidate triplets sourced from this memory (each call
        // writes its own WAL op so replay stays consistent).
        let sourced_triplets = self.triplet_store.read().ids_by_source_memory(memory_id);
        for tid in sourced_triplets {
            let _ = self.invalidate_triplet(tid);
        }

        // Remove all association edges FROM and TO this memory.
        {
            let mut edges = self.assoc_edges.write();
            edges.remove(&memory_id);
            for outgoing in edges.values_mut() {
                outgoing.retain(|e| e.dst != memory_id);
            }
        }

        // Prune coactivation_stats pairs that reference this memory.
        self.coactivation_stats
            .write()
            .retain(|(a, b), _| *a != memory_id && *b != memory_id);

        // Span edge: drop this memory's atom links. A span survives if still
        // referenced by a transcript locator or another memory; it is GC'd only
        // when its refcount hits zero. O(this memory's spans), not a full scan.
        {
            let mut s = self.span_store.write();
            if s.unlink_memory(memory_id) > 0 {
                s.save();
            }
        }

        // Clear payload content bytes to reclaim memory (keep state/metadata).
        if let Some(p) = self.payloads.write().get_mut(&memory_id) {
            p.content = Vec::new();
        }

        Ok(())
    }

    /// Add a directed association edge between two memories.
    pub fn add_assoc_edge(
        &self,
        src: MemoryId,
        dst: MemoryId,
        edge_type: EdgeType,
        weight: f32,
    ) -> Result<()> {
        {
            let states = self.states.read();
            let src_state = states.get(&src).ok_or(FieldError::NotFound(src))?;
            if src_state.deleted {
                return Err(FieldError::Deleted(src));
            }
            let dst_state = states.get(&dst).ok_or(FieldError::NotFound(dst))?;
            if dst_state.deleted {
                return Err(FieldError::Deleted(dst));
            }
        }

        // Phase 16/17: edge-legality + candidate-band checks. SemanticNeighbor is
        // a structural similarity link, not an evidence citation, so it is exempt
        // from the kind lattice — otherwise a bidirectional similarity backfill
        // emits an illegal-edge/laundering triplet for every blocked pair.
        if !matches!(edge_type, EdgeType::SemanticNeighbor) {
            let payloads = self.payloads.read();
            let src_payload = payloads.get(&src);
            let dst_payload = payloads.get(&dst);
            if let (Some(sp), Some(dp)) = (src_payload, dst_payload) {
                let src_kind = MemoryKind::infer(&sp.kind, &sp.realm,
                    std::str::from_utf8(&sp.content).unwrap_or("").get(..200).unwrap_or(""));
                let dst_kind = MemoryKind::infer(&dp.kind, &dp.realm,
                    std::str::from_utf8(&dp.content).unwrap_or("").get(..200).unwrap_or(""));

                // Phase 17: candidate citing established = laundering
                if sp.candidate && !dp.candidate {
                    eprintln!("[cec:p17] candidate→established edge blocked: {}→{}", src, dst);
                    let _ = self.add_triplet(
                        "cec:contradiction_yield".into(),
                        "candidate_laundering_blocked".into(),
                        format!("id={src}→{dst}"),
                        1.0, None, None,
                    );
                    return Err(FieldError::NotFound(dst));
                }

                if !edge_legal(src_kind, dst_kind) {
                    eprintln!("[cec:p16] illegal edge blocked: {}({})→{}({})",
                        src_kind.label(), src, dst_kind.label(), dst);
                    let _ = self.add_triplet(
                        "cec:contradiction_yield".into(),
                        "illegal_edge_blocked".into(),
                        format!("{}→{} id={src}→{dst}", src_kind.label(), dst_kind.label()),
                        1.0, None, None,
                    );
                    return Err(FieldError::NotFound(dst));
                }
            }
        }

        let op = Op::AddAssocEdge(AddAssocEdgeOp {
            src,
            dst,
            edge_type: edge_type.clone(),
            weight,
        });
        let _seqno = self.log.write().append(&op)?;

        {
            let mut edges = self.assoc_edges.write();
            let list = edges.entry(src).or_default();
            if let Some(existing) = list.iter_mut().find(|e| e.dst == dst && e.edge_type == edge_type) {
                existing.weight = existing.weight.max(weight);
            } else {
                list.push(AssocEdge { dst, edge_type, weight });
            }
        }

        Ok(())
    }

    /// Return outbound association edges for a memory.
    pub fn list_neighbors(&self, memory_id: MemoryId) -> Result<Vec<AssocEdge>> {
        Ok(self
            .assoc_edges
            .read()
            .get(&memory_id)
            .cloned()
            .unwrap_or_default())
    }

    /// Total count of non-deleted memories.
    pub fn memory_count(&self) -> usize {
        self.states.read().values().filter(|s| !s.deleted).count()
    }

    /// O(1) upper-bound count — includes soft-deleted entries.
    /// Use for latency-sensitive paths (health_check fast path).
    pub fn raw_memory_count(&self) -> usize {
        self.memory_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// O(1) pending-embedding count. Maintained by put_memory/backfill_embedding.
    pub fn raw_pending_count(&self) -> usize {
        self.pending_embed_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn enqueue_recall_effects(&self, hit_ids: &[MemoryId]) {
        const MAX_PENDING_STRENGTHEN: usize = 100_000;
        const MAX_PENDING_PAIRS: usize = 200_000;
        const MAX_PENDING_WINDOWS: usize = 20_000;

        if hit_ids.is_empty() {
            return;
        }

        let strengthen_ids = &hit_ids[..hit_ids.len().min(16)];
        let window_ids = hit_ids[..hit_ids.len().min(5)].to_vec();

        let mut pending = self.pending_recall.lock();
        for &id in strengthen_ids {
            if pending.strengthen.len() >= MAX_PENDING_STRENGTHEN {
                break;
            }
            pending.strengthen.insert(id);
        }

        for i in 0..window_ids.len() {
            for j in (i + 1)..window_ids.len() {
                if pending.co_retrieval_pairs.len() >= MAX_PENDING_PAIRS
                    && !pending
                        .co_retrieval_pairs
                        .contains_key(&(window_ids[i], window_ids[j]))
                {
                    continue;
                }
                *pending
                    .co_retrieval_pairs
                    .entry((window_ids[i], window_ids[j]))
                    .or_insert(0.0) += 0.05;
            }
        }

        if !window_ids.is_empty() && pending.proto_windows.len() < MAX_PENDING_WINDOWS {
            pending.proto_windows.push(window_ids);
        }
        drop(pending);

        // Hot-path: record access sequence for predictive memory (Layer 3)
        let mut predictor = self.predictor.write();
        for &id in hit_ids.iter().take(8) {
            predictor.record_access(id);
        }
    }

    pub(crate) fn drain_pending_recall_effects(&self) -> Result<()> {
        let pending = {
            let mut guard = self.pending_recall.lock();
            if guard.strengthen.is_empty()
                && guard.co_retrieval_pairs.is_empty()
                && guard.proto_windows.is_empty()
            {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };

        for memory_id in pending.strengthen {
            // Recall no longer inflates strength (S0: the uniform +0.01 tick is
            // rank-inert noise). Touch-only keeps recency/spacing signals intact.
            let _ = self.update_state(memory_id, None, None, None, true, None);
            let mut states = self.states.write();
            if let Some(st) = states.get_mut(&memory_id) {
                st.recompute_spacing_quality();
                let strength = st.strength;
                drop(states);
                self.cortical_idx
                    .write()
                    .update_strength(memory_id, strength);
            }
        }

        // Gate B inflow floor: add_assoc_edge merges by MAX, so a pair seen once
        // per drain window stays frozen at 0.05 forever — that class is 21M of
        // the 28.6M CoRetrieved edges. Requiring >=0.1 (two co-occurrences in
        // one window) stops the noise band at the source; repeat singles on an
        // existing edge were max-merge no-ops anyway (minus the WAL append).
        let floor = if self
            .plasticity_decay_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0.1
        } else {
            0.0
        };
        for ((src, dst), weight) in pending.co_retrieval_pairs {
            if weight < floor {
                continue;
            }
            let _ = self.add_assoc_edge(src, dst, EdgeType::CoRetrieved, weight);
        }

        if !pending.proto_windows.is_empty() {
            let mut cortical_idx = self.cortical_idx.write();
            for window in pending.proto_windows {
                cortical_idx.strengthen_proto_transitions(&window);
            }
        }

        Ok(())
    }

    /// Semantic recall: find k most similar memories to a query embedding.
    ///
    /// Applies realm filter and strength-weighted final scoring:
    ///   `score = semantic_score × (0.5 + 0.5 × effective_strength) × confidence`
    ///
    /// Uses the ANN semantic index directly, with optional realm filtering.
    /// Recall-side maintenance effects are deferred until flush/snapshot.
    pub fn recall_semantic(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_semantic_ctx(query_embedding, k, realm, None, None, true)
    }

    /// Measurement recall: identical ranking, but skips co-retrieval
    /// strengthening so an eval/diagnostic never mutates the store it reads.
    pub fn recall_semantic_measure(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_semantic_ctx(query_embedding, k, realm, None, None, false)
    }

    /// Lane-0 atom bridge as a first-class retrieval leg. Given the recall's
    /// dense-lead anchor, return its saturating-IDF co-atom partners (id, weight)
    /// so a caller (the C++ multi-lane RRF in tool_recall) can fuse them ALONGSIDE
    /// dense/keyword/HDC — a keyword-absent partner then competes on the strength
    /// of a rare shared atom instead of being drowned as a single-dense-lane hit.
    /// Weight is the BM25 IDF of the rarest shared atom (already summed across
    /// shared atoms by `bridge_candidates_saturating`). Silent (empty) when the
    /// bridge is disabled, the anchor carries no atom, or nothing co-occurs.
    pub fn bridge_lane_scored(
        &self,
        anchor: MemoryId,
        realm: Option<&str>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        // NOT gated by bridge_lane_enabled (that flag governs the in-Rust
        // dense-lane tau bridge): this leg is a distinct C++ lane-0 feature,
        // gated caller-side by CHITTA_BRIDGE_LANE0 so a single flag flips it.
        let anchor_paths = self.artifact_idx.read().paths_for_memory(anchor);
        if anchor_paths.is_empty() {
            return Vec::new();
        }
        let n_total = self
            .memory_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let cands = self.artifact_idx.read().bridge_candidates_saturating(
            &anchor_paths,
            n_total,
            anchor,
            k,
        );
        let payloads = self.payloads.read();
        cands
            .into_iter()
            .filter(|(m, _)| {
                match payloads.get(m) {
                    // Mirror build_hit's soul: bleed gate: a soul-realm memory only
                    // surfaces into a soul-realm query.
                    Some(p) => {
                        !(p.realm.starts_with("soul:")
                            && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true))
                    }
                    None => false,
                }
            })
            .collect()
    }

    /// Semantic recall with affective context.
    ///
    /// `query_valence` / `query_arousal`: caller's current affect state.
    /// Enables mood-congruent recall (Bower 1981) and frustration-escalation
    /// detection (boost corrections when caller is frustrated).
    pub fn recall_semantic_ctx(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
        query_valence: Option<f32>,
        query_arousal: Option<f32>,
        strengthen: bool,
    ) -> Result<Vec<RecallHit>> {
        let mut profile = crate::profile::RecallProfile::new("semantic_search");
        if query_embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim {
                expected: EMBED_DIM,
                actual: query_embedding.len(),
            });
        }

        let now = now_ms();
        // Realm-filtered HNSW traversal needs a wider beam: with an allowed-set
        // the graph skips most neighbors, and at k*3 a memory that is the true
        // in-realm nearest neighbor can be unreachable (observed: rank 0 at
        // ef~1920, absent at ef~180 on a 109k store). Floor of 128 when filtered.
        let result_limit = if realm.is_some() {
            k.saturating_mul(3).max(128)
        } else {
            k.saturating_mul(3).max(k)
        };
        // realm_members guard scoped to the search: holding it (a late-order
        // lock) across the states/idx acquisitions below deadlocks against
        // put_memory, which holds states.write while taking realm_members.write.
        let semantic_hits = {
            let realm_members = self.realm_members.read();
            let allowed = realm.and_then(|r| realm_members.get(r));
            self.semantic_idx
                .read()
                .search(query_embedding, result_limit, allowed, realm)
        };

        profile.next("cw_refresh");
        // Refresh competitive_weight for each candidate using the *current* HNSW neighborhood.
        // The write-time value is stale for memories ingested when the store was sparse.
        // Two-phase to avoid holding states.write() during HNSW searches.
        let (dedup_upper, cw_refresh_interval_ms, cw_refresh_budget) = {
            let pipeline = self.scoring_pipeline.read();
            (
                pipeline.config.dedup_cosine_upper,
                pipeline.config.cw_refresh_interval_ms,
                pipeline.config.cw_refresh_max_per_query,
            )
        };
        // Phase A — find candidates that need refresh, clone embeddings.
        // Uses read locks only; the inflight check here is a cheap pre-filter,
        // the authoritative check-and-reserve happens under the write guard below.
        let candidates: Vec<(MemoryId, Vec<f32>)> = {
            // Lock order: states before semantic_idx (struct order) — the
            // inverse deadlocked against sync_foreign in production.
            let states_r = self.states.read();
            let idx = self.semantic_idx.read();
            let inflight_r = self.cw_refresh_inflight.read();
            semantic_hits.iter().filter_map(|hit| {
                // Skip if refreshed recently by this or another session.
                if let Some(st) = states_r.get(&hit.memory_id) {
                    if now - st.last_cw_refresh_ms < cw_refresh_interval_ms { return None; }
                }
                if let Some(&ts) = inflight_r.get(&hit.memory_id) {
                    if now - ts < cw_refresh_interval_ms { return None; }
                }
                // Clone embedding so we can drop all locks before searching.
                let emb = idx.get_embedding(hit.memory_id)?.to_vec();
                Some((hit.memory_id, emb))
            })
            // Budget: each refresh is a full ANN/flat search; the rest are
            // picked up by later queries (amortized refresh).
            .take(cw_refresh_budget)
            .collect()
        };
        // Atomically re-check and reserve under a single write guard: concurrent
        // sessions can all pass the read-lock pre-filter above, but only one wins
        // each slot here.
        let candidates: Vec<(MemoryId, Vec<f32>)> = if candidates.is_empty() {
            candidates
        } else {
            let mut inflight_w = self.cw_refresh_inflight.write();
            // Evict expired reservations (sessions that died mid-search) on every
            // pass, not only on rounds that produce updates.
            inflight_w.retain(|_, ts| now - *ts < cw_refresh_interval_ms);
            candidates
                .into_iter()
                .filter(|(id, _)| {
                    if inflight_w.contains_key(id) { return false; }
                    inflight_w.insert(*id, now);
                    true
                })
                .collect()
        };
        // Snapshot the prepared index under a brief read, then release the guard
        // before the independent neighborhood probes. Scalar/HNSW/dirty cases
        // retain the original path and its scoring semantics.
        let refresh_plan = self.semantic_idx.read().plan_refresh_searches(&candidates, realm);
        let neighborhoods = if let Some(plan) = refresh_plan {
            plan.search_all(&candidates)
        } else {
            let idx = self.semantic_idx.read();
            candidates.iter().map(|(_, emb)| idx.search(emb, 9, None, realm)).collect()
        };
        let cw_updates: Vec<(MemoryId, f32)> = candidates.iter().zip(neighborhoods.iter())
            .filter_map(|((memory_id, _), neighbors)| {
                if neighbors.len() <= 1 { return None; }
                let mut cos_sum = 0.0f32;
                let mut n = 0u32;
                for nb in neighbors {
                    if nb.memory_id == *memory_id { continue; }
                    if nb.cosine_similarity >= dedup_upper { continue; }
                    cos_sum += nb.cosine_similarity;
                    n += 1;
                }
                if n > 0 { Some((*memory_id, cos_sum / n as f32)) } else { None }
            }).collect();
        // Phase B — apply under brief states.write(), then release reservations.
        // Every searched candidate is marked refreshed even when its neighborhood
        // produced no update, so isolated memories aren't re-searched on every
        // recall; reservations are always released so empty rounds don't leak.
        if !candidates.is_empty() {
            let cw_by_id: std::collections::HashMap<MemoryId, f32> =
                cw_updates.into_iter().collect();
            {
                let mut states_w = self.states.write();
                for (memory_id, _) in &candidates {
                    if let Some(st) = states_w.get_mut(memory_id) {
                        if let Some(&cw) = cw_by_id.get(memory_id) {
                            st.competitive_weight = cw;
                        }
                        st.last_cw_refresh_ms = now;
                    }
                }
            }
            let mut inflight_w = self.cw_refresh_inflight.write();
            for (id, _) in &candidates {
                inflight_w.remove(id);
            }
        }

        profile.next("semantic_score_format");
        let payloads = self.payloads.read();
        let states = self.states.read();
        let learners = self.learners.read();
        let pipeline = self.scoring_pipeline.read();
        let ack_scores = self.ack_scores.read();
        let recall_prov = self.recall_provenance.read();
        let mut util_rng = utility_rng(now);

        let mut hits: Vec<RecallHit> = semantic_hits
            .into_iter()
            .filter_map(|hit| {
                let memory_id = hit.memory_id;
                let state = states.get(&memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&memory_id)?;
                let content_str = String::from_utf8(payload.content.clone()).unwrap_or_default();
                if content_str.trim().is_empty() {
                    return None;
                }
                // soul:* realms are internal — exclude from unscoped queries.
                if payload.realm.starts_with("soul:") && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true) {
                    return None;
                }
                let ctx = ScoringContext {
                    relevance_score: hit.cosine_similarity,
                    recall_mode: RecallMode::Semantic,
                    state,
                    kind: &payload.kind,
                    realm: &payload.realm,
                    realm_reliability: learners.domain_reliability.reliability(&payload.realm),
                    now_ms: now,
                    query_valence,
                    query_arousal,
                    prediction_prob: None,
                    surprise_role: None,
                    has_open_debt: false,
                    integration_weight: None,
                    ack_score: ack_scores.get(&memory_id).copied().unwrap_or(0),
                    max_query_idf: 0.0,
                };
                let (score, decomp) = pipeline.score(&ctx)?;
                // Cross-context generality (THEORY.md §6): recall from N
                // distinct daemons is evidence the memory generalizes beyond
                // one session/node. Multiplicative, config-gated boost.
                let score = {
                    let w = pipeline.config.cross_context_weight;
                    if w > 0.0 {
                        let distinct = recall_prov
                            .get(&memory_id)
                            .map(|s| s.len())
                            .unwrap_or(0) as f32;
                        score
                            * (1.0
                                + w * (distinct - 1.0)
                                    .max(0.0)
                                    .min(pipeline.config.cross_context_max))
                    } else {
                        score
                    }
                };
                // Outcome utility (flag-gated): did recalling this memory
                // actually help? Multiplier is exactly 1.0 when disabled.
                let score = score * utility_multiplier(state, &mut util_rng);
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id,
                    score,
                    semantic_score: hit.cosine_similarity,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: content_str,
                    semantic_weight: decomp.semantic_weight,
                    status_mul: decomp.status_mul,
                    epistemic_mul: decomp.epistemic_mul,
                    strength_factor: decomp.strength_factor,
                    affect_valence: state.affect_valence,
                    affect_arousal: state.affect_arousal,
                    actr_activation: decomp.actr_activation,
                    surprise_boost: decomp.surprise_boost,
                    arousal_boost: decomp.arousal_boost,
                    mood_congruence: decomp.mood_congruence,
                    frustration_boost: decomp.frustration_boost,
                    interference_factor: decomp.interference_factor,
                    spacing_boost: decomp.spacing_boost,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id))
        });

        // Lure detection (Price of Meaning no-escape theorem):
        // Suppress high-lure-risk candidates that could be false recalls.
        // Only suppress from the tail — never remove the top-scoring hit.
        let lure_threshold = pipeline.config.lure_risk_threshold;
        let max_suppressed = pipeline.config.lure_max_suppressed;
        if max_suppressed > 0 && hits.len() > 1 {
            let mut suppressed = 0usize;
            let mut i = hits.len();
            while i > 1 && suppressed < max_suppressed {
                i -= 1;
                if states.get(&hits[i].memory_id)
                    .map(|s| s.lure_risk >= lure_threshold)
                    .unwrap_or(false)
                {
                    hits.remove(i);
                    suppressed += 1;
                }
            }
        }

        // Factual-bridge recall leg (df-gated IDF atom-postings join). Reserves a
        // few top-k slots for co-atom neighbours of the top dense hit that share a
        // rare identity atom (df <= tau) — the multi-hop partner the dense pool
        // buried. ZERO cosine in the ranking; A's atoms are used only as the join
        // KEY (non-circular). Silent when the anchor carries no gated atom, so the
        // dense result is untouched on the ~96% of queries that can't fire.
        if self.bridge_lane_enabled.load(std::sync::atomic::Ordering::Relaxed)
            && !hits.is_empty()
            && k > 0
        {
            let anchor = hits[0].memory_id;
            let anchor_paths = self.artifact_idx.read().paths_for_memory(anchor);
            if !anchor_paths.is_empty() {
                let n_total = self
                    .memory_count
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .max(1);
                let tau: usize = std::env::var("CHITTA_BRIDGE_TAU")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(32);
                let reserve: usize = std::env::var("CHITTA_BRIDGE_RESERVE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4)
                    .min(k);
                // Lane-0 (CHITTA_BRIDGE_LANE0): saturating-IDF bridge fused as a
                // scored RRF peer instead of the positional reserve injection.
                let lane0 = std::env::var("CHITTA_BRIDGE_LANE0")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let cands = if lane0 {
                    self.artifact_idx.read().bridge_candidates_saturating(
                        &anchor_paths,
                        n_total,
                        anchor,
                        k.saturating_mul(4).max(64),
                    )
                } else {
                    self.artifact_idx.read().bridge_candidates(
                        &anchor_paths,
                        n_total,
                        tau,
                        anchor,
                        reserve * 4,
                    )
                };
                if !cands.is_empty() {
                    // Score a bridge-injected memory into a well-formed RecallHit.
                    // semantic_score = 0.0: it was surfaced by the postings join, not
                    // cosine. Final ordering is positional (reserved slots), so the
                    // low relevance score never re-sorts it above the dense head.
                    let build_hit = |memory_id: MemoryId| -> Option<RecallHit> {
                        let state = states.get(&memory_id)?;
                        if state.deleted {
                            return None;
                        }
                        let payload = payloads.get(&memory_id)?;
                        let content_str =
                            String::from_utf8(payload.content.clone()).unwrap_or_default();
                        if content_str.trim().is_empty() {
                            return None;
                        }
                        if payload.realm.starts_with("soul:")
                            && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true)
                        {
                            return None;
                        }
                        let ctx = ScoringContext {
                            relevance_score: 0.0,
                            recall_mode: RecallMode::Semantic,
                            state,
                            kind: &payload.kind,
                            realm: &payload.realm,
                            realm_reliability: learners
                                .domain_reliability
                                .reliability(&payload.realm),
                            now_ms: now,
                            query_valence,
                            query_arousal,
                            prediction_prob: None,
                            surprise_role: None,
                            has_open_debt: false,
                            integration_weight: None,
                            ack_score: ack_scores.get(&memory_id).copied().unwrap_or(0),
                            max_query_idf: 0.0,
                        };
                        let (score, decomp) = pipeline.score(&ctx)?;
                        let eff_strength = state.effective_strength(now);
                        Some(RecallHit {
                            memory_id,
                            score,
                            semantic_score: 0.0,
                            ts_ms: payload.authored_at_ms,
                            kind: payload.kind.clone(),
                            realm: payload.realm.clone(),
                            strength: eff_strength,
                            confidence: state.confidence,
                            access_count: state.access_count,
                            content: content_str,
                            semantic_weight: decomp.semantic_weight,
                            status_mul: decomp.status_mul,
                            epistemic_mul: decomp.epistemic_mul,
                            strength_factor: decomp.strength_factor,
                            affect_valence: state.affect_valence,
                            affect_arousal: state.affect_arousal,
                            actr_activation: decomp.actr_activation,
                            surprise_boost: decomp.surprise_boost,
                            arousal_boost: decomp.arousal_boost,
                            mood_congruence: decomp.mood_congruence,
                            frustration_boost: decomp.frustration_boost,
                            interference_factor: decomp.interference_factor,
                            spacing_boost: decomp.spacing_boost,
                        })
                    };
                    if lane0 {
                        // Scored RRF peer: reciprocal-rank-fuse every saturating-IDF
                        // co-atom candidate against the dense/keyword result so a
                        // strong bridge competes on merit, not a reserved slot.
                        let bridge_hits: Vec<RecallHit> =
                            cands.iter().filter_map(|(m, _)| build_hit(*m)).collect();
                        if !bridge_hits.is_empty() {
                            let rrf_k = pipeline.config.rrf_k;
                            hits = rrf_merge(std::mem::take(&mut hits), bridge_hits, k, rrf_k);
                        }
                    } else {
                    // No-evict fusion. The dense top-k is the base. We only inject
                    // GENUINELY-NEW bridges (co-atom partners the dense pool buried
                    // past k) and pay for each by evicting a *weak* dense-tail hit —
                    // never a dense hit that is itself a co-atom bridge (that hit is
                    // reinforced by both signals; evicting it is the pure loss the
                    // fixed-reserve version caused, e.g. a gold B already at dense
                    // rank 7). Deduped: a bridge already in the dense top-k costs no
                    // slot. Bounded by `reserve`.
                    let cand_ids: std::collections::HashSet<MemoryId> =
                        cands.iter().map(|(m, _)| *m).collect();
                    let dense_topk: std::collections::HashSet<MemoryId> =
                        hits.iter().take(k).map(|h| h.memory_id).collect();
                    let mut new_bridge: Vec<RecallHit> = Vec::new();
                    for (mid, _w) in &cands {
                        if new_bridge.len() >= reserve {
                            break;
                        }
                        if dense_topk.contains(mid) {
                            continue; // already shown by dense — no slot needed
                        }
                        if let Some(h) = build_hit(*mid) {
                            new_bridge.push(h);
                        }
                    }
                    if !new_bridge.is_empty() {
                        let mut kept: Vec<RecallHit> = hits.iter().take(k).cloned().collect();
                        let need = new_bridge.len();
                        let mut room = 0usize;
                        // Pass 1: evict weakest dense-tail hits that are NOT co-atom
                        // bridges (protects a present B, which shares the anchor atom).
                        let mut i = kept.len();
                        while room < need && i > 0 {
                            i -= 1;
                            if !cand_ids.contains(&kept[i].memory_id) {
                                kept.remove(i);
                                room += 1;
                            }
                        }
                        // Pass 2: if the whole tail was co-atom bridges, drop plain
                        // tail down to the reserve floor so a truly-new bridge can enter.
                        while room < need && kept.len() > k.saturating_sub(reserve) {
                            kept.pop();
                            room += 1;
                        }
                        kept.extend(new_bridge.into_iter().take(room));
                        hits = kept;
                    }
                    }
                }
            }
        }

        hits.truncate(k);

        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        drop(pipeline);
        drop(learners);
        drop(ack_scores);
        if strengthen {
            self.enqueue_recall_effects(&hit_ids);
        }

        Ok(hits)
    }

    /// SOTA HippoRAG-style Personalized-PageRank lane over the association graph.
    ///
    /// Seeds are the query's top fused hits, weighted by their fused score. A
    /// single PPR pass (power iteration with restart) spreads that mass over a
    /// lazily-expanded neighborhood, ranking graph-reachable memories the
    /// lexical/semantic lanes never surfaced — the multi-hop bridge that shares
    /// no query terms with the question. Returns the top-`top_g` stationary
    /// nodes EXCLUDING the seeds (the injection); seeds are already ranked.
    ///
    /// Edge transition weight = type_prior * stored_weight, row-normalized per
    /// source. CoRetrieved (the weak Hebbian lane) is down-weighted vs the
    /// strong DerivedFrom provenance lane. All knobs are env-gated. The frontier
    /// is capped so the walk stays local and fast — it never materializes the
    /// whole node graph, only the seeds' bounded neighborhood.
    pub fn ppr_lane(
        &self,
        seed_ids: &[MemoryId],
        seed_weights: &[f32],
        top_g: usize,
    ) -> Vec<(MemoryId, f32)> {
        if seed_ids.is_empty() || top_g == 0 {
            return Vec::new();
        }
        let get_f = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let get_u = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let alpha = get_f("CHITTA_PPR_ALPHA", 0.15).clamp(0.01, 0.99);
        let w_derived = get_f("CHITTA_PPR_W_DERIVED", 1.0);
        let w_sameartifact = get_f("CHITTA_PPR_W_SAMEARTIFACT", 0.8);
        let w_samesession = get_f("CHITTA_PPR_W_SAMESESSION", 0.6);
        let w_coretrieved = get_f("CHITTA_PPR_W_CORETRIEVED", 0.35);
        let w_semneighbor = get_f("CHITTA_PPR_W_SEMNEIGHBOR", 0.7);
        let frontier_cap = get_u("CHITTA_PPR_FRONTIER_CAP", 4000).max(seed_ids.len());
        let max_iters = get_u("CHITTA_PPR_MAX_ITERS", 25).max(1);
        // Per-node out-degree cap: a few high-degree hubs (one memory here has
        // >6k assoc edges) make the in-set adjacency dense, so the power
        // iteration degrades to O(iters * n^2) — the measured 35s recall stall.
        // Keep only each node's top-K edges by transition weight; a hub's mass is
        // diffuse and the tail edges contribute negligibly to the stationary rank.
        let max_degree = get_u("CHITTA_PPR_MAX_DEGREE", 64).max(1);
        const HOP_BUDGET: usize = 4;

        let type_prior = |et: &EdgeType| -> f32 {
            match et {
                EdgeType::DerivedFrom => w_derived,
                EdgeType::SameArtifact => w_sameartifact,
                EdgeType::SameSession => w_samesession,
                EdgeType::CoRetrieved => w_coretrieved,
                EdgeType::Supports => 0.4,
                EdgeType::Contradicts => 0.3,
                EdgeType::SemanticNeighbor => w_semneighbor,
            }
        };

        let pipeline_config = self.scoring_pipeline.read().config.clone();
        let states = self.states.read();
        let assoc_edges = self.assoc_edges.read();

        let legal = |id: &MemoryId| -> bool {
            match states.get(id) {
                None => false,
                Some(s) if s.deleted => false,
                Some(s) => {
                    crate::scoring::status_multiplier(&s.status, &pipeline_config).is_some()
                }
            }
        };

        // Lazy BFS: build the local node set (seeds first) bounded by a small hop
        // budget and a hard frontier cap. Nearest-first exploration keeps the walk
        // concentrated where PPR mass actually lands.
        let mut index: std::collections::HashMap<MemoryId, usize> =
            std::collections::HashMap::new();
        let mut nodes: Vec<MemoryId> = Vec::new();
        for &s in seed_ids {
            index.entry(s).or_insert_with(|| {
                let i = nodes.len();
                nodes.push(s);
                i
            });
        }
        let seed_n = nodes.len();
        let mut queue: std::collections::VecDeque<(MemoryId, usize)> =
            nodes.iter().map(|&id| (id, 0usize)).collect();
        while let Some((node, hop)) = queue.pop_front() {
            if hop >= HOP_BUDGET || nodes.len() >= frontier_cap {
                continue;
            }
            let neighbors = match assoc_edges.get(&node) {
                Some(v) => v,
                None => continue,
            };
            // Cap fan-out to the top-`max_degree` neighbors by transition weight so
            // a single hub cannot flood the frontier with its low-weight tail.
            let mut cand: Vec<(f32, MemoryId)> = neighbors
                .iter()
                .map(|e| (type_prior(&e.edge_type) * e.weight.max(0.0), e.dst))
                .filter(|&(w, _)| w > 0.0)
                .collect();
            if cand.len() > max_degree {
                cand.select_nth_unstable_by(max_degree, |a, b| {
                    b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
                });
                cand.truncate(max_degree);
            }
            cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_, dst) in cand {
                if index.contains_key(&dst) || !legal(&dst) {
                    continue;
                }
                if nodes.len() >= frontier_cap {
                    break;
                }
                index.insert(dst, nodes.len());
                nodes.push(dst);
                queue.push_back((dst, hop + 1));
            }
        }

        let n = nodes.len();
        // Row-normalized transition rows restricted to the explored set.
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (i, &src) in nodes.iter().enumerate() {
            if let Some(edges) = assoc_edges.get(&src) {
                let mut row: Vec<(usize, f32)> = Vec::new();
                for edge in edges {
                    if let Some(&j) = index.get(&edge.dst) {
                        let w = type_prior(&edge.edge_type) * edge.weight.max(0.0);
                        if w > 0.0 {
                            row.push((j, w));
                        }
                    }
                }
                // Bound the row to top-`max_degree` by weight: this is what keeps
                // the power iteration O(iters * n * max_degree) instead of O(n^2).
                if row.len() > max_degree {
                    row.select_nth_unstable_by(max_degree, |a, b| {
                        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                    });
                    row.truncate(max_degree);
                }
                row.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
                let sum: f32 = row.iter().map(|e| e.1).sum();
                if sum > 0.0 {
                    for e in &mut row {
                        e.1 /= sum;
                    }
                    adj[i] = row;
                }
            }
        }
        drop(assoc_edges);
        drop(states);

        // Restart distribution: seed_weights normalized over the seeds, 0 else.
        let mut restart = vec![0.0f32; n];
        let mut wsum = 0.0f32;
        for (k, &s) in seed_ids.iter().enumerate() {
            if let Some(&i) = index.get(&s) {
                let w = seed_weights.get(k).copied().unwrap_or(1.0).max(0.0);
                restart[i] += w;
                wsum += w;
            }
        }
        if wsum <= 0.0 {
            let u = 1.0 / seed_n as f32;
            for r in restart.iter_mut().take(seed_n) {
                *r = u;
            }
        } else {
            for r in &mut restart {
                *r /= wsum;
            }
        }

        // Power iteration: p_{t+1} = (1-alpha) Pᵀ p_t + alpha r. Dangling mass
        // (nodes with no in-set out-edges) is redistributed through the restart
        // vector so p stays a proper distribution and the walk converges.
        let mut p = restart.clone();
        let mut next = vec![0.0f32; n];
        for _ in 0..max_iters {
            for i in 0..n {
                next[i] = alpha * restart[i];
            }
            let mut dangling = 0.0f32;
            for i in 0..n {
                if adj[i].is_empty() {
                    dangling += p[i];
                    continue;
                }
                let out = (1.0 - alpha) * p[i];
                for &(j, w) in &adj[i] {
                    next[j] += out * w;
                }
            }
            if dangling > 0.0 {
                let spread = (1.0 - alpha) * dangling;
                for i in 0..n {
                    next[i] += spread * restart[i];
                }
            }
            let mut delta = 0.0f32;
            for i in 0..n {
                delta += (next[i] - p[i]).abs();
            }
            std::mem::swap(&mut p, &mut next);
            if delta < 1e-4 {
                break;
            }
        }

        // Rank non-seed nodes by stationary score; return top_g.
        let mut ranked: Vec<(MemoryId, f32)> = nodes
            .iter()
            .enumerate()
            .skip(seed_n)
            .filter(|&(i, _)| p[i] > 0.0)
            .map(|(i, &id)| (id, p[i]))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_g);
        ranked
    }

    /// Gated PPR injection for the MCP hybrid-recall path. Seed the PPR lane from
    /// the current top hits and RRF-merge its stationary ranking back in,
    /// INJECTING graph-reachable memories the lexical/semantic lanes missed.
    /// Default OFF: the dev ablation showed it regresses single_hop nDCG for a
    /// noise-level multihop gain (net-negative), so it ships dormant. Enable with
    /// `CHITTA_PPR_LANE=1`. Injected nodes carry `semantic_score` 0.0, so any
    /// cosine abstain gate is unaffected. Mirrors the C++ tier-2 PPR lane in
    /// field_memory_recall.cpp.
    fn ppr_inject(&self, hits: &mut Vec<RecallHit>) {
        if std::env::var("CHITTA_PPR_LANE").ok().as_deref() != Some("1") {
            return;
        }
        if hits.len() < 2 {
            return;
        }
        let get_u = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let get_f = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let seeds_n = get_u("CHITTA_PPR_SEEDS", 5);
        let top_g = get_u("CHITTA_PPR_G", 15);
        let lane_w = get_f("CHITTA_PPR_LANE_W", 1.0);

        let seed_ids: Vec<MemoryId> = hits.iter().take(seeds_n).map(|h| h.memory_id).collect();
        let seed_w: Vec<f32> = hits.iter().take(seeds_n).map(|h| h.score.max(1e-6)).collect();
        let lane = self.ppr_lane(&seed_ids, &seed_w, top_g);
        if lane.is_empty() {
            return;
        }

        const RRF_K: f32 = 60.0;
        let mut fused: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();
        let mut by_id: std::collections::HashMap<MemoryId, RecallHit> =
            std::collections::HashMap::new();
        for (rank, h) in hits.iter().enumerate() {
            *fused.entry(h.memory_id).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f32);
            by_id.entry(h.memory_id).or_insert_with(|| h.clone());
        }

        let payloads = self.payloads.read();
        let states = self.states.read();
        let pipeline_config = self.scoring_pipeline.read().config.clone();
        let now = now_ms();
        for (rank, (mid, _)) in lane.iter().enumerate() {
            let contrib = lane_w * (1.0 / (RRF_K + (rank + 1) as f32));
            if by_id.contains_key(mid) {
                *fused.entry(*mid).or_insert(0.0) += contrib;
                continue;
            }
            // Hydrate an injected node into a full RecallHit (graph-derived: no
            // semantic score/weight, mirrors expand_associations' construction).
            let state = match states.get(mid) {
                Some(s) if !s.deleted => s,
                _ => continue,
            };
            let status_mul =
                match crate::scoring::status_multiplier(&state.status, &pipeline_config) {
                    Some(m) => m,
                    None => continue,
                };
            let payload = match payloads.get(mid) {
                Some(p) => p,
                None => continue,
            };
            *fused.entry(*mid).or_insert(0.0) += contrib;
            let eff_strength = state.effective_strength(now);
            by_id.insert(
                *mid,
                RecallHit {
                    memory_id: *mid,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: payload.created_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                },
            );
        }
        drop(payloads);
        drop(states);

        let mut order: Vec<(MemoryId, f32)> = fused
            .into_iter()
            .filter(|(id, _)| by_id.contains_key(id))
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut merged = Vec::with_capacity(order.len());
        for (mid, s) in order {
            if let Some(mut h) = by_id.remove(&mid) {
                h.score = s;
                merged.push(h);
            }
        }
        *hits = merged;
    }

    /// Expand from seed memory IDs via typed association edges (max 2 hops).
    ///
    /// Returns memories discovered via the association graph, scored by
    /// spreading activation with hop decay (×0.55 per hop).
    ///
    /// Edge type priors:
    ///   DerivedFrom=1.0, SameArtifact=0.8, SameSession=0.6, CoRetrieved=0.5,
    ///   Supports=0.4, Contradicts=0.3
    pub fn expand_associations(
        &self,
        seed_ids: &[MemoryId],
        max_hops: usize,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        const HOP_DECAY: f32 = 0.55;
        const FANOUT_CAP: usize = 16;
        let max_hops = max_hops.min(2);

        let edge_prior = |et: &EdgeType| -> f32 {
            match et {
                EdgeType::DerivedFrom => 1.0,
                EdgeType::SameArtifact => 0.8,
                EdgeType::SameSession => 0.6,
                EdgeType::CoRetrieved => 0.5,
                EdgeType::Supports => 0.4,
                EdgeType::Contradicts => 0.3,
                EdgeType::SemanticNeighbor => 0.7,
            }
        };

        let seed_set: HashSet<MemoryId> = seed_ids.iter().copied().collect();
        let pipeline_config = self.scoring_pipeline.read().config.clone();
        // activation accumulator: memory_id -> max activation score seen
        let mut activation: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        // frontier: (memory_id, activation_score, hops_remaining)
        let mut frontier: Vec<(MemoryId, f32, usize)> =
            seed_ids.iter().map(|&id| (id, 1.0, max_hops)).collect();

        // Lock order: payloads → states → assoc_edges (struct order; matches
        // sync_foreign's write-guard acquisition). payloads is needed only
        // after the walk but must be acquired first to keep the global order.
        let payloads = self.payloads.read();
        let states = self.states.read();
        let assoc_edges = self.assoc_edges.read();

        while let Some((node, act, hops_left)) = frontier.pop() {
            if hops_left == 0 {
                continue;
            }
            let neighbors = match assoc_edges.get(&node) {
                Some(v) => v,
                None => continue,
            };
            for edge in neighbors.iter().take(FANOUT_CAP) {
                let dst = edge.dst;
                // Skip deleted or status-suppressed memories so assoc expansion
                // honours the same suppression as semantic recall.
                match states.get(&dst) {
                    None => continue,
                    Some(s) if s.deleted => continue,
                    Some(s) if crate::scoring::status_multiplier(&s.status, &pipeline_config).is_none() => continue,
                    Some(_) => {}
                }
                let edge_act = act * HOP_DECAY * edge_prior(&edge.edge_type) * edge.weight;
                let entry = activation.entry(dst).or_insert(0.0);
                if edge_act > *entry {
                    *entry = edge_act;
                    // Only continue expanding if this is a new or improved path.
                    if hops_left > 1 {
                        frontier.push((dst, edge_act, hops_left - 1));
                    }
                }
            }
        }

        drop(assoc_edges);

        let now = now_ms();

        let mut hits: Vec<RecallHit> = activation
            .into_iter()
            .filter(|(id, _)| !seed_set.contains(id))
            .filter_map(|(id, act_score)| {
                let state = states.get(&id)?;
                if state.deleted { return None; }
                // Belt-and-braces: frontier filtered most suppressed memories, but
                // edges added after a status flip can still land in activation.
                let status_mul = crate::scoring::status_multiplier(&state.status, &pipeline_config)?;
                let payload = payloads.get(&id)?;
                let eff_strength = state.effective_strength(now);
                let score = act_score * eff_strength * status_mul;
                Some(RecallHit {
                    memory_id: id,
                    score,
                    semantic_score: 0.0,
                    ts_ms: payload.created_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id))
        });
        hits.truncate(limit);

        Ok(hits)
    }

    /// Register a file artifact, returning its ArtifactId (idempotent by path).
    pub fn upsert_artifact(
        &self,
        normalized_path: &str,
        repo_root: Option<String>,
    ) -> Result<ArtifactId> {
        // Fast path: already exists.
        if let Some(&id) = self.artifacts.read().get(normalized_path) {
            return Ok(id);
        }

        let artifact_id = self.artifact_id_alloc.next_id();
        let op = Op::UpsertArtifact(UpsertArtifactOp {
            artifact_id,
            normalized_path: normalized_path.to_string(),
            repo_root,
        });
        let _seqno = self.log.write().append(&op)?;

        // Double-checked insert: another thread may have raced past the read guard.
        let id = {
            let mut artifacts = self.artifacts.write();
            *artifacts
                .entry(normalized_path.to_string())
                .or_insert(artifact_id)
        };
        // Keep reverse map in sync.
        self.artifact_paths
            .write()
            .entry(id)
            .or_insert_with(|| normalized_path.to_string());

        Ok(id)
    }

    /// Derive artifact refs from content by extracting identity atoms and
    /// upserting each as an artifact. Used at write time when the caller supplies
    /// none. Non-fatal: a failed upsert just drops that atom.
    fn derive_artifact_refs(&self, content: &[u8]) -> Vec<ArtifactRef> {
        let text = String::from_utf8_lossy(content);
        let mut refs = Vec::new();
        for path in crate::organ::artifact::extract_artifact_paths(&text) {
            if let Ok(artifact_id) = self.upsert_artifact(&path, None) {
                refs.push(ArtifactRef {
                    artifact_id,
                    relation: crate::ops::ArtifactRelation::Mentioned,
                    line_start: 0,
                    line_end: 0,
                });
            }
        }
        refs
    }

    /// One-shot bootstrap: scan every stored payload, extract identity atoms, and
    /// populate the artifact index (upsert + associate) for memories written
    /// before write-time derivation existed. In-memory associations persist via
    /// the next snapshot; upserts are WAL-logged. Returns (memories, associations).
    pub fn backfill_artifact_refs(&self) -> (u64, u64) {
        let ids: Vec<MemoryId> = self.payloads.read().keys().copied().collect();
        let mut mems = 0u64;
        let mut assoc = 0u64;
        for mid in ids {
            let content = match self.payloads.read().get(&mid) {
                Some(p) => String::from_utf8_lossy(&p.content).into_owned(),
                None => continue,
            };
            let paths = crate::organ::artifact::extract_artifact_paths(&content);
            if paths.is_empty() {
                continue;
            }
            let mut any = false;
            for path in paths {
                if let Ok(artifact_id) = self.upsert_artifact(&path, None) {
                    self.artifact_idx.write().associate(mid, artifact_id, &path, 1.0);
                    assoc += 1;
                    any = true;
                }
            }
            if any {
                mems += 1;
            }
        }
        (mems, assoc)
    }

    /// Recall memories within a time range [start_ms, end_ms].
    pub fn recall_temporal(
        &self,
        start_ms: i64,
        end_ms: i64,
        realm: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        let entries = self
            .time_idx
            .read()
            .range_query(start_ms, end_ms, realm, limit);
        let now = now_ms();
        let payloads = self.payloads.read();
        let states = self.states.read();

        let hits = entries
            .into_iter()
            .filter_map(|entry| {
                let state = states.get(&entry.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&entry.memory_id)?;
                if payload.content.is_empty() {
                    return None;
                }
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: entry.memory_id,
                    score: eff_strength * state.confidence,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Keyword (BM25) recall.
    pub fn recall_keyword(&self, query: &str, k: usize) -> Result<Vec<RecallHit>> {
        self.recall_keyword_ctx(query, k, None, None, None, true)
    }
    /// Realm-scoped keyword recall: drops hits outside `realm` (None = unscoped) so the BM25
    /// lane honours --realm and never bleeds other projects' memories into a scoped query.
    pub fn recall_keyword_realm(
        &self,
        query: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_keyword_ctx(query, k, None, None, realm, true)
    }

    /// Measurement keyword recall: skips co-retrieval strengthening.
    pub fn recall_keyword_measure(
        &self,
        query: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_keyword_ctx(query, k, None, None, realm, false)
    }

    /// HDC recall: O(n) Hamming-distance search over binary hypervectors.
    /// Returns hits ordered by ascending Hamming distance (smaller = more similar).
    /// Converts to `RecallHit` with `semantic_score = 1 - hamming/8192`.
    pub fn recall_hdc(&self, query: &str, k: usize, realm: Option<&str>) -> Result<Vec<RecallHit>> {
        let hdc_hits = self.hdc_idx.read().query(query, k * 2, realm);
        if hdc_hits.is_empty() {
            return Ok(vec![]);
        }
        let payloads = self.payloads.read();
        let states   = self.states.read();
        let mut hits = Vec::with_capacity(hdc_hits.len());
        for (id, hamming_dist) in hdc_hits {
            let Some(payload) = payloads.get(&id) else { continue };
            let Some(state)   = states.get(&id)   else { continue };
            if state.deleted { continue; }
            let sim = 1.0 - hamming_dist as f32 / (128 * 64) as f32;
            hits.push(RecallHit {
                memory_id:          id,
                score:              sim,
                semantic_score:     sim,
                ts_ms:              payload.authored_at_ms,
                kind:               payload.kind.clone(),
                realm:              payload.realm.clone(),
                strength:           state.strength,
                confidence:         state.confidence,
                access_count:       state.access_count,
                content:            std::str::from_utf8(&payload.content)
                                        .unwrap_or("").to_string(),
                semantic_weight:    1.0,
                status_mul:         1.0,
                epistemic_mul:      1.0,
                strength_factor:    state.strength,
                affect_valence:     state.affect_valence,
                affect_arousal:     state.affect_arousal,
                actr_activation:    0.0,
                surprise_boost:     1.0,
                arousal_boost:      1.0,
                mood_congruence:    1.0,
                frustration_boost:  1.0,
                interference_factor: 1.0,
                spacing_boost:      1.0,
            });
        }
        hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
        hits.truncate(k);
        Ok(hits)
    }

    /// Bridge query: find entities active in [start_ms, end_ms] via EventTape, then
    /// recall their memories from time_idx. Unifies the action-event plane (EventTape)
    /// with the memory-content plane (time_idx) so temporal queries work without
    /// knowing the realm in advance.
    pub fn recall_temporal_events(
        &self,
        start_ms: i64,
        end_ms: i64,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        // Collect unique entity names from EventTape events in [start_ms, end_ms]
        let active_entities: Vec<String> = {
            let tape = self.event_tape.read();
            let mut seen = std::collections::HashSet::new();
            tape.events.iter()
                .filter(|e| e.ts_ms >= start_ms && e.ts_ms <= end_ms)
                .filter_map(|e| {
                    let name = tape.entity_name(e.entity_key).to_owned();
                    if seen.insert(name.clone()) { Some(name) } else { None }
                })
                .take(32)
                .collect()
        };

        let per_entity = (limit / active_entities.len().max(1)).max(1);
        let mut all_hits: Vec<RecallHit> = Vec::new();
        for entity in &active_entities {
            if let Ok(hits) = self.recall_temporal(start_ms, end_ms, Some(entity.as_str()), per_entity) {
                all_hits.extend(hits);
            }
        }
        // Also include realm-less global window (entity = None)
        if let Ok(hits) = self.recall_temporal(start_ms, end_ms, None, limit / 4) {
            all_hits.extend(hits);
        }

        // Deduplicate by memory_id, sort recency-first
        let mut seen_ids = std::collections::HashSet::new();
        all_hits.retain(|h| seen_ids.insert(h.memory_id));
        all_hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        all_hits.truncate(limit);
        Ok(all_hits)
    }

    /// Log a structured action event to the CEC tape and extend the CDAWG.
    /// `outcome`: 0=success 1=fail 2=error 3=partial.
    /// Also pushes TD(λ) eligibility-trace credit for non-synthetic events.
    pub fn log_event(&self, tool: &str, entity: &str, outcome: u8, session_id: u64, ts_ms: i64) {
        let (sym, turn, last_n) = {
            let mut tape = self.event_tape.write();
            let s = tape.log(tool, entity, outcome, session_id, ts_ms);
            let t = tape.events.len() as u32 - 1;
            let n = tape.last_n_syms(16);
            (s, t, n)
        };
        let mut cdawg = self.cdawg.write();
        cdawg.extend(sym, turn);
        if tool != "legacy" && tool != "remember" {
            let delta = if outcome == 0 { 0.1_f32 } else { -0.2_f32 };
            cdawg.push_td_credit(&last_n, delta, 0.9);
            // Regret-shaped utility (Phase 10): base reward minus cost axes.
            // token_cost/latency_ms/retry_count default to 0 in basic log_event path.
            let base = if outcome == 0 { 1.0_f32 } else { -1.0_f32 };
            cdawg.update_q(sym, base, 0.05, 0.95);
        }
        drop(cdawg);
        self.episode_hdc.write().log_episode(tool, entity, outcome);
        // Refutation ledger: observe the (prev, curr) bigram
        if turn > 0 {
            let tape = self.event_tape.read();
            if let (Some(prev_ev), Some(curr_ev)) = (tape.events.get(turn as usize - 1), tape.events.get(turn as usize)) {
                let prev_sym = prev_ev.pack();
                let curr_sym = curr_ev.pack();
                let ts = curr_ev.ts_ms;
                drop(tape);
                let changed = self.refutation_ledger.write().observe(prev_sym, curr_sym, ts);
                for (rule_id, status) in changed {
                    if let crate::organ::refutation_ledger::RefutStatus::Refuted(_) = status {
                        let _ = self.add_triplet(
                            format!("rule_{rule_id}"), "refuted_by".into(),
                            format!("contradict_at_ts={ts}"),
                            1.0, None, None
                        );
                        // Auto-file a task so the executor pathway has a concrete work item.
                        self.task_registry.write().create(
                            format!("cec-refuted-rule-{rule_id}"),
                            "cec-refutation".into(),
                            format!("{{\"rule_id\":{rule_id},\"ts\":{ts}}}"),
                            ts, rule_id as u64,
                        );
                        // Propose an intervention policy in shadow mode.
                        use crate::organ::intervention_store::InterventionKind;
                        self.cec_policy_store.write().propose(
                            rule_id,
                            InterventionKind::TurnInjection {
                                message: format!("⚠ CEC rule_{rule_id} refuted — this pattern's predictions are no longer reliable"),
                                priority: 80,
                            },
                            ts,
                        );
                    }
                }
            }
        }
        // Note: consolidation_pass() is NOT called inline here — it runs Sequitur over
        // the full tape and does O(rules × 4) triplet writes, which would hold rpc_mutex_
        // for seconds and stall all other tools. Call it explicitly via the MCP tool instead.
    }

    /// Record an outcome (success/failure) for the most recent action on (tool, entity).
    pub fn record_action_outcome(&self, tool: &str, entity: &str, outcome: u8, success: bool) {
        let sym = self.event_tape.write().symbol_of(tool, entity, outcome);
        self.cdawg.write().record_outcome(&[sym], success);
    }

    /// Causal recall: return the last N events matching (tool, entity) as RecallHit stubs.
    /// Content field contains a human-readable description of the event sequence.
    pub fn recall_causal(&self, tool: &str, entity: &str, k: usize) -> Result<Vec<RecallHit>> {
        let sym = {
            let mut tape = self.event_tape.write();
            tape.symbol_of(tool, entity, 0) // outcome=0 as probe; CDAWG walk ignores outcome bits via partial match
        };
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();

        // Try exact match first, then fall back to tool-only by zeroing entity_key bits.
        let turns = if let Some(state) = cdawg.walk(&[sym]) {
            cdawg.collect_endpos(state)
        } else {
            Vec::new()
        };

        let mut hits: Vec<RecallHit> = turns.iter().rev().take(k).filter_map(|&t| {
            let ev = tape.events.get(t as usize)?;
            let tool_name   = tape.tool_name(ev.tool_id);
            let entity_name = tape.entity_name(ev.entity_key);
            let outcome_str = match ev.outcome_class {
                0 => "success", 1 => "fail", 2 => "error", _ => "partial"
            };
            let content = format!(
                "[turn {}] {} on {} → {} (ts: {})",
                t, tool_name, entity_name, outcome_str, ev.ts_ms
            );
            Some(RecallHit {
                memory_id:          0,
                score:              1.0 - (t as f32 / (tape.events.len() as f32 + 1.0)),
                semantic_score:     0.0,
                ts_ms:              ev.ts_ms,
                kind:               "event".to_string(),
                realm:              "cec".to_string(),
                strength:           1.0,
                confidence:         1.0,
                access_count:       0,
                content,
                semantic_weight:    0.0,
                status_mul:         1.0,
                epistemic_mul:      1.0,
                strength_factor:    1.0,
                affect_valence:     0.0,
                affect_arousal:     0.0,
                actr_activation:    0.0,
                surprise_boost:     0.0,
                arousal_boost:      0.0,
                mood_congruence:    1.0,
                frustration_boost:  0.0,
                interference_factor:1.0,
                spacing_boost:      1.0,
            })
        }).collect();

        hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        Ok(hits)
    }

    /// Return top-k failure patterns from the CDAWG as RecallHit stubs.
    pub fn recall_failure_pattern(&self, k: usize) -> Result<Vec<RecallHit>> {
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();
        let patterns = cdawg.failure_patterns(3, k);
        let hits = patterns.into_iter().filter_map(|(state_id, fail_count, ratio)| {
            let turns = cdawg.collect_endpos(state_id);
            let last_ts = turns.iter()
                .filter_map(|&t| tape.events.get(t as usize).map(|e| e.ts_ms))
                .max()
                .unwrap_or(0);
            let content = format!(
                "[failure-pattern state={}] fail_count={} fail_ratio={:.2} last_seen_turn={}",
                state_id, fail_count, ratio, turns.iter().max().copied().unwrap_or(0)
            );
            Some(RecallHit {
                memory_id:          state_id as u64,
                score:              ratio * fail_count as f32,
                semantic_score:     0.0,
                ts_ms:              last_ts,
                kind:               "failure-pattern".to_string(),
                realm:              "cec".to_string(),
                strength:           ratio,
                confidence:         ratio,
                access_count:       fail_count,
                content,
                semantic_weight:    0.0,
                status_mul:         1.0,
                epistemic_mul:      1.0,
                strength_factor:    1.0,
                affect_valence:    -ratio,
                affect_arousal:     ratio,
                actr_activation:    0.0,
                surprise_boost:     0.0,
                arousal_boost:      0.0,
                mood_congruence:    1.0 - ratio,
                frustration_boost:  ratio,
                interference_factor:1.0,
                spacing_boost:      1.0,
            })
        }).collect();
        Ok(hits)
    }

    /// PMI-ranked causal antecedents: what actions typically precede (tool, entity)?
    /// Returns RecallHit stubs ranked by pointwise mutual information.
    pub fn recall_causal_antecedent(&self, tool: &str, entity: &str, k: usize) -> Result<Vec<RecallHit>> {
        let sym = {
            let mut tape = self.event_tape.write();
            tape.symbol_of(tool, entity, 0)
        };
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();
        let antecedents = cdawg.causal_antecedents(&[sym], k, &tape);
        let hits = antecedents
            .into_iter()
            .enumerate()
            .map(|(i, (syms, count, pmi))| {
                let desc = syms
                    .iter()
                    .map(|&s| {
                        let tool_id    = (s >> 40) as u16;
                        let outcome_cl = ((s >> 32) & 0xff) as u8;
                        let entity_k   = (s & 0xffff_ffff) as u32;
                        let outcome_str = match outcome_cl {
                            0 => "success", 1 => "fail", 2 => "error", _ => "partial",
                        };
                        format!("{} on {} → {}", tape.tool_name(tool_id), tape.entity_name(entity_k), outcome_str)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let content = format!(
                    "[causal-antecedent rank={}] {} (count={} pmi={:.3})",
                    i + 1, desc, count, pmi
                );
                RecallHit {
                    memory_id:           i as u64,
                    score:               pmi.max(0.0),
                    semantic_score:      0.0,
                    ts_ms:               0,
                    kind:                "causal-antecedent".to_string(),
                    realm:               "cec".to_string(),
                    strength:            count as f32,
                    confidence:          (pmi / 10.0).clamp(0.0, 1.0),
                    access_count:        count,
                    content,
                    semantic_weight:     0.0,
                    status_mul:          1.0,
                    epistemic_mul:       1.0,
                    strength_factor:     1.0,
                    affect_valence:      0.0,
                    affect_arousal:      0.0,
                    actr_activation:     0.0,
                    surprise_boost:      0.0,
                    arousal_boost:       0.0,
                    mood_congruence:     1.0,
                    frustration_boost:   0.0,
                    interference_factor: 1.0,
                    spacing_boost:       1.0,
                }
            })
            .collect();
        Ok(hits)
    }

    /// Heteroassociative HDC recall: given a known role/value, infer the unknown role.
    /// known_role: "tool" | "entity" | "outcome"
    /// query_role: "tool" | "entity" | "outcome"
    pub fn recall_hdcbind(
        &self,
        known_role: &str,
        known_val: &str,
        query_role: &str,
        k: usize,
    ) -> Result<Vec<RecallHit>> {
        let results = self.episode_hdc.read().recall_hdcbind(known_role, known_val, query_role, k);
        let hits = results
            .into_iter()
            .enumerate()
            .map(|(i, (name, sim))| {
                let content = format!(
                    "[hdcbind] given {}={} → {}={} (sim={:.3})",
                    known_role, known_val, query_role, name, sim
                );
                RecallHit {
                    memory_id:           i as u64,
                    score:               sim,
                    semantic_score:      0.0,
                    ts_ms:               0,
                    kind:                "hdcbind".to_string(),
                    realm:               "cec".to_string(),
                    strength:            sim,
                    confidence:          sim,
                    access_count:        0,
                    content,
                    semantic_weight:     0.0,
                    status_mul:          1.0,
                    epistemic_mul:       1.0,
                    strength_factor:     1.0,
                    affect_valence:      0.0,
                    affect_arousal:      0.0,
                    actr_activation:     0.0,
                    surprise_boost:      0.0,
                    arousal_boost:       0.0,
                    mood_congruence:     1.0,
                    frustration_boost:   0.0,
                    interference_factor: 1.0,
                    spacing_boost:       1.0,
                }
            })
            .collect();
        Ok(hits)
    }

    /// Counterfactual recall: given (tool, entity, outcome), what alternative tools/entities
    /// would have had a lower failure rate in the same context?
    pub fn recall_counterfactual(
        &self,
        tool:    &str,
        entity:  &str,
        outcome: u8,
        k:       usize,
    ) -> Result<Vec<RecallHit>> {
        use crate::organ::cdawg::CounterfactualHit;
        let taken_sym = self.event_tape.write().symbol_of(tool, entity, outcome);
        let context   = self.event_tape.read().last_n_syms(4);
        let tape      = self.event_tape.read();
        let cdawg     = self.cdawg.read();
        let hits: Vec<CounterfactualHit> = cdawg.counterfactual_alternatives(&context, taken_sym, 5, k);
        let result = hits.into_iter().enumerate().map(|(i, h)| {
            let tool_id   = (h.symbol >> 40) as u16;
            let out_cl    = ((h.symbol >> 32) & 0xff) as u8;
            let entity_k  = (h.symbol & 0xffff_ffff) as u32;
            let alt_tool  = tape.tool_name(tool_id);
            let alt_ent   = tape.entity_name(entity_k);
            let out_str   = match out_cl { 0=>"success",1=>"fail",2=>"error",_=>"partial" };
            let content = format!(
                "[counterfactual rank={}] use {}({}) → {} instead: fail_rate {:.0}% vs {:.0}% taken (Δ={:+.0}%, n={})",
                i+1, alt_tool, alt_ent, out_str,
                h.fail_ratio * 100.0, h.taken_fail_ratio * 100.0, h.delta * 100.0, h.support
            );
            RecallHit {
                memory_id:           i as u64,
                score:               h.delta.max(0.0),
                semantic_score:      0.0,
                ts_ms:               0,
                kind:                "counterfactual".to_string(),
                realm:               "cec".to_string(),
                strength:            h.delta.max(0.0),
                confidence:          1.0 - h.wilson_fail_lower,
                access_count:        h.support,
                content,
                semantic_weight:     0.0,
                status_mul:          1.0,
                epistemic_mul:       1.0,
                strength_factor:     1.0,
                affect_valence:      h.delta,
                affect_arousal:      h.delta.abs(),
                actr_activation:     0.0,
                surprise_boost:      0.0,
                arousal_boost:       0.0,
                mood_congruence:     1.0,
                frustration_boost:   0.0,
                interference_factor: 1.0,
                spacing_boost:       1.0,
            }
        }).collect();
        Ok(result)
    }

    /// Preview the top-k rules that consolidation_pass would promote (no writes).
    pub fn consolidation_preview(&self, k: usize) -> Vec<(String, u32)> {
        use crate::organ::sequitur::run_sequitur;
        let tape = self.event_tape.read();
        let rules = run_sequitur(&tape, 5);
        rules.iter().take(k).map(|r| (r.rule_key(&tape), r.support)).collect()
    }

    /// Sequitur consolidation: find frequent bigrams in EventTape, promote to triplet KG.
    /// Returns (rules_found, rules_promoted).
    pub fn consolidation_pass(&self) -> Result<(usize, usize)> {
        use crate::organ::sequitur::run_sequitur;
        const MIN_SUPPORT: u32 = 5;

        // Operator kill-switch. consolidation_pass is expensive (run_sequitur + FEP
        // rebuild over the whole tape); when many consolidate_request ops pile up in
        // the queue it can monopolize the daemon. Setting CHITTA_DISABLE_CONSOLIDATION
        // makes every trigger (queue, sleep, manual RPC) a no-op.
        if std::env::var_os("CHITTA_DISABLE_CONSOLIDATION").is_some() {
            return Ok((0, 0));
        }

        // Single-flight. A pass can take a long time on a large tape, while the sleep
        // timer + queued consolidate_request ops fire far more often. Without this guard
        // the triggers STACK — each acquires the daemon's RPC mutex in turn and they
        // pile up, turning a slow pass into an unbounded recall outage. Skip any trigger
        // that arrives while a pass is in flight; the next timer tick picks up the work.
        static CONSOLIDATING: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if CONSOLIDATING.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return Ok((0, 0));
        }
        struct InFlightGuard;
        impl Drop for InFlightGuard {
            fn drop(&mut self) {
                CONSOLIDATING.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _in_flight = InFlightGuard;

        let (rule_data, rules_for_ledger): (Vec<(String, String, String, String, String, String)>, Vec<crate::organ::sequitur::SequiturRule>) = {
            // Clone the tape under a brief read, then release the lock before the
            // expensive run_sequitur (~30s on a large tape). Holding event_tape.read()
            // across that span starves recall: a queued tape writer (put_memory/log_event)
            // blocks, and parking_lot task-fairness then blocks every subsequent reader.
            let tape = self.event_tape.read().clone();
            let rules = run_sequitur(&tape, MIN_SUPPORT);
            let data = rules.iter().map(|r| (
                r.rule_key(&tape),
                r.seq_repr(&tape),
                r.avg_outcome_label().to_string(),
                r.support.to_string(),
                format!("{}:{}", r.tape_start, r.tape_end),
                r.verbalize(&tape),
            )).collect();
            (data, rules)
        };

        // Seed refutation ledger with current rule set
        self.refutation_ledger.write().seed_from_rules(&rules_for_ledger);
        // Rebuild hypothesis market from updated ledger (Phase 10)
        self.hypothesis_market.write().update_from_ledger(&self.refutation_ledger.read());

        let total = rule_data.len();
        let mut promoted = 0usize;
        let now = now_ms();
        for (key, seq, outcome, support, range, verbalized) in rule_data {
            // Skip if this rule key is already in the KG (dedup across runs).
            if !self.triplet_store.read().query_subject(&key, now).is_empty() { continue; }
            if self.add_triplet(key.clone(), "compresses".into(),    seq,        1.0, None, None).is_err() { continue; }
            let _ = self.add_triplet(key.clone(), "avg_outcome".into(),  outcome,    1.0, None, None);
            let _ = self.add_triplet(key.clone(), "support".into(),      support,    1.0, None, None);
            let _ = self.add_triplet(key.clone(), "tape_range".into(),   range,      1.0, None, None);
            let _ = self.add_triplet(key,         "verbalized_as".into(), verbalized, 1.0, None, None);
            promoted += 1;
        }
        // Phase 12: compress low-surprisal events from the tape — WITHOUT holding any
        // lock across the heavy O(n) cdawg.surprisal sweep, which previously stalled
        // recall for 13-35s (the tape write blocked log_event, and parking_lot's
        // writer-fairness then starved every new tape reader). Snapshot tape+cdawg under
        // one brief read (tape→cdawg order, matching put_memory — the clones are cheap:
        // TurnEvent is 32 bytes), compute the removal mask off-lock, then apply it under a
        // short write. Events appended during the sweep (indices past the mask) are kept.
        {
            let (tape_snapshot, cdawg_snapshot) = {
                let tape = self.event_tape.read();
                let cdawg = self.cdawg.read();
                (tape.clone(), cdawg.clone())
            };
            let remove = tape_snapshot.compute_low_surprisal_removals(&cdawg_snapshot, 0.85);
            drop(tape_snapshot);
            drop(cdawg_snapshot);
            let removed = if remove.iter().any(|&r| r) {
                self.event_tape.write().apply_removals(&remove)
            } else {
                0
            };
            if removed > 0 {
                self.tape_tombstoned.fetch_add(removed as u64, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[cec] temporal compression: tombstoned {removed} low-surprisal events");
            }
        }

        // Phase 15: rebuild FEP model from (compressed) tape. Clone under a brief read so
        // the O(events) rebuild runs off-lock (same starvation reasoning as run_sequitur).
        {
            let tape = self.event_tape.read().clone();
            self.fep_prior.write().rebuild_from_tape(&tape);
            eprintln!("[cec] fep rebuilt: {} states modeled, drift={:.3}, shock={:.3}",
                self.fep_prior.read().state_emission_len(),
                self.fep_prior.read().ewma_drift,
                self.fep_prior.read().ewma_shock);
        }

        // Phase 11: take a Turīya health sample after each consolidation_pass.
        let diagnosis = {
            let ts = now_ms();
            let tape    = self.event_tape.read();
            let cdawg   = self.cdawg.read();
            let ledger  = self.refutation_ledger.read();
            let market  = self.hypothesis_market.read();
            let fep     = self.fep_prior.read();
            self.turiya_monitor.write().sample(ts, &cdawg, &tape, &ledger, &market, &fep);
            self.turiya_monitor.read().latest()
                .map(|s| s.diagnose())
                .unwrap_or(crate::organ::turiya_monitor::Diagnosis::Healthy)
        };

        // Phase 14: auto-queue experiments when Turīya detects high uncertainty.
        if diagnosis == crate::organ::turiya_monitor::Diagnosis::HighUncertainty {
            let result = self.queue_experiments(5);
            eprintln!("[cec] turiya→HighUncertainty: auto-queued experiments: {result}");
        }

        // Phase 17: reconcile pass — detect and log illegal edges + contradictions.
        {
            let reconcile_json = self.reconcile_pass();
            eprintln!("[cec:p17] reconcile: {reconcile_json}");
        }

        // Phase 16: write falsifiability metrics to triplet KG.
        {
            let now = now_ms();
            let tape_len = self.event_tape.read().events.len();
            let contradiction_count = self.triplet_store.read()
                .query_predicate("illegal_edge_blocked", now).len();
            let _ = self.add_triplet(
                "cec:contradiction_yield".into(), "total_blocked".into(),
                contradiction_count.to_string(), 1.0, None, None,
            );
            let _ = self.add_triplet(
                "cec:router_ready".into(), "tape_events".into(),
                tape_len.to_string(), 1.0, None, None,
            );
            eprintln!("[cec:p16] metrics: contradiction_yield={contradiction_count} tape={tape_len}");
        }

        Ok((total, promoted))
    }

    /// Return the current Turīya health vector as a JSON string.
    pub fn turiya_status(&self) -> String {
        self.turiya_monitor.read().status_json()
    }

    /// Return EventTape statistics including compression totals.
    pub fn fep_status(&self) -> String {
        self.fep_prior.read().status_json()
    }

    /// Phase 16: CPU-native routed recall. Dispatches to the cheapest lane that
    /// can answer the request without an LLM call. Returns JSON with dispatch_label
    /// and the resulting recall hits.
    pub fn routed_recall(&self, req: RecallRequest) -> String {
        let router = QueryRouter::new();
        let dispatch = router.route(&req);
        let label = dispatch.label();
        let k = req.k.max(1);

        let hits: Vec<String> = match dispatch {
            DispatchKind::Exact => {
                let now = now_ms();
                let ts = self.triplet_store.read();
                let entries = match (&req.subject, &req.predicate) {
                    (Some(s), _) => ts.query_subject(s, now),
                    (_, Some(p)) => ts.query_predicate(p, now),
                    _ => vec![],
                };
                entries.iter().take(k).map(|e| {
                    format!(r#"{{"subject":"{}","predicate":"{}","object":"{}","weight":{:.3}}}"#,
                        e.subject.replace('"', "\\\""),
                        e.predicate.replace('"', "\\\""),
                        e.object.replace('"', "\\\""),
                        e.weight)
                }).collect()
            }
            DispatchKind::Fuzzy => {
                self.recall_keyword(req.freetext.as_deref().unwrap_or(""), k)
                    .unwrap_or_default()
                    .iter().map(|h| format!(
                        r#"{{"memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )).collect()
            }
            DispatchKind::Temporal => {
                let from = req.time_from_ms.unwrap_or(0);
                let to   = req.time_to_ms.unwrap_or(i64::MAX);
                self.recall_temporal(from, to, None, k)
                    .unwrap_or_default()
                    .iter().map(|h| format!(
                        r#"{{"memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )).collect()
            }
            DispatchKind::Causal => {
                let tool   = req.causal_tool.as_deref().unwrap_or("");
                let entity = req.causal_entity.as_deref().unwrap_or("");
                self.recall_causal(tool, entity, k)
                    .unwrap_or_default()
                    .iter().map(|h| format!(
                        r#"{{"memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )).collect()
            }
            DispatchKind::Hybrid => {
                // Exact lane first, fuzzy fills remaining slots.
                let now = now_ms();
                let ts = self.triplet_store.read();
                let mut out: Vec<String> = match (&req.subject, &req.predicate) {
                    (Some(s), _) => ts.query_subject(s, now),
                    (_, Some(p)) => ts.query_predicate(p, now),
                    _ => vec![],
                }.iter().take(k / 2 + 1).map(|e| {
                    format!(r#"{{"lane":"exact","subject":"{}","predicate":"{}","object":"{}"}}"#,
                        e.subject.replace('"', "\\\""),
                        e.predicate.replace('"', "\\\""),
                        e.object.replace('"', "\\\""))
                }).collect();
                drop(ts);
                if out.len() < k {
                    let fuzzy = self.recall_keyword(
                        req.freetext.as_deref().unwrap_or(""), k - out.len()
                    ).unwrap_or_default();
                    out.extend(fuzzy.iter().map(|h| format!(
                        r#"{{"lane":"fuzzy","memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )));
                }
                out
            }
            DispatchKind::NeedsDisambiguation(slots) => {
                let slot_json: Vec<String> = slots.iter().map(|s| {
                    format!(r#"{{"slot":"{}","context":"{}"}}"#,
                        s.name, s.context.replace('"', "\\\""))
                }).collect();
                return format!(
                    r#"{{"dispatch":"needs_disambiguation","unbound_slots":[{}],"hits":[]}}"#,
                    slot_json.join(",")
                );
            }
        };

        format!(
            r#"{{"dispatch":"{label}","token_cost":0,"hits":[{}]}}"#,
            hits.join(",")
        )
    }

    pub fn tape_stats(&self) -> String {
        let tombstoned = self.tape_tombstoned.load(std::sync::atomic::Ordering::Relaxed);
        self.event_tape.read().stats_json(tombstoned)
    }

    /// Phase 14 — Queue deliberate micro-experiments for uncertain Sequitur rules.
    ///
    /// Reads HypothesisMarket::top_probes(k) and files an OpenTask intervention
    /// for each rule whose probe_value > 0.4 AND refutation_ratio < 0.3 (safety gate).
    /// Returns JSON: {"queued": N, "skipped_refuted": M, "skipped_certain": L}
    pub fn queue_experiments(&self, k: usize) -> String {
        use crate::organ::intervention_store::InterventionKind;
        let probes = {
            let market = self.hypothesis_market.read();
            market.top_probes(k.max(1)).to_vec()
        };
        let ledger = self.refutation_ledger.read();
        let mut store = self.cec_policy_store.write();
        let ts = now_ms();
        let mut queued = 0usize;
        let mut skipped_refuted = 0usize;
        let mut skipped_certain = 0usize;
        for h in &probes {
            if h.probe_value <= 0.4 {
                skipped_certain += 1;
                continue;
            }
            // Adversarial gate: don't experiment on rules being actively refuted.
            let refute_ratio = ledger.refute_ratio_for_rule(h.rule_id);
            if refute_ratio >= 0.3 {
                skipped_refuted += 1;
                continue;
            }
            let title = format!("probe_rule_{}", h.rule_id);
            let desc = format!(
                "CEC Phase 14 experiment: rule {} has p_hat={:.3} (probe_value={:.3}). \
                 Deliberately execute its antecedent and observe whether the consequent follows.",
                h.rule_id, h.p_hat, h.probe_value
            );
            store.propose(h.rule_id, InterventionKind::OpenTask { title, description: desc }, ts);
            queued += 1;
        }
        format!(
            r#"{{"queued":{queued},"skipped_refuted":{skipped_refuted},"skipped_certain":{skipped_certain}}}"#
        )
    }

    /// Phase 13 — Return top-k verbalized Sequitur rules ranked by support.
    pub fn verbalize_rules(&self, k: usize) -> String {
        use crate::organ::sequitur::run_sequitur;
        const MIN_SUPPORT: u32 = 3;
        let tape = self.event_tape.read();
        let mut rules = run_sequitur(&tape, MIN_SUPPORT);
        rules.sort_by(|a, b| b.support.cmp(&a.support));
        rules.truncate(k.max(1));
        let items: Vec<String> = rules.iter().map(|r| {
            let v = r.verbalize(&tape);
            let key = r.rule_key(&tape);
            format!(
                r#"{{"rule_id":{},"support":{},"avg_outcome":"{}","key":"{}","text":"{}"}}"#,
                r.id, r.support, r.avg_outcome_label(), key,
                v.replace('"', "\\\"")
            )
        }).collect();
        format!(r#"{{"total":{},"rules":[{}]}}"#, items.len(), items.join(","))
    }

    /// Phase 17: Promote a candidate memory to established band once a witness arrives.
    /// Returns JSON status.
    pub fn witness_memory(&self, memory_id: MemoryId, witness_kind: &str) -> String {
        let wk = WitnessKind::from_str(witness_kind);
        let mut payloads = self.payloads.write();
        let Some(payload) = payloads.get_mut(&memory_id) else {
            return format!(r#"{{"ok":false,"error":"not_found","memory_id":{memory_id}}}"#);
        };
        if !payload.candidate {
            return format!(r#"{{"ok":true,"status":"already_established","memory_id":{memory_id}}}"#);
        }
        if wk.is_none() {
            return format!(r#"{{"ok":false,"error":"unknown_witness_kind","memory_id":{memory_id}}}"#);
        }
        payload.candidate = false;
        eprintln!("[cec:p17] memory {memory_id} promoted from candidate via witness={witness_kind}");
        let _ = self.add_triplet(
            format!("cec:witness:{memory_id}"),
            "promoted_by".into(),
            witness_kind.to_string(),
            1.0, None, None,
        );
        format!(r#"{{"ok":true,"status":"promoted","memory_id":{memory_id},"witness_kind":"{witness_kind}"}}"#)
    }

    /// Phase 17: Run the R0 reconcile pass — scan assoc_edges for legality violations
    /// and detect content contradictions. Returns JSON summary.
    pub fn reconcile_pass(&self) -> String {
        let payloads    = self.payloads.read();
        let assoc_edges = self.assoc_edges.read();
        let rec = Reconciler::new();

        let result = rec.reconcile_all(&payloads, &assoc_edges);
        let contras = rec.detect_contradictions(&payloads);

        let now = now_ms();
        // Log illegal edges to triplet KG
        for (src, dst, reason) in &result.illegal_edges {
            let _ = self.add_triplet(
                format!("cec:reconcile:{src}→{dst}"),
                "illegal_reason".into(),
                reason.clone(),
                1.0, None, None,
            );
        }
        // Log contradictions
        for (a, b, score) in &contras {
            let _ = self.add_triplet(
                format!("contradiction:{a}-{b}"),
                "conflict_score".into(),
                format!("{score:.2}"),
                1.0, None, None,
            );
        }
        let _ = now; // suppress unused warning

        format!(
            r#"{{"illegal_edges":{},"contradictions":{},"unresolved":{},"ok":true}}"#,
            result.illegal_edges.len(),
            contras.len(),
            result.unresolved.len(),
        )
    }

    /// Phase 17: Produce a harvest scope document from current Turīya anomalies
    /// and router miss patterns. Used by `scripts/harvest_ow.py` to target extraction.
    pub fn harvest_scope(&self) -> String {
        let turiya_json = self.turiya_status();
        let turiya: serde_json::Value = serde_json::from_str(&turiya_json)
            .unwrap_or(serde_json::Value::Null);
        let diagnosis = turiya.get("diagnosis")
            .and_then(|v| v.as_str()).unwrap_or("unknown");

        // Top router misses from contradiction_yield triplets
        let now = now_ms();
        let ts_guard = self.triplet_store.read();
        let miss_entries = ts_guard.query_predicate("illegal_edge_blocked", now);
        let miss_count = miss_entries.len();

        let sample_misses: Vec<serde_json::Value> = miss_entries.iter().take(5).map(|e| {
            serde_json::json!({
                "pattern": e.object,
                "miss_count": 1,
                "suggested_corpus": if e.object.contains("code") {
                    "code_editing_failures"
                } else {
                    "session_continuity"
                }
            })
        }).collect();

        let scope = serde_json::json!({
            "generated_at_ms": now,
            "turiya_diagnosis": diagnosis,
            "top_router_misses": sample_misses,
            "total_router_misses": miss_count,
            "harvest_budget_items": 500_usize.min(miss_count * 10 + 50),
        });

        scope.to_string()
    }

    pub fn seed_hdc_geometry(&self, json_path: &str) -> String {
        let result = self.hdc_idx.write().seed_from_geometry(json_path);
        match result {
            Ok(n) => {
                let codebook_len = self.hdc_idx.read().codebook_len();
                serde_json::json!({
                    "ok": true,
                    "seeded_tokens": n,
                    "codebook_len": codebook_len,
                    "source": json_path,
                }).to_string()
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }).to_string(),
        }
    }

    /// Return top-k CDAWG states reachable from (tool, entity) ranked by Q-value.
    pub fn recall_motif_value(&self, tool: &str, entity: &str, k: usize) -> Result<Vec<RecallHit>> {
        let sym = {
            let mut tape = self.event_tape.write();
            tape.symbol_of(tool, entity, 0)
        };
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();
        let rows = cdawg.top_q_states(&[sym], k.max(1));
        let hits = rows.into_iter().map(|(state_id, q_val, support)| {
            let next_syms: Vec<String> = cdawg.states.get(state_id as usize)
                .map(|s| s.transitions.keys().take(3).map(|&sym| {
                    let tn = tape.tool_name((sym >> 40) as u16);
                    let en = tape.entity_name((sym & 0xffff_ffff) as u32);
                    format!("{tn}({en})")
                }).collect())
                .unwrap_or_default();
            let content = format!(
                "[motif state={}] q={:.3} support={} next=[{}]",
                state_id, q_val, support, next_syms.join(", ")
            );
            RecallHit {
                memory_id:           state_id as u64,
                score:               (q_val + 1.0) / 2.0,
                semantic_score:      0.0,
                ts_ms:               0,
                kind:                "motif".to_string(),
                realm:               "cec".to_string(),
                strength:            (q_val + 1.0) / 2.0,
                confidence:          support as f32 / (support as f32 + 1.0),
                access_count:        support,
                content,
                semantic_weight:     0.0,
                status_mul:          1.0,
                epistemic_mul:       1.0,
                strength_factor:     1.0,
                affect_valence:      q_val.clamp(-1.0, 1.0),
                affect_arousal:      q_val.abs().clamp(0.0, 1.0),
                actr_activation:     0.0,
                surprise_boost:      0.0,
                arousal_boost:       0.0,
                mood_congruence:     (q_val + 1.0) / 2.0,
                frustration_boost:   0.0,
                interference_factor: 1.0,
                spacing_boost:       1.0,
            }
        }).collect();
        Ok(hits)
    }

    /// Return top-k rules by refute_ratio as a plain-text summary.
    pub fn refutation_stats(&self, k: usize) -> String {
        let tape   = self.event_tape.read();
        let ledger = self.refutation_ledger.read();
        ledger.stats_json(&tape, k)
    }

    /// Promote eligible shadow policies, demote drifted ones, return JSON summary.
    pub fn executor_flush(&self) -> String {
        let ledger = self.refutation_ledger.read();
        let mut store = self.cec_policy_store.write();
        let promoted = store.promote_eligible();
        let demoted  = store.auto_demote_drifted(&ledger);
        let stats    = store.stats_json();
        format!(
            "{{\"promoted\":{:?},\"demoted\":{:?},\"store\":{}}}",
            promoted, demoted, stats
        )
    }

    /// List intervention policies as JSON.
    pub fn list_policies(&self, active_only: bool) -> String {
        self.cec_policy_store.read().list_json(active_only)
    }

    /// Record an explicit decision point: what was chosen, what was rejected and why.
    /// `rejected` is a slice of (packed_symbol, RejectionReason as u8).
    pub fn log_decision(
        &self,
        chosen_tool: &str, chosen_entity: &str, chosen_outcome: u8,
        rejected: Vec<(u64, u8)>,
        confidence_delta: f32,
        ts_ms: i64,
    ) {
        let chosen_sym = self.event_tape.read().symbol_of_ro(chosen_tool, chosen_entity, chosen_outcome);
        let turn_id = self.event_tape.read().events.len() as u32;
        self.decision_tape.write().log(turn_id, chosen_sym, rejected, confidence_delta, ts_ms);
    }

    /// Log an event with cost metadata for regret-shaped Q-value update (Phase 10 Part B).
    pub fn log_event_ex(
        &self,
        tool: &str, entity: &str, outcome: u8,
        session_id: u64, ts_ms: i64,
        token_cost: u32, latency_ms: u32, retry_count: u8,
    ) {
        const ALPHA_COST:    f32 = 0.001;
        const BETA_LATENCY:  f32 = 0.00001;
        const GAMMA_RETRIES: f32 = 0.1;
        let (sym, turn, last_n) = {
            let mut tape = self.event_tape.write();
            let s = tape.log(tool, entity, outcome, session_id, ts_ms);
            let t = tape.events.len() as u32 - 1;
            let n = tape.last_n_syms(16);
            (s, t, n)
        };
        let mut cdawg = self.cdawg.write();
        cdawg.extend(sym, turn);
        if tool != "legacy" && tool != "remember" {
            let base = if outcome == 0 { 1.0_f32 } else { -1.0_f32 };
            let utility = base
                - ALPHA_COST    * token_cost  as f32
                - BETA_LATENCY  * latency_ms  as f32
                - GAMMA_RETRIES * retry_count as f32;
            let delta = if outcome == 0 { 0.1_f32 } else { -0.2_f32 };
            cdawg.push_td_credit(&last_n, delta, 0.9);
            cdawg.update_q(sym, utility, 0.05, 0.95);
        }
        drop(cdawg);
        self.episode_hdc.write().log_episode(tool, entity, outcome);
    }

    /// True counterfactual recall: use DecisionTape to find cases where (tool, entity) was
    /// explicitly considered and rejected, and report the chosen alternative's outcome.
    pub fn recall_true_counterfactual(
        &self, tool: &str, entity: &str, outcome: u8, k: usize,
    ) -> Result<Vec<RecallHit>> {
        let sym = self.event_tape.read().symbol_of_ro(tool, entity, outcome);
        let etape = self.event_tape.read();
        let tape = self.decision_tape.read();
        let hits = tape.rejected_alternatives(sym, k)
            .into_iter()
            .enumerate()
            .map(|(i, (dp, reason_u8))| {
                let reason = crate::organ::decision_tape::RejectionReason::from_u8(reason_u8);
                let chosen_tool_id  = (dp.chosen_sym >> 40) as u16;
                let chosen_entity_k = (dp.chosen_sym & 0xffff_ffff) as u32;
                let chosen_tool_name   = etape.tool_name(chosen_tool_id);
                let chosen_entity_name = etape.entity_name(chosen_entity_k);
                let content = format!(
                    "[counterfactual turn={}] rejected {}({}) reason={} → chose {}({}) confidence_delta={:.3}",
                    dp.turn_id, tool, entity, reason.label(),
                    chosen_tool_name, chosen_entity_name, dp.confidence_delta
                );
                RecallHit {
                    memory_id:           dp.turn_id as u64,
                    score:               1.0 - i as f32 * 0.1,
                    semantic_score:      0.0,
                    ts_ms:               dp.ts_ms,
                    kind:                "counterfactual".to_string(),
                    realm:               "cec".to_string(),
                    strength:            dp.confidence_delta.abs(),
                    confidence:          dp.confidence_delta.abs().min(1.0),
                    access_count:        0,
                    content,
                    semantic_weight:     0.0,
                    status_mul:          1.0,
                    epistemic_mul:       1.0,
                    strength_factor:     1.0,
                    affect_valence:      dp.confidence_delta.clamp(-1.0, 1.0),
                    affect_arousal:      dp.confidence_delta.abs().clamp(0.0, 1.0),
                    actr_activation:     0.0,
                    surprise_boost:      0.0,
                    arousal_boost:       0.0,
                    mood_congruence:     1.0,
                    frustration_boost:   0.0,
                    interference_factor: 1.0,
                    spacing_boost:       1.0,
                }
            })
            .collect();
        Ok(hits)
    }

    /// Top-k rules by expected information gain (Wilson probe_value). Highest = most uncertain.
    pub fn hypothesis_probes(&self, k: usize) -> String {
        self.hypothesis_market.read().stats_json(k)
    }

    /// Keyword (BM25) recall with affective context.
    pub fn recall_keyword_ctx(
        &self,
        query: &str,
        k: usize,
        query_valence: Option<f32>,
        query_arousal: Option<f32>,
        realm: Option<&str>,
        strengthen: bool,
    ) -> Result<Vec<RecallHit>> {
        let max_query_idf = self.keyword_idx.read().query_max_idf(query);
        // Realm-scoped queries need a larger global BM25 fetch so small-realm
        // memories aren't squeezed out by cross-realm hits in the global corpus.
        let bm25_fetch = if realm.is_some() { k * 12 } else { k * 3 };
        let keyword_hits = self.keyword_idx.read().search(query, bm25_fetch);

        let now = now_ms();
        let payloads = self.payloads.read();
        let states = self.states.read();
        let learners = self.learners.read();
        let pipeline = self.scoring_pipeline.read();
        let ack_scores = self.ack_scores.read();
        let mut util_rng = utility_rng(now);

        let mut hits: Vec<RecallHit> = keyword_hits
            .into_iter()
            .filter_map(|hit| {
                let state = states.get(&hit.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&hit.memory_id)?;
                let content_str = String::from_utf8(payload.content.clone()).unwrap_or_default();
                if content_str.trim().is_empty() {
                    return None;
                }
                if payload.realm.starts_with("soul:") {
                    return None;
                }
                // Realm scoping: the BM25 lane must not leak other projects' memories into a
                // realm-scoped recall (the cross-realm injection bleed).
                if let Some(want) = realm {
                    if payload.realm != want {
                        return None;
                    }
                }
                let ctx = ScoringContext {
                    relevance_score: hit.bm25_score,
                    recall_mode: RecallMode::Keyword,
                    state,
                    kind: &payload.kind,
                    realm: &payload.realm,
                    realm_reliability: learners.domain_reliability.reliability(&payload.realm),
                    now_ms: now,
                    query_valence,
                    query_arousal,
                    prediction_prob: None,
                    surprise_role: None,
                    has_open_debt: false,
                    integration_weight: None,
                    ack_score: ack_scores.get(&hit.memory_id).copied().unwrap_or(0),
                    max_query_idf,
                };
                let (score, decomp) = pipeline.score(&ctx)?;
                let score = score * utility_multiplier(state, &mut util_rng);
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: hit.memory_id,
                    score,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: content_str,
                    semantic_weight: decomp.semantic_weight,
                    status_mul: decomp.status_mul,
                    epistemic_mul: decomp.epistemic_mul,
                    strength_factor: decomp.strength_factor,
                    affect_valence: state.affect_valence,
                    affect_arousal: state.affect_arousal,
                    actr_activation: decomp.actr_activation,
                    surprise_boost: decomp.surprise_boost,
                    arousal_boost: decomp.arousal_boost,
                    mood_congruence: decomp.mood_congruence,
                    frustration_boost: decomp.frustration_boost,
                    interference_factor: decomp.interference_factor,
                    spacing_boost: decomp.spacing_boost,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id))
        });
        hits.truncate(k);

        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        drop(pipeline);
        drop(learners);
        drop(ack_scores);
        if strengthen {
            self.enqueue_recall_effects(&hit_ids);
        }

        Ok(hits)
    }

    /// Session-level recall: aggregates chunk-level hits per source_session using noisy-OR.
    /// Returns sessions ranked by combined evidence strength.
    /// `query_embedding` — pre-computed by caller (C++ embed layer); None skips semantic lane.
    pub fn recall_session(
        &self,
        query_embedding: Option<&[f32]>,
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<crate::recall::SessionRecallHit>> {
        use std::collections::HashMap;
        use crate::recall::SessionRecallHit;

        // Fetch candidate chunks from both semantic and keyword lanes
        let fetch_limit = k * 20;
        let mut candidates: Vec<crate::recall::RecallHit> = if let Some(emb) = query_embedding {
            self.recall_semantic_ctx(emb, fetch_limit, realm, None, None, true)?
        } else {
            Vec::new()
        };

        // Merge in keyword hits, deduplicating by memory_id (keep max score)
        let kw_hits = self.recall_keyword_ctx(query_text, fetch_limit, None, None, None, true)?;
        let mut seen: std::collections::HashSet<crate::ids::MemoryId> =
            candidates.iter().map(|h| h.memory_id).collect();
        for h in kw_hits {
            if seen.insert(h.memory_id) {
                candidates.push(h);
            }
        }

        // Group by source_session; skip memories without a session
        let payloads = self.payloads.read();
        struct SessionAcc {
            scores: Vec<f32>,
            best_score: f32,
            best_content: String,
            realm: String,
        }
        let mut sessions: HashMap<String, SessionAcc> = HashMap::new();

        for hit in &candidates {
            if let Some(payload) = payloads.get(&hit.memory_id) {
                if let Some(ref sid) = payload.source_session {
                    let acc = sessions.entry(sid.clone()).or_insert_with(|| SessionAcc {
                        scores: Vec::new(),
                        best_score: 0.0,
                        best_content: String::new(),
                        realm: payload.realm.clone(),
                    });
                    acc.scores.push(hit.score);
                    if hit.score > acc.best_score {
                        acc.best_score = hit.score;
                        acc.best_content = hit.content.clone();
                    }
                }
            }
        }
        drop(payloads);

        // Score: max_chunk_score dominates; small noisy-OR bonus from remaining evidence.
        // Avoids multi-mediocre-chunk sessions beating a single high-score gold chunk.
        let mut session_hits: Vec<SessionRecallHit> = sessions
            .into_iter()
            .map(|(session_id, mut acc)| {
                acc.scores.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let max_s = acc.scores[0].min(1.0).max(0.0);
                // noisy-OR of chunks beyond the best (corroborating evidence only)
                let corroboration = acc.scores.iter().skip(1).take(4).fold(0.0f32, |combined, &s| {
                    1.0 - (1.0 - combined) * (1.0 - s.min(1.0).max(0.0))
                });
                let session_score = max_s + 0.15 * corroboration * (1.0 - max_s);
                SessionRecallHit {
                    session_id,
                    score: session_score,
                    chunk_count: acc.scores.len() as u32,
                    max_chunk_score: acc.best_score,
                    best_evidence: acc.best_content,
                    realm: acc.realm,
                }
            })
            .collect();

        session_hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.session_id.cmp(&b.session_id))
        });
        session_hits.truncate(k);
        Ok(session_hits)
    }

    pub fn recall_with_fallback(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_with_fallback_windowed(query_embedding, query_text, k, realm, None)
    }

    /// Time-windowed hybrid recall: the window GATES candidates (authored_at_ms
    /// membership in time_idx), semantic relevance still RANKS. Never
    /// recency-sorts — recency-ordered temporal recall floods context with
    /// operationally-fresh noise (stored correction). window=None is exactly
    /// recall_with_fallback.
    pub fn recall_with_fallback_windowed(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
        window: Option<(i64, i64)>,
    ) -> Result<Vec<RecallHit>> {
        // Window membership set, built from the temporal B-tree and dropped
        // before any lane runs (taken alone: no lock-ordering pair created).
        let allowed: Option<std::collections::HashSet<MemoryId>> = window.map(|(f, t)| {
            self.time_idx
                .read()
                .range_query(f, t, realm, usize::MAX)
                .into_iter()
                .map(|e| e.memory_id)
                .collect()
        });
        let gate = |v: Vec<RecallHit>| -> Vec<RecallHit> {
            match &allowed {
                Some(a) => v.into_iter().filter(|h| a.contains(&h.memory_id)).collect(),
                None => v,
            }
        };
        // Windowed queries over-fetch harder: the gate discards out-of-window
        // candidates, so lanes need more raw width to fill k.
        let window_mul: usize = if window.is_some() { 2 } else { 1 };
        // Stratified recall: cap any one realm on UNSCOPED queries so a dominant
        // realm (e.g. compliance:auto BM25 noise) can't flood results. The cap is
        // Thompson-sampled per realm from its Beta posterior (G8) so reliable
        // realms earn more slots. Targeted queries (realm=Some) are never reshaped.
        let stratify = realm.is_none();
        // Over-fetch the lane when capping so the cap can backfill from
        // non-dominant realms and still return up to k. The over-fetch factor
        // tracks the legacy divisor knob purely as a fetch-width hint.
        let fetch_k = if stratify {
            let factor = self
                .scoring_pipeline
                .read()
                .config
                .recall_realm_cap_divisor
                .max(1);
            k.saturating_mul(factor).max(k).saturating_mul(window_mul)
        } else {
            // Over-fetch globally so realm post-filter has enough candidates.
            let mul = self.scoring_pipeline.read().config.dam_fetch_mul.max(4);
            k.saturating_mul(mul).max(k).saturating_mul(window_mul)
        };
        // Snapshot the per-realm reliability learner so the stratify pass can
        // Thompson-sample without holding the learner lock during scoring.
        let reliability = if stratify {
            Some(self.learners.read().domain_reliability.clone())
        } else {
            None
        };

        // RRF hybrid: on UNSCOPED queries, fuse semantic + BM25 ranks instead of
        // using BM25 only as an empty-semantic fallback. Targeted queries
        // (realm=Some) keep the legacy semantic-then-fallback path.
        let use_rrf = self.scoring_pipeline.read().config.use_rrf;
        if use_rrf {
            let rrf_k = self.scoring_pipeline.read().config.rrf_k;
            // For realm-scoped queries, use global HNSW then post-filter instead of
            // filtered HNSW (search_candidates). search_candidates skips memories whose
            // flat embedding is absent even when the HNSW graph node has it, causing
            // false sim=0 for recently-ingested memories in small realms.
            let sem = if !stratify {
                let global = self.recall_semantic(query_embedding, fetch_k, None)?;
                if let Some(r) = realm {
                    let rm = self.realm_members.read();
                    let allowed = rm.get(r);
                    global
                        .into_iter()
                        .filter(|h| allowed.map(|a| a.contains(&h.memory_id)).unwrap_or(true))
                        .collect()
                } else {
                    global
                }
            } else {
                self.recall_semantic(query_embedding, fetch_k, realm)?
            };
            let sem = gate(sem);
            let kw = gate(self.recall_keyword_realm(query_text, fetch_k, realm)?);
            if !sem.is_empty() || !kw.is_empty() {
                let mut merged = rrf_merge(sem, kw, fetch_k, rrf_k);
                // PPR injection lane: seed from the top fused hits and RRF-merge
                // the association-graph stationary ranking back in, injecting
                // topic-bridged multi-hop memories. Gated by CHITTA_PPR_LANE.
                self.ppr_inject(&mut merged);

                // Cortical SDR re-rank: second-pass RRF over the already-merged candidate set.
                // Gated by use_cortical (default false) and a non-empty cortical index.
                // Re-ranker shape: cortical can only reorder merged candidates, never inject new ones.
                let use_cortical = self.scoring_pipeline.read().config.use_cortical;
                if use_cortical && !self.cortical_idx.read().is_empty() {
                    let cortical_rrf_k = self.scoring_pipeline.read().config.cortical_rrf_k;
                    let code = self.sparse_encoder.read().encode(query_embedding);
                    if !code.is_empty() {
                        let candidate_ids: std::collections::HashSet<MemoryId> =
                            merged.iter().map(|h| h.memory_id).collect();
                        let cortical_ranked: Vec<RecallHit> = {
                            let cortical = self.cortical_idx.read();
                            let by_id: std::collections::HashMap<MemoryId, &RecallHit> =
                                merged.iter().map(|h| (h.memory_id, h)).collect();
                            cortical
                                .search(&code, merged.len(), Some(&candidate_ids))
                                .into_iter()
                                .filter_map(|(mid, _)| by_id.get(&mid).map(|h| (*h).clone()))
                                .collect()
                        };
                        if !cortical_ranked.is_empty() {
                            merged = rrf_merge(merged, cortical_ranked, fetch_k, cortical_rrf_k);
                        }
                    }
                }

                // Stratify only for unscoped queries — targeted queries are already realm-filtered.
                let mut out = if stratify {
                    stratify_recall_hits(merged, k, reliability.as_ref())
                } else {
                    merged.into_iter().take(k).collect()
                };
                // Starvation backfill: the window gate can leave the lanes short.
                // Fill from window members ranked BY COSINE to the query — the
                // window gates, semantic still ranks.
                if out.len() < k {
                    if let Some((f, t)) = window {
                        self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
                    }
                }
                return Ok(out);
            }
            // Both lanes empty → fall through to recency below.
            let store_size = self.memory_count();
            if store_size < 10 {
                return Ok(vec![]);
            }
            if let Some((f, t)) = window {
                // Windowed query with empty lanes: rank window members by cosine,
                // never by recency.
                let mut out = Vec::new();
                self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
                return Ok(out);
            }
            log::warn!("RRF hybrid empty (semantic+BM25), falling back to recency");
            let now = now_ms();
            let temporal = self.recall_temporal(0, now, realm, fetch_k)?;
            return Ok(if stratify {
                stratify_recall_hits(temporal, k, reliability.as_ref())
            } else {
                temporal.into_iter().take(k).collect()
            });
        }

        let hits = gate(self.recall_semantic(query_embedding, fetch_k, realm)?);
        if !hits.is_empty() {
            let mut out = stratify_recall_hits(hits, k, reliability.as_ref());
            if out.len() < k {
                if let Some((f, t)) = window {
                    self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
                }
            }
            return Ok(out);
        }

        let store_size = self.memory_count();
        if store_size < 10 {
            return Ok(vec![]);
        }

        if let Some((f, t)) = window {
            let mut out = Vec::new();
            self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
            return Ok(out);
        }

        log::warn!(
            "recall_semantic returned empty with {} memories, falling back to BM25",
            store_size
        );
        let bm25_hits = self.recall_keyword(query_text, fetch_k)?;
        if !bm25_hits.is_empty() {
            return Ok(stratify_recall_hits(bm25_hits, k, reliability.as_ref()));
        }

        log::warn!("BM25 fallback also empty, falling back to recency");
        let now = now_ms();
        let temporal = self.recall_temporal(0, now, realm, fetch_k)?;
        Ok(stratify_recall_hits(temporal, k, reliability.as_ref()))
    }

    /// Fill `out` up to `k` with window members ranked by cosine similarity to
    /// the query embedding (embeddings are L2-normalized: dot == cosine).
    /// recall_temporal builds the payload-backed hits; its recency ORDER is
    /// discarded — cosine ranks. Cost bounded: at most 4k window members scored.
    fn window_backfill(
        &self,
        out: &mut Vec<RecallHit>,
        k: usize,
        from_ms: i64,
        to_ms: i64,
        realm: Option<&str>,
        query_embedding: &[f32],
    ) -> Result<()> {
        let have: std::collections::HashSet<MemoryId> =
            out.iter().map(|h| h.memory_id).collect();
        let mut pool: Vec<RecallHit> = self
            .recall_temporal(from_ms, to_ms, realm, k.saturating_mul(4).max(64))?
            .into_iter()
            .filter(|h| !have.contains(&h.memory_id))
            .collect();
        for h in pool.iter_mut() {
            h.semantic_score = self
                .embedding_of(h.memory_id)
                .map(|e| e.iter().zip(query_embedding).map(|(a, b)| a * b).sum())
                .unwrap_or(0.0);
            h.score = h.semantic_score;
        }
        pool.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
        out.extend(pool.into_iter().take(k - out.len()));
        Ok(())
    }

    /// Field-RAG / Modern Hopfield recall with optional multi-hop expansion.
    ///
    /// Each hop: fetch candidates via RRF, run T-step DAM relaxation over the
    /// candidate submatrix (s(t+1) = X @ softmax(β·Xᵀs(t))), re-rank by cosine(s_T, X).
    /// With dam_hops > 1: use s_T as the refined query vector for the next hop,
    /// excluding already-fetched IDs. Final output is the union re-sorted by score.
    pub fn recall_field(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        let (beta, steps, fetch_mul, hops) = {
            let cfg = self.scoring_pipeline.read();
            (cfg.config.dam_beta, cfg.config.dam_steps, cfg.config.dam_fetch_mul,
             cfg.config.dam_hops.max(1))
        };
        let fetch_k = k.saturating_mul(fetch_mul).max(k);
        let dim = query_embedding.len();

        let mut all_hits: Vec<RecallHit> = Vec::new();
        let mut query: Vec<f32> = query_embedding.to_vec();

        for _hop in 0..hops {
            // Exclude IDs already collected in prior hops.
            let seen: std::collections::HashSet<u64> =
                all_hits.iter().map(|h| h.memory_id).collect();

            let mut hits = self.recall_with_fallback(&query, query_text, fetch_k, realm)?;
            if _hop > 0 {
                hits.retain(|h| !seen.contains(&h.memory_id));
            }
            if hits.len() < 2 {
                all_hits.extend(hits);
                break;
            }

            // Collect embeddings for the candidate submatrix X.
            let embeddings: Vec<(usize, Vec<f32>)> = {
                let idx = self.semantic_idx.read();
                hits.iter()
                    .enumerate()
                    .filter_map(|(i, h)| {
                        idx.get_embedding(h.memory_id).map(|e| (i, e.to_vec()))
                    })
                    .collect()
            };
            if embeddings.len() < 2 {
                all_hits.extend(hits);
                break;
            }

            // Build a 4-bit TurboQuant index over the candidate submatrix once
            // per hop. turbovec inner-product over unit vectors == cosine, which
            // matches the scalar dots below (stored embs are unit). Used for the
            // DAM energies (all candidates, k = n) and the final re-rank. Falls
            // back to scalar when construction is unavailable (dim % 8 != 0).
            let turbo: Option<(turbovec::TurboQuantIndex, Vec<usize>)> = (|| {
                let n = embeddings.len();
                if dim == 0 || dim % 8 != 0 { return None; }
                let mut idx = turbovec::TurboQuantIndex::new(dim, 4).ok()?;
                let mut flat = Vec::with_capacity(n * dim);
                let mut row_hit: Vec<usize> = Vec::with_capacity(n);
                for (i, emb) in &embeddings {
                    let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm < 1e-12 { return None; }
                    flat.extend(emb.iter().map(|x| x / norm));
                    row_hit.push(*i);
                }
                idx.add(&flat);
                idx.prepare();
                Some((idx, row_hit))
            })();

            // T-step DAM relaxation.
            let mut s: Vec<f32> = query.clone();
            for _ in 0..steps {
                // Energies = beta * (X · s). turbovec needs a unit query to
                // return cosine; `s` is unit after the first step's renormalize,
                // but the initial `s = query` may not be — normalize for scoring.
                let n = embeddings.len();
                let mut energies: Vec<f32> = vec![0.0; n];
                let mut max_e = f32::NEG_INFINITY;
                let scored = turbo.as_ref().and_then(|(idx, row_hit)| {
                    let sn = {
                        let nrm = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if nrm < 1e-12 { return None; }
                        s.iter().map(|x| x / nrm).collect::<Vec<f32>>()
                    };
                    let res = idx.search(&sn, n);
                    let idxs = res.indices_for_query(0);
                    let scs = res.scores_for_query(0);
                    for (row, sc) in idxs.iter().zip(scs.iter()) {
                        if *row < 0 { continue; }
                        // row indexes the submatrix; row_hit maps it to the
                        // candidate slot (== position in `embeddings`).
                        let pos = *row as usize;
                        if let Some(slot) = row_hit.get(pos).copied() {
                            // find energies position: energies is keyed by
                            // embeddings-vec order, which equals row order.
                            let _ = slot;
                        }
                        energies[pos] = beta * *sc;
                    }
                    Some(())
                });
                if scored.is_none() {
                    for (j, (_, emb)) in embeddings.iter().enumerate() {
                        energies[j] = beta * emb.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    }
                }
                for &e in &energies { if e > max_e { max_e = e; } }

                let sum_exp: f32 = energies.iter().map(|&e| (e - max_e).exp()).sum();
                let weights: Vec<f32> =
                    energies.iter().map(|&e| (e - max_e).exp() / sum_exp).collect();

                let mut s_new = vec![0.0f32; dim];
                for (j, (_, emb)) in embeddings.iter().enumerate() {
                    let w = weights[j];
                    for (sn, &xv) in s_new.iter_mut().zip(emb.iter()) {
                        *sn += w * xv;
                    }
                }

                let norm = s_new.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-12 { break; }
                for x in &mut s_new { *x /= norm; }
                let delta: f32 = s_new
                    .iter().zip(s.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                s = s_new;
                if delta < 1e-6 { break; }
            }

            // Re-rank this hop's hits by cosine(s_T, X_j).
            // cosine(s_T, X_j) via turbovec when available, else scalar.
            let final_scored = turbo.as_ref().and_then(|(idx, _)| {
                let nrm = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                if nrm < 1e-12 { return None; }
                let sn: Vec<f32> = s.iter().map(|x| x / nrm).collect();
                let res = idx.search(&sn, embeddings.len());
                let idxs = res.indices_for_query(0);
                let scs = res.scores_for_query(0);
                for (row, sc) in idxs.iter().zip(scs.iter()) {
                    if *row < 0 { continue; }
                    let pos = *row as usize;
                    if let Some((i, _)) = embeddings.get(pos) {
                        hits[*i].score = *sc;
                        hits[*i].semantic_score = *sc;
                    }
                }
                Some(())
            });
            if final_scored.is_none() {
                for (i, emb) in &embeddings {
                    let cos = emb.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    hits[*i].score = cos;
                    hits[*i].semantic_score = cos;
                }
            }
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
            all_hits.extend(hits);

            // s_T becomes the query for the next hop.
            query = s;
        }

        // Merge: sort by score, dedup keeping highest, truncate.
        all_hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
        let mut seen = std::collections::HashSet::new();
        all_hits.retain(|h| seen.insert(h.memory_id));
        Ok(all_hits.into_iter().take(k).collect())
    }

    /// Recall memories associated with a file path (exact match).
    pub fn recall_artifact(&self, path: &str, limit: usize) -> Result<Vec<RecallHit>> {
        let entries = self.artifact_idx.read().query_path(path, limit);
        let now = now_ms();
        let payloads = self.payloads.read();
        let states = self.states.read();

        let hits = entries
            .into_iter()
            .filter_map(|entry| {
                let state = states.get(&entry.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&entry.memory_id)?;
                if payload.content.is_empty() {
                    return None;
                }
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: entry.memory_id,
                    score: entry.strength * eff_strength * state.confidence,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Add a triplet fact. Returns the triplet ID.
    pub fn add_triplet(
        &self,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) -> Result<u64> {
        let triplet_id = self.triplet_id_alloc.next_id();
        let valid_from_ms = now_ms();

        let op = Op::AddTriplet(AddTripletOp {
            triplet_id,
            subject: subject.clone(),
            predicate: predicate.clone(),
            object: object.clone(),
            weight,
            valid_from_ms,
            source_memory_id,
            source_file: source_file.clone(),
        });
        let _seqno = self.log.write().append(&op)?;

        self.triplet_store.write().replay_add(
            triplet_id,
            subject,
            predicate,
            object,
            weight,
            valid_from_ms,
            source_memory_id,
            source_file,
        );

        Ok(triplet_id)
    }

    /// Set the lifecycle status of a memory (Active/Superseded/Contradicted/Archived).
    /// Durable: writes UpdateState op to WAL.
    pub fn set_memory_status(&self, memory_id: MemoryId, status: crate::state::MemoryStatus) -> Result<()> {
        use crate::state::MemoryStatus;
        let status_u8: u8 = match status {
            MemoryStatus::Active       => 0,
            MemoryStatus::Superseded   => 1,
            MemoryStatus::Contradicted => 2,
            MemoryStatus::Archived     => 3,
            MemoryStatus::Proposed     => 4,
            MemoryStatus::Observed     => 5,
            MemoryStatus::Verified     => 6,
        };
        // Check existence before writing to WAL
        {
            let states = self.states.read();
            if !states.contains_key(&memory_id) {
                return Err(FieldError::NotFound(memory_id));
            }
        }
        let delta = crate::ops::StateDeltaOp {
            memory_id,
            strength_delta: None,
            confidence_delta: None,
            decay_rate: None,
            touch: false,
            pin: None,
            op_ts_ms: now_ms(),
            status: Some(status_u8),
            epistemic_status: None,
            staged: None,
            invalidated_by: None,
        };
        self.log.write().append(&Op::UpdateState(delta.clone()))?;
        let _ = self.log.write().sync(); // status transitions are critical lifecycle events
        let mut states = self.states.write();
        if let Some(st) = states.get_mut(&memory_id) {
            st.apply_delta(&delta, now_ms());
            Ok(())
        } else {
            Err(FieldError::NotFound(memory_id))
        }
    }

    /// Set the epistemic status of a memory (UserStated/ToolDerived/ModelInferred/AutonomousSynthesis).
    /// Durable: writes UpdateState op to WAL.
    pub fn set_epistemic_status(&self, memory_id: MemoryId, es: crate::state::EpistemicStatus) -> Result<()> {
        use crate::state::EpistemicStatus;
        let es_u8: u8 = match es {
            EpistemicStatus::UserStated          => 0,
            EpistemicStatus::ToolDerived         => 1,
            EpistemicStatus::ModelInferred       => 2,
            EpistemicStatus::AutonomousSynthesis => 3,
        };
        // Check existence before writing to WAL
        {
            let states = self.states.read();
            if !states.contains_key(&memory_id) {
                return Err(FieldError::NotFound(memory_id));
            }
        }
        let delta = crate::ops::StateDeltaOp {
            memory_id,
            strength_delta: None,
            confidence_delta: None,
            decay_rate: None,
            touch: false,
            pin: None,
            op_ts_ms: now_ms(),
            status: None,
            epistemic_status: Some(es_u8),
            staged: None,
            invalidated_by: None,
        };
        self.log.write().append(&Op::UpdateState(delta.clone()))?;
        let _ = self.log.write().sync();
        let mut states = self.states.write();
        if let Some(st) = states.get_mut(&memory_id) {
            st.apply_delta(&delta, now_ms());
            Ok(())
        } else {
            Err(FieldError::NotFound(memory_id))
        }
    }

    /// Set affect dimensions on a memory (valence: -1..+1, arousal: 0..1).
    /// In-memory only (not WAL-persisted) — affect is re-derived from content on reload.
    pub fn set_affect(&self, memory_id: MemoryId, valence: f32, arousal: f32) -> Result<()> {
        let mut states = self.states.write();
        if let Some(st) = states.get_mut(&memory_id) {
            st.affect_valence = valence.clamp(-1.0, 1.0);
            st.affect_arousal = arousal.clamp(0.0, 1.0);
            Ok(())
        } else {
            Err(FieldError::NotFound(memory_id))
        }
    }

    /// Set the source_session tag on a memory payload. In-memory only; persisted at next snapshot.
    pub fn set_source_session(&self, memory_id: MemoryId, session_id: &str) -> Result<()> {
        let realm = {
            let mut payloads = self.payloads.write();
            let p = payloads.get_mut(&memory_id).ok_or(FieldError::NotFound(memory_id))?;
            p.source_session = Some(session_id.to_string());
            p.realm.clone()
        };
        // #13 densification: the daemon path puts with session=None and attaches
        // the session here (cf_put_memory carries no session), so this is the
        // moment the SameSession chain can form. Idempotent: skips ids already
        // in the ring, and add_assoc_edge dedups edges by (dst, type).
        self.densify_write_edges(memory_id, &realm, Some(session_id));
        Ok(())
    }

    /// Extract entity seeds from a query string: capitalized words (≥3 chars),
    /// @tag references, and double-quoted strings.
    fn extract_seeds(query: &str) -> Vec<String> {
        let mut seeds: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        // @tag references
        for cap in query.split_whitespace() {
            let w = cap.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '_');
            if w.starts_with('@') && w.len() > 1 {
                let tag = w[1..].to_string();
                if seen.insert(tag.clone()) { seeds.push(tag); }
            }
        }
        // Quoted strings
        let mut in_quote = false;
        let mut buf = String::new();
        for c in query.chars() {
            if c == '"' {
                if in_quote && !buf.trim().is_empty() {
                    let s = buf.trim().to_string();
                    if seen.insert(s.clone()) { seeds.push(s); }
                    buf.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                buf.push(c);
            }
        }
        // Capitalized words ≥3 chars (skip first word of query which may be sentence-start)
        let words: Vec<&str> = query.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let w = word.trim_matches(|c: char| !c.is_alphabetic());
            if w.len() < 3 { continue; }
            let mut chars = w.chars();
            if let Some(first) = chars.next() {
                if first.is_uppercase() && i > 0 {
                    if seen.insert(w.to_string()) { seeds.push(w.to_string()); }
                }
            }
        }
        seeds
    }

    /// Spreading-activation recall: traverse triplet graph from query entities,
    /// return top-k memories ranked by accumulated activation.
    pub fn recall_spreading(
        &self,
        query: &str,
        k: usize,
        realm: Option<&str>,
        max_nodes: usize,
        max_entries_per_entity: usize,
        depth: u8,
    ) -> Vec<SpreadingRecallHit> {
        let seeds = Self::extract_seeds(query);
        if seeds.is_empty() { return Vec::new(); }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let memory_scores = match self.triplet_store.try_read_for(std::time::Duration::from_secs(5)) {
            Some(ts) => ts.spreading_activation(
                &seeds, depth, 0.6, now_ms, max_nodes, max_entries_per_entity),
            None => return Vec::new(),
        };
        if memory_scores.is_empty() { return Vec::new(); }

        // Sort by score descending, take top k
        let mut ranked: Vec<(MemoryId, f32)> = memory_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k * 4); // fetch extra for realm filtering

        let payloads = match self.payloads.try_read_for(std::time::Duration::from_secs(5)) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let mut results: Vec<SpreadingRecallHit> = Vec::new();
        for (mid, score) in ranked {
            if let Some(p) = payloads.get(&mid) {
                if let Some(r) = realm {
                    if p.realm.as_str() != r { continue; }
                }
                results.push(SpreadingRecallHit {
                    memory_id: mid,
                    score,
                    text: String::from_utf8_lossy(&p.content).chars().take(300).collect::<String>(),
                    kind: p.kind.clone(),
                    realm: p.realm.clone(),
                });
                if results.len() >= k { break; }
            }
        }
        results
    }

    /// Invalidate a triplet (marks it as expired at the current time).
    /// Backfill embedding for a memory stored with embed_pending=true.
    /// Durable: writes UpdateMemoryContent op to WAL (content unchanged, new embedding).
    pub fn backfill_embedding(&self, memory_id: MemoryId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim { expected: EMBED_DIM, actual: embedding.len() });
        }
        let existing_content = {
            // Lock order: payloads before states (matches sync_foreign).
            let payloads = self.payloads.read();
            let states = self.states.read();
            match states.get(&memory_id) {
                None => return Err(FieldError::NotFound(memory_id)),
                Some(st) if !st.embed_pending => return Ok(()),
                Some(_) => {
                    payloads.get(&memory_id)
                        .map(|p| p.content.clone())
                        .unwrap_or_default()
                }
            }
        };

        // Persist to WAL via UpdateMemoryContent (empty content = reuse existing)
        let op = Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
            memory_id,
            content: existing_content.clone(),
            embedding: embedding.to_vec(),
            op_ts_ms: now_ms(),
        });
        self.log.write().append(&op)?;

        // Update payload embedding metadata; the vector itself lives only in
        // the semantic index (upserted below) — see ChittaField::embedding_of.
        {
            let mut payloads = self.payloads.write();
            if let Some(p) = payloads.get_mut(&memory_id) {
                p.embedding = Vec::new();
                p.embedding_model = EMBED_MODEL_ID.to_string();
                p.embedding_model_id = EMBED_MODEL_ID.to_string();
                p.embedding_dim = EMBED_DIM as u32;
            }
        }

        // Update semantic index (get realm for per-realm routing)
        let realm_for_upsert = self.payloads.read()
            .get(&memory_id)
            .map(|p| p.realm.clone());

        // Semantic dedup — the check that store.rs:1321 CANNOT run on the normal path.
        // On the production path every memory is written embed_pending with an empty
        // vector, so that write-time guard (`if !embed_pending`) is false and
        // near-duplicates would otherwise accumulate unbounded. (It does still fire for
        // callers that supply their own vector — see the CONSTRAINT note at that site.)
        // This is the first moment the vector exists; search runs BEFORE this
        // memory's own vector is upserted, so k=1 returns the nearest OTHER memory.
        // A hit supersedes THIS (newer) memory rather than deleting it: reversible, keeps
        // provenance/triplet references, excluded from recall, and reinforces the original.
        let mut superseded_by: Option<MemoryId> = None;
        {
            let (dedup_thresh, dedup_upper) = {
                let cfg = &self.scoring_pipeline.read().config;
                (cfg.dedup_cosine_threshold, cfg.dedup_cosine_upper)
            };
            let neighbors = self.semantic_idx.read().search(embedding, 1, None, None);
            if let Some(top) = neighbors.first() {
                if top.memory_id != memory_id
                    && top.cosine_similarity >= dedup_thresh
                    && top.cosine_similarity < dedup_upper
                {
                    let same_realm = self.payloads.read()
                        .get(&top.memory_id)
                        .map(|p| Some(&p.realm) == realm_for_upsert.as_ref())
                        .unwrap_or(false);
                    let alive = self.states.read()
                        .get(&top.memory_id)
                        .map(|s| !s.deleted)
                        .unwrap_or(false);
                    if same_realm && alive {
                        superseded_by = Some(top.memory_id);
                    }
                }
            }
        }

        self.semantic_idx.write().upsert(
            memory_id,
            embedding.to_vec(),
            realm_for_upsert.as_deref(),
        );

        if let Some(original) = superseded_by {
            let _ = self.set_memory_status(memory_id, crate::state::MemoryStatus::Superseded);
            let _ = self.update_state(original, Some(0.0), Some(0.02), None, true, None);
        }

        // Re-encode cortical sparse index (non-fatal)
        let _ = self.encode_memory(memory_id);

        // Clear embed_pending in state
        {
            let mut states = self.states.write();
            if let Some(st) = states.get_mut(&memory_id) {
                if st.embed_pending {
                    st.embed_pending = false;
                    self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    // ── Deferred batched backfill (Step-1 re-architecture) ─────────────────────
    //
    // Splits the per-item `backfill_embedding` (WAL + payload + O(log N) HNSW insert,
    // all under one `semantic_idx.write()`) into three phases so the expensive HNSW
    // neighbor search runs OFF the write lock:
    //
    //   stage(items) — WAL + payload meta + metadata upsert (coarse/LSH/binary/embeddings
    //                  map), but NOT the global HNSW insert. Brief write. MUST be called
    //                  under the C++ rpc_mutex (acquire_lock), same as backfill_embedding.
    //   plan()       — compute HNSW insert plans under `semantic_idx.read()` (concurrent
    //                  with recall). MUST be called WITHOUT the rpc_mutex, so recall (which
    //                  needs the shared rpc_mutex) is not blocked while the search runs.
    //   apply()      — wire the plans into the graph + clear embed_pending. Brief write.
    //                  MUST be called under the rpc_mutex — preserves the rpc→semantic_idx
    //                  lock order, so no new ABBA vs sync_foreign.
    //
    // Correctness: the durable state (WAL + payload) lands in stage(); a crash between
    // stage and apply just leaves the memory embed_pending (re-picked next drain) — no
    // torn graph, no lost write. A concurrent writer inserting between plan and apply is
    // untouched by apply (see apply_delta_batch). Recall parity: id-seeded levels match
    // the load-time backfill_hnsw_delta_parallel path, so the resulting graph is
    // build-order-independent and equivalent to the per-item path.

    /// Phase 1: durable stage of a backfill batch. Returns the number of ids that
    /// still need a global-HNSW plan (drives whether `plan()` has work to do).
    pub fn backfill_stage(&self, items: &[(MemoryId, Vec<f32>)]) -> Result<usize> {
        // Pass 1 (WAL + payload meta): collect the valid (id, embedding, realm) tuples.
        let mut valid: Vec<(MemoryId, &Vec<f32>, Option<String>)> = Vec::with_capacity(items.len());
        for (memory_id, embedding) in items {
            let memory_id = *memory_id;
            if embedding.len() != EMBED_DIM {
                return Err(FieldError::InvalidEmbedDim { expected: EMBED_DIM, actual: embedding.len() });
            }
            // Skip ids that are gone or already embedded (mirrors backfill_embedding).
            let existing_content = {
                let payloads = self.payloads.read();
                let states = self.states.read();
                match states.get(&memory_id) {
                    None => continue,
                    Some(st) if !st.embed_pending => continue,
                    Some(_) => payloads.get(&memory_id).map(|p| p.content.clone()).unwrap_or_default(),
                }
            };
            // WAL: persist the embedding via UpdateMemoryContent (empty content reuses existing).
            let op = Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
                memory_id,
                content: existing_content,
                embedding: embedding.clone(),
                op_ts_ms: now_ms(),
            });
            self.log.write().append(&op)?;
            // Payload embedding metadata — the vector itself lives only in the semantic index.
            let realm_for_upsert = {
                let mut payloads = self.payloads.write();
                if let Some(p) = payloads.get_mut(&memory_id) {
                    p.embedding = Vec::new();
                    p.embedding_model = EMBED_MODEL_ID.to_string();
                    p.embedding_model_id = EMBED_MODEL_ID.to_string();
                    p.embedding_dim = EMBED_DIM as u32;
                    Some(p.realm.clone())
                } else {
                    None
                }
            };
            valid.push((memory_id, embedding, realm_for_upsert));
        }
        self.log.write().flush_buf().ok();
        // Pass 2 (metadata upsert): ONE semantic_idx.write() for the whole batch, not one
        // per item — a single acquisition means a waiting recall reader starves for one
        // brief metadata write, not the full burst (parking_lot writer-preference would
        // otherwise let per-item re-acquisition starve readers across the whole chunk).
        let mut staged_ids: Vec<MemoryId> = Vec::with_capacity(valid.len());
        let mut plan_ids: Vec<MemoryId> = Vec::new();
        {
            let mut idx = self.semantic_idx.write();
            for (memory_id, embedding, realm) in &valid {
                let needs_plan = idx.upsert_deferred(*memory_id, (*embedding).clone(), realm.as_deref());
                staged_ids.push(*memory_id);
                if needs_plan { plan_ids.push(*memory_id); }
            }
        }
        let n_plan = plan_ids.len();
        *self.backfill_plan_stage.lock() = Some(BackfillStage { staged_ids, plan_ids, plan: None });
        Ok(n_plan)
    }

    /// Phase 2: compute HNSW insert plans off the write lock. No-op if nothing staged.
    /// Call WITHOUT the C++ rpc_mutex so recall is not blocked during the search.
    pub fn backfill_plan(&self) {
        let plan_ids = {
            let stage = self.backfill_plan_stage.lock();
            match stage.as_ref() {
                Some(s) if !s.plan_ids.is_empty() => s.plan_ids.clone(),
                _ => return,
            }
        };
        let plan = self.semantic_idx.read().plan_delta_batch(&plan_ids);
        if let Some(s) = self.backfill_plan_stage.lock().as_mut() {
            s.plan = Some(plan);
        }
    }

    /// Phase 3: wire the staged plan into the graph and clear embed_pending. Returns the
    /// number of memories whose embed_pending was cleared. Call under the C++ rpc_mutex.
    pub fn backfill_apply(&self) -> usize {
        let stage = match self.backfill_plan_stage.lock().take() {
            Some(s) => s,
            None => return 0,
        };
        if let Some(plan) = stage.plan {
            self.semantic_idx.write().apply_delta_batch(plan);
        }
        for &id in &stage.staged_ids {
            let _ = self.encode_memory(id);
        }
        let mut states = self.states.write();
        for &id in &stage.staged_ids {
            if let Some(st) = states.get_mut(&id) {
                if st.embed_pending {
                    st.embed_pending = false;
                    self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        stage.staged_ids.len()
    }

    /// Convenience: run the full stage → plan → apply sequence in one call. Used by the
    /// lock-free embed_loop drain and by tests. Does NOT take the C++ rpc_mutex, so it
    /// relies solely on the internal semantic_idx lock discipline (safe for the embed_loop
    /// path, which never held acquire_lock). Returns the number of memories backfilled.
    pub fn backfill_embeddings_batch(&self, items: &[(MemoryId, Vec<f32>)]) -> Result<usize> {
        self.backfill_stage(items)?;
        self.backfill_plan();
        Ok(self.backfill_apply())
    }

    /// Persist the live delta HNSW tier to its sidecar so the NEXT restart LOADS the
    /// delta graph instead of cold-reinserting every runtime-added node from scratch
    /// (single-threaded, ~minutes once the corpus grows). Pure optimization, never a
    /// hard dependency: on startup a missing/stale/unreadable delta.hnsw silently falls
    /// back to the existing backfill/rebuild path (normalize_all's hnsw_len==total gate).
    ///
    /// Writes to the DETERMINISTIC snapshot stem this instance loads on open
    /// (`chitta.<instance>.delta.hnsw`, mirroring save_full_snapshot's path at
    /// store.rs:7639), via save_delta_hnsw's write-temp + atomic-rename (NFS-safe on
    /// Isilon — no delete-then-write). The n==0 guard skips the write entirely when the
    /// delta is empty, avoiding save_delta_hnsw's remove_file branch (an NFS delete).
    ///
    /// Takes only semantic_idx.read() (like a recall) — safe to call OFF the rpc_mutex,
    /// so it never blocks live recalls. Returns the number of delta nodes persisted
    /// (0 = nothing to persist or write failed; both are non-fatal).
    pub fn persist_delta_hnsw(&self) -> usize {
        let idx = self.semantic_idx.read();
        let n = idx.delta_len();
        if n == 0 {
            return 0;
        }
        let path = self
            .data_dir
            .join(format!("chitta.{:08x}.delta.hnsw", self.instance_id));
        if let Err(e) = idx.save_delta_hnsw(&path) {
            eprintln!("[chitta-field] WARNING: persist_delta_hnsw failed: {e}");
            return 0;
        }
        n
    }

    /// Force a full rebuild of all derived search structures (binary codes, coarse,
    /// LSH, HNSW) from the current embeddings. Required after an embedding-dimension
    /// migration, where re-embedding updates the float vectors but leaves the ANN
    /// indices built in the old vector space.
    pub fn force_reindex(&self) {
        self.semantic_idx.write().force_reindex();
    }

    /// Return memory IDs with embed_pending=true, sorted oldest first, up to limit.
    pub fn pending_embeddings(&self, limit: usize) -> Vec<MemoryId> {
        let s = self.states.read();
        let mut pending: Vec<(i64, MemoryId)> = s
            .iter()
            .filter(|(_, st)| st.embed_pending && !st.deleted)
            .map(|(id, st)| (st.created_at_ms, *id))
            .collect();
        pending.sort_by_key(|(ts, _)| *ts); // oldest first
        if pending.len() <= limit {
            return pending.into_iter().map(|(_, id)| id).collect();
        }
        // Backlog larger than one batch: embed the NEWEST (limit - half) first so a
        // just-written memory becomes recallable within ONE batch instead of waiting behind
        // the entire backlog (the write->recall lag), while still draining the OLDEST `half`
        // each batch so nothing starves. Minimises lag without dropping any pending memory.
        let half = limit / 2;
        let newest: Vec<MemoryId> =
            pending.iter().rev().take(limit - half).map(|(_, id)| *id).collect();
        let oldest: Vec<MemoryId> =
            pending.iter().take(half).map(|(_, id)| *id).collect();
        newest.into_iter().chain(oldest).collect()
    }

    /// Clear embed_pending for specific memory IDs, regardless of content.
    /// Returns count actually cleared (skips IDs not in pending state).
    pub fn purge_orphan_embed_pending(&self) -> usize {
        // Collect all embed_pending IDs
        let pending: Vec<MemoryId> = {
            let states = self.states.read();
            states.iter()
                .filter(|(_, st)| st.embed_pending && !st.deleted)
                .map(|(id, _)| *id)
                .collect()
        };
        if pending.is_empty() { return 0; }

        // Check which ones get_memory() would fail for (not in payloads or error)
        let to_clear: Vec<MemoryId> = pending.iter()
            .filter(|id| self.get_memory(**id).is_err())
            .copied()
            .collect();

        let n = to_clear.len();
        if n > 0 {
            let mut states = self.states.write();
            for id in &to_clear {
                if let Some(st) = states.get_mut(id) {
                    if st.embed_pending {
                        st.embed_pending = false;
                        self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
        n
    }

    /// Semantic-index coverage: (eligible, embedded).
    ///
    /// `eligible` = live memories with content long enough to embed — the ones that SHOULD
    /// have a vector. `embedded` = those the index actually holds one for. The gap is the
    /// number of memories reachable by BM25 only, and no ranker change can ever surface them.
    ///
    /// This exists because `pending_count` CANNOT answer the question: the embed queue is
    /// drained by failure paths as well as success paths, so a lost embedding leaves the queue
    /// empty and the store reports perfect health over a half-empty index. Derive coverage from
    /// the index; never infer it from the queue.
    /// The stored vector for a memory, exactly as the index holds it.
    /// Nothing else exposes this — which is why a document embedded into the wrong
    /// vector space has been undetectable from outside the process.
    pub fn get_embedding(&self, id: MemoryId) -> Option<Vec<f32>> {
        self.semantic_idx.read().get_embedding(id).map(|v| v.to_vec())
    }

    pub fn embed_coverage(&self) -> (usize, usize) {
        const MIN_EMBED_CHARS: usize = 20;
        // Locks are taken one at a time, never nested (get_memory() re-locks `states`).
        let live: Vec<MemoryId> = {
            let states = self.states.read();
            states.iter().filter(|(_, st)| !st.deleted).map(|(id, _)| *id).collect()
        };
        let eligible: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            live.into_iter()
                .filter(|id| payloads.get(id)
                    .map(|p| p.content.len() >= MIN_EMBED_CHARS)
                    .unwrap_or(false))
                .collect()
        };
        let idx = self.semantic_idx.read();
        let embedded = eligible.iter().filter(|id| idx.has_embedding(**id)).count();
        (eligible.len(), embedded)
    }

    /// Force-clear embed_pending for specific IDs (maintenance tool).
    pub fn force_clear_embed_pending(&self, ids: &[MemoryId]) -> usize {
        let mut states = self.states.write();
        let mut n = 0usize;
        for id in ids {
            if let Some(st) = states.get_mut(id) {
                if st.embed_pending {
                    st.embed_pending = false;
                    self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    n += 1;
                }
            }
        }
        n
    }

    /// Re-queue memories that have no embedding but are embeddable (ghost backfill failures).
    /// A ghost has embed_pending=false, embedding_model="none", empty embedding, and content
    /// long enough for BGE. Sets embed_pending=true so the backfill thread picks them up.
    /// Returns count of memories re-queued.
    pub fn requeue_ghost_embeddings(&self) -> usize {
        const MIN_EMBED_CHARS: usize = 20;

        // Pass 1: collect candidates — states+payloads locked briefly, then released.
        let candidates: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states   = self.states.read();
            states.iter()
                .filter(|(id, st)| {
                    !st.deleted && !st.embed_pending &&
                    payloads.get(id)
                        .map(|p| p.content.len() >= MIN_EMBED_CHARS)
                        .unwrap_or(false)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        if candidates.is_empty() { return 0; }

        // Pass 2: filter to those absent from HNSW — semantic_idx locked briefly, then released.
        let ghost_ids: Vec<MemoryId> = {
            let idx = self.semantic_idx.read();
            candidates.into_iter().filter(|id| !idx.contains(*id)).collect()
        };
        if ghost_ids.is_empty() { return 0; }

        // Pass 3: mark ghosts as embed_pending — states write-locked briefly.
        let mut states = self.states.write();
        let mut count = 0usize;
        for id in &ghost_ids {
            if let Some(st) = states.get_mut(id) {
                if !st.embed_pending && !st.deleted {
                    st.embed_pending = true;
                    self.pending_embed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    count += 1;
                }
            }
        }
        count
    }

    /// Mark every non-deleted memory with sufficient content as `embed_pending = true`
    /// so the backfill thread re-embeds them with the new model (e.g. after a dim change).
    /// Returns the count of memories marked.
    pub fn requeue_all_embeddings(&self, _model_id: &str) -> Result<usize> {
        const MIN_EMBED_CHARS: usize = 10;

        // Pass 1: collect candidates — brief shared lock on both maps.
        let candidates: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states   = self.states.read();
            states.iter()
                .filter(|(id, st)| {
                    !st.deleted &&
                    payloads.get(id)
                        .map(|p| p.content.len() >= MIN_EMBED_CHARS)
                        .unwrap_or(false)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        if candidates.is_empty() { return Ok(0); }

        // Pass 2: mark as embed_pending — exclusive write lock.
        let mut states = self.states.write();
        let mut count = 0usize;
        for id in &candidates {
            if let Some(st) = states.get_mut(id) {
                if !st.deleted {
                    if !st.embed_pending {
                        st.embed_pending = true;
                        self.pending_embed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Remove triplet by subject+predicate+object (invalidates first matching entry).
    pub fn forget_triplet(&self, subject: &str, predicate: &str, object: &str) -> Result<bool> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let matches: Vec<u64> = store
            .query_subject(subject, at_ms)
            .into_iter()
            .filter(|t| t.predicate == predicate && t.object == object)
            .map(|t| t.id)
            .collect();
        drop(store);
        for id in &matches {
            self.invalidate_triplet(*id)?;
        }
        Ok(!matches.is_empty())
    }

    pub fn invalidate_triplet(&self, triplet_id: u64) -> Result<()> {
        let now = now_ms();
        let op = Op::InvalidateTriplet(InvalidateTripletOp {
            triplet_id,
            invalidated_at_ms: now,
        });
        let _seqno = self.log.write().append(&op)?;

        self.triplet_store.write().invalidate(triplet_id, now);

        Ok(())
    }

    /// Query all currently-valid triplets with the given subject.
    pub fn query_subject(&self, subject: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store
            .query_subject(subject, at_ms)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Query all currently-valid triplets with the given object.
    pub fn query_object(&self, object: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store
            .query_object(object, at_ms)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Query all currently-valid triplets where subject OR object matches.
    pub fn query_entity(&self, entity: &str) -> Result<Vec<TripletEntry>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        Ok(store
            .query_entity(entity, at_ms)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Query triplets about `subject` valid in the world at `world_ms`, excluding superseded.
    pub fn query_subject_as_of(&self, subject: &str, world_ms: i64) -> Result<Vec<TripletEntry>> {
        let store = self.triplet_store.read();
        Ok(store.query_as_of(subject, world_ms).into_iter().cloned().collect())
    }

    /// Query what the agent believed about `subject` at ingestion-time `ingest_ms`.
    pub fn query_subject_believed_at(&self, subject: &str, ingest_ms: i64) -> Result<Vec<TripletEntry>> {
        let store = self.triplet_store.read();
        Ok(store.query_believed_at(subject, ingest_ms).into_iter().cloned().collect())
    }

    /// Mark triplet `old_id` as superseded by `new_id` at `at_ms`.
    /// Durable: writes a SupersedeTriplet op to the WAL, then applies in-memory.
    /// Was RAM-only (persisted solely via the .sup.json snapshot sidecar), so a
    /// segment-only replay silently lost the revision — this restores THEORY.md
    /// invariant #1 (state = f(op set)) for belief revision.
    pub fn triplet_supersede(&self, old_id: u64, new_id: u64, at_ms: i64) -> Result<()> {
        let op = Op::SupersedeTriplet(crate::ops::SupersedeTripletOp {
            old_id,
            new_id,
            superseded_at_ms: at_ms,
        });
        let _seqno = self.log.write().append(&op)?;
        self.triplet_store.write().supersede(old_id, new_id, at_ms);
        Ok(())
    }

    /// BFS graph traversal from `start` node.
    pub fn graph_traverse(
        &self,
        start: &str,
        edge_types: &[&str],
        max_hops: usize,
        max_results: usize,
        direction: crate::graph::Direction,
        max_edges: usize,
    ) -> Vec<crate::graph::TraversalHit> {
        self.triplet_store.read().graph_traverse(
            start, edge_types, max_hops, max_results, direction, max_edges)
    }

    /// Personalized PageRank over the triplet graph.
    pub fn graph_pagerank(
        &self,
        seeds: &[&str],
        edge_types: &[&str],
        damping: f32,
        iterations: u8,
        top_k: usize,
        max_nodes: usize,
        max_edges: usize,
    ) -> Vec<(String, f32)> {
        self.triplet_store.read().graph_pagerank(
            seeds, edge_types, damping, iterations, top_k, max_nodes, max_edges)
    }

    /// Get memory IDs that contradict the given memory (bidirectional).
    pub fn get_conflicts(&self, memory_id: MemoryId) -> Result<Vec<MemoryId>> {
        let id_str = memory_id.to_string();
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let mut result = Vec::new();
        for entry in store.query_subject(&id_str, at_ms) {
            if entry.predicate == "contradicts" {
                if let Ok(id) = entry.object.parse::<u64>() {
                    result.push(id);
                }
            }
        }
        for entry in store.query_object(&id_str, at_ms) {
            if entry.predicate == "contradicts" {
                if let Ok(id) = entry.subject.parse::<u64>() {
                    result.push(id);
                }
            }
        }
        Ok(result)
    }

    /// Follow "supersedes" edges to build the full supersession chain.
    /// Chain starts with memory_id itself. Max depth 20, cycle-safe.
    pub fn get_supersession_chain(&self, memory_id: MemoryId) -> Result<Vec<MemoryId>> {
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let mut chain = vec![memory_id];
        let mut visited = std::collections::HashSet::new();
        visited.insert(memory_id);
        let mut current = memory_id;
        for _ in 0..20 {
            let id_str = current.to_string();
            let next = store
                .query_object(&id_str, at_ms)
                .into_iter()
                .find(|e| e.predicate == "supersedes")
                .and_then(|e| e.subject.parse::<u64>().ok());
            match next {
                Some(n) if !visited.contains(&n) => {
                    visited.insert(n);
                    chain.push(n);
                    current = n;
                }
                _ => break,
            }
        }
        Ok(chain)
    }

    /// Get memory IDs that confirm the given memory.
    pub fn get_confirmations(&self, memory_id: MemoryId) -> Result<Vec<MemoryId>> {
        let id_str = memory_id.to_string();
        let at_ms = now_ms();
        let store = self.triplet_store.read();
        let mut result = Vec::new();
        for entry in store.query_object(&id_str, at_ms) {
            if entry.predicate == "confirms" {
                if let Ok(id) = entry.subject.parse::<u64>() {
                    result.push(id);
                }
            }
        }
        Ok(result)
    }

    /// Apply feedback for a recall episode (route learning).
    pub fn feedback(&self, episode_id: u64, reward: f32) -> Result<()> {
        self.learners.write().route.feedback(episode_id, reward);
        Ok(())
    }

    /// Get recommended window size for a session type.
    pub fn recommended_window(&self, session_type: &str) -> usize {
        self.learners
            .read()
            .context
            .recommended_window(session_type)
    }

    /// Record context outcome for a session type and window size.
    pub fn record_context_outcome(&self, session_type: &str, size: usize, outcome: f32) {
        self.learners
            .write()
            .context
            .record_outcome(session_type, size, outcome);
    }

    /// Select a retrieval route using Thompson sampling. Returns (episode_id, route).
    pub fn select_route(&self, query: &str) -> (u64, Route) {
        let intent = RouteLearner::detect_intent(query);
        let now_ms = now_ms() as u64;
        self.learners.write().route.select_route(intent, now_ms)
    }

    // ── Code Intelligence ────────────────────────────────────────────────────

    /// Upsert a symbol. Deduplicates by (kind, name, file_path, line_start).
    /// Returns the SymbolId.
    pub fn upsert_symbol(
        &self,
        kind: &str,
        name: &str,
        signature: &str,
        file_path: &str,
        line_start: u32,
        line_end: u32,
        repo_id: u64,
        embedding: &[f32],
        description: Option<String>,
        memory_id: Option<MemoryId>,
    ) -> Result<u64> {
        let symbol_id = self.symbol_id_alloc.next_id();
        let op = Op::UpsertSymbol(UpsertSymbolOp {
            symbol_id,
            kind: kind.to_string(),
            name: name.to_string(),
            signature: signature.to_string(),
            file_path: file_path.to_string(),
            line_start,
            line_end,
            repo_id,
            embedding: embedding.to_vec(),
            description: description.clone(),
            memory_id,
        });
        let _seqno = self.log.write().append(&op)?;

        let entry = SymbolEntry {
            id: symbol_id,
            kind: kind.to_string(),
            name: name.to_string(),
            signature: signature.to_string(),
            file_path: file_path.to_string(),
            line_start,
            line_end,
            repo_id,
            embedding: embedding.to_vec(),
            description,
            memory_id,
        };
        let actual_id = self.symbol_idx.write().upsert(entry);
        Ok(actual_id)
    }

    /// Remove a symbol and all its call edges.
    pub fn remove_symbol(&self, symbol_id: u64) -> Result<()> {
        let op = Op::RemoveSymbol(RemoveSymbolOp { symbol_id });
        let _seqno = self.log.write().append(&op)?;

        self.symbol_idx.write().remove(symbol_id);
        self.call_graph.write().remove_symbol(symbol_id);
        Ok(())
    }

    /// Get a symbol by ID.
    pub fn get_symbol(&self, symbol_id: u64) -> Result<Option<SymbolEntry>> {
        Ok(self.symbol_idx.read().get(symbol_id).cloned())
    }

    /// Search symbols by name (exact or prefix match).
    pub fn search_symbols_by_name(&self, query: &str, limit: usize) -> Vec<SymbolEntry> {
        self.search_symbols_by_name_scoped(query, limit, None)
    }

    /// Search symbols by name, optionally restricted to file paths containing
    /// `path_filter` (repo/directory scoping).
    pub fn search_symbols_by_name_scoped(
        &self,
        query: &str,
        limit: usize,
        path_filter: Option<&str>,
    ) -> Vec<SymbolEntry> {
        self.symbol_idx
            .read()
            .search_by_name_scoped(query, limit, path_filter)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Remove every symbol in a file (per-file invalidation before re-extract).
    /// Each removal is WAL-logged via Op::RemoveSymbol, so replay converges.
    pub fn remove_symbols_by_file(&self, file_path: &str) -> Result<usize> {
        let ids: Vec<u64> = self
            .symbol_idx
            .read()
            .by_file(file_path)
            .iter()
            .map(|e| e.id)
            .collect();
        for &id in &ids {
            self.remove_symbol(id)?;
        }
        Ok(ids.len())
    }

    /// GC the symbol index: stale-line duplicates, excluded paths (plugin
    /// caches, build deps), and dead files. Dry-run returns what WOULD be
    /// removed. Removals go through Op::RemoveSymbol (WAL-durable, reversible
    /// only via re-index — callers should snapshot first).
    pub fn dedupe_symbols(
        &self,
        dry_run: bool,
        check_fs: bool,
        path_excludes: &[String],
    ) -> Result<serde_json::Value> {
        let (cand, total) = {
            let idx = self.symbol_idx.read();
            (idx.collect_gc_candidates(check_fs, path_excludes), idx.count())
        };
        let would_remove = cand.dup.len() + cand.excluded.len() + cand.dead.len();
        let mut removed = 0usize;
        if !dry_run {
            for id in cand
                .dup
                .iter()
                .chain(cand.excluded.iter())
                .chain(cand.dead.iter())
            {
                self.remove_symbol(*id)?;
                removed += 1;
            }
        }
        Ok(serde_json::json!({
            "total": total,
            "dup": cand.dup.len(),
            "excluded_path": cand.excluded.len(),
            "dead_path": cand.dead.len(),
            "would_remove": would_remove,
            "removed": removed,
            "dry_run": dry_run,
            "top_dirs": cand.top_dirs.iter()
                .map(|(d, n)| serde_json::json!({"dir": d, "count": n}))
                .collect::<Vec<_>>(),
        }))
    }

    /// Semantic symbol search: find k nearest by cosine similarity.
    pub fn search_symbols_semantic(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.symbol_idx.read().search_semantic(query, k)
    }

    /// Get all symbols in a file.
    pub fn symbols_in_file(&self, file_path: &str) -> Vec<SymbolEntry> {
        self.symbol_idx
            .read()
            .by_file(file_path)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Add a call edge between two symbols (idempotent).
    pub fn add_call_edge(&self, caller_id: u64, callee_id: u64) -> Result<()> {
        let op = Op::AddSymCallEdge(AddSymCallEdgeOp {
            caller_id,
            callee_id,
        });
        let _seqno = self.log.write().append(&op)?;
        self.call_graph.write().add_edge(caller_id, callee_id);
        Ok(())
    }

    /// Get symbols called by the given symbol.
    pub fn get_callees(&self, symbol_id: u64) -> Vec<u64> {
        self.call_graph.read().get_callees(symbol_id)
    }

    /// Get symbols that call the given symbol.
    pub fn get_callers(&self, symbol_id: u64) -> Vec<u64> {
        self.call_graph.read().get_callers(symbol_id)
    }

    /// Upsert a code file record. Returns (CodeFileId, was_updated).
    /// `was_updated` is true when content_hash changed or was absent.
    /// WAL is written before the in-memory update to ensure crash consistency.
    pub fn upsert_code_file(
        &self,
        path: &str,
        project: &str,
        mtime: i64,
        content_hash: Option<String>,
        git_commit: Option<String>,
        git_author: Option<String>,
        git_timestamp_ms: Option<i64>,
    ) -> Result<(u64, bool)> {
        let existing_id = self.code_files.read().get_by_path(path).map(|f| f.id);
        let file_id = existing_id.unwrap_or_else(|| self.code_file_id_alloc.next_id());

        let op = Op::UpsertCodeFile(UpsertCodeFileOp {
            file_id,
            path: path.to_string(),
            project: project.to_string(),
            mtime,
            content_hash: content_hash.clone(),
            git_commit: git_commit.clone(),
            git_author: git_author.clone(),
            git_timestamp_ms,
        });
        let _seqno = self.log.write().append(&op)?;

        let (id, was_updated) = self.code_files.write().upsert(
            path, project, mtime,
            content_hash, git_commit,
            git_author, git_timestamp_ms,
            || file_id,
        );
        Ok((id, was_updated))
    }

    /// Invalidate all active triplets associated with a source file.
    /// Returns the IDs of invalidated triplets.
    pub fn invalidate_triplets_by_source_file(&self, source_file: &str) -> Result<Vec<u64>> {
        let now = now_ms();
        let ids = self.triplet_store.write().invalidate_by_source_file(source_file, now);
        let op = Op::InvalidateTripletsBySourceFile(
            crate::ops::InvalidateTripletsBySourceFileOp {
                source_file: source_file.to_string(),
                invalidated_at_ms: now,
            },
        );
        let _seqno = self.log.write().append(&op)?;
        Ok(ids)
    }

    pub fn symbol_count(&self) -> usize {
        self.symbol_idx.read().count()
    }

    pub fn code_file_count(&self) -> usize {
        self.code_files.read().count()
    }

    pub fn cortical_count(&self) -> usize {
        self.cortical_idx.read().len()
    }

    pub fn prototype_count(&self) -> usize {
        self.cortical_idx.read().prototype_count()
    }

    /// Per-realm embedding geometry stats (inspired by "Geometry of Forgetting").
    /// Returns JSON: `{"by_realm": [...], "by_kind": [...], "anomalies": [...]}`
    pub fn spectral_stats_by_realm(&self) -> String {
        let realm_stats = self.realm_stats.read();
        let kind_stats  = self.kind_stats.read();

        let mut realm_results: Vec<serde_json::Value> = realm_stats
            .iter()
            .filter_map(|(name, stats)| stats.geometry(name))
            .collect();
        realm_results.sort_by(|a, b| {
            b["count"].as_u64().unwrap_or(0).cmp(&a["count"].as_u64().unwrap_or(0))
        });

        let mut kind_results: Vec<serde_json::Value> = kind_stats
            .iter()
            .filter_map(|(name, stats)| stats.geometry(name))
            .collect();
        kind_results.sort_by(|a, b| {
            a["group"].as_str().unwrap_or("").cmp(b["group"].as_str().unwrap_or(""))
        });

        let mut anomalies: Vec<serde_json::Value> = Vec::new();
        for entry in realm_results.iter().chain(kind_results.iter()) {
            let label = entry["group"].as_str().unwrap_or("?");
            let cos   = entry["mean_cosine_sim"].as_f64().unwrap_or(0.0);
            let iso   = entry["isotropy"].as_f64().unwrap_or(1.0);
            let count = entry["count"].as_u64().unwrap_or(0);
            let has_newline = label.contains('\n') || label.contains('\r');
            if cos > 0.95 && count >= 5 {
                anomalies.push(serde_json::json!({
                    "group": label, "issue": "high_similarity",
                    "detail": format!("cos={:.3} across {} memories — likely duplicates", cos, count)
                }));
            }
            if iso < 0.3 && count >= 5 {
                anomalies.push(serde_json::json!({
                    "group": label, "issue": "collapsed_embeddings",
                    "detail": format!("isotropy={:.3} — embeddings occupy narrow subspace", iso)
                }));
            }
            if has_newline {
                anomalies.push(serde_json::json!({
                    "group": label.trim(), "issue": "dirty_realm_name",
                    "detail": "realm contains trailing whitespace/newline"
                }));
            }
        }

        serde_json::to_string(&serde_json::json!({
            "by_realm": realm_results,
            "by_kind":  kind_results,
            "anomalies": anomalies,
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Fix realm names that contain trailing whitespace/newlines.
    /// Returns the number of memories whose realm was trimmed.
    pub fn trim_realm_names(&self) -> usize {
        // Collect dirty memories: (memory_id, old_realm, trimmed_realm)
        let dirty: Vec<(MemoryId, String, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter_map(|(mid, p)| {
                    if states.get(mid).map(|s| s.deleted).unwrap_or(true) {
                        return None;
                    }
                    let trimmed = p.realm.trim().to_string();
                    if trimmed != p.realm {
                        Some((*mid, p.realm.clone(), trimmed))
                    } else {
                        None
                    }
                })
                .collect()
        };

        let count = dirty.len();
        for (mid, old_realm, new_realm) in dirty {
            // Update payload realm
            if let Some(p) = self.payloads.write().get_mut(&mid) {
                p.realm = new_realm.clone();
            }
            // Update realm_members index
            let mut rm = self.realm_members.write();
            if let Some(set) = rm.get_mut(&old_realm) {
                set.remove(&mid);
                if set.is_empty() {
                    rm.remove(&old_realm);
                }
            }
            rm.entry(new_realm).or_default().insert(mid);
        }
        count
    }

    /// Bulk-remap realms per an explicit `{old_realm -> new_realm}` mapping.
    /// `dry_run` computes the move census without mutating. On a real run it
    /// updates payload.realm + realm_members, then forces a full snapshot for
    /// durability (cf_set_realm emits no WAL, so the snapshot is the only durable
    /// record). Returns a JSON summary string.
    pub fn remap_realms(
        &self,
        mapping: &std::collections::HashMap<String, String>,
        dry_run: bool,
    ) -> String {
        let moves: Vec<(MemoryId, String, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter_map(|(mid, p)| {
                    if states.get(mid).map(|s| s.deleted).unwrap_or(true) {
                        return None;
                    }
                    match mapping.get(&p.realm) {
                        Some(new_realm) if new_realm != &p.realm => {
                            Some((*mid, p.realm.clone(), new_realm.clone()))
                        }
                        _ => None,
                    }
                })
                .collect()
        };
        let moved = moves.len();
        let mut per_target: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (_, _, new_realm) in &moves {
            *per_target.entry(new_realm.clone()).or_insert(0) += 1;
        }

        if !dry_run {
            for (mid, old_realm, new_realm) in &moves {
                if let Some(p) = self.payloads.write().get_mut(mid) {
                    p.realm = new_realm.clone();
                }
                let mut rm = self.realm_members.write();
                if let Some(set) = rm.get_mut(old_realm) {
                    set.remove(mid);
                    if set.is_empty() {
                        rm.remove(old_realm);
                    }
                }
                rm.entry(new_realm.clone()).or_default().insert(*mid);
            }
        }

        let final_realms = self.realm_members.read().len();
        let snapshot_ok = if !dry_run {
            self.save_full_snapshot().is_ok()
        } else {
            false
        };
        let mut targets: Vec<(String, usize)> = per_target.into_iter().collect();
        targets.sort_by(|a, b| b.1.cmp(&a.1));
        let targets_json: Vec<serde_json::Value> = targets
            .iter()
            .map(|(t, c)| serde_json::json!({"realm": t, "count": c}))
            .collect();
        serde_json::json!({
            "dry_run": dry_run,
            "moved": moved,
            "targets": targets_json,
            "final_realms": final_realms,
            "snapshot_ok": snapshot_ok,
        })
        .to_string()
    }

    /// Save a spectral stats snapshot for temporal drift tracking.
    /// Writes `spectral_snapshot_{timestamp}.json` to the data dir.
    pub fn save_spectral_snapshot(&self) -> Result<String> {
        let stats_json = self.spectral_stats_by_realm();
        let ts = now_ms();
        let filename = format!("spectral_snapshot_{}.json", ts);
        let path = self.data_dir.join(&filename);
        let wrapped = serde_json::json!({
            "ts_ms": ts,
            "stats": serde_json::from_str::<serde_json::Value>(&stats_json).unwrap_or_default(),
        });
        let content = serde_json::to_string_pretty(&wrapped)
            .map_err(|e| FieldError::Serialization(e.to_string()))?;
        std::fs::write(&path, content)
            .map_err(FieldError::Io)?;
        Ok(filename)
    }

    /// Load spectral drift: compare current stats with most recent snapshot.
    /// Returns JSON with per-realm/kind delta for isotropy and mean_cosine_sim.
    pub fn spectral_drift(&self) -> String {
        // Find most recent snapshot
        let entries = match std::fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(_) => return "{}".to_string(),
        };
        let mut snapshots: Vec<(i64, std::path::PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("spectral_snapshot_") && name.ends_with(".json") {
                    let ts_str = name
                        .strip_prefix("spectral_snapshot_")?
                        .strip_suffix(".json")?;
                    let ts: i64 = ts_str.parse().ok()?;
                    Some((ts, e.path()))
                } else {
                    None
                }
            })
            .collect();
        snapshots.sort_by_key(|(ts, _)| -*ts);

        let prev_snap = match snapshots.first() {
            Some((_, path)) => {
                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => return "{}".to_string(),
                };
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(v) => v,
                    Err(_) => return "{}".to_string(),
                }
            }
            None => return serde_json::json!({"error": "no previous snapshot"}).to_string(),
        };

        let prev_ts = prev_snap["ts_ms"].as_i64().unwrap_or(0);
        let prev_stats = &prev_snap["stats"];

        // Current stats
        let current_json = self.spectral_stats_by_realm();
        let current: serde_json::Value =
            serde_json::from_str(&current_json).unwrap_or_default();

        let mut drifts: Vec<serde_json::Value> = Vec::new();

        for section in &["by_realm", "by_kind"] {
            let prev_arr = prev_stats[section].as_array();
            let curr_arr = current[section].as_array();
            if let (Some(prev_items), Some(curr_items)) = (prev_arr, curr_arr) {
                let prev_map: std::collections::HashMap<&str, &serde_json::Value> = prev_items
                    .iter()
                    .filter_map(|v| v["group"].as_str().map(|g| (g, v)))
                    .collect();
                for curr in curr_items {
                    let group = match curr["group"].as_str() {
                        Some(g) => g,
                        None => continue,
                    };
                    if let Some(prev) = prev_map.get(group) {
                        let iso_prev = prev["isotropy"].as_f64().unwrap_or(0.0);
                        let iso_curr = curr["isotropy"].as_f64().unwrap_or(0.0);
                        let cos_prev = prev["mean_cosine_sim"].as_f64().unwrap_or(0.0);
                        let cos_curr = curr["mean_cosine_sim"].as_f64().unwrap_or(0.0);
                        let iso_delta = iso_curr - iso_prev;
                        let cos_delta = cos_curr - cos_prev;
                        if iso_delta.abs() > 0.005 || cos_delta.abs() > 0.005 {
                            drifts.push(serde_json::json!({
                                "section": section,
                                "group": group,
                                "isotropy_delta": (iso_delta * 1000.0).round() / 1000.0,
                                "cosine_delta": (cos_delta * 1000.0).round() / 1000.0,
                                "isotropy_now": iso_curr,
                                "cosine_now": cos_curr,
                            }));
                        }
                    }
                }
            }
        }

        let hours_since = (now_ms() - prev_ts) as f64 / 3_600_000.0;
        serde_json::to_string(&serde_json::json!({
            "snapshot_age_hours": (hours_since * 10.0).round() / 10.0,
            "drifts": drifts,
            "total_drifted": drifts.len(),
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Encode a memory's embedding into sparse codes and index into the cortical index.
    /// Persists via UpdateSparseCode op.
    /// Embedding for a memory. The semantic index is the single in-RAM home
    /// (payload copies are cleared at write/replay since v2.7.0 — keeping
    /// them duplicated ~600MB of RSS); the payload copy survives only for
    /// memories the index does not hold (unindexed / foreign-dim).
    pub(crate) fn embedding_of(&self, memory_id: MemoryId) -> Option<Vec<f32>> {
        // Lock order: payloads before semantic_idx.
        let payloads = self.payloads.read();
        let idx = self.semantic_idx.read();
        if let Some(e) = idx.get_embedding(memory_id) {
            if !e.is_empty() {
                return Some(e.to_vec());
            }
        }
        payloads.get(&memory_id).and_then(|p| {
            if p.embedding.is_empty() {
                None
            } else {
                Some(p.embedding.clone())
            }
        })
    }

    pub fn encode_memory(&self, memory_id: MemoryId) -> Result<()> {
        let embedding = self.embedding_of(memory_id);
        let Some(embedding) = embedding else {
            return Ok(());
        };
        if embedding.len() != EMBED_DIM {
            return Ok(());
        }

        let encoder = self.sparse_encoder.read();
        let code = encoder.encode(&embedding);
        if code.is_empty() {
            drop(encoder);
            self.encode_skip.write().insert(memory_id);
            return Ok(());
        }

        // Compute surprise (reconstruction error) before updating encoder
        let surprise = encoder.reconstruction_error(&embedding, &code);
        drop(encoder);

        // Update surprise in memory state and plasticity learner
        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.surprise = surprise;
            }
            let mut learners = self.learners.write();
            learners.plasticity.update_surprise(memory_id, surprise);
        }

        // FEP-derived update (accuracy + complexity + orthogonalization)
        self.sparse_encoder.write().update(&embedding, &code);

        let ts_ms = now_ms();
        let op = Op::UpdateSparseCode(UpdateSparseCodeOp {
            memory_id,
            feature_ids: code.feature_ids.clone(),
            activations: code.activations.clone(),
            ts_ms,
        });
        self.log.write().append(&op)?;

        // Index in cortical index
        let (strength, state_arousal) = {
            let st = self.states.read();
            let s = st.get(&memory_id);
            (s.map(|s| s.strength).unwrap_or(0.5), s.map(|s| s.affect_arousal).unwrap_or(0.0))
        };
        let kind = self
            .payloads
            .read()
            .get(&memory_id)
            .map(|p| p.kind.clone())
            .unwrap_or_default();
        let authored_at = self
            .payloads
            .read()
            .get(&memory_id)
            .map(|p| p.authored_at_ms)
            .unwrap_or(ts_ms);
        let affect_arousal = if state_arousal > 0.01 {
            state_arousal
        } else if kind.to_ascii_lowercase().contains("correction") {
            0.85
        } else {
            match kind.as_str() {
                "wisdom" => 0.70,
                "insight" => 0.50,
                "signal" => 0.30,
                "episode" | "observation" => 0.15,
                _ => 0.25,
            }
        };
        self.cortical_idx
            .write()
            .index_with_affect(memory_id, &code, strength, authored_at, &kind, affect_arousal);

        Ok(())
    }

    /// Save the cortical index + encoder + prototype state to a binary snapshot.
    /// After this, on next open the snapshot covers all UpdateSparseCode ops
    /// up to the current log position, so those ops can be skipped in replay.
    pub fn save_snapshot(&self) -> Result<()> {
        let _touch_drain = self.touch_drain.lock();
        self.drain_pending_touches_locked()?;
        self.drain_pending_recall_effects()?;
        self.sync_wal()?;
        let seqno = self.log.read().last_seqno();
        let path = self
            .data_dir
            .join(format!("cortex.{:08x}.snapshot", self.instance_id));
        self.cortical_idx.read().save_snapshot(&path, seqno)
    }

    // ── Layer 1: Executable Constraints ────────────────────────────────────

    pub fn assert_constraint(
        &self,
        subject: String,
        predicate: String,
        object: String,
        confidence: f32,
        scope: String,
        branch_id: u64,
        provenance: crate::organ::constraint::Provenance,
        source_memory_id: Option<u64>,
    ) -> Result<crate::organ::constraint::AssertResult> {
        let now = now_ms();
        let result = self.constraint_store.write().assert_fact(
            subject.clone(), predicate.clone(), object.clone(),
            confidence, scope.clone(), branch_id, provenance.clone(),
            now, source_memory_id,
        );
        let op = Op::AssertConstraint(crate::ops::AssertConstraintOp {
            fact_id: result.fact_id,
            subject, predicate, object, confidence, scope, branch_id,
            provenance_source: provenance.source,
            provenance_session: provenance.session_id,
            provenance_basis: provenance.confidence_basis,
            valid_from_ms: now,
            source_memory_id,
        });
        self.log.write().append(&op)?;
        Ok(result)
    }

    pub fn retract_constraint(&self, fact_id: u64) -> Result<bool> {
        let now = now_ms();
        let ok = self.constraint_store.write().retract(fact_id, now);
        if ok {
            let op = Op::RetractConstraint(crate::ops::RetractConstraintOp {
                fact_id, retracted_at_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn query_constraints(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
        scope: Option<&str>,
    ) -> Vec<crate::organ::constraint::Constraint> {
        self.constraint_store.read().query_unify(subject, predicate, object, scope)
            .into_iter().cloned().collect()
    }

    pub fn query_constraint_chain(
        &self, subject: &str, predicates: &[&str], max_depth: usize,
    ) -> Vec<Vec<crate::organ::constraint::Constraint>> {
        self.constraint_store.read().query_chain(subject, predicates, max_depth)
            .into_iter().map(|v| v.into_iter().cloned().collect()).collect()
    }

    pub fn explain_constraint(&self, fact_id: u64) -> Option<crate::organ::constraint::Explanation> {
        self.constraint_store.read().explain(fact_id)
    }

    pub fn create_constraint_branch(&self, parent_id: u64, scope: String) -> Result<u64> {
        let now = now_ms();
        let branch_id = self.constraint_store.write().create_branch(parent_id, scope.clone(), now);
        let op = Op::CreateBranch(crate::ops::CreateBranchOp {
            branch_id, parent_id, scope, created_ms: now,
        });
        self.log.write().append(&op)?;
        Ok(branch_id)
    }

    pub fn resolve_constraint_branch(&self, winner_id: u64, loser_id: u64) -> Result<bool> {
        let now = now_ms();
        let ok = self.constraint_store.write().resolve_branch(winner_id, loser_id, now);
        if ok {
            let op = Op::ResolveBranch(crate::ops::ResolveBranchOp {
                winner_id, loser_id, resolved_at_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn constraint_stats(&self) -> (usize, usize) {
        let store = self.constraint_store.read();
        (store.count(), store.branch_count())
    }

    // ── Layer 2: Trigger Tissue ─────────────────────────────────────────

    pub fn add_trigger(
        &self,
        name: String,
        condition: crate::organ::trigger::TriggerCondition,
        action: crate::organ::trigger::TriggerAction,
        deadline_ms: i64,
        tension_threshold: f32,
        gain: f32,
        realm: String,
        source_session: Option<String>,
    ) -> Result<u64> {
        let now = now_ms();
        let id = self.trigger_store.write().add_trigger(
            name, condition.clone(), action.clone(),
            deadline_ms, tension_threshold, gain, realm.clone(), source_session.clone(), now,
        );
        let trigger = self.trigger_store.read().get(id).cloned();
        if let Some(t) = trigger {
            let json = serde_json::to_vec(&t).unwrap_or_default();
            let op = Op::AddTrigger(crate::ops::AddTriggerOp { trigger_json: json });
            self.log.write().append(&op)?;
        }
        Ok(id)
    }

    pub fn fire_trigger(&self, trigger_id: u64) -> Result<Option<crate::organ::trigger::FireResult>> {
        let now = now_ms();
        let result = self.trigger_store.write().fire(trigger_id, now);
        if result.is_some() {
            let op = Op::FireTrigger(crate::ops::FireTriggerOp {
                trigger_id, fired_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(result)
    }

    pub fn dismiss_trigger(&self, trigger_id: u64) -> Result<bool> {
        let now = now_ms();
        let ok = self.trigger_store.write().dismiss(trigger_id, now);
        if ok {
            let op = Op::UpdateTrigger(crate::ops::UpdateTriggerOp {
                trigger_id, status: 2, fired_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn list_triggers(&self) -> Vec<crate::organ::trigger::TriggerAutomaton> {
        self.trigger_store.read().list_all().to_vec()
    }

    pub fn evaluate_triggers(&self) -> Result<Vec<crate::organ::trigger::FireResult>> {
        let now = now_ms();
        let ready_ids = self.trigger_store.read().evaluate_time_triggers(now);
        let mut results = Vec::new();
        for id in ready_ids {
            if let Some(result) = self.fire_trigger(id)? {
                results.push(result);
            }
        }
        Ok(results)
    }

    pub fn trigger_stats(&self) -> usize {
        self.trigger_store.read().count_armed()
    }

    // ── Layer 3: Predictive Memory ──────────────────────────────────────

    pub fn predict_needed(&self, k: usize) -> Vec<(MemoryId, f32)> {
        self.predictor.read().predict(k)
    }

    pub fn retrain_predictor(&self) {
        let now = now_ms();
        self.predictor.write().retrain(now);
    }

    pub fn predictor_stats(&self) -> (u64, usize, usize) {
        let p = self.predictor.read();
        (p.total_transitions(), p.transition_count(), p.recent_access_len())
    }

    // ── Layer 4: Surprise Memory ──────────────────────────────────────

    pub fn record_surprise(
        &self,
        context_sketch: String,
        action: String,
        expected: Option<String>,
        actual: String,
        surprise_magnitude: f32,
        domain: String,
        realm: String,
        session_id: Option<String>,
        source_memory_id: Option<u64>,
    ) -> Result<u64> {
        let now = now_ms();
        let event_id = {
            let mut store = self.surprise_store.write();
            store.record(
                context_sketch.clone(), action.clone(), expected.clone(),
                actual.clone(), surprise_magnitude, domain.clone(),
                realm.clone(), session_id.clone(), source_memory_id, now,
            )
        };
        let domain_ref = domain.clone();
        let action_ref = action.clone();
        let op = Op::RecordSurprise(crate::ops::RecordSurpriseOp {
            event_id,
            context_sketch,
            action,
            expected,
            actual,
            surprise_magnitude,
            domain,
            timestamp_ms: now,
            realm,
            session_id,
            source_memory_id,
        });
        self.log.write().append(&op)?;

        // ── Move 1: auto-strengthen/weaken via surprise credit ────────
        if let Some(source_id) = source_memory_id {
            // source_memory_id was the "expected" memory → weaken direction
            let credit_result = self.surprise_learning.write().update_credit(
                source_id, event_id, surprise_magnitude, -1, now,
            );
            if let Some(cr) = credit_result {
                // Apply strength delta via existing UpdateState
                let delta_op = crate::ops::StateDeltaOp {
                    memory_id: cr.memory_id,
                    strength_delta: Some(cr.strength_delta),
                    confidence_delta: None,
                    decay_rate: None,
                    touch: false,
                    pin: None,
                    op_ts_ms: now,
                    status: None,
                    epistemic_status: None,
                    staged: None,
                    invalidated_by: None,
                };
                if let Some(state) = self.states.write().get_mut(&cr.memory_id) {
                    state.apply_delta(&delta_op, now);
                }
                self.log.write().append(&Op::UpdateState(delta_op))?;
                // WAL the credit state
                let sl = self.surprise_learning.read();
                if let Some(st) = sl.get_state(cr.memory_id) {
                    self.log.write().append(&Op::UpdateSurpriseCredit(
                        crate::ops::UpdateSurpriseCreditOp {
                            memory_id: st.memory_id,
                            credit: st.credit,
                            last_dir: st.last_dir,
                            same_dir_streak: st.same_dir_streak,
                            last_surprise_id: st.last_surprise_id,
                            updated_ms: st.updated_ms,
                        },
                    ))?;
                }
            }
        }

        // ── Move 2: auto-feed integration kernel ──────────────────────
        {
            let should_neg = self.surprise_learning.read()
                .should_send_negative_feedback(&domain_ref, "semantic", surprise_magnitude);
            if should_neg {
                self.surprise_learning.write().record_failure(&domain_ref, "semantic", event_id);
                let _ = self.record_feedback(&domain_ref, "semantic", false);
            }
            let should_pos = self.surprise_learning.read()
                .should_send_positive_feedback(surprise_magnitude);
            if should_pos {
                let _ = self.record_feedback(&domain_ref, "keyword", true);
            }
        }

        // ── Layer 9: adjudicate wisdom lineages by envelope overlap ───
        {
            use crate::organ::wisdom_lineage::CONTRADICTION_DELTA_HIT;
            let matching = self.wisdom_lineage_store.read()
                .find_by_envelope(&domain_ref, &action_ref);
            for lineage_id in matching {
                let new_state = self.wisdom_lineage_store.write().adjudicate(
                    lineage_id, 0.0,
                    surprise_magnitude * CONTRADICTION_DELTA_HIT,
                    0.0, now,
                );
                if let Some(l) = self.wisdom_lineage_store.read().get(lineage_id) {
                    self.log.write().append(&Op::AdjudicateLineage(
                        crate::ops::AdjudicateLineageOp {
                            lineage_id,
                            support_mass: l.support_mass,
                            contradiction_mass: l.contradiction_mass,
                            staleness_mass: l.staleness_mass,
                            last_supported_ms: l.last_supported_ms,
                            last_challenged_ms: l.last_challenged_ms,
                            adjudicated_ms: now,
                        },
                    ))?;
                    if let Some(ns) = new_state {
                        self.log.write().append(&Op::TransitionLineage(
                            crate::ops::TransitionLineageOp {
                                lineage_id,
                                old_state: l.state.as_u8(),
                                new_state: ns.as_u8(),
                                reason: "surprise_adjudication".to_string(),
                                rederive_task_id: None,
                                transitioned_ms: now,
                            },
                        ))?;
                    }
                }
                // Record surprise as challenger evidence
                let _ = self.wisdom_lineage_store.write().record_challenger(
                    lineage_id,
                    crate::organ::wisdom_lineage::ChallengerEvidence {
                        intervention_id: None,
                        surprise_id: Some(event_id),
                        outcome_summary: format!("surprise magnitude {:.2}", surprise_magnitude),
                        attached_ms: now,
                    },
                    now,
                );
            }
        }

        Ok(event_id)
    }

    pub fn query_surprises(
        &self,
        domain: Option<&str>,
        realm: Option<&str>,
        min_magnitude: Option<f32>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Vec<crate::organ::surprise::SurpriseEvent> {
        self.surprise_store
            .read()
            .query(domain, realm, min_magnitude, since_ms, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn get_blind_spots(
        &self,
        realm: Option<&str>,
        limit: usize,
    ) -> Vec<crate::organ::surprise::BlindSpot> {
        self.surprise_store.read().get_blind_spots(realm, limit)
    }

    pub fn surprise_stats(&self) -> crate::organ::surprise::SurpriseStats {
        self.surprise_store.read().stats()
    }

    // ── Layer 5: Epistemic Debt ───────────────────────────────────────

    pub fn register_debt(
        &self,
        pattern: String,
        competing_hypotheses: Vec<String>,
        discriminating_test: Option<String>,
        fragility_score: f32,
        domain: String,
        realm: String,
        source_session: Option<String>,
    ) -> Result<u64> {
        let now = now_ms();
        let debt_id = {
            let mut store = self.epistemic_debt_store.write();
            store.register(
                pattern.clone(), competing_hypotheses.clone(),
                discriminating_test.clone(), fragility_score,
                domain.clone(), realm.clone(), source_session.clone(), now,
            )
        };
        let op = Op::RegisterDebt(crate::ops::RegisterDebtOp {
            debt_id,
            pattern,
            competing_hypotheses,
            discriminating_test,
            fragility_score,
            domain,
            created_ms: now,
            realm,
            source_session,
        });
        self.log.write().append(&op)?;
        Ok(debt_id)
    }

    pub fn resolve_debt(&self, debt_id: u64, resolution: String) -> Result<bool> {
        let now = now_ms();
        let ok = self.epistemic_debt_store.write().resolve(debt_id, resolution.clone(), now);
        if ok {
            let op = Op::UpdateDebt(crate::ops::UpdateDebtOp {
                debt_id,
                status: 1,
                resolved_ms: now,
                resolution: Some(resolution),
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn defer_debt(&self, debt_id: u64) -> Result<bool> {
        let ok = self.epistemic_debt_store.write().defer(debt_id);
        if ok {
            let op = Op::UpdateDebt(crate::ops::UpdateDebtOp {
                debt_id,
                status: 2,
                resolved_ms: 0,
                resolution: None,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn query_debts(
        &self,
        status: Option<crate::organ::epistemic_debt::DebtStatus>,
        domain: Option<&str>,
        realm: Option<&str>,
        min_fragility: Option<f32>,
        limit: usize,
    ) -> Vec<crate::organ::epistemic_debt::EpistemicDebt> {
        self.epistemic_debt_store
            .read()
            .query(status, domain, realm, min_fragility, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn get_fragile_decisions(
        &self,
        threshold: f32,
        limit: usize,
    ) -> Vec<crate::organ::epistemic_debt::EpistemicDebt> {
        self.epistemic_debt_store
            .read()
            .get_fragile_decisions(threshold, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn debt_stats(&self) -> crate::organ::epistemic_debt::DebtStats {
        self.epistemic_debt_store.read().stats()
    }

    // ── Layer 6: Integration Kernel ───────────────────────────────────

    pub fn record_feedback(
        &self,
        query_domain: &str,
        source: &str,
        was_useful: bool,
    ) -> Result<crate::organ::integration::SourceWeight> {
        let sw = self.integration_kernel.write().record_feedback(query_domain, source, was_useful);
        let op = Op::RecordFeedback(crate::ops::RecordFeedbackOp {
            source: sw.source.clone(),
            query_domain: sw.query_domain.clone(),
            was_useful,
            new_weight: sw.weight,
            success_count: sw.success_count,
            total_count: sw.total_count,
        });
        self.log.write().append(&op)?;
        Ok(sw)
    }

    pub fn get_source_weights(
        &self,
        domain: Option<&str>,
    ) -> Vec<crate::organ::integration::SourceWeight> {
        self.integration_kernel
            .read()
            .get_source_weights(domain)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn update_source_weight(
        &self,
        source: &str,
        domain: &str,
        weight: f32,
    ) -> Result<bool> {
        let ok = self.integration_kernel.write().update_source_weight(source, domain, weight);
        let op = Op::UpdateSourceWeight(crate::ops::UpdateSourceWeightOp {
            source: source.to_string(),
            query_domain: domain.to_string(),
            weight,
        });
        self.log.write().append(&op)?;
        Ok(ok)
    }

    pub fn integration_stats(&self) -> crate::organ::integration::IntegrationStats {
        self.integration_kernel.read().stats()
    }

    // ── Surprise Learning (Moves 1-2) ────────────────────────────────

    pub fn surprise_learning_stats(&self) -> crate::organ::surprise_learning::SurpriseLearningStats {
        self.surprise_learning.read().stats()
    }

    // ── Wisdom Promotion (Move 5) ────────────────────────────────────

    pub fn upsert_wisdom_candidate(
        &self,
        cluster_key: String,
        domain: String,
        action: String,
        summary: String,
        episode_ids: Vec<u64>,
        debt_ids: Vec<u64>,
        support_count: u32,
        cross_session_count: u32,
        mean_surprise: f32,
        promotion_score: f32,
    ) -> Result<u64> {
        let now = now_ms();
        let candidate_id = {
            let mut store = self.wisdom_promotion.write();
            store.upsert_candidate(
                cluster_key.clone(), domain.clone(), action.clone(), summary.clone(),
                episode_ids.clone(), debt_ids.clone(), support_count,
                cross_session_count, mean_surprise, promotion_score, now,
            )
        };
        let op = Op::UpsertWisdomCandidate(crate::ops::UpsertWisdomCandidateOp {
            candidate_id,
            cluster_key,
            domain,
            action,
            summary,
            episode_ids,
            debt_ids,
            support_count,
            cross_session_count,
            mean_surprise,
            promotion_score,
            created_ms: now,
        });
        self.log.write().append(&op)?;
        Ok(candidate_id)
    }

    pub fn update_wisdom_lifecycle(
        &self,
        candidate_id: u64,
        new_state: crate::organ::wisdom_promotion::WisdomLifecycle,
        memory_id: Option<u64>,
        contradiction_count: u32,
    ) -> Result<bool> {
        let now = now_ms();
        let old_state = self.wisdom_promotion.read()
            .get(candidate_id)
            .map(|c| c.lifecycle.as_u8())
            .unwrap_or(0);
        let ok = self.wisdom_promotion.write().update_lifecycle(
            candidate_id, new_state, memory_id, contradiction_count, now,
        );
        if ok {
            let op = Op::UpdateWisdomLifecycle(crate::ops::UpdateWisdomLifecycleOp {
                candidate_id,
                memory_id,
                old_state,
                new_state: new_state.as_u8(),
                contradiction_count,
                updated_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn query_wisdom_candidates(
        &self,
        lifecycle: Option<crate::organ::wisdom_promotion::WisdomLifecycle>,
        domain: Option<&str>,
        limit: usize,
    ) -> Vec<crate::organ::wisdom_promotion::WisdomCandidate> {
        self.wisdom_promotion
            .read()
            .query(lifecycle, domain, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn wisdom_promotion_stats(&self) -> crate::organ::wisdom_promotion::WisdomPromotionStats {
        self.wisdom_promotion.read().stats()
    }

    // ── Debt Evidence (Move 3) ───────────────────────────────────────

    pub fn attach_debt_evidence(
        &self,
        debt_id: u64,
        evidence_memory_ids: Vec<u64>,
        confidence: f32,
        note: Option<String>,
    ) -> Result<bool> {
        let now = now_ms();
        let ok = self.epistemic_debt_store.write().attach_evidence(
            debt_id, evidence_memory_ids.clone(), confidence, note.clone(), now,
        );
        if ok {
            let op = Op::AttachDebtEvidence(crate::ops::AttachDebtEvidenceOp {
                debt_id,
                evidence_memory_ids,
                confidence,
                note,
                attached_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    /// Auto-resolve debts with sufficient evidence. Returns count resolved.
    pub fn auto_resolve_debts(&self, threshold: f32) -> Result<usize> {
        let open_ids: Vec<u64> = self.epistemic_debt_store.read()
            .open_debts_with_evidence()
            .iter()
            .filter(|d| !d.evidence.is_empty())
            .map(|d| d.id)
            .collect();

        let now = now_ms();
        let mut resolved_count = 0usize;
        for id in open_ids {
            let resolved = self.epistemic_debt_store.write()
                .auto_resolve_if_ready(id, threshold, now);
            if resolved {
                let op = Op::UpdateDebt(crate::ops::UpdateDebtOp {
                    debt_id: id,
                    status: 1,
                    resolved_ms: now,
                    resolution: Some(format!("auto-resolved: evidence >= {:.2}", threshold)),
                });
                self.log.write().append(&op)?;
                resolved_count += 1;
            }
        }
        Ok(resolved_count)
    }

    // ── Learned Scorer (Move 6) ──────────────────────────────────────

    pub fn update_scorer_model(
        &self,
        weights_json: String,
        model_version: u64,
        mean_loss: f32,
        outcome_count: u64,
    ) -> Result<()> {
        let now = now_ms();
        self.learned_scorer.write().apply_update(
            &weights_json, model_version, mean_loss, outcome_count, now,
        );
        let op = Op::UpdateScorerModel(crate::ops::UpdateScorerModelOp {
            model_version,
            baseline_version: self.learned_scorer.read().baseline_version.clone(),
            weights_json,
            applied_at_ms: now,
            outcome_count,
            mean_loss,
        });
        self.log.write().append(&op)?;
        Ok(())
    }

    pub fn learned_scorer_stats(&self) -> crate::scoring::learned::LearnedScoringStats {
        self.learned_scorer.read().stats()
    }

    pub fn effective_scorer_weight(&self, factor_name: &str, baseline: f32) -> f32 {
        self.learned_scorer.read().effective_weight(factor_name, baseline)
    }

    // ── Layer 7: Intervention Ledger ─────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn start_intervention(
        &self,
        realm: String,
        session_id: String,
        task_id: Option<u64>,
        agent_id: String,
        domain: String,
        intent: String,
        action_type: crate::organ::intervention::ActionType,
        action_ref: String,
        preconditions: Vec<String>,
        expected_observables: Vec<String>,
        reversal_cost: crate::organ::intervention::ReversalCost,
    ) -> Result<u64> {
        let now = now_ms();
        let id = self.intervention_store.write().start_intervention(
            realm.clone(), session_id.clone(), task_id, agent_id.clone(),
            domain.clone(), intent.clone(), action_type, action_ref.clone(),
            preconditions.clone(), expected_observables.clone(), reversal_cost, now,
        );
        self.log.write().append(&crate::ops::Op::StartIntervention(
            crate::ops::StartInterventionOp {
                id, realm, session_id, task_id, agent_id, domain, intent,
                action_type: action_type.to_u8(), action_ref,
                preconditions, expected_observables,
                reversal_cost: reversal_cost.to_u8(), started_ms: now,
            }
        ))?;
        Ok(id)
    }

    pub fn add_observation(
        &self,
        intervention_id: u64,
        kind: crate::organ::intervention::ObservationKind,
        evidence_refs: Vec<u64>,
        summary: String,
        confidence: f32,
    ) -> Result<Option<u64>> {
        let now = now_ms();
        let obs_id = self.intervention_store.write().add_observation(
            intervention_id, kind, evidence_refs.clone(), summary.clone(), confidence, now,
        );
        if let Some(oid) = obs_id {
            self.log.write().append(&crate::ops::Op::AddObservation(
                crate::ops::AddObservationOp {
                    id: oid, intervention_id, kind: kind.to_u8(),
                    evidence_refs, summary, confidence, timestamp_ms: now,
                }
            ))?;
        }
        Ok(obs_id)
    }

    pub fn close_intervention(
        &self,
        intervention_id: u64,
        status: crate::organ::intervention::InterventionStatus,
    ) -> Result<bool> {
        use crate::organ::intervention::InterventionStatus;
        use crate::organ::wisdom_lineage::{SUPPORT_DELTA_HIT, CONTRADICTION_DELTA_HIT};
        let now = now_ms();
        let (domain, action_type) = {
            let store = self.intervention_store.read();
            store.get(intervention_id)
                .map(|r| (r.domain.clone(), format!("{:?}", r.action_type).to_lowercase()))
                .unwrap_or_default()
        };
        let ok = self.intervention_store.write().close_intervention(intervention_id, status, now);
        if ok {
            self.log.write().append(&crate::ops::Op::CloseIntervention(
                crate::ops::CloseInterventionOp {
                    intervention_id, status: status.to_u8(), closed_ms: now,
                }
            ))?;

            // ── Layer 9: adjudicate wisdom lineages by outcome ────────
            if !domain.is_empty() {
                let matching = self.wisdom_lineage_store.read()
                    .find_by_envelope(&domain, &action_type);
                let (support_delta, contradiction_delta) = match status {
                    InterventionStatus::Succeeded => (SUPPORT_DELTA_HIT, 0.0f32),
                    InterventionStatus::Failed | InterventionStatus::Aborted => (0.0f32, CONTRADICTION_DELTA_HIT),
                    InterventionStatus::Partial => (SUPPORT_DELTA_HIT * 0.3, CONTRADICTION_DELTA_HIT * 0.3),
                    InterventionStatus::Open => (0.0f32, 0.0f32),
                };
                for lineage_id in matching {
                    let new_state = self.wisdom_lineage_store.write().adjudicate(
                        lineage_id, support_delta, contradiction_delta, 0.0, now,
                    );
                    if let Some(l) = self.wisdom_lineage_store.read().get(lineage_id) {
                        self.log.write().append(&Op::AdjudicateLineage(
                            crate::ops::AdjudicateLineageOp {
                                lineage_id,
                                support_mass: l.support_mass,
                                contradiction_mass: l.contradiction_mass,
                                staleness_mass: l.staleness_mass,
                                last_supported_ms: l.last_supported_ms,
                                last_challenged_ms: l.last_challenged_ms,
                                adjudicated_ms: now,
                            },
                        ))?;
                        if let Some(ns) = new_state {
                            self.log.write().append(&Op::TransitionLineage(
                                crate::ops::TransitionLineageOp {
                                    lineage_id,
                                    old_state: l.state.as_u8(),
                                    new_state: ns.as_u8(),
                                    reason: "intervention_outcome".to_string(),
                                    rederive_task_id: None,
                                    transitioned_ms: now,
                                },
                            ))?;
                        }
                    }
                    if matches!(status, InterventionStatus::Failed | InterventionStatus::Aborted) {
                        let _ = self.wisdom_lineage_store.write().record_challenger(
                            lineage_id,
                            crate::organ::wisdom_lineage::ChallengerEvidence {
                                intervention_id: Some(intervention_id),
                                surprise_id: None,
                                outcome_summary: format!("intervention {} {:?}", intervention_id, status),
                                attached_ms: now,
                            },
                            now,
                        );
                    }
                }
            }
        }
        Ok(ok)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_attribution(
        &self,
        intervention_id: u64,
        primary_class: crate::organ::intervention::AttributionClass,
        secondary_class: Option<crate::organ::intervention::AttributionClass>,
        confidence_delta: f32,
        surprise_id: Option<u64>,
        debt_ids: Vec<u64>,
        source_memory_ids: Vec<u64>,
        skill_memory_ids: Vec<u64>,
        note: Option<String>,
    ) -> Result<bool> {
        let now = now_ms();
        // Look up intervention domain before releasing write lock
        let domain = {
            let store = self.intervention_store.read();
            store.get(intervention_id).map(|r| r.domain.clone()).unwrap_or_default()
        };
        let ok = self.intervention_store.write().record_attribution(
            intervention_id, primary_class, secondary_class,
            confidence_delta, surprise_id, debt_ids.clone(),
            source_memory_ids.clone(), skill_memory_ids.clone(), note.clone(), now,
        );
        if !ok { return Ok(false); }
        self.log.write().append(&crate::ops::Op::RecordAttribution(
            crate::ops::RecordAttributionOp {
                intervention_id,
                primary_class: primary_class.to_u8(),
                secondary_class: secondary_class.map(|c| c.to_u8()),
                confidence_delta, surprise_id,
                debt_ids: debt_ids.clone(),
                source_memory_ids: source_memory_ids.clone(),
                skill_memory_ids: skill_memory_ids.clone(),
                note, timestamp_ms: now,
            }
        ))?;
        // Route to learning subsystems
        self.route_attribution(&domain, primary_class, confidence_delta,
            surprise_id, &source_memory_ids, &skill_memory_ids);
        if let Some(sec) = secondary_class {
            self.route_attribution(&domain, sec, confidence_delta * 0.5,
                surprise_id, &source_memory_ids, &skill_memory_ids);
        }
        Ok(true)
    }

    fn route_attribution(
        &self,
        domain: &str,
        class: crate::organ::intervention::AttributionClass,
        confidence_delta: f32,
        surprise_id: Option<u64>,
        source_memory_ids: &[u64],
        skill_memory_ids: &[u64],
    ) {
        use crate::organ::intervention::AttributionClass::*;
        let now = now_ms();
        match class {
            MemoryRecallError => {
                if let Some(sid) = surprise_id {
                    let mut sl = self.surprise_learning.write();
                    for &mid in source_memory_ids {
                        let _ = sl.update_credit(mid, sid, confidence_delta.abs(), -1, now);
                    }
                }
            }
            SourceTrustError => {
                let _ = self.integration_kernel.write().record_feedback(domain, "memory", false);
            }
            ProcedureError => {
                for &mid in skill_memory_ids {
                    let _ = self.update_state(
                        mid, Some(-confidence_delta.abs()), None, None, false, None,
                    );
                }
            }
            ToolExecutionError | EnvironmentShift | HiddenPrecondition
            | AmbiguousState | GoalSpecError | UserOverride | ExternalNondeterminism => {
                // No automatic side-effect; caller handles debt/task repair at MCP layer
            }
        }
    }

    pub fn get_intervention(
        &self, id: u64,
    ) -> Option<crate::organ::intervention::InterventionRecord> {
        self.intervention_store.read().get(id).cloned()
    }

    pub fn query_interventions(
        &self,
        realm: Option<&str>,
        session_id: Option<&str>,
        status: Option<crate::organ::intervention::InterventionStatus>,
        limit: usize,
    ) -> Vec<crate::organ::intervention::InterventionRecord> {
        self.intervention_store.read()
            .query(realm, session_id, status, limit)
            .into_iter().cloned().collect()
    }

    pub fn list_open_interventions(
        &self,
    ) -> Vec<crate::organ::intervention::InterventionRecord> {
        self.intervention_store.read().list_open().into_iter().cloned().collect()
    }

    pub fn intervention_stats(&self) -> crate::organ::intervention::InterventionStats {
        self.intervention_store.read().stats()
    }

    pub fn close_stale_interventions(&self, threshold_ms: i64) -> Result<usize> {
        let now = now_ms();
        let stale_ids = self.intervention_store.read().stale_open(threshold_ms, now);
        let mut closed = 0usize;
        for id in stale_ids {
            let ok = self.intervention_store.write().close_intervention(
                id, crate::organ::intervention::InterventionStatus::Aborted, now,
            );
            if ok {
                self.log.write().append(&crate::ops::Op::CloseIntervention(
                    crate::ops::CloseInterventionOp {
                        intervention_id: id,
                        status: crate::organ::intervention::InterventionStatus::Aborted.to_u8(),
                        closed_ms: now,
                    }
                ))?;
                closed += 1;
            }
        }
        Ok(closed)
    }

    // ── Agent Protocol Memory (Layer 8) ──────────────────────────────────────

    pub fn register_task(
        &self,
        goal: String,
        constraints: Vec<String>,
        acceptance_criteria: Vec<String>,
        realm: String,
        session_id: String,
        priority: u8,
        parent_task_id: Option<u64>,
        deadline_ms: Option<i64>,
        tags: Vec<String>,
    ) -> Result<u64> {
        let now = now_ms();
        let id = self.agent_protocol_store.write().register_task(
            goal.clone(), constraints.clone(), acceptance_criteria.clone(),
            realm.clone(), session_id.clone(), priority, parent_task_id,
            deadline_ms, tags.clone(), now,
        );
        self.log.write().append(&crate::ops::Op::RegisterTask(crate::ops::RegisterTaskOp {
            id, session_id, realm, goal, constraints, acceptance_criteria,
            priority, parent_task_id, tags, deadline_ms, created_ms: now,
        }))?;
        Ok(id)
    }

    pub fn update_task(
        &self,
        task_id: u64,
        status: Option<u8>,
        add_intervention_id: Option<u64>,
        add_tag: Option<String>,
    ) -> Result<bool> {
        use crate::organ::agent_protocol::TaskStatus;
        let now = now_ms();
        let status_enum = status.map(TaskStatus::from_u8);
        let ok = self.agent_protocol_store.write().update_task(
            task_id, status_enum, add_intervention_id, add_tag.clone(), now,
        );
        if ok {
            self.log.write().append(&crate::ops::Op::UpdateTask(crate::ops::UpdateTaskOp {
                task_id,
                status: status.unwrap_or(0),
                add_intervention_id,
                add_tag,
                updated_ms: now,
            }))?;
        }
        Ok(ok)
    }

    pub fn add_delegation(
        &self,
        task_id: u64,
        from_agent: String,
        to_agent: String,
        handoff_note: Option<String>,
    ) -> Result<Option<u64>> {
        let now = now_ms();
        let opt_id = self.agent_protocol_store.write().add_delegation(
            task_id, from_agent.clone(), to_agent.clone(), handoff_note.clone(), now,
        );
        if let Some(id) = opt_id {
            self.log.write().append(&crate::ops::Op::AddDelegation(crate::ops::AddDelegationOp {
                id, task_id, from_agent, to_agent, handoff_note, delegated_at: now,
            }))?;
        }
        Ok(opt_id)
    }

    pub fn link_evidence(
        &self,
        task_id: u64,
        memory_id: u64,
        produced_by: String,
        evidence_kind: u8,
        relevance: f32,
    ) -> Result<Option<u64>> {
        use crate::organ::agent_protocol::EvidenceKind;
        let now = now_ms();
        let opt_id = self.agent_protocol_store.write().link_evidence(
            task_id, memory_id, produced_by.clone(),
            EvidenceKind::from_u8(evidence_kind), relevance, now,
        );
        if let Some(id) = opt_id {
            self.log.write().append(&crate::ops::Op::LinkEvidence(crate::ops::LinkEvidenceOp {
                id, task_id, memory_id, produced_by, evidence_kind, relevance, created_ms: now,
            }))?;
        }
        Ok(opt_id)
    }

    pub fn add_probe(
        &self,
        task_id: u64,
        question: String,
        expected_answerer: Option<String>,
        priority: u8,
    ) -> Result<Option<u64>> {
        let now = now_ms();
        let opt_id = self.agent_protocol_store.write().add_probe(
            task_id, question.clone(), expected_answerer.clone(), priority, now,
        );
        if let Some(id) = opt_id {
            self.log.write().append(&crate::ops::Op::AddProbe(crate::ops::AddProbeOp {
                id, task_id, question, expected_answerer, priority, created_ms: now,
            }))?;
        }
        Ok(opt_id)
    }

    pub fn resolve_probe(
        &self,
        probe_id: u64,
        status: u8,
        answer: Option<String>,
    ) -> Result<bool> {
        use crate::organ::agent_protocol::ProbeStatus;
        let now = now_ms();
        let ok = self.agent_protocol_store.write().resolve_probe(
            probe_id, ProbeStatus::from_u8(status), answer.clone(), now,
        );
        if ok {
            self.log.write().append(&crate::ops::Op::ResolveProbe(crate::ops::ResolveProbeOp {
                probe_id, status, answer, resolved_ms: now,
            }))?;
        }
        Ok(ok)
    }

    pub fn set_criterion(
        &self,
        task_id: u64,
        criterion: String,
        is_met: bool,
        evidence_note: Option<String>,
    ) -> Result<Option<u64>> {
        let now = now_ms();
        let opt_id = self.agent_protocol_store.write().set_criterion(
            task_id, criterion.clone(), is_met, evidence_note.clone(), now,
        );
        if let Some(id) = opt_id {
            self.log.write().append(&crate::ops::Op::SetCriterion(crate::ops::SetCriterionOp {
                id, task_id, criterion, is_met, evidence_note, checked_ms: now,
            }))?;
        }
        Ok(opt_id)
    }

    pub fn get_task_full(&self, task_id: u64)
        -> Option<crate::organ::agent_protocol::TaskFullView>
    {
        self.agent_protocol_store.read().get_task_full(task_id)
    }

    pub fn query_tasks(
        &self,
        realm: Option<&str>,
        session_id: Option<&str>,
        status: Option<u8>,
        priority: Option<u8>,
        limit: usize,
    ) -> Vec<crate::organ::agent_protocol::TaskContract> {
        use crate::organ::agent_protocol::TaskStatus;
        self.agent_protocol_store
            .read()
            .query_tasks(realm, session_id, status.map(TaskStatus::from_u8), priority, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn agent_protocol_stats(&self) -> crate::organ::agent_protocol::AgentProtocolStats {
        self.agent_protocol_store.read().stats()
    }

    pub fn auto_complete_tasks(&self) -> Result<usize> {
        let task_ids = self.agent_protocol_store.read().tasks_with_all_criteria_met();
        let mut completed = 0usize;
        for tid in task_ids {
            if self.update_task(tid, Some(2), None, None)? {
                completed += 1;
            }
        }
        Ok(completed)
    }

    // ── Interaction Ledger ──────────────────────────────────────────────────────

    pub fn ledger_append(&self, ev: crate::organ::interaction_ledger::InteractionEvent) -> Result<u64> {
        Ok(self.interaction_ledger.write().append(ev))
    }

    pub fn ledger_query(
        &self,
        kind: Option<crate::organ::interaction_ledger::EventKind>,
        session_id: Option<&str>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<crate::organ::interaction_ledger::InteractionEvent>> {
        Ok(self.interaction_ledger.read()
            .query(kind.as_ref(), session_id, since_ms, limit)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn ledger_compile(&self) -> Result<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let mut ledger = self.interaction_ledger.write();
        let before = ledger.assertions.len();
        ledger.compile(now);
        Ok(ledger.assertions.len() - before)
    }

    pub fn ledger_contradictions(&self) -> Result<Vec<(String, String, Vec<u64>)>> {
        Ok(self.interaction_ledger.read().contested())
    }

    pub fn predicate_attach(&self, memory_id: u64, check_cmd: String) -> Result<u64> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Ok(self.predicate_store.write().attach(memory_id, check_cmd, now_ms))
    }

    /// Run all predicates for a memory. Returns JSON with per-predicate results.
    /// Weakens memory confidence by 0.1 for each failing predicate (min 0.1).
    pub fn predicate_run(&self, memory_id: u64) -> Result<String> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Phase 1: collect commands under read lock — no subprocess yet.
        let cmds: Vec<(u64, String)> = self.predicate_store.read().collect_cmds(memory_id);

        // Phase 2: run subprocesses outside any lock to avoid blocking predicate_attach.
        let results: Vec<(u64, bool, String)> = cmds.iter()
            .map(|(id, cmd)| {
                let (ok, out) = crate::organ::predicate_store::run_cmd(cmd);
                (*id, ok, out)
            })
            .collect();

        // Phase 3: write results back under write lock.
        let (passed, failed) = self.predicate_store.write().apply_results(&results, now_ms);

        if failed > 0 {
            let decay = 0.1 * failed as f32;
            if let Some(state) = self.states.write().get_mut(&crate::ids::MemoryId::from(memory_id)) {
                state.confidence = (state.confidence - decay).max(0.1);
            }
        }
        let result = serde_json::json!({
            "memory_id": memory_id,
            "passed": passed,
            "failed": failed,
            "epistemic_status": format!("{:?}", self.predicate_store.read().epistemic_status(memory_id)),
        });
        Ok(result.to_string())
    }

    pub fn predicate_list(&self, memory_id: u64) -> Result<String> {
        let store = self.predicate_store.read();
        let preds = store.for_memory(memory_id);
        let result = serde_json::json!({
            "memory_id": memory_id,
            "epistemic_status": format!("{:?}", store.epistemic_status(memory_id)),
            "predicates": preds.iter().map(|p| serde_json::json!({
                "predicate_id": p.predicate_id,
                "check_cmd": p.check_cmd,
                "status": format!("{:?}", p.status),
                "last_checked_ms": p.last_checked_ms,
                "last_output": p.last_output,
            })).collect::<Vec<_>>(),
        });
        Ok(result.to_string())
    }

    /// Save full in-memory state to a binary snapshot (chitta.snapshot).
    /// After this, on next open only ops after snapshot_seqno need to be replayed.
    /// Budgeted background competitive-weight refresh (THEORY.md §8 Phase 3:
    /// consolidation as deliberate merge). Refreshes up to `budget` memories
    /// whose last refresh is older than the configured interval, using the
    /// same reservation discipline as the recall-path refresh — so recalls
    /// (whose own budget is small) almost never pay refresh cost themselves.
    /// Called from the subconscious sleep-consolidation cycle. Returns the
    /// number refreshed.
    pub fn cw_refresh_sweep(&self, budget: usize) -> usize {
        if budget == 0 {
            return 0;
        }
        let now = now_ms();
        let (dedup_upper, interval_ms) = {
            let pipeline = self.scoring_pipeline.read();
            (
                pipeline.config.dedup_cosine_upper,
                pipeline.config.cw_refresh_interval_ms,
            )
        };
        // Collect stale candidates + embeddings under read locks (struct order).
        let candidates: Vec<(MemoryId, Vec<f32>)> = {
            let states_r = self.states.read();
            let idx = self.semantic_idx.read();
            let inflight_r = self.cw_refresh_inflight.read();
            states_r
                .iter()
                .filter(|(_, st)| !st.deleted && now - st.last_cw_refresh_ms >= interval_ms)
                .filter(|(id, _)| {
                    inflight_r
                        .get(id)
                        .map(|&ts| now - ts >= interval_ms)
                        .unwrap_or(true)
                })
                .filter_map(|(id, _)| idx.get_embedding(*id).map(|e| (*id, e.to_vec())))
                .take(budget)
                .collect()
        };
        // Atomic re-check + reserve (same as the recall path).
        let candidates: Vec<(MemoryId, Vec<f32>)> = if candidates.is_empty() {
            return 0;
        } else {
            let mut inflight_w = self.cw_refresh_inflight.write();
            inflight_w.retain(|_, ts| now - *ts < interval_ms);
            candidates
                .into_iter()
                .filter(|(id, _)| {
                    if inflight_w.contains_key(id) {
                        return false;
                    }
                    inflight_w.insert(*id, now);
                    true
                })
                .collect()
        };
        let cw_updates: Vec<(MemoryId, f32)> = {
            let idx = self.semantic_idx.read();
            candidates
                .iter()
                .filter_map(|(memory_id, emb)| {
                    let neighbors = idx.search(emb, 9, None, None);
                    if neighbors.len() <= 1 {
                        return None;
                    }
                    let mut cos_sum = 0.0f32;
                    let mut n = 0u32;
                    for nb in &neighbors {
                        if nb.memory_id == *memory_id {
                            continue;
                        }
                        if nb.cosine_similarity >= dedup_upper {
                            continue;
                        }
                        cos_sum += nb.cosine_similarity;
                        n += 1;
                    }
                    if n > 0 {
                        Some((*memory_id, cos_sum / n as f32))
                    } else {
                        None
                    }
                })
                .collect()
        };
        let refreshed = candidates.len();
        let cw_by_id: std::collections::HashMap<MemoryId, f32> = cw_updates.into_iter().collect();
        {
            let mut states_w = self.states.write();
            for (memory_id, _) in &candidates {
                if let Some(st) = states_w.get_mut(memory_id) {
                    if let Some(&cw) = cw_by_id.get(memory_id) {
                        st.competitive_weight = cw;
                    }
                    st.last_cw_refresh_ms = now;
                }
            }
        }
        let mut inflight_w = self.cw_refresh_inflight.write();
        for (id, _) in &candidates {
            inflight_w.remove(id);
        }
        refreshed
    }

    pub fn save_full_snapshot(&self) -> Result<()> {
        use crate::snapshot::FullSnapshot;
        let _touch_drain = self.touch_drain.lock();
        self.drain_pending_touches_locked()?;
        self.drain_pending_recall_effects()?;
        self.sync_wal()?;
        let seqno = self.log.read().last_seqno();
        // Compact the delta into the base under a BRIEF write so the snapshot clone below
        // is canonical (delta empty). Decoupled from the sidecar disk writes, which now run
        // lock-free off the clone (see the sidecar block further down). Previously the merge
        // AND all six ~600MB sidecar writes were held under a single semantic_idx.write(),
        // blocking recall (which needs semantic_idx.read()) for the entire disk-write window
        // — the sleep-consolidation stall that forced --no-hygiene.
        {
            let mut idx = self.semantic_idx.write();
            if idx.delta_needs_merge() {
                idx.merge_delta_into_base();
            }
        }
        // Read BEFORE the payloads clone: any content mutation racing this
        // save then differs from the stored value and forces the next .pld
        // rewrite (see the field doc on pld_mutations).
        let pld_mutations_at_clone = self
            .pld_mutations
            .load(std::sync::atomic::Ordering::Relaxed);
        // Each clone in its OWN statement: a struct-literal initializer's
        // temporary guard lives until the end of the whole statement, so the
        // previous form held ~20 read guards at once — and read `states`
        // TWICE in one statement (cw_refresh_ts), which self-deadlocks when a
        // writer queues between the two acquisitions (parking_lot writer
        // preference blocks same-thread reacquisition). Production deadlock
        // 2026-06-11, caught by the deadlock-detection build.
        let payloads = {
            // Phase 1: collect embedded IDs without holding payloads or states locks.
            // Releasing semantic_idx.read() before acquiring payloads.read() prevents
            // parking_lot write-preference from cascading across all three locks at once.
            // ceiling: payloads.read() is still held during Phase 2 clone — put_memory
            // writes will still queue, but the hold time is ~10x shorter (no embedding alloc).
            // upgrade: Arc<MemoryPayload> in the store HashMap would make clone O(1).
            let embedded_ids: std::collections::HashSet<MemoryId> = {
                let idx = self.semantic_idx.read();
                idx.all_ids().collect()
            };

            // Phase 2: clone payloads under payloads+states only (2 locks, not 3).
            // Build MemoryPayload literals that skip embedding.clone() for HNSW-indexed
            // memories: saves ~12KB × N transient Vec<f32> allocs inside the lock.
            let payloads_r = self.payloads.read();
            let states_r = self.states.read();
            payloads_r
                .iter()
                .filter(|(id, _)| !states_r.get(id).is_some_and(|s| s.deleted))
                .map(|(id, p)| {
                    let q = MemoryPayload {
                        embedding: if embedded_ids.contains(id) {
                            Vec::new()
                        } else {
                            p.embedding.clone()
                        },
                        memory_id: p.memory_id,
                        version: p.version,
                        chunk_hash: p.chunk_hash,
                        created_at_ms: p.created_at_ms,
                        authored_at_ms: p.authored_at_ms,
                        kind: p.kind.clone(),
                        realm: p.realm.clone(),
                        content: p.content.clone(),
                        embedding_model: p.embedding_model.clone(),
                        artifact_refs: p.artifact_refs.clone(),
                        source_session: p.source_session.clone(),
                        source_tool: p.source_tool.clone(),
                        harness: p.harness.clone(),
                        provenance: p.provenance.clone(),
                        candidate: p.candidate,
                        embedding_model_id: p.embedding_model_id.clone(),
                        embedding_dim: p.embedding_dim,
                    };
                    (*id, q)
                })
                .collect()
        };
        let (states, cw_refresh_ts, utility_posteriors) = {
            let states_r = self.states.read();
            let cw: std::collections::HashMap<MemoryId, i64> = states_r
                .iter()
                .filter(|(_, st)| !st.deleted && st.last_cw_refresh_ms > 0)
                .map(|(&id, st)| (id, st.last_cw_refresh_ms))
                .collect();
            // Only memories that actually carry an observation — an untouched
            // (1, 1) prior is what a missing entry already means.
            let util: std::collections::HashMap<MemoryId, (f32, f32)> = states_r
                .iter()
                .filter(|(_, st)| !st.deleted && (st.utility_alpha > 1.0 || st.utility_beta > 1.0))
                .map(|(&id, st)| (id, (st.utility_alpha, st.utility_beta)))
                .collect();
            // Compact out deleted memories — they must not appear in the snapshot
            // so that snapshot resurrection cannot undo a forget().
            let live: std::collections::HashMap<MemoryId, _> = states_r
                .iter()
                .filter(|(_, st)| !st.deleted)
                .map(|(id, st)| (*id, st.clone()))
                .collect();
            (live, cw, util)
        };
        let assoc_edges = self.assoc_edges.read().clone();
        let artifacts = self.artifacts.read().clone();
        let artifact_paths = self.artifact_paths.read().clone();
        let time_idx = self.time_idx.read().clone();
        let keyword_idx = self.keyword_idx.read().clone();
        let artifact_idx = self.artifact_idx.read().clone();
        let triplet_store = self.triplet_store.read().clone();
        let symbol_idx = self.symbol_idx.read().clone();
        let call_graph = self.call_graph.read().clone();
        let code_files = self.code_files.read().clone();
        let semantic_idx = self.semantic_idx.read().clone();
        let coactivation_stats = {
            let mut cs = self.coactivation_stats.read().clone();
            let before = cs.len();
            let removed = crate::field::prune_coactivation_stats(&mut cs, 20);
            eprintln!("[chitta-field] coactivation_stats: {} pairs before prune, {} removed (cap=20/memory)", before, removed);
            cs
        };
        let ack_scores = self.ack_scores.read().clone();
        let correction_states = self.triplet_store.read().correction_states.clone();
        let event_tape = self.event_tape.read().clone();
        let decision_tape = self.decision_tape.read().clone();
        let turiya_monitor = self.turiya_monitor.read().clone();
        let observer_state = self.observer_state.read().clone();
        let interaction_ledger = self.interaction_ledger.read().clone();
        let predicate_store = self.predicate_store.read().clone();
        let recall_provenance = self.recall_provenance.read().clone();
        let mut snap = FullSnapshot {
            snapshot_seqno: seqno,
            payloads,
            states,
            assoc_edges,
            artifacts,
            artifact_paths,
            time_idx,
            keyword_idx,
            artifact_idx,
            triplet_store,
            symbol_idx,
            call_graph,
            code_files,
            semantic_idx,
            coactivation_stats,
            ack_scores,
            correction_states,
            event_tape,
            decision_tape,
            turiya_monitor,
            observer_state,
            interaction_ledger,
            predicate_store,
            recall_provenance,
            cw_refresh_ts,
            utility_posteriors,
            ledger_session_events: self.msg_registry.read().ledger_session_events(),
        };
        let path = self
            .data_dir
            .join(format!("chitta.{:08x}.snapshot", self.instance_id));
        // Write embedding and binary-code sidecars from the live index (before clearing clone).
        let emb_path   = path.with_extension("emb");
        let hdc_path   = path.with_extension("hdc");
        let bin_path   = path.with_extension("bin");
        let mu_path    = path.with_extension("mu");
        let hnsw_path       = path.with_extension("hnsw");
        let delta_path      = path.with_extension("delta.hnsw");
        let realm_hnsw_path = path.with_extension("realm_hnsw");
        let pld_path   = path.with_extension("pld");
        let sup_path   = path.with_extension("sup.json");
        let shdr_path  = path.with_extension("shdr");
        // Store-identity sidecar (PR3): records the vector space (model/dim/text-format) +
        // lineage so snapshot selection and WAL replay can fence foreign-dim/model data.
        {
            let hdr = crate::snapshot::StoreHeader::current(self.lineage_epoch, self.writer_uuid);
            if let Err(e) = hdr.save(&shdr_path) {
                eprintln!("[chitta-field] WARNING: .shdr sidecar save failed: {e}");
            }
        }
        {
            // Embedding/code sidecars saved from the cloned index (snap.semantic_idx) with
            // NO semantic_idx lock held — the six ~600MB disk writes must not block recall,
            // which needs semantic_idx.read(). The delta was merged into base under a brief
            // write above, so this clone is canonical (delta empty). clear_embeddings()
            // below runs after this block, so the clone still carries its vectors here.
            //
            // Dirty-skip: if the index hasn't mutated since this instance's
            // last successful sidecar write, the files on disk are already
            // current — skip the ~800MB of rewrites (THEORY.md §8 Phase 2).
            // Promote delta→base on the clone so save_hnsw() serialises the full
            // graph.  The live index is untouched; only the snapshot clone is swapped.
            snap.semantic_idx.promote_delta_to_base_if_empty();
            let idx = &snap.semantic_idx;
            let idx_mutations = idx.mutation_count();
            let last = self
                .idx_sidecars_saved_at
                .load(std::sync::atomic::Ordering::Relaxed);
            // Existence guard on .emb only: the other sidecars are written
            // conditionally (empty HNSW/centroid produce no file), so their
            // absence matches the previous save rather than invalidating it.
            // Exception: if the .hnsw on disk is a 9-byte stub (empty base, nodes
            // are all in delta) the promote above has now swapped them — force a
            // rewrite even if the mutation counter hasn't changed.
            // Stub guard: only relevant when the HNSW graph has actual nodes
            // (i.e. total_embedding_count >= HNSW_THRESHOLD and promote ran).
            // Below the threshold the HNSW is intentionally not built and
            // save_hnsw() legitimately writes a ≤9-byte stub — that is correct
            // and must not force a sidecar rewrite.
            let hnsw_stub = idx.hnsw_len() > 0
                && hnsw_path.metadata().map(|m| m.len() <= 9).unwrap_or(false);
            let clean = last == idx_mutations && emb_path.exists() && !hnsw_stub;
            if clean {
                eprintln!("[chitta-field] index sidecars unchanged since last save — skipping rewrite");
            } else {
                let _ = idx.save_embeddings_sidecar(&emb_path);
                let _ = idx.save_binary_sidecar(&bin_path);
                let _ = idx.save_centroid_sidecar(&mu_path);
                // HNSW sidecars: skipped when CHITTA_NO_HNSW_SIDECAR=1. Below
                // flat_scan_max the graph is never consulted, so re-serialising it
                // each snapshot is dead I/O; the flat-scan path is unaffected.
                if !crate::hnsw::skip_hnsw_sidecar() {
                    let _ = idx.save_hnsw(&hnsw_path);
                    let _ = idx.save_delta_hnsw(&delta_path);
                    let _ = idx.save_realm_hnsw(&realm_hnsw_path);
                }
                self.idx_sidecars_saved_at
                    .store(idx_mutations, std::sync::atomic::Ordering::Relaxed);
            }
            // Optional startup caches follow the same dirty-skip discipline.
            // The clone still owns the embeddings here; .emb was saved BEFORE
            // this applies the next loader's normalization. No store guards held.
            if !clean || ["lsh", "turbo", "turbo.meta"].iter().any(|ext| !path.with_extension(ext).exists()) {
                snap.semantic_idx.prepare_startup_caches(&path);
            }
        }
        // Tape identity includes same-length edits and dictionary changes. The
        // helper skips reconstruction when unchanged; derived organs stay optional.
        let _ = crate::startup_cache::load_or_rebuild_organs(
            &snap.event_tape, Some(&path.with_extension("organs")));
        // Save HDC sidecar — avoids tokenize+encode rebuild on next startup.
        // Dirty-skipped when the store hasn't mutated since the last write
        // (the sidecar is a lossy cache; a same-content file stays valid).
        {
            let hdc = self.hdc_idx.read();
            let count = hdc.mutation_count();
            let last = self
                .hdc_sidecar_saved_at
                .load(std::sync::atomic::Ordering::Relaxed);
            if last == count && hdc_path.exists() {
                eprintln!("[chitta-field] hdc sidecar unchanged since last save — skipping rewrite");
            } else {
                let n = hdc.save_sidecar(&hdc_path)
                    .map(|n| n.to_string())
                    .unwrap_or_else(|e| format!("err:{e}"));
                self.hdc_sidecar_saved_at
                    .store(count, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[chitta-field] hdc sidecar: {} memories written to {:?}", n, hdc_path);
            }
        }
        // Save payload content sidecar (.pld) before clearing from bincode. If this
        // fails we MUST NOT strip content from the bincode body — otherwise the content
        // would exist in neither place (silent total content loss). Abort the snapshot.
        let mut snap = snap;
        // .pld dirty-skip: content is the ONE thing with no fallback copy, so
        // skip only when no content mutation happened since the last write by
        // this instance AND the existing file is plausibly intact.
        let pld_clean = self
            .pld_saved_at
            .load(std::sync::atomic::Ordering::Relaxed)
            == pld_mutations_at_clone
            && std::fs::metadata(&pld_path).map(|m| m.len() >= 16).unwrap_or(false);
        if pld_clean {
            eprintln!("[chitta-field] .pld sidecar unchanged since last save — skipping rewrite");
        } else {
            FullSnapshot::save_payload_sidecar(&pld_path, &snap.payloads)
                .map_err(|e| FieldError::Manifest(format!(
                    "payload (.pld) sidecar save failed, aborting snapshot to avoid content loss: {e}"
                )))?;
            self.pld_saved_at
                .store(pld_mutations_at_clone, std::sync::atomic::Ordering::Relaxed);
        }
        // Stage B: persist retrieval surfaces to the .rsf sidecar (never bincode).
        // Best-effort: content is the authoritative copy, so a failed .rsf just means
        // recall falls back to embedding content — no data loss, unlike .pld.
        {
            let surfaces = self.retrieval_surfaces.read();
            if !surfaces.is_empty() {
                let rsf_path = path.with_extension("rsf");
                if let Err(e) = FullSnapshot::save_retrieval_surface_sidecar(&rsf_path, &surfaces) {
                    eprintln!("[chitta-field] .rsf sidecar save failed (non-fatal): {e}");
                }
            }
        }
        // Strip content and embeddings from bincode (live in sidecars now).
        for payload in snap.payloads.values_mut() {
            payload.content.clear();
            payload.content.shrink_to_fit();
        }
        snap.semantic_idx.clear_embeddings();
        let _ = snap.triplet_store.save_supersession_sidecar(&sup_path);
        snap.triplet_store.clean_for_load();
        snap.triplet_store.clear_indexes_for_save();
        // Diagnostic: per-field serialized sizes to identify snapshot bloat.
        {
            let sz = |v: u64| format!("{:.1}MB", v as f64 / 1_000_000.0);
            eprintln!("[size] payloads:          {}", sz(bincode::serialized_size(&snap.payloads).unwrap_or(0)));
            eprintln!("[size] states:            {}", sz(bincode::serialized_size(&snap.states).unwrap_or(0)));
            eprintln!("[size] assoc_edges:       {}", sz(bincode::serialized_size(&snap.assoc_edges).unwrap_or(0)));
            eprintln!("[size] artifacts:         {}", sz(bincode::serialized_size(&snap.artifacts).unwrap_or(0)));
            eprintln!("[size] time_idx:          {}", sz(bincode::serialized_size(&snap.time_idx).unwrap_or(0)));
            eprintln!("[size] keyword_idx:       {}", sz(bincode::serialized_size(&snap.keyword_idx).unwrap_or(0)));
            eprintln!("[size] triplet_store:     {}", sz(bincode::serialized_size(&snap.triplet_store).unwrap_or(0)));
            eprintln!("[size] symbol_idx:        {}", sz(bincode::serialized_size(&snap.symbol_idx).unwrap_or(0)));
            eprintln!("[size] call_graph:        {}", sz(bincode::serialized_size(&snap.call_graph).unwrap_or(0)));
            eprintln!("[size] code_files:        {}", sz(bincode::serialized_size(&snap.code_files).unwrap_or(0)));
            eprintln!("[size] semantic_idx:      {}", sz(bincode::serialized_size(&snap.semantic_idx).unwrap_or(0)));
            eprintln!("[size] coactivation:      {}", sz(bincode::serialized_size(&snap.coactivation_stats).unwrap_or(0)));
            eprintln!("[size] ack_scores:        {}", sz(bincode::serialized_size(&snap.ack_scores).unwrap_or(0)));
            eprintln!("[size] artifact_paths:    {}", sz(bincode::serialized_size(&snap.artifact_paths).unwrap_or(0)));
            eprintln!("[size] artifact_idx:      {}", sz(bincode::serialized_size(&snap.artifact_idx).unwrap_or(0)));
        }
        snap.save(&path)?;
        // Commit record: the manifest ties the snapshot + sidecars into one
        // committed family (open() prefers a validated family over fence-based
        // selection). Written LAST — a crash anywhere above leaves the previous
        // generation's manifest pointing at the previous intact family.
        {
            use crate::manifest::{CheckpointSet, FileRef, Manifest};
            let file_ref = |p: &std::path::Path| -> Option<FileRef> {
                let name = p.file_name()?.to_string_lossy().into_owned();
                let size_bytes = std::fs::metadata(p).ok()?.len();
                Some(FileRef { name, size_bytes })
            };
            if let Some(snapshot_ref) = file_ref(&path) {
                let mut manifest = Manifest::load(&self.data_dir)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| {
                        Manifest::new_empty(
                            crate::ops::EMBED_MODEL_ID,
                            crate::ops::EMBED_DIM as u16,
                        )
                    });
                manifest.generation += 1;
                manifest.last_seqno = seqno;
                // Per-writer coverage vector (THEORY.md §4): everything the
                // in-memory state contains (open replay ⊔ sync_foreign) plus
                // our own ops up to this save.
                let covered: std::collections::BTreeMap<String, u64> = {
                    let mut cov = self.wal_coverage.read().clone();
                    let own = cov.entry(self.instance_id).or_insert(0);
                    if seqno > *own { *own = seqno; }
                    cov.iter().map(|(i, s)| (format!("{:08x}", i), *s)).collect()
                };
                // Only files that actually exist are recorded (e.g. the .sup
                // sidecar save is best-effort) — validation checks what the
                // commit promised, nothing more.
                let family = CheckpointSet {
                    snapshot: snapshot_ref,
                    sidecars: [
                        &emb_path, &hdc_path, &bin_path, &mu_path, &hnsw_path,
                        &delta_path, &realm_hnsw_path, &pld_path, &sup_path, &shdr_path,
                    ]
                    .iter()
                    .filter_map(|p| file_ref(p))
                    .collect(),
                    snapshot_seqno: seqno,
                    covered,
                };
                manifest
                    .families
                    .insert(format!("{:08x}", self.instance_id), family.clone());
                manifest.checkpoints = Some(family);
                if let Err(e) = manifest.save(&self.data_dir) {
                    eprintln!(
                        "[chitta-field] WARNING: manifest commit failed (snapshot itself is durable): {e}"
                    );
                }
            }
        }
        // Prune old families ONLY after the new snapshot + .pld are durably written
        // (save() fsyncs the file and parent dir; save_payload_sidecar fsyncs the .pld).
        // Pruning before durability would delete the fallback the new snapshot replaces.
        prune_old_snapshots(&self.data_dir, 2);
        // Ghost janitor: dead-instance residue + resurrection accounting
        // (7-day age gate protects live peers' seen_offsets).
        janitor_sweep(&self.data_dir, self.instance_id, 7 * 86_400);
        Ok(())
    }


    /// Compact WAL: save full snapshot then delete WAL segments covered by it.
    /// Coverage is per-writer (THEORY.md §4) — see prune_covered_segments.
    /// This bounds WAL growth and speeds up startup replay.
    pub fn compact_wal(&self) -> Result<usize> {
        let count = {
            let states = self.states.read();
            states.values().filter(|s| !s.deleted).count()
        };
        if count < 100 {
            return Err(FieldError::Other(format!(
                "refusing compact_wal on near-empty store ({} live memories, minimum 100)", count
            )));
        }
        self.save_full_snapshot()?;
        // Safe pruning rule (THEORY.md §4): a segment of instance i may be
        // deleted iff our coverage vector dominates it — i.e. every op in it
        // is provably contained in the snapshot we just committed. The old
        // scalar rule (first_seqno < snapshot_seqno) compared seqnos across
        // writers, which can delete a concurrent writer's UNCOVERED ops:
        // overlapping seqno ranges make a foreign segment look covered.
        let mut covered = self.wal_coverage.read().clone();
        let own = covered.entry(self.instance_id).or_insert(0);
        let last = self.log.read().last_seqno();
        if last > *own { *own = last; }

        let seg_dir = self.data_dir.join("segments");
        let deleted = prune_covered_segments(&seg_dir, &covered, self.instance_id);
        Ok(deleted)
    }

    /// Count WAL segment files in the segments/ directory.
    pub fn wal_segment_count(&self) -> usize {
        let seg_dir = self.data_dir.join("segments");
        std::fs::read_dir(&seg_dir)
            .map(|rd| rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "seg"))
                .count())
            .unwrap_or(0)
    }

    /// Compact WAL if segment count exceeds `threshold`, with a 1-hour cooldown.
    /// Returns Ok(true) if compaction ran, Ok(false) if skipped (under threshold or cooldown).
    pub fn maybe_compact_wal(&self, threshold: usize) -> Result<bool> {
        if self.wal_segment_count() <= threshold { return Ok(false); }
        let now = now_ms();
        let last = self.last_compact_ms.load(std::sync::atomic::Ordering::Relaxed);
        if now - last < 3_600_000 { return Ok(false); }
        self.last_compact_ms.store(now, std::sync::atomic::Ordering::Relaxed);
        self.compact_wal()?;
        eprintln!("[store] maybe_compact_wal: compacted (segments > {})", threshold);
        Ok(true)
    }

    /// Prune episode memories: delete those older than `max_age_days` with strength < 0.3,
    /// then cap total episode count to `max_count` by removing the oldest.
    pub fn prune_episodes(&self, max_age_days: u64, max_count: usize) -> Result<usize> {
        let cutoff_ms = now_ms() - (max_age_days as i64) * 86_400_000;
        let mut episodes: Vec<(i64, MemoryId)> = {
            let payloads = self.payloads.read();
            payloads.iter()
                .filter(|(_, p)| p.kind == "episode")
                .map(|(id, p)| (p.created_at_ms, *id))
                .collect()
        };

        let mut deleted = 0usize;
        for &(created_ms, id) in &episodes {
            if created_ms < cutoff_ms {
                if let Ok(state) = self.get_state(id) {
                    if state.strength < 0.3 {
                        let _ = self.forget(id);
                        deleted += 1;
                    }
                }
            }
        }

        episodes.retain(|(_, id)| self.get_memory(*id).is_ok());
        if episodes.len() > max_count {
            episodes.sort_unstable_by_key(|(ts, _)| *ts);
            let target = (max_count as f64 * 0.8) as usize;
            let to_delete = episodes.len().saturating_sub(target);
            for &(_, id) in &episodes[..to_delete] {
                let _ = self.forget(id);
                deleted += 1;
            }
        }
        if deleted > 0 {
            eprintln!("[store] prune_episodes: deleted {} episode memories", deleted);
        }
        Ok(deleted)
    }

    /// Promote staged memories that have been recalled (access_count >= 1),
    /// and prune staged memories older than 7 days that were never recalled.
    /// Returns (promoted, pruned).
    pub fn promote_staged_memories(&self) -> Result<(usize, usize)> {
        let cutoff_ms = now_ms() - 7 * 86_400_000;
        let candidates: Vec<(MemoryId, i64, u32)> = {
            let states = self.states.read();
            states.values()
                .filter(|s| !s.deleted && s.staged)
                .map(|s| (s.memory_id, s.created_at_ms, s.access_count))
                .collect()
        };

        let mut promoted = 0usize;
        let mut pruned   = 0usize;
        let ts = now_ms();
        for (id, created_ms, access_count) in candidates {
            if access_count >= 1 {
                let delta = crate::ops::StateDeltaOp {
                    memory_id: id,
                    strength_delta: None,
                    confidence_delta: None,
                    decay_rate: None,
                    touch: false,
                    pin: None,
                    op_ts_ms: ts,
                    status: None,
                    epistemic_status: None,
                    staged: Some(false),
                    invalidated_by: None,
                };
                let _ = self.log.write().append(&crate::ops::Op::UpdateState(delta.clone()));
                if let Some(s) = self.states.write().get_mut(&id) {
                    s.staged = false;
                }
                promoted += 1;
            } else if created_ms < cutoff_ms {
                let _ = self.forget(id);
                pruned += 1;
            }
        }
        if promoted > 0 || pruned > 0 {
            eprintln!("[store] write_gate: promoted={} pruned={}", promoted, pruned);
        }
        Ok((promoted, pruned))
    }

    pub fn log_symbol_event(
        &self,
        symbol_name: String,
        file_path: String,
        symbol_id: Option<u64>,
        kind: crate::organ::symbol_events::SymbolEventKind,
        session_id: String,
        harness: String,
        memory_id: Option<crate::ids::MemoryId>,
        notes: Option<String>,
        timestamp_ms: i64,
    ) -> u64 {
        let ev = crate::organ::symbol_events::SymbolEvent {
            id: 0,
            symbol_name: symbol_name.clone(),
            file_path: file_path.clone(),
            symbol_id,
            kind,
            session_id: session_id.clone(),
            harness: harness.clone(),
            memory_id,
            timestamp_ms,
            notes: notes.clone(),
        };
        let id = self.symbol_event_log.write().log(ev);
        let _ = self.log.write().append(&crate::ops::Op::SymbolEvent(
            crate::ops::SymbolEventOp {
                id,
                symbol_name,
                file_path,
                symbol_id,
                kind: kind.to_u8(),
                session_id,
                harness,
                memory_id,
                timestamp_ms,
                notes,
            },
        ));
        id
    }

    pub fn query_symbol_events(
        &self,
        symbol_name: Option<&str>,
        file_path: Option<&str>,
        limit: usize,
    ) -> String {
        let log = self.symbol_event_log.read();
        let hits = log.query(symbol_name, file_path, limit);
        let arr: Vec<serde_json::Value> = hits.iter().map(|e| serde_json::json!({
            "id": e.id,
            "symbol_name": e.symbol_name,
            "file_path": e.file_path,
            "symbol_id": e.symbol_id,
            "kind": e.kind.as_str(),
            "session_id": e.session_id,
            "harness": e.harness,
            "memory_id": e.memory_id,
            "timestamp_ms": e.timestamp_ms,
            "notes": e.notes,
        })).collect();
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn symbol_stale_for_memory(&self, id: crate::ids::MemoryId) -> Option<String> {
        // 1. Definitive WAL-written invalidation.
        {
            let states = self.states.read();
            match states.get(&id) {
                None => return Some("memory not found".to_string()),
                Some(s) => if let Some(ref reason) = s.invalidated_by {
                    return Some(reason.clone());
                },
            }
        }
        // 2. Check artifact refs: if source file no longer indexed or line range gone.
        let artifact_refs = {
            let payloads = self.payloads.read();
            match payloads.get(&id) {
                Some(p) => p.artifact_refs.clone(),
                None => return None,
            }
        };
        if artifact_refs.is_empty() { return None; }
        let artifact_paths = self.artifact_paths.read();
        let symbol_idx = self.symbol_idx.read();
        for aref in &artifact_refs {
            if let Some(file_path) = artifact_paths.get(&aref.artifact_id) {
                let syms = symbol_idx.by_file(file_path);
                if syms.is_empty() {
                    return Some(format!("source file {} no longer indexed", file_path));
                }
                if aref.line_start > 0 {
                    let covered = syms.iter().any(|s| {
                        s.line_start <= aref.line_start && s.line_end >= aref.line_end
                    });
                    if !covered {
                        return Some(format!("symbol at {}:{} no longer present", file_path, aref.line_start));
                    }
                }
            }
        }
        None
    }

    pub fn memory_claim_info_json(&self, id: crate::ids::MemoryId, now_ms: i64) -> String {
        let payloads = self.payloads.read();
        let states = self.states.read();
        let state = match states.get(&id) { Some(s) => s, None => return "{}".to_string() };
        let payload = match payloads.get(&id) { Some(p) => p, None => return "{}".to_string() };
        let age_days = (now_ms - state.created_at_ms).max(0) as f64 / 86_400_000.0;
        let j = serde_json::json!({
            "staged": state.staged,
            "invalidated_by": state.invalidated_by,
            "source_session": payload.source_session,
            "source_tool": payload.source_tool,
            "harness": payload.harness,
            "created_at_ms": state.created_at_ms,
            "age_days": age_days,
        });
        j.to_string()
    }

    /// Find pairs of memories that disagree across harnesses (claude-code vs codex).
    /// Returns JSON array of conflict pairs sorted by cosine distance (most divergent first).
    pub fn query_cross_harness_conflicts(&self, realm: &str, limit: usize, min_score: f32) -> String {
        let payloads = self.payloads.read();
        let states   = self.states.read();
        let idx      = self.semantic_idx.read();

        // Collect all live, non-staged memories with a harness tag
        let candidates: Vec<(crate::ids::MemoryId, &[f32], &str)> = payloads.iter()
            .filter_map(|(id, p)| {
                let s = states.get(id)?;
                if s.deleted || s.staged { return None; }
                if !realm.is_empty() && p.realm != realm { return None; }
                let harness = p.harness.as_deref()?;
                let emb = idx
                    .get_embedding(*id)
                    .or_else(|| (!p.embedding.is_empty()).then_some(p.embedding.as_slice()))?;
                Some((*id, emb, harness))
            })
            .collect();

        // For each candidate, find its nearest cross-harness neighbour
        let mut conflicts: Vec<serde_json::Value> = Vec::new();
        let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();

        for (id_a, emb_a, harness_a) in &candidates {
            let neighbors = idx.search(emb_a, 20, None, None);
            for nb in &neighbors {
                if nb.memory_id == *id_a { continue; }
                let id_b = nb.memory_id;
                let key = if *id_a < id_b { (*id_a, id_b) } else { (id_b, *id_a) };
                if seen.contains(&key) { continue; }
                let harness_b = match payloads.get(&id_b) {
                    Some(p) => match p.harness.as_deref() { Some(h) => h, None => continue },
                    None => continue,
                };
                if harness_a == &harness_b { continue; }
                // semantic similarity → disagreement = 1 - similarity
                let disagreement = 1.0 - nb.cosine_similarity;
                if disagreement < min_score { continue; }
                seen.insert(key);
                let content_a = String::from_utf8_lossy(payloads[id_a].content.as_slice()).into_owned();
                let content_b = String::from_utf8_lossy(payloads[&id_b].content.as_slice()).into_owned();
                // char-boundary-safe truncation: byte-slicing (&s[..200]) panics when byte
                // 200 splits a multibyte codepoint, and this runs under an extern "C" FFI
                // call (cf_query_cross_harness_conflicts) — an unwind there aborts the daemon.
                let snippet_a: String = content_a.chars().take(200).collect();
                let snippet_b: String = content_b.chars().take(200).collect();
                conflicts.push(serde_json::json!({
                    "harness_a": harness_a,
                    "memory_a_id": id_a,
                    "harness_b": harness_b,
                    "memory_b_id": id_b,
                    "disagreement_score": disagreement,
                    "snippet_a": snippet_a,
                    "snippet_b": snippet_b,
                }));
                if conflicts.len() >= limit { break; }
            }
            if conflicts.len() >= limit { break; }
        }

        conflicts.sort_by(|a, b| {
            let da = a["disagreement_score"].as_f64().unwrap_or(0.0);
            let db = b["disagreement_score"].as_f64().unwrap_or(0.0);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        serde_json::to_string(&conflicts).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn mark_memory_invalidated(&self, memory_id: crate::ids::MemoryId, reason: String) -> bool {
        let exists = self.states.read().contains_key(&memory_id);
        if !exists { return false; }
        let now = now_ms();
        let delta = crate::ops::StateDeltaOp {
            memory_id,
            op_ts_ms: now,
            strength_delta: None,
            confidence_delta: None,
            decay_rate: None,
            touch: false,
            pin: None,
            status: None,
            epistemic_status: None,
            staged: None,
            invalidated_by: Some(reason.clone()),
        };
        let _ = self.log.write().append(&crate::ops::Op::UpdateState(delta.clone()));
        if let Some(s) = self.states.write().get_mut(&memory_id) {
            s.invalidated_by = Some(reason);
        }
        true
    }

    /// Run a single tier demotion pass over all memories.
    /// Returns `(demoted_count, deleted_count)`.
    ///
    /// Tiers: 0=L1 (hippocampus), 1=L2 (cortex), 2=L3 (archive), then delete.
    /// Uses `access_count` as rehearsal proxy and `strength` as utility proxy.
    pub fn run_demotion_pass(&self, now_ms: i64) -> Result<(usize, usize)> {
        const L1_TO_L2_AGE_MS: i64 = 7 * 24 * 3600 * 1000;
        const L1_TO_L2_LAST_ACCESS_MS: i64 = 2 * 24 * 3600 * 1000;
        const L1_TO_L2_MAX_STRENGTH: f32 = 0.80;

        const L2_TO_L3_AGE_BASE_MS: i64 = 45 * 24 * 3600 * 1000;
        const L2_TO_L3_REHEARSAL_BONUS_MS: i64 = 7 * 24 * 3600 * 1000;
        const L2_TO_L3_LAST_ACCESS_MS: i64 = 14 * 24 * 3600 * 1000;
        const L2_TO_L3_MAX_STRENGTH: f32 = 0.50;

        const L3_DELETE_AGE_BASE_MS: i64 = 365 * 24 * 3600 * 1000;
        const L3_DELETE_REHEARSAL_BONUS_MS: i64 = 30 * 24 * 3600 * 1000;
        const L3_DELETE_LAST_ACCESS_MS: i64 = 120 * 24 * 3600 * 1000;
        const L3_DELETE_MAX_STRENGTH: f32 = 0.12;
        const L3_DELETE_MAX_UTILITY: f32 = 0.80;

        let mut to_demote: Vec<(MemoryId, u8)> = Vec::new();
        let mut to_delete: Vec<MemoryId> = Vec::new();

        {
            let states = self.states.read();
            for (&memory_id, state) in states.iter() {
                if state.deleted || state.pinned {
                    continue;
                }

                // Strength >= L3_DELETE_MAX_UTILITY means never delete
                let age_ms = now_ms - state.created_at_ms;
                let last_access_ago = now_ms - state.last_accessed_ms;
                // access_count serves as rehearsal proxy; cap at 8 for bonus calc
                let rehearsal = state.access_count.min(8) as i64;

                match state.tier {
                    0 => {
                        // L1 → L2
                        if age_ms >= L1_TO_L2_AGE_MS
                            && last_access_ago >= L1_TO_L2_LAST_ACCESS_MS
                            && state.strength < L1_TO_L2_MAX_STRENGTH
                        {
                            to_demote.push((memory_id, 1));
                        }
                    }
                    1 => {
                        // L2 → L3
                        let threshold =
                            L2_TO_L3_AGE_BASE_MS + rehearsal * L2_TO_L3_REHEARSAL_BONUS_MS;
                        if age_ms >= threshold
                            && last_access_ago >= L2_TO_L3_LAST_ACCESS_MS
                            && state.strength < L2_TO_L3_MAX_STRENGTH
                        {
                            to_demote.push((memory_id, 2));
                        }
                    }
                    2 => {
                        // L3 → delete
                        let threshold =
                            L3_DELETE_AGE_BASE_MS + rehearsal * L3_DELETE_REHEARSAL_BONUS_MS;
                        if age_ms >= threshold
                            && last_access_ago >= L3_DELETE_LAST_ACCESS_MS
                            && state.strength < L3_DELETE_MAX_STRENGTH
                            && state.strength < L3_DELETE_MAX_UTILITY
                        {
                            to_delete.push(memory_id);
                        }
                    }
                    _ => {}
                }
            }
        }

        let demoted = to_demote.len();
        let deleted = to_delete.len();

        for (id, new_tier) in to_demote {
            let op = Op::DemoteMemory(DemoteMemoryOp {
                memory_id: id,
                new_tier,
            });
            self.log.write().append(&op)?;
            if let Some(state) = self.states.write().get_mut(&id) {
                state.tier = new_tier;
            }
        }
        for id in to_delete {
            self.forget(id)?;
        }

        // Gate B outflow: hourly CoRetrieved decay + floor-prune. An untouched
        // co-retrieval edge halves in ~34 passes (0.98^34); anything that decays
        // below 0.05 is noise and is dropped. Paired with the drain-side 0.1
        // materialization floor so decay isn't fighting a firehose.
        if self
            .plasticity_decay_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            let (survivors, pruned) = self.assoc_decay(EdgeType::CoRetrieved, 0.98, 0.05, true);
            if pruned > 0 {
                eprintln!("[plasticity] CoRetrieved decay: {survivors} survive, {pruned} pruned");
            }
        }

        Ok((demoted, deleted))
    }

    /// Encode all memories that don't yet have a sparse code.
    pub fn encode_all_unindexed(&self) -> Result<usize> {
        // Collect only memories that CAN encode: skip deleted (payloads
        // outlive soft-delete), empty/foreign-dim embeddings (the stripped-
        // snapshot rehydrator deliberately leaves deleted ones empty), and
        // ids whose sparse code came back empty before (runtime skip-set —
        // retried after restart). Without these filters the same ~7.6k
        // unencodable memories re-encoded on every consolidation cycle.
        let ids: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            let idx = self.semantic_idx.read();
            let cortical = self.cortical_idx.read();
            let skip = self.encode_skip.read();
            payloads
                .iter()
                .filter(|(id, p)| {
                    !cortical.mem_codes.contains_key(*id)
                        && !skip.contains(*id)
                        && (idx.get_embedding(**id).is_some()
                            || p.embedding.len() == EMBED_DIM)
                        && states.get(*id).map(|s| !s.deleted).unwrap_or(false)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        let count = ids.len();
        for id in ids {
            self.encode_memory(id)?;
        }
        Ok(count)
    }

    /// Train a ProductQuantizer from the residuals of all encoded memories.
    /// Requires at least 256 memories with sparse codes.
    pub fn train_pq(&self) -> Result<()> {
        // Collect residuals: for each memory with a sparse code, decode and subtract
        let residuals: Vec<Vec<f32>> = {
            let payloads = self.payloads.read();
            let idx = self.semantic_idx.read();
            let encoder = self.sparse_encoder.read();
            let cortical = self.cortical_idx.read();

            cortical
                .mem_codes
                .iter()
                .filter_map(|(&memory_id, code)| {
                    let embedding = idx
                        .get_embedding(memory_id)
                        .map(|e| e.to_vec())
                        .or_else(|| payloads.get(&memory_id).map(|p| p.embedding.clone()))?;
                    if embedding.len() != crate::ops::EMBED_DIM {
                        return None;
                    }
                    let decoded = encoder.decode(code);
                    let residual: Vec<f32> = embedding
                        .iter()
                        .zip(decoded.iter())
                        .map(|(e, d)| e - d)
                        .collect();
                    Some(residual)
                })
                .collect()
        };

        let pq = ProductQuantizer::train(&residuals, 20)
            .map_err(crate::error::FieldError::Manifest)?;

        let codebook_bytes = bincode::serialize(&pq)
            .map_err(|e| crate::error::FieldError::Serialization(e.to_string()))?;

        let op = Op::TrainPQ(TrainPQOp { codebook_bytes });
        self.log.write().append(&op)?;

        self.cortical_idx.write().set_pq(pq);

        Ok(())
    }

    /// Encode PQ residual for a single memory. The PQ must already be trained.
    pub fn encode_pq_memory(&self, memory_id: MemoryId) -> Result<()> {
        let embedding = self.embedding_of(memory_id);
        let Some(embedding) = embedding else {
            return Ok(());
        };
        if embedding.len() != crate::ops::EMBED_DIM {
            return Ok(());
        }

        let decoded = {
            let encoder = self.sparse_encoder.read();
            let cortical = self.cortical_idx.read();
            let code = match cortical.mem_codes.get(&memory_id) {
                Some(c) => c.clone(),
                None => return Ok(()),
            };
            encoder.decode(&code)
        };

        let residual: Vec<f32> = embedding
            .iter()
            .zip(decoded.iter())
            .map(|(e, d)| e - d)
            .collect();

        let codes = {
            let cortical = self.cortical_idx.read();
            let pq = match &cortical.pq {
                Some(pq) => pq,
                None => return Ok(()),
            };
            pq.quantize(&residual)
        };

        let pq_bytes: Vec<u8> = codes.to_vec();
        let op = Op::UpdateResidualPQ(UpdateResidualPQOp {
            memory_id,
            pq_bytes,
        });
        self.log.write().append(&op)?;

        self.cortical_idx.write().index_pq(memory_id, codes);

        Ok(())
    }

    /// Encode PQ residuals for all memories that have sparse codes but no PQ code.
    /// If PQ is not yet trained, trains it first.
    /// Returns the count of memories PQ-encoded.
    pub fn encode_all_pq(&self) -> Result<usize> {
        if !self.cortical_idx.read().is_pq_trained() {
            self.train_pq()?;
        }

        let ids: Vec<MemoryId> = {
            let cortical = self.cortical_idx.read();
            cortical
                .mem_codes
                .keys()
                .filter(|id| !cortical.mem_pq.contains_key(id))
                .copied()
                .collect()
        };

        let count = ids.len();
        for id in ids {
            self.encode_pq_memory(id)?;
        }

        Ok(count)
    }

    /// Return how many memories have PQ residual codes.
    pub fn pq_count(&self) -> usize {
        self.cortical_idx.read().pq_count()
    }

    // ── Layer 9: Wisdom Homeostasis ───────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn enroll_wisdom_lineage(
        &self,
        wisdom_candidate_id: u64,
        claim: String,
        envelope_json: String,
        seed_episode_ids: Vec<u64>,
        seed_surprise_ids: Vec<u64>,
        seed_intervention_ids: Vec<u64>,
        seed_debt_ids: Vec<u64>,
        ancestor_lineage_id: Option<u64>,
        derivation_relation: Option<String>,
    ) -> Result<u64> {
        use crate::organ::wisdom_lineage::ApplicabilityEnvelope;
        let now = now_ms();
        let envelope: ApplicabilityEnvelope =
            serde_json::from_str(&envelope_json).unwrap_or_default();
        let lineage_id = self.wisdom_lineage_store.write().enroll(
            wisdom_candidate_id, claim.clone(), envelope, seed_episode_ids.clone(),
            seed_surprise_ids.clone(), seed_intervention_ids.clone(), seed_debt_ids.clone(),
            ancestor_lineage_id, derivation_relation.clone(), now,
        );
        self.log.write().append(&Op::UpsertWisdomLineage(
            crate::ops::UpsertWisdomLineageOp {
                lineage_id,
                wisdom_candidate_id,
                claim,
                envelope_json,
                seed_episode_ids,
                seed_surprise_ids,
                seed_intervention_ids,
                seed_debt_ids,
                ancestor_lineage_id,
                derivation_version: 0,
                derivation_relation,
                rederive_ttl_ms: crate::organ::wisdom_lineage::DEFAULT_REDERIVE_TTL_MS,
                created_ms: now,
                updated_ms: now,
            },
        ))?;
        Ok(lineage_id)
    }

    pub fn transition_wisdom_lineage(
        &self,
        lineage_id: u64,
        new_state: u8,
        reason: String,
        rederive_task_id: Option<u64>,
    ) -> Result<bool> {
        use crate::organ::wisdom_lineage::LineageState;
        let now = now_ms();
        let old_state = self.wisdom_lineage_store.read()
            .get(lineage_id).map(|l| l.state.as_u8()).unwrap_or(0);
        let ok = self.wisdom_lineage_store.write().transition_state(
            lineage_id, LineageState::from_u8(new_state), &reason, rederive_task_id, now,
        );
        if ok {
            self.log.write().append(&Op::TransitionLineage(
                crate::ops::TransitionLineageOp {
                    lineage_id, old_state, new_state,
                    reason, rederive_task_id, transitioned_ms: now,
                },
            ))?;
        }
        Ok(ok)
    }

    pub fn close_rederive(
        &self,
        lineage_id: u64,
        action: u8,
        new_envelope_json: Option<String>,
        fork_claim: Option<String>,
        fork_lineage_id: Option<u64>,
    ) -> Result<()> {
        use crate::organ::wisdom_lineage::{ApplicabilityEnvelope, RederiveAction};
        let now = now_ms();
        let new_envelope = new_envelope_json.as_deref()
            .and_then(|j| serde_json::from_str::<ApplicabilityEnvelope>(j).ok());
        self.wisdom_lineage_store.write().close_rederive(
            lineage_id, RederiveAction::from_u8(action),
            new_envelope, fork_claim.clone(), fork_lineage_id, now,
        );
        self.log.write().append(&Op::CloseRederive(
            crate::ops::CloseRederiveOp {
                lineage_id, action,
                new_envelope_json, fork_claim, fork_lineage_id, closed_ms: now,
            },
        ))?;
        Ok(())
    }

    pub fn query_wisdom_lineages(
        &self,
        state_str: Option<&str>,
        domain: Option<&str>,
        limit: usize,
    ) -> Vec<crate::organ::wisdom_lineage::WisdomLineage> {
        use crate::organ::wisdom_lineage::LineageState;
        let state_filter = state_str.and_then(|s| match s {
            "trusted" => Some(LineageState::Trusted),
            "watch" => Some(LineageState::Watch),
            "inflamed" => Some(LineageState::Inflamed),
            "demoted" => Some(LineageState::Demoted),
            _ => None,
        });
        self.wisdom_lineage_store.read()
            .query(state_filter, domain, limit)
            .into_iter().cloned().collect()
    }

    pub fn get_wisdom_lineage(
        &self, id: u64,
    ) -> Option<crate::organ::wisdom_lineage::WisdomLineage> {
        self.wisdom_lineage_store.read().get(id).cloned()
    }

    pub fn wisdom_lineage_stats(&self) -> crate::organ::wisdom_lineage::WisdomLineageStats {
        self.wisdom_lineage_store.read().stats()
    }

    /// Grow staleness on stale lineages and return IDs that transitioned.
    pub fn tick_lineage_staleness(&self) -> Result<Vec<u64>> {
        let now = now_ms();
        let transitioned = self.wisdom_lineage_store.write().tick_staleness(now);
        for &lineage_id in &transitioned {
            if let Some(l) = self.wisdom_lineage_store.read().get(lineage_id) {
                self.log.write().append(&Op::AdjudicateLineage(
                    crate::ops::AdjudicateLineageOp {
                        lineage_id,
                        support_mass: l.support_mass,
                        contradiction_mass: l.contradiction_mass,
                        staleness_mass: l.staleness_mass,
                        last_supported_ms: l.last_supported_ms,
                        last_challenged_ms: l.last_challenged_ms,
                        adjudicated_ms: now,
                    },
                ))?;
                self.log.write().append(&Op::TransitionLineage(
                    crate::ops::TransitionLineageOp {
                        lineage_id,
                        old_state: 0,
                        new_state: l.state.as_u8(),
                        reason: "staleness_tick".to_string(),
                        rederive_task_id: None,
                        transitioned_ms: now,
                    },
                ))?;
            }
        }
        Ok(transitioned)
    }

    /// Return IDs of Inflamed lineages whose re-derive TTL has expired.
    pub fn lineage_expiry_check(&self) -> Vec<u64> {
        self.wisdom_lineage_store.read().expiry_check(now_ms())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_test_field() -> (ChittaField, TempDir) {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let field = ChittaField::open(data_dir).unwrap();
        (field, tmp)
    }

    // Direct tag selection is global. Callers must scope the selected payloads,
    // using the same exact-realm rule as semantic and keyword recall.
    #[test]
    fn test_tag_selection_realm_contract() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let mut ids = Vec::new();
        for realm in ["project:a", "project:b", "brahman"] {
            let (id, _) = field.put_memory("wisdom", realm,
                b"tagrealmcanary shared fact", &emb, 1.0, 0.001, 0,
                vec![], None, None).unwrap();
            field.add_triplet(id.to_string(), "tagged".into(), "shared-tag".into(),
                1.0, None, None).unwrap();
            ids.push(id);
        }
        let tagged = field.query_object("shared-tag").unwrap();
        assert_eq!(tagged.len(), 3);
        for (realm, expected) in ["project:a", "project:b", "brahman"].iter().zip(ids) {
            let selected: Vec<_> = tagged.iter().filter_map(|t| {
                let id = t.subject.parse::<u64>().ok()?;
                let payload = field.get_memory(id).ok()?;
                (payload.realm == *realm).then_some(id)
            }).collect();
            assert_eq!(selected, vec![expected]);
            let semantic = field.recall_semantic_measure(&emb, 10, Some(realm)).unwrap();
            let keyword = field.recall_keyword_measure("tagrealmcanary", 10, Some(realm)).unwrap();
            assert!(!semantic.is_empty() && !keyword.is_empty());
            assert!(semantic.iter().chain(keyword.iter()).all(|h| h.realm == *realm));
        }
    }

    #[test]
    fn test_put_get_roundtrip() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id, hash) = field
            .put_memory(
                "wisdom",
                "test",
                b"hello world",
                &embedding,
                0.9,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();

        let payload = field.get_memory(id).unwrap();
        assert_eq!(payload.content, b"hello world");
        assert_eq!(payload.kind, "wisdom");
        assert_eq!(payload.chunk_hash, hash);
    }

    // Windowed hybrid recall: the ts window GATES (authored_at_ms membership),
    // semantic similarity RANKS. In-window low-relevance noise must not outrank
    // in-window relevant hits, and out-of-window hits must never appear.
    #[test]
    fn test_windowed_recall_gates_by_time_ranks_by_semantic() {
        let (field, _tmp) = open_test_field();
        let day = 86_400_000i64;
        let now = now_ms();
        // Same embedding direction = relevant; orthogonal = noise.
        let mut rel = vec![0.0f32; crate::ops::EMBED_DIM];
        rel[0] = 1.0;
        let mut noise = vec![0.0f32; crate::ops::EMBED_DIM];
        noise[1] = 1.0;
        let (old_rel, _) = field
            .put_memory("wisdom", "wtest", b"relevant fact from three weeks ago window test", &rel,
                0.9, 0.001, now - 21 * day, vec![], None, None)
            .unwrap();
        let (in_rel, _) = field
            .put_memory("wisdom", "wtest", b"relevant fact from two days ago window test", &rel,
                0.9, 0.001, now - 2 * day, vec![], None, None)
            .unwrap();
        let (in_noise, _) = field
            .put_memory("wisdom", "wtest", b"unrelated compliance chatter fresh entry", &noise,
                0.9, 0.001, now - 1 * day, vec![], None, None)
            .unwrap();
        let window = Some((now - 7 * day, now));
        let hits = field
            .recall_with_fallback_windowed(&rel, "relevant fact window test", 3, Some("wtest"), window)
            .unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.memory_id).collect();
        assert!(!ids.contains(&old_rel), "out-of-window hit leaked through the gate: {ids:?}");
        assert!(ids.contains(&in_rel), "in-window relevant hit missing: {ids:?}");
        assert_eq!(ids.first(), Some(&in_rel),
            "semantic must rank above fresher noise (recency-sort pollution): {ids:?}");
        // Freshest-but-irrelevant may appear via backfill, but never above the relevant hit.
        if let Some(pos_noise) = ids.iter().position(|i| *i == in_noise) {
            assert!(pos_noise > 0);
        }
        // No window → out-of-window memory is reachable again (no frozen state).
        let all = field
            .recall_with_fallback(&rel, "relevant fact window test", 5, Some("wtest"))
            .unwrap();
        assert!(all.iter().any(|h| h.memory_id == old_rel));
    }

    // Span-lane live path: a NEW memory's atoms must be queryable and edge-linked
    // immediately after put_memory — no manual backfill (the exact gap the owner
    // hit: a fresh memory with a unique path stayed invisible to the lane).
    #[test]
    fn test_put_memory_auto_links_spans() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let content = b"results live at /projects/unique/spanlane/auto_link_probe.tsv.gz now";
        let (id, _) = field
            .put_memory("wisdom", "spantest", content, &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();

        let hits = field.span_query("auto_link_probe", Some("spantest"), 6);
        assert!(!hits.is_empty(), "new memory's atom not auto-ingested");
        assert!(
            hits.iter().any(|h| h.0.contains("auto_link_probe.tsv.gz") && h.8.contains(&id)),
            "atom missing the memory_id reverse edge: {hits:?}"
        );
        // Forward edge: the memory expands to its verbatim atom.
        let fwd = field.span_for_memory(id, 4);
        assert!(fwd.iter().any(|a| a.0.contains("auto_link_probe.tsv.gz")));

        // forget() unlinks; the memory-only span hits refcount zero → gone.
        field.forget(id).unwrap();
        let hits = field.span_query("auto_link_probe", Some("spantest"), 6);
        assert!(hits.is_empty(), "forgotten memory's atoms must be GC'd: {hits:?}");
    }

    // ── Utility posteriors ──────────────────────────────────────────────────

    #[test]
    fn record_outcome_accrues_and_caps_weight() {
        let mut st = MemoryState::new(1, [0u8; 32], 0);
        assert_eq!((st.utility_alpha, st.utility_beta), (1.0, 1.0));
        assert_eq!(st.utility_mean(), 0.5);

        st.record_outcome(true, 1.0);
        st.record_outcome(true, 2.5);
        st.record_outcome(false, 0.5);
        assert_eq!((st.utility_alpha, st.utility_beta), (4.5, 1.5));

        // Above the cap saturates at 5; non-positive and NaN are not observations.
        st.record_outcome(true, 100.0);
        assert_eq!(st.utility_alpha, 9.5);
        st.record_outcome(false, 0.0);
        st.record_outcome(false, -3.0);
        st.record_outcome(false, f32::NAN);
        assert_eq!(st.utility_beta, 1.5);
    }

    #[test]
    fn thompson_theta_is_neutral_below_observation_floor() {
        let mut rng = UtilityRng::new(0xC0FFEE);
        // Untested prior and everything short of 3 real observations: exactly
        // 0.5, no draw taken — an untested memory is neither boosted nor punished.
        for (a, b) in [(1.0, 1.0), (3.0, 1.0), (1.0, 3.0), (2.0, 2.5)] {
            assert_eq!(thompson_theta(a, b, &mut rng), 0.5, "alpha={a} beta={b}");
        }
        // At the floor a real draw happens, and it stays a probability.
        for _ in 0..200 {
            let t = thompson_theta(9.0, 1.0, &mut rng);
            assert!(t > 0.0 && t < 1.0, "theta out of range: {t}");
        }
    }

    #[test]
    fn thompson_theta_is_deterministic_and_tracks_the_posterior() {
        // Same seed → same draw, so an eval run is reproducible.
        let a = thompson_theta(20.0, 5.0, &mut UtilityRng::new(7));
        let b = thompson_theta(20.0, 5.0, &mut UtilityRng::new(7));
        assert_eq!(a, b);

        // A useful memory samples above a useless one on average.
        let mut rng = UtilityRng::new(1234);
        let good: f32 = (0..500).map(|_| thompson_theta(40.0, 2.0, &mut rng)).sum::<f32>() / 500.0;
        let bad: f32 = (0..500).map(|_| thompson_theta(2.0, 40.0, &mut rng)).sum::<f32>() / 500.0;
        assert!(good > 0.85, "good posterior mean too low: {good}");
        assert!(bad < 0.15, "bad posterior mean too high: {bad}");
    }

    #[test]
    fn utility_multiplier_is_identity_while_the_flag_is_off() {
        // The default process env has CHITTA_UTILITY_RECALL unset, so every
        // candidate multiplies by exactly 1.0 and recall ranking is unchanged.
        assert!(!utility_recall_config().enabled);
        let mut rng = UtilityRng::new(99);
        let mut st = MemoryState::new(1, [0u8; 32], 0);
        st.record_outcome(false, 5.0);
        st.record_outcome(false, 5.0);
        assert_eq!(utility_multiplier(&st, &mut rng), 1.0);
        for score in [0.0f32, 0.371_23, 12.5, f32::MAX] {
            assert_eq!(score * utility_multiplier(&st, &mut rng), score);
        }
    }

    #[test]
    fn utility_multiplier_with_weight_separates_proven_from_untested() {
        let mut rng = UtilityRng::new(4242);
        let w = 0.3f32;

        // Untested and under-observed memories both land on the same neutral
        // 1 - w/2, so turning the flag on cannot reorder them among themselves.
        let untested = MemoryState::new(1, [0u8; 32], 0);
        assert_eq!(utility_multiplier_with(w, &untested, &mut rng), 0.85);
        let mut thin = MemoryState::new(2, [0u8; 32], 0);
        thin.record_outcome(true, 2.0);
        assert_eq!(utility_multiplier_with(w, &thin, &mut rng), 0.85);

        let mut good = MemoryState::new(3, [0u8; 32], 0);
        let mut bad = MemoryState::new(4, [0u8; 32], 0);
        for _ in 0..10 {
            good.record_outcome(true, 4.0);
            bad.record_outcome(false, 4.0);
        }
        let mean = |st: &MemoryState, rng: &mut UtilityRng| {
            (0..300).map(|_| utility_multiplier_with(w, st, rng)).sum::<f32>() / 300.0
        };
        let good_mul = mean(&good, &mut rng);
        let bad_mul = mean(&bad, &mut rng);
        assert!(good_mul > 0.98, "proven memory barely boosted: {good_mul}");
        assert!(bad_mul < 0.72, "disproven memory barely damped: {bad_mul}");
        // The whole effect stays inside [1-w, 1], a 1.43x spread at w=0.3.
        assert!(bad_mul >= 1.0 - w && good_mul <= 1.0);
    }

    /// Characterization: pins the write-time semantic dedup guard in put_memory.
    ///
    /// The guard is `if !embed_pending`, and `embed_pending` is
    /// `embedding.is_empty() && content.len() >= MIN_EMBED_CHARS`. So the branch
    /// is skipped on the production FFI path (C++ passes no vector, content is
    /// long) but RUNS whenever a caller supplies an embedding — which every test
    /// here does, and which `cf_put_memory` permits. It is therefore live code,
    /// not dead code, and removing it would change write-time behavior for
    /// embedding-supplying callers. This test fails if that is ever done.
    #[test]
    fn put_memory_write_time_dedup_collapses_supplied_near_duplicates() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        // Two vectors whose cosine lands inside [dedup_cosine_threshold,
        // dedup_cosine_upper) — near-duplicate, but not an exact match.
        let (thresh, upper) = {
            let cfg = &field.scoring_pipeline.read().config;
            (cfg.dedup_cosine_threshold, cfg.dedup_cosine_upper)
        };
        let mut a = vec![0.0f32; crate::ops::EMBED_DIM];
        let mut b = vec![0.0f32; crate::ops::EMBED_DIM];
        let target = (thresh + upper) / 2.0;
        a[0] = 1.0;
        b[0] = target;
        b[1] = (1.0 - target * target).sqrt();

        let (id_a, _) = field
            .put_memory("wisdom", "dedup", b"the first near duplicate memory", &a,
                        0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        let (id_b, _) = field
            .put_memory("wisdom", "dedup", b"the second near duplicate memory", &b,
                        0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        assert_eq!(
            id_b, id_a,
            "write-time dedup must collapse a supplied near-duplicate onto the original; \
             if this fails the `if !embed_pending` branch in put_memory was removed"
        );

        // Cross-realm near-duplicates must stay independent (no silent
        // cross-realm reinforcement).
        let (id_c, _) = field
            .put_memory("wisdom", "other-realm", b"the third near duplicate memory", &b,
                        0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        assert_ne!(id_c, id_a, "dedup must not cross realms");
    }

    /// Companion: on the production path (no supplied vector, content long
    /// enough to embed) the write-time guard is skipped, so near-duplicates get
    /// distinct ids at write time and are only collapsed later by the
    /// backfill-time supersede pass in `backfill_embedding`.
    #[test]
    fn put_memory_without_embedding_defers_dedup_to_backfill() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();
        // Distinct bodies: identical content is collapsed earlier by the
        // chunk-hash exact-duplicate check, which is a different mechanism.
        let (id_a, _) = field
            .put_memory("wisdom", "dedup", b"a sufficiently long memory body, variant one",
                        &[], 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        let (id_b, _) = field
            .put_memory("wisdom", "dedup", b"a sufficiently long memory body, variant two",
                        &[], 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        assert_ne!(id_b, id_a, "no vector at write time means no write-time dedup");
        for id in [id_a, id_b] {
            assert!(
                field.states.read().get(&id).map(|s| s.embed_pending).unwrap_or(false),
                "production-path writes must be embed_pending"
            );
        }
    }

    #[test]
    fn record_outcome_rejects_unknown_ids_and_survives_snapshot_reopen() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.4f32; crate::ops::EMBED_DIM];

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let (id, _) = field
                .put_memory("wisdom", "test", b"outcome persist", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            assert!(field.record_outcome(id + 9_999, true, 1.0).is_err());
            assert_eq!(field.record_outcome(id, true, 2.0).unwrap(), (3.0, 1.0));
            assert_eq!(field.record_outcome(id, false, 1.0).unwrap(), (3.0, 2.0));
            field.save_full_snapshot().unwrap();
            id
        };

        let field = ChittaField::open(data_dir).unwrap();
        let states = field.states.read();
        let s = states.get(&id).unwrap();
        assert_eq!(
            (s.utility_alpha, s.utility_beta),
            (3.0, 2.0),
            "utility posterior must survive a snapshot save/reopen cycle"
        );
    }

    #[test]
    fn record_outcome_survives_wal_only_replay() {
        // Regression: outcomes were RAM-only until the next periodic snapshot
        // wrote the V23 `utility_posteriors` section, so a restart in between
        // reverted the posterior (observed α 4 → 3). With no snapshot at all,
        // the reopen below is a pure WAL replay — it restores only if
        // RecordOutcome is a real op.
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.4f32; crate::ops::EMBED_DIM];

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let (id, _) = field
                .put_memory("wisdom", "test", b"outcome wal", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            assert_eq!(field.record_outcome(id, true, 3.0).unwrap(), (4.0, 1.0));
            assert_eq!(field.record_outcome(id, false, 2.0).unwrap(), (4.0, 3.0));
            // Non-observations must not reach the WAL and must not move (α, β).
            assert_eq!(field.record_outcome(id, true, 0.0).unwrap(), (4.0, 3.0));
            assert_eq!(field.record_outcome(id, true, -1.0).unwrap(), (4.0, 3.0));
            id
        };

        let field = ChittaField::open(data_dir).unwrap();
        let states = field.states.read();
        let s = states.get(&id).unwrap();
        assert_eq!(
            (s.utility_alpha, s.utility_beta),
            (4.0, 3.0),
            "utility posterior must survive a WAL-only replay"
        );
    }

    #[test]
    fn record_outcome_replay_does_not_double_count_snapshot_section() {
        // Precedence: the V23 snapshot section carries (α, β) as of the
        // snapshot, and replay applies only the WAL suffix the snapshot does
        // not cover. Outcomes before the save must therefore be counted once
        // (via the section) and outcomes after it once (via the WAL).
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.4f32; crate::ops::EMBED_DIM];

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let (id, _) = field
                .put_memory("wisdom", "test", b"outcome mixed", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            field.record_outcome(id, true, 2.0).unwrap();
            field.record_outcome(id, false, 1.0).unwrap();
            field.save_full_snapshot().unwrap();
            // Post-snapshot suffix: lives only in the WAL until the next save.
            assert_eq!(field.record_outcome(id, true, 1.5).unwrap(), (4.5, 2.0));
            assert_eq!(field.record_outcome(id, false, 0.5).unwrap(), (4.5, 2.5));
            id
        };

        let field = ChittaField::open(data_dir).unwrap();
        let states = field.states.read();
        let s = states.get(&id).unwrap();
        assert_eq!(
            (s.utility_alpha, s.utility_beta),
            (4.5, 2.5),
            "snapshot-covered outcomes must not be replayed a second time"
        );
    }

    // ── Replay confluence (THEORY.md §2) ────────────────────────────────────
    // Multi-node daemons write instance-partitioned WALs; replay applies all
    // segments in instance-id sort order, not causal order. These tests pin
    // down the convergence envelope and the two known loss modes so neither
    // can drift silently.

    fn theory_put_op(memory_id: u64, ts: i64, content: &str) -> crate::ops::Op {
        let mut emb = vec![0.1f32; crate::ops::EMBED_DIM];
        emb[0] = (memory_id as f32) / 100.0;
        crate::ops::Op::PutPayload(crate::ops::PutPayloadOp {
            memory_id,
            version: 0,
            chunk_hash: [memory_id as u8; 32],
            created_at_ms: ts,
            authored_at_ms: ts,
            kind: "wisdom".to_string(),
            realm: "test".to_string(),
            content: content.as_bytes().to_vec(),
            embedding_model: "test".to_string(),
            embedding: emb,
            artifact_refs: vec![],
            source_session: None,
            source_tool: None,
            harness: None,
            embedding_model_id: String::new(),
            embedding_dim: crate::ops::EMBED_DIM as u32,
        })
    }

    fn theory_delta_op(memory_id: u64, strength_delta: f32, ts: i64) -> crate::ops::Op {
        crate::ops::Op::UpdateState(crate::ops::StateDeltaOp {
            memory_id,
            strength_delta: Some(strength_delta),
            confidence_delta: None,
            decay_rate: None,
            touch: true,
            pin: None,
            op_ts_ms: ts,
            status: None,
            epistemic_status: None,
            staged: None,
            invalidated_by: None,
        })
    }

    fn write_instance_segment(data_dir: &std::path::Path, instance: u32, ops: &[crate::ops::Op]) {
        let mut log = crate::log::OpLog::open(data_dir, instance, 1).unwrap();
        for op in ops {
            log.append(op).unwrap();
        }
        log.flush_buf().unwrap();
    }

    fn state_fingerprint(field: &ChittaField, id: u64) -> (f32, u32) {
        let states = field.states.read();
        let st = states.get(&id).expect("memory state must exist after replay");
        (st.strength, st.access_count)
    }

    /// The safe envelope: per-memory single-writer op sets converge no matter
    /// which instance id (= segment sort position) each writer was assigned.
    #[test]
    fn replay_confluent_for_disjoint_memories() {
        let set_a = vec![theory_put_op(11, 1_000, "alpha"), theory_delta_op(11, -0.2, 2_000)];
        let set_b = vec![theory_put_op(22, 1_500, "beta"), theory_delta_op(22, -0.4, 2_500)];

        let mut results = Vec::new();
        for (inst_a, inst_b) in [(0x1000_0001u32, 0x2000_0002u32), (0x2000_0002, 0x1000_0001)] {
            let tmp = TempDir::new().unwrap();
            let data_dir = tmp.path().join("data");
            std::fs::create_dir_all(&data_dir).unwrap();
            write_instance_segment(&data_dir, inst_a, &set_a);
            write_instance_segment(&data_dir, inst_b, &set_b);
            let field = ChittaField::open(data_dir).unwrap();
            results.push((state_fingerprint(&field, 11), state_fingerprint(&field, 22)));
        }
        assert_eq!(
            results[0], results[1],
            "disjoint-memory replay must be insensitive to instance assignment"
        );
    }

    /// THEORY.md §3: merge replay orders ops by (op_ts, instance, seqno), so
    /// cross-instance deltas apply in timestamp order even when instance-id
    /// sort order inverts it. Before merge replay, the ts=2000 delta below was
    /// wholly discarded by apply_delta's monotonicity guard (loss mode §2.2).
    #[test]
    fn merge_replay_applies_cross_instance_deltas_in_timestamp_order() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        write_instance_segment(&data_dir, 0x1000_0001, &[theory_put_op(33, 1_000, "gamma")]);
        write_instance_segment(&data_dir, 0x2000_0002, &[theory_delta_op(33, -0.2, 3_000)]);
        write_instance_segment(&data_dir, 0x3000_0003, &[theory_delta_op(33, -0.4, 2_000)]);

        let field = ChittaField::open(data_dir).unwrap();
        let (strength, access_count) = state_fingerprint(&field, 33);
        assert!(
            (strength - 0.4).abs() < 1e-6,
            "both deltas must apply in ts order (got strength {strength})"
        );
        assert_eq!(access_count, 2);
    }

    /// THEORY.md §2.3/§3: an UpdateState merge-ordered before its memory's
    /// PutPayload (possible under cross-writer clock skew) lands in the
    /// orphan-delta buffer and is applied after the creates. Before merge
    /// replay + the buffer, it was silently dropped.
    #[test]
    fn orphan_delta_before_create_is_buffered_and_applied() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        // Skewed clock: the delta's ts predates the create's ts, so merge
        // order applies it first — the orphan buffer must catch it.
        write_instance_segment(&data_dir, 0x1000_0001, &[theory_delta_op(44, -0.4, 500)]);
        write_instance_segment(&data_dir, 0x2000_0002, &[theory_put_op(44, 1_000, "delta")]);

        let field = ChittaField::open(data_dir).unwrap();
        let (strength, access_count) = state_fingerprint(&field, 44);
        assert!(
            (strength - 0.6).abs() < 1e-6,
            "orphaned delta must be applied after creates (got strength {strength})"
        );
        assert_eq!(access_count, 1);
    }

    /// THEORY.md §3: with merge replay, state is a function of the op SET —
    /// any partition of the ops across writers, under any instance-id
    /// assignment, converges. Creates carry the earliest timestamps so the
    /// orphan path stays out of this test (covered separately).
    #[test]
    fn replay_confluent_under_random_instance_permutations() {
        let mut ops: Vec<crate::ops::Op> = Vec::new();
        for m in 0..4u64 {
            ops.push(theory_put_op(100 + m, 1_000 + m as i64, "perm"));
        }
        for i in 0..8u64 {
            let mem = 100 + (i % 4);
            let d = -0.05 * ((i % 3) as f32 + 1.0);
            ops.push(theory_delta_op(mem, d, 2_000 + 100 * i as i64));
        }
        let instances = [0x1000_0001u32, 0x2000_0002, 0x3000_0003];

        let mut seed = 0x9E37_79B9_u64;
        let mut xorshift = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let mut fingerprints = Vec::new();
        for _ in 0..4 {
            let tmp = TempDir::new().unwrap();
            let data_dir = tmp.path().join("data");
            std::fs::create_dir_all(&data_dir).unwrap();
            let mut groups: Vec<Vec<crate::ops::Op>> = vec![Vec::new(), Vec::new(), Vec::new()];
            for op in &ops {
                groups[(xorshift() % 3) as usize].push(op.clone());
            }
            for (group, inst) in groups.into_iter().zip(instances) {
                if !group.is_empty() {
                    write_instance_segment(&data_dir, inst, &group);
                }
            }
            let field = ChittaField::open(data_dir).unwrap();
            let fp: Vec<(f32, u32)> =
                (100..104).map(|m| state_fingerprint(&field, m)).collect();
            fingerprints.push(fp);
        }
        for fp in &fingerprints[1..] {
            assert_eq!(
                &fingerprints[0], fp,
                "state must be a function of the op set, not the instance assignment"
            );
        }
    }

    /// THEORY.md §4: seqno ranges overlap across writers, so the scalar
    /// `seqno <= snapshot_seqno` skip silently dropped foreign ops the
    /// snapshot never contained. The per-writer coverage vector applies them.
    #[test]
    fn reopen_applies_uncovered_foreign_ops_with_overlapping_seqnos() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.3f32; crate::ops::EMBED_DIM];

        {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            field
                .put_memory("wisdom", "test", b"local memory", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            field.save_full_snapshot().unwrap();
        }

        // A foreign writer's ops with LOW seqnos (overlapping the snapshot's
        // scalar seqno range) that the snapshot does NOT contain.
        write_instance_segment(
            &data_dir,
            0xF000_000F,
            &[theory_put_op(777, 5_000, "foreign uncovered")],
        );

        let field = ChittaField::open(data_dir).unwrap();
        assert!(
            field.states.read().contains_key(&777),
            "uncovered foreign op must be applied on reopen (was skipped by the scalar filter)"
        );
    }

    /// THEORY.md §4: prune only segments the coverage vector dominates; an
    /// instance's open-ended last segment is never pruned. Header-only (empty)
    /// segments of a dead instance are prunable by size regardless of coverage.
    #[test]
    fn prune_covered_segments_respects_coverage_vector() {
        let tmp = TempDir::new().unwrap();
        let seg_dir = tmp.path().join("segments");
        std::fs::create_dir_all(&seg_dir).unwrap();
        // Op-bearing segments must exceed the header size, else the empty-segment
        // rule would reclaim them; write header + a byte of "op" payload.
        let ops = vec![0u8; crate::log::V3_HEADER_SIZE + 1];
        for name in [
            "10000001_000000000001.seg",
            "10000001_000000000050.seg",
            "20000002_000000000001.seg",
        ] {
            std::fs::write(seg_dir.join(name), &ops).unwrap();
        }

        // Instance 1 is the LIVE writer here: its open tail (…_50) is never pruned.
        let live = 0x1000_0001u32;

        // Not covered far enough: nothing prunable.
        let mut covered = std::collections::BTreeMap::new();
        covered.insert(0x1000_0001u32, 10u64);
        assert_eq!(prune_covered_segments(&seg_dir, &covered, live), 0);

        // Covered through the first segment's end (next_first - 1 = 49):
        // only instance 1's first segment goes; the live tail + the dead
        // instance-2 segment (absent from `covered`) stay.
        covered.insert(0x1000_0001, 49);
        assert_eq!(prune_covered_segments(&seg_dir, &covered, live), 1);
        assert!(!seg_dir.join("10000001_000000000001.seg").exists());
        assert!(seg_dir.join("10000001_000000000050.seg").exists());
        assert!(
            seg_dir.join("20000002_000000000001.seg").exists(),
            "a foreign writer's segment must never be pruned without coverage"
        );

        // A DEAD instance's final segment IS prunable once it appears in the
        // coverage vector (fully folded into the snapshot). This is the common
        // one-segment-per-lifetime case the interior windows(2) rule can't reach.
        covered.insert(0x2000_0002, 1);
        assert_eq!(prune_covered_segments(&seg_dir, &covered, live), 1);
        assert!(!seg_dir.join("20000002_000000000001.seg").exists());
        // The live instance's tail still survives — never pruned even when covered.
        assert!(seg_dir.join("10000001_000000000050.seg").exists());

        // Header-only (empty) segments: a dead instance's empty segment is
        // reclaimed WITHOUT any coverage entry (coverage is op-derived and can
        // never prove it); the live instance's empty segment is preserved.
        let empty = vec![0u8; crate::log::V3_HEADER_SIZE];
        std::fs::write(seg_dir.join("30000003_000000000001.seg"), &empty).unwrap(); // dead, empty
        std::fs::write(seg_dir.join("10000001_000000000999.seg"), &empty).unwrap(); // live, empty
        let empty_cov = std::collections::BTreeMap::new(); // no coverage at all
        assert_eq!(prune_covered_segments(&seg_dir, &empty_cov, live), 1);
        assert!(!seg_dir.join("30000003_000000000001.seg").exists());
        assert!(
            seg_dir.join("10000001_000000000999.seg").exists(),
            "the live instance's freshly-opened (header-only) segment must survive"
        );
    }

    fn theory_content_op(memory_id: u64, content: &str, ts: i64) -> crate::ops::Op {
        crate::ops::Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
            memory_id,
            content: content.as_bytes().to_vec(),
            embedding: Vec::new(),
            op_ts_ms: ts,
        })
    }

    /// THEORY.md §2.1 class (c): absolute writes are LWW registers under
    /// merge replay — newest op_ts_ms wins regardless of instance assignment.
    #[test]
    fn content_updates_are_lww_by_timestamp() {
        for (inst_b, inst_c) in [(0x2000_0002u32, 0x3000_0003u32), (0x3000_0003, 0x2000_0002)] {
            let tmp = TempDir::new().unwrap();
            let data_dir = tmp.path().join("data");
            std::fs::create_dir_all(&data_dir).unwrap();
            write_instance_segment(&data_dir, 0x1000_0001, &[theory_put_op(55, 1_000, "orig")]);
            write_instance_segment(&data_dir, inst_b, &[theory_content_op(55, "newest", 3_000)]);
            write_instance_segment(&data_dir, inst_c, &[theory_content_op(55, "middle", 2_000)]);

            let field = ChittaField::open(data_dir).unwrap();
            let payloads = field.payloads.read();
            assert_eq!(
                payloads.get(&55).unwrap().content,
                b"newest".to_vec(),
                "newest op_ts_ms must win under any instance assignment"
            );
        }
    }

    /// The semantic index is the embedding's single in-RAM home: the payload
    /// copy is cleared at write, stripped from the snapshot body, NOT
    /// rehydrated at open — and embedding_of() serves every reader.
    #[test]
    fn payload_embeddings_stripped_from_body_and_rehydrated() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.7f32; crate::ops::EMBED_DIM];

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let (id, _) = field
                .put_memory("wisdom", "test", b"strip me", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            field.save_full_snapshot().unwrap();
            id
        };

        // Raw body: embedding stripped.
        let snap_path = std::fs::read_dir(&data_dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                let n = p.file_name().unwrap().to_string_lossy().into_owned();
                n.starts_with("chitta.") && n.ends_with(".snapshot")
            })
            .expect("snapshot file");
        let raw = crate::snapshot::FullSnapshot::load(&snap_path).unwrap();
        assert!(
            raw.payloads.get(&id).unwrap().embedding.is_empty(),
            "body must not carry an embedding the .emb sidecar owns"
        );

        // Full open: the payload copy STAYS empty (no ~600MB duplicate);
        // embedding_of serves the vector from the index.
        let field = ChittaField::open(data_dir).unwrap();
        {
            let payloads = field.payloads.read();
            assert!(
                payloads.get(&id).unwrap().embedding.is_empty(),
                "payload embedding must NOT be rehydrated into the heap"
            );
        }
        assert_eq!(
            field.embedding_of(id).map(|e| e.len()),
            Some(crate::ops::EMBED_DIM),
            "embedding_of must serve the vector from the index"
        );
        assert_eq!(
            field.states.read().get(&id).map(|s| s.embed_pending),
            Some(false),
            "index-held embeddings must not be requeued for re-embed"
        );
    }

    /// Phase 2 (THEORY.md §8): index sidecars are not rewritten when the
    /// index hasn't mutated since the last save (dirty-skip).
    #[test]
    fn index_sidecars_skipped_when_clean() {
        use std::os::unix::fs::MetadataExt;
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.5f32; crate::ops::EMBED_DIM];

        let field = ChittaField::open(data_dir.clone()).unwrap();
        field
            .put_memory("wisdom", "test", b"dirty one", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.save_full_snapshot().unwrap();
        let sidecar = |ext: &str| {
            std::fs::read_dir(&data_dir)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .find(|p| p.extension().map(|e| e == ext).unwrap_or(false))
                .unwrap_or_else(|| panic!(".{ext} sidecar"))
        };
        let emb_path = sidecar("emb");
        let hdc_path = sidecar("hdc");
        let pld_path = sidecar("pld");
        let ino_first = std::fs::metadata(&emb_path).unwrap().ino();
        let hdc_ino_first = std::fs::metadata(&hdc_path).unwrap().ino();
        let pld_ino_first = std::fs::metadata(&pld_path).unwrap().ino();
        let emb_len_first = std::fs::metadata(&emb_path).unwrap().len();
        let hdc_len_first = std::fs::metadata(&hdc_path).unwrap().len();
        let pld_len_first = std::fs::metadata(&pld_path).unwrap().len();

        // No mutation between saves → same inodes (skipped rewrites).
        field.save_full_snapshot().unwrap();
        assert_eq!(
            std::fs::metadata(&emb_path).unwrap().ino(),
            ino_first,
            "clean index must not rewrite sidecars"
        );
        assert_eq!(
            std::fs::metadata(&hdc_path).unwrap().ino(),
            hdc_ino_first,
            "clean hdc store must not rewrite its sidecar"
        );
        assert_eq!(
            std::fs::metadata(&pld_path).unwrap().ino(),
            pld_ino_first,
            "unchanged content must not rewrite the .pld sidecar"
        );

        // A new memory mutates the index → rewrite (fresh inode via rename).
        // Orthogonal-ish embedding so the write-path dedup doesn't merge it.
        let mut emb2 = emb.clone();
        for v in emb2.iter_mut().take(crate::ops::EMBED_DIM / 2) {
            *v = -0.5;
        }
        field
            .put_memory("wisdom", "test", b"dirty two", &emb2, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.save_full_snapshot().unwrap();
        // Size, not inode: tmpfs reuses freed inode numbers, so a rename can
        // land on the same ino. Two embeddings serialize larger than one.
        assert!(
            std::fs::metadata(&emb_path).unwrap().len() > emb_len_first,
            "mutated index must rewrite sidecars"
        );
        assert!(
            std::fs::metadata(&hdc_path).unwrap().len() > hdc_len_first,
            "mutated hdc store must rewrite its sidecar"
        );
        assert!(
            std::fs::metadata(&pld_path).unwrap().len() > pld_len_first,
            "new content must rewrite the .pld sidecar"
        );
    }

    fn theory_recall_op(memory_ids: &[u64], ts: i64) -> crate::ops::Op {
        crate::ops::Op::RecordRecallBatch(crate::ops::RecordRecallBatchOp {
            memory_ids: memory_ids.to_vec(),
            centroid_q: Vec::new(),
            centroid_scale: 0.0,
            context_hash: ts as u64,
            ts_ms: ts,
            base_assoc_delta: 0.0,
        })
    }

    /// THEORY.md §6: recalls from distinct daemons accrue as cross-context
    /// provenance, and the evidence survives a snapshot save/reopen cycle
    /// via the V23 "recall_provenance" section (added with zero migration).
    #[test]
    fn recall_provenance_accrues_across_instances_and_persists() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        write_instance_segment(&data_dir, 0x1000_0001, &[theory_put_op(66, 1_000, "general")]);
        write_instance_segment(&data_dir, 0x2000_0002, &[theory_recall_op(&[66], 2_000)]);
        write_instance_segment(&data_dir, 0x3000_0003, &[theory_recall_op(&[66], 3_000)]);
        write_instance_segment(&data_dir, 0x4000_0004, &[theory_recall_op(&[66], 4_000)]);

        let distinct = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let n = field.recall_provenance.read().get(&66).map(|s| s.len());
            field.save_full_snapshot().unwrap();
            n
        };
        assert_eq!(distinct, Some(3), "three distinct recalling instances");

        let field = ChittaField::open(data_dir).unwrap();
        assert_eq!(
            field.recall_provenance.read().get(&66).map(|s| s.len()),
            Some(3),
            "provenance must survive snapshot save/reopen"
        );
    }

    /// THEORY.md §6: cross-context evidence raises recall score
    /// (multiplicative, config-gated).
    #[test]
    fn cross_context_provenance_boosts_recall_score() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.6f32; crate::ops::EMBED_DIM];
        let (id, _) = field
            .put_memory("wisdom", "test", b"generalizes", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();

        let s1 = field.recall_semantic(&emb, 3, Some("test")).unwrap()[0].score;
        {
            let mut prov = field.recall_provenance.write();
            let set = prov.entry(id).or_default();
            for inst in [0x1u32, 0x2, 0x3, 0x4] {
                set.insert(inst);
            }
        }
        let s2 = field.recall_semantic(&emb, 3, Some("test")).unwrap()[0].score;
        assert!(
            s2 > s1,
            "4 distinct recalling instances must boost score ({s1} → {s2})"
        );
    }

    /// THEORY.md §6, the falsifiable claim at the recall level: the ranked
    /// result of a query must not depend on which writer wrote what.
    #[test]
    fn recall_ranking_invariant_under_writer_permutation() {
        let mut emb_q = vec![0.1f32; crate::ops::EMBED_DIM];
        emb_q[0] = 1.0;
        let mut ops: Vec<crate::ops::Op> = Vec::new();
        for m in 0..5u64 {
            let mut e = vec![0.1f32; crate::ops::EMBED_DIM];
            e[0] = 1.0;
            e[1 + m as usize] = 0.3 + 0.1 * m as f32;
            ops.push(crate::ops::Op::PutPayload(match theory_put_op(200 + m, 1_000 + m as i64, "rank") {
                crate::ops::Op::PutPayload(mut p) => {
                    p.embedding = e;
                    p
                }
                _ => unreachable!(),
            }));
            ops.push(theory_delta_op(200 + m, -0.05 * (m as f32 + 1.0), 2_000 + m as i64));
        }

        let mut rankings = Vec::new();
        for (a, b) in [(0x1000_0001u32, 0x2000_0002u32), (0x2000_0002, 0x1000_0001)] {
            let tmp = TempDir::new().unwrap();
            let data_dir = tmp.path().join("data");
            std::fs::create_dir_all(&data_dir).unwrap();
            let (left, right): (Vec<_>, Vec<_>) =
                ops.iter().cloned().enumerate().partition(|(i, _)| i % 2 == 0);
            write_instance_segment(&data_dir, a, &left.into_iter().map(|(_, o)| o).collect::<Vec<_>>());
            write_instance_segment(&data_dir, b, &right.into_iter().map(|(_, o)| o).collect::<Vec<_>>());
            let field = ChittaField::open(data_dir).unwrap();
            let ids: Vec<u64> = field
                .recall_semantic(&emb_q, 5, Some("test"))
                .unwrap()
                .iter()
                .map(|h| h.memory_id)
                .collect();
            rankings.push(ids);
        }
        assert_eq!(
            rankings[0], rankings[1],
            "recall ranking must be writer-assignment invariant"
        );
    }

    /// THEORY.md §6/§8: the consolidation sweep refreshes stale competitive
    /// weights up to its budget and stamps them, so recall-path budgets
    /// rarely trigger.
    #[test]
    fn cw_refresh_sweep_respects_budget_and_stamps() {
        let (field, _tmp) = open_test_field();
        // Orthogonal square waves (different frequencies) — pairwise cosine 0,
        // so the write-path dedup can't merge them.
        let mut embs = Vec::new();
        for m in 0..4usize {
            let e: Vec<f32> = (0..crate::ops::EMBED_DIM)
                .map(|i| if (i / (64 << m)) % 2 == 0 { 0.5 } else { -0.5 })
                .collect();
            embs.push(e);
        }
        let mut ids = Vec::new();
        for (m, e) in embs.iter().enumerate() {
            let (id, _) = field
                .put_memory("wisdom", "test", format!("sweep {m}").as_bytes(), e, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            ids.push(id);
        }
        // Force staleness.
        {
            let mut states = field.states.write();
            for id in &ids {
                states.get_mut(id).unwrap().last_cw_refresh_ms = 0;
            }
        }
        assert_eq!(field.cw_refresh_sweep(2), 2, "budget must cap the sweep");
        let stamped = field
            .states
            .read()
            .values()
            .filter(|st| st.last_cw_refresh_ms > 0)
            .count();
        assert_eq!(stamped, 2);
        assert_eq!(field.cw_refresh_sweep(10), 2, "remaining stale memories swept");
        assert_eq!(field.cw_refresh_sweep(10), 0, "nothing stale left");
    }

    /// Regression for the consolidation re-encode loop: unencodable memories
    /// (deleted, empty-code) must not be re-collected on every pass.
    #[test]
    fn encode_all_unindexed_converges_to_zero() {
        let (field, _tmp) = open_test_field();
        let emb_a = vec![0.5f32; crate::ops::EMBED_DIM];
        let mut emb_b = emb_a.clone();
        for v in emb_b.iter_mut().take(crate::ops::EMBED_DIM / 2) {
            *v = -0.5;
        }
        let (id_a, _) = field
            .put_memory("wisdom", "test", b"encodable", &emb_a, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field
            .put_memory("wisdom", "test", b"to be deleted", &emb_b, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();

        // Soft-delete one: it must never be collected for encoding.
        let _ = field.forget(id_a);

        let first = field.encode_all_unindexed().unwrap();
        assert!(first <= 1, "deleted memory must not be collected (got {first})");
        // Whatever was attempted is now coded or skip-set: the pass converges.
        assert_eq!(
            field.encode_all_unindexed().unwrap(),
            0,
            "second pass must collect nothing — the re-encode loop"
        );
    }

    /// Janitor: dead-instance residue goes, protected classes stay, and
    /// resurrected (previously-deleted) files are counted via the ledger.
    #[test]
    fn janitor_sweep_removes_ghosts_and_tracks_resurrection() {
        let tmp = TempDir::new().unwrap();
        let d = tmp.path();
        let touch = |name: &str| std::fs::write(d.join(name), b"x").unwrap();

        // Protected: own seen_offsets, a live family + its sidecar.
        touch("seen_offsets.aaaa0001.json");
        touch("chitta.bbbb0002.snapshot");
        touch("chitta.bbbb0002.emb");
        touch("cortex.bbbb0002.snapshot");
        // Ghosts: dead reader, orphan cortex, orphan sidecar.
        touch("seen_offsets.dead0003.json");
        touch("cortex.dead0004.snapshot");
        touch("chitta.dead0005.emb");

        // max_age 0 → everything is old enough (mtime gate test inverse:
        // a huge max_age must delete nothing).
        janitor_sweep(d, 0xaaaa_0001, u64::MAX);
        assert!(d.join("seen_offsets.dead0003.json").exists(), "age gate must protect");

        janitor_sweep(d, 0xaaaa_0001, 0);
        assert!(d.join("seen_offsets.aaaa0001.json").exists(), "own file protected");
        assert!(d.join("chitta.bbbb0002.snapshot").exists(), "families are prune's job");
        assert!(d.join("chitta.bbbb0002.emb").exists(), "family sidecar protected");
        assert!(d.join("cortex.bbbb0002.snapshot").exists(), "family cortex protected");
        assert!(!d.join("seen_offsets.dead0003.json").exists(), "dead reader removed");
        assert!(!d.join("cortex.dead0004.snapshot").exists(), "orphan cortex removed");
        assert!(!d.join("chitta.dead0005.emb").exists(), "orphan sidecar removed");

        // Resurrection: re-create a deleted ghost; ledger must count it (the
        // count is logged; behaviorally it gets deleted again).
        touch("seen_offsets.dead0003.json");
        janitor_sweep(d, 0xaaaa_0001, 0);
        assert!(!d.join("seen_offsets.dead0003.json").exists(), "resurrected ghost re-deleted");
        let ledger = std::fs::read_to_string(d.join(".janitor.json")).unwrap();
        assert!(ledger.contains("seen_offsets.dead0003.json"));
    }

    #[test]
    fn test_manifest_commits_snapshot_family() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.2f32; crate::ops::EMBED_DIM];

        {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            field
                .put_memory("wisdom", "test", b"manifest commit", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            // Bypass the compact_wal 100-memory guard: save directly.
            field.save_full_snapshot().unwrap();
        }

        let manifest = crate::manifest::Manifest::load(&data_dir).unwrap().unwrap();
        assert!(manifest.generation >= 1);
        let committed = manifest
            .validated_snapshot_path(&data_dir)
            .expect("freshly committed family must validate");
        assert!(committed.exists());

        // Tamper with a recorded sidecar: validation must fail and open must
        // still succeed via fence-based fallback.
        let cp = manifest.checkpoints.as_ref().unwrap();
        let side = data_dir.join(&cp.sidecars[0].name);
        {
            let f = std::fs::OpenOptions::new().write(true).open(&side).unwrap();
            f.set_len(cp.sidecars[0].size_bytes + 7).unwrap();
        }
        assert!(manifest.validated_snapshot_path(&data_dir).is_none());
        let field = ChittaField::open(data_dir).unwrap();
        assert_eq!(field.memory_count(), 1);
    }

    #[test]
    fn test_cw_refresh_ts_survives_snapshot_reopen() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let emb = vec![0.4f32; crate::ops::EMBED_DIM];

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let (id, _) = field
                .put_memory("wisdom", "test", b"cw persist", &emb, 0.9, 0.001, 0, vec![], None, None)
                .unwrap();
            field.states.write().get_mut(&id).unwrap().last_cw_refresh_ms = 1_234_567;
            field.save_full_snapshot().unwrap();
            id
        };

        let field = ChittaField::open(data_dir).unwrap();
        assert_eq!(
            field.states.read().get(&id).unwrap().last_cw_refresh_ms,
            1_234_567,
            "last_cw_refresh_ms must survive a snapshot save/reopen cycle"
        );
    }

    #[test]
    fn test_cw_refresh_releases_inflight_reservations() {
        // Neighborhood with a real update: reservations must be released.
        let (field, _tmp) = open_test_field();
        let mut emb_a = vec![0.0f32; crate::ops::EMBED_DIM];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0f32; crate::ops::EMBED_DIM];
        emb_b[0] = 1.0;
        emb_b[1] = 1.0;
        field
            .put_memory("wisdom", "test", b"cw refresh a", &emb_a, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field
            .put_memory("wisdom", "test", b"cw refresh b", &emb_b, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.recall_semantic(&emb_a, 5, Some("test")).unwrap();
        assert!(
            field.cw_refresh_inflight.read().is_empty(),
            "reservations must be released after a refresh round"
        );

        // Isolated memory: the round produces no cw update — reservations must
        // still be released (empty rounds must not leak entries).
        let (field2, _tmp2) = open_test_field();
        field2
            .put_memory("wisdom", "test", b"isolated", &emb_a, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field2.recall_semantic(&emb_a, 5, Some("test")).unwrap();
        assert!(
            field2.cw_refresh_inflight.read().is_empty(),
            "empty update rounds must not leak reservations"
        );
    }

    #[test]
    fn test_densify_write_edges_pair() {
        // #13 controlled prospective pair test (drift-independent, binary gate):
        // a symptom memory and its root-cause memory written in the same session
        // + realm must be linked by a bidirectional SameSession edge, making each
        // 1-hop reachable from the other. Rollback removes exactly those edges.
        use std::sync::atomic::Ordering;
        let (field, _tmp) = open_test_field();
        field.densify_enabled.store(true, Ordering::Relaxed);

        // Distinct embeddings (cos 0.707 < dedup threshold) so B is not merged.
        let mut emb_a = vec![0.0f32; crate::ops::EMBED_DIM];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0f32; crate::ops::EMBED_DIM];
        emb_b[0] = 1.0;
        emb_b[1] = 1.0;
        let sess = Some("pairtest-session".to_string());

        let (a, _) = field
            .put_memory("episode", "test", b"symptom: recall degrades over days",
                &emb_a, 0.9, 0.001, 0, vec![], sess.clone(), None)
            .unwrap();
        let (b, _) = field
            .put_memory("episode", "test", b"root cause: eval strengthens wrong edges",
                &emb_b, 0.9, 0.001, 0, vec![], sess.clone(), None)
            .unwrap();
        assert_ne!(a, b, "B must be a distinct memory, not deduped into A");

        let na = field.list_neighbors(a).unwrap();
        let nb = field.list_neighbors(b).unwrap();
        assert!(
            na.iter().any(|e| e.dst == b && e.edge_type == EdgeType::SameSession),
            "root-cause must be reachable from symptom via a SameSession edge",
        );
        assert!(
            nb.iter().any(|e| e.dst == a && e.edge_type == EdgeType::SameSession),
            "densification edge must be bidirectional",
        );

        // Surgical rollback: removes exactly the two densification edges.
        let removed = field.remove_assoc_edges_by_type(EdgeType::SameSession);
        assert_eq!(removed, 2, "both edge directions must be removed");
        assert!(field.list_neighbors(a).unwrap().is_empty());
        assert!(field.list_neighbors(b).unwrap().is_empty());
    }

    #[test]
    fn test_densify_via_set_source_session() {
        // Production daemon flow: cf_put_memory carries session=None; the C++
        // handler attaches it via set_source_session right after. The chain
        // must form at that attach point, and must not double-fire when the
        // session was already known at put time.
        use std::sync::atomic::Ordering;
        let (field, _tmp) = open_test_field();
        field.densify_enabled.store(true, Ordering::Relaxed);

        let mut emb_a = vec![0.0f32; crate::ops::EMBED_DIM];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0f32; crate::ops::EMBED_DIM];
        emb_b[1] = 1.0;

        let (a, _) = field
            .put_memory("episode", "test", b"daemon-path write A", &emb_a, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        let (b, _) = field
            .put_memory("episode", "test", b"daemon-path write B", &emb_b, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        // No session at put time → no edges yet.
        assert!(field.list_neighbors(a).unwrap().is_empty());

        field.set_source_session(a, "daemon-sess").unwrap();
        field.set_source_session(b, "daemon-sess").unwrap();
        let nb = field.list_neighbors(b).unwrap();
        assert!(
            nb.iter().any(|e| e.dst == a && e.edge_type == EdgeType::SameSession),
            "attach-time densification must link B to its session sibling A",
        );

        // Re-attach must not duplicate edges (ring guard).
        field.set_source_session(b, "daemon-sess").unwrap();
        let nb2 = field.list_neighbors(b).unwrap();
        let same_count = nb2.iter().filter(|e| e.dst == a && e.edge_type == EdgeType::SameSession).count();
        assert_eq!(same_count, 1, "re-attach must not duplicate the edge");
    }

    #[test]
    fn test_densify_backfill() {
        // Retro-backfill over historical memories: session-tagged writes made
        // while the write-hook was OFF (densify_enabled=false — the pre-#13
        // world) must gain the same K=3 decaying SameSession chain when
        // densify_backfill(apply=true) runs. Dry run writes nothing; re-apply
        // is idempotent; untagged memories are ignored.
        let (field, _tmp) = open_test_field();
        let sess = Some("hist-sess".to_string());

        let mut ids = Vec::new();
        for i in 0..4u8 {
            let mut emb = vec![0.0f32; crate::ops::EMBED_DIM];
            emb[i as usize] = 1.0;
            let (id, _) = field
                .put_memory("episode", "test", format!("hist write {i}").as_bytes(),
                    &emb, 0.9, 0.001, 0, vec![], sess.clone(), None)
                .unwrap();
            ids.push(id);
        }
        // One untagged memory: must not join any group.
        let mut emb = vec![0.0f32; crate::ops::EMBED_DIM];
        emb[5] = 1.0;
        field
            .put_memory("episode", "test", b"untagged", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();

        // Write-hook off → no edges yet despite session tags.
        assert!(field.list_neighbors(ids[3]).unwrap().is_empty());

        // Dry run: 1 session group of 4 → pairs = 1+2+3 = 6; nothing written.
        let (sessions, mems, pairs, hist) = field.densify_backfill(false);
        assert_eq!((sessions, mems, pairs), (1, 4, 6));
        assert_eq!(hist[2], 1, "group of 4 lands in the 3-5 bucket");
        assert!(field.list_neighbors(ids[3]).unwrap().is_empty(), "dry run must write nothing");

        // Apply: newest links to its 3 priors with decaying weight.
        let (_, _, pairs_applied, _) = field.densify_backfill(true);
        assert_eq!(pairs_applied, 6);
        let n3 = field.list_neighbors(ids[3]).unwrap();
        let same: Vec<_> = n3.iter().filter(|e| e.edge_type == EdgeType::SameSession).collect();
        assert_eq!(same.len(), 3, "newest must link to all 3 priors");
        let w = |dst| same.iter().find(|e| e.dst == dst).unwrap().weight;
        assert!((w(ids[2]) - 0.6).abs() < 1e-6);
        assert!((w(ids[1]) - 0.42).abs() < 1e-6);
        assert!((w(ids[0]) - 0.294).abs() < 1e-6);

        // Idempotent: re-apply changes nothing.
        field.densify_backfill(true);
        assert_eq!(
            field.list_neighbors(ids[3]).unwrap().iter()
                .filter(|e| e.edge_type == EdgeType::SameSession).count(),
            3,
            "re-apply must not duplicate edges"
        );
    }

    #[test]
    fn test_forget() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.0f32; crate::ops::EMBED_DIM];
        let (id, _) = field
            .put_memory(
                "wisdom",
                "test",
                b"to forget",
                &embedding,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field.forget(id).unwrap();
        assert!(matches!(
            field.get_memory(id),
            Err(crate::error::FieldError::Deleted(_))
        ));
    }

    #[test]
    fn test_replay_on_reopen() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let embedding = vec![0.5f32; crate::ops::EMBED_DIM];
            let (id, _) = field
                .put_memory(
                    "episode",
                    "test",
                    b"persisted",
                    &embedding,
                    0.8,
                    0.002,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
            id
        };

        // Reopen and verify data survived.
        let field2 = ChittaField::open(data_dir).unwrap();
        let payload = field2.get_memory(id).unwrap();
        assert_eq!(payload.content, b"persisted");
    }

    #[test]
    fn test_assoc_edge() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id1, _) = field
            .put_memory(
                "wisdom",
                "test",
                b"a",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        let (id2, _) = field
            .put_memory(
                "wisdom",
                "test",
                b"b",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field
            .add_assoc_edge(id1, id2, EdgeType::CoRetrieved, 0.7)
            .unwrap();
        let neighbors = field.list_neighbors(id1).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].dst, id2);
    }

    #[test]
    fn test_integration_add_triplet() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        let id = field
            .add_triplet(
                "chitta".into(),
                "replaces".into(),
                "duckdb".into(),
                1.0,
                None,
                None,
            )
            .unwrap();
        assert!(id > 0);

        let results = field.query_subject("chitta").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].object, "duckdb");
    }

    #[test]
    fn test_replay_triplets() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            field
                .add_triplet("a".into(), "b".into(), "c".into(), 1.0, None, None)
                .unwrap();
        }

        let field2 = ChittaField::open(data_dir).unwrap();
        let results = field2.query_subject("a").unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_integration_invalidate_triplet() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        let id = field
            .add_triplet(
                "chitta".into(),
                "uses".into(),
                "duckdb".into(),
                1.0,
                None,
                None,
            )
            .unwrap();

        let before = field.query_subject("chitta").unwrap();
        assert_eq!(before.len(), 1);

        field.invalidate_triplet(id).unwrap();

        let after = field.query_subject("chitta").unwrap();
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_supersede_survives_wal_replay() {
        // Regression (SOTA review 2b): supersession was RAM-only + serde(skip),
        // persisted solely via the .sup.json snapshot sidecar. With no snapshot
        // taken, a reopen replays purely from the WAL — so the revision survives
        // ONLY if SupersedeTriplet is a real op. Before the fix, old_id reappeared.
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        let old_id;
        {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            old_id = field
                .add_triplet("chitta".into(), "uses".into(), "duckdb".into(), 1.0, None, None)
                .unwrap();
            let new_id = field
                .add_triplet("chitta".into(), "uses".into(), "sqlite".into(), 1.0, None, None)
                .unwrap();
            field.triplet_supersede(old_id, new_id, now_ms()).unwrap();

            let live = field.query_subject_as_of("chitta", i64::MAX).unwrap();
            assert!(!live.iter().any(|e| e.id == old_id), "superseded pre-reopen");
        }

        // Reopen: no snapshot was written, so this is a pure WAL replay.
        let field2 = ChittaField::open(data_dir.clone()).unwrap();
        let after = field2.query_subject_as_of("chitta", i64::MAX).unwrap();
        assert!(
            !after.iter().any(|e| e.id == old_id),
            "supersession must survive WAL-only replay"
        );
    }

    #[test]
    fn test_integration_query_entity() {
        let tmp = TempDir::new().unwrap();
        let field = ChittaField::open(tmp.path().join("data")).unwrap();

        field
            .add_triplet(
                "alice".into(),
                "knows".into(),
                "bob".into(),
                1.0,
                None,
                None,
            )
            .unwrap();
        field
            .add_triplet(
                "charlie".into(),
                "knows".into(),
                "alice".into(),
                1.0,
                None,
                None,
            )
            .unwrap();
        field
            .add_triplet(
                "alice".into(),
                "works_at".into(),
                "anthropic".into(),
                1.0,
                None,
                None,
            )
            .unwrap();

        let results = field.query_entity("alice").unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_integration_recall_keyword() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];

        field
            .put_memory(
                "wisdom",
                "test",
                b"rust ownership model prevents memory leaks automatically",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field
            .put_memory(
                "wisdom",
                "test",
                b"python garbage collector handles memory management",
                &emb,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();

        let hits = field.recall_keyword("rust ownership", 5).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].kind, "wisdom");
        // "rust" and "ownership" only in doc 1
        assert!(hits[0].content.contains("rust"));
    }

    #[test]
    fn test_recall_effects_are_deferred_until_flush() {
        let (field, _tmp) = open_test_field();

        let mut emb1 = vec![0.0f32; crate::ops::EMBED_DIM];
        emb1[0] = 1.0;
        let mut emb2 = vec![0.0f32; crate::ops::EMBED_DIM];
        emb2[1] = 1.0;

        field
            .put_memory(
                "wisdom",
                "test",
                b"alpha memory",
                &emb1,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        field
            .put_memory(
                "wisdom",
                "test",
                b"beta memory",
                &emb2,
                1.0,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();

        let seqno_before = field.log.read().last_seqno();
        let hits = field.recall_semantic(&emb1, 2, Some("test")).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(field.log.read().last_seqno(), seqno_before);
        assert!(!field.pending_recall.lock().strengthen.is_empty());

        field.flush().unwrap();
        assert!(field.log.read().last_seqno() > seqno_before);
        assert!(field.pending_recall.lock().strengthen.is_empty());
    }

    // ── Status-aware recall tests ─────────────────────────────────────────────

    /// Superseded/Contradicted/Archived memories must be excluded from semantic recall.
    #[test]
    fn test_recall_excludes_invalidated_statuses() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; crate::ops::EMBED_DIM];

        let (id_active, _)     = field.put_memory("wisdom", "test", b"active memory",     &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_superseded, _) = field.put_memory("wisdom", "test", b"superseded memory", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_contradicted,_)= field.put_memory("wisdom", "test", b"contradicted memory",&emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_archived, _)   = field.put_memory("wisdom", "test", b"archived memory",   &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        field.set_memory_status(id_superseded,    crate::state::MemoryStatus::Superseded).unwrap();
        field.set_memory_status(id_contradicted,  crate::state::MemoryStatus::Contradicted).unwrap();
        field.set_memory_status(id_archived,      crate::state::MemoryStatus::Archived).unwrap();

        let hits = field.recall_semantic(&emb, 20, None).unwrap();
        let ids: Vec<_> = hits.iter().map(|h| h.memory_id).collect();

        assert!(ids.contains(&id_active),        "active memory must be recalled");
        assert!(!ids.contains(&id_superseded),   "superseded must be excluded");
        assert!(!ids.contains(&id_contradicted), "contradicted must be excluded");
        assert!(!ids.contains(&id_archived),     "archived must be excluded");
    }

    /// Verified memories score higher than Active; Proposed score lower.
    #[test]
    fn test_recall_status_score_ordering() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; crate::ops::EMBED_DIM];

        let (id_active,   _) = field.put_memory("wisdom", "test", b"active",   &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_verified, _) = field.put_memory("wisdom", "test", b"verified", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_proposed, _) = field.put_memory("wisdom", "test", b"proposed", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        field.set_memory_status(id_verified, crate::state::MemoryStatus::Verified).unwrap();
        field.set_memory_status(id_proposed, crate::state::MemoryStatus::Proposed).unwrap();

        let hits = field.recall_semantic(&emb, 20, None).unwrap();
        let score = |id: MemoryId| hits.iter().find(|h| h.memory_id == id).map(|h| h.score).unwrap_or(0.0);

        assert!(score(id_verified) > score(id_active),  "verified must outscore active");
        assert!(score(id_active)   > score(id_proposed), "active must outscore proposed");
    }

    // ── Recall explainability tests ─────────────────────────────────────────

    #[test]
    fn test_recall_explain_fields_populated() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; crate::ops::EMBED_DIM];

        let (id, _) = field.put_memory("wisdom", "test", b"tool derived memory", &emb, 0.9, 0.001, 0, vec![], None, None).unwrap();
        field.set_epistemic_status(id, crate::state::EpistemicStatus::ToolDerived).unwrap();

        let hits = field.recall_semantic(&emb, 5, Some("test")).unwrap();
        let hit = hits.iter().find(|h| h.memory_id == id).expect("memory must be recalled");

        assert!(hit.semantic_weight > 0.0, "semantic_weight must be > 0");
        assert!((hit.status_mul - 1.0).abs() < f32::EPSILON, "Active status_mul must be 1.0");
        assert!((hit.epistemic_mul - 0.95).abs() < f32::EPSILON, "ToolDerived epistemic_mul must be 0.95");
        assert!(hit.strength_factor >= 0.5 && hit.strength_factor <= 1.0, "strength_factor must be in [0.5, 1.0]");
    }

    #[test]
    fn test_recall_explain_score_decomposition() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.5f32; crate::ops::EMBED_DIM];

        let (id, _) = field.put_memory("wisdom", "test", b"decomposition test", &emb, 0.8, 0.001, 0, vec![], None, None).unwrap();

        let hits = field.recall_semantic(&emb, 5, Some("test")).unwrap();
        let hit = hits.iter().find(|h| h.memory_id == id).expect("memory must be recalled");

        // Score is the product of all pipeline factors:
        // relevance × actr × strength × confidence × surprise × arousal × mood × frustration
        // × status × epistemic × kind × realm_reliability
        // For a fresh memory with default config, most boosts are 1.0.
        // Just verify score is positive and decomp fields are populated.
        assert!(hit.score > 0.0, "score must be positive");
        assert!(hit.strength_factor >= 0.5, "strength_factor must be >= 0.5");
        assert!(hit.semantic_weight > 0.0, "semantic_weight must be > 0");
        assert!(hit.status_mul > 0.0, "status_mul must be > 0");
        assert!(hit.epistemic_mul > 0.0, "epistemic_mul must be > 0");
    }

    #[test]
    fn test_recall_keyword_explain_fields() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];

        field.put_memory("wisdom", "test", b"rust ownership borrow checker lifetime", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        let hits = field.recall_keyword("rust ownership", 5).unwrap();
        assert!(!hits.is_empty(), "keyword recall must return results");
        let hit = &hits[0];

        assert!(hit.semantic_weight > 0.0, "semantic_weight must be bm25_score > 0");
        assert!(hit.status_mul > 0.0, "status_mul must be populated");
        assert!(hit.epistemic_mul > 0.0, "epistemic_mul must be populated");
        assert!(hit.strength_factor >= 0.5, "strength_factor must be >= 0.5");
    }

    // ── Contradiction engine tests ──────────────────────────────────────────

    #[test]
    fn test_get_conflicts_bidirectional() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id_a, _) = field.put_memory("wisdom", "test", b"memory A", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_b, _) = field.put_memory("wisdom", "test", b"memory B", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        field.add_triplet(id_a.to_string(), "contradicts".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

        let conflicts_a = field.get_conflicts(id_a).unwrap();
        let conflicts_b = field.get_conflicts(id_b).unwrap();
        assert!(conflicts_a.contains(&id_b), "A must see B as conflict");
        assert!(conflicts_b.contains(&id_a), "B must see A as conflict");
    }

    #[test]
    fn test_get_supersession_chain_follows_edges() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id_a, _) = field.put_memory("wisdom", "test", b"original", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_b, _) = field.put_memory("wisdom", "test", b"revision 1", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_c, _) = field.put_memory("wisdom", "test", b"revision 2", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        // "B supersedes A" means subject=B, predicate="supersedes", object=A
        field.add_triplet(id_b.to_string(), "supersedes".to_string(), id_a.to_string(), 1.0, None, None).unwrap();
        field.add_triplet(id_c.to_string(), "supersedes".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

        let chain = field.get_supersession_chain(id_a).unwrap();
        assert_eq!(chain, vec![id_a, id_b, id_c], "chain must follow A -> B -> C");
    }

    #[test]
    fn test_get_supersession_chain_cycle_safe() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id_a, _) = field.put_memory("wisdom", "test", b"cycle A", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_b, _) = field.put_memory("wisdom", "test", b"cycle B", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        // Create a cycle: B supersedes A, A supersedes B
        field.add_triplet(id_b.to_string(), "supersedes".to_string(), id_a.to_string(), 1.0, None, None).unwrap();
        field.add_triplet(id_a.to_string(), "supersedes".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

        let chain = field.get_supersession_chain(id_a).unwrap();
        assert!(chain.len() <= 21, "cycle must terminate within max depth");
        assert_eq!(chain[0], id_a, "chain must start with self");
    }

    #[test]
    fn test_get_confirmations() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id_x, _) = field.put_memory("wisdom", "test", b"confirmer", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
        let (id_y, _) = field.put_memory("wisdom", "test", b"confirmed", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        // "X confirms Y" means subject=X, predicate="confirms", object=Y
        field.add_triplet(id_x.to_string(), "confirms".to_string(), id_y.to_string(), 1.0, None, None).unwrap();

        let confs = field.get_confirmations(id_y).unwrap();
        assert_eq!(confs, vec![id_x], "Y must show X as confirmer");
    }

    #[test]
    fn test_get_conflicts_empty() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        let (id, _) = field.put_memory("wisdom", "test", b"lonely memory", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

        let conflicts = field.get_conflicts(id).unwrap();
        assert!(conflicts.is_empty(), "no contradictions should return empty vec");
    }

    // ── Regression tests for replay/contract correctness ─────────────────────

    fn put_test_memory(field: &ChittaField, content: &[u8]) -> MemoryId {
        let emb = vec![0.1f32; crate::ops::EMBED_DIM];
        field.put_memory("wisdom", "test", content, &emb, 1.0, 0.001, 0, vec![], None, None)
            .unwrap().0
    }

    /// Bug fix: UpdateState replay used now_ms=0, corrupting last_accessed_ms and
    /// last_strengthened_ms. After reopen the timestamps must reflect op_ts_ms, not epoch 0.
    #[test]
    fn test_replay_update_state_timestamps_nonzero() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let id = put_test_memory(&field, b"state-replay");
            // touch=true writes an UpdateState op with real op_ts_ms
            field.update_state(id, None, None, None, true, None).unwrap();
            field.flush().unwrap();
            id
        };

        let field2 = ChittaField::open(data_dir).unwrap();
        let state = field2.get_state(id).unwrap();
        assert!(
            state.last_accessed_ms > 0,
            "last_accessed_ms must not be 0 after replay, got {}",
            state.last_accessed_ms
        );
    }

    /// Bug fix: UpdateMemoryContent replay did not clear embed_pending, so backfilled
    /// memories were re-queued as pending after every restart.
    #[test]
    fn test_replay_backfill_clears_embed_pending() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");

        let id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            // Empty embedding slice → embed_pending = true
            let (id, _) = field.put_memory("wisdom", "test", b"this memory needs an embedding backfill", &[], 1.0, 0.001, 0, vec![], None, None).unwrap();
            let emb = vec![0.2f32; crate::ops::EMBED_DIM];
            field.backfill_embedding(id, &emb).unwrap();
            field.flush().unwrap();
            id
        };

        let field2 = ChittaField::open(data_dir).unwrap();
        assert!(
            !field2.pending_embeddings(100).contains(&id),
            "backfilled memory must not appear in pending_embeddings after replay"
        );
    }

    /// Bug fix: backfill_embedding() previously returned Ok(()) for nonexistent IDs.
    #[test]
    fn test_backfill_nonexistent_returns_not_found() {
        let (field, _tmp) = open_test_field();
        let emb = vec![0.0f32; crate::ops::EMBED_DIM];
        let fake_id: MemoryId = 0xdeadbeef_cafebabe;
        let result = field.backfill_embedding(fake_id, &emb);
        assert!(
            matches!(result, Err(crate::error::FieldError::NotFound(_))),
            "expected NotFound, got {:?}", result
        );
    }

    /// Bug fix: set_memory_status() and set_epistemic_status() wrote WAL before
    /// confirming the memory exists, leaving orphaned WAL entries on invalid IDs.
    #[test]
    fn test_set_status_invalid_id_no_wal_mutation() {
        let (field, _tmp) = open_test_field();
        let fake_id: MemoryId = 0xdeadbeef_00000001;
        let seqno_before = field.log.read().last_seqno();

        let r1 = field.set_memory_status(fake_id, crate::state::MemoryStatus::Archived);
        let r2 = field.set_epistemic_status(fake_id, crate::state::EpistemicStatus::ModelInferred);

        assert!(matches!(r1, Err(crate::error::FieldError::NotFound(_))));
        assert!(matches!(r2, Err(crate::error::FieldError::NotFound(_))));
        assert_eq!(
            field.log.read().last_seqno(), seqno_before,
            "WAL must not grow when ID is invalid"
        );
    }

    #[test]
    fn test_compact_wal_guard_rejects_small_store() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
        for i in 0..50 {
            field
                .put_memory(
                    "wisdom",
                    "test",
                    format!("memory {}", i).as_bytes(),
                    &embedding,
                    0.9,
                    0.001,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
        }
        let result = field.compact_wal();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("refusing compact_wal"),
            "expected guard error, got: {}", err_msg
        );
    }

    #[test]
    fn test_compact_wal_guard_allows_large_store() {
        let (field, _tmp) = open_test_field();
        let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
        for i in 0..100 {
            field
                .put_memory(
                    "wisdom",
                    "test",
                    format!("memory {}", i).as_bytes(),
                    &embedding,
                    0.9,
                    0.001,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
        }
        let result = field.compact_wal();
        assert!(result.is_ok(), "compact_wal should succeed with 100+ memories, got: {:?}", result);
    }

    #[test]
    fn test_filter_level_signatures_reduces_terms() {
        let (field, _tmp) = open_test_field();
        field.set_filter_level(FilterLevel::Signatures);
        let code = b"fn foo(x: i32) -> i32 {\n    let y = x + 1;\n    y\n}";
        let (id, _) = field
            .put_memory("code", "test", code, &[], 0.8, 0.001, 0, vec![], None, None)
            .unwrap();
        let hits = field.recall_keyword("fn foo", 5).unwrap();
        assert!(hits.iter().any(|h| h.memory_id == id));
        let body_hits = field.recall_keyword("let y", 5).unwrap();
        assert!(!body_hits.iter().any(|h| h.memory_id == id));
    }

    #[test]
    fn test_recall_fallback_to_bm25() {
        let (field, _tmp) = open_test_field();
        for i in 0..15 {
            field
                .put_memory(
                    "wisdom",
                    "test",
                    format!("unique_term_{i} content here").as_bytes(),
                    &[],
                    0.8,
                    0.001,
                    0,
                    vec![],
                    None,
                    None,
                )
                .unwrap();
        }
        let hits = field
            .recall_with_fallback(&vec![0.0f32; crate::ops::EMBED_DIM], "unique_term_0", 5, None)
            .unwrap();
        assert!(!hits.is_empty(), "fallback should return results");
    }

    /// Capability #1 (anti-reprocessing): the keyed provenance lane resolves a
    /// `[done]` record by content-hash OR input path through an exact O(1) lookup
    /// that never touches the fuzzy retriever, keeps the earliest record when a
    /// sha is re-registered under different surrounding text, survives snapshot
    /// save/reopen (rebuilt deterministically from live payloads), and misses
    /// cleanly for unknown keys.
    #[test]
    fn provenance_keyed_lane_exact_lookup_and_persists() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let put = |field: &ChittaField, body: &str| {
            field
                .put_memory("signal", "cc-soul", body.as_bytes(), &[], 0.8, 0.001, 0, vec![], None, None)
                .unwrap()
                .0
        };

        let (first_id, second_id) = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let a = put(&field, "[done] input:/data/x.fastq sha:deadbeef12 task:qc output:/out/x.tsv status:ok");
            // Same sha, different task text: escapes the byte-identical string gate,
            // so it stores as a second live record — the keyed lane must still
            // resolve to the EARLIEST record.
            let b = put(&field, "[done] input:/data/x.fastq sha:deadbeef12 task:requant output:/out/x2.tsv status:ok");
            assert_ne!(a, b, "same-sha different-text must be two records");

            assert_eq!(field.provenance_lookup("deadbeef12", "").unwrap().0, a, "sha lookup -> earliest");
            assert_eq!(field.provenance_lookup("", "/data/x.fastq").unwrap().0, a, "path lookup -> earliest");
            assert_eq!(field.provenance_lookup("sha:deadbeef12", "").unwrap().0, a, "prefixed arg resolves");
            assert!(field.provenance_lookup("deadbeef12", "").unwrap().1.contains("task:qc"), "returns earliest content");
            assert!(field.provenance_lookup("cafef00d99", "/nope").is_none(), "unknown key misses cleanly");

            field.save_full_snapshot().unwrap();
            (a, b)
        };
        assert_ne!(first_id, second_id);

        // Reopen: lane rebuilt from live payloads, deterministically earliest.
        let field = ChittaField::open(data_dir).unwrap();
        assert_eq!(field.provenance_lookup("deadbeef12", "").unwrap().0, first_id, "rebuilt lane keeps earliest after reopen");
        assert!(field.provenance_lookup("deadbeef12", "").unwrap().1.contains("task:qc"));
        assert!(field.provenance_lookup("cafef00d99", "").is_none());
    }

    /// Capability #2: durable corrections with override semantics. Proves the
    /// correction keyed lane fires deterministically when the corrected
    /// mistake's trigger recurs in a turn (an exact bigram probe, no fuzzy
    /// retriever), applies LATEST-WINS + SUPERSEDE when a newer correction
    /// shares a trigger, misses cleanly on unrelated turns, and rebuilds
    /// deterministically (newest-wins) across a snapshot save/reopen.
    #[test]
    fn correction_keyed_lane_fires_latest_wins_and_persists() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let put = |field: &ChittaField, body: &str| {
            field
                .put_memory("correction", "cc-soul", body.as_bytes(), &[], 0.9, 0.001, 0, vec![], None, None)
                .unwrap()
                .0
        };

        let old_id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            // A correction whose mistake is "cp over the running binary".
            let a = put(&field, "[correction] USE: install atomic rename\nNOT: cp over the running binary");

            // Trigger recurs in a turn -> the correction FIRES (the exact bigram
            // "running_binary"/"cp_over"... survives after stopword strip).
            let hit = field.correction_check("can I just cp over the running binary to deploy?");
            assert!(hit.is_some(), "recurring mistake trigger must fire the correction");
            assert_eq!(hit.as_ref().unwrap().0, a, "fires the stored correction id");
            assert!(hit.unwrap().1.contains("install atomic rename"), "surfaces the USE: fix");

            // Unrelated turn -> clean miss (no fuzzy false-positive).
            assert!(field.correction_check("what's the weather in the terminal today").is_none(),
                    "unrelated turn must miss cleanly");

            // Too-short correction (< 2 significant tokens) never indexes -> stays
            // on the fuzzy lane, so it can't be found by the keyed probe.
            let _short = put(&field, "[correction] USE: yes\nNOT: no");
            assert!(field.correction_check("no").is_none(), "sub-bigram correction is not keyed");

            // LATEST-WINS / SUPERSEDE: a newer correction sharing the trigger
            // replaces the old one for that key.
            let b = put(&field, "[correction] USE: install -m 0755 atomic\nNOT: cp over the running binary ETXTBSY");
            let hit2 = field.correction_check("cp over the running binary now").unwrap();
            assert_eq!(hit2.0, b, "newest correction supersedes older for a shared trigger");
            assert!(hit2.1.contains("ETXTBSY"), "surfaces the superseding correction body");
            assert_ne!(a, b);

            field.save_full_snapshot().unwrap();
            a
        };

        // Reopen: lane rebuilt from live payloads, deterministically newest-wins.
        let field = ChittaField::open(data_dir).unwrap();
        let hit = field.correction_check("cp over the running binary now").unwrap();
        assert_ne!(hit.0, old_id, "rebuilt lane keeps the SUPERSEDING correction after reopen");
        assert!(hit.1.contains("ETXTBSY"), "rebuilt lane surfaces newest body");
        assert!(field.correction_check("totally unrelated question").is_none());
    }

    #[test]
    fn correction_uppercase_header_form_is_keyed() {
        // The free-form `[CORRECTION to memory #… — topic]` header (uppercase,
        // extra words) is how most real corrections are stored. The old
        // lowercase-exact gate rejected it, silently disabling the keyed lane.
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let field = ChittaField::open(data_dir).unwrap();
        let body = "[CORRECTION to memory #2411395996631171237 — reuse background] \
                    realm:pan-barley. My earlier claim \"reuse has 0 AFDB background by \
                    construction\" was WRONG. Cause: spine_member_col omits AFDB scaffold.";
        let id = field
            .put_memory("correction", "brahman", body.as_bytes(), &[], 0.9, 0.001, 0, vec![], None, None)
            .unwrap()
            .0;
        let hit = field.correction_check("reuse has 0 AFDB background by construction");
        assert!(hit.is_some(), "uppercase-header correction must fire on its mistake restatement");
        assert_eq!(hit.unwrap().0, id, "fires the stored uppercase-header correction id");
    }

    #[test]
    fn correction_multivalued_credits_shared_bigrams() {
        // Older correction A's mistake shares 2 of its 3 bigrams with a NEWER
        // correction B. Single-valued latest-wins would let B steal both keys, so
        // A could never reach the 2-bigram threshold on its own restatement (only
        // B — the wrong correction — would fire). The multi-valued index credits A
        // for every bigram it owns, so A's restatement still surfaces A.
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let field = ChittaField::open(data_dir).unwrap();
        let put = |body: &str| field
            .put_memory("correction", "cc-soul", body.as_bytes(), &[], 0.9, 0.001, 0, vec![], None, None)
            .unwrap()
            .0;
        put("[correction] USE: quartz lattice mica\nNOT: alpha beta gamma delta");
        let _newer = put("[correction] USE: other fix\nNOT: alpha beta gamma zeta");
        let hit = field
            .correction_check("alpha beta gamma delta")
            .expect("A's restatement must fire despite B stealing shared bigrams");
        assert!(hit.1.contains("quartz lattice mica"),
                "multi-valued index must credit older A for its shared bigrams (single-valued drops it)");
    }

    /// Capability #3: task hand-off across discontinuous sessions. Proves the
    /// task-state keyed lane resolves a `[task]` record by its slug through an
    /// exact O(1) lookup, applies LATEST-WINS when the status evolves
    /// (in-progress -> done), returns the DONE record after the update, misses
    /// cleanly on an unknown id, and rebuilds deterministically (newest-wins)
    /// across a snapshot save/reopen.
    #[test]
    fn task_state_keyed_lane_latest_wins_and_persists() {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Stored kind:signal like the [job]/[done] rituals; the lane self-gates
        // on the [task] prefix, not the kind.
        let put = |field: &ChittaField, body: &str| {
            field
                .put_memory("signal", "cc-soul", body.as_bytes(), &[], 0.8, 0.001, 0, vec![], None, None)
                .unwrap()
                .0
        };

        let done_id = {
            let field = ChittaField::open(data_dir.clone()).unwrap();
            let a = put(&field, "[task] id:foo status:in-progress next:wire the FFI layer");
            // Same task, evolved status -> a second live record; the lane must
            // resolve to the LATEST (done), not the first (in-progress).
            let b = put(&field, "[task] id:foo status:done next:ship it");
            assert_ne!(a, b, "evolving status stores as two records");

            let hit = field.task_state_lookup("foo").unwrap();
            assert_eq!(hit.0, b, "slug lookup -> LATEST record (recency wins)");
            assert!(hit.1.contains("status:done"), "returns the newest status");
            assert_eq!(field.task_state_lookup("task:foo").unwrap().0, b, "prefixed arg resolves");
            assert!(field.task_state_lookup("bar").is_none(), "unknown id misses cleanly");

            field.save_full_snapshot().unwrap();
            b
        };

        // Reopen: lane rebuilt from live payloads, deterministically newest-wins.
        let field = ChittaField::open(data_dir).unwrap();
        let hit = field.task_state_lookup("foo").unwrap();
        assert_eq!(hit.0, done_id, "rebuilt lane keeps the LATEST record after reopen");
        assert!(hit.1.contains("status:done"), "rebuilt lane surfaces newest status");
        assert!(field.task_state_lookup("bar").is_none());
    }

    // ── Deferred-batched-insert (Step-1 re-architecture) proofs ────────────────

    fn lcg_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..dim).map(|_| {
            s ^= s >> 33; s = s.wrapping_mul(0xff51afd7ed558ccd); s ^= s >> 33;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        }).collect()
    }
    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        v.iter().map(|x| x / n).collect()
    }

    /// Recall@k PARITY gate: the deferred-batched insert path must match the per-item
    /// path. For the PRODUCTION recall path (flat scan over the embeddings map, active
    /// below FLAT_SCAN_MAX=2M) the two are bitwise-identical (same `upsert_meta`); for
    /// the HNSW fallback graph they must be approximation-equivalent.
    #[test]
    fn test_deferred_batch_recall_parity() {
        use crate::hnsw::SemanticIndex;
        use crate::ops::EMBED_DIM;
        const N: u64 = 5200;   // > HNSW_TIER2_THRESHOLD (5000): exercises the delta tier
        const Q: usize = 30;
        const K: usize = 10;

        let items: Vec<(u64, Vec<f32>)> = (1..=N).map(|i| (i, lcg_vec(i, EMBED_DIM))).collect();

        // Model production: a pre-existing base graph (built at load by the epoch-based
        // backfill_hnsw_delta_parallel), then INCREMENTAL backfill of new memories. Seed
        // both paths identically, then diverge: A per-item upsert, B deferred-batched.
        const SEED: usize = 3000;
        let mut a = SemanticIndex::new();
        let mut b = SemanticIndex::new();
        // Build the HNSW graph regardless of size: below flat_scan_max() production
        // serves via flat scan and never builds it, but this test probes the graph
        // directly (hnsw_search_test) to compare per-item vs batched construction.
        a.force_hnsw_build_for_test();
        b.force_hnsw_build_for_test();
        for (id, e) in &items[..SEED] {
            a.upsert(*id, e.clone(), Some("test"));
            b.upsert(*id, e.clone(), Some("test"));
        }
        // Incremental tail — the path under test.
        for (id, e) in &items[SEED..] { a.upsert(*id, e.clone(), Some("test")); }
        for (id, e) in &items[SEED..] { b.upsert_deferred(*id, e.clone(), Some("test")); }
        for chunk in items[SEED..].chunks(100) {   // 100 = daemon pending_embeddings(100) batch
            let ids: Vec<u64> = chunk.iter().map(|(id, _)| *id).collect();
            let plan = b.plan_delta_batch(&ids);
            b.apply_delta_batch(plan);
        }

        // Ground-truth brute-force cosine top-k.
        let normed: Vec<(u64, Vec<f32>)> = items.iter().map(|(id, e)| (*id, unit(e))).collect();
        let (mut flat_identical, mut ra, mut rb) = (0usize, 0.0f64, 0.0f64);
        for qi in 0..Q {
            let q = unit(&lcg_vec(1_000_000 + qi as u64, EMBED_DIM));
            let mut bf: Vec<(f32, u64)> = normed.iter()
                .map(|(id, e)| (e.iter().zip(&q).map(|(x, y)| x * y).sum::<f32>(), *id)).collect();
            bf.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
            let truth: std::collections::HashSet<u64> = bf.iter().take(K).map(|(_, id)| *id).collect();

            // (a) Production path (default env = flat scan): must be IDENTICAL A vs B.
            let fa: Vec<u64> = a.search(&q, K, None, None).into_iter().map(|h| h.memory_id).collect();
            let fb: Vec<u64> = b.search(&q, K, None, None).into_iter().map(|h| h.memory_id).collect();
            if fa == fb { flat_identical += 1; }

            // (b) HNSW fallback graph: approximation-equivalent quality.
            let ha: std::collections::HashSet<u64> = a.hnsw_search_test(&q, K).into_iter().map(|h| h.memory_id).collect();
            let hb: std::collections::HashSet<u64> = b.hnsw_search_test(&q, K).into_iter().map(|h| h.memory_id).collect();
            ra += truth.intersection(&ha).count() as f64 / K as f64;
            rb += truth.intersection(&hb).count() as f64 / K as f64;
        }
        ra /= Q as f64; rb /= Q as f64;
        eprintln!("[parity] flat-path identical: {flat_identical}/{Q}  |  HNSW recall@{K}: per-item={ra:.3} batched={rb:.3}");
        assert_eq!(flat_identical, Q, "production flat-scan recall must be identical A vs B");
        assert!(rb > 0.0, "batched HNSW path returns hits");
        assert!(rb >= ra - 0.05, "batched HNSW recall {rb:.3} within 0.05 of per-item {ra:.3}");
    }

    /// WRITES-DON'T-BLOCK-READS proof: measure the total EXCLUSIVE (write-lock) hold time
    /// to insert a batch — this is exactly what blocks recall (recall needs the shared
    /// lock; a held write lock stalls it). Per-item holds the write lock across the
    /// O(log N) global + per-realm HNSW neighbor SEARCH for every item. The deferred path
    /// holds it only for cheap metadata + the pointer-wire apply; the search runs under a
    /// READ lock (plan), which recall can share. Stable metric (a sum, not a noisy
    /// worst-case) — mirrors the daemon's [lockprof] EXCLUSIVE-hold reduction.
    #[test]
    fn test_deferred_batch_exclusive_hold() {
        use crate::hnsw::SemanticIndex;
        use crate::ops::EMBED_DIM;
        const SEED: u64 = 5000;    // > HNSW_TIER2_THRESHOLD: global delta tier active
        const BATCH: usize = 500;
        // 3 realms, each ~1/3 of the corpus (> 500): per-realm HNSW active too, so both
        // graph inserts (global + per-realm) are on the write path in the per-item case.
        let realm_of = |i: u64| -> &'static str { ["ra", "rb", "rc"][(i % 3) as usize] };
        let seed: Vec<(u64, Vec<f32>)> = (1..=SEED).map(|i| (i, lcg_vec(i, EMBED_DIM))).collect();
        let tail: Vec<(u64, Vec<f32>)> =
            (SEED + 1..=SEED + BATCH as u64).map(|i| (i, lcg_vec(i, EMBED_DIM))).collect();
        let build = || {
            let mut s = SemanticIndex::new();
            for (id, e) in &seed { s.upsert(*id, e.clone(), Some(realm_of(*id))); }
            s
        };

        // Per-item: every upsert holds the write lock across the HNSW neighbor search.
        let mut a = build();
        let mut hold_a = std::time::Duration::ZERO;
        for (id, e) in &tail {
            let t = std::time::Instant::now();
            a.upsert(*id, e.clone(), Some(realm_of(*id)));
            hold_a += t.elapsed();
        }

        // Batched: EXCLUSIVE hold = metadata write + apply write. The neighbor search
        // (plan_delta_batch) runs OFF the write lock and is NOT counted — the whole point.
        let mut b = build();
        let ids: Vec<u64> = tail.iter().map(|(id, _)| *id).collect();
        let t0 = std::time::Instant::now();
        for (id, e) in &tail { b.upsert_deferred(*id, e.clone(), Some(realm_of(*id))); }
        let mut hold_b = t0.elapsed();                 // metadata (exclusive)
        let plan = b.plan_delta_batch(&ids);           // OFF-LOCK (read) — not counted
        let t1 = std::time::Instant::now();
        b.apply_delta_batch(plan);
        hold_b += t1.elapsed();                         // apply (exclusive)

        eprintln!("[exclusive-hold] per-item={:?} batched={:?} (search moved off-lock into plan)",
                  hold_a, hold_b);
        assert!(hold_b.as_micros() * 3 < hold_a.as_micros(),
            "batched exclusive hold {hold_b:?} must be <⅓ of per-item {hold_a:?}");
    }
}
