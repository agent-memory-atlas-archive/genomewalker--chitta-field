mod organs;

mod maintenance;

mod recall;

mod keyed;

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
pub(crate) fn audited_remove(path: impl AsRef<std::path::Path>, reason: &str) -> std::io::Result<()> {
    let path = path.as_ref();
    let result = std::fs::remove_file(path);
    eprintln!("[chitta-field] unlink path={} reason={} result={:?}", path.display(), reason, result);
    result
}

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
                if empty && audited_remove(path, "wal-dead-empty").is_ok() {
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
            if max_covered >= seg_end && audited_remove(path, "wal-covered").is_ok() {
                deleted += 1;
            }
        }
        // The instance's final (open-ended) segment: prune only for a dead
        // instance present in `covered` (fully folded). Never the live tail.
        if inst != live_instance {
            if let Some((first, path)) = segs.last() {
                if covered.get(&inst).is_some_and(|&c| c >= *first)
                    && audited_remove(path, "wal-covered").is_ok()
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
            if audited_remove(&p, "snapshot-family-prune").is_ok() { removed += 1; }
        }
        let p = data_dir.join(format!("{}.{}", stem, delta_ext));
        if audited_remove(&p, "snapshot-family-prune").is_ok() { removed += 1; }
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
                if audited_remove(&p, "snapshot-family-prune").is_ok() { removed += 1; }
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
            if audited_remove(p, "snapshot-family-prune").is_ok() { removed += 1; }
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
                if audited_remove(entry.path(), "snapshot-family-prune").is_ok() { removed += 1; }
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
            if audited_remove(p, "janitor-aged-orphan").is_ok() {
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
                    self.record_replication_session(existing_id, source_session.as_deref())?;
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
                    self.record_replication_session(existing_id, source_session.as_deref())?;
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
                        self.record_replication_session(top.memory_id, source_session.as_deref())?;
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
        self.anchors.write().upsert(memory_id, &payload);
        self.payloads.write().insert(memory_id, payload);
        self.pld_mutations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.memory_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.states.write().insert(memory_id, state);
        self.refresh_replications(&[memory_id]);
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
        let observer_canonicals = if self.ablations.disabled("observer") || self.ablations.disabled("observer_state") {
            Vec::new()
        } else {
            self.observer.extract(&content_str, memory_id, authored_at_ms, &mut self.observer_state.write())
        };
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
        if !self.ablations.disabled("event_tape") {
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
            if !self.ablations.disabled("cdawg") { cdawg.extend(sym, turn); }
            // Phase 15: update FEP model and blend surprisal signal.
            let fep_free_energy = if self.ablations.disabled("fep_prior") { 0.0 }
                else { self.fep_prior.write().observe_packed(sym, &cdawg).free_energy };
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
        if kind == "process" && !self.ablations.disabled("archive") {
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

    // ── Soul REPL session persistence ──────────────────────────────────────────

    pub fn repl_session_get(&self, id: &str) -> Option<String> {
        if self.ablations.disabled("repl_sessions") { return None; }
        self.repl_sessions.read().get(id).map(|s| s.namespace_json.clone())
    }

    pub fn repl_session_set(&self, id: &str, namespace_json: &str, updated_ms: i64) {
        if self.ablations.disabled("repl_sessions") { return (); }
        self.repl_sessions.write().set(id.to_string(), namespace_json.to_string(), updated_ms);
    }

    pub fn repl_session_delete(&self, id: &str) -> bool {
        if self.ablations.disabled("repl_sessions") { return false; }
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
        if self.ablations.disabled("repl_sessions") { return String::from("{}"); }
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
        if self.ablations.disabled("repl_sessions") { return String::from("[]"); }
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
        self.refresh_replications(&[memory_id]);
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
        self.anchors.write().entries.remove(&memory_id);

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

    /// Log a structured action event to the CEC tape and extend the CDAWG.
    /// `outcome`: 0=success 1=fail 2=error 3=partial.
    /// Also pushes TD(λ) eligibility-trace credit for non-synthetic events.
    pub fn log_event(&self, tool: &str, entity: &str, outcome: u8, session_id: u64, ts_ms: i64) {
        if self.ablations.disabled("event_tape") { return (); }
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
        if self.ablations.disabled("event_tape") { return (); }
        let sym = self.event_tape.write().symbol_of(tool, entity, outcome);
        self.cdawg.write().record_outcome(&[sym], success);
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

        let replication_ids = crate::replication::affected_edge(&subject, &predicate, &object);
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

        self.refresh_replications(&replication_ids);
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

    /// Set the source_session payload tag (snapshot-persisted). Preserve both
    /// old and new session witnesses in the WAL before changing the tag.
    pub fn set_source_session(&self, memory_id: MemoryId, session_id: &str) -> Result<()> {
        let previous = self.payloads.read().get(&memory_id)
            .ok_or(FieldError::NotFound(memory_id))?.source_session.clone();
        self.record_replication_session(memory_id, previous.as_deref())?;
        self.record_replication_session(memory_id, Some(session_id))?;
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
        self.refresh_replications(&[memory_id]);
        self.densify_write_edges(memory_id, &realm, Some(session_id));
        Ok(())
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

        let replication_ids = self.replication_edge_ids(triplet_id);
        self.triplet_store.write().invalidate(triplet_id, now);
        self.refresh_replications(&replication_ids);

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
        let replication_ids = self.replication_edge_ids(old_id);
        self.triplet_store.write().supersede(old_id, new_id, at_ms);
        self.refresh_replications(&replication_ids);
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
        if self.ablations.disabled("learners") { return Ok(()); }
        self.learners.write().route.feedback(episode_id, reward);
        Ok(())
    }

    /// Get recommended window size for a session type.
    pub fn recommended_window(&self, session_type: &str) -> usize {
        if self.ablations.disabled("learners") { return 0; }
        self.learners
            .read()
            .context
            .recommended_window(session_type)
    }

    /// Record context outcome for a session type and window size.
    pub fn record_context_outcome(&self, session_type: &str, size: usize, outcome: f32) {
        if self.ablations.disabled("learners") { return (); }
        self.learners
            .write()
            .context
            .record_outcome(session_type, size, outcome);
    }

    /// Select a retrieval route using Thompson sampling. Returns (episode_id, route).
    pub fn select_route(&self, query: &str) -> (u64, Route) {
        if self.ablations.disabled("learners") { return (0, Route::Hybrid); }
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
        if self.ablations.disabled("cortical_idx") { return 0; }
        self.cortical_idx.read().len()
    }

    pub fn prototype_count(&self) -> usize {
        if self.ablations.disabled("cortical_idx") { return 0; }
        self.cortical_idx.read().prototype_count()
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(0); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(false); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(None); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(None); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(None); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(false); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(None); }
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
        if self.ablations.disabled("agent_protocol_store") { return None; }
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
        if self.ablations.disabled("agent_protocol_store") { return Vec::new(); }
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
        if self.ablations.disabled("agent_protocol_store") { return Ok(0); }
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
        if self.ablations.disabled("interaction_ledger") { return Ok(0); }
        Ok(self.interaction_ledger.write().append(ev))
    }

    pub fn ledger_query(
        &self,
        kind: Option<crate::organ::interaction_ledger::EventKind>,
        session_id: Option<&str>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<crate::organ::interaction_ledger::InteractionEvent>> {
        if self.ablations.disabled("interaction_ledger") { return Ok(Vec::new()); }
        Ok(self.interaction_ledger.read()
            .query(kind.as_ref(), session_id, since_ms, limit)
            .into_iter()
            .cloned()
            .collect())
    }

    pub fn ledger_compile(&self) -> Result<usize> {
        if self.ablations.disabled("interaction_ledger") { return Ok(0); }
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
        if self.ablations.disabled("interaction_ledger") { return Ok(Vec::new()); }
        Ok(self.interaction_ledger.read().contested())
    }

    pub fn predicate_attach(&self, memory_id: u64, check_cmd: String) -> Result<u64> {
        if self.ablations.disabled("predicate_store") { return Ok(0); }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Ok(self.predicate_store.write().attach(memory_id, check_cmd, now_ms))
    }

    /// Run all predicates for a memory. Returns JSON with per-predicate results.
    /// Weakens memory confidence by 0.1 for each failing predicate (min 0.1).
    pub fn predicate_run(&self, memory_id: u64) -> Result<String> {
        if self.ablations.disabled("predicate_store") { return Ok(String::from("[]")); }
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
        if self.ablations.disabled("predicate_store") { return Ok(String::from("[]")); }
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
        if self.ablations.disabled("symbol_event_log") { return 0; }
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
        if self.ablations.disabled("symbol_event_log") { return String::from("[]"); }
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

}

#[cfg(test)]
mod tests;
