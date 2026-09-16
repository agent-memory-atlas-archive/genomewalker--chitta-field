use crate::ids::{ChunkHash, MemoryId};
use crate::ops::StateDeltaOp;
use serde::{Deserialize, Serialize};

/// Compact quantized sketch of a query that retrieved this memory.
/// 32 i8 values + scale reconstruct a ~32-dim projected embedding.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalContext {
    pub centroid_q: Vec<i8>,   // 32 dims, quantized
    pub scale: f32,
    pub context_hash: u64,     // hash of original query embedding + realm
    pub ts_ms: i64,
}

/// Rolling history of up to 8 query contexts that retrieved a memory.
/// `signature` is the cached mean of the stored centroids (full f32, 32 dims).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalHistory {
    pub entries: Vec<RetrievalContext>,  // capped at 8, FIFO oldest-drop
    pub signature: Vec<f32>,             // cached mean centroid (32 dims)
}

impl RetrievalHistory {
    pub const MAX_ENTRIES: usize = 8;
    pub const CENTROID_DIMS: usize = 32;

    /// Append a new retrieval context; if full, drop the oldest.
    /// Recomputes the cached mean signature.
    pub fn push(&mut self, ctx: RetrievalContext) {
        if self.entries.len() >= Self::MAX_ENTRIES {
            self.entries.remove(0);
        }
        self.entries.push(ctx);
        self.recompute_signature();
    }

    fn recompute_signature(&mut self) {
        if self.entries.is_empty() {
            self.signature.clear();
            return;
        }
        let n = self.entries.len() as f32;
        let dims = Self::CENTROID_DIMS;
        let mut sig = vec![0f32; dims];
        for entry in &self.entries {
            for (i, &q) in entry.centroid_q.iter().take(dims).enumerate() {
                sig[i] += (q as f32 * entry.scale) / n;
            }
        }
        self.signature = sig;
    }
}

/// Mutable overlay state for a memory — lives separately from the immutable payload.
/// First-class memory lifecycle status.
/// `Active` is the default; `Superseded` is set when a correction replaces this memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum MemoryStatus {
    #[default]
    Active,
    Superseded,
    Contradicted,
    Archived,
    Proposed,
    Observed,
    Verified,
}

/// How a memory was obtained — orthogonal to confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum EpistemicStatus {
    UserStated = 0,
    #[default]
    ToolDerived = 1,
    ModelInferred = 2,
    AutonomousSynthesis = 3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryState {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: ChunkHash,
    pub deleted: bool,

    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,

    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,

    pub pinned: bool,
    pub tier: u8, // 0=L1 (hippocampus), 1=L2 (cortex), 2=L3 (archive)
    #[serde(default)]
    pub last_state_op_ts_ms: i64,
    #[serde(default)]
    pub retrieval_history: RetrievalHistory,
    #[serde(default)]
    pub embed_pending: bool,
    #[serde(default)]
    pub status: MemoryStatus,
    #[serde(default)]
    pub epistemic_status: EpistemicStatus,
    /// Reconstruction surprise: how poorly the sparse encoder predicts this memory.
    /// High surprise = unique information, low surprise = redundant. FEP §2.3.
    #[serde(default)]
    pub surprise: f32,
    /// Affect valence: -1.0 (negative/frustration) to +1.0 (positive/eureka). 0 = neutral.
    /// Inspired by Anthropic's emotion vector research (2026).
    #[serde(default)]
    pub affect_valence: f32,
    /// Affect arousal: 0.0 (calm) to 1.0 (intense). High arousal = flashbulb memory effect.
    #[serde(default)]
    pub affect_arousal: f32,
    /// ACT-R access timestamps: last 16 retrieval wall-clock times (ms since epoch).
    /// Used for power-law base-level activation: B_i = ln(Σ t_j^(-d)).
    #[serde(default)]
    pub access_timestamps: Vec<i64>,

    // ── Interference-aware fields (Price of Meaning / Geometry of Forgetting) ──
    /// Local competitor density: mean cosine similarity to k-nearest neighbors [0,1].
    /// High = crowded neighborhood → more interference-driven forgetting.
    /// Precomputed on write path; O(1) read during scoring.
    #[serde(default)]
    pub competitive_weight: f32,
    /// False-recall risk: competitive_weight weighted by same-kind neighbor ratio [0,1].
    /// Used by post-pipeline LureDetector to suppress confusable candidates.
    #[serde(default)]
    pub lure_risk: f32,
    /// Retrieval spacing regularity [0,1]. Derived from coefficient of variation
    /// of inter-retrieval intervals in access_timestamps. Well-spaced = high quality.
    #[serde(default)]
    pub spacing_quality: f32,

    /// True until recalled at least once. Staged memories rank 0.80× lower;
    /// auto-pruned after 7 days if never recalled.
    #[serde(default = "default_true")]
    pub staged: bool,

    /// Commit hash or symbol-change ID that deterministically invalidated this memory.
    #[serde(default)]
    pub invalidated_by: Option<String>,

    /// Wall-clock ms timestamp of last competitive_weight refresh. Skipped here
    /// to keep the positional bincode layout of V12+ snapshots stable; persisted
    /// via `FullSnapshot::cw_refresh_ts` (V22) and hydrated on load.
    #[serde(skip)]
    pub last_cw_refresh_ms: i64,

    /// Outcome utility posterior Beta(α, β): α accrues success weight, β failure
    /// weight, both via `record_outcome`. Prior (1, 1) = untested. Access is not
    /// value, so this is independent of `strength`/`access_count`.
    /// Skipped for the same reason as `last_cw_refresh_ms` (frozen positional
    /// bincode layout); persisted via `FullSnapshot::utility_posteriors` (V23
    /// section) and hydrated on load, with `Op::RecordOutcome` in the WAL
    /// covering every observation made since that snapshot (without it a
    /// restart between two periodic saves reverted α/β).
    /// `sanitize()` restores the (1, 1) prior,
    /// which is what makes the serde-skip zero-default and a snapshot written
    /// before the section both read back as untested.
    #[serde(skip)]
    pub utility_alpha: f32,
    #[serde(skip)]
    pub utility_beta: f32,

    /// Derived from durable session provenance after load; never changes snapshot layout.
    #[serde(skip)]
    pub replication_count: u32,
}

fn default_true() -> bool { true }

impl MemoryState {
    pub fn new(memory_id: MemoryId, chunk_hash: ChunkHash, created_at_ms: i64) -> Self {
        Self {
            memory_id,
            current_version: 0,
            current_chunk_hash: chunk_hash,
            deleted: false,
            strength: 1.0,
            decay_rate: 0.001,
            confidence: 1.0,
            access_count: 0,
            last_accessed_ms: created_at_ms,
            last_strengthened_ms: created_at_ms,
            created_at_ms,
            pinned: false,
            tier: 0,
            last_state_op_ts_ms: 0,
            retrieval_history: RetrievalHistory::default(),
            embed_pending: false,
            status: MemoryStatus::Active,
            epistemic_status: EpistemicStatus::ToolDerived,
            surprise: 0.0,
            affect_valence: 0.0,
            affect_arousal: 0.0,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: true,
            invalidated_by: None,
            last_cw_refresh_ms: 0,
            utility_alpha: 1.0,
            utility_beta: 1.0,
            replication_count: 1,
        }
    }

    /// Clamp all float fields to valid ranges. Fixes corrupt data from V7→V8 migration.
    pub fn sanitize(&mut self) {
        if !self.confidence.is_finite() || self.confidence < 0.0 || self.confidence > 1.0 {
            self.confidence = 0.5;
        }
        if !self.strength.is_finite() || self.strength < 0.0 || self.strength > 1.0 {
            self.strength = 0.5;
        }
        if !self.affect_valence.is_finite() || self.affect_valence < -1.0 || self.affect_valence > 1.0 {
            self.affect_valence = 0.0;
        }
        if !self.affect_arousal.is_finite() || self.affect_arousal < 0.0 || self.affect_arousal > 1.0 {
            self.affect_arousal = 0.0;
        }
        if !self.surprise.is_finite() || self.surprise < 0.0 || self.surprise > 1.0 {
            self.surprise = 0.0;
        }
        if !self.decay_rate.is_finite() || self.decay_rate < 0.0 {
            self.decay_rate = 0.001;
        }
        if !self.competitive_weight.is_finite() {
            self.competitive_weight = 0.0;
        }
        if !self.lure_risk.is_finite() {
            self.lure_risk = 0.0;
        }
        if !self.spacing_quality.is_finite() {
            self.spacing_quality = 0.0;
        }
        // Posterior counts start at the (1, 1) prior and only accrue, so
        // anything below 1 is either the serde-skip zero or corruption.
        if !self.utility_alpha.is_finite() || self.utility_alpha < 1.0 {
            self.utility_alpha = 1.0;
        }
        if !self.utility_beta.is_finite() || self.utility_beta < 1.0 {
            self.utility_beta = 1.0;
        }
    }

    /// Accrue one outcome observation into the utility posterior. Weight is
    /// capped at 5 so a single report can move the posterior but not swamp it;
    /// a non-positive or non-finite weight is not an observation and is dropped.
    // `!(weight > 0.0)` is the NaN-catching form and is deliberate: it is true for
    // NaN, whereas the `weight <= 0.0` that clippy's refactor invites is false for
    // NaN, which would let a NaN weight through and poison the posterior
    // irrecoverably (α/β are persisted). The negation is the guard, not an
    // oversight.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn record_outcome(&mut self, success: bool, weight: f32) {
        if !(weight > 0.0) {
            return;
        }
        let w = weight.min(5.0);
        if success {
            self.utility_alpha += w;
        } else {
            self.utility_beta += w;
        }
    }

    /// Posterior mean utility, α/(α+β). 0.5 at the untouched prior.
    pub fn utility_mean(&self) -> f32 {
        self.utility_alpha / (self.utility_alpha + self.utility_beta)
    }

    pub fn apply_delta(&mut self, delta: &StateDeltaOp, now_ms: i64) {
        if delta.op_ts_ms > 0 && delta.op_ts_ms <= self.last_state_op_ts_ms {
            return;
        }
        if delta.op_ts_ms > 0 {
            self.last_state_op_ts_ms = delta.op_ts_ms;
        }
        if let Some(d) = delta.strength_delta {
            self.strength = (self.strength + d).clamp(0.0, 1.0);
            if d > 0.0 {
                self.last_strengthened_ms = now_ms;
            }
        }
        if let Some(d) = delta.confidence_delta {
            self.confidence = (self.confidence + d).clamp(0.0, 1.0);
        }
        if let Some(r) = delta.decay_rate {
            self.decay_rate = r.max(0.0);
        }
        if delta.touch {
            self.access_count += 1;
            self.last_accessed_ms = now_ms;
            // ACT-R: record access timestamp for power-law decay
            if self.access_timestamps.len() >= 16 {
                self.access_timestamps.remove(0);
            }
            self.access_timestamps.push(now_ms);
        }
        if let Some(p) = delta.pin {
            self.pinned = p;
        }
        if let Some(s) = delta.status {
            self.status = match s {
                1 => MemoryStatus::Superseded,
                2 => MemoryStatus::Contradicted,
                3 => MemoryStatus::Archived,
                4 => MemoryStatus::Proposed,
                5 => MemoryStatus::Observed,
                6 => MemoryStatus::Verified,
                _ => MemoryStatus::Active,
            };
        }
        if let Some(es) = delta.epistemic_status {
            self.epistemic_status = match es {
                0 => EpistemicStatus::UserStated,
                2 => EpistemicStatus::ModelInferred,
                3 => EpistemicStatus::AutonomousSynthesis,
                _ => EpistemicStatus::ToolDerived,
            };
        }
        if let Some(s) = delta.staged {
            self.staged = s;
        }
        if delta.invalidated_by.is_some() {
            self.invalidated_by = delta.invalidated_by.clone();
        }
    }

    /// Compute effective strength accounting for time-based decay.
    pub fn effective_strength(&self, now_ms: i64) -> f32 {
        let age_days = (now_ms - self.last_strengthened_ms).max(0) as f64 / 86_400_000.0;
        let decayed = self.strength as f64 * (-self.decay_rate as f64 * age_days).exp();
        decayed.clamp(0.0, 1.0) as f32
    }

    /// Recompute spacing_quality from access_timestamps.
    /// Uses coefficient of variation of inter-retrieval intervals.
    /// Well-spaced (regular intervals) → high quality; bursty → low quality.
    pub fn recompute_spacing_quality(&mut self) {
        if self.access_timestamps.len() < 3 {
            self.spacing_quality = 0.0;
            return;
        }
        let intervals: Vec<f64> = self.access_timestamps.windows(2)
            .map(|w| (w[1] - w[0]).max(1) as f64)
            .collect();
        let n = intervals.len() as f64;
        let mean = intervals.iter().sum::<f64>() / n;
        if mean < 1.0 {
            self.spacing_quality = 0.0;
            return;
        }
        let var = intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let cv = var.sqrt() / mean; // coefficient of variation
        // Low CV = regular spacing = high quality. CV→0 means perfect spacing → 1.0
        // High CV = bursty = low quality → 0.0
        self.spacing_quality = (1.0 / (1.0 + cv)).min(1.0) as f32;
    }

    /// ACT-R base-level activation: B_i = ln(Σ_j t_j^(-d))
    ///
    /// Power-law decay over access history (Anderson & Schooler 1991).
    /// For recorded timestamps: exact sum. For older unrecorded accesses:
    /// uniform-distribution approximation over memory lifetime.
    /// Returns [0, 1] via sigmoid transform.
    pub fn actr_base_level_activation(&self, now_ms: i64) -> f32 {
        const D: f64 = 0.5;          // power-law decay exponent
        const TAU: f64 = 1.5;        // sigmoid temperature
        const THRESHOLD: f64 = -1.0; // sigmoid midpoint
        const MIN_AGE_MS: i64 = 60_000; // 1 minute floor to avoid singularity

        let mut sum = 0.0f64;

        // Exact power-law sum over recorded access timestamps
        for &ts in &self.access_timestamps {
            let age_ms = (now_ms - ts).max(MIN_AGE_MS);
            let age_days = age_ms as f64 / 86_400_000.0;
            sum += age_days.powf(-D);
        }

        // Approximate older unrecorded accesses (uniform over lifetime)
        let n_exact = self.access_timestamps.len() as u32;
        let n_missing = self.access_count.saturating_sub(n_exact);
        if n_missing > 0 {
            let lifetime_ms = (now_ms - self.created_at_ms).max(MIN_AGE_MS);
            let lifetime_days = lifetime_ms as f64 / 86_400_000.0;
            // ∫_0^L (n/L) × t^(-d) dt = n × L^(-d) / (1-d)
            sum += (n_missing as f64) * lifetime_days.powf(-D) / (1.0 - D);
        }

        // Ensure at least creation event contributes
        if sum <= 0.0 {
            let age_ms = (now_ms - self.created_at_ms).max(MIN_AGE_MS);
            let age_days = age_ms as f64 / 86_400_000.0;
            sum = age_days.powf(-D);
        }

        let activation = sum.ln();
        let factor = 1.0 / (1.0 + (-(activation - THRESHOLD) / TAU).exp());
        (factor as f32).clamp(0.0, 1.0)
    }
}
