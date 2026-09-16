use serde::{Deserialize, Serialize};
use std::path::Path;

/// Runtime-configurable scoring parameters.
///
/// Loaded from `scoring.json` in the data directory. If absent, defaults
/// are used. Every constant that was previously hardcoded in the scoring
/// pipeline now lives here — tunable without recompilation.
///
/// The config file is a developmental scaffold: as the cognitive system
/// matures, parameters can shift (e.g. increasing surprise weight as the
/// memory store grows, or reducing frustration sensitivity as corrections
/// accumulate). This is the "neuroplasticity" layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoringConfig {
    // ── ACT-R base-level activation ─────────────────────────────────────
    /// Power-law decay exponent (d in B_i = ln(Σ t_j^(-d))). Default 0.5.
    pub actr_decay_exponent: f32,
    /// Sigmoid temperature for activation → [0,1] mapping.
    pub actr_sigmoid_tau: f32,
    /// Sigmoid midpoint (activation value that maps to 0.5).
    pub actr_sigmoid_threshold: f32,
    /// Floor of the ACT-R factor (minimum contribution even for zero activation).
    pub actr_floor: f32,
    /// Range above floor (factor = floor + range * activation).
    pub actr_range: f32,

    // ── Prediction error (FEP) ──────────────────────────────────────────
    /// Surprise boost weight: factor = 1 + weight * surprise.
    pub surprise_weight: f32,

    // ── Flashbulb / arousal ─────────────────────────────────────────────
    /// Arousal boost weight: factor = 1 + weight * arousal.
    pub arousal_weight: f32,

    // ── Mood-congruent recall ───────────────────────────────────────────
    /// Maximum mood-congruence boost.
    pub mood_weight: f32,
    /// Minimum query arousal to activate mood matching.
    pub mood_min_arousal: f32,

    // ── Frustration-escalation ──────────────────────────────────────────
    /// Query valence threshold below which frustration activates.
    pub frustration_valence_threshold: f32,
    /// Query arousal threshold above which frustration activates.
    pub frustration_arousal_threshold: f32,
    /// Frustration boost weight for corrections.
    pub frustration_correction_weight: f32,
    /// Frustration boost weight for preferences.
    pub frustration_preference_weight: f32,

    // ── Status multipliers ──────────────────────────────────────────────
    pub status_verified: f32,
    pub status_active: f32,
    pub status_observed: f32,
    pub status_proposed: f32,

    // ── Kind multipliers ────────────────────────────────────────────────
    pub kind_correction: f32,
    pub kind_wisdom: f32,
    pub kind_insight: f32,
    pub kind_episode: f32,
    /// Distiller fact fragments lifted verbatim from transcripts: evidence, not
    /// conclusions, so they rank below curated kinds at equal similarity.
    pub kind_operational: f32,
    pub kind_default: f32,

    // ── Epistemic multipliers ───────────────────────────────────────────
    pub epistemic_user_stated: f32,
    pub epistemic_tool_derived: f32,
    pub epistemic_model_inferred: f32,
    pub epistemic_autonomous: f32,

    // ── Strength decay ──────────────────────────────────────────────────
    /// Floor of strength factor: factor = floor + range * effective_strength.
    pub strength_floor: f32,
    /// Range of strength factor above floor.
    pub strength_range: f32,
    /// Default per-day exponential decay rate for new memories.
    pub default_decay_rate: f32,

    // ── Deduplication ───────────────────────────────────────────────────
    /// Cosine similarity threshold for semantic dedup at ingestion.
    pub dedup_cosine_threshold: f32,
    /// Cosine similarity upper bound (above = exact duplicate, handled by chunk hash).
    pub dedup_cosine_upper: f32,

    // ── Competitive-weight refresh ──────────────────────────────────────
    /// Minimum interval (ms) between competitive_weight refreshes per memory.
    pub cw_refresh_interval_ms: i64,

    // ── Association spreading ───────────────────────────────────────────
    /// Per-hop decay factor for spreading activation.
    pub assoc_hop_decay: f32,

    // ── Cross-context generality (THEORY.md §6) ─────────────────────────
    /// Boost per additional distinct recalling instance:
    /// factor = 1 + weight * min(distinct − 1, cross_context_max). 0 disables.
    pub cross_context_weight: f32,
    pub cross_context_max: f32,

    /// Saturating independent-session multiplier; 0 disables. Opt-in value: 1.15.
    pub replication_max: f32,

    /// Max competitive-weight refreshes per recall. Each refresh costs one
    /// ANN/flat search; unbounded refresh after a restart with stale
    /// timestamps turned k-large recalls into minutes-long scan convoys
    /// (production 2026-06-11). Refresh is amortized across queries instead.
    pub cw_refresh_max_per_query: usize,

    // ── Interference density (Price of Meaning) ────────────────────────
    /// Penalty weight for local competitor crowding.
    /// Factor = 1 / (1 + interference_penalty * competitive_weight).
    pub interference_penalty: f32,

    // ── Spacing boost (Geometry of Forgetting) ─────────────────────────
    /// Floor of spacing factor (minimum contribution).
    pub spacing_floor: f32,
    /// Range above floor: factor = floor + range * spacing_quality.
    pub spacing_range: f32,

    // ── Prediction boost (Markov chain access predictor) ──────────────
    /// Prediction boost weight: factor = 1 + weight * prediction_probability.
    pub prediction_weight: f32,

    // ── Surprise domain (Layer 4) ──────────────────────────────────────
    #[serde(default = "default_surprise_domain_actual_weight")]
    pub surprise_domain_actual_weight: f32,
    #[serde(default = "default_surprise_domain_expected_weight")]
    pub surprise_domain_expected_weight: f32,

    // ── Epistemic debt (Layer 5) ───────────────────────────────────────
    #[serde(default = "default_epistemic_debt_boost")]
    pub epistemic_debt_boost: f32,

    // ── Lure detection (post-pipeline) ─────────────────────────────────
    /// Lure risk threshold above which candidates may be suppressed.
    pub lure_risk_threshold: f32,
    /// Maximum number of lure-flagged candidates to suppress per query.
    pub lure_max_suppressed: usize,

    // ── Rare-entity surprisal ─────────────────────────────────────────
    #[serde(default = "default_rare_entity_weight")]
    pub rare_entity_weight: f32,

    // ── Stratified recall (anti-flooding) ────────────────────────
    /// On unscoped recall, cap any single realm to ceil(k / divisor) hits so a
    /// dominant realm (e.g. compliance:auto BM25 noise) can't flood results.
    /// 0 disables the cap. Only applies when the caller passes realm=None.
    #[serde(default = "default_recall_realm_cap_divisor")]
    pub recall_realm_cap_divisor: usize,

    // ── Reciprocal Rank Fusion (hybrid recall) ───────────────────────
    /// RRF constant k; standard value is 60.0. Larger k flattens the rank
    /// weighting so deeper positions contribute more evenly.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// Enable RRF hybrid merge of HNSW semantic + BM25 keyword results on
    /// unscoped recall. When false, BM25 is only a fallback for empty semantic.
    #[serde(default = "default_use_rrf")]
    pub use_rrf: bool,

    // ── Field-RAG / Modern Hopfield recall ───────────────────────────
    /// Inverse temperature β for DAM update s(t+1) = X@softmax(β·Xᵀs(t)).
    /// Probe validated β=5-10 as the sweet spot for bge-large-en-v1.5 embeddings.
    #[serde(default = "default_dam_beta")]
    pub dam_beta: f32,
    /// Number of relaxation steps T. Convergence typically within 8-10 steps.
    #[serde(default = "default_dam_steps")]
    pub dam_steps: usize,
    /// Candidate pool multiplier: fetch k * dam_fetch_mul candidates for DAM reranking.
    #[serde(default = "default_dam_fetch_mul")]
    pub dam_fetch_mul: usize,
    /// Multi-hop count: after DAM relaxation produces s_T, use it as a new query for
    /// additional retrieval passes (excluding already-fetched IDs). 1 = single-hop
    /// (current behaviour). 2 = one extra hop using s_T as the refined query vector.
    #[serde(default = "default_dam_hops")]
    pub dam_hops: usize,

    // ── Cortical SDR re-rank (third RRF pass) ────────────────────────
    /// Enable cortical posting-index re-rank as a second RRF pass over the
    /// already-merged HNSW+BM25 candidate set. Safe to enable once the
    /// SparseEncoder has seen ≥1000 memories. Default: false.
    #[serde(default)]
    pub use_cortical: bool,
    /// RRF constant k for the cortical re-rank pass (default 60.0).
    #[serde(default = "default_cortical_rrf_k")]
    pub cortical_rrf_k: f32,
}

fn default_cortical_rrf_k() -> f32 { 60.0 }
fn default_rare_entity_weight() -> f32 { 0.15 }
fn default_recall_realm_cap_divisor() -> usize { 4 }
fn default_rrf_k() -> f32 { 60.0 }
fn default_use_rrf() -> bool { true }
fn default_dam_beta() -> f32 { 6.0 }
fn default_dam_steps() -> usize { 2 }
fn default_dam_fetch_mul() -> usize { 4 }
fn default_dam_hops() -> usize { 1 }
fn default_surprise_domain_actual_weight() -> f32 { 0.15 }
fn default_surprise_domain_expected_weight() -> f32 { 0.10 }
fn default_epistemic_debt_boost() -> f32 { 1.1 }

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            // ACT-R
            actr_decay_exponent: 0.5,
            actr_sigmoid_tau: 1.5,
            actr_sigmoid_threshold: -1.0,
            actr_floor: 0.3,
            actr_range: 0.7,

            // Surprise
            surprise_weight: 0.25,

            // Arousal
            arousal_weight: 0.15,

            // Mood congruence
            mood_weight: 0.2,
            mood_min_arousal: 0.1,

            // Frustration
            frustration_valence_threshold: -0.3,
            frustration_arousal_threshold: 0.4,
            frustration_correction_weight: 1.5,
            frustration_preference_weight: 1.0,

            // Status
            status_verified: 1.15,
            status_active: 1.0,
            status_observed: 0.85,
            status_proposed: 0.65,

            // Kind
            kind_correction: 1.3,
            kind_wisdom: 1.1,
            kind_insight: 1.05,
            kind_episode: 0.7,
            kind_operational: 0.8,
            kind_default: 1.0,

            // Epistemic
            epistemic_user_stated: 1.0,
            epistemic_tool_derived: 0.95,
            epistemic_model_inferred: 0.85,
            epistemic_autonomous: 0.75,

            // Strength
            strength_floor: 0.5,
            strength_range: 0.5,
            default_decay_rate: 0.001,

            // Dedup
            dedup_cosine_threshold: 0.88,
            dedup_cosine_upper: 0.9999,

            // Competitive-weight refresh
            cw_refresh_interval_ms: 60_000,

            // Association
            assoc_hop_decay: 0.55,

            // Cross-context generality
            cross_context_weight: 0.05,
            replication_max: 0.0,
            cross_context_max: 3.0,

            // Competitive-weight refresh budget
            cw_refresh_max_per_query: 16,

            // Interference (Price of Meaning)
            interference_penalty: 0.3,

            // Spacing (Geometry of Forgetting)
            spacing_floor: 0.85,
            spacing_range: 0.15,

            // Prediction
            prediction_weight: 0.3,

            // Surprise domain
            surprise_domain_actual_weight: 0.15,
            surprise_domain_expected_weight: 0.10,

            // Epistemic debt
            epistemic_debt_boost: 1.1,

            // Lure detection
            lure_risk_threshold: 0.7,
            lure_max_suppressed: 2,

            // Rare-entity surprisal
            rare_entity_weight: 0.15,

            // Stratified recall (anti-flooding)
            recall_realm_cap_divisor: 4,

            // Reciprocal Rank Fusion (hybrid recall)
            rrf_k: 60.0,
            use_rrf: true,

            // Field-RAG / Modern Hopfield
            // β=6, steps=2: grade-recall sweep shows β=2/steps=10 over-mixes (candidate
            // centroid drift zeros out secondary gold). Higher β sharpens the attractor;
            // fewer steps prevents drift-to-centroid. Sweep {4,5,6,8,10}×{1,2,3} to tune.
            dam_beta: 6.0,
            dam_steps: 2,
            dam_fetch_mul: 4,
            dam_hops: 1,

            // Cortical SDR re-rank
            use_cortical: false,
            cortical_rrf_k: 60.0,
        }
    }
}

impl ScoringConfig {
    /// Load config from `scoring.json` in the given directory.
    /// Falls back to defaults if the file doesn't exist or is malformed.
    pub fn load(data_dir: &Path) -> Self {
        let mut config = Self::load_file(data_dir);
        if let Ok(value) = std::env::var("CHITTA_REPLICATION_MAX") {
            if let Ok(value) = value.parse::<f32>() {
                if value.is_finite() && (value == 0.0 || value >= 1.0) {
                    config.replication_max = value;
                }
            }
        }
        config
    }

    fn load_file(data_dir: &Path) -> Self {
        let path = data_dir.join("scoring.json");
        if !path.exists() {
            log::info!("No scoring.json found, using default scoring config");
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(config) => {
                    log::info!("Loaded scoring config from {}", path.display());
                    config
                }
                Err(e) => {
                    log::warn!("Malformed scoring.json: {}, using defaults", e);
                    Self::default()
                }
            },
            Err(e) => {
                log::warn!("Cannot read scoring.json: {}, using defaults", e);
                Self::default()
            }
        }
    }

    /// Save current config to `scoring.json` for inspection/editing.
    pub fn save(&self, data_dir: &Path) -> std::io::Result<()> {
        let path = data_dir.join("scoring.json");
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}
