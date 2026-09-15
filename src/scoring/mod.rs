pub mod config;
pub mod factors;
pub mod learned;

use crate::state::{EpistemicStatus, MemoryState, MemoryStatus};
use config::ScoringConfig;

/// Everything a scoring factor needs to compute its contribution.
/// Borrowed references avoid cloning during the hot recall path.
pub struct ScoringContext<'a> {
    /// Raw relevance signal: cosine similarity (semantic) or BM25 score (keyword).
    pub relevance_score: f32,
    /// Which recall path produced this candidate.
    pub recall_mode: RecallMode,
    /// The memory's mutable state (strength, decay, access history, affect, etc.).
    pub state: &'a MemoryState,
    /// Memory kind (e.g. "correction", "wisdom", "episode").
    pub kind: &'a str,
    /// Memory realm (namespace).
    pub realm: &'a str,
    /// Per-realm PoE reliability score.
    pub realm_reliability: f32,
    /// Current wall-clock time in ms since epoch.
    pub now_ms: i64,
    /// Caller's affect state (None = not provided).
    pub query_valence: Option<f32>,
    pub query_arousal: Option<f32>,
    /// Prediction probability from Markov chain predictor (None = not predicted).
    pub prediction_prob: Option<f32>,
    /// Surprise role: was this memory involved in a surprise event? (Layer 4)
    pub surprise_role: Option<SurpriseRole>,
    /// Does this memory's domain have open epistemic debt? (Layer 5)
    pub has_open_debt: bool,
    /// Learned source weight from integration kernel (Layer 6)
    pub integration_weight: Option<f32>,
    /// Cumulative ack/nack score for this memory (positive = proven useful, negative = stale/wrong).
    pub ack_score: i32,
    /// Max IDF of query tokens that appear in the keyword index — rare-entity signal.
    pub max_query_idf: f32,
}

#[derive(Debug, Clone)]
pub enum SurpriseRole {
    WasActual(f32),
    WasExpected(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallMode {
    Semantic,
    Keyword,
}

/// Decomposed score — one field per built-in factor, matching CfRecallHit FFI layout.
#[derive(Debug, Clone, Default)]
pub struct ScoreDecomposition {
    pub semantic_weight: f32,
    pub strength_factor: f32,
    pub status_mul: f32,
    pub epistemic_mul: f32,
    pub actr_activation: f32,
    pub surprise_boost: f32,
    pub arousal_boost: f32,
    pub mood_congruence: f32,
    pub frustration_boost: f32,
    pub interference_factor: f32,
    pub spacing_boost: f32,
    pub surprise_domain_factor: f32,
    pub epistemic_debt_factor: f32,
    pub integration_weight_factor: f32,
    pub rare_entity_boost: f32,
}

/// A single scoring factor in the pipeline.
///
/// Each factor computes a multiplicative contribution to the final score.
/// Returning `None` excludes the memory entirely (e.g. Superseded status).
pub trait ScoringFactor: Send + Sync {
    fn name(&self) -> &'static str;
    fn compute(
        &self,
        ctx: &ScoringContext,
        config: &ScoringConfig,
        decomp: &mut ScoreDecomposition,
    ) -> Option<f32>;
}

/// Composable scoring pipeline. Factors are evaluated in order; their
/// contributions are multiplied together. Any factor returning None
/// vetoes the memory (excluded from results).
pub struct ScoringPipeline {
    factors: Vec<Box<dyn ScoringFactor>>,
    pub config: ScoringConfig,
}

impl ScoringPipeline {
    /// Build the default pipeline with all built-in cognitive factors.
    pub fn new(config: ScoringConfig) -> Self {
        use factors::*;
        let factors: Vec<Box<dyn ScoringFactor>> = vec![
            Box::new(RelevanceFactor),
            Box::new(ACTRActivationFactor),
            Box::new(StrengthFactor),
            Box::new(ConfidenceFactor),
            Box::new(SurpriseFactor),
            Box::new(ArousalFactor),
            Box::new(MoodCongruenceFactor),
            Box::new(FrustrationEscalationFactor),
            Box::new(StatusFactor),
            Box::new(EpistemicFactor),
            Box::new(KindFactor),
            Box::new(RealmReliabilityFactor),
            Box::new(InterferenceDensityFactor),
            Box::new(SpacingBoostFactor),
            Box::new(PredictionFactor),
            Box::new(SurpriseDomainFactor),
            Box::new(EpistemicDebtFactor),
            Box::new(IntegrationWeightFactor),
            Box::new(AckScoreFactor),
            Box::new(RareEntityFactor),
            Box::new(StagedFactor),
        ];
        Self { factors, config }
    }

    /// Score a candidate memory. Returns None if any factor vetoes it.
    pub fn score(&self, ctx: &ScoringContext) -> Option<(f32, ScoreDecomposition)> {
        // Bounded-envelope composite (NOT a pure product). Relevance is the anchor; all
        // secondary factors (recency/ACT-R, confidence, status, epistemic, kind, …) modulate
        // it within +/-30% via the envelope (0.7 + 0.3*secondary). A pure product let the
        // secondaries multiply a highly-relevant but old/low-confidence memory down to ~0 —
        // the "3% inversion" that surfaced once mean-centering compressed the cosine spread
        // (with raw cosine's larger magnitude masking it). Relevance still caps the product,
        // so an irrelevant memory cannot be re-inflated. Design room room-f366bf5a (2026-06-02).
        let mut relevance = 1.0f32;
        let mut secondary = 1.0f32;
        let mut decomp = ScoreDecomposition::default();

        for factor in &self.factors {
            match factor.compute(ctx, &self.config, &mut decomp) {
                Some(val) => {
                    if factor.name() == "relevance" {
                        relevance *= val;
                    } else {
                        secondary *= val;
                    }
                }
                None => return None,
            }
        }

        let envelope = 0.7 + 0.3 * secondary;
        Some((relevance * envelope, decomp))
    }

    /// Replace the config (hot-reload). Thread-safe when called under write lock.
    pub fn reload_config(&mut self, config: ScoringConfig) {
        self.config = config;
    }
}

/// Helper: status multiplier from config. Returns None for excluded statuses.
pub fn status_multiplier(status: &MemoryStatus, config: &ScoringConfig) -> Option<f32> {
    match status {
        MemoryStatus::Active     => Some(config.status_active),
        MemoryStatus::Verified   => Some(config.status_verified),
        MemoryStatus::Observed   => Some(config.status_observed),
        MemoryStatus::Proposed   => Some(config.status_proposed),
        MemoryStatus::Superseded | MemoryStatus::Contradicted | MemoryStatus::Archived => None,
    }
}

/// Helper: epistemic multiplier from config.
pub fn epistemic_multiplier(es: &EpistemicStatus, config: &ScoringConfig) -> f32 {
    match es {
        EpistemicStatus::UserStated          => config.epistemic_user_stated,
        EpistemicStatus::ToolDerived         => config.epistemic_tool_derived,
        EpistemicStatus::ModelInferred       => config.epistemic_model_inferred,
        EpistemicStatus::AutonomousSynthesis => config.epistemic_autonomous,
    }
}

/// Helper: kind multiplier from config.
pub fn kind_multiplier(kind: &str, config: &ScoringConfig) -> f32 {
    match kind {
        "correction" | "preference" => config.kind_correction,
        "wisdom"                    => config.kind_wisdom,
        "insight"                   => config.kind_insight,
        "episode"                   => config.kind_episode,
        "operational"               => config.kind_operational,
        _                           => config.kind_default,
    }
}

#[cfg(test)]
mod kind_tests {
    use super::*;

    #[test]
    fn operational_fragments_rank_below_curated_kinds() {
        let c = ScoringConfig::default();
        let k = |kind| kind_multiplier(kind, &c);
        assert!(k("operational") < k("wisdom"));
        assert!(k("operational") < k("correction"));
        assert!(k("operational") < k("signal"));
        assert!(k("operational") > k("episode"));
    }
}
