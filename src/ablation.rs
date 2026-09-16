//! Per-open experimental ablations. Never serialized: rollback data is separate
//! from the empty runtime view, and existing snapshot codecs remain authoritative.
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub(crate) const ORGANS: &[&str] = &[
    "session_registry",
    "transcript_registry",
    "task_registry",
    "user_model_registry",
    "theme_organ",
    "analytics_registry",
    "msg_registry",
    "skill_registry",
    "agent_registry",
    "constraint_store",
    "trigger_store",
    "surprise_store",
    "epistemic_debt_store",
    "intervention_store",
    "agent_protocol_store",
    "wisdom_lineage_store",
    "symbol_event_log",
    "repl_sessions",
    "span_store",
    "event_tape",
    "cdawg",
    "episode_hdc",
    "refutation_ledger",
    "cec_policy_store",
    "decision_tape",
    "hypothesis_market",
    "turiya_monitor",
    "fep_prior",
    "observer",
    "observer_state",
    "interaction_ledger",
    "predicate_store",
    "archive",
    "cortical_idx",
    "hdc_idx",
    "lite_encoder",
    "sparse_encoder",
    "learners",
    "predictor",
    "integration_kernel",
    "surprise_learning",
    "wisdom_promotion",
    "learned_scorer",
];

#[derive(Clone, Default)]
pub(crate) struct Ablations(HashSet<String>);

impl Ablations {
    pub(crate) fn parse(value: &str) -> crate::error::Result<Self> {
        let mut names = HashSet::new();
        for name in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if !ORGANS.contains(&name) {
                return Err(crate::FieldError::Other(format!("unknown CHITTA_ABLATE_ORGANS organ: {name}")));
            }
            names.insert(name.to_owned());
        }
        Ok(Self(names))
    }

    pub(crate) fn from_env() -> crate::error::Result<Self> {
        Self::parse(&std::env::var("CHITTA_ABLATE_ORGANS").unwrap_or_default())
    }

    /// Suppress only NEW organ-owned WAL records. Existing records always replay
    /// into preserved state, including during ablation, so rollback loses nothing.
    pub(crate) fn suppresses(&self, op: &crate::ops::Op) -> bool {
        use crate::ops::Op;
        let organ = match op {
            Op::SessionEvent(_) => "session_registry",
            Op::TranscriptEvent(_) => "transcript_registry",
            Op::TaskEvent(_) => "task_registry",
            Op::UserModelEvent(_) => "user_model_registry",
            Op::ThemeEvent(_) => "theme_organ",
            Op::AnalyticsEvent(_) => "analytics_registry",
            Op::MsgEvent(_) => "msg_registry",
            Op::SkillUpload(_) | Op::SkillDeprecate(_) => "skill_registry",
            Op::AgentUpsert(_) | Op::AgentDisable(_) => "agent_registry",
            Op::AssertConstraint(_) | Op::RetractConstraint(_) | Op::CreateBranch(_) | Op::ResolveBranch(_) => "constraint_store",
            Op::AddTrigger(_) | Op::UpdateTrigger(_) | Op::FireTrigger(_) => "trigger_store",
            Op::RecordSurprise(_) => "surprise_store",
            Op::RegisterDebt(_) | Op::UpdateDebt(_) | Op::AttachDebtEvidence(_) => "epistemic_debt_store",
            Op::UpdateSourceWeight(_) | Op::RecordFeedback(_) => "integration_kernel",
            Op::UpdateSurpriseCredit(_) => "surprise_learning",
            Op::UpsertWisdomCandidate(_) | Op::UpdateWisdomLifecycle(_) => "wisdom_promotion",
            Op::UpdateScorerModel(_) => "learned_scorer",
            Op::StartIntervention(_) | Op::AddObservation(_) | Op::CloseIntervention(_) | Op::RecordAttribution(_) => "intervention_store",
            Op::RegisterTask(_) | Op::UpdateTask(_) | Op::AddDelegation(_) | Op::LinkEvidence(_) | Op::AddProbe(_) | Op::ResolveProbe(_) | Op::SetCriterion(_) => "agent_protocol_store",
            Op::UpsertWisdomLineage(_) | Op::AdjudicateLineage(_) | Op::TransitionLineage(_) | Op::RecordChallenger(_) | Op::CloseRederive(_) => "wisdom_lineage_store",
            Op::SymbolEvent(_) => "symbol_event_log",
            Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_) => "cortical_idx",
            _ => return false,
        };
        self.disabled(organ)
    }

    pub(crate) fn disabled(&self, name: &str) -> bool { self.0.contains(name) }
}

/// With ablation off this is an ordinary parking_lot lock. When ablated,
/// readers see an empty organ and writes have only guard-local scratch state.
/// Persistence and foreign WAL replay explicitly use the preserved state.
/// Keeping those accesses explicit prevents a maintenance save from discarding
/// a disabled organ's data and makes rollback independent of optional defaults.
pub(crate) struct Organ<T> {
    runtime: RwLock<T>,
    preserved: Option<RwLock<T>>,
    empty: fn() -> T,
}

impl<T> Organ<T> {
    pub(crate) fn new(value: T, empty: fn() -> T, disabled: bool) -> Self {
        if disabled {
            Self { runtime: RwLock::new(empty()), preserved: Some(RwLock::new(value)), empty }
        } else {
            Self { runtime: RwLock::new(value), preserved: None, empty }
        }
    }

    pub(crate) fn read(&self) -> RwLockReadGuard<'_, T> { self.runtime.read() }

    pub(crate) fn write(&self) -> OrganWrite<'_, T> {
        if self.preserved.is_some() {
            OrganWrite::Discard(Box::new((self.empty)()))
        } else {
            OrganWrite::Live(self.runtime.write())
        }
    }

    pub(crate) fn persisted_read(&self) -> RwLockReadGuard<'_, T> {
        self.preserved.as_ref().unwrap_or(&self.runtime).read()
    }

    pub(crate) fn replay_write(&self) -> RwLockWriteGuard<'_, T> {
        self.preserved.as_ref().unwrap_or(&self.runtime).write()
    }
}

pub(crate) enum OrganWrite<'a, T> {
    Live(RwLockWriteGuard<'a, T>),
    Discard(Box<T>),
}
impl<T> Deref for OrganWrite<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { match self { Self::Live(g) => g, Self::Discard(v) => v } }
}
impl<T> DerefMut for OrganWrite<'_, T> {
    fn deref_mut(&mut self) -> &mut T { match self { Self::Live(g) => g, Self::Discard(v) => v } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_validated_and_deduplicated() {
        let flags = Ablations::parse(" session_registry,session_registry, ,archive ").unwrap();
        assert!(flags.disabled("archive"));
        assert!(!flags.disabled("hdc_idx"));
        assert!(Ablations::parse("session_regsitry").is_err());
    }

    #[test]
    fn ablation_hides_state_but_preserves_rollback_and_foreign_replay() {
        let organ = Organ::new(vec![1], Vec::new, true);
        assert!(organ.read().is_empty());
        organ.write().push(2);
        assert!(organ.read().is_empty());
        assert_eq!(*organ.persisted_read(), vec![1]);
        organ.replay_write().push(3);
        assert_eq!(*organ.persisted_read(), vec![1, 3]);
        assert!(organ.read().is_empty());
    }

    #[test]
    fn control_preserves_normal_lock_semantics() {
        let organ = Organ::new(vec![1], Vec::new, false);
        organ.write().push(2);
        assert_eq!(*organ.read(), vec![1, 2]);
        assert_eq!(*organ.persisted_read(), vec![1, 2]);
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use crate::ChittaField;

    fn open(path: &std::path::Path, flags: &str) -> ChittaField {
        ChittaField::open_with_ablations(path.to_owned(), true, Ablations::parse(flags).unwrap()).unwrap()
    }

    #[test]
    fn disabled_snapshot_sections_survive_save_and_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let field = open(dir.path(), "");
        field.event_tape.write().log("kept", "rollback", 1, 7, 1000);
        field.decision_tape.write().log(1, 2, vec![(3, 0)], 0.5, 1000);
        field.observer_state.write().assert_fact(crate::organ::observer::ObservedFact {
            predicate_id: 0, object: "rollback".into(),
            polarity: crate::organ::observer::Polarity::Affirmed,
            time_scope: crate::organ::observer::TimeScope::Current,
            confidence: 1.0, source_id: 7, valid_from: 1000, valid_to: None,
        }, 1000);
        field.save_full_snapshot().unwrap();
        let original = crate::snapshot::FullSnapshot::load(&dir.path().join(format!("chitta.{:08x}.snapshot", field.instance_id))).unwrap();
        drop(field);
        let field = open(dir.path(), "event_tape,decision_tape,observer_state,turiya_monitor,interaction_ledger,predicate_store,msg_registry");
        assert!(field.event_tape.read().events.is_empty());
        assert!(field.decision_tape.read().points.is_empty());
        assert!(field.observer_state.read().current_facts().is_empty());
        field.event_tape.write().log("discarded", "rollback", 1, 7, 2000);
        field.decision_tape.write().log(9, 9, vec![], 0.0, 2000);
        field.save_full_snapshot().unwrap();
        let saved = crate::snapshot::FullSnapshot::load(&dir.path().join(format!("chitta.{:08x}.snapshot", field.instance_id))).unwrap();
        macro_rules! preserved {
            ($name:ident) => { assert_eq!(bincode::serialize(&original.$name).unwrap(), bincode::serialize(&saved.$name).unwrap(), stringify!($name)); };
        }
        preserved!(event_tape);
        preserved!(decision_tape);
        preserved!(observer_state);
        preserved!(turiya_monitor);
        preserved!(interaction_ledger);
        preserved!(predicate_store);
        preserved!(ledger_session_events);
        drop(field);
        let field = open(dir.path(), "");
        assert_eq!(field.decision_tape.read().points.len(), 1);
        assert_eq!(field.event_tape.read().events.len(), original.event_tape.events.len());
    }

    #[test]
    fn ablated_writes_do_not_resurrect_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        let field = open(dir.path(), "agent_protocol_store");
        let before = field.chain_head();
        assert_eq!(field.register_task("disabled".into(), vec![], vec![], "test".into(), "test".into(), 1, None, None, vec![]).unwrap(), 0);
        assert_eq!(field.chain_head(), before);
        drop(field);
        let field = open(dir.path(), "");
        assert!(field.query_tasks(None, None, None, None, 10).is_empty());
    }

    #[test]
    fn all_organ_ablations_preserve_core_writes_and_keyed_lanes() {
        let dir = tempfile::tempdir().unwrap();
        let flags = ORGANS.join(",");
        let field = open(dir.path(), &flags);
        let records = [
            ("signal", "[done] sha:abc123def456 input:/tmp/ablation-fixture verified"),
            ("wisdom", "[task] task:ablation-canary status:running next:verify"),
            ("correction", "[correction] USE: amber kettle\nNOT: zephyr marmalade orchard"),
        ];
        let mut ids = Vec::new();
        for (kind, text) in records {
            ids.push(field.put_memory(kind, "test", text.as_bytes(), &[], 1.0, 0.001, 1000, vec![], None, None).unwrap().0);
        }
        assert_eq!(field.provenance_lookup("abc123def456", "").unwrap().0, ids[0]);
        assert_eq!(field.task_state_lookup("ablation-canary").unwrap().0, ids[1]);
        assert!(field.correction_check("zephyr marmalade orchard").is_some());
        field.sync_wal().unwrap();
        drop(field);
        let field = open(dir.path(), &flags);
        for id in &ids { assert!(field.get_memory(*id).is_ok()); }
        assert_eq!(field.provenance_lookup("abc123def456", "").unwrap().0, ids[0]);
        assert_eq!(field.task_state_lookup("ablation-canary").unwrap().0, ids[1]);
        assert!(field.correction_check("zephyr marmalade orchard").is_some());
    }
}
