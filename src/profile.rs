//! Opt-in recall stage timing. Load timing is always emitted.
use std::time::Instant;

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};

/// Always-on component timing, including uncontended acquisitions. Logging is
/// outside the protected section; first/new maxima make quiet logs meaningful.
struct LockMetrics {
    component: &'static str,
    acquisitions: AtomicU64,
    max_wait: AtomicU64,
    max_hold: AtomicU64,
}
impl LockMetrics {
    fn new(component: &'static str) -> Self {
        Self { component, acquisitions: AtomicU64::new(0),
            max_wait: AtomicU64::new(0), max_hold: AtomicU64::new(0) }
    }
    fn record(&self, mode: &str, wait_us: u64, held_us: u64) {
        let n = self.acquisitions.fetch_add(1, Ordering::Relaxed) + 1;
        let old_wait = self.max_wait.fetch_max(wait_us, Ordering::Relaxed);
        let old_hold = self.max_hold.fetch_max(held_us, Ordering::Relaxed);
        if n == 1 || wait_us > old_wait || held_us > old_hold
            || wait_us > 50_000 || held_us > 50_000 || n % 4096 == 0 {
            let line = format!("[lockprof] RUST component={} mode={} held_us={} wait_us={} acquisitions={} max_hold_us={} max_wait_us={}\n",
                self.component, mode, held_us, wait_us, n,
                self.max_hold.load(Ordering::Relaxed), self.max_wait.load(Ordering::Relaxed));
            // stderr is shared with C++/llama writers that do not take Rust's
            // stdio lock. One short write keeps a record intact through the
            // daemon's timestamp pipe (well below PIPE_BUF).
            let _ = std::io::Write::write_all(&mut std::io::stderr().lock(), line.as_bytes());
        }
    }
}

pub(crate) struct ProfiledGuard<'a, G> {
    guard: Option<G>,
    metrics: &'a LockMetrics,
    mode: &'static str,
    acquired: Instant,
    wait_us: u64,
}
impl<'a, G> ProfiledGuard<'a, G> {
    fn new(guard: G, metrics: &'a LockMetrics, mode: &'static str, started: Instant) -> Self {
        let acquired = Instant::now();
        Self { guard: Some(guard), metrics, mode, acquired,
            wait_us: acquired.duration_since(started).as_micros() as u64 }
    }
}
impl<G: Deref> Deref for ProfiledGuard<'_, G> {
    type Target = G::Target;
    fn deref(&self) -> &Self::Target { self.guard.as_ref().unwrap().deref() }
}
impl<G: DerefMut> DerefMut for ProfiledGuard<'_, G> {
    fn deref_mut(&mut self) -> &mut Self::Target { self.guard.as_mut().unwrap().deref_mut() }
}
impl<G> Drop for ProfiledGuard<'_, G> {
    fn drop(&mut self) {
        drop(self.guard.take());
        let held_us = self.acquired.elapsed().as_micros() as u64;
        self.metrics.record(self.mode, self.wait_us, held_us);
    }
}

pub(crate) struct ProfiledRwLock<T> {
    inner: parking_lot::RwLock<T>,
    metrics: LockMetrics,
}
impl<T> ProfiledRwLock<T> {
    pub(crate) fn new(component: &'static str, value: T) -> Self {
        Self { inner: parking_lot::RwLock::new(value), metrics: LockMetrics::new(component) }
    }
    pub(crate) fn read(&self) -> ProfiledGuard<'_, parking_lot::RwLockReadGuard<'_, T>> {
        let started = Instant::now();
        ProfiledGuard::new(self.inner.read(), &self.metrics, "read", started)
    }
    pub(crate) fn write(&self) -> ProfiledGuard<'_, parking_lot::RwLockWriteGuard<'_, T>> {
        let started = Instant::now();
        ProfiledGuard::new(self.inner.write(), &self.metrics, "write", started)
    }
    // Startup builds under shared ownership, then upgrades only for publication.
    // Return the native guard so timed upgrades preserve that ownership.
    pub(crate) fn try_upgradable_read_for(&self, timeout: std::time::Duration)
        -> Option<parking_lot::RwLockUpgradableReadGuard<'_, T>> {
        let started = Instant::now();
        let guard = self.inner.try_upgradable_read_for(timeout);
        self.metrics.record(if guard.is_some() { "upgradable_read" } else { "read_timeout" },
            started.elapsed().as_micros() as u64, 0);
        guard
    }

    pub(crate) fn try_read_for(&self, timeout: std::time::Duration)
        -> Option<ProfiledGuard<'_, parking_lot::RwLockReadGuard<'_, T>>> {
        let started = Instant::now();
        match self.inner.try_read_for(timeout) {
            Some(guard) => Some(ProfiledGuard::new(guard, &self.metrics, "read", started)),
            None => {
                self.metrics.record("read_timeout", started.elapsed().as_micros() as u64, 0);
                None
            }
        }
    }
}

/// The archive deliberately retains std's poisoning semantics.
pub(crate) struct ProfiledStdRwLock<T> {
    inner: std::sync::RwLock<T>,
    metrics: LockMetrics,
}
impl<T> ProfiledStdRwLock<T> {
    pub(crate) fn new(component: &'static str, value: T) -> Self {
        Self { inner: std::sync::RwLock::new(value), metrics: LockMetrics::new(component) }
    }
    pub(crate) fn write(&self) -> std::sync::LockResult<ProfiledGuard<'_, std::sync::RwLockWriteGuard<'_, T>>> {
        let started = Instant::now();
        self.inner.write()
            .map(|guard| ProfiledGuard::new(guard, &self.metrics, "write", started))
            .map_err(|error| std::sync::PoisonError::new(ProfiledGuard::new(
                error.into_inner(), &self.metrics, "write", started)))
    }
}

#[cfg(test)]
mod lock_tests {
    use super::*;
    #[test]
    fn guards_count_reads_writes_and_contended_waits() {
        let lock = ProfiledRwLock::new("test", 0);
        std::thread::scope(|scope| {
            let mut writer = lock.write();
            *writer = 7;
            let (tx, rx) = std::sync::mpsc::channel();
            let reader_lock = &lock;
            let reader = scope.spawn(move || {
                tx.send(()).unwrap();
                assert_eq!(*reader_lock.read(), 7);
            });
            rx.recv().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(60));
            drop(writer);
            reader.join().unwrap();
        });
        assert_eq!(lock.metrics.acquisitions.load(Ordering::Relaxed), 2);
        assert!(lock.metrics.max_hold.load(Ordering::Relaxed) >= 50_000);
        assert!(lock.metrics.max_wait.load(Ordering::Relaxed) > 0);
    }
    #[test]
    fn unwinding_unlocks_and_preserves_archive_poison() {
        let lock = ProfiledStdRwLock::new("archive_test", 0);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut value = lock.write().unwrap();
            *value = 9;
            panic!("test unwind");
        }));
        assert_eq!(*lock.write().err().unwrap().into_inner(), 9);
        assert_eq!(lock.metrics.acquisitions.load(Ordering::Relaxed), 2);
    }
}

pub(crate) struct RecallProfile {
    stage: &'static str,
    start: Option<Instant>,
}
impl RecallProfile {
    pub(crate) fn new(stage: &'static str) -> Self {
        Self { stage, start: enabled().then(Instant::now) }
    }
    pub(crate) fn next(&mut self, stage: &'static str) {
        self.emit();
        self.stage = stage;
        if self.start.is_some() { self.start = Some(Instant::now()); }
    }
    fn emit(&self) {
        if let Some(start) = self.start {
            eprintln!("[chitta-field] recall stage={} us={} thread={:?}", self.stage,
                start.elapsed().as_micros(), std::thread::current().id());
        }
    }
}
impl Drop for RecallProfile { fn drop(&mut self) { self.emit(); } }

pub(crate) struct LoadPhase(pub(crate) &'static str, Instant);
impl LoadPhase {
    pub(crate) fn new(name: &'static str) -> Self { Self(name, Instant::now()) }
}
impl Drop for LoadPhase {
    fn drop(&mut self) {
        eprintln!("[chitta-field] load phase={} ms={}", self.0, self.1.elapsed().as_millis());
    }
}

/// Capacity-based allocation estimate; excludes allocator headers and fragmentation.
pub(crate) fn map_bytes<K, V>(map: &std::collections::HashMap<K, V>) -> usize {
    // hashbrown capacity is usable slots (7/8 of buckets); one control byte per bucket.
    (map.capacity() * 8 / 7) * (std::mem::size_of::<(K, V)>() + 1)
}

impl crate::field::ChittaField {
    /// Each guard is acquired alone. This diagnostic never serializes or clones organs.
    pub fn memory_breakdown(&self) -> String {
        let payloads = {
            let p = self.payloads.read();
            map_bytes(&p) + p.values().map(|v| v.kind.capacity() + v.realm.capacity()
                + v.content.capacity() + v.embedding_model.capacity() + v.embedding.capacity() * 4
                + v.embedding_model_id.capacity() + v.provenance.capacity()
                + v.source_session.as_ref().map_or(0, String::capacity)
                + v.source_tool.as_ref().map_or(0, String::capacity)
                + v.harness.as_ref().map_or(0, String::capacity)
                + v.artifact_refs.capacity() * std::mem::size_of::<crate::ops::ArtifactRef>()).sum::<usize>()
        };
        let states = map_bytes(&self.states.read());
        let (embeddings, hnsw, semantic_caches) = self.semantic_idx.read().allocated_bytes();
        let hdc = self.hdc_idx.read().allocated_bytes();
        let triplets = self.triplet_store.read().allocated_bytes();
        let (keyword_postings, keyword_reverse, keyword_reverse_legacy_estimate) = self.keyword_idx.read().allocated_bytes();
        let (episode_hdc, episode_hdc_legacy_estimate) = self.episode_hdc.read().allocated_bytes();
        let spans = self.span_store.read().allocated_bytes();
        let cdawg = {
            let g = self.cdawg.read();
            g.states.capacity() * std::mem::size_of::<crate::organ::cdawg::CdawgState>()
                + g.states.iter().map(|s| map_bytes(&s.transitions)
                    + s.endpos.serialized_size() + s.sl_children.capacity() * 4).sum::<usize>()
        };
        let lite_encoder = self.lite_encoder.read().as_ref().map_or(0, |e|
            map_bytes(&e.vocab) + e.vocab.keys().map(String::capacity).sum::<usize>()
            + e.word_left.iter().chain(e.word_right.iter()).map(|v| v.capacity() * 4).sum::<usize>());
        let rss_bytes = std::fs::read_to_string("/proc/self/status").ok().and_then(|text|
            text.lines().find_map(|line| line.strip_prefix("VmRSS:")
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|kb| kb.parse::<usize>().ok()).map(|kb| kb * 1024)));
        let allocation_estimate_total = payloads + states + embeddings + hnsw + hdc
            + triplets + spans + cdawg + semantic_caches + lite_encoder + episode_hdc
            + keyword_postings + keyword_reverse;
        serde_json::json!({"units":"bytes", "basis":"capacity_estimate_not_rss",
            "rss_bytes":rss_bytes, "allocation_estimate_total":allocation_estimate_total,
            "payloads":payloads, "states":states, "embeddings":embeddings, "hnsw":hnsw,
            "keyword_postings":keyword_postings, "keyword_reverse":keyword_reverse, "keyword_reverse_legacy_estimate":keyword_reverse_legacy_estimate,
            "hdc":hdc, "episode_hdc":episode_hdc, "episode_hdc_legacy_estimate":episode_hdc_legacy_estimate, "triplets":triplets, "spans":spans, "cdawg":cdawg,
            "semantic_caches":semantic_caches, "lite_encoder":lite_encoder,
            "unaccounted":"turbo internals, other organs, model, threads, allocator fragmentation"}).to_string()
    }
}

pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CHITTA_RECALL_PROFILE").as_deref() == Ok("1"))
}

/// Detailed snapshot timings are opt-in; phase names also identify decode jobs.
pub(crate) struct SnapshotPhase<'a>(&'a str, Option<Instant>);
impl<'a> SnapshotPhase<'a> {
    pub(crate) fn new(name: &'a str) -> Self {
        Self(name, std::env::var_os("CHITTA_PROFILE_SNAPSHOT").is_some().then(Instant::now))
    }
    /// Time this phase whether or not CHITTA_PROFILE_SNAPSHOT is set. For the
    /// handful of phases that must be readable from a production log: on
    /// 2026-09-20 `load phase=normalize ms=41017` had no breakdown under it,
    /// because every sub-timer in normalize_with_cache was opt-in and the
    /// daemon does not set that variable.
    pub(crate) fn always(name: &'a str) -> Self { Self(name, Some(Instant::now())) }
}
impl Drop for SnapshotPhase<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            eprintln!("[chitta-field] snapshot section={} decode_ms={}", self.0, start.elapsed().as_millis());
        }
    }
}

/// Opt-in replay application timing. Coverage skips and decoding are excluded.
/// Deferred orphan retries are reported separately from their source records.
pub(crate) struct ReplayApplyProfile {
    kinds: Option<std::collections::BTreeMap<&'static str, (u64, u128)>>,
}
impl ReplayApplyProfile {
    pub(crate) fn new() -> Self {
        Self { kinds: (std::env::var("CHITTA_PROFILE_REPLAY").as_deref() == Ok("1"))
            .then(std::collections::BTreeMap::new) }
    }
    pub(crate) fn begin(&mut self, op: &crate::ops::Op) -> Option<ReplayApplyTimer<'_>> {
        self.kinds.as_ref()?;
        let kind = replay_kind(op);
        self.named(kind)
    }
    pub(crate) fn named(&mut self, kind: &'static str) -> Option<ReplayApplyTimer<'_>> {
        let totals = self.kinds.as_mut()?.entry(kind).or_default();
        Some(ReplayApplyTimer { totals, started: Instant::now() })
    }
}
pub(crate) struct ReplayApplyTimer<'a> {
    totals: &'a mut (u64, u128),
    started: Instant,
}
impl Drop for ReplayApplyTimer<'_> {
    fn drop(&mut self) {
        self.totals.0 += 1;
        self.totals.1 += self.started.elapsed().as_nanos();
    }
}
impl Drop for ReplayApplyProfile {
    fn drop(&mut self) {
        if let Some(kinds) = &self.kinds {
            for (kind, (records, apply_ns)) in kinds {
                eprintln!("[chitta-field] replay_apply kind={} records={} apply_ns={}", kind, records, apply_ns);
            }
        }
    }
}

fn replay_kind(op: &crate::ops::Op) -> &'static str {
    match op {
            crate::ops::Op::PutPayload(..) => "PutPayload",
            crate::ops::Op::UpdateState(..) => "UpdateState",
            crate::ops::Op::UpdateStateBatch(..) => "UpdateStateBatch",
            crate::ops::Op::DeleteMemory(..) => "DeleteMemory",
            crate::ops::Op::AddAssocEdge(..) => "AddAssocEdge",
            crate::ops::Op::UpsertArtifact(..) => "UpsertArtifact",
            crate::ops::Op::AddTriplet(..) => "AddTriplet",
            crate::ops::Op::InvalidateTriplet(..) => "InvalidateTriplet",
            crate::ops::Op::UpsertSymbol(..) => "UpsertSymbol",
            crate::ops::Op::RemoveSymbol(..) => "RemoveSymbol",
            crate::ops::Op::AddSymCallEdge(..) => "AddSymCallEdge",
            crate::ops::Op::RemoveSymCallEdge(..) => "RemoveSymCallEdge",
            crate::ops::Op::UpsertCodeFile(..) => "UpsertCodeFile",
            crate::ops::Op::UpdateSparseCode(..) => "UpdateSparseCode",
            crate::ops::Op::DemoteMemory(..) => "DemoteMemory",
            crate::ops::Op::TrainPQ(..) => "TrainPQ",
            crate::ops::Op::UpdateResidualPQ(..) => "UpdateResidualPQ",
            crate::ops::Op::SessionEvent(..) => "SessionEvent",
            crate::ops::Op::TranscriptEvent(..) => "TranscriptEvent",
            crate::ops::Op::TaskEvent(..) => "TaskEvent",
            crate::ops::Op::UserModelEvent(..) => "UserModelEvent",
            crate::ops::Op::ThemeEvent(..) => "ThemeEvent",
            crate::ops::Op::AnalyticsEvent(..) => "AnalyticsEvent",
            crate::ops::Op::ClearProject(..) => "ClearProject",
            crate::ops::Op::UpdateSymbolDescription(..) => "UpdateSymbolDescription",
            crate::ops::Op::UpdateMemoryContent(..) => "UpdateMemoryContent",
            crate::ops::Op::RecordRecallBatch(..) => "RecordRecallBatch",
            crate::ops::Op::StrengthenAssocEdge(..) => "StrengthenAssocEdge",
            crate::ops::Op::MsgEvent(..) => "MsgEvent",
            crate::ops::Op::SkillUpload(..) => "SkillUpload",
            crate::ops::Op::SkillDeprecate(..) => "SkillDeprecate",
            crate::ops::Op::AgentUpsert(..) => "AgentUpsert",
            crate::ops::Op::AgentDisable(..) => "AgentDisable",
            crate::ops::Op::AssertConstraint(..) => "AssertConstraint",
            crate::ops::Op::RetractConstraint(..) => "RetractConstraint",
            crate::ops::Op::CreateBranch(..) => "CreateBranch",
            crate::ops::Op::ResolveBranch(..) => "ResolveBranch",
            crate::ops::Op::AddTrigger(..) => "AddTrigger",
            crate::ops::Op::UpdateTrigger(..) => "UpdateTrigger",
            crate::ops::Op::FireTrigger(..) => "FireTrigger",
            crate::ops::Op::RecordSurprise(..) => "RecordSurprise",
            crate::ops::Op::RegisterDebt(..) => "RegisterDebt",
            crate::ops::Op::UpdateDebt(..) => "UpdateDebt",
            crate::ops::Op::UpdateSourceWeight(..) => "UpdateSourceWeight",
            crate::ops::Op::RecordFeedback(..) => "RecordFeedback",
            crate::ops::Op::UpdateSurpriseCredit(..) => "UpdateSurpriseCredit",
            crate::ops::Op::UpsertWisdomCandidate(..) => "UpsertWisdomCandidate",
            crate::ops::Op::UpdateWisdomLifecycle(..) => "UpdateWisdomLifecycle",
            crate::ops::Op::UpdateScorerModel(..) => "UpdateScorerModel",
            crate::ops::Op::AttachDebtEvidence(..) => "AttachDebtEvidence",
            crate::ops::Op::StartIntervention(..) => "StartIntervention",
            crate::ops::Op::AddObservation(..) => "AddObservation",
            crate::ops::Op::CloseIntervention(..) => "CloseIntervention",
            crate::ops::Op::RecordAttribution(..) => "RecordAttribution",
            crate::ops::Op::RegisterTask(..) => "RegisterTask",
            crate::ops::Op::UpdateTask(..) => "UpdateTask",
            crate::ops::Op::AddDelegation(..) => "AddDelegation",
            crate::ops::Op::LinkEvidence(..) => "LinkEvidence",
            crate::ops::Op::AddProbe(..) => "AddProbe",
            crate::ops::Op::ResolveProbe(..) => "ResolveProbe",
            crate::ops::Op::SetCriterion(..) => "SetCriterion",
            crate::ops::Op::UpsertWisdomLineage(..) => "UpsertWisdomLineage",
            crate::ops::Op::AdjudicateLineage(..) => "AdjudicateLineage",
            crate::ops::Op::TransitionLineage(..) => "TransitionLineage",
            crate::ops::Op::RecordChallenger(..) => "RecordChallenger",
            crate::ops::Op::CloseRederive(..) => "CloseRederive",
            crate::ops::Op::InvalidateTripletsBySourceFile(..) => "InvalidateTripletsBySourceFile",
            crate::ops::Op::UpdateMemoryKind(..) => "UpdateMemoryKind",
            crate::ops::Op::SymbolEvent(..) => "SymbolEvent",
            crate::ops::Op::SupersedeTriplet(..) => "SupersedeTriplet",
            crate::ops::Op::RecordOutcome(..) => "RecordOutcome",
        }
}

/// Full reader/merge costs, including covered records (unlike replay_apply).
#[derive(Default)]
struct ReplayCosts { records: u64, decode_ns: u128, verify_ns: u128, apply_ns: u128, other_ns: u128 }
pub(crate) struct ReplayLoopProfile {
    kinds: Option<std::collections::BTreeMap<&'static str, ReplayCosts>>,
}
impl ReplayLoopProfile {
    pub(crate) fn new() -> Self {
        Self { kinds: (std::env::var("CHITTA_PROFILE_REPLAY").as_deref() == Ok("1"))
            .then(std::collections::BTreeMap::new) }
    }
    pub(crate) fn start(&self) -> Option<Instant> { self.kinds.as_ref().map(|_| Instant::now()) }
    pub(crate) fn kind(&self, op: &crate::ops::Op) -> &'static str { replay_kind(op) }
    pub(crate) fn record(&mut self, kind: &'static str, total: u128, decode: u128, verify: u128) {
        if let Some(kinds) = self.kinds.as_mut() {
            let row = kinds.entry(kind).or_default();
            row.records += 1; row.decode_ns += decode; row.verify_ns += verify;
            row.other_ns += total.saturating_sub(decode + verify);
        }
    }
    pub(crate) fn applied(&mut self, kind: &'static str, ns: u128) {
        if let Some(kinds) = self.kinds.as_mut() { kinds.entry(kind).or_default().apply_ns += ns; }
    }
}
impl Drop for ReplayLoopProfile {
    fn drop(&mut self) {
        if let Some(kinds) = &self.kinds {
            for (kind, r) in kinds {
                eprintln!("[chitta-field] replay_loop kind={} records={} decode_ns={} verify_ns={} apply_ns={} other_ns={}",
                    kind, r.records, r.decode_ns, r.verify_ns, r.apply_ns, r.other_ns);
            }
        }
    }
}
