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
            eprintln!("[lockprof] RUST component={} mode={} held_us={} wait_us={} acquisitions={} max_hold_us={} max_wait_us={}",
                self.component, mode, held_us, wait_us, n,
                self.max_hold.load(Ordering::Relaxed), self.max_wait.load(Ordering::Relaxed));
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
}
impl Drop for SnapshotPhase<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            eprintln!("[chitta-field] snapshot section={} decode_ms={}", self.0, start.elapsed().as_millis());
        }
    }
}
