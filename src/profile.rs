//! Opt-in recall stage timing. Load timing is always emitted.
use std::time::Instant;

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
