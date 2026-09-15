//! Span Lane — verbatim high-value atoms extracted from raw transcripts.
//!
//! Complements distill (which paraphrases and belief-gates specifics away): this
//! organ keeps the exact bytes — paths, ids, file:line, commands, errors — so
//! they are retrievable verbatim with NO LLM and NO GPU at query time.
//!
//! Decay-immunity is by construction: the query path is either a full linear
//! scan (no persistent index at all) or a trigram accelerator rebuilt lazily
//! when the `mutations` counter advances (the `ensure_turbo` pattern). There is
//! no frozen state that rots with uptime.
//!
//! Space: globally-deduped atoms + ≤4 locators/atom + a rebuilt-on-demand
//! trigram map. ~100k unique atoms ≈ a few MB, versus ≥0.9 GB to embed every
//! transcript chunk.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const SIDECAR: &str = "spans.bin";

// Span classes (u8, stable on disk).
pub const CLASS_PATH: u8 = 0;
pub const CLASS_URL: u8 = 1;
pub const CLASS_UUID: u8 = 2;
pub const CLASS_HEX: u8 = 3;
pub const CLASS_ISSUE: u8 = 4;
pub const CLASS_FILELINE: u8 = 5;
pub const CLASS_BASH: u8 = 6;
pub const CLASS_ERROR: u8 = 7;

const MAX_LOCATORS: usize = 4;
const PER_LINE_CAP: usize = 64; // anti-pathological: cap spans harvested per JSONL line
const MAX_LINE_BYTES: usize = 200_000; // scan at most this many bytes of a line
const CLAMP_TEXT: usize = 512; // paths/urls/errors
const CLAMP_BASH: usize = 300; // command strings
const TRIGRAM_MIN: usize = 200_000; // below this, full scan is already sub-10ms

fn class_weight(c: u8) -> f32 {
    match c {
        CLASS_FILELINE => 1.4,
        CLASS_PATH => 1.3,
        CLASS_ERROR => 1.3,
        CLASS_BASH => 1.1,
        CLASS_URL => 1.1,
        CLASS_UUID => 1.0,
        CLASS_ISSUE => 1.0,
        CLASS_HEX => 0.8,
        _ => 1.0,
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Locator {
    session_idx: u32,
    line: u32,
    realm_id: u16,
    ts_ms: i64,
}

#[derive(Clone, Serialize, Deserialize)]
struct RealmOcc {
    realm_id: u16,
    count: u32,
}

#[derive(Clone, Serialize, Deserialize)]
struct SpanEntry {
    text: String,
    class: u8,
    count: u32,
    first_ms: i64,
    last_ms: i64,
    realms: SmallVec<[RealmOcc; 2]>,
    locators: SmallVec<[Locator; MAX_LOCATORS]>,
    /// memory_ids whose distilled TEXT contains this atom (the memory↔span edge).
    /// A span with empty locators AND empty mem_refs is dead → GC-eligible.
    #[serde(default)]
    mem_refs: SmallVec<[u64; 2]>,
}

impl SpanEntry {
    /// A span is alive while any reference (transcript locator or memory link)
    /// still points at it. Tombstones (text cleared on GC) are also dead.
    fn is_dead(&self) -> bool {
        self.text.is_empty() || (self.locators.is_empty() && self.mem_refs.is_empty())
    }
}

/// One retrieval result. Verbatim `text` is complete on its own; the locator
/// (`session`, `line`) lets the caller page context via read_transcript, but a
/// dead locator never invalidates the atom.
#[derive(Debug, Clone)]
pub struct SpanHit {
    pub text: String,
    pub class: u8,
    pub count: u32,
    pub last_ms: i64,
    pub realm: String,
    pub session: String,
    pub line: u32,
    pub score: f32,
    /// memory_ids whose text references this atom — the reverse edge, lets a
    /// matched span jump to the distilled beliefs that mention it.
    pub memory_ids: Vec<u64>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct IngestStats {
    pub lines: u64,
    pub raw_spans: u64,
    pub new_spans: u64,
    pub redacted: u64,
    pub skipped_injection: u64,
}

impl std::ops::AddAssign for IngestStats {
    fn add_assign(&mut self, o: Self) {
        self.lines += o.lines;
        self.raw_spans += o.raw_spans;
        self.new_spans += o.new_spans;
        self.redacted += o.redacted;
        self.skipped_injection += o.skipped_injection;
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct SpanStore {
    spans: Vec<SpanEntry>,
    realm_dict: Vec<String>,
    session_dict: Vec<String>,
    /// path -> (bytes consumed, lines consumed) — incremental watermark.
    watermarks: HashMap<String, (u64, u32)>,
    mutations: u64,
    /// memory_id -> fnv1a(content) at last ingest — idempotent memory backfill.
    /// A changed hash means the memory was re-distilled/superseded → relink.
    #[serde(default)]
    mem_watermarks: HashMap<u64, u64>,

    // Rebuilt on load / never serialized.
    #[serde(skip)]
    by_hash: HashMap<u64, u32>,
    /// memory_id -> span indices it links to (forward edge). Derived from each
    /// span's mem_refs; rebuilt in reindex(), never persisted.
    #[serde(skip)]
    mem_adjacency: HashMap<u64, SmallVec<[u32; 4]>>,
    #[serde(skip)]
    realm_idx: HashMap<String, u16>,
    #[serde(skip)]
    session_idx: HashMap<String, u32>,
    #[serde(skip)]
    trigram: HashMap<u32, roaring::RoaringBitmap>,
    #[serde(skip)]
    trigram_built_at: u64,
    /// Number of leading spans whose trigrams are in `trigram`; spans only append
    /// (tombstones keep their index), so the index extends incrementally.
    #[serde(skip)]
    trigram_indexed: usize,
    #[serde(skip)]
    redacted_total: u64,
    #[serde(skip)]
    path: PathBuf,
    /// Unsaved in-RAM changes (live-path ingest defers the ~50MB serialize off
    /// the memory-write hot path; the daemon's periodic span flush persists).
    #[serde(skip)]
    dirty: bool,
}

fn fnv1a(class: u8, s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    h = (h ^ class as u64).wrapping_mul(0x100000001b3);
    for b in s.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

impl SpanStore {
    pub(crate) fn allocated_bytes(&self) -> usize {
        use crate::profile::map_bytes;
        self.spans.capacity() * std::mem::size_of::<SpanEntry>()
            + self.spans.iter().map(|s| s.text.capacity()
                + if s.mem_refs.spilled() { s.mem_refs.capacity() * 8 } else { 0 }).sum::<usize>()
            + self.realm_dict.iter().chain(self.session_dict.iter()).map(|s| std::mem::size_of::<String>() + s.capacity()).sum::<usize>()
            + map_bytes(&self.by_hash) + map_bytes(&self.mem_watermarks)
            + map_bytes(&self.mem_adjacency) + map_bytes(&self.watermarks)
            + map_bytes(&self.realm_idx) + map_bytes(&self.session_idx)
            + self.trigram.values().map(|b| b.serialized_size()).sum::<usize>()
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join(SIDECAR);
        let mut store: Self = std::fs::read(&path)
            .ok()
            .and_then(|b| bincode::deserialize(&b).ok())
            .unwrap_or_default();
        store.path = path;
        store.reindex();
        store
    }

    /// Rebuild all serde-skipped lookup tables from the persisted vectors.
    fn reindex(&mut self) {
        // Positions changed: the incremental trigram index is void.
        self.trigram.clear();
        self.trigram_indexed = 0;
        self.by_hash.clear();
        self.mem_adjacency.clear();
        for (i, s) in self.spans.iter().enumerate() {
            if s.text.is_empty() {
                continue; // tombstone — not indexed, not queryable
            }
            self.by_hash.insert(fnv1a(s.class, &s.text), i as u32);
            for &mid in &s.mem_refs {
                self.mem_adjacency.entry(mid).or_default().push(i as u32);
            }
        }
        self.realm_idx.clear();
        for (i, r) in self.realm_dict.iter().enumerate() {
            self.realm_idx.insert(r.clone(), i as u16);
        }
        self.session_idx.clear();
        for (i, s) in self.session_dict.iter().enumerate() {
            self.session_idx.insert(s.clone(), i as u32);
        }
    }

    /// Physically drop tombstoned spans and reindex. O(n); only called from
    /// save() when the dead fraction is high — never on the per-prune hot path.
    fn maybe_compact(&mut self) {
        let dead = self.spans.iter().filter(|s| s.text.is_empty()).count();
        if dead == 0 || dead * 4 <= self.spans.len() {
            return;
        }
        self.spans.retain(|s| !s.text.is_empty());
        self.mutations = self.mutations.wrapping_add(1);
        self.reindex();
    }

    pub fn save(&mut self) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        self.maybe_compact();
        if let Ok(bytes) = bincode::serialize(self) {
            let tmp = self.path.with_extension("bin.tmp");
            if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &self.path).is_ok() {
                self.dirty = false;
            }
        }
    }

    /// Persist only if there are unsaved changes. Returns true iff a save ran.
    pub fn save_if_dirty(&mut self) -> bool {
        if !self.dirty {
            return false;
        }
        self.save();
        !self.dirty
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
    pub fn redacted_total(&self) -> u64 {
        self.redacted_total
    }
    pub fn on_disk_bytes(&self) -> u64 {
        std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0)
    }

    /// Per-class diagnostics: (class, total, singletons(count==1), text_bytes).
    pub fn histogram(&self) -> Vec<(u8, u64, u64, u64)> {
        let mut h: HashMap<u8, (u64, u64, u64)> = HashMap::new();
        for s in &self.spans {
            let e = h.entry(s.class).or_default();
            e.0 += 1;
            if s.count == 1 {
                e.1 += 1;
            }
            e.2 += s.text.len() as u64;
        }
        let mut v: Vec<(u8, u64, u64, u64)> =
            h.into_iter().map(|(c, (a, b, d))| (c, a, b, d)).collect();
        v.sort_by_key(|x| std::cmp::Reverse(x.1));
        v
    }

    fn intern_realm(&mut self, realm: &str) -> u16 {
        if let Some(&id) = self.realm_idx.get(realm) {
            return id;
        }
        let id = self.realm_dict.len() as u16;
        self.realm_dict.push(realm.to_string());
        self.realm_idx.insert(realm.to_string(), id);
        id
    }

    fn intern_session(&mut self, session: &str) -> u32 {
        if let Some(&id) = self.session_idx.get(session) {
            return id;
        }
        let id = self.session_dict.len() as u32;
        self.session_dict.push(session.to_string());
        self.session_idx.insert(session.to_string(), id);
        id
    }

    /// Insert one (already redacted+clamped) atom. Global dedup by (class,text).
    fn insert_span(
        &mut self,
        class: u8,
        text: String,
        realm_id: u16,
        session_idx: u32,
        line: u32,
        ts_ms: i64,
    ) -> bool {
        let h = fnv1a(class, &text);
        if let Some(&idx) = self.by_hash.get(&h) {
            let i = idx as usize;
            if self.spans[i].class == class && self.spans[i].text == text {
                let e = &mut self.spans[i];
                e.count = e.count.saturating_add(1);
                if ts_ms < e.first_ms {
                    e.first_ms = ts_ms;
                }
                if ts_ms > e.last_ms {
                    e.last_ms = ts_ms;
                }
                // realm occurrence
                match e.realms.iter_mut().find(|r| r.realm_id == realm_id) {
                    Some(r) => r.count = r.count.saturating_add(1),
                    None => e.realms.push(RealmOcc { realm_id, count: 1 }),
                }
                add_locator(&mut e.locators, session_idx, line, realm_id, ts_ms);
                return false;
            }
            // 64-bit hash collision with a genuinely different atom (P~1e-10 at
            // 1e5 atoms): fall through and store separately; the map keeps the
            // first — the collider just won't dedup. Correctness preserved.
        }
        let idx = self.spans.len() as u32;
        let mut locators: SmallVec<[Locator; MAX_LOCATORS]> = SmallVec::new();
        locators.push(Locator { session_idx, line, realm_id, ts_ms });
        let mut realms: SmallVec<[RealmOcc; 2]> = SmallVec::new();
        realms.push(RealmOcc { realm_id, count: 1 });
        self.spans.push(SpanEntry {
            text,
            class,
            count: 1,
            first_ms: ts_ms,
            last_ms: ts_ms,
            realms,
            locators,
            mem_refs: SmallVec::new(),
        });
        self.by_hash.insert(h, idx);
        true
    }

    /// Insert one (already redacted+clamped) atom sourced from a MEMORY's text
    /// rather than a transcript. Same global dedup by (class,text): if the atom
    /// already exists (from a transcript or another memory), we only attach the
    /// memory_id edge — no transcript locator is created. Returns true iff a new
    /// span row was allocated.
    fn insert_span_mem(&mut self, class: u8, text: String, realm_id: u16, memory_id: u64) -> bool {
        let h = fnv1a(class, &text);
        if let Some(&idx) = self.by_hash.get(&h) {
            let i = idx as usize;
            if self.spans[i].class == class && self.spans[i].text == text {
                if !self.spans[i].mem_refs.contains(&memory_id) {
                    self.spans[i].mem_refs.push(memory_id);
                    self.mem_adjacency.entry(memory_id).or_default().push(i as u32);
                }
                match self.spans[i].realms.iter_mut().find(|r| r.realm_id == realm_id) {
                    Some(r) => r.count = r.count.saturating_add(1),
                    None => self.spans[i].realms.push(RealmOcc { realm_id, count: 1 }),
                }
                return false;
            }
        }
        let idx = self.spans.len() as u32;
        let mut realms: SmallVec<[RealmOcc; 2]> = SmallVec::new();
        realms.push(RealmOcc { realm_id, count: 1 });
        let mut mem_refs: SmallVec<[u64; 2]> = SmallVec::new();
        mem_refs.push(memory_id);
        self.spans.push(SpanEntry {
            text,
            class,
            count: 1,
            first_ms: 0,
            last_ms: 0,
            realms,
            locators: SmallVec::new(),
            mem_refs,
        });
        self.by_hash.insert(h, idx);
        self.mem_adjacency.entry(memory_id).or_default().push(idx);
        true
    }

    /// Link one memory's distilled text into the span store (the memory→span
    /// edge). Idempotent via a per-memory content hash: unchanged text is a
    /// no-op; changed text (re-distill / supersede) unlinks the stale atoms
    /// first, then relinks. `realm` is the memory's realm string directly.
    pub fn ingest_memory(&mut self, memory_id: u64, text: &str, realm: &str) -> IngestStats {
        let mut stats = IngestStats::default();
        let content_hash = fnv1a(0, text);
        match self.mem_watermarks.get(&memory_id) {
            Some(&h) if h == content_hash => return stats, // unchanged → idempotent skip
            Some(_) => {
                self.unlink_memory(memory_id); // superseded → drop stale edges first
            }
            None => {}
        }
        let realm_id = self.intern_realm(realm);
        for (class, raw) in extract_atoms(text) {
            stats.raw_spans += 1;
            let clamp = if class == CLASS_BASH { CLAMP_BASH } else { CLAMP_TEXT };
            let clamped = clamp_utf8(&raw, clamp);
            match redact(&clamped) {
                Some((red, was_red)) => {
                    if was_red {
                        stats.redacted += 1;
                        self.redacted_total += 1;
                    }
                    if self.insert_span_mem(class, red, realm_id, memory_id) {
                        stats.new_spans += 1;
                    }
                }
                None => {
                    stats.redacted += 1;
                    self.redacted_total += 1;
                }
            }
        }
        self.mem_watermarks.insert(memory_id, content_hash);
        self.mutations = self.mutations.wrapping_add(stats.new_spans.max(1));
        self.dirty = true;
        stats
    }

    /// Remove all edges from a memory (forget / prune / supersede). For each
    /// linked span, drop this memory_id; if the span then has no transcript
    /// locator and no other memory ref, tombstone it (refcount hit zero — GC).
    /// O(this memory's span count), never a full scan. Returns the number of
    /// span edges touched (0 = memory had no links → caller need not persist).
    pub fn unlink_memory(&mut self, memory_id: u64) -> u64 {
        let indices = match self.mem_adjacency.remove(&memory_id) {
            Some(v) => v,
            None => {
                if self.mem_watermarks.remove(&memory_id).is_some() {
                    self.dirty = true;
                }
                return 0;
            }
        };
        let touched = indices.len() as u64;
        for idx in indices {
            let i = idx as usize;
            if let Some(s) = self.spans.get_mut(i) {
                s.mem_refs.retain(|m| *m != memory_id);
                if s.is_dead() && !s.text.is_empty() {
                    // refcount zero → tombstone (physical reclaim deferred to compaction)
                    self.by_hash.remove(&fnv1a(s.class, &s.text));
                    s.text.clear();
                    s.realms.clear();
                    s.locators.clear();
                    s.mem_refs.clear();
                }
            }
        }
        self.mem_watermarks.remove(&memory_id);
        if touched > 0 {
            self.mutations = self.mutations.wrapping_add(touched);
        }
        self.dirty = true;
        touched
    }

    /// True iff this memory currently has ≥1 live span edge.
    pub fn has_memory_link(&self, memory_id: u64) -> bool {
        self.mem_adjacency.get(&memory_id).map(|v| !v.is_empty()).unwrap_or(false)
    }

    /// Forward expansion: the verbatim atoms a given memory's text references.
    /// Lets a recalled belief hyperlink to the exact path/command it is about.
    pub fn spans_for_memory(&self, memory_id: u64, k: usize) -> Vec<SpanHit> {
        let indices = match self.mem_adjacency.get(&memory_id) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mut hits: Vec<SpanHit> = Vec::new();
        for &idx in indices {
            let e = match self.spans.get(idx as usize) {
                Some(e) if !e.text.is_empty() => e,
                _ => continue,
            };
            hits.push(SpanHit {
                text: e.text.clone(),
                class: e.class,
                count: e.count,
                last_ms: e.last_ms,
                realm: e
                    .realms
                    .first()
                    .and_then(|r| self.realm_dict.get(r.realm_id as usize).cloned())
                    .unwrap_or_default(),
                session: String::new(),
                line: 0,
                score: class_weight(e.class) * (1.0 + 0.10 * (e.count.max(1) as f32).ln()),
                memory_ids: e.mem_refs.to_vec(),
            });
        }
        // Most distinctive class first (file:line/path/error over url/issue), then popularity.
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }

    // ── ingest ────────────────────────────────────────────────────────────

    /// Incrementally ingest one transcript file from its byte watermark.
    /// If the file shrank since last seen (compaction/truncation), the watermark
    /// resets to 0 and the file is re-scanned; hash-dedup keeps that idempotent
    /// for atom identity (count reflects re-observation, which is fine).
    pub fn ingest_transcript(&mut self, path: &Path) -> IngestStats {
        let mut stats = IngestStats::default();
        let meta = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return stats,
        };
        let size = meta.len();
        let ts_ms = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let path_str = path.to_string_lossy().to_string();

        let (mut off, mut line_no) = *self.watermarks.get(&path_str).unwrap_or(&(0, 0));
        if off > size {
            off = 0;
            line_no = 0;
        }
        if off == size {
            return stats;
        }

        let data = match read_from(path, off) {
            Some(d) => d,
            None => return stats,
        };
        let session_idx = self.intern_session(&path_str);

        // Consume only complete lines; leave any trailing partial for next time.
        let mut consumed = 0usize;
        let mut start = 0usize;
        let bytes = data.as_slice();
        while let Some(nl) = memchr_nl(&bytes[start..]) {
            let end = start + nl; // index of '\n'
            let line = &bytes[start..end];
            self.ingest_line(line, session_idx, line_no, ts_ms, &mut stats);
            line_no = line_no.wrapping_add(1);
            consumed = end + 1;
            start = end + 1;
            stats.lines += 1;
        }

        self.watermarks
            .insert(path_str, (off + consumed as u64, line_no));
        if stats.new_spans > 0 {
            self.mutations = self.mutations.wrapping_add(stats.new_spans);
        }
        if consumed > 0 || stats.new_spans > 0 {
            self.dirty = true;
        }
        stats
    }

    fn ingest_line(
        &mut self,
        line: &[u8],
        session_idx: u32,
        line_no: u32,
        ts_ms: i64,
        stats: &mut IngestStats,
    ) {
        if line.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(line);
        // Skip chitta's own injected context to avoid a recall→transcript→recall
        // feedback loop (case 11b).
        if text.contains("UserPromptSubmit hook additional context")
            || text.contains("[soul-context]")
            || text.contains("[recall:")
        {
            stats.skipped_injection += 1;
            return;
        }
        // realm from this record's cwd (same rule the hooks use).
        let realm = extract_cwd(&text)
            .map(|c| realm_from_cwd(&c))
            .unwrap_or_else(|| "brahman".to_string());
        let realm_id = self.intern_realm(&realm);

        let scan = if text.len() > MAX_LINE_BYTES {
            &text[..text
                .char_indices()
                .take_while(|(i, _)| *i < MAX_LINE_BYTES)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0)]
        } else {
            &text
        };

        let mut harvested = 0usize;
        for (class, raw) in extract_atoms(scan) {
            if harvested >= PER_LINE_CAP {
                break;
            }
            stats.raw_spans += 1;
            let clamp = if class == CLASS_BASH { CLAMP_BASH } else { CLAMP_TEXT };
            let clamped = clamp_utf8(&raw, clamp);
            match redact(&clamped) {
                Some((red, was_red)) => {
                    if was_red {
                        stats.redacted += 1;
                        self.redacted_total += 1;
                    }
                    if self.insert_span(class, red, realm_id, session_idx, line_no, ts_ms) {
                        stats.new_spans += 1;
                    }
                    harvested += 1;
                }
                None => {
                    // entirely a secret — dropped
                    stats.redacted += 1;
                    self.redacted_total += 1;
                }
            }
        }
    }

    /// Full backfill over every *.jsonl under the Claude projects dir.
    pub fn ingest_dir(&mut self, projects_dir: &Path) -> IngestStats {
        let mut total = IngestStats::default();
        let mut files: Vec<PathBuf> = Vec::new();
        collect_jsonl(projects_dir, &mut files);
        for f in files {
            total += self.ingest_transcript(&f);
        }
        self.save();
        total
    }

    // ── query (no LLM, no GPU) ──────────────────────────────────────────────

    /// Rebuild the trigram accelerator iff the mutation counter advanced since
    /// the last build (the ensure_turbo pattern). Only used above TRIGRAM_MIN;
    /// below that a full scan is already sub-10ms and needs no index.
    fn ensure_trigram(&mut self) {
        if self.spans.len() < TRIGRAM_MIN {
            return;
        }
        // Incremental: index only the spans appended since the last query. The
        // previous full rebuild on every mutation cost seconds under the span
        // store write lock after each memory write (stall watch 2026-09-15:
        // 10-13 s recall stalls, queued put_memory holding the exclusive RPC lock
        // behind it). Spans never change text except to become tombstones,
        // which the query skips, so old entries stay valid.
        if self.trigram_indexed > self.spans.len() {
            self.trigram.clear();
            self.trigram_indexed = 0;
        }
        for i in self.trigram_indexed..self.spans.len() {
            let text = self.spans[i].text.to_lowercase();
            for tg in trigrams(&text) {
                self.trigram.entry(tg).or_default().insert(i as u32);
            }
        }
        self.trigram_indexed = self.spans.len();
        self.trigram_built_at = self.mutations;
    }

    pub fn query(&mut self, q: &str, realm: Option<&str>, k: usize) -> Vec<SpanHit> {
        if q.trim().is_empty() || self.spans.is_empty() {
            return Vec::new();
        }
        self.ensure_trigram();
        let ql = q.to_lowercase();
        let realm_filter: Option<u16> = match realm {
            Some(r) => match self.realm_idx.get(r) {
                Some(&id) => Some(id),
                None => return Vec::new(), // realm has no atoms → nothing leaks in
            },
            None => None,
        };
        // Token = maximal run of identifier/path chars. A pasted full path is a
        // single long token; that length is what makes a match distinctive.
        let qtokens: Vec<&str> = ql
            .split(|c: char| !c.is_alphanumeric() && c != '/' && c != '.' && c != '-' && c != '_')
            .filter(|t| t.len() >= 4 && !is_stopword(t))
            .collect();
        if qtokens.is_empty() {
            return Vec::new();
        }

        // Candidate set: trigram-narrowed above TRIGRAM_MIN, else all spans.
        let candidates: Vec<u32> = if self.spans.len() >= TRIGRAM_MIN {
            let mut set = roaring::RoaringBitmap::new();
            for tg in trigrams(&ql) {
                if let Some(bm) = self.trigram.get(&tg) {
                    set |= bm;
                }
            }
            set.iter().collect()
        } else {
            (0..self.spans.len() as u32).collect()
        };

        let now = now_ms();
        let mut hits: Vec<SpanHit> = Vec::new();
        for idx in candidates {
            let e = &self.spans[idx as usize];
            if e.text.is_empty() {
                continue; // tombstone (GC'd span awaiting compaction)
            }
            // realm scoping at source; capture the per-realm occurrence count so a
            // scoped query reports in-realm frequency, not the global total.
            let occ: Option<(u16, u32)> = match realm_filter {
                Some(rid) => match e.realms.iter().find(|r| r.realm_id == rid) {
                    Some(ro) => Some((rid, ro.count)),
                    None => continue,
                },
                None => None,
            };
            let tl = e.text.to_lowercase();
            // Match strength = the number of characters that matched distinctively.
            //   (a) whole atom present verbatim in the query -> atom length, or
            //   (b) longest query token that is a substring of the atom.
            // Distinctiveness (a 31-char filename token) beats popularity: a path
            // seen 5000×  is not more relevant to a specific query than one seen 6×.
            let mut strength = 0usize;
            if tl.len() >= 6 && ql.contains(&tl) {
                strength = tl.len();
            }
            for t in &qtokens {
                if t.len() > strength && tl.contains(*t) {
                    strength = t.len();
                }
            }
            if strength < 5 {
                continue;
            }
            // Coverage: how much of the atom the match explains. A 6-char generic
            // token ("chitta") inside a 60-char repo-root path covers 10% and is
            // diluted; a distinctive filename covering 40% is not. This stops
            // high-count generic paths from surfacing on a single common token.
            let coverage = (strength as f32 / tl.len().max(1) as f32).clamp(0.15, 1.0);
            let freq = 1.0 + 0.10 * (e.count as f32).ln(); // mild popularity prior
            let age_days = ((now - e.last_ms).max(0) as f32) / 86_400_000.0;
            let recency = 0.5 + 0.5 / (1.0 + age_days / 60.0); // soft, in [0.5, 1.0]
            let realm_boost = if occ.is_some() { 1.3 } else { 1.0 };
            let score =
                (strength as f32).sqrt() * coverage * class_weight(e.class) * freq * recency * realm_boost;

            // pick the most-recent locator, preferring one in the queried realm
            let loc = pick_locator(&e.locators, realm_filter);
            let (session, line, loc_realm) = loc
                .map(|l| {
                    (
                        self.session_dict
                            .get(l.session_idx as usize)
                            .cloned()
                            .unwrap_or_default(),
                        l.line,
                        self.realm_dict
                            .get(l.realm_id as usize)
                            .cloned()
                            .unwrap_or_default(),
                    )
                })
                .unwrap_or_default();
            hits.push(SpanHit {
                text: e.text.clone(),
                class: e.class,
                count: occ.map(|(_, c)| c).unwrap_or(e.count),
                last_ms: e.last_ms,
                realm: loc_realm,
                session,
                line,
                score,
                memory_ids: e.mem_refs.to_vec(),
            });
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }
}

fn add_locator(
    locs: &mut SmallVec<[Locator; MAX_LOCATORS]>,
    session_idx: u32,
    line: u32,
    realm_id: u16,
    ts_ms: i64,
) {
    if locs.iter().any(|l| l.session_idx == session_idx && l.line == line) {
        return;
    }
    if locs.len() < MAX_LOCATORS {
        locs.push(Locator { session_idx, line, realm_id, ts_ms });
        return;
    }
    // Keep the 2 oldest + 2 newest so provenance spans DIFFERENT sessions over
    // time rather than collapsing into one busy session.
    locs.push(Locator { session_idx, line, realm_id, ts_ms });
    locs.sort_by_key(|l| l.ts_ms);
    // drop a middle element
    let mid = locs.len() / 2;
    locs.remove(mid);
}

fn pick_locator(locs: &SmallVec<[Locator; MAX_LOCATORS]>, realm: Option<u16>) -> Option<&Locator> {
    let mut best: Option<&Locator> = None;
    for l in locs.iter() {
        if let Some(rid) = realm {
            if l.realm_id != rid {
                continue;
            }
        }
        match best {
            Some(b) if b.ts_ms >= l.ts_ms => {}
            _ => best = Some(l),
        }
    }
    // fall back to newest of any realm if none matched the filter
    best.or_else(|| locs.iter().max_by_key(|l| l.ts_ms))
}

// ── atom extraction ─────────────────────────────────────────────────────────

struct Extractors {
    path: regex::Regex,
    home: regex::Regex,
    url: regex::Regex,
    fileline: regex::Regex,
    issue: regex::Regex,
    bash: regex::Regex,
    error: regex::Regex,
}

fn extractors() -> &'static Extractors {
    static E: OnceLock<Extractors> = OnceLock::new();
    E.get_or_init(|| Extractors {
        path: regex::Regex::new(r"(?:/[\w.@+-]+){3,}").unwrap(),
        home: regex::Regex::new(r"~(?:/[\w.@+-]+){2,}").unwrap(),
        url: regex::Regex::new(r#"https?://[^\s)"'<>\\]+"#).unwrap(),
        fileline: regex::Regex::new(
            r"\b[\w./-]+\.(?:rs|py|cpp|hpp|cc|cxx|c|h|js|ts|tsx|jsx|go|java|sh|toml|json|md|txt|yaml|yml|rb|php|lua|sql|proto):\d+\b",
        )
        .unwrap(),
        issue: regex::Regex::new(r"#\d{8,}").unwrap(),
        // "command": "…"  — the Bash tool_use payload, captured structurally.
        bash: regex::Regex::new(r#""command"\s*:\s*"((?:[^"\\]|\\.){1,4000})""#).unwrap(),
        error: regex::Regex::new(
            r"(?:error\[E\d+\]|panicked at|Traceback \(most recent|thread '.*' panicked|[A-Za-z.]+Error:|[A-Za-z.]+Exception:)[^\n]{0,300}",
        )
        .unwrap(),
    })
}

/// Extract all high-value atoms from one line of raw transcript text.
/// Raw-line scanning works because JSON does not escape forward slashes, so
/// Unix paths / urls / uuids / file:line appear literally.
fn extract_atoms(text: &str) -> Vec<(u8, String)> {
    let e = extractors();
    let mut out: Vec<(u8, String)> = Vec::new();
    for m in e.fileline.find_iter(text) {
        out.push((CLASS_FILELINE, m.as_str().to_string()));
    }
    for m in e.path.find_iter(text) {
        if !is_noise_path(m.as_str()) {
            out.push((CLASS_PATH, m.as_str().to_string()));
        }
    }
    for m in e.home.find_iter(text) {
        if !is_noise_path(m.as_str()) {
            out.push((CLASS_PATH, m.as_str().to_string()));
        }
    }
    for m in e.url.find_iter(text) {
        out.push((CLASS_URL, m.as_str().to_string()));
    }
    // UUID and bare-HEX classes are intentionally NOT extracted: measured at
    // 793k + 812k unique (82% of all atoms) for near-zero recall value — opaque
    // session/message ids and content hashes nobody queries semantically. See
    // backfill histogram. They can be re-enabled gated on count>=2 if needed.
    for m in e.issue.find_iter(text) {
        out.push((CLASS_ISSUE, m.as_str().to_string()));
    }
    for c in e.bash.captures_iter(text) {
        if let Some(cmd) = c.get(1) {
            out.push((CLASS_BASH, unescape_json(cmd.as_str())));
        }
    }
    for m in e.error.find_iter(text) {
        out.push((CLASS_ERROR, m.as_str().to_string()));
    }
    out
}

/// Ephemeral / machine-generated paths that never help recall: temp roots,
/// build artifacts, vendored deps, VCS internals.
fn is_noise_path(p: &str) -> bool {
    const NOISE: [&str; 11] = [
        "/tmp/", "/proc/", "/sys/", "/dev/", "/.cache/", "/node_modules/", "/.git/",
        "/target/", "/__pycache__/", "/.venv/", "/site-packages/",
    ];
    if p.len() > 300 {
        return true; // absurdly long -> junk
    }
    NOISE.iter().any(|n| p.contains(n))
}

fn unescape_json(s: &str) -> String {
    s.replace("\\\"", "\"")
        .replace("\\n", " ")
        .replace("\\t", " ")
        .replace("\\\\", "\\")
}

// ── redaction (runs before store; fail-closed) ─────────────────────────────

struct Redactors {
    url_creds: regex::Regex,
    aws: regex::Regex,
    gh: regex::Regex,
    jwt: regex::Regex,
    kv: regex::Regex,
}

fn redactors() -> &'static Redactors {
    static R: OnceLock<Redactors> = OnceLock::new();
    R.get_or_init(|| Redactors {
        url_creds: regex::Regex::new(r"([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@").unwrap(),
        aws: regex::Regex::new(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b").unwrap(),
        gh: regex::Regex::new(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b").unwrap(),
        jwt: regex::Regex::new(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}\b")
            .unwrap(),
        kv: regex::Regex::new(
            r"(?i)\b([a-z_]*(?:secret|token|passwd|password|api[_-]?key|access[_-]?key|private[_-]?key|bearer)[a-z_]*)\s*[=:]\s*\S+",
        )
        .unwrap(),
    })
}

/// Redact known secret shapes. Returns `None` if nothing survives (the whole
/// span was a secret), else `(redacted, was_anything_redacted)`.
fn redact(s: &str) -> Option<(String, bool)> {
    let r = redactors();
    let mut out = r.url_creds.replace_all(s, "$1<redacted:creds>@").to_string();
    let mut hit = out != s;
    for (re, tag) in [
        (&r.aws, "<redacted:aws>"),
        (&r.gh, "<redacted:ghtoken>"),
        (&r.jwt, "<redacted:jwt>"),
    ] {
        let next = re.replace_all(&out, tag).to_string();
        if next != out {
            hit = true;
        }
        out = next;
    }
    let next = r.kv.replace_all(&out, "$1=<redacted>").to_string();
    if next != out {
        hit = true;
    }
    out = next;
    let residue = out.replace("<redacted:creds>", "").replace("<redacted:aws>", "")
        .replace("<redacted:ghtoken>", "").replace("<redacted:jwt>", "")
        .replace("<redacted>", "");
    if residue.trim().len() < 3 {
        return None;
    }
    Some((out, hit))
}

// ── small helpers ───────────────────────────────────────────────────────────

fn is_stopword(t: &str) -> bool {
    // Common query-prose words that are long enough to pass the length gate but
    // carry no atom identity. Keeps "where/what/output/..." from matching junk.
    matches!(
        t,
        "where" | "what" | "when" | "which" | "whose" | "there" | "here" | "this" | "that"
            | "with" | "from" | "have" | "were" | "will" | "would" | "could" | "should"
            | "about" | "into" | "than" | "then" | "them" | "they" | "your" | "you're"
            | "does" | "did" | "the" | "and" | "for" | "output" | "directory" | "file"
            | "files" | "path" | "command" | "value" | "using" | "used" | "please"
    )
}

fn realm_from_cwd(cwd: &str) -> String {
    // Same rule the hooks use: /repos/<name> -> <name>; else basename; else brahman.
    if let Some(pos) = cwd.find("/repos/") {
        let rest = &cwd[pos + 7..];
        let name = rest.split('/').next().unwrap_or("");
        if !name.is_empty() {
            return name.to_string();
        }
    }
    cwd.rsplit('/')
        .find(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "brahman".to_string())
}

fn extract_cwd(line: &str) -> Option<String> {
    // "cwd":"/some/path"
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| regex::Regex::new(r#""cwd"\s*:\s*"([^"]+)""#).unwrap());
    re.captures(line).and_then(|c| c.get(1)).map(|m| m.as_str().to_string())
}

fn clamp_utf8(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .take_while(|(i, _)| *i < max)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    s[..end].to_string()
}

fn trigrams(s: &str) -> Vec<u32> {
    let b: Vec<u8> = s.bytes().collect();
    if b.len() < 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(b.len() - 2);
    for w in b.windows(3) {
        out.push(((w[0] as u32) << 16) | ((w[1] as u32) << 8) | (w[2] as u32));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn memchr_nl(b: &[u8]) -> Option<usize> {
    b.iter().position(|&c| c == b'\n')
}

fn read_from(path: &Path, off: u64) -> Option<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(off)).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    Some(buf)
}

fn collect_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                collect_jsonl(&p, out);
            } else if p.extension().map(|x| x == "jsonl").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_classes() {
        let line = r#"{"cwd":"/maps/projects/x/repos/foo","content":"see /maps/projects/foo/bar/baz.rs and store.rs:762 at https://github.com/a/b uuid de809142-9597-4df4-a783-e87dadb4ba58"}"#;
        let atoms = extract_atoms(line);
        let classes: Vec<u8> = atoms.iter().map(|(c, _)| *c).collect();
        assert!(classes.contains(&CLASS_PATH), "path missing: {:?}", atoms);
        assert!(classes.contains(&CLASS_FILELINE), "fileline missing: {:?}", atoms);
        assert!(classes.contains(&CLASS_URL));
        // UUID/HEX are intentionally not extracted (space): confirm absent.
        assert!(!classes.contains(&CLASS_UUID), "uuid should be dropped: {:?}", atoms);
        assert!(!classes.contains(&CLASS_HEX));
    }

    #[test]
    fn noise_paths_filtered() {
        let line = r#"{"content":"real /maps/projects/foo/src and junk /tmp/x/y/z and /a/node_modules/pkg/index.js"}"#;
        let atoms = extract_atoms(line);
        let paths: Vec<&String> = atoms
            .iter()
            .filter(|(c, _)| *c == CLASS_PATH)
            .map(|(_, t)| t)
            .collect();
        assert!(paths.iter().any(|p| p.contains("/maps/projects/foo/src")));
        assert!(!paths.iter().any(|p| p.contains("/tmp/")), "tmp leaked: {:?}", paths);
        assert!(!paths.iter().any(|p| p.contains("node_modules")), "node_modules leaked: {:?}", paths);
    }

    #[test]
    fn redaction_drops_secrets() {
        // a bare AWS key is entirely a secret -> dropped
        assert!(redact("AKIAIOSFODNN7EXAMPLE").is_none());
        // KEY=secret redacted but surrounding survives
        let (out, hit) = redact("export API_KEY=sk-livesecret123").unwrap();
        assert!(hit);
        assert!(out.contains("<redacted>"), "{}", out);
        assert!(!out.contains("sk-livesecret123"));
        // a normal path is untouched
        let (out, hit) = redact("/home/kbd606/.claude/mind").unwrap();
        assert!(!hit);
        assert_eq!(out, "/home/kbd606/.claude/mind");
    }

    #[test]
    fn global_dedup_counts() {
        let mut s = SpanStore::new();
        let ra = s.intern_realm("a");
        let sess = s.intern_session("t1");
        assert!(s.insert_span(CLASS_PATH, "/a/b/c".into(), ra, sess, 1, 100));
        assert!(!s.insert_span(CLASS_PATH, "/a/b/c".into(), ra, sess, 2, 200));
        assert_eq!(s.spans.len(), 1);
        assert_eq!(s.spans[0].count, 2);
    }

    #[test]
    fn realm_scoping_no_leak() {
        let mut s = SpanStore::new();
        let ra = s.intern_realm("proj_a");
        let rb = s.intern_realm("proj_b");
        let sess = s.intern_session("t1");
        // atom only in A
        s.insert_span(CLASS_PATH, "/proj/a/only".into(), ra, sess, 1, 100);
        // cross-project atom in A and B
        s.insert_span(CLASS_PATH, "/shared/tool".into(), ra, sess, 2, 100);
        s.insert_span(CLASS_PATH, "/shared/tool".into(), rb, sess, 3, 100);

        // B-scoped query for the A-only atom -> nothing (no leak)
        let hits = s.query("what is /proj/a/only path", Some("proj_b"), 5);
        assert!(hits.is_empty(), "A-only atom leaked into B: {:?}", hits);
        // B-scoped query for the shared atom -> found
        let hits = s.query("run /shared/tool now", Some("proj_b"), 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "/shared/tool");
    }

    #[test]
    fn verbatim_query_recovers_atom() {
        let mut s = SpanStore::new();
        let ra = s.intern_realm("r");
        let sess = s.intern_session("t1");
        s.insert_span(CLASS_PATH, "/maps/data/bamdam/run42".into(), ra, sess, 5, 100);
        let hits = s.query("where did /maps/data/bamdam/run42 go", None, 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].text, "/maps/data/bamdam/run42");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn memory_edge_bidirectional() {
        let mut s = SpanStore::new();
        // memory 42's text mentions a path — link it.
        s.ingest_memory(42, "the truth table lives at /maps/ellesmere/flb_stats.tsv now", "ellesmere");
        // forward: memory -> its atoms
        let fwd = s.spans_for_memory(42, 5);
        assert!(fwd.iter().any(|h| h.text.contains("flb_stats.tsv")), "forward edge missing: {:?}", fwd.iter().map(|h| &h.text).collect::<Vec<_>>());
        // reverse: a query hit carries the memory_id
        let hits = s.query("where is /maps/ellesmere/flb_stats.tsv", None, 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].memory_ids.contains(&42), "reverse edge missing: {:?}", hits[0].memory_ids);
    }

    #[test]
    fn transcript_and_memory_share_one_span() {
        let mut s = SpanStore::new();
        let ra = s.intern_realm("r");
        let sess = s.intern_session("t1");
        // same atom seen in a transcript AND mentioned by memory 7 -> ONE span, both edges.
        s.insert_span(CLASS_PATH, "/shared/atom/path".into(), ra, sess, 1, 100);
        s.ingest_memory(7, "see /shared/atom/path", "r");
        assert_eq!(s.spans.iter().filter(|e| !e.text.is_empty()).count(), 1, "should dedup to one span");
        let hits = s.query("open /shared/atom/path", None, 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].memory_ids.contains(&7));
    }

    #[test]
    fn unlink_gc_when_refcount_zero_but_survives_if_referenced() {
        let mut s = SpanStore::new();
        let ra = s.intern_realm("r");
        let sess = s.intern_session("t1");
        // memory-only atom -> GC'd on unlink
        s.ingest_memory(1, "only in memory: /mem/only/path", "r");
        // transcript+memory atom -> survives unlink (transcript locator keeps it)
        s.insert_span(CLASS_PATH, "/kept/by/transcript".into(), ra, sess, 2, 100);
        s.ingest_memory(1, "also /kept/by/transcript", "r");
        s.unlink_memory(1);
        // memory-only atom gone
        assert!(s.query("/mem/only/path here", None, 5).is_empty(), "memory-only atom should be GC'd");
        // transcript atom still queryable, but no longer carries memory 1
        let hits = s.query("/kept/by/transcript run", None, 5);
        assert_eq!(hits.len(), 1);
        assert!(!hits[0].memory_ids.contains(&1), "stale memory edge not cleared");
    }

    #[test]
    fn memory_ingest_idempotent_and_supersede() {
        let mut s = SpanStore::new();
        s.ingest_memory(9, "path /a/b/c/d", "r");
        let before = s.spans.iter().filter(|e| !e.text.is_empty()).count();
        // same text again -> no-op (idempotent)
        let st = s.ingest_memory(9, "path /a/b/c/d", "r");
        assert_eq!(st.raw_spans, 0, "unchanged memory should be skipped");
        // changed text (supersede) -> old atom unlinked, new atom linked
        s.ingest_memory(9, "path /x/y/z/w", "r");
        assert!(s.query("/x/y/z/w go", None, 5).iter().any(|h| h.memory_ids.contains(&9)));
        assert!(s.query("/a/b/c/d go", None, 5).is_empty(), "superseded atom should be gone");
        let _ = before;
    }

    #[test]
    fn locator_cap_spans_sessions() {
        let mut locs: SmallVec<[Locator; MAX_LOCATORS]> = SmallVec::new();
        for i in 0..8i64 {
            add_locator(&mut locs, i as u32, 0, 0, i);
        }
        assert_eq!(locs.len(), MAX_LOCATORS);
        // oldest and newest retained
        let min = locs.iter().map(|l| l.ts_ms).min().unwrap();
        let max = locs.iter().map(|l| l.ts_ms).max().unwrap();
        assert_eq!(min, 0);
        assert_eq!(max, 7);
    }
}
