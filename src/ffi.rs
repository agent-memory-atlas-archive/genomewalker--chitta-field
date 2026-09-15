//! C FFI for chitta-field.
//! Uses typed POD structs for hot-path calls. No JSON in recall path.
//! All functions return 0 on success, negative on error.
//! Errors readable via cf_last_error().

use crate::field::ChittaField;
use crate::ops::{
    AgentDisableOp, AgentUpsertOp, AnalyticsEventOp, ClearProjectOp, EdgeType, MsgEventOp, Op,
    RecordRecallBatchOp, SessionEventOp, SkillDeprecateOp, SkillUploadOp, TaskEventOp,
    ThemeEventOp, TranscriptEventOp, UpdateMemoryContentOp, UpdateSymbolDescriptionOp,
    UserModelEventOp,
};
use crate::recall::RecallHit;
use serde_json;
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

// Thread-local last error, errno-style. Concurrent FFI calls on the same
// CfHandle no longer race on a shared error slot; each thread reads its own.
// Pointer returned by cf_last_error remains valid until the next FFI call
// on the same thread overwrites it (same contract as errno / strerror).
thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Opaque handle. C code holds *mut CfHandle, but the Rust side accesses it
/// through shared references (`&*h`) so concurrent FFI calls are sound;
/// interior mutability inside ChittaField (parking_lot RwLocks) protects the
/// actual data.
pub struct CfHandle {
    field: std::sync::Arc<ChittaField>,
    maintenance: Vec<(std::sync::mpsc::Sender<()>, std::thread::JoinHandle<()>)>,

}

impl CfHandle {
    fn ok(&self) -> c_int {
        LAST_ERROR.with(|le| *le.borrow_mut() = None);
        0
    }
    fn err(&self, e: impl std::fmt::Display) -> c_int {
        LAST_ERROR.with(|le| *le.borrow_mut() = CString::new(e.to_string()).ok());
        -1
    }
}

/// Error return for JSON-returning FFI fns: NULL + LAST_ERROR set, so C++
/// can always distinguish (and log) failure via cf_last_error instead of
/// treating NULL as ambiguous "empty". One envelope for all 90+ JSON fns.
fn json_null(e: impl std::fmt::Display) -> *mut c_char {
    LAST_ERROR.with(|le| *le.borrow_mut() = CString::new(e.to_string()).ok());
    std::ptr::null_mut()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn edge_type_from_u8(v: u8) -> EdgeType {
    match v {
        0 => EdgeType::DerivedFrom,
        1 => EdgeType::SameSession,
        2 => EdgeType::SameArtifact,
        3 => EdgeType::CoRetrieved,
        4 => EdgeType::Supports,
        6 => EdgeType::SemanticNeighbor,
        _ => EdgeType::Contradicts,
    }
}

fn write_hits(hits: Vec<RecallHit>, buf: *mut CfRecallHit, cap: usize, written: *mut usize) {
    let n = hits.len().min(cap);
    for (i, h) in hits.iter().take(n).enumerate() {
        unsafe {
            *buf.add(i) = CfRecallHit {
                memory_id: h.memory_id,
                score: h.score,
                semantic_score: h.semantic_score,
                ts_ms: h.ts_ms,
                strength: h.strength,
                confidence: h.confidence,
                access_count: h.access_count,
                semantic_weight: h.semantic_weight,
                status_mul: h.status_mul,
                epistemic_mul: h.epistemic_mul,
                strength_factor: h.strength_factor,
                affect_valence: h.affect_valence,
                affect_arousal: h.affect_arousal,
                actr_activation: h.actr_activation,
                surprise_boost: h.surprise_boost,
                arousal_boost: h.arousal_boost,
                mood_congruence: h.mood_congruence,
                frustration_boost: h.frustration_boost,
                interference_factor: h.interference_factor,
                spacing_boost: h.spacing_boost,
            };
        }
    }
    unsafe {
        *written = n;
    }
}

/// Share the profiling flag with the C++ stages around the FFI calls.
#[no_mangle]
pub extern "C" fn cf_recall_profile_enabled() -> c_int { crate::profile::enabled() as c_int }

/// Caller owns the returned string and releases it with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_memory_breakdown(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return std::ptr::null_mut(); }
    let field = &unsafe { &*h }.field;
    CString::new(field.memory_breakdown()).map_or(std::ptr::null_mut(), CString::into_raw)
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_open(data_dir: *const c_char, _lock_dir: *const c_char) -> *mut CfHandle {
    // lock_dir is ignored — the Upanishads model needs no locks.
    if data_dir.is_null() { return std::ptr::null_mut(); }
    let data_dir = unsafe {
        match CStr::from_ptr(data_dir).to_str() {
            Ok(s) => PathBuf::from(s),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    match ChittaField::open(data_dir) {
        Ok(field) => {
            let field = std::sync::Arc::new(field);
            let mut maintenance = Vec::new();
            // WAL has its own timer: a long index build must not delay group commit.
            for wal_only in [true, false] {
                let field = field.clone();
                let (stop, stopped) = std::sync::mpsc::channel();
                let mut interval = if wal_only { field.log.read().sync_interval() }
                    else { std::time::Duration::from_millis(100) };
                let worker = std::thread::Builder::new()
                    .name(if wal_only { "chitta-wal" } else { "chitta-maint" }.into())
                    .spawn(move || {
                        let touch_interval = std::time::Duration::from_secs(
                            std::env::var("CHITTA_TOUCH_FLUSH_S").ok().and_then(|s| s.parse().ok()).unwrap_or(5).max(1));
                        let min_mutations = std::env::var("CHITTA_TURBO_REBUILD_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
                        let mut last_touch = std::time::Instant::now();
                        while matches!(stopped.recv_timeout(interval), Err(std::sync::mpsc::RecvTimeoutError::Timeout)) {
                            if wal_only {
                                let mut log = field.log.write();
                                if let Err(e) = log.sync_if_due() {
                                    eprintln!("[field] WAL timer sync failed: {e}");
                                }
                                interval = log.next_sync_delay();
                            } else {
                                if last_touch.elapsed() >= touch_interval {
                                    if let Err(e) = field.drain_pending_touches() {
                                        eprintln!("[field] touch drain failed: {e}");
                                    }
                                    last_touch = std::time::Instant::now();
                                }
                                let plan = field.semantic_idx.read().plan_turbo_rebuild(min_mutations);
                                if let Some(plan) = plan {
                                    plan.build();
                                    field.semantic_idx.write().prune_turbo_changes();
                                }
                            }
                        }
                    });
                match worker {
                    Ok(worker) => maintenance.push((stop, worker)),
                    Err(e) => {
                        for (stop, worker) in maintenance { let _ = stop.send(()); let _ = worker.join(); }
                        let error = json_null(format!("cf_open maintenance: {e}"));
                        if !error.is_null() { unsafe { drop(CString::from_raw(error)); } }
                        return std::ptr::null_mut();
                    }
                }
            }
            Box::into_raw(Box::new(CfHandle { field, maintenance }))
        },
        Err(e) => {
            // The daemon only sees a null handle; without this line a refused
            // open (lock held, bad manifest, unreadable segment) is undiagnosable.
            eprintln!("[chitta-field] open failed: {e}");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn cf_close(h: *mut CfHandle) {
    if !h.is_null() {
        let mut b = unsafe { Box::from_raw(h) };
        for (stop, _) in &b.maintenance { let _ = stop.send(()); }
        for (_, worker) in b.maintenance.drain(..) {
            if worker.join().is_err() { eprintln!("[field] maintenance thread panicked"); }
        }
        // Persist any span links deferred off the write hot path.
        b.field.span_flush();
        drop(b);
    }
}

#[no_mangle]
pub extern "C" fn cf_last_error(_h: *const CfHandle) -> *const c_char {
    LAST_ERROR.with(|le| {
        le.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

// ── Chain integrity ──────────────────────────────────────────────────────────

/// Copy the current chain tip hash (32 bytes) into `out`.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn cf_chain_head(h: *const CfHandle, out: *mut u8) -> c_int {
    if h.is_null() || out.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let head = handle.field.chain_head();
    unsafe {
        std::ptr::copy_nonoverlapping(head.as_ptr(), out, 32);
    }
    0
}

/// One-shot backfill of the artifact (bridge-lane) index from stored payloads.
/// Writes memory-count to *out_mems and association-count to *out_assoc.
/// Returns 0 on success, -1 on null handle.
#[no_mangle]
pub extern "C" fn cf_backfill_artifact_refs(
    h: *const CfHandle,
    out_mems: *mut u64,
    out_assoc: *mut u64,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let (mems, assoc) = handle.field.backfill_artifact_refs();
    if !out_mems.is_null() {
        unsafe { *out_mems = mems };
    }
    if !out_assoc.is_null() {
        unsafe { *out_assoc = assoc };
    }
    0
}

/// Compiled store vector-space identity = hash(model_id, embed_dim, text_format_version).
/// Handle-less: lets a running daemon probe a replacement binary's vector space (via the
/// `format-id` subcommand) before execv'ing into it on self-update (PR4 gate).
#[no_mangle]
pub extern "C" fn cf_compiled_vector_space_id() -> u64 {
    crate::snapshot::StoreHeader::compiled_vector_space_id()
}

/// Compiled embedding model identifier (build.rs-generated EMBED_MODEL_ID) as a
/// NUL-terminated static C string. The C++ daemon stamps re_embed metadata with this so
/// the recorded embedding_model never drifts from the Rust store identity / vsid.
#[no_mangle]
pub extern "C" fn cf_embed_model_id() -> *const c_char {
    crate::ops::EMBED_MODEL_ID_CSTR.as_ptr() as *const c_char
}

/// Free a string returned by cf_* functions (e.g. cf_skill_read, cf_agent_get).
#[no_mangle]
pub extern "C" fn cf_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)); }
    }
}

// ── Write operations ──────────────────────────────────────────────────────────

/// Store a new memory. Returns MemoryId via out_memory_id (must be non-null).
/// embedding_ptr/embedding_len: pointer to f32 array, len must be 768.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_put_memory(
    h: *mut CfHandle,
    kind: *const c_char,
    realm: *const c_char,
    content_ptr: *const u8,
    content_len: usize,
    embedding_ptr: *const f32,
    embedding_len: usize,
    confidence: f32,
    decay_rate: f32,
    authored_at_ms: i64,
    out_memory_id: *mut u64,
) -> c_int {
    if h.is_null() || out_memory_id.is_null() || kind.is_null() || realm.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let realm_str = unsafe {
        match CStr::from_ptr(realm).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    // Null/zero-length content → empty slice (from_raw_parts on a null ptr is UB even for len 0).
    let content: &[u8] = if content_ptr.is_null() || content_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(content_ptr, content_len) }
    };
    let embedding: &[f32] = if embedding_ptr.is_null() || embedding_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(embedding_ptr, embedding_len) }
    };

    match handle.field.put_memory(
        kind_str,
        realm_str,
        content,
        embedding,
        confidence,
        decay_rate,
        authored_at_ms,
        vec![],
        None,
        None,
    ) {
        Ok((memory_id, _)) => {
            unsafe {
                *out_memory_id = memory_id;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Update mutable state of a memory.
/// Pass NaN for deltas you don't want to apply (use f32::NAN as sentinel).
#[no_mangle]
pub extern "C" fn cf_update_state(
    h: *mut CfHandle,
    memory_id: u64,
    strength_delta: f32,
    confidence_delta: f32,
    decay_rate: f32,
    touch: u8,
    pin: i8,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let strength_delta = if strength_delta.is_nan() {
        None
    } else {
        Some(strength_delta)
    };
    let confidence_delta = if confidence_delta.is_nan() {
        None
    } else {
        Some(confidence_delta)
    };
    let decay_rate = if decay_rate.is_nan() {
        None
    } else {
        Some(decay_rate)
    };
    let pin_opt = match pin {
        -1 => None,
        0 => Some(false),
        _ => Some(true),
    };

    match handle.field.update_state(
        memory_id,
        strength_delta,
        confidence_delta,
        decay_rate,
        touch != 0,
        pin_opt,
    ) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Soft-delete a memory.
#[no_mangle]
pub extern "C" fn cf_forget(h: *mut CfHandle, memory_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.forget(memory_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Record a positive ack signal for a memory (raises its recall ack_score).
#[no_mangle]
pub extern "C" fn cf_ack_memory(h: *mut CfHandle, memory_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.ack_memory(memory_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Record a negative nack signal for a memory (lowers its recall ack_score).
#[no_mangle]
pub extern "C" fn cf_nack_memory(h: *mut CfHandle, memory_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.nack_memory(memory_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Record an outcome observation against a memory's utility posterior.
/// `success` is 0/1; `weight` is capped at 5 and dropped when non-positive.
/// `out_alpha`/`out_beta` receive the updated posterior when non-null.
#[no_mangle]
pub extern "C" fn cf_record_outcome(
    h: *mut CfHandle,
    memory_id: u64,
    success: c_int,
    weight: f32,
    out_alpha: *mut f32,
    out_beta: *mut f32,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.record_outcome(memory_id, success != 0, weight) {
        Ok((alpha, beta)) => {
            if !out_alpha.is_null() {
                unsafe { *out_alpha = alpha };
            }
            if !out_beta.is_null() {
                unsafe { *out_beta = beta };
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Add an association edge between two memories.
/// edge_type: 0=DerivedFrom, 1=SameSession, 2=SameArtifact, 3=CoRetrieved, 4=Supports, 5=Contradicts
#[no_mangle]
pub extern "C" fn cf_add_assoc_edge(
    h: *mut CfHandle,
    src: u64,
    dst: u64,
    edge_type: u8,
    weight: f32,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let et = edge_type_from_u8(edge_type);
    match handle.field.add_assoc_edge(src, dst, et, weight) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Register a file artifact, returns its ArtifactId via out_artifact_id.
#[no_mangle]
pub extern "C" fn cf_upsert_artifact(
    h: *mut CfHandle,
    normalized_path: *const c_char,
    out_artifact_id: *mut u64,
) -> c_int {
    if h.is_null() || out_artifact_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let path_str = unsafe {
        match CStr::from_ptr(normalized_path).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    match handle.field.upsert_artifact(path_str, None) {
        Ok(artifact_id) => {
            unsafe {
                *out_artifact_id = artifact_id;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

// ── Read operations ───────────────────────────────────────────────────────────

/// A single recall result. Layout must match the C header exactly.
#[repr(C)]
#[derive(Clone)]
pub struct CfRecallHit {
    pub memory_id: u64,
    pub score: f32,
    pub semantic_score: f32,
    pub ts_ms: i64,
    pub strength: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub semantic_weight: f32,
    pub status_mul: f32,
    pub epistemic_mul: f32,
    pub strength_factor: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
    pub actr_activation: f32,
    pub surprise_boost: f32,
    pub arousal_boost: f32,
    pub mood_congruence: f32,
    pub frustration_boost: f32,
    pub interference_factor: f32,
    pub spacing_boost: f32,
}

/// Output buffer for recall results. Caller allocates hits_buf with capacity hits_cap.
/// On return, *hits_written contains number of results written.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_recall_semantic(
    h: *mut CfHandle,
    query_embedding: *const f32,
    embedding_len: usize,
    realm: *const c_char,
    k: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
    no_learn: bool,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let embedding = unsafe { std::slice::from_raw_parts(query_embedding, embedding_len) };
    let realm_str = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => Some(s),
                Err(e) => return handle.err(e),
            }
        }
    };

    let recalled = if no_learn {
        handle.field.recall_semantic_measure(embedding, k, realm_str)
    } else {
        handle.field.recall_semantic(embedding, k, realm_str)
    };
    match recalled {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Hybrid recall: HNSW semantic + BM25 keyword fused via RRF.
/// This is the real hybrid path — unlike cf_recall_semantic it calls
/// recall_with_fallback which runs stratified RRF merge internally.
/// start_ms/end_ms: optional authored_at_ms window GATE (0/0 = disabled);
/// the window gates candidates, semantic relevance still ranks.
#[no_mangle]
pub extern "C" fn cf_recall_with_fallback(
    h: *mut CfHandle,
    query_embedding: *const f32,
    embedding_len: usize,
    query_text: *const c_char,
    realm: *const c_char,
    k: usize,
    start_ms: i64,
    end_ms: i64,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let embedding = unsafe { std::slice::from_raw_parts(query_embedding, embedding_len) };
    let query_str = if query_text.is_null() {
        ""
    } else {
        match unsafe { CStr::from_ptr(query_text).to_str() } {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let realm_str = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => Some(s),
                Err(e) => return handle.err(e),
            }
        }
    };
    let window = if start_ms == 0 && end_ms == 0 { None } else { Some((start_ms, end_ms)) };
    match handle
        .field
        .recall_with_fallback_windowed(embedding, query_str, k, realm_str, window)
    {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Parse a natural-language temporal phrase out of a query. Deterministic,
/// no LLM. Returns a JSON string `{"from_ms":..,"to_ms":..,"stripped":".."}`
/// (caller frees with cf_free_string) or null when no phrase matches.
/// tz_offset_min: caller's local UTC offset in minutes (calendar math runs
/// in local wall time).
#[no_mangle]
pub extern "C" fn cf_parse_time_window(
    query: *const c_char,
    now_ms: i64,
    tz_offset_min: i32,
) -> *mut c_char {
    if query.is_null() {
        return std::ptr::null_mut();
    }
    let q = match unsafe { CStr::from_ptr(query).to_str() } {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match crate::organ::time_window::parse_time_window(q, now_ms, tz_offset_min) {
        Some(w) => {
            let json = serde_json::json!({
                "from_ms": w.from_ms.max(0),
                "to_ms": w.to_ms,
                "stripped": w.stripped,
            });
            CString::new(json.to_string())
                .map(|s| s.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
        None => std::ptr::null_mut(),
    }
}

/// Semantic recall with affective context for mood-congruent retrieval.
/// `query_valence` and `query_arousal` are NaN to disable affect matching.
#[no_mangle]
pub extern "C" fn cf_recall_semantic_ctx(
    h: *mut CfHandle,
    query_embedding: *const f32,
    embedding_len: usize,
    realm: *const c_char,
    k: usize,
    query_valence: f32,
    query_arousal: f32,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let embedding = unsafe { std::slice::from_raw_parts(query_embedding, embedding_len) };
    let realm_str = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => Some(s),
                Err(e) => return handle.err(e),
            }
        }
    };
    let qv = if query_valence.is_nan() { None } else { Some(query_valence) };
    let qa = if query_arousal.is_nan() { None } else { Some(query_arousal) };

    match handle.field.recall_semantic_ctx(embedding, k, realm_str, qv, qa, true) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Lane-0 atom bridge: given the recall anchor (dense-lead memory), write the
/// saturating-IDF co-atom partner ids and their BM25-IDF weights into parallel
/// out buffers, ranked by weight desc. Returns 0 on success; `written` <= cap.
/// The caller fuses these as a first-class RRF lane alongside dense/keyword.
#[no_mangle]
pub extern "C" fn cf_bridge_lane(
    h: *mut CfHandle,
    anchor_memory_id: u64,
    realm: *const c_char,
    k: usize,
    out_ids: *mut u64,
    out_weights: *mut f32,
    cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || out_ids.is_null() || out_weights.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let realm_str = if realm.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(realm).to_str() } {
            Ok(s) => Some(s),
            Err(e) => return handle.err(e),
        }
    };
    let cands = handle
        .field
        .bridge_lane_scored(anchor_memory_id, realm_str, k);
    let n = cands.len().min(cap);
    for (i, (m, w)) in cands.into_iter().take(n).enumerate() {
        unsafe {
            *out_ids.add(i) = m;
            *out_weights.add(i) = w;
        }
    }
    unsafe {
        *written = n;
    }
    handle.ok()
}

/// Field-RAG recall: Modern Hopfield / DAM relaxation over the HNSW candidate submatrix.
/// query_text is used for RRF BM25 lane (pass "" to skip BM25).
#[no_mangle]
pub extern "C" fn cf_recall_field(
    h: *mut CfHandle,
    query_embedding: *const f32,
    embedding_len: usize,
    query_text: *const c_char,
    realm: *const c_char,
    k: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let embedding = unsafe { std::slice::from_raw_parts(query_embedding, embedding_len) };
    let qtext = if query_text.is_null() {
        ""
    } else {
        match unsafe { CStr::from_ptr(query_text).to_str() } {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let realm_str = if realm.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(realm).to_str() } {
            Ok(s) => Some(s),
            Err(e) => return handle.err(e),
        }
    };
    match handle.field.recall_field(embedding, qtext, k, realm_str) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_temporal(
    h: *mut CfHandle,
    start_ms: i64,
    end_ms: i64,
    realm: *const c_char,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let realm_str = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => Some(s),
                Err(e) => return handle.err(e),
            }
        }
    };

    match handle
        .field
        .recall_temporal(start_ms, end_ms, realm_str, limit)
    {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_temporal_events(
    h: *mut CfHandle,
    start_ms: i64,
    end_ms: i64,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.recall_temporal_events(start_ms, end_ms, limit) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_artifact(
    h: *mut CfHandle,
    normalized_path: *const c_char,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || normalized_path.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let path_str = unsafe {
        match CStr::from_ptr(normalized_path).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.recall_artifact(path_str, limit) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_expand_associations(
    h: *mut CfHandle,
    seed_ids: *const u64,
    seed_count: usize,
    max_hops: usize,
    limit: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || seed_ids.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let seeds = unsafe { std::slice::from_raw_parts(seed_ids, seed_count) };

    match handle.field.expand_associations(seeds, max_hops, limit) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Personalized-PageRank injection lane. Seeds (ids + weights) come from the
/// caller's top fused hits; returns up to `top_g` graph-reachable (id, score)
/// pairs with the seeds excluded, written into the parallel out_ids/out_scores
/// arrays. *out_written is the count actually written (<= out_cap).
#[no_mangle]
pub extern "C" fn cf_ppr_lane(
    h: *mut CfHandle,
    seed_ids: *const u64,
    seed_weights: *const f32,
    seed_count: usize,
    top_g: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_cap: usize,
    out_written: *mut usize,
) -> c_int {
    if h.is_null()
        || seed_ids.is_null()
        || seed_weights.is_null()
        || out_ids.is_null()
        || out_scores.is_null()
        || out_written.is_null()
    {
        return -1;
    }
    let handle = unsafe { &*h };
    let seeds = unsafe { std::slice::from_raw_parts(seed_ids, seed_count) };
    let weights = unsafe { std::slice::from_raw_parts(seed_weights, seed_count) };

    let lane = handle.field.ppr_lane(seeds, weights, top_g);
    let n = lane.len().min(out_cap);
    unsafe {
        for (i, (id, score)) in lane.iter().take(n).enumerate() {
            *out_ids.add(i) = *id;
            *out_scores.add(i) = *score;
        }
        *out_written = n;
    }
    handle.ok()
}

/// Get payload content for a memory. Writes UTF-8 content into buf (null-terminated).
/// Returns 0 on success, -1 if not found/deleted, -2 if buf too small — in that
/// case *written is set to the required content length (excluding the NUL) so the
/// caller can retry with a buffer of at least written+1 bytes.
#[no_mangle]
pub extern "C" fn cf_get_content(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    get_content_impl(h, memory_id, buf, buf_cap, written, true)
}

/// Read-only payload hydration, including buffer-size probes and retries.
#[no_mangle]
pub extern "C" fn cf_peek_content(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    get_content_impl(h, memory_id, buf, buf_cap, written, false)
}

fn get_content_impl(h: *mut CfHandle, memory_id: u64, buf: *mut u8,
    buf_cap: usize, written: *mut usize, touch: bool) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    match if touch { handle.field.get_memory(memory_id) } else { handle.field.peek_memory(memory_id) } {
        Ok(payload) => {
            let content = &payload.content;
            // Require room for the trailing NUL the contract promises.
            if content.len() >= buf_cap {
                unsafe { *written = content.len(); }
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(content.as_ptr(), buf, content.len());
                *buf.add(content.len()) = 0;
                *written = content.len();
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Stage B: get the natural-language retrieval surface for a memory into buf.
/// written=0 (with a NUL) when no surface is stored — caller then embeds content.
/// Returns -2 (with required len in `written`) if the surface exceeds buf_cap.
#[no_mangle]
pub extern "C" fn cf_get_retrieval_surface(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() || buf_cap == 0 {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.get_retrieval_surface(memory_id) {
        Some(s) => {
            let bytes = s.as_bytes();
            if bytes.len() >= buf_cap {
                unsafe { *written = bytes.len(); }
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
                *written = bytes.len();
            }
            handle.ok()
        }
        None => {
            unsafe {
                *buf = 0;
                *written = 0;
            }
            handle.ok()
        }
    }
}

/// Stage B: set (surface non-empty) or clear (surface empty/null) a memory's
/// natural-language retrieval surface. `surface` is a UTF-8 byte buffer of `len`.
#[no_mangle]
pub extern "C" fn cf_set_retrieval_surface(
    h: *mut CfHandle,
    memory_id: u64,
    surface: *const u8,
    len: usize,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let s = if surface.is_null() || len == 0 {
        String::new()
    } else {
        let slice = unsafe { std::slice::from_raw_parts(surface, len) };
        String::from_utf8_lossy(slice).into_owned()
    };
    handle.field.set_retrieval_surface(memory_id, &s);
    handle.ok()
}

/// Get kind string for a memory into buf.
#[no_mangle]
pub extern "C" fn cf_get_kind(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
) -> c_int {
    if h.is_null() || buf.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    match handle.field.peek_memory(memory_id) {
        Ok(payload) => {
            let bytes = payload.kind.as_bytes();
            if bytes.len() >= buf_cap {
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Get realm string for a memory into buf.
#[no_mangle]
pub extern "C" fn cf_get_realm(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
) -> c_int {
    if h.is_null() || buf.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    match handle.field.peek_memory(memory_id) {
        Ok(payload) => {
            let bytes = payload.realm.as_bytes();
            if bytes.len() >= buf_cap {
                return -2;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_keyword(
    h: *mut CfHandle,
    query: *const c_char,
    k: usize,
    realm: *const c_char,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
    no_learn: bool,
) -> c_int {
    if h.is_null() || query.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    // Null/empty realm → unscoped; otherwise the BM25 lane is filtered to this realm.
    let realm_opt = if realm.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(realm) }.to_str() {
            Ok(s) if !s.is_empty() => Some(s),
            _ => None,
        }
    };

    let recalled = if no_learn {
        handle.field.recall_keyword_measure(query_str, k, realm_opt)
    } else {
        handle.field.recall_keyword_realm(query_str, k, realm_opt)
    };
    match recalled {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Deterministic provenance lookup (keyed lane, capability #1).
/// Returns 0 and fills `out_id` + `buf` (NUL-terminated record content) on a
/// hit; returns 1 (no fields written) on a clean miss; negative on error.
/// `sha` is tried first (content identity), then `input` path. Either may be
/// NULL/empty. No embedding, no ranking — an exact-key HashMap lookup.
#[no_mangle]
pub extern "C" fn cf_provenance_lookup(
    h: *mut CfHandle,
    sha: *const c_char,
    input: *const c_char,
    out_id: *mut u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || out_id.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let to_str = |p: *const c_char| -> &str {
        if p.is_null() {
            ""
        } else {
            unsafe { CStr::from_ptr(p) }.to_str().unwrap_or("")
        }
    };
    let sha_str = to_str(sha);
    let input_str = to_str(input);

    match handle.field.provenance_lookup(sha_str, input_str) {
        Some((id, content)) => {
            let bytes = content.as_bytes();
            if bytes.len() >= buf_cap {
                unsafe { *written = bytes.len(); }
                return -2;
            }
            unsafe {
                *out_id = id;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
                *written = bytes.len();
            }
            handle.ok()
        }
        None => 1,
    }
}

/// Deterministic correction lookup (keyed lane, capability #2 — durable
/// corrections with override semantics). Given free turn/context text, fills
/// `out_id` (newest fired correction) + `buf` (NUL-terminated; up to 3 fired
/// corrections joined by "\n---\n") on a hit; returns 1 (no fields written) on
/// a clean miss; negative on error. `text` may be NULL/empty. An exact-key
/// bigram probe — no embedding, no ranking, no fuzzy miss.
#[no_mangle]
pub extern "C" fn cf_correction_check(
    h: *mut CfHandle,
    text: *const c_char,
    out_id: *mut u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || out_id.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let text_str = if text.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(text) }.to_str().unwrap_or("")
    };

    match handle.field.correction_check(text_str) {
        Some((id, content)) => {
            let bytes = content.as_bytes();
            if bytes.len() >= buf_cap {
                unsafe { *written = bytes.len(); }
                return -2;
            }
            unsafe {
                *out_id = id;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
                *written = bytes.len();
            }
            handle.ok()
        }
        None => 1,
    }
}

/// Deterministic task-state lookup (keyed lane, capability #3 — task hand-off
/// across discontinuous sessions). Given a task slug, fills `out_id` + `buf`
/// (NUL-terminated record content) with the LATEST live `[task]` record for
/// that id on a hit; returns 1 (no fields written) on a clean miss; negative on
/// error. `id` may be NULL/empty. An exact-key HashMap read — no embedding, no
/// ranking. Latest-wins: status evolves, so the newest record is returned.
#[no_mangle]
pub extern "C" fn cf_task_state_lookup(
    h: *mut CfHandle,
    id: *const c_char,
    out_id: *mut u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || out_id.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let id_str = if id.is_null() {
        ""
    } else {
        unsafe { CStr::from_ptr(id) }.to_str().unwrap_or("")
    };

    match handle.field.task_state_lookup(id_str) {
        Some((mid, content)) => {
            let bytes = content.as_bytes();
            if bytes.len() >= buf_cap {
                unsafe { *written = bytes.len(); }
                return -2;
            }
            unsafe {
                *out_id = mid;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
                *buf.add(bytes.len()) = 0;
                *written = bytes.len();
            }
            handle.ok()
        }
        None => 1,
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_hdc(
    h: *mut CfHandle,
    query: *const c_char,
    realm: *const c_char,
    k: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || query.is_null() || hits_buf.is_null() || hits_written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let realm_opt = if realm.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(realm).to_str().ok() }
    };

    match handle.field.recall_hdc(query_str, k, realm_opt) {
        Ok(hits) => {
            write_hits(hits, hits_buf, hits_cap, hits_written);
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

// ── CEC: Event tape + CDAWG ──────────────────────────────────────────────────

/// Log a structured action event to the CEC tape and extend the CDAWG.
/// outcome: 0=success 1=fail 2=error 3=partial.
#[no_mangle]
pub extern "C" fn cf_log_event(
    h: *mut CfHandle,
    tool: *const c_char,
    entity: *const c_char,
    outcome: u8,
    session_id: u64,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || tool.is_null() || entity.is_null() { return -1; }
    let handle = unsafe { &*h };
    let tool_str = unsafe { match CStr::from_ptr(tool).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let entity_str = unsafe { match CStr::from_ptr(entity).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    handle.field.log_event(tool_str, entity_str, outcome, session_id, ts_ms);
    handle.ok()
}

/// Recall the last k occurrences of (tool, entity) from the CDAWG event tape.
#[no_mangle]
pub extern "C" fn cf_recall_last_action(
    h: *mut CfHandle,
    tool: *const c_char,
    entity: *const c_char,
    k: usize,
    hits_buf: *mut CfRecallHit,
    hits_cap: usize,
    hits_written: *mut usize,
) -> c_int {
    if h.is_null() || tool.is_null() || entity.is_null() || hits_buf.is_null() || hits_written.is_null() { return -1; }
    let handle = unsafe { &*h };
    let tool_str   = unsafe { match CStr::from_ptr(tool).to_str()   { Ok(s) => s, Err(e) => return handle.err(e) } };
    let entity_str = unsafe { match CStr::from_ptr(entity).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    match handle.field.recall_causal(tool_str, entity_str, k) {
        Ok(hits) => { write_hits(hits, hits_buf, hits_cap, hits_written); handle.ok() }
        Err(e)   => handle.err(e),
    }
}

/// Return top-k failure patterns from the CDAWG as JSON.
#[no_mangle]
pub extern "C" fn cf_recall_failure_pattern(
    h: *mut CfHandle,
    k: usize,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_recall_failure_pattern: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.recall_failure_pattern(k) {
        Ok(hits) => {
            let arr: Vec<serde_json::Value> = hits.iter().map(|h| serde_json::json!({
                "state_id":   h.memory_id,
                "fail_ratio": h.strength,
                "content":    h.content,
                "ts_ms":      h.ts_ms,
            })).collect();
            match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
                Ok(s) => s.into_raw(),
                Err(e) => json_null(format!("cf_recall_failure_pattern: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_recall_failure_pattern: {e}")),
    }
}

/// Prune (or down-weight) memories whose content matches any pattern, for the
/// `prune-memories` maintenance command. `patterns_json` is a JSON array of strings.
/// `apply`: 0=dry-run, nonzero=apply. `action`: 0=delete (forget), 1=archive (down-weight).
/// Returns a JSON array [{"id","kind","content"}] of matched memories; NULL on error.
/// Caller frees the result with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_prune_memories(
    h: *mut CfHandle,
    patterns_json: *const c_char,
    apply: c_int,
    action: c_int,
) -> *mut c_char {
    if h.is_null() || patterns_json.is_null() { return json_null("cf_prune_memories: null argument"); }
    let handle = unsafe { &*h };
    let pj = unsafe { std::ffi::CStr::from_ptr(patterns_json) }.to_string_lossy().into_owned();
    let patterns: Vec<String> = serde_json::from_str(&pj).unwrap_or_default();
    match handle.field.prune_by_content(&patterns, apply != 0, action as u8) {
        Ok(matches) => {
            let arr: Vec<serde_json::Value> = matches.iter().map(|(id, kind, content)| serde_json::json!({
                "id": id,
                "kind": kind,
                "content": content,
            })).collect();
            match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
                Ok(s) => s.into_raw(),
                Err(e) => json_null(format!("cf_prune_memories: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_prune_memories: {e}")),
    }
}

/// Return top-k PMI-ranked causal antecedents for (tool, entity) as JSON.
/// JSON array: [{rank, content, pmi, count}]
#[no_mangle]
pub extern "C" fn cf_recall_causal_antecedent(
    h: *mut CfHandle,
    tool: *const c_char,
    entity: *const c_char,
    k: usize,
) -> *mut c_char {
    if h.is_null() || tool.is_null() || entity.is_null() {
        return json_null("cf_recall_causal_antecedent: null argument");
    }
    let handle = unsafe { &*h };
    let tool_str   = unsafe { match CStr::from_ptr(tool).to_str()   { Ok(s) => s, Err(_) => return json_null("cf_recall_causal_antecedent: no data") } };
    let entity_str = unsafe { match CStr::from_ptr(entity).to_str() { Ok(s) => s, Err(_) => return json_null("cf_recall_causal_antecedent: no data") } };
    match handle.field.recall_causal_antecedent(tool_str, entity_str, k) {
        Ok(hits) => {
            let arr: Vec<serde_json::Value> = hits.iter().map(|h| serde_json::json!({
                "rank":    h.memory_id + 1,
                "content": h.content,
                "pmi":     h.score,
                "count":   h.access_count,
            })).collect();
            match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
                Ok(s) => s.into_raw(),
                Err(e) => json_null(format!("cf_recall_causal_antecedent: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_recall_causal_antecedent: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_hdcbind(
    h: *mut CfHandle,
    known_role: *const c_char,
    known_val: *const c_char,
    query_role: *const c_char,
    k: usize,
) -> *mut c_char {
    if h.is_null() || known_role.is_null() || known_val.is_null() || query_role.is_null() {
        return json_null("cf_recall_hdcbind: null argument");
    }
    let handle = unsafe { &*h };
    let kr = unsafe { match CStr::from_ptr(known_role).to_str() { Ok(s) => s, Err(_) => return json_null("cf_recall_hdcbind: no data") } };
    let kv = unsafe { match CStr::from_ptr(known_val).to_str()  { Ok(s) => s, Err(_) => return json_null("cf_recall_hdcbind: no data") } };
    let qr = unsafe { match CStr::from_ptr(query_role).to_str() { Ok(s) => s, Err(_) => return json_null("cf_recall_hdcbind: no data") } };
    match handle.field.recall_hdcbind(kr, kv, qr, k) {
        Ok(hits) => {
            let arr: Vec<serde_json::Value> = hits.iter().map(|h| {
                // content: "[hdcbind] given role=val → qrole=NAME (sim=...)"
                // extract NAME: after "→ ", take everything after '=' before ' ' or '('
                let name = h.content.split("→ ").nth(1)
                    .and_then(|s| s.splitn(2, '=').nth(1))
                    .and_then(|s| s.split(|c| c == ' ' || c == '(').next())
                    .unwrap_or("");
                serde_json::json!({
                    "rank":       h.memory_id + 1,
                    "name":       name,
                    "similarity": h.score,
                    "content":    h.content,
                })
            }).collect();
            match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
                Ok(s) => s.into_raw(),
                Err(e) => json_null(format!("cf_recall_hdcbind: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_recall_hdcbind: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_recall_counterfactual(
    h: *mut CfHandle,
    tool: *const c_char,
    entity: *const c_char,
    outcome: u8,
    k: usize,
) -> *mut c_char {
    if h.is_null() || tool.is_null() || entity.is_null() { return json_null("cf_recall_counterfactual: null argument"); }
    let handle = unsafe { &*h };
    let tool_s   = unsafe { match CStr::from_ptr(tool).to_str()   { Ok(s)=>s, Err(_)=>return json_null("cf_recall_counterfactual: no data") } };
    let entity_s = unsafe { match CStr::from_ptr(entity).to_str() { Ok(s)=>s, Err(_)=>return json_null("cf_recall_counterfactual: no data") } };
    match handle.field.recall_counterfactual(tool_s, entity_s, outcome, k) {
        Ok(hits) => {
            let arr: Vec<serde_json::Value> = hits.iter().map(|h| serde_json::json!({
                "rank":             h.memory_id + 1,
                "content":          h.content,
                "delta_pct":        h.score * 100.0,
                "confidence":       h.confidence,
                "support":          h.access_count,
            })).collect();
            match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
                Ok(s) => s.into_raw(),
                Err(e) => json_null(format!("cf_recall_counterfactual: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_recall_counterfactual: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_consolidation_preview(h: *mut CfHandle, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_consolidation_preview: null argument"); }
    let handle = unsafe { &*h };
    let items = handle.field.consolidation_preview(k);
    let arr: Vec<serde_json::Value> = items.iter().map(|(key, support)| {
        serde_json::json!({"key": key, "support": support})
    }).collect();
    match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
        Ok(s) => s.into_raw(),
        Err(e) => json_null(format!("cf_consolidation_preview: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_consolidation_pass(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_consolidation_pass: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.consolidation_pass() {
        Ok((total, promoted)) => {
            let s = format!("{{\"rules_found\":{total},\"rules_promoted\":{promoted}}}");
            match CString::new(s) {
                Ok(cs) => cs.into_raw(),
                Err(e) => json_null(format!("cf_consolidation_pass: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_consolidation_pass: {e}")),
    }
}

/// Return top-k CDAWG motif states ranked by Q-value as a JSON array.
#[no_mangle]
pub extern "C" fn cf_recall_motif_value(
    h: *mut CfHandle,
    tool: *const c_char,
    entity: *const c_char,
    k: usize,
) -> *mut c_char {
    if h.is_null() || tool.is_null() || entity.is_null() { return json_null("cf_recall_motif_value: null argument"); }
    let handle = unsafe { &*h };
    let tool_s   = unsafe { match CStr::from_ptr(tool).to_str()   { Ok(s)=>s, Err(_)=>return json_null("cf_recall_motif_value: no data") } };
    let entity_s = unsafe { match CStr::from_ptr(entity).to_str() { Ok(s)=>s, Err(_)=>return json_null("cf_recall_motif_value: no data") } };
    match handle.field.recall_motif_value(tool_s, entity_s, if k == 0 { 5 } else { k }) {
        Ok(hits) => {
            let arr: Vec<serde_json::Value> = hits.iter().map(|h| serde_json::json!({
                "state_id": h.memory_id,
                "q_value":  h.score * 2.0 - 1.0,
                "support":  h.access_count,
                "content":  h.content,
            })).collect();
            match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
                Ok(s) => s.into_raw(),
                Err(e) => json_null(format!("cf_recall_motif_value: {e}")),
            }
        }
        Err(e) => json_null(format!("cf_recall_motif_value: {e}")),
    }
}

/// Return top-k refuted/live Sequitur rules as a plain-text stats string.
#[no_mangle]
pub extern "C" fn cf_refutation_stats(h: *mut CfHandle, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_refutation_stats: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.refutation_stats(if k == 0 { 10 } else { k });
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_refutation_stats: {e}")),
    }
}

/// Log an event with cost metadata for regret-shaped Q-value update (Phase 10).
#[no_mangle]
pub extern "C" fn cf_log_event_ex(
    h: *mut CfHandle,
    tool: *const c_char, entity: *const c_char,
    outcome: u8, session_id: u64, ts_ms: i64,
    token_cost: u32, latency_ms: u32, retry_count: u8,
) -> c_int {
    if h.is_null() || tool.is_null() || entity.is_null() { return -1; }
    let handle = unsafe { &*h };
    let t = unsafe { match CStr::from_ptr(tool).to_str()   { Ok(s) => s, Err(e) => return handle.err(e) } };
    let e = unsafe { match CStr::from_ptr(entity).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    handle.field.log_event_ex(t, e, outcome, session_id, ts_ms, token_cost, latency_ms, retry_count);
    handle.ok()
}

/// Record an explicit decision point into the DecisionTape.
/// `rejected_json`: JSON array of [sym_u64, reason_u8] pairs.
#[no_mangle]
pub extern "C" fn cf_log_decision(
    h: *mut CfHandle,
    chosen_tool: *const c_char, chosen_entity: *const c_char, chosen_outcome: u8,
    rejected_json: *const c_char,
    confidence_delta: f32, ts_ms: i64,
) -> c_int {
    if h.is_null() || chosen_tool.is_null() || chosen_entity.is_null() { return -1; }
    let handle = unsafe { &*h };
    let ct = unsafe { match CStr::from_ptr(chosen_tool).to_str()   { Ok(s) => s, Err(e) => return handle.err(e) } };
    let ce = unsafe { match CStr::from_ptr(chosen_entity).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let rejected: Vec<(u64, u8)> = if rejected_json.is_null() {
        Vec::new()
    } else {
        let raw = unsafe { CStr::from_ptr(rejected_json) }.to_string_lossy();
        serde_json::from_str::<Vec<(u64, u8)>>(&raw).unwrap_or_default()
    };
    handle.field.log_decision(ct, ce, chosen_outcome, rejected, confidence_delta, ts_ms);
    handle.ok()
}

/// Recall true counterfactuals from DecisionTape for (tool, entity, outcome).
#[no_mangle]
pub extern "C" fn cf_recall_true_counterfactual(
    h: *mut CfHandle,
    tool: *const c_char, entity: *const c_char, outcome: u8, k: usize,
) -> *mut c_char {
    if h.is_null() || tool.is_null() || entity.is_null() { return json_null("cf_recall_true_counterfactual: null argument"); }
    let handle = unsafe { &*h };
    let t = unsafe { CStr::from_ptr(tool).to_string_lossy() };
    let e = unsafe { CStr::from_ptr(entity).to_string_lossy() };
    let hits = match handle.field.recall_true_counterfactual(&t, &e, outcome, if k == 0 { 5 } else { k }) {
        Ok(h) => h,
        Err(_) => return json_null("cf_recall_true_counterfactual: no data"),
    };
    let json = serde_json::to_string(&hits.iter().map(|h| serde_json::json!({
        "turn_id": h.memory_id,
        "score": h.score,
        "ts_ms": h.ts_ms,
        "content": h.content,
        "confidence_delta": h.confidence,
    })).collect::<Vec<_>>()).unwrap_or_else(|_| "[]".to_string());
    match CString::new(json) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_recall_true_counterfactual: {e}")),
    }
}

/// Return top-k hypothesis probes as JSON (rules with highest expected info gain).
#[no_mangle]
pub extern "C" fn cf_hypothesis_probes(h: *mut CfHandle, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_hypothesis_probes: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.hypothesis_probes(if k == 0 { 10 } else { k });
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_hypothesis_probes: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_executor_flush(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_executor_flush: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.executor_flush();
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_executor_flush: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_list_policies(h: *mut CfHandle, active_only: bool) -> *mut c_char {
    if h.is_null() { return json_null("cf_list_policies: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.list_policies(active_only);
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_list_policies: {e}")),
    }
}

/// CEC Phase 11: Turīya Monitor status. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_turiya_status(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_turiya_status: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.turiya_status();
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_turiya_status: {e}")),
    }
}

/// CEC Phase 14: queue deliberate micro-experiments for uncertain Sequitur rules.
/// Returns JSON: {"queued":N,"skipped_refuted":M,"skipped_certain":L}. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_queue_experiments(h: *mut CfHandle, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_queue_experiments: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.queue_experiments(if k == 0 { 5 } else { k });
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_queue_experiments: {e}")),
    }
}

/// CEC Phase 13: top-k verbalized Sequitur rules. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_verbalize_rules(h: *const CfHandle, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_verbalize_rules: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.verbalize_rules(if k == 0 { 10 } else { k });
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_verbalize_rules: {e}")),
    }
}

/// CEC Phase 12: EventTape statistics + compression totals. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_tape_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_tape_stats: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.tape_stats();
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_tape_stats: {e}")),
    }
}

/// Return FEP prior organ status as a JSON string (caller must cf_free_string).
#[no_mangle]
pub extern "C" fn cf_fep_status(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_fep_status: null argument"); }
    let handle = unsafe { &*h };
    let s = handle.field.fep_status();
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_fep_status: {e}")),
    }
}

/// Phase 16: CPU-native routed recall. Accepts a JSON RecallRequest, dispatches
/// to the cheapest lane (exact/fuzzy/temporal/causal/hybrid), returns JSON hits.
/// Caller must free with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_routed_recall(
    h: *const CfHandle,
    request_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || request_json.is_null() { return json_null("cf_routed_recall: null argument"); }
    let handle = unsafe { &*h };
    let req_str = unsafe { std::ffi::CStr::from_ptr(request_json) }.to_string_lossy();
    let req: crate::organ::query_router::RecallRequest =
        match serde_json::from_str(&req_str) {
            Ok(r) => r,
            Err(e) => {
                let s = format!(r#"{{"error":"bad request: {}"}}"#, e);
                return CString::new(s).map(|cs| cs.into_raw()).unwrap_or(json_null("cf_routed_recall: no data"));
            }
        };
    let result = handle.field.routed_recall(req);
    CString::new(result).map(|cs| cs.into_raw()).unwrap_or(json_null("cf_routed_recall: no data"))
}

// ── Phase 17 exports ─────────────────────────────────────────────────────────

/// Promote a candidate memory to established band. witness_kind: correction|outcome|hit_rate_delta.
/// Returns JSON. Caller must free with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_witness_memory(
    h: *mut CfHandle,
    memory_id: u64,
    witness_kind: *const c_char,
) -> *mut c_char {
    if h.is_null() || witness_kind.is_null() { return json_null("cf_witness_memory: null argument"); }
    let handle = unsafe { &*h };
    let wk = unsafe { std::ffi::CStr::from_ptr(witness_kind) }.to_string_lossy();
    let json = handle.field.witness_memory(memory_id, &wk);
    CString::new(json).map(|cs| cs.into_raw()).unwrap_or(json_null("cf_witness_memory: no data"))
}

/// R0 reconcile pass: scan assoc_edges for legality violations + detect contradictions.
/// Returns JSON summary. Caller must free with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_reconcile_pass(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_reconcile_pass: null argument"); }
    let handle = unsafe { &*h };
    let json = handle.field.reconcile_pass();
    CString::new(json).map(|cs| cs.into_raw()).unwrap_or(json_null("cf_reconcile_pass: no data"))
}

/// Force a full rebuild of all derived search indices (binary codes, coarse, LSH,
/// HNSW) from current embeddings. Used after an embedding-dimension migration.
/// Returns 0 on success, -1 on null handle.
#[no_mangle]
pub extern "C" fn cf_force_reindex(h: *const CfHandle) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    handle.field.force_reindex();
    0
}

/// Produce a harvest scope document from Turīya anomalies + router misses.
/// Returns JSON. Caller must free with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_harvest_scope(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_harvest_scope: null argument"); }
    let handle = unsafe { &*h };
    let json = handle.field.harvest_scope();
    CString::new(json).map(|cs| cs.into_raw()).unwrap_or(json_null("cf_harvest_scope: no data"))
}

/// Seed HDC codebook from a vocab_geometry harvest JSON (Phase 17 Part D).
/// json_path: path to harvest_ow.py vocab_geometry output file.
/// Returns JSON: {"ok":true,"seeded_tokens":N,"codebook_len":M,"source":"..."}.
#[no_mangle]
pub extern "C" fn cf_seed_hdc_geometry(
    h: *const CfHandle,
    json_path: *const c_char,
) -> *mut c_char {
    if h.is_null() || json_path.is_null() { return json_null("cf_seed_hdc_geometry: null argument"); }
    let handle = unsafe { &*h };
    let path = unsafe { std::ffi::CStr::from_ptr(json_path) }.to_string_lossy();
    let json = handle.field.seed_hdc_geometry(&path);
    CString::new(json).map(|cs| cs.into_raw()).unwrap_or(json_null("cf_seed_hdc_geometry: no data"))
}

// ── Triplet operations ────────────────────────────────────────────────────────

/// Add a triplet fact. Returns triplet_id via out_triplet_id.
/// source_memory_id: pass 0 for no source memory.
#[no_mangle]
pub extern "C" fn cf_add_triplet(
    h: *mut CfHandle,
    subject: *const c_char,
    predicate: *const c_char,
    object: *const c_char,
    weight: f32,
    source_memory_id: u64,
    out_triplet_id: *mut u64,
) -> c_int {
    cf_add_triplet_with_source(
        h, subject, predicate, object, weight,
        source_memory_id, std::ptr::null(), out_triplet_id,
    )
}

/// Invalidate a triplet.
#[no_mangle]
pub extern "C" fn cf_invalidate_triplet(h: *mut CfHandle, triplet_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.invalidate_triplet(triplet_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Select retrieval route for a query. Returns (episode_id, route_int).
/// route_int: 0=Semantic, 1=Keyword, 2=Temporal, 3=Artifact, 4=Hybrid, 5=Full
#[no_mangle]
pub extern "C" fn cf_select_route(
    h: *mut CfHandle, query: *const c_char,
    out_episode_id: *mut u64, out_route: *mut u8,
) -> c_int {
    if h.is_null() || query.is_null() || out_episode_id.is_null() || out_route.is_null() { return -1; }
    let handle = unsafe { &*h };
    let q = unsafe { std::ffi::CStr::from_ptr(query) }.to_string_lossy();
    let (episode_id, route) = handle.field.select_route(&q);
    use crate::learner::route::Route;
    let route_int: u8 = match route {
        Route::Semantic  => 0,
        Route::Keyword   => 1,
        Route::Temporal  => 2,
        Route::Artifact  => 3,
        Route::Hybrid    => 4,
        Route::Full      => 5,
        Route::Attractor => 6,
    };
    unsafe { *out_episode_id = episode_id; *out_route = route_int; }
    handle.ok()
}

/// Record outcome for a retrieval episode. reward in [-1, 1].
#[no_mangle]
pub extern "C" fn cf_route_feedback(
    h: *mut CfHandle, episode_id: u64, reward: f32,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.feedback(episode_id, reward) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_forget_triplet(
    h: *mut CfHandle, subject: *const c_char,
    predicate: *const c_char, object: *const c_char,
) -> c_int {
    if h.is_null() || subject.is_null() || predicate.is_null() || object.is_null() { return -1; }
    let handle = unsafe { &*h };
    let s = unsafe { std::ffi::CStr::from_ptr(subject) }.to_string_lossy();
    let p = unsafe { std::ffi::CStr::from_ptr(predicate) }.to_string_lossy();
    let o = unsafe { std::ffi::CStr::from_ptr(object) }.to_string_lossy();
    match handle.field.forget_triplet(&s, &p, &o) {
        Ok(_) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_backfill_embedding(
    h: *mut CfHandle, memory_id: u64,
    embedding_ptr: *const f32, embedding_len: usize,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let embedding = if embedding_ptr.is_null() || embedding_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(embedding_ptr, embedding_len) }
    };
    match handle.field.backfill_embedding(memory_id, embedding) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

// ── Deferred batched backfill (Step-1) — three-phase FFI ───────────────────────
// The C++ backfill thread calls stage → plan → apply per chunk, holding the
// exclusive rpc_mutex (acquire_lock) around stage and apply ONLY, so the O(log N)
// neighbor search in plan runs off the rpc_mutex (recall not blocked). See
// ChittaField::backfill_stage/plan/apply for the lock discipline.

/// Phase 1 (call under rpc_mutex). `embeddings` is `n * embed_dim` row-major f32.
/// Writes the number of ids still needing a global-HNSW plan to `out_plan_count`.
#[no_mangle]
pub extern "C" fn cf_backfill_stage(
    h: *mut CfHandle,
    ids: *const u64,
    embeddings: *const f32,
    n: usize,
    embed_dim: usize,
    out_plan_count: *mut usize,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    if (ids.is_null() || embeddings.is_null()) && n != 0 { return -1; }
    let ids_slice = if n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(ids, n) } };
    let emb_slice = if n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(embeddings, n * embed_dim) } };
    let items: Vec<(u64, Vec<f32>)> = ids_slice.iter().enumerate()
        .map(|(i, &id)| (id, emb_slice[i * embed_dim..(i + 1) * embed_dim].to_vec()))
        .collect();
    match handle.field.backfill_stage(&items) {
        Ok(plan_count) => {
            if !out_plan_count.is_null() { unsafe { *out_plan_count = plan_count; } }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Phase 2 (call WITHOUT rpc_mutex): compute HNSW plans off the write lock.
#[no_mangle]
pub extern "C" fn cf_backfill_plan(h: *mut CfHandle) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    handle.field.backfill_plan();
    handle.ok()
}

/// Phase 3 (call under rpc_mutex): apply plans + clear embed_pending. Writes the
/// number of memories whose embed_pending was cleared to `out_applied`.
#[no_mangle]
pub extern "C" fn cf_backfill_apply(h: *mut CfHandle, out_applied: *mut usize) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let n = handle.field.backfill_apply();
    if !out_applied.is_null() { unsafe { *out_applied = n; } }
    handle.ok()
}

/// Persist the delta HNSW sidecar so the next restart LOADS it instead of cold-
/// reinserting the delta from scratch. Off-lock safe (takes only semantic_idx.read()).
/// Writes the number of delta nodes persisted to `out_persisted` (0 = nothing/failed).
#[no_mangle]
pub extern "C" fn cf_persist_delta_hnsw(h: *mut CfHandle, out_persisted: *mut usize) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let n = handle.field.persist_delta_hnsw();
    if !out_persisted.is_null() { unsafe { *out_persisted = n; } }
    handle.ok()
}

#[no_mangle]
pub extern "C" fn cf_pending_embeddings(
    h: *mut CfHandle, out_ids: *mut u64, max_ids: usize, out_count: *mut usize,
) -> c_int {
    if h.is_null() || out_ids.is_null() || out_count.is_null() { return -1; }
    let handle = unsafe { &*h };
    let ids = handle.field.pending_embeddings(max_ids);
    let n = ids.len().min(max_ids);
    unsafe {
        std::ptr::copy_nonoverlapping(ids.as_ptr(), out_ids, n);
        *out_count = n;
    }
    handle.ok()
}

fn write_triplets_json(
    entries: Vec<crate::organ::triplet::TripletEntry>,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    use serde_json::json;

    let json_val: Vec<_> = entries
        .iter()
        .map(|e| {
            json!({
                "id": e.id,
                "subject": e.subject,
                "predicate": e.predicate,
                "object": e.object,
                "weight": e.weight,
            })
        })
        .collect();

    let s = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let bytes = s.as_bytes();
    // +1 for null terminator
    if bytes.len() + 1 > buf_cap {
        return -2;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
        *(buf as *mut u8).add(bytes.len()) = 0;
        *written = bytes.len();
    }

    0
}

/// Query by subject. Writes results as null-terminated JSON into buf.
/// JSON format: [{"id":1,"subject":"...","predicate":"...","object":"...","weight":0.9}]
/// Returns 0 on success, -1 on error, -2 if buf too small.
#[no_mangle]
pub extern "C" fn cf_query_subject(
    h: *mut CfHandle,
    subject: *const c_char,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || subject.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let subject_str = unsafe {
        match CStr::from_ptr(subject).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.query_subject(subject_str) {
        Ok(entries) => write_triplets_json(entries, buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

/// Query by object. Writes results as null-terminated JSON into buf.
#[no_mangle]
pub extern "C" fn cf_query_object(
    h: *mut CfHandle,
    object: *const c_char,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || object.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let object_str = unsafe {
        match CStr::from_ptr(object).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.query_object(object_str) {
        Ok(entries) => write_triplets_json(entries, buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

/// Query by entity (subject OR object). Writes results as null-terminated JSON into buf.
#[no_mangle]
pub extern "C" fn cf_query_entity(
    h: *mut CfHandle,
    entity: *const c_char,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || entity.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let entity_str = unsafe {
        match CStr::from_ptr(entity).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.query_entity(entity_str) {
        Ok(entries) => write_triplets_json(entries, buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

// ── Learner operations ────────────────────────────────────────────────────────

/// Apply feedback reward to a pending recall episode (route learning).
/// episode_id: returned by cf_select_route. reward: 0.0 = bad, 1.0 = perfect.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_feedback(h: *mut CfHandle, episode_id: u64, reward: f32) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.feedback(episode_id, reward) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Get recommended context window size for a session type.
/// session_type: null-terminated string (e.g., "code", "general").
/// Returns the recommended window size, or 10 on error.
#[no_mangle]
pub extern "C" fn cf_recommended_window(h: *mut CfHandle, session_type: *const c_char) -> usize {
    if h.is_null() || session_type.is_null() {
        return 10;
    }
    let handle = unsafe { &*h };
    let session_str = unsafe {
        match CStr::from_ptr(session_type).to_str() {
            Ok(s) => s,
            Err(_) => return 10,
        }
    };
    handle.field.recommended_window(session_str)
}

// ── Maintenance ───────────────────────────────────────────────────────────────

/// Ingest new ops from all foreign-instance segment files.
/// Reads bytes appended since the last call, applies ops to in-memory state.
/// Returns the count of ops applied, or -1 on error (see cf_last_error).
#[no_mangle]
pub extern "C" fn cf_sync_foreign(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.sync_foreign() {
        Ok(count) => count as c_int,
        Err(e) => handle.err(e),
    }
}

/// Phase 1 of sync_foreign: read new peer ops off disk into an internal pending buffer.
/// Does disk I/O; on a hard-mounted NFS volume it can block indefinitely. Call it WITHOUT
/// holding the daemon's rpc lock. Returns the number of ops now pending, or -1 on error.
#[no_mangle]
pub extern "C" fn cf_sync_foreign_collect(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.sync_foreign_collect() {
        Ok(count) => count as c_int,
        Err(e) => handle.err(e),
    }
}

/// Phase 2 of sync_foreign: apply the pending ops to in-memory state. Takes ~40 write guards,
/// does no disk reads. Call it WITH the daemon's exclusive rpc lock held.
/// Returns the count of ops applied, or -1 on error.
#[no_mangle]
pub extern "C" fn cf_sync_foreign_apply(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.sync_foreign_apply() {
        Ok(count) => count as c_int,
        Err(e) => handle.err(e),
    }
}

/// Flush write buffer to OS.
#[no_mangle]
pub extern "C" fn cf_flush(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.flush() {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// fdatasync the WAL to disk (durable). Separated from cf_flush so the disk sync can
/// run OFF the C++ rpc_mutex — put_memory now only flush_buf()s under the lock, and the
/// caller fdatasyncs after releasing the lock, so recall is no longer blocked by the
/// per-write fsync (was ~200-330ms on NFS /home).
#[no_mangle]
pub extern "C" fn cf_sync(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.sync_wal() {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Get stats.
#[no_mangle]
pub extern "C" fn cf_memory_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.memory_count() }
}

/// O(1) upper-bound count (includes soft-deleted). Safe for latency-sensitive paths.
#[no_mangle]
pub extern "C" fn cf_raw_memory_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.raw_memory_count() }
}

/// O(1) count of memories awaiting embedding. Maintained atomically.
#[no_mangle]
pub extern "C" fn cf_purge_orphan_embed_pending(h: *mut CfHandle, out_cleared: *mut usize) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let n = handle.field.purge_orphan_embed_pending();
    if !out_cleared.is_null() { unsafe { *out_cleared = n; } }
    handle.ok()
}

/// Semantic-index coverage: how many memories SHOULD have a vector vs how many DO.
/// `pending_count` cannot answer this — the queue drains on failure as well as success.
#[no_mangle]
pub extern "C" fn cf_embed_coverage(
    h: *mut CfHandle, out_eligible: *mut usize, out_embedded: *mut usize,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let (eligible, embedded) = handle.field.embed_coverage();
    if !out_eligible.is_null() { unsafe { *out_eligible = eligible; } }
    if !out_embedded.is_null() { unsafe { *out_embedded = embedded; } }
    handle.ok()
}

/// Copy a memory's stored vector into `out` (capacity `cap` floats).
/// Returns the number of floats written, or 0 if the memory has no embedding.
#[no_mangle]
pub extern "C" fn cf_get_embedding(
    h: *mut CfHandle, id: u64, out: *mut f32, cap: usize,
) -> usize {
    if h.is_null() || out.is_null() { return 0; }
    let handle = unsafe { &*h };
    let Some(v) = handle.field.get_embedding(id) else { return 0 };
    let n = v.len().min(cap);
    unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out, n); }
    n
}

#[no_mangle]
pub extern "C" fn cf_force_clear_embed_pending(
    h: *mut CfHandle, ids: *const u64, count: usize, out_cleared: *mut usize,
) -> c_int {
    if h.is_null() || ids.is_null() { return -1; }
    let handle = unsafe { &*h };
    let id_slice = unsafe { std::slice::from_raw_parts(ids, count) };
    let n = handle.field.force_clear_embed_pending(id_slice);
    if !out_cleared.is_null() { unsafe { *out_cleared = n; } }
    handle.ok()
}

#[no_mangle]
pub extern "C" fn cf_pending_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.raw_pending_count() }
}

#[no_mangle]
pub extern "C" fn cf_requeue_ghost_embeddings(h: *const CfHandle) -> usize {
    if h.is_null() { return 0; }
    unsafe { &*h }.field.requeue_ghost_embeddings()
}

#[no_mangle]
pub extern "C" fn cf_requeue_all_embeddings(
    h: *const CfHandle,
    model_id_ptr: *const std::os::raw::c_char,
    model_id_len: usize,
) -> i64 {
    if h.is_null() { return -1; }
    let model_id = if model_id_ptr.is_null() || model_id_len == 0 {
        crate::ops::EMBED_MODEL_ID.to_string()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(model_id_ptr as *const u8, model_id_len) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    match unsafe { &*h }.field.requeue_all_embeddings(&model_id) {
        Ok(n) => n as i64,
        Err(_) => -1,
    }
}

// ── Code Intelligence ─────────────────────────────────────────────────────────

/// A single symbol search result. Fixed-size POD struct for C interop.
#[repr(C)]
#[derive(Clone)]
pub struct CfSymbolHit {
    pub symbol_id: u64,
    pub score: f32,
    pub kind: [u8; 64],
    pub name: [u8; 256],
    pub signature: [u8; 512],
    pub file_path: [u8; 1024],
    pub line_start: u32,
    pub line_end: u32,
    pub repo_id: u64,
}

impl CfSymbolHit {
    fn from_entry(entry: &crate::organ::symbol::SymbolEntry, score: f32) -> Self {
        fn copy_str(dst: &mut [u8], src: &str) {
            let bytes = src.as_bytes();
            let n = bytes.len().min(dst.len() - 1);
            dst[..n].copy_from_slice(&bytes[..n]);
            dst[n] = 0;
        }
        let mut hit = CfSymbolHit {
            symbol_id: entry.id,
            score,
            kind: [0u8; 64],
            name: [0u8; 256],
            signature: [0u8; 512],
            file_path: [0u8; 1024],
            line_start: entry.line_start,
            line_end: entry.line_end,
            repo_id: entry.repo_id,
        };
        copy_str(&mut hit.kind, &entry.kind);
        copy_str(&mut hit.name, &entry.name);
        copy_str(&mut hit.signature, &entry.signature);
        copy_str(&mut hit.file_path, &entry.file_path);
        hit
    }
}

fn write_symbol_hits(
    entries: &[crate::organ::symbol::SymbolEntry],
    score: f32,
    buf: *mut CfSymbolHit,
    cap: usize,
    written: *mut usize,
) {
    let n = entries.len().min(cap);
    for (i, e) in entries.iter().take(n).enumerate() {
        unsafe {
            *buf.add(i) = CfSymbolHit::from_entry(e, score);
        }
    }
    unsafe {
        *written = n;
    }
}

/// Upsert a symbol. Returns 0 on success, -1 on error. Writes symbol_id to out_id.
#[no_mangle]
pub extern "C" fn cf_upsert_symbol(
    h: *mut CfHandle,
    kind: *const c_char,
    name: *const c_char,
    signature: *const c_char,
    file_path: *const c_char,
    line_start: u32,
    line_end: u32,
    repo_id: u64,
    embedding: *const f32,
    embed_len: usize,
    description: *const c_char,
    memory_id: u64,
    out_id: *mut u64,
) -> c_int {
    if h.is_null() || out_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    macro_rules! parse_str {
        ($ptr:expr) => {
            if $ptr.is_null() {
                return -1;
            } else {
                match unsafe { CStr::from_ptr($ptr).to_str() } {
                    Ok(s) => s,
                    Err(e) => return handle.err(e),
                }
            }
        };
    }

    let kind_str = parse_str!(kind);
    let name_str = parse_str!(name);
    let sig_str = parse_str!(signature);
    let path_str = parse_str!(file_path);
    let desc = if description.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(description).to_str() } {
            Ok(s) if !s.is_empty() => Some(s.to_string()),
            _ => None,
        }
    };
    let mem_id = if memory_id == 0 {
        None
    } else {
        Some(memory_id)
    };
    let emb: &[f32] = if embedding.is_null() || embed_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(embedding, embed_len) }
    };

    match handle.field.upsert_symbol(
        kind_str, name_str, sig_str, path_str, line_start, line_end, repo_id, emb, desc, mem_id,
    ) {
        Ok(id) => {
            unsafe {
                *out_id = id;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Remove a symbol.
#[no_mangle]
pub extern "C" fn cf_remove_symbol(h: *mut CfHandle, symbol_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.remove_symbol(symbol_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Search symbols by name (exact or prefix). Returns number written via *written.
#[no_mangle]
pub extern "C" fn cf_search_symbols_by_name(
    h: *mut CfHandle,
    query: *const c_char,
    limit: usize,
    buf: *mut CfSymbolHit,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || query.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let query_str = match unsafe { CStr::from_ptr(query).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let results = handle.field.search_symbols_by_name(query_str, limit);
    write_symbol_hits(&results, 1.0, buf, buf_len, written);
    handle.ok()
}

/// Semantic symbol search. Returns number written via *written.
#[no_mangle]
pub extern "C" fn cf_search_symbols_semantic(
    h: *mut CfHandle,
    query: *const f32,
    embed_len: usize,
    k: usize,
    buf: *mut CfSymbolHit,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || query.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let emb = unsafe { std::slice::from_raw_parts(query, embed_len) };

    let scored = handle.field.search_symbols_semantic(emb, k);
    let n = scored.len().min(buf_len);
    for (i, (sym_id, score)) in scored.iter().take(n).enumerate() {
        if let Ok(Some(entry)) = handle.field.get_symbol(*sym_id) {
            unsafe {
                *buf.add(i) = CfSymbolHit::from_entry(&entry, *score);
            }
        }
    }
    unsafe {
        *written = n;
    }
    handle.ok()
}

/// Get all symbols in a file.
#[no_mangle]
pub extern "C" fn cf_symbols_in_file(
    h: *mut CfHandle,
    file_path: *const c_char,
    buf: *mut CfSymbolHit,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || file_path.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let path_str = match unsafe { CStr::from_ptr(file_path).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let results = handle.field.symbols_in_file(path_str);
    write_symbol_hits(&results, 1.0, buf, buf_len, written);
    handle.ok()
}

/// Search symbols by name restricted to file paths containing `path_filter`
/// (NULL/empty = unscoped). Returns number written via *written.
#[no_mangle]
pub extern "C" fn cf_search_symbols_by_name_scoped(
    h: *mut CfHandle,
    query: *const c_char,
    limit: usize,
    path_filter: *const c_char,
    buf: *mut CfSymbolHit,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || query.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let query_str = match unsafe { CStr::from_ptr(query).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let filter = if path_filter.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(path_filter).to_str() } {
            Ok(s) if !s.is_empty() => Some(s),
            Ok(_) => None,
            Err(e) => return handle.err(e),
        }
    };

    let results = handle
        .field
        .search_symbols_by_name_scoped(query_str, limit, filter);
    write_symbol_hits(&results, 1.0, buf, buf_len, written);
    handle.ok()
}

/// Remove every symbol in a file (per-file invalidation before re-extract).
#[no_mangle]
pub extern "C" fn cf_remove_symbols_by_file(
    h: *mut CfHandle,
    file_path: *const c_char,
    out_removed: *mut usize,
) -> c_int {
    if h.is_null() || file_path.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let path_str = match unsafe { CStr::from_ptr(file_path).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    match handle.field.remove_symbols_by_file(path_str) {
        Ok(n) => {
            if !out_removed.is_null() {
                unsafe { *out_removed = n };
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// GC the symbol index. `path_excludes` is a comma-separated substring list
/// (NULL/empty = none). Writes a JSON stats object to buf.
#[no_mangle]
pub extern "C" fn cf_dedupe_symbols(
    h: *mut CfHandle,
    dry_run: c_int,
    check_fs: c_int,
    path_excludes: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let excludes: Vec<String> = if path_excludes.is_null() {
        Vec::new()
    } else {
        match unsafe { CStr::from_ptr(path_excludes).to_str() } {
            Ok(s) => s
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect(),
            Err(e) => return handle.err(e),
        }
    };
    match handle
        .field
        .dedupe_symbols(dry_run != 0, check_fs != 0, &excludes)
    {
        Ok(stats) => {
            let json_str = match serde_json::to_string(&stats) {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            };
            write_json_buf(&json_str, buf, buf_cap, written)
        }
        Err(e) => handle.err(e),
    }
}

/// Add a call edge between two symbols.
#[no_mangle]
pub extern "C" fn cf_add_sym_call_edge(h: *mut CfHandle, caller_id: u64, callee_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.add_call_edge(caller_id, callee_id) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Get symbols called by caller_id. Returns count via *written.
#[no_mangle]
pub extern "C" fn cf_get_callees(
    h: *mut CfHandle,
    symbol_id: u64,
    buf: *mut u64,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let ids = handle.field.get_callees(symbol_id);
    let n = ids.len().min(buf_len);
    for (i, &id) in ids.iter().take(n).enumerate() {
        unsafe {
            *buf.add(i) = id;
        }
    }
    unsafe {
        *written = n;
    }
    handle.ok()
}

/// Get symbols that call symbol_id. Returns count via *written.
#[no_mangle]
pub extern "C" fn cf_get_callers(
    h: *mut CfHandle,
    symbol_id: u64,
    buf: *mut u64,
    buf_len: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let ids = handle.field.get_callers(symbol_id);
    let n = ids.len().min(buf_len);
    for (i, &id) in ids.iter().take(n).enumerate() {
        unsafe {
            *buf.add(i) = id;
        }
    }
    unsafe {
        *written = n;
    }
    handle.ok()
}

// ── Contradiction engine ─────────────────────────────────────────────────────

/// Get memory IDs that contradict the given memory (bidirectional).
#[no_mangle]
pub extern "C" fn cf_get_conflicts(
    h: *mut CfHandle,
    memory_id: u64,
    out_ids: *mut u64,
    max_ids: usize,
    out_count: *mut usize,
) -> c_int {
    if h.is_null() || out_ids.is_null() || out_count.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.get_conflicts(memory_id) {
        Ok(ids) => {
            let n = ids.len().min(max_ids);
            for (i, &id) in ids.iter().take(n).enumerate() {
                unsafe { *out_ids.add(i) = id; }
            }
            unsafe { *out_count = n; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Follow supersession chain from memory_id. Returns chain including self.
#[no_mangle]
pub extern "C" fn cf_get_supersession_chain(
    h: *mut CfHandle,
    memory_id: u64,
    out_ids: *mut u64,
    max_ids: usize,
    out_count: *mut usize,
) -> c_int {
    if h.is_null() || out_ids.is_null() || out_count.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.get_supersession_chain(memory_id) {
        Ok(ids) => {
            let n = ids.len().min(max_ids);
            for (i, &id) in ids.iter().take(n).enumerate() {
                unsafe { *out_ids.add(i) = id; }
            }
            unsafe { *out_count = n; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Get memory IDs that confirm the given memory.
#[no_mangle]
pub extern "C" fn cf_get_confirmations(
    h: *mut CfHandle,
    memory_id: u64,
    out_ids: *mut u64,
    max_ids: usize,
    out_count: *mut usize,
) -> c_int {
    if h.is_null() || out_ids.is_null() || out_count.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.get_confirmations(memory_id) {
        Ok(ids) => {
            let n = ids.len().min(max_ids);
            for (i, &id) in ids.iter().take(n).enumerate() {
                unsafe { *out_ids.add(i) = id; }
            }
            unsafe { *out_count = n; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Upsert a code file (legacy). Returns its file_id via *out_id.
#[no_mangle]
pub extern "C" fn cf_upsert_code_file(
    h: *mut CfHandle,
    path: *const c_char,
    project: *const c_char,
    mtime: i64,
    out_id: *mut u64,
) -> c_int {
    cf_upsert_code_file_v2(
        h, path, project, mtime,
        std::ptr::null(), std::ptr::null(), std::ptr::null(), 0,
        std::ptr::null_mut(),
        out_id,
    )
}

/// Upsert a code file with content hash and git provenance.
/// Nullable params: pass null for absent. out_changed: set to 1 if content changed, 0 if hash matched.
#[no_mangle]
pub extern "C" fn cf_upsert_code_file_v2(
    h: *mut CfHandle,
    path: *const c_char,
    project: *const c_char,
    mtime: i64,
    content_hash: *const c_char,
    git_commit: *const c_char,
    git_author: *const c_char,
    git_timestamp_ms: i64,
    out_changed: *mut c_int,
    out_id: *mut u64,
) -> c_int {
    if h.is_null() || path.is_null() || project.is_null() || out_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let path_str = match unsafe { CStr::from_ptr(path).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let project_str = match unsafe { CStr::from_ptr(project).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let hash_opt = if content_hash.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(content_hash).to_str().ok().map(|s| s.to_string()) }
    };
    let commit_opt = if git_commit.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(git_commit).to_str().ok().map(|s| s.to_string()) }
    };
    let author_opt = if git_author.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(git_author).to_str().ok().map(|s| s.to_string()) }
    };
    let ts_opt = if git_timestamp_ms < 0 { None } else { Some(git_timestamp_ms) };

    match handle.field.upsert_code_file(
        path_str, project_str, mtime,
        hash_opt, commit_opt, author_opt, ts_opt,
    ) {
        Ok((id, was_updated)) => {
            unsafe { *out_id = id; }
            if !out_changed.is_null() {
                unsafe { *out_changed = if was_updated { 1 } else { 0 }; }
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Invalidate all active triplets associated with a source file.
#[no_mangle]
pub extern "C" fn cf_invalidate_triplets_by_source_file(
    h: *mut CfHandle,
    source_file: *const c_char,
) -> c_int {
    if h.is_null() || source_file.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let sf_str = match unsafe { CStr::from_ptr(source_file).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    match handle.field.invalidate_triplets_by_source_file(sf_str) {
        Ok(_) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Add a triplet with optional source_file. Returns triplet_id via out_triplet_id.
#[no_mangle]
pub extern "C" fn cf_add_triplet_with_source(
    h: *mut CfHandle,
    subject: *const c_char,
    predicate: *const c_char,
    object: *const c_char,
    weight: f32,
    source_memory_id: u64,
    source_file: *const c_char,
    out_triplet_id: *mut u64,
) -> c_int {
    if h.is_null() || subject.is_null() || predicate.is_null()
        || object.is_null() || out_triplet_id.is_null()
    {
        return -1;
    }
    let handle = unsafe { &*h };

    let subject_str = unsafe {
        match CStr::from_ptr(subject).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let predicate_str = unsafe {
        match CStr::from_ptr(predicate).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let object_str = unsafe {
        match CStr::from_ptr(object).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let src_mem = if source_memory_id == 0 { None } else { Some(source_memory_id) };
    let src_file = if source_file.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(source_file).to_str().ok().map(|s| s.to_string()) }
    };

    match handle.field.add_triplet(
        subject_str.to_string(),
        predicate_str.to_string(),
        object_str.to_string(),
        weight,
        src_mem,
        src_file,
    ) {
        Ok(triplet_id) => {
            unsafe { *out_triplet_id = triplet_id; }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_symbol_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.symbol_count() }
}

#[no_mangle]
pub extern "C" fn cf_code_file_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.code_file_count() }
}

/// Encode all unindexed memories into sparse codes. Returns count encoded.
#[no_mangle]
pub extern "C" fn cf_encode_all(h: *mut CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    let handle = unsafe { &*h };
    match handle.field.encode_all_unindexed() {
        Ok(n) => n,
        Err(e) => {
            handle.err(e);
            0
        }
    }
}

/// Get cortical index size (how many memories have sparse codes).
#[no_mangle]
pub extern "C" fn cf_cortical_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.cortical_count() }
}

/// Get number of prototype clusters in the CorticalIndex.
#[no_mangle]
pub extern "C" fn cf_prototype_count(h: *mut CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.prototype_count() }
}

/// Train product quantizer on accumulated residuals. Returns true on success.
/// Requires at least 256 encoded memories.
#[no_mangle]
pub extern "C" fn cf_train_pq(h: *mut CfHandle) -> bool {
    if h.is_null() {
        return false;
    }
    let handle = unsafe { &*h };
    handle.field.train_pq().is_ok()
}

/// Encode PQ residuals for all memories not yet PQ-encoded.
/// Trains PQ first if needed. Returns count encoded, or 0 on error.
#[no_mangle]
pub extern "C" fn cf_encode_all_pq(h: *mut CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    let handle = unsafe { &*h };
    match handle.field.encode_all_pq() {
        Ok(n) => n,
        Err(e) => {
            handle.err(e);
            0
        }
    }
}

/// Return how many memories have PQ residual codes.
#[no_mangle]
pub extern "C" fn cf_pq_count(h: *mut CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.pq_count() }
}

/// Save the cortical index to a binary snapshot file. Returns true on success.
#[no_mangle]
pub extern "C" fn cf_save_snapshot(h: *mut CfHandle) -> bool {
    if h.is_null() {
        return false;
    }
    let handle = unsafe { &*h };
    handle.field.save_snapshot().is_ok()
}

/// Save the full in-memory state to a binary snapshot file (chitta.snapshot).
/// Returns true on success.
#[no_mangle]
pub extern "C" fn cf_save_full_snapshot(h: *mut CfHandle) -> bool {
    if h.is_null() {
        return false;
    }
    let handle = unsafe { &*h };
    handle.field.save_full_snapshot().is_ok()
}

// ── Lite Encoder ─────────────────────────────────────────────────────────────

/// Train the lite encoder from all existing memories with sparse codes.
/// Returns number of training examples or -1 on error.
#[no_mangle]
pub extern "C" fn cf_train_lite_encoder(h: *mut CfHandle) -> i32 {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.train_lite_encoder() {
        Ok(n) => n as i32,
        Err(e) => handle.err(e),
    }
}

/// Save the lite encoder to disk (<data_dir>/lite_encoder.bin).
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_save_lite_encoder(h: *mut CfHandle) -> i32 {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.save_lite_encoder() {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Check if lite encoder is trained and ready.
/// Returns 1 if ready, 0 if not.
#[no_mangle]
pub extern "C" fn cf_lite_encoder_ready(h: *const CfHandle) -> u8 {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.lite_encoder_ready() as u8 }
}

/// Encode text via lite encoder into sparse feature indices and weights.
/// out_atoms: caller-allocated array of at least K_ACTIVE uint32 values.
/// out_weights: caller-allocated array of at least K_ACTIVE f32 values.
/// Returns the number of active features written (≤ K_ACTIVE), or -1 on failure.
#[no_mangle]
pub extern "C" fn cf_encode_lite(
    h: *const CfHandle,
    text_ptr: *const u8,
    text_len: usize,
    out_atoms: *mut u32,
    out_weights: *mut f32,
) -> i32 {
    if h.is_null() || text_ptr.is_null() || out_atoms.is_null() || out_weights.is_null() {
        return -1;
    }
    let text = unsafe {
        match std::str::from_utf8(std::slice::from_raw_parts(text_ptr, text_len)) {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    let field = unsafe { &(*h).field };
    match field.encode_lite(text) {
        Some(code) => {
            let n = code.feature_ids.len();
            unsafe {
                for (i, (&atom, &weight)) in code
                    .feature_ids
                    .iter()
                    .zip(code.activations.iter())
                    .enumerate()
                {
                    *out_atoms.add(i) = atom;
                    *out_weights.add(i) = weight;
                }
            }
            n as i32
        }
        None => -1,
    }
}

/// Run a tier demotion pass. Returns demoted_count (low 32 bits) | deleted_count (high 32 bits).
#[no_mangle]
pub extern "C" fn cf_run_demotion(h: *mut CfHandle, now_ms: i64) -> u64 {
    if h.is_null() {
        return 0;
    }
    let handle = unsafe { &*h };
    match handle.field.run_demotion_pass(now_ms) {
        Ok((demoted, deleted)) => (demoted as u64) | ((deleted as u64) << 32),
        Err(e) => {
            handle.err(e);
            0
        }
    }
}

// ── Domain Event Log ──────────────────────────────────────────────────────────

/// Iterate log ops starting from `from_seqno`.
/// Callback receives: op serialized as JSON bytes, length, seqno, user ctx.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_iterate_log(
    h: *mut CfHandle,
    from_seqno: u64,
    callback: extern "C" fn(*const u8, usize, u64, *mut std::ffi::c_void),
    ctx: *mut std::ffi::c_void,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let result = handle.field.log.write().replay(from_seqno, |_inst, seqno, op| {
        let json = serde_json::to_vec(&op).unwrap_or_default();
        callback(json.as_ptr(), json.len(), seqno, ctx);
        Ok(())
    });
    match result {
        Ok(_) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Emit a domain event into the chitta-field log.
/// domain: "session", "transcript", "task", "theme", "analytics"
/// Returns assigned event_id via *out_event_id, or 0 on error.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_emit_event(
    h: *mut CfHandle,
    domain: *const c_char,
    kind: *const c_char,
    entity_id: *const c_char,
    payload_json: *const u8,
    payload_len: usize,
    realm: *const c_char,
    fencing_token: u64,
    out_event_id: *mut u64,
) -> c_int {
    if h.is_null() || domain.is_null() || kind.is_null() || out_event_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let domain_str = unsafe {
        match CStr::from_ptr(domain).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let entity_id_str = if entity_id.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(entity_id).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };
    let realm_str = if realm.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };
    let payload = if payload_json.is_null() || payload_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(payload_json, payload_len) }.to_vec()
    };

    // Capture payload as string before it is moved into the op (for immediate in-memory apply).
    let payload_str_for_apply = String::from_utf8(payload.clone()).unwrap_or_default();

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);

    use std::time::{SystemTime, UNIX_EPOCH};
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let op = match domain_str {
        "session" => Op::SessionEvent(SessionEventOp {
            event_id,
            session_id: entity_id_str.to_string(),
            kind: kind_str.to_string(),
            payload_json: payload,
            realm: realm_str.to_string(),
            ts_ms,
        }),
        "transcript" => Op::TranscriptEvent(TranscriptEventOp {
            event_id,
            session_id: entity_id_str.to_string(),
            kind: kind_str.to_string(),
            payload_json: payload,
            realm: realm_str.to_string(),
            ts_ms,
        }),
        "task" => Op::TaskEvent(TaskEventOp {
            event_id,
            task_type: entity_id_str.to_string(),
            task_id: entity_id_str.to_string(),
            kind: kind_str.to_string(),
            payload_json: payload,
            realm: realm_str.to_string(),
            ts_ms,
            fencing_token,
        }),
        "theme" => Op::ThemeEvent(ThemeEventOp {
            event_id,
            kind: kind_str.to_string(),
            theme_id: fencing_token,
            payload_json: payload,
            ts_ms,
        }),
        "analytics" => Op::AnalyticsEvent(AnalyticsEventOp {
            event_id,
            kind: kind_str.to_string(),
            session_id: entity_id_str.to_string(),
            payload_json: payload,
            ts_ms,
        }),
        "msg" | "sadhana" | "dream" | "ledger" => Op::MsgEvent(MsgEventOp {
            event_id,
            domain: domain_str.to_string(),
            kind: kind_str.to_string(),
            target: entity_id_str.to_string(),
            payload_json: payload,
            realm: realm_str.to_string(),
            ts_ms,
        }),
        "user_model" => {
            return handle.err("cf_emit_event: use cf_user_model_upsert/cf_user_model_observe for user_model events");
        }
        _ => return handle.err(format!("unknown domain: {}", domain_str)),
    };

    let result = handle.field.log.write().append(&op);
    match result {
        Ok(_seqno) => {
            // Immediately apply to in-memory state for same-instance reads.
            // (WAL replay only fires for foreign ops from other instances.)
            if domain_str == "transcript" {
                use crate::organ::OrganApply;
                // Apply the full typed event, not only its exact-lookup payload.
                // Otherwise a just-registered transcript is absent from
                // cf_transcript_list until the next daemon/WAL replay.
                let _ = handle.field.transcript_registry.write().apply(op.clone());
            }
            if domain_str == "session" {
                let mut reg = handle.field.session_registry.write();
                match kind_str {
                    "register" => {
                        let session_kind = serde_json::from_str::<serde_json::Value>(&payload_str_for_apply)
                            .ok()
                            .and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(|s| s.to_string()))
                            .unwrap_or_default();
                        reg.register(entity_id_str.to_string(), session_kind, realm_str.to_string(), ts_ms);
                    }
                    "heartbeat" => reg.heartbeat(entity_id_str, ts_ms),
                    "deregister" => reg.deregister(entity_id_str),
                    _ => {}
                }
                // Mirror session events into msg_registry so get_events_by_domain_kind works.
                use crate::organ::msg::MsgEvent;
                handle.field.msg_registry.write().insert(MsgEvent {
                    event_id,
                    domain: domain_str.to_string(),
                    kind: kind_str.to_string(),
                    target: entity_id_str.to_string(),
                    payload_json: payload_str_for_apply.clone(),
                    realm: realm_str.to_string(),
                    ts_ms,
                });
            }
            if matches!(domain_str, "msg" | "sadhana" | "dream" | "ledger") {
                use crate::organ::msg::MsgEvent;
                handle.field.msg_registry.write().insert(MsgEvent {
                    event_id,
                    domain: domain_str.to_string(),
                    kind: kind_str.to_string(),
                    target: entity_id_str.to_string(),
                    payload_json: payload_str_for_apply,
                    realm: realm_str.to_string(),
                    ts_ms,
                });
            }
            unsafe {
                *out_event_id = event_id;
            }
            handle.ok()
        }
        Err(e) => handle.err(e),
    }
}

/// Get the payload of the most recent domain event matching domain+kind+entity_id.
/// Supports domain="user_model" (kind = entity_type) and domain="transcript"
/// (kind = transcript event kind, entity_id = session_id).
/// Returns 0 and writes JSON payload to buf if found, 1 if not found,
/// -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_get_latest_event(
    h: *mut CfHandle,
    domain: *const c_char,
    kind: *const c_char,
    entity_id: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null()
        || domain.is_null()
        || kind.is_null()
        || entity_id.is_null()
        || buf.is_null()
        || written.is_null()
    {
        return -1;
    }
    let handle = unsafe { &*h };

    let domain_str = unsafe {
        match CStr::from_ptr(domain).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let entity_id_str = unsafe {
        match CStr::from_ptr(entity_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let payload: Option<String> = match domain_str {
        "user_model" => {
            let registry = handle.field.user_model_registry.read();
            match registry.get(entity_id_str) {
                Some(entry) if entry.entity_type == kind_str => Some(entry.payload_json.clone()),
                _ => None,
            }
        }
        "transcript" => {
            let registry = handle.field.transcript_registry.read();
            registry
                .get_session_event(entity_id_str, kind_str)
                .map(|s| s.to_string())
        }
        "ledger" | "msg" | "sadhana" | "dream" => {
            let registry = handle.field.msg_registry.read();
            registry
                .get_events(domain_str, kind_str, entity_id_str, usize::MAX)
                .last()
                .map(|e| e.payload_json.clone())
        }
        _ => {
            return handle.err(format!(
                "cf_get_latest_event: unsupported domain '{}'",
                domain_str
            ))
        }
    };

    match payload {
        Some(p) => write_json_buf(&p, buf, buf_cap, written),
        None => 1,
    }
}

/// Query events by domain, kind, and target (e.g. session_id for msg delivery).
/// Returns JSON array of event objects into buf: `[{"event_id":..., "kind":..., "target":..., "payload_json":..., "realm":..., "ts_ms":...}, ...]`
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_get_events_by_target(
    h: *mut CfHandle,
    domain: *const c_char,
    kind: *const c_char,
    target: *const c_char,
    limit: usize,
    out_buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || domain.is_null() || kind.is_null() || target.is_null()
        || out_buf.is_null() || written.is_null()
    {
        return -1;
    }
    let handle = unsafe { &*h };

    let domain_str = unsafe {
        match CStr::from_ptr(domain).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let target_str = unsafe {
        match CStr::from_ptr(target).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let registry = handle.field.msg_registry.read();
    let events = registry.get_events(domain_str, kind_str, target_str, limit);

    let json_arr: Vec<serde_json::Value> = events
        .iter()
        .map(|ev| {
            let payload: serde_json::Value =
                serde_json::from_str(&ev.payload_json).unwrap_or(serde_json::Value::Null);
            serde_json::json!({
                "event_id": ev.event_id,
                "kind": ev.kind,
                "target": ev.target,
                "payload": payload,
                "realm": ev.realm,
                "ts_ms": ev.ts_ms,
            })
        })
        .collect();

    let json_str = serde_json::to_string(&json_arr).unwrap_or_else(|_| "[]".to_string());
    write_json_buf(&json_str, out_buf, buf_cap, written)
}

/// Query all events matching domain+kind across all targets.
/// Returns a JSON array sorted by ts_ms descending (newest first), up to `limit` entries.
/// Each element: {event_id, kind, target, payload, realm, ts_ms}
#[no_mangle]
pub extern "C" fn cf_get_events_by_domain_kind(
    h: *mut CfHandle,
    domain: *const c_char,
    kind: *const c_char,
    limit: usize,
    out_buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || domain.is_null() || kind.is_null() || out_buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let domain_str = unsafe {
        match CStr::from_ptr(domain).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let registry = handle.field.msg_registry.read();
    let events = registry.get_events_by_domain_kind(domain_str, kind_str, limit);

    let json_arr: Vec<serde_json::Value> = events
        .iter()
        .map(|ev| {
            let payload: serde_json::Value =
                serde_json::from_str(&ev.payload_json).unwrap_or(serde_json::Value::Null);
            serde_json::json!({
                "event_id": ev.event_id,
                "kind": ev.kind,
                "target": ev.target,
                "payload": payload,
                "realm": ev.realm,
                "ts_ms": ev.ts_ms,
            })
        })
        .collect();

    let json_str = serde_json::to_string(&json_arr).unwrap_or_else(|_| "[]".to_string());
    write_json_buf(&json_str, out_buf, buf_cap, written)
}

/// Check whether any event exists for (domain, kind, target). Returns 1 if found, 0 if not, -1 on error.
#[no_mangle]
pub extern "C" fn cf_has_event(
    h: *mut CfHandle,
    domain: *const c_char,
    kind: *const c_char,
    target: *const c_char,
) -> c_int {
    if h.is_null() || domain.is_null() || kind.is_null() || target.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let domain_str = unsafe { CStr::from_ptr(domain).to_str().unwrap_or("") };
    let kind_str   = unsafe { CStr::from_ptr(kind).to_str().unwrap_or("") };
    let target_str = unsafe { CStr::from_ptr(target).to_str().unwrap_or("") };
    let registry = handle.field.msg_registry.read();
    if registry.has_event(domain_str, kind_str, target_str) { 1 } else { 0 }
}

/// Look up a single event by event_id. Returns JSON object: {event_id, kind, target, payload, realm, ts_ms}
/// or empty object {} if not found.
#[no_mangle]
pub extern "C" fn cf_get_event_by_id(
    h: *mut CfHandle,
    event_id: u64,
    out_buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || out_buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let registry = handle.field.msg_registry.read();
    let json_str = match registry.get_event_by_id(event_id) {
        Some(ev) => {
            let payload: serde_json::Value =
                serde_json::from_str(&ev.payload_json).unwrap_or(serde_json::Value::Null);
            serde_json::json!({
                "event_id": ev.event_id,
                "kind": ev.kind,
                "target": ev.target,
                "payload": payload,
                "realm": ev.realm,
                "ts_ms": ev.ts_ms,
            })
            .to_string()
        }
        None => "{}".to_string(),
    };
    write_json_buf(&json_str, out_buf, buf_cap, written)
}

// ── Session high-level FFI ────────────────────────────────────────────────────

/// Register a new session. kind is the session type (e.g. "claude", "sadhana").
/// Emits a SessionEvent("register") op to the log and updates the in-memory registry.
#[no_mangle]
pub extern "C" fn cf_session_register(
    h: *mut CfHandle,
    session_id: *const c_char,
    kind: *const c_char,
    realm: *const c_char,
    now_ms: i64,
) -> c_int {
    if h.is_null() || session_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let session_id_str = unsafe {
        match CStr::from_ptr(session_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let kind_str = if kind.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(kind).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };
    let realm_str = if realm.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };

    let payload_json = format!(r#"{{"kind":"{}"}}"#, kind_str).into_bytes();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::SessionEvent(SessionEventOp {
        event_id,
        session_id: session_id_str.to_string(),
        kind: "register".to_string(),
        payload_json,
        realm: realm_str.to_string(),
        ts_ms: now_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle.field.session_registry.write().register(
        session_id_str.to_string(),
        kind_str.to_string(),
        realm_str.to_string(),
        now_ms,
    );
    handle.ok()
}

/// Update the last-heartbeat timestamp for an active session.
#[no_mangle]
pub extern "C" fn cf_session_heartbeat(
    h: *mut CfHandle,
    session_id: *const c_char,
    now_ms: i64,
) -> c_int {
    if h.is_null() || session_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let session_id_str = unsafe {
        match CStr::from_ptr(session_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::SessionEvent(SessionEventOp {
        event_id,
        session_id: session_id_str.to_string(),
        kind: "heartbeat".to_string(),
        payload_json: Vec::new(),
        realm: String::new(),
        ts_ms: now_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .session_registry
        .write()
        .heartbeat(session_id_str, now_ms);
    handle.ok()
}

/// Mark a session as closed.
#[no_mangle]
pub extern "C" fn cf_session_deregister(
    h: *mut CfHandle,
    session_id: *const c_char,
    now_ms: i64,
) -> c_int {
    if h.is_null() || session_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let session_id_str = unsafe {
        match CStr::from_ptr(session_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::SessionEvent(SessionEventOp {
        event_id,
        session_id: session_id_str.to_string(),
        kind: "deregister".to_string(),
        payload_json: Vec::new(),
        realm: String::new(),
        ts_ms: now_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .session_registry
        .write()
        .deregister(session_id_str);
    handle.ok()
}

// ── Transcript high-level FFI ─────────────────────────────────────────────────

/// Register a new transcript for a session.
#[no_mangle]
pub extern "C" fn cf_transcript_register(
    h: *mut CfHandle,
    transcript_id: *const c_char,
    session_id: *const c_char,
) -> c_int {
    if h.is_null() || transcript_id.is_null() || session_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let transcript_id_str = unsafe {
        match CStr::from_ptr(transcript_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let session_id_str = unsafe {
        match CStr::from_ptr(session_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    use std::time::{SystemTime, UNIX_EPOCH};
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let payload_str = format!(r#"{{"transcript_id":"{}"}}"#, transcript_id_str);
    let payload_json = payload_str.as_bytes().to_vec();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::TranscriptEvent(TranscriptEventOp {
        event_id,
        session_id: session_id_str.to_string(),
        kind: "register".to_string(),
        payload_json,
        realm: String::new(),
        ts_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    {
        let mut registry = handle.field.transcript_registry.write();
        registry.set_session_event(session_id_str, "register", payload_str);
        registry.register(transcript_id_str.to_string(), session_id_str.to_string());
    }
    handle.ok()
}

/// Update transcript completion progress (0.0–100.0).
#[no_mangle]
pub extern "C" fn cf_transcript_update_progress(
    h: *mut CfHandle,
    transcript_id: *const c_char,
    progress_pct: f32,
) -> c_int {
    if h.is_null() || transcript_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let transcript_id_str = unsafe {
        match CStr::from_ptr(transcript_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    use std::time::{SystemTime, UNIX_EPOCH};
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let session_id = {
        let registry = handle.field.transcript_registry.read();
        registry
            .get(transcript_id_str)
            .map(|record| record.session_id.clone())
    };
    let session_id = match session_id {
        Some(session_id) => session_id,
        None => return handle.err(format!("unknown transcript_id: {}", transcript_id_str)),
    };

    let payload_str = format!(
        r#"{{"transcript_id":"{}","progress_pct":{}}}"#,
        transcript_id_str, progress_pct
    );
    let payload_json = payload_str.as_bytes().to_vec();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::TranscriptEvent(TranscriptEventOp {
        event_id,
        session_id: session_id.clone(),
        kind: "update_progress".to_string(),
        payload_json,
        realm: String::new(),
        ts_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    {
        let mut registry = handle.field.transcript_registry.write();
        registry.set_session_event(&session_id, "update_progress", payload_str);
        registry.update_progress(transcript_id_str, progress_pct);
    }
    handle.ok()
}

/// Add a turn to a transcript. Returns the assigned turn_id via out_turn_id.
/// content_ptr/content_len are the UTF-8 turn content (not NUL-terminated).
#[no_mangle]
pub extern "C" fn cf_transcript_add_turn(
    h: *mut CfHandle,
    transcript_id: *const c_char,
    role: *const c_char,
    content_ptr: *const u8,
    content_len: usize,
    ts_ms: i64,
    out_turn_id: *mut u64,
) -> c_int {
    if h.is_null() || transcript_id.is_null() || out_turn_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let transcript_id_str = unsafe {
        match CStr::from_ptr(transcript_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let role_str = if role.is_null() {
        "user"
    } else {
        unsafe {
            match CStr::from_ptr(role).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };
    let content = if content_ptr.is_null() || content_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(content_ptr, content_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(e) => return handle.err(e),
        }
    };

    let session_id = {
        let registry = handle.field.transcript_registry.read();
        registry
            .get(transcript_id_str)
            .map(|record| record.session_id.clone())
    };
    let session_id = match session_id {
        Some(session_id) => session_id,
        None => return handle.err(format!("unknown transcript_id: {}", transcript_id_str)),
    };

    let payload_str = serde_json::json!({
        "transcript_id": transcript_id_str,
        "role": role_str,
        "content": content,
    })
    .to_string();
    let payload_json = payload_str.as_bytes().to_vec();

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::TranscriptEvent(TranscriptEventOp {
        event_id,
        session_id: session_id.clone(),
        kind: "add_turn".to_string(),
        payload_json,
        realm: String::new(),
        ts_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    let turn_id = {
        let mut registry = handle.field.transcript_registry.write();
        registry.set_session_event(&session_id, "add_turn", payload_str);
        registry.add_turn(transcript_id_str, role_str.to_string(), content, ts_ms)
    };
    unsafe {
        *out_turn_id = turn_id;
    }
    handle.ok()
}

// ── Task / Sadhana / Dream high-level FFI ────────────────────────────────────

/// Create a task, sadhana, or dream.
/// kind: "task" | "sadhana" | "dream" (or any custom type).
/// payload_json: arbitrary UTF-8 JSON metadata (may be null/0 for empty).
#[no_mangle]
pub extern "C" fn cf_task_create(
    h: *mut CfHandle,
    task_id: *const c_char,
    kind: *const c_char,
    payload_json: *const u8,
    payload_len: usize,
    now_ms: i64,
    fencing_token: u64,
) -> c_int {
    if h.is_null() || task_id.is_null() || kind.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let task_id_str = unsafe {
        match CStr::from_ptr(task_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let payload_str = if payload_json.is_null() || payload_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(payload_json, payload_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(e) => return handle.err(e),
        }
    };

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::TaskEvent(TaskEventOp {
        event_id,
        task_type: kind_str.to_string(),
        task_id: task_id_str.to_string(),
        kind: "create".to_string(),
        payload_json: payload_str.as_bytes().to_vec(),
        realm: String::new(),
        ts_ms: now_ms,
        fencing_token,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle.field.task_registry.write().create(
        task_id_str.to_string(),
        kind_str.to_string(),
        payload_str,
        now_ms,
        fencing_token,
    );
    handle.ok()
}

/// Transition a task's status.
/// new_status: "start" | "pause" | "resume" | "complete" | "fail"
#[no_mangle]
pub extern "C" fn cf_task_transition(
    h: *mut CfHandle,
    task_id: *const c_char,
    new_status: *const c_char,
    now_ms: i64,
    fencing_token: u64,
) -> c_int {
    if h.is_null() || task_id.is_null() || new_status.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let task_id_str = unsafe {
        match CStr::from_ptr(task_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let status_str = unsafe {
        match CStr::from_ptr(new_status).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    // Determine task_type from registry for the log op.
    let task_type = handle
        .field
        .task_registry
        .read()
        .get(task_id_str)
        .map(|t| t.kind.clone())
        .unwrap_or_default();

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::TaskEvent(TaskEventOp {
        event_id,
        task_type,
        task_id: task_id_str.to_string(),
        kind: status_str.to_string(),
        payload_json: Vec::new(),
        realm: String::new(),
        ts_ms: now_ms,
        fencing_token,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    let transitioned = handle
        .field
        .task_registry
        .write()
        .transition(task_id_str, status_str, now_ms, fencing_token);
    if !transitioned {
        // Transition rejected: stale fencing token or unknown task_id — signal error to caller
        return handle.err("task transition rejected: stale fencing token or unknown task_id");
    }
    handle.ok()
}

/// List tasks as a JSON array written into buf.
/// kind_filter: null means all; "task"/"sadhana"/"dream" filters by kind.
/// active_only: 1 = only pending/running/paused; 0 = all.
/// Returns 0 on success, -2 if buf too small.
#[no_mangle]
pub extern "C" fn cf_task_list(
    h: *mut CfHandle,
    kind_filter: *const c_char,
    active_only: u8,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let filter_str = if kind_filter.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(kind_filter).to_str() {
                Ok(s) if !s.is_empty() => Some(s),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };

    // Collect owned data before dropping the read guard so we can call handle methods.
    let json_val: Vec<serde_json::Value> = {
        let registry = handle.field.task_registry.read();
        let records: Vec<_> = if active_only != 0 {
            if let Some(kind) = filter_str {
                registry
                    .list_active()
                    .into_iter()
                    .filter(|t| t.kind == kind)
                    .collect()
            } else {
                registry.list_active()
            }
        } else if let Some(kind) = filter_str {
            registry.list_by_kind(kind)
        } else {
            registry.list_all()
        };
        records
            .iter()
            .map(|t| {
                serde_json::json!({
                    "task_id": t.task_id,
                    "kind": t.kind,
                    "status": t.status.as_str(),
                    "payload_json": t.payload_json,
                    "created_at_ms": t.created_at_ms,
                    "updated_at_ms": t.updated_at_ms,
                })
            })
            .collect()
    }; // registry guard dropped here

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

// ── User Model FFI ────────────────────────────────────────────────────────────

/// Upsert a user model entity (profile, goal, habit, anticipation, calibration).
/// payload_json may be null/0 for empty payload.
#[no_mangle]
pub extern "C" fn cf_user_model_upsert(
    h: *mut CfHandle,
    entity_id: *const c_char,
    entity_type: *const c_char,
    payload_json: *const u8,
    payload_len: usize,
    now_ms: i64,
) -> c_int {
    if h.is_null() || entity_id.is_null() || entity_type.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let entity_id_str = unsafe {
        match CStr::from_ptr(entity_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let entity_type_str = unsafe {
        match CStr::from_ptr(entity_type).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let payload_str = if payload_json.is_null() || payload_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(payload_json, payload_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(e) => return handle.err(e),
        }
    };

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::UserModelEvent(UserModelEventOp {
        event_id,
        entity_type: entity_type_str.to_string(),
        entity_id: entity_id_str.to_string(),
        kind: "upsert".to_string(),
        payload_json: payload_str.as_bytes().to_vec(),
        ts_ms: now_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle.field.user_model_registry.write().upsert(
        entity_id_str.to_string(),
        entity_type_str.to_string(),
        payload_str,
        now_ms,
    );
    handle.ok()
}

/// Record an observation of a user model entity (increments count, updates timestamp).
#[no_mangle]
pub extern "C" fn cf_user_model_observe(
    h: *mut CfHandle,
    entity_id: *const c_char,
    now_ms: i64,
) -> c_int {
    if h.is_null() || entity_id.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let entity_id_str = unsafe {
        match CStr::from_ptr(entity_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::UserModelEvent(UserModelEventOp {
        event_id,
        entity_type: String::new(),
        entity_id: entity_id_str.to_string(),
        kind: "observe".to_string(),
        payload_json: Vec::new(),
        ts_ms: now_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .user_model_registry
        .write()
        .observe(entity_id_str, now_ms);
    handle.ok()
}

/// List user model entries as a JSON array written into buf.
/// entity_type_filter: null means all; otherwise filter by entity_type.
/// Returns 0 on success, -2 if buf too small.
#[no_mangle]
pub extern "C" fn cf_user_model_list(
    h: *mut CfHandle,
    entity_type_filter: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let filter_str = if entity_type_filter.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(entity_type_filter).to_str() {
                Ok(s) if !s.is_empty() => Some(s),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };

    let json_val: Vec<serde_json::Value> = {
        let registry = handle.field.user_model_registry.read();
        let entries: Vec<_> = if let Some(etype) = filter_str {
            registry.list_by_type(etype)
        } else {
            registry.list_all()
        };
        entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "payload_json": e.payload_json,
                    "updated_at_ms": e.updated_at_ms,
                    "observation_count": e.observation_count,
                })
            })
            .collect()
    };

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

// ── Theme FFI ─────────────────────────────────────────────────────────────────

/// Create a new named theme.
#[no_mangle]
pub extern "C" fn cf_theme_create(h: *mut CfHandle, theme_id: u64, name: *const c_char) -> c_int {
    if h.is_null() || name.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let payload = serde_json::json!({ "name": name_str }).to_string();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::ThemeEvent(ThemeEventOp {
        event_id,
        kind: "create".to_string(),
        theme_id,
        payload_json: payload.as_bytes().to_vec(),
        ts_ms: 0,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .theme_organ
        .write()
        .create(theme_id, name_str.to_string());
    handle.ok()
}

/// Update the centroid (JSON array of floats) for a theme.
/// centroid_json may be null/0 to clear the centroid.
#[no_mangle]
pub extern "C" fn cf_theme_update_centroid(
    h: *mut CfHandle,
    theme_id: u64,
    centroid_json: *const u8,
    centroid_len: usize,
) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let centroid_str = if centroid_json.is_null() || centroid_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(centroid_json, centroid_len) };
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(e) => return handle.err(e),
        }
    };

    let payload = serde_json::json!({ "centroid_json": centroid_str }).to_string();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::ThemeEvent(ThemeEventOp {
        event_id,
        kind: "update_centroid".to_string(),
        theme_id,
        payload_json: payload.as_bytes().to_vec(),
        ts_ms: 0,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .theme_organ
        .write()
        .update_centroid(theme_id, centroid_str);
    handle.ok()
}

/// Assign a memory to a theme.
#[no_mangle]
pub extern "C" fn cf_theme_assign_member(h: *mut CfHandle, theme_id: u64, memory_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let payload = serde_json::json!({ "memory_id": memory_id }).to_string();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::ThemeEvent(ThemeEventOp {
        event_id,
        kind: "assign_member".to_string(),
        theme_id,
        payload_json: payload.as_bytes().to_vec(),
        ts_ms: 0,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .theme_organ
        .write()
        .assign_member(theme_id, memory_id);
    handle.ok()
}

/// Remove a memory from a theme.
#[no_mangle]
pub extern "C" fn cf_theme_remove_member(h: *mut CfHandle, theme_id: u64, memory_id: u64) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let payload = serde_json::json!({ "memory_id": memory_id }).to_string();
    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::ThemeEvent(ThemeEventOp {
        event_id,
        kind: "remove_member".to_string(),
        theme_id,
        payload_json: payload.as_bytes().to_vec(),
        ts_ms: 0,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle
        .field
        .theme_organ
        .write()
        .remove_member(theme_id, memory_id);
    handle.ok()
}

/// List all themes as a JSON array written into buf.
/// Each element: {theme_id, name, member_count, centroid_json}
/// Returns 0 on success, -2 if buf too small.
#[no_mangle]
pub extern "C" fn cf_theme_list(
    h: *mut CfHandle,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let json_val: Vec<serde_json::Value> = {
        let organ = handle.field.theme_organ.read();
        organ
            .list_all()
            .iter()
            .map(|t| {
                serde_json::json!({
                    "theme_id": t.theme_id,
                    "name": t.name,
                    "member_count": t.member_ids.len(),
                    "centroid_json": t.centroid,
                })
            })
            .collect()
    };

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

/// Get a single theme by ID as JSON. Returns 0 on success, 1 if not found, -1 on error.
#[no_mangle]
pub extern "C" fn cf_theme_get(
    h: *mut CfHandle,
    theme_id: u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let theme_val = {
        let organ = handle.field.theme_organ.read();
        organ.get(theme_id).map(|t| {
            serde_json::json!({
                "theme_id": t.theme_id,
                "name": t.name,
                "realm": t.realm,
                "coherence": t.coherence,
                "member_count": t.member_ids.len(),
                "created_at": t.created_at,
            })
        })
    };
    let json_str = match theme_val {
        None => return 1,
        Some(v) => match serde_json::to_string(&v) {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        },
    };

    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

/// Get theme stats as JSON. realm="" means all realms.
#[no_mangle]
pub extern "C" fn cf_theme_stats(
    h: *mut CfHandle,
    realm: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let realm_str = if realm.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };

    let total_memory_count = handle.field.payloads.read().len();
    let stats = handle
        .field
        .theme_organ
        .read()
        .stats(realm_str, total_memory_count);

    let json_str = match serde_json::to_string(&serde_json::json!({
        "total_themes": stats.total_themes,
        "total_memberships": stats.total_memberships,
        "orphan_count": stats.orphan_count,
        "avg_size": stats.avg_size,
        "avg_coherence": stats.avg_coherence,
    })) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

/// Find top-k themes by embedding similarity. Returns JSON array of {theme_id, score}.
#[no_mangle]
pub extern "C" fn cf_theme_recall(
    h: *mut CfHandle,
    embedding_ptr: *const f32,
    embedding_len: usize,
    k: usize,
    realm: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || embedding_ptr.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let embedding = unsafe { std::slice::from_raw_parts(embedding_ptr, embedding_len) };

    let realm_str = if realm.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };

    let hits = handle
        .field
        .theme_organ
        .read()
        .recall_by_embedding(embedding, k, realm_str);

    let json_val: Vec<serde_json::Value> = hits
        .iter()
        .map(|(tid, score)| {
            serde_json::json!({
                "theme_id": tid,
                "score": score,
            })
        })
        .collect();

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

/// Run theme maintenance (split/merge). Returns JSON ThemeMaintenanceResult.
#[no_mangle]
pub extern "C" fn cf_theme_maintain(
    h: *mut CfHandle,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let embeddings: std::collections::HashMap<u64, Vec<f32>> = {
        let payloads = handle.field.payloads.read();
        let idx = handle.field.semantic_idx.read();
        payloads
            .iter()
            .map(|(id, p)| {
                let e = idx
                    .get_embedding(*id)
                    .map(|e| e.to_vec())
                    .unwrap_or_else(|| p.embedding.clone());
                (*id, e)
            })
            .collect()
    };

    let result = handle.field.theme_organ.write().maintain(&embeddings);

    let json_str = match serde_json::to_string(&serde_json::json!({
        "themes_split": result.themes_split,
        "themes_merged": result.themes_merged,
        "memories_reassigned": result.memories_reassigned,
    })) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

/// Assign orphan memories to themes. Returns JSON {assigned, remaining}.
#[no_mangle]
pub extern "C" fn cf_theme_assign_orphans(
    h: *mut CfHandle,
    batch_size: usize,
    realm: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let realm_str: String = if realm.is_null() {
        String::new()
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) => s.to_string(),
                Err(e) => return handle.err(e),
            }
        }
    };

    let (all_memory_ids, embeddings): (Vec<u64>, std::collections::HashMap<u64, Vec<f32>>) = {
        let payloads = handle.field.payloads.read();
        let idx = handle.field.semantic_idx.read();
        let ids: Vec<u64> = payloads.keys().copied().collect();
        let embs: std::collections::HashMap<u64, Vec<f32>> = payloads
            .iter()
            .map(|(id, p)| {
                let e = idx
                    .get_embedding(*id)
                    .map(|e| e.to_vec())
                    .unwrap_or_else(|| p.embedding.clone());
                (*id, e)
            })
            .collect();
        (ids, embs)
    };

    let (assigned, remaining) = handle.field.theme_organ.write().assign_orphans(
        &all_memory_ids,
        &embeddings,
        &realm_str,
        batch_size,
        0.7,
    );

    let json_str = match serde_json::to_string(&serde_json::json!({
        "assigned": assigned,
        "remaining": remaining,
    })) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };

    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

// ── Analytics high-level FFI ──────────────────────────────────────────────────

/// Append an analytics entry. Emits an AnalyticsEvent op to the log and
/// appends to the in-memory AnalyticsRegistry.
/// kind: event kind string (e.g. "exposure", "recall_query").
/// entity_id: session id or other entity identifier.
/// payload_json / payload_len: JSON payload bytes.
/// ts_ms: timestamp in milliseconds since Unix epoch.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_analytics_append(
    h: *mut CfHandle,
    kind: *const c_char,
    entity_id: *const c_char,
    payload_json: *const u8,
    payload_len: usize,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || kind.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let entity_id_str = if entity_id.is_null() {
        ""
    } else {
        unsafe {
            match CStr::from_ptr(entity_id).to_str() {
                Ok(s) => s,
                Err(e) => return handle.err(e),
            }
        }
    };
    let payload = if payload_json.is_null() || payload_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(payload_json, payload_len) }.to_vec()
    };
    let payload_str = match std::str::from_utf8(&payload) {
        Ok(s) => s.to_string(),
        Err(e) => return handle.err(e),
    };

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::AnalyticsEvent(AnalyticsEventOp {
        event_id,
        kind: kind_str.to_string(),
        session_id: entity_id_str.to_string(),
        payload_json: payload,
        ts_ms,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    handle.field.analytics_registry.write().append(
        kind_str.to_string(),
        entity_id_str.to_string(),
        payload_str,
        ts_ms,
    );
    handle.ok()
}

/// Write the most recent `limit` analytics entries as a JSON array into buf.
/// Returns 0 on success, -2 if buf too small, -1 on other error.
#[no_mangle]
pub extern "C" fn cf_analytics_recent(
    h: *mut CfHandle,
    limit: usize,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let json_val: Vec<serde_json::Value> = {
        let registry = handle.field.analytics_registry.read();
        registry
            .recent(limit)
            .into_iter()
            .map(|e| {
                serde_json::json!({
                    "id":           e.id,
                    "kind":         e.kind,
                    "entity_id":    e.entity_id,
                    "payload_json": e.payload_json,
                    "ts_ms":        e.ts_ms,
                })
            })
            .collect()
    };

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    handle.ok()
}

// ── New high-level query FFI (Phase 0 migration) ─────────────────────────────

/// Helper: serialize JSON string into caller-allocated buffer.
/// Returns 0 on success, -2 if buf too small.
fn write_json_buf(json_str: &str, buf: *mut u8, buf_cap: usize, written: *mut usize) -> c_int {
    let bytes = json_str.as_bytes();
    if bytes.len() > buf_cap {
        return -2;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
        *written = bytes.len();
    }
    0
}

/// 1. Filtered recall — returns JSON array of {id, content, kind, realm, confidence, strength, ts_ms}.
/// Filters by kind (null = any), realm (null = any), min_confidence, min_strength.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_recall_filtered(
    h: *mut CfHandle,
    kind: *const c_char,
    realm: *const c_char,
    min_confidence: f32,
    min_strength: f32,
    limit: usize,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let kind_filter = if kind.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(kind).to_str() {
                Ok(s) if !s.is_empty() => Some(s),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };
    let realm_filter = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) if !s.is_empty() => Some(s),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };

    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let payloads = handle.field.payloads.read();
    let states = handle.field.states.read();

    let mut results: Vec<serde_json::Value> = Vec::new();
    for (mid, payload) in payloads.iter() {
        if let Some(state) = states.get(mid) {
            if state.deleted {
                continue;
            }
            if state.confidence < min_confidence {
                continue;
            }
            let eff_strength = state.effective_strength(now);
            if eff_strength < min_strength {
                continue;
            }
            if let Some(k) = kind_filter {
                if payload.kind != k {
                    continue;
                }
            }
            if let Some(r) = realm_filter {
                if payload.realm != r {
                    continue;
                }
            }
            let content_str = String::from_utf8_lossy(&payload.content);
            results.push(serde_json::json!({
                "id": mid,
                "content": content_str,
                "kind": payload.kind,
                "realm": payload.realm,
                "confidence": state.confidence,
                "strength": eff_strength,
                "ts_ms": payload.created_at_ms,
            }));
            if results.len() >= limit {
                break;
            }
        }
    }

    drop(payloads);
    drop(states);

    let json_str = match serde_json::to_string(&results) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// 2. Paginated memory listing sorted by strength/recency/confidence.
/// sort_by: "strength" | "recency" | "confidence" (default: "recency").
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_list_memories(
    h: *mut CfHandle,
    kind: *const c_char,
    realm: *const c_char,
    sort_by: *const c_char,
    limit: usize,
    offset: usize,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let kind_filter = if kind.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(kind).to_str() {
                Ok(s) if !s.is_empty() => Some(s.to_string()),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };
    let realm_filter = if realm.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm).to_str() {
                Ok(s) if !s.is_empty() => Some(s.to_string()),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };
    let sort_str = if sort_by.is_null() {
        "recency"
    } else {
        unsafe {
            match CStr::from_ptr(sort_by).to_str() {
                Ok(s) if !s.is_empty() => s,
                Ok(_) => "recency",
                Err(e) => return handle.err(e),
            }
        }
    };

    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let payloads = handle.field.payloads.read();
    let states = handle.field.states.read();

    let mut entries: Vec<(
        u64,
        &crate::payload::MemoryPayload,
        &crate::state::MemoryState,
    )> = Vec::new();
    for (mid, payload) in payloads.iter() {
        if let Some(state) = states.get(mid) {
            if state.deleted {
                continue;
            }
            if let Some(ref k) = kind_filter {
                if payload.kind != *k {
                    continue;
                }
            }
            if let Some(ref r) = realm_filter {
                if payload.realm != *r {
                    continue;
                }
            }
            entries.push((*mid, payload, state));
        }
    }

    match sort_str {
        "strength" => entries.sort_by(|a, b| {
            b.2.effective_strength(now)
                .partial_cmp(&a.2.effective_strength(now))
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        "confidence" => entries.sort_by(|a, b| {
            b.2.confidence
                .partial_cmp(&a.2.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        _ => entries.sort_by(|a, b| b.1.created_at_ms.cmp(&a.1.created_at_ms)),
    }

    let triplets = handle.field.triplet_store.read();
    let page: Vec<serde_json::Value> = entries
        .iter()
        .skip(offset)
        .take(limit)
        .map(|(mid, payload, state)| {
            let content_str = String::from_utf8_lossy(&payload.content);
            let mid_str = mid.to_string();
            let tags: Vec<&str> = triplets
                .query_subject(&mid_str, now)
                .into_iter()
                .filter(|t| t.predicate == "tagged")
                .map(|t| t.object.as_str())
                .collect();
            serde_json::json!({
                "id": mid,
                "content": content_str,
                "kind": payload.kind,
                "realm": payload.realm,
                "confidence": state.confidence,
                "strength": state.effective_strength(now),
                "ts_ms": payload.created_at_ms,
                "pinned": state.pinned,
                "tags": tags,
            })
        })
        .collect();

    drop(triplets);
    drop(payloads);
    drop(states);

    let json_str = match serde_json::to_string(&page) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// 3. Aggregate stats: count_by_kind, avg_confidence, avg_strength, total.
/// realm_filter: null = all realms, otherwise filter by realm.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_memory_stats(
    h: *mut CfHandle,
    realm_filter: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let realm_str = if realm_filter.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(realm_filter).to_str() {
                Ok(s) if !s.is_empty() => Some(s),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };

    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let payloads = handle.field.payloads.read();
    let states = handle.field.states.read();

    let mut count_by_kind: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut total: usize = 0;
    let mut sum_confidence: f64 = 0.0;
    let mut sum_strength: f64 = 0.0;

    for (mid, payload) in payloads.iter() {
        if let Some(state) = states.get(mid) {
            if state.deleted {
                continue;
            }
            if let Some(r) = realm_str {
                if payload.realm != r {
                    continue;
                }
            }
            total += 1;
            sum_confidence += state.confidence as f64;
            sum_strength += state.effective_strength(now) as f64;
            *count_by_kind.entry(payload.kind.clone()).or_insert(0) += 1;
        }
    }

    drop(payloads);
    drop(states);

    let avg_confidence = if total > 0 {
        sum_confidence / total as f64
    } else {
        0.0
    };
    let avg_strength = if total > 0 {
        sum_strength / total as f64
    } else {
        0.0
    };

    let triplet_count = handle.field.triplet_store.read().triplet_count();

    let json_str = match serde_json::to_string(&serde_json::json!({
        "total": total,
        "count_by_kind": count_by_kind,
        "avg_confidence": avg_confidence,
        "avg_strength": avg_strength,
        "total_triplets": triplet_count,
    })) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// Per-realm embedding geometry stats (effective dimensionality, isotropy, mean cosine sim).
/// Returns JSON array into buf. Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_spectral_stats_by_realm(
    h: *mut CfHandle,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let json_str = handle.field.spectral_stats_by_realm();
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// Save spectral snapshot for temporal drift tracking.
/// Returns 0 on success. Writes filename into buf.
#[no_mangle]
pub extern "C" fn cf_save_spectral_snapshot(
    h: *mut CfHandle,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    match handle.field.save_spectral_snapshot() {
        Ok(filename) => write_json_buf(&format!("\"{}\"", filename), buf, buf_cap, written),
        Err(e) => handle.err(e),
    }
}

/// Get spectral drift since last snapshot. Returns JSON into buf.
#[no_mangle]
pub extern "C" fn cf_spectral_drift(
    h: *mut CfHandle,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let json_str = handle.field.spectral_drift();
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// Trim trailing whitespace from realm names. Returns count of fixed memories.
#[no_mangle]
pub extern "C" fn cf_trim_realm_names(h: *mut CfHandle) -> i64 {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    handle.field.trim_realm_names() as i64
}

/// Bulk-remap realms per a JSON `{old_realm: new_realm}` object. `dry_run`
/// computes the census without mutating; a real run mutates + forces a snapshot.
/// Writes a JSON summary to `buf`. Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_remap_realms(
    h: *mut CfHandle,
    mapping_json: *const c_char,
    dry_run: bool,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || mapping_json.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let map_str = match unsafe { CStr::from_ptr(mapping_json).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let mapping: std::collections::HashMap<String, String> =
        match serde_json::from_str(map_str) {
            Ok(m) => m,
            Err(e) => return handle.err(e),
        };
    let json_str = handle.field.remap_realms(&mapping, dry_run);
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// 4. Get single task by ID (JSON payload).
/// Returns 0 on success, 1 if not found, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_task_get(
    h: *mut CfHandle,
    task_id: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || task_id.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let task_id_str = unsafe {
        match CStr::from_ptr(task_id).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let json_val = {
        let registry = handle.field.task_registry.read();
        match registry.get(task_id_str) {
            Some(t) => serde_json::json!({
                "task_id": t.task_id,
                "kind": t.kind,
                "status": t.status.as_str(),
                "payload_json": t.payload_json,
                "created_at_ms": t.created_at_ms,
                "updated_at_ms": t.updated_at_ms,
            }),
            None => return 1,
        }
    };

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// 5. Update task payload (returns 0=ok, -1=not found/error).
#[no_mangle]
pub extern "C" fn cf_task_update_payload(
    h: *mut CfHandle,
    task_id: *const c_char,
    payload_json: *const c_char,
    now_ms: i64,
) -> i32 {
    if h.is_null() || task_id.is_null() || payload_json.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let task_id_str = unsafe {
        match CStr::from_ptr(task_id).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.err(e);
                return -1;
            }
        }
    };
    let payload_str = unsafe {
        match CStr::from_ptr(payload_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.err(e);
                return -1;
            }
        }
    };

    let task_type = handle
        .field
        .task_registry
        .read()
        .get(task_id_str)
        .map(|t| t.kind.clone())
        .unwrap_or_default();

    if task_type.is_empty() {
        handle.err("task not found");
        return -1;
    }

    let event_id = handle.field.event_id_alloc.fetch_add(1, Ordering::Relaxed);
    let op = Op::TaskEvent(TaskEventOp {
        event_id,
        task_type,
        task_id: task_id_str.to_string(),
        kind: "update_payload".to_string(),
        payload_json: payload_str.as_bytes().to_vec(),
        realm: String::new(),
        ts_ms: now_ms,
        fencing_token: 0,
    });

    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        handle.err(e);
        return -1;
    }
    if handle.field.task_registry.write().update_payload(
        task_id_str,
        payload_str.to_string(),
        now_ms,
    ) {
        handle.ok();
        0
    } else {
        handle.err("task not found");
        -1
    }
}

/// 6. List sessions (JSON array). active_only=1 filters by active status.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_session_list(
    h: *mut CfHandle,
    active_only: i32,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let json_val: Vec<serde_json::Value> = {
        let registry = handle.field.session_registry.read();
        let records: Vec<&crate::organ::session::SessionRecord> = if active_only != 0 {
            registry.list_active()
        } else {
            registry.list_all()
        };
        records
            .iter()
            .map(|s| {
                serde_json::json!({
                    "session_id": s.session_id,
                    "kind": s.kind,
                    "realm": s.realm,
                    "started_at_ms": s.started_at_ms,
                    "last_heartbeat_ms": s.last_heartbeat_ms,
                    "status": s.status,
                })
            })
            .collect()
    };

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// 7. List transcripts (JSON array, most recent first).
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_transcript_list(
    h: *mut CfHandle,
    limit: usize,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let json_val: Vec<serde_json::Value> = {
        let registry = handle.field.transcript_registry.read();
        let mut records: Vec<&crate::organ::transcript::TranscriptRecord> = registry.list_all();
        records.sort_by(|a, b| {
            let a_ts = a.turns.last().map(|t| t.ts_ms).unwrap_or(0);
            let b_ts = b.turns.last().map(|t| t.ts_ms).unwrap_or(0);
            b_ts.cmp(&a_ts)
        });
        records
            .iter()
            .take(limit)
            .map(|t| {
                serde_json::json!({
                    "transcript_id": t.transcript_id,
                    "session_id": t.session_id,
                    "progress_pct": t.progress_pct,
                    "turn_count": t.turns.len(),
                })
            })
            .collect()
    };

    let json_str = match serde_json::to_string(&json_val) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// 8. Get memory metadata by ID (JSON: kind, realm, confidence, strength, ts_ms, pinned).
/// Returns 0 on success, 1 if not found/deleted, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_get_memory_metadata(
    h: *mut CfHandle,
    memory_id: u64,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let payloads = handle.field.payloads.read();
    let states = handle.field.states.read();

    let payload = match payloads.get(&memory_id) {
        Some(p) => p,
        None => return 1,
    };
    let state = match states.get(&memory_id) {
        Some(s) if !s.deleted => s,
        _ => return 1,
    };

    let status_str = format!("{:?}", state.status);
    let epistemic_str = format!("{:?}", state.epistemic_status);

    let json_str = match serde_json::to_string(&serde_json::json!({
        "id": memory_id,
        "kind": payload.kind,
        "realm": payload.realm,
        "confidence": state.confidence,
        "strength": state.effective_strength(now),
        "ts_ms": payload.created_at_ms,
        "pinned": state.pinned,
        "tier": state.tier,
        "access_count": state.access_count,
        "decay_rate": state.decay_rate,
        "status": status_str,
        "epistemic_status": epistemic_str,
        "last_accessed_ms": state.last_accessed_ms,
        "last_strengthened_ms": state.last_strengthened_ms,
        "created_at_ms": state.created_at_ms,
        "last_state_op_ts_ms": state.last_state_op_ts_ms,
    })) {
        Ok(s) => s,
        Err(e) => {
            drop(payloads);
            drop(states);
            return handle.err(e);
        }
    };

    let rc = write_json_buf(&json_str, buf, buf_cap, written);
    drop(payloads);
    drop(states);
    if rc == 0 {
        handle.ok()
    } else {
        rc
    }
}

/// 9. Update memory kind field (returns 0=ok, -1=not found/error).
#[no_mangle]
pub extern "C" fn cf_set_realm(
    h: *mut CfHandle,
    memory_id: u64,
    new_realm: *const c_char,
) -> i32 {
    if h.is_null() || new_realm.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let realm_str = unsafe {
        match CStr::from_ptr(new_realm).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => { handle.err(e); return -1; }
        }
    };
    let old_realm = {
        let payloads = handle.field.payloads.read();
        match payloads.get(&memory_id) {
            Some(p) => p.realm.clone(),
            None => { drop(payloads); handle.err("memory not found"); return -1; }
        }
    };
    // Update payload realm
    {
        let mut payloads = handle.field.payloads.write();
        if let Some(p) = payloads.get_mut(&memory_id) {
            p.realm = realm_str.clone();
        }
    }
    // Update realm_members: remove from old, insert into new
    {
        let mut rm = handle.field.realm_members.write();
        if let Some(set) = rm.get_mut(&old_realm) {
            set.remove(&memory_id);
        }
        rm.entry(realm_str).or_default().insert(memory_id);
    }
    handle.ok();
    0
}

#[no_mangle]
pub extern "C" fn cf_update_memory_kind(
    h: *mut CfHandle,
    memory_id: u64,
    new_kind: *const c_char,
) -> i32 {
    if h.is_null() || new_kind.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let kind_str = unsafe {
        match CStr::from_ptr(new_kind).to_str() {
            Ok(s) => s,
            Err(e) => {
                handle.err(e);
                return -1;
            }
        }
    };

    // Log first for durability — without this, kind changes are lost on
    // crash between mutation and snapshot.
    let op = crate::ops::Op::UpdateMemoryKind(crate::ops::UpdateMemoryKindOp {
        memory_id,
        new_kind: kind_str.to_string(),
        op_ts_ms: crate::store::now_ms(),
    });
    let log_result = handle.field.log.write().append(&op);
    if let Err(e) = log_result {
        return handle.err(e);
    }

    let mut payloads = handle.field.payloads.write();
    match payloads.get_mut(&memory_id) {
        Some(p) => {
            p.kind = kind_str.to_string();
            drop(payloads);
            handle.ok();
            0
        }
        None => {
            drop(payloads);
            handle.err("memory not found");
            -1
        }
    }
}

/// 10. List all triplets where entity is subject OR object, with limit.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_list_triplets_for_entity(
    h: *mut CfHandle,
    entity: *const c_char,
    limit: usize,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || entity.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    let entity_str = unsafe {
        match CStr::from_ptr(entity).to_str() {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };

    let mut entries = handle.field.query_entity(entity_str).unwrap_or_default();
    entries.truncate(limit);

    // Reuse write_triplets_json (already defined above)
    write_triplets_json(entries, buf, buf_cap, written)
}

// ── Additional FFI for DuckDB removal migration ──────────────────────────────

/// List code files, optionally filtered by project. Returns JSON array.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_list_code_files(
    h: *mut CfHandle,
    project_filter: *const c_char,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let project = if project_filter.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(project_filter).to_str() {
                Ok(s) if !s.is_empty() => Some(s.to_string()),
                Ok(_) => None,
                Err(e) => return handle.err(e),
            }
        }
    };
    let files = handle.field.code_files.read();
    let result: Vec<serde_json::Value> = files
        .iter()
        .filter(|f| project.as_ref().map(|p| f.project == *p).unwrap_or(true))
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "path": f.path,
                "project": f.project,
                "mtime": f.mtime,
            })
        })
        .collect();
    drop(files);
    let json_str = match serde_json::to_string(&result) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    let rc = write_json_buf(&json_str, buf, buf_cap, written);
    if rc == 0 {
        handle.ok();
    }
    rc
}

/// Remove all code files and their associated symbols for a project.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_clear_project(h: *mut CfHandle, project: *const c_char) -> c_int {
    if h.is_null() || project.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let project_str = unsafe {
        match CStr::from_ptr(project).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return handle.err(e),
        }
    };
    // Log op first for durability
    let op = Op::ClearProject(ClearProjectOp {
        project: project_str.clone(),
    });
    let log_result = handle.field.log.write().append(&op);
    if let Err(e) = log_result {
        return handle.err(e);
    }
    // Apply in-memory
    let mut files = handle.field.code_files.write();
    let removed_paths = files.remove_by_project(&project_str);
    drop(files);
    let mut syms = handle.field.symbol_idx.write();
    let removed_ids = syms.remove_by_file_paths(&removed_paths);
    drop(syms);
    let mut cg = handle.field.call_graph.write();
    for id in removed_ids {
        cg.remove_symbol(id);
    }
    drop(cg);
    handle.ok()
}

/// Update description for a symbol by ID.
/// Returns 0 on success, 1 if not found, -1 on error.
#[no_mangle]
pub extern "C" fn cf_set_symbol_description(
    h: *mut CfHandle,
    symbol_id: u64,
    description: *const c_char,
    description_len: usize,
) -> c_int {
    if h.is_null() || description.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let desc = match unsafe {
        std::str::from_utf8(std::slice::from_raw_parts(
            description as *const u8,
            description_len,
        ))
    } {
        Ok(s) => s.to_string(),
        Err(e) => return handle.err(e),
    };
    {
        let syms = handle.field.symbol_idx.read();
        if syms.get(symbol_id).is_none() {
            return 1;
        }
    }
    let op = Op::UpdateSymbolDescription(UpdateSymbolDescriptionOp {
        symbol_id,
        description: desc.clone(),
        op_ts_ms: crate::store::now_ms(),
    });
    let log_result = handle.field.log.write().append(&op);
    if let Err(e) = log_result {
        return handle.err(e);
    }
    if let Some(sym) = handle.field.symbol_idx.write().get_mut(symbol_id) {
        sym.description = Some(desc);
    }
    handle.ok()
}

/// Update content + embedding for an existing memory.
/// Returns 0 on success, 1 if not found, -1 on error.
#[no_mangle]
pub extern "C" fn cf_update_memory_content(
    h: *mut CfHandle,
    id: u64,
    content: *const u8,
    content_len: usize,
    embedding: *const f32,
    embedding_len: usize,
) -> c_int {
    if h.is_null() || content.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let new_content = unsafe { std::slice::from_raw_parts(content, content_len) }.to_vec();
    let new_embedding: Vec<f32> = if embedding.is_null() || embedding_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(embedding, embedding_len) }.to_vec()
    };
    if !new_embedding.is_empty() && new_embedding.len() != crate::ops::EMBED_DIM {
        return handle.err(format!(
            "embedding length {} != EMBED_DIM {}",
            new_embedding.len(),
            crate::ops::EMBED_DIM
        ));
    }
    // Check existence before logging op
    if !handle.field.payloads.read().contains_key(&id) {
        return 1;
    }
    // Log for durability
    let op = Op::UpdateMemoryContent(UpdateMemoryContentOp {
        memory_id: id,
        content: new_content.clone(),
        embedding: new_embedding.clone(),
        op_ts_ms: crate::store::now_ms(),
    });
    let log_result = handle.field.log.write().append(&op);
    if let Err(e) = log_result {
        return handle.err(e);
    }
    // Apply in-memory; capture the memory's realm so the semantic-index upsert routes
    // to the correct per-realm HNSW. Must match apply_op's WAL-replay path (Some(realm)),
    // otherwise recall realm-filtering diverges for edited memories across a restart.
    let mem_realm = {
        let mut payloads = handle.field.payloads.write();
        match payloads.get_mut(&id) {
            Some(payload) => {
                payload.content = new_content.clone();
                handle
                    .field
                    .pld_mutations
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !new_embedding.is_empty() {
                    // semantic_idx (upserted below) is the embedding's single
                    // in-RAM home; drop any stale payload copy.
                    payload.embedding = Vec::new();
                }
                payload.realm.clone()
            }
            None => String::new(),
        }
    };
    if !new_embedding.is_empty() {
        handle.field.semantic_idx.write().upsert(id, new_embedding, Some(mem_realm.as_str()));
    }
    let content_str = String::from_utf8_lossy(&new_content).to_string();
    handle.field.keyword_idx.write().index(id, &content_str);
    // Re-encode cortical sparse code for updated content/embedding
    let _ = handle.field.encode_memory(id);
    // Span-lane re-link: changed content unlinks stale atoms then relinks
    // (watermark supersede). In-RAM; the periodic span flush persists.
    handle.field.span_link_memory(id, &content_str, &mem_realm);
    handle.ok()
}

/// List distinct realm names from non-deleted memories. Returns JSON string array.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_realm_list(
    h: *mut CfHandle,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let payloads = handle.field.payloads.read();
    let states = handle.field.states.read();
    let mut realms: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (mid, payload) in payloads.iter() {
        if let Some(state) = states.get(mid) {
            if !state.deleted {
                realms.insert(payload.realm.clone());
            }
        }
    }
    drop(payloads);
    drop(states);
    let mut list: Vec<String> = realms.into_iter().collect();
    list.sort_unstable();
    let json_str = match serde_json::to_string(&list) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// Recall memories filtered by kind, sorted by confidence descending.
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_recall_by_kind(
    h: *mut CfHandle,
    kind: *const c_char,
    limit: usize,
    buf: *mut u8,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || kind.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let kind_str = unsafe {
        match CStr::from_ptr(kind).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => return handle.err(e),
        }
    };
    // O(K log limit) via kind_members index — only iterate members of this kind,
    // and keep a min-heap of size `limit` on confidence.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Lock order: payloads → states → kind_members (struct order; matches
    // sync_foreign's write-guard acquisition).
    let payloads = handle.field.payloads.read();
    let states = handle.field.states.read();
    let kind_members = handle.field.kind_members.read();

    let empty_set;
    let members: &std::collections::HashSet<u64> = match kind_members.get(&kind_str) {
        Some(s) => s,
        None => { empty_set = std::collections::HashSet::new(); &empty_set }
    };

    let mut heap: BinaryHeap<Reverse<(u32, u64)>> = BinaryHeap::with_capacity(limit + 1);
    for &mid in members.iter() {
        let st = match states.get(&mid) { Some(s) if !s.deleted => s, _ => continue };
        let conf_bits = if st.confidence.is_nan() { 0 } else { st.confidence.to_bits() };
        if heap.len() < limit {
            heap.push(Reverse((conf_bits, mid)));
        } else if let Some(&Reverse((min_bits, _))) = heap.peek() {
            if conf_bits > min_bits {
                heap.pop();
                heap.push(Reverse((conf_bits, mid)));
            }
        }
    }
    let mut top: Vec<(u32, u64)> = heap.into_iter().map(|Reverse(t)| t).collect();
    top.sort_by(|a, b| b.0.cmp(&a.0));

    let page: Vec<serde_json::Value> = top
        .into_iter()
        .filter_map(|(conf_bits, mid)| {
            let payload = payloads.get(&mid)?;
            Some(serde_json::json!({
                "id": mid,
                "confidence": f32::from_bits(conf_bits),
                "content": String::from_utf8_lossy(&payload.content),
            }))
        })
        .collect();
    drop(payloads);
    drop(states);
    drop(kind_members);
    let json_str = match serde_json::to_string(&page) {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    write_json_buf(&json_str, buf, buf_cap, written)
}

/// Purge corrupt memories: empty/whitespace content or non-finite affect values.
/// Writes the count of purged memories to *out_purged.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_purge_corrupt(h: *mut CfHandle, out_purged: *mut usize) -> c_int {
    if h.is_null() || out_purged.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let to_purge: Vec<u64> = {
        let payloads = handle.field.payloads.read();
        let states = handle.field.states.read();
        payloads
            .iter()
            .filter(|(mid, payload)| {
                let deleted = states.get(mid).map(|s| s.deleted).unwrap_or(false);
                if deleted {
                    return false;
                }
                let content_str = String::from_utf8(payload.content.clone()).unwrap_or_default();
                let empty = content_str.trim().is_empty();
                let av = states.get(mid).map(|s| s.affect_valence).unwrap_or(0.0);
                let aa = states.get(mid).map(|s| s.affect_arousal).unwrap_or(0.0);
                let corrupt_affect = !av.is_finite() || !aa.is_finite()
                    || av.abs() > 1000.0 || aa.abs() > 1000.0;
                empty || corrupt_affect
            })
            .map(|(mid, _)| *mid)
            .collect()
    };
    let count = to_purge.len();
    for id in &to_purge {
        let _ = handle.field.forget(*id);
    }

    // Also remove orphaned semantic index entries (embedding exists but no payload)
    let orphaned: Vec<u64> = {
        let payloads = handle.field.payloads.read();
        let idx = handle.field.semantic_idx.read();
        idx.all_ids()
            .filter(|id| !payloads.contains_key(id))
            .collect()
    };
    let orphan_count = orphaned.len();
    for id in orphaned {
        handle.field.semantic_idx.write().remove(id);
    }

    unsafe { *out_purged = count + orphan_count; }
    handle.ok()
}

/// Return the 768-dim embeddings for a batch of memory IDs as JSON.
/// Output: {"embeddings": {"<id>": [f32, ...], ...}}
/// Missing IDs are silently omitted.
#[no_mangle]
pub unsafe extern "C" fn cf_get_memory_embeddings_batch(
    handle: *const CfHandle,
    ids: *const u64,
    ids_len: usize,
    out_buf: *mut u8,
    out_buf_len: usize,
    written: *mut usize,
) -> i32 {
    if handle.is_null() || ids.is_null() || out_buf.is_null() || written.is_null() {
        return -1;
    }
    let field = &(*handle).field;
    let id_slice = std::slice::from_raw_parts(ids, ids_len);
    let payloads = field.payloads.read();
    let idx = field.semantic_idx.read();
    let mut result: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();
    for &id in id_slice {
        if let Some(e) = idx.get_embedding(id) {
            result.insert(id.to_string(), e.to_vec());
        } else if let Some(payload) = payloads.get(&id) {
            result.insert(id.to_string(), payload.embedding.clone());
        }
    }
    let json = serde_json::json!({"embeddings": result});
    let json_str = match serde_json::to_string(&json) {
        Ok(s) => s,
        Err(_) => return -1,
    };
    write_json_buf(&json_str, out_buf, out_buf_len, written)
}

/// Durably record a completed recall event. Appends a RecordRecallBatchOp to
/// the WAL and applies the effects (touch, retrieval history, co-activation
/// stats, Hebbian edge strengthening) to in-memory state.
#[no_mangle]
pub unsafe extern "C" fn cf_record_recall_batch(
    handle: *mut CfHandle,
    ids: *const u64,
    ids_len: usize,
    centroid_q: *const i8,
    centroid_q_len: usize,
    centroid_scale: f32,
    context_hash: u64,
    ts_ms: i64,
    base_assoc_delta: f32,
) -> i32 {
    if handle.is_null() || ids.is_null() {
        return -1;
    }
    let h = &*handle;
    let id_slice = std::slice::from_raw_parts(ids, ids_len);
    // centroid_q is optional: null/zero-len → empty slice
    let cq_slice: &[i8] = if centroid_q.is_null() || centroid_q_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(centroid_q, centroid_q_len)
    };

    let op = Op::RecordRecallBatch(RecordRecallBatchOp {
        memory_ids: id_slice.to_vec(),
        centroid_q: cq_slice.to_vec(),
        centroid_scale,
        context_hash,
        ts_ms,
        base_assoc_delta,
    });

    let append_result = h.field.log.write().append(&op);
    match append_result {
        Ok(_seqno) => {
            // Apply to in-memory state immediately.
            let ctx = crate::state::RetrievalContext {
                centroid_q: cq_slice.to_vec(),
                scale: centroid_scale,
                context_hash,
                ts_ms,
            };
            {
                let mut states = h.field.states.write();
                for &mid in id_slice {
                    if let Some(state) = states.get_mut(&mid) {
                        state.access_count += 1;
                        state.last_accessed_ms = ts_ms;
                        state.retrieval_history.push(ctx.clone());
                    }
                }
            }
            {
                // Cross-context generality evidence (THEORY.md §6).
                let mut prov = h.field.recall_provenance.write();
                for &mid in id_slice {
                    let set = prov.entry(mid).or_default();
                    if set.len() < 8 {
                        set.insert(h.field.instance_id);
                    }
                }
            }
            {
                let mut assoc = h.field.assoc_edges.write();
                let mut coact = h.field.coactivation_stats.write();
                for i in 0..id_slice.len() {
                    for j in (i + 1)..id_slice.len() {
                        let key = (
                            id_slice[i].min(id_slice[j]),
                            id_slice[i].max(id_slice[j]),
                        );
                        let stats = coact.entry(key).or_default();
                        stats.record(context_hash, ts_ms);
                        let multiplier = stats.hebbian_multiplier();
                        let delta = base_assoc_delta * multiplier;
                        crate::field::strengthen_assoc_edge_map(
                            &mut assoc,
                            id_slice[i],
                            id_slice[j],
                            crate::ops::EdgeType::CoRetrieved,
                            delta,
                        );
                    }
                }
            }
            h.ok()
        }
        Err(e) => h.err(e),
    }
}

// ── Association edge query ────────────────────────────────────────────────────

/// Return association edges for a memory as a null-terminated JSON array.
/// JSON: [{"src":id,"dst":id,"edge_type":0,"weight":0.5}, ...]
/// Returns 0 on success, -2 if buf too small, -1 on error.
#[no_mangle]
pub extern "C" fn cf_get_assoc_edges(
    h: *mut CfHandle,
    memory_id: u64,
    limit: usize,
    buf: *mut c_char,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };

    fn et_u8(et: &crate::ops::EdgeType) -> u8 {
        match et {
            crate::ops::EdgeType::DerivedFrom  => 0,
            crate::ops::EdgeType::SameSession  => 1,
            crate::ops::EdgeType::SameArtifact => 2,
            crate::ops::EdgeType::CoRetrieved  => 3,
            crate::ops::EdgeType::Supports     => 4,
            crate::ops::EdgeType::Contradicts  => 5,
            crate::ops::EdgeType::SemanticNeighbor => 6,
        }
    }

    // Collect edges while holding lock, serialize, then drop lock before handle.ok()
    let serialized: Result<String, ()> = {
        let assoc_edges = handle.field.assoc_edges.read();
        let mut results: Vec<serde_json::Value> = Vec::new();

        if let Some(edges) = assoc_edges.get(&memory_id) {
            for e in edges.iter().take(limit) {
                results.push(serde_json::json!({
                    "src": memory_id,
                    "dst": e.dst,
                    "edge_type": et_u8(&e.edge_type),
                    "weight": e.weight,
                }));
            }
        }

        let remaining = limit.saturating_sub(results.len());
        if remaining > 0 {
            'outer: for (&src_id, edges) in assoc_edges.iter() {
                if src_id == memory_id { continue; }
                for e in edges {
                    if e.dst == memory_id {
                        results.push(serde_json::json!({
                            "src": src_id,
                            "dst": memory_id,
                            "edge_type": et_u8(&e.edge_type),
                            "weight": e.weight,
                        }));
                        if results.len() >= limit { break 'outer; }
                    }
                }
            }
        }

        serde_json::to_string(&results).map_err(|_| ())
        // lock dropped here
    };

    let s = match serialized {
        Ok(s) => s,
        Err(()) => return handle.err("failed to serialize assoc edges"),
    };

    let bytes = s.as_bytes();
    if bytes.len() + 1 > buf_cap {
        return -2;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
        *(buf as *mut u8).add(bytes.len()) = 0;
        *written = bytes.len();
    }

    handle.ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::TempDir;

    unsafe fn open_tmp() -> (*mut CfHandle, TempDir) {
        let tmp = TempDir::new().unwrap();
        let data = CString::new(tmp.path().join("data").to_str().unwrap()).unwrap();
        let lock = CString::new(tmp.path().join("lock").to_str().unwrap()).unwrap();
        let h = cf_open(data.as_ptr(), lock.as_ptr());
        assert!(!h.is_null());
        (h, tmp)
    }

    unsafe fn get_latest_event(
        h: *mut CfHandle,
        domain: &str,
        kind: &str,
        entity_id: &str,
    ) -> Result<Option<String>, c_int> {
        let domain = CString::new(domain).unwrap();
        let kind = CString::new(kind).unwrap();
        let entity_id = CString::new(entity_id).unwrap();
        let mut buf = vec![0u8; 4096];
        let mut written = 0usize;
        let rc = cf_get_latest_event(
            h,
            domain.as_ptr(),
            kind.as_ptr(),
            entity_id.as_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
            &mut written,
        );
        match rc {
            0 => Ok(Some(String::from_utf8(buf[..written].to_vec()).unwrap())),
            1 => Ok(None),
            other => Err(other),
        }
    }

    #[test]
    fn test_ffi_get_content_too_small_reports_required_size() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let content = [b'x'; 100];
            let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            let r = cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                content.as_ptr(),
                content.len(),
                embedding.as_ptr(),
                embedding.len(),
                0.9,
                0.001,
                0,
                &mut id,
            );
            assert_eq!(r, 0);

            // Exact-fit buffer (no room for the NUL) → -2, *written = required size.
            let mut buf = vec![0u8; 100];
            let mut written = 0usize;
            let r = cf_get_content(h, id, buf.as_mut_ptr(), buf.len(), &mut written);
            assert_eq!(r, -2);
            assert_eq!(written, 100);

            // Retry with written+1 succeeds and NUL-terminates.
            let mut buf = vec![0u8; written + 1];
            let r = cf_get_content(h, id, buf.as_mut_ptr(), buf.len(), &mut written);
            assert_eq!(r, 0);
            assert_eq!(written, 100);
            assert_eq!(&buf[..100], &content[..]);
            assert_eq!(buf[100], 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_put_recall() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"ffi test memory";
            let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;

            let r = cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                content.as_ptr(),
                content.len(),
                embedding.as_ptr(),
                embedding.len(),
                0.9,
                0.001,
                0,
                &mut id,
            );
            assert_eq!(r, 0);
            assert!(id > 0);

            // recall it back
            let mut hits = vec![
                CfRecallHit {
                    memory_id: 0,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: 0,
                    strength: 0.0,
                    confidence: 0.0,
                    access_count: 0,
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 0.0,
                    arousal_boost: 0.0,
                    mood_congruence: 0.0,
                    frustration_boost: 0.0,
                    interference_factor: 0.0,
                    spacing_boost: 0.0,
                };
                10
            ];
            let mut written: usize = 0;
            let r = cf_recall_semantic(
                h,
                embedding.as_ptr(),
                embedding.len(),
                realm.as_ptr(),
                5,
                hits.as_mut_ptr(),
                hits.len(),
                &mut written,
                false,
            );
            assert_eq!(r, 0);
            assert_eq!(written, 1);
            assert_eq!(hits[0].memory_id, id);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_forget() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("episode").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"to forget";
            let embedding = vec![0.5f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                content.as_ptr(),
                content.len(),
                embedding.as_ptr(),
                embedding.len(),
                1.0,
                0.001,
                0,
                &mut id,
            );

            let r = cf_forget(h, id);
            assert_eq!(r, 0);

            // should not appear in recall
            let mut hits = vec![
                CfRecallHit {
                    memory_id: 0,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: 0,
                    strength: 0.0,
                    confidence: 0.0,
                    access_count: 0,
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 0.0,
                    arousal_boost: 0.0,
                    mood_congruence: 0.0,
                    frustration_boost: 0.0,
                    interference_factor: 0.0,
                    spacing_boost: 0.0,
                };
                10
            ];
            let mut written: usize = 0;
            cf_recall_semantic(
                h,
                embedding.as_ptr(),
                embedding.len(),
                std::ptr::null(),
                10,
                hits.as_mut_ptr(),
                hits.len(),
                &mut written,
                false,
            );
            assert_eq!(written, 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_get_content() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("correction").unwrap();
            let realm = CString::new("r").unwrap();
            let content = b"hello from ffi";
            let embedding = vec![0.3f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                content.as_ptr(),
                content.len(),
                embedding.as_ptr(),
                embedding.len(),
                1.0,
                0.001,
                0,
                &mut id,
            );

            let mut buf = vec![0u8; 256];
            let mut written: usize = 0;
            let r = cf_get_content(h, id, buf.as_mut_ptr(), buf.len(), &mut written);
            assert_eq!(r, 0);
            assert_eq!(&buf[..written], content);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_update_state() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"state test";
            let embedding = vec![0.2f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                content.as_ptr(),
                content.len(),
                embedding.as_ptr(),
                embedding.len(),
                1.0,
                0.001,
                0,
                &mut id,
            );

            // Apply strength delta
            let r = cf_update_state(h, id, 0.1, f32::NAN, f32::NAN, 1, -1);
            assert_eq!(r, 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_assoc_edge() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id1: u64 = 0;
            let mut id2: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"a".as_ptr(),
                1,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                0,
                &mut id1,
            );
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"b".as_ptr(),
                1,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                0,
                &mut id2,
            );

            let r = cf_add_assoc_edge(h, id1, id2, 0, 0.8); // DerivedFrom
            assert_eq!(r, 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_upsert_artifact() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let path = CString::new("src/main.cpp").unwrap();
            let mut art_id: u64 = 0;
            let r = cf_upsert_artifact(h, path.as_ptr(), &mut art_id);
            assert_eq!(r, 0);
            assert!(art_id > 0);

            // Idempotent: second call returns same id
            let mut art_id2: u64 = 0;
            let r2 = cf_upsert_artifact(h, path.as_ptr(), &mut art_id2);
            assert_eq!(r2, 0);
            assert_eq!(art_id, art_id2);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_recall_temporal() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("episode").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.4f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"temporal".as_ptr(),
                8,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                1000,
                &mut id,
            );

            let mut hits = vec![
                CfRecallHit {
                    memory_id: 0,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: 0,
                    strength: 0.0,
                    confidence: 0.0,
                    access_count: 0,
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 0.0,
                    arousal_boost: 0.0,
                    mood_congruence: 0.0,
                    frustration_boost: 0.0,
                    interference_factor: 0.0,
                    spacing_boost: 0.0,
                };
                10
            ];
            let mut written: usize = 0;
            let r = cf_recall_temporal(
                h,
                0,
                10000,
                std::ptr::null(),
                10,
                hits.as_mut_ptr(),
                hits.len(),
                &mut written,
            );
            assert_eq!(r, 0);
            assert_eq!(written, 1);
            assert_eq!(hits[0].memory_id, id);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_flush() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let r = cf_flush(h);
            assert_eq!(r, 0);
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_memory_count() {
        unsafe {
            let (h, _tmp) = open_tmp();
            assert_eq!(cf_memory_count(h as *const CfHandle), 0);

            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"x".as_ptr(),
                1,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                0,
                &mut id,
            );
            assert_eq!(cf_memory_count(h as *const CfHandle), 1);

            cf_forget(h, id);
            assert_eq!(cf_memory_count(h as *const CfHandle), 0);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_get_kind_realm() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("correction").unwrap();
            let realm = CString::new("myproject").unwrap();
            let emb = vec![0.7f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"content".as_ptr(),
                7,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                0,
                &mut id,
            );

            let mut buf = [0u8; 64];
            let r = cf_get_kind(h, id, buf.as_mut_ptr(), buf.len());
            assert_eq!(r, 0);
            let kind_result = std::ffi::CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_str()
                .unwrap();
            assert_eq!(kind_result, "correction");

            let mut buf2 = [0u8; 64];
            let r2 = cf_get_realm(h, id, buf2.as_mut_ptr(), buf2.len());
            assert_eq!(r2, 0);
            let realm_result = std::ffi::CStr::from_ptr(buf2.as_ptr() as *const c_char)
                .to_str()
                .unwrap();
            assert_eq!(realm_result, "myproject");

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_expand_associations() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let emb = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id1: u64 = 0;
            let mut id2: u64 = 0;
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"seed".as_ptr(),
                4,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                0,
                &mut id1,
            );
            cf_put_memory(
                h,
                kind.as_ptr(),
                realm.as_ptr(),
                b"linked".as_ptr(),
                6,
                emb.as_ptr(),
                emb.len(),
                1.0,
                0.001,
                0,
                &mut id2,
            );
            cf_add_assoc_edge(h, id1, id2, 0, 1.0); // DerivedFrom

            let seeds = [id1];
            let mut hits = vec![
                CfRecallHit {
                    memory_id: 0,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: 0,
                    strength: 0.0,
                    confidence: 0.0,
                    access_count: 0,
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 0.0,
                    arousal_boost: 0.0,
                    mood_congruence: 0.0,
                    frustration_boost: 0.0,
                    interference_factor: 0.0,
                    spacing_boost: 0.0,
                };
                10
            ];
            let mut written: usize = 0;
            let r = cf_expand_associations(
                h,
                seeds.as_ptr(),
                seeds.len(),
                2,
                10,
                hits.as_mut_ptr(),
                hits.len(),
                &mut written,
            );
            assert_eq!(r, 0);
            assert_eq!(written, 1);
            assert_eq!(hits[0].memory_id, id2);

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_last_error_on_failure() {
        unsafe {
            let (h, _tmp) = open_tmp();
            // Try to forget a non-existent memory
            let r = cf_forget(h, 99999);
            assert_eq!(r, -1);
            let err_ptr = cf_last_error(h as *const CfHandle);
            assert!(!err_ptr.is_null());
            let err_msg = std::ffi::CStr::from_ptr(err_ptr).to_str().unwrap();
            assert!(!err_msg.is_empty());
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_null_handle_returns_error() {
        let r = cf_forget(std::ptr::null_mut(), 1);
        assert_eq!(r, -1);
        assert!(cf_last_error(std::ptr::null()).is_null());
        assert_eq!(cf_memory_count(std::ptr::null()), 0);
    }

    #[test]
    fn test_ffi_clear_project() {
        unsafe {
            let (h, _tmp) = open_tmp();
            // Register two code files in the same project
            let path1 = CString::new("/proj/a.cpp").unwrap();
            let path2 = CString::new("/proj/b.cpp").unwrap();
            let proj = CString::new("proj").unwrap();
            let mut file_id: u64 = 0;
            assert_eq!(
                cf_upsert_code_file(h, path1.as_ptr(), proj.as_ptr(), 0, &mut file_id),
                0
            );
            assert_eq!(
                cf_upsert_code_file(h, path2.as_ptr(), proj.as_ptr(), 0, &mut file_id),
                0
            );

            // Verify files are listed
            let mut buf = vec![0u8; 4096];
            let mut written = 0usize;
            assert_eq!(
                cf_list_code_files(h, proj.as_ptr(), buf.as_mut_ptr(), buf.len(), &mut written),
                0
            );
            let json = std::str::from_utf8(&buf[..written]).unwrap();
            assert!(json.contains("a.cpp") && json.contains("b.cpp"));

            // Clear the project
            assert_eq!(cf_clear_project(h, proj.as_ptr()), 0);

            // Files should be gone
            let mut written2 = 0usize;
            assert_eq!(
                cf_list_code_files(h, proj.as_ptr(), buf.as_mut_ptr(), buf.len(), &mut written2),
                0
            );
            let json2 = std::str::from_utf8(&buf[..written2]).unwrap();
            assert_eq!(json2, "[]");

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_update_memory_content() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let kind = CString::new("wisdom").unwrap();
            let realm = CString::new("test").unwrap();
            let content = b"original content";
            let emb = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            assert_eq!(
                cf_put_memory(
                    h,
                    kind.as_ptr(),
                    realm.as_ptr(),
                    content.as_ptr(),
                    content.len(),
                    emb.as_ptr(),
                    emb.len(),
                    0.9,
                    0.001,
                    0,
                    &mut id
                ),
                0
            );
            assert!(id > 0);

            // Update content + embedding
            let new_content = b"updated content";
            let new_emb = vec![0.9f32; crate::ops::EMBED_DIM];
            assert_eq!(
                cf_update_memory_content(
                    h,
                    id,
                    new_content.as_ptr(),
                    new_content.len(),
                    new_emb.as_ptr(),
                    new_emb.len()
                ),
                0
            );

            // Wrong embedding size must fail
            let bad_emb = [0.5f32; 3];
            assert_eq!(
                cf_update_memory_content(
                    h,
                    id,
                    new_content.as_ptr(),
                    new_content.len(),
                    bad_emb.as_ptr(),
                    bad_emb.len()
                ),
                -1
            );

            // Non-existent ID must return 1
            assert_eq!(
                cf_update_memory_content(
                    h,
                    99999,
                    new_content.as_ptr(),
                    new_content.len(),
                    std::ptr::null(),
                    0
                ),
                1
            );

            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_realm_list_sorted() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let emb = vec![0.1f32; crate::ops::EMBED_DIM];
            let mut id: u64 = 0;
            for realm_name in &["zebra", "alpha", "middle"] {
                let kind = CString::new("wisdom").unwrap();
                let realm = CString::new(*realm_name).unwrap();
                let txt = realm_name.as_bytes();
                cf_put_memory(
                    h,
                    kind.as_ptr(),
                    realm.as_ptr(),
                    txt.as_ptr(),
                    txt.len(),
                    emb.as_ptr(),
                    emb.len(),
                    0.9,
                    0.001,
                    0,
                    &mut id,
                );
            }
            let mut buf = vec![0u8; 4096];
            let mut written = 0usize;
            assert_eq!(
                cf_realm_list(h, buf.as_mut_ptr(), buf.len(), &mut written),
                0
            );
            let json = std::str::from_utf8(&buf[..written]).unwrap();
            let realms: Vec<String> = serde_json::from_str(json).unwrap();
            let mut sorted = realms.clone();
            sorted.sort_unstable();
            assert_eq!(realms, sorted, "realm list must be sorted");
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_recall_artifact() {
        unsafe {
            let (h, _tmp) = open_tmp();
            // No memories associated yet — should return 0 hits, not an error
            let path = CString::new("src/main.cpp").unwrap();
            let mut hits = vec![
                CfRecallHit {
                    memory_id: 0,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: 0,
                    strength: 0.0,
                    confidence: 0.0,
                    access_count: 0,
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 0.0,
                    arousal_boost: 0.0,
                    mood_congruence: 0.0,
                    frustration_boost: 0.0,
                    interference_factor: 0.0,
                    spacing_boost: 0.0,
                };
                10
            ];
            let mut written: usize = 0;
            let r = cf_recall_artifact(
                h,
                path.as_ptr(),
                10,
                hits.as_mut_ptr(),
                hits.len(),
                &mut written,
            );
            assert_eq!(r, 0);
            assert_eq!(written, 0);
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_transcript_latest_event_roundtrip_and_reopen() {
        unsafe {
            let (h, tmp) = open_tmp();
            let session_id = CString::new("sess-1").unwrap();
            let transcript_id = CString::new("tx-1").unwrap();
            let role = CString::new("assistant").unwrap();
            let mut turn_id = 0u64;

            assert_eq!(
                cf_transcript_register(h, transcript_id.as_ptr(), session_id.as_ptr()),
                0
            );
            assert_eq!(
                cf_transcript_update_progress(h, transcript_id.as_ptr(), 42.5),
                0
            );
            assert_eq!(
                cf_transcript_add_turn(
                    h,
                    transcript_id.as_ptr(),
                    role.as_ptr(),
                    b"hello".as_ptr(),
                    5,
                    1234,
                    &mut turn_id,
                ),
                0
            );
            assert_eq!(turn_id, 0);

            let progress = get_latest_event(h, "transcript", "update_progress", "sess-1")
                .unwrap()
                .unwrap();
            assert!(progress.contains(r#""transcript_id":"tx-1""#));
            assert!(progress.contains(r#""progress_pct":42.5"#));

            let turn = get_latest_event(h, "transcript", "add_turn", "sess-1")
                .unwrap()
                .unwrap();
            assert!(turn.contains(r#""transcript_id":"tx-1""#));
            assert!(turn.contains(r#""role":"assistant""#));
            assert!(turn.contains(r#""content":"hello""#));

            cf_close(h);

            let data = CString::new(tmp.path().join("data").to_str().unwrap()).unwrap();
            let lock = CString::new(tmp.path().join("lock").to_str().unwrap()).unwrap();
            let reopened = cf_open(data.as_ptr(), lock.as_ptr());
            assert!(!reopened.is_null());

            let reopened_progress =
                get_latest_event(reopened, "transcript", "update_progress", "sess-1")
                    .unwrap()
                    .unwrap();
            assert!(reopened_progress.contains(r#""progress_pct":42.5"#));

            let reopened_turn = get_latest_event(reopened, "transcript", "add_turn", "sess-1")
                .unwrap()
                .unwrap();
            assert!(reopened_turn.contains(r#""content":"hello""#));

            cf_close(reopened);
        }
    }

    #[test]
    fn test_ffi_transcript_update_requires_registered_transcript() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let transcript_id = CString::new("missing").unwrap();
            assert_eq!(
                cf_transcript_update_progress(h, transcript_id.as_ptr(), 1.0),
                -1
            );
            assert_eq!(
                cf_transcript_add_turn(
                    h,
                    transcript_id.as_ptr(),
                    std::ptr::null(),
                    b"x".as_ptr(),
                    1,
                    0,
                    &mut 0u64,
                ),
                -1
            );
            cf_close(h);
        }
    }

    #[test]
    fn test_ffi_get_latest_event_returns_buf_too_small() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let entity_id = CString::new("profile-1").unwrap();
            let entity_type = CString::new("profile").unwrap();
            assert_eq!(
                cf_user_model_upsert(
                    h,
                    entity_id.as_ptr(),
                    entity_type.as_ptr(),
                    br#"{"name":"abcdef"}"#.as_ptr(),
                    br#"{"name":"abcdef"}"#.len(),
                    100,
                ),
                0
            );

            let domain = CString::new("user_model").unwrap();
            let kind = CString::new("profile").unwrap();
            let mut buf = [0u8; 4];
            let mut written = 0usize;
            assert_eq!(
                cf_get_latest_event(
                    h,
                    domain.as_ptr(),
                    kind.as_ptr(),
                    entity_id.as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len(),
                    &mut written,
                ),
                -2
            );
            assert_eq!(written, 0);
            cf_close(h);
        }
    }

    #[test]
    fn test_ledger_session_snapshot_and_wal_suffix() {
        unsafe {
            let (h, tmp) = open_tmp();
            let timestamp = 1779913619.9907227_f64;
            let parsed: serde_json::Value = serde_json::from_str("1779913619.9907227").unwrap();
            assert_eq!(parsed.as_f64().unwrap(), timestamp, "ledger timestamps require exact JSON parsing");
            let emit = |h, domain: &str, kind: &str, target: &str, payload: &[u8]| {
                let mut id = 0;
                assert_eq!(cf_emit_event(h, CString::new(domain).unwrap().as_ptr(),
                    CString::new(kind).unwrap().as_ptr(), CString::new(target).unwrap().as_ptr(),
                    payload.as_ptr(), payload.len(), std::ptr::null(), 0, &mut id), 0);
            };
            emit(h, "ledger", "task_records", "task-ledger", br#"{"revision":1,"changes":[]}"#);
            emit(h, "session", "register", "owner", br#"{"kind":"codex"}"#);
            assert!(cf_save_full_snapshot(h));
            emit(h, "ledger", "task_records", "task-ledger", br#"{"revision":2,"changes":[]}"#);
            emit(h, "session", "deregister", "owner", b"{}");
            let data_dir = (&*h).field.data_dir.clone();
            cf_close(h);
            let field = crate::field::ChittaField::open(data_dir).unwrap();
            let registry = field.msg_registry.read();
            let events = registry.get_events_by_domain_kind("ledger", "task_records", 100);
            assert_eq!(events.len(), 2, "snapshot prefix and WAL suffix each restored once");
            assert!(events.iter().any(|e| e.payload_json.contains("\"revision\":1")));
            assert!(events.iter().any(|e| e.payload_json.contains("\"revision\":2")));
            assert_eq!(registry.get_events_by_domain_kind("session", "register", 100).len(), 1);
            assert_eq!(field.session_registry.read().list_active().len(), 0);
            drop(tmp);
        }
    }

    #[test]
    fn test_ffi_emit_event_rejects_user_model_domain() {
        unsafe {
            let (h, _tmp) = open_tmp();
            let domain = CString::new("user_model").unwrap();
            let kind = CString::new("upsert").unwrap();
            let entity_id = CString::new("profile-1").unwrap();
            let mut event_id = 0u64;
            assert_eq!(
                cf_emit_event(
                    h,
                    domain.as_ptr(),
                    kind.as_ptr(),
                    entity_id.as_ptr(),
                    br#"{}"#.as_ptr(),
                    2,
                    std::ptr::null(),
                    0,
                    &mut event_id,
                ),
                -1
            );
            assert_eq!(event_id, 0);
            cf_close(h);
        }
    }
}

/// Set memory lifecycle status. status: 0=Active, 1=Superseded, 2=Contradicted, 3=Archived, 4=Proposed, 5=Observed, 6=Verified
#[no_mangle]
pub extern "C" fn cf_set_memory_status(h: *mut CfHandle, memory_id: u64, status: u8) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    use crate::state::MemoryStatus;
    let s = match status {
        1 => MemoryStatus::Superseded,
        2 => MemoryStatus::Contradicted,
        3 => MemoryStatus::Archived,
        4 => MemoryStatus::Proposed,
        5 => MemoryStatus::Observed,
        6 => MemoryStatus::Verified,
        _ => MemoryStatus::Active,
    };
    match handle.field.set_memory_status(memory_id, s) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Set epistemic status. es: 0=UserStated, 1=ToolDerived, 2=ModelInferred, 3=AutonomousSynthesis
#[no_mangle]
pub extern "C" fn cf_set_epistemic_status(h: *mut CfHandle, memory_id: u64, es: u8) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    use crate::state::EpistemicStatus;
    let status = match es {
        0 => EpistemicStatus::UserStated,
        2 => EpistemicStatus::ModelInferred,
        3 => EpistemicStatus::AutonomousSynthesis,
        _ => EpistemicStatus::ToolDerived,
    };
    match handle.field.set_epistemic_status(memory_id, status) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Set affect dimensions on a memory. valence: -1.0 to +1.0, arousal: 0.0 to 1.0.
#[no_mangle]
pub extern "C" fn cf_set_affect(h: *mut CfHandle, memory_id: u64, valence: f32, arousal: f32) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.set_affect(memory_id, valence, arousal) {
        Ok(()) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

/// Compact WAL: save full snapshot then delete segments covered by it.
/// Returns number of deleted segments, or -1 on error.
#[no_mangle]
pub extern "C" fn cf_compact_wal(h: *mut CfHandle) -> i64 {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.compact_wal() {
        Ok(n) => n as i64,
        Err(e) => { handle.err(e); -1 }
    }
}

/// Count WAL segment files. Returns segment count.
#[no_mangle]
pub extern "C" fn cf_wal_segment_count(h: *const CfHandle) -> usize {
    if h.is_null() { return 0; }
    unsafe { (*h).field.wal_segment_count() }
}

/// Compact WAL if segment count > threshold and cooldown elapsed.
/// Returns 1 if compacted, 0 if skipped, -1 on error.
#[no_mangle]
pub extern "C" fn cf_maybe_compact_wal(h: *mut CfHandle, threshold: usize) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.maybe_compact_wal(threshold) {
        Ok(true)  => 1,
        Ok(false) => 0,
        Err(e)    => { handle.err(e); -1 }
    }
}

/// Prune old/excess episode memories.
/// Returns deleted count, or -1 on error.
#[no_mangle]
pub extern "C" fn cf_prune_episodes(h: *mut CfHandle, max_age_days: u64, max_count: usize) -> i64 {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.prune_episodes(max_age_days, max_count) {
        Ok(n)  => n as i64,
        Err(e) => { handle.err(e); -1 }
    }
}

/// Return JSON: {"staged_count": N, "oldest_staged_age_days": F}.
#[no_mangle]
pub extern "C" fn cf_write_gate_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_write_gate_stats: null argument"); }
    let handle = unsafe { &*h };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    let states = handle.field.states.read();
    let mut count = 0usize;
    let mut oldest_ms: i64 = now;
    for s in states.values() {
        if !s.deleted && s.staged {
            count += 1;
            if s.created_at_ms < oldest_ms {
                oldest_ms = s.created_at_ms;
            }
        }
    }
    drop(states);
    let oldest_days = if count > 0 {
        (now - oldest_ms).max(0) as f64 / 86_400_000.0
    } else { 0.0 };
    let j = format!(r#"{{"staged_count":{},"oldest_staged_age_days":{:.2}}}"#, count, oldest_days);
    match std::ffi::CString::new(j) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_write_gate_stats: {e}")),
    }
}

/// Promote staged memories that have been recalled; prune stale ones.
/// Returns (promoted as u32) << 32 | (pruned as u32). Returns 0 on error.
#[no_mangle]
pub extern "C" fn cf_promote_staged_memories(h: *mut CfHandle) -> u64 {
    if h.is_null() { return 0; }
    let handle = unsafe { &*h };
    match handle.field.promote_staged_memories() {
        Ok((promoted, pruned)) => ((promoted as u64) << 32) | (pruned as u64),
        Err(e) => { handle.err(e); 0 }
    }
}

// ── Scoring Pipeline Config FFI ───────────────────────────────────────────────

/// Reload scoring config from scoring.json in the data directory.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_reload_scoring_config(h: *mut CfHandle) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let config = crate::scoring::config::ScoringConfig::load(&handle.field.data_dir);
    handle.field.scoring_pipeline.write().reload_config(config);
    0
}

/// Save current scoring config to scoring.json for inspection/editing.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_save_scoring_config(h: *const CfHandle) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let pipeline = handle.field.scoring_pipeline.read();
    match pipeline.config.save(&handle.field.data_dir) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ── FEP Attractor Network FFI ────────────────────────────────────────────────

/// Get reconstruction error (surprise) for a memory. Returns value in [0,1].
/// -1.0 on error. Used by C++ consolidation for free-energy merge criterion.
#[no_mangle]
pub extern "C" fn cf_reconstruction_error(h: *const CfHandle, memory_id: u64) -> f32 {
    if h.is_null() { return -1.0; }
    let handle = unsafe { &*h };
    let embedding = match handle.field.embedding_of(memory_id) {
        Some(e) if e.len() == crate::ops::EMBED_DIM => e,
        _ => return -1.0,
    };
    let encoder = handle.field.sparse_encoder.read();
    let code = encoder.encode(&embedding);
    encoder.reconstruction_error(&embedding, &code)
}

/// Get the surprise score cached in memory state. Returns -1.0 if not found.
#[no_mangle]
pub extern "C" fn cf_memory_surprise(h: *const CfHandle, memory_id: u64) -> f32 {
    if h.is_null() { return -1.0; }
    let handle = unsafe { &*h };
    handle.field.states.read()
        .get(&memory_id)
        .map(|s| s.surprise)
        .unwrap_or(-1.0)
}

/// Cortical attractor search: settle query embedding then search.
/// Writes results to buf, returns count written.
#[no_mangle]
pub extern "C" fn cf_search_attractor(
    h: *const CfHandle,
    embedding: *const f32,
    dim: usize,
    k: usize,
    settle_steps: usize,
    buf: *mut CfRecallHit,
    buf_cap: usize,
    written: *mut usize,
) -> c_int {
    if h.is_null() || embedding.is_null() || buf.is_null() || written.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    if dim != crate::ops::EMBED_DIM { return -1; }
    let emb = unsafe { std::slice::from_raw_parts(embedding, dim) };

    let encoder = handle.field.sparse_encoder.read();
    let code = encoder.encode(emb);
    drop(encoder);

    let cortical = handle.field.cortical_idx.read();
    let results = cortical.search_attractor(&code, k, None, settle_steps);
    drop(cortical);

    let states = handle.field.states.read();
    let n = results.len().min(buf_cap);
    for (i, (mem_id, score)) in results.iter().take(n).enumerate() {
        let state = states.get(mem_id);
        unsafe {
            *buf.add(i) = CfRecallHit {
                memory_id: *mem_id,
                score: *score,
                semantic_score: *score,
                ts_ms: state.map(|s| s.last_accessed_ms).unwrap_or(0),
                strength: state.map(|s| s.strength).unwrap_or(0.0),
                confidence: state.map(|s| s.confidence).unwrap_or(0.0),
                access_count: state.map(|s| s.access_count).unwrap_or(0),
                semantic_weight: 1.0,
                status_mul: 1.0,
                epistemic_mul: 1.0,
                strength_factor: 1.0,
                affect_valence: 0.0,
                affect_arousal: 0.0,
                actr_activation: 0.0,
                surprise_boost: 1.0,
                arousal_boost: 1.0,
                mood_congruence: 1.0,
                frustration_boost: 1.0,
                interference_factor: 0.0,
                spacing_boost: 0.0,
            };
        }
    }
    unsafe { *written = n; }
    0
}

/// Record co-retrieval in the Hopfield network.
#[no_mangle]
pub extern "C" fn cf_hopfield_co_retrieval(
    h: *mut CfHandle,
    ids: *const u64,
    count: usize,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || ids.is_null() || count == 0 { return -1; }
    let handle = unsafe { &*h };
    let id_slice = unsafe { std::slice::from_raw_parts(ids, count) };
    handle.field.hopfield.write().record_co_retrieval(id_slice, ts_ms);
    0
}

/// Get Hopfield network statistics as JSON string.
#[no_mangle]
pub extern "C" fn cf_hopfield_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_hopfield_stats: null argument"); }
    let handle = unsafe { &*h };
    let net = handle.field.hopfield.read();
    let json = format!(
        r#"{{"couplings":{},"settles":{}}}"#,
        net.coupling_count(),
        net.settle_count()
    );
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_hopfield_stats: no data"))
}

/// Adapt cortical vigilance based on aggregate reconstruction error.
#[no_mangle]
pub extern "C" fn cf_adapt_vigilance(h: *mut CfHandle, avg_error: f32) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    handle.field.cortical_idx.write().adapt_vigilance(avg_error);
    0
}

// ── Skill Registry FFI ──────────────────────────────────────────────────────

/// Upload a new skill version. Returns the assigned version number, or -1 on error.
#[no_mangle]
pub extern "C" fn cf_skill_upload(
    h: *mut CfHandle,
    skill_id: *const c_char,
    content: *const c_char,
    uploaded_by: *const c_char,
    tags_json: *const c_char,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || skill_id.is_null() || content.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let skill_id_str = unsafe { match CStr::from_ptr(skill_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let content_str = unsafe { match CStr::from_ptr(content).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let uploaded_by_str = if uploaded_by.is_null() { "" } else {
        unsafe { match CStr::from_ptr(uploaded_by).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } }
    };
    let tags: Vec<String> = if tags_json.is_null() {
        Vec::new()
    } else {
        let tags_str = unsafe { match CStr::from_ptr(tags_json).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
        serde_json::from_str(tags_str).unwrap_or_default()
    };

    let op = Op::SkillUpload(SkillUploadOp {
        skill_id: skill_id_str.to_string(),
        content: content_str.to_string(),
        uploaded_by: uploaded_by_str.to_string(),
        tags: tags.clone(),
        ts_ms,
    });
    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    let version = handle.field.skill_registry.write().upload(skill_id_str, content_str, uploaded_by_str, &tags, ts_ms);
    version as c_int
}

/// Read a skill version as JSON. version=0 means latest.
#[no_mangle]
pub extern "C" fn cf_skill_read(
    h: *const CfHandle,
    skill_id: *const c_char,
    version: u32,
) -> *mut c_char {
    if h.is_null() || skill_id.is_null() { return json_null("cf_skill_read: null argument"); }
    let handle = unsafe { &*h };
    let skill_id_str = unsafe { match CStr::from_ptr(skill_id).to_str() { Ok(s) => s, Err(_) => return json_null("cf_skill_read: no data") } };
    let reg = handle.field.skill_registry.read();
    match reg.read(skill_id_str, version) {
        Some(sv) => {
            let json = serde_json::to_string(sv).unwrap_or_default();
            CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_skill_read: no data"))
        }
        None => json_null("cf_skill_read: no data"),
    }
}

/// List all skills as JSON array of {skill_id, latest_version}.
#[no_mangle]
pub extern "C" fn cf_skill_list(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_skill_list: null argument"); }
    let handle = unsafe { &*h };
    let reg = handle.field.skill_registry.read();
    let list: Vec<serde_json::Value> = reg.list().iter().map(|(id, ver)| {
        serde_json::json!({"skill_id": id, "latest_version": ver})
    }).collect();
    let json = serde_json::to_string(&list).unwrap_or_default();
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_skill_list: no data"))
}

/// Search skills by query. Returns JSON array.
#[no_mangle]
pub extern "C" fn cf_skill_search(
    h: *const CfHandle,
    query: *const c_char,
    limit: usize,
) -> *mut c_char {
    if h.is_null() || query.is_null() { return json_null("cf_skill_search: null argument"); }
    let handle = unsafe { &*h };
    let query_str = unsafe { match CStr::from_ptr(query).to_str() { Ok(s) => s, Err(_) => return json_null("cf_skill_search: no data") } };
    let reg = handle.field.skill_registry.read();
    let results = reg.search(query_str, if limit == 0 { 20 } else { limit });
    let json = serde_json::to_string(&results).unwrap_or_default();
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_skill_search: no data"))
}

/// Deprecate a skill.
#[no_mangle]
pub extern "C" fn cf_skill_deprecate(
    h: *mut CfHandle,
    skill_id: *const c_char,
) -> c_int {
    if h.is_null() || skill_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let skill_id_str = unsafe { match CStr::from_ptr(skill_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let op = Op::SkillDeprecate(SkillDeprecateOp { skill_id: skill_id_str.to_string() });
    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    if handle.field.skill_registry.write().deprecate(skill_id_str) { 0 } else { -1 }
}

// ── Agent Registry FFI ──────────────────────────────────────────────────────

/// Register or update an agent. Returns 1 if newly created, 0 if updated, -1 on error.
#[no_mangle]
pub extern "C" fn cf_agent_upsert(
    h: *mut CfHandle,
    agent_id: *const c_char,
    display_name: *const c_char,
    description: *const c_char,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || agent_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let agent_id_str = unsafe { match CStr::from_ptr(agent_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let name_str = if display_name.is_null() { "" } else {
        unsafe { match CStr::from_ptr(display_name).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } }
    };
    let desc_str = if description.is_null() { "" } else {
        unsafe { match CStr::from_ptr(description).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } }
    };

    let op = Op::AgentUpsert(AgentUpsertOp {
        agent_id: agent_id_str.to_string(),
        display_name: name_str.to_string(),
        description: desc_str.to_string(),
        ts_ms,
    });
    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    let is_new = handle.field.agent_registry.write().upsert(agent_id_str, name_str, desc_str, ts_ms);
    if is_new { 1 } else { 0 }
}

/// Record activity for an agent (increments memory count).
#[no_mangle]
pub extern "C" fn cf_agent_record_activity(
    h: *mut CfHandle,
    agent_id: *const c_char,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || agent_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let agent_id_str = unsafe { match CStr::from_ptr(agent_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    handle.field.agent_registry.write().record_activity(agent_id_str, ts_ms);
    handle.ok()
}

/// Record a new session for an agent.
#[no_mangle]
pub extern "C" fn cf_agent_record_session(
    h: *mut CfHandle,
    agent_id: *const c_char,
    ts_ms: i64,
) -> c_int {
    if h.is_null() || agent_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let agent_id_str = unsafe { match CStr::from_ptr(agent_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    handle.field.agent_registry.write().record_session(agent_id_str, ts_ms);
    handle.ok()
}

/// Get an agent record as JSON. Returns null if not found.
#[no_mangle]
pub extern "C" fn cf_agent_get(
    h: *const CfHandle,
    agent_id: *const c_char,
) -> *mut c_char {
    if h.is_null() || agent_id.is_null() { return json_null("cf_agent_get: null argument"); }
    let handle = unsafe { &*h };
    let agent_id_str = unsafe { match CStr::from_ptr(agent_id).to_str() { Ok(s) => s, Err(_) => return json_null("cf_agent_get: no data") } };
    let reg = handle.field.agent_registry.read();
    match reg.get(agent_id_str) {
        Some(rec) => {
            let json = serde_json::to_string(rec).unwrap_or_default();
            CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_agent_get: no data"))
        }
        None => json_null("cf_agent_get: no data"),
    }
}

/// List all agents as JSON array.
#[no_mangle]
pub extern "C" fn cf_agent_list(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_agent_list: null argument"); }
    let handle = unsafe { &*h };
    let reg = handle.field.agent_registry.read();
    let list = reg.list();
    let json = serde_json::to_string(&list).unwrap_or_default();
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_agent_list: no data"))
}

/// Disable (revoke) an agent.
#[no_mangle]
pub extern "C" fn cf_agent_disable(
    h: *mut CfHandle,
    agent_id: *const c_char,
) -> c_int {
    if h.is_null() || agent_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let agent_id_str = unsafe { match CStr::from_ptr(agent_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let op = Op::AgentDisable(AgentDisableOp { agent_id: agent_id_str.to_string() });
    let result = handle.field.log.write().append(&op);
    if let Err(e) = result {
        return handle.err(e);
    }
    if handle.field.agent_registry.write().disable(agent_id_str) { 0 } else { -1 }
}

// ── Layer 1: Executable Constraints ─────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_assert_constraint(
    h: *mut CfHandle,
    params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_assert_constraint: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_assert_constraint: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_assert_constraint: no data") };

    let subject = params["subject"].as_str().unwrap_or("").to_string();
    let predicate = params["predicate"].as_str().unwrap_or("").to_string();
    let object = params["object"].as_str().unwrap_or("").to_string();
    let confidence = params["confidence"].as_f64().unwrap_or(0.8) as f32;
    let scope = params["scope"].as_str().unwrap_or("global").to_string();
    let branch_id = params["branch_id"].as_u64().unwrap_or(0);
    let provenance = crate::organ::constraint::Provenance {
        source: params["provenance_source"].as_str().unwrap_or("tool").to_string(),
        session_id: params["session_id"].as_str().map(|s| s.to_string()),
        confidence_basis: params["confidence_basis"].as_str().unwrap_or("observed").to_string(),
    };
    let source_memory_id = params["source_memory_id"].as_u64();

    match handle.field.assert_constraint(subject, predicate, object, confidence, scope, branch_id, provenance, source_memory_id) {
        Ok(result) => {
            let json = serde_json::json!({
                "fact_id": result.fact_id,
                "conflict": result.conflict.map(|c| serde_json::json!({
                    "rival_fact_id": c.rival_fact_id,
                    "rival_object": c.rival_object,
                    "new_branch_id": c.new_branch_id,
                })),
            });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_assert_constraint: no data"))
        }
        Err(e) => json_null(format!("cf_assert_constraint: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_retract_constraint(h: *mut CfHandle, fact_id: u64) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.retract_constraint(fact_id) {
        Ok(true) => handle.ok(),
        Ok(false) => handle.err("fact not found"),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_query_constraints(
    h: *const CfHandle,
    params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_query_constraints: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_query_constraints: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_query_constraints: no data") };

    let subject = params["subject"].as_str();
    let predicate = params["predicate"].as_str();
    let object = params["object"].as_str();
    let scope = params["scope"].as_str();

    let results = handle.field.query_constraints(subject, predicate, object, scope);
    let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_constraints: no data"))
}

#[no_mangle]
pub extern "C" fn cf_explain_constraint(h: *const CfHandle, fact_id: u64) -> *mut c_char {
    if h.is_null() { return json_null("cf_explain_constraint: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.explain_constraint(fact_id) {
        Some(explanation) => {
            let json = serde_json::json!({
                "fact": {
                    "id": explanation.fact.id,
                    "subject": explanation.fact.subject,
                    "predicate": explanation.fact.predicate,
                    "object": explanation.fact.object,
                    "confidence": explanation.fact.confidence,
                    "scope": explanation.fact.scope,
                    "branch_id": explanation.fact.branch_id,
                    "provenance": {
                        "source": explanation.fact.provenance.source,
                        "session_id": explanation.fact.provenance.session_id,
                        "confidence_basis": explanation.fact.provenance.confidence_basis,
                    },
                },
                "supporting": explanation.supporting,
                "conflicting": explanation.conflicting,
                "branch": explanation.branch.map(|b| serde_json::json!({
                    "id": b.id, "parent_id": b.parent_id, "scope": b.scope,
                    "status": format!("{:?}", b.status),
                })),
            });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_explain_constraint: no data"))
        }
        None => json_null("cf_explain_constraint: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_create_constraint_branch(
    h: *mut CfHandle, parent_id: u64, scope: *const c_char,
) -> i64 {
    if h.is_null() || scope.is_null() { return -1; }
    let handle = unsafe { &*h };
    let scope_str = unsafe { match CStr::from_ptr(scope).to_str() { Ok(s) => s.to_string(), Err(_) => return -1 } };
    match handle.field.create_constraint_branch(parent_id, scope_str) {
        Ok(id) => id as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_resolve_constraint_branch(
    h: *mut CfHandle, winner_id: u64, loser_id: u64,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.resolve_constraint_branch(winner_id, loser_id) {
        Ok(true) => handle.ok(),
        Ok(false) => handle.err("branch not found"),
        Err(e) => handle.err(e),
    }
}

// ── Layer 2: Trigger Tissue ─────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_add_trigger(
    h: *mut CfHandle, params_json: *const c_char,
) -> i64 {
    if h.is_null() || params_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return -1 } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return -1 };

    let name = params["name"].as_str().unwrap_or("").to_string();
    let condition: crate::organ::trigger::TriggerCondition = match serde_json::from_value(params["condition"].clone()) {
        Ok(c) => c, Err(_) => return -1,
    };
    let action: crate::organ::trigger::TriggerAction = match serde_json::from_value(params["action"].clone()) {
        Ok(a) => a, Err(_) => return -1,
    };
    let deadline_ms = params["deadline_ms"].as_i64().unwrap_or(0);
    let tension_threshold = params["tension_threshold"].as_f64().unwrap_or(0.8) as f32;
    let gain = params["gain"].as_f64().unwrap_or(0.5) as f32;
    let realm = params["realm"].as_str().unwrap_or("global").to_string();
    let source_session = params["session_id"].as_str().map(|s| s.to_string());

    match handle.field.add_trigger(name, condition, action, deadline_ms, tension_threshold, gain, realm, source_session) {
        Ok(id) => id as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_fire_trigger(h: *mut CfHandle, trigger_id: u64) -> *mut c_char {
    if h.is_null() { return json_null("cf_fire_trigger: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.fire_trigger(trigger_id) {
        Ok(Some(result)) => {
            let json = serde_json::json!({
                "trigger_id": result.trigger_id,
                "action": result.action,
            });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_fire_trigger: no data"))
        }
        _ => json_null("cf_fire_trigger: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_dismiss_trigger(h: *mut CfHandle, trigger_id: u64) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.dismiss_trigger(trigger_id) {
        Ok(true) => handle.ok(),
        Ok(false) => handle.err("trigger not found or not armed"),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_list_triggers(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_list_triggers: null argument"); }
    let handle = unsafe { &*h };
    let triggers = handle.field.list_triggers();
    let json = serde_json::to_string(&triggers).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_list_triggers: no data"))
}

#[no_mangle]
pub extern "C" fn cf_evaluate_triggers(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_evaluate_triggers: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.evaluate_triggers() {
        Ok(results) => {
            let json = serde_json::json!(results.iter().map(|r| serde_json::json!({
                "trigger_id": r.trigger_id,
            })).collect::<Vec<_>>());
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_evaluate_triggers: no data"))
        }
        Err(e) => json_null(format!("cf_evaluate_triggers: {e}")),
    }
}

// ── Layer 3: Predictive Memory ──────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_predict_needed(h: *const CfHandle, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_predict_needed: null argument"); }
    let handle = unsafe { &*h };
    let predictions = handle.field.predict_needed(k);
    let json = serde_json::json!(predictions.iter().map(|(id, prob)| {
        serde_json::json!({"memory_id": id, "probability": prob})
    }).collect::<Vec<_>>());
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_predict_needed: no data"))
}

#[no_mangle]
pub extern "C" fn cf_retrain_predictor(h: *mut CfHandle) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    handle.field.retrain_predictor();
    handle.ok()
}

#[no_mangle]
pub extern "C" fn cf_constraint_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_constraint_stats: null argument"); }
    let handle = unsafe { &*h };
    let (facts, branches) = handle.field.constraint_stats();
    let armed = handle.field.trigger_stats();
    let (transitions, transition_sources, recent_accesses) = handle.field.predictor_stats();
    let surprise = handle.field.surprise_stats();
    let debt = handle.field.debt_stats();
    let integration = handle.field.integration_stats();
    let json = serde_json::json!({
        "constraints": {"facts": facts, "branches": branches},
        "triggers": {"armed": armed},
        "predictor": {"transitions": transitions, "sources": transition_sources, "recent_accesses": recent_accesses},
        "surprise": {"events": surprise.total_events, "avg_magnitude": surprise.avg_magnitude},
        "epistemic_debt": {"total": debt.total, "open": debt.open, "resolved": debt.resolved, "deferred": debt.deferred, "avg_fragility_open": debt.avg_fragility_open},
        "integration": {"total_queries": integration.total_queries, "sources": integration.source_rates.len()},
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_constraint_stats: no data"))
}

// ── Layer 4: Surprise Memory ──────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_record_surprise(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_record_surprise: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_record_surprise: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_record_surprise: no data") };

    let context_sketch = params["context_sketch"].as_str().unwrap_or("").to_string();
    let action = params["action"].as_str().unwrap_or("").to_string();
    let expected = params["expected"].as_str().map(|s| s.to_string());
    let actual = params["actual"].as_str().unwrap_or("").to_string();
    let surprise_magnitude = params["surprise_magnitude"].as_f64().unwrap_or(0.5) as f32;
    let domain = params["domain"].as_str().unwrap_or("general").to_string();
    let realm = params["realm"].as_str().unwrap_or("global").to_string();
    let session_id = params["session_id"].as_str().map(|s| s.to_string());
    let source_memory_id = params["source_memory_id"].as_u64();

    match handle.field.record_surprise(
        context_sketch, action, expected, actual,
        surprise_magnitude, domain, realm, session_id, source_memory_id,
    ) {
        Ok(event_id) => {
            let json = serde_json::json!({"event_id": event_id});
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_record_surprise: no data"))
        }
        Err(e) => json_null(format!("cf_record_surprise: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_query_surprises(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_query_surprises: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_query_surprises: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_query_surprises: no data") };

    let domain = params["domain"].as_str();
    let realm = params["realm"].as_str();
    let min_magnitude = params["min_magnitude"].as_f64().map(|v| v as f32);
    let since_ms = params["since_ms"].as_i64();
    let limit = params["limit"].as_u64().unwrap_or(50) as usize;

    let events = handle.field.query_surprises(domain, realm, min_magnitude, since_ms, limit);
    let json = serde_json::to_string(&events).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_surprises: no data"))
}

#[no_mangle]
pub extern "C" fn cf_get_blind_spots(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_get_blind_spots: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_get_blind_spots: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_get_blind_spots: no data") };

    let realm = params["realm"].as_str();
    let limit = params["limit"].as_u64().unwrap_or(10) as usize;

    let spots = handle.field.get_blind_spots(realm, limit);
    let json = serde_json::to_string(&spots).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_get_blind_spots: no data"))
}

#[no_mangle]
pub extern "C" fn cf_surprise_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_surprise_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.surprise_stats();
    let json = serde_json::json!({
        "total_events": stats.total_events,
        "avg_magnitude": stats.avg_magnitude,
        "by_domain": stats.by_domain,
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_surprise_stats: no data"))
}

// ── Layer 5: Epistemic Debt ───────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_register_debt(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_register_debt: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_register_debt: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_register_debt: no data") };

    let pattern = params["pattern"].as_str().unwrap_or("").to_string();
    let competing_hypotheses: Vec<String> = params["competing_hypotheses"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let discriminating_test = params["discriminating_test"].as_str().map(|s| s.to_string());
    let fragility_score = params["fragility_score"].as_f64().unwrap_or(0.5) as f32;
    let domain = params["domain"].as_str().unwrap_or("general").to_string();
    let realm = params["realm"].as_str().unwrap_or("global").to_string();
    let source_session = params["session_id"].as_str().map(|s| s.to_string());

    match handle.field.register_debt(
        pattern, competing_hypotheses, discriminating_test,
        fragility_score, domain, realm, source_session,
    ) {
        Ok(debt_id) => {
            let json = serde_json::json!({"debt_id": debt_id});
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_register_debt: no data"))
        }
        Err(e) => json_null(format!("cf_register_debt: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_resolve_debt(
    h: *mut CfHandle, debt_id: u64, resolution_json: *const c_char,
) -> c_int {
    if h.is_null() || resolution_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let resolution = unsafe { match CStr::from_ptr(resolution_json).to_str() { Ok(s) => s.to_string(), Err(_) => return -1 } };
    match handle.field.resolve_debt(debt_id, resolution) {
        Ok(true) => handle.ok(),
        Ok(false) => handle.err("debt not found"),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_defer_debt(h: *mut CfHandle, debt_id: u64) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.defer_debt(debt_id) {
        Ok(true) => handle.ok(),
        Ok(false) => handle.err("debt not found"),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_query_debts(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_query_debts: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_query_debts: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_query_debts: no data") };

    let status = params["status"].as_str().map(|s| match s {
        "open" | "Open" => crate::organ::epistemic_debt::DebtStatus::Open,
        "resolved" | "Resolved" => crate::organ::epistemic_debt::DebtStatus::Resolved,
        _ => crate::organ::epistemic_debt::DebtStatus::Deferred,
    });
    let domain = params["domain"].as_str();
    let realm = params["realm"].as_str();
    let min_fragility = params["min_fragility"].as_f64().map(|v| v as f32);
    let limit = params["limit"].as_u64().unwrap_or(50) as usize;

    let debts = handle.field.query_debts(status, domain, realm, min_fragility, limit);
    let json = serde_json::to_string(&debts).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_debts: no data"))
}

#[no_mangle]
pub extern "C" fn cf_get_fragile_decisions(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_get_fragile_decisions: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_get_fragile_decisions: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_get_fragile_decisions: no data") };

    let threshold = params["threshold"].as_f64().unwrap_or(0.5) as f32;
    let limit = params["limit"].as_u64().unwrap_or(20) as usize;

    let debts = handle.field.get_fragile_decisions(threshold, limit);
    let json = serde_json::to_string(&debts).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_get_fragile_decisions: no data"))
}

#[no_mangle]
pub extern "C" fn cf_debt_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_debt_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.debt_stats();
    let json = serde_json::json!({
        "total": stats.total,
        "open": stats.open,
        "resolved": stats.resolved,
        "deferred": stats.deferred,
        "avg_fragility_open": stats.avg_fragility_open,
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_debt_stats: no data"))
}

// ── Layer 6: Integration Kernel ───────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_record_feedback(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_record_feedback: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_record_feedback: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_record_feedback: no data") };

    let query_domain = params["query_domain"].as_str().unwrap_or("general");
    let source = params["source"].as_str().unwrap_or("");
    let was_useful = params["was_useful"].as_bool().unwrap_or(true);

    match handle.field.record_feedback(query_domain, source, was_useful) {
        Ok(sw) => {
            let json = serde_json::json!({
                "source": sw.source,
                "query_domain": sw.query_domain,
                "weight": sw.weight,
                "success_count": sw.success_count,
                "total_count": sw.total_count,
            });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_record_feedback: no data"))
        }
        Err(e) => json_null(format!("cf_record_feedback: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_get_source_weights(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_get_source_weights: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return json_null("cf_get_source_weights: no data") } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return json_null("cf_get_source_weights: no data") };

    let domain = params["domain"].as_str();
    let weights = handle.field.get_source_weights(domain);
    let json = serde_json::to_string(&weights).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_get_source_weights: no data"))
}

#[no_mangle]
pub extern "C" fn cf_update_source_weight(
    h: *mut CfHandle, params_json: *const c_char,
) -> c_int {
    if h.is_null() || params_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() { Ok(s) => s, Err(_) => return -1 } };
    let params: serde_json::Value = match serde_json::from_str(json_str) { Ok(v) => v, Err(_) => return -1 };

    let source = params["source"].as_str().unwrap_or("");
    let domain = params["domain"].as_str().unwrap_or("general");
    let weight = params["weight"].as_f64().unwrap_or(1.0) as f32;

    match handle.field.update_source_weight(source, domain, weight) {
        Ok(_) => handle.ok(),
        Err(e) => handle.err(e),
    }
}

#[no_mangle]
pub extern "C" fn cf_integration_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_integration_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.integration_stats();
    let json = serde_json::json!({
        "total_queries": stats.total_queries,
        "source_rates": stats.source_rates.iter().map(|(source, domain, rate, count)| {
            serde_json::json!({"source": source, "domain": domain, "success_rate": rate, "total_count": count})
        }).collect::<Vec<_>>(),
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_integration_stats: no data"))
}

// ── Autonomous Learning FFI ───────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_surprise_learning_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_surprise_learning_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.surprise_learning_stats();
    let json = serde_json::json!({
        "tracked_memories": stats.tracked_memories,
        "tracked_failure_pairs": stats.tracked_failure_pairs,
        "total_gates_passed": stats.total_gates_passed,
        "total_credits_updated": stats.total_credits_updated,
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_surprise_learning_stats: no data"))
}

#[no_mangle]
pub extern "C" fn cf_upsert_wisdom_candidate(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_upsert_wisdom_candidate: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_upsert_wisdom_candidate: no data")
    }};
    let params: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_upsert_wisdom_candidate: no data")
    };
    let cluster_key = params["cluster_key"].as_str().unwrap_or("").to_string();
    let domain = params["domain"].as_str().unwrap_or("").to_string();
    let action = params["action"].as_str().unwrap_or("").to_string();
    let summary = params["summary"].as_str().unwrap_or("").to_string();
    let episode_ids: Vec<u64> = params["episode_ids"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let debt_ids: Vec<u64> = params["debt_ids"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let support_count = params["support_count"].as_u64().unwrap_or(0) as u32;
    let cross_session_count = params["cross_session_count"].as_u64().unwrap_or(0) as u32;
    let mean_surprise = params["mean_surprise"].as_f64().unwrap_or(0.0) as f32;
    let promotion_score = params["promotion_score"].as_f64().unwrap_or(0.0) as f32;

    match handle.field.upsert_wisdom_candidate(
        cluster_key, domain, action, summary, episode_ids, debt_ids,
        support_count, cross_session_count, mean_surprise, promotion_score,
    ) {
        Ok(candidate_id) => {
            let json = serde_json::json!({"candidate_id": candidate_id});
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_upsert_wisdom_candidate: no data"))
        }
        Err(e) => json_null(format!("cf_upsert_wisdom_candidate: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_update_wisdom_lifecycle(
    h: *mut CfHandle, candidate_id: u64, new_state: u8,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let lifecycle = crate::organ::wisdom_promotion::WisdomLifecycle::from_u8(new_state);
    match handle.field.update_wisdom_lifecycle(candidate_id, lifecycle, None, 0) {
        Ok(true) => 0,
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_query_wisdom_candidates(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_query_wisdom_candidates: null argument"); }
    let handle = unsafe { &*h };
    let params: serde_json::Value = if params_json.is_null() {
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s, Err(_) => return json_null("cf_query_wisdom_candidates: no data")
        }};
        serde_json::from_str(json_str).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
    };
    let lifecycle = params["lifecycle"].as_u64()
        .map(|v| crate::organ::wisdom_promotion::WisdomLifecycle::from_u8(v as u8));
    let domain = params["domain"].as_str();
    let limit = params["limit"].as_u64().unwrap_or(50) as usize;
    let results = handle.field.query_wisdom_candidates(lifecycle, domain, limit);
    let json = serde_json::to_value(&results).unwrap_or(serde_json::Value::Array(vec![]));
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_wisdom_candidates: no data"))
}

#[no_mangle]
pub extern "C" fn cf_wisdom_promotion_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_wisdom_promotion_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.wisdom_promotion_stats();
    let json = serde_json::to_value(&stats).unwrap_or(serde_json::Value::Null);
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_wisdom_promotion_stats: no data"))
}

#[no_mangle]
pub extern "C" fn cf_attach_debt_evidence(
    h: *mut CfHandle, debt_id: u64, evidence_json: *const c_char,
) -> c_int {
    if h.is_null() || evidence_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(evidence_json).to_str() {
        Ok(s) => s, Err(_) => return -1
    }};
    let params: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return -1
    };
    let memory_ids: Vec<u64> = params["memory_ids"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
        .unwrap_or_default();
    let confidence = params["confidence"].as_f64().unwrap_or(0.5) as f32;
    let note = params["note"].as_str().map(|s| s.to_string());
    match handle.field.attach_debt_evidence(debt_id, memory_ids, confidence, note) {
        Ok(true) => 0,
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_update_scorer_model(
    h: *mut CfHandle, model_json: *const c_char,
) -> c_int {
    if h.is_null() || model_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(model_json).to_str() {
        Ok(s) => s, Err(_) => return -1
    }};
    let params: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return -1
    };
    let weights_json = params["weights"].to_string();
    let model_version = params["model_version"].as_u64().unwrap_or(0);
    let mean_loss = params["mean_loss"].as_f64().unwrap_or(0.0) as f32;
    let outcome_count = params["outcome_count"].as_u64().unwrap_or(0);
    match handle.field.update_scorer_model(weights_json, model_version, mean_loss, outcome_count) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_learned_scorer_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_learned_scorer_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.learned_scorer_stats();
    let json = serde_json::to_value(&stats).unwrap_or(serde_json::Value::Null);
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_learned_scorer_stats: no data"))
}

#[no_mangle]
pub extern "C" fn cf_effective_scorer_weights(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_effective_scorer_weights: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.learned_scorer_stats();
    let mut weights = serde_json::Map::new();
    for f in &stats.factors {
        weights.insert(f.name.clone(), serde_json::json!({
            "delta": f.delta,
            "min_delta": f.min_delta,
            "max_delta": f.max_delta,
        }));
    }
    let json = serde_json::json!({
        "model_version": stats.model_version,
        "baseline_version": stats.baseline_version,
        "factors": weights,
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_effective_scorer_weights: no data"))
}

// ── Layer 7: Intervention Ledger ─────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_start_intervention(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_start_intervention: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_start_intervention: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_start_intervention: no data")
    };
    use crate::organ::intervention::{ActionType, ReversalCost};
    let realm = p["realm"].as_str().unwrap_or("coding").to_string();
    let session_id = p["session_id"].as_str().unwrap_or("").to_string();
    let task_id = p["task_id"].as_u64();
    let agent_id = p["agent_id"].as_str().unwrap_or("").to_string();
    let domain = p["domain"].as_str().unwrap_or("").to_string();
    let intent = p["intent"].as_str().unwrap_or("").to_string();
    let action_type = ActionType::from_u8(p["action_type"].as_u64().unwrap_or(0) as u8);
    let action_ref = p["action_ref"].as_str().unwrap_or("").to_string();
    let preconditions = p["preconditions"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let expected_observables = p["expected_observables"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let reversal_cost = ReversalCost::from_u8(p["reversal_cost"].as_u64().unwrap_or(0) as u8);
    match handle.field.start_intervention(
        realm, session_id, task_id, agent_id, domain, intent,
        action_type, action_ref, preconditions, expected_observables, reversal_cost,
    ) {
        Ok(id) => {
            let json = serde_json::json!({ "intervention_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_start_intervention: no data"))
        }
        Err(e) => json_null(format!("cf_start_intervention: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_add_observation(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_add_observation: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_add_observation: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_add_observation: no data")
    };
    use crate::organ::intervention::ObservationKind;
    let intervention_id = match p["intervention_id"].as_u64() {
        Some(id) => id, None => return json_null("cf_add_observation: no data")
    };
    let kind = ObservationKind::from_u8(p["kind"].as_u64().unwrap_or(0) as u8);
    let evidence_refs = p["evidence_refs"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()).unwrap_or_default();
    let summary = p["summary"].as_str().unwrap_or("").to_string();
    let confidence = p["confidence"].as_f64().unwrap_or(1.0) as f32;
    match handle.field.add_observation(intervention_id, kind, evidence_refs, summary, confidence) {
        Ok(Some(oid)) => {
            let json = serde_json::json!({ "observation_id": oid });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_add_observation: no data"))
        }
        _ => json_null("cf_add_observation: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_close_intervention(
    h: *mut CfHandle, intervention_id: u64, status: u8,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    use crate::organ::intervention::InterventionStatus;
    match handle.field.close_intervention(intervention_id, InterventionStatus::from_u8(status)) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_record_attribution(
    h: *mut CfHandle, params_json: *const c_char,
) -> c_int {
    if h.is_null() || params_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return -1
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return -1
    };
    use crate::organ::intervention::AttributionClass;
    let intervention_id = match p["intervention_id"].as_u64() { Some(id) => id, None => return -1 };
    let primary_class = AttributionClass::from_u8(p["primary_class"].as_u64().unwrap_or(9) as u8);
    let secondary_class = p["secondary_class"].as_u64().map(|v| AttributionClass::from_u8(v as u8));
    let confidence_delta = p["confidence_delta"].as_f64().unwrap_or(0.5) as f32;
    let surprise_id = p["surprise_id"].as_u64();
    let debt_ids = p["debt_ids"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()).unwrap_or_default();
    let source_memory_ids = p["source_memory_ids"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()).unwrap_or_default();
    let skill_memory_ids = p["skill_memory_ids"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()).unwrap_or_default();
    let note = p["note"].as_str().map(|s| s.to_string());
    match handle.field.record_attribution(
        intervention_id, primary_class, secondary_class, confidence_delta,
        surprise_id, debt_ids, source_memory_ids, skill_memory_ids, note,
    ) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_query_interventions(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_query_interventions: null argument"); }
    let handle = unsafe { &*h };
    let p: serde_json::Value = if params_json.is_null() {
        serde_json::Value::Null
    } else {
        let s = unsafe { match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s, Err(_) => return json_null("cf_query_interventions: no data")
        }};
        serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
    };
    use crate::organ::intervention::InterventionStatus;
    let realm = p["realm"].as_str();
    let session_id = p["session_id"].as_str();
    let status = p["status"].as_u64().map(|v| InterventionStatus::from_u8(v as u8));
    let limit = p["limit"].as_u64().unwrap_or(50) as usize;
    let results = handle.field.query_interventions(realm, session_id, status, limit);
    let json = serde_json::to_value(&results).unwrap_or(serde_json::Value::Array(vec![]));
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_interventions: no data"))
}

#[no_mangle]
pub extern "C" fn cf_get_intervention(
    h: *const CfHandle, intervention_id: u64,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_get_intervention: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.get_intervention(intervention_id) {
        Some(rec) => {
            let json = serde_json::to_value(&rec).unwrap_or(serde_json::Value::Null);
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_get_intervention: no data"))
        }
        None => json_null("cf_get_intervention: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_intervention_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_intervention_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.intervention_stats();
    let json = serde_json::to_value(&stats).unwrap_or(serde_json::Value::Null);
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_intervention_stats: no data"))
}

#[no_mangle]
pub extern "C" fn cf_list_open_interventions(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_list_open_interventions: null argument"); }
    let handle = unsafe { &*h };
    let results = handle.field.list_open_interventions();
    let json = serde_json::to_value(&results).unwrap_or(serde_json::Value::Array(vec![]));
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_list_open_interventions: no data"))
}

#[no_mangle]
pub extern "C" fn cf_close_stale_interventions(
    h: *mut CfHandle, threshold_ms: i64,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.close_stale_interventions(threshold_ms) {
        Ok(count) => count as c_int,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_auto_resolve_debts(h: *mut CfHandle, threshold: f32) -> *mut c_char {
    if h.is_null() { return json_null("cf_auto_resolve_debts: null argument"); }
    let handle = unsafe { &*h };
    let resolved = handle.field.auto_resolve_debts(threshold).unwrap_or(0);
    let json = serde_json::json!({ "resolved_count": resolved });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_auto_resolve_debts: no data"))
}

// ── Agent Protocol Memory (Layer 8) ──────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_register_task(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_register_task: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_register_task: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_register_task: no data")
    };
    let goal = p["goal"].as_str().unwrap_or("").to_string();
    let constraints: Vec<String> = p["constraints"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let acceptance_criteria: Vec<String> = p["acceptance_criteria"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    let realm = p["realm"].as_str().unwrap_or("coding").to_string();
    let session_id = p["session_id"].as_str().unwrap_or("").to_string();
    let priority = p["priority"].as_u64().unwrap_or(5) as u8;
    let parent_task_id = p["parent_task_id"].as_u64();
    let deadline_ms = p["deadline_ms"].as_i64();
    let tags: Vec<String> = p["tags"].as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();
    match handle.field.register_task(
        goal, constraints, acceptance_criteria,
        realm, session_id, priority, parent_task_id, deadline_ms, tags,
    ) {
        Ok(id) => {
            let json = serde_json::json!({ "task_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_register_task: no data"))
        }
        Err(e) => json_null(format!("cf_register_task: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_update_task(
    h: *mut CfHandle, params_json: *const c_char,
) -> c_int {
    if h.is_null() || params_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return -1
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return -1
    };
    let task_id = match p["task_id"].as_u64() { Some(id) => id, None => return -1 };
    let status: Option<u8> = p["status"].as_u64().map(|v| v as u8);
    let add_intervention_id = p["add_intervention_id"].as_u64();
    let add_tag = p["add_tag"].as_str().map(|s| s.to_string());
    match handle.field.update_task(task_id, status, add_intervention_id, add_tag) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_add_delegation(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_add_delegation: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_add_delegation: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_add_delegation: no data")
    };
    let task_id = match p["task_id"].as_u64() { Some(id) => id, None => return json_null("cf_add_delegation: no data") };
    let from_agent = p["from_agent"].as_str().unwrap_or("").to_string();
    let to_agent = p["to_agent"].as_str().unwrap_or("").to_string();
    let handoff_note = p["handoff_note"].as_str().map(|s| s.to_string());
    match handle.field.add_delegation(task_id, from_agent, to_agent, handoff_note) {
        Ok(Some(id)) => {
            let json = serde_json::json!({ "delegation_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_add_delegation: no data"))
        }
        _ => json_null("cf_add_delegation: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_link_evidence(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_link_evidence: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_link_evidence: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_link_evidence: no data")
    };
    let task_id = match p["task_id"].as_u64() { Some(id) => id, None => return json_null("cf_link_evidence: no data") };
    let memory_id = match p["memory_id"].as_u64() { Some(id) => id, None => return json_null("cf_link_evidence: no data") };
    let produced_by = p["produced_by"].as_str().unwrap_or("").to_string();
    let evidence_kind = p["evidence_kind"].as_u64().unwrap_or(0) as u8;
    let relevance = p["relevance"].as_f64().unwrap_or(1.0) as f32;
    match handle.field.link_evidence(task_id, memory_id, produced_by, evidence_kind, relevance) {
        Ok(Some(id)) => {
            let json = serde_json::json!({ "evidence_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_link_evidence: no data"))
        }
        _ => json_null("cf_link_evidence: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_add_probe(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_add_probe: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_add_probe: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_add_probe: no data")
    };
    let task_id = match p["task_id"].as_u64() { Some(id) => id, None => return json_null("cf_add_probe: no data") };
    let question = p["question"].as_str().unwrap_or("").to_string();
    let expected_answerer = p["expected_answerer"].as_str().map(|s| s.to_string());
    let priority = p["priority"].as_u64().unwrap_or(5) as u8;
    match handle.field.add_probe(task_id, question, expected_answerer, priority) {
        Ok(Some(id)) => {
            let json = serde_json::json!({ "probe_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_add_probe: no data"))
        }
        _ => json_null("cf_add_probe: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_resolve_probe(
    h: *mut CfHandle, params_json: *const c_char,
) -> c_int {
    if h.is_null() || params_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return -1
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return -1
    };
    let probe_id = match p["probe_id"].as_u64() { Some(id) => id, None => return -1 };
    let status = p["status"].as_u64().unwrap_or(1) as u8;
    let answer = p["answer"].as_str().map(|s| s.to_string());
    match handle.field.resolve_probe(probe_id, status, answer) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_set_criterion(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_set_criterion: null argument"); }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_set_criterion: no data")
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return json_null("cf_set_criterion: no data")
    };
    let task_id = match p["task_id"].as_u64() { Some(id) => id, None => return json_null("cf_set_criterion: no data") };
    let criterion = p["criterion"].as_str().unwrap_or("").to_string();
    let is_met = p["is_met"].as_bool().unwrap_or(false);
    let evidence_note = p["evidence_note"].as_str().map(|s| s.to_string());
    match handle.field.set_criterion(task_id, criterion, is_met, evidence_note) {
        Ok(Some(id)) => {
            let json = serde_json::json!({ "criterion_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_set_criterion: no data"))
        }
        _ => json_null("cf_set_criterion: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_get_task(
    h: *const CfHandle, task_id: u64,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_get_task: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.get_task_full(task_id) {
        Some(view) => {
            let json = serde_json::to_value(&view).unwrap_or(serde_json::Value::Null);
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_get_task: no data"))
        }
        None => json_null("cf_get_task: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_query_tasks(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_query_tasks: null argument"); }
    let handle = unsafe { &*h };
    let p: serde_json::Value = if params_json.is_null() {
        serde_json::Value::Null
    } else {
        let s = unsafe { match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s, Err(_) => return json_null("cf_query_tasks: no data")
        }};
        serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
    };
    let realm = p["realm"].as_str();
    let session_id = p["session_id"].as_str();
    let status: Option<u8> = p["status"].as_u64().map(|v| v as u8);
    let priority: Option<u8> = p["priority"].as_u64().map(|v| v as u8);
    let limit = p["limit"].as_u64().unwrap_or(50) as usize;
    let results = handle.field.query_tasks(realm, session_id, status, priority, limit);
    let json = serde_json::to_value(&results).unwrap_or(serde_json::Value::Array(vec![]));
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_tasks: no data"))
}

#[no_mangle]
pub extern "C" fn cf_agent_protocol_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_agent_protocol_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.agent_protocol_stats();
    let json = serde_json::to_value(&stats).unwrap_or(serde_json::Value::Null);
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_agent_protocol_stats: no data"))
}

#[no_mangle]
pub extern "C" fn cf_auto_complete_tasks(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_auto_complete_tasks: null argument"); }
    let handle = unsafe { &*h };
    let completed = handle.field.auto_complete_tasks().unwrap_or(0);
    let json = serde_json::json!({ "completed_count": completed });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_auto_complete_tasks: no data"))
}

// ── Layer 9: Wisdom Homeostasis ───────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_enroll_wisdom_lineage(
    h: *mut CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() || params_json.is_null() { return json_null("cf_enroll_wisdom_lineage: null argument"); }
    let handle = unsafe { &*h };
    let s = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return json_null("cf_enroll_wisdom_lineage: no data")
    }};
    let p: serde_json::Value = serde_json::from_str(s).unwrap_or(serde_json::Value::Null);
    let candidate_id = match p["wisdom_candidate_id"].as_u64() {
        Some(v) => v, None => return json_null("cf_enroll_wisdom_lineage: no data")
    };
    let claim = p["claim"].as_str().unwrap_or("").to_string();
    let envelope_json = p["envelope"].to_string();
    let to_u64_vec = |v: &serde_json::Value| -> Vec<u64> {
        v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect()).unwrap_or_default()
    };
    match handle.field.enroll_wisdom_lineage(
        candidate_id, claim, envelope_json,
        to_u64_vec(&p["seed_episode_ids"]),
        to_u64_vec(&p["seed_surprise_ids"]),
        to_u64_vec(&p["seed_intervention_ids"]),
        to_u64_vec(&p["seed_debt_ids"]),
        p["ancestor_lineage_id"].as_u64(),
        p["derivation_relation"].as_str().map(|s| s.to_string()),
    ) {
        Ok(id) => {
            let json = serde_json::json!({ "lineage_id": id });
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_enroll_wisdom_lineage: no data"))
        }
        Err(e) => json_null(format!("cf_enroll_wisdom_lineage: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_transition_wisdom_lineage(
    h: *mut CfHandle, lineage_id: u64, new_state: u8,
    reason: *const c_char, task_id: u64,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    let reason_str = if reason.is_null() { "manual".to_string() } else {
        unsafe { CStr::from_ptr(reason).to_str().unwrap_or("manual").to_string() }
    };
    let tid = if task_id == 0 { None } else { Some(task_id) };
    match handle.field.transition_wisdom_lineage(lineage_id, new_state, reason_str, tid) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_close_rederive(
    h: *mut CfHandle, params_json: *const c_char,
) -> c_int {
    if h.is_null() || params_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let s = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return -1
    }};
    let p: serde_json::Value = serde_json::from_str(s).unwrap_or(serde_json::Value::Null);
    let lineage_id = match p["lineage_id"].as_u64() { Some(v) => v, None => return -1 };
    let action = p["action"].as_u64().unwrap_or(3) as u8;
    let new_envelope_json = if p["new_envelope"].is_null() { None } else {
        Some(p["new_envelope"].to_string())
    };
    let fork_claim = p["fork_claim"].as_str().map(|s| s.to_string());
    let fork_lineage_id = p["fork_lineage_id"].as_u64();
    match handle.field.close_rederive(lineage_id, action, new_envelope_json, fork_claim, fork_lineage_id) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_query_wisdom_lineages(
    h: *const CfHandle, params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_query_wisdom_lineages: null argument"); }
    let handle = unsafe { &*h };
    let p: serde_json::Value = if params_json.is_null() {
        serde_json::Value::Null
    } else {
        let s = unsafe { match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s, Err(_) => return json_null("cf_query_wisdom_lineages: no data")
        }};
        serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
    };
    let state_str = p["state"].as_str();
    let domain = p["domain"].as_str();
    let limit = p["limit"].as_u64().unwrap_or(50) as usize;
    let results = handle.field.query_wisdom_lineages(state_str, domain, limit);
    let json = serde_json::to_value(&results).unwrap_or(serde_json::Value::Array(vec![]));
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_wisdom_lineages: no data"))
}

#[no_mangle]
pub extern "C" fn cf_get_wisdom_lineage(
    h: *const CfHandle, lineage_id: u64,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_get_wisdom_lineage: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.get_wisdom_lineage(lineage_id) {
        Some(l) => {
            let json = serde_json::to_value(&l).unwrap_or(serde_json::Value::Null);
            CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_get_wisdom_lineage: no data"))
        }
        None => json_null("cf_get_wisdom_lineage: no data"),
    }
}

#[no_mangle]
pub extern "C" fn cf_wisdom_lineage_stats(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_wisdom_lineage_stats: null argument"); }
    let handle = unsafe { &*h };
    let stats = handle.field.wisdom_lineage_stats();
    let json = serde_json::to_value(&stats).unwrap_or(serde_json::Value::Null);
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_wisdom_lineage_stats: no data"))
}

#[no_mangle]
pub extern "C" fn cf_tick_lineage_staleness(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_tick_lineage_staleness: null argument"); }
    let handle = unsafe { &*h };
    let ids = handle.field.tick_lineage_staleness().unwrap_or_default();
    let count = ids.len();
    let json = serde_json::json!({ "transitioned_ids": ids, "count": count });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_tick_lineage_staleness: no data"))
}

#[no_mangle]
pub extern "C" fn cf_lineage_expiry_check(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_lineage_expiry_check: null argument"); }
    let handle = unsafe { &*h };
    let ids = handle.field.lineage_expiry_check();
    let count = ids.len();
    let json = serde_json::json!({ "expired_ids": ids, "count": count });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_lineage_expiry_check: no data"))
}

// ── Soul REPL Session Store FFI ─────────────────────────────────────────────

/// Get a REPL session namespace as JSON string. Returns null if not found.
/// Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_repl_session_get(
    h: *const CfHandle,
    session_id: *const c_char,
) -> *mut c_char {
    if h.is_null() || session_id.is_null() { return json_null("cf_repl_session_get: null argument"); }
    let handle = unsafe { &*h };
    let id_str = unsafe { match CStr::from_ptr(session_id).to_str() { Ok(s) => s, Err(_) => return json_null("cf_repl_session_get: no data") } };
    match handle.field.repl_session_get(id_str) {
        Some(ns) => CString::new(ns).map(|s| s.into_raw()).unwrap_or(json_null("cf_repl_session_get: no data")),
        None => json_null("cf_repl_session_get: no data"),
    }
}

/// Set/update a REPL session namespace. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_repl_session_set(
    h: *mut CfHandle,
    session_id: *const c_char,
    namespace_json: *const c_char,
    updated_ms: i64,
) -> c_int {
    if h.is_null() || session_id.is_null() || namespace_json.is_null() { return -1; }
    let handle = unsafe { &*h };
    let id_str = unsafe { match CStr::from_ptr(session_id).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    let ns_str = unsafe { match CStr::from_ptr(namespace_json).to_str() { Ok(s) => s, Err(e) => return handle.err(e) } };
    handle.field.repl_session_set(id_str, ns_str, updated_ms);
    0
}

/// Delete a REPL session. Returns 1 if deleted, 0 if not found.
#[no_mangle]
pub extern "C" fn cf_repl_session_delete(
    h: *mut CfHandle,
    session_id: *const c_char,
) -> c_int {
    if h.is_null() || session_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let id_str = unsafe { match CStr::from_ptr(session_id).to_str() { Ok(e) => e, Err(e) => return handle.err(e) } };
    if handle.field.repl_session_delete(id_str) { 1 } else { 0 }
}

/// Execute Python code in the REPL sandbox. Atomically gets session namespace,
/// executes code, persists updated namespace.
/// Returns JSON: {success, output, error, session_id, trajectory}.
/// Caller must free result with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_repl_execute(
    h: *mut CfHandle,
    session_id: *const c_char,
    code: *const c_char,
    reset: c_int,
    socket_path: *const c_char,
    max_output: c_int,
) -> *mut c_char {
    if h.is_null() || session_id.is_null() || code.is_null() { return json_null("cf_repl_execute: null argument"); }
    let handle = unsafe { &*h };
    let sid  = unsafe { match CStr::from_ptr(session_id).to_str() { Ok(s) => s, Err(_) => return json_null("cf_repl_execute: no data") } };
    let code = unsafe { match CStr::from_ptr(code).to_str()       { Ok(s) => s, Err(_) => return json_null("cf_repl_execute: no data") } };
    let sp   = if socket_path.is_null() { "" } else {
        unsafe { CStr::from_ptr(socket_path).to_str().unwrap_or("") }
    };
    let max = if max_output > 0 { max_output as usize } else { 10_000 };
    let json = handle.field.repl_execute(sid, code, reset != 0, sp, max);
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_repl_execute: no data"))
}

/// Set source_session on an existing memory (in-memory; persisted at next snapshot).
#[no_mangle]
pub extern "C" fn cf_set_source_session(
    h: *mut CfHandle,
    memory_id: u64,
    session_id: *const c_char,
) -> c_int {
    if h.is_null() || session_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let sid = match unsafe { CStr::from_ptr(session_id).to_str() } {
        Ok(s) => s,
        Err(e) => return handle.err(e),
    };
    match handle.field.set_source_session(memory_id, sid) {
        Ok(()) => 0,
        Err(e) => handle.err(e),
    }
}

#[repr(C)]
pub struct CfSpreadingHit {
    pub memory_id: u64,
    pub score:     f32,
}

#[no_mangle]
pub extern "C" fn cf_recall_spreading(
    handle: *mut CfHandle,
    query: *const c_char,
    k:     usize,
    realm: *const c_char,
    max_nodes: usize,
    max_entries_per_entity: usize,
    depth: u8,
    out_json: *mut c_char,
    out_json_len: usize,
) -> c_int {
    if handle.is_null() || query.is_null() || out_json.is_null() || out_json_len == 0 {
        return -1;
    }
    let h = unsafe { &*handle };
    let query_str = unsafe { std::ffi::CStr::from_ptr(query).to_string_lossy() };
    let realm_opt: Option<String> = if realm.is_null() {
        None
    } else {
        let s = unsafe { std::ffi::CStr::from_ptr(realm).to_string_lossy() };
        if s.is_empty() { None } else { Some(s.into_owned()) }
    };
    let results = h.field.recall_spreading(
        &query_str,
        k.clamp(1, 100),
        realm_opt.as_deref(),
        max_nodes.clamp(1, 512),
        max_entries_per_entity.clamp(1, 256),
        depth.min(4),
    );
    let arr: Vec<serde_json::Value> = results.iter().map(|r| serde_json::json!({
        "memory_id": r.memory_id,
        "score":     r.score,
        "text":      r.text,
        "kind":      r.kind,
        "realm":     r.realm,
    })).collect();
    let json = serde_json::json!({ "results": arr });
    let s = json.to_string();
    let bytes = s.as_bytes();
    let copy_len = bytes.len().min(out_json_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_json as *mut u8, copy_len);
        *out_json.add(copy_len) = 0;
    }
    results.len() as c_int
}

/// Session-level recall hit returned by cf_recall_session.
/// Scalar-only struct: session identifiers are NOT carried here — they are returned
/// separately as a JSON array via the `session_ids_json_out` out-param of cf_recall_session.
#[repr(C)]
#[derive(Clone)]
pub struct CfSessionHit {
    pub score: f32,
    pub chunk_count: u32,
    pub max_chunk_score: f32,
}

/// Session-level recall: groups chunk hits by source_session and scores with noisy-OR.
/// `query_embedding` — pre-computed embedding (may be null to use keyword-only path).
/// On return `session_ids_json` is set to a JSON array of session IDs in score order.
/// Returns number of sessions written (≤ hits_cap), or -1 on error.
#[no_mangle]
pub extern "C" fn cf_recall_session(
    h: *mut CfHandle,
    query_embedding: *const f32,
    embedding_len: usize,
    query_text: *const c_char,
    realm: *const c_char,
    k: usize,
    hits_buf: *mut CfSessionHit,
    hits_cap: usize,
    hits_written: *mut usize,
    session_ids_json_out: *mut *mut c_char,
) -> c_int {
    if h.is_null() || hits_buf.is_null() || hits_written.is_null() { return -1; }
    let handle = unsafe { &*h };

    let embedding: Option<&[f32]> = if query_embedding.is_null() || embedding_len == 0 {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(query_embedding, embedding_len) })
    };
    let qtext = if query_text.is_null() { "" } else {
        match unsafe { CStr::from_ptr(query_text).to_str() } {
            Ok(s) => s,
            Err(e) => return handle.err(e),
        }
    };
    let realm_str = if realm.is_null() { None } else {
        match unsafe { CStr::from_ptr(realm).to_str() } {
            Ok(s) => Some(s),
            Err(e) => return handle.err(e),
        }
    };

    match handle.field.recall_session(embedding, qtext, k, realm_str) {
        Ok(hits) => {
            let n = hits.len().min(hits_cap);
            unsafe { *hits_written = n; }
            let session_ids: Vec<&str> = hits[..n].iter().map(|h| h.session_id.as_str()).collect();
            let ids_json = serde_json::to_string(&session_ids).unwrap_or_else(|_| "[]".into());

            for (i, hit) in hits[..n].iter().enumerate() {
                unsafe {
                    let slot = &mut *hits_buf.add(i);
                    slot.score = hit.score;
                    slot.chunk_count = hit.chunk_count;
                    slot.max_chunk_score = hit.max_chunk_score;
                }
            }
            // Best evidence strings packed as JSON array
            let evidence: Vec<&str> = hits[..n].iter().map(|h| h.best_evidence.as_str()).collect();
            let evidence_json = serde_json::to_string(&evidence).unwrap_or_else(|_| "[]".into());
            // Return both as a single JSON object
            let combined = format!(r#"{{"session_ids":{},"evidence":{}}}"#, ids_json, evidence_json);
            if !session_ids_json_out.is_null() {
                match CString::new(combined) {
                    Ok(cs) => unsafe { *session_ids_json_out = cs.into_raw(); },
                    Err(_) => unsafe { *session_ids_json_out = std::ptr::null_mut(); },
                }
            }
            0
        }
        Err(e) => handle.err(e),
    }
}

/// List all REPL sessions as JSON array. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_repl_session_list(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_repl_session_list: null argument"); }
    let handle = unsafe { &*h };
    let json = handle.field.repl_session_list();
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_repl_session_list: no data"))
}

// ── Contradiction detection FFI ───────────────────────────────────────────────

/// Detect contradictions for a memory already stored in the field.
/// Builds a transient ContradictionIndex from realm peers, then runs
/// detect_for_new_memory. Returns JSON array of ContradictionCandidate.
/// Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_detect_contradictions(
    h: *const CfHandle,
    memory_id: u64,
    realm_ptr: *const c_char,
) -> *mut c_char {
    if h.is_null() || realm_ptr.is_null() {
        return json_null("cf_detect_contradictions: null argument");
    }
    let handle = unsafe { &*h };
    let realm = match unsafe { CStr::from_ptr(realm_ptr) }.to_str() {
        Ok(s) => s,
        Err(_) => return json_null("cf_detect_contradictions: no data"),
    };

    // Gather peers and target content in a single lock window
    let (peer_pairs, target_content): (Vec<(u64, Vec<u8>)>, Vec<u8>) = {
        let payloads = handle.field.payloads.read();
        let members = handle.field.realm_members.read();
        let ids: Vec<u64> = match members.get(realm) {
            Some(s) => s.iter().copied().collect(),
            None => vec![],
        };
        let peers = ids.iter()
            .filter(|&&id| id != memory_id)
            .filter_map(|&id| payloads.get(&id).map(|p| (id, p.content.clone())))
            .collect();
        let target = match payloads.get(&memory_id) {
            Some(p) => p.content.clone(),
            None => return CString::new("[]").map(|s| s.into_raw()).unwrap_or(json_null("cf_detect_contradictions: no data")),
        };
        (peers, target)
    };

    use crate::contradiction::{ContradictionIndex, parse_claim_atoms};
    let mut index = ContradictionIndex::new();

    // Register peers first
    for (pid, content) in &peer_pairs {
        let atoms = parse_claim_atoms(*pid, realm, content);
        if !atoms.is_empty() {
            index.register_atoms(*pid, atoms);
        }
    }

    let candidates = index.detect_for_new_memory(memory_id, &target_content, realm);
    let json = serde_json::to_string(&candidates).unwrap_or_else(|_| "[]".into());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_detect_contradictions: no data"))
}

/// Background scan: detect contradictions across all memories in a realm.
/// Returns JSON array of ContradictionCandidate (up to `limit`).
/// Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_scan_contradictions(
    h: *const CfHandle,
    realm_ptr: *const c_char,
    limit: u32,
) -> *mut c_char {
    if h.is_null() || realm_ptr.is_null() {
        return json_null("cf_scan_contradictions: null argument");
    }
    let handle = unsafe { &*h };
    let realm = match unsafe { CStr::from_ptr(realm_ptr) }.to_str() {
        Ok(s) => s,
        Err(_) => return json_null("cf_scan_contradictions: no data"),
    };

    let pairs: Vec<(u64, Vec<u8>)> = {
        let payloads = handle.field.payloads.read();
        let members = handle.field.realm_members.read();
        let ids = match members.get(realm) {
            Some(s) => s.iter().copied().collect::<Vec<_>>(),
            None => vec![],
        };
        ids.into_iter()
            .filter_map(|id| payloads.get(&id).map(|p| (id, p.content.clone())))
            .collect()
    };

    use crate::contradiction::ContradictionIndex;
    let mut index = ContradictionIndex::new();
    let candidates = index.scan_realm(realm, &pairs, limit as usize);
    let json = serde_json::to_string(&candidates).unwrap_or_else(|_| "[]".into());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_scan_contradictions: no data"))
}

/// Resolve a contradiction pair by declaring a winner and loser.
/// Returns JSON ResolutionOps for the C++ handler to apply.
/// Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_resolve_contradiction(
    h: *const CfHandle,
    winner_id: u64,
    loser_id: u64,
    reason_ptr: *const c_char,
) -> *mut c_char {
    if h.is_null() {
        return json_null("cf_resolve_contradiction: null argument");
    }
    let reason = if reason_ptr.is_null() {
        "manual"
    } else {
        unsafe { CStr::from_ptr(reason_ptr) }.to_str().unwrap_or("manual")
    };

    // Build a minimal ContradictionIndex with just the two memories' content
    let (winner_content, loser_content) = {
        let payloads = handle_from(h).field.payloads.read();
        let wc = payloads.get(&winner_id).map(|p| p.content.clone()).unwrap_or_default();
        let lc = payloads.get(&loser_id).map(|p| p.content.clone()).unwrap_or_default();
        (wc, lc)
    };

    // Infer realm from winner
    let realm = {
        let payloads = handle_from(h).field.payloads.read();
        payloads.get(&winner_id).map(|p| p.realm.clone()).unwrap_or_default()
    };

    use crate::contradiction::{ContradictionIndex, ContradictionCandidate, CandidateStatus, parse_claim_atoms};
    let mut index = ContradictionIndex::new();

    let winner_atoms = parse_claim_atoms(winner_id, &realm, &winner_content);
    let loser_atoms  = parse_claim_atoms(loser_id,  &realm, &loser_content);
    if !winner_atoms.is_empty() { index.register_atoms(winner_id, winner_atoms); }
    if !loser_atoms.is_empty()  { index.register_atoms(loser_id,  loser_atoms); }

    // Synthesise a candidate with id=0 to resolve against
    let candidate_id = index.add_candidate(ContradictionCandidate {
        id: 0,
        memory_a: winner_id,
        memory_b: loser_id,
        score: 1.0,
        same_score: 1.0,
        opposition_score: 1.0,
        reason: reason.to_string(),
        status: CandidateStatus::Open,
        created_at_ms: 0,
    });

    match index.resolve(candidate_id, winner_id, loser_id, reason) {
        Some(ops) => {
            let json = serde_json::to_string(&ops).unwrap_or_else(|_| "{}".into());
            CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_resolve_contradiction: no data"))
        }
        None => CString::new("{}").map(|s| s.into_raw()).unwrap_or(json_null("cf_resolve_contradiction: no data")),
    }
}

// ── Introspection FFI ────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_symbol_stale_for_memory(
    h: *const CfHandle,
    memory_id: u64,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_symbol_stale_for_memory: null argument"); }
    let handle = unsafe { &*h };
    let reason = handle.field.symbol_stale_for_memory(memory_id);
    let json = serde_json::json!({
        "stale": reason.is_some(),
        "reason": reason,
    });
    CString::new(json.to_string()).map(|s| s.into_raw()).unwrap_or(json_null("cf_symbol_stale_for_memory: no data"))
}

#[no_mangle]
pub extern "C" fn cf_memory_claim_info(
    h: *const CfHandle,
    memory_id: u64,
    now_ms: i64,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_memory_claim_info: null argument"); }
    let handle = unsafe { &*h };
    let json = handle.field.memory_claim_info_json(memory_id, now_ms);
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_memory_claim_info: no data"))
}

// ── Symbol event log FFI ─────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn cf_log_symbol_event(
    h: *mut CfHandle,
    params_json: *const c_char,
) -> u64 {
    if h.is_null() || params_json.is_null() { return 0; }
    let handle = unsafe { &*h };
    let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s, Err(_) => return 0,
    }};
    let p: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v, Err(_) => return 0,
    };
    use crate::organ::symbol_events::SymbolEventKind;
    let symbol_name = p["symbol_name"].as_str().unwrap_or("").to_string();
    let file_path   = p["file_path"].as_str().unwrap_or("").to_string();
    let symbol_id   = p["symbol_id"].as_u64();
    let kind        = SymbolEventKind::from_u8(p["kind"].as_u64().unwrap_or(0) as u8);
    let session_id  = p["session_id"].as_str().unwrap_or("").to_string();
    let harness     = p["harness"].as_str().unwrap_or("").to_string();
    let memory_id   = p["memory_id"].as_u64().filter(|&v| v != 0);
    let notes       = p["notes"].as_str().map(|s| s.to_string());
    let timestamp_ms = p["timestamp_ms"].as_i64().unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    });
    handle.field.log_symbol_event(
        symbol_name, file_path, symbol_id, kind,
        session_id, harness, memory_id, notes, timestamp_ms,
    )
}

#[no_mangle]
pub extern "C" fn cf_query_symbol_events(
    h: *const CfHandle,
    params_json: *const c_char,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_query_symbol_events: null argument"); }
    let handle = unsafe { &*h };
    let (symbol_name, file_path, limit) = if params_json.is_null() {
        (None, None, 50usize)
    } else {
        let json_str = unsafe { match CStr::from_ptr(params_json).to_str() {
            Ok(s) => s, Err(_) => return json_null("cf_query_symbol_events: no data"),
        }};
        let p: serde_json::Value = serde_json::from_str(json_str).unwrap_or_default();
        let sn = p["symbol_name"].as_str().map(|s| s.to_string());
        let fp = p["file_path"].as_str().map(|s| s.to_string());
        let lim = p["limit"].as_u64().unwrap_or(50) as usize;
        (sn, fp, lim)
    };
    let json = handle.field.query_symbol_events(
        symbol_name.as_deref(),
        file_path.as_deref(),
        limit,
    );
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_query_symbol_events: no data"))
}

#[no_mangle]
pub extern "C" fn cf_mark_memory_invalidated(
    h: *mut CfHandle,
    memory_id: u64,
    reason: *const c_char,
) -> i32 {
    if h.is_null() || reason.is_null() { return -1; }
    let handle = unsafe { &*h };
    let reason_str = unsafe { match CStr::from_ptr(reason).to_str() {
        Ok(s) => s.to_string(), Err(_) => return -1,
    }};
    if handle.field.mark_memory_invalidated(memory_id, reason_str) { 0 } else { -1 }
}

#[no_mangle]
pub extern "C" fn cf_query_cross_harness_conflicts(
    h: *const CfHandle,
    realm: *const c_char,
    limit: u32,
    min_score: f32,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_query_cross_harness_conflicts: null argument"); }
    let handle = unsafe { &*h };
    let realm_str = if realm.is_null() { "".to_string() } else {
        unsafe { CStr::from_ptr(realm).to_str().unwrap_or("").to_string() }
    };
    let lim = if limit == 0 { 20 } else { limit as usize };
    let json = handle.field.query_cross_harness_conflicts(&realm_str, lim, min_score);
    match CString::new(json) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_query_cross_harness_conflicts: {e}")),
    }
}

/// Query triplets for `subject` valid in the world at `world_ms`.
/// Excludes tombstoned and superseded entries. Returns JSON array of TripletEntry objects.
/// Caller must free the returned string with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_triplet_query_as_of(
    h: *mut CfHandle,
    subject: *const c_char,
    world_ms: i64,
) -> *mut c_char {
    if h.is_null() || subject.is_null() { return json_null("cf_triplet_query_as_of: null argument"); }
    let handle = unsafe { &*h };
    let subject_str = unsafe {
        match CStr::from_ptr(subject).to_str() {
            Ok(s) => s,
            Err(_) => return json_null("cf_triplet_query_as_of: no data"),
        }
    };
    let entries = match handle.field.query_subject_as_of(subject_str, world_ms) {
        Ok(e) => e,
        Err(_) => return json_null("cf_triplet_query_as_of: no data"),
    };
    let json = serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string());
    match CString::new(json) {
        Ok(cs) => cs.into_raw(),
        Err(e) => json_null(format!("cf_triplet_query_as_of: {e}")),
    }
}

/// Mark triplet `old_id` as superseded by `new_id` at ingestion-time `at_ms`.
/// Returns 0 on success, -1 if handle is null.
#[no_mangle]
pub extern "C" fn cf_triplet_supersede(
    h: *mut CfHandle,
    old_id: u64,
    new_id: u64,
    at_ms: i64,
) -> c_int {
    if h.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.triplet_supersede(old_id, new_id, at_ms) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// BFS graph traversal from `start`. Returns JSON array of TraversalHit.
/// edge_types_json: JSON array of strings ([] = all). direction: "outgoing"|"incoming"|"both".
/// Caller must free with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_graph_traverse(
    h: *mut CfHandle,
    start: *const c_char,
    edge_types_json: *const c_char,
    max_hops: usize,
    max_results: usize,
    direction: *const c_char,
    max_edges: usize,
) -> *mut c_char {
    if h.is_null() || start.is_null() { return json_null("cf_graph_traverse: null argument"); }
    let handle = unsafe { &*h };
    let start_str = unsafe {
        match CStr::from_ptr(start).to_str() {
            Ok(s) => s,
            Err(_) => return json_null("cf_graph_traverse: no data"),
        }
    };
    let edge_types_str: Vec<String> = if edge_types_json.is_null() {
        vec![]
    } else {
        unsafe {
            CStr::from_ptr(edge_types_json).to_str().ok()
                .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                .unwrap_or_default()
        }
    };
    let edge_refs: Vec<&str> = edge_types_str.iter().map(|s| s.as_str()).collect();
    let dir = if direction.is_null() {
        crate::graph::Direction::Outgoing
    } else {
        unsafe {
            CStr::from_ptr(direction).to_str().ok()
                .map(crate::graph::Direction::from_str)
                .unwrap_or(crate::graph::Direction::Outgoing)
        }
    };
    let hits = handle.field.graph_traverse(
        start_str, &edge_refs, max_hops.min(4), max_results.clamp(1, 100), dir,
        max_edges.clamp(1, 50_000));
    let json = serde_json::to_string(&hits).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_graph_traverse: no data"))
}

/// Personalized PageRank over the triplet graph. Returns JSON array of [node, score] pairs.
/// seeds_json: JSON array of seed node strings. Caller must free with cf_free_string().
#[no_mangle]
pub extern "C" fn cf_graph_pagerank(
    h: *mut CfHandle,
    seeds_json: *const c_char,
    edge_types_json: *const c_char,
    damping: f32,
    iterations: u8,
    top_k: usize,
    max_nodes: usize,
    max_edges: usize,
) -> *mut c_char {
    if h.is_null() || seeds_json.is_null() { return json_null("cf_graph_pagerank: null argument"); }
    let handle = unsafe { &*h };
    let seeds_str: Vec<String> = unsafe {
        CStr::from_ptr(seeds_json).to_str().ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
            .unwrap_or_default()
    };
    let seed_refs: Vec<&str> = seeds_str.iter().take(32).map(|s| s.as_str()).collect();
    let edge_types_str: Vec<String> = if edge_types_json.is_null() {
        vec![]
    } else {
        unsafe {
            CStr::from_ptr(edge_types_json).to_str().ok()
                .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
                .unwrap_or_default()
        }
    };
    let edge_refs: Vec<&str> = edge_types_str.iter().map(|s| s.as_str()).collect();
    let results = handle.field.graph_pagerank(
        &seed_refs, &edge_refs, damping, iterations.min(100), top_k.clamp(1, 100),
        max_nodes.clamp(1, 512), max_edges.clamp(1, 100_000));
    let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_graph_pagerank: no data"))
}

// ── Interaction Ledger FFI ─────────────────────────────────────────────────────

/// Append an interaction event. `json_in` is a JSON object matching InteractionEvent
/// (event_id ignored; assigned by ledger). Returns assigned event_id via `out_event_id`.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn cf_ledger_append(
    h: *const CfHandle,
    json_in: *const c_char,
    out_event_id: *mut u64,
) -> c_int {
    if h.is_null() || json_in.is_null() || out_event_id.is_null() { return -1; }
    let handle = unsafe { &*h };
    let s = match unsafe { CStr::from_ptr(json_in) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let ev: crate::organ::interaction_ledger::InteractionEvent = match serde_json::from_str(s) {
        Ok(e) => e,
        Err(_) => return -1,
    };
    match handle.field.ledger_append(ev) {
        Ok(id) => { unsafe { *out_event_id = id; } 0 }
        Err(_) => -1,
    }
}

/// Query interaction events. `json_in` = `{"kind":"Retrieve","session_id":"...","since_ms":0,"limit":20}`.
/// All fields optional. Returns JSON array. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_ledger_query(
    h: *const CfHandle,
    json_in: *const c_char,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_ledger_query: null argument"); }
    let handle = unsafe { &*h };
    let args: serde_json::Value = if json_in.is_null() {
        serde_json::Value::Object(Default::default())
    } else {
        match unsafe { CStr::from_ptr(json_in) }.to_str().ok()
            .and_then(|s| serde_json::from_str(s).ok()) {
            Some(v) => v,
            None => return json_null("cf_ledger_query: no data"),
        }
    };
    use crate::organ::interaction_ledger::EventKind;
    let kind: Option<EventKind> = args["kind"].as_str().and_then(|k| match k {
        "Retrieve" => Some(EventKind::Retrieve),
        "Inject"   => Some(EventKind::Inject),
        "Outcome"  => Some(EventKind::Outcome),
        "Override" => Some(EventKind::Override),
        _ => None,
    });
    let session_id_owned = args["session_id"].as_str().map(|s| s.to_owned());
    let since_ms = args["since_ms"].as_i64();
    let limit = args["limit"].as_u64().unwrap_or(50) as usize;
    let results = match handle.field.ledger_query(kind, session_id_owned.as_deref(), since_ms, limit) {
        Ok(r) => r,
        Err(_) => return json_null("cf_ledger_query: no data"),
    };
    let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_ledger_query: no data"))
}

/// Compile Override events into versioned assertions. Returns number of new assertions via `out_count`.
#[no_mangle]
pub extern "C" fn cf_ledger_compile(h: *const CfHandle, out_count: *mut u32) -> c_int {
    if h.is_null() || out_count.is_null() { return -1; }
    let handle = unsafe { &*h };
    match handle.field.ledger_compile() {
        Ok(n) => { unsafe { *out_count = n as u32; } 0 }
        Err(_) => -1,
    }
}

/// Return contested assertion pairs as JSON array of [subject, predicate, [assertion_ids]].
/// Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_ledger_contradictions(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_ledger_contradictions: null argument"); }
    let handle = unsafe { &*h };
    let results = match handle.field.ledger_contradictions() {
        Ok(r) => r,
        Err(_) => return json_null("cf_ledger_contradictions: no data"),
    };
    let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_ledger_contradictions: no data"))
}


#[no_mangle]
pub extern "C" fn cf_predicate_attach(
    h: *const CfHandle,
    memory_id: u64,
    check_cmd: *const c_char,
) -> i64 {
    if h.is_null() || check_cmd.is_null() { return -1; }
    let handle = unsafe { &*h };
    let cmd = match unsafe { std::ffi::CStr::from_ptr(check_cmd) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    match handle.field.predicate_attach(memory_id, cmd) {
        Ok(id) => id as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn cf_predicate_run(h: *const CfHandle, memory_id: u64) -> *mut c_char {
    if h.is_null() { return json_null("cf_predicate_run: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.predicate_run(memory_id) {
        Ok(json) => CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_predicate_run: no data")),
        Err(e) => json_null(format!("cf_predicate_run: {e}")),
    }
}

#[no_mangle]
pub extern "C" fn cf_predicate_list(h: *const CfHandle, memory_id: u64) -> *mut c_char {
    if h.is_null() { return json_null("cf_predicate_list: null argument"); }
    let handle = unsafe { &*h };
    match handle.field.predicate_list(memory_id) {
        Ok(json) => CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_predicate_list: no data")),
        Err(e) => json_null(format!("cf_predicate_list: {e}")),
    }
}

#[inline]
fn handle_from(h: *const CfHandle) -> &'static CfHandle {
    unsafe { &*h }
}

/// Budgeted background competitive-weight refresh (consolidation sweep).
/// Returns the number refreshed, or -1 on null handle.
#[no_mangle]
pub extern "C" fn cf_cw_refresh_sweep(h: *mut CfHandle, budget: usize) -> i64 {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    handle.field.cw_refresh_sweep(budget) as i64
}

// ── Span Lane FFI ───────────────────────────────────────────────────────────

/// Query the span lane. Returns a JSON array of {text,class,count,last_ms,realm,
/// session,line,score}. `realm` NULL/empty = unscoped. No GPU, no LLM. Free with
/// cf_free_string.
#[no_mangle]
pub extern "C" fn cf_span_query(
    h: *mut CfHandle,
    query: *const c_char,
    realm: *const c_char,
    k: usize,
) -> *mut c_char {
    if h.is_null() || query.is_null() {
        return json_null("cf_span_query: null argument");
    }
    let handle = unsafe { &*h };
    let q = unsafe { match CStr::from_ptr(query).to_str() { Ok(s)=>s, Err(_)=>return json_null("cf_span_query: bad utf8") } };
    let realm_s = if realm.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(realm).to_str().ok().filter(|s| !s.is_empty()) }
    };
    let hits = handle.field.span_query(q, realm_s, if k == 0 { 6 } else { k });
    let arr: Vec<serde_json::Value> = hits.into_iter().map(|(text, class, count, last_ms, realm, session, line, score, memory_ids)| {
        serde_json::json!({
            "text": text, "class": class, "count": count, "last_ms": last_ms,
            "realm": realm, "session": session, "line": line, "score": score,
            "memory_ids": memory_ids,
        })
    }).collect();
    match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
        Ok(s) => s.into_raw(),
        Err(e) => json_null(format!("cf_span_query: {e}")),
    }
}

/// Directed relation transfer. JSON retains mode/indexed/results; reason is
/// null on success, no_source_relation/no_target_relation on abstention.
/// Each result includes all supporting target edges; relations cites a→b.
/// Free with cf_free_string. Errors are JSON so wrappers preserve explanations.
#[no_mangle]
pub extern "C" fn cf_recall_analogy(h: *const CfHandle, json_in: *const c_char) -> *mut c_char {
    let reply = |body: serde_json::Value| CString::new(body.to_string()).unwrap().into_raw();
    let error = |reason: &str| reply(serde_json::json!({"error": reason}));
    if h.is_null() || json_in.is_null() { return error("recall_analogy: null argument"); }
    let handle = unsafe { &*h };
    let args: serde_json::Value = match unsafe { CStr::from_ptr(json_in) }.to_str().ok()
        .and_then(|s| serde_json::from_str(s).ok()) {
        Some(v) => v, None => return error("recall_analogy: bad JSON"),
    };
    if args["mode"].as_str().unwrap_or("proportional") != "proportional" {
        return error("recall_analogy supports proportional mode only; use query_graph for graph queries");
    }
    let a = args["a"].as_str().unwrap_or("");
    let b = args["b"].as_str().unwrap_or("");
    let c = args["c"].as_str().unwrap_or("");
    if [a, b, c].iter().any(|s| s.trim().is_empty()) {
        return error("proportional mode needs a, b and c (a:b :: c:?)");
    }
    let limit = args["limit"].as_u64().unwrap_or(8).clamp(1, 100) as usize;
    let mut transfer = {
        let store = handle.field.triplet_store.read();
        crate::analogy::proportional(&store, a, b, c, crate::store::now_ms())
    };
    transfer.rank(limit);
    let edge_json = |e: &crate::organ::triplet::TripletEntry| serde_json::json!({
        "subject": e.subject, "predicate": e.predicate, "object": e.object,
        "memory_id": e.source_memory_id, "triplet_id": e.id,
        "weight": if e.weight.is_finite() { e.weight } else { 0.0 },
        "valid_from_ms": e.valid_from_ms,
    });
    let relations: Vec<_> = transfer.relations.iter().map(&edge_json).collect();
    let results: Vec<_> = transfer.results.iter().map(|edges| {
        let best = &edges[0];
        let payload = best.source_memory_id.and_then(|id| handle.field.get_memory(id).ok());
        let support: Vec<_> = edges.iter().map(&edge_json).collect();
        serde_json::json!({
            "id": best.source_memory_id.unwrap_or(0),
            "score": if best.weight.is_finite() { best.weight } else { 0.0 },
            "answer": best.object, "predicate": best.predicate,
            "realm": payload.as_ref().map(|p| p.realm.as_str()).unwrap_or(""),
            "text": payload.as_ref().map(|p| String::from_utf8_lossy(&p.content).into_owned()).unwrap_or_default(),
            "edges": support,
        })
    }).collect();
    reply(serde_json::json!({"mode": "proportional", "indexed": transfer.indexed,
        "results": results, "reason": transfer.reason, "relations": relations}))
}

/// Forward edge: verbatim atoms a recalled memory references. JSON array of
/// {text,class,count,realm}. Free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_span_for_memory(h: *mut CfHandle, memory_id: u64, k: usize) -> *mut c_char {
    if h.is_null() { return json_null("cf_span_for_memory: null argument"); }
    let handle = unsafe { &*h };
    let atoms = handle.field.span_for_memory(memory_id, if k == 0 { 6 } else { k });
    let arr: Vec<serde_json::Value> = atoms.into_iter().map(|(text, class, count, realm)| {
        serde_json::json!({ "text": text, "class": class, "count": count, "realm": realm })
    }).collect();
    match CString::new(serde_json::to_string(&arr).unwrap_or_default()) {
        Ok(s) => s.into_raw(),
        Err(e) => json_null(format!("cf_span_for_memory: {e}")),
    }
}

/// Link one memory's text into the span store (idempotent). Returns new spans.
#[no_mangle]
pub extern "C" fn cf_span_ingest_memory(
    h: *mut CfHandle,
    memory_id: u64,
    text: *const c_char,
    realm: *const c_char,
) -> i64 {
    if h.is_null() || text.is_null() { return -1; }
    let handle = unsafe { &*h };
    let t = unsafe { match CStr::from_ptr(text).to_str() { Ok(s)=>s, Err(_)=>return -1 } };
    let r = if realm.is_null() { "brahman" } else {
        unsafe { CStr::from_ptr(realm).to_str().unwrap_or("brahman") }
    };
    handle.field.span_ingest_memory(memory_id, t, r) as i64
}

/// Backfill the memory→span edge over all live memories. Returns JSON
/// {linked,new_spans}.
#[no_mangle]
pub extern "C" fn cf_span_backfill_memories(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_span_backfill_memories: null argument"); }
    let handle = unsafe { &*h };
    let (linked, new_spans) = handle.field.span_backfill_memories();
    let s = format!("{{\"linked\":{linked},\"new_spans\":{new_spans}}}");
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_span_backfill_memories: {e}")) }
}

/// #13 retro-backfill of SameSession edges. apply=false is a dry run (counts
/// only). Returns JSON {apply,sessions,memories_in_sessions,pairs,directed_edges,histogram}.
#[no_mangle]
pub extern "C" fn cf_densify_backfill(h: *mut CfHandle, apply: bool) -> *mut c_char {
    if h.is_null() { return json_null("cf_densify_backfill: null argument"); }
    let handle = unsafe { &*h };
    let (sessions, mems, pairs, hist) = handle.field.densify_backfill(apply);
    let s = format!(
        "{{\"apply\":{apply},\"sessions\":{sessions},\"memories_in_sessions\":{mems},\"pairs\":{pairs},\"directed_edges\":{},\"histogram\":{{\"n1\":{},\"n2\":{},\"n3_5\":{},\"n6_10\":{},\"n11_50\":{},\"n51plus\":{}}}}}",
        pairs * 2, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]
    );
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_densify_backfill: {e}")) }
}

/// Dense-kNN SemanticNeighbor edge backfill. apply=false is a dry run (counts
/// only). Returns JSON {apply,k,min_cos,memories_scanned,memories_with_neighbors,directed_edges}.
#[no_mangle]
pub extern "C" fn cf_semantic_backfill(h: *mut CfHandle, apply: bool, k: usize, min_cos: f32) -> *mut c_char {
    if h.is_null() { return json_null("cf_semantic_backfill: null argument"); }
    let handle = unsafe { &*h };
    let (scanned, with_nb, edges) = handle.field.semantic_backfill(apply, k, min_cos);
    let s = format!(
        "{{\"apply\":{apply},\"k\":{k},\"min_cos\":{min_cos},\"memories_scanned\":{scanned},\"memories_with_neighbors\":{with_nb},\"directed_edges\":{edges}}}"
    );
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_semantic_backfill: {e}")) }
}

/// Assoc-graph census. Returns JSON array of 6 per-EdgeType entries (wire
/// numbering: DerivedFrom=0..Contradicts=5), each {type,count,weights:[5]}
/// with weight buckets <0.05, 0.05-0.2, 0.2-0.5, 0.5-0.8, >=0.8.
#[no_mangle]
pub extern "C" fn cf_assoc_census(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_assoc_census: null argument"); }
    let handle = unsafe { &*h };
    let census = handle.field.assoc_census();
    const NAMES: [&str; 7] = ["DerivedFrom", "SameSession", "SameArtifact", "CoRetrieved", "Supports", "Contradicts", "SemanticNeighbor"];
    let entries: Vec<String> = census.iter().enumerate().map(|(i, (count, w))| {
        format!(
            "{{\"type\":\"{}\",\"count\":{count},\"weights\":[{},{},{},{},{}]}}",
            NAMES[i], w[0], w[1], w[2], w[3], w[4]
        )
    }).collect();
    let s = format!("[{}]", entries.join(","));
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_assoc_census: {e}")) }
}

/// Gate B: decay + floor-prune one assoc EdgeType (wire numbering, see
/// edge_type_from_u8). apply=false is a dry run. Returns JSON
/// {apply,edge_type,factor,prune_below,survivors,pruned}.
#[no_mangle]
pub extern "C" fn cf_assoc_decay(
    h: *mut CfHandle,
    edge_type: u8,
    factor: f32,
    prune_below: f32,
    apply: bool,
) -> *mut c_char {
    if h.is_null() { return json_null("cf_assoc_decay: null argument"); }
    let handle = unsafe { &*h };
    let et = edge_type_from_u8(edge_type);
    let (survivors, pruned) = handle.field.assoc_decay(et, factor, prune_below, apply);
    let s = format!(
        "{{\"apply\":{apply},\"edge_type\":{edge_type},\"factor\":{factor},\"prune_below\":{prune_below},\"survivors\":{survivors},\"pruned\":{pruned}}}"
    );
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_assoc_decay: {e}")) }
}

/// Full backfill over `projects_dir`. Returns JSON {unique,new,redacted}.
#[no_mangle]
pub extern "C" fn cf_span_backfill(h: *mut CfHandle, projects_dir: *const c_char) -> *mut c_char {
    if h.is_null() || projects_dir.is_null() {
        return json_null("cf_span_backfill: null argument");
    }
    let handle = unsafe { &*h };
    let dir = unsafe { match CStr::from_ptr(projects_dir).to_str() { Ok(s)=>s, Err(_)=>return json_null("cf_span_backfill: bad utf8") } };
    let (unique, new, redacted) = handle.field.span_backfill(std::path::Path::new(dir));
    let s = format!("{{\"unique\":{unique},\"new\":{new},\"redacted\":{redacted}}}");
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_span_backfill: {e}")) }
}

/// Incrementally ingest one transcript file. Returns count of new atoms.
#[no_mangle]
pub extern "C" fn cf_span_ingest(h: *mut CfHandle, transcript_path: *const c_char) -> i64 {
    if h.is_null() || transcript_path.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    let p = unsafe { match CStr::from_ptr(transcript_path).to_str() { Ok(s)=>s, Err(_)=>return -1 } };
    handle.field.span_ingest_transcript(std::path::Path::new(p)) as i64
}

/// Persist the span store iff it has unsaved live-path changes. Returns 1 if
/// a save ran, 0 if clean, -1 on null handle.
#[no_mangle]
pub extern "C" fn cf_span_flush(h: *mut CfHandle) -> c_int {
    if h.is_null() {
        return -1;
    }
    let handle = unsafe { &*h };
    handle.field.span_flush() as c_int
}

/// Returns JSON {unique,disk_bytes,redacted_total}.
#[no_mangle]
pub extern "C" fn cf_span_stats(h: *mut CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_span_stats: null argument"); }
    let handle = unsafe { &*h };
    let (unique, disk, redacted) = handle.field.span_stats();
    let s = format!("{{\"unique\":{unique},\"disk_bytes\":{disk},\"redacted_total\":{redacted}}}");
    match CString::new(s) { Ok(cs)=>cs.into_raw(), Err(e)=>json_null(format!("cf_span_stats: {e}")) }
}

/// Diagnostic WAL status; sync_count counts actual sync_data attempts.
#[no_mangle]
pub extern "C" fn cf_wal_status(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_wal_status: null argument"); }
    let field = &unsafe { &*h }.field;
    let log = field.log.read();
    let value = serde_json::json!({
        "pending_sync_count": log.pending_sync_count(),
        "sync_data_count": log.sync_count(),
    });
    CString::new(value.to_string()).unwrap().into_raw()
}

#[cfg(test)]
mod read_maintenance_tests {
    use super::*;

    #[test]
    fn recall_hydration_never_schedules_accesses_including_buffer_retry() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let h = cf_open(path.as_ptr(), std::ptr::null());
        assert!(!h.is_null());
        let field = &unsafe { &*h }.field;
        let embedding = vec![0.1; crate::ops::EMBED_DIM];
        let id = field.put_memory("wisdom", "test", b"hydration", &embedding,
            1.0, 0.001, 0, vec![], None, None).unwrap().0;
        let seq = field.log.read().last_seqno();
        for _ in 0..8 {
            let mut hits = vec![unsafe { std::mem::zeroed::<CfRecallHit>() }; 4];
            let mut n = 0;
            assert_eq!(cf_recall_semantic(h, embedding.as_ptr(), embedding.len(),
                std::ptr::null(), 4, hits.as_mut_ptr(), 4, &mut n, true), 0);
            assert_eq!(n, 1);
            let mut buf = [0u8; 128]; let mut written = 0;
            assert_eq!(cf_peek_content(h, id, buf.as_mut_ptr(), 1, &mut written), -2);
            assert_eq!(cf_peek_content(h, id, buf.as_mut_ptr(), buf.len(), &mut written), 0);
            assert_eq!(cf_get_kind(h, id, buf.as_mut_ptr(), buf.len()), 0);
            assert_eq!(cf_get_realm(h, id, buf.as_mut_ptr(), buf.len()), 0);
            assert!(field.pending_touches.lock().is_empty());
            field.drain_pending_touches().unwrap();
            assert_eq!(field.get_state(id).unwrap().access_count, 0);
            assert_eq!(field.log.read().last_seqno(), seq);
        }
        cf_close(h);
    }

    #[test]
    fn ffi_worker_drains_touches_and_syncs_without_a_queue() {
        let dir = tempfile::tempdir().unwrap();
        let path = CString::new(dir.path().to_str().unwrap()).unwrap();
        let h = cf_open(path.as_ptr(), std::ptr::null());
        assert!(!h.is_null());
        let field = &unsafe { &*h }.field;
        let id = field.put_memory("wisdom", "test", b"timer", &vec![0.1; crate::ops::EMBED_DIM],
            1.0, 0.001, 0, vec![], None, None).unwrap().0;
        field.sync_wal().unwrap();
        let seq = field.log.read().last_seqno();
        let syncs = field.log.read().sync_count();
        let mut buf = [0u8; 8192]; let mut written = 0;
        assert_eq!(cf_get_content(h, id, buf.as_mut_ptr(), buf.len(), &mut written), 0);
        assert_eq!(cf_get_memory_metadata(h, id, buf.as_mut_ptr(), buf.len(), &mut written), 0);
        assert_eq!(field.log.read().last_seqno(), seq);
        assert_eq!(field.log.read().sync_count(), syncs, "no per-recall sync");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(8);
        while std::time::Instant::now() < deadline {
            if field.get_state(id).unwrap().access_count == 1 && field.log.read().pending_sync_count() == 0 { break; }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(field.get_state(id).unwrap().access_count, 1);
        assert_eq!(field.log.read().pending_sync_count(), 0);
        assert_eq!(field.log.read().last_seqno(), seq + 1);
        cf_close(h);
    }
}
