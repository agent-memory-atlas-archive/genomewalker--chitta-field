//! C FFI for chitta-field.
//! Uses typed POD structs for hot-path calls. No JSON in recall path.
//! All functions return 0 on success, negative on error.
//! Errors readable via cf_last_error().

mod sessions_ledger;
pub use sessions_ledger::*;

mod organs;
pub use organs::*;

mod recall;
pub use recall::*;

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
// Optional derived-index work owns only its inputs/publication slot, never the
// field or instance lock. Shutdown may abandon it without waiting for native
// preparation; a late result can only update the abandoned publication slot.
enum StartupWork<T> { Complete(T), Stopped, Failed }
fn startup_work<T: Send + 'static>(
    stopped: &std::sync::mpsc::Receiver<()>,
    work: impl FnOnce() -> T + Send + 'static,
) -> StartupWork<T> {
    let (send, done) = std::sync::mpsc::channel();
    if let Err(error) = std::thread::Builder::new().name("chitta-startup-index".into())
        .spawn(move || { let _ = send.send(work()); }) {
        eprintln!("[field] deferred worker spawn failed: {error}");
        return StartupWork::Failed;
    }
    loop {
        match stopped.recv_timeout(std::time::Duration::from_millis(25)) {
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {},
            _ => return StartupWork::Stopped,
        }
        match done.try_recv() {
            Ok(value) => return StartupWork::Complete(value),
            Err(std::sync::mpsc::TryRecvError::Empty) => {},
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                eprintln!("[field] deferred worker exited without a result");
                return StartupWork::Failed;
            }
        }
    }
}

// Keep the field on the maintenance thread: cancellation releases every guard
// before cf_close joins it, so no detached worker can retain the instance lock.
fn prepare_startup_keywords(field: &ChittaField, stopped: &std::sync::mpsc::Receiver<()>) -> bool {
    let cancelled = || !matches!(stopped.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty));
    let tick = std::time::Duration::from_millis(25);
    let mut guard = loop {
        if cancelled() { return false; }
        if let Some(guard) = field.keyword_idx.try_upgradable_read_for(tick) { break guard; }
    };
    let Some(prepared) = guard.prepare_reverse_index(cancelled) else { return false; };
    loop {
        if cancelled() { return false; }
        match parking_lot::RwLockUpgradableReadGuard::try_upgrade_for(guard, tick) {
            Ok(mut writer) => { writer.publish_reverse_index(prepared); return true; }
            Err(reader) => guard = reader,
        }
    }
}

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
    // Diagnostic eager control for replica parity; production is staged.
    let opened = if std::env::var("CHITTA_STARTUP_EAGER").as_deref() == Ok("1") {
        ChittaField::open(data_dir)
    } else { ChittaField::open_for_serving(data_dir) };
    match opened {
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
                        if !wal_only {
                            // Triplets first: the largest snapshot section, and
                            // until 2026-09-20 its 1,899 ms index rebuild sat on
                            // the open path. cf_startup_indexes_ready covers it,
                            // so ordering it first keeps that gate's window from
                            // growing. The replication counts walk the graph and
                            // therefore follow immediately.
                            let begin = std::time::Instant::now();
                            if !matches!(startup_work(&stopped, field.triplet_store.startup_job()), StartupWork::Complete(())) { return; }
                            eprintln!("[field] deferred phase=triplets duration_ms={}", begin.elapsed().as_millis());
                            // Timed apart: it is the open path's other graph cost
                            // and the larger unknown in the pre-ready budget.
                            let begin = std::time::Instant::now();
                            let replication = field.clone();
                            if !matches!(startup_work(&stopped, move || replication.rebuild_replications_if_pending()), StartupWork::Complete(())) { return; }
                            eprintln!("[field] deferred phase=triplet_replication duration_ms={}", begin.elapsed().as_millis());
                            for (phase, job) in [
                                ("symbols", field.symbol_idx.startup_job()),
                                ("span_store", field.span_store.startup_job()),
                                ("hdc", field.hdc_idx.startup_job()),
                                ("event_tape_organs", field.cdawg.startup_job()),
                                ("episode_hdc", field.episode_hdc.startup_job()),
                            ] {
                                let begin = std::time::Instant::now();
                                if !matches!(startup_work(&stopped, job), StartupWork::Complete(())) { return; }
                                eprintln!("[field] deferred phase={phase} duration_ms={}", begin.elapsed().as_millis());
                            }
                            let begin = std::time::Instant::now();
                            if !prepare_startup_keywords(&field, &stopped) { return; }
                            eprintln!("[field] deferred phase=keyword_reverse duration_ms={}", begin.elapsed().as_millis());
                            let begin = std::time::Instant::now();
                            let snapshot = field.startup_turbo_snapshot.lock().take();
                            let cache_plan = snapshot.as_deref().and_then(|p| {
                                field.semantic_idx.read().plan_startup_turbo_cache(p)
                            });
                            let cached = match startup_work(&stopped, move || cache_plan.is_some_and(|plan| plan.load())) {
                                StartupWork::Complete(hit) => hit,
                                StartupWork::Stopped => return,
                                StartupWork::Failed => false,
                            };
                            if !cached {
                                let plan = field.semantic_idx.read().plan_turbo_rebuild(0);
                                if let Some(plan) = plan {
                                    if matches!(startup_work(&stopped, move || plan.build()), StartupWork::Stopped) { return; }
                                }
                            }
                            field.semantic_idx.write().prune_turbo_changes();
                            eprintln!("[field] deferred phase=turbo duration_ms={} cache_hit={cached}", begin.elapsed().as_millis());
                        }
                        let touch_interval = std::time::Duration::from_secs(
                            std::env::var("CHITTA_TOUCH_FLUSH_S").ok().and_then(|s| s.parse().ok()).unwrap_or(5).max(1));
                        let min_mutations = std::env::var("CHITTA_TURBO_REBUILD_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
                        let checkpoint_bytes = std::env::var("CHITTA_CHECKPOINT_WAL_MB")
                            .ok().and_then(|s| s.parse::<u64>().ok()).filter(|n| *n > 0)
                            .unwrap_or(16).saturating_mul(1024 * 1024);
                        let checkpoint_records = std::env::var("CHITTA_WAL_SNAPSHOT_RECORDS")
                            .ok().and_then(|s| s.parse::<u64>().ok()).filter(|n| *n > 0)
                            .unwrap_or(20_000);
                        let mut last_checkpoint_check = std::time::Instant::now();
                        let mut last_checkpoint_attempt = None::<std::time::Instant>;
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
                                if last_checkpoint_check.elapsed() >= std::time::Duration::from_secs(1) {
                                    // Never hold a log guard across the family writer. Failed saves
                                    // keep their watermark and retry with backoff, not every tick.
                                    let now = std::time::Instant::now();
                                    let due = {
                                        let log = field.log.read();
                                        log.checkpoint_ready(now)
                                            && log.checkpoint_due(checkpoint_bytes, checkpoint_records)
                                            && last_checkpoint_attempt.is_none_or(|last| now.duration_since(last) >= std::time::Duration::from_secs(60))
                                    };
                                    if due {
                                        let begin = std::time::Instant::now();
                                        last_checkpoint_attempt = Some(begin);
                                        let result = field.save_full_snapshot();
                                        eprintln!("[checkpoint] reason=wal_budget duration_ms={} result={result:?}", begin.elapsed().as_millis());
                                    }
                                    last_checkpoint_check = std::time::Instant::now();
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
        if let Err(e) = b.field.flush().and_then(|_| b.field.sync_wal()) {
            eprintln!("[field] shutdown flush failed: {e}");
        }
        eprintln!("[checkpoint] shutdown_wal_tail_records={}", b.field.log.read().checkpoint_tail_records());
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

// ── CEC: Event tape + CDAWG ──────────────────────────────────────────────────

// ── Phase 17 exports ─────────────────────────────────────────────────────────

// ── Triplet operations ────────────────────────────────────────────────────────

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

#[no_mangle]
pub extern "C" fn cf_symbol_count(h: *const CfHandle) -> usize {
    if h.is_null() {
        return 0;
    }
    unsafe { (*h).field.symbol_count() }
}

// ── Lite Encoder ─────────────────────────────────────────────────────────────

// ── Domain Event Log ──────────────────────────────────────────────────────────

// ── Session high-level FFI ────────────────────────────────────────────────────

// ── Transcript high-level FFI ─────────────────────────────────────────────────

// ── Task / Sadhana / Dream high-level FFI ────────────────────────────────────

// ── User Model FFI ────────────────────────────────────────────────────────────

// ── Theme FFI ─────────────────────────────────────────────────────────────────

// ── Analytics high-level FFI ──────────────────────────────────────────────────

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

// ── Additional FFI for DuckDB removal migration ──────────────────────────────

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

// ── Skill Registry FFI ──────────────────────────────────────────────────────

// ── Agent Registry FFI ──────────────────────────────────────────────────────

// ── Layer 1: Executable Constraints ─────────────────────────────────────────

// ── Layer 2: Trigger Tissue ─────────────────────────────────────────────────

// ── Layer 3: Predictive Memory ──────────────────────────────────────────────

// ── Layer 4: Surprise Memory ──────────────────────────────────────────────

// ── Layer 5: Epistemic Debt ───────────────────────────────────────────────

// ── Layer 6: Integration Kernel ───────────────────────────────────────────

// ── Autonomous Learning FFI ───────────────────────────────────────────────

// ── Layer 7: Intervention Ledger ─────────────────────────────────────────

// ── Agent Protocol Memory (Layer 8) ──────────────────────────────────────────

// ── Layer 9: Wisdom Homeostasis ───────────────────────────────────────────────

// ── Soul REPL Session Store FFI ─────────────────────────────────────────────

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

// ── Contradiction detection FFI ───────────────────────────────────────────────

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

// ── Interaction Ledger FFI ─────────────────────────────────────────────────────

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

#[cfg(test)]
mod deferred_startup_tests {
    use super::*;

    #[test]
    fn keyword_startup_stop_releases_field_before_same_process_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let field = ChittaField::open(dir.path().to_path_buf()).unwrap();
        let (stop, stopped) = std::sync::mpsc::channel();
        stop.send(()).unwrap();
        assert!(!prepare_startup_keywords(&field, &stopped));
        drop(field);
        let reopened = ChittaField::open(dir.path().to_path_buf()).unwrap();
        drop(reopened);
    }

    #[test]
    fn stop_does_not_join_blocked_optional_work() {
        let (stop, stopped) = std::sync::mpsc::channel();
        let (entered, started) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let (finished, finish) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let result = startup_work(&stopped, move || {
                entered.send(()).unwrap();
                released.recv().unwrap();
            });
            finished.send(matches!(result, StartupWork::Stopped)).unwrap();
        });
        started.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        stop.send(()).unwrap();
        // The optional job remains blocked until after maintenance has stopped.
        let stopped_promptly = finish.recv_timeout(std::time::Duration::from_secs(1));
        release.send(()).unwrap();
        waiter.join().unwrap();
        assert_eq!(stopped_promptly.unwrap(), true);
    }
}

/// Whether tools depending on secondary startup indexes may run without loading.
#[no_mangle]
pub unsafe extern "C" fn cf_startup_indexes_ready(handle: *mut CfHandle) -> bool {
    let Some(handle) = handle.as_ref() else { return false; };
    let f = &handle.field;
    f.symbol_idx.is_ready() && f.span_store.is_ready() && f.hdc_idx.is_ready()
        && f.cdawg.is_ready() && f.episode_hdc.is_ready() && f.triplet_store.is_ready()
        // The graph being queryable is not the same as the counts derived from
        // it being current: a deferring open skips replication::rebuild, so
        // until the maintenance thread has redone it every state still carries
        // the snapshot's replication_count. Holding the gate keeps that stale
        // count out of every tool routed through it.
        && !f.triplet_replication_pending.load(std::sync::atomic::Ordering::Acquire)
}

/// Whether the triplet graph is queryable without blocking on its index
/// rebuild. The exact/hybrid triplet lanes and spreading activation answer
/// `loading` until this is true, the way the code-intel tools already do.
#[no_mangle]
pub unsafe extern "C" fn cf_triplets_ready(handle: *mut CfHandle) -> bool {
    let Some(handle) = handle.as_ref() else { return false; };
    handle.field.triplet_store.is_ready()
}
