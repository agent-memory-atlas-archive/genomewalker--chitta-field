//! organs C entry points.

use super::*;

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
