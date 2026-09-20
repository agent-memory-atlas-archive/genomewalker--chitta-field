//! recall C entry points.

use super::*;

/// Internal derived-index access; no new daemon tool or snapshot field.
#[no_mangle]
pub extern "C" fn cf_source_anchors(h: *mut CfHandle, request: *const c_char,
    buf: *mut u8, cap: usize, written: *mut usize) -> c_int {
    if h.is_null() || request.is_null() || buf.is_null() || written.is_null() { return -1; }
    let handle = unsafe { &*h };
    let request = unsafe { CStr::from_ptr(request) }.to_bytes();
    let request: serde_json::Value = match serde_json::from_slice(request) { Ok(v) => v, Err(e) => return handle.err(e) };
    let output = handle.field.source_anchors(&request).to_string();
    unsafe { *written = output.len(); }
    if output.len() >= cap { return -2; }
    unsafe { std::ptr::copy_nonoverlapping(output.as_ptr(), buf, output.len()); *buf.add(output.len()) = 0; }
    handle.ok()
}

/// Share the profiling flag with the C++ stages around the FFI calls.
#[no_mangle]
pub extern "C" fn cf_recall_profile_enabled() -> c_int { crate::profile::enabled() as c_int }

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
    // Ungated by choice, unlike the exact/hybrid recall lanes. Correct either
    // way (Deref drains first), and unreachable during the deferred window over
    // RPC: recall_analogy is not on the exempt list in field_handler.hpp, so the
    // startup-indexes gate -- which covers triplets since 2026-09-20 -- answers
    // loading before this runs. A direct FFI caller blocks for the rebuild
    // instead, which is the right trade for a structural query.
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
