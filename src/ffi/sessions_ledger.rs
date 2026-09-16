//! sessions ledger C entry points.

use super::*;

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

    if handle.field.ablations.suppresses(&op) {
        unsafe { *out_event_id = 0; }
        return handle.ok();
    }

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

/// List all REPL sessions as JSON array. Caller must free with cf_free_string.
#[no_mangle]
pub extern "C" fn cf_repl_session_list(h: *const CfHandle) -> *mut c_char {
    if h.is_null() { return json_null("cf_repl_session_list: null argument"); }
    let handle = unsafe { &*h };
    let json = handle.field.repl_session_list();
    CString::new(json).map(|s| s.into_raw()).unwrap_or(json_null("cf_repl_session_list: no data"))
}

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
