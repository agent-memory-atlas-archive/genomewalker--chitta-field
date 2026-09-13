use std::collections::HashMap;

/// A single message event stored in the registry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MsgEvent {
    pub event_id: u64,
    pub domain: String,
    pub kind: String,
    pub target: String,
    pub payload_json: String,
    pub realm: String,
    pub ts_ms: i64,
}

/// In-memory registry of domain events indexed by (domain, kind, target).
/// Populated during WAL replay and live via cf_emit_event.
#[derive(Debug, Default)]
pub struct MsgRegistry {
    /// Key: target (entity_id), Value: events in insertion order.
    by_target: HashMap<String, Vec<MsgEvent>>,
}

impl MsgRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Durable task ledger and native session history, in per-target causal order.
    pub fn ledger_session_events(&self) -> Vec<MsgEvent> {
        let mut events: Vec<_> = self.by_target.values().flat_map(|v| v.iter())
            .filter(|e| e.domain == "session" || (e.domain == "ledger" && e.kind == "task_records"))
            .cloned().collect();
        events.sort_by_key(|e| e.ts_ms); // stable: preserve same-target timestamp ties
        events
    }

    pub fn insert(&mut self, event: MsgEvent) {
        self.by_target
            .entry(event.target.clone())
            .or_default()
            .push(event);
    }

    /// Query events by domain, kind, and target. Returns up to `limit` events
    /// in insertion order (oldest first).
    pub fn get_events(
        &self,
        domain: &str,
        kind: &str,
        target: &str,
        limit: usize,
    ) -> Vec<&MsgEvent> {
        let Some(events) = self.by_target.get(target) else {
            return Vec::new();
        };
        events
            .iter()
            .filter(|e| e.domain == domain && e.kind == kind)
            .take(limit)
            .collect()
    }

    /// Query all events matching domain+kind across all targets.
    /// Returns up to `limit` events sorted by ts_ms descending (newest first).
    pub fn get_events_by_domain_kind(
        &self,
        domain: &str,
        kind: &str,
        limit: usize,
    ) -> Vec<&MsgEvent> {
        let mut result: Vec<&MsgEvent> = self
            .by_target
            .values()
            .flat_map(|events| events.iter())
            .filter(|e| e.domain == domain && e.kind == kind)
            .collect();
        result.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        result.truncate(limit);
        result
    }

    /// Look up a single event by its event_id across all targets.
    pub fn get_event_by_id(&self, event_id: u64) -> Option<&MsgEvent> {
        self.by_target
            .values()
            .flat_map(|events| events.iter())
            .find(|e| e.event_id == event_id)
    }

    /// Check whether any event for `target` matches domain+kind.
    pub fn has_event(&self, domain: &str, kind: &str, target: &str) -> bool {
        self.by_target
            .get(target)
            .map(|events| events.iter().any(|e| e.domain == domain && e.kind == kind))
            .unwrap_or(false)
    }
}

impl crate::organ::OrganApply for MsgRegistry {
    /// Organ-owned WAL replay (THEORY.md §8 Phase 2).
    fn apply(&mut self, op: crate::ops::Op) -> Option<crate::ops::Op> {
        use crate::ops::Op;
        match op {
            Op::MsgEvent(ev) => {
                use crate::organ::msg::MsgEvent;
                let payload_str = String::from_utf8(ev.payload_json).unwrap_or_default();
                self.insert(MsgEvent {
                    event_id: ev.event_id,
                    domain: ev.domain,
                    kind: ev.kind,
                    target: ev.target,
                    payload_json: payload_str,
                    realm: ev.realm,
                    ts_ms: ev.ts_ms,
                });
                    None
                }
            other => Some(other),
        }
    }
}
