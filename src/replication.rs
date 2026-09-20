//! Independent replication is a union of session identities, never a count of
//! memories, source tools or graph edges. Provenance may follow source/derived_from
//! memory references (cycle-safe); only direct confirmations contribute, so a
//! chain of agreement cannot recursively amplify itself. Unknown provenance adds
//! no invented identity. The neutral floor is one, even with zero known sessions.
use crate::{field::ChittaField, ids::MemoryId, organ::triplet::TripletStore,
            payload::MemoryPayload, state::MemoryState};
use std::collections::{BTreeSet, HashMap, HashSet};

fn now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_millis() as i64
}

fn provenance_edge(predicate: &str) -> bool {
    matches!(predicate, "source" | "derived_from" | "merged_from" | "replication_session")
}

pub(crate) fn affected_edge(subject: &str, predicate: &str, object: &str) -> Vec<MemoryId> {
    if provenance_edge(predicate) { subject.parse().ok().into_iter().collect() }
    else if predicate == "confirms" { object.parse().ok().into_iter().collect() }
    else { Vec::new() }
}

fn insert_session(sessions: &mut BTreeSet<String>, value: &str) {
    let value = value.trim().strip_prefix("session:").unwrap_or(value.trim()).trim();
    if !value.is_empty() { sessions.insert(value.to_string()); }
}

fn provenance(id: MemoryId, payloads: &HashMap<MemoryId, MemoryPayload>,
              states: &HashMap<MemoryId, MemoryState>, graph: &TripletStore,
              at: i64, sessions: &mut BTreeSet<String>) {
    let mut pending = vec![id];
    let mut visited = HashSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) || states.get(&id).map_or(true, |s| s.deleted) { continue; }
        if let Some(session) = payloads.get(&id).and_then(|p| p.source_session.as_deref()) {
            insert_session(sessions, session);
        }
        for edge in graph.query_subject(&id.to_string(), at) {
            if graph.is_superseded_crate(edge.id) { continue; }
            match edge.predicate.as_str() {
                "replication_session" => insert_session(sessions, &edge.object),
                "source" | "derived_from" | "merged_from" => {
                    if let Ok(parent) = edge.object.parse::<MemoryId>() {
                        pending.push(parent);
                    } else if edge.predicate == "derived_from" || edge.object.starts_with("session:") {
                        // derived_from's nonnumeric target is a session (field_session.cpp).
                        // source's bare labels (mcp_tool, distillation, ...) are writers.
                        insert_session(sessions, &edge.object);
                    }
                }
                _ => {}
            }
        }
    }
}

fn sessions(id: MemoryId, payloads: &HashMap<MemoryId, MemoryPayload>,
            states: &HashMap<MemoryId, MemoryState>, graph: &TripletStore, at: i64) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    provenance(id, payloads, states, graph, at, &mut result);
    for edge in graph.query_object(&id.to_string(), at) {
        if edge.predicate == "confirms" && !graph.is_superseded_crate(edge.id) {
            if let Ok(witness) = edge.subject.parse() {
                provenance(witness, payloads, states, graph, at, &mut result);
            }
        }
    }
    result
}

pub(crate) fn rebuild(payloads: &HashMap<MemoryId, MemoryPayload>, graph: &TripletStore,
                      states: &mut HashMap<MemoryId, MemoryState>) {
    let at = now();
    let counts: Vec<_> = states.keys().map(|&id|
        (id, sessions(id, payloads, states, graph, at).len().max(1) as u32)).collect();
    for (id, count) in counts { states.get_mut(&id).unwrap().replication_count = count; }
}

impl ChittaField {
    /// Recompute every state's replication count. Walks the triplet graph, so a
    /// serving open that deferred the triplet rebuild skips it and the
    /// maintenance thread calls this once the graph is ready. No-op when the
    /// open already did it. Lock order matches field.rs: payloads -> states ->
    /// triplet_store.
    ///
    /// The flag clears only after the counts are in `states`, because
    /// `cf_startup_indexes_ready` reads it: clearing on entry would open that
    /// gate while every state still carried the snapshot's stale count. The
    /// maintenance thread is the only caller and calls it once.
    pub(crate) fn rebuild_replications_if_pending(&self) {
        use std::sync::atomic::Ordering;
        if !self.triplet_replication_pending.load(Ordering::Acquire) { return; }
        // Force the deferred triplet load while holding nothing, so the
        // payloads -> states -> triplet_store order below is a plain lock
        // acquisition and never a multi-second rebuild under states.write().
        drop(self.triplet_store.read());
        {
            let payloads = self.payloads.read();
            let mut states = self.states.write();
            let graph = self.triplet_store.read();
            rebuild(&payloads, &graph, &mut states);
        }
        self.triplet_replication_pending.store(false, Ordering::Release);
    }

    /// Recompute only the mutation's provenance dependents, outside recall. Lock
    /// order matches field.rs: payloads -> states -> triplet_store.
    pub(crate) fn refresh_replications(&self, seeds: &[MemoryId]) {
        if seeds.is_empty() { return; }
        let payloads = self.payloads.read();
        let mut states = self.states.write();
        let graph = self.triplet_store.read();
        let at = now();
        let mut pending = seeds.to_vec();
        let mut affected = HashSet::new();
        while let Some(id) = pending.pop() {
            if !affected.insert(id) { continue; }
            let key = id.to_string();
            for edge in graph.query_object(&key, at) {
                if provenance_edge(&edge.predicate) {
                    if let Ok(child) = edge.subject.parse() { pending.push(child); }
                }
            }
            for edge in graph.query_subject(&key, at) {
                if edge.predicate == "confirms" {
                    if let Ok(target) = edge.object.parse() { pending.push(target); }
                }
            }
        }
        for id in affected {
            let count = sessions(id, &payloads, &states, &graph, at).len().max(1) as u32;
            if let Some(state) = states.get_mut(&id) { state.replication_count = count; }
        }
    }

    /// Existing AddTriplet WAL/snapshot format preserves dedup observations even
    /// when no new payload is created, including sessions attached by the daemon.
    pub(crate) fn record_replication_session(&self, id: MemoryId, session: Option<&str>) -> crate::error::Result<()> {
        let mut normalized = BTreeSet::new();
        if let Some(session) = session { insert_session(&mut normalized, session); }
        for session in normalized {
            let exists = {
                let graph = self.triplet_store.read();
                graph.query_subject(&id.to_string(), now()).iter().any(|e|
                    e.predicate == "replication_session" && e.object == session
                        && !graph.is_superseded_crate(e.id))
            };
            if !exists {
                self.add_triplet(id.to_string(), "replication_session".into(), session, 1.0, None, None)?;
            }
        }
        Ok(())
    }

    pub(crate) fn replication_edge_ids(&self, id: u64) -> Vec<MemoryId> {
        self.triplet_store.read().entry_by_id_crate(id)
            .map(|e| affected_edge(&e.subject, &e.predicate, &e.object)).unwrap_or_default()
    }

    /// Offline audit of cached counts plus their evidence. Caller must open only
    /// a private store copy: open takes an exclusive lock and can write caches.
    pub fn replication_audit(&self) -> serde_json::Value {
        let payloads = self.payloads.read();
        let states = self.states.read();
        let graph = self.triplet_store.read();
        let at = now();
        let mut rows = Vec::new();
        let mut known_sessions = 0usize;
        let mut replicated = 0usize;
        for (&id, state) in states.iter().filter(|(_, s)| !s.deleted) {
            replicated += usize::from(state.replication_count >= 2);
            if let Some(p) = payloads.get(&id) {
                let sources: BTreeSet<_> = graph.query_subject(&id.to_string(), at).iter()
                    .filter(|e| e.predicate == "source").map(|e| e.object.clone()).collect();
                let derived = graph.query_subject(&id.to_string(), at).iter()
                    .any(|e| e.predicate == "derived_from");
                let evidence = sessions(id, &payloads, &states, &graph, at);
                known_sessions += usize::from(!evidence.is_empty());
                rows.push(serde_json::json!({"id":id.to_string(), "kind":p.kind,
                    "count":state.replication_count, "source_tool":p.source_tool,
                    "provenance":p.provenance, "sources":sources, "derived_from":derived,
                    "sessions":evidence}));
            }
        }
        rows.sort_by(|a,b| b["count"].as_u64().cmp(&a["count"].as_u64())
            .then_with(|| a["id"].as_str().unwrap().parse::<u64>().unwrap()
                .cmp(&b["id"].as_str().unwrap().parse::<u64>().unwrap())));
        let top20: Vec<_> = rows.iter().take(20).cloned().collect();
        let replicated_rows: Vec<_> = rows.into_iter().filter(|r| r["count"].as_u64().unwrap() >= 2).collect();
        serde_json::json!({"live_memories":states.values().filter(|s| !s.deleted).count(),
            "with_identifiable_sessions":known_sessions, "replicated_memories":replicated,
            "memories":replicated_rows, "top20":top20})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn put(field: &ChittaField, text: &str, session: Option<&str>) -> MemoryId {
        field.put_memory("wisdom", "test", text.as_bytes(), &[], 1.0, 0.001, 0,
            vec![], session.map(str::to_string), None).unwrap().0
    }
    fn count(field: &ChittaField, id: MemoryId) -> u32 {
        field.states.read()[&id].replication_count
    }
    fn edge(field: &ChittaField, a: MemoryId, pred: &str, b: impl ToString) -> u64 {
        field.add_triplet(a.to_string(), pred.into(), b.to_string(), 1.0, None, None).unwrap()
    }
    #[test]
    fn distinct_sessions_not_memories_or_writers_and_no_confirmation_echo() {
        let dir = tempfile::tempdir().unwrap();
        let f = ChittaField::open(dir.path().into()).unwrap();
        let a = put(&f, "a", Some("one"));
        let b = put(&f, "b", Some("one"));
        let c = put(&f, "c", Some("two"));
        let d = put(&f, "d", Some("three"));
        edge(&f, a, "source", "distillation");
        edge(&f, b, "confirms", a);
        assert_eq!(count(&f, a), 1);
        let confirmation = edge(&f, c, "confirms", a);
        edge(&f, d, "confirms", c);
        assert_eq!(count(&f, a), 2); // not 3: no recursive confirmations
        f.invalidate_triplet(confirmation).unwrap();
        assert_eq!(count(&f, a), 1);
        edge(&f, c, "confirms", a);
        f.forget(c).unwrap();
        assert_eq!(count(&f, a), 1);
    }
    #[test]
    fn dedup_and_late_sessions_survive_wal_and_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let id;
        {
            let f = ChittaField::open(dir.path().into()).unwrap();
            id = put(&f, "same", Some("one"));
            assert_eq!(id, put(&f, "same", Some("one")));
            assert_eq!(count(&f, id), 1);
            assert_eq!(id, put(&f, "same", Some("two")));
            assert_eq!(count(&f, id), 2);
            f.set_source_session(id, "three").unwrap();
            f.set_source_session(id, "three").unwrap();
            assert_eq!(count(&f, id), 3);
            f.sync_wal().unwrap();
        }
        {
            let f = ChittaField::open(dir.path().into()).unwrap();
            assert_eq!(count(&f, id), 3);
            f.save_full_snapshot().unwrap();
        }
        let f = ChittaField::open(dir.path().into()).unwrap();
        assert_eq!(count(&f, id), 3);
    }
    #[test]
    fn inherited_provenance_cycles_and_late_updates() {
        let dir = tempfile::tempdir().unwrap();
        let f = ChittaField::open(dir.path().into()).unwrap();
        let a = put(&f, "a", None);
        let b = put(&f, "b", None);
        let c = put(&f, "c", None);
        assert_eq!(count(&f, a), 1);
        edge(&f, a, "derived_from", b);
        edge(&f, b, "derived_from", a); // cycles cannot inflate counts
        edge(&f, b, "derived_from", "session:one");
        edge(&f, c, "confirms", a);
        f.set_source_session(c, "one").unwrap();
        assert_eq!(count(&f, a), 1);
        f.set_source_session(c, "two").unwrap();
        assert_eq!(count(&f, a), 2);
        let x = edge(&f, a, "source", "session:three");
        assert_eq!(count(&f, a), 3);
        let y = edge(&f, a, "source", "mcp_tool");
        f.triplet_supersede(x, y, now()).unwrap();
        assert_eq!(count(&f, a), 2);
    }
    #[test]
    fn provenance_and_semantic_dedup_retain_incoming_session() {
        let dir = tempfile::tempdir().unwrap();
        let f = ChittaField::open(dir.path().into()).unwrap();
        let put_signal = |realm: &str, session: &str| f.put_memory("signal", realm,
            b"[done] identical provenance", &[], 1.0, 0.001, 0, vec![], Some(session.into()), None).unwrap().0;
        let a = put_signal("a", "one");
        assert_eq!(a, put_signal("b", "two"));
        assert_eq!(count(&f, a), 2);
        let mut x = vec![0.0; crate::ops::EMBED_DIM]; x[0] = 1.0;
        let mut y = x.clone(); y[0] = 0.95; y[1] = (1.0f32 - 0.95 * 0.95).sqrt();
        let first = f.put_memory("wisdom", "test", b"first", &x, 1.0, 0.001, 0,
            vec![], Some("one".into()), None).unwrap().0;
        let merged = f.put_memory("wisdom", "test", b"near", &y, 1.0, 0.001, 0,
            vec![], Some("two".into()), None).unwrap().0;
        assert_eq!(first, merged);
        assert_eq!(count(&f, first), 2);
        let deferred = put(&f, "long deferred near duplicate observation", Some("three"));
        f.backfill_embedding(deferred, &y).unwrap();
        assert_eq!(count(&f, first), 3);
    }
}
