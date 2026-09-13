use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CorrectionState {
    #[default]
    Emitted,
    Acknowledged,
    Applied,
    Verified,
}

/// A single subject-predicate-object fact.
/// `weight` is the forward (subject→object) strength.
/// `reverse_weight` is the backward (object→subject) strength.
/// Asymmetric weights emerge from sequential observations (FEP §3.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletEntry {
    pub id: u64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub weight: f32,
    #[serde(default = "default_reverse_weight")]
    pub reverse_weight: f32,
    pub valid_from_ms: i64,
    pub valid_to_ms: i64, // 0 = still valid
    pub source_memory_id: Option<MemoryId>,
    pub source_file: Option<String>,
}

fn default_reverse_weight() -> f32 {
    -1.0 // sentinel: -1 means "use weight" (backward compat)
}

impl TripletEntry {
    /// Forward weight (subject→object).
    pub fn forward_weight(&self) -> f32 {
        self.weight
    }
    /// Reverse weight (object→subject). Falls back to weight for legacy entries.
    pub fn reverse_weight(&self) -> f32 {
        if self.reverse_weight < 0.0 { self.weight } else { self.reverse_weight }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripletStore {
    next_id: u64,
    entries: Vec<TripletEntry>,

    // Derived indexes. Serialized for backward compat with old snapshots, but cleared
    // before save (save_full_snapshot calls clear_indexes_for_save()) so new snapshots
    // store empty maps here. rebuild_indexes() must be called after deserialization.
    id_to_index: HashMap<u64, usize>,
    by_subject: HashMap<String, Vec<u64>>,
    by_object: HashMap<String, Vec<u64>>,
    by_predicate: HashMap<String, Vec<u64>>,

    // Ephemeral — absent entries are implicitly Emitted.
    #[serde(skip)]
    pub correction_states: HashMap<u64, CorrectionState>,
    // Bi-temporal supersession: old_id → (new_id, superseded_at_ingest_ms).
    // Stored in a .sup.json sidecar; not in the bincode snapshot.
    #[serde(skip)]
    supersession_map: HashMap<u64, (u64, i64)>,
    // Ingestion timestamps: id → ms when the agent first stored the fact.
    // Backfilled from valid_from_ms on load if sidecar absent.
    #[serde(skip)]
    ingestion_times: HashMap<u64, i64>,
}

impl TripletStore {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            id_to_index: HashMap::new(),
            by_subject: HashMap::new(),
            by_object: HashMap::new(),
            by_predicate: HashMap::new(),
            correction_states: HashMap::new(),
            supersession_map: HashMap::new(),
            ingestion_times: HashMap::new(),
        }
    }

    /// Clear derived indexes before serialization so new snapshots stay small.
    /// Call rebuild_indexes() after any deserialization to restore them.
    pub fn clear_indexes_for_save(&mut self) {
        self.id_to_index.clear();
        self.id_to_index.shrink_to_fit();
        self.by_subject.clear();
        self.by_subject.shrink_to_fit();
        self.by_object.clear();
        self.by_object.shrink_to_fit();
        self.by_predicate.clear();
        self.by_predicate.shrink_to_fit();
    }

    /// Rebuild all derived indexes from `entries`. Must be called after deserialization.
    pub fn rebuild_indexes(&mut self) {
        self.id_to_index = HashMap::with_capacity(self.entries.len());
        self.by_subject   = HashMap::new();
        self.by_object    = HashMap::new();
        self.by_predicate = HashMap::new();
        for (idx, e) in self.entries.iter().enumerate() {
            self.id_to_index.insert(e.id, idx);
            self.by_subject.entry(e.subject.clone()).or_default().push(e.id);
            self.by_object.entry(e.object.clone()).or_default().push(e.id);
            self.by_predicate.entry(e.predicate.clone()).or_default().push(e.id);
        }
    }

    /// Remove invalidated (valid_to_ms != 0) entries. Returns removed count.
    pub fn purge_invalidated(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.valid_to_ms == 0);
        let removed = before - self.entries.len();
        if removed > 0 {
            self.entries.shrink_to_fit();
            self.rebuild_indexes();
            self.ingestion_times.retain(|id, _| self.id_to_index.contains_key(id));
            self.ingestion_times.shrink_to_fit();
        }
        removed
    }

    /// Deduplicate entries in-place: for live (valid_to_ms==0) entries with identical
    /// (subject, predicate, object), keep only the highest-weight one. Returns removed count.
    pub fn dedup_entries(&mut self) -> usize {
        use std::collections::hash_map::Entry;
        // Owned keys to avoid borrow conflicts during retain.
        let mut live_best: HashMap<(String, String, String), usize> = HashMap::new();
        let mut to_remove = std::collections::HashSet::new();

        for (idx, e) in self.entries.iter().enumerate() {
            if e.valid_to_ms != 0 { continue; }
            let key = (e.subject.clone(), e.predicate.clone(), e.object.clone());
            match live_best.entry(key) {
                Entry::Vacant(v) => { v.insert(idx); }
                Entry::Occupied(mut o) => {
                    let best_idx = *o.get();
                    if e.weight > self.entries[best_idx].weight {
                        to_remove.insert(best_idx);
                        *o.get_mut() = idx;
                    } else {
                        to_remove.insert(idx);
                    }
                }
            }
        }

        let removed = to_remove.len();
        if removed == 0 { return 0; }

        let mut i = 0usize;
        self.entries.retain(|_| { let keep = !to_remove.contains(&i); i += 1; keep });
        self.rebuild_indexes();
        self.entries.shrink_to_fit();
        self.ingestion_times.retain(|id, _| self.id_to_index.contains_key(id));
        self.ingestion_times.shrink_to_fit();
        removed
    }

    pub fn correction_state(&self, id: u64) -> CorrectionState {
        self.correction_states.get(&id).copied().unwrap_or_default()
    }

    /// Add a triplet fact, allocating a new ID. Returns the new triplet ID.
    /// Deduplicates: if an identical (subject, predicate, object) with valid_to_ms==0
    /// already exists, bumps its weight and returns the existing ID instead.
    pub fn add(
        &mut self,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) -> u64 {
        // Check for existing live (valid_to_ms==0) entry with same (s,p,o).
        if let Some(existing_id) = self.find_exact_live(&subject, &predicate, &object) {
            if let Some(&idx) = self.id_to_index.get(&existing_id) {
                if let Some(e) = self.entries.get_mut(idx) {
                    e.weight = e.weight.max(weight);
                }
            }
            return existing_id;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.insert_with_id(id, subject, predicate, object, weight, valid_from_ms,
            source_memory_id, source_file);
        id
    }

    fn find_exact_live(&self, subject: &str, predicate: &str, object: &str) -> Option<u64> {
        let ids = self.by_subject.get(subject)?;
        for &id in ids {
            if let Some(&idx) = self.id_to_index.get(&id) {
                if let Some(e) = self.entries.get(idx) {
                    if e.valid_to_ms == 0 && e.predicate == predicate && e.object == object {
                        return Some(id);
                    }
                }
            }
        }
        None
    }

    /// Add a triplet with an explicit ID (used during log replay).
    /// Advances next_id past the given id if necessary.
    pub fn replay_add(
        &mut self,
        id: u64,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) {
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.insert_with_id(
            id,
            subject,
            predicate,
            object,
            weight,
            valid_from_ms,
            source_memory_id,
            source_file,
        );
    }

    fn insert_with_id(
        &mut self,
        id: u64,
        subject: String,
        predicate: String,
        object: String,
        weight: f32,
        valid_from_ms: i64,
        source_memory_id: Option<MemoryId>,
        source_file: Option<String>,
    ) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        self.ingestion_times.insert(id, now_ms);
        let idx = self.entries.len();
        self.id_to_index.insert(id, idx);

        let entry = TripletEntry {
            id,
            subject: subject.clone(),
            predicate: predicate.clone(),
            object: object.clone(),
            weight,
            reverse_weight: weight * 0.3, // asymmetric by default: reverse is attenuated
            valid_from_ms,
            valid_to_ms: 0,
            source_memory_id,
            source_file,
        };
        self.entries.push(entry);

        self.by_subject
            .entry(subject)
            .or_insert_with(Vec::new)
            .push(id);
        self.by_object
            .entry(object)
            .or_insert_with(Vec::new)
            .push(id);
        self.by_predicate
            .entry(predicate)
            .or_insert_with(Vec::new)
            .push(id);
    }

    /// Invalidate a triplet (set valid_to_ms = now_ms).
    pub fn invalidate(&mut self, triplet_id: u64, now_ms: i64) {
        if let Some(&idx) = self.id_to_index.get(&triplet_id) {
            if let Some(entry) = self.entries.get_mut(idx) {
                entry.valid_to_ms = now_ms;
            }
        }
    }

    fn entry_by_id(&self, id: u64) -> Option<&TripletEntry> {
        let &idx = self.id_to_index.get(&id)?;
        self.entries.get(idx)
    }

    fn is_valid(entry: &TripletEntry, at_ms: i64) -> bool {
        // valid_to_ms == 0 means still valid (no expiry set).
        // Otherwise, the triplet is valid while at_ms < valid_to_ms.
        entry.valid_to_ms == 0 || at_ms < entry.valid_to_ms
    }

    fn resolve_ids<'a>(&'a self, ids: &[u64], at_ms: i64) -> Vec<&'a TripletEntry> {
        ids.iter()
            .filter_map(|&id| self.entry_by_id(id))
            .filter(|e| Self::is_valid(e, at_ms))
            .collect()
    }

    /// Query all valid triplets with the given subject.
    pub fn query_subject(&self, subject: &str, at_ms: i64) -> Vec<&TripletEntry> {
        match self.by_subject.get(subject) {
            Some(ids) => self.resolve_ids(ids, at_ms),
            None => Vec::new(),
        }
    }

    /// Query all valid triplets with the given object.
    pub fn query_object(&self, object: &str, at_ms: i64) -> Vec<&TripletEntry> {
        match self.by_object.get(object) {
            Some(ids) => self.resolve_ids(ids, at_ms),
            None => Vec::new(),
        }
    }

    /// Query all valid triplets with the given predicate.
    pub fn query_predicate(&self, predicate: &str, at_ms: i64) -> Vec<&TripletEntry> {
        match self.by_predicate.get(predicate) {
            Some(ids) => self.resolve_ids(ids, at_ms),
            None => Vec::new(),
        }
    }

    /// Query all valid triplets where subject OR object matches the given string.
    pub fn query_entity(&self, entity: &str, at_ms: i64) -> Vec<&TripletEntry> {
        self.query_entity_limited(entity, at_ms, usize::MAX)
    }

    /// Query valid triplets touching `entity`, stopping before allocating more
    /// than `max_results` references. This is the serving-path variant used by
    /// spreading activation; slicing query_entity() after collection is too late.
    pub fn query_entity_limited(
        &self,
        entity: &str,
        at_ms: i64,
        max_results: usize,
    ) -> Vec<&TripletEntry> {
        if max_results == 0 { return Vec::new(); }
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::with_capacity(max_results.min(64));
        let mut inspected = 0usize;

        let subject_ids = self
            .by_subject
            .get(entity)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let object_ids = self
            .by_object
            .get(entity)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        for &id in subject_ids.iter().chain(object_ids.iter()) {
            if inspected >= max_results { break; }
            inspected += 1;
            if seen.insert(id) {
                if let Some(entry) = self.entry_by_id(id) {
                    if Self::is_valid(entry, at_ms) {
                        result.push(entry);
                        if result.len() >= max_results {
                            break;
                        }
                    }
                }
            }
        }

        result
    }

    /// Find all objects connected to `subject` via `predicate`.
    pub fn objects_of(&self, subject: &str, predicate: &str, at_ms: i64) -> Vec<String> {
        self.query_subject(subject, at_ms)
            .into_iter()
            .filter(|e| e.predicate == predicate)
            .map(|e| e.object.clone())
            .collect()
    }

    /// Find all subjects that have `predicate -> object`.
    pub fn subjects_of(&self, predicate: &str, object: &str, at_ms: i64) -> Vec<String> {
        self.query_object(object, at_ms)
            .into_iter()
            .filter(|e| e.predicate == predicate)
            .map(|e| e.subject.clone())
            .collect()
    }

    /// Invalidate all active triplets whose source_file matches.
    /// Returns the IDs of invalidated triplets.
    pub fn ids_by_source_memory(&self, memory_id: MemoryId) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|e| e.valid_to_ms == 0 && e.source_memory_id == Some(memory_id))
            .map(|e| e.id)
            .collect()
    }

    pub fn invalidate_by_source_file(&mut self, source_file: &str, now_ms: i64) -> Vec<u64> {
        let mut invalidated = Vec::new();
        for entry in self.entries.iter_mut() {
            if entry.valid_to_ms == 0 {
                if let Some(ref sf) = entry.source_file {
                    if sf == source_file {
                        entry.valid_to_ms = now_ms;
                        invalidated.push(entry.id);
                    }
                }
            }
        }
        invalidated
    }

    pub fn set_correction_state(&mut self, id: u64, state: CorrectionState) -> bool {
        if self.id_to_index.contains_key(&id) {
            self.correction_states.insert(id, state);
            true
        } else {
            false
        }
    }

    pub fn all_subjects(&self) -> Vec<String> {
        self.by_subject.keys().cloned().collect()
    }

    pub fn triplet_count(&self) -> usize {
        self.entries.len()
    }

    /// Copy at most `max_facts` live facts for analogy indexing without first
    /// cloning the entire subject vocabulary.
    pub fn analogy_facts(&self, at_ms: i64, max_facts: usize) -> Vec<crate::analogy::Fact> {
        self.entries
            .iter()
            .take(max_facts)
            .filter(|entry| Self::is_valid(entry, at_ms))
            .map(|entry| crate::analogy::Fact {
                subject: entry.subject.clone(),
                predicate: entry.predicate.clone(),
                object: entry.object.clone(),
                memory_id: entry.source_memory_id,
            })
            .collect()
    }

    /// BFS spreading activation from seed entities. Returns memory_id → max activation score.
    /// depth=2, decay=0.6 gives two hops with diminishing strength.
    pub fn spreading_activation(
        &self,
        seeds: &[String],
        depth: u8,
        decay: f32,
        at_ms: i64,
        max_nodes: usize,
        max_entries_per_entity: usize,
    ) -> HashMap<MemoryId, f32> {
        use std::collections::HashSet;
        if max_nodes == 0 || max_entries_per_entity == 0 {
            return HashMap::new();
        }
        let mut memory_scores: HashMap<MemoryId, f32> = HashMap::new();
        let mut visited: HashSet<String> = seeds.iter().take(max_nodes).cloned().collect();
        let mut current_layer: Vec<(String, f32)> =
            seeds.iter().take(max_nodes).map(|s| (s.clone(), 1.0f32)).collect();
        for d in 0u8..=depth {
            let mut next_layer: Vec<(String, f32)> = Vec::new();
            for (entity, activation) in &current_layer {
                let entries = self.query_entity_limited(entity, at_ms, max_entries_per_entity);
                for entry in entries {
                    if let Some(mid) = entry.source_memory_id {
                        if memory_scores.contains_key(&mid) || memory_scores.len() < max_nodes {
                            let s = memory_scores.entry(mid).or_insert(0.0);
                            if *activation > *s { *s = *activation; }
                        }
                    }
                    if d >= depth { continue; }
                    let (neighbor, w) = if entry.subject == *entity {
                        (entry.object.clone(), entry.forward_weight())
                    } else {
                        (entry.subject.clone(), entry.reverse_weight())
                    };
                    let next_act = (*activation) * w.max(0.0_f32) * decay;
                    if next_act < 0.01_f32 { continue; }
                    if visited.insert(neighbor.clone()) {
                        next_layer.push((neighbor, next_act));
                    }
                }
            }
            if next_layer.len() > max_nodes {
                next_layer.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                next_layer.truncate(max_nodes);
            }
            current_layer = next_layer;
        }
        memory_scores
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    // ── pub(crate) accessors for graph.rs ─────────────────────────────────────

    pub(crate) fn subject_ids(&self, node: &str) -> Option<&Vec<u64>> {
        self.by_subject.get(node)
    }

    pub(crate) fn object_ids(&self, node: &str) -> Option<&Vec<u64>> {
        self.by_object.get(node)
    }

    pub(crate) fn entry_by_id_crate(&self, id: u64) -> Option<&TripletEntry> {
        self.entry_by_id(id)
    }

    pub(crate) fn is_superseded_crate(&self, id: u64) -> bool {
        self.supersession_map.contains_key(&id)
    }

    /// Mark `old_id` as superseded by `new_id` at ingestion-time `at_ms`.
    /// `query_as_of` and `query_believed_at` will exclude superseded entries.
    pub fn supersede(&mut self, old_id: u64, new_id: u64, at_ms: i64) {
        self.supersession_map.insert(old_id, (new_id, at_ms));
    }

    /// Query facts about `subject` valid in the world at world-clock `world_ms`,
    /// excluding entries that have been superseded.
    pub fn query_as_of(&self, subject: &str, world_ms: i64) -> Vec<&TripletEntry> {
        match self.by_subject.get(subject) {
            None => Vec::new(),
            Some(ids) => ids.iter()
                .filter_map(|&id| self.entry_by_id(id))
                .filter(|e| {
                    let world_valid = (e.valid_from_ms == 0 || e.valid_from_ms <= world_ms)
                        && (e.valid_to_ms == 0 || world_ms < e.valid_to_ms);
                    // Supersession excludes only once it has taken effect by world_ms —
                    // an as-of query for an earlier time must still see the fact. Gating
                    // on membership alone (the old behaviour) retroactively erased facts
                    // from every past world-time. Mirrors query_believed_at's sup_at gate.
                    let not_superseded = match self.supersession_map.get(&e.id) {
                        Some(&(_, sup_at)) => sup_at > world_ms,
                        None => true,
                    };
                    world_valid && not_superseded
                })
                .collect(),
        }
    }

    /// Query what the agent believed about `subject` at ingestion-time `ingest_ms`:
    /// entries ingested on or before `ingest_ms` that had not yet been superseded.
    pub fn query_believed_at(&self, subject: &str, ingest_ms: i64) -> Vec<&TripletEntry> {
        match self.by_subject.get(subject) {
            None => Vec::new(),
            Some(ids) => ids.iter()
                .filter_map(|&id| self.entry_by_id(id))
                .filter(|e| {
                    let ingested = self.ingestion_times
                        .get(&e.id)
                        .copied()
                        .unwrap_or(e.valid_from_ms); // fallback for pre-migration entries
                    if ingested > ingest_ms { return false; }
                    // Not yet superseded as of ingest_ms?
                    match self.supersession_map.get(&e.id) {
                        Some(&(_, sup_at)) => sup_at > ingest_ms,
                        None => true,
                    }
                })
                .collect(),
        }
    }

    /// Persist supersession + ingestion data to a JSON sidecar alongside the snapshot.
    pub fn save_supersession_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        let data = serde_json::json!({
            "supersession_map": self.supersession_map.iter()
                .map(|(k, v)| (k.to_string(), [v.0, v.1 as u64]))
                .collect::<std::collections::HashMap<_,_>>(),
            "ingestion_times": self.ingestion_times.iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect::<std::collections::HashMap<_,_>>(),
        });
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, data.to_string())?;
        std::fs::rename(&tmp, path)
    }

    /// Load supersession + ingestion data from a JSON sidecar. No-op if file absent.
    pub fn load_supersession_sidecar(&mut self, path: &std::path::Path) {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return,
        };
        let v: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(_) => return,
        };
        if let Some(obj) = v["supersession_map"].as_object() {
            for (k, arr) in obj {
                if let (Ok(old_id), Some(arr)) = (k.parse::<u64>(), arr.as_array()) {
                    if arr.len() == 2 {
                        let new_id = arr[0].as_u64().unwrap_or(0);
                        let at_ms  = arr[1].as_i64().unwrap_or(0);
                        self.supersession_map.insert(old_id, (new_id, at_ms));
                    }
                }
            }
        }
        if let Some(obj) = v["ingestion_times"].as_object() {
            for (k, ts) in obj {
                if let (Ok(id), Some(ms)) = (k.parse::<u64>(), ts.as_i64()) {
                    self.ingestion_times.insert(id, ms);
                }
            }
        }
        eprintln!("[chitta-field] .sup sidecar: {} supersessions, {} ingestion times",
            self.supersession_map.len(), self.ingestion_times.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_query_subject() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "has".into(),
            "memory".into(),
            0.9,
            0,
            None,
            None,
        );
        store.add(
            "duckdb".into(),
            "is_a".into(),
            "database".into(),
            1.0,
            0,
            None,
            None,
        );

        let results = store.query_subject("chitta", 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_temporal_invalidation() {
        let mut store = TripletStore::new();
        let id = store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );

        // Valid at time 500
        let results = store.query_subject("chitta", 500);
        assert_eq!(results.len(), 1);

        // Invalidate at time 1000
        store.invalidate(id, 1000);

        // Now invalid at time 1500
        let results = store.query_subject("chitta", 1500);
        assert_eq!(results.len(), 0);

        // But was still valid at time 999
        let results = store.query_subject("chitta", 999);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_as_of_supersession_is_time_gated() {
        // Regression: query_as_of used to exclude any ever-superseded entry,
        // retroactively erasing the fact from every past world-time. The gate
        // must respect the supersession's own timestamp.
        let mut store = TripletStore::new();
        let old_id = store.add(
            "daemon".into(),
            "listens_on".into(),
            "7432".into(),
            1.0,
            100, // valid_from
            None,
            None,
        );
        let new_id = store.add(
            "daemon".into(),
            "listens_on".into(),
            "8000".into(),
            1.0,
            2000,
            None,
            None,
        );
        // Superseded at world-time 2000.
        store.supersede(old_id, new_id, 2000);

        // As of 1000 (before supersession) the old fact must still be visible.
        let before = store.query_as_of("daemon", 1000);
        assert!(
            before.iter().any(|e| e.id == old_id),
            "as-of before supersession must still see the fact"
        );
        // As of 3000 (after supersession) the old fact is gone.
        let after = store.query_as_of("daemon", 3000);
        assert!(
            !after.iter().any(|e| e.id == old_id),
            "as-of after supersession must exclude the fact"
        );
    }

    #[test]
    fn test_query_entity() {
        let mut store = TripletStore::new();
        store.add(
            "alice".into(),
            "knows".into(),
            "bob".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "charlie".into(),
            "knows".into(),
            "alice".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "alice".into(),
            "works_at".into(),
            "anthropic".into(),
            1.0,
            0,
            None,
            None,
        );

        let results = store.query_entity("alice", 0);
        assert_eq!(results.len(), 3); // alice appears as subject twice, object once
    }

    #[test]
    fn test_query_object() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "amber".into(),
            "uses".into(),
            "duckdb".into(),
            0.8,
            0,
            None,
            None,
        );

        let results = store.query_object("duckdb", 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_predicate() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "uses".into(),
            "hnsw".into(),
            0.9,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "has".into(),
            "memory".into(),
            1.0,
            0,
            None,
            None,
        );

        let results = store.query_predicate("uses", 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_objects_of() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "uses".into(),
            "hnsw".into(),
            0.9,
            0,
            None,
            None,
        );
        store.add(
            "chitta".into(),
            "has".into(),
            "memory".into(),
            1.0,
            0,
            None,
            None,
        );

        let objects = store.objects_of("chitta", "uses", 0);
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&"duckdb".to_string()));
        assert!(objects.contains(&"hnsw".to_string()));
    }

    #[test]
    fn test_subjects_of() {
        let mut store = TripletStore::new();
        store.add(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            0,
            None,
            None,
        );
        store.add(
            "amber".into(),
            "uses".into(),
            "duckdb".into(),
            0.8,
            0,
            None,
            None,
        );

        let subjects = store.subjects_of("uses", "duckdb", 0);
        assert_eq!(subjects.len(), 2);
        assert!(subjects.contains(&"chitta".to_string()));
        assert!(subjects.contains(&"amber".to_string()));
    }

    #[test]
    fn test_replay_add() {
        let mut store = TripletStore::new();
        // Simulate replay with a specific id
        store.replay_add(42, "a".into(), "b".into(), "c".into(), 1.0, 0, None, None);
        assert_eq!(store.next_id(), 43);
        let results = store.query_subject("a", 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
    }
}
