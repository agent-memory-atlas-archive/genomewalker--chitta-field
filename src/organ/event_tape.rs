use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn fnv1a_64(s: &[u8]) -> u64 {
    s.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        h.wrapping_mul(0x100000001b3) ^ b as u64
    })
}

fn canonicalize_entity(entity: &str) -> String {
    if entity.contains('/') || entity.contains('\\') {
        // File path: strip HOME prefix, keep repo-relative
        if let Ok(home) = std::env::var("HOME") {
            if let Some(rel) = entity.strip_prefix(&*home) {
                return rel.trim_start_matches('/').to_string();
            }
        }
        return entity.trim_start_matches('/').to_string();
    }
    if entity.starts_with("http://") || entity.starts_with("https://") {
        // URL: hostname only
        if let Some(host) = entity.split("//").nth(1).and_then(|s| s.split('/').next()) {
            return host.to_string();
        }
    }
    if entity.contains("::") {
        // FQN: keep lowercase
        return entity.to_lowercase();
    }
    // Freeform: first 40 alphanum chars, lowercase
    entity
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .take(40)
        .collect::<String>()
        .to_lowercase()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvent {
    pub turn_id:       u32,
    pub tool_id:       u16,
    pub entity_key:    u32,
    pub outcome_class: u8,  // 0=success 1=fail 2=error 3=partial/legacy
    pub session_id:    u64,
    pub ts_ms:         i64,
}

impl TurnEvent {
    pub fn pack(&self) -> u64 {
        ((self.tool_id as u64) << 40)
            | ((self.outcome_class as u64) << 32)
            | (self.entity_key as u64)
    }
}

/// How much of a tape a derived organ was built from. Stored inside the organs
/// cache so a later open can tell "the same tape plus a tail" from "a different
/// tape", and apply only the tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TapeCut {
    pub events: u64,
    pub tools: u64,
    pub entities: u64,
    pub key: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventTape {
    pub events:      Vec<TurnEvent>,
    tool_names:      Vec<String>,
    tool_interner:   HashMap<String, u16>,
    entity_names:    Vec<String>,
    entity_interner: HashMap<u64, u32>,
}

impl EventTape {
    /// Hash ordered events plus the dictionaries used to interpret them. HashMap
    /// serialization order must not affect identity across process restarts.
    pub(crate) fn startup_key(&self) -> [u8; 32] { self.cut().key }

    /// Identity of the whole tape, as a prefix of itself.
    pub(crate) fn cut(&self) -> TapeCut {
        self.cut_at(self.events.len(), self.tool_names.len(), self.entity_names.len())
            .expect("full-length cut is always in range")
    }

    /// Identity of the first `events` events read through the first `tools` and
    /// `entities` dictionary entries. Live appends only ever push onto all three,
    /// so a cache built at an earlier cut stays reusable; an in-place edit or a
    /// removal inside the prefix changes this key and forces a full rebuild.
    pub(crate) fn cut_at(&self, events: usize, tools: usize, entities: usize) -> Option<TapeCut> {
        let (e, t, n) = (self.events.get(..events)?, self.tool_names.get(..tools)?,
            self.entity_names.get(..entities)?);
        Some(TapeCut {
            events: events as u64, tools: tools as u64, entities: entities as u64,
            key: crate::startup_cache::digest(
                &bincode::serialize(&(1u32, e, t, n)).expect("serialize tape key")),
        })
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_tool(&mut self, tool: &str) -> u16 {
        if let Some(&id) = self.tool_interner.get(tool) {
            return id;
        }
        let id = self.tool_names.len() as u16;
        self.tool_names.push(tool.to_string());
        self.tool_interner.insert(tool.to_string(), id);
        id
    }

    pub fn intern_entity(&mut self, entity: &str) -> u32 {
        let canon = canonicalize_entity(entity);
        let hash = fnv1a_64(canon.as_bytes());
        if let Some(&id) = self.entity_interner.get(&hash) {
            return id;
        }
        let id = self.entity_names.len() as u32;
        self.entity_names.push(canon);
        self.entity_interner.insert(hash, id);
        id
    }

    /// Append one event, return its packed symbol.
    pub fn log(
        &mut self,
        tool: &str,
        entity: &str,
        outcome: u8,
        session_id: u64,
        ts_ms: i64,
    ) -> u64 {
        let tool_id    = self.intern_tool(tool);
        let entity_key = self.intern_entity(entity);
        let turn_id    = self.events.len() as u32;
        let ev = TurnEvent { turn_id, tool_id, entity_key, outcome_class: outcome, session_id, ts_ms };
        let sym = ev.pack();
        self.events.push(ev);
        sym
    }

    /// Compute the packed symbol without appending (lookup existing interners).
    /// Interns tool/entity if not present (side effect on interner, not tape).
    /// Read-only symbol probe: returns 0 if tool/entity are not yet interned.
    pub fn symbol_of_ro(&self, tool: &str, entity: &str, outcome: u8) -> u64 {
        let tool_id = match self.tool_interner.get(tool) {
            Some(&id) => id,
            None => return 0,
        };
        let canon = canonicalize_entity(entity);
        let hash  = fnv1a_64(canon.as_bytes());
        let entity_key = match self.entity_interner.get(&hash) {
            Some(&k) => k,
            None => return 0,
        };
        TurnEvent { turn_id: 0, tool_id, entity_key, outcome_class: outcome, session_id: 0, ts_ms: 0 }.pack()
    }

    pub fn symbol_of(&mut self, tool: &str, entity: &str, outcome: u8) -> u64 {
        let tool_id    = self.intern_tool(tool);
        let entity_key = self.intern_entity(entity);
        TurnEvent { turn_id: 0, tool_id, entity_key, outcome_class: outcome, session_id: 0, ts_ms: 0 }.pack()
    }

    /// Seed entity interner from existing triplet subjects/objects.
    pub fn seed_from_triplets<'a>(&mut self, subjects: impl Iterator<Item = &'a str>) {
        for s in subjects {
            self.intern_entity(s);
        }
    }

    /// Synthesize a legacy event for an existing memory (warm-start CDAWG migration).
    pub fn synthesize_legacy(&mut self, realm: &str, ts_ms: i64) {
        self.log("legacy", realm, 3, 0, ts_ms);
    }

    pub fn tool_name(&self, id: u16) -> &str {
        self.tool_names.get(id as usize).map(|s| s.as_str()).unwrap_or("unknown")
    }

    pub fn entity_name(&self, key: u32) -> &str {
        self.entity_names.get(key as usize).map(|s| s.as_str()).unwrap_or("unknown")
    }

    /// Return the last n packed symbols from the tape (oldest first).
    pub fn last_n_syms(&self, n: usize) -> Vec<u64> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..].iter().map(|e| e.pack()).collect()
    }

    /// Phase 12 — Temporal compression.
    ///
    /// Tombstones (removes) events whose PPM prediction confidence exceeds
    /// `threshold` (e.g. 0.85), meaning the CDAWG predicted them with high
    /// confidence — they carry little new information.
    ///
    /// Preserves: failure events (outcome_class != 0), the first and last
    /// event of each session, and any event the CDAWG cannot predict (novel).
    ///
    /// Returns the number of events removed.
    /// Read-only: compute the low-surprisal removal mask without mutating. This is the
    /// heavy half (O(n) cdawg.surprisal sweep, ~tens of seconds on a large tape). Run it
    /// off an immutable clone of the tape + cdawg so NO write lock is held across the
    /// sweep — holding `event_tape.write()` here was the dominant recall stall (a queued
    /// tape writer made parking_lot starve every new reader). Returns a per-index mask
    /// over the events present at call time.
    pub fn compute_low_surprisal_removals(
        &self,
        cdawg: &crate::organ::cdawg::CdawgOrgan,
        threshold: f32,
    ) -> Vec<bool> {
        const MIN_TAPE: usize = 200;
        const CTX_WINDOW: usize = 4;

        let n = self.events.len();
        let mut remove = vec![false; n];
        if n < MIN_TAPE { return remove; }

        // Collect session boundary indices (first and last per session_id).
        let mut session_bounds = std::collections::HashSet::new();
        {
            let mut last_session = u64::MAX;
            for (i, e) in self.events.iter().enumerate() {
                if e.session_id != last_session {
                    session_bounds.insert(i);
                    if i > 0 { session_bounds.insert(i - 1); }
                    last_session = e.session_id;
                }
            }
            session_bounds.insert(n - 1);
        }

        for i in CTX_WINDOW..n.saturating_sub(1) {
            // Always keep failure/error events and session boundaries.
            if self.events[i].outcome_class != 0 { continue; }
            if session_bounds.contains(&i) { continue; }

            let ctx: Vec<u64> = self.events[i.saturating_sub(CTX_WINDOW)..i]
                .iter().map(|e| e.pack()).collect();
            let sym = self.events[i].pack();

            if let Some(surprisal) = cdawg.surprisal(&ctx, sym) {
                // surprisal = -log2(p) → confidence = 2^(-surprisal)
                let confidence = 2f32.powf(-surprisal);
                if confidence > threshold {
                    remove[i] = true;
                }
            }
        }
        remove
    }

    /// Apply a removal mask from `compute_low_surprisal_removals`. Events appended after
    /// the mask was computed (index >= mask.len()) are always kept — the tape is
    /// append-only between compute and apply (consolidation is the sole remover and runs
    /// single-threaded), so lower indices stay stable. This is the brief half, the only
    /// mutation, held under a short write lock. Returns count removed.
    pub fn apply_removals(&mut self, remove: &[bool]) -> usize {
        let removed = remove.iter().filter(|&&r| r).count();
        if removed == 0 { return 0; }
        let keep_cap = self.events.len().saturating_sub(removed);
        let mut new_events = Vec::with_capacity(keep_cap);
        for (i, e) in self.events.drain(..).enumerate() {
            if i >= remove.len() || !remove[i] { new_events.push(e); }
        }
        self.events = new_events;
        removed
    }

    /// Compute + apply in one call. Holds `&mut self` across the heavy sweep, so prefer
    /// the split (compute off-clone, then apply) on the live daemon path — see
    /// consolidation_pass Phase 12. Kept for tests and manual/offline callers.
    pub fn compress_low_surprisal(
        &mut self,
        cdawg: &crate::organ::cdawg::CdawgOrgan,
        threshold: f32,
    ) -> usize {
        let remove = self.compute_low_surprisal_removals(cdawg, threshold);
        self.apply_removals(&remove)
    }

    /// Return tape statistics as a JSON string.
    pub fn stats_json(&self, tombstoned_total: u64) -> String {
        let n = self.events.len();
        let failures = self.events.iter().filter(|e| e.outcome_class == 1).count();
        let sessions: std::collections::HashSet<u64> =
            self.events.iter().map(|e| e.session_id).collect();
        format!(
            r#"{{"events":{n},"tombstoned_total":{tombstoned_total},"failure_events":{failures},"sessions":{ses},"tool_count":{tc},"entity_count":{ec}}}"#,
            ses = sessions.len(),
            tc  = self.tool_names.len(),
            ec  = self.entity_names.len(),
        )
    }
}
