//! Keyed operations.

use super::*;

impl ChittaField {
    /// Deterministic provenance lookup — the keyed lane for capability #1
    /// (anti-reprocessing). Given a content-hash and/or an input path, returns
    /// the earliest live `[done]` record that registered a matching key, or
    /// `None`. This is an O(1) exact-key lookup that BYPASSES the fuzzy
    /// BM25+HNSW retriever entirely: no embedding, no ranking, no false
    /// neighbours. `sha` is tried first (content identity), then `input` path.
    /// Both args accept a raw value or an already-prefixed key.
    pub fn provenance_lookup(&self, sha: &str, input: &str) -> Option<(MemoryId, String)> {
        let idx = self.prov_key_idx.read();
        let mut candidates: Vec<String> = Vec::new();
        let sha = sha.trim();
        if !sha.is_empty() {
            let s = sha.strip_prefix("sha:").unwrap_or(sha).to_ascii_lowercase();
            candidates.push(format!("sha:{s}"));
        }
        let input = input.trim();
        if !input.is_empty() {
            let p = input.strip_prefix("input:").unwrap_or(input);
            candidates.push(format!("input:{p}"));
        }
        for key in candidates {
            if let Some(&id) = idx.get(&key) {
                let alive = self
                    .states
                    .read()
                    .get(&id)
                    .map(|s| !s.deleted)
                    .unwrap_or(false);
                if alive {
                    let content = self
                        .payloads
                        .read()
                        .get(&id)
                        .map(|p| String::from_utf8_lossy(&p.content).into_owned())
                        .unwrap_or_default();
                    return Some((id, content));
                }
            }
        }
        None
    }

    /// Deterministic correction lookup — the keyed lane for capability #2
    /// (durable corrections with override semantics). Given free turn/context
    /// text, generates the same normalized bigram keys and probes the
    /// correction index. Returns up to `MAX_FIRED` distinct *live* corrections
    /// whose corrected-mistake trigger recurs in the text, NEWEST FIRST
    /// (latest-wins), joined by `\n---\n`. This is an O(turn_tokens) set of
    /// exact HashMap reads that BYPASSES the fuzzy BM25+HNSW retriever entirely
    /// — the correction can no longer lose on cosine similarity and be dropped
    /// from the top-k. The first element of the tuple is the newest fired
    /// correction's id (for exposed-correction / M-metric tracking).
    pub fn correction_check(&self, text: &str) -> Option<(MemoryId, String)> {
        use std::collections::HashMap;
        const MAX_FIRED: usize = 3;
        // AND firing rule: a correction fires only when >= 2 of ITS distinct
        // bigrams recur in this turn. A single shared/generic bigram is NOT
        // enough — that OR rule made ~44% of normal turns inject an irrelevant
        // stored correction (precision sweep). Requiring 2 distinct bigrams of
        // the same correction cut the false-fire rate to single digits while
        // keeping the vast majority of corrections fireable; the rest fall
        // through to the fuzzy lane (no regression).
        let mut keys = correction_bigram_keys(&norm_correction_tokens(text));
        keys.sort();
        keys.dedup();
        if keys.len() < 2 {
            return None;
        }
        // Count how many of the turn's DISTINCT bigrams map to each correction.
        // The index is multi-valued: each key resolves to every correction that
        // carries that trigger, so a record is credited for EACH of its bigrams
        // present in the turn even when a newer correction shares one. Turn keys
        // are deduped and each id appears once per key, so a per-id count of 2
        // means two distinct trigger bigrams of that correction co-occur.
        let mut per_id: HashMap<MemoryId, usize> = HashMap::new();
        {
            let idx = self.correction_key_idx.read();
            for k in &keys {
                if let Some(ids) = idx.get(k) {
                    for &id in ids {
                        *per_id.entry(id).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut ids: Vec<MemoryId> = per_id
            .into_iter()
            .filter(|&(_, n)| n >= 2)
            .map(|(id, _)| id)
            .collect();
        if ids.is_empty() {
            return None;
        }
        // Monotonic ids => descending == newest correction first.
        ids.sort_unstable_by(|a, b| b.cmp(a));
        let payloads = self.payloads.read();
        let states = self.states.read();
        let mut out: Vec<String> = Vec::new();
        let mut newest: Option<MemoryId> = None;
        for id in ids {
            if states.get(&id).map(|s| s.deleted).unwrap_or(true) {
                continue;
            }
            if let Some(p) = payloads.get(&id) {
                if newest.is_none() {
                    newest = Some(id);
                }
                out.push(String::from_utf8_lossy(&p.content).into_owned());
                if out.len() >= MAX_FIRED {
                    break;
                }
            }
        }
        newest.map(|id| (id, out.join("\n---\n")))
    }

    /// Deterministic task-state lookup — the keyed lane for capability #3 (task
    /// hand-off across discontinuous sessions). Given a task slug, returns the
    /// LATEST live `[task]` record for that id, or `None`. An O(1) exact-key
    /// HashMap read that BYPASSES the fuzzy BM25+HNSW retriever entirely: no
    /// embedding, no ranking. Unlike provenance (write-once, earliest wins), a
    /// task's status EVOLVES, so the index is latest-wins (SUPERSEDE) — the same
    /// discipline as the correction lane. `id` accepts a raw slug or an
    /// already-prefixed `task:<slug>` key.
    pub fn task_state_lookup(&self, id: &str) -> Option<(MemoryId, String)> {
        let raw = id.trim().strip_prefix("task:").unwrap_or(id.trim());
        let key = norm_task_key(raw)?;
        let mem_id = *self.task_key_idx.read().get(&key)?;
        let alive = self
            .states
            .read()
            .get(&mem_id)
            .map(|s| !s.deleted)
            .unwrap_or(false);
        if !alive {
            return None;
        }
        let content = self
            .payloads
            .read()
            .get(&mem_id)
            .map(|p| String::from_utf8_lossy(&p.content).into_owned())
            .unwrap_or_default();
        Some((mem_id, content))
    }

}
