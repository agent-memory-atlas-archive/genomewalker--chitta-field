//! organs operations.

use super::*;

impl ChittaField {

    // ── Analogy Lane (VSA over the triplet store) ──────────────────────────────

    /// Copy the live triplet lane out for `crate::analogy`. Both guards are
    /// dropped before the caller does any hypervector work — scoring under the
    /// triplet read lock starves every queued writer (parking_lot is
    /// writer-preferring; see the lock-order note in field.rs).
    ///
    /// Returns (facts, source memory → realm, live triplet count). The count is
    /// the analogy index's staleness key.
    ///
    /// Realm only, deliberately: realm is what the caller filters on, and it is
    /// a short, highly-repeated string. Payload *text* is fetched separately by
    /// `analogy_texts` for the ranked subset alone — carrying every source
    /// memory's full content through here cost ~17 MB per call at 33k memories,
    /// all of it discarded except the handful of rows that survive ranking.
    pub fn analogy_snapshot(
        &self,
        max_facts: usize,
    ) -> (Vec<crate::analogy::Fact>, std::collections::HashMap<MemoryId, String>, usize) {
        let now = now_ms();
        let (facts, count) = {
            let ts = self.triplet_store.read();
            let facts = ts.analogy_facts(now, max_facts);
            (facts, ts.triplet_count())
        };
        let mut facts = facts;
        let meta = {
            let payloads = self.payloads.read();
            // Most production triplets reach the lane through add_triplet's
            // 3-argument form, which leaves source_memory_id unset (0 -> None):
            // native_distiller.cpp and ingester.cpp instead encode the source as
            // the SUBJECT, `add_triplet(std::to_string(mem_id), "has_flag", ...)`.
            // Without recovering that, the structural index is empty on a live
            // store even though the fact table is full. Only subjects that
            // resolve to a real payload are accepted, so an entity that merely
            // looks numeric is never mistaken for a memory.
            for f in facts.iter_mut() {
                if f.memory_id.is_none() {
                    if let Ok(id) = f.subject.parse::<MemoryId>() {
                        if payloads.contains_key(&id) {
                            f.memory_id = Some(id);
                        }
                    }
                }
            }
            let wanted: HashSet<MemoryId> = facts.iter().filter_map(|f| f.memory_id).collect();
            wanted
                .iter()
                .filter_map(|id| payloads.get(id).map(|p| (*id, p.realm.clone())))
                .collect()
        };
        (facts, meta, count)
    }

    /// Payload text for the ranked subset of an analogy result, keyed by id.
    /// Companion to `analogy_snapshot`, which carries realms only.
    ///
    /// Takes `payloads` alone and holds it only for the copy — a different lock
    /// from the triplet guard `analogy_snapshot` uses, and never held with it.
    /// Call this AFTER ranking and realm-filtering, so the copy is bounded by
    /// the result limit rather than by the size of the store.
    pub fn analogy_texts(&self, ids: &[MemoryId]) -> std::collections::HashMap<MemoryId, String> {
        if ids.is_empty() {
            return std::collections::HashMap::new();
        }
        let payloads = self.payloads.read();
        ids.iter()
            .filter_map(|id| {
                payloads
                    .get(id)
                    .map(|p| (*id, String::from_utf8_lossy(&p.content).into_owned()))
            })
            .collect()
    }

    // ── Span Lane (verbatim transcript atoms) ──────────────────────────────────

    /// Query the span lane. No embedding, no GPU, no LLM. `realm=None` is
    /// unscoped; a realm that has no atoms returns empty (no cross-project leak).
    /// Returns (text, class, count, last_ms, realm, session, line, score,
    /// memory_ids) tuples. `memory_ids` is the reverse edge — beliefs referencing
    /// this atom — so a matched span can jump to the memories that mention it.
    pub fn span_query(
        &self,
        query: &str,
        realm: Option<&str>,
        k: usize,
    ) -> Vec<(String, u8, u32, i64, String, String, u32, f32, Vec<u64>)> {
        if self.ablations.disabled("span_store") { return Vec::new(); }
        self.span_store
            .write()
            .query(query, realm, k)
            .into_iter()
            .map(|h| {
                (h.text, h.class, h.count, h.last_ms, h.realm, h.session, h.line, h.score, h.memory_ids)
            })
            .collect()
    }

    /// Forward edge: the verbatim atoms a recalled memory's text references.
    /// Returns (text, class, count, realm) tuples, most-distinctive first.
    pub fn span_for_memory(&self, memory_id: u64, k: usize) -> Vec<(String, u8, u32, String)> {
        if self.ablations.disabled("span_store") { return Vec::new(); }
        self.span_store
            .read()
            .spans_for_memory(memory_id, k)
            .into_iter()
            .map(|h| (h.text, h.class, h.count, h.realm))
            .collect()
    }

    /// Link one memory's text into the span store (idempotent by content hash),
    /// persisting immediately. For write hot paths use span_link_memory instead.
    pub fn span_ingest_memory(&self, memory_id: u64, text: &str, realm: &str) -> u64 {
        if self.ablations.disabled("span_store") { return 0; }
        let mut s = self.span_store.write();
        let stats = s.ingest_memory(memory_id, text, realm);
        s.save_if_dirty();
        stats.new_spans
    }

    /// Deferred-persistence memory link for write hot paths (put_memory /
    /// content update): links in RAM only; span_flush persists periodically.
    pub fn span_link_memory(&self, memory_id: u64, text: &str, realm: &str) {
        if self.ablations.disabled("span_store") { return (); }
        self.span_store.write().ingest_memory(memory_id, text, realm);
    }

    /// Persist the span store iff it has unsaved changes. Called periodically
    /// by the queue processor and on daemon shutdown. Returns true iff saved.
    pub fn span_flush(&self) -> bool {
        if self.ablations.disabled("span_store") { return false; }
        self.span_store.write().save_if_dirty()
    }

    /// Backfill the memory→span edge over every live memory. Idempotent: a
    /// memory whose text is unchanged since last link is skipped. Returns
    /// (memories_linked, new_spans).
    pub fn span_backfill_memories(&self) -> (u64, u64) {
        if self.ablations.disabled("span_store") { return Default::default(); }
        // Snapshot (id, realm, text) under the payloads read-lock, then release it
        // before taking the span_store write-lock to avoid holding both at once.
        let snapshot: Vec<(u64, String, String)> = {
            // payloads is ordered before states — acquire in that order (lock-order audit).
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .map(|(id, p)| {
                    (*id, p.realm.clone(), String::from_utf8_lossy(&p.content).into_owned())
                })
                .collect()
        };
        let mut linked = 0u64;
        let mut new_spans = 0u64;
        {
            let mut s = self.span_store.write();
            for (id, realm, text) in &snapshot {
                let stats = s.ingest_memory(*id, text, realm);
                new_spans += stats.new_spans;
                if s.has_memory_link(*id) {
                    linked += 1;
                }
            }
            s.save();
        }
        (linked, new_spans)
    }

    /// Incrementally ingest one transcript from its watermark. Idempotent.
    /// In-RAM only (called from the queue thread on register/distill); the
    /// periodic span_flush persists spans and watermark together.
    pub fn span_ingest_transcript(&self, path: &std::path::Path) -> u64 {
        if self.ablations.disabled("span_store") { return 0; }
        self.span_store.write().ingest_transcript(path).new_spans
    }

    /// Full backfill over a projects dir. Returns (unique_total, new, redacted).
    pub fn span_backfill(&self, projects_dir: &std::path::Path) -> (usize, u64, u64) {
        if self.ablations.disabled("span_store") { return Default::default(); }
        let mut s = self.span_store.write();
        let stats = s.ingest_dir(projects_dir);
        (s.len(), stats.new_spans, stats.redacted)
    }

    /// (unique_total, on_disk_bytes, redacted_total).
    pub fn span_stats(&self) -> (usize, u64, u64) {
        if self.ablations.disabled("span_store") { return Default::default(); }
        let s = self.span_store.read();
        (s.len(), s.on_disk_bytes(), s.redacted_total())
    }

    /// Preview the top-k rules that consolidation_pass would promote (no writes).
    pub fn consolidation_preview(&self, k: usize) -> Vec<(String, u32)> {
        if self.ablations.disabled("event_tape") { return Vec::new(); }
        use crate::organ::sequitur::run_sequitur;
        let tape = self.event_tape.read();
        let rules = run_sequitur(&tape, 5);
        rules.iter().take(k).map(|r| (r.rule_key(&tape), r.support)).collect()
    }

    /// Sequitur consolidation: find frequent bigrams in EventTape, promote to triplet KG.
    /// Returns (rules_found, rules_promoted).
    pub fn consolidation_pass(&self) -> Result<(usize, usize)> {
        if self.ablations.disabled("event_tape") { return Ok((0, 0)); }
        use crate::organ::sequitur::run_sequitur;
        const MIN_SUPPORT: u32 = 5;

        // Operator kill-switch. consolidation_pass is expensive (run_sequitur + FEP
        // rebuild over the whole tape); when many consolidate_request ops pile up in
        // the queue it can monopolize the daemon. Setting CHITTA_DISABLE_CONSOLIDATION
        // makes every trigger (queue, sleep, manual RPC) a no-op.
        if std::env::var_os("CHITTA_DISABLE_CONSOLIDATION").is_some() {
            return Ok((0, 0));
        }

        // Single-flight. A pass can take a long time on a large tape, while the sleep
        // timer + queued consolidate_request ops fire far more often. Without this guard
        // the triggers STACK — each acquires the daemon's RPC mutex in turn and they
        // pile up, turning a slow pass into an unbounded recall outage. Skip any trigger
        // that arrives while a pass is in flight; the next timer tick picks up the work.
        static CONSOLIDATING: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if CONSOLIDATING.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return Ok((0, 0));
        }
        struct InFlightGuard;
        impl Drop for InFlightGuard {
            fn drop(&mut self) {
                CONSOLIDATING.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _in_flight = InFlightGuard;

        let (rule_data, rules_for_ledger): (Vec<(String, String, String, String, String, String)>, Vec<crate::organ::sequitur::SequiturRule>) = {
            // Clone the tape under a brief read, then release the lock before the
            // expensive run_sequitur (~30s on a large tape). Holding event_tape.read()
            // across that span starves recall: a queued tape writer (put_memory/log_event)
            // blocks, and parking_lot task-fairness then blocks every subsequent reader.
            let tape = self.event_tape.read().clone();
            let rules = run_sequitur(&tape, MIN_SUPPORT);
            let data = rules.iter().map(|r| (
                r.rule_key(&tape),
                r.seq_repr(&tape),
                r.avg_outcome_label().to_string(),
                r.support.to_string(),
                format!("{}:{}", r.tape_start, r.tape_end),
                r.verbalize(&tape),
            )).collect();
            (data, rules)
        };

        // Seed refutation ledger with current rule set
        self.refutation_ledger.write().seed_from_rules(&rules_for_ledger);
        // Rebuild hypothesis market from updated ledger (Phase 10)
        self.hypothesis_market.write().update_from_ledger(&self.refutation_ledger.read());

        let total = rule_data.len();
        let mut promoted = 0usize;
        let now = now_ms();
        for (key, seq, outcome, support, range, verbalized) in rule_data {
            // Skip if this rule key is already in the KG (dedup across runs).
            if !self.triplet_store.read().query_subject(&key, now).is_empty() { continue; }
            if self.add_triplet(key.clone(), "compresses".into(),    seq,        1.0, None, None).is_err() { continue; }
            let _ = self.add_triplet(key.clone(), "avg_outcome".into(),  outcome,    1.0, None, None);
            let _ = self.add_triplet(key.clone(), "support".into(),      support,    1.0, None, None);
            let _ = self.add_triplet(key.clone(), "tape_range".into(),   range,      1.0, None, None);
            let _ = self.add_triplet(key,         "verbalized_as".into(), verbalized, 1.0, None, None);
            promoted += 1;
        }
        // Phase 12: compress low-surprisal events from the tape — WITHOUT holding any
        // lock across the heavy O(n) cdawg.surprisal sweep, which previously stalled
        // recall for 13-35s (the tape write blocked log_event, and parking_lot's
        // writer-fairness then starved every new tape reader). Snapshot tape+cdawg under
        // one brief read (tape→cdawg order, matching put_memory — the clones are cheap:
        // TurnEvent is 32 bytes), compute the removal mask off-lock, then apply it under a
        // short write. Events appended during the sweep (indices past the mask) are kept.
        {
            let (tape_snapshot, cdawg_snapshot) = {
                let tape = self.event_tape.read();
                let cdawg = self.cdawg.read();
                (tape.clone(), cdawg.clone())
            };
            let remove = tape_snapshot.compute_low_surprisal_removals(&cdawg_snapshot, 0.85);
            drop(tape_snapshot);
            drop(cdawg_snapshot);
            let removed = if remove.iter().any(|&r| r) {
                self.event_tape.write().apply_removals(&remove)
            } else {
                0
            };
            if removed > 0 {
                self.tape_tombstoned.fetch_add(removed as u64, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[cec] temporal compression: tombstoned {removed} low-surprisal events");
            }
        }

        // Phase 15: rebuild FEP model from (compressed) tape. Clone under a brief read so
        // the O(events) rebuild runs off-lock (same starvation reasoning as run_sequitur).
        {
            let tape = self.event_tape.read().clone();
            self.fep_prior.write().rebuild_from_tape(&tape);
            eprintln!("[cec] fep rebuilt: {} states modeled, drift={:.3}, shock={:.3}",
                self.fep_prior.read().state_emission_len(),
                self.fep_prior.read().ewma_drift,
                self.fep_prior.read().ewma_shock);
        }

        // Phase 11: take a Turīya health sample after each consolidation_pass.
        let diagnosis = {
            let ts = now_ms();
            let tape    = self.event_tape.read();
            let cdawg   = self.cdawg.read();
            let ledger  = self.refutation_ledger.read();
            let market  = self.hypothesis_market.read();
            let fep     = self.fep_prior.read();
            self.turiya_monitor.write().sample(ts, &cdawg, &tape, &ledger, &market, &fep);
            self.turiya_monitor.read().latest()
                .map(|s| s.diagnose())
                .unwrap_or(crate::organ::turiya_monitor::Diagnosis::Healthy)
        };

        // Phase 14: auto-queue experiments when Turīya detects high uncertainty.
        if diagnosis == crate::organ::turiya_monitor::Diagnosis::HighUncertainty {
            let result = self.queue_experiments(5);
            eprintln!("[cec] turiya→HighUncertainty: auto-queued experiments: {result}");
        }

        // Phase 17: reconcile pass — detect and log illegal edges + contradictions.
        {
            let reconcile_json = self.reconcile_pass();
            eprintln!("[cec:p17] reconcile: {reconcile_json}");
        }

        // Phase 16: write falsifiability metrics to triplet KG.
        {
            let now = now_ms();
            let tape_len = self.event_tape.read().events.len();
            let contradiction_count = self.triplet_store.read()
                .query_predicate("illegal_edge_blocked", now).len();
            let _ = self.add_triplet(
                "cec:contradiction_yield".into(), "total_blocked".into(),
                contradiction_count.to_string(), 1.0, None, None,
            );
            let _ = self.add_triplet(
                "cec:router_ready".into(), "tape_events".into(),
                tape_len.to_string(), 1.0, None, None,
            );
            eprintln!("[cec:p16] metrics: contradiction_yield={contradiction_count} tape={tape_len}");
        }

        Ok((total, promoted))
    }

    /// Return the current Turīya health vector as a JSON string.
    pub fn turiya_status(&self) -> String {
        if self.ablations.disabled("turiya_monitor") { return String::from("[]"); }
        self.turiya_monitor.read().status_json()
    }

    /// Return EventTape statistics including compression totals.
    pub fn fep_status(&self) -> String {
        if self.ablations.disabled("fep_prior") { return String::from("[]"); }
        self.fep_prior.read().status_json()
    }

    pub fn tape_stats(&self) -> String {
        if self.ablations.disabled("event_tape") { return String::from("[]"); }
        let tombstoned = self.tape_tombstoned.load(std::sync::atomic::Ordering::Relaxed);
        self.event_tape.read().stats_json(tombstoned)
    }

    /// Phase 14 — Queue deliberate micro-experiments for uncertain Sequitur rules.
    ///
    /// Reads HypothesisMarket::top_probes(k) and files an OpenTask intervention
    /// for each rule whose probe_value > 0.4 AND refutation_ratio < 0.3 (safety gate).
    /// Returns JSON: {"queued": N, "skipped_refuted": M, "skipped_certain": L}
    pub fn queue_experiments(&self, k: usize) -> String {
        if self.ablations.disabled("cec_policy_store") || self.ablations.disabled("hypothesis_market") { return String::from("{}"); }
        use crate::organ::intervention_store::InterventionKind;
        let probes = {
            let market = self.hypothesis_market.read();
            market.top_probes(k.max(1)).to_vec()
        };
        let ledger = self.refutation_ledger.read();
        let mut store = self.cec_policy_store.write();
        let ts = now_ms();
        let mut queued = 0usize;
        let mut skipped_refuted = 0usize;
        let mut skipped_certain = 0usize;
        for h in &probes {
            if h.probe_value <= 0.4 {
                skipped_certain += 1;
                continue;
            }
            // Adversarial gate: don't experiment on rules being actively refuted.
            let refute_ratio = ledger.refute_ratio_for_rule(h.rule_id);
            if refute_ratio >= 0.3 {
                skipped_refuted += 1;
                continue;
            }
            let title = format!("probe_rule_{}", h.rule_id);
            let desc = format!(
                "CEC Phase 14 experiment: rule {} has p_hat={:.3} (probe_value={:.3}). \
                 Deliberately execute its antecedent and observe whether the consequent follows.",
                h.rule_id, h.p_hat, h.probe_value
            );
            store.propose(h.rule_id, InterventionKind::OpenTask { title, description: desc }, ts);
            queued += 1;
        }
        format!(
            r#"{{"queued":{queued},"skipped_refuted":{skipped_refuted},"skipped_certain":{skipped_certain}}}"#
        )
    }

    /// Phase 13 — Return top-k verbalized Sequitur rules ranked by support.
    pub fn verbalize_rules(&self, k: usize) -> String {
        if self.ablations.disabled("event_tape") { return String::from("[]"); }
        use crate::organ::sequitur::run_sequitur;
        const MIN_SUPPORT: u32 = 3;
        let tape = self.event_tape.read();
        let mut rules = run_sequitur(&tape, MIN_SUPPORT);
        rules.sort_by(|a, b| b.support.cmp(&a.support));
        rules.truncate(k.max(1));
        let items: Vec<String> = rules.iter().map(|r| {
            let v = r.verbalize(&tape);
            let key = r.rule_key(&tape);
            format!(
                r#"{{"rule_id":{},"support":{},"avg_outcome":"{}","key":"{}","text":"{}"}}"#,
                r.id, r.support, r.avg_outcome_label(), key,
                v.replace('"', "\\\"")
            )
        }).collect();
        format!(r#"{{"total":{},"rules":[{}]}}"#, items.len(), items.join(","))
    }

    /// Phase 17: Promote a candidate memory to established band once a witness arrives.
    /// Returns JSON status.
    pub fn witness_memory(&self, memory_id: MemoryId, witness_kind: &str) -> String {
        let wk = WitnessKind::from_str(witness_kind);
        let mut payloads = self.payloads.write();
        let Some(payload) = payloads.get_mut(&memory_id) else {
            return format!(r#"{{"ok":false,"error":"not_found","memory_id":{memory_id}}}"#);
        };
        if !payload.candidate {
            return format!(r#"{{"ok":true,"status":"already_established","memory_id":{memory_id}}}"#);
        }
        if wk.is_none() {
            return format!(r#"{{"ok":false,"error":"unknown_witness_kind","memory_id":{memory_id}}}"#);
        }
        payload.candidate = false;
        eprintln!("[cec:p17] memory {memory_id} promoted from candidate via witness={witness_kind}");
        let _ = self.add_triplet(
            format!("cec:witness:{memory_id}"),
            "promoted_by".into(),
            witness_kind.to_string(),
            1.0, None, None,
        );
        format!(r#"{{"ok":true,"status":"promoted","memory_id":{memory_id},"witness_kind":"{witness_kind}"}}"#)
    }

    /// Phase 17: Run the R0 reconcile pass — scan assoc_edges for legality violations
    /// and detect content contradictions. Returns JSON summary.
    pub fn reconcile_pass(&self) -> String {
        let payloads    = self.payloads.read();
        let assoc_edges = self.assoc_edges.read();
        let rec = Reconciler::new();

        let result = rec.reconcile_all(&payloads, &assoc_edges);
        let contras = rec.detect_contradictions(&payloads);

        let now = now_ms();
        // Log illegal edges to triplet KG
        for (src, dst, reason) in &result.illegal_edges {
            let _ = self.add_triplet(
                format!("cec:reconcile:{src}→{dst}"),
                "illegal_reason".into(),
                reason.clone(),
                1.0, None, None,
            );
        }
        // Log contradictions
        for (a, b, score) in &contras {
            let _ = self.add_triplet(
                format!("contradiction:{a}-{b}"),
                "conflict_score".into(),
                format!("{score:.2}"),
                1.0, None, None,
            );
        }
        let _ = now; // suppress unused warning

        format!(
            r#"{{"illegal_edges":{},"contradictions":{},"unresolved":{},"ok":true}}"#,
            result.illegal_edges.len(),
            contras.len(),
            result.unresolved.len(),
        )
    }

    /// Phase 17: Produce a harvest scope document from current Turīya anomalies
    /// and router miss patterns. Used by `scripts/harvest_ow.py` to target extraction.
    pub fn harvest_scope(&self) -> String {
        let turiya_json = self.turiya_status();
        let turiya: serde_json::Value = serde_json::from_str(&turiya_json)
            .unwrap_or(serde_json::Value::Null);
        let diagnosis = turiya.get("diagnosis")
            .and_then(|v| v.as_str()).unwrap_or("unknown");

        // Top router misses from contradiction_yield triplets
        let now = now_ms();
        let ts_guard = self.triplet_store.read();
        let miss_entries = ts_guard.query_predicate("illegal_edge_blocked", now);
        let miss_count = miss_entries.len();

        let sample_misses: Vec<serde_json::Value> = miss_entries.iter().take(5).map(|e| {
            serde_json::json!({
                "pattern": e.object,
                "miss_count": 1,
                "suggested_corpus": if e.object.contains("code") {
                    "code_editing_failures"
                } else {
                    "session_continuity"
                }
            })
        }).collect();

        let scope = serde_json::json!({
            "generated_at_ms": now,
            "turiya_diagnosis": diagnosis,
            "top_router_misses": sample_misses,
            "total_router_misses": miss_count,
            "harvest_budget_items": 500_usize.min(miss_count * 10 + 50),
        });

        scope.to_string()
    }

    pub fn seed_hdc_geometry(&self, json_path: &str) -> String {
        if self.ablations.disabled("hdc_idx") { return String::from("[]"); }
        let result = self.hdc_idx.write().seed_from_geometry(json_path);
        match result {
            Ok(n) => {
                let codebook_len = self.hdc_idx.read().codebook_len();
                serde_json::json!({
                    "ok": true,
                    "seeded_tokens": n,
                    "codebook_len": codebook_len,
                    "source": json_path,
                }).to_string()
            }
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }).to_string(),
        }
    }

    /// Return top-k rules by refute_ratio as a plain-text summary.
    pub fn refutation_stats(&self, k: usize) -> String {
        if self.ablations.disabled("refutation_ledger") { return String::from("[]"); }
        let tape   = self.event_tape.read();
        let ledger = self.refutation_ledger.read();
        ledger.stats_json(&tape, k)
    }

    /// Promote eligible shadow policies, demote drifted ones, return JSON summary.
    pub fn executor_flush(&self) -> String {
        if self.ablations.disabled("cec_policy_store") { return String::from("{}"); }
        let ledger = self.refutation_ledger.read();
        let mut store = self.cec_policy_store.write();
        let promoted = store.promote_eligible();
        let demoted  = store.auto_demote_drifted(&ledger);
        let stats    = store.stats_json();
        format!(
            "{{\"promoted\":{:?},\"demoted\":{:?},\"store\":{}}}",
            promoted, demoted, stats
        )
    }

    /// List intervention policies as JSON.
    pub fn list_policies(&self, active_only: bool) -> String {
        if self.ablations.disabled("cec_policy_store") { return String::from("[]"); }
        self.cec_policy_store.read().list_json(active_only)
    }

    /// Record an explicit decision point: what was chosen, what was rejected and why.
    /// `rejected` is a slice of (packed_symbol, RejectionReason as u8).
    pub fn log_decision(
        &self,
        chosen_tool: &str, chosen_entity: &str, chosen_outcome: u8,
        rejected: Vec<(u64, u8)>,
        confidence_delta: f32,
        ts_ms: i64,
    ) {
        let chosen_sym = self.event_tape.read().symbol_of_ro(chosen_tool, chosen_entity, chosen_outcome);
        let turn_id = self.event_tape.read().events.len() as u32;
        self.decision_tape.write().log(turn_id, chosen_sym, rejected, confidence_delta, ts_ms);
    }

    /// Log an event with cost metadata for regret-shaped Q-value update (Phase 10 Part B).
    pub fn log_event_ex(
        &self,
        tool: &str, entity: &str, outcome: u8,
        session_id: u64, ts_ms: i64,
        token_cost: u32, latency_ms: u32, retry_count: u8,
    ) {
        if self.ablations.disabled("event_tape") { return (); }
        const ALPHA_COST:    f32 = 0.001;
        const BETA_LATENCY:  f32 = 0.00001;
        const GAMMA_RETRIES: f32 = 0.1;
        let (sym, turn, last_n) = {
            let mut tape = self.event_tape.write();
            let s = tape.log(tool, entity, outcome, session_id, ts_ms);
            let t = tape.events.len() as u32 - 1;
            let n = tape.last_n_syms(16);
            (s, t, n)
        };
        let mut cdawg = self.cdawg.write();
        cdawg.extend(sym, turn);
        if tool != "legacy" && tool != "remember" {
            let base = if outcome == 0 { 1.0_f32 } else { -1.0_f32 };
            let utility = base
                - ALPHA_COST    * token_cost  as f32
                - BETA_LATENCY  * latency_ms  as f32
                - GAMMA_RETRIES * retry_count as f32;
            let delta = if outcome == 0 { 0.1_f32 } else { -0.2_f32 };
            cdawg.push_td_credit(&last_n, delta, 0.9);
            cdawg.update_q(sym, utility, 0.05, 0.95);
        }
        drop(cdawg);
        self.episode_hdc.write().log_episode(tool, entity, outcome);
    }

    /// Top-k rules by expected information gain (Wilson probe_value). Highest = most uncertain.
    pub fn hypothesis_probes(&self, k: usize) -> String {
        if self.ablations.disabled("hypothesis_market") { return String::from("[]"); }
        self.hypothesis_market.read().stats_json(k)
    }

    // ── Layer 1: Executable Constraints ────────────────────────────────────

    pub fn assert_constraint(
        &self,
        subject: String,
        predicate: String,
        object: String,
        confidence: f32,
        scope: String,
        branch_id: u64,
        provenance: crate::organ::constraint::Provenance,
        source_memory_id: Option<u64>,
    ) -> Result<crate::organ::constraint::AssertResult> {
        let now = now_ms();
        let result = self.constraint_store.write().assert_fact(
            subject.clone(), predicate.clone(), object.clone(),
            confidence, scope.clone(), branch_id, provenance.clone(),
            now, source_memory_id,
        );
        let op = Op::AssertConstraint(crate::ops::AssertConstraintOp {
            fact_id: result.fact_id,
            subject, predicate, object, confidence, scope, branch_id,
            provenance_source: provenance.source,
            provenance_session: provenance.session_id,
            provenance_basis: provenance.confidence_basis,
            valid_from_ms: now,
            source_memory_id,
        });
        self.log.write().append(&op)?;
        Ok(result)
    }

    pub fn retract_constraint(&self, fact_id: u64) -> Result<bool> {
        if self.ablations.disabled("constraint_store") { return Ok(false); }
        let now = now_ms();
        let ok = self.constraint_store.write().retract(fact_id, now);
        if ok {
            let op = Op::RetractConstraint(crate::ops::RetractConstraintOp {
                fact_id, retracted_at_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn query_constraints(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
        scope: Option<&str>,
    ) -> Vec<crate::organ::constraint::Constraint> {
        if self.ablations.disabled("constraint_store") { return Vec::new(); }
        self.constraint_store.read().query_unify(subject, predicate, object, scope)
            .into_iter().cloned().collect()
    }

    pub fn query_constraint_chain(
        &self, subject: &str, predicates: &[&str], max_depth: usize,
    ) -> Vec<Vec<crate::organ::constraint::Constraint>> {
        if self.ablations.disabled("constraint_store") { return Vec::new(); }
        self.constraint_store.read().query_chain(subject, predicates, max_depth)
            .into_iter().map(|v| v.into_iter().cloned().collect()).collect()
    }

    pub fn explain_constraint(&self, fact_id: u64) -> Option<crate::organ::constraint::Explanation> {
        if self.ablations.disabled("constraint_store") { return None; }
        self.constraint_store.read().explain(fact_id)
    }

    pub fn create_constraint_branch(&self, parent_id: u64, scope: String) -> Result<u64> {
        if self.ablations.disabled("constraint_store") { return Ok(0); }
        let now = now_ms();
        let branch_id = self.constraint_store.write().create_branch(parent_id, scope.clone(), now);
        let op = Op::CreateBranch(crate::ops::CreateBranchOp {
            branch_id, parent_id, scope, created_ms: now,
        });
        self.log.write().append(&op)?;
        Ok(branch_id)
    }

    pub fn resolve_constraint_branch(&self, winner_id: u64, loser_id: u64) -> Result<bool> {
        if self.ablations.disabled("constraint_store") { return Ok(false); }
        let now = now_ms();
        let ok = self.constraint_store.write().resolve_branch(winner_id, loser_id, now);
        if ok {
            let op = Op::ResolveBranch(crate::ops::ResolveBranchOp {
                winner_id, loser_id, resolved_at_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn constraint_stats(&self) -> (usize, usize) {
        if self.ablations.disabled("constraint_store") { return Default::default(); }
        let store = self.constraint_store.read();
        (store.count(), store.branch_count())
    }

    // ── Layer 2: Trigger Tissue ─────────────────────────────────────────

    pub fn add_trigger(
        &self,
        name: String,
        condition: crate::organ::trigger::TriggerCondition,
        action: crate::organ::trigger::TriggerAction,
        deadline_ms: i64,
        tension_threshold: f32,
        gain: f32,
        realm: String,
        source_session: Option<String>,
    ) -> Result<u64> {
        if self.ablations.disabled("trigger_store") { return Ok(0); }
        let now = now_ms();
        let id = self.trigger_store.write().add_trigger(
            name, condition.clone(), action.clone(),
            deadline_ms, tension_threshold, gain, realm.clone(), source_session.clone(), now,
        );
        let trigger = self.trigger_store.read().get(id).cloned();
        if let Some(t) = trigger {
            let json = serde_json::to_vec(&t).unwrap_or_default();
            let op = Op::AddTrigger(crate::ops::AddTriggerOp { trigger_json: json });
            self.log.write().append(&op)?;
        }
        Ok(id)
    }

    pub fn fire_trigger(&self, trigger_id: u64) -> Result<Option<crate::organ::trigger::FireResult>> {
        if self.ablations.disabled("trigger_store") { return Ok(None); }
        let now = now_ms();
        let result = self.trigger_store.write().fire(trigger_id, now);
        if result.is_some() {
            let op = Op::FireTrigger(crate::ops::FireTriggerOp {
                trigger_id, fired_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(result)
    }

    pub fn dismiss_trigger(&self, trigger_id: u64) -> Result<bool> {
        if self.ablations.disabled("trigger_store") { return Ok(false); }
        let now = now_ms();
        let ok = self.trigger_store.write().dismiss(trigger_id, now);
        if ok {
            let op = Op::UpdateTrigger(crate::ops::UpdateTriggerOp {
                trigger_id, status: 2, fired_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn list_triggers(&self) -> Vec<crate::organ::trigger::TriggerAutomaton> {
        if self.ablations.disabled("trigger_store") { return Vec::new(); }
        self.trigger_store.read().list_all().to_vec()
    }

    pub fn evaluate_triggers(&self) -> Result<Vec<crate::organ::trigger::FireResult>> {
        if self.ablations.disabled("trigger_store") { return Ok(Vec::new()); }
        let now = now_ms();
        let ready_ids = self.trigger_store.read().evaluate_time_triggers(now);
        let mut results = Vec::new();
        for id in ready_ids {
            if let Some(result) = self.fire_trigger(id)? {
                results.push(result);
            }
        }
        Ok(results)
    }

    pub fn trigger_stats(&self) -> usize {
        if self.ablations.disabled("trigger_store") { return 0; }
        self.trigger_store.read().count_armed()
    }

    // ── Layer 3: Predictive Memory ──────────────────────────────────────

    pub fn predict_needed(&self, k: usize) -> Vec<(MemoryId, f32)> {
        if self.ablations.disabled("predictor") { return Vec::new(); }
        self.predictor.read().predict(k)
    }

    pub fn retrain_predictor(&self) {
        if self.ablations.disabled("predictor") { return (); }
        let now = now_ms();
        self.predictor.write().retrain(now);
    }

    pub fn predictor_stats(&self) -> (u64, usize, usize) {
        if self.ablations.disabled("predictor") { return Default::default(); }
        let p = self.predictor.read();
        (p.total_transitions(), p.transition_count(), p.recent_access_len())
    }

    // ── Layer 4: Surprise Memory ──────────────────────────────────────

    pub fn record_surprise(
        &self,
        context_sketch: String,
        action: String,
        expected: Option<String>,
        actual: String,
        surprise_magnitude: f32,
        domain: String,
        realm: String,
        session_id: Option<String>,
        source_memory_id: Option<u64>,
    ) -> Result<u64> {
        if self.ablations.disabled("surprise_store") { return Ok(0); }
        let now = now_ms();
        let event_id = {
            let mut store = self.surprise_store.write();
            store.record(
                context_sketch.clone(), action.clone(), expected.clone(),
                actual.clone(), surprise_magnitude, domain.clone(),
                realm.clone(), session_id.clone(), source_memory_id, now,
            )
        };
        let domain_ref = domain.clone();
        let action_ref = action.clone();
        let op = Op::RecordSurprise(crate::ops::RecordSurpriseOp {
            event_id,
            context_sketch,
            action,
            expected,
            actual,
            surprise_magnitude,
            domain,
            timestamp_ms: now,
            realm,
            session_id,
            source_memory_id,
        });
        self.log.write().append(&op)?;

        // ── Move 1: auto-strengthen/weaken via surprise credit ────────
        if let Some(source_id) = source_memory_id.filter(|_| !self.ablations.disabled("surprise_learning")) {
            // source_memory_id was the "expected" memory → weaken direction
            let credit_result = self.surprise_learning.write().update_credit(
                source_id, event_id, surprise_magnitude, -1, now,
            );
            if let Some(cr) = credit_result {
                // Apply strength delta via existing UpdateState
                let delta_op = crate::ops::StateDeltaOp {
                    memory_id: cr.memory_id,
                    strength_delta: Some(cr.strength_delta),
                    confidence_delta: None,
                    decay_rate: None,
                    touch: false,
                    pin: None,
                    op_ts_ms: now,
                    status: None,
                    epistemic_status: None,
                    staged: None,
                    invalidated_by: None,
                };
                if let Some(state) = self.states.write().get_mut(&cr.memory_id) {
                    state.apply_delta(&delta_op, now);
                }
                self.log.write().append(&Op::UpdateState(delta_op))?;
                // WAL the credit state
                let sl = self.surprise_learning.read();
                if let Some(st) = sl.get_state(cr.memory_id) {
                    self.log.write().append(&Op::UpdateSurpriseCredit(
                        crate::ops::UpdateSurpriseCreditOp {
                            memory_id: st.memory_id,
                            credit: st.credit,
                            last_dir: st.last_dir,
                            same_dir_streak: st.same_dir_streak,
                            last_surprise_id: st.last_surprise_id,
                            updated_ms: st.updated_ms,
                        },
                    ))?;
                }
            }
        }

        // ── Move 2: auto-feed integration kernel ──────────────────────
        if !self.ablations.disabled("surprise_learning") {
            let should_neg = self.surprise_learning.read()
                .should_send_negative_feedback(&domain_ref, "semantic", surprise_magnitude);
            if should_neg {
                self.surprise_learning.write().record_failure(&domain_ref, "semantic", event_id);
                let _ = self.record_feedback(&domain_ref, "semantic", false);
            }
            let should_pos = self.surprise_learning.read()
                .should_send_positive_feedback(surprise_magnitude);
            if should_pos {
                let _ = self.record_feedback(&domain_ref, "keyword", true);
            }
        }

        // ── Layer 9: adjudicate wisdom lineages by envelope overlap ───
        {
            use crate::organ::wisdom_lineage::CONTRADICTION_DELTA_HIT;
            let matching = self.wisdom_lineage_store.read()
                .find_by_envelope(&domain_ref, &action_ref);
            for lineage_id in matching {
                let new_state = self.wisdom_lineage_store.write().adjudicate(
                    lineage_id, 0.0,
                    surprise_magnitude * CONTRADICTION_DELTA_HIT,
                    0.0, now,
                );
                if let Some(l) = self.wisdom_lineage_store.read().get(lineage_id) {
                    self.log.write().append(&Op::AdjudicateLineage(
                        crate::ops::AdjudicateLineageOp {
                            lineage_id,
                            support_mass: l.support_mass,
                            contradiction_mass: l.contradiction_mass,
                            staleness_mass: l.staleness_mass,
                            last_supported_ms: l.last_supported_ms,
                            last_challenged_ms: l.last_challenged_ms,
                            adjudicated_ms: now,
                        },
                    ))?;
                    if let Some(ns) = new_state {
                        self.log.write().append(&Op::TransitionLineage(
                            crate::ops::TransitionLineageOp {
                                lineage_id,
                                old_state: l.state.as_u8(),
                                new_state: ns.as_u8(),
                                reason: "surprise_adjudication".to_string(),
                                rederive_task_id: None,
                                transitioned_ms: now,
                            },
                        ))?;
                    }
                }
                // Record surprise as challenger evidence
                let _ = self.wisdom_lineage_store.write().record_challenger(
                    lineage_id,
                    crate::organ::wisdom_lineage::ChallengerEvidence {
                        intervention_id: None,
                        surprise_id: Some(event_id),
                        outcome_summary: format!("surprise magnitude {:.2}", surprise_magnitude),
                        attached_ms: now,
                    },
                    now,
                );
            }
        }

        Ok(event_id)
    }

    pub fn query_surprises(
        &self,
        domain: Option<&str>,
        realm: Option<&str>,
        min_magnitude: Option<f32>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Vec<crate::organ::surprise::SurpriseEvent> {
        if self.ablations.disabled("surprise_store") { return Vec::new(); }
        self.surprise_store
            .read()
            .query(domain, realm, min_magnitude, since_ms, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn get_blind_spots(
        &self,
        realm: Option<&str>,
        limit: usize,
    ) -> Vec<crate::organ::surprise::BlindSpot> {
        if self.ablations.disabled("surprise_store") { return Vec::new(); }
        self.surprise_store.read().get_blind_spots(realm, limit)
    }

    pub fn surprise_stats(&self) -> crate::organ::surprise::SurpriseStats {
        self.surprise_store.read().stats()
    }

    // ── Layer 5: Epistemic Debt ───────────────────────────────────────

    pub fn register_debt(
        &self,
        pattern: String,
        competing_hypotheses: Vec<String>,
        discriminating_test: Option<String>,
        fragility_score: f32,
        domain: String,
        realm: String,
        source_session: Option<String>,
    ) -> Result<u64> {
        if self.ablations.disabled("epistemic_debt_store") { return Ok(0); }
        let now = now_ms();
        let debt_id = {
            let mut store = self.epistemic_debt_store.write();
            store.register(
                pattern.clone(), competing_hypotheses.clone(),
                discriminating_test.clone(), fragility_score,
                domain.clone(), realm.clone(), source_session.clone(), now,
            )
        };
        let op = Op::RegisterDebt(crate::ops::RegisterDebtOp {
            debt_id,
            pattern,
            competing_hypotheses,
            discriminating_test,
            fragility_score,
            domain,
            created_ms: now,
            realm,
            source_session,
        });
        self.log.write().append(&op)?;
        Ok(debt_id)
    }

    pub fn resolve_debt(&self, debt_id: u64, resolution: String) -> Result<bool> {
        if self.ablations.disabled("epistemic_debt_store") { return Ok(false); }
        let now = now_ms();
        let ok = self.epistemic_debt_store.write().resolve(debt_id, resolution.clone(), now);
        if ok {
            let op = Op::UpdateDebt(crate::ops::UpdateDebtOp {
                debt_id,
                status: 1,
                resolved_ms: now,
                resolution: Some(resolution),
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn defer_debt(&self, debt_id: u64) -> Result<bool> {
        if self.ablations.disabled("epistemic_debt_store") { return Ok(false); }
        let ok = self.epistemic_debt_store.write().defer(debt_id);
        if ok {
            let op = Op::UpdateDebt(crate::ops::UpdateDebtOp {
                debt_id,
                status: 2,
                resolved_ms: 0,
                resolution: None,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn query_debts(
        &self,
        status: Option<crate::organ::epistemic_debt::DebtStatus>,
        domain: Option<&str>,
        realm: Option<&str>,
        min_fragility: Option<f32>,
        limit: usize,
    ) -> Vec<crate::organ::epistemic_debt::EpistemicDebt> {
        if self.ablations.disabled("epistemic_debt_store") { return Vec::new(); }
        self.epistemic_debt_store
            .read()
            .query(status, domain, realm, min_fragility, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn get_fragile_decisions(
        &self,
        threshold: f32,
        limit: usize,
    ) -> Vec<crate::organ::epistemic_debt::EpistemicDebt> {
        if self.ablations.disabled("epistemic_debt_store") { return Vec::new(); }
        self.epistemic_debt_store
            .read()
            .get_fragile_decisions(threshold, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn debt_stats(&self) -> crate::organ::epistemic_debt::DebtStats {
        self.epistemic_debt_store.read().stats()
    }

    // ── Layer 6: Integration Kernel ───────────────────────────────────

    pub fn record_feedback(
        &self,
        query_domain: &str,
        source: &str,
        was_useful: bool,
    ) -> Result<crate::organ::integration::SourceWeight> {
        if self.ablations.disabled("integration_kernel") {
            return Ok(crate::organ::integration::SourceWeight {
                source: String::new(), query_domain: String::new(), weight: 1.0,
                success_count: 0, total_count: 0,
            });
        }
        let sw = self.integration_kernel.write().record_feedback(query_domain, source, was_useful);
        let op = Op::RecordFeedback(crate::ops::RecordFeedbackOp {
            source: sw.source.clone(),
            query_domain: sw.query_domain.clone(),
            was_useful,
            new_weight: sw.weight,
            success_count: sw.success_count,
            total_count: sw.total_count,
        });
        self.log.write().append(&op)?;
        Ok(sw)
    }

    pub fn get_source_weights(
        &self,
        domain: Option<&str>,
    ) -> Vec<crate::organ::integration::SourceWeight> {
        if self.ablations.disabled("integration_kernel") { return Vec::new(); }
        self.integration_kernel
            .read()
            .get_source_weights(domain)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn update_source_weight(
        &self,
        source: &str,
        domain: &str,
        weight: f32,
    ) -> Result<bool> {
        if self.ablations.disabled("integration_kernel") { return Ok(false); }
        let ok = self.integration_kernel.write().update_source_weight(source, domain, weight);
        let op = Op::UpdateSourceWeight(crate::ops::UpdateSourceWeightOp {
            source: source.to_string(),
            query_domain: domain.to_string(),
            weight,
        });
        self.log.write().append(&op)?;
        Ok(ok)
    }

    pub fn integration_stats(&self) -> crate::organ::integration::IntegrationStats {
        self.integration_kernel.read().stats()
    }

    // ── Surprise Learning (Moves 1-2) ────────────────────────────────

    pub fn surprise_learning_stats(&self) -> crate::organ::surprise_learning::SurpriseLearningStats {
        self.surprise_learning.read().stats()
    }

    // ── Wisdom Promotion (Move 5) ────────────────────────────────────

    pub fn upsert_wisdom_candidate(
        &self,
        cluster_key: String,
        domain: String,
        action: String,
        summary: String,
        episode_ids: Vec<u64>,
        debt_ids: Vec<u64>,
        support_count: u32,
        cross_session_count: u32,
        mean_surprise: f32,
        promotion_score: f32,
    ) -> Result<u64> {
        if self.ablations.disabled("wisdom_promotion") { return Ok(0); }
        let now = now_ms();
        let candidate_id = {
            let mut store = self.wisdom_promotion.write();
            store.upsert_candidate(
                cluster_key.clone(), domain.clone(), action.clone(), summary.clone(),
                episode_ids.clone(), debt_ids.clone(), support_count,
                cross_session_count, mean_surprise, promotion_score, now,
            )
        };
        let op = Op::UpsertWisdomCandidate(crate::ops::UpsertWisdomCandidateOp {
            candidate_id,
            cluster_key,
            domain,
            action,
            summary,
            episode_ids,
            debt_ids,
            support_count,
            cross_session_count,
            mean_surprise,
            promotion_score,
            created_ms: now,
        });
        self.log.write().append(&op)?;
        Ok(candidate_id)
    }

    pub fn update_wisdom_lifecycle(
        &self,
        candidate_id: u64,
        new_state: crate::organ::wisdom_promotion::WisdomLifecycle,
        memory_id: Option<u64>,
        contradiction_count: u32,
    ) -> Result<bool> {
        if self.ablations.disabled("wisdom_promotion") { return Ok(false); }
        let now = now_ms();
        let old_state = self.wisdom_promotion.read()
            .get(candidate_id)
            .map(|c| c.lifecycle.as_u8())
            .unwrap_or(0);
        let ok = self.wisdom_promotion.write().update_lifecycle(
            candidate_id, new_state, memory_id, contradiction_count, now,
        );
        if ok {
            let op = Op::UpdateWisdomLifecycle(crate::ops::UpdateWisdomLifecycleOp {
                candidate_id,
                memory_id,
                old_state,
                new_state: new_state.as_u8(),
                contradiction_count,
                updated_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    pub fn query_wisdom_candidates(
        &self,
        lifecycle: Option<crate::organ::wisdom_promotion::WisdomLifecycle>,
        domain: Option<&str>,
        limit: usize,
    ) -> Vec<crate::organ::wisdom_promotion::WisdomCandidate> {
        if self.ablations.disabled("wisdom_promotion") { return Vec::new(); }
        self.wisdom_promotion
            .read()
            .query(lifecycle, domain, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn wisdom_promotion_stats(&self) -> crate::organ::wisdom_promotion::WisdomPromotionStats {
        self.wisdom_promotion.read().stats()
    }

    // ── Debt Evidence (Move 3) ───────────────────────────────────────

    pub fn attach_debt_evidence(
        &self,
        debt_id: u64,
        evidence_memory_ids: Vec<u64>,
        confidence: f32,
        note: Option<String>,
    ) -> Result<bool> {
        if self.ablations.disabled("epistemic_debt_store") { return Ok(false); }
        let now = now_ms();
        let ok = self.epistemic_debt_store.write().attach_evidence(
            debt_id, evidence_memory_ids.clone(), confidence, note.clone(), now,
        );
        if ok {
            let op = Op::AttachDebtEvidence(crate::ops::AttachDebtEvidenceOp {
                debt_id,
                evidence_memory_ids,
                confidence,
                note,
                attached_ms: now,
            });
            self.log.write().append(&op)?;
        }
        Ok(ok)
    }

    /// Auto-resolve debts with sufficient evidence. Returns count resolved.
    pub fn auto_resolve_debts(&self, threshold: f32) -> Result<usize> {
        if self.ablations.disabled("epistemic_debt_store") { return Ok(0); }
        let open_ids: Vec<u64> = self.epistemic_debt_store.read()
            .open_debts_with_evidence()
            .iter()
            .filter(|d| !d.evidence.is_empty())
            .map(|d| d.id)
            .collect();

        let now = now_ms();
        let mut resolved_count = 0usize;
        for id in open_ids {
            let resolved = self.epistemic_debt_store.write()
                .auto_resolve_if_ready(id, threshold, now);
            if resolved {
                let op = Op::UpdateDebt(crate::ops::UpdateDebtOp {
                    debt_id: id,
                    status: 1,
                    resolved_ms: now,
                    resolution: Some(format!("auto-resolved: evidence >= {:.2}", threshold)),
                });
                self.log.write().append(&op)?;
                resolved_count += 1;
            }
        }
        Ok(resolved_count)
    }

    // ── Learned Scorer (Move 6) ──────────────────────────────────────

    pub fn update_scorer_model(
        &self,
        weights_json: String,
        model_version: u64,
        mean_loss: f32,
        outcome_count: u64,
    ) -> Result<()> {
        if self.ablations.disabled("learned_scorer") { return Ok(()); }
        let now = now_ms();
        self.learned_scorer.write().apply_update(
            &weights_json, model_version, mean_loss, outcome_count, now,
        );
        let op = Op::UpdateScorerModel(crate::ops::UpdateScorerModelOp {
            model_version,
            baseline_version: self.learned_scorer.read().baseline_version.clone(),
            weights_json,
            applied_at_ms: now,
            outcome_count,
            mean_loss,
        });
        self.log.write().append(&op)?;
        Ok(())
    }

    pub fn learned_scorer_stats(&self) -> crate::scoring::learned::LearnedScoringStats {
        self.learned_scorer.read().stats()
    }

    pub fn effective_scorer_weight(&self, factor_name: &str, baseline: f32) -> f32 {
        if self.ablations.disabled("learned_scorer") { return baseline; }
        self.learned_scorer.read().effective_weight(factor_name, baseline)
    }

    // ── Layer 7: Intervention Ledger ─────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn start_intervention(
        &self,
        realm: String,
        session_id: String,
        task_id: Option<u64>,
        agent_id: String,
        domain: String,
        intent: String,
        action_type: crate::organ::intervention::ActionType,
        action_ref: String,
        preconditions: Vec<String>,
        expected_observables: Vec<String>,
        reversal_cost: crate::organ::intervention::ReversalCost,
    ) -> Result<u64> {
        if self.ablations.disabled("intervention_store") { return Ok(0); }
        let now = now_ms();
        let id = self.intervention_store.write().start_intervention(
            realm.clone(), session_id.clone(), task_id, agent_id.clone(),
            domain.clone(), intent.clone(), action_type, action_ref.clone(),
            preconditions.clone(), expected_observables.clone(), reversal_cost, now,
        );
        self.log.write().append(&crate::ops::Op::StartIntervention(
            crate::ops::StartInterventionOp {
                id, realm, session_id, task_id, agent_id, domain, intent,
                action_type: action_type.to_u8(), action_ref,
                preconditions, expected_observables,
                reversal_cost: reversal_cost.to_u8(), started_ms: now,
            }
        ))?;
        Ok(id)
    }

    pub fn add_observation(
        &self,
        intervention_id: u64,
        kind: crate::organ::intervention::ObservationKind,
        evidence_refs: Vec<u64>,
        summary: String,
        confidence: f32,
    ) -> Result<Option<u64>> {
        if self.ablations.disabled("intervention_store") { return Ok(None); }
        let now = now_ms();
        let obs_id = self.intervention_store.write().add_observation(
            intervention_id, kind, evidence_refs.clone(), summary.clone(), confidence, now,
        );
        if let Some(oid) = obs_id {
            self.log.write().append(&crate::ops::Op::AddObservation(
                crate::ops::AddObservationOp {
                    id: oid, intervention_id, kind: kind.to_u8(),
                    evidence_refs, summary, confidence, timestamp_ms: now,
                }
            ))?;
        }
        Ok(obs_id)
    }

    pub fn close_intervention(
        &self,
        intervention_id: u64,
        status: crate::organ::intervention::InterventionStatus,
    ) -> Result<bool> {
        if self.ablations.disabled("intervention_store") { return Ok(false); }
        use crate::organ::intervention::InterventionStatus;
        use crate::organ::wisdom_lineage::{SUPPORT_DELTA_HIT, CONTRADICTION_DELTA_HIT};
        let now = now_ms();
        let (domain, action_type) = {
            let store = self.intervention_store.read();
            store.get(intervention_id)
                .map(|r| (r.domain.clone(), format!("{:?}", r.action_type).to_lowercase()))
                .unwrap_or_default()
        };
        let ok = self.intervention_store.write().close_intervention(intervention_id, status, now);
        if ok {
            self.log.write().append(&crate::ops::Op::CloseIntervention(
                crate::ops::CloseInterventionOp {
                    intervention_id, status: status.to_u8(), closed_ms: now,
                }
            ))?;

            // ── Layer 9: adjudicate wisdom lineages by outcome ────────
            if !domain.is_empty() {
                let matching = self.wisdom_lineage_store.read()
                    .find_by_envelope(&domain, &action_type);
                let (support_delta, contradiction_delta) = match status {
                    InterventionStatus::Succeeded => (SUPPORT_DELTA_HIT, 0.0f32),
                    InterventionStatus::Failed | InterventionStatus::Aborted => (0.0f32, CONTRADICTION_DELTA_HIT),
                    InterventionStatus::Partial => (SUPPORT_DELTA_HIT * 0.3, CONTRADICTION_DELTA_HIT * 0.3),
                    InterventionStatus::Open => (0.0f32, 0.0f32),
                };
                for lineage_id in matching {
                    let new_state = self.wisdom_lineage_store.write().adjudicate(
                        lineage_id, support_delta, contradiction_delta, 0.0, now,
                    );
                    if let Some(l) = self.wisdom_lineage_store.read().get(lineage_id) {
                        self.log.write().append(&Op::AdjudicateLineage(
                            crate::ops::AdjudicateLineageOp {
                                lineage_id,
                                support_mass: l.support_mass,
                                contradiction_mass: l.contradiction_mass,
                                staleness_mass: l.staleness_mass,
                                last_supported_ms: l.last_supported_ms,
                                last_challenged_ms: l.last_challenged_ms,
                                adjudicated_ms: now,
                            },
                        ))?;
                        if let Some(ns) = new_state {
                            self.log.write().append(&Op::TransitionLineage(
                                crate::ops::TransitionLineageOp {
                                    lineage_id,
                                    old_state: l.state.as_u8(),
                                    new_state: ns.as_u8(),
                                    reason: "intervention_outcome".to_string(),
                                    rederive_task_id: None,
                                    transitioned_ms: now,
                                },
                            ))?;
                        }
                    }
                    if matches!(status, InterventionStatus::Failed | InterventionStatus::Aborted) {
                        let _ = self.wisdom_lineage_store.write().record_challenger(
                            lineage_id,
                            crate::organ::wisdom_lineage::ChallengerEvidence {
                                intervention_id: Some(intervention_id),
                                surprise_id: None,
                                outcome_summary: format!("intervention {} {:?}", intervention_id, status),
                                attached_ms: now,
                            },
                            now,
                        );
                    }
                }
            }
        }
        Ok(ok)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_attribution(
        &self,
        intervention_id: u64,
        primary_class: crate::organ::intervention::AttributionClass,
        secondary_class: Option<crate::organ::intervention::AttributionClass>,
        confidence_delta: f32,
        surprise_id: Option<u64>,
        debt_ids: Vec<u64>,
        source_memory_ids: Vec<u64>,
        skill_memory_ids: Vec<u64>,
        note: Option<String>,
    ) -> Result<bool> {
        if self.ablations.disabled("intervention_store") { return Ok(false); }
        let now = now_ms();
        // Look up intervention domain before releasing write lock
        let domain = {
            let store = self.intervention_store.read();
            store.get(intervention_id).map(|r| r.domain.clone()).unwrap_or_default()
        };
        let ok = self.intervention_store.write().record_attribution(
            intervention_id, primary_class, secondary_class,
            confidence_delta, surprise_id, debt_ids.clone(),
            source_memory_ids.clone(), skill_memory_ids.clone(), note.clone(), now,
        );
        if !ok { return Ok(false); }
        self.log.write().append(&crate::ops::Op::RecordAttribution(
            crate::ops::RecordAttributionOp {
                intervention_id,
                primary_class: primary_class.to_u8(),
                secondary_class: secondary_class.map(|c| c.to_u8()),
                confidence_delta, surprise_id,
                debt_ids: debt_ids.clone(),
                source_memory_ids: source_memory_ids.clone(),
                skill_memory_ids: skill_memory_ids.clone(),
                note, timestamp_ms: now,
            }
        ))?;
        // Route to learning subsystems
        self.route_attribution(&domain, primary_class, confidence_delta,
            surprise_id, &source_memory_ids, &skill_memory_ids);
        if let Some(sec) = secondary_class {
            self.route_attribution(&domain, sec, confidence_delta * 0.5,
                surprise_id, &source_memory_ids, &skill_memory_ids);
        }
        Ok(true)
    }

    pub(super) fn route_attribution(
        &self,
        domain: &str,
        class: crate::organ::intervention::AttributionClass,
        confidence_delta: f32,
        surprise_id: Option<u64>,
        source_memory_ids: &[u64],
        skill_memory_ids: &[u64],
    ) {
        use crate::organ::intervention::AttributionClass::*;
        let now = now_ms();
        match class {
            MemoryRecallError => {
                if let Some(sid) = surprise_id {
                    let mut sl = self.surprise_learning.write();
                    for &mid in source_memory_ids {
                        let _ = sl.update_credit(mid, sid, confidence_delta.abs(), -1, now);
                    }
                }
            }
            SourceTrustError => {
                let _ = self.integration_kernel.write().record_feedback(domain, "memory", false);
            }
            ProcedureError => {
                for &mid in skill_memory_ids {
                    let _ = self.update_state(
                        mid, Some(-confidence_delta.abs()), None, None, false, None,
                    );
                }
            }
            ToolExecutionError | EnvironmentShift | HiddenPrecondition
            | AmbiguousState | GoalSpecError | UserOverride | ExternalNondeterminism => {
                // No automatic side-effect; caller handles debt/task repair at MCP layer
            }
        }
    }

    pub fn get_intervention(
        &self, id: u64,
    ) -> Option<crate::organ::intervention::InterventionRecord> {
        if self.ablations.disabled("intervention_store") { return None; }
        self.intervention_store.read().get(id).cloned()
    }

    pub fn query_interventions(
        &self,
        realm: Option<&str>,
        session_id: Option<&str>,
        status: Option<crate::organ::intervention::InterventionStatus>,
        limit: usize,
    ) -> Vec<crate::organ::intervention::InterventionRecord> {
        if self.ablations.disabled("intervention_store") { return Vec::new(); }
        self.intervention_store.read()
            .query(realm, session_id, status, limit)
            .into_iter().cloned().collect()
    }

    pub fn list_open_interventions(
        &self,
    ) -> Vec<crate::organ::intervention::InterventionRecord> {
        if self.ablations.disabled("intervention_store") { return Vec::new(); }
        self.intervention_store.read().list_open().into_iter().cloned().collect()
    }

    pub fn intervention_stats(&self) -> crate::organ::intervention::InterventionStats {
        self.intervention_store.read().stats()
    }

    pub fn close_stale_interventions(&self, threshold_ms: i64) -> Result<usize> {
        if self.ablations.disabled("intervention_store") { return Ok(0); }
        let now = now_ms();
        let stale_ids = self.intervention_store.read().stale_open(threshold_ms, now);
        let mut closed = 0usize;
        for id in stale_ids {
            let ok = self.intervention_store.write().close_intervention(
                id, crate::organ::intervention::InterventionStatus::Aborted, now,
            );
            if ok {
                self.log.write().append(&crate::ops::Op::CloseIntervention(
                    crate::ops::CloseInterventionOp {
                        intervention_id: id,
                        status: crate::organ::intervention::InterventionStatus::Aborted.to_u8(),
                        closed_ms: now,
                    }
                ))?;
                closed += 1;
            }
        }
        Ok(closed)
    }

    // ── Layer 9: Wisdom Homeostasis ───────────────────────────────

    #[allow(clippy::too_many_arguments)]
    pub fn enroll_wisdom_lineage(
        &self,
        wisdom_candidate_id: u64,
        claim: String,
        envelope_json: String,
        seed_episode_ids: Vec<u64>,
        seed_surprise_ids: Vec<u64>,
        seed_intervention_ids: Vec<u64>,
        seed_debt_ids: Vec<u64>,
        ancestor_lineage_id: Option<u64>,
        derivation_relation: Option<String>,
    ) -> Result<u64> {
        if self.ablations.disabled("wisdom_lineage_store") { return Ok(0); }
        use crate::organ::wisdom_lineage::ApplicabilityEnvelope;
        let now = now_ms();
        let envelope: ApplicabilityEnvelope =
            serde_json::from_str(&envelope_json).unwrap_or_default();
        let lineage_id = self.wisdom_lineage_store.write().enroll(
            wisdom_candidate_id, claim.clone(), envelope, seed_episode_ids.clone(),
            seed_surprise_ids.clone(), seed_intervention_ids.clone(), seed_debt_ids.clone(),
            ancestor_lineage_id, derivation_relation.clone(), now,
        );
        self.log.write().append(&Op::UpsertWisdomLineage(
            crate::ops::UpsertWisdomLineageOp {
                lineage_id,
                wisdom_candidate_id,
                claim,
                envelope_json,
                seed_episode_ids,
                seed_surprise_ids,
                seed_intervention_ids,
                seed_debt_ids,
                ancestor_lineage_id,
                derivation_version: 0,
                derivation_relation,
                rederive_ttl_ms: crate::organ::wisdom_lineage::DEFAULT_REDERIVE_TTL_MS,
                created_ms: now,
                updated_ms: now,
            },
        ))?;
        Ok(lineage_id)
    }

    pub fn transition_wisdom_lineage(
        &self,
        lineage_id: u64,
        new_state: u8,
        reason: String,
        rederive_task_id: Option<u64>,
    ) -> Result<bool> {
        if self.ablations.disabled("wisdom_lineage_store") { return Ok(false); }
        use crate::organ::wisdom_lineage::LineageState;
        let now = now_ms();
        let old_state = self.wisdom_lineage_store.read()
            .get(lineage_id).map(|l| l.state.as_u8()).unwrap_or(0);
        let ok = self.wisdom_lineage_store.write().transition_state(
            lineage_id, LineageState::from_u8(new_state), &reason, rederive_task_id, now,
        );
        if ok {
            self.log.write().append(&Op::TransitionLineage(
                crate::ops::TransitionLineageOp {
                    lineage_id, old_state, new_state,
                    reason, rederive_task_id, transitioned_ms: now,
                },
            ))?;
        }
        Ok(ok)
    }

    pub fn close_rederive(
        &self,
        lineage_id: u64,
        action: u8,
        new_envelope_json: Option<String>,
        fork_claim: Option<String>,
        fork_lineage_id: Option<u64>,
    ) -> Result<()> {
        if self.ablations.disabled("wisdom_lineage_store") { return Ok(()); }
        use crate::organ::wisdom_lineage::{ApplicabilityEnvelope, RederiveAction};
        let now = now_ms();
        let new_envelope = new_envelope_json.as_deref()
            .and_then(|j| serde_json::from_str::<ApplicabilityEnvelope>(j).ok());
        self.wisdom_lineage_store.write().close_rederive(
            lineage_id, RederiveAction::from_u8(action),
            new_envelope, fork_claim.clone(), fork_lineage_id, now,
        );
        self.log.write().append(&Op::CloseRederive(
            crate::ops::CloseRederiveOp {
                lineage_id, action,
                new_envelope_json, fork_claim, fork_lineage_id, closed_ms: now,
            },
        ))?;
        Ok(())
    }

    pub fn query_wisdom_lineages(
        &self,
        state_str: Option<&str>,
        domain: Option<&str>,
        limit: usize,
    ) -> Vec<crate::organ::wisdom_lineage::WisdomLineage> {
        if self.ablations.disabled("wisdom_lineage_store") { return Vec::new(); }
        use crate::organ::wisdom_lineage::LineageState;
        let state_filter = state_str.and_then(|s| match s {
            "trusted" => Some(LineageState::Trusted),
            "watch" => Some(LineageState::Watch),
            "inflamed" => Some(LineageState::Inflamed),
            "demoted" => Some(LineageState::Demoted),
            _ => None,
        });
        self.wisdom_lineage_store.read()
            .query(state_filter, domain, limit)
            .into_iter().cloned().collect()
    }

    pub fn get_wisdom_lineage(
        &self, id: u64,
    ) -> Option<crate::organ::wisdom_lineage::WisdomLineage> {
        if self.ablations.disabled("wisdom_lineage_store") { return None; }
        self.wisdom_lineage_store.read().get(id).cloned()
    }

    pub fn wisdom_lineage_stats(&self) -> crate::organ::wisdom_lineage::WisdomLineageStats {
        self.wisdom_lineage_store.read().stats()
    }

    /// Grow staleness on stale lineages and return IDs that transitioned.
    pub fn tick_lineage_staleness(&self) -> Result<Vec<u64>> {
        if self.ablations.disabled("wisdom_lineage_store") { return Ok(Vec::new()); }
        let now = now_ms();
        let transitioned = self.wisdom_lineage_store.write().tick_staleness(now);
        for &lineage_id in &transitioned {
            if let Some(l) = self.wisdom_lineage_store.read().get(lineage_id) {
                self.log.write().append(&Op::AdjudicateLineage(
                    crate::ops::AdjudicateLineageOp {
                        lineage_id,
                        support_mass: l.support_mass,
                        contradiction_mass: l.contradiction_mass,
                        staleness_mass: l.staleness_mass,
                        last_supported_ms: l.last_supported_ms,
                        last_challenged_ms: l.last_challenged_ms,
                        adjudicated_ms: now,
                    },
                ))?;
                self.log.write().append(&Op::TransitionLineage(
                    crate::ops::TransitionLineageOp {
                        lineage_id,
                        old_state: 0,
                        new_state: l.state.as_u8(),
                        reason: "staleness_tick".to_string(),
                        rederive_task_id: None,
                        transitioned_ms: now,
                    },
                ))?;
            }
        }
        Ok(transitioned)
    }

    /// Return IDs of Inflamed lineages whose re-derive TTL has expired.
    pub fn lineage_expiry_check(&self) -> Vec<u64> {
        if self.ablations.disabled("wisdom_lineage_store") { return Vec::new(); }
        self.wisdom_lineage_store.read().expiry_check(now_ms())
    }
}
