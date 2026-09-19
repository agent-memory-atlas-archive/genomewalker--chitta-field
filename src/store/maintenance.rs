//! maintenance operations.

use super::*;

impl ChittaField {

    /// Invalidate a triplet (marks it as expired at the current time).
    /// Backfill embedding for a memory stored with embed_pending=true.
    /// Durable: writes UpdateMemoryContent op to WAL (content unchanged, new embedding).
    pub fn backfill_embedding(&self, memory_id: MemoryId, embedding: &[f32]) -> Result<()> {
        if embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim { expected: EMBED_DIM, actual: embedding.len() });
        }
        let existing_content = {
            // Lock order: payloads before states (matches sync_foreign).
            let payloads = self.payloads.read();
            let states = self.states.read();
            match states.get(&memory_id) {
                None => return Err(FieldError::NotFound(memory_id)),
                Some(st) if !st.embed_pending => return Ok(()),
                Some(_) => {
                    payloads.get(&memory_id)
                        .map(|p| p.content.clone())
                        .unwrap_or_default()
                }
            }
        };

        // Persist to WAL via UpdateMemoryContent (empty content = reuse existing)
        let op = Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
            memory_id,
            content: existing_content.clone(),
            embedding: embedding.to_vec(),
            op_ts_ms: now_ms(),
        });
        self.log.write().append(&op)?;

        // Update payload embedding metadata; the vector itself lives only in
        // the semantic index (upserted below) — see ChittaField::embedding_of.
        {
            let mut payloads = self.payloads.write();
            if let Some(p) = payloads.get_mut(&memory_id) {
                p.embedding = Vec::new();
                p.embedding_model = EMBED_MODEL_ID.to_string();
                p.embedding_model_id = EMBED_MODEL_ID.to_string();
                p.embedding_dim = EMBED_DIM as u32;
            }
        }

        // Update semantic index (get realm for per-realm routing)
        let realm_for_upsert = self.payloads.read()
            .get(&memory_id)
            .map(|p| p.realm.clone());

        // Semantic dedup — the check that store.rs:1321 CANNOT run on the normal path.
        // On the production path every memory is written embed_pending with an empty
        // vector, so that write-time guard (`if !embed_pending`) is false and
        // near-duplicates would otherwise accumulate unbounded. (It does still fire for
        // callers that supply their own vector — see the CONSTRAINT note at that site.)
        // This is the first moment the vector exists; search runs BEFORE this
        // memory's own vector is upserted, so k=1 returns the nearest OTHER memory.
        // A hit supersedes THIS (newer) memory rather than deleting it: reversible, keeps
        // provenance/triplet references, excluded from recall, and reinforces the original.
        let mut superseded_by: Option<MemoryId> = None;
        {
            let (dedup_thresh, dedup_upper) = {
                let cfg = &self.scoring_pipeline.read().config;
                (cfg.dedup_cosine_threshold, cfg.dedup_cosine_upper)
            };
            let neighbors = self.semantic_idx.read().search(embedding, 1, None, None);
            if let Some(top) = neighbors.first() {
                if top.memory_id != memory_id
                    && top.cosine_similarity >= dedup_thresh
                    && top.cosine_similarity < dedup_upper
                {
                    let same_realm = self.payloads.read()
                        .get(&top.memory_id)
                        .map(|p| Some(&p.realm) == realm_for_upsert.as_ref())
                        .unwrap_or(false);
                    let alive = self.states.read()
                        .get(&top.memory_id)
                        .map(|s| !s.deleted)
                        .unwrap_or(false);
                    if same_realm && alive {
                        superseded_by = Some(top.memory_id);
                    }
                }
            }
        }

        self.semantic_idx.write().upsert(
            memory_id,
            embedding.to_vec(),
            realm_for_upsert.as_deref(),
        );

        if let Some(original) = superseded_by {
            self.add_triplet(original.to_string(), "merged_from".into(), memory_id.to_string(), 1.0, None, None)?;
            let _ = self.set_memory_status(memory_id, crate::state::MemoryStatus::Superseded);
            let _ = self.update_state(original, Some(0.0), Some(0.02), None, true, None);
        }

        // Re-encode cortical sparse index (non-fatal)
        let _ = self.encode_memory(memory_id);

        // Clear embed_pending in state
        {
            let mut states = self.states.write();
            if let Some(st) = states.get_mut(&memory_id) {
                if st.embed_pending {
                    st.embed_pending = false;
                    self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        Ok(())
    }

    // ── Deferred batched backfill (Step-1 re-architecture) ─────────────────────
    //
    // Splits the per-item `backfill_embedding` (WAL + payload + O(log N) HNSW insert,
    // all under one `semantic_idx.write()`) into three phases so the expensive HNSW
    // neighbor search runs OFF the write lock:
    //
    //   stage(items) — WAL + payload meta + metadata upsert (coarse/LSH/binary/embeddings
    //                  map), but NOT the global HNSW insert. Brief write. MUST be called
    //                  under the C++ rpc_mutex (acquire_lock), same as backfill_embedding.
    //   plan()       — compute HNSW insert plans under `semantic_idx.read()` (concurrent
    //                  with recall). MUST be called WITHOUT the rpc_mutex, so recall (which
    //                  needs the shared rpc_mutex) is not blocked while the search runs.
    //   apply()      — wire the plans into the graph + clear embed_pending. Brief write.
    //                  MUST be called under the rpc_mutex — preserves the rpc→semantic_idx
    //                  lock order, so no new ABBA vs sync_foreign.
    //
    // Correctness: the durable state (WAL + payload) lands in stage(); a crash between
    // stage and apply just leaves the memory embed_pending (re-picked next drain) — no
    // torn graph, no lost write. A concurrent writer inserting between plan and apply is
    // untouched by apply (see apply_delta_batch). Recall parity: id-seeded levels match
    // the load-time backfill_hnsw_delta_parallel path, so the resulting graph is
    // build-order-independent and equivalent to the per-item path.

    /// Phase 1: durable stage of a backfill batch. Returns the number of ids that
    /// still need a global-HNSW plan (drives whether `plan()` has work to do).
    pub fn backfill_stage(&self, items: &[(MemoryId, Vec<f32>)]) -> Result<usize> {
        // Pass 1 (WAL + payload meta): collect the valid (id, embedding, realm) tuples.
        let mut valid: Vec<(MemoryId, &Vec<f32>, Option<String>)> = Vec::with_capacity(items.len());
        for (memory_id, embedding) in items {
            let memory_id = *memory_id;
            if embedding.len() != EMBED_DIM {
                return Err(FieldError::InvalidEmbedDim { expected: EMBED_DIM, actual: embedding.len() });
            }
            // Skip ids that are gone or already embedded (mirrors backfill_embedding).
            let existing_content = {
                let payloads = self.payloads.read();
                let states = self.states.read();
                match states.get(&memory_id) {
                    None => continue,
                    Some(st) if !st.embed_pending => continue,
                    Some(_) => payloads.get(&memory_id).map(|p| p.content.clone()).unwrap_or_default(),
                }
            };
            // WAL: persist the embedding via UpdateMemoryContent (empty content reuses existing).
            let op = Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
                memory_id,
                content: existing_content,
                embedding: embedding.clone(),
                op_ts_ms: now_ms(),
            });
            self.log.write().append(&op)?;
            // Payload embedding metadata — the vector itself lives only in the semantic index.
            let realm_for_upsert = {
                let mut payloads = self.payloads.write();
                if let Some(p) = payloads.get_mut(&memory_id) {
                    p.embedding = Vec::new();
                    p.embedding_model = EMBED_MODEL_ID.to_string();
                    p.embedding_model_id = EMBED_MODEL_ID.to_string();
                    p.embedding_dim = EMBED_DIM as u32;
                    Some(p.realm.clone())
                } else {
                    None
                }
            };
            valid.push((memory_id, embedding, realm_for_upsert));
        }
        self.log.write().flush_buf().ok();
        // Pass 2 (metadata upsert): ONE semantic_idx.write() for the whole batch, not one
        // per item — a single acquisition means a waiting recall reader starves for one
        // brief metadata write, not the full burst (parking_lot writer-preference would
        // otherwise let per-item re-acquisition starve readers across the whole chunk).
        let mut staged_ids: Vec<MemoryId> = Vec::with_capacity(valid.len());
        let mut plan_ids: Vec<MemoryId> = Vec::new();
        {
            let mut idx = self.semantic_idx.write();
            for (memory_id, embedding, realm) in &valid {
                let needs_plan = idx.upsert_deferred(*memory_id, (*embedding).clone(), realm.as_deref());
                staged_ids.push(*memory_id);
                if needs_plan { plan_ids.push(*memory_id); }
            }
        }
        let n_plan = plan_ids.len();
        *self.backfill_plan_stage.lock() = Some(BackfillStage { staged_ids, plan_ids, plan: None });
        Ok(n_plan)
    }

    /// Phase 2: compute HNSW insert plans off the write lock. No-op if nothing staged.
    /// Call WITHOUT the C++ rpc_mutex so recall is not blocked during the search.
    pub fn backfill_plan(&self) {
        let plan_ids = {
            let stage = self.backfill_plan_stage.lock();
            match stage.as_ref() {
                Some(s) if !s.plan_ids.is_empty() => s.plan_ids.clone(),
                _ => return,
            }
        };
        let plan = self.semantic_idx.read().plan_delta_batch(&plan_ids);
        if let Some(s) = self.backfill_plan_stage.lock().as_mut() {
            s.plan = Some(plan);
        }
    }

    /// Phase 3: wire the staged plan into the graph and clear embed_pending. Returns the
    /// number of memories whose embed_pending was cleared. Call under the C++ rpc_mutex.
    pub fn backfill_apply(&self) -> usize {
        let stage = match self.backfill_plan_stage.lock().take() {
            Some(s) => s,
            None => return 0,
        };
        if let Some(plan) = stage.plan {
            self.semantic_idx.write().apply_delta_batch(plan);
        }
        for &id in &stage.staged_ids {
            let _ = self.encode_memory(id);
        }
        let mut states = self.states.write();
        for &id in &stage.staged_ids {
            if let Some(st) = states.get_mut(&id) {
                if st.embed_pending {
                    st.embed_pending = false;
                    self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        stage.staged_ids.len()
    }

    /// Convenience: run the full stage → plan → apply sequence in one call. Used by the
    /// lock-free embed_loop drain and by tests. Does NOT take the C++ rpc_mutex, so it
    /// relies solely on the internal semantic_idx lock discipline (safe for the embed_loop
    /// path, which never held acquire_lock). Returns the number of memories backfilled.
    pub fn backfill_embeddings_batch(&self, items: &[(MemoryId, Vec<f32>)]) -> Result<usize> {
        self.backfill_stage(items)?;
        self.backfill_plan();
        Ok(self.backfill_apply())
    }

    /// Persist the live delta HNSW tier to its sidecar so the NEXT restart LOADS the
    /// delta graph instead of cold-reinserting every runtime-added node from scratch
    /// (single-threaded, ~minutes once the corpus grows). Pure optimization, never a
    /// hard dependency: on startup a missing/stale/unreadable delta.hnsw silently falls
    /// back to the existing backfill/rebuild path (normalize_all's hnsw_len==total gate).
    ///
    /// Writes to the DETERMINISTIC snapshot stem this instance loads on open
    /// (`chitta.<instance>.delta.hnsw`, mirroring save_full_snapshot's path at
    /// store.rs:7639), via save_delta_hnsw's write-temp + atomic-rename (NFS-safe on
    /// Isilon — no delete-then-write). The n==0 guard skips the write entirely when the
    /// delta is empty, avoiding save_delta_hnsw's remove_file branch (an NFS delete).
    ///
    /// Takes only semantic_idx.read() (like a recall) — safe to call OFF the rpc_mutex,
    /// so it never blocks live recalls. Returns the number of delta nodes persisted
    /// (0 = nothing to persist or write failed; both are non-fatal).
    pub fn persist_delta_hnsw(&self) -> usize {
        let idx = self.semantic_idx.read();
        let n = idx.delta_len();
        if n == 0 {
            return 0;
        }
        let path = self
            .data_dir
            .join(format!("chitta.{:08x}.delta.hnsw", self.instance_id));
        if let Err(e) = idx.save_delta_hnsw(&path) {
            eprintln!("[chitta-field] WARNING: persist_delta_hnsw failed: {e}");
            return 0;
        }
        n
    }

    /// Force a full rebuild of all derived search structures (binary codes, coarse,
    /// LSH, HNSW) from the current embeddings. Required after an embedding-dimension
    /// migration, where re-embedding updates the float vectors but leaves the ANN
    /// indices built in the old vector space.
    pub fn force_reindex(&self) {
        self.semantic_idx.write().force_reindex();
    }

    /// Return memory IDs with embed_pending=true, sorted oldest first, up to limit.
    pub fn pending_embeddings(&self, limit: usize) -> Vec<MemoryId> {
        let s = self.states.read();
        let mut pending: Vec<(i64, MemoryId)> = s
            .iter()
            .filter(|(_, st)| st.embed_pending && !st.deleted)
            .map(|(id, st)| (st.created_at_ms, *id))
            .collect();
        pending.sort_by_key(|(ts, _)| *ts); // oldest first
        if pending.len() <= limit {
            return pending.into_iter().map(|(_, id)| id).collect();
        }
        // Backlog larger than one batch: embed the NEWEST (limit - half) first so a
        // just-written memory becomes recallable within ONE batch instead of waiting behind
        // the entire backlog (the write->recall lag), while still draining the OLDEST `half`
        // each batch so nothing starves. Minimises lag without dropping any pending memory.
        let half = limit / 2;
        let newest: Vec<MemoryId> =
            pending.iter().rev().take(limit - half).map(|(_, id)| *id).collect();
        let oldest: Vec<MemoryId> =
            pending.iter().take(half).map(|(_, id)| *id).collect();
        newest.into_iter().chain(oldest).collect()
    }

    /// Clear embed_pending for specific memory IDs, regardless of content.
    /// Returns count actually cleared (skips IDs not in pending state).
    pub fn purge_orphan_embed_pending(&self) -> usize {
        // Collect all embed_pending IDs
        let pending: Vec<MemoryId> = {
            let states = self.states.read();
            states.iter()
                .filter(|(_, st)| st.embed_pending && !st.deleted)
                .map(|(id, _)| *id)
                .collect()
        };
        if pending.is_empty() { return 0; }

        // Check which ones get_memory() would fail for (not in payloads or error)
        let to_clear: Vec<MemoryId> = pending.iter()
            .filter(|id| self.get_memory(**id).is_err())
            .copied()
            .collect();

        let n = to_clear.len();
        if n > 0 {
            let mut states = self.states.write();
            for id in &to_clear {
                if let Some(st) = states.get_mut(id) {
                    if st.embed_pending {
                        st.embed_pending = false;
                        self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
        n
    }

    /// Semantic-index coverage: (eligible, embedded).
    ///
    /// `eligible` = live memories with content long enough to embed — the ones that SHOULD
    /// have a vector. `embedded` = those the index actually holds one for. The gap is the
    /// number of memories reachable by BM25 only, and no ranker change can ever surface them.
    ///
    /// This exists because `pending_count` CANNOT answer the question: the embed queue is
    /// drained by failure paths as well as success paths, so a lost embedding leaves the queue
    /// empty and the store reports perfect health over a half-empty index. Derive coverage from
    /// the index; never infer it from the queue.
    /// The stored vector for a memory, exactly as the index holds it.
    /// Nothing else exposes this — which is why a document embedded into the wrong
    /// vector space has been undetectable from outside the process.
    pub fn get_embedding(&self, id: MemoryId) -> Option<Vec<f32>> {
        self.semantic_idx.read().get_embedding(id).map(|v| v.to_vec())
    }

    pub fn embed_coverage(&self) -> (usize, usize) {
        const MIN_EMBED_CHARS: usize = 20;
        // Locks are taken one at a time, never nested (get_memory() re-locks `states`).
        let live: Vec<MemoryId> = {
            let states = self.states.read();
            states.iter().filter(|(_, st)| !st.deleted).map(|(id, _)| *id).collect()
        };
        let eligible: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            live.into_iter()
                .filter(|id| payloads.get(id)
                    .map(|p| p.content.len() >= MIN_EMBED_CHARS)
                    .unwrap_or(false))
                .collect()
        };
        let idx = self.semantic_idx.read();
        let embedded = eligible.iter().filter(|id| idx.has_embedding(**id)).count();
        (eligible.len(), embedded)
    }

    /// Force-clear embed_pending for specific IDs (maintenance tool).
    pub fn force_clear_embed_pending(&self, ids: &[MemoryId]) -> usize {
        let mut states = self.states.write();
        let mut n = 0usize;
        for id in ids {
            if let Some(st) = states.get_mut(id) {
                if st.embed_pending {
                    st.embed_pending = false;
                    self.pending_embed_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    n += 1;
                }
            }
        }
        n
    }

    /// Re-queue memories that have no embedding but are embeddable (ghost backfill failures).
    /// A ghost has embed_pending=false, embedding_model="none", empty embedding, and content
    /// long enough for BGE. Sets embed_pending=true so the backfill thread picks them up.
    /// Returns count of memories re-queued.
    pub fn requeue_ghost_embeddings(&self) -> usize {
        const MIN_EMBED_CHARS: usize = 20;

        // Pass 1: collect candidates — states+payloads locked briefly, then released.
        let candidates: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states   = self.states.read();
            states.iter()
                .filter(|(id, st)| {
                    !st.deleted && !st.embed_pending &&
                    payloads.get(id)
                        .map(|p| p.content.len() >= MIN_EMBED_CHARS)
                        .unwrap_or(false)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        if candidates.is_empty() { return 0; }

        // Pass 2: filter to those absent from HNSW — semantic_idx locked briefly, then released.
        let ghost_ids: Vec<MemoryId> = {
            let idx = self.semantic_idx.read();
            candidates.into_iter().filter(|id| !idx.contains(*id)).collect()
        };
        if ghost_ids.is_empty() { return 0; }

        // Pass 3: mark ghosts as embed_pending — states write-locked briefly.
        let mut states = self.states.write();
        let mut count = 0usize;
        for id in &ghost_ids {
            if let Some(st) = states.get_mut(id) {
                if !st.embed_pending && !st.deleted {
                    st.embed_pending = true;
                    self.pending_embed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    count += 1;
                }
            }
        }
        count
    }

    /// Mark every non-deleted memory with sufficient content as `embed_pending = true`
    /// so the backfill thread re-embeds them with the new model (e.g. after a dim change).
    /// Returns the count of memories marked.
    pub fn requeue_all_embeddings(&self, _model_id: &str) -> Result<usize> {
        const MIN_EMBED_CHARS: usize = 10;

        // Pass 1: collect candidates — brief shared lock on both maps.
        let candidates: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states   = self.states.read();
            states.iter()
                .filter(|(id, st)| {
                    !st.deleted &&
                    payloads.get(id)
                        .map(|p| p.content.len() >= MIN_EMBED_CHARS)
                        .unwrap_or(false)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        if candidates.is_empty() { return Ok(0); }

        // Pass 2: mark as embed_pending — exclusive write lock.
        let mut states = self.states.write();
        let mut count = 0usize;
        for id in &candidates {
            if let Some(st) = states.get_mut(id) {
                if !st.deleted {
                    if !st.embed_pending {
                        st.embed_pending = true;
                        self.pending_embed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Per-realm embedding geometry stats (inspired by "Geometry of Forgetting").
    /// Returns JSON: `{"by_realm": [...], "by_kind": [...], "anomalies": [...]}`
    pub fn spectral_stats_by_realm(&self) -> String {
        let realm_stats = self.realm_stats.read();
        let kind_stats  = self.kind_stats.read();

        let mut realm_results: Vec<serde_json::Value> = realm_stats
            .iter()
            .filter_map(|(name, stats)| stats.geometry(name))
            .collect();
        realm_results.sort_by(|a, b| {
            b["count"].as_u64().unwrap_or(0).cmp(&a["count"].as_u64().unwrap_or(0))
        });

        let mut kind_results: Vec<serde_json::Value> = kind_stats
            .iter()
            .filter_map(|(name, stats)| stats.geometry(name))
            .collect();
        kind_results.sort_by(|a, b| {
            a["group"].as_str().unwrap_or("").cmp(b["group"].as_str().unwrap_or(""))
        });

        let mut anomalies: Vec<serde_json::Value> = Vec::new();
        for entry in realm_results.iter().chain(kind_results.iter()) {
            let label = entry["group"].as_str().unwrap_or("?");
            let cos   = entry["mean_cosine_sim"].as_f64().unwrap_or(0.0);
            let iso   = entry["isotropy"].as_f64().unwrap_or(1.0);
            let count = entry["count"].as_u64().unwrap_or(0);
            let has_newline = label.contains('\n') || label.contains('\r');
            if cos > 0.95 && count >= 5 {
                anomalies.push(serde_json::json!({
                    "group": label, "issue": "high_similarity",
                    "detail": format!("cos={:.3} across {} memories — likely duplicates", cos, count)
                }));
            }
            if iso < 0.3 && count >= 5 {
                anomalies.push(serde_json::json!({
                    "group": label, "issue": "collapsed_embeddings",
                    "detail": format!("isotropy={:.3} — embeddings occupy narrow subspace", iso)
                }));
            }
            if has_newline {
                anomalies.push(serde_json::json!({
                    "group": label.trim(), "issue": "dirty_realm_name",
                    "detail": "realm contains trailing whitespace/newline"
                }));
            }
        }

        serde_json::to_string(&serde_json::json!({
            "by_realm": realm_results,
            "by_kind":  kind_results,
            "anomalies": anomalies,
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Fix realm names that contain trailing whitespace/newlines.
    /// Returns the number of memories whose realm was trimmed.
    pub fn trim_realm_names(&self) -> usize {
        // Collect dirty memories: (memory_id, old_realm, trimmed_realm)
        let dirty: Vec<(MemoryId, String, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter_map(|(mid, p)| {
                    if states.get(mid).map(|s| s.deleted).unwrap_or(true) {
                        return None;
                    }
                    let trimmed = p.realm.trim().to_string();
                    if trimmed != p.realm {
                        Some((*mid, p.realm.clone(), trimmed))
                    } else {
                        None
                    }
                })
                .collect()
        };

        let count = dirty.len();
        for (mid, old_realm, new_realm) in dirty {
            // Update payload realm
            if let Some(p) = self.payloads.write().get_mut(&mid) {
                p.realm = new_realm.clone();
            }
            // Update realm_members index
            let mut rm = self.realm_members.write();
            if let Some(set) = rm.get_mut(&old_realm) {
                set.remove(&mid);
                if set.is_empty() {
                    rm.remove(&old_realm);
                }
            }
            rm.entry(new_realm).or_default().insert(mid);
        }
        count
    }

    /// Bulk-remap realms per an explicit `{old_realm -> new_realm}` mapping.
    /// `dry_run` computes the move census without mutating. On a real run it
    /// updates payload.realm + realm_members, then forces a full snapshot for
    /// durability (cf_set_realm emits no WAL, so the snapshot is the only durable
    /// record). Returns a JSON summary string.
    pub fn remap_realms(
        &self,
        mapping: &std::collections::HashMap<String, String>,
        dry_run: bool,
    ) -> String {
        let moves: Vec<(MemoryId, String, String)> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            payloads
                .iter()
                .filter_map(|(mid, p)| {
                    if states.get(mid).map(|s| s.deleted).unwrap_or(true) {
                        return None;
                    }
                    match mapping.get(&p.realm) {
                        Some(new_realm) if new_realm != &p.realm => {
                            Some((*mid, p.realm.clone(), new_realm.clone()))
                        }
                        _ => None,
                    }
                })
                .collect()
        };
        let moved = moves.len();
        let mut per_target: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (_, _, new_realm) in &moves {
            *per_target.entry(new_realm.clone()).or_insert(0) += 1;
        }

        if !dry_run {
            for (mid, old_realm, new_realm) in &moves {
                if let Some(p) = self.payloads.write().get_mut(mid) {
                    p.realm = new_realm.clone();
                }
                let mut rm = self.realm_members.write();
                if let Some(set) = rm.get_mut(old_realm) {
                    set.remove(mid);
                    if set.is_empty() {
                        rm.remove(old_realm);
                    }
                }
                rm.entry(new_realm.clone()).or_default().insert(*mid);
            }
        }

        let final_realms = self.realm_members.read().len();
        let snapshot_ok = if !dry_run {
            self.save_full_snapshot().is_ok()
        } else {
            false
        };
        let mut targets: Vec<(String, usize)> = per_target.into_iter().collect();
        targets.sort_by(|a, b| b.1.cmp(&a.1));
        let targets_json: Vec<serde_json::Value> = targets
            .iter()
            .map(|(t, c)| serde_json::json!({"realm": t, "count": c}))
            .collect();
        serde_json::json!({
            "dry_run": dry_run,
            "moved": moved,
            "targets": targets_json,
            "final_realms": final_realms,
            "snapshot_ok": snapshot_ok,
        })
        .to_string()
    }

    /// Save a spectral stats snapshot for temporal drift tracking.
    /// Writes `spectral_snapshot_{timestamp}.json` to the data dir.
    pub fn save_spectral_snapshot(&self) -> Result<String> {
        let stats_json = self.spectral_stats_by_realm();
        let ts = now_ms();
        let filename = format!("spectral_snapshot_{}.json", ts);
        let path = self.data_dir.join(&filename);
        let wrapped = serde_json::json!({
            "ts_ms": ts,
            "stats": serde_json::from_str::<serde_json::Value>(&stats_json).unwrap_or_default(),
        });
        let content = serde_json::to_string_pretty(&wrapped)
            .map_err(|e| FieldError::Serialization(e.to_string()))?;
        std::fs::write(&path, content)
            .map_err(FieldError::Io)?;
        Ok(filename)
    }

    /// Load spectral drift: compare current stats with most recent snapshot.
    /// Returns JSON with per-realm/kind delta for isotropy and mean_cosine_sim.
    pub fn spectral_drift(&self) -> String {
        // Find most recent snapshot
        let entries = match std::fs::read_dir(&self.data_dir) {
            Ok(e) => e,
            Err(_) => return "{}".to_string(),
        };
        let mut snapshots: Vec<(i64, std::path::PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("spectral_snapshot_") && name.ends_with(".json") {
                    let ts_str = name
                        .strip_prefix("spectral_snapshot_")?
                        .strip_suffix(".json")?;
                    let ts: i64 = ts_str.parse().ok()?;
                    Some((ts, e.path()))
                } else {
                    None
                }
            })
            .collect();
        snapshots.sort_by_key(|(ts, _)| -*ts);

        let prev_snap = match snapshots.first() {
            Some((_, path)) => {
                let content = match std::fs::read_to_string(path) {
                    Ok(c) => c,
                    Err(_) => return "{}".to_string(),
                };
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(v) => v,
                    Err(_) => return "{}".to_string(),
                }
            }
            None => return serde_json::json!({"error": "no previous snapshot"}).to_string(),
        };

        let prev_ts = prev_snap["ts_ms"].as_i64().unwrap_or(0);
        let prev_stats = &prev_snap["stats"];

        // Current stats
        let current_json = self.spectral_stats_by_realm();
        let current: serde_json::Value =
            serde_json::from_str(&current_json).unwrap_or_default();

        let mut drifts: Vec<serde_json::Value> = Vec::new();

        for section in &["by_realm", "by_kind"] {
            let prev_arr = prev_stats[section].as_array();
            let curr_arr = current[section].as_array();
            if let (Some(prev_items), Some(curr_items)) = (prev_arr, curr_arr) {
                let prev_map: std::collections::HashMap<&str, &serde_json::Value> = prev_items
                    .iter()
                    .filter_map(|v| v["group"].as_str().map(|g| (g, v)))
                    .collect();
                for curr in curr_items {
                    let group = match curr["group"].as_str() {
                        Some(g) => g,
                        None => continue,
                    };
                    if let Some(prev) = prev_map.get(group) {
                        let iso_prev = prev["isotropy"].as_f64().unwrap_or(0.0);
                        let iso_curr = curr["isotropy"].as_f64().unwrap_or(0.0);
                        let cos_prev = prev["mean_cosine_sim"].as_f64().unwrap_or(0.0);
                        let cos_curr = curr["mean_cosine_sim"].as_f64().unwrap_or(0.0);
                        let iso_delta = iso_curr - iso_prev;
                        let cos_delta = cos_curr - cos_prev;
                        if iso_delta.abs() > 0.005 || cos_delta.abs() > 0.005 {
                            drifts.push(serde_json::json!({
                                "section": section,
                                "group": group,
                                "isotropy_delta": (iso_delta * 1000.0).round() / 1000.0,
                                "cosine_delta": (cos_delta * 1000.0).round() / 1000.0,
                                "isotropy_now": iso_curr,
                                "cosine_now": cos_curr,
                            }));
                        }
                    }
                }
            }
        }

        let hours_since = (now_ms() - prev_ts) as f64 / 3_600_000.0;
        serde_json::to_string(&serde_json::json!({
            "snapshot_age_hours": (hours_since * 10.0).round() / 10.0,
            "drifts": drifts,
            "total_drifted": drifts.len(),
        }))
        .unwrap_or_else(|_| "{}".to_string())
    }

    /// Encode a memory's embedding into sparse codes and index into the cortical index.
    /// Persists via UpdateSparseCode op.
    /// Embedding for a memory. The semantic index is the single in-RAM home
    /// (payload copies are cleared at write/replay since v2.7.0 — keeping
    /// them duplicated ~600MB of RSS); the payload copy survives only for
    /// memories the index does not hold (unindexed / foreign-dim).
    pub(crate) fn embedding_of(&self, memory_id: MemoryId) -> Option<Vec<f32>> {
        // Lock order: payloads before semantic_idx.
        let payloads = self.payloads.read();
        let idx = self.semantic_idx.read();
        if let Some(e) = idx.get_embedding(memory_id) {
            if !e.is_empty() {
                return Some(e.to_vec());
            }
        }
        payloads.get(&memory_id).and_then(|p| {
            if p.embedding.is_empty() {
                None
            } else {
                Some(p.embedding.clone())
            }
        })
    }

    pub fn encode_memory(&self, memory_id: MemoryId) -> Result<()> {
        if self.ablations.disabled("cortical_idx") || self.ablations.disabled("sparse_encoder") { return Ok(()); }
        let embedding = self.embedding_of(memory_id);
        let Some(embedding) = embedding else {
            return Ok(());
        };
        if embedding.len() != EMBED_DIM {
            return Ok(());
        }

        let encoder = self.sparse_encoder.read();
        let code = encoder.encode(&embedding);
        if code.is_empty() {
            drop(encoder);
            self.encode_skip.write().insert(memory_id);
            return Ok(());
        }

        // Compute surprise (reconstruction error) before updating encoder
        let surprise = encoder.reconstruction_error(&embedding, &code);
        drop(encoder);

        // Update surprise in memory state and plasticity learner
        {
            let mut states = self.states.write();
            if let Some(state) = states.get_mut(&memory_id) {
                state.surprise = surprise;
            }
            let mut learners = self.learners.write();
            learners.plasticity.update_surprise(memory_id, surprise);
        }

        // FEP-derived update (accuracy + complexity + orthogonalization)
        self.sparse_encoder.write().update(&embedding, &code);

        let ts_ms = now_ms();
        let op = Op::UpdateSparseCode(UpdateSparseCodeOp {
            memory_id,
            feature_ids: code.feature_ids.clone(),
            activations: code.activations.clone(),
            ts_ms,
        });
        self.log.write().append(&op)?;

        // Index in cortical index
        let (strength, state_arousal) = {
            let st = self.states.read();
            let s = st.get(&memory_id);
            (s.map(|s| s.strength).unwrap_or(0.5), s.map(|s| s.affect_arousal).unwrap_or(0.0))
        };
        let kind = self
            .payloads
            .read()
            .get(&memory_id)
            .map(|p| p.kind.clone())
            .unwrap_or_default();
        let authored_at = self
            .payloads
            .read()
            .get(&memory_id)
            .map(|p| p.authored_at_ms)
            .unwrap_or(ts_ms);
        let affect_arousal = if state_arousal > 0.01 {
            state_arousal
        } else if kind.to_ascii_lowercase().contains("correction") {
            0.85
        } else {
            match kind.as_str() {
                "wisdom" => 0.70,
                "insight" => 0.50,
                "signal" => 0.30,
                "episode" | "observation" => 0.15,
                _ => 0.25,
            }
        };
        self.cortical_idx
            .write()
            .index_with_affect(memory_id, &code, strength, authored_at, &kind, affect_arousal);

        Ok(())
    }

    /// Save the cortical index + encoder + prototype state to a binary snapshot.
    /// After this, on next open the snapshot covers all UpdateSparseCode ops
    /// up to the current log position, so those ops can be skipped in replay.
    pub fn save_snapshot(&self) -> Result<()> {
        let _touch_drain = self.touch_drain.lock();
        self.drain_pending_touches_locked()?;
        self.drain_pending_recall_effects()?;
        self.sync_wal()?;
        let seqno = self.log.read().last_seqno();
        let path = self
            .data_dir
            .join(format!("cortex.{:08x}.snapshot", self.instance_id));
        self.cortical_idx.persisted_read().save_snapshot(&path, seqno)
    }

    /// Save full in-memory state to a binary snapshot (chitta.snapshot).
    /// After this, on next open only ops after snapshot_seqno need to be replayed.
    /// Budgeted background competitive-weight refresh (THEORY.md §8 Phase 3:
    /// consolidation as deliberate merge). Refreshes up to `budget` memories
    /// whose last refresh is older than the configured interval, using the
    /// same reservation discipline as the recall-path refresh — so recalls
    /// (whose own budget is small) almost never pay refresh cost themselves.
    /// Called from the subconscious sleep-consolidation cycle. Returns the
    /// number refreshed.
    pub fn cw_refresh_sweep(&self, budget: usize) -> usize {
        if budget == 0 {
            return 0;
        }
        let now = now_ms();
        let (dedup_upper, interval_ms) = {
            let pipeline = self.scoring_pipeline.read();
            (
                pipeline.config.dedup_cosine_upper,
                pipeline.config.cw_refresh_interval_ms,
            )
        };
        // Collect stale candidates + embeddings under read locks (struct order).
        let candidates: Vec<(MemoryId, Vec<f32>)> = {
            let states_r = self.states.read();
            let idx = self.semantic_idx.read();
            let inflight_r = self.cw_refresh_inflight.read();
            states_r
                .iter()
                .filter(|(_, st)| !st.deleted && now - st.last_cw_refresh_ms >= interval_ms)
                .filter(|(id, _)| {
                    inflight_r
                        .get(id)
                        .map(|&ts| now - ts >= interval_ms)
                        .unwrap_or(true)
                })
                .filter_map(|(id, _)| idx.get_embedding(*id).map(|e| (*id, e.to_vec())))
                .take(budget)
                .collect()
        };
        // Atomic re-check + reserve (same as the recall path).
        let candidates: Vec<(MemoryId, Vec<f32>)> = if candidates.is_empty() {
            return 0;
        } else {
            let mut inflight_w = self.cw_refresh_inflight.write();
            inflight_w.retain(|_, ts| now - *ts < interval_ms);
            candidates
                .into_iter()
                .filter(|(id, _)| {
                    if inflight_w.contains_key(id) {
                        return false;
                    }
                    inflight_w.insert(*id, now);
                    true
                })
                .collect()
        };
        let cw_updates: Vec<(MemoryId, f32)> = {
            let idx = self.semantic_idx.read();
            candidates
                .iter()
                .filter_map(|(memory_id, emb)| {
                    let neighbors = idx.search(emb, 9, None, None);
                    if neighbors.len() <= 1 {
                        return None;
                    }
                    let mut cos_sum = 0.0f32;
                    let mut n = 0u32;
                    for nb in &neighbors {
                        if nb.memory_id == *memory_id {
                            continue;
                        }
                        if nb.cosine_similarity >= dedup_upper {
                            continue;
                        }
                        cos_sum += nb.cosine_similarity;
                        n += 1;
                    }
                    if n > 0 {
                        Some((*memory_id, cos_sum / n as f32))
                    } else {
                        None
                    }
                })
                .collect()
        };
        let refreshed = candidates.len();
        let cw_by_id: std::collections::HashMap<MemoryId, f32> = cw_updates.into_iter().collect();
        {
            let mut states_w = self.states.write();
            for (memory_id, _) in &candidates {
                if let Some(st) = states_w.get_mut(memory_id) {
                    if let Some(&cw) = cw_by_id.get(memory_id) {
                        st.competitive_weight = cw;
                    }
                    st.last_cw_refresh_ms = now;
                }
            }
        }
        let mut inflight_w = self.cw_refresh_inflight.write();
        for (id, _) in &candidates {
            inflight_w.remove(id);
        }
        refreshed
    }

    pub fn save_full_snapshot(&self) -> Result<()> {
        self.save_full_snapshot_certified().map(|_| ())
    }

    fn save_full_snapshot_certified(&self) -> Result<usize> {
        let mut pruned = 0;
        use crate::snapshot::FullSnapshot;
        let _touch_drain = self.touch_drain.lock();
        self.drain_pending_touches_locked()?;
        self.drain_pending_recall_effects()?;
        self.sync_wal()?;
        let seqno = self.log.read().last_seqno();
        let snapshot_coverage = self.wal_coverage.read().clone();
        // Compact the delta into the base under a BRIEF write so the snapshot clone below
        // is canonical (delta empty). Decoupled from the sidecar disk writes, which now run
        // lock-free off the clone (see the sidecar block further down). Previously the merge
        // AND all six ~600MB sidecar writes were held under a single semantic_idx.write(),
        // blocking recall (which needs semantic_idx.read()) for the entire disk-write window
        // — the sleep-consolidation stall that forced --no-hygiene.
        {
            let mut idx = self.semantic_idx.write();
            if idx.delta_needs_merge() {
                idx.merge_delta_into_base();
            }
        }
        // Read BEFORE the payloads clone: any content mutation racing this
        // save then differs from the stored value and forces the next .pld
        // rewrite (see the field doc on pld_mutations).
        let pld_mutations_at_clone = self
            .pld_mutations
            .load(std::sync::atomic::Ordering::Relaxed);
        // Each clone in its OWN statement: a struct-literal initializer's
        // temporary guard lives until the end of the whole statement, so the
        // previous form held ~20 read guards at once — and read `states`
        // TWICE in one statement (cw_refresh_ts), which self-deadlocks when a
        // writer queues between the two acquisitions (parking_lot writer
        // preference blocks same-thread reacquisition). Production deadlock
        // 2026-06-11, caught by the deadlock-detection build.
        let payloads = {
            // Phase 1: collect embedded IDs without holding payloads or states locks.
            // Releasing semantic_idx.read() before acquiring payloads.read() prevents
            // parking_lot write-preference from cascading across all three locks at once.
            // ceiling: payloads.read() is still held during Phase 2 clone — put_memory
            // writes will still queue, but the hold time is ~10x shorter (no embedding alloc).
            // upgrade: Arc<MemoryPayload> in the store HashMap would make clone O(1).
            let embedded_ids: std::collections::HashSet<MemoryId> = {
                let idx = self.semantic_idx.read();
                idx.all_ids().collect()
            };

            // Phase 2: clone payloads under payloads+states only (2 locks, not 3).
            // Build MemoryPayload literals that skip embedding.clone() for HNSW-indexed
            // memories: saves ~12KB × N transient Vec<f32> allocs inside the lock.
            let payloads_r = self.payloads.read();
            let states_r = self.states.read();
            payloads_r
                .iter()
                .filter(|(id, _)| !states_r.get(id).is_some_and(|s| s.deleted))
                .map(|(id, p)| {
                    let q = MemoryPayload {
                        embedding: if embedded_ids.contains(id) {
                            Vec::new()
                        } else {
                            p.embedding.clone()
                        },
                        memory_id: p.memory_id,
                        version: p.version,
                        chunk_hash: p.chunk_hash,
                        created_at_ms: p.created_at_ms,
                        authored_at_ms: p.authored_at_ms,
                        kind: p.kind.clone(),
                        realm: p.realm.clone(),
                        content: p.content.clone(),
                        embedding_model: p.embedding_model.clone(),
                        artifact_refs: p.artifact_refs.clone(),
                        source_session: p.source_session.clone(),
                        source_tool: p.source_tool.clone(),
                        harness: p.harness.clone(),
                        provenance: p.provenance.clone(),
                        candidate: p.candidate,
                        embedding_model_id: p.embedding_model_id.clone(),
                        embedding_dim: p.embedding_dim,
                    };
                    (*id, q)
                })
                .collect()
        };
        let (states, cw_refresh_ts, utility_posteriors) = {
            let states_r = self.states.read();
            let cw: std::collections::HashMap<MemoryId, i64> = states_r
                .iter()
                .filter(|(_, st)| !st.deleted && st.last_cw_refresh_ms > 0)
                .map(|(&id, st)| (id, st.last_cw_refresh_ms))
                .collect();
            // Only memories that actually carry an observation — an untouched
            // (1, 1) prior is what a missing entry already means.
            let util: std::collections::HashMap<MemoryId, (f32, f32)> = states_r
                .iter()
                .filter(|(_, st)| !st.deleted && (st.utility_alpha > 1.0 || st.utility_beta > 1.0))
                .map(|(&id, st)| (id, (st.utility_alpha, st.utility_beta)))
                .collect();
            // Compact out deleted memories — they must not appear in the snapshot
            // so that snapshot resurrection cannot undo a forget().
            let live: std::collections::HashMap<MemoryId, _> = states_r
                .iter()
                .filter(|(_, st)| !st.deleted)
                .map(|(id, st)| (*id, st.clone()))
                .collect();
            (live, cw, util)
        };
        let assoc_edges = self.assoc_edges.read().clone();
        let artifacts = self.artifacts.read().clone();
        let artifact_paths = self.artifact_paths.read().clone();
        let time_idx = self.time_idx.read().clone();
        let keyword_idx = self.keyword_idx.read().clone();
        let artifact_idx = self.artifact_idx.read().clone();
        let triplet_store = self.triplet_store.read().clone();
        let symbol_idx = self.symbol_idx.read().clone();
        let call_graph = self.call_graph.read().clone();
        let code_files = self.code_files.read().clone();
        let semantic_idx = self.semantic_idx.read().clone();
        let coactivation_stats = {
            let mut cs = self.coactivation_stats.read().clone();
            let before = cs.len();
            let removed = crate::field::prune_coactivation_stats(&mut cs, 20);
            eprintln!("[chitta-field] coactivation_stats: {} pairs before prune, {} removed (cap=20/memory)", before, removed);
            cs
        };
        let ack_scores = self.ack_scores.read().clone();
        let correction_states = self.triplet_store.read().correction_states.clone();
        let event_tape = self.event_tape.persisted_read().clone();
        let decision_tape = self.decision_tape.persisted_read().clone();
        let turiya_monitor = self.turiya_monitor.persisted_read().clone();
        let observer_state = self.observer_state.persisted_read().clone();
        let interaction_ledger = self.interaction_ledger.persisted_read().clone();
        let predicate_store = self.predicate_store.persisted_read().clone();
        let recall_provenance = self.recall_provenance.read().clone();
        let mut snap = FullSnapshot {
            snapshot_seqno: seqno,
            payloads,
            states,
            assoc_edges,
            artifacts,
            artifact_paths,
            time_idx,
            keyword_idx,
            artifact_idx,
            triplet_store,
            symbol_idx,
            call_graph,
            code_files,
            semantic_idx,
            coactivation_stats,
            ack_scores,
            correction_states,
            event_tape,
            decision_tape,
            turiya_monitor,
            observer_state,
            interaction_ledger,
            predicate_store,
            recall_provenance,
            cw_refresh_ts,
            utility_posteriors,
            ledger_session_events: self.msg_registry.persisted_read().ledger_session_events(),
        };
        let path = self
            .data_dir
            .join(format!("chitta.{:08x}.snapshot", self.instance_id));
        // Write embedding and binary-code sidecars from the live index (before clearing clone).
        let emb_path   = path.with_extension("emb");
        let hdc_path   = path.with_extension("hdc");
        let bin_path   = path.with_extension("bin");
        let mu_path    = path.with_extension("mu");
        let hnsw_path       = path.with_extension("hnsw");
        let delta_path      = path.with_extension("delta.hnsw");
        let realm_hnsw_path = path.with_extension("realm_hnsw");
        let pld_path   = path.with_extension("pld");
        let sup_path   = path.with_extension("sup.json");
        let shdr_path  = path.with_extension("shdr");
        // Store-identity sidecar (PR3): records the vector space (model/dim/text-format) +
        // lineage so snapshot selection and WAL replay can fence foreign-dim/model data.
        {
            let hdr = crate::snapshot::StoreHeader::current(self.lineage_epoch, self.writer_uuid);
            if let Err(e) = hdr.save(&shdr_path) {
                eprintln!("[chitta-field] WARNING: .shdr sidecar save failed: {e}");
            }
        }
        {
            // Embedding/code sidecars saved from the cloned index (snap.semantic_idx) with
            // NO semantic_idx lock held — the six ~600MB disk writes must not block recall,
            // which needs semantic_idx.read(). The delta was merged into base under a brief
            // write above, so this clone is canonical (delta empty). clear_embeddings()
            // below runs after this block, so the clone still carries its vectors here.
            //
            // Dirty-skip: if the index hasn't mutated since this instance's
            // last successful sidecar write, the files on disk are already
            // current — skip the ~800MB of rewrites (THEORY.md §8 Phase 2).
            // Promote delta→base on the clone so save_hnsw() serialises the full
            // graph.  The live index is untouched; only the snapshot clone is swapped.
            snap.semantic_idx.promote_delta_to_base_if_empty();
            let idx = &snap.semantic_idx;
            let idx_mutations = idx.mutation_count();
            let last = self
                .idx_sidecars_saved_at
                .load(std::sync::atomic::Ordering::Relaxed);
            // Existence guard on .emb only: the other sidecars are written
            // conditionally (empty HNSW/centroid produce no file), so their
            // absence matches the previous save rather than invalidating it.
            // Exception: if the .hnsw on disk is a 9-byte stub (empty base, nodes
            // are all in delta) the promote above has now swapped them — force a
            // rewrite even if the mutation counter hasn't changed.
            // Stub guard: only relevant when the HNSW graph has actual nodes
            // (i.e. total_embedding_count >= HNSW_THRESHOLD and promote ran).
            // Below the threshold the HNSW is intentionally not built and
            // save_hnsw() legitimately writes a ≤9-byte stub — that is correct
            // and must not force a sidecar rewrite.
            let hnsw_stub = idx.hnsw_len() > 0
                && hnsw_path.metadata().map(|m| m.len() <= 9).unwrap_or(false);
            let clean = last == idx_mutations && emb_path.exists() && !hnsw_stub;
            if clean {
                eprintln!("[chitta-field] index sidecars unchanged since last save — skipping rewrite");
            } else {
                let _ = idx.save_embeddings_sidecar(&emb_path);
                let _ = idx.save_binary_sidecar(&bin_path);
                let _ = idx.save_centroid_sidecar(&mu_path);
                // HNSW sidecars: skipped when CHITTA_NO_HNSW_SIDECAR=1. Below
                // flat_scan_max the graph is never consulted, so re-serialising it
                // each snapshot is dead I/O; the flat-scan path is unaffected.
                if !crate::hnsw::skip_hnsw_sidecar() {
                    let _ = idx.save_hnsw(&hnsw_path);
                    let _ = idx.save_delta_hnsw(&delta_path);
                    let _ = idx.save_realm_hnsw(&realm_hnsw_path);
                }
                self.idx_sidecars_saved_at
                    .store(idx_mutations, std::sync::atomic::Ordering::Relaxed);
            }
            // Optional startup caches follow the same dirty-skip discipline.
            // The clone still owns the embeddings here; .emb was saved BEFORE
            // this applies the next loader's normalization. No store guards held.
            if !clean || ["lsh", "turbo", "turbo.meta"].iter().any(|ext| !path.with_extension(ext).exists()) {
                snap.semantic_idx.prepare_startup_caches(&path);
            }
        }
        // Tape identity includes same-length edits and dictionary changes. The
        // helper skips reconstruction when unchanged; derived organs stay optional.
        let _ = crate::startup_cache::load_or_rebuild_organs(
            &snap.event_tape, Some(&path.with_extension("organs")));
        // Save HDC sidecar — avoids tokenize+encode rebuild on next startup.
        // Dirty-skipped when the store hasn't mutated since the last write
        // (the sidecar is a lossy cache; a same-content file stays valid).
        {
            let hdc = self.hdc_idx.persisted_read();
            let count = hdc.mutation_count();
            let last = self
                .hdc_sidecar_saved_at
                .load(std::sync::atomic::Ordering::Relaxed);
            if last == count && hdc_path.exists() {
                eprintln!("[chitta-field] hdc sidecar unchanged since last save — skipping rewrite");
            } else {
                let n = hdc.save_sidecar(&hdc_path)
                    .map(|n| n.to_string())
                    .unwrap_or_else(|e| format!("err:{e}"));
                self.hdc_sidecar_saved_at
                    .store(count, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[chitta-field] hdc sidecar: {} memories written to {:?}", n, hdc_path);
            }
        }
        // Save payload content sidecar (.pld) before clearing from bincode. If this
        // fails we MUST NOT strip content from the bincode body — otherwise the content
        // would exist in neither place (silent total content loss). Abort the snapshot.
        let mut snap = snap;
        // .pld dirty-skip: content is the ONE thing with no fallback copy, so
        // skip only when no content mutation happened since the last write by
        // this instance AND the existing file is plausibly intact.
        let pld_clean = self
            .pld_saved_at
            .load(std::sync::atomic::Ordering::Relaxed)
            == pld_mutations_at_clone
            && std::fs::metadata(&pld_path).map(|m| m.len() >= 16).unwrap_or(false);
        if pld_clean {
            eprintln!("[chitta-field] .pld sidecar unchanged since last save — skipping rewrite");
        } else {
            FullSnapshot::save_payload_sidecar(&pld_path, &snap.payloads)
                .map_err(|e| FieldError::Manifest(format!(
                    "payload (.pld) sidecar save failed, aborting snapshot to avoid content loss: {e}"
                )))?;
            self.pld_saved_at
                .store(pld_mutations_at_clone, std::sync::atomic::Ordering::Relaxed);
        }
        // Stage B: persist retrieval surfaces to the .rsf sidecar (never bincode).
        // Best-effort: content is the authoritative copy, so a failed .rsf just means
        // recall falls back to embedding content — no data loss, unlike .pld.
        {
            let surfaces = self.retrieval_surfaces.read();
            if !surfaces.is_empty() {
                let rsf_path = path.with_extension("rsf");
                if let Err(e) = FullSnapshot::save_retrieval_surface_sidecar(&rsf_path, &surfaces) {
                    eprintln!("[chitta-field] .rsf sidecar save failed (non-fatal): {e}");
                }
            }
        }
        // Strip content and embeddings from bincode (live in sidecars now).
        for payload in snap.payloads.values_mut() {
            payload.content.clear();
            payload.content.shrink_to_fit();
        }
        snap.semantic_idx.clear_embeddings();
        let _ = snap.triplet_store.save_supersession_sidecar(&sup_path);
        snap.triplet_store.clean_for_load();
        snap.triplet_store.clear_indexes_for_save();
        // Diagnostic: per-field serialized sizes to identify snapshot bloat.
        {
            let sz = |v: u64| format!("{:.1}MB", v as f64 / 1_000_000.0);
            eprintln!("[size] payloads:          {}", sz(bincode::serialized_size(&snap.payloads).unwrap_or(0)));
            eprintln!("[size] states:            {}", sz(bincode::serialized_size(&snap.states).unwrap_or(0)));
            eprintln!("[size] assoc_edges:       {}", sz(bincode::serialized_size(&snap.assoc_edges).unwrap_or(0)));
            eprintln!("[size] artifacts:         {}", sz(bincode::serialized_size(&snap.artifacts).unwrap_or(0)));
            eprintln!("[size] time_idx:          {}", sz(bincode::serialized_size(&snap.time_idx).unwrap_or(0)));
            eprintln!("[size] keyword_idx:       {}", sz(bincode::serialized_size(&snap.keyword_idx).unwrap_or(0)));
            eprintln!("[size] triplet_store:     {}", sz(bincode::serialized_size(&snap.triplet_store).unwrap_or(0)));
            eprintln!("[size] symbol_idx:        {}", sz(bincode::serialized_size(&snap.symbol_idx).unwrap_or(0)));
            eprintln!("[size] call_graph:        {}", sz(bincode::serialized_size(&snap.call_graph).unwrap_or(0)));
            eprintln!("[size] code_files:        {}", sz(bincode::serialized_size(&snap.code_files).unwrap_or(0)));
            eprintln!("[size] semantic_idx:      {}", sz(bincode::serialized_size(&snap.semantic_idx).unwrap_or(0)));
            eprintln!("[size] coactivation:      {}", sz(bincode::serialized_size(&snap.coactivation_stats).unwrap_or(0)));
            eprintln!("[size] ack_scores:        {}", sz(bincode::serialized_size(&snap.ack_scores).unwrap_or(0)));
            eprintln!("[size] artifact_paths:    {}", sz(bincode::serialized_size(&snap.artifact_paths).unwrap_or(0)));
            eprintln!("[size] artifact_idx:      {}", sz(bincode::serialized_size(&snap.artifact_idx).unwrap_or(0)));
        }
        snap.save(&path)?;
        // Commit record: the manifest ties the snapshot + sidecars into one
        // committed family (open() prefers a validated family over fence-based
        // selection). Written LAST — a crash anywhere above leaves the previous
        // generation's manifest pointing at the previous intact family.
        {
            use crate::manifest::{CheckpointSet, FileRef, Manifest};
            let file_ref = |p: &std::path::Path| -> Option<FileRef> {
                let name = p.file_name()?.to_string_lossy().into_owned();
                let size_bytes = std::fs::metadata(p).ok()?.len();
                Some(FileRef { name, size_bytes })
            };
            if let Some(snapshot_ref) = file_ref(&path) {
                let mut manifest = Manifest::load(&self.data_dir)
                    .ok()
                    .flatten()
                    .unwrap_or_else(|| {
                        Manifest::new_empty(
                            crate::ops::EMBED_MODEL_ID,
                            crate::ops::EMBED_DIM as u16,
                        )
                    });
                manifest.generation += 1;
                manifest.last_seqno = seqno;
                // Per-writer coverage vector (THEORY.md §4): everything the
                // in-memory state contains (open replay ⊔ sync_foreign) plus
                // our own ops up to this save.
                let covered: std::collections::BTreeMap<String, u64> = {
                    let mut cov = snapshot_coverage.clone();
                    let own = cov.entry(self.instance_id).or_insert(0);
                    if seqno > *own { *own = seqno; }
                    cov.iter().map(|(i, s)| (format!("{:08x}", i), *s)).collect()
                };
                // Only files that actually exist are recorded (e.g. the .sup
                // sidecar save is best-effort) — validation checks what the
                // commit promised, nothing more.
                let cortical_path = path.with_extension("cortex");
                self.cortical_idx.persisted_read().save_snapshot(&cortical_path, seqno)?;
                // Certificates may delete the WAL: cortex data and its rename
                // must be durable before the manifest makes that promise.
                std::fs::File::open(&cortical_path)?.sync_all()?;
                std::fs::File::open(&self.data_dir)?.sync_all()?;
                let cortical = file_ref(&cortical_path).ok_or_else(||
                    FieldError::Manifest("missing committed cortical snapshot".into()))?;
                let lineage = crate::snapshot::StoreHeader::compiled_vector_space_id();
                let segments = self.log.read().segment_inventory();
                manifest.segments = segments.clone();
                let family = CheckpointSet {
                    segments,
                    cortical: Some(cortical),
                    cortical_covered: covered.clone(),
                    vector_space_id: Some(lineage),
                    snapshot: snapshot_ref,
                    sidecars: [
                        &emb_path, &hdc_path, &bin_path, &mu_path, &hnsw_path,
                        &delta_path, &realm_hnsw_path, &pld_path, &sup_path, &shdr_path, &cortical_path,
                    ]
                    .iter()
                    .filter_map(|p| file_ref(p))
                    .collect(),
                    snapshot_seqno: seqno,
                    covered,
                };
                manifest
                    .families
                    .insert(format!("{:08x}", self.instance_id), family.clone());
                manifest.checkpoints = Some(family);
                if let Err(e) = manifest.save(&self.data_dir) {
                    eprintln!(
                        "[chitta-field] WARNING: manifest commit failed (snapshot itself is durable): {e}"
                    );
                } else {
                    pruned = self.prune_certified_wal()?;
                }
            }
        }
        // Prune old families ONLY after the new snapshot + .pld are durably written
        // (save() fsyncs the file and parent dir; save_payload_sidecar fsyncs the .pld).
        // Pruning before durability would delete the fallback the new snapshot replaces.
        prune_old_snapshots(&self.data_dir, 2);
        // Ghost janitor: dead-instance residue + resurrection accounting
        // (7-day age gate protects live peers' seen_offsets).
        janitor_sweep(&self.data_dir, self.instance_id, 7 * 86_400);
        Ok(pruned)
    }


    /// Compact WAL: save full snapshot then delete WAL segments covered by it.
    /// Coverage is per-writer (THEORY.md §4) — see prune_covered_segments.
    /// This bounds WAL growth and speeds up startup replay.
    pub fn compact_wal(&self) -> Result<usize> {
        let count = {
            let states = self.states.read();
            states.values().filter(|s| !s.deleted).count()
        };
        if count < 100 {
            return Err(FieldError::Other(format!(
                "refusing compact_wal on near-empty store ({} live memories, minimum 100)", count
            )));
        }
        self.save_full_snapshot_certified()
    }

    fn prune_certified_wal(&self) -> Result<usize> {
        // Safe pruning rule (THEORY.md §4): a segment of instance i may be
        // deleted iff our coverage vector dominates it — i.e. every op in it
        // is provably contained in the snapshot we just committed. The old
        // scalar rule (first_seqno < snapshot_seqno) compared seqnos across
        // writers, which can delete a concurrent writer's UNCOVERED ops:
        // overlapping seqno ranges make a foreign segment look covered.
        let manifest = crate::manifest::Manifest::load(&self.data_dir)?
            .ok_or_else(|| FieldError::Manifest("WAL pruning requires committed coverage".into()))?;
        let family = manifest.families.get(&format!("{:08x}", self.instance_id))
            .ok_or_else(|| FieldError::Manifest("WAL pruning requires writer snapshot family".into()))?;
        for file in std::iter::once(&family.snapshot).chain(family.sidecars.iter()) {
            if std::fs::metadata(self.data_dir.join(&file.name))?.len() != file.size_bytes {
                return Err(FieldError::Manifest("WAL pruning family failed validation".into()));
            }
        }
        if family.vector_space_id != Some(crate::snapshot::StoreHeader::compiled_vector_space_id()) {
            return Ok(0);
        }
        let Some(cortical) = &family.cortical else { return Ok(0) };
        if std::fs::metadata(self.data_dir.join(&cortical.name))?.len() != cortical.size_bytes {
            return Err(FieldError::Manifest("WAL pruning cortical family failed validation".into()));
        }
        let full = crate::wal_certificate::vector(&family.covered);
        let cortical = crate::wal_certificate::vector(&family.cortical_covered);
        let log = self.log.read();
        let mut deleted = 0;
        for entry in &family.segments {
            let path = self.data_dir.join(&entry.path);
            if crate::wal_certificate::sealed_candidate(&path, log.writer_path())
                && crate::wal_certificate::covered_segment(&self.data_dir, entry, &full, &cortical, log.writer_path())
                && audited_remove(&path, "wal-family-certified-full-and-cortical-covered").is_ok() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Count WAL segment files in the segments/ directory.
    pub fn wal_segment_count(&self) -> usize {
        let seg_dir = self.data_dir.join("segments");
        std::fs::read_dir(&seg_dir)
            .map(|rd| rd
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|x| x == "seg"))
                .count())
            .unwrap_or(0)
    }

    /// Compact WAL if segment count exceeds `threshold`, with a 1-hour cooldown.
    /// Returns Ok(true) if compaction ran, Ok(false) if skipped (under threshold or cooldown).
    pub fn maybe_compact_wal(&self, threshold: usize) -> Result<bool> {
        if self.wal_segment_count() <= threshold { return Ok(false); }
        let now = now_ms();
        let last = self.last_compact_ms.load(std::sync::atomic::Ordering::Relaxed);
        if now - last < 3_600_000 { return Ok(false); }
        self.last_compact_ms.store(now, std::sync::atomic::Ordering::Relaxed);
        self.compact_wal()?;
        eprintln!("[store] maybe_compact_wal: compacted (segments > {})", threshold);
        Ok(true)
    }

    /// Prune episode memories: delete those older than `max_age_days` with strength < 0.3,
    /// then cap total episode count to `max_count` by removing the oldest.
    pub fn prune_episodes(&self, max_age_days: u64, max_count: usize) -> Result<usize> {
        let cutoff_ms = now_ms() - (max_age_days as i64) * 86_400_000;
        let mut episodes: Vec<(i64, MemoryId)> = {
            let payloads = self.payloads.read();
            payloads.iter()
                .filter(|(_, p)| p.kind == "episode")
                .map(|(id, p)| (p.created_at_ms, *id))
                .collect()
        };

        let mut deleted = 0usize;
        for &(created_ms, id) in &episodes {
            if created_ms < cutoff_ms {
                if let Ok(state) = self.get_state(id) {
                    if state.strength < 0.3 {
                        let _ = self.forget(id);
                        deleted += 1;
                    }
                }
            }
        }

        episodes.retain(|(_, id)| self.get_memory(*id).is_ok());
        if episodes.len() > max_count {
            episodes.sort_unstable_by_key(|(ts, _)| *ts);
            let target = (max_count as f64 * 0.8) as usize;
            let to_delete = episodes.len().saturating_sub(target);
            for &(_, id) in &episodes[..to_delete] {
                let _ = self.forget(id);
                deleted += 1;
            }
        }
        if deleted > 0 {
            eprintln!("[store] prune_episodes: deleted {} episode memories", deleted);
        }
        Ok(deleted)
    }

    /// Promote staged memories that have been recalled (access_count >= 1),
    /// and prune staged memories older than 7 days that were never recalled.
    /// Returns (promoted, pruned).
    pub fn promote_staged_memories(&self) -> Result<(usize, usize)> {
        let cutoff_ms = now_ms() - 7 * 86_400_000;
        let candidates: Vec<(MemoryId, i64, u32)> = {
            let states = self.states.read();
            states.values()
                .filter(|s| !s.deleted && s.staged)
                .map(|s| (s.memory_id, s.created_at_ms, s.access_count))
                .collect()
        };

        let mut promoted = 0usize;
        let mut pruned   = 0usize;
        let ts = now_ms();
        for (id, created_ms, access_count) in candidates {
            if access_count >= 1 {
                let delta = crate::ops::StateDeltaOp {
                    memory_id: id,
                    strength_delta: None,
                    confidence_delta: None,
                    decay_rate: None,
                    touch: false,
                    pin: None,
                    op_ts_ms: ts,
                    status: None,
                    epistemic_status: None,
                    staged: Some(false),
                    invalidated_by: None,
                };
                let _ = self.log.write().append(&crate::ops::Op::UpdateState(delta.clone()));
                if let Some(s) = self.states.write().get_mut(&id) {
                    s.staged = false;
                }
                promoted += 1;
            } else if created_ms < cutoff_ms {
                let _ = self.forget(id);
                pruned += 1;
            }
        }
        if promoted > 0 || pruned > 0 {
            eprintln!("[store] write_gate: promoted={} pruned={}", promoted, pruned);
        }
        Ok((promoted, pruned))
    }

    /// Run a single tier demotion pass over all memories.
    /// Returns `(demoted_count, deleted_count)`.
    ///
    /// Tiers: 0=L1 (hippocampus), 1=L2 (cortex), 2=L3 (archive), then delete.
    /// Uses `access_count` as rehearsal proxy and `strength` as utility proxy.
    pub fn run_demotion_pass(&self, now_ms: i64) -> Result<(usize, usize)> {
        const L1_TO_L2_AGE_MS: i64 = 7 * 24 * 3600 * 1000;
        const L1_TO_L2_LAST_ACCESS_MS: i64 = 2 * 24 * 3600 * 1000;
        const L1_TO_L2_MAX_STRENGTH: f32 = 0.80;

        const L2_TO_L3_AGE_BASE_MS: i64 = 45 * 24 * 3600 * 1000;
        const L2_TO_L3_REHEARSAL_BONUS_MS: i64 = 7 * 24 * 3600 * 1000;
        const L2_TO_L3_LAST_ACCESS_MS: i64 = 14 * 24 * 3600 * 1000;
        const L2_TO_L3_MAX_STRENGTH: f32 = 0.50;

        const L3_DELETE_AGE_BASE_MS: i64 = 365 * 24 * 3600 * 1000;
        const L3_DELETE_REHEARSAL_BONUS_MS: i64 = 30 * 24 * 3600 * 1000;
        const L3_DELETE_LAST_ACCESS_MS: i64 = 120 * 24 * 3600 * 1000;
        const L3_DELETE_MAX_STRENGTH: f32 = 0.12;
        const L3_DELETE_MAX_UTILITY: f32 = 0.80;

        let mut to_demote: Vec<(MemoryId, u8)> = Vec::new();
        let mut to_delete: Vec<MemoryId> = Vec::new();

        {
            let states = self.states.read();
            for (&memory_id, state) in states.iter() {
                if state.deleted || state.pinned {
                    continue;
                }

                // Strength >= L3_DELETE_MAX_UTILITY means never delete
                let age_ms = now_ms - state.created_at_ms;
                let last_access_ago = now_ms - state.last_accessed_ms;
                // access_count serves as rehearsal proxy; cap at 8 for bonus calc
                let rehearsal = state.access_count.min(8) as i64;

                match state.tier {
                    0 => {
                        // L1 → L2
                        if age_ms >= L1_TO_L2_AGE_MS
                            && last_access_ago >= L1_TO_L2_LAST_ACCESS_MS
                            && state.strength < L1_TO_L2_MAX_STRENGTH
                        {
                            to_demote.push((memory_id, 1));
                        }
                    }
                    1 => {
                        // L2 → L3
                        let threshold =
                            L2_TO_L3_AGE_BASE_MS + rehearsal * L2_TO_L3_REHEARSAL_BONUS_MS;
                        if age_ms >= threshold
                            && last_access_ago >= L2_TO_L3_LAST_ACCESS_MS
                            && state.strength < L2_TO_L3_MAX_STRENGTH
                        {
                            to_demote.push((memory_id, 2));
                        }
                    }
                    2 => {
                        // L3 → delete
                        let threshold =
                            L3_DELETE_AGE_BASE_MS + rehearsal * L3_DELETE_REHEARSAL_BONUS_MS;
                        if age_ms >= threshold
                            && last_access_ago >= L3_DELETE_LAST_ACCESS_MS
                            && state.strength < L3_DELETE_MAX_STRENGTH
                            && state.strength < L3_DELETE_MAX_UTILITY
                        {
                            to_delete.push(memory_id);
                        }
                    }
                    _ => {}
                }
            }
        }

        let demoted = to_demote.len();
        let deleted = to_delete.len();

        for (id, new_tier) in to_demote {
            let op = Op::DemoteMemory(DemoteMemoryOp {
                memory_id: id,
                new_tier,
            });
            self.log.write().append(&op)?;
            if let Some(state) = self.states.write().get_mut(&id) {
                state.tier = new_tier;
            }
        }
        for id in to_delete {
            self.forget(id)?;
        }

        // Gate B outflow: hourly CoRetrieved decay + floor-prune. An untouched
        // co-retrieval edge halves in ~34 passes (0.98^34); anything that decays
        // below 0.05 is noise and is dropped. Paired with the drain-side 0.1
        // materialization floor so decay isn't fighting a firehose.
        if self
            .plasticity_decay_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            let (survivors, pruned) = self.assoc_decay(EdgeType::CoRetrieved, 0.98, 0.05, true);
            if pruned > 0 {
                eprintln!("[plasticity] CoRetrieved decay: {survivors} survive, {pruned} pruned");
            }
        }

        Ok((demoted, deleted))
    }

    /// Encode all memories that don't yet have a sparse code.
    pub fn encode_all_unindexed(&self) -> Result<usize> {
        if self.ablations.disabled("cortical_idx") || self.ablations.disabled("sparse_encoder") { return Ok(0); }
        // Collect only memories that CAN encode: skip deleted (payloads
        // outlive soft-delete), empty/foreign-dim embeddings (the stripped-
        // snapshot rehydrator deliberately leaves deleted ones empty), and
        // ids whose sparse code came back empty before (runtime skip-set —
        // retried after restart). Without these filters the same ~7.6k
        // unencodable memories re-encoded on every consolidation cycle.
        let ids: Vec<MemoryId> = {
            let payloads = self.payloads.read();
            let states = self.states.read();
            let idx = self.semantic_idx.read();
            let cortical = self.cortical_idx.read();
            let skip = self.encode_skip.read();
            payloads
                .iter()
                .filter(|(id, p)| {
                    !cortical.mem_codes.contains_key(*id)
                        && !skip.contains(*id)
                        && (idx.get_embedding(**id).is_some()
                            || p.embedding.len() == EMBED_DIM)
                        && states.get(*id).map(|s| !s.deleted).unwrap_or(false)
                })
                .map(|(id, _)| *id)
                .collect()
        };
        let count = ids.len();
        for id in ids {
            self.encode_memory(id)?;
        }
        Ok(count)
    }

    /// Train a ProductQuantizer from the residuals of all encoded memories.
    /// Requires at least 256 memories with sparse codes.
    pub fn train_pq(&self) -> Result<()> {
        if self.ablations.disabled("cortical_idx") || self.ablations.disabled("sparse_encoder") { return Ok(()); }
        // Collect residuals: for each memory with a sparse code, decode and subtract
        let residuals: Vec<Vec<f32>> = {
            let payloads = self.payloads.read();
            let idx = self.semantic_idx.read();
            let encoder = self.sparse_encoder.read();
            let cortical = self.cortical_idx.read();

            cortical
                .mem_codes
                .iter()
                .filter_map(|(&memory_id, code)| {
                    let embedding = idx
                        .get_embedding(memory_id)
                        .map(|e| e.to_vec())
                        .or_else(|| payloads.get(&memory_id).map(|p| p.embedding.clone()))?;
                    if embedding.len() != crate::ops::EMBED_DIM {
                        return None;
                    }
                    let decoded = encoder.decode(code);
                    let residual: Vec<f32> = embedding
                        .iter()
                        .zip(decoded.iter())
                        .map(|(e, d)| e - d)
                        .collect();
                    Some(residual)
                })
                .collect()
        };

        let pq = ProductQuantizer::train(&residuals, 20)
            .map_err(crate::error::FieldError::Manifest)?;

        let codebook_bytes = bincode::serialize(&pq)
            .map_err(|e| crate::error::FieldError::Serialization(e.to_string()))?;

        let op = Op::TrainPQ(TrainPQOp { codebook_bytes });
        self.log.write().append(&op)?;

        self.cortical_idx.write().set_pq(pq);

        Ok(())
    }

    /// Encode PQ residual for a single memory. The PQ must already be trained.
    pub fn encode_pq_memory(&self, memory_id: MemoryId) -> Result<()> {
        if self.ablations.disabled("cortical_idx") || self.ablations.disabled("sparse_encoder") { return Ok(()); }
        let embedding = self.embedding_of(memory_id);
        let Some(embedding) = embedding else {
            return Ok(());
        };
        if embedding.len() != crate::ops::EMBED_DIM {
            return Ok(());
        }

        let decoded = {
            let encoder = self.sparse_encoder.read();
            let cortical = self.cortical_idx.read();
            let code = match cortical.mem_codes.get(&memory_id) {
                Some(c) => c.clone(),
                None => return Ok(()),
            };
            encoder.decode(&code)
        };

        let residual: Vec<f32> = embedding
            .iter()
            .zip(decoded.iter())
            .map(|(e, d)| e - d)
            .collect();

        let codes = {
            let cortical = self.cortical_idx.read();
            let pq = match &cortical.pq {
                Some(pq) => pq,
                None => return Ok(()),
            };
            pq.quantize(&residual)
        };

        let pq_bytes: Vec<u8> = codes.to_vec();
        let op = Op::UpdateResidualPQ(UpdateResidualPQOp {
            memory_id,
            pq_bytes,
        });
        self.log.write().append(&op)?;

        self.cortical_idx.write().index_pq(memory_id, codes);

        Ok(())
    }

    /// Encode PQ residuals for all memories that have sparse codes but no PQ code.
    /// If PQ is not yet trained, trains it first.
    /// Returns the count of memories PQ-encoded.
    pub fn encode_all_pq(&self) -> Result<usize> {
        if self.ablations.disabled("cortical_idx") || self.ablations.disabled("sparse_encoder") { return Ok(0); }
        if !self.cortical_idx.read().is_pq_trained() {
            self.train_pq()?;
        }

        let ids: Vec<MemoryId> = {
            let cortical = self.cortical_idx.read();
            cortical
                .mem_codes
                .keys()
                .filter(|id| !cortical.mem_pq.contains_key(id))
                .copied()
                .collect()
        };

        let count = ids.len();
        for id in ids {
            self.encode_pq_memory(id)?;
        }

        Ok(count)
    }

    /// Return how many memories have PQ residual codes.
    pub fn pq_count(&self) -> usize {
        if self.ablations.disabled("cortical_idx") { return 0; }
        self.cortical_idx.read().pq_count()
    }

}
