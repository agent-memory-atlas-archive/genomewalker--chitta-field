//! Recall operations.

use super::*;

// Evaluation clock for recall only: never change WAL, access or write timestamps.
fn parse_recall_now(value: Option<&str>) -> Option<i64> {
    value?.parse::<i64>().ok().filter(|now| *now > 0)
}

fn recall_now_ms() -> i64 {
    static FIXED: std::sync::OnceLock<Option<i64>> = std::sync::OnceLock::new();
    FIXED.get_or_init(|| parse_recall_now(std::env::var("CHITTA_RECALL_NOW").ok().as_deref()))
        .unwrap_or_else(super::now_ms)
}

#[cfg(test)]
#[test]
fn evaluation_recall_clock_accepts_only_positive_unix_milliseconds() {
    assert_eq!(parse_recall_now(Some("1789560000123")), Some(1789560000123));
    for value in [None, Some(""), Some("0"), Some("-1"), Some("123oops"), Some("99999999999999999999")] {
        assert_eq!(parse_recall_now(value), None);
    }
}

impl ChittaField {

    pub(super) fn enqueue_recall_effects(&self, hit_ids: &[MemoryId]) {
        const MAX_PENDING_STRENGTHEN: usize = 100_000;
        const MAX_PENDING_PAIRS: usize = 200_000;
        const MAX_PENDING_WINDOWS: usize = 20_000;

        if hit_ids.is_empty() {
            return;
        }

        let strengthen_ids = &hit_ids[..hit_ids.len().min(16)];
        let window_ids = hit_ids[..hit_ids.len().min(5)].to_vec();

        let mut pending = self.pending_recall.lock();
        for &id in strengthen_ids {
            if pending.strengthen.len() >= MAX_PENDING_STRENGTHEN {
                break;
            }
            pending.strengthen.insert(id);
        }

        for i in 0..window_ids.len() {
            for j in (i + 1)..window_ids.len() {
                if pending.co_retrieval_pairs.len() >= MAX_PENDING_PAIRS
                    && !pending
                        .co_retrieval_pairs
                        .contains_key(&(window_ids[i], window_ids[j]))
                {
                    continue;
                }
                *pending
                    .co_retrieval_pairs
                    .entry((window_ids[i], window_ids[j]))
                    .or_insert(0.0) += 0.05;
            }
        }

        if !window_ids.is_empty() && pending.proto_windows.len() < MAX_PENDING_WINDOWS {
            pending.proto_windows.push(window_ids);
        }
        drop(pending);

        // Hot-path: record access sequence for predictive memory (Layer 3)
        let mut predictor = self.predictor.write();
        for &id in hit_ids.iter().take(8) {
            predictor.record_access(id);
        }
    }

    pub(crate) fn drain_pending_recall_effects(&self) -> Result<()> {
        let pending = {
            let mut guard = self.pending_recall.lock();
            if guard.strengthen.is_empty()
                && guard.co_retrieval_pairs.is_empty()
                && guard.proto_windows.is_empty()
            {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };

        for memory_id in pending.strengthen {
            // Recall no longer inflates strength (S0: the uniform +0.01 tick is
            // rank-inert noise). Touch-only keeps recency/spacing signals intact.
            let _ = self.update_state(memory_id, None, None, None, true, None);
            let mut states = self.states.write();
            if let Some(st) = states.get_mut(&memory_id) {
                st.recompute_spacing_quality();
                let strength = st.strength;
                drop(states);
                self.cortical_idx
                    .write()
                    .update_strength(memory_id, strength);
            }
        }

        // Gate B inflow floor: add_assoc_edge merges by MAX, so a pair seen once
        // per drain window stays frozen at 0.05 forever — that class is 21M of
        // the 28.6M CoRetrieved edges. Requiring >=0.1 (two co-occurrences in
        // one window) stops the noise band at the source; repeat singles on an
        // existing edge were max-merge no-ops anyway (minus the WAL append).
        let floor = if self
            .plasticity_decay_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0.1
        } else {
            0.0
        };
        for ((src, dst), weight) in pending.co_retrieval_pairs {
            if weight < floor {
                continue;
            }
            let _ = self.add_assoc_edge(src, dst, EdgeType::CoRetrieved, weight);
        }

        if !pending.proto_windows.is_empty() {
            let mut cortical_idx = self.cortical_idx.write();
            for window in pending.proto_windows {
                cortical_idx.strengthen_proto_transitions(&window);
            }
        }

        Ok(())
    }

    /// Semantic recall: find k most similar memories to a query embedding.
    ///
    /// Applies realm filter and strength-weighted final scoring:
    ///   `score = semantic_score × (0.5 + 0.5 × effective_strength) × confidence`
    ///
    /// Uses the ANN semantic index directly, with optional realm filtering.
    /// Recall-side maintenance effects are deferred until flush/snapshot.
    pub fn recall_semantic(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_semantic_ctx(query_embedding, k, realm, None, None, true)
    }

    /// Measurement recall: identical ranking, but skips co-retrieval
    /// strengthening so an eval/diagnostic never mutates the store it reads.
    pub fn recall_semantic_measure(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_semantic_ctx(query_embedding, k, realm, None, None, false)
    }

    /// Lane-0 atom bridge as a first-class retrieval leg. Given the recall's
    /// dense-lead anchor, return its saturating-IDF co-atom partners (id, weight)
    /// so a caller (the C++ multi-lane RRF in tool_recall) can fuse them ALONGSIDE
    /// dense/keyword/HDC — a keyword-absent partner then competes on the strength
    /// of a rare shared atom instead of being drowned as a single-dense-lane hit.
    /// Weight is the BM25 IDF of the rarest shared atom (already summed across
    /// shared atoms by `bridge_candidates_saturating`). Silent (empty) when the
    /// bridge is disabled, the anchor carries no atom, or nothing co-occurs.
    pub fn bridge_lane_scored(
        &self,
        anchor: MemoryId,
        realm: Option<&str>,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        // NOT gated by bridge_lane_enabled (that flag governs the in-Rust
        // dense-lane tau bridge): this leg is a distinct C++ lane-0 feature,
        // gated caller-side by CHITTA_BRIDGE_LANE0 so a single flag flips it.
        let anchor_paths = self.artifact_idx.read().paths_for_memory(anchor);
        if anchor_paths.is_empty() {
            return Vec::new();
        }
        let n_total = self
            .memory_count
            .load(std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let cands = self.artifact_idx.read().bridge_candidates_saturating(
            &anchor_paths,
            n_total,
            anchor,
            k,
        );
        let payloads = self.payloads.read();
        cands
            .into_iter()
            .filter(|(m, _)| {
                match payloads.get(m) {
                    // Mirror build_hit's soul: bleed gate: a soul-realm memory only
                    // surfaces into a soul-realm query.
                    Some(p) => {
                        !(p.realm.starts_with("soul:")
                            && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true))
                    }
                    None => false,
                }
            })
            .collect()
    }

    /// Semantic recall with affective context.
    ///
    /// `query_valence` / `query_arousal`: caller's current affect state.
    /// Enables mood-congruent recall (Bower 1981) and frustration-escalation
    /// detection (boost corrections when caller is frustrated).
    pub fn recall_semantic_ctx(
        &self,
        query_embedding: &[f32],
        k: usize,
        realm: Option<&str>,
        query_valence: Option<f32>,
        query_arousal: Option<f32>,
        strengthen: bool,
    ) -> Result<Vec<RecallHit>> {
        let mut profile = crate::profile::RecallProfile::new("semantic_search");
        if query_embedding.len() != EMBED_DIM {
            return Err(FieldError::InvalidEmbedDim {
                expected: EMBED_DIM,
                actual: query_embedding.len(),
            });
        }

        let now = recall_now_ms();
        // Realm-filtered HNSW traversal needs a wider beam: with an allowed-set
        // the graph skips most neighbors, and at k*3 a memory that is the true
        // in-realm nearest neighbor can be unreachable (observed: rank 0 at
        // ef~1920, absent at ef~180 on a 109k store). Floor of 128 when filtered.
        let result_limit = if realm.is_some() {
            k.saturating_mul(3).max(128)
        } else {
            k.saturating_mul(3).max(k)
        };
        // realm_members guard scoped to the search: holding it (a late-order
        // lock) across the states/idx acquisitions below deadlocks against
        // put_memory, which holds states.write while taking realm_members.write.
        let semantic_hits = {
            let realm_members = self.realm_members.read();
            let allowed = realm.and_then(|r| realm_members.get(r));
            self.semantic_idx
                .read()
                .search(query_embedding, result_limit, allowed, realm)
        };

        profile.next("cw_refresh");
        // Refresh competitive_weight for each candidate using the *current* HNSW neighborhood.
        // The write-time value is stale for memories ingested when the store was sparse.
        // Two-phase to avoid holding states.write() during HNSW searches.
        let (dedup_upper, cw_refresh_interval_ms, cw_refresh_budget) = {
            let pipeline = self.scoring_pipeline.read();
            (
                pipeline.config.dedup_cosine_upper,
                pipeline.config.cw_refresh_interval_ms,
                pipeline.config.cw_refresh_max_per_query,
            )
        };
        // Phase A — find candidates that need refresh, clone embeddings.
        // Uses read locks only; the inflight check here is a cheap pre-filter,
        // the authoritative check-and-reserve happens under the write guard below.
        let candidates: Vec<(MemoryId, Vec<f32>)> = if !strengthen {
            // Measurement reads must use the stored scoring inputs. Refreshing a
            // budgeted subset here changes later no_learn queries and restart
            // identity even though no access count or WAL entry is written.
            Vec::new()
        } else {
            // Lock order: states before semantic_idx (struct order) — the
            // inverse deadlocked against sync_foreign in production.
            let states_r = self.states.read();
            let idx = self.semantic_idx.read();
            let inflight_r = self.cw_refresh_inflight.read();
            semantic_hits.iter().filter_map(|hit| {
                // Skip if refreshed recently by this or another session.
                if let Some(st) = states_r.get(&hit.memory_id) {
                    if now - st.last_cw_refresh_ms < cw_refresh_interval_ms { return None; }
                }
                if let Some(&ts) = inflight_r.get(&hit.memory_id) {
                    if now - ts < cw_refresh_interval_ms { return None; }
                }
                // Clone embedding so we can drop all locks before searching.
                let emb = idx.get_embedding(hit.memory_id)?.to_vec();
                Some((hit.memory_id, emb))
            })
            // Budget: each refresh is a full ANN/flat search; the rest are
            // picked up by later queries (amortized refresh).
            .take(cw_refresh_budget)
            .collect()
        };
        // Atomically re-check and reserve under a single write guard: concurrent
        // sessions can all pass the read-lock pre-filter above, but only one wins
        // each slot here.
        let candidates: Vec<(MemoryId, Vec<f32>)> = if candidates.is_empty() {
            candidates
        } else {
            let mut inflight_w = self.cw_refresh_inflight.write();
            // Evict expired reservations (sessions that died mid-search) on every
            // pass, not only on rounds that produce updates.
            inflight_w.retain(|_, ts| now - *ts < cw_refresh_interval_ms);
            candidates
                .into_iter()
                .filter(|(id, _)| {
                    if inflight_w.contains_key(id) { return false; }
                    inflight_w.insert(*id, now);
                    true
                })
                .collect()
        };
        // Snapshot the prepared index under a brief read, then release the guard
        // before the independent neighborhood probes. Scalar/HNSW/dirty cases
        // retain the original path and its scoring semantics.
        let refresh_plan = self.semantic_idx.read().plan_refresh_searches(&candidates, realm);
        let neighborhoods = if let Some(plan) = refresh_plan {
            plan.search_all(&candidates)
        } else {
            let idx = self.semantic_idx.read();
            candidates.iter().map(|(_, emb)| idx.search(emb, 9, None, realm)).collect()
        };
        let cw_updates: Vec<(MemoryId, f32)> = candidates.iter().zip(neighborhoods.iter())
            .filter_map(|((memory_id, _), neighbors)| {
                if neighbors.len() <= 1 { return None; }
                let mut cos_sum = 0.0f32;
                let mut n = 0u32;
                for nb in neighbors {
                    if nb.memory_id == *memory_id { continue; }
                    if nb.cosine_similarity >= dedup_upper { continue; }
                    cos_sum += nb.cosine_similarity;
                    n += 1;
                }
                if n > 0 { Some((*memory_id, cos_sum / n as f32)) } else { None }
            }).collect();
        // Phase B — apply under brief states.write(), then release reservations.
        // Every searched candidate is marked refreshed even when its neighborhood
        // produced no update, so isolated memories aren't re-searched on every
        // recall; reservations are always released so empty rounds don't leak.
        if !candidates.is_empty() {
            let cw_by_id: std::collections::HashMap<MemoryId, f32> =
                cw_updates.into_iter().collect();
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
        }

        profile.next("semantic_score_format");
        let payloads = self.payloads.read();
        let states = self.states.read();
        let learners = self.learners.read();
        let pipeline = self.scoring_pipeline.read();
        let ack_scores = self.ack_scores.read();
        let recall_prov = self.recall_provenance.read();
        let mut util_rng = utility_rng(now);

        let mut hits: Vec<RecallHit> = semantic_hits
            .into_iter()
            .filter_map(|hit| {
                let memory_id = hit.memory_id;
                let state = states.get(&memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&memory_id)?;
                let content_str = String::from_utf8(payload.content.clone()).unwrap_or_default();
                if content_str.trim().is_empty() {
                    return None;
                }
                // soul:* realms are internal — exclude from unscoped queries.
                if payload.realm.starts_with("soul:") && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true) {
                    return None;
                }
                let ctx = ScoringContext {
                    relevance_score: hit.cosine_similarity,
                    recall_mode: RecallMode::Semantic,
                    state,
                    kind: &payload.kind,
                    realm: &payload.realm,
                    realm_reliability: if self.ablations.disabled("learners") { 1.0 } else { learners.domain_reliability.reliability(&payload.realm) },
                    now_ms: now,
                    query_valence,
                    query_arousal,
                    prediction_prob: None,
                    surprise_role: None,
                    has_open_debt: false,
                    integration_weight: None,
                    ack_score: ack_scores.get(&memory_id).copied().unwrap_or(0),
                    max_query_idf: 0.0,
                };
                let (score, decomp) = pipeline.score(&ctx)?;
                // Cross-context generality (THEORY.md §6): recall from N
                // distinct daemons is evidence the memory generalizes beyond
                // one session/node. Multiplicative, config-gated boost.
                let score = {
                    let w = pipeline.config.cross_context_weight;
                    if w > 0.0 {
                        let distinct = recall_prov
                            .get(&memory_id)
                            .map(|s| s.len())
                            .unwrap_or(0) as f32;
                        score
                            * (1.0
                                + w * (distinct - 1.0)
                                    .max(0.0)
                                    .min(pipeline.config.cross_context_max))
                    } else {
                        score
                    }
                };
                // Outcome utility (flag-gated): did recalling this memory
                // actually help? Multiplier is exactly 1.0 when disabled.
                let score = score * utility_multiplier(state, &mut util_rng);
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id,
                    score,
                    semantic_score: hit.cosine_similarity,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: content_str,
                    semantic_weight: decomp.semantic_weight,
                    status_mul: decomp.status_mul,
                    epistemic_mul: decomp.epistemic_mul,
                    strength_factor: decomp.strength_factor,
                    affect_valence: state.affect_valence,
                    affect_arousal: state.affect_arousal,
                    actr_activation: decomp.actr_activation,
                    surprise_boost: decomp.surprise_boost,
                    arousal_boost: decomp.arousal_boost,
                    mood_congruence: decomp.mood_congruence,
                    frustration_boost: decomp.frustration_boost,
                    interference_factor: decomp.interference_factor,
                    spacing_boost: decomp.spacing_boost,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id))
        });

        // Lure detection (Price of Meaning no-escape theorem):
        // Suppress high-lure-risk candidates that could be false recalls.
        // Only suppress from the tail — never remove the top-scoring hit.
        let lure_threshold = pipeline.config.lure_risk_threshold;
        let max_suppressed = pipeline.config.lure_max_suppressed;
        if max_suppressed > 0 && hits.len() > 1 {
            let mut suppressed = 0usize;
            let mut i = hits.len();
            while i > 1 && suppressed < max_suppressed {
                i -= 1;
                if states.get(&hits[i].memory_id)
                    .map(|s| s.lure_risk >= lure_threshold)
                    .unwrap_or(false)
                {
                    hits.remove(i);
                    suppressed += 1;
                }
            }
        }

        // Factual-bridge recall leg (df-gated IDF atom-postings join). Reserves a
        // few top-k slots for co-atom neighbours of the top dense hit that share a
        // rare identity atom (df <= tau) — the multi-hop partner the dense pool
        // buried. ZERO cosine in the ranking; A's atoms are used only as the join
        // KEY (non-circular). Silent when the anchor carries no gated atom, so the
        // dense result is untouched on the ~96% of queries that can't fire.
        if self.bridge_lane_enabled.load(std::sync::atomic::Ordering::Relaxed)
            && !hits.is_empty()
            && k > 0
        {
            let anchor = hits[0].memory_id;
            let anchor_paths = self.artifact_idx.read().paths_for_memory(anchor);
            if !anchor_paths.is_empty() {
                let n_total = self
                    .memory_count
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .max(1);
                let tau: usize = std::env::var("CHITTA_BRIDGE_TAU")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(32);
                let reserve: usize = std::env::var("CHITTA_BRIDGE_RESERVE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4)
                    .min(k);
                // Lane-0 (CHITTA_BRIDGE_LANE0): saturating-IDF bridge fused as a
                // scored RRF peer instead of the positional reserve injection.
                let lane0 = std::env::var("CHITTA_BRIDGE_LANE0")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let cands = if lane0 {
                    self.artifact_idx.read().bridge_candidates_saturating(
                        &anchor_paths,
                        n_total,
                        anchor,
                        k.saturating_mul(4).max(64),
                    )
                } else {
                    self.artifact_idx.read().bridge_candidates(
                        &anchor_paths,
                        n_total,
                        tau,
                        anchor,
                        reserve * 4,
                    )
                };
                if !cands.is_empty() {
                    // Score a bridge-injected memory into a well-formed RecallHit.
                    // semantic_score = 0.0: it was surfaced by the postings join, not
                    // cosine. Final ordering is positional (reserved slots), so the
                    // low relevance score never re-sorts it above the dense head.
                    let build_hit = |memory_id: MemoryId| -> Option<RecallHit> {
                        let state = states.get(&memory_id)?;
                        if state.deleted {
                            return None;
                        }
                        let payload = payloads.get(&memory_id)?;
                        let content_str =
                            String::from_utf8(payload.content.clone()).unwrap_or_default();
                        if content_str.trim().is_empty() {
                            return None;
                        }
                        if payload.realm.starts_with("soul:")
                            && realm.map(|r| !r.starts_with("soul:")).unwrap_or(true)
                        {
                            return None;
                        }
                        let ctx = ScoringContext {
                            relevance_score: 0.0,
                            recall_mode: RecallMode::Semantic,
                            state,
                            kind: &payload.kind,
                            realm: &payload.realm,
                            realm_reliability: if self.ablations.disabled("learners") { 1.0 } else { learners.domain_reliability.reliability(&payload.realm) },
                            now_ms: now,
                            query_valence,
                            query_arousal,
                            prediction_prob: None,
                            surprise_role: None,
                            has_open_debt: false,
                            integration_weight: None,
                            ack_score: ack_scores.get(&memory_id).copied().unwrap_or(0),
                            max_query_idf: 0.0,
                        };
                        let (score, decomp) = pipeline.score(&ctx)?;
                        let eff_strength = state.effective_strength(now);
                        Some(RecallHit {
                            memory_id,
                            score,
                            semantic_score: 0.0,
                            ts_ms: payload.authored_at_ms,
                            kind: payload.kind.clone(),
                            realm: payload.realm.clone(),
                            strength: eff_strength,
                            confidence: state.confidence,
                            access_count: state.access_count,
                            content: content_str,
                            semantic_weight: decomp.semantic_weight,
                            status_mul: decomp.status_mul,
                            epistemic_mul: decomp.epistemic_mul,
                            strength_factor: decomp.strength_factor,
                            affect_valence: state.affect_valence,
                            affect_arousal: state.affect_arousal,
                            actr_activation: decomp.actr_activation,
                            surprise_boost: decomp.surprise_boost,
                            arousal_boost: decomp.arousal_boost,
                            mood_congruence: decomp.mood_congruence,
                            frustration_boost: decomp.frustration_boost,
                            interference_factor: decomp.interference_factor,
                            spacing_boost: decomp.spacing_boost,
                        })
                    };
                    if lane0 {
                        // Scored RRF peer: reciprocal-rank-fuse every saturating-IDF
                        // co-atom candidate against the dense/keyword result so a
                        // strong bridge competes on merit, not a reserved slot.
                        let bridge_hits: Vec<RecallHit> =
                            cands.iter().filter_map(|(m, _)| build_hit(*m)).collect();
                        if !bridge_hits.is_empty() {
                            let rrf_k = pipeline.config.rrf_k;
                            hits = rrf_merge(std::mem::take(&mut hits), bridge_hits, k, rrf_k);
                        }
                    } else {
                    // No-evict fusion. The dense top-k is the base. We only inject
                    // GENUINELY-NEW bridges (co-atom partners the dense pool buried
                    // past k) and pay for each by evicting a *weak* dense-tail hit —
                    // never a dense hit that is itself a co-atom bridge (that hit is
                    // reinforced by both signals; evicting it is the pure loss the
                    // fixed-reserve version caused, e.g. a gold B already at dense
                    // rank 7). Deduped: a bridge already in the dense top-k costs no
                    // slot. Bounded by `reserve`.
                    let cand_ids: std::collections::HashSet<MemoryId> =
                        cands.iter().map(|(m, _)| *m).collect();
                    let dense_topk: std::collections::HashSet<MemoryId> =
                        hits.iter().take(k).map(|h| h.memory_id).collect();
                    let mut new_bridge: Vec<RecallHit> = Vec::new();
                    for (mid, _w) in &cands {
                        if new_bridge.len() >= reserve {
                            break;
                        }
                        if dense_topk.contains(mid) {
                            continue; // already shown by dense — no slot needed
                        }
                        if let Some(h) = build_hit(*mid) {
                            new_bridge.push(h);
                        }
                    }
                    if !new_bridge.is_empty() {
                        let mut kept: Vec<RecallHit> = hits.iter().take(k).cloned().collect();
                        let need = new_bridge.len();
                        let mut room = 0usize;
                        // Pass 1: evict weakest dense-tail hits that are NOT co-atom
                        // bridges (protects a present B, which shares the anchor atom).
                        let mut i = kept.len();
                        while room < need && i > 0 {
                            i -= 1;
                            if !cand_ids.contains(&kept[i].memory_id) {
                                kept.remove(i);
                                room += 1;
                            }
                        }
                        // Pass 2: if the whole tail was co-atom bridges, drop plain
                        // tail down to the reserve floor so a truly-new bridge can enter.
                        while room < need && kept.len() > k.saturating_sub(reserve) {
                            kept.pop();
                            room += 1;
                        }
                        kept.extend(new_bridge.into_iter().take(room));
                        hits = kept;
                    }
                    }
                }
            }
        }

        hits.truncate(k);

        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        drop(pipeline);
        drop(learners);
        drop(ack_scores);
        if strengthen {
            self.enqueue_recall_effects(&hit_ids);
        }

        Ok(hits)
    }

    /// SOTA HippoRAG-style Personalized-PageRank lane over the association graph.
    ///
    /// Seeds are the query's top fused hits, weighted by their fused score. A
    /// single PPR pass (power iteration with restart) spreads that mass over a
    /// lazily-expanded neighborhood, ranking graph-reachable memories the
    /// lexical/semantic lanes never surfaced — the multi-hop bridge that shares
    /// no query terms with the question. Returns the top-`top_g` stationary
    /// nodes EXCLUDING the seeds (the injection); seeds are already ranked.
    ///
    /// Edge transition weight = type_prior * stored_weight, row-normalized per
    /// source. CoRetrieved (the weak Hebbian lane) is down-weighted vs the
    /// strong DerivedFrom provenance lane. All knobs are env-gated. The frontier
    /// is capped so the walk stays local and fast — it never materializes the
    /// whole node graph, only the seeds' bounded neighborhood.
    pub fn ppr_lane(
        &self,
        seed_ids: &[MemoryId],
        seed_weights: &[f32],
        top_g: usize,
    ) -> Vec<(MemoryId, f32)> {
        if seed_ids.is_empty() || top_g == 0 {
            return Vec::new();
        }
        let get_f = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let get_u = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let alpha = get_f("CHITTA_PPR_ALPHA", 0.15).clamp(0.01, 0.99);
        let w_derived = get_f("CHITTA_PPR_W_DERIVED", 1.0);
        let w_sameartifact = get_f("CHITTA_PPR_W_SAMEARTIFACT", 0.8);
        let w_samesession = get_f("CHITTA_PPR_W_SAMESESSION", 0.6);
        let w_coretrieved = get_f("CHITTA_PPR_W_CORETRIEVED", 0.35);
        let w_semneighbor = get_f("CHITTA_PPR_W_SEMNEIGHBOR", 0.7);
        let frontier_cap = get_u("CHITTA_PPR_FRONTIER_CAP", 4000).max(seed_ids.len());
        let max_iters = get_u("CHITTA_PPR_MAX_ITERS", 25).max(1);
        // Per-node out-degree cap: a few high-degree hubs (one memory here has
        // >6k assoc edges) make the in-set adjacency dense, so the power
        // iteration degrades to O(iters * n^2) — the measured 35s recall stall.
        // Keep only each node's top-K edges by transition weight; a hub's mass is
        // diffuse and the tail edges contribute negligibly to the stationary rank.
        let max_degree = get_u("CHITTA_PPR_MAX_DEGREE", 64).max(1);
        const HOP_BUDGET: usize = 4;

        let type_prior = |et: &EdgeType| -> f32 {
            match et {
                EdgeType::DerivedFrom => w_derived,
                EdgeType::SameArtifact => w_sameartifact,
                EdgeType::SameSession => w_samesession,
                EdgeType::CoRetrieved => w_coretrieved,
                EdgeType::Supports => 0.4,
                EdgeType::Contradicts => 0.3,
                EdgeType::SemanticNeighbor => w_semneighbor,
            }
        };

        let pipeline_config = self.scoring_pipeline.read().config.clone();
        let states = self.states.read();
        let assoc_edges = self.assoc_edges.read();

        let legal = |id: &MemoryId| -> bool {
            match states.get(id) {
                None => false,
                Some(s) if s.deleted => false,
                Some(s) => {
                    crate::scoring::status_multiplier(&s.status, &pipeline_config).is_some()
                }
            }
        };

        // Lazy BFS: build the local node set (seeds first) bounded by a small hop
        // budget and a hard frontier cap. Nearest-first exploration keeps the walk
        // concentrated where PPR mass actually lands.
        let mut index: std::collections::HashMap<MemoryId, usize> =
            std::collections::HashMap::new();
        let mut nodes: Vec<MemoryId> = Vec::new();
        for &s in seed_ids {
            index.entry(s).or_insert_with(|| {
                let i = nodes.len();
                nodes.push(s);
                i
            });
        }
        let seed_n = nodes.len();
        let mut queue: std::collections::VecDeque<(MemoryId, usize)> =
            nodes.iter().map(|&id| (id, 0usize)).collect();
        while let Some((node, hop)) = queue.pop_front() {
            if hop >= HOP_BUDGET || nodes.len() >= frontier_cap {
                continue;
            }
            let neighbors = match assoc_edges.get(&node) {
                Some(v) => v,
                None => continue,
            };
            // Cap fan-out to the top-`max_degree` neighbors by transition weight so
            // a single hub cannot flood the frontier with its low-weight tail.
            let mut cand: Vec<(f32, MemoryId)> = neighbors
                .iter()
                .map(|e| (type_prior(&e.edge_type) * e.weight.max(0.0), e.dst))
                .filter(|&(w, _)| w > 0.0)
                .collect();
            if cand.len() > max_degree {
                cand.select_nth_unstable_by(max_degree, |a, b| {
                    b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
                });
                cand.truncate(max_degree);
            }
            cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_, dst) in cand {
                if index.contains_key(&dst) || !legal(&dst) {
                    continue;
                }
                if nodes.len() >= frontier_cap {
                    break;
                }
                index.insert(dst, nodes.len());
                nodes.push(dst);
                queue.push_back((dst, hop + 1));
            }
        }

        let n = nodes.len();
        // Row-normalized transition rows restricted to the explored set.
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (i, &src) in nodes.iter().enumerate() {
            if let Some(edges) = assoc_edges.get(&src) {
                let mut row: Vec<(usize, f32)> = Vec::new();
                for edge in edges {
                    if let Some(&j) = index.get(&edge.dst) {
                        let w = type_prior(&edge.edge_type) * edge.weight.max(0.0);
                        if w > 0.0 {
                            row.push((j, w));
                        }
                    }
                }
                // Bound the row to top-`max_degree` by weight: this is what keeps
                // the power iteration O(iters * n * max_degree) instead of O(n^2).
                if row.len() > max_degree {
                    row.select_nth_unstable_by(max_degree, |a, b| {
                        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                    });
                    row.truncate(max_degree);
                }
                row.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
                let sum: f32 = row.iter().map(|e| e.1).sum();
                if sum > 0.0 {
                    for e in &mut row {
                        e.1 /= sum;
                    }
                    adj[i] = row;
                }
            }
        }
        drop(assoc_edges);
        drop(states);

        // Restart distribution: seed_weights normalized over the seeds, 0 else.
        let mut restart = vec![0.0f32; n];
        let mut wsum = 0.0f32;
        for (k, &s) in seed_ids.iter().enumerate() {
            if let Some(&i) = index.get(&s) {
                let w = seed_weights.get(k).copied().unwrap_or(1.0).max(0.0);
                restart[i] += w;
                wsum += w;
            }
        }
        if wsum <= 0.0 {
            let u = 1.0 / seed_n as f32;
            for r in restart.iter_mut().take(seed_n) {
                *r = u;
            }
        } else {
            for r in &mut restart {
                *r /= wsum;
            }
        }

        // Power iteration: p_{t+1} = (1-alpha) Pᵀ p_t + alpha r. Dangling mass
        // (nodes with no in-set out-edges) is redistributed through the restart
        // vector so p stays a proper distribution and the walk converges.
        let mut p = restart.clone();
        let mut next = vec![0.0f32; n];
        for _ in 0..max_iters {
            for i in 0..n {
                next[i] = alpha * restart[i];
            }
            let mut dangling = 0.0f32;
            for i in 0..n {
                if adj[i].is_empty() {
                    dangling += p[i];
                    continue;
                }
                let out = (1.0 - alpha) * p[i];
                for &(j, w) in &adj[i] {
                    next[j] += out * w;
                }
            }
            if dangling > 0.0 {
                let spread = (1.0 - alpha) * dangling;
                for i in 0..n {
                    next[i] += spread * restart[i];
                }
            }
            let mut delta = 0.0f32;
            for i in 0..n {
                delta += (next[i] - p[i]).abs();
            }
            std::mem::swap(&mut p, &mut next);
            if delta < 1e-4 {
                break;
            }
        }

        // Rank non-seed nodes by stationary score; return top_g.
        let mut ranked: Vec<(MemoryId, f32)> = nodes
            .iter()
            .enumerate()
            .skip(seed_n)
            .filter(|&(i, _)| p[i] > 0.0)
            .map(|(i, &id)| (id, p[i]))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_g);
        ranked
    }

    /// Gated PPR injection for the MCP hybrid-recall path. Seed the PPR lane from
    /// the current top hits and RRF-merge its stationary ranking back in,
    /// INJECTING graph-reachable memories the lexical/semantic lanes missed.
    /// Default OFF: the dev ablation showed it regresses single_hop nDCG for a
    /// noise-level multihop gain (net-negative), so it ships dormant. Enable with
    /// `CHITTA_PPR_LANE=1`. Injected nodes carry `semantic_score` 0.0, so any
    /// cosine abstain gate is unaffected. Mirrors the C++ tier-2 PPR lane in
    /// field_memory_recall.cpp.
    pub(super) fn ppr_inject(&self, hits: &mut Vec<RecallHit>) {
        if std::env::var("CHITTA_PPR_LANE").ok().as_deref() != Some("1") {
            return;
        }
        if hits.len() < 2 {
            return;
        }
        let get_u = |k: &str, d: usize| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let get_f = |k: &str, d: f32| {
            std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let seeds_n = get_u("CHITTA_PPR_SEEDS", 5);
        let top_g = get_u("CHITTA_PPR_G", 15);
        let lane_w = get_f("CHITTA_PPR_LANE_W", 1.0);

        let seed_ids: Vec<MemoryId> = hits.iter().take(seeds_n).map(|h| h.memory_id).collect();
        let seed_w: Vec<f32> = hits.iter().take(seeds_n).map(|h| h.score.max(1e-6)).collect();
        let lane = self.ppr_lane(&seed_ids, &seed_w, top_g);
        if lane.is_empty() {
            return;
        }

        const RRF_K: f32 = 60.0;
        let mut fused: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();
        let mut by_id: std::collections::HashMap<MemoryId, RecallHit> =
            std::collections::HashMap::new();
        for (rank, h) in hits.iter().enumerate() {
            *fused.entry(h.memory_id).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f32);
            by_id.entry(h.memory_id).or_insert_with(|| h.clone());
        }

        let payloads = self.payloads.read();
        let states = self.states.read();
        let pipeline_config = self.scoring_pipeline.read().config.clone();
        let now = recall_now_ms();
        for (rank, (mid, _)) in lane.iter().enumerate() {
            let contrib = lane_w * (1.0 / (RRF_K + (rank + 1) as f32));
            if by_id.contains_key(mid) {
                *fused.entry(*mid).or_insert(0.0) += contrib;
                continue;
            }
            // Hydrate an injected node into a full RecallHit (graph-derived: no
            // semantic score/weight, mirrors expand_associations' construction).
            let state = match states.get(mid) {
                Some(s) if !s.deleted => s,
                _ => continue,
            };
            let status_mul =
                match crate::scoring::status_multiplier(&state.status, &pipeline_config) {
                    Some(m) => m,
                    None => continue,
                };
            let payload = match payloads.get(mid) {
                Some(p) => p,
                None => continue,
            };
            *fused.entry(*mid).or_insert(0.0) += contrib;
            let eff_strength = state.effective_strength(now);
            by_id.insert(
                *mid,
                RecallHit {
                    memory_id: *mid,
                    score: 0.0,
                    semantic_score: 0.0,
                    ts_ms: payload.created_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                },
            );
        }
        drop(payloads);
        drop(states);

        let mut order: Vec<(MemoryId, f32)> = fused
            .into_iter()
            .filter(|(id, _)| by_id.contains_key(id))
            .collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut merged = Vec::with_capacity(order.len());
        for (mid, s) in order {
            if let Some(mut h) = by_id.remove(&mid) {
                h.score = s;
                merged.push(h);
            }
        }
        *hits = merged;
    }

    /// Expand from seed memory IDs via typed association edges (max 2 hops).
    ///
    /// Returns memories discovered via the association graph, scored by
    /// spreading activation with hop decay (×0.55 per hop).
    ///
    /// Edge type priors:
    ///   DerivedFrom=1.0, SameArtifact=0.8, SameSession=0.6, CoRetrieved=0.5,
    ///   Supports=0.4, Contradicts=0.3
    pub fn expand_associations(
        &self,
        seed_ids: &[MemoryId],
        max_hops: usize,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        const HOP_DECAY: f32 = 0.55;
        const FANOUT_CAP: usize = 16;
        let max_hops = max_hops.min(2);

        let edge_prior = |et: &EdgeType| -> f32 {
            match et {
                EdgeType::DerivedFrom => 1.0,
                EdgeType::SameArtifact => 0.8,
                EdgeType::SameSession => 0.6,
                EdgeType::CoRetrieved => 0.5,
                EdgeType::Supports => 0.4,
                EdgeType::Contradicts => 0.3,
                EdgeType::SemanticNeighbor => 0.7,
            }
        };

        let seed_set: HashSet<MemoryId> = seed_ids.iter().copied().collect();
        let pipeline_config = self.scoring_pipeline.read().config.clone();
        // activation accumulator: memory_id -> max activation score seen
        let mut activation: std::collections::HashMap<MemoryId, f32> =
            std::collections::HashMap::new();

        // frontier: (memory_id, activation_score, hops_remaining)
        let mut frontier: Vec<(MemoryId, f32, usize)> =
            seed_ids.iter().map(|&id| (id, 1.0, max_hops)).collect();

        // Lock order: payloads → states → assoc_edges (struct order; matches
        // sync_foreign's write-guard acquisition). payloads is needed only
        // after the walk but must be acquired first to keep the global order.
        let payloads = self.payloads.read();
        let states = self.states.read();
        let assoc_edges = self.assoc_edges.read();

        while let Some((node, act, hops_left)) = frontier.pop() {
            if hops_left == 0 {
                continue;
            }
            let neighbors = match assoc_edges.get(&node) {
                Some(v) => v,
                None => continue,
            };
            for edge in neighbors.iter().take(FANOUT_CAP) {
                let dst = edge.dst;
                // Skip deleted or status-suppressed memories so assoc expansion
                // honours the same suppression as semantic recall.
                match states.get(&dst) {
                    None => continue,
                    Some(s) if s.deleted => continue,
                    Some(s) if crate::scoring::status_multiplier(&s.status, &pipeline_config).is_none() => continue,
                    Some(_) => {}
                }
                let edge_act = act * HOP_DECAY * edge_prior(&edge.edge_type) * edge.weight;
                let entry = activation.entry(dst).or_insert(0.0);
                if edge_act > *entry {
                    *entry = edge_act;
                    // Only continue expanding if this is a new or improved path.
                    if hops_left > 1 {
                        frontier.push((dst, edge_act, hops_left - 1));
                    }
                }
            }
        }

        drop(assoc_edges);

        let now = recall_now_ms();

        let mut hits: Vec<RecallHit> = activation
            .into_iter()
            .filter(|(id, _)| !seed_set.contains(id))
            .filter_map(|(id, act_score)| {
                let state = states.get(&id)?;
                if state.deleted { return None; }
                // Belt-and-braces: frontier filtered most suppressed memories, but
                // edges added after a status flip can still land in activation.
                let status_mul = crate::scoring::status_multiplier(&state.status, &pipeline_config)?;
                let payload = payloads.get(&id)?;
                let eff_strength = state.effective_strength(now);
                let score = act_score * eff_strength * status_mul;
                Some(RecallHit {
                    memory_id: id,
                    score,
                    semantic_score: 0.0,
                    ts_ms: payload.created_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id))
        });
        hits.truncate(limit);

        Ok(hits)
    }

    /// Recall memories within a time range [start_ms, end_ms].
    pub fn recall_temporal(
        &self,
        start_ms: i64,
        end_ms: i64,
        realm: Option<&str>,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        let entries = self
            .time_idx
            .read()
            .range_query(start_ms, end_ms, realm, limit);
        let now = recall_now_ms();
        let payloads = self.payloads.read();
        let states = self.states.read();

        let hits = entries
            .into_iter()
            .filter_map(|entry| {
                let state = states.get(&entry.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&entry.memory_id)?;
                if payload.content.is_empty() {
                    return None;
                }
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: entry.memory_id,
                    score: eff_strength * state.confidence,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Keyword (BM25) recall.
    pub fn recall_keyword(&self, query: &str, k: usize) -> Result<Vec<RecallHit>> {
        self.recall_keyword_ctx(query, k, None, None, None, true)
    }
    /// Realm-scoped keyword recall: drops hits outside `realm` (None = unscoped) so the BM25
    /// lane honours --realm and never bleeds other projects' memories into a scoped query.
    pub fn recall_keyword_realm(
        &self,
        query: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_keyword_ctx(query, k, None, None, realm, true)
    }

    /// Measurement keyword recall: skips co-retrieval strengthening.
    pub fn recall_keyword_measure(
        &self,
        query: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_keyword_ctx(query, k, None, None, realm, false)
    }

    /// HDC recall: O(n) Hamming-distance search over binary hypervectors.
    /// Returns hits ordered by ascending Hamming distance (smaller = more similar).
    /// Converts to `RecallHit` with `semantic_score = 1 - hamming/8192`.
    pub fn recall_hdc(&self, query: &str, k: usize, realm: Option<&str>) -> Result<Vec<RecallHit>> {
        let hdc_hits = self.hdc_idx.read().query(query, k * 2, realm);
        if hdc_hits.is_empty() {
            return Ok(vec![]);
        }
        let payloads = self.payloads.read();
        let states   = self.states.read();
        let mut hits = Vec::with_capacity(hdc_hits.len());
        for (id, hamming_dist) in hdc_hits {
            let Some(payload) = payloads.get(&id) else { continue };
            let Some(state)   = states.get(&id)   else { continue };
            if state.deleted { continue; }
            let sim = 1.0 - hamming_dist as f32 / (128 * 64) as f32;
            hits.push(RecallHit {
                memory_id:          id,
                score:              sim,
                semantic_score:     sim,
                ts_ms:              payload.authored_at_ms,
                kind:               payload.kind.clone(),
                realm:              payload.realm.clone(),
                strength:           state.strength,
                confidence:         state.confidence,
                access_count:       state.access_count,
                content:            std::str::from_utf8(&payload.content)
                                        .unwrap_or("").to_string(),
                semantic_weight:    1.0,
                status_mul:         1.0,
                epistemic_mul:      1.0,
                strength_factor:    state.strength,
                affect_valence:     state.affect_valence,
                affect_arousal:     state.affect_arousal,
                actr_activation:    0.0,
                surprise_boost:     1.0,
                arousal_boost:      1.0,
                mood_congruence:    1.0,
                frustration_boost:  1.0,
                interference_factor: 1.0,
                spacing_boost:      1.0,
            });
        }
        hits.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
        hits.truncate(k);
        Ok(hits)
    }

    /// Bridge query: find entities active in [start_ms, end_ms] via EventTape, then
    /// recall their memories from time_idx. Unifies the action-event plane (EventTape)
    /// with the memory-content plane (time_idx) so temporal queries work without
    /// knowing the realm in advance.
    pub fn recall_temporal_events(
        &self,
        start_ms: i64,
        end_ms: i64,
        limit: usize,
    ) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") { return Ok(Vec::new()); }
        // Collect unique entity names from EventTape events in [start_ms, end_ms]
        let active_entities: Vec<String> = {
            let tape = self.event_tape.read();
            let mut seen = std::collections::HashSet::new();
            tape.events.iter()
                .filter(|e| e.ts_ms >= start_ms && e.ts_ms <= end_ms)
                .filter_map(|e| {
                    let name = tape.entity_name(e.entity_key).to_owned();
                    if seen.insert(name.clone()) { Some(name) } else { None }
                })
                .take(32)
                .collect()
        };

        let per_entity = (limit / active_entities.len().max(1)).max(1);
        let mut all_hits: Vec<RecallHit> = Vec::new();
        for entity in &active_entities {
            if let Ok(hits) = self.recall_temporal(start_ms, end_ms, Some(entity.as_str()), per_entity) {
                all_hits.extend(hits);
            }
        }
        // Also include realm-less global window (entity = None)
        if let Ok(hits) = self.recall_temporal(start_ms, end_ms, None, limit / 4) {
            all_hits.extend(hits);
        }

        // Deduplicate by memory_id, sort recency-first
        let mut seen_ids = std::collections::HashSet::new();
        all_hits.retain(|h| seen_ids.insert(h.memory_id));
        all_hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        all_hits.truncate(limit);
        Ok(all_hits)
    }

    /// Causal recall: return the last N events matching (tool, entity) as RecallHit stubs.
    /// Content field contains a human-readable description of the event sequence.
    pub fn recall_causal(&self, tool: &str, entity: &str, k: usize) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") || self.ablations.disabled("cdawg") { return Ok(Vec::new()); }
        let sym = {
            let mut tape = self.event_tape.write();
            tape.symbol_of(tool, entity, 0) // outcome=0 as probe; CDAWG walk ignores outcome bits via partial match
        };
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();

        // Try exact match first, then fall back to tool-only by zeroing entity_key bits.
        let turns = if let Some(state) = cdawg.walk(&[sym]) {
            cdawg.collect_endpos(state)
        } else {
            Vec::new()
        };

        let mut hits: Vec<RecallHit> = turns.iter().rev().take(k).filter_map(|&t| {
            let ev = tape.events.get(t as usize)?;
            let tool_name   = tape.tool_name(ev.tool_id);
            let entity_name = tape.entity_name(ev.entity_key);
            let outcome_str = match ev.outcome_class {
                0 => "success", 1 => "fail", 2 => "error", _ => "partial"
            };
            let content = format!(
                "[turn {}] {} on {} → {} (ts: {})",
                t, tool_name, entity_name, outcome_str, ev.ts_ms
            );
            Some(RecallHit {
                memory_id:          0,
                score:              1.0 - (t as f32 / (tape.events.len() as f32 + 1.0)),
                semantic_score:     0.0,
                ts_ms:              ev.ts_ms,
                kind:               "event".to_string(),
                realm:              "cec".to_string(),
                strength:           1.0,
                confidence:         1.0,
                access_count:       0,
                content,
                semantic_weight:    0.0,
                status_mul:         1.0,
                epistemic_mul:      1.0,
                strength_factor:    1.0,
                affect_valence:     0.0,
                affect_arousal:     0.0,
                actr_activation:    0.0,
                surprise_boost:     0.0,
                arousal_boost:      0.0,
                mood_congruence:    1.0,
                frustration_boost:  0.0,
                interference_factor:1.0,
                spacing_boost:      1.0,
            })
        }).collect();

        hits.sort_by(|a, b| b.ts_ms.cmp(&a.ts_ms));
        Ok(hits)
    }

    /// Return top-k failure patterns from the CDAWG as RecallHit stubs.
    pub fn recall_failure_pattern(&self, k: usize) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") || self.ablations.disabled("cdawg") { return Ok(Vec::new()); }
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();
        let patterns = cdawg.failure_patterns(3, k);
        let hits = patterns.into_iter().filter_map(|(state_id, fail_count, ratio)| {
            let turns = cdawg.collect_endpos(state_id);
            let last_ts = turns.iter()
                .filter_map(|&t| tape.events.get(t as usize).map(|e| e.ts_ms))
                .max()
                .unwrap_or(0);
            let content = format!(
                "[failure-pattern state={}] fail_count={} fail_ratio={:.2} last_seen_turn={}",
                state_id, fail_count, ratio, turns.iter().max().copied().unwrap_or(0)
            );
            Some(RecallHit {
                memory_id:          state_id as u64,
                score:              ratio * fail_count as f32,
                semantic_score:     0.0,
                ts_ms:              last_ts,
                kind:               "failure-pattern".to_string(),
                realm:              "cec".to_string(),
                strength:           ratio,
                confidence:         ratio,
                access_count:       fail_count,
                content,
                semantic_weight:    0.0,
                status_mul:         1.0,
                epistemic_mul:      1.0,
                strength_factor:    1.0,
                affect_valence:    -ratio,
                affect_arousal:     ratio,
                actr_activation:    0.0,
                surprise_boost:     0.0,
                arousal_boost:      0.0,
                mood_congruence:    1.0 - ratio,
                frustration_boost:  ratio,
                interference_factor:1.0,
                spacing_boost:      1.0,
            })
        }).collect();
        Ok(hits)
    }

    /// PMI-ranked causal antecedents: what actions typically precede (tool, entity)?
    /// Returns RecallHit stubs ranked by pointwise mutual information.
    pub fn recall_causal_antecedent(&self, tool: &str, entity: &str, k: usize) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") || self.ablations.disabled("cdawg") { return Ok(Vec::new()); }
        let sym = {
            let mut tape = self.event_tape.write();
            tape.symbol_of(tool, entity, 0)
        };
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();
        let antecedents = cdawg.causal_antecedents(&[sym], k, &tape);
        let hits = antecedents
            .into_iter()
            .enumerate()
            .map(|(i, (syms, count, pmi))| {
                let desc = syms
                    .iter()
                    .map(|&s| {
                        let tool_id    = (s >> 40) as u16;
                        let outcome_cl = ((s >> 32) & 0xff) as u8;
                        let entity_k   = (s & 0xffff_ffff) as u32;
                        let outcome_str = match outcome_cl {
                            0 => "success", 1 => "fail", 2 => "error", _ => "partial",
                        };
                        format!("{} on {} → {}", tape.tool_name(tool_id), tape.entity_name(entity_k), outcome_str)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let content = format!(
                    "[causal-antecedent rank={}] {} (count={} pmi={:.3})",
                    i + 1, desc, count, pmi
                );
                RecallHit {
                    memory_id:           i as u64,
                    score:               pmi.max(0.0),
                    semantic_score:      0.0,
                    ts_ms:               0,
                    kind:                "causal-antecedent".to_string(),
                    realm:               "cec".to_string(),
                    strength:            count as f32,
                    confidence:          (pmi / 10.0).clamp(0.0, 1.0),
                    access_count:        count,
                    content,
                    semantic_weight:     0.0,
                    status_mul:          1.0,
                    epistemic_mul:       1.0,
                    strength_factor:     1.0,
                    affect_valence:      0.0,
                    affect_arousal:      0.0,
                    actr_activation:     0.0,
                    surprise_boost:      0.0,
                    arousal_boost:       0.0,
                    mood_congruence:     1.0,
                    frustration_boost:   0.0,
                    interference_factor: 1.0,
                    spacing_boost:       1.0,
                }
            })
            .collect();
        Ok(hits)
    }

    /// Heteroassociative HDC recall: given a known role/value, infer the unknown role.
    /// known_role: "tool" | "entity" | "outcome"
    /// query_role: "tool" | "entity" | "outcome"
    pub fn recall_hdcbind(
        &self,
        known_role: &str,
        known_val: &str,
        query_role: &str,
        k: usize,
    ) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("episode_hdc") { return Ok(Vec::new()); }
        let results = self.episode_hdc.read().recall_hdcbind(known_role, known_val, query_role, k);
        let hits = results
            .into_iter()
            .enumerate()
            .map(|(i, (name, sim))| {
                let content = format!(
                    "[hdcbind] given {}={} → {}={} (sim={:.3})",
                    known_role, known_val, query_role, name, sim
                );
                RecallHit {
                    memory_id:           i as u64,
                    score:               sim,
                    semantic_score:      0.0,
                    ts_ms:               0,
                    kind:                "hdcbind".to_string(),
                    realm:               "cec".to_string(),
                    strength:            sim,
                    confidence:          sim,
                    access_count:        0,
                    content,
                    semantic_weight:     0.0,
                    status_mul:          1.0,
                    epistemic_mul:       1.0,
                    strength_factor:     1.0,
                    affect_valence:      0.0,
                    affect_arousal:      0.0,
                    actr_activation:     0.0,
                    surprise_boost:      0.0,
                    arousal_boost:       0.0,
                    mood_congruence:     1.0,
                    frustration_boost:   0.0,
                    interference_factor: 1.0,
                    spacing_boost:       1.0,
                }
            })
            .collect();
        Ok(hits)
    }

    /// Counterfactual recall: given (tool, entity, outcome), what alternative tools/entities
    /// would have had a lower failure rate in the same context?
    pub fn recall_counterfactual(
        &self,
        tool:    &str,
        entity:  &str,
        outcome: u8,
        k:       usize,
    ) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") || self.ablations.disabled("cdawg") { return Ok(Vec::new()); }
        use crate::organ::cdawg::CounterfactualHit;
        let taken_sym = self.event_tape.write().symbol_of(tool, entity, outcome);
        let context   = self.event_tape.read().last_n_syms(4);
        let tape      = self.event_tape.read();
        let cdawg     = self.cdawg.read();
        let hits: Vec<CounterfactualHit> = cdawg.counterfactual_alternatives(&context, taken_sym, 5, k);
        let result = hits.into_iter().enumerate().map(|(i, h)| {
            let tool_id   = (h.symbol >> 40) as u16;
            let out_cl    = ((h.symbol >> 32) & 0xff) as u8;
            let entity_k  = (h.symbol & 0xffff_ffff) as u32;
            let alt_tool  = tape.tool_name(tool_id);
            let alt_ent   = tape.entity_name(entity_k);
            let out_str   = match out_cl { 0=>"success",1=>"fail",2=>"error",_=>"partial" };
            let content = format!(
                "[counterfactual rank={}] use {}({}) → {} instead: fail_rate {:.0}% vs {:.0}% taken (Δ={:+.0}%, n={})",
                i+1, alt_tool, alt_ent, out_str,
                h.fail_ratio * 100.0, h.taken_fail_ratio * 100.0, h.delta * 100.0, h.support
            );
            RecallHit {
                memory_id:           i as u64,
                score:               h.delta.max(0.0),
                semantic_score:      0.0,
                ts_ms:               0,
                kind:                "counterfactual".to_string(),
                realm:               "cec".to_string(),
                strength:            h.delta.max(0.0),
                confidence:          1.0 - h.wilson_fail_lower,
                access_count:        h.support,
                content,
                semantic_weight:     0.0,
                status_mul:          1.0,
                epistemic_mul:       1.0,
                strength_factor:     1.0,
                affect_valence:      h.delta,
                affect_arousal:      h.delta.abs(),
                actr_activation:     0.0,
                surprise_boost:      0.0,
                arousal_boost:       0.0,
                mood_congruence:     1.0,
                frustration_boost:   0.0,
                interference_factor: 1.0,
                spacing_boost:       1.0,
            }
        }).collect();
        Ok(result)
    }

    /// Phase 16: CPU-native routed recall. Dispatches to the cheapest lane that
    /// can answer the request without an LLM call. Returns JSON with dispatch_label
    /// and the resulting recall hits.
    pub fn routed_recall(&self, req: RecallRequest) -> String {
        let router = QueryRouter::new();
        let dispatch = router.route(&req);
        let label = dispatch.label();
        let k = req.k.max(1);

        let hits: Vec<String> = match dispatch {
            DispatchKind::Exact => {
                let now = recall_now_ms();
                let ts = self.triplet_store.read();
                let entries = match (&req.subject, &req.predicate) {
                    (Some(s), _) => ts.query_subject(s, now),
                    (_, Some(p)) => ts.query_predicate(p, now),
                    _ => vec![],
                };
                entries.iter().take(k).map(|e| {
                    format!(r#"{{"subject":"{}","predicate":"{}","object":"{}","weight":{:.3}}}"#,
                        e.subject.replace('"', "\\\""),
                        e.predicate.replace('"', "\\\""),
                        e.object.replace('"', "\\\""),
                        e.weight)
                }).collect()
            }
            DispatchKind::Fuzzy => {
                self.recall_keyword(req.freetext.as_deref().unwrap_or(""), k)
                    .unwrap_or_default()
                    .iter().map(|h| format!(
                        r#"{{"memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )).collect()
            }
            DispatchKind::Temporal => {
                let from = req.time_from_ms.unwrap_or(0);
                let to   = req.time_to_ms.unwrap_or(i64::MAX);
                self.recall_temporal(from, to, None, k)
                    .unwrap_or_default()
                    .iter().map(|h| format!(
                        r#"{{"memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )).collect()
            }
            DispatchKind::Causal => {
                let tool   = req.causal_tool.as_deref().unwrap_or("");
                let entity = req.causal_entity.as_deref().unwrap_or("");
                self.recall_causal(tool, entity, k)
                    .unwrap_or_default()
                    .iter().map(|h| format!(
                        r#"{{"memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )).collect()
            }
            DispatchKind::Hybrid => {
                // Exact lane first, fuzzy fills remaining slots.
                let now = recall_now_ms();
                let ts = self.triplet_store.read();
                let mut out: Vec<String> = match (&req.subject, &req.predicate) {
                    (Some(s), _) => ts.query_subject(s, now),
                    (_, Some(p)) => ts.query_predicate(p, now),
                    _ => vec![],
                }.iter().take(k / 2 + 1).map(|e| {
                    format!(r#"{{"lane":"exact","subject":"{}","predicate":"{}","object":"{}"}}"#,
                        e.subject.replace('"', "\\\""),
                        e.predicate.replace('"', "\\\""),
                        e.object.replace('"', "\\\""))
                }).collect();
                drop(ts);
                if out.len() < k {
                    let fuzzy = self.recall_keyword(
                        req.freetext.as_deref().unwrap_or(""), k - out.len()
                    ).unwrap_or_default();
                    out.extend(fuzzy.iter().map(|h| format!(
                        r#"{{"lane":"fuzzy","memory_id":{},"score":{:.4},"content":"{}"}}"#,
                        h.memory_id, h.score,
                        h.content.replace('"', "\\\"").chars().take(120).collect::<String>()
                    )));
                }
                out
            }
            DispatchKind::NeedsDisambiguation(slots) => {
                let slot_json: Vec<String> = slots.iter().map(|s| {
                    format!(r#"{{"slot":"{}","context":"{}"}}"#,
                        s.name, s.context.replace('"', "\\\""))
                }).collect();
                return format!(
                    r#"{{"dispatch":"needs_disambiguation","unbound_slots":[{}],"hits":[]}}"#,
                    slot_json.join(",")
                );
            }
        };

        format!(
            r#"{{"dispatch":"{label}","token_cost":0,"hits":[{}]}}"#,
            hits.join(",")
        )
    }

    /// Return top-k CDAWG states reachable from (tool, entity) ranked by Q-value.
    pub fn recall_motif_value(&self, tool: &str, entity: &str, k: usize) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") || self.ablations.disabled("cdawg") { return Ok(Vec::new()); }
        let sym = {
            let mut tape = self.event_tape.write();
            tape.symbol_of(tool, entity, 0)
        };
        let tape  = self.event_tape.read();
        let cdawg = self.cdawg.read();
        let rows = cdawg.top_q_states(&[sym], k.max(1));
        let hits = rows.into_iter().map(|(state_id, q_val, support)| {
            let next_syms: Vec<String> = cdawg.states.get(state_id as usize)
                .map(|s| s.transitions.keys().take(3).map(|&sym| {
                    let tn = tape.tool_name((sym >> 40) as u16);
                    let en = tape.entity_name((sym & 0xffff_ffff) as u32);
                    format!("{tn}({en})")
                }).collect())
                .unwrap_or_default();
            let content = format!(
                "[motif state={}] q={:.3} support={} next=[{}]",
                state_id, q_val, support, next_syms.join(", ")
            );
            RecallHit {
                memory_id:           state_id as u64,
                score:               (q_val + 1.0) / 2.0,
                semantic_score:      0.0,
                ts_ms:               0,
                kind:                "motif".to_string(),
                realm:               "cec".to_string(),
                strength:            (q_val + 1.0) / 2.0,
                confidence:          support as f32 / (support as f32 + 1.0),
                access_count:        support,
                content,
                semantic_weight:     0.0,
                status_mul:          1.0,
                epistemic_mul:       1.0,
                strength_factor:     1.0,
                affect_valence:      q_val.clamp(-1.0, 1.0),
                affect_arousal:      q_val.abs().clamp(0.0, 1.0),
                actr_activation:     0.0,
                surprise_boost:      0.0,
                arousal_boost:       0.0,
                mood_congruence:     (q_val + 1.0) / 2.0,
                frustration_boost:   0.0,
                interference_factor: 1.0,
                spacing_boost:       1.0,
            }
        }).collect();
        Ok(hits)
    }

    /// True counterfactual recall: use DecisionTape to find cases where (tool, entity) was
    /// explicitly considered and rejected, and report the chosen alternative's outcome.
    pub fn recall_true_counterfactual(
        &self, tool: &str, entity: &str, outcome: u8, k: usize,
    ) -> Result<Vec<RecallHit>> {
        if self.ablations.disabled("event_tape") || self.ablations.disabled("decision_tape") { return Ok(Vec::new()); }
        let sym = self.event_tape.read().symbol_of_ro(tool, entity, outcome);
        let etape = self.event_tape.read();
        let tape = self.decision_tape.read();
        let hits = tape.rejected_alternatives(sym, k)
            .into_iter()
            .enumerate()
            .map(|(i, (dp, reason_u8))| {
                let reason = crate::organ::decision_tape::RejectionReason::from_u8(reason_u8);
                let chosen_tool_id  = (dp.chosen_sym >> 40) as u16;
                let chosen_entity_k = (dp.chosen_sym & 0xffff_ffff) as u32;
                let chosen_tool_name   = etape.tool_name(chosen_tool_id);
                let chosen_entity_name = etape.entity_name(chosen_entity_k);
                let content = format!(
                    "[counterfactual turn={}] rejected {}({}) reason={} → chose {}({}) confidence_delta={:.3}",
                    dp.turn_id, tool, entity, reason.label(),
                    chosen_tool_name, chosen_entity_name, dp.confidence_delta
                );
                RecallHit {
                    memory_id:           dp.turn_id as u64,
                    score:               1.0 - i as f32 * 0.1,
                    semantic_score:      0.0,
                    ts_ms:               dp.ts_ms,
                    kind:                "counterfactual".to_string(),
                    realm:               "cec".to_string(),
                    strength:            dp.confidence_delta.abs(),
                    confidence:          dp.confidence_delta.abs().min(1.0),
                    access_count:        0,
                    content,
                    semantic_weight:     0.0,
                    status_mul:          1.0,
                    epistemic_mul:       1.0,
                    strength_factor:     1.0,
                    affect_valence:      dp.confidence_delta.clamp(-1.0, 1.0),
                    affect_arousal:      dp.confidence_delta.abs().clamp(0.0, 1.0),
                    actr_activation:     0.0,
                    surprise_boost:      0.0,
                    arousal_boost:       0.0,
                    mood_congruence:     1.0,
                    frustration_boost:   0.0,
                    interference_factor: 1.0,
                    spacing_boost:       1.0,
                }
            })
            .collect();
        Ok(hits)
    }

    /// Keyword (BM25) recall with affective context.
    pub fn recall_keyword_ctx(
        &self,
        query: &str,
        k: usize,
        query_valence: Option<f32>,
        query_arousal: Option<f32>,
        realm: Option<&str>,
        strengthen: bool,
    ) -> Result<Vec<RecallHit>> {
        let max_query_idf = self.keyword_idx.read().query_max_idf(query);
        // Realm-scoped queries need a larger global BM25 fetch so small-realm
        // memories aren't squeezed out by cross-realm hits in the global corpus.
        let bm25_fetch = if realm.is_some() { k * 12 } else { k * 3 };
        let keyword_hits = self.keyword_idx.read().search(query, bm25_fetch);

        let now = recall_now_ms();
        let payloads = self.payloads.read();
        let states = self.states.read();
        let learners = self.learners.read();
        let pipeline = self.scoring_pipeline.read();
        let ack_scores = self.ack_scores.read();
        let mut util_rng = utility_rng(now);

        let mut hits: Vec<RecallHit> = keyword_hits
            .into_iter()
            .filter_map(|hit| {
                let state = states.get(&hit.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&hit.memory_id)?;
                let content_str = String::from_utf8(payload.content.clone()).unwrap_or_default();
                if content_str.trim().is_empty() {
                    return None;
                }
                if payload.realm.starts_with("soul:") {
                    return None;
                }
                // Realm scoping: the BM25 lane must not leak other projects' memories into a
                // realm-scoped recall (the cross-realm injection bleed).
                if let Some(want) = realm {
                    if payload.realm != want {
                        return None;
                    }
                }
                let ctx = ScoringContext {
                    relevance_score: hit.bm25_score,
                    recall_mode: RecallMode::Keyword,
                    state,
                    kind: &payload.kind,
                    realm: &payload.realm,
                    realm_reliability: if self.ablations.disabled("learners") { 1.0 } else { learners.domain_reliability.reliability(&payload.realm) },
                    now_ms: now,
                    query_valence,
                    query_arousal,
                    prediction_prob: None,
                    surprise_role: None,
                    has_open_debt: false,
                    integration_weight: None,
                    ack_score: ack_scores.get(&hit.memory_id).copied().unwrap_or(0),
                    max_query_idf,
                };
                let (score, decomp) = pipeline.score(&ctx)?;
                let score = score * utility_multiplier(state, &mut util_rng);
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: hit.memory_id,
                    score,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: content_str,
                    semantic_weight: decomp.semantic_weight,
                    status_mul: decomp.status_mul,
                    epistemic_mul: decomp.epistemic_mul,
                    strength_factor: decomp.strength_factor,
                    affect_valence: state.affect_valence,
                    affect_arousal: state.affect_arousal,
                    actr_activation: decomp.actr_activation,
                    surprise_boost: decomp.surprise_boost,
                    arousal_boost: decomp.arousal_boost,
                    mood_congruence: decomp.mood_congruence,
                    frustration_boost: decomp.frustration_boost,
                    interference_factor: decomp.interference_factor,
                    spacing_boost: decomp.spacing_boost,
                })
            })
            .collect();

        hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id))
        });
        hits.truncate(k);

        let hit_ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
        drop(states);
        drop(payloads);
        drop(pipeline);
        drop(learners);
        drop(ack_scores);
        if strengthen {
            self.enqueue_recall_effects(&hit_ids);
        }

        Ok(hits)
    }

    /// Session-level recall: aggregates chunk-level hits per source_session using noisy-OR.
    /// Returns sessions ranked by combined evidence strength.
    /// `query_embedding` — pre-computed by caller (C++ embed layer); None skips semantic lane.
    pub fn recall_session(
        &self,
        query_embedding: Option<&[f32]>,
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<crate::recall::SessionRecallHit>> {
        use std::collections::HashMap;
        use crate::recall::SessionRecallHit;

        // Fetch candidate chunks from both semantic and keyword lanes
        let fetch_limit = k * 20;
        let mut candidates: Vec<crate::recall::RecallHit> = if let Some(emb) = query_embedding {
            self.recall_semantic_ctx(emb, fetch_limit, realm, None, None, true)?
        } else {
            Vec::new()
        };

        // Merge in keyword hits, deduplicating by memory_id (keep max score)
        let kw_hits = self.recall_keyword_ctx(query_text, fetch_limit, None, None, None, true)?;
        let mut seen: std::collections::HashSet<crate::ids::MemoryId> =
            candidates.iter().map(|h| h.memory_id).collect();
        for h in kw_hits {
            if seen.insert(h.memory_id) {
                candidates.push(h);
            }
        }

        // Group by source_session; skip memories without a session
        let payloads = self.payloads.read();
        struct SessionAcc {
            scores: Vec<f32>,
            best_score: f32,
            best_content: String,
            realm: String,
        }
        let mut sessions: HashMap<String, SessionAcc> = HashMap::new();

        for hit in &candidates {
            if let Some(payload) = payloads.get(&hit.memory_id) {
                if let Some(ref sid) = payload.source_session {
                    let acc = sessions.entry(sid.clone()).or_insert_with(|| SessionAcc {
                        scores: Vec::new(),
                        best_score: 0.0,
                        best_content: String::new(),
                        realm: payload.realm.clone(),
                    });
                    acc.scores.push(hit.score);
                    if hit.score > acc.best_score {
                        acc.best_score = hit.score;
                        acc.best_content = hit.content.clone();
                    }
                }
            }
        }
        drop(payloads);

        // Score: max_chunk_score dominates; small noisy-OR bonus from remaining evidence.
        // Avoids multi-mediocre-chunk sessions beating a single high-score gold chunk.
        let mut session_hits: Vec<SessionRecallHit> = sessions
            .into_iter()
            .map(|(session_id, mut acc)| {
                acc.scores.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let max_s = acc.scores[0].min(1.0).max(0.0);
                // noisy-OR of chunks beyond the best (corroborating evidence only)
                let corroboration = acc.scores.iter().skip(1).take(4).fold(0.0f32, |combined, &s| {
                    1.0 - (1.0 - combined) * (1.0 - s.min(1.0).max(0.0))
                });
                let session_score = max_s + 0.15 * corroboration * (1.0 - max_s);
                SessionRecallHit {
                    session_id,
                    score: session_score,
                    chunk_count: acc.scores.len() as u32,
                    max_chunk_score: acc.best_score,
                    best_evidence: acc.best_content,
                    realm: acc.realm,
                }
            })
            .collect();

        session_hits.sort_unstable_by(|a, b| {
            b.score.total_cmp(&a.score).then_with(|| a.session_id.cmp(&b.session_id))
        });
        session_hits.truncate(k);
        Ok(session_hits)
    }

    pub fn recall_with_fallback(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        self.recall_with_fallback_windowed(query_embedding, query_text, k, realm, None)
    }

    /// Time-windowed hybrid recall: the window GATES candidates (authored_at_ms
    /// membership in time_idx), semantic relevance still RANKS. Never
    /// recency-sorts — recency-ordered temporal recall floods context with
    /// operationally-fresh noise (stored correction). window=None is exactly
    /// recall_with_fallback.
    pub fn recall_with_fallback_windowed(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
        window: Option<(i64, i64)>,
    ) -> Result<Vec<RecallHit>> {
        // Window membership set, built from the temporal B-tree and dropped
        // before any lane runs (taken alone: no lock-ordering pair created).
        let allowed: Option<std::collections::HashSet<MemoryId>> = window.map(|(f, t)| {
            self.time_idx
                .read()
                .range_query(f, t, realm, usize::MAX)
                .into_iter()
                .map(|e| e.memory_id)
                .collect()
        });
        let gate = |v: Vec<RecallHit>| -> Vec<RecallHit> {
            match &allowed {
                Some(a) => v.into_iter().filter(|h| a.contains(&h.memory_id)).collect(),
                None => v,
            }
        };
        // Windowed queries over-fetch harder: the gate discards out-of-window
        // candidates, so lanes need more raw width to fill k.
        let window_mul: usize = if window.is_some() { 2 } else { 1 };
        // Stratified recall: cap any one realm on UNSCOPED queries so a dominant
        // realm (e.g. compliance:auto BM25 noise) can't flood results. The cap is
        // Thompson-sampled per realm from its Beta posterior (G8) so reliable
        // realms earn more slots. Targeted queries (realm=Some) are never reshaped.
        let stratify = realm.is_none();
        // Over-fetch the lane when capping so the cap can backfill from
        // non-dominant realms and still return up to k. The over-fetch factor
        // tracks the legacy divisor knob purely as a fetch-width hint.
        let fetch_k = if stratify {
            let factor = self
                .scoring_pipeline
                .read()
                .config
                .recall_realm_cap_divisor
                .max(1);
            k.saturating_mul(factor).max(k).saturating_mul(window_mul)
        } else {
            // Over-fetch globally so realm post-filter has enough candidates.
            let mul = self.scoring_pipeline.read().config.dam_fetch_mul.max(4);
            k.saturating_mul(mul).max(k).saturating_mul(window_mul)
        };
        // Snapshot the per-realm reliability learner so the stratify pass can
        // Thompson-sample without holding the learner lock during scoring.
        let reliability = if stratify && !self.ablations.disabled("learners") {
            Some(self.learners.read().domain_reliability.clone())
        } else {
            None
        };

        // RRF hybrid: on UNSCOPED queries, fuse semantic + BM25 ranks instead of
        // using BM25 only as an empty-semantic fallback. Targeted queries
        // (realm=Some) keep the legacy semantic-then-fallback path.
        let use_rrf = self.scoring_pipeline.read().config.use_rrf;
        if use_rrf {
            let rrf_k = self.scoring_pipeline.read().config.rrf_k;
            // For realm-scoped queries, use global HNSW then post-filter instead of
            // filtered HNSW (search_candidates). search_candidates skips memories whose
            // flat embedding is absent even when the HNSW graph node has it, causing
            // false sim=0 for recently-ingested memories in small realms.
            let sem = if !stratify {
                let global = self.recall_semantic(query_embedding, fetch_k, None)?;
                if let Some(r) = realm {
                    let rm = self.realm_members.read();
                    let allowed = rm.get(r);
                    global
                        .into_iter()
                        .filter(|h| allowed.map(|a| a.contains(&h.memory_id)).unwrap_or(true))
                        .collect()
                } else {
                    global
                }
            } else {
                self.recall_semantic(query_embedding, fetch_k, realm)?
            };
            let sem = gate(sem);
            let kw = gate(self.recall_keyword_realm(query_text, fetch_k, realm)?);
            if !sem.is_empty() || !kw.is_empty() {
                let mut merged = rrf_merge(sem, kw, fetch_k, rrf_k);
                // PPR injection lane: seed from the top fused hits and RRF-merge
                // the association-graph stationary ranking back in, injecting
                // topic-bridged multi-hop memories. Gated by CHITTA_PPR_LANE.
                self.ppr_inject(&mut merged);

                // Cortical SDR re-rank: second-pass RRF over the already-merged candidate set.
                // Gated by use_cortical (default false) and a non-empty cortical index.
                // Re-ranker shape: cortical can only reorder merged candidates, never inject new ones.
                let use_cortical = self.scoring_pipeline.read().config.use_cortical;
                if use_cortical && !self.ablations.disabled("sparse_encoder") && !self.cortical_idx.read().is_empty() {
                    let cortical_rrf_k = self.scoring_pipeline.read().config.cortical_rrf_k;
                    let code = self.sparse_encoder.read().encode(query_embedding);
                    if !code.is_empty() {
                        let candidate_ids: std::collections::HashSet<MemoryId> =
                            merged.iter().map(|h| h.memory_id).collect();
                        let cortical_ranked: Vec<RecallHit> = {
                            let cortical = self.cortical_idx.read();
                            let by_id: std::collections::HashMap<MemoryId, &RecallHit> =
                                merged.iter().map(|h| (h.memory_id, h)).collect();
                            cortical
                                .search(&code, merged.len(), Some(&candidate_ids))
                                .into_iter()
                                .filter_map(|(mid, _)| by_id.get(&mid).map(|h| (*h).clone()))
                                .collect()
                        };
                        if !cortical_ranked.is_empty() {
                            merged = rrf_merge(merged, cortical_ranked, fetch_k, cortical_rrf_k);
                        }
                    }
                }

                // Stratify only for unscoped queries — targeted queries are already realm-filtered.
                let mut out = if stratify {
                    stratify_recall_hits(merged, k, reliability.as_ref())
                } else {
                    merged.into_iter().take(k).collect()
                };
                // Starvation backfill: the window gate can leave the lanes short.
                // Fill from window members ranked BY COSINE to the query — the
                // window gates, semantic still ranks.
                if out.len() < k {
                    if let Some((f, t)) = window {
                        self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
                    }
                }
                return Ok(out);
            }
            // Both lanes empty → fall through to recency below.
            let store_size = self.memory_count();
            if store_size < 10 {
                return Ok(vec![]);
            }
            if let Some((f, t)) = window {
                // Windowed query with empty lanes: rank window members by cosine,
                // never by recency.
                let mut out = Vec::new();
                self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
                return Ok(out);
            }
            log::warn!("RRF hybrid empty (semantic+BM25), falling back to recency");
            let now = recall_now_ms();
            let temporal = self.recall_temporal(0, now, realm, fetch_k)?;
            return Ok(if stratify {
                stratify_recall_hits(temporal, k, reliability.as_ref())
            } else {
                temporal.into_iter().take(k).collect()
            });
        }

        let hits = gate(self.recall_semantic(query_embedding, fetch_k, realm)?);
        if !hits.is_empty() {
            let mut out = stratify_recall_hits(hits, k, reliability.as_ref());
            if out.len() < k {
                if let Some((f, t)) = window {
                    self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
                }
            }
            return Ok(out);
        }

        let store_size = self.memory_count();
        if store_size < 10 {
            return Ok(vec![]);
        }

        if let Some((f, t)) = window {
            let mut out = Vec::new();
            self.window_backfill(&mut out, k, f, t, realm, query_embedding)?;
            return Ok(out);
        }

        log::warn!(
            "recall_semantic returned empty with {} memories, falling back to BM25",
            store_size
        );
        let bm25_hits = self.recall_keyword(query_text, fetch_k)?;
        if !bm25_hits.is_empty() {
            return Ok(stratify_recall_hits(bm25_hits, k, reliability.as_ref()));
        }

        log::warn!("BM25 fallback also empty, falling back to recency");
        let now = recall_now_ms();
        let temporal = self.recall_temporal(0, now, realm, fetch_k)?;
        Ok(stratify_recall_hits(temporal, k, reliability.as_ref()))
    }

    /// Fill `out` up to `k` with window members ranked by cosine similarity to
    /// the query embedding (embeddings are L2-normalized: dot == cosine).
    /// recall_temporal builds the payload-backed hits; its recency ORDER is
    /// discarded — cosine ranks. Cost bounded: at most 4k window members scored.
    pub(super) fn window_backfill(
        &self,
        out: &mut Vec<RecallHit>,
        k: usize,
        from_ms: i64,
        to_ms: i64,
        realm: Option<&str>,
        query_embedding: &[f32],
    ) -> Result<()> {
        let have: std::collections::HashSet<MemoryId> =
            out.iter().map(|h| h.memory_id).collect();
        let mut pool: Vec<RecallHit> = self
            .recall_temporal(from_ms, to_ms, realm, k.saturating_mul(4).max(64))?
            .into_iter()
            .filter(|h| !have.contains(&h.memory_id))
            .collect();
        for h in pool.iter_mut() {
            h.semantic_score = self
                .embedding_of(h.memory_id)
                .map(|e| e.iter().zip(query_embedding).map(|(a, b)| a * b).sum())
                .unwrap_or(0.0);
            h.score = h.semantic_score;
        }
        pool.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
        out.extend(pool.into_iter().take(k - out.len()));
        Ok(())
    }

    /// Field-RAG / Modern Hopfield recall with optional multi-hop expansion.
    ///
    /// Each hop: fetch candidates via RRF, run T-step DAM relaxation over the
    /// candidate submatrix (s(t+1) = X @ softmax(β·Xᵀs(t))), re-rank by cosine(s_T, X).
    /// With dam_hops > 1: use s_T as the refined query vector for the next hop,
    /// excluding already-fetched IDs. Final output is the union re-sorted by score.
    pub fn recall_field(
        &self,
        query_embedding: &[f32],
        query_text: &str,
        k: usize,
        realm: Option<&str>,
    ) -> Result<Vec<RecallHit>> {
        let (beta, steps, fetch_mul, hops) = {
            let cfg = self.scoring_pipeline.read();
            (cfg.config.dam_beta, cfg.config.dam_steps, cfg.config.dam_fetch_mul,
             cfg.config.dam_hops.max(1))
        };
        let fetch_k = k.saturating_mul(fetch_mul).max(k);
        let dim = query_embedding.len();

        let mut all_hits: Vec<RecallHit> = Vec::new();
        let mut query: Vec<f32> = query_embedding.to_vec();

        for _hop in 0..hops {
            // Exclude IDs already collected in prior hops.
            let seen: std::collections::HashSet<u64> =
                all_hits.iter().map(|h| h.memory_id).collect();

            let mut hits = self.recall_with_fallback(&query, query_text, fetch_k, realm)?;
            if _hop > 0 {
                hits.retain(|h| !seen.contains(&h.memory_id));
            }
            if hits.len() < 2 {
                all_hits.extend(hits);
                break;
            }

            // Collect embeddings for the candidate submatrix X.
            let embeddings: Vec<(usize, Vec<f32>)> = {
                let idx = self.semantic_idx.read();
                hits.iter()
                    .enumerate()
                    .filter_map(|(i, h)| {
                        idx.get_embedding(h.memory_id).map(|e| (i, e.to_vec()))
                    })
                    .collect()
            };
            if embeddings.len() < 2 {
                all_hits.extend(hits);
                break;
            }

            // Build a 4-bit TurboQuant index over the candidate submatrix once
            // per hop. turbovec inner-product over unit vectors == cosine, which
            // matches the scalar dots below (stored embs are unit). Used for the
            // DAM energies (all candidates, k = n) and the final re-rank. Falls
            // back to scalar when construction is unavailable (dim % 8 != 0).
            let turbo: Option<(turbovec::TurboQuantIndex, Vec<usize>)> = (|| {
                let n = embeddings.len();
                if dim == 0 || dim % 8 != 0 { return None; }
                let mut idx = turbovec::TurboQuantIndex::new(dim, 4).ok()?;
                let mut flat = Vec::with_capacity(n * dim);
                let mut row_hit: Vec<usize> = Vec::with_capacity(n);
                for (i, emb) in &embeddings {
                    let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm < 1e-12 { return None; }
                    flat.extend(emb.iter().map(|x| x / norm));
                    row_hit.push(*i);
                }
                idx.add(&flat);
                idx.prepare();
                Some((idx, row_hit))
            })();

            // T-step DAM relaxation.
            let mut s: Vec<f32> = query.clone();
            for _ in 0..steps {
                // Energies = beta * (X · s). turbovec needs a unit query to
                // return cosine; `s` is unit after the first step's renormalize,
                // but the initial `s = query` may not be — normalize for scoring.
                let n = embeddings.len();
                let mut energies: Vec<f32> = vec![0.0; n];
                let mut max_e = f32::NEG_INFINITY;
                let scored = turbo.as_ref().and_then(|(idx, row_hit)| {
                    let sn = {
                        let nrm = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if nrm < 1e-12 { return None; }
                        s.iter().map(|x| x / nrm).collect::<Vec<f32>>()
                    };
                    let res = idx.search(&sn, n);
                    let idxs = res.indices_for_query(0);
                    let scs = res.scores_for_query(0);
                    for (row, sc) in idxs.iter().zip(scs.iter()) {
                        if *row < 0 { continue; }
                        // row indexes the submatrix; row_hit maps it to the
                        // candidate slot (== position in `embeddings`).
                        let pos = *row as usize;
                        if let Some(slot) = row_hit.get(pos).copied() {
                            // find energies position: energies is keyed by
                            // embeddings-vec order, which equals row order.
                            let _ = slot;
                        }
                        energies[pos] = beta * *sc;
                    }
                    Some(())
                });
                if scored.is_none() {
                    for (j, (_, emb)) in embeddings.iter().enumerate() {
                        energies[j] = beta * emb.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    }
                }
                for &e in &energies { if e > max_e { max_e = e; } }

                let sum_exp: f32 = energies.iter().map(|&e| (e - max_e).exp()).sum();
                let weights: Vec<f32> =
                    energies.iter().map(|&e| (e - max_e).exp() / sum_exp).collect();

                let mut s_new = vec![0.0f32; dim];
                for (j, (_, emb)) in embeddings.iter().enumerate() {
                    let w = weights[j];
                    for (sn, &xv) in s_new.iter_mut().zip(emb.iter()) {
                        *sn += w * xv;
                    }
                }

                let norm = s_new.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm < 1e-12 { break; }
                for x in &mut s_new { *x /= norm; }
                let delta: f32 = s_new
                    .iter().zip(s.iter()).map(|(&a, &b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                s = s_new;
                if delta < 1e-6 { break; }
            }

            // Re-rank this hop's hits by cosine(s_T, X_j).
            // cosine(s_T, X_j) via turbovec when available, else scalar.
            let final_scored = turbo.as_ref().and_then(|(idx, _)| {
                let nrm = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                if nrm < 1e-12 { return None; }
                let sn: Vec<f32> = s.iter().map(|x| x / nrm).collect();
                let res = idx.search(&sn, embeddings.len());
                let idxs = res.indices_for_query(0);
                let scs = res.scores_for_query(0);
                for (row, sc) in idxs.iter().zip(scs.iter()) {
                    if *row < 0 { continue; }
                    let pos = *row as usize;
                    if let Some((i, _)) = embeddings.get(pos) {
                        hits[*i].score = *sc;
                        hits[*i].semantic_score = *sc;
                    }
                }
                Some(())
            });
            if final_scored.is_none() {
                for (i, emb) in &embeddings {
                    let cos = emb.iter().zip(s.iter()).map(|(&a, &b)| a * b).sum::<f32>();
                    hits[*i].score = cos;
                    hits[*i].semantic_score = cos;
                }
            }
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
            all_hits.extend(hits);

            // s_T becomes the query for the next hop.
            query = s;
        }

        // Merge: sort by score, dedup keeping highest, truncate.
        all_hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.memory_id.cmp(&b.memory_id)));
        let mut seen = std::collections::HashSet::new();
        all_hits.retain(|h| seen.insert(h.memory_id));
        Ok(all_hits.into_iter().take(k).collect())
    }

    /// Recall memories associated with a file path (exact match).
    pub fn recall_artifact(&self, path: &str, limit: usize) -> Result<Vec<RecallHit>> {
        let entries = self.artifact_idx.read().query_path(path, limit);
        let now = recall_now_ms();
        let payloads = self.payloads.read();
        let states = self.states.read();

        let hits = entries
            .into_iter()
            .filter_map(|entry| {
                let state = states.get(&entry.memory_id)?;
                if state.deleted {
                    return None;
                }
                let payload = payloads.get(&entry.memory_id)?;
                if payload.content.is_empty() {
                    return None;
                }
                let eff_strength = state.effective_strength(now);
                Some(RecallHit {
                    memory_id: entry.memory_id,
                    score: entry.strength * eff_strength * state.confidence,
                    semantic_score: 0.0,
                    ts_ms: payload.authored_at_ms,
                    kind: payload.kind.clone(),
                    realm: payload.realm.clone(),
                    strength: eff_strength,
                    confidence: state.confidence,
                    access_count: state.access_count,
                    content: String::from_utf8(payload.content.clone()).unwrap_or_default(),
                    semantic_weight: 0.0,
                    status_mul: 0.0,
                    epistemic_mul: 0.0,
                    strength_factor: 0.0,
                    affect_valence: 0.0,
                    affect_arousal: 0.0,
                    actr_activation: 0.0,
                    surprise_boost: 1.0,
                    arousal_boost: 1.0,
                    mood_congruence: 1.0,
                    frustration_boost: 1.0,
                    interference_factor: 1.0,
                    spacing_boost: 1.0,
                })
            })
            .collect();

        Ok(hits)
    }

    /// Extract entity seeds from a query string: capitalized words (≥3 chars),
    /// @tag references, and double-quoted strings.
    pub(super) fn extract_seeds(query: &str) -> Vec<String> {
        let mut seeds: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        // @tag references
        for cap in query.split_whitespace() {
            let w = cap.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '_');
            if w.starts_with('@') && w.len() > 1 {
                let tag = w[1..].to_string();
                if seen.insert(tag.clone()) { seeds.push(tag); }
            }
        }
        // Quoted strings
        let mut in_quote = false;
        let mut buf = String::new();
        for c in query.chars() {
            if c == '"' {
                if in_quote && !buf.trim().is_empty() {
                    let s = buf.trim().to_string();
                    if seen.insert(s.clone()) { seeds.push(s); }
                    buf.clear();
                }
                in_quote = !in_quote;
            } else if in_quote {
                buf.push(c);
            }
        }
        // Capitalized words ≥3 chars (skip first word of query which may be sentence-start)
        let words: Vec<&str> = query.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let w = word.trim_matches(|c: char| !c.is_alphabetic());
            if w.len() < 3 { continue; }
            let mut chars = w.chars();
            if let Some(first) = chars.next() {
                if first.is_uppercase() && i > 0 {
                    if seen.insert(w.to_string()) { seeds.push(w.to_string()); }
                }
            }
        }
        seeds
    }

    /// Spreading-activation recall: traverse triplet graph from query entities,
    /// return top-k memories ranked by accumulated activation.
    pub fn recall_spreading(
        &self,
        query: &str,
        k: usize,
        realm: Option<&str>,
        max_nodes: usize,
        max_entries_per_entity: usize,
        depth: u8,
    ) -> Vec<SpreadingRecallHit> {
        let seeds = Self::extract_seeds(query);
        if seeds.is_empty() { return Vec::new(); }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let memory_scores = match self.triplet_store.try_read_for(std::time::Duration::from_secs(5)) {
            Some(ts) => ts.spreading_activation(
                &seeds, depth, 0.6, now_ms, max_nodes, max_entries_per_entity),
            None => return Vec::new(),
        };
        if memory_scores.is_empty() { return Vec::new(); }

        // Sort by score descending, take top k
        let mut ranked: Vec<(MemoryId, f32)> = memory_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(k * 4); // fetch extra for realm filtering

        let payloads = match self.payloads.try_read_for(std::time::Duration::from_secs(5)) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let mut results: Vec<SpreadingRecallHit> = Vec::new();
        for (mid, score) in ranked {
            if let Some(p) = payloads.get(&mid) {
                if let Some(r) = realm {
                    if p.realm.as_str() != r { continue; }
                }
                results.push(SpreadingRecallHit {
                    memory_id: mid,
                    score,
                    text: String::from_utf8_lossy(&p.content).chars().take(300).collect::<String>(),
                    kind: p.kind.clone(),
                    realm: p.realm.clone(),
                });
                if results.len() >= k { break; }
            }
        }
        results
    }

}
