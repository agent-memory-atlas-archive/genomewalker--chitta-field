use super::*;
use tempfile::TempDir;

fn open_test_field() -> (ChittaField, TempDir) {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let field = ChittaField::open(data_dir).unwrap();
    (field, tmp)
}

// Direct tag selection is global. Callers must scope the selected payloads,
// using the same exact-realm rule as semantic and keyword recall.
#[test]
fn test_tag_selection_realm_contract() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let mut ids = Vec::new();
    for realm in ["project:a", "project:b", "brahman"] {
        let (id, _) = field.put_memory("wisdom", realm,
            b"tagrealmcanary shared fact", &emb, 1.0, 0.001, 0,
            vec![], None, None).unwrap();
        field.add_triplet(id.to_string(), "tagged".into(), "shared-tag".into(),
            1.0, None, None).unwrap();
        ids.push(id);
    }
    let tagged = field.query_object("shared-tag").unwrap();
    assert_eq!(tagged.len(), 3);
    for (realm, expected) in ["project:a", "project:b", "brahman"].iter().zip(ids) {
        let selected: Vec<_> = tagged.iter().filter_map(|t| {
            let id = t.subject.parse::<u64>().ok()?;
            let payload = field.get_memory(id).ok()?;
            (payload.realm == *realm).then_some(id)
        }).collect();
        assert_eq!(selected, vec![expected]);
        let semantic = field.recall_semantic_measure(&emb, 10, Some(realm)).unwrap();
        let keyword = field.recall_keyword_measure("tagrealmcanary", 10, Some(realm)).unwrap();
        assert!(!semantic.is_empty() && !keyword.is_empty());
        assert!(semantic.iter().chain(keyword.iter()).all(|h| h.realm == *realm));
    }
}

#[test]
fn test_put_get_roundtrip() {
    let (field, _tmp) = open_test_field();
    let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id, hash) = field
        .put_memory(
            "wisdom",
            "test",
            b"hello world",
            &embedding,
            0.9,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();

    let payload = field.get_memory(id).unwrap();
    assert_eq!(payload.content, b"hello world");
    assert_eq!(payload.kind, "wisdom");
    assert_eq!(payload.chunk_hash, hash);
}

// Windowed hybrid recall: the ts window GATES (authored_at_ms membership),
// semantic similarity RANKS. In-window low-relevance noise must not outrank
// in-window relevant hits, and out-of-window hits must never appear.
#[test]
fn test_windowed_recall_gates_by_time_ranks_by_semantic() {
    let (field, _tmp) = open_test_field();
    let day = 86_400_000i64;
    let now = now_ms();
    // Same embedding direction = relevant; orthogonal = noise.
    let mut rel = vec![0.0f32; crate::ops::EMBED_DIM];
    rel[0] = 1.0;
    let mut noise = vec![0.0f32; crate::ops::EMBED_DIM];
    noise[1] = 1.0;
    let (old_rel, _) = field
        .put_memory("wisdom", "wtest", b"relevant fact from three weeks ago window test", &rel,
            0.9, 0.001, now - 21 * day, vec![], None, None)
        .unwrap();
    let (in_rel, _) = field
        .put_memory("wisdom", "wtest", b"relevant fact from two days ago window test", &rel,
            0.9, 0.001, now - 2 * day, vec![], None, None)
        .unwrap();
    let (in_noise, _) = field
        .put_memory("wisdom", "wtest", b"unrelated compliance chatter fresh entry", &noise,
            0.9, 0.001, now - 1 * day, vec![], None, None)
        .unwrap();
    let window = Some((now - 7 * day, now));
    let hits = field
        .recall_with_fallback_windowed(&rel, "relevant fact window test", 3, Some("wtest"), window)
        .unwrap();
    let ids: Vec<_> = hits.iter().map(|h| h.memory_id).collect();
    assert!(!ids.contains(&old_rel), "out-of-window hit leaked through the gate: {ids:?}");
    assert!(ids.contains(&in_rel), "in-window relevant hit missing: {ids:?}");
    assert_eq!(ids.first(), Some(&in_rel),
        "semantic must rank above fresher noise (recency-sort pollution): {ids:?}");
    // Freshest-but-irrelevant may appear via backfill, but never above the relevant hit.
    if let Some(pos_noise) = ids.iter().position(|i| *i == in_noise) {
        assert!(pos_noise > 0);
    }
    // No window → out-of-window memory is reachable again (no frozen state).
    let all = field
        .recall_with_fallback(&rel, "relevant fact window test", 5, Some("wtest"))
        .unwrap();
    assert!(all.iter().any(|h| h.memory_id == old_rel));
}

// Span-lane live path: a NEW memory's atoms must be queryable and edge-linked
// immediately after put_memory — no manual backfill (the exact gap the owner
// hit: a fresh memory with a unique path stayed invisible to the lane).
#[test]
fn test_put_memory_auto_links_spans() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let content = b"results live at /projects/unique/spanlane/auto_link_probe.tsv.gz now";
    let (id, _) = field
        .put_memory("wisdom", "spantest", content, &emb, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();

    let hits = field.span_query("auto_link_probe", Some("spantest"), 6);
    assert!(!hits.is_empty(), "new memory's atom not auto-ingested");
    assert!(
        hits.iter().any(|h| h.0.contains("auto_link_probe.tsv.gz") && h.8.contains(&id)),
        "atom missing the memory_id reverse edge: {hits:?}"
    );
    // Forward edge: the memory expands to its verbatim atom.
    let fwd = field.span_for_memory(id, 4);
    assert!(fwd.iter().any(|a| a.0.contains("auto_link_probe.tsv.gz")));

    // forget() unlinks; the memory-only span hits refcount zero → gone.
    field.forget(id).unwrap();
    let hits = field.span_query("auto_link_probe", Some("spantest"), 6);
    assert!(hits.is_empty(), "forgotten memory's atoms must be GC'd: {hits:?}");
}

// ── Utility posteriors ──────────────────────────────────────────────────

#[test]
fn record_outcome_accrues_and_caps_weight() {
    let mut st = MemoryState::new(1, [0u8; 32], 0);
    assert_eq!((st.utility_alpha, st.utility_beta), (1.0, 1.0));
    assert_eq!(st.utility_mean(), 0.5);

    st.record_outcome(true, 1.0);
    st.record_outcome(true, 2.5);
    st.record_outcome(false, 0.5);
    assert_eq!((st.utility_alpha, st.utility_beta), (4.5, 1.5));

    // Above the cap saturates at 5; non-positive and NaN are not observations.
    st.record_outcome(true, 100.0);
    assert_eq!(st.utility_alpha, 9.5);
    st.record_outcome(false, 0.0);
    st.record_outcome(false, -3.0);
    st.record_outcome(false, f32::NAN);
    assert_eq!(st.utility_beta, 1.5);
}

#[test]
fn thompson_theta_is_neutral_below_observation_floor() {
    let mut rng = UtilityRng::new(0xC0FFEE);
    // Untested prior and everything short of 3 real observations: exactly
    // 0.5, no draw taken — an untested memory is neither boosted nor punished.
    for (a, b) in [(1.0, 1.0), (3.0, 1.0), (1.0, 3.0), (2.0, 2.5)] {
        assert_eq!(thompson_theta(a, b, &mut rng), 0.5, "alpha={a} beta={b}");
    }
    // At the floor a real draw happens, and it stays a probability.
    for _ in 0..200 {
        let t = thompson_theta(9.0, 1.0, &mut rng);
        assert!(t > 0.0 && t < 1.0, "theta out of range: {t}");
    }
}

#[test]
fn thompson_theta_is_deterministic_and_tracks_the_posterior() {
    // Same seed → same draw, so an eval run is reproducible.
    let a = thompson_theta(20.0, 5.0, &mut UtilityRng::new(7));
    let b = thompson_theta(20.0, 5.0, &mut UtilityRng::new(7));
    assert_eq!(a, b);

    // A useful memory samples above a useless one on average.
    let mut rng = UtilityRng::new(1234);
    let good: f32 = (0..500).map(|_| thompson_theta(40.0, 2.0, &mut rng)).sum::<f32>() / 500.0;
    let bad: f32 = (0..500).map(|_| thompson_theta(2.0, 40.0, &mut rng)).sum::<f32>() / 500.0;
    assert!(good > 0.85, "good posterior mean too low: {good}");
    assert!(bad < 0.15, "bad posterior mean too high: {bad}");
}

#[test]
fn utility_multiplier_is_identity_while_the_flag_is_off() {
    // The default process env has CHITTA_UTILITY_RECALL unset, so every
    // candidate multiplies by exactly 1.0 and recall ranking is unchanged.
    assert!(!utility_recall_config().enabled);
    let mut rng = UtilityRng::new(99);
    let mut st = MemoryState::new(1, [0u8; 32], 0);
    st.record_outcome(false, 5.0);
    st.record_outcome(false, 5.0);
    assert_eq!(utility_multiplier(&st, &mut rng), 1.0);
    for score in [0.0f32, 0.371_23, 12.5, f32::MAX] {
        assert_eq!(score * utility_multiplier(&st, &mut rng), score);
    }
}

#[test]
fn utility_multiplier_with_weight_separates_proven_from_untested() {
    let mut rng = UtilityRng::new(4242);
    let w = 0.3f32;

    // Untested and under-observed memories both land on the same neutral
    // 1 - w/2, so turning the flag on cannot reorder them among themselves.
    let untested = MemoryState::new(1, [0u8; 32], 0);
    assert_eq!(utility_multiplier_with(w, &untested, &mut rng), 0.85);
    let mut thin = MemoryState::new(2, [0u8; 32], 0);
    thin.record_outcome(true, 2.0);
    assert_eq!(utility_multiplier_with(w, &thin, &mut rng), 0.85);

    let mut good = MemoryState::new(3, [0u8; 32], 0);
    let mut bad = MemoryState::new(4, [0u8; 32], 0);
    for _ in 0..10 {
        good.record_outcome(true, 4.0);
        bad.record_outcome(false, 4.0);
    }
    let mean = |st: &MemoryState, rng: &mut UtilityRng| {
        (0..300).map(|_| utility_multiplier_with(w, st, rng)).sum::<f32>() / 300.0
    };
    let good_mul = mean(&good, &mut rng);
    let bad_mul = mean(&bad, &mut rng);
    assert!(good_mul > 0.98, "proven memory barely boosted: {good_mul}");
    assert!(bad_mul < 0.72, "disproven memory barely damped: {bad_mul}");
    // The whole effect stays inside [1-w, 1], a 1.43x spread at w=0.3.
    assert!(bad_mul >= 1.0 - w && good_mul <= 1.0);
}

/// Characterization: pins the write-time semantic dedup guard in put_memory.
///
/// The guard is `if !embed_pending`, and `embed_pending` is
/// `embedding.is_empty() && content.len() >= MIN_EMBED_CHARS`. So the branch
/// is skipped on the production FFI path (C++ passes no vector, content is
/// long) but RUNS whenever a caller supplies an embedding — which every test
/// here does, and which `cf_put_memory` permits. It is therefore live code,
/// not dead code, and removing it would change write-time behavior for
/// embedding-supplying callers. This test fails if that is ever done.
#[test]
fn put_memory_write_time_dedup_collapses_supplied_near_duplicates() {
    let tmp = TempDir::new().unwrap();
    let field = ChittaField::open(tmp.path().join("data")).unwrap();

    // Two vectors whose cosine lands inside [dedup_cosine_threshold,
    // dedup_cosine_upper) — near-duplicate, but not an exact match.
    let (thresh, upper) = {
        let cfg = &field.scoring_pipeline.read().config;
        (cfg.dedup_cosine_threshold, cfg.dedup_cosine_upper)
    };
    let mut a = vec![0.0f32; crate::ops::EMBED_DIM];
    let mut b = vec![0.0f32; crate::ops::EMBED_DIM];
    let target = (thresh + upper) / 2.0;
    a[0] = 1.0;
    b[0] = target;
    b[1] = (1.0 - target * target).sqrt();

    let (id_a, _) = field
        .put_memory("wisdom", "dedup", b"the first near duplicate memory", &a,
                    0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    let (id_b, _) = field
        .put_memory("wisdom", "dedup", b"the second near duplicate memory", &b,
                    0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    assert_eq!(
        id_b, id_a,
        "write-time dedup must collapse a supplied near-duplicate onto the original; \
         if this fails the `if !embed_pending` branch in put_memory was removed"
    );

    // Cross-realm near-duplicates must stay independent (no silent
    // cross-realm reinforcement).
    let (id_c, _) = field
        .put_memory("wisdom", "other-realm", b"the third near duplicate memory", &b,
                    0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    assert_ne!(id_c, id_a, "dedup must not cross realms");
}

/// Companion: on the production path (no supplied vector, content long
/// enough to embed) the write-time guard is skipped, so near-duplicates get
/// distinct ids at write time and are only collapsed later by the
/// backfill-time supersede pass in `backfill_embedding`.
#[test]
fn put_memory_without_embedding_defers_dedup_to_backfill() {
    let tmp = TempDir::new().unwrap();
    let field = ChittaField::open(tmp.path().join("data")).unwrap();
    // Distinct bodies: identical content is collapsed earlier by the
    // chunk-hash exact-duplicate check, which is a different mechanism.
    let (id_a, _) = field
        .put_memory("wisdom", "dedup", b"a sufficiently long memory body, variant one",
                    &[], 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    let (id_b, _) = field
        .put_memory("wisdom", "dedup", b"a sufficiently long memory body, variant two",
                    &[], 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    assert_ne!(id_b, id_a, "no vector at write time means no write-time dedup");
    for id in [id_a, id_b] {
        assert!(
            field.states.read().get(&id).map(|s| s.embed_pending).unwrap_or(false),
            "production-path writes must be embed_pending"
        );
    }
}

#[test]
fn record_outcome_rejects_unknown_ids_and_survives_snapshot_reopen() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.4f32; crate::ops::EMBED_DIM];

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let (id, _) = field
            .put_memory("wisdom", "test", b"outcome persist", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        assert!(field.record_outcome(id + 9_999, true, 1.0).is_err());
        assert_eq!(field.record_outcome(id, true, 2.0).unwrap(), (3.0, 1.0));
        assert_eq!(field.record_outcome(id, false, 1.0).unwrap(), (3.0, 2.0));
        field.save_full_snapshot().unwrap();
        id
    };

    let field = ChittaField::open(data_dir).unwrap();
    let states = field.states.read();
    let s = states.get(&id).unwrap();
    assert_eq!(
        (s.utility_alpha, s.utility_beta),
        (3.0, 2.0),
        "utility posterior must survive a snapshot save/reopen cycle"
    );
}

#[test]
fn record_outcome_survives_wal_only_replay() {
    // Regression: outcomes were RAM-only until the next periodic snapshot
    // wrote the V23 `utility_posteriors` section, so a restart in between
    // reverted the posterior (observed α 4 → 3). With no snapshot at all,
    // the reopen below is a pure WAL replay — it restores only if
    // RecordOutcome is a real op.
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.4f32; crate::ops::EMBED_DIM];

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let (id, _) = field
            .put_memory("wisdom", "test", b"outcome wal", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        assert_eq!(field.record_outcome(id, true, 3.0).unwrap(), (4.0, 1.0));
        assert_eq!(field.record_outcome(id, false, 2.0).unwrap(), (4.0, 3.0));
        // Non-observations must not reach the WAL and must not move (α, β).
        assert_eq!(field.record_outcome(id, true, 0.0).unwrap(), (4.0, 3.0));
        assert_eq!(field.record_outcome(id, true, -1.0).unwrap(), (4.0, 3.0));
        id
    };

    let field = ChittaField::open(data_dir).unwrap();
    let states = field.states.read();
    let s = states.get(&id).unwrap();
    assert_eq!(
        (s.utility_alpha, s.utility_beta),
        (4.0, 3.0),
        "utility posterior must survive a WAL-only replay"
    );
}

#[test]
fn record_outcome_replay_does_not_double_count_snapshot_section() {
    // Precedence: the V23 snapshot section carries (α, β) as of the
    // snapshot, and replay applies only the WAL suffix the snapshot does
    // not cover. Outcomes before the save must therefore be counted once
    // (via the section) and outcomes after it once (via the WAL).
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.4f32; crate::ops::EMBED_DIM];

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let (id, _) = field
            .put_memory("wisdom", "test", b"outcome mixed", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.record_outcome(id, true, 2.0).unwrap();
        field.record_outcome(id, false, 1.0).unwrap();
        field.save_full_snapshot().unwrap();
        // Post-snapshot suffix: lives only in the WAL until the next save.
        assert_eq!(field.record_outcome(id, true, 1.5).unwrap(), (4.5, 2.0));
        assert_eq!(field.record_outcome(id, false, 0.5).unwrap(), (4.5, 2.5));
        id
    };

    let field = ChittaField::open(data_dir).unwrap();
    let states = field.states.read();
    let s = states.get(&id).unwrap();
    assert_eq!(
        (s.utility_alpha, s.utility_beta),
        (4.5, 2.5),
        "snapshot-covered outcomes must not be replayed a second time"
    );
}

// ── Replay confluence (THEORY.md §2) ────────────────────────────────────
// Multi-node daemons write instance-partitioned WALs; replay applies all
// segments in instance-id sort order, not causal order. These tests pin
// down the convergence envelope and the two known loss modes so neither
// can drift silently.

fn theory_put_op(memory_id: u64, ts: i64, content: &str) -> crate::ops::Op {
    let mut emb = vec![0.1f32; crate::ops::EMBED_DIM];
    emb[0] = (memory_id as f32) / 100.0;
    crate::ops::Op::PutPayload(crate::ops::PutPayloadOp {
        memory_id,
        version: 0,
        chunk_hash: [memory_id as u8; 32],
        created_at_ms: ts,
        authored_at_ms: ts,
        kind: "wisdom".to_string(),
        realm: "test".to_string(),
        content: content.as_bytes().to_vec(),
        embedding_model: "test".to_string(),
        embedding: emb,
        artifact_refs: vec![],
        source_session: None,
        source_tool: None,
        harness: None,
        embedding_model_id: String::new(),
        embedding_dim: crate::ops::EMBED_DIM as u32,
    })
}

fn theory_delta_op(memory_id: u64, strength_delta: f32, ts: i64) -> crate::ops::Op {
    crate::ops::Op::UpdateState(crate::ops::StateDeltaOp {
        memory_id,
        strength_delta: Some(strength_delta),
        confidence_delta: None,
        decay_rate: None,
        touch: true,
        pin: None,
        op_ts_ms: ts,
        status: None,
        epistemic_status: None,
        staged: None,
        invalidated_by: None,
    })
}

fn write_instance_segment(data_dir: &std::path::Path, instance: u32, ops: &[crate::ops::Op]) {
    let mut log = crate::log::OpLog::open(data_dir, instance, 1).unwrap();
    for op in ops {
        log.append(op).unwrap();
    }
    log.flush_buf().unwrap();
}

fn state_fingerprint(field: &ChittaField, id: u64) -> (f32, u32) {
    let states = field.states.read();
    let st = states.get(&id).expect("memory state must exist after replay");
    (st.strength, st.access_count)
}

/// The safe envelope: per-memory single-writer op sets converge no matter
/// which instance id (= segment sort position) each writer was assigned.
#[test]
fn replay_confluent_for_disjoint_memories() {
    let set_a = vec![theory_put_op(11, 1_000, "alpha"), theory_delta_op(11, -0.2, 2_000)];
    let set_b = vec![theory_put_op(22, 1_500, "beta"), theory_delta_op(22, -0.4, 2_500)];

    let mut results = Vec::new();
    for (inst_a, inst_b) in [(0x1000_0001u32, 0x2000_0002u32), (0x2000_0002, 0x1000_0001)] {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        write_instance_segment(&data_dir, inst_a, &set_a);
        write_instance_segment(&data_dir, inst_b, &set_b);
        let field = ChittaField::open(data_dir).unwrap();
        results.push((state_fingerprint(&field, 11), state_fingerprint(&field, 22)));
    }
    assert_eq!(
        results[0], results[1],
        "disjoint-memory replay must be insensitive to instance assignment"
    );
}

/// THEORY.md §3: merge replay orders ops by (op_ts, instance, seqno), so
/// cross-instance deltas apply in timestamp order even when instance-id
/// sort order inverts it. Before merge replay, the ts=2000 delta below was
/// wholly discarded by apply_delta's monotonicity guard (loss mode §2.2).
#[test]
fn merge_replay_applies_cross_instance_deltas_in_timestamp_order() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    write_instance_segment(&data_dir, 0x1000_0001, &[theory_put_op(33, 1_000, "gamma")]);
    write_instance_segment(&data_dir, 0x2000_0002, &[theory_delta_op(33, -0.2, 3_000)]);
    write_instance_segment(&data_dir, 0x3000_0003, &[theory_delta_op(33, -0.4, 2_000)]);

    let field = ChittaField::open(data_dir).unwrap();
    let (strength, access_count) = state_fingerprint(&field, 33);
    assert!(
        (strength - 0.4).abs() < 1e-6,
        "both deltas must apply in ts order (got strength {strength})"
    );
    assert_eq!(access_count, 2);
}

/// THEORY.md §2.3/§3: an UpdateState merge-ordered before its memory's
/// PutPayload (possible under cross-writer clock skew) lands in the
/// orphan-delta buffer and is applied after the creates. Before merge
/// replay + the buffer, it was silently dropped.
#[test]
fn orphan_delta_before_create_is_buffered_and_applied() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    // Skewed clock: the delta's ts predates the create's ts, so merge
    // order applies it first — the orphan buffer must catch it.
    write_instance_segment(&data_dir, 0x1000_0001, &[theory_delta_op(44, -0.4, 500)]);
    write_instance_segment(&data_dir, 0x2000_0002, &[theory_put_op(44, 1_000, "delta")]);

    let field = ChittaField::open(data_dir).unwrap();
    let (strength, access_count) = state_fingerprint(&field, 44);
    assert!(
        (strength - 0.6).abs() < 1e-6,
        "orphaned delta must be applied after creates (got strength {strength})"
    );
    assert_eq!(access_count, 1);
}

/// THEORY.md §3: with merge replay, state is a function of the op SET —
/// any partition of the ops across writers, under any instance-id
/// assignment, converges. Creates carry the earliest timestamps so the
/// orphan path stays out of this test (covered separately).
#[test]
fn replay_confluent_under_random_instance_permutations() {
    let mut ops: Vec<crate::ops::Op> = Vec::new();
    for m in 0..4u64 {
        ops.push(theory_put_op(100 + m, 1_000 + m as i64, "perm"));
    }
    for i in 0..8u64 {
        let mem = 100 + (i % 4);
        let d = -0.05 * ((i % 3) as f32 + 1.0);
        ops.push(theory_delta_op(mem, d, 2_000 + 100 * i as i64));
    }
    let instances = [0x1000_0001u32, 0x2000_0002, 0x3000_0003];

    let mut seed = 0x9E37_79B9_u64;
    let mut xorshift = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let mut fingerprints = Vec::new();
    for _ in 0..4 {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let mut groups: Vec<Vec<crate::ops::Op>> = vec![Vec::new(), Vec::new(), Vec::new()];
        for op in &ops {
            groups[(xorshift() % 3) as usize].push(op.clone());
        }
        for (group, inst) in groups.into_iter().zip(instances) {
            if !group.is_empty() {
                write_instance_segment(&data_dir, inst, &group);
            }
        }
        let field = ChittaField::open(data_dir).unwrap();
        let fp: Vec<(f32, u32)> =
            (100..104).map(|m| state_fingerprint(&field, m)).collect();
        fingerprints.push(fp);
    }
    for fp in &fingerprints[1..] {
        assert_eq!(
            &fingerprints[0], fp,
            "state must be a function of the op set, not the instance assignment"
        );
    }
}

/// THEORY.md §4: seqno ranges overlap across writers, so the scalar
/// `seqno <= snapshot_seqno` skip silently dropped foreign ops the
/// snapshot never contained. The per-writer coverage vector applies them.
#[test]
fn reopen_applies_uncovered_foreign_ops_with_overlapping_seqnos() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.3f32; crate::ops::EMBED_DIM];

    {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        field
            .put_memory("wisdom", "test", b"local memory", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.save_full_snapshot().unwrap();
    }

    // A foreign writer's ops with LOW seqnos (overlapping the snapshot's
    // scalar seqno range) that the snapshot does NOT contain.
    write_instance_segment(
        &data_dir,
        0xF000_000F,
        &[theory_put_op(777, 5_000, "foreign uncovered")],
    );

    let field = ChittaField::open(data_dir).unwrap();
    assert!(
        field.states.read().contains_key(&777),
        "uncovered foreign op must be applied on reopen (was skipped by the scalar filter)"
    );
}

/// THEORY.md §4: prune only segments the coverage vector dominates; an
/// instance's open-ended last segment is never pruned. Header-only (empty)
/// segments of a dead instance are prunable by size regardless of coverage.
#[test]
fn prune_covered_segments_respects_coverage_vector() {
    let tmp = TempDir::new().unwrap();
    let seg_dir = tmp.path().join("segments");
    std::fs::create_dir_all(&seg_dir).unwrap();
    // Op-bearing segments must exceed the header size, else the empty-segment
    // rule would reclaim them; write header + a byte of "op" payload.
    let ops = vec![0u8; crate::log::V3_HEADER_SIZE + 1];
    for name in [
        "10000001_000000000001.seg",
        "10000001_000000000050.seg",
        "20000002_000000000001.seg",
    ] {
        std::fs::write(seg_dir.join(name), &ops).unwrap();
    }

    // Instance 1 is the LIVE writer here: its open tail (…_50) is never pruned.
    let live = 0x1000_0001u32;

    // Not covered far enough: nothing prunable.
    let mut covered = std::collections::BTreeMap::new();
    covered.insert(0x1000_0001u32, 10u64);
    assert_eq!(prune_covered_segments(&seg_dir, &covered, live), 0);

    // Covered through the first segment's end (next_first - 1 = 49):
    // only instance 1's first segment goes; the live tail + the dead
    // instance-2 segment (absent from `covered`) stay.
    covered.insert(0x1000_0001, 49);
    assert_eq!(prune_covered_segments(&seg_dir, &covered, live), 1);
    assert!(!seg_dir.join("10000001_000000000001.seg").exists());
    assert!(seg_dir.join("10000001_000000000050.seg").exists());
    assert!(
        seg_dir.join("20000002_000000000001.seg").exists(),
        "a foreign writer's segment must never be pruned without coverage"
    );

    // A DEAD instance's final segment IS prunable once it appears in the
    // coverage vector (fully folded into the snapshot). This is the common
    // one-segment-per-lifetime case the interior windows(2) rule can't reach.
    covered.insert(0x2000_0002, 1);
    assert_eq!(prune_covered_segments(&seg_dir, &covered, live), 1);
    assert!(!seg_dir.join("20000002_000000000001.seg").exists());
    // The live instance's tail still survives — never pruned even when covered.
    assert!(seg_dir.join("10000001_000000000050.seg").exists());

    // Header-only (empty) segments: a dead instance's empty segment is
    // reclaimed WITHOUT any coverage entry (coverage is op-derived and can
    // never prove it); the live instance's empty segment is preserved.
    let empty = vec![0u8; crate::log::V3_HEADER_SIZE];
    std::fs::write(seg_dir.join("30000003_000000000001.seg"), &empty).unwrap(); // dead, empty
    std::fs::write(seg_dir.join("10000001_000000000999.seg"), &empty).unwrap(); // live, empty
    let empty_cov = std::collections::BTreeMap::new(); // no coverage at all
    assert_eq!(prune_covered_segments(&seg_dir, &empty_cov, live), 1);
    assert!(!seg_dir.join("30000003_000000000001.seg").exists());
    assert!(
        seg_dir.join("10000001_000000000999.seg").exists(),
        "the live instance's freshly-opened (header-only) segment must survive"
    );
}

fn theory_content_op(memory_id: u64, content: &str, ts: i64) -> crate::ops::Op {
    crate::ops::Op::UpdateMemoryContent(crate::ops::UpdateMemoryContentOp {
        memory_id,
        content: content.as_bytes().to_vec(),
        embedding: Vec::new(),
        op_ts_ms: ts,
    })
}

/// THEORY.md §2.1 class (c): absolute writes are LWW registers under
/// merge replay — newest op_ts_ms wins regardless of instance assignment.
#[test]
fn content_updates_are_lww_by_timestamp() {
    for (inst_b, inst_c) in [(0x2000_0002u32, 0x3000_0003u32), (0x3000_0003, 0x2000_0002)] {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        write_instance_segment(&data_dir, 0x1000_0001, &[theory_put_op(55, 1_000, "orig")]);
        write_instance_segment(&data_dir, inst_b, &[theory_content_op(55, "newest", 3_000)]);
        write_instance_segment(&data_dir, inst_c, &[theory_content_op(55, "middle", 2_000)]);

        let field = ChittaField::open(data_dir).unwrap();
        let payloads = field.payloads.read();
        assert_eq!(
            payloads.get(&55).unwrap().content,
            b"newest".to_vec(),
            "newest op_ts_ms must win under any instance assignment"
        );
    }
}

/// The semantic index is the embedding's single in-RAM home: the payload
/// copy is cleared at write, stripped from the snapshot body, NOT
/// rehydrated at open — and embedding_of() serves every reader.
#[test]
fn payload_embeddings_stripped_from_body_and_rehydrated() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.7f32; crate::ops::EMBED_DIM];

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let (id, _) = field
            .put_memory("wisdom", "test", b"strip me", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.save_full_snapshot().unwrap();
        id
    };

    // Raw body: embedding stripped.
    let snap_path = std::fs::read_dir(&data_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            n.starts_with("chitta.") && n.ends_with(".snapshot")
        })
        .expect("snapshot file");
    let raw = crate::snapshot::FullSnapshot::load(&snap_path).unwrap();
    assert!(
        raw.payloads.get(&id).unwrap().embedding.is_empty(),
        "body must not carry an embedding the .emb sidecar owns"
    );

    // Full open: the payload copy STAYS empty (no ~600MB duplicate);
    // embedding_of serves the vector from the index.
    let field = ChittaField::open(data_dir).unwrap();
    {
        let payloads = field.payloads.read();
        assert!(
            payloads.get(&id).unwrap().embedding.is_empty(),
            "payload embedding must NOT be rehydrated into the heap"
        );
    }
    assert_eq!(
        field.embedding_of(id).map(|e| e.len()),
        Some(crate::ops::EMBED_DIM),
        "embedding_of must serve the vector from the index"
    );
    assert_eq!(
        field.states.read().get(&id).map(|s| s.embed_pending),
        Some(false),
        "index-held embeddings must not be requeued for re-embed"
    );
}

/// Phase 2 (THEORY.md §8): index sidecars are not rewritten when the
/// index hasn't mutated since the last save (dirty-skip).
#[test]
fn index_sidecars_skipped_when_clean() {
    use std::os::unix::fs::MetadataExt;
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.5f32; crate::ops::EMBED_DIM];

    let field = ChittaField::open(data_dir.clone()).unwrap();
    field
        .put_memory("wisdom", "test", b"dirty one", &emb, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    field.save_full_snapshot().unwrap();
    let sidecar = |ext: &str| {
        std::fs::read_dir(&data_dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().map(|e| e == ext).unwrap_or(false))
            .unwrap_or_else(|| panic!(".{ext} sidecar"))
    };
    let emb_path = sidecar("emb");
    let hdc_path = sidecar("hdc");
    let pld_path = sidecar("pld");
    let ino_first = std::fs::metadata(&emb_path).unwrap().ino();
    let hdc_ino_first = std::fs::metadata(&hdc_path).unwrap().ino();
    let pld_ino_first = std::fs::metadata(&pld_path).unwrap().ino();
    let emb_len_first = std::fs::metadata(&emb_path).unwrap().len();
    let hdc_len_first = std::fs::metadata(&hdc_path).unwrap().len();
    let pld_len_first = std::fs::metadata(&pld_path).unwrap().len();

    // No mutation between saves → same inodes (skipped rewrites).
    field.save_full_snapshot().unwrap();
    assert_eq!(
        std::fs::metadata(&emb_path).unwrap().ino(),
        ino_first,
        "clean index must not rewrite sidecars"
    );
    assert_eq!(
        std::fs::metadata(&hdc_path).unwrap().ino(),
        hdc_ino_first,
        "clean hdc store must not rewrite its sidecar"
    );
    assert_eq!(
        std::fs::metadata(&pld_path).unwrap().ino(),
        pld_ino_first,
        "unchanged content must not rewrite the .pld sidecar"
    );

    // A new memory mutates the index → rewrite (fresh inode via rename).
    // Orthogonal-ish embedding so the write-path dedup doesn't merge it.
    let mut emb2 = emb.clone();
    for v in emb2.iter_mut().take(crate::ops::EMBED_DIM / 2) {
        *v = -0.5;
    }
    field
        .put_memory("wisdom", "test", b"dirty two", &emb2, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    field.save_full_snapshot().unwrap();
    // Size, not inode: tmpfs reuses freed inode numbers, so a rename can
    // land on the same ino. Two embeddings serialize larger than one.
    assert!(
        std::fs::metadata(&emb_path).unwrap().len() > emb_len_first,
        "mutated index must rewrite sidecars"
    );
    assert!(
        std::fs::metadata(&hdc_path).unwrap().len() > hdc_len_first,
        "mutated hdc store must rewrite its sidecar"
    );
    assert!(
        std::fs::metadata(&pld_path).unwrap().len() > pld_len_first,
        "new content must rewrite the .pld sidecar"
    );
}

fn theory_recall_op(memory_ids: &[u64], ts: i64) -> crate::ops::Op {
    crate::ops::Op::RecordRecallBatch(crate::ops::RecordRecallBatchOp {
        memory_ids: memory_ids.to_vec(),
        centroid_q: Vec::new(),
        centroid_scale: 0.0,
        context_hash: ts as u64,
        ts_ms: ts,
        base_assoc_delta: 0.0,
    })
}

/// THEORY.md §6: recalls from distinct daemons accrue as cross-context
/// provenance, and the evidence survives a snapshot save/reopen cycle
/// via the V23 "recall_provenance" section (added with zero migration).
#[test]
fn recall_provenance_accrues_across_instances_and_persists() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    write_instance_segment(&data_dir, 0x1000_0001, &[theory_put_op(66, 1_000, "general")]);
    write_instance_segment(&data_dir, 0x2000_0002, &[theory_recall_op(&[66], 2_000)]);
    write_instance_segment(&data_dir, 0x3000_0003, &[theory_recall_op(&[66], 3_000)]);
    write_instance_segment(&data_dir, 0x4000_0004, &[theory_recall_op(&[66], 4_000)]);

    let distinct = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let n = field.recall_provenance.read().get(&66).map(|s| s.len());
        field.save_full_snapshot().unwrap();
        n
    };
    assert_eq!(distinct, Some(3), "three distinct recalling instances");

    let field = ChittaField::open(data_dir).unwrap();
    assert_eq!(
        field.recall_provenance.read().get(&66).map(|s| s.len()),
        Some(3),
        "provenance must survive snapshot save/reopen"
    );
}

/// THEORY.md §6: cross-context evidence raises recall score
/// (multiplicative, config-gated).
#[test]
fn cross_context_provenance_boosts_recall_score() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.6f32; crate::ops::EMBED_DIM];
    let (id, _) = field
        .put_memory("wisdom", "test", b"generalizes", &emb, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();

    let s1 = field.recall_semantic(&emb, 3, Some("test")).unwrap()[0].score;
    {
        let mut prov = field.recall_provenance.write();
        let set = prov.entry(id).or_default();
        for inst in [0x1u32, 0x2, 0x3, 0x4] {
            set.insert(inst);
        }
    }
    let s2 = field.recall_semantic(&emb, 3, Some("test")).unwrap()[0].score;
    assert!(
        s2 > s1,
        "4 distinct recalling instances must boost score ({s1} → {s2})"
    );
}

/// THEORY.md §6, the falsifiable claim at the recall level: the ranked
/// result of a query must not depend on which writer wrote what.
#[test]
fn recall_ranking_invariant_under_writer_permutation() {
    let mut emb_q = vec![0.1f32; crate::ops::EMBED_DIM];
    emb_q[0] = 1.0;
    let mut ops: Vec<crate::ops::Op> = Vec::new();
    for m in 0..5u64 {
        let mut e = vec![0.1f32; crate::ops::EMBED_DIM];
        e[0] = 1.0;
        e[1 + m as usize] = 0.3 + 0.1 * m as f32;
        ops.push(crate::ops::Op::PutPayload(match theory_put_op(200 + m, 1_000 + m as i64, "rank") {
            crate::ops::Op::PutPayload(mut p) => {
                p.embedding = e;
                p
            }
            _ => unreachable!(),
        }));
        ops.push(theory_delta_op(200 + m, -0.05 * (m as f32 + 1.0), 2_000 + m as i64));
    }

    let mut rankings = Vec::new();
    for (a, b) in [(0x1000_0001u32, 0x2000_0002u32), (0x2000_0002, 0x1000_0001)] {
        let tmp = TempDir::new().unwrap();
        let data_dir = tmp.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        let (left, right): (Vec<_>, Vec<_>) =
            ops.iter().cloned().enumerate().partition(|(i, _)| i % 2 == 0);
        write_instance_segment(&data_dir, a, &left.into_iter().map(|(_, o)| o).collect::<Vec<_>>());
        write_instance_segment(&data_dir, b, &right.into_iter().map(|(_, o)| o).collect::<Vec<_>>());
        let field = ChittaField::open(data_dir).unwrap();
        let ids: Vec<u64> = field
            .recall_semantic(&emb_q, 5, Some("test"))
            .unwrap()
            .iter()
            .map(|h| h.memory_id)
            .collect();
        rankings.push(ids);
    }
    assert_eq!(
        rankings[0], rankings[1],
        "recall ranking must be writer-assignment invariant"
    );
}

/// THEORY.md §6/§8: the consolidation sweep refreshes stale competitive
/// weights up to its budget and stamps them, so recall-path budgets
/// rarely trigger.
#[test]
fn cw_refresh_sweep_respects_budget_and_stamps() {
    let (field, _tmp) = open_test_field();
    // Orthogonal square waves (different frequencies) — pairwise cosine 0,
    // so the write-path dedup can't merge them.
    let mut embs = Vec::new();
    for m in 0..4usize {
        let e: Vec<f32> = (0..crate::ops::EMBED_DIM)
            .map(|i| if (i / (64 << m)) % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        embs.push(e);
    }
    let mut ids = Vec::new();
    for (m, e) in embs.iter().enumerate() {
        let (id, _) = field
            .put_memory("wisdom", "test", format!("sweep {m}").as_bytes(), e, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        ids.push(id);
    }
    // Force staleness.
    {
        let mut states = field.states.write();
        for id in &ids {
            states.get_mut(id).unwrap().last_cw_refresh_ms = 0;
        }
    }
    assert_eq!(field.cw_refresh_sweep(2), 2, "budget must cap the sweep");
    let stamped = field
        .states
        .read()
        .values()
        .filter(|st| st.last_cw_refresh_ms > 0)
        .count();
    assert_eq!(stamped, 2);
    assert_eq!(field.cw_refresh_sweep(10), 2, "remaining stale memories swept");
    assert_eq!(field.cw_refresh_sweep(10), 0, "nothing stale left");
}

/// Regression for the consolidation re-encode loop: unencodable memories
/// (deleted, empty-code) must not be re-collected on every pass.
#[test]
fn encode_all_unindexed_converges_to_zero() {
    let (field, _tmp) = open_test_field();
    let emb_a = vec![0.5f32; crate::ops::EMBED_DIM];
    let mut emb_b = emb_a.clone();
    for v in emb_b.iter_mut().take(crate::ops::EMBED_DIM / 2) {
        *v = -0.5;
    }
    let (id_a, _) = field
        .put_memory("wisdom", "test", b"encodable", &emb_a, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    field
        .put_memory("wisdom", "test", b"to be deleted", &emb_b, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();

    // Soft-delete one: it must never be collected for encoding.
    let _ = field.forget(id_a);

    let first = field.encode_all_unindexed().unwrap();
    assert!(first <= 1, "deleted memory must not be collected (got {first})");
    // Whatever was attempted is now coded or skip-set: the pass converges.
    assert_eq!(
        field.encode_all_unindexed().unwrap(),
        0,
        "second pass must collect nothing — the re-encode loop"
    );
}

/// Janitor: dead-instance residue goes, protected classes stay, and
/// resurrected (previously-deleted) files are counted via the ledger.
#[test]
fn janitor_sweep_removes_ghosts_and_tracks_resurrection() {
    let tmp = TempDir::new().unwrap();
    let d = tmp.path();
    let touch = |name: &str| std::fs::write(d.join(name), b"x").unwrap();

    // Protected: own seen_offsets, a live family + its sidecar.
    touch("seen_offsets.aaaa0001.json");
    touch("chitta.bbbb0002.snapshot");
    touch("chitta.bbbb0002.emb");
    touch("cortex.bbbb0002.snapshot");
    // Ghosts: dead reader, orphan cortex, orphan sidecar.
    touch("seen_offsets.dead0003.json");
    touch("cortex.dead0004.snapshot");
    touch("chitta.dead0005.emb");

    // max_age 0 → everything is old enough (mtime gate test inverse:
    // a huge max_age must delete nothing).
    janitor_sweep(d, 0xaaaa_0001, u64::MAX);
    assert!(d.join("seen_offsets.dead0003.json").exists(), "age gate must protect");

    janitor_sweep(d, 0xaaaa_0001, 0);
    assert!(d.join("seen_offsets.aaaa0001.json").exists(), "own file protected");
    assert!(d.join("chitta.bbbb0002.snapshot").exists(), "families are prune's job");
    assert!(d.join("chitta.bbbb0002.emb").exists(), "family sidecar protected");
    assert!(d.join("cortex.bbbb0002.snapshot").exists(), "family cortex protected");
    assert!(!d.join("seen_offsets.dead0003.json").exists(), "dead reader removed");
    assert!(!d.join("cortex.dead0004.snapshot").exists(), "orphan cortex removed");
    assert!(!d.join("chitta.dead0005.emb").exists(), "orphan sidecar removed");

    // Resurrection: re-create a deleted ghost; ledger must count it (the
    // count is logged; behaviorally it gets deleted again).
    touch("seen_offsets.dead0003.json");
    janitor_sweep(d, 0xaaaa_0001, 0);
    assert!(!d.join("seen_offsets.dead0003.json").exists(), "resurrected ghost re-deleted");
    let ledger = std::fs::read_to_string(d.join(".janitor.json")).unwrap();
    assert!(ledger.contains("seen_offsets.dead0003.json"));
}

#[test]
fn test_manifest_commits_snapshot_family() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.2f32; crate::ops::EMBED_DIM];

    {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        field
            .put_memory("wisdom", "test", b"manifest commit", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        // Bypass the compact_wal 100-memory guard: save directly.
        field.save_full_snapshot().unwrap();
    }

    let manifest = crate::manifest::Manifest::load(&data_dir).unwrap().unwrap();
    assert!(manifest.generation >= 1);
    let committed = manifest
        .validated_snapshot_path(&data_dir)
        .expect("freshly committed family must validate");
    assert!(committed.exists());

    // Tamper with a recorded sidecar: validation must fail and open must
    // still succeed via fence-based fallback.
    let cp = manifest.checkpoints.as_ref().unwrap();
    let side = data_dir.join(&cp.sidecars[0].name);
    {
        let f = std::fs::OpenOptions::new().write(true).open(&side).unwrap();
        f.set_len(cp.sidecars[0].size_bytes + 7).unwrap();
    }
    assert!(manifest.validated_snapshot_path(&data_dir).is_none());
    let field = ChittaField::open(data_dir).unwrap();
    assert_eq!(field.memory_count(), 1);
}

#[test]
fn test_cw_refresh_ts_survives_snapshot_reopen() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let emb = vec![0.4f32; crate::ops::EMBED_DIM];

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let (id, _) = field
            .put_memory("wisdom", "test", b"cw persist", &emb, 0.9, 0.001, 0, vec![], None, None)
            .unwrap();
        field.states.write().get_mut(&id).unwrap().last_cw_refresh_ms = 1_234_567;
        field.save_full_snapshot().unwrap();
        id
    };

    let field = ChittaField::open(data_dir).unwrap();
    assert_eq!(
        field.states.read().get(&id).unwrap().last_cw_refresh_ms,
        1_234_567,
        "last_cw_refresh_ms must survive a snapshot save/reopen cycle"
    );
}

#[test]
fn test_cw_refresh_releases_inflight_reservations() {
    // Neighborhood with a real update: reservations must be released.
    let (field, _tmp) = open_test_field();
    let mut emb_a = vec![0.0f32; crate::ops::EMBED_DIM];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0f32; crate::ops::EMBED_DIM];
    emb_b[0] = 1.0;
    emb_b[1] = 1.0;
    field
        .put_memory("wisdom", "test", b"cw refresh a", &emb_a, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    field
        .put_memory("wisdom", "test", b"cw refresh b", &emb_b, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    field.recall_semantic(&emb_a, 5, Some("test")).unwrap();
    assert!(
        field.cw_refresh_inflight.read().is_empty(),
        "reservations must be released after a refresh round"
    );

    // Isolated memory: the round produces no cw update — reservations must
    // still be released (empty rounds must not leak entries).
    let (field2, _tmp2) = open_test_field();
    field2
        .put_memory("wisdom", "test", b"isolated", &emb_a, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    field2.recall_semantic(&emb_a, 5, Some("test")).unwrap();
    assert!(
        field2.cw_refresh_inflight.read().is_empty(),
        "empty update rounds must not leak reservations"
    );
}

#[test]
fn test_densify_write_edges_pair() {
    // #13 controlled prospective pair test (drift-independent, binary gate):
    // a symptom memory and its root-cause memory written in the same session
    // + realm must be linked by a bidirectional SameSession edge, making each
    // 1-hop reachable from the other. Rollback removes exactly those edges.
    use std::sync::atomic::Ordering;
    let (field, _tmp) = open_test_field();
    field.densify_enabled.store(true, Ordering::Relaxed);

    // Distinct embeddings (cos 0.707 < dedup threshold) so B is not merged.
    let mut emb_a = vec![0.0f32; crate::ops::EMBED_DIM];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0f32; crate::ops::EMBED_DIM];
    emb_b[0] = 1.0;
    emb_b[1] = 1.0;
    let sess = Some("pairtest-session".to_string());

    let (a, _) = field
        .put_memory("episode", "test", b"symptom: recall degrades over days",
            &emb_a, 0.9, 0.001, 0, vec![], sess.clone(), None)
        .unwrap();
    let (b, _) = field
        .put_memory("episode", "test", b"root cause: eval strengthens wrong edges",
            &emb_b, 0.9, 0.001, 0, vec![], sess.clone(), None)
        .unwrap();
    assert_ne!(a, b, "B must be a distinct memory, not deduped into A");

    let na = field.list_neighbors(a).unwrap();
    let nb = field.list_neighbors(b).unwrap();
    assert!(
        na.iter().any(|e| e.dst == b && e.edge_type == EdgeType::SameSession),
        "root-cause must be reachable from symptom via a SameSession edge",
    );
    assert!(
        nb.iter().any(|e| e.dst == a && e.edge_type == EdgeType::SameSession),
        "densification edge must be bidirectional",
    );

    // Surgical rollback: removes exactly the two densification edges.
    let removed = field.remove_assoc_edges_by_type(EdgeType::SameSession);
    assert_eq!(removed, 2, "both edge directions must be removed");
    assert!(field.list_neighbors(a).unwrap().is_empty());
    assert!(field.list_neighbors(b).unwrap().is_empty());
}

#[test]
fn test_densify_via_set_source_session() {
    // Production daemon flow: cf_put_memory carries session=None; the C++
    // handler attaches it via set_source_session right after. The chain
    // must form at that attach point, and must not double-fire when the
    // session was already known at put time.
    use std::sync::atomic::Ordering;
    let (field, _tmp) = open_test_field();
    field.densify_enabled.store(true, Ordering::Relaxed);

    let mut emb_a = vec![0.0f32; crate::ops::EMBED_DIM];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0f32; crate::ops::EMBED_DIM];
    emb_b[1] = 1.0;

    let (a, _) = field
        .put_memory("episode", "test", b"daemon-path write A", &emb_a, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    let (b, _) = field
        .put_memory("episode", "test", b"daemon-path write B", &emb_b, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();
    // No session at put time → no edges yet.
    assert!(field.list_neighbors(a).unwrap().is_empty());

    field.set_source_session(a, "daemon-sess").unwrap();
    field.set_source_session(b, "daemon-sess").unwrap();
    let nb = field.list_neighbors(b).unwrap();
    assert!(
        nb.iter().any(|e| e.dst == a && e.edge_type == EdgeType::SameSession),
        "attach-time densification must link B to its session sibling A",
    );

    // Re-attach must not duplicate edges (ring guard).
    field.set_source_session(b, "daemon-sess").unwrap();
    let nb2 = field.list_neighbors(b).unwrap();
    let same_count = nb2.iter().filter(|e| e.dst == a && e.edge_type == EdgeType::SameSession).count();
    assert_eq!(same_count, 1, "re-attach must not duplicate the edge");
}

#[test]
fn test_densify_backfill() {
    // Retro-backfill over historical memories: session-tagged writes made
    // while the write-hook was OFF (densify_enabled=false — the pre-#13
    // world) must gain the same K=3 decaying SameSession chain when
    // densify_backfill(apply=true) runs. Dry run writes nothing; re-apply
    // is idempotent; untagged memories are ignored.
    let (field, _tmp) = open_test_field();
    let sess = Some("hist-sess".to_string());

    let mut ids = Vec::new();
    for i in 0..4u8 {
        let mut emb = vec![0.0f32; crate::ops::EMBED_DIM];
        emb[i as usize] = 1.0;
        let (id, _) = field
            .put_memory("episode", "test", format!("hist write {i}").as_bytes(),
                &emb, 0.9, 0.001, 0, vec![], sess.clone(), None)
            .unwrap();
        ids.push(id);
    }
    // One untagged memory: must not join any group.
    let mut emb = vec![0.0f32; crate::ops::EMBED_DIM];
    emb[5] = 1.0;
    field
        .put_memory("episode", "test", b"untagged", &emb, 0.9, 0.001, 0, vec![], None, None)
        .unwrap();

    // Write-hook off → no edges yet despite session tags.
    assert!(field.list_neighbors(ids[3]).unwrap().is_empty());

    // Dry run: 1 session group of 4 → pairs = 1+2+3 = 6; nothing written.
    let (sessions, mems, pairs, hist) = field.densify_backfill(false);
    assert_eq!((sessions, mems, pairs), (1, 4, 6));
    assert_eq!(hist[2], 1, "group of 4 lands in the 3-5 bucket");
    assert!(field.list_neighbors(ids[3]).unwrap().is_empty(), "dry run must write nothing");

    // Apply: newest links to its 3 priors with decaying weight.
    let (_, _, pairs_applied, _) = field.densify_backfill(true);
    assert_eq!(pairs_applied, 6);
    let n3 = field.list_neighbors(ids[3]).unwrap();
    let same: Vec<_> = n3.iter().filter(|e| e.edge_type == EdgeType::SameSession).collect();
    assert_eq!(same.len(), 3, "newest must link to all 3 priors");
    let w = |dst| same.iter().find(|e| e.dst == dst).unwrap().weight;
    assert!((w(ids[2]) - 0.6).abs() < 1e-6);
    assert!((w(ids[1]) - 0.42).abs() < 1e-6);
    assert!((w(ids[0]) - 0.294).abs() < 1e-6);

    // Idempotent: re-apply changes nothing.
    field.densify_backfill(true);
    assert_eq!(
        field.list_neighbors(ids[3]).unwrap().iter()
            .filter(|e| e.edge_type == EdgeType::SameSession).count(),
        3,
        "re-apply must not duplicate edges"
    );
}

#[test]
fn test_forget() {
    let (field, _tmp) = open_test_field();
    let embedding = vec![0.0f32; crate::ops::EMBED_DIM];
    let (id, _) = field
        .put_memory(
            "wisdom",
            "test",
            b"to forget",
            &embedding,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();
    field.forget(id).unwrap();
    assert!(matches!(
        field.get_memory(id),
        Err(crate::error::FieldError::Deleted(_))
    ));
}

#[test]
fn test_replay_on_reopen() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let embedding = vec![0.5f32; crate::ops::EMBED_DIM];
        let (id, _) = field
            .put_memory(
                "episode",
                "test",
                b"persisted",
                &embedding,
                0.8,
                0.002,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
        id
    };

    // Reopen and verify data survived.
    let field2 = ChittaField::open(data_dir).unwrap();
    let payload = field2.get_memory(id).unwrap();
    assert_eq!(payload.content, b"persisted");
}

#[test]
fn test_assoc_edge() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id1, _) = field
        .put_memory(
            "wisdom",
            "test",
            b"a",
            &emb,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();
    let (id2, _) = field
        .put_memory(
            "wisdom",
            "test",
            b"b",
            &emb,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();
    field
        .add_assoc_edge(id1, id2, EdgeType::CoRetrieved, 0.7)
        .unwrap();
    let neighbors = field.list_neighbors(id1).unwrap();
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].dst, id2);
}

#[test]
fn test_integration_add_triplet() {
    let tmp = TempDir::new().unwrap();
    let field = ChittaField::open(tmp.path().join("data")).unwrap();

    let id = field
        .add_triplet(
            "chitta".into(),
            "replaces".into(),
            "duckdb".into(),
            1.0,
            None,
            None,
        )
        .unwrap();
    assert!(id > 0);

    let results = field.query_subject("chitta").unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].object, "duckdb");
}

#[test]
fn test_replay_triplets() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");

    {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        field
            .add_triplet("a".into(), "b".into(), "c".into(), 1.0, None, None)
            .unwrap();
    }

    let field2 = ChittaField::open(data_dir).unwrap();
    let results = field2.query_subject("a").unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_integration_invalidate_triplet() {
    let tmp = TempDir::new().unwrap();
    let field = ChittaField::open(tmp.path().join("data")).unwrap();

    let id = field
        .add_triplet(
            "chitta".into(),
            "uses".into(),
            "duckdb".into(),
            1.0,
            None,
            None,
        )
        .unwrap();

    let before = field.query_subject("chitta").unwrap();
    assert_eq!(before.len(), 1);

    field.invalidate_triplet(id).unwrap();

    let after = field.query_subject("chitta").unwrap();
    assert_eq!(after.len(), 0);
}

#[test]
fn test_supersede_survives_wal_replay() {
    // Regression (SOTA review 2b): supersession was RAM-only + serde(skip),
    // persisted solely via the .sup.json snapshot sidecar. With no snapshot
    // taken, a reopen replays purely from the WAL — so the revision survives
    // ONLY if SupersedeTriplet is a real op. Before the fix, old_id reappeared.
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let old_id;
    {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        old_id = field
            .add_triplet("chitta".into(), "uses".into(), "duckdb".into(), 1.0, None, None)
            .unwrap();
        let new_id = field
            .add_triplet("chitta".into(), "uses".into(), "sqlite".into(), 1.0, None, None)
            .unwrap();
        field.triplet_supersede(old_id, new_id, now_ms()).unwrap();

        let live = field.query_subject_as_of("chitta", i64::MAX).unwrap();
        assert!(!live.iter().any(|e| e.id == old_id), "superseded pre-reopen");
    }

    // Reopen: no snapshot was written, so this is a pure WAL replay.
    let field2 = ChittaField::open(data_dir.clone()).unwrap();
    let after = field2.query_subject_as_of("chitta", i64::MAX).unwrap();
    assert!(
        !after.iter().any(|e| e.id == old_id),
        "supersession must survive WAL-only replay"
    );
}

#[test]
fn test_integration_query_entity() {
    let tmp = TempDir::new().unwrap();
    let field = ChittaField::open(tmp.path().join("data")).unwrap();

    field
        .add_triplet(
            "alice".into(),
            "knows".into(),
            "bob".into(),
            1.0,
            None,
            None,
        )
        .unwrap();
    field
        .add_triplet(
            "charlie".into(),
            "knows".into(),
            "alice".into(),
            1.0,
            None,
            None,
        )
        .unwrap();
    field
        .add_triplet(
            "alice".into(),
            "works_at".into(),
            "anthropic".into(),
            1.0,
            None,
            None,
        )
        .unwrap();

    let results = field.query_entity("alice").unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_integration_recall_keyword() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];

    field
        .put_memory(
            "wisdom",
            "test",
            b"rust ownership model prevents memory leaks automatically",
            &emb,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();
    field
        .put_memory(
            "wisdom",
            "test",
            b"python garbage collector handles memory management",
            &emb,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();

    let hits = field.recall_keyword("rust ownership", 5).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].kind, "wisdom");
    // "rust" and "ownership" only in doc 1
    assert!(hits[0].content.contains("rust"));
}

#[test]
fn test_recall_effects_are_deferred_until_flush() {
    let (field, _tmp) = open_test_field();

    let mut emb1 = vec![0.0f32; crate::ops::EMBED_DIM];
    emb1[0] = 1.0;
    let mut emb2 = vec![0.0f32; crate::ops::EMBED_DIM];
    emb2[1] = 1.0;

    field
        .put_memory(
            "wisdom",
            "test",
            b"alpha memory",
            &emb1,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();
    field
        .put_memory(
            "wisdom",
            "test",
            b"beta memory",
            &emb2,
            1.0,
            0.001,
            0,
            vec![],
            None,
            None,
        )
        .unwrap();

    let seqno_before = field.log.read().last_seqno();
    let hits = field.recall_semantic(&emb1, 2, Some("test")).unwrap();
    assert!(!hits.is_empty());
    assert_eq!(field.log.read().last_seqno(), seqno_before);
    assert!(!field.pending_recall.lock().strengthen.is_empty());

    field.flush().unwrap();
    assert!(field.log.read().last_seqno() > seqno_before);
    assert!(field.pending_recall.lock().strengthen.is_empty());
}

// ── Status-aware recall tests ─────────────────────────────────────────────

/// Superseded/Contradicted/Archived memories must be excluded from semantic recall.
#[test]
fn test_recall_excludes_invalidated_statuses() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.5f32; crate::ops::EMBED_DIM];

    let (id_active, _)     = field.put_memory("wisdom", "test", b"active memory",     &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_superseded, _) = field.put_memory("wisdom", "test", b"superseded memory", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_contradicted,_)= field.put_memory("wisdom", "test", b"contradicted memory",&emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_archived, _)   = field.put_memory("wisdom", "test", b"archived memory",   &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    field.set_memory_status(id_superseded,    crate::state::MemoryStatus::Superseded).unwrap();
    field.set_memory_status(id_contradicted,  crate::state::MemoryStatus::Contradicted).unwrap();
    field.set_memory_status(id_archived,      crate::state::MemoryStatus::Archived).unwrap();

    let hits = field.recall_semantic(&emb, 20, None).unwrap();
    let ids: Vec<_> = hits.iter().map(|h| h.memory_id).collect();

    assert!(ids.contains(&id_active),        "active memory must be recalled");
    assert!(!ids.contains(&id_superseded),   "superseded must be excluded");
    assert!(!ids.contains(&id_contradicted), "contradicted must be excluded");
    assert!(!ids.contains(&id_archived),     "archived must be excluded");
}

/// Verified memories score higher than Active; Proposed score lower.
#[test]
fn test_recall_status_score_ordering() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.5f32; crate::ops::EMBED_DIM];

    let (id_active,   _) = field.put_memory("wisdom", "test", b"active",   &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_verified, _) = field.put_memory("wisdom", "test", b"verified", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_proposed, _) = field.put_memory("wisdom", "test", b"proposed", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    field.set_memory_status(id_verified, crate::state::MemoryStatus::Verified).unwrap();
    field.set_memory_status(id_proposed, crate::state::MemoryStatus::Proposed).unwrap();

    let hits = field.recall_semantic(&emb, 20, None).unwrap();
    let score = |id: MemoryId| hits.iter().find(|h| h.memory_id == id).map(|h| h.score).unwrap_or(0.0);

    assert!(score(id_verified) > score(id_active),  "verified must outscore active");
    assert!(score(id_active)   > score(id_proposed), "active must outscore proposed");
}

// ── Recall explainability tests ─────────────────────────────────────────

#[test]
fn test_recall_explain_fields_populated() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.5f32; crate::ops::EMBED_DIM];

    let (id, _) = field.put_memory("wisdom", "test", b"tool derived memory", &emb, 0.9, 0.001, 0, vec![], None, None).unwrap();
    field.set_epistemic_status(id, crate::state::EpistemicStatus::ToolDerived).unwrap();

    let hits = field.recall_semantic(&emb, 5, Some("test")).unwrap();
    let hit = hits.iter().find(|h| h.memory_id == id).expect("memory must be recalled");

    assert!(hit.semantic_weight > 0.0, "semantic_weight must be > 0");
    assert!((hit.status_mul - 1.0).abs() < f32::EPSILON, "Active status_mul must be 1.0");
    assert!((hit.epistemic_mul - 0.95).abs() < f32::EPSILON, "ToolDerived epistemic_mul must be 0.95");
    assert!(hit.strength_factor >= 0.5 && hit.strength_factor <= 1.0, "strength_factor must be in [0.5, 1.0]");
}

#[test]
fn test_recall_explain_score_decomposition() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.5f32; crate::ops::EMBED_DIM];

    let (id, _) = field.put_memory("wisdom", "test", b"decomposition test", &emb, 0.8, 0.001, 0, vec![], None, None).unwrap();

    let hits = field.recall_semantic(&emb, 5, Some("test")).unwrap();
    let hit = hits.iter().find(|h| h.memory_id == id).expect("memory must be recalled");

    // Score is the product of all pipeline factors:
    // relevance × actr × strength × confidence × surprise × arousal × mood × frustration
    // × status × epistemic × kind × realm_reliability
    // For a fresh memory with default config, most boosts are 1.0.
    // Just verify score is positive and decomp fields are populated.
    assert!(hit.score > 0.0, "score must be positive");
    assert!(hit.strength_factor >= 0.5, "strength_factor must be >= 0.5");
    assert!(hit.semantic_weight > 0.0, "semantic_weight must be > 0");
    assert!(hit.status_mul > 0.0, "status_mul must be > 0");
    assert!(hit.epistemic_mul > 0.0, "epistemic_mul must be > 0");
}

#[test]
fn test_recall_keyword_explain_fields() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];

    field.put_memory("wisdom", "test", b"rust ownership borrow checker lifetime", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    let hits = field.recall_keyword("rust ownership", 5).unwrap();
    assert!(!hits.is_empty(), "keyword recall must return results");
    let hit = &hits[0];

    assert!(hit.semantic_weight > 0.0, "semantic_weight must be bm25_score > 0");
    assert!(hit.status_mul > 0.0, "status_mul must be populated");
    assert!(hit.epistemic_mul > 0.0, "epistemic_mul must be populated");
    assert!(hit.strength_factor >= 0.5, "strength_factor must be >= 0.5");
}

// ── Contradiction engine tests ──────────────────────────────────────────

#[test]
fn test_get_conflicts_bidirectional() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id_a, _) = field.put_memory("wisdom", "test", b"memory A", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_b, _) = field.put_memory("wisdom", "test", b"memory B", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    field.add_triplet(id_a.to_string(), "contradicts".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

    let conflicts_a = field.get_conflicts(id_a).unwrap();
    let conflicts_b = field.get_conflicts(id_b).unwrap();
    assert!(conflicts_a.contains(&id_b), "A must see B as conflict");
    assert!(conflicts_b.contains(&id_a), "B must see A as conflict");
}

#[test]
fn test_get_supersession_chain_follows_edges() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id_a, _) = field.put_memory("wisdom", "test", b"original", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_b, _) = field.put_memory("wisdom", "test", b"revision 1", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_c, _) = field.put_memory("wisdom", "test", b"revision 2", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    // "B supersedes A" means subject=B, predicate="supersedes", object=A
    field.add_triplet(id_b.to_string(), "supersedes".to_string(), id_a.to_string(), 1.0, None, None).unwrap();
    field.add_triplet(id_c.to_string(), "supersedes".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

    let chain = field.get_supersession_chain(id_a).unwrap();
    assert_eq!(chain, vec![id_a, id_b, id_c], "chain must follow A -> B -> C");
}

#[test]
fn test_get_supersession_chain_cycle_safe() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id_a, _) = field.put_memory("wisdom", "test", b"cycle A", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_b, _) = field.put_memory("wisdom", "test", b"cycle B", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    // Create a cycle: B supersedes A, A supersedes B
    field.add_triplet(id_b.to_string(), "supersedes".to_string(), id_a.to_string(), 1.0, None, None).unwrap();
    field.add_triplet(id_a.to_string(), "supersedes".to_string(), id_b.to_string(), 1.0, None, None).unwrap();

    let chain = field.get_supersession_chain(id_a).unwrap();
    assert!(chain.len() <= 21, "cycle must terminate within max depth");
    assert_eq!(chain[0], id_a, "chain must start with self");
}

#[test]
fn test_get_confirmations() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id_x, _) = field.put_memory("wisdom", "test", b"confirmer", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();
    let (id_y, _) = field.put_memory("wisdom", "test", b"confirmed", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    // "X confirms Y" means subject=X, predicate="confirms", object=Y
    field.add_triplet(id_x.to_string(), "confirms".to_string(), id_y.to_string(), 1.0, None, None).unwrap();

    let confs = field.get_confirmations(id_y).unwrap();
    assert_eq!(confs, vec![id_x], "Y must show X as confirmer");
}

#[test]
fn test_get_conflicts_empty() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    let (id, _) = field.put_memory("wisdom", "test", b"lonely memory", &emb, 1.0, 0.001, 0, vec![], None, None).unwrap();

    let conflicts = field.get_conflicts(id).unwrap();
    assert!(conflicts.is_empty(), "no contradictions should return empty vec");
}

// ── Regression tests for replay/contract correctness ─────────────────────

fn put_test_memory(field: &ChittaField, content: &[u8]) -> MemoryId {
    let emb = vec![0.1f32; crate::ops::EMBED_DIM];
    field.put_memory("wisdom", "test", content, &emb, 1.0, 0.001, 0, vec![], None, None)
        .unwrap().0
}

#[test]
fn chaos_unlinked_wal_preserves_n_plus_m_memories() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    let mut ids = Vec::new();
    {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        for n in 0..5 {
            ids.push(put_test_memory(&field, format!("before unlink {n}").as_bytes()));
        }
        field.flush().unwrap();
        // One writer, one cycle, no snapshot: all N records must come from
        // the unlinked descriptor, not a snapshot-covered prefix.
        for entry in std::fs::read_dir(data_dir.join("segments")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "seg") {
                std::fs::remove_file(path).unwrap();
            }
        }
        for n in 0..7 {
            ids.push(put_test_memory(&field, format!("after unlink {n}").as_bytes()));
        }
        field.flush().unwrap();
    }
    assert_eq!(ids.iter().copied().collect::<std::collections::HashSet<_>>().len(), 12);
    let reopened = ChittaField::open(data_dir).unwrap();
    for (n, id) in ids.into_iter().enumerate() {
        let text = if n < 5 { format!("before unlink {n}") }
                   else { format!("after unlink {}", n - 5) };
        assert_eq!(reopened.get_memory(id).unwrap().content, text.as_bytes());
    }
}

/// Bug fix: UpdateState replay used now_ms=0, corrupting last_accessed_ms and
/// last_strengthened_ms. After reopen the timestamps must reflect op_ts_ms, not epoch 0.
#[test]
fn test_replay_update_state_timestamps_nonzero() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let id = put_test_memory(&field, b"state-replay");
        // touch=true writes an UpdateState op with real op_ts_ms
        field.update_state(id, None, None, None, true, None).unwrap();
        field.flush().unwrap();
        id
    };

    let field2 = ChittaField::open(data_dir).unwrap();
    let state = field2.get_state(id).unwrap();
    assert!(
        state.last_accessed_ms > 0,
        "last_accessed_ms must not be 0 after replay, got {}",
        state.last_accessed_ms
    );
}

/// Bug fix: UpdateMemoryContent replay did not clear embed_pending, so backfilled
/// memories were re-queued as pending after every restart.
#[test]
fn test_replay_backfill_clears_embed_pending() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");

    let id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        // Empty embedding slice → embed_pending = true
        let (id, _) = field.put_memory("wisdom", "test", b"this memory needs an embedding backfill", &[], 1.0, 0.001, 0, vec![], None, None).unwrap();
        let emb = vec![0.2f32; crate::ops::EMBED_DIM];
        field.backfill_embedding(id, &emb).unwrap();
        field.flush().unwrap();
        id
    };

    let field2 = ChittaField::open(data_dir).unwrap();
    assert!(
        !field2.pending_embeddings(100).contains(&id),
        "backfilled memory must not appear in pending_embeddings after replay"
    );
}

/// Bug fix: backfill_embedding() previously returned Ok(()) for nonexistent IDs.
#[test]
fn test_backfill_nonexistent_returns_not_found() {
    let (field, _tmp) = open_test_field();
    let emb = vec![0.0f32; crate::ops::EMBED_DIM];
    let fake_id: MemoryId = 0xdeadbeef_cafebabe;
    let result = field.backfill_embedding(fake_id, &emb);
    assert!(
        matches!(result, Err(crate::error::FieldError::NotFound(_))),
        "expected NotFound, got {:?}", result
    );
}

/// Bug fix: set_memory_status() and set_epistemic_status() wrote WAL before
/// confirming the memory exists, leaving orphaned WAL entries on invalid IDs.
#[test]
fn test_set_status_invalid_id_no_wal_mutation() {
    let (field, _tmp) = open_test_field();
    let fake_id: MemoryId = 0xdeadbeef_00000001;
    let seqno_before = field.log.read().last_seqno();

    let r1 = field.set_memory_status(fake_id, crate::state::MemoryStatus::Archived);
    let r2 = field.set_epistemic_status(fake_id, crate::state::EpistemicStatus::ModelInferred);

    assert!(matches!(r1, Err(crate::error::FieldError::NotFound(_))));
    assert!(matches!(r2, Err(crate::error::FieldError::NotFound(_))));
    assert_eq!(
        field.log.read().last_seqno(), seqno_before,
        "WAL must not grow when ID is invalid"
    );
}

#[test]
fn test_compact_wal_guard_rejects_small_store() {
    let (field, _tmp) = open_test_field();
    let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
    for i in 0..50 {
        field
            .put_memory(
                "wisdom",
                "test",
                format!("memory {}", i).as_bytes(),
                &embedding,
                0.9,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
    }
    let result = field.compact_wal();
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("refusing compact_wal"),
        "expected guard error, got: {}", err_msg
    );
}

#[test]
fn test_compact_wal_guard_allows_large_store() {
    let (field, _tmp) = open_test_field();
    let embedding = vec![0.1f32; crate::ops::EMBED_DIM];
    for i in 0..100 {
        field
            .put_memory(
                "wisdom",
                "test",
                format!("memory {}", i).as_bytes(),
                &embedding,
                0.9,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
    }
    let result = field.compact_wal();
    assert!(result.is_ok(), "compact_wal should succeed with 100+ memories, got: {:?}", result);
}

#[test]
fn test_filter_level_signatures_reduces_terms() {
    let (field, _tmp) = open_test_field();
    field.set_filter_level(FilterLevel::Signatures);
    let code = b"fn foo(x: i32) -> i32 {\n    let y = x + 1;\n    y\n}";
    let (id, _) = field
        .put_memory("code", "test", code, &[], 0.8, 0.001, 0, vec![], None, None)
        .unwrap();
    let hits = field.recall_keyword("fn foo", 5).unwrap();
    assert!(hits.iter().any(|h| h.memory_id == id));
    let body_hits = field.recall_keyword("let y", 5).unwrap();
    assert!(!body_hits.iter().any(|h| h.memory_id == id));
}

#[test]
fn test_recall_fallback_to_bm25() {
    let (field, _tmp) = open_test_field();
    for i in 0..15 {
        field
            .put_memory(
                "wisdom",
                "test",
                format!("unique_term_{i} content here").as_bytes(),
                &[],
                0.8,
                0.001,
                0,
                vec![],
                None,
                None,
            )
            .unwrap();
    }
    let hits = field
        .recall_with_fallback(&vec![0.0f32; crate::ops::EMBED_DIM], "unique_term_0", 5, None)
        .unwrap();
    assert!(!hits.is_empty(), "fallback should return results");
}

/// Capability #1 (anti-reprocessing): the keyed provenance lane resolves a
/// `[done]` record by content-hash OR input path through an exact O(1) lookup
/// that never touches the fuzzy retriever, keeps the earliest record when a
/// sha is re-registered under different surrounding text, survives snapshot
/// save/reopen (rebuilt deterministically from live payloads), and misses
/// cleanly for unknown keys.
#[test]
fn provenance_keyed_lane_exact_lookup_and_persists() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    let put = |field: &ChittaField, body: &str| {
        field
            .put_memory("signal", "cc-soul", body.as_bytes(), &[], 0.8, 0.001, 0, vec![], None, None)
            .unwrap()
            .0
    };

    let (first_id, second_id) = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let a = put(&field, "[done] input:/data/x.fastq sha:deadbeef12 task:qc output:/out/x.tsv status:ok");
        // Same sha, different task text: escapes the byte-identical string gate,
        // so it stores as a second live record — the keyed lane must still
        // resolve to the EARLIEST record.
        let b = put(&field, "[done] input:/data/x.fastq sha:deadbeef12 task:requant output:/out/x2.tsv status:ok");
        assert_ne!(a, b, "same-sha different-text must be two records");

        assert_eq!(field.provenance_lookup("deadbeef12", "").unwrap().0, a, "sha lookup -> earliest");
        assert_eq!(field.provenance_lookup("", "/data/x.fastq").unwrap().0, a, "path lookup -> earliest");
        assert_eq!(field.provenance_lookup("sha:deadbeef12", "").unwrap().0, a, "prefixed arg resolves");
        assert!(field.provenance_lookup("deadbeef12", "").unwrap().1.contains("task:qc"), "returns earliest content");
        assert!(field.provenance_lookup("cafef00d99", "/nope").is_none(), "unknown key misses cleanly");

        field.save_full_snapshot().unwrap();
        (a, b)
    };
    assert_ne!(first_id, second_id);

    // Reopen: lane rebuilt from live payloads, deterministically earliest.
    let field = ChittaField::open(data_dir).unwrap();
    assert_eq!(field.provenance_lookup("deadbeef12", "").unwrap().0, first_id, "rebuilt lane keeps earliest after reopen");
    assert!(field.provenance_lookup("deadbeef12", "").unwrap().1.contains("task:qc"));
    assert!(field.provenance_lookup("cafef00d99", "").is_none());
}

/// Capability #2: durable corrections with override semantics. Proves the
/// correction keyed lane fires deterministically when the corrected
/// mistake's trigger recurs in a turn (an exact bigram probe, no fuzzy
/// retriever), applies LATEST-WINS + SUPERSEDE when a newer correction
/// shares a trigger, misses cleanly on unrelated turns, and rebuilds
/// deterministically (newest-wins) across a snapshot save/reopen.
#[test]
fn correction_keyed_lane_fires_latest_wins_and_persists() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    let put = |field: &ChittaField, body: &str| {
        field
            .put_memory("correction", "cc-soul", body.as_bytes(), &[], 0.9, 0.001, 0, vec![], None, None)
            .unwrap()
            .0
    };

    let old_id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        // A correction whose mistake is "cp over the running binary".
        let a = put(&field, "[correction] USE: install atomic rename\nNOT: cp over the running binary");

        // Trigger recurs in a turn -> the correction FIRES (the exact bigram
        // "running_binary"/"cp_over"... survives after stopword strip).
        let hit = field.correction_check("can I just cp over the running binary to deploy?");
        assert!(hit.is_some(), "recurring mistake trigger must fire the correction");
        assert_eq!(hit.as_ref().unwrap().0, a, "fires the stored correction id");
        assert!(hit.unwrap().1.contains("install atomic rename"), "surfaces the USE: fix");

        // Unrelated turn -> clean miss (no fuzzy false-positive).
        assert!(field.correction_check("what's the weather in the terminal today").is_none(),
                "unrelated turn must miss cleanly");

        // Too-short correction (< 2 significant tokens) never indexes -> stays
        // on the fuzzy lane, so it can't be found by the keyed probe.
        let _short = put(&field, "[correction] USE: yes\nNOT: no");
        assert!(field.correction_check("no").is_none(), "sub-bigram correction is not keyed");

        // LATEST-WINS / SUPERSEDE: a newer correction sharing the trigger
        // replaces the old one for that key.
        let b = put(&field, "[correction] USE: install -m 0755 atomic\nNOT: cp over the running binary ETXTBSY");
        let hit2 = field.correction_check("cp over the running binary now").unwrap();
        assert_eq!(hit2.0, b, "newest correction supersedes older for a shared trigger");
        assert!(hit2.1.contains("ETXTBSY"), "surfaces the superseding correction body");
        assert_ne!(a, b);

        field.save_full_snapshot().unwrap();
        a
    };

    // Reopen: lane rebuilt from live payloads, deterministically newest-wins.
    let field = ChittaField::open(data_dir).unwrap();
    let hit = field.correction_check("cp over the running binary now").unwrap();
    assert_ne!(hit.0, old_id, "rebuilt lane keeps the SUPERSEDING correction after reopen");
    assert!(hit.1.contains("ETXTBSY"), "rebuilt lane surfaces newest body");
    assert!(field.correction_check("totally unrelated question").is_none());
}

#[test]
fn correction_uppercase_header_form_is_keyed() {
    // The free-form `[CORRECTION to memory #… — topic]` header (uppercase,
    // extra words) is how most real corrections are stored. The old
    // lowercase-exact gate rejected it, silently disabling the keyed lane.
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let field = ChittaField::open(data_dir).unwrap();
    let body = "[CORRECTION to memory #2411395996631171237 — reuse background] \
                realm:pan-barley. My earlier claim \"reuse has 0 AFDB background by \
                construction\" was WRONG. Cause: spine_member_col omits AFDB scaffold.";
    let id = field
        .put_memory("correction", "brahman", body.as_bytes(), &[], 0.9, 0.001, 0, vec![], None, None)
        .unwrap()
        .0;
    let hit = field.correction_check("reuse has 0 AFDB background by construction");
    assert!(hit.is_some(), "uppercase-header correction must fire on its mistake restatement");
    assert_eq!(hit.unwrap().0, id, "fires the stored uppercase-header correction id");
}

#[test]
fn correction_multivalued_credits_shared_bigrams() {
    // Older correction A's mistake shares 2 of its 3 bigrams with a NEWER
    // correction B. Single-valued latest-wins would let B steal both keys, so
    // A could never reach the 2-bigram threshold on its own restatement (only
    // B — the wrong correction — would fire). The multi-valued index credits A
    // for every bigram it owns, so A's restatement still surfaces A.
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let field = ChittaField::open(data_dir).unwrap();
    let put = |body: &str| field
        .put_memory("correction", "cc-soul", body.as_bytes(), &[], 0.9, 0.001, 0, vec![], None, None)
        .unwrap()
        .0;
    put("[correction] USE: quartz lattice mica\nNOT: alpha beta gamma delta");
    let _newer = put("[correction] USE: other fix\nNOT: alpha beta gamma zeta");
    let hit = field
        .correction_check("alpha beta gamma delta")
        .expect("A's restatement must fire despite B stealing shared bigrams");
    assert!(hit.1.contains("quartz lattice mica"),
            "multi-valued index must credit older A for its shared bigrams (single-valued drops it)");
}

/// Capability #3: task hand-off across discontinuous sessions. Proves the
/// task-state keyed lane resolves a `[task]` record by its slug through an
/// exact O(1) lookup, applies LATEST-WINS when the status evolves
/// (in-progress -> done), returns the DONE record after the update, misses
/// cleanly on an unknown id, and rebuilds deterministically (newest-wins)
/// across a snapshot save/reopen.
#[test]
fn task_state_keyed_lane_latest_wins_and_persists() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();

    // Stored kind:signal like the [job]/[done] rituals; the lane self-gates
    // on the [task] prefix, not the kind.
    let put = |field: &ChittaField, body: &str| {
        field
            .put_memory("signal", "cc-soul", body.as_bytes(), &[], 0.8, 0.001, 0, vec![], None, None)
            .unwrap()
            .0
    };

    let done_id = {
        let field = ChittaField::open(data_dir.clone()).unwrap();
        let a = put(&field, "[task] id:foo status:in-progress next:wire the FFI layer");
        // Same task, evolved status -> a second live record; the lane must
        // resolve to the LATEST (done), not the first (in-progress).
        let b = put(&field, "[task] id:foo status:done next:ship it");
        assert_ne!(a, b, "evolving status stores as two records");

        let hit = field.task_state_lookup("foo").unwrap();
        assert_eq!(hit.0, b, "slug lookup -> LATEST record (recency wins)");
        assert!(hit.1.contains("status:done"), "returns the newest status");
        assert_eq!(field.task_state_lookup("task:foo").unwrap().0, b, "prefixed arg resolves");
        assert!(field.task_state_lookup("bar").is_none(), "unknown id misses cleanly");

        field.save_full_snapshot().unwrap();
        b
    };

    // Reopen: lane rebuilt from live payloads, deterministically newest-wins.
    let field = ChittaField::open(data_dir).unwrap();
    let hit = field.task_state_lookup("foo").unwrap();
    assert_eq!(hit.0, done_id, "rebuilt lane keeps the LATEST record after reopen");
    assert!(hit.1.contains("status:done"), "rebuilt lane surfaces newest status");
    assert!(field.task_state_lookup("bar").is_none());
}

// ── Deferred-batched-insert (Step-1 re-architecture) proofs ────────────────

fn lcg_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..dim).map(|_| {
        s ^= s >> 33; s = s.wrapping_mul(0xff51afd7ed558ccd); s ^= s >> 33;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    }).collect()
}
fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// Recall@k PARITY gate: the deferred-batched insert path must match the per-item
/// path. For the PRODUCTION recall path (flat scan over the embeddings map, active
/// below FLAT_SCAN_MAX=2M) the two are bitwise-identical (same `upsert_meta`); for
/// the HNSW fallback graph they must be approximation-equivalent.
#[test]
fn test_deferred_batch_recall_parity() {
    use crate::hnsw::SemanticIndex;
    use crate::ops::EMBED_DIM;
    const N: u64 = 5200;   // > HNSW_TIER2_THRESHOLD (5000): exercises the delta tier
    const Q: usize = 30;
    const K: usize = 10;

    let items: Vec<(u64, Vec<f32>)> = (1..=N).map(|i| (i, lcg_vec(i, EMBED_DIM))).collect();

    // Model production: a pre-existing base graph (built at load by the epoch-based
    // backfill_hnsw_delta_parallel), then INCREMENTAL backfill of new memories. Seed
    // both paths identically, then diverge: A per-item upsert, B deferred-batched.
    const SEED: usize = 3000;
    let mut a = SemanticIndex::new();
    let mut b = SemanticIndex::new();
    // Build the HNSW graph regardless of size: below flat_scan_max() production
    // serves via flat scan and never builds it, but this test probes the graph
    // directly (hnsw_search_test) to compare per-item vs batched construction.
    a.force_hnsw_build_for_test();
    b.force_hnsw_build_for_test();
    for (id, e) in &items[..SEED] {
        a.upsert(*id, e.clone(), Some("test"));
        b.upsert(*id, e.clone(), Some("test"));
    }
    // Incremental tail — the path under test.
    for (id, e) in &items[SEED..] { a.upsert(*id, e.clone(), Some("test")); }
    for (id, e) in &items[SEED..] { b.upsert_deferred(*id, e.clone(), Some("test")); }
    for chunk in items[SEED..].chunks(100) {   // 100 = daemon pending_embeddings(100) batch
        let ids: Vec<u64> = chunk.iter().map(|(id, _)| *id).collect();
        let plan = b.plan_delta_batch(&ids);
        b.apply_delta_batch(plan);
    }

    // Ground-truth brute-force cosine top-k.
    let normed: Vec<(u64, Vec<f32>)> = items.iter().map(|(id, e)| (*id, unit(e))).collect();
    let (mut flat_identical, mut ra, mut rb) = (0usize, 0.0f64, 0.0f64);
    for qi in 0..Q {
        let q = unit(&lcg_vec(1_000_000 + qi as u64, EMBED_DIM));
        let mut bf: Vec<(f32, u64)> = normed.iter()
            .map(|(id, e)| (e.iter().zip(&q).map(|(x, y)| x * y).sum::<f32>(), *id)).collect();
        bf.sort_unstable_by(|x, y| y.0.total_cmp(&x.0));
        let truth: std::collections::HashSet<u64> = bf.iter().take(K).map(|(_, id)| *id).collect();

        // (a) Production path (default env = flat scan): must be IDENTICAL A vs B.
        let fa: Vec<u64> = a.search(&q, K, None, None).into_iter().map(|h| h.memory_id).collect();
        let fb: Vec<u64> = b.search(&q, K, None, None).into_iter().map(|h| h.memory_id).collect();
        if fa == fb { flat_identical += 1; }

        // (b) HNSW fallback graph: approximation-equivalent quality.
        let ha: std::collections::HashSet<u64> = a.hnsw_search_test(&q, K).into_iter().map(|h| h.memory_id).collect();
        let hb: std::collections::HashSet<u64> = b.hnsw_search_test(&q, K).into_iter().map(|h| h.memory_id).collect();
        ra += truth.intersection(&ha).count() as f64 / K as f64;
        rb += truth.intersection(&hb).count() as f64 / K as f64;
    }
    ra /= Q as f64; rb /= Q as f64;
    eprintln!("[parity] flat-path identical: {flat_identical}/{Q}  |  HNSW recall@{K}: per-item={ra:.3} batched={rb:.3}");
    assert_eq!(flat_identical, Q, "production flat-scan recall must be identical A vs B");
    assert!(rb > 0.0, "batched HNSW path returns hits");
    assert!(rb >= ra - 0.05, "batched HNSW recall {rb:.3} within 0.05 of per-item {ra:.3}");
}

/// WRITES-DON'T-BLOCK-READS proof: measure the total EXCLUSIVE (write-lock) hold time
/// to insert a batch — this is exactly what blocks recall (recall needs the shared
/// lock; a held write lock stalls it). Per-item holds the write lock across the
/// O(log N) global + per-realm HNSW neighbor SEARCH for every item. The deferred path
/// holds it only for cheap metadata + the pointer-wire apply; the search runs under a
/// READ lock (plan), which recall can share. Stable metric (a sum, not a noisy
/// worst-case) — mirrors the daemon's [lockprof] EXCLUSIVE-hold reduction.
#[test]
fn test_deferred_batch_exclusive_hold() {
    use crate::hnsw::SemanticIndex;
    use crate::ops::EMBED_DIM;
    const SEED: u64 = 5000;    // > HNSW_TIER2_THRESHOLD: global delta tier active
    const BATCH: usize = 500;
    // 3 realms, each ~1/3 of the corpus (> 500): per-realm HNSW active too, so both
    // graph inserts (global + per-realm) are on the write path in the per-item case.
    let realm_of = |i: u64| -> &'static str { ["ra", "rb", "rc"][(i % 3) as usize] };
    let seed: Vec<(u64, Vec<f32>)> = (1..=SEED).map(|i| (i, lcg_vec(i, EMBED_DIM))).collect();
    let tail: Vec<(u64, Vec<f32>)> =
        (SEED + 1..=SEED + BATCH as u64).map(|i| (i, lcg_vec(i, EMBED_DIM))).collect();
    let build = || {
        let mut s = SemanticIndex::new();
        for (id, e) in &seed { s.upsert(*id, e.clone(), Some(realm_of(*id))); }
        s
    };

    // Per-item: every upsert holds the write lock across the HNSW neighbor search.
    let mut a = build();
    let mut hold_a = std::time::Duration::ZERO;
    for (id, e) in &tail {
        let t = std::time::Instant::now();
        a.upsert(*id, e.clone(), Some(realm_of(*id)));
        hold_a += t.elapsed();
    }

    // Batched: EXCLUSIVE hold = metadata write + apply write. The neighbor search
    // (plan_delta_batch) runs OFF the write lock and is NOT counted — the whole point.
    let mut b = build();
    let ids: Vec<u64> = tail.iter().map(|(id, _)| *id).collect();
    let t0 = std::time::Instant::now();
    for (id, e) in &tail { b.upsert_deferred(*id, e.clone(), Some(realm_of(*id))); }
    let mut hold_b = t0.elapsed();                 // metadata (exclusive)
    let plan = b.plan_delta_batch(&ids);           // OFF-LOCK (read) — not counted
    let t1 = std::time::Instant::now();
    b.apply_delta_batch(plan);
    hold_b += t1.elapsed();                         // apply (exclusive)

    eprintln!("[exclusive-hold] per-item={:?} batched={:?} (search moved off-lock into plan)",
              hold_a, hold_b);
    assert!(hold_b.as_micros() * 3 < hold_a.as_micros(),
        "batched exclusive hold {hold_b:?} must be <⅓ of per-item {hold_a:?}");
}


#[test]
fn measurement_recall_preserves_competitive_weights_and_refresh_timestamps() {
    let (field, _tmp) = open_test_field();
    let mut queries = Vec::new();
    for i in 0..24usize {
        let mut embedding = vec![0.0; crate::ops::EMBED_DIM];
        embedding[i] = 1.0;
        let (id, _) = field.put_memory("wisdom", "readonly",
            format!("independent measurement candidate {i}").as_bytes(),
            &embedding, 0.9, 0.0, 0, vec![], None, None).unwrap();
        let mut states = field.states.write();
        let state = states.get_mut(&id).unwrap();
        state.competitive_weight = 0.8;
        state.last_cw_refresh_ms = 0;
        queries.push(embedding);
    }
    let snapshot = || {
        let mut rows: Vec<_> = field.states.read().iter().map(|(&id, state)| {
            (id, state.competitive_weight.to_bits(), state.last_cw_refresh_ms,
             state.access_count, state.last_accessed_ms, state.strength.to_bits())
        }).collect();
        rows.sort_unstable();
        rows
    };
    let before = snapshot();
    for query in &queries {
        assert!(!field.recall_semantic_measure(query, 24, Some("readonly")).unwrap().is_empty());
    }
    assert_eq!(before, snapshot(), "no_learn must not alter scoring inputs for later queries");
    assert!(field.cw_refresh_inflight.read().is_empty());
    // Learning still refreshes stale weights; only the measurement path is read-only.
    field.recall_semantic(&queries[0], 24, Some("readonly")).unwrap();
    assert_ne!(before, snapshot());
}
