//! Read-path lock and persistence regressions.
use crate::{ChittaField, ops::EMBED_DIM};

fn put(field: &ChittaField) -> u64 {
    field.put_memory("wisdom", "test", b"read-path regression", &vec![0.1; EMBED_DIM],
        1.0, 0.001, 0, vec![], None, None).unwrap().0
}

#[test]
fn concurrent_get_and_span_need_no_write_or_log_lock() {
    let dir = tempfile::tempdir().unwrap();
    let field = ChittaField::open(dir.path().to_owned()).unwrap();
    let id = put(&field);
    let seq = field.log.read().last_seqno();
    let syncs = field.log.read().sync_count();
    // A regression to any of these writes deadlocks the child: bounded receive
    // ensures the test reports failure instead of hanging the whole test suite.
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        let states = field.states.read();
        let learners = field.learners.read();
        let spans = field.span_store.read();
        let log = field.log.write();
        for _ in 0..12 {
            let tx = tx.clone(); let field = &field;
            scope.spawn(move || {
                for _ in 0..17 {
                    assert!(!field.get_memory(id).unwrap().content.is_empty());
                    field.span_for_memory(id, 4);
                }
                tx.send(()).unwrap();
            });
        }
        let results: Vec<_> = (0..12).map(|_| rx.recv_timeout(std::time::Duration::from_secs(3))).collect();
        drop((states, learners, spans, log));
        assert!(results.iter().all(Result::is_ok));
    });
    assert_eq!(field.log.read().last_seqno(), seq);
    assert_eq!(field.log.read().sync_count(), syncs);
    assert_eq!(field.get_state(id).unwrap().access_count, 0);
    field.drain_pending_touches().unwrap();
    assert_eq!(field.log.read().last_seqno(), seq + 1, "ONE batch WAL record");
    assert_eq!(field.get_state(id).unwrap().access_count, 204);
    field.drain_pending_touches().unwrap();
    assert_eq!(field.log.read().last_seqno(), seq + 1, "empty drain writes nothing");
}

#[test]
fn access_batch_snapshot_and_shutdown_replay_exact_counts() {
    let dir = tempfile::tempdir().unwrap();
    let id;
    {
        let field = ChittaField::open(dir.path().to_owned()).unwrap();
        id = put(&field);
        for _ in 0..40 { field.get_memory(id).unwrap(); }
        field.save_full_snapshot().unwrap();
        assert_eq!(field.get_state(id).unwrap().access_count, 40);
        for _ in 0..7 { field.get_memory(id).unwrap(); }
    }
    for _ in 0..2 {
        let field = ChittaField::open(dir.path().to_owned()).unwrap();
        let state = field.get_state(id).unwrap();
        assert_eq!(state.access_count, 47, "snapshot coverage must avoid double application");
        assert_eq!(state.access_timestamps.len(), 16);
        assert!(state.last_accessed_ms > 0);
    }
}

#[test]
fn deferred_access_does_not_fence_out_explicit_strengthen() {
    let dir = tempfile::tempdir().unwrap();
    let field = ChittaField::open(dir.path().to_owned()).unwrap();
    let id = put(&field);
    field.pending_touches.lock().extend([(id, 100), (id, 100), (id, 101)]);
    field.update_state(id, Some(-0.1), None, None, true, None).unwrap();
    let before = field.get_state(id).unwrap();
    field.drain_pending_touches().unwrap();
    let after = field.get_state(id).unwrap();
    assert_eq!(after.access_count, before.access_count + 3);
    assert_eq!(after.strength, before.strength);
    assert_eq!(after.last_state_op_ts_ms, before.last_state_op_ts_ms);
    assert_eq!(after.last_accessed_ms, before.last_accessed_ms);
}

#[test]
fn second_open_of_the_same_store_is_refused() {
    let tmp = tempfile::TempDir::new().unwrap();
    let first = ChittaField::open(tmp.path().to_path_buf()).unwrap();
    let second = ChittaField::open(tmp.path().to_path_buf());
    let err = match second {
        Ok(_) => panic!("second instance must not open the same directory"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("another chitta-field instance"), "{err}");
    drop(first);
    ChittaField::open(tmp.path().to_path_buf()).expect("lock released with the first instance");
}

#[test]
fn append_recovers_when_the_segment_is_deleted_underneath() {
    let tmp = tempfile::TempDir::new().unwrap();
    let field = ChittaField::open(tmp.path().to_path_buf()).unwrap();
    let emb = vec![0.1f32; EMBED_DIM];
    let (first_id, _) = field
        .put_memory("wisdom", "project:t", b"before deletion", &emb, 1.0, 0.001, 0, vec![], None, None)
        .unwrap();
    let seg_dir = tmp.path().join("segments");
    for entry in std::fs::read_dir(&seg_dir).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
    let (second_id, _) = field
        .put_memory("wisdom", "project:t", b"after deletion", &emb, 1.0, 0.001, 0, vec![], None, None)
        .unwrap();
    field.log.write().sync().unwrap();
    let segments: Vec<_> = std::fs::read_dir(&seg_dir).unwrap().map(|e| e.unwrap().path()).collect();
    assert_eq!(segments.len(), 1, "a fresh segment replaces the deleted one: {segments:?}");
    assert!(std::fs::metadata(&segments[0]).unwrap().len() > crate::log::V3_HEADER_SIZE as u64);
    drop(field);
    let reopened = ChittaField::open(tmp.path().to_path_buf()).unwrap();
    assert!(reopened.get_memory(second_id).is_ok(), "op written after recovery must replay");
    assert!(reopened.get_memory(first_id).is_err(), "the op in the deleted segment is gone, by construction");
}

#[test]
fn open_prepares_turbo_before_returning() {
    let tmp = tempfile::tempdir().unwrap();
    let field = ChittaField::open(tmp.path().to_path_buf()).unwrap();
    // A real WAL operation gives the snapshot a nonzero sequence and payload identity.
    put(&field);
    let mut embedding = vec![0.0; crate::ops::EMBED_DIM];
    embedding[0] = 1.0;
    {
        let mut index = field.semantic_idx.write();
        for id in 1..=2001 { index.upsert(id, embedding.clone(), None); }
    }
    field.save_full_snapshot().unwrap();
    drop(field);
    let reopened = ChittaField::open(tmp.path().to_path_buf()).unwrap();
    let index = reopened.semantic_idx.read();
    assert!(index.plan_turbo_rebuild(0).is_none(), "open must publish the prepared index");
    assert!(!index.search(&embedding, 5, None, None).is_empty());
}
