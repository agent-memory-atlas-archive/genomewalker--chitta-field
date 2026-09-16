//! Ordered startup phases used by ChittaField::open_with_lock.
//! Keep snapshot selection, sidecar loading, WAL replay and index rebuilding in order.

use super::*;

pub(super) struct LoadedSnapshot {
    pub(super) payloads: HashMap<MemoryId, MemoryPayload>,
    pub(super) retrieval_surfaces: HashMap<MemoryId, Vec<u8>>,
    pub(super) recall_provenance: HashMap<MemoryId, std::collections::BTreeSet<crate::ids::InstanceId>>,
    pub(super) states: HashMap<MemoryId, MemoryState>,
    pub(super) assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub(super) artifacts: HashMap<String, ArtifactId>,
    pub(super) artifact_paths: HashMap<ArtifactId, String>,
    pub(super) semantic_idx: SemanticIndex,
    pub(super) time_idx: TemporalIndex,
    pub(super) artifact_idx: ArtifactIndex,
    pub(super) keyword_idx: KeywordIndex,
    pub(super) triplet_store: TripletStore,
    pub(super) symbol_idx: SymbolIndex,
    pub(super) call_graph: CallGraph,
    pub(super) code_files: CodeFileIndex,
    pub(super) cortical_idx: CorticalIndex,
    pub(super) session_registry: SessionRegistry,
    pub(super) transcript_registry: TranscriptRegistry,
    pub(super) task_registry: TaskRegistry,
    pub(super) user_model_registry: UserModelRegistry,
    pub(super) theme_organ: ThemeOrgan,
    pub(super) analytics_registry: AnalyticsRegistry,
    pub(super) msg_registry: MsgRegistry,
    pub(super) skill_registry: SkillRegistry,
    pub(super) agent_registry: AgentRegistry,
    pub(super) constraint_store: ConstraintStore,
    pub(super) trigger_store: TriggerStore,
    pub(super) predictor: AccessPredictor,
    pub(super) surprise_store: SurpriseStore,
    pub(super) epistemic_debt_store: EpistemicDebtStore,
    pub(super) integration_kernel: IntegrationKernel,
    pub(super) surprise_learning: SurpriseLearningStore,
    pub(super) wisdom_promotion: WisdomPromotionStore,
    pub(super) learned_scorer: LearnedScoringModel,
    pub(super) intervention_store: InterventionStore,
    pub(super) agent_protocol_store: AgentProtocolStore,
    pub(super) wisdom_lineage_store: WisdomLineageStore,
    pub(super) symbol_event_log: SymbolEventLog,
    pub(super) chunk_hash_idx: HashMap<crate::ids::ChunkHash, MemoryId>,
    pub(super) snapshot_coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub(super) snap_ack_scores: HashMap<MemoryId, i32>,
    pub(super) snap_correction_states: HashMap<u64, crate::organ::triplet::CorrectionState>,
    pub(super) snap_event_tape: crate::organ::event_tape::EventTape,
    pub(super) snap_decision_tape: crate::organ::decision_tape::DecisionTape,
    pub(super) snap_interaction_ledger: crate::organ::interaction_ledger::InteractionLedger,
    pub(super) snap_predicate_store: crate::organ::predicate_store::PredicateStore,
    pub(super) snapshot_seqno: u64,
    pub(super) full_snapshot_seqno: u64,
    pub(super) best_full_path: Option<PathBuf>,
    pub(super) loaded_manifest: Option<crate::manifest::Manifest>,
    pub(super) loaded_snapshot_name: Option<String>,
    pub(super) loaded_header: Option<crate::snapshot::StoreHeader>,
    pub(super) migrate_reembed: bool,
    pub(super) reindex_mode: bool,
}

pub(super) fn load_snapshots(data_dir: &std::path::Path) -> Result<LoadedSnapshot> {
    let mut payloads: HashMap<MemoryId, MemoryPayload> = HashMap::new();
    let mut retrieval_surfaces: HashMap<MemoryId, Vec<u8>> = HashMap::new();
    let mut recall_provenance: HashMap<MemoryId, std::collections::BTreeSet<crate::ids::InstanceId>> = HashMap::new();
    let mut states: HashMap<MemoryId, MemoryState> = HashMap::new();
    let mut assoc_edges: HashMap<MemoryId, Vec<AssocEdge>> = HashMap::new();
    let mut artifacts: HashMap<String, ArtifactId> = HashMap::new();
    let mut artifact_paths: HashMap<ArtifactId, String> = HashMap::new();
    let mut semantic_idx = SemanticIndex::new();
    let mut time_idx = TemporalIndex::new();
    let mut artifact_idx = ArtifactIndex::new();
    let mut keyword_idx = KeywordIndex::new();
    let mut triplet_store = TripletStore::new();
    let mut symbol_idx = SymbolIndex::new();
    let mut call_graph = CallGraph::new();
    let mut code_files = CodeFileIndex::new();
    let mut cortical_idx = CorticalIndex::new();
    let mut session_registry = SessionRegistry::new();
    let transcript_registry = TranscriptRegistry::new();
    let task_registry = TaskRegistry::new();
    let user_model_registry = UserModelRegistry::new();
    let theme_organ = ThemeOrgan::new();
    let analytics_registry = AnalyticsRegistry::new();
    let mut msg_registry = MsgRegistry::new();
    let skill_registry = SkillRegistry::new();
    let agent_registry = AgentRegistry::new();
    let constraint_store = ConstraintStore::new();
    let trigger_store = TriggerStore::new();
    let predictor = AccessPredictor::new();
    let surprise_store = SurpriseStore::new();
    let epistemic_debt_store = EpistemicDebtStore::new();
    let integration_kernel = IntegrationKernel::new();
    let surprise_learning = SurpriseLearningStore::new();
    let wisdom_promotion = WisdomPromotionStore::new();
    let learned_scorer = LearnedScoringModel::new("v5.14".to_string());
    let intervention_store = InterventionStore::new();
    let agent_protocol_store = AgentProtocolStore::new();
    let wisdom_lineage_store = WisdomLineageStore::new();
    let symbol_event_log = SymbolEventLog::new();
    let chunk_hash_idx: HashMap<crate::ids::ChunkHash, MemoryId> = HashMap::new();
    let mut snapshot_coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats> = HashMap::new();
    let mut snap_ack_scores: HashMap<MemoryId, i32> = HashMap::new();
    let mut snap_correction_states: HashMap<u64, crate::organ::triplet::CorrectionState> = HashMap::new();
    let mut snap_event_tape    = crate::organ::event_tape::EventTape::new();
    let mut snap_decision_tape = crate::organ::decision_tape::DecisionTape::new();
    let mut snap_interaction_ledger = crate::organ::interaction_ledger::InteractionLedger::default();
    let mut snap_predicate_store    = crate::organ::predicate_store::PredicateStore::default();

    // Find best cortical snapshot by peeking seqno (16-byte read per file), then load only that one.
    let mut snapshot_seqno: u64 = 0;
    let mut best_cortex_path: Option<std::path::PathBuf> = None;
    let mut stale_cortex_paths: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("cortex.") && name_str.ends_with(".snapshot") {
                match CorticalIndex::peek_snapshot_seqno(&entry.path()) {
                    Ok(seqno) if seqno > snapshot_seqno => {
                        if let Some(prev) = best_cortex_path.replace(entry.path()) {
                            stale_cortex_paths.push(prev);
                        }
                        snapshot_seqno = seqno;
                    }
                    Ok(_) => stale_cortex_paths.push(entry.path()),
                    Err(e) => {
                        eprintln!(
                            "[chitta-field] skipping corrupt cortical snapshot {:?}: {}",
                            entry.path(),
                            e
                        );
                        stale_cortex_paths.push(entry.path());
                    }
                }
            }
        }
    }
    if let Some(ref path) = best_cortex_path {
        match CorticalIndex::load_snapshot(path) {
            Ok((loaded, seqno)) => {
                cortical_idx = loaded;
                snapshot_seqno = seqno;
                eprintln!(
                    "[chitta-field] loaded cortical snapshot seqno={} from {:?}",
                    seqno, path
                );
            }
            Err(e) => eprintln!(
                "[chitta-field] failed to load cortical snapshot {:?}: {}",
                path, e
            ),
        }
    }
    // Keep the 1 most recent stale cortex snapshot as a safety net.
    // Cortex is a cache (rebuilt from segments), so 1 backup is enough.
    let mut stale_cortex_by_seqno: Vec<(u64, &std::path::PathBuf)> = stale_cortex_paths
        .iter()
        .filter_map(|p| CorticalIndex::peek_snapshot_seqno(p).ok().map(|s| (s, p)))
        .collect();
    stale_cortex_by_seqno.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, path) in stale_cortex_by_seqno.iter().skip(1) {
        let _ = std::fs::remove_file(path);
    }

    // Find best full snapshot by peeking seqno (16-byte read per file), then load only that one.
    let mut full_snapshot_seqno: u64 = 0;
    let mut best_full_path: Option<std::path::PathBuf> = None;
    let mut stale_full_paths: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("chitta.") && name_str.ends_with(".snapshot") {
                match FullSnapshot::peek_seqno(&entry.path()) {
                    Ok(seqno) if seqno > full_snapshot_seqno => {
                        if let Some(prev) = best_full_path.replace(entry.path()) {
                            stale_full_paths.push(prev);
                        }
                        full_snapshot_seqno = seqno;
                    }
                    Ok(_) => stale_full_paths.push(entry.path()),
                    Err(e) => {
                        eprintln!(
                            "[chitta-field] skipping corrupt full snapshot {:?}: {}",
                            entry.path(),
                            e
                        );
                        stale_full_paths.push(entry.path());
                    }
                }
            }
        }
    }
    let had_full_snapshots = best_full_path.is_some() || !stale_full_paths.is_empty();
    let mut full_snapshot_loaded = false;
    // Try best snapshot first, then fall back to stale ones (sorted by seqno descending).
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(ref path) = best_full_path {
        candidates.push(path.clone());
    }
    // Sort stale paths by seqno descending so we try the most recent first.
    let mut stale_with_seqno: Vec<(u64, std::path::PathBuf)> = stale_full_paths
        .iter()
        .filter_map(|p| FullSnapshot::peek_seqno(p).ok().map(|s| (s, p.clone())))
        .collect();
    stale_with_seqno.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, p) in stale_with_seqno {
        candidates.push(p);
    }
    // Manifest-committed family takes precedence: a validated commit record
    // beats heuristic seqno-peek selection. Stores without a manifest (or
    // with a stale/invalid one) fall back to the fence stack below. Kept
    // around for the WAL coverage vector of the loaded family (§ replay).
    let loaded_manifest = crate::manifest::Manifest::load(&data_dir).ok().flatten();
    if let Some(manifest) = loaded_manifest.as_ref() {
        match manifest.validated_snapshot_path(&data_dir) {
            Some(committed) => {
                candidates.retain(|p| p != &committed);
                candidates.insert(0, committed);
                eprintln!(
                    "[chitta-field] manifest generation {}: committed snapshot family preferred",
                    manifest.generation
                );
            }
            None if manifest.checkpoints.is_some() => {
                eprintln!(
                    "[chitta-field] manifest family failed validation — using fence-based selection"
                );
            }
            None => {}
        }
    }
    let mut loaded_header: Option<crate::snapshot::StoreHeader> = None;
    let mut loaded_snapshot_name: Option<String> = None;
    // CHITTA_MIGRATE_REEMBED=1: one-shot embedding-space migration. Bypasses the dim/vsid
    // load fences and skips the foreign-dim embedding sidecars so a new-vsid binary can load
    // an old-vsid store's content + metadata (embeddings dropped → re_embed refills the
    // compiled vector space from content). OFF by default; never set in normal daemon operation.
    let migrate_reembed = std::env::var_os("CHITTA_MIGRATE_REEMBED").is_some();
    // Like migrate_reembed for the shdr model-ID fence, but still loads .emb so
    // ANN indices can be rebuilt from existing embeddings. For `chittad reindex`.
    let reindex_mode = std::env::var_os("CHITTA_REINDEX_MODE").is_some();
    for candidate in &candidates {
        let snapshot_phase = crate::profile::LoadPhase::new("snapshot");
        let loaded = FullSnapshot::load(candidate);
        drop(snapshot_phase);
        match loaded {
            Ok(mut snap) => {
                full_snapshot_seqno = snap.snapshot_seqno;
                // v11+: content in .pld sidecar; v10: content already in bincode (no-op).
                let pld_phase = crate::profile::LoadPhase::new("pld");
                let pld_loaded = FullSnapshot::load_payload_sidecar(&candidate.with_extension("pld"), &mut snap.payloads);
                drop(pld_phase);
                if !pld_loaded {
                    // A missing/torn .pld is harmless only if content is still in the
                    // bincode body (≤v10). For content-stripped snapshots (v11+) the .pld
                    // is the ONLY copy of content — reject this candidate and fall back to
                    // an older snapshot rather than silently serving empty content.
                    let stripped = !snap.payloads.is_empty()
                        && snap.payloads.values().all(|p| p.content.is_empty());
                    if stripped {
                        eprintln!(
                            "[chitta-field] REFUSING snapshot {:?}: .pld sidecar missing/corrupt \
                             but payload content was stripped — would serve empty content; trying fallback",
                            candidate
                        );
                        continue;
                    }
                }
                // Count embedded payloads by dim once — both the dim-fence and the
                // stale-.shdr check below need it.
                let mut embedded = 0usize;
                let mut matching = 0usize;
                let mut foreign_dim = 0u32;
                for p in snap.payloads.values() {
                    if p.embedding_dim > 0 {
                        embedded += 1;
                        if p.embedding_dim == crate::ops::EMBED_DIM as u32 {
                            matching += 1;
                        } else {
                            foreign_dim = p.embedding_dim;
                        }
                    }
                }
                // Dim-aware fencing: refuse a snapshot whose embedded payloads were
                // ALL written at a different EMBED_DIM than this binary compiles for.
                // Catches a wholesale 768-d snapshot winning on seqno after a 1536-d
                // migration (the contamination that motivated this guard); a partially
                // migrated store passes because any payload matching EMBED_DIM lets it
                // through, and a fresh store passes because it has no embedded payloads.
                if embedded > 0 && matching == 0 {
                    if !migrate_reembed {
                        eprintln!(
                            "[chitta-field] REFUSING snapshot {:?}: {} embedded payloads all at \
                             dim={} but this binary compiles EMBED_DIM={}; trying fallback",
                            candidate, embedded, foreign_dim, crate::ops::EMBED_DIM
                        );
                        continue;
                    }
                    eprintln!(
                        "[chitta-field] [migrate] accepting foreign-dim snapshot {:?} ({} payloads \
                         at dim={}) for re-embed to EMBED_DIM={}",
                        candidate, embedded, foreign_dim, crate::ops::EMBED_DIM
                    );
                }
                // Store-identity fencing (PR3): refuse a snapshot whose .shdr records a
                // different vector space (foreign model/dim/text-format) than this binary.
                // Fail-safe: a missing/legacy/corrupt .shdr is treated as same-lineage and
                // allowed (the dim-fence above still guards wholesale foreign-dim snapshots).
                {
                    let shdr_path = candidate.with_extension("shdr");
                    if let Some(hdr) = crate::snapshot::StoreHeader::load(&shdr_path) {
                        // A .shdr whose recorded dim disagrees with the snapshot's actual
                        // payload dim (which the dim-fence just validated against the compiled
                        // space) is a STALE stamp — left behind when a re-embed rewrote the data
                        // but an older writer's .shdr survived under a reused family id. The
                        // payload data is ground truth, so disregard the lying stamp and let the
                        // next save re-stamp it. Without this a correct-data store bricks on load.
                        let shdr_stale =
                            hdr.embed_dim != crate::ops::EMBED_DIM as u32 && matching > 0;
                        if hdr.matches_compiled() {
                            loaded_header = Some(hdr);
                        } else if shdr_stale {
                            eprintln!(
                                "[chitta-field] .shdr at {:?} is STALE (records dim={} but {} payloads \
                                 are at compiled dim={}); disregarding stamp, re-stamping on next save",
                                candidate, hdr.embed_dim, matching, crate::ops::EMBED_DIM
                            );
                            // accept: leave loaded_header = None → a fresh lineage is minted,
                            // which also escapes the stale writer's reused family id.
                        } else if !migrate_reembed && !reindex_mode {
                            eprintln!(
                                "[chitta-field] REFUSING snapshot {:?}: .shdr vector_space \
                                 (model={} dim={} tfv={}) != compiled (model={} dim={} tfv={}); \
                                 trying fallback",
                                candidate, hdr.model_id, hdr.embed_dim, hdr.text_format_version,
                                crate::ops::EMBED_MODEL_ID, crate::ops::EMBED_DIM,
                                crate::ops::TEXT_FORMAT_VERSION
                            );
                            continue;
                        } else if reindex_mode {
                            eprintln!(
                                "[chitta-field] [reindex] accepting foreign-vsid snapshot {:?} \
                                 (model={} dim={} tfv={}) to rebuild ANN from existing embeddings",
                                candidate, hdr.model_id, hdr.embed_dim, hdr.text_format_version,
                            );
                        } else {
                            eprintln!(
                                "[chitta-field] [migrate] accepting foreign-vsid snapshot {:?} (.shdr \
                                 model={} dim={} tfv={}) for re-embed to compiled (model={} dim={} tfv={})",
                                candidate, hdr.model_id, hdr.embed_dim, hdr.text_format_version,
                                crate::ops::EMBED_MODEL_ID, crate::ops::EMBED_DIM,
                                crate::ops::TEXT_FORMAT_VERSION
                            );
                            // Do NOT adopt the foreign header as our identity; the migration
                            // re-embeds + re-stamps a fresh .shdr at the compiled vector space.
                        }
                    }
                }
                snap.triplet_store.load_supersession_sidecar(&candidate.with_extension("sup.json"));
                for ev in snap.ledger_session_events {
                    if ev.domain == "session" {
                        match ev.kind.as_str() {
                            "register" => {
                                let kind = serde_json::from_str::<serde_json::Value>(&ev.payload_json)
                                    .ok().and_then(|v| v.get("kind").and_then(|k| k.as_str()).map(str::to_owned))
                                    .unwrap_or_default();
                                session_registry.register(ev.target.clone(), kind, ev.realm.clone(), ev.ts_ms);
                            }
                            "heartbeat" => session_registry.heartbeat(&ev.target, ev.ts_ms),
                            "deregister" => session_registry.deregister(&ev.target),
                            _ => {}
                        }
                    }
                    msg_registry.insert(ev);
                }
                payloads = snap.payloads;
                // Stage B: load retrieval surfaces alongside content. Absent on
                // pre-Stage-B families → empty map → embed falls back to content.
                retrieval_surfaces = FullSnapshot::load_retrieval_surface_sidecar(&candidate.with_extension("rsf"));
                recall_provenance = snap.recall_provenance;
                states = snap.states;
                assoc_edges = snap.assoc_edges;
                artifacts = snap.artifacts;
                artifact_paths = snap.artifact_paths;
                time_idx = snap.time_idx;
                keyword_idx = snap.keyword_idx;
                artifact_idx = snap.artifact_idx;
                triplet_store = snap.triplet_store;
                symbol_idx = snap.symbol_idx;
                call_graph = snap.call_graph;
                code_files = snap.code_files;
                semantic_idx = snap.semantic_idx;
                snapshot_coactivation_stats = snap.coactivation_stats;
                snap_ack_scores = snap.ack_scores;
                snap_correction_states = snap.correction_states;
                snap_event_tape         = snap.event_tape;
                snap_decision_tape      = snap.decision_tape;
                snap_interaction_ledger = snap.interaction_ledger;
                snap_predicate_store    = snap.predicate_store;
                eprintln!(
                    "[chitta-field] loaded full snapshot seqno={} ({} memories) from {:?}",
                    full_snapshot_seqno, payloads.len(), candidate
                );
                loaded_snapshot_name =
                    candidate.file_name().map(|n| n.to_string_lossy().into_owned());
                full_snapshot_loaded = true;
                break;
            }
            Err(e) => eprintln!(
                "[chitta-field] failed to load full snapshot {:?}: {}",
                candidate, e
            ),
        }
    }
    if had_full_snapshots && !full_snapshot_loaded {
        return Err(FieldError::Manifest(
            "all full snapshots failed to load — refusing to start with empty store \
             (this prevents data loss; fix the snapshot format or restore from backup)"
                .to_string(),
        ));
    }
    // Only clean up stale snapshots if we successfully loaded one.
    // Keep the 1 most recent stale snapshot as a safety net against format bugs.
    if full_snapshot_loaded {
        let mut stale_by_seqno: Vec<(u64, &std::path::PathBuf)> = stale_full_paths
            .iter()
            .filter(|p| best_full_path.as_ref() != Some(p))
            .filter_map(|p| FullSnapshot::peek_seqno(p).ok().map(|s| (s, p)))
            .collect();
        stale_by_seqno.sort_by(|a, b| b.0.cmp(&a.0));
        // Skip the most recent stale (keep as backup), delete the rest + their sidecars.
        for (_, path) in stale_by_seqno.iter().skip(1) {
            let _ = std::fs::remove_file(path);
            for ext in &["emb", "bin", "mu", "shdr", "hnsw", "pld", "snapshot.tmp", "lsh", "organs", "turbo", "turbo.meta"] {
                let _ = std::fs::remove_file(path.with_extension(ext));
            }
            // delta.hnsw: with_extension replaces only last component, handle separately
            let delta = path.with_extension("delta.hnsw");
            let _ = std::fs::remove_file(&delta);
        }

        // Prune orphaned sidecars: files whose snapshot hash is not current best or backup.
        let mut live_hashes = std::collections::HashSet::new();
        let extract_hash = |p: &std::path::Path| -> Option<String> {
            p.file_name()?.to_str()?.splitn(3, '.').nth(1).map(String::from)
        };
        if let Some(ref p) = best_full_path {
            if let Some(h) = extract_hash(p) { live_hashes.insert(h); }
        }
        if let Some((_, p)) = stale_by_seqno.first() {
            if let Some(h) = extract_hash(p) { live_hashes.insert(h); }
        }
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if !name.starts_with("chitta.") { continue; }
                let parts: Vec<&str> = name.splitn(3, '.').collect();
                if parts.len() < 3 { continue; }
                let hash = parts[1];
                if live_hashes.contains(hash) { continue; }
                let ext = parts[2];
                if matches!(ext, "emb" | "bin" | "hnsw" | "pld" | "delta.hnsw" | "snapshot.tmp" | "lsh" | "organs" | "turbo" | "turbo.meta") {
                    let removed = std::fs::remove_file(entry.path()).is_ok();
                    if removed {
                        eprintln!("[chitta-field] pruned orphaned sidecar: {}", name);
                    }
                }
            }
        }
    }

    Ok(LoadedSnapshot {
        payloads,
        retrieval_surfaces,
        recall_provenance,
        states,
        assoc_edges,
        artifacts,
        artifact_paths,
        semantic_idx,
        time_idx,
        artifact_idx,
        keyword_idx,
        triplet_store,
        symbol_idx,
        call_graph,
        code_files,
        cortical_idx,
        session_registry,
        transcript_registry,
        task_registry,
        user_model_registry,
        theme_organ,
        analytics_registry,
        msg_registry,
        skill_registry,
        agent_registry,
        constraint_store,
        trigger_store,
        predictor,
        surprise_store,
        epistemic_debt_store,
        integration_kernel,
        surprise_learning,
        wisdom_promotion,
        learned_scorer,
        intervention_store,
        agent_protocol_store,
        wisdom_lineage_store,
        symbol_event_log,
        chunk_hash_idx,
        snapshot_coactivation_stats,
        snap_ack_scores,
        snap_correction_states,
        snap_event_tape,
        snap_decision_tape,
        snap_interaction_ledger,
        snap_predicate_store,
        snapshot_seqno,
        full_snapshot_seqno,
        best_full_path,
        loaded_manifest,
        loaded_snapshot_name,
        loaded_header,
        migrate_reembed,
        reindex_mode,
    })
}

pub(super) fn replay_wal(
    log: &mut OpLog,
    mut ctx: ApplyCtx<'_>,
    recall_provenance: &mut HashMap<MemoryId, std::collections::BTreeSet<InstanceId>>,
    snapshot_seqno: u64,
    full_snapshot_seqno: u64,
    loaded_manifest: &Option<crate::manifest::Manifest>,
    loaded_snapshot_name: &Option<String>,
) -> Result<std::collections::BTreeMap<u32, u64>> {
    // WAL coverage of the loaded snapshot (THEORY.md §4): per-writer max
    // seqno the snapshot provably contains, from its manifest family.
    // Under multi-writer overlap, the scalar `seqno <= full_snapshot_seqno`
    // filter silently skips foreign ops the snapshot never saw — the
    // per-writer vector is the correct test. Legacy snapshots without a
    // family fall back to the scalar (today's approximate behavior).
    let full_covered: std::collections::BTreeMap<u32, u64> = loaded_manifest
        .as_ref()
        .zip(loaded_snapshot_name.as_ref())
        .and_then(|(m, name)| {
            m.families
                .values()
                .chain(m.checkpoints.iter())
                .find(|cp| &cp.snapshot.name == name)
                .map(|cp| {
                    cp.covered
                        .iter()
                        .filter_map(|(k, v)| u32::from_str_radix(k, 16).ok().map(|i| (i, *v)))
                        .collect()
                })
        })
        .unwrap_or_default();
    let covered_by_full = |inst: u32, seqno: u64| -> bool {
        match full_covered.get(&inst) {
            Some(&max) => seqno <= max,
            None => full_covered.is_empty() && seqno <= full_snapshot_seqno,
        }
    };
    let mut max_replayed_seqno = full_snapshot_seqno;
    // Orphan-delta buffer (THEORY.md §2.3): clock skew across writers can
    // merge-order an UpdateState before its memory's PutPayload; buffer
    // and retry once all creates are applied. Merge replay already yields
    // timestamp order, so collection order is application order.
    let mut orphan_deltas: Vec<crate::ops::StateDeltaOp> = Vec::new();
    let mut orphan_accesses = Vec::new();
    let wal_replay_phase = crate::profile::LoadPhase::new("wal_replay");
    let replayed_coverage = log.replay(0, |inst, seqno, op| {
        if seqno > max_replayed_seqno { max_replayed_seqno = seqno; }
        if covered_by_full(inst, seqno) {
            // This op is covered by the full snapshot.
            // Still apply cortical ops not covered by the cortical snapshot.
            match &op {
                Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_)
                    if seqno <= snapshot_seqno =>
                {
                    return Ok(())
                }
                Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_) => {
                    // fall through: apply to cortical index only
                }
                _ => return Ok(()), // covered by full snapshot
            }
        } else if seqno <= snapshot_seqno {
            // Covered by cortical snapshot but not by full snapshot (shouldn't normally happen,
            // but handle gracefully by skipping cortical ops).
            if matches!(
                op,
                Op::UpdateSparseCode(_) | Op::TrainPQ(_) | Op::UpdateResidualPQ(_)
            ) {
                return Ok(());
            }
        }
        if let Op::RecordRecallBatch(b) = &op {
            for mid in &b.memory_ids {
                let set = recall_provenance.entry(*mid).or_default();
                if set.len() < 8 {
                    set.insert(inst);
                }
            }
        }
        if let Op::UpdateStateBatch(deltas) = &op {
            for d in deltas {
                if let Some(state) = ctx.states.get_mut(&d.memory_id) {
                    apply_access_delta(state, d);
                } else {
                    orphan_accesses.push(d.clone());
                }
            }
            return Ok(());
        }
        if let Op::UpdateState(d) = &op {
            if !ctx.states.contains_key(&d.memory_id) {
                orphan_deltas.push(d.clone());
                return Ok(());
            }
        }
        apply_op(op, ctx.reborrow());
        Ok(())
    })?;
    for d in orphan_deltas {
        if let Some(state) = ctx.states.get_mut(&d.memory_id) {
            let replay_now = if d.op_ts_ms > 0 { d.op_ts_ms } else { state.created_at_ms };
            state.apply_delta(&d, replay_now);
        }
    }
    for d in orphan_accesses {
        if let Some(state) = ctx.states.get_mut(&d.memory_id) { apply_access_delta(state, &d); }
    }
    // State coverage = snapshot coverage ⊔ walked WAL maxima (per-writer
    // max). Carried on the field and written into the next manifest commit.
    let mut wal_coverage = full_covered;
    for (inst, max) in replayed_coverage {
        let e = wal_coverage.entry(inst).or_insert(0);
        if max > *e { *e = max; }
    }
    log.set_next_seqno(max_replayed_seqno + 1);
    drop(wal_replay_phase);
    Ok(wal_coverage)
}

pub(super) struct KeyedIndexes {
    pub(super) content_prov_idx: HashMap<u64, MemoryId>,
    pub(super) prov_key_idx: HashMap<String, MemoryId>,
    pub(super) correction_key_idx: HashMap<String, Vec<MemoryId>>,
    pub(super) task_key_idx: HashMap<String, MemoryId>,
}

pub(super) fn load_embedding_sidecars(
    semantic_idx: &mut SemanticIndex,
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &mut HashMap<MemoryId, MemoryState>,
    best_full_path: &Option<PathBuf>,
    migrate_reembed: bool,
    reindex_mode: bool,
) {
    // Load embedding + binary-code sidecars, then HNSW — all before WAL replay.
    if let Some(snap_path) = best_full_path.as_ref().filter(|_| !migrate_reembed || reindex_mode) {
        // [migrate] when CHITTA_MIGRATE_REEMBED is set (and not reindex_mode), this whole block
        // is skipped: the foreign-vector .emb/.bin/.mu/.hnsw sidecars are never loaded, so every
        // memory has no embedding and re_embed refills the compiled vector space from content (.pld).
        // [reindex] CHITTA_REINDEX_MODE accepts the snapshot despite model-ID mismatch AND loads
        // existing embeddings so force_reindex() can rebuild ANN indices from them.
        // .emb: flat binary embeddings (v10+). For v9 snapshots the sidecar won't exist
        // yet, so this is a no-op and embeddings remain populated from bincode.
        let emb_phase = crate::profile::LoadPhase::new("emb");
        let emb_loaded = semantic_idx.load_embeddings_sidecar(&snap_path.with_extension("emb"));
        drop(emb_phase);
        if !emb_loaded && semantic_idx.embeddings_count() == 0 {
            eprintln!("[chitta-field] WARNING: v10 snapshot but .emb sidecar missing — embeddings will be empty until backfill");
        }
        // Payload embeddings stay OUT of the heap — the semantic index is
        // their single in-RAM home (the rehydration this block used to do
        // duplicated ~600MB of RSS; readers go through embedding_of).
        // Payloads whose embedding exists in neither place self-heal via
        // embed_pending backfill from content.
        {
            let mut requeued = 0usize;
            for (id, p) in payloads.iter() {
                if !p.embedding.is_empty() || p.embedding_dim == 0 {
                    continue;
                }
                if semantic_idx.get_embedding(*id).is_some() {
                    continue;
                }
                if let Some(st) = states.get_mut(id) {
                    if !st.deleted && !p.content.is_empty() {
                        st.embed_pending = true;
                        requeued += 1;
                    }
                }
            }
            if requeued > 0 {
                eprintln!(
                    "[chitta-field] payload embeddings: {} requeued for re-embed (index holds the rest)",
                    requeued
                );
            }
        }
        // .bin: binary codes sidecar — skip O(N×256) reconstruction in normalize_all.
        let _ = semantic_idx.load_binary_sidecar(&snap_path.with_extension("bin"));
        // .mu: corpus-mean centroid (anisotropy correction). Must load before
        // normalize_all() so any rebuilt binary codes are centered, and before any
        // search so queries are centered. Absent on legacy snapshots → raw cosine.
        let _ = semantic_idx.load_centroid_sidecar(&snap_path.with_extension("mu"));
        // .emb mmap: only for large stores (> EMB_MMAP_MIN). Above that, recall uses the
        // binary-Hamming prefilter which reads few embeddings, so mmap saves a large heap
        // copy. Below it, recall is a flat heap scan over every embedding — serving that from
        // mmap both wastes nothing and courts a SIGBUS: a consolidation prune can unlink the
        // .emb file out from under the mapping while a scan is faulting its pages. Keeping the
        // embeddings in the heap removes that race entirely.
        if emb_loaded && semantic_idx.embeddings_count() > crate::hnsw::EMB_MMAP_MIN {
            let _ = semantic_idx.activate_mmap_embeddings(&snap_path.with_extension("emb"));
        }
        // .hnsw + .delta.hnsw: load both tiers; backfill handles WAL-replay additions.
        // Skipped when CHITTA_NO_HNSW_SIDECAR=1: below flat_scan_max the flat scan
        // serves every query, so loading the stale graph is dead RAM. rebuild_hnsw()
        // self-heals lazily if the corpus ever crosses the cap.
        if !crate::hnsw::skip_hnsw_sidecar() {
            let _ = semantic_idx.load_hnsw(&snap_path.with_extension("hnsw"));
            let _ = semantic_idx.load_delta_hnsw(&snap_path.with_extension("delta.hnsw"));
        }
    }
}

pub(super) fn reconcile_embeddings(
    semantic_idx: &mut SemanticIndex,
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &mut HashMap<MemoryId, MemoryState>,
    best_full_path: &Option<PathBuf>,
    data_dir: &std::path::Path,
) {
    semantic_idx.set_inhibit_hnsw(false);
    let purged_ids = semantic_idx.purge_wrong_dim();
    for id in &purged_ids {
        if let Some(state) = states.get_mut(id) {
            if !state.deleted {
                state.embed_pending = true;
            }
        }
    }
    // Embed reconciliation — runs EVERY start, deliberately unguarded.
    // purge_wrong_dim() above re-queues embeddings of the WRONG dim. Nothing re-queued
    // MISSING ones: an embedding lost to a decode failure (vak_llama embed_one returns a
    // zero vector) or to force_clear_embed_pending was gone for good, because embed_pending
    // was already false and no startup check ever looked for absent vectors. The memory
    // stayed keyword-only forever while pending_count reported 0 — the queue drains on
    // failure as well as success, so it cannot be used as a coverage metric. Derive
    // coverage from the index instead of trusting the flag.
    {
        const MIN_EMBED_CHARS: usize = 20;
        let mut requeued = 0usize;
        for (id, payload) in payloads {
            if payload.content.len() < MIN_EMBED_CHARS { continue; }
            if semantic_idx.has_embedding(*id) { continue; }
            if let Some(state) = states.get_mut(id) {
                if !state.embed_pending && !state.deleted {
                    state.embed_pending = true;
                    requeued += 1;
                }
            }
        }
        if requeued > 0 {
            eprintln!("[chitta-field] embed reconciliation: {requeued} content-bearing \
                       memories had NO embedding — re-queued for backfill");
        }
    }
    {
        let _phase = crate::profile::LoadPhase::new("normalize");
        semantic_idx.normalize_with_cache(best_full_path.as_ref().map(|p| p.with_extension("lsh")).as_deref());
    }
    // After WAL replay, the HNSW may have been backfilled with entries beyond the snapshot.
    // Persist the updated HNSW sidecar now so the next restart loads a complete graph
    // instead of spending O(N log N) re-inserting the delta.
    if let Some(ref snap_path) = best_full_path {
        let hnsw_count = semantic_idx.hnsw_len();
        let total      = semantic_idx.total_embedding_count();
        if hnsw_count > 0 && hnsw_count == total {
            // Promote delta→base (O(1) swap) so save_hnsw serialises the full
            // graph, not a 9-byte empty-base stub — mirrors save_full_snapshot.
            // Without this the next restart re-backfills the whole delta tier.
            semantic_idx.promote_delta_to_base_if_empty();
            if let Err(e) = semantic_idx.save_hnsw(&snap_path.with_extension("hnsw")) {
                eprintln!("[chitta-field] WARNING: failed to save post-replay HNSW: {e}");
            } else {
                eprintln!("[chitta-field] post-replay HNSW saved ({hnsw_count} nodes)");
            }
            let _ = semantic_idx.save_delta_hnsw(&snap_path.with_extension("delta.hnsw"));
        }
    }

    // One-time migration: mark SSL memories (content with →) for gloss-baked re-embed.
    // Guard file prevents re-running after backfill completes.
    {
        let flag = data_dir.join("ssl_gloss_v1.migrated");
        if !flag.exists() {
            let arrow: &[u8] = b"\xe2\x86\x92"; // UTF-8 →
            let mut ssl_count = 0usize;
            for (id, payload) in payloads {
                if payload.content.windows(3).any(|w| w == arrow) {
                    if let Some(state) = states.get_mut(id) {
                        if !state.embed_pending && !state.deleted {
                            state.embed_pending = true;
                            ssl_count += 1;
                        }
                    }
                }
            }
            if ssl_count > 0 {
                eprintln!("[chitta-field] SSL gloss migration: marked {ssl_count} memories for re-embed with gloss");
            }
            let _ = std::fs::write(&flag, "done");
        }
    }

    // One-time migration: 768→1536 embedding dimension (ssl_distiller_dpo).
    // The old 768-d .emb/.bin sidecars are skipped via the bumped EMB_MAGIC/BIN_MAGIC,
    // so on the first start with EMBED_DIM=1536 every memory has no embedding and
    // purge_wrong_dim() finds nothing to mark. Explicitly mark every content-bearing
    // memory embed_pending so the background backfill thread re-embeds it at 1536-d.
    // Guard file makes this run exactly once (after which a 1536-d snapshot is saved).
    {
        let flag = data_dir.join("embed_1536_v1.migrated");
        if !flag.exists() {
            const MIN_EMBED_CHARS: usize = 20;
            let mut n = 0usize;
            for (id, payload) in payloads {
                if payload.content.len() >= MIN_EMBED_CHARS {
                    if let Some(state) = states.get_mut(id) {
                        if !state.embed_pending && !state.deleted {
                            state.embed_pending = true;
                            n += 1;
                        }
                    }
                }
            }
            eprintln!("[chitta-field] 768→1536 migration: marked {n} memories for re-embed at 1536-d");
            let _ = std::fs::write(&flag, "done");
        }
    }

}

pub(super) fn repair_temporal_entries(
    time_idx: &mut TemporalIndex,
    payloads: &HashMap<MemoryId, MemoryPayload>,
) {
    // Fix temporal entries that have ts_ms=0 (stored before authored_at_ms default fix)
    {
        let zero_entries = time_idx.entries_with_ts(0);
        let fixed_count = zero_entries.len();
        for entry in zero_entries {
            if let Some(payload) = payloads.get(&entry.memory_id) {
                let correct_ts = if payload.authored_at_ms != 0 {
                    payload.authored_at_ms
                } else {
                    payload.created_at_ms
                };
                if correct_ts != 0 {
                    time_idx.remove(entry.memory_id, 0);
                    time_idx.upsert(TemporalEntry {
                        memory_id: entry.memory_id,
                        ts_ms: correct_ts,
                        kind: entry.kind.clone(),
                        realm: entry.realm.clone(),
                        strength: entry.strength,
                    });
                }
            }
        }
        if fixed_count > 0 {
            eprintln!(
                "[chitta-field] fixed {} temporal entries with ts_ms=0",
                fixed_count
            );
        }
    }

}

pub(super) fn warm_startup_indexes(
    keyword_idx: &mut KeywordIndex,
    semantic_idx: &mut SemanticIndex,
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
    best_full_path: &Option<PathBuf>,
    data_dir: &std::path::Path,
) -> (Option<LiteEncoder>, crate::hdc::HdcStore) {
    // These inputs are immutable after WAL replay + normalization. No store
    // guards exist yet: quantization, HDC and lite I/O can run independently.
    let (loaded_lite_encoder, hdc_store) = std::thread::scope(|scope| {
        // Reverse postings are independent of the semantic/HDC/lite inputs.
        // Finish this before publishing the field, so mutation/removal APIs
        // observe the same complete keyword index as the serial loader.
        let keyword = scope.spawn(|| {
            let _phase = crate::profile::LoadPhase::new("keyword_reverse");
            keyword_idx.rebuild_reverse_index();
        });
        let lite = scope.spawn(|| {
            let _phase = crate::profile::LoadPhase::new("lite_encoder");
            ChittaField::load_lite_encoder(data_dir)
        });
        let turbo = scope.spawn(|| {
            semantic_idx.warm_turbo_with_cache(best_full_path.as_deref());
        });
    // Build HDC index — load from sidecar if available (fast path), else rebuild.
    let hdc_phase = crate::profile::LoadPhase::new("hdc");
    let mut hdc_store = crate::hdc::HdcStore::new();
    {
        let hdc_sidecar = best_full_path.as_ref().map(|p| p.with_extension("hdc"));
        let loaded = hdc_sidecar.as_ref()
            .and_then(|p| hdc_store.load_sidecar(p).ok())
            .unwrap_or(0);
        if loaded > 0 {
            eprintln!("[chitta-field] hdc sidecar: loaded {} memories (skipped rebuild)", loaded);
        } else {
            eprintln!("[chitta-field] hdc sidecar: not found or stale — rebuilding from payloads");
            let entries = payloads.iter()
                .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
                .map(|(id, p)| (*id, std::str::from_utf8(&p.content).unwrap_or(""), p.realm.as_str()));
            hdc_store.rebuild(entries);
        }
    }

    drop(hdc_phase);
    keyword.join().expect("startup keyword worker panicked");
    turbo.join().expect("startup Turbo worker panicked");
    let loaded_lite_encoder = lite.join().expect("startup lite encoder worker panicked");
    (loaded_lite_encoder, hdc_store)

    });
    semantic_idx.prune_turbo_changes();
    (loaded_lite_encoder, hdc_store)
}

pub(super) fn rebuild_event_organs(
    snap_event_tape: crate::organ::event_tape::EventTape,
    triplet_store: &TripletStore,
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
    best_full_path: &Option<PathBuf>,
) -> (crate::organ::event_tape::EventTape, crate::organ::cdawg::CdawgOrgan, crate::hdc::EpisodeHdcStore) {
    // Build EventTape from snapshot, seed entity interner from triplets, synthesize
    // legacy events for existing memories, then rebuild CDAWG from the tape.
    // Use persisted EventTape from snapshot if available; otherwise synthesize from memories.
    let event_tape = if !snap_event_tape.events.is_empty() {
        snap_event_tape
    } else {
        let mut tape = crate::organ::event_tape::EventTape::new();
        let subjects: Vec<String> = triplet_store.all_subjects();
        tape.seed_from_triplets(subjects.iter().map(|s| s.as_str()));
        let mut sorted_payloads: Vec<_> = payloads.iter()
            .filter(|(id, _)| states.get(id).map(|s| !s.deleted).unwrap_or(false))
            .collect();
        sorted_payloads.sort_by_key(|(_, p)| p.authored_at_ms);
        // Cap at 5000 most-recent memories to bound CDAWG rebuild cost on migration.
        const LEGACY_CAP: usize = 5_000;
        let skip = sorted_payloads.len().saturating_sub(LEGACY_CAP);
        for (_, p) in sorted_payloads.into_iter().skip(skip) {
            tape.synthesize_legacy(&p.realm, p.authored_at_ms);
        }
        tape
    };
    let tape_phase = crate::profile::LoadPhase::new("event_tape_organs");
    let organs_path = best_full_path.as_ref().map(|p| p.with_extension("organs"));
    let (cdawg, episode_hdc) = crate::startup_cache::load_or_rebuild_organs(
        &event_tape, organs_path.as_deref());
    drop(tape_phase);
    (event_tape, cdawg, episode_hdc)
}

pub(super) fn rebuild_keyed_indexes(
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
) -> KeyedIndexes {
    // Rebuild content_prov_idx from live signal/[done] payloads so the dedup gate
    // survives daemon restarts without requiring snapshot format changes.
    let content_prov_idx: HashMap<u64, MemoryId> = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        payloads.iter()
            .filter(|(id, p)| {
                p.kind == "signal"
                    && p.content.starts_with(b"[done]")
                    && !states.get(id).map(|s| s.deleted).unwrap_or(false)
            })
            .map(|(id, p)| {
                let mut h = DefaultHasher::new();
                p.content.hash(&mut h);
                (h.finish(), *id)
            })
            .collect()
    };
    eprintln!("[chitta-field] content_prov_idx rebuilt: {} [done] provenance records", content_prov_idx.len());
    // Rebuild the keyed provenance lane from live [done] payloads. Sorted by
    // created_at ascending + or_insert => earliest record wins per key,
    // matching the live put_memory path deterministically (HashMap iteration
    // order is otherwise nondeterministic, and multiple records can share a
    // sha/path key).
    let prov_key_idx: HashMap<String, MemoryId> = {
        let mut live: Vec<_> = payloads
            .iter()
            .filter(|(id, p)| {
                p.kind == "signal"
                    && p.content.starts_with(b"[done]")
                    && !states.get(id).map(|s| s.deleted).unwrap_or(false)
            })
            .collect();
        // (created_at, memory_id) so equal-timestamp records break ties by
        // allocation order — matching the live put_memory call order exactly.
        live.sort_by_key(|(id, p)| (p.created_at_ms, **id));
        let mut idx: HashMap<String, MemoryId> = HashMap::new();
        for (id, p) in live {
            for pk in crate::store::parse_prov_keys(&p.content) {
                idx.entry(pk).or_insert(*id);
            }
        }
        idx
    };
    eprintln!("[chitta-field] prov_key_idx rebuilt: {} keyed provenance entries", prov_key_idx.len());
    // Rebuild the correction keyed lane from live [correction] payloads.
    // Multi-valued: each trigger key -> ALL live correction ids carrying it,
    // in ASCENDING (created_at, id) order so newest is last. Pre-filter is
    // case-insensitive `[correction` to admit the free-form header form
    // `[CORRECTION to memory #… — topic]` (most real corrections) — mirroring
    // the internal gate in parse_correction_keys; the old lowercase-exact
    // `[correction]` filter silently dropped ~80% of corrections here.
    let correction_key_idx: HashMap<String, Vec<MemoryId>> = {
        let mut live: Vec<_> = payloads
            .iter()
            .filter(|(id, p)| {
                let lead = p.content.get(..11).unwrap_or_default().to_ascii_lowercase();
                lead.starts_with(b"[correction")
                    && !states.get(id).map(|s| s.deleted).unwrap_or(false)
            })
            .collect();
        live.sort_by_key(|(id, p)| (p.created_at_ms, **id));
        let mut idx: HashMap<String, Vec<MemoryId>> = HashMap::new();
        for (id, p) in live {
            for ck in crate::store::parse_correction_keys(&p.content) {
                let v = idx.entry(ck).or_default();
                if !v.contains(id) {
                    v.push(*id);
                }
            }
        }
        idx
    };
    eprintln!("[chitta-field] correction_key_idx rebuilt: {} keyed correction triggers", correction_key_idx.len());
    // Rebuild the task-state keyed lane from live [task] payloads. Sorted
    // ASCENDING by (created_at, id) + `insert` (overwrite) => the NEWEST
    // record wins per task slug, matching the live put_memory insert order
    // exactly (LATEST-WINS / SUPERSEDE) — same discipline as the correction
    // rebuild (status evolves, so newest state wins, unlike provenance).
    let task_key_idx: HashMap<String, MemoryId> = {
        let mut live: Vec<_> = payloads
            .iter()
            .filter(|(id, p)| {
                p.content.starts_with(b"[task]")
                    && !states.get(id).map(|s| s.deleted).unwrap_or(false)
            })
            .collect();
        live.sort_by_key(|(id, p)| (p.created_at_ms, **id));
        let mut idx: HashMap<String, MemoryId> = HashMap::new();
        for (id, p) in live {
            for tk in crate::store::parse_task_keys(&p.content) {
                idx.insert(tk, *id);
            }
        }
        idx
    };
    eprintln!("[chitta-field] task_key_idx rebuilt: {} keyed task-state entries", task_key_idx.len());
    KeyedIndexes { content_prov_idx, prov_key_idx, correction_key_idx, task_key_idx }
}
