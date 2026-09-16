use crate::error::{FieldError, Result};
use crate::hnsw::SemanticIndex;
use crate::ids::{
    new_instance_id, ArtifactId, ArtifactIdAllocator, InstanceId, MemoryId, MemoryIdAllocator,
    TripletIdAllocator,
};
use crate::learner::LearnerSet;
use crate::log::OpLog;
use crate::ops::{EdgeType, Op};
use crate::organ::analytics::AnalyticsRegistry;
use crate::organ::artifact::ArtifactIndex;
use crate::organ::callgraph::CallGraph;
use crate::organ::codefile::CodeFileIndex;
use crate::organ::cortex::{CorticalIndex, SparseCode, SparseEncoder};
use crate::organ::hopfield::HopfieldNetwork;
use crate::organ::keyword::KeywordIndex;
use crate::organ::lite_encoder::LiteEncoder;
use crate::organ::agent::AgentRegistry;
use crate::organ::constraint::ConstraintStore;
use crate::organ::msg::MsgRegistry;
use crate::organ::predictor::AccessPredictor;
use crate::organ::skill::SkillRegistry;
use crate::organ::trigger::TriggerStore;
use crate::organ::surprise::SurpriseStore;
use crate::organ::agent_protocol::AgentProtocolStore;
use crate::organ::wisdom_lineage::WisdomLineageStore;
use crate::organ::epistemic_debt::EpistemicDebtStore;
use crate::organ::integration::IntegrationKernel;
use crate::organ::surprise_learning::SurpriseLearningStore;
use crate::organ::wisdom_promotion::WisdomPromotionStore;
use crate::organ::intervention::InterventionStore;
use crate::organ::symbol_events::SymbolEventLog;
use crate::scoring::learned::LearnedScoringModel;
use crate::organ::pq::ProductQuantizer;
use crate::organ::session::SessionRegistry;
use crate::organ::symbol::{SymbolEntry, SymbolIndex};
use crate::organ::task::TaskRegistry;
use crate::organ::temporal::{TemporalEntry, TemporalIndex};
use crate::organ::theme_organ::ThemeOrgan;
use crate::organ::transcript::TranscriptRegistry;
use crate::organ::triplet::TripletStore;
use crate::organ::user_model::UserModelRegistry;
use crate::payload::MemoryPayload;
use crate::snapshot::FullSnapshot;
use crate::state::MemoryState;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::sync::Arc;

/// A single directed association edge stored in memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssocEdge {
    pub dst: MemoryId,
    pub edge_type: EdgeType,
    pub weight: f32,
}

/// Tracks how often two memories were co-retrieved and in how many
/// distinct query contexts. Used to weight Hebbian edge strengthening.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct CoActivationStats {
    /// Total number of co-retrieval events.
    pub sim_count: u32,
    /// Distinct context_hash values seen (capped at 64).
    pub recent_context_hashes: Vec<u64>,
    /// Number of distinct contexts (len of deduplicated recent_context_hashes).
    pub diversity_count: u16,
    pub last_seen_ms: i64,
}

impl CoActivationStats {
    const MAX_HASHES: usize = 64;

    pub fn record(&mut self, context_hash: u64, ts_ms: i64) {
        self.sim_count += 1;
        self.last_seen_ms = ts_ms;
        if !self.recent_context_hashes.contains(&context_hash) {
            if self.recent_context_hashes.len() >= Self::MAX_HASHES {
                self.recent_context_hashes.remove(0);
            }
            self.recent_context_hashes.push(context_hash);
        }
        self.diversity_count = self.recent_context_hashes.len() as u16;
    }

    /// Hebbian multiplier: sim_count * diversity_count, capped at 16.0
    pub fn hebbian_multiplier(&self) -> f32 {
        let raw = self.sim_count as f32 * self.diversity_count as f32;
        raw.min(16.0)
    }
}

/// Cap coactivation_stats to `max_per_memory` strongest pairs per memory.
/// Returns the number of entries removed.
pub(crate) fn prune_coactivation_stats(
    stats: &mut HashMap<(MemoryId, MemoryId), CoActivationStats>,
    max_per_memory: usize,
) -> usize {
    use std::collections::HashSet;
    let mut per_memory: HashMap<MemoryId, Vec<((MemoryId, MemoryId), u32)>> =
        HashMap::with_capacity(stats.len().min(65536));
    for (key, stat) in stats.iter() {
        per_memory.entry(key.0).or_default().push((*key, stat.sim_count));
        per_memory.entry(key.1).or_default().push((*key, stat.sim_count));
    }
    let mut to_remove: HashSet<(MemoryId, MemoryId)> = HashSet::new();
    for (_mem, mut pairs) in per_memory {
        if pairs.len() > max_per_memory {
            pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (key, _) in pairs.into_iter().skip(max_per_memory) {
                to_remove.insert(key);
            }
        }
    }
    let removed = to_remove.len();
    for key in &to_remove { stats.remove(key); }
    removed
}

#[derive(Default)]
pub(crate) struct PendingRecallEffects {
    pub strengthen: HashSet<MemoryId>,
    pub co_retrieval_pairs: HashMap<(MemoryId, MemoryId), f32>,
    pub proto_windows: Vec<Vec<MemoryId>>,
}

// ── Lock-acquisition order ────────────────────────────────────────────────────
// parking_lot RwLocks are writer-preferring and NOT reentrant: a queued writer
// blocks all later readers, so acquisition order across fields is what stands
// between us and both deadlocks and convoys (one already required SIGKILL —
// see commit a20070a). When holding more than one lock, acquire in ascending
// tier order and release before acquiring anything from a lower tier:
//
//   Tier 0  log, id_alloc                      (append/allocate primitives)
//   Tier 1  payloads, states, assoc_edges      (core memory data)
//   Tier 2  semantic_idx, time_idx, keyword_idx, artifact_idx, hdc_idx,
//           cortical maps, realm_members       (derived indexes)
//   Tier 3  triplet_store, symbol_idx, call_graph, code_files (knowledge graph)
//   Tier 4  scoring_pipeline, learners, ack_scores, coactivation_stats,
//           cw_refresh_inflight                (scoring / stats)
//   Tier 5  organs: event_tape, decision_tape, turiya_monitor, observer_state,
//           interaction_ledger, predicate_store, ...  (append-mostly organs)
//
// Prefer the two-phase pattern (collect under reads → drop → apply under one
// brief write; see recall_semantic_ctx's CW refresh) over holding write locks
// across computation. Long-lived multi-lock holds on the recall path are bugs.
// Enforcement: build with `--features deadlock-detection` (CI does) to get a
// checker thread that dumps lock cycles; there is no static ordering check, so
// keep new multi-lock sites tier-ordered.
/// Peer ops read off disk by `sync_foreign_collect`, not yet applied to state by
/// `sync_foreign_apply`. Buffering between the two phases is what lets the disk reads run
/// outside the daemon's global rpc_mutex_ without losing ops: `seen_offsets` advances at
/// collect time, so an unapplied batch would otherwise never be re-read.
#[derive(Default)]
pub(crate) struct PendingForeign {
    pub(crate) ops: Vec<(crate::ids::InstanceId, Op)>,
    pub(crate) coverage: std::collections::BTreeMap<crate::ids::InstanceId, u64>,
}

/// Take an exclusive, non-blocking advisory lock on `<data_dir>/.instance.lock`.
/// Fails fast when another live instance holds it. flock() works on NFS via the
/// lock daemon and is released automatically when the holder exits.
fn acquire_instance_lock(data_dir: &std::path::Path) -> Result<Option<std::fs::File>> {
    if std::env::var("CHITTA_STORE_LOCK").map(|v| v == "0").unwrap_or(false) {
        return Ok(None);
    }
    use std::io::{Read, Seek, Write};
    use std::os::unix::io::AsRawFd;
    let path = data_dir.join(".instance.lock");
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    // Two attempts: the second runs only after a provably stale lock file
    // (same host, holder pid gone) has been replaced. On NFSv3 a daemon killed
    // mid-snapshot (2026-09-15, TimeoutStopSec=30) left a server-side lock on
    // the old inode that no process owned; 49 restarts failed until the file
    // was renamed by hand. A holder on another host is never overridden.
    for attempt in 0..2 {
        let mut file = std::fs::OpenOptions::new().read(true).write(true).create(true).open(&path)?;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc == 0 {
            let _ = file.set_len(0);
            let _ = file.seek(std::io::SeekFrom::Start(0));
            let _ = writeln!(file, "{} {}", std::process::id(), hostname);
            let _ = file.sync_all();
            return Ok(Some(file));
        }
        let err = std::io::Error::last_os_error();
        let would_block = err.raw_os_error() == Some(libc::EWOULDBLOCK)
            || err.raw_os_error() == Some(libc::EAGAIN);
        if !would_block {
            return Err(FieldError::Io(err));
        }
        let mut holder = String::new();
        let _ = file.read_to_string(&mut holder);
        let mut parts = holder.split_whitespace();
        let holder_pid: Option<u32> = parts.next().and_then(|p| p.parse().ok());
        let holder_host = parts.next().unwrap_or("");
        let stale = attempt == 0
            && !hostname.is_empty()
            && holder_host == hostname
            && holder_pid.map(|pid| unsafe { libc::kill(pid as i32, 0) } != 0
                && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH))
                .unwrap_or(false);
        if stale {
            eprintln!(
                "[chitta-field] stale instance lock {} (holder pid {} on this host is gone); replacing it",
                path.display(), holder_pid.unwrap_or(0)
            );
            let _ = std::fs::remove_file(&path);
            continue;
        }
        return Err(FieldError::Io(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            format!(
                "another chitta-field instance holds {} (recorded holder: {}) — refusing to open the same store twice",
                path.display(), if holder.trim().is_empty() { "unknown" } else { holder.trim() }
            ),
        )));
    }
    unreachable!("instance lock: two attempts exhausted")
}

pub struct ChittaField {
    /// Exclusive advisory lock on `<data_dir>/.instance.lock`, held for the life
    /// of this instance. A second instance on the same directory would compact
    /// away this one's live WAL segment (2026-09-14: the CLI auto-started a
    /// second chittad on ~/.claude/mind and 15 minutes of appends went to a
    /// deleted file). `CHITTA_STORE_LOCK=0` disables the check.
    #[allow(dead_code)]
    pub(crate) instance_lock: Option<std::fs::File>,
    #[allow(dead_code)]
    pub(crate) data_dir: PathBuf,
    #[allow(dead_code)]
    pub(crate) instance_id: InstanceId,
    /// Store-identity lineage (PR3): bumped each open; persisted in the `.shdr` sidecar.
    pub(crate) lineage_epoch: u64,
    /// Stable writer identity for this lineage; carried forward across opens.
    pub(crate) writer_uuid: u128,
    pub(crate) log: RwLock<OpLog>,
    pub(crate) pending_touches: Mutex<Vec<(MemoryId, i64)>>,
    pub(crate) touch_drain: Mutex<()>,
    pub(crate) id_alloc: Arc<MemoryIdAllocator>,
    pub(crate) artifact_id_alloc: Arc<ArtifactIdAllocator>,
    pub(crate) payloads: RwLock<HashMap<MemoryId, MemoryPayload>>,
    /// Stage B dual-store: id -> natural-language retrieval surface. When present,
    /// this (not the telegraphic content) is what gets embedded, so recall matches
    /// natural-language queries. Persisted to the .rsf sidecar, never into bincode.
    pub(crate) retrieval_surfaces: RwLock<HashMap<MemoryId, Vec<u8>>>,
    pub(crate) states: RwLock<HashMap<MemoryId, MemoryState>>,
    pub(crate) assoc_edges: RwLock<HashMap<MemoryId, Vec<AssocEdge>>>,
    pub(crate) artifacts: RwLock<HashMap<String, ArtifactId>>,
    pub(crate) artifact_paths: RwLock<HashMap<ArtifactId, String>>,
    pub(crate) semantic_idx: RwLock<SemanticIndex>,
    pub(crate) time_idx: RwLock<TemporalIndex>,
    pub(crate) artifact_idx: RwLock<ArtifactIndex>,
    pub(crate) keyword_idx: RwLock<KeywordIndex>,
    pub(crate) triplet_store: RwLock<TripletStore>,
    pub(crate) triplet_id_alloc: Arc<TripletIdAllocator>,
    pub(crate) symbol_idx: RwLock<SymbolIndex>,
    pub(crate) call_graph: RwLock<CallGraph>,
    pub(crate) code_files: RwLock<CodeFileIndex>,
    pub(crate) symbol_id_alloc: Arc<TripletIdAllocator>,
    pub(crate) code_file_id_alloc: Arc<TripletIdAllocator>,
    pub(crate) learners: RwLock<LearnerSet>,
    pub(crate) sparse_encoder: RwLock<SparseEncoder>,
    pub(crate) cortical_idx: RwLock<CorticalIndex>,
    pub(crate) event_id_alloc: Arc<AtomicU64>,
    pub(crate) session_registry: RwLock<SessionRegistry>,
    pub(crate) transcript_registry: RwLock<TranscriptRegistry>,
    pub(crate) task_registry: RwLock<TaskRegistry>,
    pub(crate) user_model_registry: RwLock<UserModelRegistry>,
    pub(crate) theme_organ: RwLock<ThemeOrgan>,
    pub(crate) analytics_registry: RwLock<AnalyticsRegistry>,
    pub(crate) msg_registry: RwLock<MsgRegistry>,
    pub(crate) skill_registry: RwLock<SkillRegistry>,
    pub(crate) agent_registry: RwLock<AgentRegistry>,
    pub(crate) constraint_store: RwLock<ConstraintStore>,
    pub(crate) trigger_store: RwLock<TriggerStore>,
    pub(crate) predictor: RwLock<AccessPredictor>,
    pub(crate) surprise_store: RwLock<SurpriseStore>,
    pub(crate) epistemic_debt_store: RwLock<EpistemicDebtStore>,
    pub(crate) integration_kernel: RwLock<IntegrationKernel>,
    pub(crate) surprise_learning: RwLock<SurpriseLearningStore>,
    pub(crate) wisdom_promotion: RwLock<WisdomPromotionStore>,
    pub(crate) learned_scorer: RwLock<LearnedScoringModel>,
    pub(crate) intervention_store: RwLock<InterventionStore>,
    pub(crate) agent_protocol_store: RwLock<AgentProtocolStore>,
    pub(crate) wisdom_lineage_store: RwLock<WisdomLineageStore>,
    pub(crate) symbol_event_log: RwLock<SymbolEventLog>,
    pub(crate) lite_encoder: RwLock<Option<LiteEncoder>>,
    /// Byte offsets for each foreign segment file, used by sync_foreign().
    pub(crate) seen_offsets: RwLock<HashMap<PathBuf, u64>>,
    /// Foreign ops read off disk by sync_foreign_collect(), awaiting sync_foreign_apply().
    /// The two phases are split so the segment reads — which hit a hard-mounted NFS volume and
    /// can block in uninterruptible D state indefinitely — never run under the daemon's global
    /// rpc_mutex_. Holding that lock across a stalled read froze every request while the socket
    /// kept accepting connections (the "wedged daemon", 2026-07-14).
    pub(crate) pending_foreign: RwLock<PendingForeign>,
    pub(crate) chunk_hash_idx: RwLock<HashMap<crate::ids::ChunkHash, MemoryId>>,
    /// Cross-realm provenance dedup for kind="signal" "[done]" records.
    /// Key: DefaultHasher(content) as u64. Prevents re-processing already-handled files.
    pub(crate) content_prov_idx: RwLock<HashMap<u64, MemoryId>>,
    /// Keyed provenance lane (capability #1, anti-reprocessing). Semantic exact
    /// keys parsed out of `[done]` records ("sha:<hex>", "input:<path>") ->
    /// earliest live record id. Powers `provenance_lookup`, a deterministic O(1)
    /// query that bypasses the fuzzy retriever. Derived from live payloads and
    /// rebuilt on load (no snapshot-format change, migration-free, rollback-safe).
    pub(crate) prov_key_idx: RwLock<HashMap<String, MemoryId>>,
    /// Correction keyed lane (capability #2, durable corrections with override).
    /// Distinctive bigram of a `[correction]` record's corrected-mistake phrase
    /// -> ALL live correction ids carrying that trigger (multi-valued). A single
    /// latest-wins slot lost distinctive bigrams to newer corrections sharing a
    /// domain-common word (`afdb_background`), so a record only fired on its rare
    /// bigrams — observed live: a mistake restatement matching 3 of a record's
    /// bigrams fired nothing. The set lets `correction_check` credit a record for
    /// every trigger it owns; supersession is preserved by filtering deleted ids
    /// and ordering newest-first at query time. Powers `correction_check`, a
    /// deterministic O(turn_tokens) exact-key probe that reserves an injection
    /// slot for a correction whose mistake recurs in a turn — replacing the fuzzy
    /// recall that loses corrections on cosine similarity ~99% of the time.
    /// Derived, rebuilt on load (no snapshot-format change, migration-free,
    /// rollback-safe — same discipline as prov_key_idx).
    /// ceiling: a bigram's id-vec is unbounded; bounded in practice by distinct
    /// corrections sharing it; upgrade: cap to newest-N if a bigram ever explodes.
    pub(crate) correction_key_idx: RwLock<HashMap<String, Vec<MemoryId>>>,
    /// Task-state keyed lane (capability #3, task hand-off across discontinuous
    /// sessions). Task slug (`task:<slug>`) -> the NEWEST live `[task]` record id
    /// for that task (LATEST-WINS / SUPERSEDE, since status evolves). Powers
    /// `task_state_lookup`, a deterministic O(1) exact-key probe that answers
    /// "what is the current state of task X?" without the fuzzy retriever.
    /// Derived, rebuilt on load (no snapshot-format change, migration-free,
    /// rollback-safe — same discipline as correction_key_idx).
    pub(crate) task_key_idx: RwLock<HashMap<String, MemoryId>>,
    pub(crate) realm_members: RwLock<HashMap<String, HashSet<MemoryId>>>,
    pub(crate) kind_members:  RwLock<HashMap<String, HashSet<MemoryId>>>,
    /// Write-time densification: per-session recency ring (last K memory ids),
    /// used to add SameSession assoc edges linking a new memory to its recent
    /// same-session siblings. Transient — not persisted (recency only spans a
    /// live session, never a restart boundary).
    pub(crate) session_recent: RwLock<HashMap<String, std::collections::VecDeque<MemoryId>>>,
    /// Write-time densification gate. Default off; enabled via CHITTA_DENSIFY=1
    /// at construction so the write rule stays reviewable before it goes live.
    pub(crate) densify_enabled: std::sync::atomic::AtomicBool,
    /// Gate B plasticity gate. Default off; enabled via CHITTA_PLASTICITY_DECAY=1
    /// at construction. When on: the hourly demotion pass decays CoRetrieved
    /// edges (x0.98, prune <0.05) and drain applies a 0.1 materialization floor
    /// to new co-retrieval pairs (single co-occurrences never become edges).
    pub(crate) plasticity_decay_enabled: std::sync::atomic::AtomicBool,
    /// Factual-bridge recall leg gate. Default off; enabled via CHITTA_BRIDGE_LANE=1
    /// at construction. When on, recall reserves a few top-k slots for df-gated
    /// IDF atom-postings neighbours of the top dense hit (silent when the anchor
    /// carries no atom with corpus df <= tau).
    pub(crate) bridge_lane_enabled: std::sync::atomic::AtomicBool,
    pub(crate) memory_count: Arc<AtomicUsize>,
    pub(crate) pending_embed_count: Arc<AtomicUsize>,
    pub(crate) last_compact_ms: Arc<std::sync::atomic::AtomicI64>,
    pub(crate) pending_recall: Mutex<PendingRecallEffects>,
    /// Staged HNSW insert plan for the deferred-batched-insert backfill path
    /// (Step-1 re-architecture). `backfill_stage()` fills the ids under a brief
    /// write; `backfill_plan()` computes the plan off the write lock;
    /// `backfill_apply()` drains it under a brief write. `None` when idle.
    pub(crate) backfill_plan_stage: Mutex<Option<crate::store::BackfillStage>>,
    pub(crate) coactivation_stats: RwLock<HashMap<(MemoryId, MemoryId), CoActivationStats>>,
    /// Asymmetric Hopfield network for energy-based attractor recall. FEP §3.2.
    pub(crate) hopfield: RwLock<HopfieldNetwork>,
    pub(crate) filter_level: std::sync::Arc<std::sync::atomic::AtomicU8>,
    pub(crate) scoring_pipeline: RwLock<crate::scoring::ScoringPipeline>,
    pub(crate) realm_stats: RwLock<HashMap<String, crate::store::GroupStats>>,
    pub(crate) kind_stats:  RwLock<HashMap<String, crate::store::GroupStats>>,
    /// Ack/nack usage scores — persisted in FullSnapshot.ack_scores (v9+).
    pub(crate) ack_scores: RwLock<HashMap<MemoryId, i32>>,
    /// Soul REPL session namespaces — persisted to repl_sessions.json (not in snapshot).
    pub(crate) repl_sessions: RwLock<crate::repl_sessions::ReplSessionStore>,
    /// Span Lane — verbatim transcript atoms; persisted to spans.bin (not in
    /// snapshot; rebuilt on write, decay-immune by construction).
    pub(crate) span_store: RwLock<crate::organ::span_store::SpanStore>,
    /// Hyperdimensional Computing index — O(n) Hamming recall, no floats.
    pub(crate) hdc_idx:    RwLock<crate::hdc::HdcStore>,
    pub(crate) event_tape:   RwLock<crate::organ::event_tape::EventTape>,
    pub(crate) cdawg:        RwLock<crate::organ::cdawg::CdawgOrgan>,
    pub(crate) episode_hdc:        RwLock<crate::hdc::EpisodeHdcStore>,
    pub(crate) refutation_ledger:  RwLock<crate::organ::refutation_ledger::RefutationLedger>,
    pub(crate) cec_policy_store:   RwLock<crate::organ::intervention_store::InterventionStore>,
    pub(crate) decision_tape:      RwLock<crate::organ::decision_tape::DecisionTape>,
    /// Ephemeral — rebuilt from refutation_ledger after each consolidation_pass.
    pub(crate) hypothesis_market:  RwLock<crate::organ::hypothesis_market::HypothesisMarket>,
    /// CEC Phase 11 — Turīya Monitor: read-only organ that watches CEC organ health.
    /// Serialized in snapshot (rolling 100-sample window persists across sessions).
    pub(crate) turiya_monitor:     RwLock<crate::organ::turiya_monitor::TuriyaMonitor>,
    /// Ephemeral — rebuilt from EventTape alongside CDAWG at load.
    pub(crate) fep_prior:          RwLock<crate::organ::fep_prior::FepPriorOrgan>,
    /// CEC Phase 12 — cumulative count of events tombstoned by temporal compression.
    /// Ephemeral (not in snapshot) — lifetime of this daemon process.
    pub(crate) tape_tombstoned:    std::sync::atomic::AtomicU64,
    pub(crate) observer:           crate::organ::observer::Observer,
    pub(crate) observer_state:     RwLock<crate::organ::observer::ObserverState>,
    pub(crate) interaction_ledger: RwLock<crate::organ::interaction_ledger::InteractionLedger>,
    pub(crate) predicate_store:    RwLock<crate::organ::predicate_store::PredicateStore>,
    /// In-flight competitive_weight refresh reservations: memory_id -> reservation_ts_ms.
    /// Prevents thundering-herd when multiple sessions refresh simultaneously.
    pub(crate) cw_refresh_inflight: parking_lot::RwLock<std::collections::HashMap<crate::ids::MemoryId, i64>>,
    /// Per-writer WAL coverage of the in-memory state: instance → max seqno
    /// applied (open replay ⊔ sync_foreign ⊔ own appends at save time).
    /// Written into the manifest family on save; the safe pruning and
    /// replay-skip vector of THEORY.md §4.
    pub(crate) wal_coverage: parking_lot::RwLock<std::collections::BTreeMap<crate::ids::InstanceId, u64>>,
    /// SemanticIndex mutation count at the last successful sidecar write by
    /// this instance. u64::MAX = never written (first save must write).
    /// Dirty-skip: unchanged index ⇒ the .emb/.bin/.mu/.hnsw/.delta.hnsw/
    /// .realm_hnsw files from the previous save are still current.
    pub(crate) idx_sidecars_saved_at: std::sync::atomic::AtomicU64,
    /// HdcStore mutation count at the last .hdc sidecar write (same scheme).
    pub(crate) hdc_sidecar_saved_at: std::sync::atomic::AtomicU64,
    /// Payload CONTENT mutation counter + the value at the last .pld write.
    /// Audited bump sites: put_memory insert, cf_update_memory_content, and
    /// sync_foreign (wholesale). Removals never need a bump: the loader fills
    /// only ids present in the snapshot body, so stale extras are ignored.
    /// The counter MUST be read before the payloads clone in
    /// save_full_snapshot — a put racing the save then forces a rewrite on
    /// the NEXT save instead of leaving a skippable stale file.
    pub(crate) pld_mutations: std::sync::atomic::AtomicU64,
    pub(crate) pld_saved_at: std::sync::atomic::AtomicU64,
    /// Memories whose sparse encode produced an empty code (runtime-only;
    /// retried after restart). Keeps encode_all_unindexed from re-encoding
    /// the same unencodable ids every consolidation cycle.
    pub(crate) encode_skip: parking_lot::RwLock<HashSet<MemoryId>>,
    /// memory → distinct instances that recalled it (cross-context
    /// generality evidence; THEORY.md §6). Capped at 8 per memory.
    /// Persisted as the V23 "recall_provenance" section.
    pub(crate) recall_provenance: parking_lot::RwLock<HashMap<MemoryId, std::collections::BTreeSet<crate::ids::InstanceId>>>,
    /// G6: quality-diversity (MAP-Elites) archive — best genome per (realm, task_type) niche.
    pub(crate) archive: std::sync::Arc<std::sync::RwLock<crate::learner::archive::QdArchive>>,
}

impl Drop for ChittaField {
    fn drop(&mut self) {
        if let Err(e) = self.flush().and_then(|_| self.sync_wal()) {
            eprintln!("[field] shutdown flush failed: {e}");
        }
    }
}

impl ChittaField {
    /// Flush the write buffer to the OS.
    pub fn flush(&self) -> Result<()> {
        self.drain_pending_touches()?;
        self.drain_pending_recall_effects()?;
        self.log.write().flush_buf()
    }

    /// fdatasync the WAL to disk (durable). Separated from flush() (buffer→OS only) so the
    /// disk sync can run OFF the C++ rpc_mutex: put_memory flush_buf()s the append under the
    /// lock, the caller calls this after releasing the lock. Recall no longer blocks on the
    /// per-write fsync.
    pub fn sync_wal(&self) -> Result<()> {
        self.log.write().sync()
    }

    /// Return the current chain tip hash (SHA256). Zero if only V1 data.
    pub fn chain_head(&self) -> crate::log::ChainHash {
        self.log.read().chain_head()
    }

    pub fn open(data_dir: PathBuf) -> Result<Self> {
        Self::open_with_lock(data_dir, true)
    }

    /// Open without the exclusive instance lock. Only for tests that model a
    /// peer instance on the same directory (sync_foreign); production callers
    /// use `open`, and a shared directory needs `CHITTA_STORE_LOCK=0` explicitly.
    pub fn open_unlocked(data_dir: PathBuf) -> Result<Self> {
        Self::open_with_lock(data_dir, false)
    }

    fn open_with_lock(data_dir: PathBuf, lock: bool) -> Result<Self> {
        #[cfg(feature = "deadlock-detection")]
        {
            static CHECKER: std::sync::Once = std::sync::Once::new();
            CHECKER.call_once(|| {
                std::thread::spawn(|| loop {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    let deadlocks = parking_lot::deadlock::check_deadlock();
                    if deadlocks.is_empty() { continue; }
                    eprintln!("[chitta-field] {} DEADLOCK(S) DETECTED", deadlocks.len());
                    for (i, threads) in deadlocks.iter().enumerate() {
                        eprintln!("deadlock #{i}");
                        for t in threads {
                            eprintln!("thread {:?}\n{:?}", t.thread_id(), t.backtrace());
                        }
                    }
                    std::process::abort();
                });
            });
        }
        std::fs::create_dir_all(&data_dir)?;
        std::fs::create_dir_all(data_dir.join("segments"))?;
        let instance_lock = if lock { acquire_instance_lock(&data_dir)? } else { None };

        // Each open() generates a fresh InstanceId — no coordination needed.
        let instance_id = new_instance_id();

        // Open this instance's write log.
        let mut log = OpLog::open(&data_dir, instance_id, 1)?;

        // Allocators are partitioned by instance_id — no collision with other instances.
        let id_alloc = Arc::new(MemoryIdAllocator::with_instance(instance_id));
        let artifact_id_alloc = Arc::new(ArtifactIdAllocator::with_instance(instance_id));

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
        let mut transcript_registry = TranscriptRegistry::new();
        let mut task_registry = TaskRegistry::new();
        let mut user_model_registry = UserModelRegistry::new();
        let mut theme_organ = ThemeOrgan::new();
        let mut analytics_registry = AnalyticsRegistry::new();
        let mut msg_registry = MsgRegistry::new();
        let mut skill_registry = SkillRegistry::new();
        let mut agent_registry = AgentRegistry::new();
        let mut constraint_store = ConstraintStore::new();
        let mut trigger_store = TriggerStore::new();
        let predictor = AccessPredictor::new();
        let mut surprise_store = SurpriseStore::new();
        let mut epistemic_debt_store = EpistemicDebtStore::new();
        let mut integration_kernel = IntegrationKernel::new();
        let mut surprise_learning = SurpriseLearningStore::new();
        let mut wisdom_promotion = WisdomPromotionStore::new();
        let mut learned_scorer = LearnedScoringModel::new("v5.14".to_string());
        let mut intervention_store = InterventionStore::new();
        let mut agent_protocol_store = AgentProtocolStore::new();
        let mut wisdom_lineage_store = WisdomLineageStore::new();
        let mut symbol_event_log = SymbolEventLog::new();
        let mut chunk_hash_idx: HashMap<crate::ids::ChunkHash, MemoryId> = HashMap::new();
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

        // Replay ALL segment files to rebuild in-memory state.
        // Skip ops already covered by the full snapshot or cortical snapshot.
        let mut replay_realm_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
        let mut replay_kind_members:  HashMap<String, HashSet<MemoryId>> = HashMap::new();
        let mut replay_coactivation_stats = snapshot_coactivation_stats;
        triplet_store.correction_states = snap_correction_states;
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
        // Inhibit HNSW inserts during replay — binary Hamming takes over after normalize_all(),
        // so building the O(N log N) HNSW graph incrementally would waste time and RAM.
        semantic_idx.set_inhibit_hnsw(true);
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
        let mut ctx = ApplyCtx {
            payloads: &mut payloads,
            states: &mut states,
            assoc_edges: &mut assoc_edges,
            artifacts: &mut artifacts,
            artifact_paths: &mut artifact_paths,
            semantic_idx: &mut semantic_idx,
            time_idx: &mut time_idx,
            artifact_idx: &mut artifact_idx,
            keyword_idx: &mut keyword_idx,
            triplet_store: &mut triplet_store,
            symbol_idx: &mut symbol_idx,
            call_graph: &mut call_graph,
            code_files: &mut code_files,
            cortical_idx: &mut cortical_idx,
            session_registry: &mut session_registry,
            transcript_registry: &mut transcript_registry,
            task_registry: &mut task_registry,
            user_model_registry: &mut user_model_registry,
            theme_organ: &mut theme_organ,
            analytics_registry: &mut analytics_registry,
            msg_registry: &mut msg_registry,
            skill_registry: &mut skill_registry,
            agent_registry: &mut agent_registry,
            constraint_store: &mut constraint_store,
            trigger_store: &mut trigger_store,
            surprise_store: &mut surprise_store,
            epistemic_debt_store: &mut epistemic_debt_store,
            integration_kernel: &mut integration_kernel,
            surprise_learning: &mut surprise_learning,
            wisdom_promotion: &mut wisdom_promotion,
            learned_scorer: &mut learned_scorer,
            intervention_store: &mut intervention_store,
            agent_protocol_store: &mut agent_protocol_store,
            wisdom_lineage_store: &mut wisdom_lineage_store,
            symbol_event_log: &mut symbol_event_log,
            chunk_hash_idx: &mut chunk_hash_idx,
            realm_members: &mut replay_realm_members,
            kind_members: &mut replay_kind_members,
            coactivation_stats: &mut replay_coactivation_stats,
        };
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
            if let Some(state) = states.get_mut(&d.memory_id) {
                let replay_now = if d.op_ts_ms > 0 { d.op_ts_ms } else { state.created_at_ms };
                state.apply_delta(&d, replay_now);
            }
        }
        for d in orphan_accesses {
            if let Some(state) = states.get_mut(&d.memory_id) { apply_access_delta(state, &d); }
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
            for (id, payload) in &payloads {
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
        {
            let _phase = crate::profile::LoadPhase::new("keyword_reverse");
            keyword_idx.rebuild_reverse_index();
        }

        // One-time migration: mark SSL memories (content with →) for gloss-baked re-embed.
        // Guard file prevents re-running after backfill completes.
        {
            let flag = data_dir.join("ssl_gloss_v1.migrated");
            if !flag.exists() {
                let arrow: &[u8] = b"\xe2\x86\x92"; // UTF-8 →
                let mut ssl_count = 0usize;
                for (id, payload) in &payloads {
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
                for (id, payload) in &payloads {
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

        let realm_members = build_realm_members(&payloads, &states);
        let kind_members  = build_kind_members(&payloads, &states);
        let init_pending = states.values().filter(|s| s.embed_pending && !s.deleted).count();

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

        let triplet_id_alloc = Arc::new(TripletIdAllocator::new(triplet_store.next_id()));

        // Heal the symbol index's derived maps (by_name/dedup) from by_id —
        // snapshots written before the dedup-key fix carry line-keyed dedup
        // entries and by_name buckets with recycled ids.
        let symbols_phase = crate::profile::LoadPhase::new("symbols");
        let symbol_dups = symbol_idx.rebuild_derived();
        drop(symbols_phase);
        if symbol_dups > 0 {
            eprintln!(
                "[chitta-field] symbol index: {} duplicate entries detected (run dedupe_symbols to GC)",
                symbol_dups
            );
        }

        // Sync symbol and code-file id allocators from the loaded indexes.
        let max_symbol_id = symbol_idx.max_id().unwrap_or(0);
        let symbol_id_alloc = Arc::new(TripletIdAllocator::new(max_symbol_id + 1));
        let max_code_file_id = code_files.max_id().unwrap_or(0);
        let code_file_id_alloc = Arc::new(TripletIdAllocator::new(max_code_file_id + 1));

        // These inputs are immutable after WAL replay + normalization. No store
        // guards exist yet: quantization, HDC and lite I/O can run independently.
        let (loaded_lite_encoder, hdc_store) = std::thread::scope(|scope| {
            let lite = scope.spawn(|| {
                let _phase = crate::profile::LoadPhase::new("lite_encoder");
                Self::load_lite_encoder(&data_dir)
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
        turbo.join().expect("startup Turbo worker panicked");
        let loaded_lite_encoder = lite.join().expect("startup lite encoder worker panicked");
        (loaded_lite_encoder, hdc_store)

        });
        semantic_idx.prune_turbo_changes();
        let loaded_seen_offsets = Self::load_seen_offsets(&data_dir, instance_id);
        let scoring_config = crate::scoring::config::ScoringConfig::load(&data_dir);
        let loaded_repl_sessions = crate::repl_sessions::ReplSessionStore::load(&data_dir);
        let span_phase = crate::profile::LoadPhase::new("span_store");
        let loaded_span_store = crate::organ::span_store::SpanStore::load(&data_dir);
        drop(span_phase);

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
        let refutation_ledger = crate::organ::refutation_ledger::RefutationLedger::new();
        let cec_policy_store  = crate::organ::intervention_store::InterventionStore::new();
        let decision_tape     = snap_decision_tape;
        let hypothesis_market = crate::organ::hypothesis_market::HypothesisMarket::new();
        let turiya_monitor    = crate::organ::turiya_monitor::TuriyaMonitor::new();
        let fep_prior         = crate::organ::fep_prior::FepPriorOrgan::new();

        let initial_memory_count = payloads.len();
        // Store identity (PR3): carry the lineage forward from the loaded .shdr — bumping
        // the epoch to mark this new write session — or mint a fresh lineage for a legacy
        // store (no/!matching .shdr). writer_uuid mixes instance_id with open-time nanos.
        let (prior_epoch, writer_uuid) = match &loaded_header {
            Some(h) => (h.lineage_epoch, h.writer_uuid),
            None => {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                (0u64, ((instance_id as u128) << 96) ^ nanos)
            }
        };
        let lineage_epoch = prior_epoch.saturating_add(1);
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
        crate::replication::rebuild(&payloads, &triplet_store, &mut states);
        Ok(Self {
            instance_lock,
            data_dir,
            instance_id,
            lineage_epoch,
            writer_uuid,
            log: RwLock::new(log),
            pending_touches: Mutex::new(Vec::new()),
            touch_drain: Mutex::new(()),
            id_alloc,
            artifact_id_alloc,
            payloads: RwLock::new(payloads),
            retrieval_surfaces: RwLock::new(retrieval_surfaces),
            states: RwLock::new(states),
            assoc_edges: RwLock::new(assoc_edges),
            artifacts: RwLock::new(artifacts),
            artifact_paths: RwLock::new(artifact_paths),
            semantic_idx: RwLock::new(semantic_idx),
            time_idx: RwLock::new(time_idx),
            artifact_idx: RwLock::new(artifact_idx),
            keyword_idx: RwLock::new(keyword_idx),
            triplet_store: RwLock::new({
                let _phase = crate::profile::LoadPhase::new("triplets");
                let before = triplet_store.triplet_count();
                let (purged, deduped) = triplet_store.clean_for_load();
                if purged > 0 || deduped > 0 {
                    eprintln!("[chitta-field] triplet migration on load: purged {} invalidated, deduped {} duplicates ({} → {})",
                        purged, deduped, before, triplet_store.triplet_count());
                }
                triplet_store
            }),
            triplet_id_alloc,
            symbol_idx: RwLock::new(symbol_idx),
            call_graph: RwLock::new(call_graph),
            code_files: RwLock::new(code_files),
            symbol_id_alloc,
            code_file_id_alloc,
            learners: RwLock::new(LearnerSet::new()),
            sparse_encoder: RwLock::new(SparseEncoder::new()),
            cortical_idx: RwLock::new(cortical_idx),
            event_id_alloc: Arc::new(AtomicU64::new(1)),
            session_registry: RwLock::new(session_registry),
            transcript_registry: RwLock::new(transcript_registry),
            task_registry: RwLock::new(task_registry),
            user_model_registry: RwLock::new(user_model_registry),
            theme_organ: RwLock::new(theme_organ),
            analytics_registry: RwLock::new(analytics_registry),
            msg_registry: RwLock::new(msg_registry),
            skill_registry: RwLock::new(skill_registry),
            agent_registry: RwLock::new(agent_registry),
            constraint_store: RwLock::new(constraint_store),
            trigger_store: RwLock::new(trigger_store),
            predictor: RwLock::new(predictor),
            surprise_store: RwLock::new(surprise_store),
            epistemic_debt_store: RwLock::new(epistemic_debt_store),
            integration_kernel: RwLock::new(integration_kernel),
            surprise_learning: RwLock::new(surprise_learning),
            wisdom_promotion: RwLock::new(wisdom_promotion),
            learned_scorer: RwLock::new(learned_scorer),
            intervention_store: RwLock::new(intervention_store),
            agent_protocol_store: RwLock::new(agent_protocol_store),
            wisdom_lineage_store: RwLock::new(wisdom_lineage_store),
            symbol_event_log: RwLock::new(symbol_event_log),
            lite_encoder: RwLock::new(loaded_lite_encoder),
            seen_offsets: RwLock::new(loaded_seen_offsets),
            pending_foreign: RwLock::new(PendingForeign::default()),
            chunk_hash_idx: RwLock::new(chunk_hash_idx),
            content_prov_idx: RwLock::new(content_prov_idx),
            prov_key_idx: RwLock::new(prov_key_idx),
            correction_key_idx: RwLock::new(correction_key_idx),
            task_key_idx: RwLock::new(task_key_idx),
            realm_members: RwLock::new(realm_members),
            kind_members:  RwLock::new(kind_members),
            session_recent: RwLock::new(HashMap::new()),
            densify_enabled: std::sync::atomic::AtomicBool::new(
                std::env::var("CHITTA_DENSIFY").map(|v| v == "1").unwrap_or(false),
            ),
            plasticity_decay_enabled: std::sync::atomic::AtomicBool::new(
                std::env::var("CHITTA_PLASTICITY_DECAY").map(|v| v == "1").unwrap_or(false),
            ),
            bridge_lane_enabled: std::sync::atomic::AtomicBool::new(
                std::env::var("CHITTA_BRIDGE_LANE").map(|v| v == "1").unwrap_or(false),
            ),
            memory_count: Arc::new(AtomicUsize::new(initial_memory_count)),
            pending_embed_count: Arc::new(AtomicUsize::new(init_pending)),
            last_compact_ms: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            realm_stats: RwLock::new(HashMap::new()),
            kind_stats:  RwLock::new(HashMap::new()),
            ack_scores:  RwLock::new(snap_ack_scores),
            repl_sessions: RwLock::new(loaded_repl_sessions),
            span_store: RwLock::new(loaded_span_store),
            pending_recall: Mutex::new(PendingRecallEffects::default()),
            backfill_plan_stage: Mutex::new(None),
            coactivation_stats: RwLock::new({
                let mut cs = replay_coactivation_stats;
                let n = cs.len();
                // Prune to 100 pairs/memory to bound startup RAM on old snapshots.
                crate::field::prune_coactivation_stats(&mut cs, 20);
                if cs.len() < n {
                    eprintln!("[chitta-field] pruned {} coactivation pairs on load", n - cs.len());
                }
                cs
            }),
            hopfield: RwLock::new(HopfieldNetwork::new()),
            filter_level: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
            scoring_pipeline: RwLock::new(crate::scoring::ScoringPipeline::new(scoring_config)),
            hdc_idx:    RwLock::new(hdc_store),
            event_tape:   RwLock::new(event_tape),
            cdawg:        RwLock::new(cdawg),
            episode_hdc:        RwLock::new(episode_hdc),
            refutation_ledger:  RwLock::new(refutation_ledger),
            cec_policy_store:   RwLock::new(cec_policy_store),
            decision_tape:      RwLock::new(decision_tape),
            hypothesis_market:  RwLock::new(hypothesis_market),
            turiya_monitor:     RwLock::new(turiya_monitor),
            fep_prior:          RwLock::new(fep_prior),
            tape_tombstoned:    std::sync::atomic::AtomicU64::new(0),
            observer:           crate::organ::observer::Observer::new(),
            observer_state:     RwLock::new(crate::organ::observer::ObserverState::default()),
            interaction_ledger: RwLock::new(snap_interaction_ledger),
            predicate_store:    RwLock::new(snap_predicate_store),
            cw_refresh_inflight: parking_lot::RwLock::new(std::collections::HashMap::new()),
            wal_coverage: parking_lot::RwLock::new(wal_coverage),
            idx_sidecars_saved_at: std::sync::atomic::AtomicU64::new(u64::MAX),
            hdc_sidecar_saved_at: std::sync::atomic::AtomicU64::new(u64::MAX),
            pld_mutations: std::sync::atomic::AtomicU64::new(0),
            pld_saved_at: std::sync::atomic::AtomicU64::new(u64::MAX),
            encode_skip: parking_lot::RwLock::new(HashSet::new()),
            recall_provenance: parking_lot::RwLock::new(recall_provenance),
            archive: std::sync::Arc::new(std::sync::RwLock::new(crate::learner::archive::QdArchive::new())),
        })
    }
}

impl ChittaField {
    pub fn set_filter_level(&self, level: crate::store::FilterLevel) {
        self.filter_level.store(level as u8, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn filter_level(&self) -> crate::store::FilterLevel {
        match self.filter_level.load(std::sync::atomic::Ordering::Relaxed) {
            1 => crate::store::FilterLevel::Signatures,
            2 => crate::store::FilterLevel::MinimalContext,
            _ => crate::store::FilterLevel::None,
        }
    }
}

impl ChittaField {
    /// Ingest new ops from all foreign-instance segment files.
    ///
    /// Scans `data_dir/segments/` for files not belonging to this instance, reads
    /// any bytes appended since the last call, and applies those ops to the in-memory
    /// state without writing to this instance's own WAL.  CRC mismatches or truncated
    /// reads at the tail of a file are treated as in-progress writes from a concurrent
    /// peer and are silently skipped.
    ///
    /// Returns the count of ops applied.
    pub fn sync_foreign(&self) -> crate::error::Result<usize> {
        self.sync_foreign_collect()?;
        self.sync_foreign_apply()
    }

    /// Phase 1: read new peer ops off disk into `pending_foreign`. Acquires NO state guards
    /// beyond `seen_offsets` (which no request path touches), so the caller must NOT hold the
    /// daemon's global rpc_mutex_ here — every read below can block forever on a hard-mounted
    /// NFS volume, and under that lock the whole daemon stops answering.
    ///
    /// Offsets advance here, so a batch that is collected but never applied would be lost.
    /// It isn't: ops accumulate in `pending_foreign` until an apply drains them.
    ///
    /// Returns the number of ops now pending.
    pub fn sync_foreign_collect(&self) -> crate::error::Result<usize> {
        use crate::log::{collect_foreign_segments, replay_from_offset};

        let foreign_segs = collect_foreign_segments(&self.data_dir, self.instance_id)?;
        let mut ops: Vec<(crate::ids::InstanceId, Op)> = Vec::new();
        let mut fresh_coverage: std::collections::BTreeMap<crate::ids::InstanceId, u64> =
            std::collections::BTreeMap::new();

        {
            let mut seen = self.seen_offsets.write();
            for seg_path in &foreign_segs {
                let inst: crate::ids::InstanceId = seg_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.split_once('_'))
                    .and_then(|(p, _)| u32::from_str_radix(p, 16).ok())
                    .unwrap_or(0);
                let offset = *seen.get(seg_path).unwrap_or(&0);
                let new_offset = replay_from_offset(seg_path, offset, |seqno, op| {
                    let cov = fresh_coverage.entry(inst).or_insert(0);
                    if seqno > *cov { *cov = seqno; }
                    ops.push((inst, op));
                    Ok(())
                })?;
                seen.insert(seg_path.clone(), new_offset);
            }
        }

        let mut pending = self.pending_foreign.write();
        pending.ops.append(&mut ops);
        for (inst, max) in fresh_coverage {
            let e = pending.coverage.entry(inst).or_insert(0);
            if max > *e { *e = max; }
        }
        Ok(pending.ops.len())
    }

    /// Phase 2: apply the ops collected by `sync_foreign_collect`. This is the part that takes
    /// ~40 write guards in canonical order, so the daemon holds its exclusive rpc_mutex_ across
    /// it (deadlock fix, 2026-06-11 — see simple_cli.cpp). Pure in-memory: no disk reads.
    ///
    /// Returns the count of ops applied.
    pub fn sync_foreign_apply(&self) -> crate::error::Result<usize> {
        let (ops, fresh_coverage) = {
            let mut pending = self.pending_foreign.write();
            if pending.ops.is_empty() {
                return Ok(0);
            }
            (
                std::mem::take(&mut pending.ops),
                std::mem::take(&mut pending.coverage),
            )
        };

        let count = ops.len();

        // Acquire all write locks and apply every foreign op in one pass,
        // reusing the same apply_op path used during startup replay.
        let mut payloads = self.payloads.write();
        let mut states = self.states.write();
        let mut assoc_edges = self.assoc_edges.write();
        let mut artifacts = self.artifacts.write();
        let mut artifact_paths = self.artifact_paths.write();
        let mut semantic_idx = self.semantic_idx.write();
        let mut time_idx = self.time_idx.write();
        let mut artifact_idx = self.artifact_idx.write();
        let mut keyword_idx = self.keyword_idx.write();
        let mut triplet_store = self.triplet_store.write();
        let mut symbol_idx = self.symbol_idx.write();
        let mut call_graph = self.call_graph.write();
        let mut code_files = self.code_files.write();
        let mut cortical_idx = self.cortical_idx.write();
        let mut session_reg = self.session_registry.write();
        let mut transcript_reg = self.transcript_registry.write();
        let mut task_reg = self.task_registry.write();
        let mut user_model_reg = self.user_model_registry.write();
        let mut theme_organ = self.theme_organ.write();
        let mut analytics_reg = self.analytics_registry.write();
        let mut msg_reg = self.msg_registry.write();
        let mut skill_reg = self.skill_registry.write();
        let mut agent_reg = self.agent_registry.write();
        let mut constraint_reg = self.constraint_store.write();
        let mut trigger_reg = self.trigger_store.write();
        let mut surprise_reg = self.surprise_store.write();
        let mut epistemic_debt_reg = self.epistemic_debt_store.write();
        let mut integration_reg = self.integration_kernel.write();
        let mut surprise_learning_reg = self.surprise_learning.write();
        let mut wisdom_promotion_reg = self.wisdom_promotion.write();
        let mut learned_scorer_reg = self.learned_scorer.write();
        let mut intervention_store_reg = self.intervention_store.write();
        let mut agent_protocol_store_reg = self.agent_protocol_store.write();
        let mut wisdom_lineage_store_reg = self.wisdom_lineage_store.write();
        let mut symbol_event_log_reg = self.symbol_event_log.write();
        let mut chunk_hash_idx = self.chunk_hash_idx.write();
        let mut realm_members = self.realm_members.write();
        let mut kind_members  = self.kind_members.write();
        let mut coactivation_stats = self.coactivation_stats.write();
        let mut recall_prov = self.recall_provenance.write();

        let mut ctx = ApplyCtx {
            payloads: &mut *payloads,
            states: &mut *states,
            assoc_edges: &mut *assoc_edges,
            artifacts: &mut *artifacts,
            artifact_paths: &mut *artifact_paths,
            semantic_idx: &mut *semantic_idx,
            time_idx: &mut *time_idx,
            artifact_idx: &mut *artifact_idx,
            keyword_idx: &mut *keyword_idx,
            triplet_store: &mut *triplet_store,
            symbol_idx: &mut *symbol_idx,
            call_graph: &mut *call_graph,
            code_files: &mut *code_files,
            cortical_idx: &mut *cortical_idx,
            session_registry: &mut *session_reg,
            transcript_registry: &mut *transcript_reg,
            task_registry: &mut *task_reg,
            user_model_registry: &mut *user_model_reg,
            theme_organ: &mut *theme_organ,
            analytics_registry: &mut *analytics_reg,
            msg_registry: &mut *msg_reg,
            skill_registry: &mut *skill_reg,
            agent_registry: &mut *agent_reg,
            constraint_store: &mut *constraint_reg,
            trigger_store: &mut *trigger_reg,
            surprise_store: &mut *surprise_reg,
            epistemic_debt_store: &mut *epistemic_debt_reg,
            integration_kernel: &mut *integration_reg,
            surprise_learning: &mut *surprise_learning_reg,
            wisdom_promotion: &mut *wisdom_promotion_reg,
            learned_scorer: &mut *learned_scorer_reg,
            intervention_store: &mut *intervention_store_reg,
            agent_protocol_store: &mut *agent_protocol_store_reg,
            wisdom_lineage_store: &mut *wisdom_lineage_store_reg,
            symbol_event_log: &mut *symbol_event_log_reg,
            chunk_hash_idx: &mut *chunk_hash_idx,
            realm_members: &mut *realm_members,
            kind_members: &mut *kind_members,
            coactivation_stats: &mut *coactivation_stats,
        };
        for (inst, op) in ops {
            if let Op::RecordRecallBatch(b) = &op {
                for mid in &b.memory_ids {
                    let set = recall_prov.entry(*mid).or_default();
                    if set.len() < 8 {
                        set.insert(inst);
                    }
                }
            }
            apply_op(op, ctx.reborrow());
        }

        crate::replication::rebuild(&payloads, &triplet_store, &mut states);
        if count > 0 {
            self.persist_seen_offsets();
            self.pld_mutations
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Ingested foreign ops are now part of in-memory state — extend
            // the coverage vector so the next snapshot commit claims them.
            let mut cov = self.wal_coverage.write();
            for (inst, max) in fresh_coverage {
                let e = cov.entry(inst).or_insert(0);
                if max > *e { *e = max; }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod sync_foreign_split_tests {
    use super::*;

    fn write_one(f: &ChittaField, content: &str) {
        f.put_memory(
            "episode",
            "test",
            content.as_bytes(),
            &[0.1f32; 768],
            1.0,
            0.0,
            0,
            Vec::new(),
            None,
            None,
        )
        .expect("put_memory");
        f.flush().expect("flush");
    }

    /// collect() advances seen_offsets, so those bytes are never re-read. If a collected batch
    /// could be dropped before apply(), the ops would be gone for good. Assert it survives:
    /// a second collect (which finds nothing new) must not discard what the first one buffered.
    #[test]
    fn collected_ops_survive_until_applied() {
        let dir = tempfile::tempdir().unwrap();
        let peer = ChittaField::open(dir.path().to_path_buf()).unwrap();
        // `local` must exist BEFORE the peer writes: a fresh instance seeds seen_offsets from
        // current segment sizes (see load_seen_offsets), so one opened afterwards starts at EOF
        // and would legitimately see nothing to sync.
        let local = ChittaField::open_unlocked(dir.path().to_path_buf()).unwrap();
        write_one(&peer, "peer wrote this");
        let before = local.memory_count();

        let pending = local.sync_foreign_collect().unwrap();
        assert!(pending > 0, "collect should buffer the peer's ops");
        assert_eq!(local.memory_count(), before, "collect must not mutate state");

        // Offsets are now past the peer's ops. A second collect finds nothing new — and must
        // not lose the batch the first one buffered.
        let still_pending = local.sync_foreign_collect().unwrap();
        assert_eq!(still_pending, pending, "second collect dropped the buffered ops");

        let applied = local.sync_foreign_apply().unwrap();
        assert_eq!(applied, pending, "apply must drain exactly what was collected");
        assert!(local.memory_count() > before, "peer memory should now be visible");

        // Draining is idempotent: nothing left to apply.
        assert_eq!(local.sync_foreign_apply().unwrap(), 0);
    }

    /// The split must be behaviour-preserving: sync_foreign() == collect() + apply().
    #[test]
    fn sync_foreign_still_applies_peer_ops() {
        let dir = tempfile::tempdir().unwrap();
        let peer = ChittaField::open(dir.path().to_path_buf()).unwrap();
        let local = ChittaField::open_unlocked(dir.path().to_path_buf()).unwrap();
        write_one(&peer, "one");
        write_one(&peer, "two");
        let before = local.memory_count();
        let applied = local.sync_foreign().unwrap();

        assert!(applied > 0, "sync_foreign applied nothing");
        assert!(local.memory_count() > before);
        assert_eq!(local.sync_foreign().unwrap(), 0, "re-sync should be a no-op");
    }
}

impl ChittaField {
    fn seen_offsets_path(
        data_dir: &std::path::Path,
        instance_id: crate::ids::InstanceId,
    ) -> std::path::PathBuf {
        data_dir.join(format!("seen_offsets.{:08x}.json", instance_id))
    }

    fn load_seen_offsets(
        data_dir: &std::path::Path,
        instance_id: crate::ids::InstanceId,
    ) -> HashMap<PathBuf, u64> {
        let path = Self::seen_offsets_path(data_dir, instance_id);
        if let Some(saved) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            return saved;
        }
        // No saved file — this is a new instance_id (fresh restart). Seed seen_offsets
        // from current segment file sizes so sync_foreign doesn't re-apply every op
        // from the startup replay. Segments that exist now were fully replayed at startup;
        // only bytes written AFTER this open() should be treated as new foreign ops.
        let mut offsets = HashMap::new();
        let seg_dir = data_dir.join("segments");
        if let Ok(rd) = std::fs::read_dir(&seg_dir) {
            for entry in rd.flatten() {
                if entry.metadata().map(|m| m.is_file()).unwrap_or(false) {
                    if let Ok(meta) = std::fs::metadata(entry.path()) {
                        offsets.insert(entry.path(), meta.len());
                    }
                }
            }
        }
        // Persist immediately so the next restart loads from file (fast path)
        // rather than re-scanning segment sizes.
        if let Ok(json) = serde_json::to_string(&offsets) {
            let path = Self::seen_offsets_path(data_dir, instance_id);
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, &json).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
        offsets
    }

    fn persist_seen_offsets(&self) {
        let path = Self::seen_offsets_path(&self.data_dir, self.instance_id);
        let offsets = self.seen_offsets.read();
        if let Ok(json) = serde_json::to_string(&*offsets) {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, &json).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }

    fn load_lite_encoder(data_dir: &std::path::Path) -> Option<LiteEncoder> {
        let path = data_dir.join("lite_encoder.bin");
        if !path.exists() {
            return None;
        }
        match std::fs::read(&path) {
            Ok(bytes) => match LiteEncoder::from_bytes(&bytes) {
                Ok(enc) => {
                    eprintln!(
                        "[chitta-field] loaded lite encoder ({} vocab, {} examples)",
                        enc.vocab.len(),
                        enc.training_examples
                    );
                    Some(enc)
                }
                Err(e) => {
                    eprintln!("[chitta-field] lite encoder load failed: {}", e);
                    None
                }
            },
            Err(_) => None,
        }
    }

    /// Train the lite encoder from all memories with sparse codes.
    /// Returns the number of training examples used.
    pub fn train_lite_encoder(&self) -> Result<usize> {
        let payloads = self.payloads.read();
        let cortical_idx = self.cortical_idx.read();

        let mut examples: Vec<(String, SparseCode)> = Vec::new();
        for (mem_id, payload) in payloads.iter() {
            if let Some(code) = cortical_idx.mem_codes.get(mem_id) {
                if !code.is_empty() {
                    let content = String::from_utf8_lossy(&payload.content).into_owned();
                    if !content.trim().is_empty() {
                        examples.push((content, code.clone()));
                    }
                }
            }
        }

        let count = examples.len();
        if count == 0 {
            return Ok(0);
        }

        let encoder = LiteEncoder::train(&examples);
        *self.lite_encoder.write() = Some(encoder);
        Ok(count)
    }

    /// Encode text via lite encoder. Returns None if not trained or no words match vocab.
    pub fn encode_lite(&self, text: &str) -> Option<SparseCode> {
        self.lite_encoder.read().as_ref()?.encode(text)
    }

    /// Save lite encoder to <data_dir>/lite_encoder.bin
    pub fn save_lite_encoder(&self) -> Result<()> {
        let guard = self.lite_encoder.read();
        let enc = guard.as_ref().ok_or_else(|| {
            crate::error::FieldError::Manifest("lite encoder not trained".to_string())
        })?;
        let bytes = enc.to_bytes();
        let path = self.data_dir.join("lite_encoder.bin");
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Check if the lite encoder is trained and ready.
    pub fn lite_encoder_ready(&self) -> bool {
        self.lite_encoder.read().is_some()
    }
}

/// Apply a single Op to the in-memory projections, including all indexes.
/// Borrow bundle for op application — every mutable structure an op can
/// touch. Built once per replay/sync batch; pass `ctx.reborrow()` per op.
/// Adding an organ = one field here, one line in reborrow(), one arm in
/// apply_op — the seam for the organ-trait migration (THEORY.md §8 Phase 2).
pub(crate) struct ApplyCtx<'a> {
    pub payloads: &'a mut HashMap<MemoryId, MemoryPayload>,
    pub states: &'a mut HashMap<MemoryId, MemoryState>,
    pub assoc_edges: &'a mut HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: &'a mut HashMap<String, ArtifactId>,
    pub artifact_paths: &'a mut HashMap<ArtifactId, String>,
    pub semantic_idx: &'a mut SemanticIndex,
    pub time_idx: &'a mut TemporalIndex,
    pub artifact_idx: &'a mut ArtifactIndex,
    pub keyword_idx: &'a mut KeywordIndex,
    pub triplet_store: &'a mut TripletStore,
    pub symbol_idx: &'a mut SymbolIndex,
    pub call_graph: &'a mut CallGraph,
    pub code_files: &'a mut CodeFileIndex,
    pub cortical_idx: &'a mut CorticalIndex,
    pub session_registry: &'a mut SessionRegistry,
    pub transcript_registry: &'a mut TranscriptRegistry,
    pub task_registry: &'a mut TaskRegistry,
    pub user_model_registry: &'a mut UserModelRegistry,
    pub theme_organ: &'a mut ThemeOrgan,
    pub analytics_registry: &'a mut AnalyticsRegistry,
    pub msg_registry: &'a mut MsgRegistry,
    pub skill_registry: &'a mut SkillRegistry,
    pub agent_registry: &'a mut AgentRegistry,
    pub constraint_store: &'a mut ConstraintStore,
    pub trigger_store: &'a mut TriggerStore,
    pub surprise_store: &'a mut SurpriseStore,
    pub epistemic_debt_store: &'a mut EpistemicDebtStore,
    pub integration_kernel: &'a mut IntegrationKernel,
    pub surprise_learning: &'a mut SurpriseLearningStore,
    pub wisdom_promotion: &'a mut WisdomPromotionStore,
    pub learned_scorer: &'a mut LearnedScoringModel,
    pub intervention_store: &'a mut InterventionStore,
    pub agent_protocol_store: &'a mut AgentProtocolStore,
    pub wisdom_lineage_store: &'a mut WisdomLineageStore,
    pub symbol_event_log: &'a mut SymbolEventLog,
    pub chunk_hash_idx: &'a mut HashMap<crate::ids::ChunkHash, MemoryId>,
    pub realm_members: &'a mut HashMap<String, HashSet<MemoryId>>,
    pub kind_members: &'a mut HashMap<String, HashSet<MemoryId>>,
    pub coactivation_stats: &'a mut HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

impl<'a> ApplyCtx<'a> {
    pub(crate) fn reborrow(&mut self) -> ApplyCtx<'_> {
        ApplyCtx {
            payloads: &mut *self.payloads,
            states: &mut *self.states,
            assoc_edges: &mut *self.assoc_edges,
            artifacts: &mut *self.artifacts,
            artifact_paths: &mut *self.artifact_paths,
            semantic_idx: &mut *self.semantic_idx,
            time_idx: &mut *self.time_idx,
            artifact_idx: &mut *self.artifact_idx,
            keyword_idx: &mut *self.keyword_idx,
            triplet_store: &mut *self.triplet_store,
            symbol_idx: &mut *self.symbol_idx,
            call_graph: &mut *self.call_graph,
            code_files: &mut *self.code_files,
            cortical_idx: &mut *self.cortical_idx,
            session_registry: &mut *self.session_registry,
            transcript_registry: &mut *self.transcript_registry,
            task_registry: &mut *self.task_registry,
            user_model_registry: &mut *self.user_model_registry,
            theme_organ: &mut *self.theme_organ,
            analytics_registry: &mut *self.analytics_registry,
            msg_registry: &mut *self.msg_registry,
            skill_registry: &mut *self.skill_registry,
            agent_registry: &mut *self.agent_registry,
            constraint_store: &mut *self.constraint_store,
            trigger_store: &mut *self.trigger_store,
            surprise_store: &mut *self.surprise_store,
            epistemic_debt_store: &mut *self.epistemic_debt_store,
            integration_kernel: &mut *self.integration_kernel,
            surprise_learning: &mut *self.surprise_learning,
            wisdom_promotion: &mut *self.wisdom_promotion,
            learned_scorer: &mut *self.learned_scorer,
            intervention_store: &mut *self.intervention_store,
            agent_protocol_store: &mut *self.agent_protocol_store,
            wisdom_lineage_store: &mut *self.wisdom_lineage_store,
            symbol_event_log: &mut *self.symbol_event_log,
            chunk_hash_idx: &mut *self.chunk_hash_idx,
            realm_members: &mut *self.realm_members,
            kind_members: &mut *self.kind_members,
            coactivation_stats: &mut *self.coactivation_stats,
        }
    }
}

pub(crate) fn apply_op(op: Op, ctx: ApplyCtx) {
    let ApplyCtx {
        payloads,
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
        realm_members,
        kind_members,
        coactivation_stats,
    } = ctx;
    // Organ-owned ops first (OrganApply, THEORY.md §8 Phase 2); the central
    // match below keeps only multi-structure ops. Migration is incremental —
    // move an organ's arms into its `impl OrganApply` and add a line here.
    use crate::organ::OrganApply;
    let op = match intervention_store.apply(op) { None => return, Some(op) => op };
    let op = match agent_protocol_store.apply(op) { None => return, Some(op) => op };
    let op = match wisdom_lineage_store.apply(op) { None => return, Some(op) => op };
    let op = match surprise_store.apply(op) { None => return, Some(op) => op };
    let op = match epistemic_debt_store.apply(op) { None => return, Some(op) => op };
    let op = match integration_kernel.apply(op) { None => return, Some(op) => op };
    let op = match surprise_learning.apply(op) { None => return, Some(op) => op };
    let op = match wisdom_promotion.apply(op) { None => return, Some(op) => op };
    let op = match learned_scorer.apply(op) { None => return, Some(op) => op };
    let op = match msg_registry.apply(op) { None => return, Some(op) => op };
    let op = match skill_registry.apply(op) { None => return, Some(op) => op };
    let op = match agent_registry.apply(op) { None => return, Some(op) => op };
    let op = match analytics_registry.apply(op) { None => return, Some(op) => op };
    let op = match trigger_store.apply(op) { None => return, Some(op) => op };
    let op = match constraint_store.apply(op) { None => return, Some(op) => op };
    let op = match transcript_registry.apply(op) { None => return, Some(op) => op };
    let op = match task_registry.apply(op) { None => return, Some(op) => op };
    let op = match user_model_registry.apply(op) { None => return, Some(op) => op };
    let op = match theme_organ.apply(op) { None => return, Some(op) => op };
    let op = match symbol_event_log.apply(op) { None => return, Some(op) => op };
    match op {
        Op::PutPayload(put) => {
            let memory_id = put.memory_id;
            let chunk_hash = put.chunk_hash;
            let created_at_ms = put.created_at_ms;
            let authored_at_ms = if put.authored_at_ms == 0 {
                created_at_ms
            } else {
                put.authored_at_ms
            };
            let version = put.version;
            let kind = put.kind.clone();
            let realm = put.realm.clone();
            let embedding = put.embedding.clone();
            let artifact_refs = put.artifact_refs.clone();
            let content_str = String::from_utf8(put.content.clone()).unwrap_or_default();

            let state = states
                .entry(memory_id)
                .or_insert_with(|| MemoryState::new(memory_id, chunk_hash, created_at_ms));
            state.current_version = version;
            state.current_chunk_hash = chunk_hash;
            let strength = state.strength;

            let mut pl = MemoryPayload::from(put);
            if !pl.embedding.is_empty() {
                // semantic_idx (upserted below) is the embedding's single
                // in-RAM home — see ChittaField::embedding_of.
                pl.embedding = Vec::new();
            }
            payloads.insert(memory_id, pl);
            chunk_hash_idx.entry(chunk_hash).or_insert(memory_id);
            semantic_idx.upsert(memory_id, embedding, Some(realm.as_str()));
            keyword_idx.index(memory_id, &content_str);
            realm_members
                .entry(realm.clone())
                .or_default()
                .insert(memory_id);
            kind_members
                .entry(kind.clone())
                .or_default()
                .insert(memory_id);

            time_idx.upsert(TemporalEntry {
                memory_id,
                ts_ms: authored_at_ms,
                kind,
                realm,
                strength,
            });

            for art_ref in &artifact_refs {
                if let Some(path) = artifact_paths.get(&art_ref.artifact_id) {
                    artifact_idx.associate(memory_id, art_ref.artifact_id, path, strength);
                }
            }
        }
        Op::UpdateStateBatch(deltas) => {
            for delta in deltas {
                if let Some(state) = states.get_mut(&delta.memory_id) {
                    apply_access_delta(state, &delta);
                }
            }
        }
        Op::UpdateState(delta) => {
            let memory_id = delta.memory_id;
            if let Some(state) = states.get_mut(&memory_id) {
                // Use op_ts_ms as the reference time during replay so that
                // last_accessed_ms / last_strengthened_ms are set to the
                // wall-clock time of the original operation, not epoch 0.
                let replay_now = if delta.op_ts_ms > 0 { delta.op_ts_ms } else { state.created_at_ms };
                state.apply_delta(&delta, replay_now);
            }
        }
        Op::DeleteMemory(del) => {
            let memory_id = del.memory_id;
            if let Some(state) = states.get_mut(&memory_id) {
                state.deleted = true;
            }
            semantic_idx.remove(memory_id);
            keyword_idx.remove(memory_id);
            if let Some(payload) = payloads.get_mut(&memory_id) {
                time_idx.remove(memory_id, payload.authored_at_ms);
                let remove_realm = if let Some(ids) = realm_members.get_mut(&payload.realm) {
                    ids.remove(&memory_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_realm {
                    realm_members.remove(&payload.realm);
                }
                let remove_kind = if let Some(ids) = kind_members.get_mut(&payload.kind) {
                    ids.remove(&memory_id);
                    ids.is_empty()
                } else {
                    false
                };
                if remove_kind {
                    kind_members.remove(&payload.kind);
                }
                // Clear content bytes so replay doesn't resurrect deleted text.
                payload.content = Vec::new();
            }
            artifact_idx.remove_memory(memory_id);
            // Transitive: remove assoc_edges from/to this memory. Triplets sourced from
            // this memory are handled by their own InvalidateTriplet WAL ops (written by
            // forget() at deletion time); no separate replay needed here.
            assoc_edges.remove(&memory_id);
            for outgoing in assoc_edges.values_mut() {
                outgoing.retain(|e| e.dst != memory_id);
            }
            coactivation_stats.retain(|(a, b), _| *a != memory_id && *b != memory_id);
        }
        Op::AddAssocEdge(edge_op) => {
            let entry = assoc_edges.entry(edge_op.src).or_insert_with(Vec::new);
            entry.push(AssocEdge {
                dst: edge_op.dst,
                edge_type: edge_op.edge_type,
                weight: edge_op.weight,
            });
        }
        Op::UpsertArtifact(art_op) => {
            artifacts
                .entry(art_op.normalized_path.clone())
                .or_insert(art_op.artifact_id);
            artifact_paths
                .entry(art_op.artifact_id)
                .or_insert(art_op.normalized_path);
        }
        Op::AddTriplet(t) => {
            triplet_store.replay_add(
                t.triplet_id,
                t.subject,
                t.predicate,
                t.object,
                t.weight,
                t.valid_from_ms,
                t.source_memory_id,
                t.source_file,
            );
        }
        Op::InvalidateTriplet(inv) => {
            triplet_store.invalidate(inv.triplet_id, inv.invalidated_at_ms);
        }
        Op::SupersedeTriplet(s) => {
            triplet_store.supersede(s.old_id, s.new_id, s.superseded_at_ms);
        }
        Op::UpsertSymbol(s) => {
            let entry = SymbolEntry {
                id: s.symbol_id,
                kind: s.kind,
                name: s.name,
                signature: s.signature,
                file_path: s.file_path,
                line_start: s.line_start,
                line_end: s.line_end,
                repo_id: s.repo_id,
                embedding: s.embedding,
                description: s.description,
                memory_id: s.memory_id,
            };
            symbol_idx.upsert(entry);
        }
        Op::RemoveSymbol(r) => {
            symbol_idx.remove(r.symbol_id);
            call_graph.remove_symbol(r.symbol_id);
        }
        Op::AddSymCallEdge(e) => {
            call_graph.add_edge(e.caller_id, e.callee_id);
        }
        Op::RemoveSymCallEdge(e) => {
            // Edges are stored in the bidirectional maps; remove individually.
            let callees = call_graph.get_callees(e.caller_id);
            if callees.contains(&e.callee_id) {
                // Reconstruct by removing and re-adding remaining edges for caller.
                let remaining: Vec<u64> =
                    callees.into_iter().filter(|&c| c != e.callee_id).collect();
                call_graph.remove_symbol(e.caller_id);
                for callee in remaining {
                    call_graph.add_edge(e.caller_id, callee);
                }
            }
        }
        Op::UpsertCodeFile(f) => {
            code_files.upsert(
                &f.path, &f.project, f.mtime,
                f.content_hash.clone(), f.git_commit.clone(),
                f.git_author.clone(), f.git_timestamp_ms,
                || f.file_id,
            );
        }
        Op::InvalidateTripletsBySourceFile(op) => {
            triplet_store.invalidate_by_source_file(&op.source_file, op.invalidated_at_ms);
        }
        Op::UpdateSparseCode(op) => {
            let code = SparseCode {
                feature_ids: op.feature_ids.clone(),
                activations: op.activations.clone(),
            };
            let strength = states.get(&op.memory_id).map(|s| s.strength).unwrap_or(0.5);
            let (kind, ts_ms) = payloads
                .get(&op.memory_id)
                .map(|p| (p.kind.as_str(), p.authored_at_ms))
                .unwrap_or(("unknown", op.ts_ms));
            cortical_idx.index(op.memory_id, &code, strength, ts_ms, kind);
        }
        Op::DemoteMemory(d) => {
            if let Some(state) = states.get_mut(&d.memory_id) {
                state.tier = d.new_tier;
            }
        }
        // Accrual, not assignment: the snapshot's `utility_posteriors` section
        // restores (α, β) as of the snapshot, and only the WAL suffix the
        // snapshot does not cover reaches here (see `covered_by_full` above),
        // so each observation lands exactly once.
        Op::RecordOutcome(r) => {
            if let Some(state) = states.get_mut(&r.memory_id) {
                state.record_outcome(r.success, r.weight);
            }
        }
        Op::TrainPQ(t) => {
            if let Ok(pq) = bincode::deserialize::<ProductQuantizer>(&t.codebook_bytes) {
                cortical_idx.set_pq(pq);
            }
        }
        Op::UpdateResidualPQ(u) => {
            if u.pq_bytes.len() == 32 {
                let mut codes = [0u8; 32];
                codes.copy_from_slice(&u.pq_bytes);
                cortical_idx.index_pq(u.memory_id, codes);
            }
        }
        Op::SessionEvent(ev) => {
            let payload_str = String::from_utf8(ev.payload_json.clone()).unwrap_or_default();
            match ev.kind.as_str() {
                "register" => {
                    let kind = serde_json::from_slice::<serde_json::Value>(&ev.payload_json)
                        .ok()
                        .and_then(|v| {
                            v.get("kind")
                                .and_then(|k| k.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_default();
                    session_registry.register(ev.session_id.clone(), kind, ev.realm.clone(), ev.ts_ms);
                }
                "heartbeat" => session_registry.heartbeat(&ev.session_id, ev.ts_ms),
                "deregister" => session_registry.deregister(&ev.session_id),
                _ => {}
            }
            // Mirror into msg_registry for get_events_by_domain_kind queries.
            use crate::organ::msg::MsgEvent;
            msg_registry.insert(MsgEvent {
                event_id: ev.event_id,
                domain: "session".to_string(),
                kind: ev.kind,
                target: ev.session_id,
                payload_json: payload_str,
                realm: ev.realm,
                ts_ms: ev.ts_ms,
            });
        }
        Op::ClearProject(cp) => {
            let removed_paths = code_files.remove_by_project(&cp.project);
            let removed_ids = symbol_idx.remove_by_file_paths(&removed_paths);
            for id in removed_ids {
                call_graph.remove_symbol(id);
            }
        }
        Op::UpdateSymbolDescription(usd) => {
            if let Some(sym) = symbol_idx.get_mut(usd.symbol_id) {
                sym.description = Some(usd.description);
            }
        }
        Op::UpdateMemoryContent(umc) => {
            let content_str = String::from_utf8(umc.content.clone()).unwrap_or_default();
            if let Some(payload) = payloads.get_mut(&umc.memory_id) {
                payload.content = umc.content;
                if !umc.embedding.is_empty() {
                    // semantic_idx (upserted below) is the embedding's single
                    // in-RAM home; drop any stale payload copy.
                    payload.embedding = Vec::new();
                }
            }
            if !umc.embedding.is_empty() {
                let umc_realm = payloads.get(&umc.memory_id).map(|p| p.realm.clone()).unwrap_or_default();
                semantic_idx.upsert(umc.memory_id, umc.embedding, Some(umc_realm.as_str()));
                // An UpdateMemoryContent with a real embedding means the backfill
                // completed. Clear embed_pending so replayed state matches live state.
                if let Some(st) = states.get_mut(&umc.memory_id) {
                    st.embed_pending = false;
                }
            }
            keyword_idx.index(umc.memory_id, &content_str);
        }
        Op::UpdateMemoryKind(umk) => {
            if let Some(payload) = payloads.get_mut(&umk.memory_id) {
                payload.kind = umk.new_kind;
            }
        }
        Op::RecordRecallBatch(op) => {
            let ctx = crate::state::RetrievalContext {
                centroid_q: op.centroid_q.clone(),
                scale: op.centroid_scale,
                context_hash: op.context_hash,
                ts_ms: op.ts_ms,
            };
            // Touch each memory and append retrieval context
            for &mid in &op.memory_ids {
                if let Some(state) = states.get_mut(&mid) {
                    state.access_count += 1;
                    state.last_accessed_ms = op.ts_ms;
                    state.retrieval_history.push(ctx.clone());
                }
            }
            // Update pairwise co-activation stats and strengthen edges
            let ids = &op.memory_ids;
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let key = (ids[i].min(ids[j]), ids[i].max(ids[j]));
                    let stats = coactivation_stats.entry(key).or_default();
                    stats.record(op.context_hash, op.ts_ms);
                    let multiplier = stats.hebbian_multiplier();
                    let delta = op.base_assoc_delta * multiplier;
                    strengthen_assoc_edge_map(assoc_edges, ids[i], ids[j], crate::ops::EdgeType::CoRetrieved, delta);
                }
            }
        }
        Op::StrengthenAssocEdge(op) => {
            strengthen_assoc_edge_map(assoc_edges, op.src, op.dst, op.edge_type, op.delta);
        }
        // ── Layer 8: Agent Protocol Memory ──────────────────────────────────
        // Layer 9: Wisdom Homeostasis

        // Consumed by the OrganApply dispatch above — unreachable by
        // construction. Listed explicitly (not `_`) so adding a new Op
        // variant still breaks this match at compile time until it gets a
        // handler or an organ.
        Op::StartIntervention(_) | Op::AddObservation(_) | Op::CloseIntervention(_)
        | Op::RecordAttribution(_) | Op::RegisterTask(_) | Op::UpdateTask(_)
        | Op::AddDelegation(_) | Op::LinkEvidence(_) | Op::AddProbe(_)
        | Op::ResolveProbe(_) | Op::SetCriterion(_) | Op::UpsertWisdomLineage(_)
        | Op::AdjudicateLineage(_) | Op::TransitionLineage(_)
        | Op::RecordChallenger(_) | Op::CloseRederive(_)
        | Op::RecordSurprise(_) | Op::RegisterDebt(_) | Op::UpdateDebt(_) | Op::AttachDebtEvidence(_) | Op::UpdateSourceWeight(_) | Op::RecordFeedback(_)
        | Op::UpdateSurpriseCredit(_) | Op::UpsertWisdomCandidate(_) | Op::UpdateWisdomLifecycle(_) | Op::UpdateScorerModel(_) | Op::MsgEvent(_) | Op::SkillUpload(_)
        | Op::SkillDeprecate(_) | Op::AgentUpsert(_) | Op::AgentDisable(_) | Op::AnalyticsEvent(_) | Op::AddTrigger(_) | Op::UpdateTrigger(_)
        | Op::FireTrigger(_) | Op::AssertConstraint(_) | Op::RetractConstraint(_) | Op::CreateBranch(_) | Op::ResolveBranch(_)
        | Op::TranscriptEvent(_) | Op::TaskEvent(_) | Op::UserModelEvent(_)
        | Op::ThemeEvent(_) | Op::SymbolEvent(_) => {
            debug_assert!(false, "organ-owned op leaked past OrganApply dispatch");
            eprintln!("[chitta-field] BUG: organ-owned op reached the central match");
        }
    }
}

/// Upsert an assoc edge: if one already exists between (src, dst, edge_type),
/// add `delta` to its weight. Otherwise insert a new edge with weight = `delta`.
/// Also maintains the reverse direction edge.
/// Upsert an assoc edge on a raw HashMap. Public so FFI can apply in-memory
/// without going through the full apply_op path.
pub fn strengthen_assoc_edge_map(
    assoc_edges: &mut HashMap<MemoryId, Vec<AssocEdge>>,
    src: MemoryId,
    dst: MemoryId,
    edge_type: EdgeType,
    delta: f32,
) {
    // Forward direction
    let edges = assoc_edges.entry(src).or_default();
    let mut found = false;
    for edge in edges.iter_mut() {
        if edge.dst == dst && std::mem::discriminant(&edge.edge_type) == std::mem::discriminant(&edge_type) {
            edge.weight = (edge.weight + delta).min(1.0);
            found = true;
            break;
        }
    }
    if !found {
        edges.push(AssocEdge {
            dst,
            edge_type: edge_type.clone(),
            weight: delta.clamp(0.0, 1.0),
        });
    }
    // Reverse direction
    let rev_edges = assoc_edges.entry(dst).or_default();
    let mut rev_found = false;
    for edge in rev_edges.iter_mut() {
        if edge.dst == src && std::mem::discriminant(&edge.edge_type) == std::mem::discriminant(&edge_type) {
            edge.weight = (edge.weight + delta).min(1.0);
            rev_found = true;
            break;
        }
    }
    if !rev_found {
        rev_edges.push(AssocEdge {
            dst: src,
            edge_type,
            weight: delta.clamp(0.0, 1.0),
        });
    }
}

fn build_realm_members(
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
) -> HashMap<String, HashSet<MemoryId>> {
    let mut realm_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
    for (&memory_id, payload) in payloads {
        let not_deleted = states.get(&memory_id).map(|s| !s.deleted).unwrap_or(false);
        if !not_deleted {
            continue;
        }
        realm_members
            .entry(payload.realm.clone())
            .or_default()
            .insert(memory_id);
    }
    realm_members
}

fn build_kind_members(
    payloads: &HashMap<MemoryId, MemoryPayload>,
    states: &HashMap<MemoryId, MemoryState>,
) -> HashMap<String, HashSet<MemoryId>> {
    let mut kind_members: HashMap<String, HashSet<MemoryId>> = HashMap::new();
    for (&memory_id, payload) in payloads {
        let not_deleted = states.get(&memory_id).map(|s| !s.deleted).unwrap_or(false);
        if !not_deleted {
            continue;
        }
        kind_members
            .entry(payload.kind.clone())
            .or_default()
            .insert(memory_id);
    }
    kind_members
}

/// Access batches are additive, deduplicated by WAL coverage, not the explicit
/// state-delta timestamp fence (many concurrent accesses share a millisecond).
/// They never advance that fence or move a newer explicit access backwards.
pub(crate) fn apply_access_delta(state: &mut MemoryState, delta: &crate::ops::StateDeltaOp) {
    state.access_count = state.access_count.saturating_add(1);
    if delta.op_ts_ms >= state.last_accessed_ms && delta.op_ts_ms >= state.last_state_op_ts_ms {
        if let Some(rate) = delta.decay_rate { state.decay_rate = rate.max(0.0); }
    }
    state.last_accessed_ms = state.last_accessed_ms.max(delta.op_ts_ms);
    state.access_timestamps.push(delta.op_ts_ms);
    state.access_timestamps.sort_unstable();
    if state.access_timestamps.len() > 16 {
        state.access_timestamps.drain(..state.access_timestamps.len() - 16);
    }
}
