use crate::error::{FieldError, Result};
use rayon::prelude::*;
use crate::field::{AssocEdge, CoActivationStats};
use crate::hnsw::SemanticIndex;
use crate::ids::{ArtifactId, ChunkHash, MemoryId};
use crate::organ::artifact::ArtifactIndex;
use crate::organ::callgraph::CallGraph;
use crate::organ::codefile::CodeFileIndex;
use crate::organ::keyword::KeywordIndex;
use crate::organ::symbol::SymbolIndex;
use crate::organ::temporal::TemporalIndex;
use crate::organ::event_tape::EventTape;
use crate::organ::decision_tape::DecisionTape;
use crate::organ::turiya_monitor::TuriyaMonitor;
use crate::organ::observer::ObserverState;
use crate::organ::interaction_ledger::InteractionLedger;
use crate::organ::predicate_store::PredicateStore;
use crate::organ::triplet::{CorrectionState, TripletStore};
use crate::payload::MemoryPayload;
use crate::state::{EpistemicStatus, MemoryState, MemoryStatus, RetrievalHistory};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic for snapshots written before the ANN SemanticIndex rewrite (v1.0.3 and earlier).
const FULL_SNAPSHOT_MAGIC_V1: u64 = 0xF011_5741_7E00_0003;
/// Magic for v1.0.4 snapshots (post-ANN, old MemoryState, no coactivation_stats).
const FULL_SNAPSHOT_MAGIC_V4: u64 = 0xF011_5741_7E00_0004;
/// Magic for v1.0.5 snapshots (with coactivation_stats, but 14-field MemoryState).
/// These were written before RetrievalHistory/embed_pending/MemoryStatus/EpistemicStatus
/// were added to MemoryState on 2026-03-26.
const FULL_SNAPSHOT_MAGIC_V5: u64 = 0xF011_5741_7E00_0005;
/// Magic for v1.0.6 snapshots: full MemoryState with RetrievalHistory, embed_pending,
/// MemoryStatus, EpistemicStatus, surprise, affect — but NO access_timestamps.
const FULL_SNAPSHOT_MAGIC_V6: u64 = 0xF011_5741_7E00_0006;
/// Magic for v1.0.7 snapshots: adds access_timestamps but NO interference fields.
const FULL_SNAPSHOT_MAGIC_V7: u64 = 0xF011_5741_7E00_0007;
/// Magic for v1.0.8 snapshots: adds competitive_weight, lure_risk, spacing_quality.
/// FullSnapshot had no top-level sidecars (ack_scores / correction_states).
const FULL_SNAPSHOT_MAGIC_V8: u64 = 0xF011_5741_7E00_0008;
/// v1.0.9: FullSnapshot with ack_scores + correction_states; embeddings in bincode.
const FULL_SNAPSHOT_MAGIC_V9: u64 = 0xF011_5741_7E00_0009;
/// v1.0.10: embeddings moved to .emb sidecar; content still in bincode.
const FULL_SNAPSHOT_MAGIC_V10: u64 = 0xF011_5741_7E00_000A;
/// v1.0.11: content moved to .pld sidecar; bincode field is empty Vec.
/// MemoryState did NOT yet have `staged`/`invalidated_by`.
const FULL_SNAPSHOT_MAGIC_V11: u64 = 0xF011_5741_7E00_000B;
/// V12: MemoryState gains `staged` + `invalidated_by`. MemoryPayload unchanged.
const FULL_SNAPSHOT_MAGIC_V12: u64 = 0xF011_5741_7E00_000C;
/// V13: MemoryPayload gains `harness: Option<String>`.
const FULL_SNAPSHOT_MAGIC_V13: u64 = 0xF011_5741_7E00_000D;
/// V14: FullSnapshot gains `event_tape: EventTape` (CEC Phase 1).
const FULL_SNAPSHOT_MAGIC_V14: u64 = 0xF011_5741_7E00_000E;
/// V15: FullSnapshot gains `decision_tape: DecisionTape` (CEC Phase 10).
const FULL_SNAPSHOT_MAGIC_V15: u64 = 0xF011_5741_7E00_000F;
/// V16 magic (v5.35–5.41): FullSnapshot gains `turiya_monitor: TuriyaMonitor` (CEC Phase 11).
const FULL_SNAPSHOT_MAGIC_V16: u64 = 0xF011_5741_7E00_0010;
/// V17 magic: MemoryPayload gains `provenance: String` + `candidate: bool` (CEC Phase 17).
const FULL_SNAPSHOT_MAGIC_V17: u64 = 0xF011_5741_7E00_0011;
/// V18 magic: FullSnapshot gains `observer_state: ObserverState` (Tier-0 observation extraction).
pub const FULL_SNAPSHOT_MAGIC_V18: u64 = 0xF011_5741_7E00_0012;
/// V19 magic: MemoryPayload gains `embedding_model_id` + `embedding_dim`; EMBED_DIM 256→768.
const FULL_SNAPSHOT_MAGIC_V19: u64     = 0xF011_5741_7E00_0013;
/// V20 magic: FullSnapshot gains `interaction_ledger: InteractionLedger`.
const FULL_SNAPSHOT_MAGIC_V20: u64     = 0xF011_5741_7E00_0014;
/// V21 magic: FullSnapshot gains `predicate_store: PredicateStore`.
const FULL_SNAPSHOT_MAGIC_V21: u64     = 0xF011_5741_7E00_0015;
/// V22 magic: FullSnapshot gains `cw_refresh_ts` — persisted competitive-weight
/// refresh timestamps, so a daemon restart does not reset every memory to
/// "never refreshed" (the restart refresh-herd trigger).
const FULL_SNAPSHOT_MAGIC_V22: u64     = 0xF011_5741_7E00_0016;
/// V23 magic: sectioned container. Layout after the magic:
///   [seqno: u64 LE]
///   then sections until EOF: [name_len: u16 LE][name: utf8]
///                            [body_len: u64 LE][body: bincode of that field]
/// One section per FullSnapshot field. Unknown sections are skipped, missing
/// sections keep their defaults — ADDING a top-level field no longer needs a
/// magic bump, a Legacy* struct, or a migration arm: write the section in
/// save(), match it in the V23 loader, done. (Changing the INTERNAL layout of
/// an existing section still needs versioning — bodies remain positional
/// bincode of that one type.) The bincode ladder is frozen at V22.
const FULL_SNAPSHOT_MAGIC: u64         = 0xF011_5741_7E00_0017; // V23

/// Magic for the payload content sidecar (.pld).
const PLD_MAGIC: u64 = 0x504C_4400_0000_0001; // "PLD\0\0\0\0\x01"
// Retrieval-surface sidecar (.rsf): id -> natural-language embed surface. Stored
// OUTSIDE the bincode Snapshot (like .pld/.emb) so adding it never breaks old
// snapshot reads. Missing sidecar = no surface = embed falls back to content.
const RSF_MAGIC: u64 = 0x5253_4600_0000_0001; // "RSF\0\0\0\0\x01"

/// Magic for the store-identity sidecar (.shdr).
const SHDR_MAGIC: u64 = 0x5348_4452_0000_0001; // "SHDR\0\0\0\x01"
/// On-disk store format version (independent of the snapshot bincode magic).
pub const STORE_FORMAT_VERSION: u32 = 1;

/// Store identity recorded alongside each snapshot family in a `.shdr` sidecar.
///
/// Purpose: make snapshot selection and WAL replay aware of the *vector space* a store
/// was written in, so a foreign-dim/foreign-model snapshot can never win on max-seqno
/// (the root cause of the 768→1536 migration contamination). `vector_space_id` is a
/// stable hash of (model_id, embed_dim, text_format_version) — deterministic from the
/// compiled constants, so it can be computed without loading anything.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoreHeader {
    pub format_version: u32,
    pub embed_dim: u32,
    pub model_id: String,
    pub text_format_version: u32,
    /// Monotonic per-write-session counter; bumped each time the store is opened for
    /// writing. Lets a reader detect a newer writer (PR4 single-writer hygiene).
    pub lineage_epoch: u64,
    /// Identity of the writer that first stamped this lineage (entropy from instance_id
    /// + open time). Stable across the lineage; carried forward on each save.
    pub writer_uuid: u128,
    /// Stable hash of (model_id, embed_dim, text_format_version).
    pub vector_space_id: u64,
}

impl StoreHeader {
    /// FNV-1a over model_id + embed_dim + text_format_version. Stable across processes.
    pub fn compute_vector_space_id(model_id: &str, embed_dim: u32, text_format_version: u32) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |b: u8| { h ^= b as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); };
        for b in model_id.bytes() { mix(b); }
        for b in embed_dim.to_le_bytes() { mix(b); }
        for b in text_format_version.to_le_bytes() { mix(b); }
        h
    }

    /// The vector_space_id this binary compiles for.
    pub fn compiled_vector_space_id() -> u64 {
        Self::compute_vector_space_id(
            crate::ops::EMBED_MODEL_ID,
            crate::ops::EMBED_DIM as u32,
            crate::ops::TEXT_FORMAT_VERSION,
        )
    }

    /// Build a header describing the compiled vector space with the given lineage.
    pub fn current(lineage_epoch: u64, writer_uuid: u128) -> Self {
        let model_id = crate::ops::EMBED_MODEL_ID.to_string();
        let embed_dim = crate::ops::EMBED_DIM as u32;
        let text_format_version = crate::ops::TEXT_FORMAT_VERSION;
        StoreHeader {
            format_version: STORE_FORMAT_VERSION,
            embed_dim,
            vector_space_id: Self::compute_vector_space_id(&model_id, embed_dim, text_format_version),
            model_id,
            text_format_version,
            lineage_epoch,
            writer_uuid,
        }
    }

    /// True when this header's vector space matches the binary's compiled constants.
    pub fn matches_compiled(&self) -> bool {
        self.embed_dim == crate::ops::EMBED_DIM as u32
            && self.model_id == crate::ops::EMBED_MODEL_ID
            && self.vector_space_id == Self::compiled_vector_space_id()
    }

    /// Persist to `path` (magic + bincode body) via tmp+rename+fsync.
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("shdr.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&SHDR_MAGIC.to_le_bytes())?;
            let body = bincode::serialize(self).map_err(|e| FieldError::Serialization(e.to_string()))?;
            f.write_all(&body)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) { let _ = d.sync_all(); }
        }
        Ok(())
    }

    /// Load from `path`. Returns None if missing/corrupt/wrong-magic (legacy store).
    pub fn load(path: &Path) -> Option<StoreHeader> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() < 8 { return None; }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        if magic != SHDR_MAGIC { return None; }
        bincode::deserialize(&bytes[8..]).ok()
    }
}

// ── Compile-time guard: MemoryState size must match snapshot format ──────────
// Since V23 the snapshot is a sectioned container (see FULL_SNAPSHOT_MAGIC):
//   * ADDING a top-level FullSnapshot field = add a write_section line in
//     save() + a match arm in the V23 loader + a default in empty(). No magic
//     bump, no Legacy* struct, no migration arm.
//   * CHANGING the internal layout of a section's type (e.g. MemoryState
//     inside "states") still breaks positional bincode WITHIN that section.
//     If you add/remove MemoryState fields you MUST write the new data under a
//     new section name (e.g. "states_v2") with a fallback reader for the old
//     one — or persist the new data as its own section keyed by MemoryId,
//     like cw_refresh_ts does.
// bincode is positional — #[serde(default)] does NOT work with it.
// V12: added `staged: bool` + `invalidated_by: Option<String>`. Exact size depends on
// compiler field-reordering (repr(Rust) may pack bools together). Guard is a range check.
const _: () = assert!(
    std::mem::size_of::<MemoryState>() >= 200,
    "MemoryState shrank below V9 baseline — check for accidental field removal"
);

// ── Legacy SemanticIndex (pre-ANN rewrite) ───────────────────────────────────

#[derive(Serialize, Deserialize)]
struct LegacySemanticIndex {
    embeddings: std::collections::HashMap<MemoryId, Vec<f32>>,
    deleted: std::collections::HashSet<MemoryId>,
}

// ── Legacy MemoryState (V4 and V5 snapshots, 14-field layout) ────────────────

/// MemoryPayload as serialized in V1–V12 snapshots (pre-Phase-4).
/// Lacks `harness: Option<String>`.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryPayloadV12 {
    pub memory_id: MemoryId,
    pub version: u32,
    pub chunk_hash: ChunkHash,
    pub created_at_ms: i64,
    pub authored_at_ms: i64,
    pub kind: String,
    pub realm: String,
    pub content: Vec<u8>,
    pub embedding_model: String,
    pub embedding: Vec<f32>,
    pub artifact_refs: Vec<crate::ops::ArtifactRef>,
    pub source_session: Option<String>,
    pub source_tool: Option<String>,
}
impl LegacyMemoryPayloadV12 {
    fn upgrade(self) -> MemoryPayload {
        let harness = self.source_tool.as_deref().map(|t| {
            if t.starts_with("codex") { "codex".to_string() } else { "claude-code".to_string() }
        });
        MemoryPayload {
            memory_id: self.memory_id,
            version: self.version,
            chunk_hash: self.chunk_hash,
            created_at_ms: self.created_at_ms,
            authored_at_ms: self.authored_at_ms,
            kind: self.kind,
            realm: self.realm,
            content: self.content,
            embedding_model: self.embedding_model,
            embedding: self.embedding,
            artifact_refs: self.artifact_refs,
            source_session: self.source_session,
            source_tool: self.source_tool,
            harness,
            provenance: "human".to_string(),
            candidate: false,
            embedding_model_id: "legacy-256".to_string(),
            embedding_dim: 256,
        }
    }
}

fn upgrade_payloads(m: std::collections::HashMap<MemoryId, LegacyMemoryPayloadV12>) -> std::collections::HashMap<MemoryId, MemoryPayload> {
    m.into_iter().map(|(id, p)| (id, p.upgrade())).collect()
}

/// MemoryState as serialized in V4 snapshots (pre-CTM, 14 fields).
/// Lacks RetrievalHistory, embed_pending, MemoryStatus, and EpistemicStatus.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV4 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
}

impl LegacyMemoryStateV4 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: RetrievalHistory::default(),
            embed_pending: false,
            status: MemoryStatus::Active,
            epistemic_status: EpistemicStatus::ToolDerived,
            surprise: 0.0,
            affect_valence: 0.0,
            affect_arousal: 0.0,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
            last_cw_refresh_ms: 0,
            utility_alpha: 1.0,
            utility_beta: 1.0,
            replication_count: 1,
        }
    }
}

/// MemoryState as serialized in V5 snapshots (15 fields: added RetrievalHistory).
/// Written by binaries from 2026-03-17 (commit c61a955) through 2026-03-25.
/// Lacks embed_pending, MemoryStatus, and EpistemicStatus.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV5 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
}

impl LegacyMemoryStateV5 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: false,
            status: MemoryStatus::Active,
            epistemic_status: EpistemicStatus::ToolDerived,
            surprise: 0.0,
            affect_valence: 0.0,
            affect_arousal: 0.0,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
            last_cw_refresh_ms: 0,
            utility_alpha: 1.0,
            utility_beta: 1.0,
            replication_count: 1,
        }
    }
}

/// MemoryState as serialized in V6 snapshots (added affect + surprise, no access_timestamps).
/// Written by v5.10.0–v5.10.1 (2026-04-04 through 2026-04-10).
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV6 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
    pub embed_pending: bool,
    pub status: MemoryStatus,
    pub epistemic_status: EpistemicStatus,
    pub surprise: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
}

impl LegacyMemoryStateV6 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: self.embed_pending,
            status: self.status,
            epistemic_status: self.epistemic_status,
            surprise: self.surprise,
            affect_valence: self.affect_valence,
            affect_arousal: self.affect_arousal,
            access_timestamps: Vec::new(),
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
            last_cw_refresh_ms: 0,
            utility_alpha: 1.0,
            utility_beta: 1.0,
            replication_count: 1,
        }
    }
}

/// MemoryState as serialized in V7 snapshots (added access_timestamps, no interference fields).
/// Written by v5.11.0–v5.11.2 (2026-04-10).
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV7 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
    pub embed_pending: bool,
    pub status: MemoryStatus,
    pub epistemic_status: EpistemicStatus,
    pub surprise: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
    pub access_timestamps: Vec<i64>,
}

impl LegacyMemoryStateV7 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: self.embed_pending,
            status: self.status,
            epistemic_status: self.epistemic_status,
            surprise: self.surprise,
            affect_valence: self.affect_valence,
            affect_arousal: self.affect_arousal,
            access_timestamps: self.access_timestamps,
            competitive_weight: 0.0,
            lure_risk: 0.0,
            spacing_quality: 0.0,
            staged: false,
            invalidated_by: None,
            last_cw_refresh_ms: 0,
            utility_alpha: 1.0,
            utility_beta: 1.0,
            replication_count: 1,
        }
    }
}

// ── Legacy snapshot structs ───────────────────────────────────────────────────

/// V1 snapshot: pre-ANN SemanticIndex + 14-field MemoryState.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV1 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV4>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: LegacySemanticIndex,
}

/// V4 snapshot: SemanticIndex (ANN), 14-field MemoryState, no coactivation_stats.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV4 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV4>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
}

/// V5 snapshot: SemanticIndex (ANN), 15-field MemoryState (with RetrievalHistory), with coactivation_stats.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV5 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV5>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

/// V6 snapshot: full MemoryState with affect/surprise but no access_timestamps.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV6 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV6>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

/// V7 snapshot: full MemoryState with access_timestamps but no interference fields.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV7 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV7>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

// ── V8 snapshot struct (no top-level sidecars) ───────────────────────────────

/// MemoryState as it existed before Phase 5 write-gate (v1.0.12).
/// Identical to current MemoryState minus `staged` and `invalidated_by`.
/// Used to deserialise V8–V11 snapshots written by older binaries.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryStateV11 {
    pub memory_id: MemoryId,
    pub current_version: u32,
    pub current_chunk_hash: crate::ids::ChunkHash,
    pub deleted: bool,
    pub strength: f32,
    pub decay_rate: f32,
    pub confidence: f32,
    pub access_count: u32,
    pub last_accessed_ms: i64,
    pub last_strengthened_ms: i64,
    pub created_at_ms: i64,
    pub pinned: bool,
    pub tier: u8,
    pub last_state_op_ts_ms: i64,
    pub retrieval_history: RetrievalHistory,
    pub embed_pending: bool,
    pub status: MemoryStatus,
    pub epistemic_status: EpistemicStatus,
    pub surprise: f32,
    pub affect_valence: f32,
    pub affect_arousal: f32,
    pub access_timestamps: Vec<i64>,
    pub competitive_weight: f32,
    pub lure_risk: f32,
    pub spacing_quality: f32,
}

impl LegacyMemoryStateV11 {
    fn upgrade(self) -> MemoryState {
        MemoryState {
            memory_id: self.memory_id,
            current_version: self.current_version,
            current_chunk_hash: self.current_chunk_hash,
            deleted: self.deleted,
            strength: self.strength,
            decay_rate: self.decay_rate,
            confidence: self.confidence,
            access_count: self.access_count,
            last_accessed_ms: self.last_accessed_ms,
            last_strengthened_ms: self.last_strengthened_ms,
            created_at_ms: self.created_at_ms,
            pinned: self.pinned,
            tier: self.tier,
            last_state_op_ts_ms: self.last_state_op_ts_ms,
            retrieval_history: self.retrieval_history,
            embed_pending: self.embed_pending,
            status: self.status,
            epistemic_status: self.epistemic_status,
            surprise: self.surprise,
            affect_valence: self.affect_valence,
            affect_arousal: self.affect_arousal,
            access_timestamps: self.access_timestamps,
            competitive_weight: self.competitive_weight,
            lure_risk: self.lure_risk,
            spacing_quality: self.spacing_quality,
            staged: false,
            invalidated_by: None,
            last_cw_refresh_ms: 0,
            utility_alpha: 1.0,
            utility_beta: 1.0,
            replication_count: 1,
        }
    }
}

/// V11 snapshot layout (magic 0xF011_5741_7E00_000B): same sidecar layout as
/// current but MemoryState lacks `staged`/`invalidated_by`.
/// Also used for V9 and V10 which share the same MemoryState binary layout.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV11 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV11>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
}

#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV8 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, LegacyMemoryStateV11>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
}

// ── Current snapshot struct (v9+) ────────────────────────────────────────────

/// V12 snapshot layout: same sidecar layout as V11; MemoryState has staged+invalidated_by;
/// MemoryPayload lacks `harness`.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV12 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, LegacyMemoryPayloadV12>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
}

#[derive(Serialize, Deserialize)]
pub struct FullSnapshot {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    /// Persistent ack/nack usage scores keyed by MemoryId.
    pub ack_scores: HashMap<MemoryId, i32>,
    /// Persistent correction lifecycle states keyed by triplet id.
    pub correction_states: HashMap<u64, CorrectionState>,
    /// CEC Phase 1: append-only structured event tape for CDAWG reconstruction.
    pub event_tape: EventTape,
    /// CEC Phase 10: branch-point memory — chosen actions and rejected alternatives.
    pub decision_tape: DecisionTape,
    /// CEC Phase 11: Turīya Monitor — rolling health samples across CEC organs.
    pub turiya_monitor: TuriyaMonitor,
    /// Tier-0 observation extraction state — structured facts extracted from conversational turns.
    pub observer_state: ObserverState,
    /// V20: Interaction Ledger — reads/writes/outcomes as first-class events + versioned assertions.
    pub interaction_ledger: InteractionLedger,
    /// V21: Predicate Store — executable checks attached to memories for falsifiability.
    pub predicate_store: PredicateStore,
    /// V22: per-memory competitive-weight refresh timestamps. Persisted so a
    /// restart does not make every memory look never-refreshed (MemoryState's
    /// `last_cw_refresh_ms` itself stays `#[serde(skip)]` to keep the positional
    /// bincode layout of all V12+ snapshots stable). Hydrated into states on load.
    pub cw_refresh_ts: HashMap<MemoryId, i64>,
    /// V23 section "recall_provenance": memory → distinct recalling instance
    /// ids (cross-context generality evidence; THEORY.md §6). Added with NO
    /// format migration — the sectioned container defaults missing sections.
    pub recall_provenance: HashMap<MemoryId, std::collections::BTreeSet<u32>>,
    /// V23 section "utility_posteriors": memory → (utility_alpha, utility_beta).
    /// Carried outside `states` because MemoryState's positional bincode layout
    /// is frozen; hydrated into states on load. A missing section leaves every
    /// posterior at the (1, 1) prior.
    pub utility_posteriors: HashMap<MemoryId, (f32, f32)>,
    /// V23 optional section: ledger transactions and native session lifecycle.
    /// MsgEvent/SessionEvent WAL opcodes and old positional formats stay frozen.
    pub ledger_session_events: Vec<crate::organ::msg::MsgEvent>,
}

/// V22 snapshot layout, frozen: the last monolithic-bincode format. Positional
/// bincode of this exact field list. Do NOT change this struct — V23+ uses the
/// sectioned container and FullSnapshot itself is free to grow.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV22 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
    pub event_tape: EventTape,
    pub decision_tape: DecisionTape,
    pub turiya_monitor: TuriyaMonitor,
    pub observer_state: ObserverState,
    pub interaction_ledger: InteractionLedger,
    pub predicate_store: PredicateStore,
    pub cw_refresh_ts: HashMap<MemoryId, i64>,
}

impl LegacyFullSnapshotV22 {
    fn upgrade(self) -> FullSnapshot {
        FullSnapshot {
            snapshot_seqno:     self.snapshot_seqno,
            payloads:           self.payloads,
            states:             self.states,
            assoc_edges:        self.assoc_edges,
            artifacts:          self.artifacts,
            artifact_paths:     self.artifact_paths,
            time_idx:           self.time_idx,
            keyword_idx:        self.keyword_idx,
            artifact_idx:       self.artifact_idx,
            triplet_store:      self.triplet_store,
            symbol_idx:         self.symbol_idx,
            call_graph:         self.call_graph,
            code_files:         self.code_files,
            semantic_idx:       self.semantic_idx,
            coactivation_stats: self.coactivation_stats,
            ack_scores:         self.ack_scores,
            correction_states:  self.correction_states,
            event_tape:         self.event_tape,
            decision_tape:      self.decision_tape,
            turiya_monitor:     self.turiya_monitor,
            observer_state:     self.observer_state,
            interaction_ledger: self.interaction_ledger,
            predicate_store:    self.predicate_store,
            cw_refresh_ts:      self.cw_refresh_ts,
            recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
        }
    }
}

/// V21 snapshot layout: identical to FullSnapshot minus `cw_refresh_ts`.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV21 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
    pub event_tape: EventTape,
    pub decision_tape: DecisionTape,
    pub turiya_monitor: TuriyaMonitor,
    pub observer_state: ObserverState,
    pub interaction_ledger: InteractionLedger,
    pub predicate_store: PredicateStore,
}

/// MemoryPayload as serialized in V16 snapshots (pre-Phase-17).
/// Lacks `provenance: String` and `candidate: bool`.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryPayloadV16 {
    pub memory_id:       crate::ids::MemoryId,
    pub version:         u32,
    pub chunk_hash:      crate::ids::ChunkHash,
    pub created_at_ms:   i64,
    pub authored_at_ms:  i64,
    pub kind:            String,
    pub realm:           String,
    pub content:         Vec<u8>,
    pub embedding_model: String,
    pub embedding:       Vec<f32>,
    pub artifact_refs:   Vec<crate::ops::ArtifactRef>,
    pub source_session:  Option<String>,
    pub source_tool:     Option<String>,
    pub harness:         Option<String>,
}
impl LegacyMemoryPayloadV16 {
    fn upgrade(self) -> crate::payload::MemoryPayload {
        crate::payload::MemoryPayload {
            memory_id:       self.memory_id,
            version:         self.version,
            chunk_hash:      self.chunk_hash,
            created_at_ms:   self.created_at_ms,
            authored_at_ms:  self.authored_at_ms,
            kind:            self.kind,
            realm:           self.realm,
            content:         self.content,
            embedding_model: self.embedding_model,
            embedding:       self.embedding,
            artifact_refs:   self.artifact_refs,
            source_session:  self.source_session,
            source_tool:     self.source_tool,
            harness:         self.harness,
            provenance:          "human".to_string(),
            candidate:           false,
            embedding_model_id:  "legacy-256".to_string(),
            embedding_dim:       256,
        }
    }
}

fn upgrade_payloads_v16(
    m: std::collections::HashMap<crate::ids::MemoryId, LegacyMemoryPayloadV16>,
) -> std::collections::HashMap<crate::ids::MemoryId, crate::payload::MemoryPayload> {
    m.into_iter().map(|(id, p)| (id, p.upgrade())).collect()
}

/// MemoryPayload as serialized in V18 snapshots (pre-V19).
/// Lacks `embedding_model_id: String` and `embedding_dim: u32`.
#[derive(Serialize, Deserialize)]
struct LegacyMemoryPayloadV18 {
    pub memory_id:       crate::ids::MemoryId,
    pub version:         u32,
    pub chunk_hash:      crate::ids::ChunkHash,
    pub created_at_ms:   i64,
    pub authored_at_ms:  i64,
    pub kind:            String,
    pub realm:           String,
    pub content:         Vec<u8>,
    pub embedding_model: String,
    pub embedding:       Vec<f32>,
    pub artifact_refs:   Vec<crate::ops::ArtifactRef>,
    pub source_session:  Option<String>,
    pub source_tool:     Option<String>,
    pub harness:         Option<String>,
    pub provenance:      String,
    pub candidate:       bool,
}
impl LegacyMemoryPayloadV18 {
    fn upgrade(self) -> crate::payload::MemoryPayload {
        crate::payload::MemoryPayload {
            memory_id:          self.memory_id,
            version:            self.version,
            chunk_hash:         self.chunk_hash,
            created_at_ms:      self.created_at_ms,
            authored_at_ms:     self.authored_at_ms,
            kind:               self.kind,
            realm:              self.realm,
            content:            self.content,
            embedding_model:    self.embedding_model,
            embedding:          vec![],   // cleared — stale 256-d vectors must not enter 768-d HNSW
            artifact_refs:      self.artifact_refs,
            source_session:     self.source_session,
            source_tool:        self.source_tool,
            harness:            self.harness,
            provenance:         self.provenance,
            candidate:          self.candidate,
            embedding_model_id: "legacy-256".to_string(),
            embedding_dim:      256,
        }
    }
}

fn upgrade_payloads_v18(
    m: std::collections::HashMap<crate::ids::MemoryId, LegacyMemoryPayloadV18>,
) -> std::collections::HashMap<crate::ids::MemoryId, crate::payload::MemoryPayload> {
    m.into_iter().map(|(id, p)| (id, p.upgrade())).collect()
}

/// V16 snapshot layout — FullSnapshot without provenance/candidate on MemoryPayload.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV16 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, LegacyMemoryPayloadV16>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
}

/// V18 snapshot layout — FullSnapshot with observer_state but without embedding_model_id/embedding_dim on payloads.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV18 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, LegacyMemoryPayloadV18>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
    pub observer_state:      ObserverState,
}

/// V19 snapshot layout — FullSnapshot without interaction_ledger.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV19 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, MemoryPayload>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
    pub observer_state:      ObserverState,
}

/// V20 snapshot layout — FullSnapshot with interaction_ledger but without predicate_store.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV20 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, MemoryPayload>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
    pub observer_state:      ObserverState,
    pub interaction_ledger:  InteractionLedger,
}

/// V17 snapshot layout — FullSnapshot without observer_state.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV17 {
    pub snapshot_seqno:      u64,
    pub payloads:            HashMap<MemoryId, MemoryPayload>,
    pub states:              HashMap<MemoryId, MemoryState>,
    pub assoc_edges:         HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts:           HashMap<String, ArtifactId>,
    pub artifact_paths:      HashMap<ArtifactId, String>,
    pub time_idx:            TemporalIndex,
    pub keyword_idx:         KeywordIndex,
    pub artifact_idx:        ArtifactIndex,
    pub triplet_store:       TripletStore,
    pub symbol_idx:          SymbolIndex,
    pub call_graph:          CallGraph,
    pub code_files:          CodeFileIndex,
    pub semantic_idx:        SemanticIndex,
    pub coactivation_stats:  HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores:          HashMap<MemoryId, i32>,
    pub correction_states:   HashMap<u64, CorrectionState>,
    pub event_tape:          EventTape,
    pub decision_tape:       DecisionTape,
    pub turiya_monitor:      TuriyaMonitor,
}

/// V15 snapshot layout — FullSnapshot without turiya_monitor.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV15 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
    pub event_tape: EventTape,
    pub decision_tape: DecisionTape,
}

/// V14 snapshot layout — FullSnapshot without decision_tape.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV14 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
    pub event_tape: EventTape,
}

/// V13 snapshot layout — FullSnapshot without event_tape.
#[derive(Serialize, Deserialize)]
struct LegacyFullSnapshotV13 {
    pub snapshot_seqno: u64,
    pub payloads: HashMap<MemoryId, MemoryPayload>,
    pub states: HashMap<MemoryId, MemoryState>,
    pub assoc_edges: HashMap<MemoryId, Vec<AssocEdge>>,
    pub artifacts: HashMap<String, ArtifactId>,
    pub artifact_paths: HashMap<ArtifactId, String>,
    pub time_idx: TemporalIndex,
    pub keyword_idx: KeywordIndex,
    pub artifact_idx: ArtifactIndex,
    pub triplet_store: TripletStore,
    pub symbol_idx: SymbolIndex,
    pub call_graph: CallGraph,
    pub code_files: CodeFileIndex,
    pub semantic_idx: SemanticIndex,
    pub coactivation_stats: HashMap<(MemoryId, MemoryId), CoActivationStats>,
    pub ack_scores: HashMap<MemoryId, i32>,
    pub correction_states: HashMap<u64, CorrectionState>,
}

/// Write one V23 section: [name_len u16][name][body_len u64][bincode body].
fn write_section<W: Write, T: serde::Serialize>(w: &mut W, name: &str, value: &T) -> Result<()> {
    let body = bincode::serialize(value)
        .map_err(|e| FieldError::Serialization(format!("section '{name}': {e}")))?;
    w.write_all(&(name.len() as u16).to_le_bytes())?;
    w.write_all(name.as_bytes())?;
    w.write_all(&(body.len() as u64).to_le_bytes())?;
    w.write_all(&body)?;
    Ok(())
}

/// Measures only buffered file refills; section timers include decode and allocation.
struct SnapshotReader {
    file: std::fs::File,
    profile: bool,
    elapsed: std::time::Duration,
    bytes: usize,
}
impl SnapshotReader {
    fn new(file: std::fs::File) -> Self {
        Self { file, profile: std::env::var_os("CHITTA_PROFILE_SNAPSHOT").is_some(),
            elapsed: std::time::Duration::ZERO, bytes: 0 }
    }
}
impl Read for SnapshotReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if !self.profile { return self.file.read(buf); }
        let start = std::time::Instant::now();
        let result = self.file.read(buf);
        self.elapsed += start.elapsed();
        self.bytes += result.as_ref().copied().unwrap_or(0);
        result
    }
}
impl Drop for SnapshotReader {
    fn drop(&mut self) {
        if self.profile {
            eprintln!("[chitta-field] snapshot io_ms={} bytes={}", self.elapsed.as_millis(), self.bytes);
        }
    }
}

/// Historical streaming decoder, retained as the compatibility test oracle.
#[cfg(test)]
fn read_section<R: Read, T: serde::de::DeserializeOwned>(r: &mut R, name: &str) -> Result<T> {
    let start = std::time::Instant::now();
    let result = bincode::deserialize_from(r)
        .map_err(|e| FieldError::Serialization(format!("section '{name}': {e}")));
    if std::env::var_os("CHITTA_PROFILE_SNAPSHOT").is_some() {
        eprintln!("[chitta-field] snapshot section={name} decode_ms={}", start.elapsed().as_millis());
    }
    result
}

/// Decode directly from the resident bytes (no per-string reader scratch copy).
/// On malformed input, replay the historical reader decoder for its exact error
/// text and category, including bincode's reader-versus-slice EOF distinction.
fn decode_body<T: serde::de::DeserializeOwned>(bytes: &[u8], name: &str) -> Result<T> {
    let _phase = crate::profile::SnapshotPhase::new(name);
    bincode::deserialize(bytes).or_else(|_| bincode::deserialize_from(bytes))
        .map_err(|e| FieldError::Serialization(format!("section '{name}': {e}")))
}

// Limit startup concurrency independently of the host's (often hundreds of) CPUs.
fn decode_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(4)
        .thread_name(|i| format!("snapshot-{i}")).build().expect("snapshot decode pool"))
}

/// Serde caps its default HashMap reservation to 1 MiB. The section byte bound
/// lets these large, trusted-length maps reserve once without trusting a corrupt
/// count alone. Entries with zero-sized encodings still grow normally if needed.
fn read_map_section<K, V>(bytes: &[u8], name: &str) -> Result<HashMap<K, V>>
where K: serde::de::DeserializeOwned + Eq + std::hash::Hash,
      V: serde::de::DeserializeOwned,
{
    use bincode::Options;
    use serde::de::{DeserializeSeed, MapAccess, Visitor};
    struct MapSeed<K, V>(usize, std::marker::PhantomData<(K, V)>);
    impl<'de, K, V> Visitor<'de> for MapSeed<K, V>
    where K: Deserialize<'de> + Eq + std::hash::Hash, V: Deserialize<'de> {
        type Value = HashMap<K, V>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a map")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> std::result::Result<Self::Value, A::Error> {
            let mut map = HashMap::with_capacity(entries.size_hint().unwrap_or(0).min(self.0));
            while let Some((key, value)) = entries.next_entry()? { map.insert(key, value); }
            Ok(map)
        }
    }
    impl<'de, K, V> DeserializeSeed<'de> for MapSeed<K, V>
    where K: Deserialize<'de> + Eq + std::hash::Hash, V: Deserialize<'de> {
        type Value = HashMap<K, V>;
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
            d.deserialize_map(self)
        }
    }
    let _phase = crate::profile::SnapshotPhase::new(name);
    let options = bincode::DefaultOptions::new().with_fixint_encoding().allow_trailing_bytes();
    let mut decoder = bincode::Deserializer::from_slice(bytes, options);
    MapSeed(bytes.len() / 8, std::marker::PhantomData).deserialize(&mut decoder)
        .or_else(|_| bincode::deserialize_from(bytes))
        .map_err(|e| FieldError::Serialization(format!("section '{name}': {e}")))
}

struct Section<'a> {
    name: String,
    body: &'a [u8],
    declared_len: u64,
}

type ApplySection = Box<dyn FnOnce(&mut FullSnapshot) + Send>;

/// Own the immutable mapping and pending results across independently scheduled
/// decode batches. Publication remains in file order, including errors and
/// duplicate sections; a later batch must never replace a WAL-updated section.
struct MappedSnapshot {
    bytes: memmap2::Mmap,
    sections: Vec<(String, std::ops::Range<usize>, u64)>,
    decoded: Vec<Option<Result<ApplySection>>>,
    table_result: Result<()>,
    seqno: u64,
}

impl MappedSnapshot {
    // The caller has validated the 16-byte V23 header.
    fn new(bytes: memmap2::Mmap, seqno: u64) -> Self {
        let (sections, table_result) = section_table(&bytes[16..]);
        let base = bytes.as_ptr() as usize;
        let sections: Vec<_> = sections.into_iter().map(|section| {
            let start = section.body.as_ptr() as usize - base;
            (section.name, start..start + section.body.len(), section.declared_len)
        }).collect();
        let decoded = (0..sections.len()).map(|_| None).collect();
        Self { bytes, sections, decoded, table_result, seqno }
    }

    fn decode_selected(&mut self, select: impl Fn(&str) -> bool) {
        let mut jobs: Vec<_> = self.sections.iter().enumerate()
            .filter(|(i, (name, _, _))| self.decoded[*i].is_none() && select(name))
            .collect();
        jobs.sort_by_key(|(_, (_, range, _))| std::cmp::Reverse(range.len()));
        let bytes = &self.bytes;
        let decoded: Vec<_> = decode_pool().install(|| jobs.par_iter()
            .with_max_len(1).map(|(i, (name, range, declared_len))| {
                let section = Section {
                    name: name.clone(), body: &bytes[range.clone()], declared_len: *declared_len,
                };
                (*i, decode_section(&section))
            }).collect());
        for (i, result) in decoded { self.decoded[i] = Some(result); }
    }

    fn finish(mut self) -> Result<FullSnapshot> {
        self.decode_selected(|_| true);
        let mut snap = FullSnapshot::empty(self.seqno);
        for result in self.decoded {
            result.expect("all snapshot sections decoded")?(&mut snap);
        }
        self.table_result?;
        for state in snap.states.values_mut() { state.sanitize(); }
        Ok(snap)
    }
}

fn decode_section(section: &Section<'_>) -> Result<ApplySection> {
    let n = section.name.as_str();
    macro_rules! decode {
        ($field:ident) => {{
            let value = decode_body(section.body, n)?;
            Ok(Box::new(move |snap: &mut FullSnapshot| snap.$field = value) as ApplySection)
        }};
    }
    macro_rules! map {
        ($field:ident) => {{
            let value = read_map_section(section.body, n)?;
            Ok(Box::new(move |snap: &mut FullSnapshot| snap.$field = value) as ApplySection)
        }};
    }
    match n {
        "payloads" => map!(payloads),
        "states" => map!(states),
        "assoc_edges" => map!(assoc_edges),
        "artifacts" => map!(artifacts),
        "artifact_paths" => map!(artifact_paths),
        "time_idx" => decode!(time_idx),
        "keyword_idx" => decode!(keyword_idx),
        "artifact_idx" => decode!(artifact_idx),
        "triplet_store" => {
            let mut value: TripletStore = decode_body(section.body, n)?;
            let _phase = crate::profile::SnapshotPhase::new("triplet_indexes");
            value.rebuild_indexes();
            Ok(Box::new(move |snap: &mut FullSnapshot| snap.triplet_store = value))
        }
        "triplets_clean" => {
            let value = decode_body(section.body, n)?;
            Ok(Box::new(move |snap: &mut FullSnapshot| snap.triplet_store.clean = value))
        }
        "symbol_idx" => decode!(symbol_idx),
        "call_graph" => decode!(call_graph),
        "code_files" => decode!(code_files),
        "semantic_idx" => decode!(semantic_idx),
        "coactivation_stats" => map!(coactivation_stats),
        "ack_scores" => map!(ack_scores),
        "correction_states" => map!(correction_states),
        "event_tape" => decode!(event_tape),
        "decision_tape" => decode!(decision_tape),
        "turiya_monitor" => decode!(turiya_monitor),
        "observer_state" => decode!(observer_state),
        "interaction_ledger" => decode!(interaction_ledger),
        "predicate_store" => decode!(predicate_store),
        "cw_refresh_ts" => map!(cw_refresh_ts),
        "recall_provenance" => map!(recall_provenance),
        "utility_posteriors" => map!(utility_posteriors),
        "ledger_session_events" => decode!(ledger_session_events),
        _ => {
            let name = section.name.clone();
            let len = section.declared_len;
            Ok(Box::new(move |_| eprintln!(
                "[chitta-field] skipping unknown snapshot section '{}' ({} bytes)", name, len)))
        }
    }
}

/// Index the length-prefixed V23 bodies without decoding them. Save a malformed
/// later header as the last result: an earlier section's decode error must win,
/// just as it did in the streaming loader. A lone final name-length byte and a
/// short unknown body were accepted by that loader and remain accepted here.
fn section_table(mut bytes: &[u8]) -> (Vec<Section<'_>>, Result<()>) {
    let mut sections = Vec::new();
    let result = (|| {
        loop {
            let mut nlen = [0; 2];
            match bytes.read_exact(&mut nlen) {
                Ok(()) => {},
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(FieldError::Io(e)),
            }
            let mut name = vec![0; u16::from_le_bytes(nlen) as usize];
            bytes.read_exact(&mut name)?;
            let mut blen = [0; 8];
            bytes.read_exact(&mut blen)?;
            let declared_len = u64::from_le_bytes(blen);
            let len = declared_len.min(bytes.len() as u64) as usize;
            let (body, rest) = bytes.split_at(len);
            sections.push(Section { name: String::from_utf8_lossy(&name).into_owned(), body, declared_len });
            bytes = rest;
        }
        Ok(())
    })();
    (sections, result)
}

impl FullSnapshot {
    /// An empty snapshot at the given seqno — the starting point for the V23
    /// sectioned loader; sections missing from the file keep these defaults.
    fn empty(snapshot_seqno: u64) -> Self {
        FullSnapshot {
            snapshot_seqno,
            payloads:           HashMap::new(),
            states:             HashMap::new(),
            assoc_edges:        HashMap::new(),
            artifacts:          HashMap::new(),
            artifact_paths:     HashMap::new(),
            time_idx:           TemporalIndex::new(),
            keyword_idx:        KeywordIndex::new(),
            artifact_idx:       ArtifactIndex::new(),
            triplet_store:      TripletStore::new(),
            symbol_idx:         SymbolIndex::new(),
            call_graph:         CallGraph::new(),
            code_files:         CodeFileIndex::new(),
            semantic_idx:       SemanticIndex::new(),
            coactivation_stats: HashMap::new(),
            ack_scores:         HashMap::new(),
            correction_states:  HashMap::new(),
            event_tape:         EventTape::new(),
            decision_tape:      DecisionTape::new(),
            turiya_monitor:     TuriyaMonitor::new(),
            observer_state:     ObserverState::default(),
            interaction_ledger: InteractionLedger::default(),
            predicate_store:    PredicateStore::default(),
            cw_refresh_ts:      HashMap::new(),
            recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("snapshot.tmp");
        {
            let f = std::fs::File::create(&tmp)?;
            let mut w = BufWriter::new(f);
            // V23 sectioned container: magic, seqno (fixed offset for
            // peek_seqno), then one section per field.
            w.write_all(&FULL_SNAPSHOT_MAGIC.to_le_bytes())?;
            w.write_all(&self.snapshot_seqno.to_le_bytes())?;
            write_section(&mut w, "payloads",           &self.payloads)?;
            write_section(&mut w, "states",             &self.states)?;
            write_section(&mut w, "assoc_edges",        &self.assoc_edges)?;
            write_section(&mut w, "artifacts",          &self.artifacts)?;
            write_section(&mut w, "artifact_paths",     &self.artifact_paths)?;
            write_section(&mut w, "time_idx",           &self.time_idx)?;
            write_section(&mut w, "keyword_idx",        &self.keyword_idx)?;
            write_section(&mut w, "artifact_idx",       &self.artifact_idx)?;
            write_section(&mut w, "triplet_store",      &self.triplet_store)?;
            write_section(&mut w, "triplets_clean",     &self.triplet_store.clean)?;
            write_section(&mut w, "symbol_idx",         &self.symbol_idx)?;
            write_section(&mut w, "call_graph",         &self.call_graph)?;
            write_section(&mut w, "code_files",         &self.code_files)?;
            write_section(&mut w, "semantic_idx",       &self.semantic_idx)?;
            write_section(&mut w, "coactivation_stats", &self.coactivation_stats)?;
            write_section(&mut w, "ack_scores",         &self.ack_scores)?;
            write_section(&mut w, "correction_states",  &self.correction_states)?;
            write_section(&mut w, "event_tape",         &self.event_tape)?;
            write_section(&mut w, "decision_tape",      &self.decision_tape)?;
            write_section(&mut w, "turiya_monitor",     &self.turiya_monitor)?;
            write_section(&mut w, "observer_state",     &self.observer_state)?;
            write_section(&mut w, "interaction_ledger", &self.interaction_ledger)?;
            write_section(&mut w, "predicate_store",    &self.predicate_store)?;
            write_section(&mut w, "cw_refresh_ts",      &self.cw_refresh_ts)?;
            write_section(&mut w, "recall_provenance",  &self.recall_provenance)?;
            write_section(&mut w, "utility_posteriors", &self.utility_posteriors)?;
            write_section(&mut w, "ledger_session_events", &self.ledger_session_events)?;
            w.flush()?;
            // fsync data+magic to disk before the rename commits the file, so a crash
            // can't leave a renamed-but-truncated snapshot whose magic still reads valid.
            w.into_inner()
                .map_err(|e| FieldError::Manifest(e.to_string()))?
                .sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        // fsync the parent directory so the rename entry itself survives a crash.
        if let Some(dir) = path.parent() {
            if let Ok(d) = std::fs::File::open(dir) { let _ = d.sync_all(); }
        }
        Ok(())
    }

    /// Save payload content to a `.pld` sidecar.
    /// Format: [PLD_MAGIC:u64][count:u64]([id:u64][len:u32][bytes:u8×len])×count
    pub fn save_payload_sidecar(path: &Path, payloads: &HashMap<MemoryId, crate::payload::MemoryPayload>) -> std::io::Result<()> {
        let tmp = path.with_extension("pld.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&PLD_MAGIC.to_le_bytes())?;
            f.write_all(&(payloads.len() as u64).to_le_bytes())?;
            for (&id, payload) in payloads {
                f.write_all(&id.to_le_bytes())?;
                f.write_all(&(payload.content.len() as u32).to_le_bytes())?;
                f.write_all(&payload.content)?;
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load payload content from a `.pld` sidecar into an existing payload map.
    /// Returns false if the file is missing or corrupt (caller falls back to bincode content).
    pub fn load_payload_sidecar(path: &Path, payloads: &mut HashMap<MemoryId, crate::payload::MemoryPayload>) -> bool {
        let bytes = match std::fs::read(path) { Ok(b) => b, Err(_) => return false };
        if bytes.len() < 16 { return false; }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != PLD_MAGIC { return false; }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut off = 16usize;
        for _ in 0..count {
            if off + 12 > bytes.len() { return false; }
            let id = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[off+8..off+12].try_into().unwrap()) as usize;
            off += 12;
            if off + len > bytes.len() { return false; }
            if let Some(payload) = payloads.get_mut(&id) {
                payload.content = bytes[off..off+len].to_vec();
            }
            off += len;
        }
        eprintln!("[chitta-field] loaded .pld sidecar: {} content entries", count);
        true
    }

    /// Save the retrieval-surface sidecar (.rsf): id -> NL embed surface bytes.
    /// Format mirrors .pld: [RSF_MAGIC:u64][count:u64]([id:u64][len:u32][bytes]×count).
    pub fn save_retrieval_surface_sidecar(path: &Path, surfaces: &HashMap<MemoryId, Vec<u8>>) -> std::io::Result<()> {
        let tmp = path.with_extension("rsf.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&RSF_MAGIC.to_le_bytes())?;
            f.write_all(&(surfaces.len() as u64).to_le_bytes())?;
            for (&id, bytes) in surfaces {
                f.write_all(&id.to_le_bytes())?;
                f.write_all(&(bytes.len() as u32).to_le_bytes())?;
                f.write_all(bytes)?;
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load the .rsf sidecar. Missing/corrupt returns an empty map (embed then
    /// falls back to content) — never fatal, since content is the authoritative copy.
    pub fn load_retrieval_surface_sidecar(path: &Path) -> HashMap<MemoryId, Vec<u8>> {
        let mut out = HashMap::new();
        let bytes = match std::fs::read(path) { Ok(b) => b, Err(_) => return out };
        if bytes.len() < 16 { return out; }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != RSF_MAGIC { return out; }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut off = 16usize;
        for _ in 0..count {
            if off + 12 > bytes.len() { break; }
            let id = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap());
            let len = u32::from_le_bytes(bytes[off+8..off+12].try_into().unwrap()) as usize;
            off += 12;
            if off + len > bytes.len() { break; }
            out.insert(id, bytes[off..off+len].to_vec());
            off += len;
        }
        eprintln!("[chitta-field] loaded .rsf sidecar: {} surface entries", out.len());
        out
    }

    /// Read only the magic and snapshot_seqno without deserializing the full snapshot.
    pub fn peek_seqno(path: &Path) -> Result<u64> {
        let f = std::fs::File::open(path)?;
        let mut r = BufReader::new(f);
        let mut buf = [0u8; 16];
        r.read_exact(&mut buf)?;
        let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        if magic != FULL_SNAPSHOT_MAGIC
            && magic != FULL_SNAPSHOT_MAGIC_V18
            && magic != FULL_SNAPSHOT_MAGIC_V17
            && magic != FULL_SNAPSHOT_MAGIC_V16
            && magic != FULL_SNAPSHOT_MAGIC_V15
            && magic != FULL_SNAPSHOT_MAGIC_V14
            && magic != FULL_SNAPSHOT_MAGIC_V13
            && magic != FULL_SNAPSHOT_MAGIC_V12
            && magic != FULL_SNAPSHOT_MAGIC_V11
            && magic != FULL_SNAPSHOT_MAGIC_V10
            && magic != FULL_SNAPSHOT_MAGIC_V9
            && magic != FULL_SNAPSHOT_MAGIC_V8
            && magic != FULL_SNAPSHOT_MAGIC_V7
            && magic != FULL_SNAPSHOT_MAGIC_V6
            && magic != FULL_SNAPSHOT_MAGIC_V5
            && magic != FULL_SNAPSHOT_MAGIC_V4
            && magic != FULL_SNAPSHOT_MAGIC_V1
            && magic != FULL_SNAPSHOT_MAGIC_V19
            && magic != FULL_SNAPSHOT_MAGIC_V20
            && magic != FULL_SNAPSHOT_MAGIC_V21
            && magic != FULL_SNAPSHOT_MAGIC_V22
        {
            return Err(FieldError::Manifest("invalid full snapshot magic".to_string()));
        }
        Ok(u64::from_le_bytes(buf[8..16].try_into().unwrap()))
    }

    /// Load a full snapshot from disk. Transparently migrates all legacy formats.
    /// V23 bodies decode concurrently from an immutable mapping; legacy layouts stream.
    pub fn load(path: &Path) -> Result<Self> {
        let (mut snap, rebuilt) = Self::load_inner(path)?;
        if !rebuilt { snap.triplet_store.rebuild_indexes(); }
        // V22: hydrate per-state refresh timestamps from the persisted map so a
        // restart does not trigger a full competitive-weight refresh sweep.
        for (id, ts) in &snap.cw_refresh_ts {
            if let Some(st) = snap.states.get_mut(id) {
                st.last_cw_refresh_ms = *ts;
            }
        }
        // V23 "utility_posteriors": hydrate after load_inner's sanitize(), which
        // has already reset every state to the (1, 1) prior.
        for (id, &(alpha, beta)) in &snap.utility_posteriors {
            if let Some(st) = snap.states.get_mut(id) {
                st.utility_alpha = alpha;
                st.utility_beta = beta;
                st.sanitize();
            }
        }
        Ok(snap)
    }

    fn load_inner(path: &Path) -> Result<(Self, bool)> {
        let mut rebuilt = false;
        let snap = Self::load_raw(path, &mut rebuilt)?;
        Ok((snap, rebuilt))
    }

    fn load_raw(path: &Path, rebuilt: &mut bool) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| FieldError::Manifest(e.to_string()))?;
        let mut r = BufReader::with_capacity(1 << 20, SnapshotReader::new(file));
        let mut magic_buf = [0u8; 8];
        r.read_exact(&mut magic_buf)
            .map_err(|_| FieldError::Manifest("snapshot too short".to_string()))?;
        let magic = u64::from_le_bytes(magic_buf);

        if magic == FULL_SNAPSHOT_MAGIC {
            // V23: sectioned container. Unknown sections are skipped (a newer
            // writer's file still loads), missing sections keep defaults (an
            // older file from a leaner writer still loads).
            let mut seq_buf = [0u8; 8];
            r.read_exact(&mut seq_buf)
                .map_err(|_| FieldError::Manifest("v23 snapshot too short".to_string()))?;
            // SAFETY: committed snapshots are immutable. save() writes a fresh
            // temporary file and atomically renames it; pruning/unlinking a file
            // cannot invalidate this open file's mapping. Keep the mapping alive
            // until every decoding worker has joined; all decoded values own
            // their data. Legacy formats keep the historical streaming reader.
            let bytes = unsafe { memmap2::MmapOptions::new().map(&r.get_ref().file)? };
            if std::env::var_os("CHITTA_PROFILE_SNAPSHOT").is_some() {
                // Mapping page faults are included in decode timings, not the
                // explicit read() timer emitted by SnapshotReader.
                eprintln!("[chitta-field] snapshot mapped_bytes={}", bytes.len());
            }
            let _phase = crate::profile::SnapshotPhase::new("parallel_sections");
            let snap = MappedSnapshot::new(bytes, u64::from_le_bytes(seq_buf)).finish()?;
            *rebuilt = true;
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V22 {
            // V22: last monolithic-bincode format (frozen layout).
            eprintln!("[chitta-field] migrating v22 snapshot → v23 (sectioned container)");
            let leg: LegacyFullSnapshotV22 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = leg.upgrade();
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V21 {
            // V21→V22: add cw_refresh_ts (default empty).
            eprintln!("[chitta-field] migrating v21 snapshot → v22 (cw refresh timestamps)");
            let leg: LegacyFullSnapshotV21 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut states = leg.states;
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: leg.interaction_ledger,
                predicate_store:    leg.predicate_store,
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V20 {
            // V20→V21: add predicate_store (default empty).
            eprintln!("[chitta-field] migrating v20 snapshot → v21 (predicate store)");
            let leg: LegacyFullSnapshotV20 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut states = leg.states;
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: leg.interaction_ledger,
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V19 {
            // V19→V20: add interaction_ledger (default empty).
            eprintln!("[chitta-field] migrating v19 snapshot → v20 (interaction ledger)");
            let leg: LegacyFullSnapshotV19 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut states = leg.states;
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V18 {
            // V18→V19: add embedding_model_id/embedding_dim; clear stale 256-d embeddings;
            // mark all non-deleted memories with content as embed_pending for 768-d re-embedding.
            eprintln!("[chitta-field] migrating v18 snapshot → v19 (Ollama 768-d re-embedding)");
            let leg: LegacyFullSnapshotV18 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let payloads = upgrade_payloads_v18(leg.payloads);
            let mut states = leg.states;
            // Mark every non-deleted memory with content as embed_pending so the
            // backfill thread re-embeds with nomic-embed-text:v1.5 at 768 dims.
            for (id, state) in states.iter_mut() {
                if state.deleted { continue; }
                let has_content = payloads.get(id)
                    .map(|p| p.content.len() >= 10)
                    .unwrap_or(false);
                if has_content {
                    state.embed_pending = true;
                }
            }
            for state in states.values_mut() { state.sanitize(); }
            return Ok(FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads,
                states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     leg.observer_state,
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V17 {
            // V17→V18: add empty ObserverState.
            eprintln!("[chitta-field] migrating v17 snapshot → v18 (adding observer_state)");
            let leg: LegacyFullSnapshotV17 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states:             leg.states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V16 {
            // V16→V17: MemoryPayload gains provenance + candidate.
            eprintln!("[chitta-field] migrating v16 snapshot → v17 (adding provenance+candidate to payloads)");
            let leg: LegacyFullSnapshotV16 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           upgrade_payloads_v16(leg.payloads),
                states:             leg.states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         leg.event_tape,
                decision_tape:      leg.decision_tape,
                turiya_monitor:     leg.turiya_monitor,
                observer_state:     ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V15 {
            // V15→V16: add empty TuriyaMonitor.
            eprintln!("[chitta-field] migrating v15 snapshot → v16 (adding turiya_monitor)");
            let leg: LegacyFullSnapshotV15 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:    leg.snapshot_seqno,
                payloads:          leg.payloads,
                states:            leg.states,
                assoc_edges:       leg.assoc_edges,
                artifacts:         leg.artifacts,
                artifact_paths:    leg.artifact_paths,
                time_idx:          leg.time_idx,
                keyword_idx:       leg.keyword_idx,
                artifact_idx:      leg.artifact_idx,
                triplet_store:     leg.triplet_store,
                symbol_idx:        leg.symbol_idx,
                call_graph:        leg.call_graph,
                code_files:        leg.code_files,
                semantic_idx:      leg.semantic_idx,
                coactivation_stats:leg.coactivation_stats,
                ack_scores:        leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape:        leg.event_tape,
                decision_tape:     leg.decision_tape,
                turiya_monitor:    TuriyaMonitor::new(),
                observer_state:    ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V14 {
            // V14→V15: add empty DecisionTape.
            eprintln!("[chitta-field] migrating v14 snapshot → v15 (adding decision_tape)");
            let leg: LegacyFullSnapshotV14 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:    leg.snapshot_seqno,
                payloads:          leg.payloads,
                states:            leg.states,
                assoc_edges:       leg.assoc_edges,
                artifacts:         leg.artifacts,
                artifact_paths:    leg.artifact_paths,
                time_idx:          leg.time_idx,
                keyword_idx:       leg.keyword_idx,
                artifact_idx:      leg.artifact_idx,
                triplet_store:     leg.triplet_store,
                symbol_idx:        leg.symbol_idx,
                call_graph:        leg.call_graph,
                code_files:        leg.code_files,
                semantic_idx:      leg.semantic_idx,
                coactivation_stats:leg.coactivation_stats,
                ack_scores:        leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape:        leg.event_tape,
                decision_tape:     DecisionTape::new(),
                turiya_monitor:    TuriyaMonitor::new(),
                observer_state:    ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V13 {
            // V13→V14: add empty EventTape.
            eprintln!("[chitta-field] migrating v13 snapshot → v14 (adding event_tape)");
            let leg: LegacyFullSnapshotV13 = bincode::deserialize_from(&mut r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno:     leg.snapshot_seqno,
                payloads:           leg.payloads,
                states:             leg.states,
                assoc_edges:        leg.assoc_edges,
                artifacts:          leg.artifacts,
                artifact_paths:     leg.artifact_paths,
                time_idx:           leg.time_idx,
                keyword_idx:        leg.keyword_idx,
                artifact_idx:       leg.artifact_idx,
                triplet_store:      leg.triplet_store,
                symbol_idx:         leg.symbol_idx,
                call_graph:         leg.call_graph,
                code_files:         leg.code_files,
                semantic_idx:       leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores:         leg.ack_scores,
                correction_states:  leg.correction_states,
                event_tape:         EventTape::new(),
                decision_tape:      DecisionTape::new(),
                turiya_monitor:     TuriyaMonitor::new(),
                observer_state:     ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V12 {
            // v12: MemoryState has staged+invalidated_by; MemoryPayload lacks harness.
            eprintln!("[chitta-field] migrating v12 snapshot → v13 (adding harness to payloads)");
            let r = &mut r;
            let leg: LegacyFullSnapshotV12 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut snap = FullSnapshot {
                snapshot_seqno: leg.snapshot_seqno,
                payloads: upgrade_payloads(leg.payloads),
                states: leg.states,
                assoc_edges: leg.assoc_edges,
                artifacts: leg.artifacts,
                artifact_paths: leg.artifact_paths,
                time_idx: leg.time_idx,
                keyword_idx: leg.keyword_idx,
                artifact_idx: leg.artifact_idx,
                triplet_store: leg.triplet_store,
                symbol_idx: leg.symbol_idx,
                call_graph: leg.call_graph,
                code_files: leg.code_files,
                semantic_idx: leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores: leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            };
            for state in snap.states.values_mut() { state.sanitize(); }
            return Ok(snap);
        }

        if magic == FULL_SNAPSHOT_MAGIC_V11
            || magic == FULL_SNAPSHOT_MAGIC_V10
            || magic == FULL_SNAPSHOT_MAGIC_V9
        {
            // v9/v10/v11: MemoryState lacks staged/invalidated_by; MemoryPayload lacks harness.
            if magic == FULL_SNAPSHOT_MAGIC_V11 {
                eprintln!("[chitta-field] migrating v11 snapshot → v13");
            } else {
                eprintln!("[chitta-field] migrating v9/v10 snapshot → v13");
            }
            let r = &mut r;
            let leg: LegacyFullSnapshotV11 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states: HashMap<MemoryId, MemoryState> = leg.states
                .into_iter()
                .map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) })
                .collect();
            return Ok(FullSnapshot {
                snapshot_seqno: leg.snapshot_seqno,
                payloads: upgrade_payloads(leg.payloads),
                states,
                assoc_edges: leg.assoc_edges,
                artifacts: leg.artifacts,
                artifact_paths: leg.artifact_paths,
                time_idx: leg.time_idx,
                keyword_idx: leg.keyword_idx,
                artifact_idx: leg.artifact_idx,
                triplet_store: leg.triplet_store,
                symbol_idx: leg.symbol_idx,
                call_graph: leg.call_graph,
                code_files: leg.code_files,
                semantic_idx: leg.semantic_idx,
                coactivation_stats: leg.coactivation_stats,
                ack_scores: leg.ack_scores,
                correction_states: leg.correction_states,
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V8 {
            // V8: no top-level sidecars; MemoryState lacks staged/invalidated_by; payload lacks harness.
            eprintln!("[chitta-field] migrating v8 snapshot → v13");
            let r = &mut r;
            let v8: LegacyFullSnapshotV8 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states: HashMap<MemoryId, MemoryState> = v8.states
                .into_iter()
                .map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) })
                .collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v8.snapshot_seqno,
                payloads: upgrade_payloads(v8.payloads),
                states,
                assoc_edges: v8.assoc_edges,
                artifacts: v8.artifacts,
                artifact_paths: v8.artifact_paths,
                time_idx: v8.time_idx,
                keyword_idx: v8.keyword_idx,
                artifact_idx: v8.artifact_idx,
                triplet_store: v8.triplet_store,
                symbol_idx: v8.symbol_idx,
                call_graph: v8.call_graph,
                code_files: v8.code_files,
                semantic_idx: v8.semantic_idx,
                coactivation_stats: v8.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V7 {
            // V7: access_timestamps but no interference fields.
            eprintln!("[chitta-field] migrating v7 snapshot → v9");
            let r = &mut r;
            let v7: LegacyFullSnapshotV7 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v7.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v7.snapshot_seqno,
                payloads: upgrade_payloads(v7.payloads),
                states,
                assoc_edges: v7.assoc_edges,
                artifacts: v7.artifacts,
                artifact_paths: v7.artifact_paths,
                time_idx: v7.time_idx,
                keyword_idx: v7.keyword_idx,
                artifact_idx: v7.artifact_idx,
                triplet_store: v7.triplet_store,
                symbol_idx: v7.symbol_idx,
                call_graph: v7.call_graph,
                code_files: v7.code_files,
                semantic_idx: v7.semantic_idx,
                coactivation_stats: v7.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V6 {
            // V6: MemoryState with affect/surprise but no access_timestamps.
            eprintln!("[chitta-field] migrating v6 snapshot → v9");
            let r = &mut r;
            let v6: LegacyFullSnapshotV6 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v6.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v6.snapshot_seqno,
                payloads: upgrade_payloads(v6.payloads),
                states,
                assoc_edges: v6.assoc_edges,
                artifacts: v6.artifacts,
                artifact_paths: v6.artifact_paths,
                time_idx: v6.time_idx,
                keyword_idx: v6.keyword_idx,
                artifact_idx: v6.artifact_idx,
                triplet_store: v6.triplet_store,
                symbol_idx: v6.symbol_idx,
                call_graph: v6.call_graph,
                code_files: v6.code_files,
                semantic_idx: v6.semantic_idx,
                coactivation_stats: v6.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V5 {
            // V5: 14-field MemoryState + coactivation_stats.
            eprintln!("[chitta-field] migrating v5 snapshot → v9");
            let r = &mut r;
            let v5: LegacyFullSnapshotV5 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v5.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v5.snapshot_seqno,
                payloads: upgrade_payloads(v5.payloads),
                states,
                assoc_edges: v5.assoc_edges,
                artifacts: v5.artifacts,
                artifact_paths: v5.artifact_paths,
                time_idx: v5.time_idx,
                keyword_idx: v5.keyword_idx,
                artifact_idx: v5.artifact_idx,
                triplet_store: v5.triplet_store,
                symbol_idx: v5.symbol_idx,
                call_graph: v5.call_graph,
                code_files: v5.code_files,
                semantic_idx: v5.semantic_idx,
                coactivation_stats: v5.coactivation_stats,
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V4 {
            // V4: 14-field MemoryState, no coactivation_stats.
            eprintln!("[chitta-field] migrating v4 snapshot → v9");
            let r = &mut r;
            let v4: LegacyFullSnapshotV4 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let states = v4.states.into_iter().map(|(id, s)| { let mut m = s.upgrade(); m.sanitize(); (id, m) }).collect();
            return Ok(FullSnapshot {
                snapshot_seqno: v4.snapshot_seqno,
                payloads: upgrade_payloads(v4.payloads),
                states,
                assoc_edges: v4.assoc_edges,
                artifacts: v4.artifacts,
                artifact_paths: v4.artifact_paths,
                time_idx: v4.time_idx,
                keyword_idx: v4.keyword_idx,
                artifact_idx: v4.artifact_idx,
                triplet_store: v4.triplet_store,
                symbol_idx: v4.symbol_idx,
                call_graph: v4.call_graph,
                code_files: v4.code_files,
                semantic_idx: v4.semantic_idx,
                coactivation_stats: HashMap::new(),
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        if magic == FULL_SNAPSHOT_MAGIC_V1 {
            // V1: pre-ANN SemanticIndex + 14-field MemoryState.
            eprintln!("[chitta-field] migrating v1 snapshot → v6 (ANN index + MemoryStatus + EpistemicStatus)");
            let r = &mut r;
            let v1: LegacyFullSnapshotV1 = bincode::deserialize_from(r)
                .map_err(|e| FieldError::Serialization(e.to_string()))?;
            let mut semantic_idx = SemanticIndex::new();
            for (mem_id, emb) in v1.semantic_idx.embeddings {
                semantic_idx.upsert(mem_id, emb, None);
            }
            let states = v1.states.into_iter().map(|(id, s)| (id, s.upgrade())).collect::<HashMap<_, _>>();
            return Ok(FullSnapshot {
                snapshot_seqno: v1.snapshot_seqno,
                payloads: upgrade_payloads(v1.payloads),
                states,
                assoc_edges: v1.assoc_edges,
                artifacts: v1.artifacts,
                artifact_paths: v1.artifact_paths,
                time_idx: v1.time_idx,
                keyword_idx: v1.keyword_idx,
                artifact_idx: v1.artifact_idx,
                triplet_store: v1.triplet_store,
                symbol_idx: v1.symbol_idx,
                call_graph: v1.call_graph,
                code_files: v1.code_files,
                semantic_idx,
                coactivation_stats: HashMap::new(),
                ack_scores: HashMap::new(),
                correction_states: HashMap::new(),
                event_tape: EventTape::new(),
                decision_tape: DecisionTape::new(),
                turiya_monitor: TuriyaMonitor::new(),
                observer_state: ObserverState::default(),
                interaction_ledger: InteractionLedger::default(),
                predicate_store:    PredicateStore::default(),
                cw_refresh_ts:      HashMap::new(),
                recall_provenance:  HashMap::new(),
            utility_posteriors: HashMap::new(),
            ledger_session_events: Vec::new(),
            });
        }

        Err(FieldError::Manifest(format!("unknown snapshot magic: {:#x}", magic)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serial_v23(bytes: &[u8]) -> Result<FullSnapshot> {
        let mut r = bytes;
        let mut snap = FullSnapshot::empty(0);
            loop {
                let mut nlen_buf = [0u8; 2];
                match r.read_exact(&mut nlen_buf) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(FieldError::Io(e)),
                }
                let nlen = u16::from_le_bytes(nlen_buf) as usize;
                let mut name_buf = vec![0u8; nlen];
                r.read_exact(&mut name_buf)?;
                let name = String::from_utf8_lossy(&name_buf).into_owned();
                let mut blen_buf = [0u8; 8];
                r.read_exact(&mut blen_buf)?;
                let blen = u64::from_le_bytes(blen_buf);
                let mut body = Read::take(&mut r, blen);
                let n = name.as_str();
                match n {
                    "payloads"           => snap.payloads           = read_section(&mut body, n)?,
                    "states"             => snap.states             = read_section(&mut body, n)?,
                    "assoc_edges"        => snap.assoc_edges        = read_section(&mut body, n)?,
                    "artifacts"          => snap.artifacts          = read_section(&mut body, n)?,
                    "artifact_paths"     => snap.artifact_paths     = read_section(&mut body, n)?,
                    "time_idx"           => snap.time_idx           = read_section(&mut body, n)?,
                    "keyword_idx"        => snap.keyword_idx        = read_section(&mut body, n)?,
                    "artifact_idx"       => snap.artifact_idx       = read_section(&mut body, n)?,
                    "triplet_store"      => snap.triplet_store      = read_section(&mut body, n)?,
                    "triplets_clean"     => snap.triplet_store.clean = read_section(&mut body, n)?,
                    "symbol_idx"         => snap.symbol_idx         = read_section(&mut body, n)?,
                    "call_graph"         => snap.call_graph         = read_section(&mut body, n)?,
                    "code_files"         => snap.code_files         = read_section(&mut body, n)?,
                    "semantic_idx"       => snap.semantic_idx       = read_section(&mut body, n)?,
                    "coactivation_stats" => snap.coactivation_stats = read_section(&mut body, n)?,
                    "ack_scores"         => snap.ack_scores         = read_section(&mut body, n)?,
                    "correction_states"  => snap.correction_states  = read_section(&mut body, n)?,
                    "event_tape"         => snap.event_tape         = read_section(&mut body, n)?,
                    "decision_tape"      => snap.decision_tape      = read_section(&mut body, n)?,
                    "turiya_monitor"     => snap.turiya_monitor     = read_section(&mut body, n)?,
                    "observer_state"     => snap.observer_state     = read_section(&mut body, n)?,
                    "interaction_ledger" => snap.interaction_ledger = read_section(&mut body, n)?,
                    "predicate_store"    => snap.predicate_store    = read_section(&mut body, n)?,
                    "cw_refresh_ts"      => snap.cw_refresh_ts      = read_section(&mut body, n)?,
                    "recall_provenance"  => snap.recall_provenance  = read_section(&mut body, n)?,
                    "utility_posteriors" => snap.utility_posteriors = read_section(&mut body, n)?,
                    "ledger_session_events" => snap.ledger_session_events = read_section(&mut body, n)?,
                    _ => {
                        eprintln!(
                            "[chitta-field] skipping unknown snapshot section '{}' ({} bytes)",
                            name, blen
                        );
                    }
                }
                // Drain whatever the deserializer left so the next section
                // header is read from the right offset.
                std::io::copy(&mut body, &mut std::io::sink()).map_err(FieldError::Io)?;
            }
        Ok(snap)
    }

    fn parallel_v23(bytes: &[u8]) -> Result<FullSnapshot> {
        let mut mapping = memmap2::MmapMut::map_anon(bytes.len() + 16).unwrap();
        mapping[16..].copy_from_slice(bytes);
        let mut pending = MappedSnapshot::new(mapping.make_read_only().unwrap(), 0);
        pending.decode_selected(|name| name == "triplets_clean");
        let count = pending.decoded.iter().filter(|r| r.is_some()).count();
        pending.decode_selected(|name| name == "triplets_clean");
        assert_eq!(count, pending.decoded.iter().filter(|r| r.is_some()).count());
        pending.finish()
    }

    #[test]
    fn staged_sections_retain_mapping_after_snapshot_unlink() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot");
        let mut bytes = vec![0; 16];
        write_section(&mut bytes, "ack_scores", &HashMap::from([(7u64, 9i32)])).unwrap();
        write_section(&mut bytes, "triplets_clean", &true).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // SAFETY: this test never modifies the immutable mapped inode.
        let mapping = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let mut pending = MappedSnapshot::new(mapping, 42);
        pending.decode_selected(|name| name == "triplets_clean");
        drop(file);
        std::fs::remove_file(path).unwrap();
        let snap = pending.finish().unwrap();
        assert_eq!(snap.snapshot_seqno, 42);
        assert_eq!(snap.ack_scores, HashMap::from([(7, 9)]));
        assert!(snap.triplet_store.clean);
    }

    #[test]
    fn parallel_sections_preserve_streaming_errors_at_every_truncation() {
        let mut bytes = Vec::new();
        write_section(&mut bytes, "ack_scores", &HashMap::from([(7u64, 5i32), (8, 9)])).unwrap();
        write_section(&mut bytes, "future_section", &vec![1u64, 2, 3]).unwrap();
        write_section(&mut bytes, "triplets_clean", &true).unwrap();
        for end in 0..=bytes.len() {
            let serial = serial_v23(&bytes[..end]);
            let parallel = parallel_v23(&bytes[..end]);
            match (serial, parallel) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.ack_scores, b.ack_scores, "end={end}");
                    assert_eq!(a.triplet_store.clean, b.triplet_store.clean, "end={end}");
                }
                (Err(a), Err(b)) => {
                    assert_eq!(std::mem::discriminant(&a), std::mem::discriminant(&b), "end={end}");
                    assert_eq!(a.to_string(), b.to_string(), "end={end}");
                }
                _ => panic!("different acceptance at end={end}"),
            }
        }
        // An earlier corrupt body wins over either a later corrupt body or header.
        let mut bad = Vec::new();
        write_section(&mut bad, "triplets_clean", &2u8).unwrap();
        write_section(&mut bad, "ack_scores", &0u8).unwrap();
        bad.extend_from_slice(&[10, 0, b'a']);
        for _ in 0..20 {
            assert_eq!(serial_v23(&bad).err().unwrap().to_string(), parallel_v23(&bad).err().unwrap().to_string());
        }
    }

    #[test]
    fn parallel_sections_preserve_duplicate_order_and_short_unknown_body() {
        let mut bytes = Vec::new();
        write_section(&mut bytes, "triplets_clean", &true).unwrap();
        write_section(&mut bytes, "triplet_store", &TripletStore::new()).unwrap();
        assert!(!parallel_v23(&bytes).unwrap().triplet_store.clean);
        write_section(&mut bytes, "triplets_clean", &true).unwrap();
        write_section(&mut bytes, "ack_scores", &HashMap::from([(1u64, 3i32)])).unwrap();
        write_section(&mut bytes, "ack_scores", &HashMap::from([(2u64, 7i32)])).unwrap();
        // Both old and new readers allow body padding and a truncated unknown body.
        bytes.extend_from_slice(&6u16.to_le_bytes());
        bytes.extend_from_slice(b"future");
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.push(9);
        let snap = parallel_v23(&bytes).unwrap();
        assert!(snap.triplet_store.clean);
        assert_eq!(snap.ack_scores, HashMap::from([(2, 7)]));
        assert_eq!(serial_v23(&bytes).unwrap().ack_scores, snap.ack_scores);
    }

    #[test]
    fn presized_maps_match_bincode_and_bound_corrupt_count() {
        let entries: Vec<_> = (0u64..40_000).map(|id| (id, id + 1)).collect();
        // A vector of pairs has the same wire layout as a bincode map. Duplicate
        // keys remain last-wins, even when the advertised count exceeds unique keys.
        let mut duplicated = entries.clone();
        duplicated.push((0, 99));
        let mut bytes = bincode::serialize(&duplicated).unwrap();
        bytes.extend_from_slice(b"padding");
        let expected: HashMap<u64, u64> = read_section(&mut &bytes[..], "test").unwrap();
        let actual: HashMap<u64, u64> = read_map_section(&bytes, "test").unwrap();
        assert_eq!(expected, actual);
        let corrupt = u64::MAX.to_le_bytes();
        let old = read_section::<_, HashMap<u64, u64>>(&mut &corrupt[..], "test").unwrap_err();
        let new = read_map_section::<u64, u64>(&corrupt, "test").unwrap_err();
        assert_eq!(old.to_string(), new.to_string());
    }


    #[test]
    fn rsf_sidecar_roundtrip_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chitta.abc.snapshot");
        let rsf = path.with_extension("rsf");

        // Missing sidecar → empty map, no panic (old snapshot family).
        assert!(FullSnapshot::load_retrieval_surface_sidecar(&rsf).is_empty());

        // Round-trip: surfaces survive save→load byte-identical, incl. UTF-8 + empty-ish.
        let mut surfaces: HashMap<MemoryId, Vec<u8>> = HashMap::new();
        surfaces.insert(1, b"The rtk submodule was removed and merged to main.".to_vec());
        surfaces.insert(2, "unicode: reduced 25GB\u{2192}7GB RSS".as_bytes().to_vec());
        surfaces.insert(u64::MAX, vec![0u8; 4096]);
        FullSnapshot::save_retrieval_surface_sidecar(&rsf, &surfaces).unwrap();

        let loaded = FullSnapshot::load_retrieval_surface_sidecar(&rsf);
        assert_eq!(loaded, surfaces);

        // Truncated/corrupt sidecar → empty map, no panic (fallback to content).
        std::fs::write(&rsf, b"short").unwrap();
        assert!(FullSnapshot::load_retrieval_surface_sidecar(&rsf).is_empty());
    }

    fn legacy_v4_state() -> LegacyMemoryStateV4 {
        LegacyMemoryStateV4 {
            memory_id: 42,
            current_version: 1,
            current_chunk_hash: [0u8; 32],
            deleted: false,
            strength: 0.8,
            decay_rate: 0.001,
            confidence: 0.9,
            access_count: 3,
            last_accessed_ms: 1_000_000,
            last_strengthened_ms: 900_000,
            created_at_ms: 800_000,
            pinned: false,
            tier: 0,
            last_state_op_ts_ms: 1_000_001,
        }
    }

    fn legacy_v5_state() -> LegacyMemoryStateV5 {
        LegacyMemoryStateV5 {
            memory_id: 99,
            current_version: 2,
            current_chunk_hash: [0u8; 32],
            deleted: false,
            strength: 0.7,
            decay_rate: 0.002,
            confidence: 0.85,
            access_count: 5,
            last_accessed_ms: 2_000_000,
            last_strengthened_ms: 1_900_000,
            created_at_ms: 1_800_000,
            pinned: true,
            tier: 1,
            last_state_op_ts_ms: 2_000_001,
            retrieval_history: RetrievalHistory::default(),
        }
    }

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "chitta-snap-test-{}-{}",
            tag,
            std::process::id()
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn v23_triplet_clean_section_is_optional_and_survives_restart() {
        let path = scratch_path("triplets-clean");
        let mut snap = FullSnapshot::empty(42);
        snap.triplet_store.replay_add(1, "s".into(), "p".into(), "o".into(), 0.5, 0, None, None);
        snap.triplet_store.replay_add(2, "s".into(), "p".into(), "o".into(), 0.8, 0, None, None);
        // Legacy V23 container has no marker. Its bincode triplet body is unchanged.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FULL_SNAPSHOT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&42u64.to_le_bytes());
        write_section(&mut bytes, "triplet_store", &snap.triplet_store).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let mut loaded = FullSnapshot::load(&path).unwrap();
        assert!(!loaded.triplet_store.clean);
        assert_eq!(loaded.triplet_store.clean_for_load(), (0, 1));
        loaded.save(&path).unwrap();
        let mut restored = FullSnapshot::load(&path).unwrap();
        assert!(restored.triplet_store.clean);
        assert_eq!(restored.triplet_store.clean_for_load(), (0, 0));
        assert_eq!(restored.triplet_store.query_subject("s", 0)[0].id, 2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn v23_roundtrip_empty_snapshot() {
        let path = scratch_path("v23-roundtrip");
        let snap = FullSnapshot::empty(42);
        snap.save(&path).unwrap();
        assert_eq!(FullSnapshot::peek_seqno(&path).unwrap(), 42);
        let loaded = FullSnapshot::load(&path).unwrap();
        assert_eq!(loaded.snapshot_seqno, 42);
        assert!(loaded.states.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v23_unknown_section_is_skipped() {
        let path = scratch_path("v23-unknown");
        // Hand-build: magic, seqno, one unknown section, then a real one.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FULL_SNAPSHOT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&7u64.to_le_bytes());
        let name = b"future_organ";
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name);
        let body = vec![0xAB; 33];
        bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&body);
        let mut ack: HashMap<MemoryId, i32> = HashMap::new();
        ack.insert(9, 3);
        let ack_body = bincode::serialize(&ack).unwrap();
        let ack_name = b"ack_scores";
        bytes.extend_from_slice(&(ack_name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(ack_name);
        bytes.extend_from_slice(&(ack_body.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&ack_body);
        std::fs::write(&path, &bytes).unwrap();

        let loaded = FullSnapshot::load(&path).unwrap();
        assert_eq!(loaded.snapshot_seqno, 7);
        assert_eq!(loaded.ack_scores.get(&9), Some(&3));
        // Missing sections defaulted.
        assert!(loaded.states.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn utility_posteriors_roundtrip_and_hydrate_states() {
        let path = scratch_path("utility-roundtrip");
        let mut snap = FullSnapshot::empty(21);
        let mut st = MemoryState::new(7, [0u8; 32], 1_000);
        st.record_outcome(true, 3.0);
        st.record_outcome(false, 1.0);
        assert_eq!((st.utility_alpha, st.utility_beta), (4.0, 2.0));
        snap.utility_posteriors.insert(7, (st.utility_alpha, st.utility_beta));
        snap.states.insert(7, st);
        // A memory with no observation is simply absent from the section.
        snap.states.insert(8, MemoryState::new(8, [0u8; 32], 1_000));
        snap.save(&path).unwrap();

        let loaded = FullSnapshot::load(&path).unwrap();
        assert_eq!(loaded.utility_posteriors.get(&7), Some(&(4.0, 2.0)));
        let hydrated = &loaded.states[&7];
        assert_eq!((hydrated.utility_alpha, hydrated.utility_beta), (4.0, 2.0));
        let untouched = &loaded.states[&8];
        assert_eq!((untouched.utility_alpha, untouched.utility_beta), (1.0, 1.0));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v23_snapshot_without_utility_section_loads_as_prior() {
        let path = scratch_path("utility-compat");
        // Hand-write a V23 container carrying only "states" — exactly what a
        // pre-section writer produced.
        let mut states: HashMap<MemoryId, MemoryState> = HashMap::new();
        let mut st = MemoryState::new(3, [0u8; 32], 1_000);
        st.record_outcome(true, 5.0);
        states.insert(3, st);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FULL_SNAPSHOT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        let name = b"states";
        let body = bincode::serialize(&states).unwrap();
        bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&(body.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&body);
        std::fs::write(&path, &bytes).unwrap();

        let loaded = FullSnapshot::load(&path).unwrap();
        assert!(loaded.utility_posteriors.is_empty());
        // `utility_alpha` is #[serde(skip)], so the written 6.0 is gone and
        // sanitize() must have restored the (1, 1) prior — not 0.
        let s = &loaded.states[&3];
        assert_eq!((s.utility_alpha, s.utility_beta), (1.0, 1.0));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v22_bincode_snapshot_migrates_to_v23() {
        let path = scratch_path("v22-migrate");
        let mut snap = FullSnapshot::empty(11);
        snap.cw_refresh_ts.insert(5, 999);
        // Serialize in the frozen V22 monolithic-bincode layout.
        let leg = LegacyFullSnapshotV22 {
            snapshot_seqno:     snap.snapshot_seqno,
            payloads:           snap.payloads,
            states:             snap.states,
            assoc_edges:        snap.assoc_edges,
            artifacts:          snap.artifacts,
            artifact_paths:     snap.artifact_paths,
            time_idx:           snap.time_idx,
            keyword_idx:        snap.keyword_idx,
            artifact_idx:       snap.artifact_idx,
            triplet_store:      snap.triplet_store,
            symbol_idx:         snap.symbol_idx,
            call_graph:         snap.call_graph,
            code_files:         snap.code_files,
            semantic_idx:       snap.semantic_idx,
            coactivation_stats: snap.coactivation_stats,
            ack_scores:         snap.ack_scores,
            correction_states:  snap.correction_states,
            event_tape:         snap.event_tape,
            decision_tape:      snap.decision_tape,
            turiya_monitor:     snap.turiya_monitor,
            observer_state:     snap.observer_state,
            interaction_ledger: snap.interaction_ledger,
            predicate_store:    snap.predicate_store,
            cw_refresh_ts:      snap.cw_refresh_ts,
        };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FULL_SNAPSHOT_MAGIC_V22.to_le_bytes());
        bytes.extend_from_slice(&bincode::serialize(&leg).unwrap());
        std::fs::write(&path, &bytes).unwrap();

        assert_eq!(FullSnapshot::peek_seqno(&path).unwrap(), 11);
        let loaded = FullSnapshot::load(&path).unwrap();
        assert_eq!(loaded.snapshot_seqno, 11);
        assert_eq!(loaded.cw_refresh_ts.get(&5), Some(&999));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn v4_upgrade_sets_surprise_zero() {
        assert_eq!(legacy_v4_state().upgrade().surprise, 0.0);
    }

    #[test]
    fn v5_upgrade_sets_surprise_zero() {
        assert_eq!(legacy_v5_state().upgrade().surprise, 0.0);
    }

    /// Bincode positional deserialization of a V4-layout byte stream into the
    /// current MemoryState via the upgrade path must yield surprise == 0.0.
    /// `#[serde(default)]` alone does not save bincode (positional); only the
    /// explicit upgrade path inserting `surprise: 0.0` does.
    #[test]
    fn v4_bincode_roundtrip_surprise_zero() {
        let legacy = legacy_v4_state();
        let bytes = bincode::serialize(&legacy).expect("serialize");
        let decoded: LegacyMemoryStateV4 = bincode::deserialize(&bytes).expect("deserialize");
        let upgraded = decoded.upgrade();
        assert_eq!(upgraded.surprise, 0.0);
        assert_eq!(upgraded.strength, 0.8);
    }

    #[test]
    fn v5_bincode_roundtrip_surprise_zero() {
        let legacy = legacy_v5_state();
        let bytes = bincode::serialize(&legacy).expect("serialize");
        let decoded: LegacyMemoryStateV5 = bincode::deserialize(&bytes).expect("deserialize");
        let upgraded = decoded.upgrade();
        assert_eq!(upgraded.surprise, 0.0);
        assert_eq!(upgraded.strength, 0.7);
        assert_eq!(upgraded.tier, 1);
    }
}
