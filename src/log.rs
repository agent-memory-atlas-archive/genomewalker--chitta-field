use crate::error::{FieldError, Result};
use crate::ids::InstanceId;
use crate::ops::Op;
use crc32fast::Hasher as CrcHasher;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const SEGMENT_MAGIC: &[u8; 8] = b"CFLOG001";
pub const SEGMENT_MAGIC_V2: &[u8; 8] = b"CFLOG002";
/// V3 = V2 record format + an 8-byte vector_space_id lineage stamp in the header.
/// Lets replay() fence out segments written in a foreign vector space (model/dim/text-format).
/// NOTE: a pre-V3 binary cannot read V3 segments (rollback hazard — compact_wal before downgrading).
pub const SEGMENT_MAGIC_V3: &[u8; 8] = b"CFLOG003";
pub const MAX_SEGMENT_SIZE: u64 = 256 * 1024 * 1024;

/// SHA256 hash of a record, used for hash-chaining.
pub type ChainHash = [u8; 32];
pub const ZERO_HASH: ChainHash = [0u8; 32];

/// Compute the hash that links this record into the chain.
/// H = SHA256(seqno || op_type || prev_hash || payload)
fn compute_record_hash(seqno: u64, op_type: u8, prev_hash: &ChainHash, payload: &[u8]) -> ChainHash {
    let mut hasher = Sha256::new();
    hasher.update(seqno.to_be_bytes());
    hasher.update([op_type]);
    hasher.update(prev_hash);
    hasher.update(payload);
    hasher.finalize().into()
}

fn hex(h: &ChainHash) -> String {
    h.iter().map(|b| format!("{:02x}", b)).collect::<String>()
}

const OP_PUT_PAYLOAD: u8 = 0;
const OP_UPDATE_STATE: u8 = 1;
const OP_DELETE_MEMORY: u8 = 2;
const OP_ADD_ASSOC_EDGE: u8 = 3;
const OP_UPSERT_ARTIFACT: u8 = 4;
const OP_ADD_TRIPLET: u8 = 5;
const OP_INVALIDATE_TRIPLET: u8 = 6;
const OP_UPSERT_SYMBOL: u8 = 7;
const OP_REMOVE_SYMBOL: u8 = 8;
const OP_ADD_SYM_CALL_EDGE: u8 = 9;
const OP_REMOVE_SYM_CALL_EDGE: u8 = 10;
const OP_UPSERT_CODE_FILE: u8 = 11;
const OP_UPDATE_SPARSE_CODE: u8 = 12;
const OP_DEMOTE_MEMORY: u8 = 13;
const OP_TRAIN_PQ: u8 = 14;
const OP_UPDATE_RESIDUAL_PQ: u8 = 15;
use crate::ops::{
    OP_AGENT_DISABLE, OP_AGENT_UPSERT, OP_ANALYTICS_EVENT, OP_CLEAR_PROJECT, OP_MSG_EVENT,
    OP_SESSION_EVENT, OP_SKILL_DEPRECATE, OP_SKILL_UPLOAD, OP_TASK_EVENT, OP_THEME_EVENT,
    OP_RECORD_RECALL_BATCH, OP_STRENGTHEN_ASSOC_EDGE, OP_TRANSCRIPT_EVENT,
    OP_UPDATE_MEMORY_CONTENT, OP_UPDATE_SYMBOL_DESCRIPTION, OP_USER_MODEL_EVENT,
    OP_ASSERT_CONSTRAINT, OP_RETRACT_CONSTRAINT, OP_CREATE_BRANCH, OP_RESOLVE_BRANCH,
    OP_ADD_TRIGGER, OP_UPDATE_TRIGGER, OP_FIRE_TRIGGER,
    OP_RECORD_SURPRISE, OP_REGISTER_DEBT, OP_UPDATE_DEBT,
    OP_UPDATE_SOURCE_WEIGHT, OP_RECORD_FEEDBACK,
    OP_UPDATE_SURPRISE_CREDIT, OP_UPSERT_WISDOM_CANDIDATE,
    OP_UPDATE_WISDOM_LIFECYCLE, OP_UPDATE_SCORER_MODEL, OP_ATTACH_DEBT_EVIDENCE,
    OP_START_INTERVENTION, OP_ADD_OBSERVATION, OP_CLOSE_INTERVENTION, OP_RECORD_ATTRIBUTION,
    OP_REGISTER_TASK, OP_UPDATE_TASK, OP_ADD_DELEGATION, OP_LINK_EVIDENCE,
    OP_ADD_PROBE, OP_RESOLVE_PROBE, OP_SET_CRITERION,
    OP_UPSERT_WISDOM_LINEAGE, OP_ADJUDICATE_LINEAGE, OP_TRANSITION_LINEAGE,
    OP_RECORD_CHALLENGER, OP_CLOSE_REDERIVE,
    OP_INVALIDATE_TRIPLETS_BY_SOURCE_FILE,
    OP_UPDATE_MEMORY_KIND,
    OP_SYMBOL_EVENT,
    OP_SUPERSEDE_TRIPLET,
    OP_RECORD_OUTCOME,
};

fn op_type_byte(op: &Op) -> u8 {
    match op {
        Op::PutPayload(_) => OP_PUT_PAYLOAD,
        Op::UpdateState(_) => OP_UPDATE_STATE,
        Op::UpdateStateBatch(_) => crate::ops::OP_UPDATE_STATE_BATCH,
        Op::DeleteMemory(_) => OP_DELETE_MEMORY,
        Op::AddAssocEdge(_) => OP_ADD_ASSOC_EDGE,
        Op::UpsertArtifact(_) => OP_UPSERT_ARTIFACT,
        Op::AddTriplet(_) => OP_ADD_TRIPLET,
        Op::InvalidateTriplet(_) => OP_INVALIDATE_TRIPLET,
        Op::UpsertSymbol(_) => OP_UPSERT_SYMBOL,
        Op::RemoveSymbol(_) => OP_REMOVE_SYMBOL,
        Op::AddSymCallEdge(_) => OP_ADD_SYM_CALL_EDGE,
        Op::RemoveSymCallEdge(_) => OP_REMOVE_SYM_CALL_EDGE,
        Op::UpsertCodeFile(_) => OP_UPSERT_CODE_FILE,
        Op::UpdateSparseCode(_) => OP_UPDATE_SPARSE_CODE,
        Op::DemoteMemory(_) => OP_DEMOTE_MEMORY,
        Op::TrainPQ(_) => OP_TRAIN_PQ,
        Op::UpdateResidualPQ(_) => OP_UPDATE_RESIDUAL_PQ,
        Op::SessionEvent(_) => OP_SESSION_EVENT,
        Op::TranscriptEvent(_) => OP_TRANSCRIPT_EVENT,
        Op::TaskEvent(_) => OP_TASK_EVENT,
        Op::UserModelEvent(_) => OP_USER_MODEL_EVENT,
        Op::ThemeEvent(_) => OP_THEME_EVENT,
        Op::AnalyticsEvent(_) => OP_ANALYTICS_EVENT,
        Op::ClearProject(_) => OP_CLEAR_PROJECT,
        Op::UpdateSymbolDescription(_) => OP_UPDATE_SYMBOL_DESCRIPTION,
        Op::UpdateMemoryContent(_) => OP_UPDATE_MEMORY_CONTENT,
        Op::RecordRecallBatch(_) => OP_RECORD_RECALL_BATCH,
        Op::StrengthenAssocEdge(_) => OP_STRENGTHEN_ASSOC_EDGE,
        Op::MsgEvent(_) => OP_MSG_EVENT,
        Op::SkillUpload(_) => OP_SKILL_UPLOAD,
        Op::SkillDeprecate(_) => OP_SKILL_DEPRECATE,
        Op::AgentUpsert(_) => OP_AGENT_UPSERT,
        Op::AgentDisable(_) => OP_AGENT_DISABLE,
        Op::AssertConstraint(_) => OP_ASSERT_CONSTRAINT,
        Op::RetractConstraint(_) => OP_RETRACT_CONSTRAINT,
        Op::CreateBranch(_) => OP_CREATE_BRANCH,
        Op::ResolveBranch(_) => OP_RESOLVE_BRANCH,
        Op::AddTrigger(_) => OP_ADD_TRIGGER,
        Op::UpdateTrigger(_) => OP_UPDATE_TRIGGER,
        Op::FireTrigger(_) => OP_FIRE_TRIGGER,
        Op::RecordSurprise(_) => OP_RECORD_SURPRISE,
        Op::RegisterDebt(_) => OP_REGISTER_DEBT,
        Op::UpdateDebt(_) => OP_UPDATE_DEBT,
        Op::UpdateSourceWeight(_) => OP_UPDATE_SOURCE_WEIGHT,
        Op::RecordFeedback(_) => OP_RECORD_FEEDBACK,
        Op::UpdateSurpriseCredit(_) => OP_UPDATE_SURPRISE_CREDIT,
        Op::UpsertWisdomCandidate(_) => OP_UPSERT_WISDOM_CANDIDATE,
        Op::UpdateWisdomLifecycle(_) => OP_UPDATE_WISDOM_LIFECYCLE,
        Op::UpdateScorerModel(_) => OP_UPDATE_SCORER_MODEL,
        Op::AttachDebtEvidence(_) => OP_ATTACH_DEBT_EVIDENCE,
        Op::StartIntervention(_) => OP_START_INTERVENTION,
        Op::AddObservation(_) => OP_ADD_OBSERVATION,
        Op::CloseIntervention(_) => OP_CLOSE_INTERVENTION,
        Op::RecordAttribution(_) => OP_RECORD_ATTRIBUTION,
        Op::RegisterTask(_) => OP_REGISTER_TASK,
        Op::UpdateTask(_) => OP_UPDATE_TASK,
        Op::AddDelegation(_) => OP_ADD_DELEGATION,
        Op::LinkEvidence(_) => OP_LINK_EVIDENCE,
        Op::AddProbe(_) => OP_ADD_PROBE,
        Op::ResolveProbe(_) => OP_RESOLVE_PROBE,
        Op::SetCriterion(_) => OP_SET_CRITERION,
        Op::UpsertWisdomLineage(_) => OP_UPSERT_WISDOM_LINEAGE,
        Op::AdjudicateLineage(_) => OP_ADJUDICATE_LINEAGE,
        Op::TransitionLineage(_) => OP_TRANSITION_LINEAGE,
        Op::RecordChallenger(_) => OP_RECORD_CHALLENGER,
        Op::CloseRederive(_) => OP_CLOSE_REDERIVE,
        Op::InvalidateTripletsBySourceFile(_) => OP_INVALIDATE_TRIPLETS_BY_SOURCE_FILE,
        Op::UpdateMemoryKind(_) => OP_UPDATE_MEMORY_KIND,
        Op::SymbolEvent(_) => OP_SYMBOL_EVENT,
        Op::SupersedeTriplet(_) => OP_SUPERSEDE_TRIPLET,
        Op::RecordOutcome(_) => OP_RECORD_OUTCOME,
    }
}

pub struct OpLog {
    pub(crate) ablations: crate::ablation::Ablations,
    data_dir: PathBuf,
    instance_id: InstanceId,
    current_segment: BufWriter<File>,
    current_segment_path: PathBuf,
    recoveries: u64,
    // Accepted bytes of only the active segment; ESTALE can make the old fd
    // unreadable. Rotation drops this bounded recovery copy.
    accepted_prefix: Vec<u8>,
    #[cfg(test)]
    fail_next_write: bool,
    current_segment_size: u64,
    next_seqno: u64,
    ops_since_sync: u64,
    sync_count: u64,
    last_sync: Instant,
    sync_interval: Duration,
    /// Hash-chain tip: SHA256 of the most recent V2 record.
    /// Zero for genesis or when only V1 segments exist.
    chain_head: ChainHash,
    /// Compiled vector-space id stamped into new V3 segments; replay() fences out
    /// segments carrying a different stamp (foreign lineage).
    vector_space_id: u64,
}

impl OpLog {
    /// Open the write log for a specific instance.
    /// Finds this instance's last segment to continue appending, or creates a new one.
    pub fn open(data_dir: &Path, instance_id: InstanceId, next_seqno: u64) -> Result<Self> {
        Self::open_with_chain(data_dir, instance_id, next_seqno, ZERO_HASH)
    }

    /// Open with a known chain_head (set by caller after replay).
    pub fn open_with_chain(
        data_dir: &Path,
        instance_id: InstanceId,
        next_seqno: u64,
        chain_head: ChainHash,
    ) -> Result<Self> {
        let seg_dir = data_dir.join("segments");
        fs::create_dir_all(&seg_dir)?;
        let vector_space_id = crate::snapshot::StoreHeader::compiled_vector_space_id();

        let existing = collect_instance_segments(&seg_dir, instance_id)?;

        if let Some(last_path) = existing.last() {
            let size = last_path.metadata()?.len();
            // Continue appending to a chained (V2/V3) segment; V1 forces a new segment.
            // The record format is identical, so a V2 tail can continue under a V3 binary.
            // Lineage guard: never append our (compiled-vsid) ops into a segment stamped with
            // a DIFFERENT vector_space_id. replay() fences foreign-stamped segments, so any ops
            // appended here would be silently dropped on the next restart. A V1/V2/legacy
            // segment carries no stamp (None) and counts as same-lineage (matches replay()).
            let lineage_ok = segment_vector_space_id(last_path)
                .is_none_or(|v| v == vector_space_id);
            if size < MAX_SEGMENT_SIZE && is_chained_segment(last_path) && lineage_ok {
                let f = OpenOptions::new()
                    .append(true)
                    .read(true)
                    .open(last_path)?;
                return Ok(Self {
                    ablations: crate::ablation::Ablations::default(),
                    data_dir: data_dir.to_path_buf(),
                    instance_id,
                    current_segment: BufWriter::new(f),
                    current_segment_path: last_path.clone(),
            recoveries: 0,
                    accepted_prefix: fs::read(&last_path)?,
                    #[cfg(test)]
                    fail_next_write: false,
                    current_segment_size: size,
                    next_seqno,
                    ops_since_sync: 0,
            sync_count: 0,
            last_sync: Instant::now(),
            sync_interval: Duration::from_millis(std::env::var("CHITTA_WAL_SYNC_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(200).max(1)),
                    chain_head,
                    vector_space_id,
                });
            }
        }

        let path = segment_path(data_dir, instance_id, next_seqno);
        let f = create_segment_v3(&path, next_seqno, &chain_head, vector_space_id)?;
        let accepted_prefix = fs::read(&path)?;
        let header_size = V3_HEADER_SIZE as u64;
        Ok(Self {
            ablations: crate::ablation::Ablations::default(),
            data_dir: data_dir.to_path_buf(),
            instance_id,
            current_segment: BufWriter::new(f),
            current_segment_path: path,
            recoveries: 0,
            accepted_prefix,
            #[cfg(test)]
            fail_next_write: false,
            current_segment_size: header_size,
            next_seqno,
            ops_since_sync: 0,
            sync_count: 0,
            last_sync: Instant::now(),
            sync_interval: Duration::from_millis(std::env::var("CHITTA_WAL_SYNC_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(200).max(1)),
            chain_head,
            vector_space_id,
        })
    }

    /// Append an op, returning its assigned seqno.
    /// V2 format: [payload_len:4][seqno:8][op_type:1][prev_hash:32][payload:N][crc32:4]
    pub fn append(&mut self, op: &Op) -> Result<u64> {
        if self.ablations.suppresses(op) { return Ok(0); }
        // A local filesystem lets writes to an unlinked file succeed silently;
        // NFS fails them with ESTALE. Detect deletion through the open handle
        // (fstat, attribute-cached on NFS) rather than a path lookup per append:
        // the per-append stat cost ~10-50 ms on a loaded NFS and every remember
        // does several appends under the exclusive RPC lock (lockprof 2026-09-15:
        // remember held 450-1100 ms).
        if self.segment_vanished() {
            eprintln!("[chitta-field] WAL segment {} vanished — restoring accepted prefix on fresh descriptor", self.current_segment_path.display());
            self.recover_segment()?;
        }
        self.rotate_if_needed()?;

        let seqno = self.next_seqno;
        let op_type = op_type_byte(op);
        let payload =
            rmp_serde::to_vec(op).map_err(|e| FieldError::Serialization(e.to_string()))?;

        let payload_len = payload.len() as u32;
        let seqno_bytes = seqno.to_be_bytes();
        let op_type_bytes = [op_type];
        let prev_hash = self.chain_head;

        // CRC covers seqno + op_type + prev_hash + payload
        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_bytes);
        hasher.update(&op_type_bytes);
        hasher.update(&prev_hash);
        hasher.update(&payload);
        let crc = hasher.finalize();

        // Write V2 record. If the segment file has gone away underneath us
        // (deleted by another process: ESTALE/ENOENT on NFS), open a fresh
        // segment and write there rather than losing every later op.
        let mut record = Vec::with_capacity(4 + 8 + 1 + 32 + payload.len() + 4);
        record.extend_from_slice(&payload_len.to_be_bytes());
        record.extend_from_slice(&seqno_bytes);
        record.extend_from_slice(&op_type_bytes);
        record.extend_from_slice(&prev_hash);
        record.extend_from_slice(&payload);
        record.extend_from_slice(&crc.to_be_bytes());
        if let Err(e) = self.write_record(&record) {
            eprintln!("[chitta-field] WAL append failed on {}: {} — restoring accepted WAL prefix", self.current_segment_path.display(), e);
            self.recover_segment()?;
            self.current_segment.write_all(&record)?;
            self.current_segment.flush()?;
        }

        self.accepted_prefix.extend_from_slice(&record);

        // Advance chain head
        self.chain_head = compute_record_hash(seqno, op_type, &prev_hash, &payload);

        // Appends flush to the OS; the maintenance timer owns group fsync.
        self.ops_since_sync += 1;

        // V2 entry: 4 + 8 + 1 + 32 + payload_len + 4
        let entry_size = 4 + 8 + 1 + 32 + payload.len() as u64 + 4;
        self.current_segment_size += entry_size;
        self.next_seqno += 1;

        Ok(seqno)
    }

    /// Force fsync — call after critical mutations (put_memory, forget, etc.)
    pub fn sync(&mut self) -> Result<()> {
        if self.segment_vanished() || !self.current_segment_path.exists() {
            eprintln!("[chitta-field] WAL segment {} vanished — restoring accepted prefix on fresh descriptor", self.current_segment_path.display());
            self.recover_segment()?;
        }
        self.current_segment.flush()?;
        self.sync_count += 1;
        if let Err(e) = self.current_segment.get_ref().sync_data() {
            // A stale/deleted segment cannot become durable; move to a fresh one
            // so the next appends are, and surface it once per recovery.
            eprintln!("[chitta-field] WAL sync failed on {}: {} — restoring accepted WAL prefix", self.current_segment_path.display(), e);
            self.recover_segment()?;
        }
        self.ops_since_sync = 0;
        self.last_sync = Instant::now();
        Ok(())
    }

    /// Number of detected segment recoveries.
    pub fn recovery_count(&self) -> u64 { self.recoveries }


    /// True when the open segment's inode has no links left (deleted underneath us).
    fn segment_vanished(&self) -> bool {
        use std::os::unix::fs::MetadataExt;
        match self.current_segment.get_ref().metadata() {
            Ok(m) => m.nlink() == 0,
            Err(_) => true,
        }
    }

    /// Install a fresh inode from accepted bytes, never from a possibly stale fd.
    /// Preserve the header/name/chain so acknowledged records remain replayable.
    fn recover_segment(&mut self) -> Result<()> {
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_nanos();
        let tmp = self.current_segment_path.with_extension(format!("recovering.{nonce}"));
        let mut f = OpenOptions::new().read(true).write(true).create_new(true).open(&tmp)?;
        f.write_all(&self.accepted_prefix)?;
        f.sync_all()?;
        eprintln!("[chitta-field] rename from={} to={} reason=wal-prefix-recovery", tmp.display(), self.current_segment_path.display());
        fs::rename(&tmp, &self.current_segment_path)?;
        File::open(self.current_segment_path.parent().unwrap())?.sync_all()?;
        let old = std::mem::replace(&mut self.current_segment, BufWriter::new(f));
        let _ = old.into_parts(); // discard failed append bytes, including on drop
        self.recoveries += 1;
        eprintln!("[chitta-field] recovered WAL segment {}: {} bytes from accepted prefix on fresh descriptor",
                  self.current_segment_path.display(), self.accepted_prefix.len());
        Ok(())
    }

    fn write_record(&mut self, record: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if std::mem::take(&mut self.fail_next_write) {
            return Err(std::io::Error::from_raw_os_error(116));
        }
        self.current_segment.write_all(record)?;
        self.current_segment.flush()
    }

    pub(crate) fn writer_path(&self) -> &Path { &self.current_segment_path }

    pub fn pending_sync_count(&self) -> u64 { self.ops_since_sync }
    pub fn sync_count(&self) -> u64 { self.sync_count }
    pub fn sync_interval(&self) -> Duration { self.sync_interval }
    pub fn next_sync_delay(&self) -> Duration {
        if self.ops_since_sync == 0 { return self.sync_interval; }
        self.sync_interval.saturating_sub(self.last_sync.elapsed()).max(Duration::from_millis(1))
    }

    pub fn sync_if_due(&mut self) -> Result<()> {
        if self.ops_since_sync > 0 && self.last_sync.elapsed() >= self.sync_interval {
            self.sync()?;
        }
        Ok(())
    }

    /// Replay ALL segment files in data_dir/segments/ (from all instances).
    /// Segments are sorted alphabetically — instance_id prefix ensures consistent ordering.
    /// Updates chain_head as V2 records are replayed.
    ///
    /// Each instance maintains an independent chain. When replay crosses an
    /// instance boundary (filename prefix change), chain_head is reset to
    /// ZERO_HASH — otherwise the next instance's first segment would always
    /// trigger a "chain continuity warning" because its stored_head is zero
    /// while the accumulated head from the prior instance is not.
    /// Replay all segments, applying ops in deterministic MERGE order:
    /// `(effective_ts, instance_id, seqno)` — THEORY.md §3. Segment walking
    /// (chain/CRC verification) stays sequential per segment; application is
    /// deferred until all ops are collected, so cross-instance interleavings
    /// converge to the same state regardless of instance-id sort order.
    /// Clock-less ops (op_timestamp == None) inherit the previous ts-bearing
    /// op's time from the same writer (carry-forward), preserving per-writer
    /// order. Memory cost is the buffered op set — bounded in practice by
    /// WAL compaction.
    ///
    /// Returns the per-instance max seqno walked (the WAL coverage vector;
    /// THEORY.md §4).
    pub fn replay<F>(
        &mut self,
        _start_seqno: u64,
        mut f: F,
    ) -> Result<std::collections::BTreeMap<InstanceId, u64>>
    where
        F: FnMut(InstanceId, u64, Op) -> Result<()>,
    {
        let seg_dir = self.data_dir.join("segments");
        let segments = collect_all_segments(&seg_dir)?;
        let mut chain_head = ZERO_HASH;
        let mut current_instance: Option<String> = None;
        let own_prefix = format!("{:08x}", self.instance_id);
        let mut own_chain_head = ZERO_HASH;
        let mut buf: Vec<(i64, InstanceId, u64, Op)> = Vec::new();
        let mut last_ts: std::collections::BTreeMap<InstanceId, i64> = std::collections::BTreeMap::new();
        let mut coverage: std::collections::BTreeMap<InstanceId, u64> = std::collections::BTreeMap::new();
        for seg_path in &segments {
            // Lineage fence: skip segments stamped (V3) with a different vector_space_id —
            // foreign-vector data must not contaminate replay. V1/V2/legacy segments carry
            // no stamp (None) and are treated as same-lineage (always replayed).
            if let Some(seg_vsid) = segment_vector_space_id(seg_path) {
                if seg_vsid != self.vector_space_id {
                    eprintln!(
                        "[chitta-field] WAL lineage fence: skipping foreign segment {:?} (vsid={:#018x} != own {:#018x})",
                        seg_path, seg_vsid, self.vector_space_id
                    );
                    continue;
                }
            }
            let instance = seg_path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.split_once('_'))
                .map(|(prefix, _)| prefix.to_string());
            if instance != current_instance {
                chain_head = ZERO_HASH;
                current_instance = instance.clone();
            }
            let inst_id: InstanceId = instance
                .as_deref()
                .and_then(|p| InstanceId::from_str_radix(p, 16).ok())
                .unwrap_or(0);
            chain_head = replay_segment_chained(seg_path, 0, chain_head, &mut |seqno, op| {
                // Carry-forward effective timestamp: monotone per writer.
                let prev = last_ts.get(&inst_id).copied().unwrap_or(0);
                let eff = crate::ops::op_timestamp(&op).unwrap_or(prev).max(prev);
                last_ts.insert(inst_id, eff);
                let cov = coverage.entry(inst_id).or_insert(0);
                if seqno > *cov { *cov = seqno; }
                buf.push((eff, inst_id, seqno, op));
                Ok(())
            })?;
            if current_instance.as_deref() == Some(own_prefix.as_str()) {
                own_chain_head = chain_head;
            }
        }
        // Only track this instance's chain tip — not the global replay result.
        // Using the cross-instance accumulated hash would cause the first append
        // to write prev_hash = <foreign tip>, triggering a warning on every
        // subsequent restart when the boundary reset sets chain_head back to zero.
        self.chain_head = own_chain_head;

        // Apply in merge order.
        buf.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
        for (_, inst, seqno, op) in buf {
            f(inst, seqno, op)?;
        }
        Ok(coverage)
    }

    /// Current chain tip hash. Zero if only V1 data exists.
    pub fn chain_head(&self) -> ChainHash {
        self.chain_head
    }

    /// Set chain_head externally (e.g. after caller-driven replay).
    pub fn set_chain_head(&mut self, h: ChainHash) {
        self.chain_head = h;
    }

    /// Flush the write buffer to the OS.
    pub fn flush_buf(&mut self) -> Result<()> {
        self.current_segment.flush()?;
        Ok(())
    }

    /// Return the seqno of the last op appended (0 if nothing written yet).
    pub fn last_seqno(&self) -> u64 {
        self.next_seqno.saturating_sub(1)
    }

    /// Advance next_seqno after WAL replay so snapshots get the correct seqno.
    pub fn set_next_seqno(&mut self, seqno: u64) {
        if seqno > self.next_seqno {
            self.next_seqno = seqno;
        }
    }

    fn rotate_if_needed(&mut self) -> Result<()> {
        if self.current_segment_size < MAX_SEGMENT_SIZE {
            return Ok(());
        }
        self.sync()?; // propagate failure before rotating the hash chain
        let new_path = segment_path(&self.data_dir, self.instance_id, self.next_seqno);
        let f = create_segment_v3(&new_path, self.next_seqno, &self.chain_head, self.vector_space_id)?;
        let accepted_prefix = fs::read(&new_path)?;
        self.current_segment = BufWriter::new(f);
        self.accepted_prefix = accepted_prefix;
        self.current_segment_path = new_path;
        self.current_segment_size = V3_HEADER_SIZE as u64;
        Ok(())
    }
}

fn segment_path(data_dir: &Path, instance_id: InstanceId, first_seqno: u64) -> PathBuf {
    data_dir
        .join("segments")
        .join(format!("{:08x}_{:012}.seg", instance_id, first_seqno))
}

/// V1 header: magic(8) + first_seqno(8) = 16
const V1_HEADER_SIZE: usize = 16;
/// V2 header: magic(8) + first_seqno(8) + chain_head(32) = 48
const V2_HEADER_SIZE: usize = 48;
/// V3 header: magic(8) + first_seqno(8) + chain_head(32) + vector_space_id(8) = 56
pub(crate) const V3_HEADER_SIZE: usize = 56;

#[allow(dead_code)] // Retained for V1 backward compatibility
fn create_segment(path: &Path, first_seqno: u64) -> Result<File> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(SEGMENT_MAGIC)?;
    f.write_all(&first_seqno.to_be_bytes())?;
    f.sync_all()?;
    Ok(f)
}

#[allow(dead_code)] // superseded by create_segment_v3; retained for reference/tooling
fn create_segment_v2(path: &Path, first_seqno: u64, chain_head: &ChainHash) -> Result<File> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(SEGMENT_MAGIC_V2)?;
    f.write_all(&first_seqno.to_be_bytes())?;
    f.write_all(chain_head)?;
    f.sync_all()?;
    Ok(f)
}

fn create_segment_v3(path: &Path, first_seqno: u64, chain_head: &ChainHash, vector_space_id: u64) -> Result<File> {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(SEGMENT_MAGIC_V3)?;
    f.write_all(&first_seqno.to_be_bytes())?;
    f.write_all(chain_head)?;
    f.write_all(&vector_space_id.to_be_bytes())?;
    f.sync_all()?;
    File::open(path.parent().unwrap())?.sync_all()?;
    Ok(f)
}

/// Read the segment magic and return the version (1, 2, or 3).
fn segment_version(magic: &[u8; 8]) -> Option<u8> {
    if magic == SEGMENT_MAGIC { Some(1) }
    else if magic == SEGMENT_MAGIC_V2 { Some(2) }
    else if magic == SEGMENT_MAGIC_V3 { Some(3) }
    else { None }
}

/// True for V2 or V3 (chained) segments — both safe to continue appending records to
/// (the record format is identical; only the header differs).
fn is_chained_segment(path: &Path) -> bool {
    let mut f = match File::open(path) { Ok(f) => f, Err(_) => return false };
    let mut magic = [0u8; 8];
    if f.read_exact(&mut magic).is_err() { return false; }
    &magic == SEGMENT_MAGIC_V2 || &magic == SEGMENT_MAGIC_V3
}

/// Read a V3 segment's vector_space_id lineage stamp. None for V1/V2/legacy/unreadable
/// segments — replay treats those as same-lineage (always replayed).
pub(crate) fn segment_vector_space_id(path: &Path) -> Option<u64> {
    let mut f = File::open(path).ok()?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != SEGMENT_MAGIC_V3 { return None; }
    let mut skip = [0u8; 40]; // first_seqno(8) + chain_head(32)
    f.read_exact(&mut skip).ok()?;
    let mut vbuf = [0u8; 8];
    f.read_exact(&mut vbuf).ok()?;
    Some(u64::from_be_bytes(vbuf))
}

fn is_valid_segment_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".seg") else { return false; };
    let Some((instance_hex, seqno)) = stem.split_once('_') else { return false; };
    instance_hex.len() == 8
        && instance_hex.chars().all(|c| c.is_ascii_hexdigit())
        && seqno.len() == 12
        && seqno.chars().all(|c| c.is_ascii_digit())
}

/// All .seg files in directory matching the canonical naming pattern, sorted.
fn collect_all_segments(seg_dir: &Path) -> Result<Vec<PathBuf>> {
    if !seg_dir.exists() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = fs::read_dir(seg_dir)?
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str())?;
            if is_valid_segment_name(name) {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Only .seg files belonging to this instance (by filename prefix).
fn collect_instance_segments(seg_dir: &Path, instance_id: InstanceId) -> Result<Vec<PathBuf>> {
    let prefix = format!("{:08x}_", instance_id);
    let all = collect_all_segments(seg_dir)?;
    Ok(all
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(&prefix))
                .unwrap_or(false)
        })
        .collect())
}

/// Collect all segment files in `data_dir/segments/` not owned by `own_instance_id`.
pub fn collect_foreign_segments(
    data_dir: &Path,
    own_instance_id: InstanceId,
) -> Result<Vec<PathBuf>> {
    let seg_dir = data_dir.join("segments");
    let own_prefix = format!("{:08x}_", own_instance_id);
    let all = collect_all_segments(&seg_dir)?;
    Ok(all
        .into_iter()
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| !n.starts_with(&own_prefix))
                .unwrap_or(false)
        })
        .collect())
}

/// Replay a single segment file starting from `byte_offset`.
/// `byte_offset=0` reads from the start (parses the 16-byte header).
/// Non-zero offset seeks directly to that position (header already consumed).
/// Returns the byte offset past the last successfully decoded entry.
/// CRC mismatches or truncated reads at end-of-file are treated as
/// in-progress writes by a concurrent peer and simply stop the scan.
pub fn replay_from_offset<F>(path: &Path, byte_offset: u64, mut f: F) -> Result<u64>
where
    F: FnMut(u64, Op) -> Result<()>,
{
    use std::io::Seek;

    let mut file = BufReader::with_capacity(64 * 1024, std::fs::File::open(path)?);

    // Determine version from magic
    let (version, header_len) = if byte_offset == 0 {
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return Ok(0);
        }
        let ver = segment_version(&magic).ok_or_else(|| {
            FieldError::Manifest(format!("bad segment magic in {}", path.display()))
        })?;
        let mut _buf = [0u8; 8];
        let _ = file.read_exact(&mut _buf);
        if ver == 2 {
            // Skip chain_head in V2 header
            let mut _chain = [0u8; 32];
            let _ = file.read_exact(&mut _chain);
            (ver, V2_HEADER_SIZE as u64)
        } else if ver == 3 {
            // Skip chain_head(32) + vector_space_id(8) in V3 header
            let mut _rest = [0u8; 40];
            let _ = file.read_exact(&mut _rest);
            (ver, V3_HEADER_SIZE as u64)
        } else {
            (ver, V1_HEADER_SIZE as u64)
        }
    } else {
        // When resuming from offset, peek magic to know version
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() {
            return Ok(byte_offset);
        }
        let ver = segment_version(&magic).unwrap_or(1);
        file.seek(std::io::SeekFrom::Start(byte_offset))?;
        (ver, byte_offset)
    };

    let mut cursor = if byte_offset == 0 { header_len } else { byte_offset };

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FieldError::Io(e)),
        }
        let payload_len = u32::from_be_bytes(len_buf) as usize;
        let prev_hash_size: u64 = if version >= 2 { 32 } else { 0 };
        let entry_size = 4u64 + 8 + 1 + prev_hash_size + payload_len as u64 + 4;

        let mut seqno_buf = [0u8; 8];
        if file.read_exact(&mut seqno_buf).is_err() {
            break;
        }
        let seqno = u64::from_be_bytes(seqno_buf);

        let mut op_type_buf = [0u8; 1];
        if file.read_exact(&mut op_type_buf).is_err() {
            break;
        }

        // V2/V3: read prev_hash
        let prev_hash = if version >= 2 {
            let mut h = [0u8; 32];
            if file.read_exact(&mut h).is_err() { break; }
            h
        } else {
            ZERO_HASH
        };

        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            break;
        }

        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() {
            break;
        }
        let stored_crc = u32::from_be_bytes(crc_buf);

        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
        if version >= 2 {
            hasher.update(&prev_hash);
        }
        hasher.update(&payload);
        if hasher.finalize() != stored_crc {
            // Partial write from a concurrent peer — stop here.
            break;
        }

        let op: Op = match rmp_serde::from_slice(&payload) {
            Ok(op) => op,
            Err(_) => break,
        };

        cursor += entry_size;
        f(seqno, op)?;
    }

    Ok(cursor)
}

/// A torn tail (power loss mid-append) leaves the segment's final record
/// incomplete. Every committed record before it is intact, so bricking the
/// store would lose nothing and cost everything: truncate at the last good
/// offset, warn, and continue. Interior corruption (a bad record with valid
/// records after it) is NOT a torn write and stays fatal.
fn truncate_torn_tail(path: &Path, good_offset: u64, seqno: u64) -> Result<()> {
    eprintln!(
        "[chitta-field] torn WAL tail in {}: record at byte {} (seqno {}) incomplete — \
         truncating to last good record and continuing",
        path.display(),
        good_offset,
        seqno
    );
    let f = OpenOptions::new().write(true).open(path)?;
    f.set_len(good_offset)?;
    f.sync_all()?;
    Ok(())
}

/// Replay a segment, tracking chain hashes for V2 segments.
/// Returns the chain_head after replaying all records.
fn replay_segment_chained<F>(
    path: &Path,
    start_seqno: u64,
    mut chain_head: ChainHash,
    f: &mut F,
) -> Result<ChainHash>
where
    F: FnMut(u64, Op) -> Result<()>,
{
    use std::io::Seek;
    let raw = File::open(path)?;
    let file_len = raw.metadata()?.len();
    // NFS must not service a read for every record field. Bound buffering per
    // reader while retaining CRC, chain checks and exact torn-tail offsets.
    let mut file = BufReader::with_capacity(64 * 1024, raw);

    let mut magic = [0u8; 8];
    if file.read_exact(&mut magic).is_err() {
        return Ok(chain_head);
    }
    let version = segment_version(&magic).ok_or_else(|| {
        FieldError::Manifest(format!("bad segment magic in {}", path.display()))
    })?;

    let mut _first_seqno_buf = [0u8; 8];
    if file.read_exact(&mut _first_seqno_buf).is_err() {
        return Ok(chain_head);
    }

    // V2/V3: read stored chain_head from header and verify continuity
    if version >= 2 {
        let mut stored_head = [0u8; 32];
        if file.read_exact(&mut stored_head).is_err() {
            return Ok(chain_head);
        }
        // V3 header carries an 8-byte vector_space_id after chain_head — consume it.
        if version == 3 {
            let mut _vsid = [0u8; 8];
            if file.read_exact(&mut _vsid).is_err() {
                return Ok(chain_head);
            }
        }
        // Verify segment continuity: stored head must match incoming chain_head.
        // Skip when chain_head is ZERO (instance boundary reset) — mismatch there is expected.
        if stored_head != chain_head && chain_head != ZERO_HASH {
            eprintln!(
                "[chitta-field] chain continuity warning in {}: expected {}, segment has {}",
                path.display(),
                &hex(&chain_head)[..16],
                &hex(&stored_head)[..16],
            );
        }
    }

    loop {
        // Offset of this record's start — the truncation point if the record
        // turns out to be torn. An EOF mid-record can only be the file tail,
        // so every read_exact failure below is a torn-tail condition.
        let record_start = file.stream_position().map_err(FieldError::Io)?;

        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(FieldError::Io(e)),
        }
        let payload_len = u32::from_be_bytes(len_buf) as usize;

        let mut seqno_buf = [0u8; 8];
        match file.read_exact(&mut seqno_buf) {
            Ok(()) => {}
            Err(_) => {
                truncate_torn_tail(path, record_start, 0)?;
                break;
            }
        }
        let seqno = u64::from_be_bytes(seqno_buf);

        let mut op_type_buf = [0u8; 1];
        if file.read_exact(&mut op_type_buf).is_err() {
            truncate_torn_tail(path, record_start, seqno)?;
            break;
        }
        let op_type = op_type_buf[0];

        // V2/V3: read prev_hash from record
        let prev_hash = if version >= 2 {
            let mut h = [0u8; 32];
            if file.read_exact(&mut h).is_err() {
                truncate_torn_tail(path, record_start, seqno)?;
                break;
            }
            h
        } else {
            ZERO_HASH
        };

        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            truncate_torn_tail(path, record_start, seqno)?;
            break;
        }

        let mut crc_buf = [0u8; 4];
        if file.read_exact(&mut crc_buf).is_err() {
            truncate_torn_tail(path, record_start, seqno)?;
            break;
        }
        let stored_crc = u32::from_be_bytes(crc_buf);

        // CRC verification (V2/V3 include prev_hash in CRC)
        let mut hasher = CrcHasher::new();
        hasher.update(&seqno_buf);
        hasher.update(&op_type_buf);
        if version >= 2 {
            hasher.update(&prev_hash);
        }
        hasher.update(&payload);
        let computed_crc = hasher.finalize();

        if computed_crc != stored_crc {
            // A CRC failure on the file's FINAL record is a torn write (e.g.
            // garbage flushed at the tail), not interior corruption — recover.
            let at_eof = file.stream_position().map_err(FieldError::Io)? >= file_len;
            if at_eof {
                truncate_torn_tail(path, record_start, seqno)?;
                break;
            }
            return Err(FieldError::CrcMismatch {
                seqno,
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        // V2: verify chain integrity (warn on mismatch, don't fail — cross-instance segments have independent chains).
        // Skip warning when chain_head is ZERO: we're at an instance boundary reset, so any prev_hash
        // mismatch is either correct (genesis) or legacy bad data from a prior bug where replay()
        // propagated a foreign instance's chain tip into the first append of a new instance.
        if version >= 2 {
            if prev_hash != chain_head && chain_head != ZERO_HASH {
                eprintln!(
                    "[chitta-field] chain record warning at seqno {}: expected {}, record has {} — resetting chain",
                    seqno,
                    &hex(&chain_head)[..16],
                    &hex(&prev_hash)[..16],
                );
            }
            chain_head = compute_record_hash(seqno, op_type, &prev_hash, &payload);
        }

        if seqno < start_seqno {
            continue;
        }

        let op: Op = rmp_serde::from_slice(&payload).map_err(|e| FieldError::CorruptLog {
            seqno,
            reason: format!("msgpack decode failed: {}", e),
        })?;

        if op_type_byte(&op) != op_type {
            return Err(FieldError::CorruptLog {
                seqno,
                reason: format!(
                    "op_type discriminant mismatch: stored {}, decoded {}",
                    op_type,
                    op_type_byte(&op)
                ),
            });
        }

        f(seqno, op)?;
    }
    Ok(chain_head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Op, StateDeltaOp};
    use std::sync::atomic::{AtomicU32, Ordering};

    // Unique temp-dir helper — avoids pulling the `tempfile` dev-dep (which
    // transitively drags getrandom/aho-corasick into the build graph).
    struct ScratchDir(PathBuf);
    impl ScratchDir {
        fn new(tag: &str) -> Self {
            static SEQ: AtomicU32 = AtomicU32::new(0);
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "chitta-field-test-{}-{}-{}",
                tag,
                std::process::id(),
                seq
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn make_op(memory_id: u64) -> Op {
        Op::UpdateState(StateDeltaOp {
            memory_id,
            strength_delta: Some(0.1),
            confidence_delta: None,
            decay_rate: None,
            touch: true,
            pin: None,
            op_ts_ms: 0,
            status: None,
            epistemic_status: None,
            staged: None,
            invalidated_by: None,
        })
    }

    // Regression for the chain-continuity warning that fired O(segments)
    // times on every daemon startup when the data dir contained segments
    // from more than one instance. Each instance owns an independent chain;
    // replay() MUST reset chain_head to ZERO_HASH at an instance-id prefix
    // boundary, otherwise the next instance's first segment is compared
    // against the prior instance's accumulated hash.
    //
    // Before the fix, this test produced ~1 warning per appended op on stderr.
    // After the fix, the replay is silent and every op is surfaced exactly once.
    fn only_segment(data_dir: &Path) -> PathBuf {
        let seg_dir = data_dir.join("segments");
        let mut segs: Vec<PathBuf> = fs::read_dir(&seg_dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        segs.sort();
        assert_eq!(segs.len(), 1, "expected exactly one segment");
        segs.pop().unwrap()
    }

    #[test]
    fn stale_write_recovers_without_reading_old_descriptor() {
        let tmp = ScratchDir::new("stale-write");
        let mut log = OpLog::open(tmp.path(), 0x7788, 1).unwrap();
        for id in 1..=5 { log.append(&make_op(id)).unwrap(); }
        // Replace the old fd with a write-only descriptor. Recovery cannot read
        // this fd (as with ESTALE), and the injected write returns NFS errno 116.
        let fd = OpenOptions::new().write(true).open(log.writer_path()).unwrap();
        let old = std::mem::replace(&mut log.current_segment, BufWriter::new(fd));
        let _ = old.into_parts();
        log.fail_next_write = true;
        for id in 6..=12 { log.append(&make_op(id)).unwrap(); }
        log.sync().unwrap();
        assert_eq!(log.recovery_count(), 1);
        drop(log);
        let mut reopened = OpLog::open(tmp.path(), 0x8899, 1).unwrap();
        let mut seen = Vec::new();
        reopened.replay(0, |_, _, op| {
            if let Op::UpdateState(s) = op { seen.push(s.memory_id); }
            Ok(())
        }).unwrap();
        assert_eq!(seen, (1..=12).collect::<Vec<_>>());
    }

    #[test]
    fn chaos_unlinked_segment_recovers_full_prefix_and_suffix() {
        let tmp = ScratchDir::new("unlinked-segment");
        let mut log = OpLog::open(tmp.path(), 0x7788, 0).unwrap();
        for id in 1..=5 { log.append(&make_op(id)).unwrap(); }
        log.sync().unwrap();
        let next = log.next_seqno;
        let chain = log.chain_head;
        drop(log);
        let mut log = OpLog::open_with_chain(tmp.path(), 0x7788, next, chain).unwrap();
        fs::remove_file(&log.current_segment_path).unwrap();
        for id in 6..=12 { log.append(&make_op(id)).unwrap(); }
        log.sync().unwrap();
        assert_eq!(log.recovery_count(), 1);
        drop(log);
        let mut reopened = OpLog::open(tmp.path(), 0x8899, 0).unwrap();
        for _ in 0..2 {
            let mut seen = Vec::new();
            reopened.replay(0, |_, _, op| {
                if let Op::UpdateState(s) = op { seen.push(s.memory_id); }
                Ok(())
            }).unwrap();
            assert_eq!(seen, (1..=12).collect::<Vec<_>>());
        }
    }

    // Power loss mid-append leaves an incomplete final record. Replay must
    // recover every committed op, truncate the torn tail, and keep the store
    // openable — a torn tail must never brick startup.
    #[test]
    fn torn_tail_is_truncated_and_replay_continues() {
        let tmp = ScratchDir::new("torn-tail");
        let data_dir = tmp.path();

        {
            let mut log = OpLog::open(data_dir, 0x3333_3333, 0).unwrap();
            log.append(&make_op(1)).unwrap();
            log.append(&make_op(2)).unwrap();
            log.append(&make_op(3)).unwrap();
            log.flush_buf().unwrap();
        }

        // Tear the tail: chop 5 bytes off the last record.
        let seg = only_segment(data_dir);
        let len = fs::metadata(&seg).unwrap().len();
        let f = OpenOptions::new().write(true).open(&seg).unwrap();
        f.set_len(len - 5).unwrap();
        f.sync_all().unwrap();

        // Replay recovers ops 1 and 2; the torn record 3 is dropped.
        let mut seen = Vec::new();
        let mut log = OpLog::open(data_dir, 0x3333_3333, 0).unwrap();
        log.replay(0, |_inst, _seq, op| {
            if let Op::UpdateState(d) = op { seen.push(d.memory_id); }
            Ok(())
        })
        .expect("torn tail must not fail replay");
        assert_eq!(seen, vec![1, 2]);

        // The segment was repaired: a second replay is clean and complete.
        let mut seen2 = Vec::new();
        log.replay(0, |_inst, _seq, op| {
            if let Op::UpdateState(d) = op { seen2.push(d.memory_id); }
            Ok(())
        })
        .unwrap();
        assert_eq!(seen2, vec![1, 2]);
        assert!(fs::metadata(&seg).unwrap().len() < len - 5 + 1);
    }

    // Interior corruption (bad record with valid records after it) is NOT a
    // torn write and must remain fatal.
    #[test]
    fn interior_corruption_still_fails_replay() {
        let tmp = ScratchDir::new("interior-corrupt");
        let data_dir = tmp.path();

        {
            let mut log = OpLog::open(data_dir, 0x4444_4444, 0).unwrap();
            log.append(&make_op(1)).unwrap();
            log.append(&make_op(2)).unwrap();
            log.append(&make_op(3)).unwrap();
            log.flush_buf().unwrap();
        }

        // Flip a byte inside record 1's payload, leaving records 2 and 3
        // intact after it. V3 header = magic(8)+first_seqno(8)+chain_head(32)
        // +vsid(8) = 56 bytes; record layout = len(4)+seqno(8)+op_type(1)
        // +prev_hash(32)+payload+crc(4).
        let seg = only_segment(data_dir);
        let mut bytes = fs::read(&seg).unwrap();
        let len1 = u32::from_be_bytes(bytes[56..60].try_into().unwrap()) as usize;
        let payload_off = 56 + 4 + 8 + 1 + 32;
        bytes[payload_off + len1 / 2] ^= 0xFF;
        fs::write(&seg, &bytes).unwrap();

        let mut log = OpLog::open(data_dir, 0x4444_4444, 0).unwrap();
        let result = log.replay(0, |_inst, _seq, _op| Ok(()));
        assert!(result.is_err(), "interior corruption must fail replay");
    }

    #[test]
    fn replay_across_independent_instance_chains() {
        let tmp = ScratchDir::new("replay-chains");
        let data_dir = tmp.path();

        // Instance A: two ops.
        {
            let mut log_a = OpLog::open(data_dir, 0x1111_1111, 0).unwrap();
            log_a.append(&make_op(1)).unwrap();
            log_a.append(&make_op(2)).unwrap();
            log_a.flush_buf().unwrap();
        }

        // Instance B: two ops, independent chain, same data_dir.
        {
            let mut log_b = OpLog::open(data_dir, 0x2222_2222, 0).unwrap();
            log_b.append(&make_op(3)).unwrap();
            log_b.append(&make_op(4)).unwrap();
            log_b.flush_buf().unwrap();
        }

        // Fresh instance C replays everything.
        let mut log_c = OpLog::open(data_dir, 0x3333_3333, 0).unwrap();
        let mut seen: Vec<u64> = Vec::new();
        log_c
            .replay(0, |_inst, _seqno, op| {
                if let Op::UpdateState(s) = op {
                    seen.push(s.memory_id);
                }
                Ok(())
            })
            .unwrap();

        // All four ops surfaced in deterministic (instance-prefix) order.
        // Each instance is a separate chain; replay must cross the prefix
        // boundary without corrupting chain_head and without aborting.
        assert_eq!(seen, vec![1, 2, 3, 4]);
    }
}

#[cfg(test)]
mod group_commit_tests {
    use super::*;

    fn op(id: u64) -> Op {
        Op::DeleteMemory(crate::ops::DeleteMemoryOp { memory_id: id, deleted_at_ms: id as i64 })
    }

    #[test]
    fn append_count_never_triggers_sync_but_timer_and_boundary_do() {
        let dir = tempfile::tempdir().unwrap();
        let mut log = OpLog::open(dir.path(), 42, 1).unwrap();
        for id in 1..=100 { log.append(&op(id)).unwrap(); }
        assert_eq!(log.pending_sync_count(), 100);
        assert_eq!(log.sync_count(), 0);
        log.last_sync = Instant::now() - log.sync_interval;
        log.sync_if_due().unwrap();
        assert_eq!(log.sync_count(), 1);
        assert_eq!(log.pending_sync_count(), 0);
        log.sync_if_due().unwrap();
        assert_eq!(log.sync_count(), 1, "idle timer never fsyncs");
        log.append(&op(101)).unwrap();
        log.sync().unwrap();
        assert_eq!(log.sync_count(), 2);
        assert_eq!(log.pending_sync_count(), 0);
    }

    #[test]
    #[ignore = "subprocess fixture launched by kill9_replays_durable_prefix_and_unsynced_tail"]
    fn crash_writer() {
        let Some(path) = std::env::var_os("CHITTA_CRASH_TEST_DIR") else { return; };
        let dir = PathBuf::from(path);
        let mut log = OpLog::open(&dir, 42, 1).unwrap();
        for id in 1..=8 { log.append(&op(id)).unwrap(); }
        log.sync().unwrap();
        for id in 9..=13 { log.append(&op(id)).unwrap(); }
        fs::write(dir.join("ready"), log.current_segment_size.to_string()).unwrap();
        loop { std::thread::park(); }
    }

    #[test]
    #[cfg(unix)]
    fn kill9_replays_durable_prefix_and_unsynced_tail() {
        let dir = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "log::group_commit_tests::crash_writer", "--ignored"])
            .env("CHITTA_CRASH_TEST_DIR", dir.path())
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null())
            .spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while !dir.path().join("ready").exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        child.kill().unwrap(); // SIGKILL on Unix, never a live daemon.
        let status = child.wait().unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(9));
        assert!(dir.path().join("ready").exists());
        let mut log = OpLog::open(dir.path(), 42, 1).unwrap();
        let mut seen = Vec::new();
        log.replay(0, |_, _, op| { if let Op::DeleteMemory(d) = op { seen.push(d.memory_id); } Ok(()) }).unwrap();
        assert!(seen.len() >= 8 && seen.len() <= 13);
        assert_eq!(seen, (1..=seen.len() as u64).collect::<Vec<_>>());
        // SIGKILL does not evict the OS cache. Also model loss of part of the
        // unsynced tail, verifying repair and subsequent hash-chain appends.
        let path = log.current_segment_path.clone();
        drop(log);
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(file.metadata().unwrap().len() - 5).unwrap();
        let mut log = OpLog::open(dir.path(), 42, 1).unwrap();
        let mut seqs = Vec::new();
        log.replay(0, |_, seq, _| { seqs.push(seq); Ok(()) }).unwrap();
        assert_eq!(seqs, (1..=12).collect::<Vec<_>>());
        log.set_next_seqno(13);
        log.append(&op(14)).unwrap();
        log.sync().unwrap();
        let mut count = 0;
        log.replay(0, |_, _, _| { count += 1; Ok(()) }).unwrap();
        assert_eq!(count, 13);
    }
}
