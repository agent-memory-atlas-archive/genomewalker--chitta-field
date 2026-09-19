use crate::error::{FieldError, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const MANIFEST_MAGIC: &str = "CHITTA_FIELD_MANIFEST_V1";
pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    pub path: String,
    pub first_seqno: u64,
    pub last_seqno: u64,
    pub size_bytes: u64,
}

/// One file of a committed snapshot family, identified by name (relative to
/// data_dir) and expected size. Size mismatch ⇒ the family is not trusted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRef {
    pub name: String,
    pub size_bytes: u64,
}

/// A committed snapshot family: the `.snapshot` plus every sidecar that
/// existed when the manifest was written. The manifest write is the single
/// commit point — a crash anywhere during the multi-file save leaves the
/// previous generation's manifest intact and the half-written family
/// unreferenced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSet {
    pub snapshot: FileRef,
    pub sidecars: Vec<FileRef>,
    pub snapshot_seqno: u64,
    /// Per-writer WAL coverage: instance-id hex → max seqno of that writer's
    /// ops included in this snapshot (THEORY.md §4). Replay skips exactly the
    /// ops this vector dominates; WAL pruning may delete exactly the segments
    /// it dominates. Empty for pre-v2.2 manifests (scalar-seqno fallback).
    #[serde(default)]
    pub covered: std::collections::BTreeMap<String, u64>,
    /// Family-bound WAL certificates; absent in legacy manifests.
    #[serde(default)]
    pub segments: Vec<SegmentInfo>,
    #[serde(default)]
    pub cortical: Option<FileRef>,
    #[serde(default)]
    pub cortical_covered: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    pub vector_space_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub magic: String,
    pub generation: u64,
    pub store_uuid: String,
    pub format_version: u32,
    pub embedding_model: String,
    pub embedding_dim: u16,

    pub next_memory_id: u64,
    pub next_artifact_id: u64,
    pub last_seqno: u64,
    pub clean_shutdown: bool,

    pub segments: Vec<SegmentInfo>,
    /// Latest committed family of the most recent saver (back-compat; also
    /// present as an entry in `families`).
    pub checkpoints: Option<CheckpointSet>,
    /// Version vector of committed families: saver instance hex → its latest
    /// committed family (THEORY.md §4). Each entry is owned by one writer and
    /// self-heals on that writer's next save; `load()` joins both slots
    /// per-entry, so concurrent commits merge instead of shadowing.
    #[serde(default)]
    pub families: std::collections::BTreeMap<String, CheckpointSet>,
}

impl Manifest {
    pub fn new_empty(embedding_model: &str, embedding_dim: u16) -> Self {
        Self {
            magic: MANIFEST_MAGIC.to_string(),
            generation: 0,
            store_uuid: generate_uuid(),
            format_version: FORMAT_VERSION,
            embedding_model: embedding_model.to_string(),
            embedding_dim,
            next_memory_id: 1,
            next_artifact_id: 1,
            last_seqno: 0,
            clean_shutdown: false,
            segments: Vec::new(),
            checkpoints: None,
            families: std::collections::BTreeMap::new(),
        }
    }

    /// Validate committed families against the filesystem (every file exists
    /// with the recorded size) and return the snapshot path of the best
    /// validating one — highest snapshot_seqno across `families` plus the
    /// legacy `checkpoints` entry. None means the caller should fall back to
    /// fence-based selection.
    pub fn validated_snapshot_path(&self, data_dir: &Path) -> Option<PathBuf> {
        let mut candidates: Vec<&CheckpointSet> =
            self.families.values().chain(self.checkpoints.iter()).collect();
        candidates.sort_by(|a, b| b.snapshot_seqno.cmp(&a.snapshot_seqno));
        candidates.dedup_by(|a, b| a.snapshot.name == b.snapshot.name);
        let ok = |fr: &FileRef| -> bool {
            fs::metadata(data_dir.join(&fr.name))
                .map(|m| m.len() == fr.size_bytes)
                .unwrap_or(false)
        };
        for cp in candidates {
            if ok(&cp.snapshot) && cp.sidecars.iter().all(|fr| ok(fr)) {
                return Some(data_dir.join(&cp.snapshot.name));
            }
            eprintln!(
                "[chitta-field] manifest family {} failed validation — trying next",
                cp.snapshot.name
            );
        }
        None
    }

    /// Load from data_dir — joins both slots. The scalar fields come from the
    /// higher generation; `families` is joined per-entry (the entry with the
    /// larger snapshot_seqno wins), so a concurrent writer's commit shadowed
    /// at the slot level is still recovered from the other slot.
    pub fn load(data_dir: &Path) -> Result<Option<Self>> {
        let slot1 = manifest_path(data_dir, 1);
        let slot2 = manifest_path(data_dir, 2);

        let m1 = try_load_slot(&slot1);
        let m2 = try_load_slot(&slot2);

        match (m1, m2) {
            (None, None) => Ok(None),
            (Some(m), None) => Ok(Some(m)),
            (None, Some(m)) => Ok(Some(m)),
            (Some(a), Some(b)) => {
                let (mut newer, older) = if a.generation >= b.generation { (a, b) } else { (b, a) };
                for (inst, fam) in older.families {
                    match newer.families.get(&inst) {
                        Some(existing) if existing.snapshot_seqno >= fam.snapshot_seqno => {}
                        _ => {
                            newer.families.insert(inst, fam);
                        }
                    }
                }
                Ok(Some(newer))
            }
        }
    }

    /// Atomically save to the inactive manifest slot.
    /// Writes to MANIFEST.{slot}.tmp, fsyncs, renames, then fsyncs the directory.
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        let slot = self.slot();
        let final_path = manifest_path(data_dir, slot);
        let tmp_path = data_dir.join(format!("MANIFEST.{}.tmp", slot));

        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| FieldError::Manifest(format!("serialize failed: {}", e)))?;

        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp_path)?;
            f.write_all(&json)?;
            f.sync_all()?;
        }

        fs::rename(&tmp_path, &final_path)?;

        // Fsync the directory to persist the rename.
        fsync_dir(data_dir)?;

        Ok(())
    }

    /// Returns which slot this generation writes to (1 or 2).
    /// Even generation → slot 1, odd generation → slot 2.
    fn slot(&self) -> u8 {
        if self.generation % 2 == 0 {
            1
        } else {
            2
        }
    }
}

fn manifest_path(data_dir: &Path, slot: u8) -> PathBuf {
    data_dir.join(format!("MANIFEST.{}", slot))
}

fn try_load_slot(path: &Path) -> Option<Manifest> {
    let mut f = File::open(path).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let m: Manifest = serde_json::from_slice(&buf).ok()?;
    if m.magic != MANIFEST_MAGIC {
        return None;
    }
    if m.format_version != FORMAT_VERSION {
        return None;
    }
    Some(m)
}

fn fsync_dir(dir: &Path) -> Result<()> {
    let f = File::open(dir)?;
    f.sync_all()?;
    Ok(())
}

/// Generate a UUID v4-like string from /dev/urandom bytes formatted as hex.
/// Format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (RFC 4122 v4 layout).
fn generate_uuid() -> String {
    let mut buf = [0u8; 16];
    // Read from /dev/urandom; fall back to a timestamp-based seed on error.
    if let Ok(mut f) = File::open("/dev/urandom") {
        let _ = f.read_exact(&mut buf);
    } else {
        // Fallback: use std::time for some entropy.
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let t_bytes = t.to_le_bytes();
        buf[..16].copy_from_slice(&t_bytes);
    }
    // Set version bits (v4) and variant bits (RFC 4122).
    buf[6] = (buf[6] & 0x0f) | 0x40;
    buf[8] = (buf[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        buf[0], buf[1], buf[2], buf[3],
        buf[4], buf[5],
        buf[6], buf[7],
        buf[8], buf[9],
        buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
    )
}
