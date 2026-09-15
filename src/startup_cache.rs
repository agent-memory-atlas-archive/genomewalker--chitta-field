//! Optional, disposable startup accelerators. Never part of the snapshot contract.
use std::{io::{self, Write}, path::Path};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use bincode::Options;

const MAGIC: &[u8; 8] = b"CFBOOT01";

pub(crate) fn digest(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }

pub(crate) fn load<T: DeserializeOwned>(path: &Path, key: &[u8; 32]) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 72 || &bytes[..8] != MAGIC || &bytes[8..40] != key
        || bytes[40..72] != digest(&bytes[72..]) { return None; }
    bincode::DefaultOptions::new().with_fixint_encoding()
        .with_limit((bytes.len() - 72) as u64).reject_trailing_bytes()
        .deserialize(&bytes[72..]).ok()
}

pub(crate) fn save<T: Serialize>(path: &Path, key: &[u8; 32], value: &T) -> io::Result<()> {
    let bytes = bincode::serialize(value).map_err(io::Error::other)?;
    let tmp = path.with_extension(format!("{}.tmp", path.extension().unwrap_or_default().to_string_lossy()));
    let result = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(MAGIC)?;
        file.write_all(key)?;
        file.write_all(&digest(&bytes))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() { let _ = std::fs::remove_file(tmp); }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn optional_cache_rejects_stale_truncated_and_corrupted_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cache");
        let key = digest(b"input");
        assert!(load::<Vec<u64>>(&path, &key).is_none());
        save(&path, &key, &vec![1u64, 2, 3]).unwrap();
        assert_eq!(load::<Vec<u64>>(&path, &key).unwrap(), vec![1, 2, 3]);
        assert!(load::<Vec<u64>>(&path, &digest(b"changed")).is_none());
        let bytes = std::fs::read(&path).unwrap();
        for n in [0, 8, 40, 72, bytes.len() - 1] {
            std::fs::write(&path, &bytes[..n]).unwrap();
            assert!(load::<Vec<u64>>(&path, &key).is_none());
        }
        let mut bad = bytes; *bad.last_mut().unwrap() ^= 1;
        std::fs::write(&path, bad).unwrap();
        assert!(load::<Vec<u64>>(&path, &key).is_none());
    }
}

#[cfg(test)]
mod organ_tests {
    use super::*;
    use crate::{organ::{event_tape::EventTape, cdawg::CdawgOrgan}, hdc::EpisodeHdcStore};

    #[test]
    fn organs_roundtrip_and_tape_edits_invalidate_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.organs");
        let mut tape = EventTape::new();
        for i in 0..40 { tape.log("Bash", "file.rs", (i % 3) as u8, 1, i); }
        let mut cdawg = CdawgOrgan::new(); cdawg.rebuild_from_tape(&tape);
        let mut hdc = EpisodeHdcStore::new(); hdc.rebuild(&tape);
        let key = tape.startup_key();
        save(&path, &key, &(&cdawg, &hdc)).unwrap();
        let (restored, mut restored_hdc): (CdawgOrgan, EpisodeHdcStore) = load(&path, &key).unwrap();
        assert_eq!(serde_json::to_value(&cdawg).unwrap(), serde_json::to_value(&restored).unwrap());
        assert_eq!(hdc.recall_hdcbind("tool", "Bash", "entity", 10), restored_hdc.recall_hdcbind("tool", "Bash", "entity", 10));
        hdc.log_episode("Bash", "file.rs", 1); restored_hdc.log_episode("Bash", "file.rs", 1);
        assert_eq!(serde_json::to_value(&hdc).unwrap(), serde_json::to_value(&restored_hdc).unwrap());
        tape.events[0].outcome_class = 2;
        assert_ne!(key, tape.startup_key());
        assert!(load::<(CdawgOrgan, EpisodeHdcStore)>(&path, &tape.startup_key()).is_none());
        let changed = tape.startup_key();
        tape.log("Read", "other.rs", 0, 2, 50);
        assert_ne!(changed, tape.startup_key());
    }
}

/// Rebuild from the tape, rather than copying runtime credit/learning state, so
/// optional cache presence cannot change the historical startup semantics.
pub(crate) fn load_or_rebuild_organs(
    tape: &crate::organ::event_tape::EventTape, path: Option<&Path>,
) -> (crate::organ::cdawg::CdawgOrgan, crate::hdc::EpisodeHdcStore) {
    use crate::{organ::cdawg::CdawgOrgan, hdc::EpisodeHdcStore};
    let key = tape.startup_key();
    let cached: Option<(CdawgOrgan, EpisodeHdcStore)> = path.and_then(|p| load(p, &key));
    if let Some((mut cdawg, hdc)) = cached {
        cdawg.rebuilt = true;
        eprintln!("[chitta-field] event tape organs cache hit=true");
        return (cdawg, hdc);
    }
    let mut cdawg = CdawgOrgan::new(); cdawg.rebuild_from_tape(tape);
    let mut hdc = EpisodeHdcStore::new(); hdc.rebuild(tape);
    if let Some(path) = path {
        if let Err(e) = save(path, &key, &(&cdawg, &hdc)) {
            eprintln!("[chitta-field] optional event tape cache write failed: {e}");
        }
    }
    eprintln!("[chitta-field] event tape organs cache hit=false");
    (cdawg, hdc)
}
