//! Conservative building blocks for manifest-certified WAL skipping.
//! Replay/pruning integration must bind BOTH loaded snapshots to these vectors.
use crate::manifest::SegmentInfo;
use std::collections::BTreeMap;
use std::path::Path;

fn segment_identity(name: &str) -> Option<(u32, u64)> {
    let (writer, first) = name.strip_suffix(".seg")?.split_once('_')?;
    if writer.len() != 8
        || first.len() != 12
        || !writer.bytes().all(|b| b.is_ascii_hexdigit())
        || !first.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some((u32::from_str_radix(writer, 16).ok()?, first.parse().ok()?))
}

/// Scan a caller-identified sealed segment. Never certify the pinned writer,
/// an incomplete record, a foreign lineage, or a file changed during scanning.
/// V1/V2 inherit the store lineage, as replay does. The caller must establish
/// sealing; unchanged length alone cannot prove it.
pub(crate) fn scan_sealed_segment(
    path: &Path,
    writer_path: &Path,
    vector_space_id: u64,
) -> Option<SegmentInfo> {
    use std::io::Read;
    if path == writer_path {
        return None;
    }
    let name = path.file_name()?.to_str()?;
    let (_, first_seqno) = segment_identity(name)?;
    let before = std::fs::metadata(path).ok()?;
    if !before.is_file() {
        return None;
    }
    // The incremental reader is permissive on unknown headers; certificates
    // must reject them rather than infer a legacy version.
    let mut header = [0u8; 8];
    std::fs::File::open(path)
        .ok()?
        .read_exact(&mut header)
        .ok()?;
    if header != *crate::log::SEGMENT_MAGIC
        && header != *crate::log::SEGMENT_MAGIC_V2
        && header != *crate::log::SEGMENT_MAGIC_V3
    {
        return None;
    }
    if header == *crate::log::SEGMENT_MAGIC_V3
        && crate::log::segment_vector_space_id(path) != Some(vector_space_id)
    {
        return None;
    }
    let mut last_seqno = None;
    let mut valid = true;
    let end = crate::log::replay_from_offset(path, 0, |seqno, _| {
        valid &= match last_seqno {
            Some(last) => seqno > last,
            None => seqno == first_seqno,
        };
        last_seqno = Some(seqno);
        Ok(())
    })
    .ok()?;
    let after = std::fs::metadata(path).ok()?;
    if !valid
        || end != before.len()
        || before.len() != after.len()
        || before.modified().ok()? != after.modified().ok()?
    {
        return None;
    }
    Some(SegmentInfo {
        path: format!("segments/{name}"),
        first_seqno,
        last_seqno: last_seqno?, // Empty files carry no useful certificate.
        size_bytes: end,
    })
}

/// Check an inventory entry against independently validated full and cortical
/// coverage. The caller must bind both vectors and the inventory to the loaded
/// family/lineage first; legacy scalar watermarks are insufficient.
/// This opens no segment. Missing metadata always means decode/retain.
pub(crate) fn covered_segment(
    data_dir: &Path,
    entry: &SegmentInfo,
    full: &BTreeMap<u32, u64>,
    cortical: &BTreeMap<u32, u64>,
    writer_path: &Path,
) -> bool {
    let Some(name) = entry.path.strip_prefix("segments/") else {
        return false;
    };
    let Some((writer, first)) = segment_identity(name) else {
        return false;
    };
    if first != entry.first_seqno || entry.last_seqno < first {
        return false;
    }
    let (Some(full), Some(cortical)) = (full.get(&writer), cortical.get(&writer)) else {
        return false;
    };
    let path = data_dir.join(&entry.path);
    path != writer_path
        && entry.last_seqno <= (*full).min(*cortical)
        && std::fs::metadata(path)
            .map(|m| m.is_file() && m.len() == entry.size_bytes)
            .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::OpLog;
    use crate::ops::{Op, UpdateMemoryContentOp};

    fn fixture() -> (tempfile::TempDir, std::path::PathBuf, SegmentInfo) {
        let dir = tempfile::tempdir().unwrap();
        let mut log = OpLog::open(dir.path(), 0x12345678, 1).unwrap();
        for seq in 1..=2 {
            log.append(&Op::UpdateMemoryContent(UpdateMemoryContentOp {
                memory_id: seq,
                content: format!("record {seq}").into_bytes(),
                embedding: Vec::new(),
                op_ts_ms: 0,
            }))
            .unwrap();
        }
        log.sync().unwrap();
        let path = log.writer_path().to_path_buf();
        drop(log); // Explicitly seal the fixture writer.
        let active = dir.path().join("segments/87654321_000000000003.seg");
        let vsid = crate::log::segment_vector_space_id(&path).unwrap();
        let entry = scan_sealed_segment(&path, &active, vsid).unwrap();
        (dir, path, entry)
    }

    #[test]
    fn certificate_requires_both_writer_vectors_and_exact_range() {
        let (dir, path, mut entry) = fixture();
        let active = dir.path().join("active");
        let full = BTreeMap::from([(0x12345678, entry.last_seqno)]);
        let mut cortex = BTreeMap::from([(0x12345678, entry.last_seqno - 1)]);
        assert!(!covered_segment(
            dir.path(),
            &entry,
            &full,
            &cortex,
            &active
        ));
        cortex.insert(0x12345678, entry.last_seqno);
        assert!(covered_segment(dir.path(), &entry, &full, &cortex, &active));
        assert!(!covered_segment(dir.path(), &entry, &full, &cortex, &path));
        assert!(!covered_segment(
            dir.path(),
            &entry,
            &BTreeMap::new(),
            &cortex,
            &active
        ));
        assert!(!covered_segment(
            dir.path(),
            &entry,
            &full,
            &BTreeMap::new(),
            &active
        ));
        entry.first_seqno += 1;
        assert!(!covered_segment(
            dir.path(),
            &entry,
            &full,
            &cortex,
            &active
        ));
    }

    #[test]
    fn certification_rejects_active_foreign_and_torn_segments() {
        use std::io::Write;
        let (dir, path, entry) = fixture();
        let active = dir.path().join("active");
        let vsid = crate::log::segment_vector_space_id(&path).unwrap();
        assert!(scan_sealed_segment(&path, &path, vsid).is_none());
        assert!(scan_sealed_segment(&path, &active, vsid ^ 1).is_none());
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&[0, 0])
            .unwrap();
        assert!(scan_sealed_segment(&path, &active, vsid).is_none());
        let covered = BTreeMap::from([(0x12345678, u64::MAX)]);
        assert!(!covered_segment(
            dir.path(),
            &entry,
            &covered,
            &covered,
            &active
        ));
    }

    #[test]
    fn certificates_reject_missing_files_and_paths_outside_inventory() {
        let (dir, _, mut entry) = fixture();
        let covered = BTreeMap::from([(0x12345678, u64::MAX)]);
        for name in [
            "../12345678_000000000001.seg",
            "12345678_1.seg",
            "12345678_000000000001.seg/other",
            "1234567g_000000000001.seg",
        ] {
            assert!(segment_identity(name).is_none());
            entry.path = format!("segments/{name}");
            assert!(!covered_segment(
                dir.path(),
                &entry,
                &covered,
                &covered,
                &dir.path().join("active")
            ));
        }
        entry.path = "segments/12345678_000000000999.seg".into();
        entry.first_seqno = 999;
        entry.last_seqno = 1000;
        assert!(!covered_segment(
            dir.path(),
            &entry,
            &covered,
            &covered,
            &dir.path().join("active")
        ));
    }
}
