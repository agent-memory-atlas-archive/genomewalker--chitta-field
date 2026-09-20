//! Phase 0 proof for docs/DESIGN-2026-09-20-mapped-snapshot.md §7.
//!
//! Reads a committed V23 family once through the existing loader, writes its
//! memories out as the §5 arrays, then measures map+validate, random
//! peek-equivalent lookups (cold and warm) and one sequential pass over the
//! embedding matrix. `sweep` measures MADV_WILLNEED plus a sequential touch
//! over a whole family directory. Nothing here changes store behaviour.
//!
//! ceiling: checksums are crc32 widened to u64, not xxh3; the V24 writer must
//! use xxh3 as the design states. crc32 is the slower of the two, so a verify
//! timing measured here is an upper bound.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chitta_field::ids::MemoryId;
use chitta_field::ops::EMBED_DIM;
use chitta_field::snapshot::FullSnapshot;

const MAGIC: u64 = 0x3432_565F_4154_5443; // "CTAT_V24" little-endian-ish tag
const FORMAT_VERSION: u32 = 1;
const ROW: usize = 64;
const HEADER_RESERVED: usize = 4096;
const SECTION_ENTRY: usize = 56;
const ALIGN: usize = 4096;

struct SectionDesc {
    name: [u8; 16],
    offset: u64,
    len: u64,
    elem_size: u32,
    elem_count: u64,
    checksum: u64,
    align: u32,
}

fn name16(s: &str) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..s.len()].copy_from_slice(s.as_bytes());
    out
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    u64::from(h.finalize())
}

// ── writer ───────────────────────────────────────────────────────────────────

/// Interned realm/kind table: `[count:u32]([len:u32][bytes])×count`.
struct Interner {
    ids: HashMap<String, u32>,
    order: Vec<String>,
}

impl Interner {
    fn new() -> Self {
        Self { ids: HashMap::new(), order: Vec::new() }
    }
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.order.len() as u32;
        self.ids.insert(s.to_string(), id);
        self.order.push(s.to_string());
        id
    }
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.order.len() as u32).to_le_bytes());
        for s in &self.order {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        out
    }
}

/// `[magic:u64][count:u64]([id:u64][f32×EMBED_DIM])×count`, hnsw.rs §sidecar.
fn read_emb_sidecar(path: &Path) -> HashMap<MemoryId, Vec<u8>> {
    let mut out = HashMap::new();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("emb sidecar unreadable: {e}");
            return out;
        }
    };
    if bytes.len() < 16 {
        return out;
    }
    let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let rec = 8 + EMBED_DIM * 4;
    let mut off = 16usize;
    for _ in 0..count {
        if off + rec > bytes.len() {
            break;
        }
        let id = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        out.insert(id, bytes[off + 8..off + rec].to_vec());
        off += rec;
    }
    out
}

fn pad_to(out: &mut Vec<u8>, align: usize) {
    while out.len() % align != 0 {
        out.push(0);
    }
}

fn build(snapshot: &Path, dest: &Path) -> std::io::Result<()> {
    let t0 = Instant::now();
    let mut snap = FullSnapshot::load(snapshot).expect("load V23 family");
    eprintln!("v23 load: {:?} memories={}", t0.elapsed(), snap.states.len());

    let t = Instant::now();
    FullSnapshot::load_payload_sidecar(&snapshot.with_extension("pld"), &mut snap.payloads);
    eprintln!("pld sidecar: {:?}", t.elapsed());

    let t = Instant::now();
    let embeddings = read_emb_sidecar(&snapshot.with_extension("emb"));
    eprintln!("emb sidecar: {:?} rows={}", t.elapsed(), embeddings.len());

    let t = Instant::now();
    let mut ids: Vec<MemoryId> = snap.states.keys().copied().collect();
    ids.sort_unstable();

    let mut interner = Interner::new();
    let mut state = Vec::with_capacity(ids.len() * ROW);
    let mut arena: Vec<u8> = Vec::with_capacity(1 << 30);
    let mut emb: Vec<u8> = Vec::with_capacity(embeddings.len() * EMBED_DIM * 4);

    const NO_PAYLOAD: &str = "";
    for id in &ids {
        let st = &snap.states[id];
        let pl = snap.payloads.get(id);
        let realm_id = interner.intern(pl.map_or(NO_PAYLOAD, |p| p.realm.as_str()));
        let kind_id = interner.intern(pl.map_or(NO_PAYLOAD, |p| p.kind.as_str()));
        let content: &[u8] = match pl { Some(p) => &p.content, None => &[] };
        let content_off = arena.len() as u64;
        arena.extend_from_slice(content);
        let content_len = content.len() as u32;
        let emb_row = match embeddings.get(id) {
            Some(v) => {
                let row = (emb.len() / (EMBED_DIM * 4)) as u32;
                emb.extend_from_slice(v);
                row
            }
            None => u32::MAX,
        };
        let mut flags = 0u32;
        if st.deleted {
            flags |= 1;
        }
        if pl.is_some_and(|p| p.candidate) {
            flags |= 2;
        }
        let ack = snap.ack_scores.get(id).copied().unwrap_or(0);

        let before = state.len();
        state.extend_from_slice(&id.to_le_bytes());
        state.extend_from_slice(&realm_id.to_le_bytes());
        state.extend_from_slice(&kind_id.to_le_bytes());
        state.extend_from_slice(&pl.map_or(0, |p| p.created_at_ms).to_le_bytes());
        state.extend_from_slice(&pl.map_or(0, |p| p.authored_at_ms).to_le_bytes());
        state.extend_from_slice(&flags.to_le_bytes());
        state.extend_from_slice(&ack.to_le_bytes());
        state.extend_from_slice(&content_off.to_le_bytes());
        state.extend_from_slice(&content_len.to_le_bytes());
        state.extend_from_slice(&emb_row.to_le_bytes());
        state.extend_from_slice(&st.strength.to_le_bytes());
        state.extend_from_slice(&st.confidence.to_le_bytes());
        assert_eq!(state.len() - before, ROW, "state row is not 64 bytes");
    }
    let strings = interner.encode();
    eprintln!(
        "arrays built: {:?} state={} arena={} emb={} strings={}",
        t.elapsed(),
        state.len(),
        arena.len(),
        emb.len(),
        strings.len()
    );

    let bodies: Vec<(&str, &[u8], u32, u64)> = vec![
        ("state", state.as_slice(), ROW as u32, ids.len() as u64),
        ("arena", arena.as_slice(), 1, arena.len() as u64),
        ("emb", emb.as_slice(), (EMBED_DIM * 4) as u32, (emb.len() / (EMBED_DIM * 4)) as u64),
        ("strings", strings.as_slice(), 1, interner.order.len() as u64),
    ];

    let mut offset = HEADER_RESERVED.max(28 + bodies.len() * SECTION_ENTRY);
    offset = offset.div_ceil(ALIGN) * ALIGN;
    let mut descs = Vec::new();
    for (name, body, elem_size, elem_count) in &bodies {
        descs.push(SectionDesc {
            name: name16(name),
            offset: offset as u64,
            len: body.len() as u64,
            elem_size: *elem_size,
            elem_count: *elem_count,
            checksum: checksum(body),
            align: ALIGN as u32,
        });
        offset += body.len();
        offset = offset.div_ceil(ALIGN) * ALIGN;
    }

    let mut head = Vec::with_capacity(HEADER_RESERVED);
    head.extend_from_slice(&MAGIC.to_le_bytes());
    head.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    head.extend_from_slice(&0u32.to_le_bytes());
    head.extend_from_slice(&snap.snapshot_seqno.to_le_bytes());
    head.extend_from_slice(&(descs.len() as u32).to_le_bytes());
    for d in &descs {
        head.extend_from_slice(&d.name);
        head.extend_from_slice(&d.offset.to_le_bytes());
        head.extend_from_slice(&d.len.to_le_bytes());
        head.extend_from_slice(&d.elem_size.to_le_bytes());
        head.extend_from_slice(&d.elem_count.to_le_bytes());
        head.extend_from_slice(&d.checksum.to_le_bytes());
        head.extend_from_slice(&d.align.to_le_bytes());
    }
    pad_to(&mut head, ALIGN);

    let t = Instant::now();
    let tmp = dest.with_extension("tmp");
    let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
    f.write_all(&head)?;
    let mut written = head.len();
    let zeros = [0u8; 4096];
    for (d, (_, body, _, _)) in descs.iter().zip(bodies.iter()) {
        while written < d.offset as usize {
            let n = (d.offset as usize - written).min(zeros.len());
            f.write_all(&zeros[..n])?;
            written += n;
        }
        f.write_all(body)?;
        written += body.len();
    }
    f.into_inner()?.sync_all()?;
    std::fs::rename(&tmp, dest)?;
    eprintln!("wrote {} ({:?})", dest.display(), t.elapsed());
    Ok(())
}

// ── mapped reader ────────────────────────────────────────────────────────────

struct Mapped {
    map: memmap2::Mmap,
    sections: Vec<SectionDesc>,
}

impl Mapped {
    fn section(&self, name: &str) -> &[u8] {
        let want = name16(name);
        let d = self.sections.iter().find(|d| d.name == want).expect("section");
        &self.map[d.offset as usize..(d.offset + d.len) as usize]
    }
}

/// Map and validate: header, version, section table bounds, and a sampled
/// checksum (first and last 64 KiB of each section) as §4 specifies.
fn map_and_validate(path: &Path, full_verify: bool) -> (Mapped, std::time::Duration) {
    let t = Instant::now();
    let file = std::fs::File::open(path).expect("open");
    let map = unsafe { memmap2::MmapOptions::new().map(&file).expect("mmap") };
    assert_eq!(u64::from_le_bytes(map[0..8].try_into().unwrap()), MAGIC, "magic");
    assert_eq!(u32::from_le_bytes(map[8..12].try_into().unwrap()), FORMAT_VERSION, "version");
    let count = u32::from_le_bytes(map[24..28].try_into().unwrap()) as usize;
    let mut sections = Vec::with_capacity(count);
    for i in 0..count {
        let b = 28 + i * SECTION_ENTRY;
        let d = SectionDesc {
            name: map[b..b + 16].try_into().unwrap(),
            offset: u64::from_le_bytes(map[b + 16..b + 24].try_into().unwrap()),
            len: u64::from_le_bytes(map[b + 24..b + 32].try_into().unwrap()),
            elem_size: u32::from_le_bytes(map[b + 32..b + 36].try_into().unwrap()),
            elem_count: u64::from_le_bytes(map[b + 36..b + 44].try_into().unwrap()),
            checksum: u64::from_le_bytes(map[b + 44..b + 52].try_into().unwrap()),
            align: u32::from_le_bytes(map[b + 52..b + 56].try_into().unwrap()),
        };
        assert!((d.offset + d.len) as usize <= map.len(), "section out of bounds");
        assert_eq!(d.offset as usize % d.align as usize, 0, "section misaligned");
        if d.elem_size > 1 {
            assert_eq!(d.len, d.elem_size as u64 * d.elem_count, "elem size/count disagree");
        }
        sections.push(d);
    }
    let sample = 64 << 10;
    for d in &sections {
        let body = &map[d.offset as usize..(d.offset + d.len) as usize];
        if full_verify {
            assert_eq!(checksum(body), d.checksum, "section checksum");
        } else {
            let head = &body[..body.len().min(sample)];
            let tail = &body[body.len().saturating_sub(sample)..];
            std::hint::black_box(checksum(head) ^ checksum(tail));
        }
    }
    let elapsed = t.elapsed();
    (Mapped { map, sections }, elapsed)
}

/// Evict this file's clean pages so the next touch is a real first-touch fault.
fn drop_cache(path: &Path) {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path).expect("open for fadvise");
    let len = f.metadata().expect("metadata").len() as i64;
    unsafe {
        libc::posix_fadvise(f.as_raw_fd(), 0, len, libc::POSIX_FADV_DONTNEED);
    }
}

/// peek_memory equivalent through the mapping: binary search the state table,
/// reject deleted, copy the content bytes out of the arena, read the embedding.
fn peek(state: &[u8], arena: &[u8], emb: &[u8], id: MemoryId) -> usize {
    let rows = state.len() / ROW;
    let mut lo = 0usize;
    let mut hi = rows;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let k = u64::from_le_bytes(state[mid * ROW..mid * ROW + 8].try_into().unwrap());
        match k.cmp(&id) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => {
                let r = &state[mid * ROW..(mid + 1) * ROW];
                let flags = u32::from_le_bytes(r[32..36].try_into().unwrap());
                if flags & 1 != 0 {
                    return 0;
                }
                let off = u64::from_le_bytes(r[40..48].try_into().unwrap()) as usize;
                let len = u32::from_le_bytes(r[48..52].try_into().unwrap()) as usize;
                let content = arena[off..off + len].to_vec();
                let row = u32::from_le_bytes(r[52..56].try_into().unwrap());
                let mut acc = content.len();
                if row != u32::MAX {
                    let base = row as usize * EMBED_DIM * 4;
                    acc += emb[base] as usize + emb[base + EMBED_DIM * 4 - 1] as usize;
                }
                return acc;
            }
        }
    }
    0
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 11
}

fn measure(path: &Path, lookups: usize) {
    // Cold map + validate: evict first so header and sampled pages really fault.
    drop_cache(path);
    let (m, cold_validate) = map_and_validate(path, false);
    let rows = m.section("state").len() / ROW;
    println!("file_bytes\t{}", m.map.len());
    println!("state_rows\t{rows}");
    println!("arena_bytes\t{}", m.section("arena").len());
    println!("emb_rows\t{}", m.section("emb").len() / (EMBED_DIM * 4));
    println!("map_validate_cold_ms\t{:.3}", cold_validate.as_secs_f64() * 1e3);
    // Same ids for every phase, drawn from the mapped state table itself.
    let mut seed = 0x5ee_d123u64;
    let ids: Vec<MemoryId> = {
        let state = m.section("state");
        (0..lookups)
            .map(|_| {
                let r = (lcg(&mut seed) as usize) % rows;
                u64::from_le_bytes(state[r * ROW..r * ROW + 8].try_into().unwrap())
            })
            .collect()
    };
    drop(m);

    let (m, warm_validate) = map_and_validate(path, false);
    println!("map_validate_warm_ms\t{:.3}", warm_validate.as_secs_f64() * 1e3);
    drop(m);

    // Full checksum verify: the concurrent background step in §4, not the gate.
    let t = Instant::now();
    let (m, _) = map_and_validate(path, true);
    println!("full_verify_warm_ms\t{:.3}", t.elapsed().as_secs_f64() * 1e3);
    drop(m);

    // Cold lookups: evict, remap, run the ids on first-touch pages.
    drop_cache(path);
    let (m, _) = map_and_validate(path, false);
    let mut acc = 0usize;
    {
        let (state, arena, emb) = (m.section("state"), m.section("arena"), m.section("emb"));
        let t = Instant::now();
        for &id in &ids {
            acc += peek(state, arena, emb, id);
        }
        let cold = t.elapsed();
        println!("lookup_cold_total_ms\t{:.3}", cold.as_secs_f64() * 1e3);
        println!("lookup_cold_us_each\t{:.3}", cold.as_secs_f64() * 1e6 / lookups as f64);

        // Warm: same ids, pages now resident.
        let t = Instant::now();
        for &id in &ids {
            acc += peek(state, arena, emb, id);
        }
        let warm = t.elapsed();
        println!("lookup_warm_total_ms\t{:.3}", warm.as_secs_f64() * 1e3);
        println!("lookup_warm_us_each\t{:.3}", warm.as_secs_f64() * 1e6 / lookups as f64);
    }
    std::hint::black_box(acc);
    drop(m);

    // One full sequential pass over the embedding matrix, cold then warm: the
    // flat-scan shape recall_semantic falls back to before the index is loaded.
    let query: Vec<f32> = (0..EMBED_DIM).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    for label in ["cold", "warm"] {
        if label == "cold" {
            drop_cache(path);
        }
        let (m, _) = map_and_validate(path, false);
        let emb = m.section("emb");
        let n = emb.len() / (EMBED_DIM * 4);
        let t = Instant::now();
        let mut best = f32::MIN;
        let mut best_row = 0usize;
        for r in 0..n {
            let base = r * EMBED_DIM * 4;
            // SAFETY: the emb section starts 4096-aligned and every row is a
            // multiple of EMBED_DIM*4 bytes, so each row start is f32-aligned.
            let row: &[f32] = unsafe {
                std::slice::from_raw_parts(emb[base..].as_ptr() as *const f32, EMBED_DIM)
            };
            let mut dot = 0f32;
            for k in 0..EMBED_DIM {
                dot += row[k] * query[k];
            }
            if dot > best {
                best = dot;
                best_row = r;
            }
        }
        let d = t.elapsed();
        println!("emb_scan_{label}_ms\t{:.3}", d.as_secs_f64() * 1e3);
        println!(
            "emb_scan_{label}_mbps\t{:.1}",
            emb.len() as f64 / (1 << 20) as f64 / d.as_secs_f64()
        );
        std::hint::black_box((best, best_row));
    }
}

// ── MADV_WILLNEED sweep ──────────────────────────────────────────────────────

fn sweep(dir: &Path) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read family dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    let total: u64 = files.iter().map(|p| p.metadata().map(|m| m.len()).unwrap_or(0)).sum();
    println!("sweep_files\t{}", files.len());
    println!("sweep_bytes\t{total}");

    for p in &files {
        drop_cache(p);
    }

    let t = Instant::now();
    let mut maps = Vec::new();
    for p in &files {
        let f = std::fs::File::open(p).expect("open");
        if f.metadata().expect("metadata").len() == 0 {
            continue;
        }
        let m = unsafe { memmap2::MmapOptions::new().map(&f).expect("mmap") };
        unsafe {
            libc::madvise(m.as_ptr() as *mut libc::c_void, m.len(), libc::MADV_WILLNEED);
        }
        maps.push(m);
    }
    let advise = t.elapsed();
    println!("madvise_call_ms\t{:.3}", advise.as_secs_f64() * 1e3);

    let t = Instant::now();
    let mut acc = 0u64;
    for m in &maps {
        let mut i = 0usize;
        while i < m.len() {
            acc += m[i] as u64;
            i += 4096;
        }
    }
    let touch = t.elapsed();
    std::hint::black_box(acc);
    println!("sequential_touch_ms\t{:.3}", touch.as_secs_f64() * 1e3);
    println!("sweep_total_ms\t{:.3}", (advise + touch).as_secs_f64() * 1e3);
    println!(
        "sweep_mbps\t{:.1}",
        total as f64 / (1 << 20) as f64 / (advise + touch).as_secs_f64()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => build(Path::new(&args[2]), Path::new(&args[3])).expect("build"),
        Some("measure") => measure(
            Path::new(&args[2]),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000),
        ),
        Some("sweep") => sweep(Path::new(&args[2])),
        _ => {
            eprintln!("usage: mapped_bench build <family.snapshot> <out>");
            eprintln!("       mapped_bench measure <out> [lookups]");
            eprintln!("       mapped_bench sweep <family-dir>");
            std::process::exit(2);
        }
    }
}
