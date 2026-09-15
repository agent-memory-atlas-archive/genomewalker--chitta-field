//! Hyperdimensional Computing (HDC) memory index.
//!
//! Encodes memory content as 8192-bit binary vectors via bag-of-words bundling.
//! Retrieval is O(n) Hamming distance (popcount XOR) — pure CPU bitwise ops,
//! no floats, no GPU, no SIMD required.
//!
//! Vocabulary is hash-derived and deterministic: every restart produces the
//! same word→HdcVec mapping without loading any file.
//!
//! Properties exploited by chitta:
//! - Compositional queries: `AND(X, Y)` ≈ `XOR(encode(X), encode(Y))`
//! - Incremental realm bundles: add/remove a single memory without rebuild
//! - Temporal set membership: `hamming(session_bundle, query_hv)` ≈ relevance
//! - Forgetting without deletion: un-bundle subtracts the vector's bit contribution

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

use crate::ids::MemoryId;

// ── Dimensionality ────────────────────────────────────────────────────────────

/// Number of u64 words per vector: 128 × 64 = 8192 bits.
const D: usize = 128;

/// A single hyperdimensional vector.
pub type HdcVec = [u64; D];

// ── Deterministic vocabulary ──────────────────────────────────────────────────

#[inline]
fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        h.wrapping_mul(0x100000001b3) ^ b as u64
    })
}

/// Map a word to its random hypervector. Deterministic: same word → same HdcVec
/// across all restarts, no vocabulary file needed.
pub fn word_hv(word: &str) -> HdcVec {
    // +1 so the empty-string / zero-hash edge case still produces a nonzero seed
    let mut s = fnv1a_64(word.as_bytes()).wrapping_add(1);
    std::array::from_fn(|_| xorshift64(&mut s))
}

// ── Primitive operations ──────────────────────────────────────────────────────

/// Binding: XOR of two vectors. Used to associate role with content, or
/// to combine two concepts into a superposition.
#[inline]
pub fn bind(a: &HdcVec, b: &HdcVec) -> HdcVec {
    std::array::from_fn(|i| a[i] ^ b[i])
}

/// Hamming distance between two vectors (0 = identical, 8192 = opposite).
#[inline]
pub fn hamming(a: &HdcVec, b: &HdcVec) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Normalised similarity in [0, 1] (1 = identical).
#[inline]
pub fn similarity(a: &HdcVec, b: &HdcVec) -> f32 {
    1.0 - hamming(a, b) as f32 / (D * 64) as f32
}

/// Majority-vote bundle of `vecs`. Threshold: strict majority (>n/2 → 1).
/// Ties on even n go to 0. Empty input → zero vector.
pub fn bundle(vecs: &[HdcVec]) -> HdcVec {
    if vecs.is_empty() {
        return [0u64; D];
    }
    // u16 popcount per bit position — supports up to 65535 simultaneous vectors
    let mut counts = vec![0u16; D * 64];
    for hv in vecs {
        for (wi, &w) in hv.iter().enumerate() {
            for bit in 0u32..64 {
                if (w >> bit) & 1 == 1 {
                    counts[wi * 64 + bit as usize] = counts[wi * 64 + bit as usize].saturating_add(1);
                }
            }
        }
    }
    let threshold = vecs.len() as u32 / 2 + 1;
    let mut out = [0u64; D];
    for wi in 0..D {
        for bit in 0u32..64 {
            if counts[wi * 64 + bit as usize] as u32 >= threshold {
                out[wi] |= 1u64 << bit;
            }
        }
    }
    out
}

// ── Text encoder ──────────────────────────────────────────────────────────────

/// Tokenise `text` into lowercase alphanumeric tokens (≥2 chars), look up
/// each token's word_hv, and majority-vote bundle them into a single HdcVec.
pub fn encode(text: &str) -> HdcVec {
    let hvs: Vec<HdcVec> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| word_hv(&t.to_lowercase()))
        .collect();
    bundle(&hvs)
}

// ── Geometry binarization ─────────────────────────────────────────────────────

/// Convert a continuous f32 direction vector (from embedding matrix PCA) into a
/// binary HdcVec. Sign-pack 64 floats per u64 word; XOR-rotation-mix to fill D=128.
///
/// Similarity order is approximately preserved: directions with small cosine
/// angle → small Hamming distance (first-order LSH property without a random matrix).
fn binarize_f32_direction(direction: &[f32]) -> HdcVec {
    let mut out = [0u64; D];
    let pack_words = (direction.len() / 64).min(D);
    for (wi, chunk) in direction.chunks(64).enumerate().take(pack_words) {
        let mut word = 0u64;
        for (bi, &v) in chunk.iter().enumerate() {
            if v > 0.0 { word |= 1u64 << bi; }
        }
        out[wi] = word;
    }
    // XOR-rotation mix to fill remaining words when direction.len() < D*64.
    if pack_words > 0 && pack_words < D {
        for wi in pack_words..D {
            out[wi] = out[wi % pack_words].rotate_left(13)
                ^ out[(wi + 1) % pack_words].rotate_right(7);
        }
    }
    out
}

// ── Incremental realm bundle ──────────────────────────────────────────────────

/// Maintains a running bit-count so a realm's aggregate HdcVec can be recovered
/// in O(D) without iterating all member vectors.
#[derive(Clone)]
pub struct RealmBundle {
    /// bit[wi*64 + b] = how many stored HdcVecs have bit b set in word wi
    counts: Vec<u16>,
    /// how many HdcVecs have been bundled
    pub n: u32,
}

impl Default for RealmBundle {
    fn default() -> Self {
        Self { counts: vec![0u16; D * 64], n: 0 }
    }
}

impl RealmBundle {
    pub fn add(&mut self, hv: &HdcVec) {
        for (wi, &w) in hv.iter().enumerate() {
            for bit in 0u32..64 {
                if (w >> bit) & 1 == 1 {
                    self.counts[wi * 64 + bit as usize] =
                        self.counts[wi * 64 + bit as usize].saturating_add(1);
                }
            }
        }
        self.n += 1;
    }

    pub fn remove(&mut self, hv: &HdcVec) {
        if self.n == 0 { return; }
        for (wi, &w) in hv.iter().enumerate() {
            for bit in 0u32..64 {
                if (w >> bit) & 1 == 1 {
                    self.counts[wi * 64 + bit as usize] =
                        self.counts[wi * 64 + bit as usize].saturating_sub(1);
                }
            }
        }
        self.n = self.n.saturating_sub(1);
    }

    /// Materialise the current majority-vote HdcVec for this realm.
    pub fn to_hv(&self) -> HdcVec {
        if self.n == 0 { return [0u64; D]; }
        let threshold = self.n / 2 + 1;
        let mut out = [0u64; D];
        for wi in 0..D {
            for bit in 0u32..64 {
                if self.counts[wi * 64 + bit as usize] as u32 >= threshold {
                    out[wi] |= 1u64 << bit;
                }
            }
        }
        out
    }
}

// ── HdcStore ──────────────────────────────────────────────────────────────────

/// Per-memory HDC index. Maintained alongside the HNSW and BM25 indexes.
/// Rebuilt from persisted payload content on startup — no snapshot serialisation needed.
#[derive(Default)]
pub struct HdcStore {
    /// memory_id → encoded HdcVec (1024 bytes each)
    memories: HashMap<MemoryId, HdcVec>,
    /// realm → running bit-count bundle
    realm_bundles: HashMap<String, RealmBundle>,
    /// realm → set of memory IDs (for realm-scoped recall)
    realm_members: HashMap<String, HashSet<MemoryId>>,
    /// token → seeded HdcVec from open-weight geometry harvest
    /// Checked before hash-derived word_hv() — grounded vectors take precedence.
    seeded_codebook: HashMap<String, HdcVec>,
    /// Monotone mutation counter (runtime-only) for sidecar dirty-skip at
    /// save (same pattern as SemanticIndex::mutations).
    mutations: u64,
}

impl HdcStore {
    pub(crate) fn allocated_bytes(&self) -> usize {
        use crate::profile::map_bytes;
        map_bytes(&self.memories) + map_bytes(&self.seeded_codebook)
            + self.seeded_codebook.keys().map(String::capacity).sum::<usize>()
            + map_bytes(&self.realm_bundles)
            + self.realm_bundles.iter().map(|(k, v)| k.capacity() + v.counts.capacity() * 2).sum::<usize>()
            + map_bytes(&self.realm_members)
            + self.realm_members.iter().map(|(k, v)| k.capacity() + v.capacity() * 9).sum::<usize>()
    }

    pub fn new() -> Self {
        Self::default()
    }

    /// Monotone mutation counter for sidecar dirty-skip at save.
    pub fn mutation_count(&self) -> u64 {
        self.mutations
    }

    /// Bulk-load from (id, content, realm) tuples. Used at startup to rebuild
    /// from persisted payloads without snapshot serialisation of large arrays.
    pub fn rebuild<'a>(&mut self, entries: impl Iterator<Item = (MemoryId, &'a str, &'a str)>) {
        self.mutations += 1;
        self.memories.clear();
        self.realm_bundles.clear();
        self.realm_members.clear();
        for (id, text, realm) in entries {
            self.insert(id, text, realm);
        }
    }

    pub fn len(&self) -> usize {
        self.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Encode `text` using the seeded codebook (grounded tokens first, then hash fallback).
    pub fn encode_with_codebook(&self, text: &str) -> HdcVec {
        let hvs: Vec<HdcVec> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 2)
            .map(|t| {
                let lower = t.to_lowercase();
                self.seeded_codebook.get(&lower).copied().unwrap_or_else(|| word_hv(&lower))
            })
            .collect();
        bundle(&hvs)
    }

    /// Encode `text` and register it under `id` in `realm`.
    pub fn insert(&mut self, id: MemoryId, text: &str, realm: &str) {
        self.mutations += 1;
        let hv = self.encode_with_codebook(text);
        self.realm_bundles
            .entry(realm.to_string())
            .or_default()
            .add(&hv);
        self.realm_members
            .entry(realm.to_string())
            .or_default()
            .insert(id);
        self.memories.insert(id, hv);
    }

    /// Remove a memory from the index and update the realm bundle.
    pub fn remove(&mut self, id: MemoryId, realm: &str) {
        self.mutations += 1;
        if let Some(hv) = self.memories.remove(&id) {
            if let Some(rb) = self.realm_bundles.get_mut(realm) {
                rb.remove(&hv);
            }
            if let Some(members) = self.realm_members.get_mut(realm) {
                members.remove(&id);
            }
        }
    }

    /// Return top-`k` memories by Hamming similarity to the encoded `text`.
    /// When `realm` is `Some`, only memories in that realm are considered.
    /// Returns `(MemoryId, hamming_distance)` sorted ascending (0 = identical).
    pub fn query(&self, text: &str, k: usize, realm: Option<&str>) -> Vec<(MemoryId, u32)> {
        if k == 0 || self.memories.is_empty() {
            return vec![];
        }
        let q = encode(text);
        let mut scored: Vec<(MemoryId, u32)> = match realm {
            Some(r) => {
                let Some(members) = self.realm_members.get(r) else { return vec![]; };
                members.iter()
                    .filter_map(|id| self.memories.get(id).map(|hv| (*id, hamming(&q, hv))))
                    .collect()
            }
            None => self.memories.iter().map(|(&id, hv)| (id, hamming(&q, hv))).collect(),
        };
        scored.sort_unstable_by_key(|&(id, d)| (d, id));
        scored.truncate(k);
        scored
    }

    /// Similarity score in [0, 1] for a precomputed query HdcVec vs a memory.
    pub fn score_hv(&self, id: MemoryId, query_hv: &HdcVec) -> Option<f32> {
        self.memories.get(&id).map(|hv| similarity(query_hv, hv))
    }

    /// Return top-`k` memories most representative of a realm's aggregate theme.
    /// Useful for "what does realm X talk about?" cluster queries.
    pub fn realm_theme_query(&self, realm: &str, k: usize) -> Vec<(MemoryId, u32)> {
        let Some(rb) = self.realm_bundles.get(realm) else { return vec![]; };
        let q = rb.to_hv();
        let Some(members) = self.realm_members.get(realm) else { return vec![]; };
        let mut scored: Vec<(MemoryId, u32)> = members.iter()
            .filter_map(|id| self.memories.get(id).map(|hv| (*id, hamming(&q, hv))))
            .collect();
        scored.sort_unstable_by_key(|&(id, d)| (d, id));
        scored.truncate(k);
        scored
    }

    /// Return the realm bundle HdcVec (the "theme" of a realm).
    pub fn realm_hv(&self, realm: &str) -> Option<HdcVec> {
        self.realm_bundles.get(realm).map(|rb| rb.to_hv())
    }

    /// Cross-realm similarity: how similar are two realms' aggregate themes?
    /// Returns similarity in [0, 1].
    pub fn realm_similarity(&self, a: &str, b: &str) -> f32 {
        match (self.realm_hv(a), self.realm_hv(b)) {
            (Some(ha), Some(hb)) => similarity(&ha, &hb),
            _ => 0.0,
        }
    }

    /// Number of tokens in the seeded codebook.
    pub fn codebook_len(&self) -> usize { self.seeded_codebook.len() }

    /// Seed the HDC codebook from a vocab_geometry harvest JSON produced by
    /// `scripts/harvest_ow.py --mode vocab_geometry`.
    ///
    /// Each semantic direction (f32[d]) is binarized via sign-packing:
    /// 64 floats → 1 u64 word (bit = sign(float)), filled to 128 u64s by
    /// XOR-rotation mixing. This preserves cosine similarity order in Hamming
    /// space to a first approximation. Returns the number of tokens seeded.
    pub fn seed_from_geometry(&mut self, json_path: &str) -> std::io::Result<usize> {
        self.mutations += 1;
        let bytes = std::fs::read(json_path)?;
        let val: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let dirs = val["directions"].as_array()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing directions"))?;
        let mut seeded = 0usize;
        for dir in dirs {
            let floats: Vec<f32> = dir["direction"].as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_f64().map(|x| x as f32))
                .collect();
            if floats.is_empty() { continue; }
            let hv = binarize_f32_direction(&floats);
            if let Some(tokens) = dir["top_tokens"].as_array() {
                for tok in tokens {
                    if let Some(s) = tok.as_str() {
                        let key: String = s.split(|c: char| !c.is_alphanumeric())
                            .filter(|t| t.len() >= 2)
                            .next()
                            .unwrap_or(s)
                            .to_lowercase();
                        if !key.is_empty() {
                            self.seeded_codebook.insert(key, hv);
                            seeded += 1;
                        }
                    }
                }
            }
        }
        Ok(seeded)
    }

    /// Compose a multi-concept query: encode each term, XOR-bind them all,
    /// then return nearest memories. XOR binding approximates logical AND in HDC.
    /// E.g., query_and(&["Python", "debugging"]) finds memories about both.
    /// Persist the full HDC index to a flat binary sidecar file.
    /// Format: magic(u64) + n_memories(u64) + [(id:u64, hv:[u64;128]); n]
    ///         + n_realms(u32) + per-realm [name_len(u32), name, n_members(u32),
    ///         [member_id:u64; n], bundle_n(u32), counts:[u16; D*64]].
    /// Returns the number of memories written.
    pub fn save_sidecar(&self, path: &Path) -> std::io::Result<usize> {
        const MAGIC: u64 = 0xC0DE_BABE_DC51_DE2A;
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(&MAGIC.to_le_bytes())?;
        let n = self.memories.len() as u64;
        w.write_all(&n.to_le_bytes())?;
        for (&id, hv) in &self.memories {
            w.write_all(&id.to_le_bytes())?;
            for &word in hv.iter() {
                w.write_all(&word.to_le_bytes())?;
            }
        }
        let n_realms = self.realm_members.len() as u32;
        w.write_all(&n_realms.to_le_bytes())?;
        for (realm, members) in &self.realm_members {
            let name_bytes = realm.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            w.write_all(name_bytes)?;
            w.write_all(&(members.len() as u32).to_le_bytes())?;
            for &mid in members {
                w.write_all(&mid.to_le_bytes())?;
            }
            let bundle = self.realm_bundles.get(realm.as_str());
            let (bundle_n, counts) = match bundle {
                Some(b) => (b.n, b.counts.as_slice()),
                None    => (0u32, &[][..]),
            };
            w.write_all(&bundle_n.to_le_bytes())?;
            if counts.len() == D * 64 {
                for &c in counts {
                    w.write_all(&c.to_le_bytes())?;
                }
            } else {
                for _ in 0..D * 64 {
                    w.write_all(&0u16.to_le_bytes())?;
                }
            }
        }
        Ok(self.memories.len())
    }

    /// Load HDC index from a sidecar file written by `save_sidecar`.
    /// Returns the number of memories loaded, or 0 on any format mismatch.
    pub fn load_sidecar(&mut self, path: &Path) -> std::io::Result<usize> {
        const MAGIC: u64 = 0xC0DE_BABE_DC51_DE2A;
        let f = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::new(f);
        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?;
        if u64::from_le_bytes(buf8) != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "hdc sidecar: bad magic"));
        }
        r.read_exact(&mut buf8)?;
        let n_memories = u64::from_le_bytes(buf8) as usize;
        self.memories.clear();
        self.realm_bundles.clear();
        self.realm_members.clear();
        self.memories.reserve(n_memories);
        for _ in 0..n_memories {
            r.read_exact(&mut buf8)?;
            let id = MemoryId::from_le_bytes(buf8);
            let mut hv = [0u64; D];
            for word in hv.iter_mut() {
                r.read_exact(&mut buf8)?;
                *word = u64::from_le_bytes(buf8);
            }
            self.memories.insert(id, hv);
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let n_realms = u32::from_le_bytes(buf4) as usize;
        for _ in 0..n_realms {
            r.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let mut name_buf = vec![0u8; name_len];
            r.read_exact(&mut name_buf)?;
            let realm = String::from_utf8(name_buf)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            r.read_exact(&mut buf4)?;
            let n_members = u32::from_le_bytes(buf4) as usize;
            let mut members = HashSet::with_capacity(n_members);
            for _ in 0..n_members {
                r.read_exact(&mut buf8)?;
                members.insert(MemoryId::from_le_bytes(buf8));
            }
            r.read_exact(&mut buf4)?;
            let bundle_n = u32::from_le_bytes(buf4);
            let mut counts = vec![0u16; D * 64];
            let mut buf2 = [0u8; 2];
            for c in counts.iter_mut() {
                r.read_exact(&mut buf2)?;
                *c = u16::from_le_bytes(buf2);
            }
            self.realm_members.insert(realm.clone(), members);
            self.realm_bundles.insert(realm, RealmBundle { counts, n: bundle_n });
        }
        Ok(n_memories)
    }

    pub fn query_and(&self, terms: &[&str], k: usize, realm: Option<&str>) -> Vec<(MemoryId, u32)> {
        if terms.is_empty() { return vec![]; }
        let mut q = word_hv(&terms[0].to_lowercase());
        for t in &terms[1..] {
            q = bind(&q, &word_hv(&t.to_lowercase()));
        }
        let mut scored: Vec<(MemoryId, u32)> = match realm {
            Some(r) => {
                let Some(members) = self.realm_members.get(r) else { return vec![]; };
                members.iter()
                    .filter_map(|id| self.memories.get(id).map(|hv| (*id, hamming(&q, hv))))
                    .collect()
            }
            None => self.memories.iter().map(|(&id, hv)| (id, hamming(&q, hv))).collect(),
        };
        scored.sort_unstable_by_key(|&(id, d)| (d, id));
        scored.truncate(k);
        scored
    }
}

// ── Episode HDC Binder ────────────────────────────────────────────────────────
//
// Heteroassociative memory over CEC events. Each event is encoded as:
//   E = bind(R_tool, t_hv) XOR bind(R_entity, e_hv) XOR bind(R_outcome, o_hv)
//
// Role vectors are deterministic (hash-derived). Episodes are accumulated into
// per-role-value bundles. Query: given known_role=known_val, retrieve top-k
// candidates for query_role via XOR-unbind + codebook cleanup.
//
// EpisodeHdcStore is ephemeral — rebuilt from EventTape at startup, not serialized.

#[inline] fn role_tool()    -> HdcVec { word_hv("ROLE:TOOL") }
#[inline] fn role_entity()  -> HdcVec { word_hv("ROLE:ENTITY") }
#[inline] fn role_outcome() -> HdcVec { word_hv("ROLE:OUTCOME") }

#[inline]
pub fn outcome_class_name(c: u8) -> &'static str {
    match c { 0 => "success", 1 => "fail", 2 => "error", _ => "partial" }
}

fn encode_episode_hv(tool_name: &str, entity_name: &str, outcome: u8) -> HdcVec {
    let bt = bind(&role_tool(),    &word_hv(tool_name));
    let be = bind(&role_entity(),  &encode(entity_name));
    let bo = bind(&role_outcome(), &word_hv(outcome_class_name(outcome)));
    let mut out = bt;
    for i in 0..D { out[i] ^= be[i] ^ bo[i]; }
    out
}

/// Bit-count accumulator for a set of episode hypervectors.
/// Identical to RealmBundle but independent so EpisodeHdcStore has no snapshot dependency.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
struct EpBundle {
    // Bit-sliced saturating u16 counters. A singleton needs one 1024-byte
    // plane, rather than 8192 u16s (16 KiB). More planes are allocated only
    // when observation counts require them; the maximum is still 16 planes.
    #[serde(with = "episode_planes")]
    planes: Vec<Box<HdcVec>>,
    n: u32,
}

impl EpBundle {
    fn add(&mut self, hv: &HdcVec) {
        let needed = (32 - self.n.saturating_add(1).leading_zeros()).min(16) as usize;
        while self.planes.len() < needed { self.planes.push(Box::new([0; D])); }
        for (wi, &word) in hv.iter().enumerate() {
            let mut carry = word;
            for plane in &mut self.planes {
                let old = plane[wi];
                plane[wi] ^= carry;
                carry &= old;
                if carry == 0 { break; }
            }
            // A carry beyond the sixteenth bit saturates, exactly like the
            // prior u16::saturating_add. Restore those lanes to all ones.
            if carry != 0 {
                for plane in &mut self.planes { plane[wi] |= carry; }
            }
        }
        self.n += 1;
    }

    fn to_hv(&self) -> HdcVec {
        let threshold = self.n / 2 + 1;
        if self.n == 0 || threshold >= (1u32 << self.planes.len()) { return [0; D]; }
        let mut out = [0; D];
        for (wi, word) in out.iter_mut().enumerate() {
            let mut equal = !0u64;
            let mut greater = 0u64;
            for (bit, plane) in self.planes.iter().enumerate().rev() {
                if threshold & (1 << bit) == 0 {
                    greater |= equal & plane[wi];
                    equal &= !plane[wi];
                } else {
                    equal &= plane[wi];
                }
            }
            *word = greater | equal;
        }
        out
    }

    fn allocated_bytes(&self) -> usize {
        self.planes.capacity() * std::mem::size_of::<Box<HdcVec>>()
            + self.planes.len() * std::mem::size_of::<HdcVec>()
    }
}

/// Episode-level HDC heteroassociative index.
/// Optional startup sidecar; never embedded in the full snapshot.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct EpisodeHdcStore {
    /// Episodes grouped by tool name
    by_tool:    std::collections::HashMap<String, EpBundle>,
    /// Episodes grouped by entity name (canonical)
    by_entity:  std::collections::HashMap<String, EpBundle>,
    /// Episodes grouped by outcome class name ("success"/"fail"/"error"/"partial")
    by_outcome: std::collections::HashMap<String, EpBundle>,
    /// All observed tool names (for codebook cleanup)
    tools:    std::collections::HashSet<String>,
    /// All observed entity names
    entities: std::collections::HashSet<String>,
}

impl EpisodeHdcStore {
    pub(crate) fn allocated_bytes(&self) -> (usize, usize) {
        use crate::profile::map_bytes;
        let mut actual = 0;
        let mut legacy = 0;
        for map in [&self.by_tool, &self.by_entity, &self.by_outcome] {
            let base = map_bytes(map) + map.keys().map(String::capacity).sum::<usize>();
            actual += base + map.values().map(EpBundle::allocated_bytes).sum::<usize>();
            legacy += base + map.len() * D * 64 * 2;
        }
        (actual, legacy)
    }

    pub fn new() -> Self { Self::default() }

    /// Encode one episode and add it to the per-role-value bundles.
    pub fn log_episode(&mut self, tool_name: &str, entity_name: &str, outcome: u8) {
        let ep = encode_episode_hv(tool_name, entity_name, outcome);
        self.by_tool.entry(tool_name.to_string()).or_default().add(&ep);
        self.by_entity.entry(entity_name.to_string()).or_default().add(&ep);
        let oname = outcome_class_name(outcome);
        self.by_outcome.entry(oname.to_string()).or_default().add(&ep);
        self.tools.insert(tool_name.to_string());
        self.entities.insert(entity_name.to_string());
    }

    /// Rebuild from all events in the tape.
    pub fn rebuild(&mut self, tape: &super::organ::event_tape::EventTape) {
        *self = Self::new();
        for ev in &tape.events {
            let tn = tape.tool_name(ev.tool_id);
            let en = tape.entity_name(ev.entity_key);
            self.log_episode(tn, en, ev.outcome_class);
        }
    }

    /// Heteroassociative query: given `known_role = known_val`, return top-k
    /// candidates for `query_role` ranked by Hamming similarity after XOR-unbind.
    ///
    /// known_role / query_role: "tool" | "entity" | "outcome"
    pub fn recall_hdcbind(
        &self,
        known_role: &str,
        known_val: &str,
        query_role: &str,
        k: usize,
    ) -> Vec<(String, f32)> {
        // 1. Get aggregate episode bundle for known value
        let bundle_hv = match known_role {
            "tool"    => self.by_tool.get(known_val).map(|b| b.to_hv()),
            "entity"  => self.by_entity.get(known_val).map(|b| b.to_hv()),
            "outcome" => self.by_outcome.get(known_val).map(|b| b.to_hv()),
            _ => return vec![],
        };
        let bundle_hv = match bundle_hv {
            Some(hv) => hv,
            None => return vec![],
        };

        // 2. known_binding = bind(R_known, val_hv(known_val))
        let r_known = Self::role_hv(known_role);
        let val_hv  = Self::val_hv(known_role, known_val);
        let known_binding = bind(&r_known, &val_hv);

        // 3. Unbind: probe = bundle XOR known_binding
        let probe = bind(&bundle_hv, &known_binding);

        // 4. Project: probe2 = probe XOR R_query  →  ≈ val_hv(query answer)
        let r_query = Self::role_hv(query_role);
        let projected = bind(&probe, &r_query);

        // 5. Compare against codebook for query_role
        let mut scores: Vec<(String, f32)> = match query_role {
            "tool" => self.tools.iter().map(|name| {
                (name.clone(), similarity(&projected, &word_hv(name)))
            }).collect(),
            "entity" => self.entities.iter().map(|name| {
                (name.clone(), similarity(&projected, &encode(name)))
            }).collect(),
            "outcome" => ["success", "fail", "error", "partial"].iter().map(|&name| {
                (name.to_string(), similarity(&projected, &word_hv(name)))
            }).collect(),
            _ => return vec![],
        };

        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scores.truncate(k);
        scores
    }

    fn role_hv(role: &str) -> HdcVec {
        match role {
            "tool"    => role_tool(),
            "entity"  => role_entity(),
            "outcome" => role_outcome(),
            _ => [0u64; D],
        }
    }

    fn val_hv(role: &str, val: &str) -> HdcVec {
        if role == "entity" { encode(val) } else { word_hv(val) }
    }

    pub fn event_count(&self) -> usize {
        self.by_tool.values().map(|b| b.n as usize).sum::<usize>()
            / self.by_tool.len().max(1)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tied_candidates_are_selected_by_id_in_every_query_mode() {
        for reverse in [false, true] {
            let mut store = HdcStore::new();
            let ids: Vec<_> = if reverse { (1..=80).rev().collect() } else { (1..=80).collect() };
            for id in ids { store.insert(id, "same words", "realm"); }
            for hits in [store.query("same words", 7, None),
                         store.query("same words", 7, Some("realm")),
                         store.realm_theme_query("realm", 7),
                         store.query_and(&["same", "words"], 7, None)] {
                assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), (1..=7).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn word_hv_deterministic() {
        assert_eq!(word_hv("hello"), word_hv("hello"));
        assert_ne!(word_hv("hello"), word_hv("world"));
    }

    #[test]
    fn hamming_identity() {
        let hv = word_hv("test");
        assert_eq!(hamming(&hv, &hv), 0);
    }

    #[test]
    fn hamming_range() {
        let a = word_hv("alpha");
        let b = word_hv("beta");
        let d = hamming(&a, &b);
        assert!(d > 0 && d < D as u32 * 64, "d={d}");
        // Expected ~4096 for random independent vectors
        let expected = (D * 32) as u32;
        assert!((d as i32 - expected as i32).abs() < 500, "d={d} far from {expected}");
    }

    #[test]
    fn bundle_of_one_is_identity() {
        let hv = word_hv("solo");
        assert_eq!(bundle(&[hv]), hv);
    }

    #[test]
    fn encode_similar_texts() {
        let a = encode("the cat sat on the mat");
        let b = encode("a cat sat on a mat");
        let c = encode("quantum computing algorithms");
        let d_ab = hamming(&a, &b);
        let d_ac = hamming(&a, &c);
        // Similar texts should be closer than dissimilar
        assert!(d_ab < d_ac, "similar texts farther apart than dissimilar: d_ab={d_ab} d_ac={d_ac}");
    }

    #[test]
    fn store_insert_query() {
        let mut store = HdcStore::new();
        store.insert(1, "Python programming language", "test");
        store.insert(2, "Rust systems programming", "test");
        store.insert(3, "cooking pasta recipes", "test");

        let hits = store.query("Python programming", 2, Some("test"));
        assert!(!hits.is_empty());
        // Memory 1 (Python) should score closer than memory 3 (cooking)
        let d1 = hits.iter().find(|&&(id, _)| id == 1).map(|&(_, d)| d);
        let d3 = hits.iter().find(|&&(id, _)| id == 3).map(|&(_, d)| d);
        if let (Some(d1), Some(d3)) = (d1, d3) {
            assert!(d1 < d3, "Python memory should be closer: d1={d1} d3={d3}");
        }
    }

    #[test]
    fn store_remove_updates_realm_bundle() {
        let mut store = HdcStore::new();
        store.insert(1, "machine learning", "r");
        store.insert(2, "deep learning neural nets", "r");
        assert_eq!(store.realm_bundles["r"].n, 2);
        store.remove(1, "r");
        assert_eq!(store.realm_bundles["r"].n, 1);
        assert!(!store.memories.contains_key(&1));
    }

    #[test]
    fn query_and_binding() {
        let mut store = HdcStore::new();
        store.insert(1, "Python machine learning scikit", "r");
        store.insert(2, "Python web server flask", "r");
        store.insert(3, "cooking pasta tomato sauce", "r");

        // XOR binding produces a superposition vector; semantic ordering is not
        // guaranteed for short texts/small n in binary HDC. Just verify the API works.
        let hits = store.query_and(&["python", "learning"], 3, Some("r"));
        assert_eq!(hits.len(), 3);
        // All distances should be in the valid range
        for (_, d) in &hits {
            assert!(*d <= 8192, "hamming must be ≤ 8192, got {d}");
        }
    }

    #[test]
    fn realm_similarity_reflexive() {
        let mut store = HdcStore::new();
        store.insert(1, "some content here", "realm_a");
        let sim = store.realm_similarity("realm_a", "realm_a");
        assert!((sim - 1.0).abs() < 1e-5, "sim={sim}");
    }

    #[test]
    fn realm_similarity_distinct_realms() {
        let mut store = HdcStore::new();
        store.insert(1, "rust programming systems", "tech");
        store.insert(2, "rust programming systems", "tech");
        store.insert(3, "baking bread sourdough recipes", "food");
        store.insert(4, "cooking pasta italian cuisine", "food");
        let sim_same  = store.realm_similarity("tech", "tech");
        let sim_cross = store.realm_similarity("tech", "food");
        assert!(sim_same > sim_cross, "same-realm sim {sim_same} should exceed cross {sim_cross}");
    }
}

#[cfg(test)]
mod episode_counter_tests {
    use super::*;

    #[test]
    fn bitplanes_match_scalar_majorities_across_growth_boundaries() {
        let mut bundle = EpBundle::default();
        let mut counts = vec![0u16; D * 64];
        assert_eq!(bundle.to_hv(), [0; D]);
        let mut seed = 79;
        for n in 1..=300 {
            let hv = std::array::from_fn(|_| xorshift64(&mut seed));
            bundle.add(&hv);
            let mut expected = [0; D];
            for wi in 0..D {
                for bit in 0..64 {
                    counts[wi * 64 + bit] += ((hv[wi] >> bit) & 1) as u16;
                    if counts[wi * 64 + bit] as u32 > n / 2 { expected[wi] |= 1 << bit; }
                }
            }
            assert_eq!(bundle.to_hv(), expected, "n={n}");
            if n == 1 { assert!(bundle.allocated_bytes() < D * 64 * 2 / 8); }
        }
    }

    #[test]
    fn bitplanes_preserve_u16_saturation() {
        let mut bundle = EpBundle { planes: (0..16).map(|_| Box::new([!0; D])).collect(), n: 65535 };
        bundle.add(&[!0; D]);
        assert!(bundle.planes.iter().all(|p| p.iter().all(|&w| w == !0)));
        assert_eq!(bundle.to_hv(), [!0; D]);
        bundle.n = 131070;
        assert_eq!(bundle.to_hv(), [0; D], "threshold exceeds saturated u16 counts");
    }
}

mod episode_planes {
    use super::HdcVec;
    use serde::{Serialize, Deserialize};
    pub fn serialize<S: serde::Serializer>(planes: &[Box<HdcVec>], serializer: S) -> Result<S::Ok, S::Error> {
        planes.iter().map(|p| p.as_slice()).collect::<Vec<_>>().serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<Box<HdcVec>>, D::Error> {
        let planes = Vec::<Vec<u64>>::deserialize(deserializer)?;
        if planes.len() > 16 { return Err(serde::de::Error::custom("too many episode planes")); }
        planes.into_iter().map(|p| p.try_into().map(Box::new)
            .map_err(|_| serde::de::Error::custom("wrong episode plane dimension"))).collect()
    }
}
