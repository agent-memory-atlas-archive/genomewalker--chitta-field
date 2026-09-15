use crate::ids::{ArtifactId, MemoryId};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

// --- Identity-atom extraction (Rust port of bench/atoms.py, validated at
// 116/140 bridge recovery + false-atom junk 4->1). An atom is a filesystem path
// OR a stable id token (DOI, version, run-id), NOT a bare prose slash-word and
// NOT a URL host. Tightened after junk_eyeball found the loose /[\w./-]{12,}
// matched /misspecification, /contamination, //www.w3.org/2000/svg as if paths.
fn kv_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:input|output|out|log|path):\s*([^\s|,;]+)").unwrap())
}
fn abs_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"/[\w./\-]{12,}").unwrap())
}
fn urlish_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^//|(?:^|/)(?:www\.|[\w-]+\.(?:org|com|net|io|gov|edu)/)").unwrap())
}

// --- grounded-entity atoms (env-gated experiment). CHITTA_BRIDGE_ATOMS:
//   unset|paths -> paths only (shipped default, byte-identical)
//   safe        -> + uuid/hash/accession (structurally unmistakable identities)
//   all|code    -> + snake_case/CamelCase code symbols (df-gate does precision)
// Entity classes are STRUCTURED tokens so they don't catch prose; the df gate in
// bridge_candidates keeps only rare ones. Read once.
fn atoms_mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("CHITTA_BRIDGE_ATOMS").as_deref() {
        Ok("safe") => 1,
        Ok("all") | Ok("code") => 2,
        _ => 0,
    })
}
fn uuid_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b").unwrap())
}
fn hex_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b[0-9a-f]{8,64}\b").unwrap())
}
fn acc_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b(?:GC[AF]_\d{9}\.\d+|[A-Z]{2,6}\d{4,}(?:\.\d+)?)\b").unwrap())
}
fn snake_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b").unwrap())
}
fn camel_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+){1,}\b").unwrap())
}
// a hex run is an id only if it mixes a-f letters AND digits (else it's a word or a number)
fn is_hash(t: &str) -> bool {
    t.chars().any(|c| ('a'..='f').contains(&c)) && t.chars().any(|c| c.is_ascii_digit())
}

fn valid_abs(p: &str) -> bool {
    if urlish_re().is_match(p) {
        return false; // protocol-relative or URL host
    }
    let body = &p[1..]; // p is guaranteed to start with '/'
    if body.contains('/') {
        return true; // >=2 segments -> real path
    }
    // single segment: keep only with an id signal (dot/version/digit), e.g. a
    // DOI 2024.11.18.624148 or file.ext; drop bare words (/misspecification).
    // ceiling: a 2-segment plain-word jargon token (/derived/inference) still
    // passes; rare, high-df -> low IDF weight. Tightening further dropped 24
    // genuine-bridge recoveries (83%->66%), so not worth it.
    body.contains('.') || body.chars().any(|c| c.is_ascii_digit())
}

/// Extract identity atoms (paths / stable ids) from memory content.
pub fn extract_artifact_paths(content: &str) -> Vec<String> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<String> = BTreeSet::new();
    for cap in kv_re().captures_iter(content) {
        let p = &cap[1];
        if p.len() >= 8 && p.contains('/') && !urlish_re().is_match(p) {
            set.insert(p.to_string());
        }
    }
    for m in abs_re().find_iter(content) {
        let p = m.as_str();
        if p.len() >= 12 && valid_abs(p) {
            set.insert(p.to_string());
        }
    }
    let mode = atoms_mode();
    if mode >= 1 {
        for m in uuid_re().find_iter(content) {
            set.insert(m.as_str().to_string());
        }
        for m in hex_re().find_iter(content) {
            let t = m.as_str();
            if is_hash(t) {
                set.insert(t.to_string());
            }
        }
        for m in acc_re().find_iter(content) {
            set.insert(m.as_str().to_string());
        }
    }
    if mode >= 2 {
        for m in snake_re().find_iter(content) {
            if m.as_str().len() >= 8 {
                set.insert(m.as_str().to_string());
            }
        }
        for m in camel_re().find_iter(content) {
            if m.as_str().len() >= 8 {
                set.insert(m.as_str().to_string());
            }
        }
    }
    set.into_iter().collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub memory_id: MemoryId,
    pub artifact_id: ArtifactId,
    pub normalized_path: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactIndex {
    /// path -> list of entries
    by_path: HashMap<String, Vec<ArtifactEntry>>,
    /// memory_id -> list of paths (for reverse lookup / cleanup)
    by_memory: HashMap<MemoryId, Vec<String>>,
}

impl ArtifactIndex {
    pub fn new() -> Self {
        Self {
            by_path: HashMap::new(),
            by_memory: HashMap::new(),
        }
    }

    /// Register an association between a memory and a file path.
    pub fn associate(
        &mut self,
        memory_id: MemoryId,
        artifact_id: ArtifactId,
        path: &str,
        strength: f32,
    ) {
        let entry = ArtifactEntry {
            memory_id,
            artifact_id,
            normalized_path: path.to_string(),
            strength,
        };

        let entries = self
            .by_path
            .entry(path.to_string())
            .or_insert_with(Vec::new);
        // Avoid duplicates: replace if same memory_id already present for this path.
        if let Some(pos) = entries.iter().position(|e| e.memory_id == memory_id) {
            entries[pos] = entry;
        } else {
            entries.push(entry);
        }

        self.by_memory
            .entry(memory_id)
            .or_insert_with(Vec::new)
            .push(path.to_string());
    }

    /// Remove all associations for a memory (on forget).
    pub fn remove_memory(&mut self, memory_id: MemoryId) {
        if let Some(paths) = self.by_memory.remove(&memory_id) {
            for path in paths {
                if let Some(entries) = self.by_path.get_mut(&path) {
                    entries.retain(|e| e.memory_id != memory_id);
                    if entries.is_empty() {
                        self.by_path.remove(&path);
                    }
                }
            }
        }
    }

    /// Query memories associated with a given path (exact match), sorted by strength desc.
    pub fn query_path(&self, path: &str, limit: usize) -> Vec<ArtifactEntry> {
        let Some(entries) = self.by_path.get(path) else {
            return Vec::new();
        };
        let mut result: Vec<ArtifactEntry> = entries.clone();
        result.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(limit);
        result
    }

    /// Query memories associated with any path matching a prefix.
    pub fn query_prefix(&self, prefix: &str, limit: usize) -> Vec<ArtifactEntry> {
        let mut result: Vec<ArtifactEntry> = self
            .by_path
            .iter()
            .filter(|(path, _)| path.starts_with(prefix))
            .flat_map(|(_, entries)| entries.iter().cloned())
            .collect();
        result.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result.truncate(limit);
        result
    }

    /// Get all paths associated with a memory.
    pub fn paths_for_memory(&self, memory_id: MemoryId) -> Vec<String> {
        self.by_memory.get(&memory_id).cloned().unwrap_or_default()
    }

    /// df-gated IDF postings join — the factual-bridge retrieval leg.
    /// Given an anchor's atom paths, returns co-atom memories scored by the sum
    /// of IDF weights ln(n_total/df) over shared atoms whose df <= tau. Atoms
    /// above the gate (common paths) are dropped to prevent flooding; `exclude`
    /// (the anchor) is removed; ZERO cosine. Returns empty when no gated atom
    /// fires — silence is the anti-flood guarantee (only ~4% of memories carry
    /// a gated atom). `by_path.get(p).len()` is the document frequency for free.
    pub fn bridge_candidates(
        &self,
        anchor_paths: &[String],
        n_total: usize,
        tau: usize,
        exclude: MemoryId,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        for path in anchor_paths {
            if let Some(entries) = self.by_path.get(path) {
                let dfreq = entries.len();
                if dfreq == 0 || dfreq > tau {
                    continue;
                }
                let w = (n_total as f32 / dfreq as f32).ln();
                for e in entries {
                    if e.memory_id == exclude {
                        continue;
                    }
                    *scores.entry(e.memory_id).or_insert(0.0) += w;
                }
            }
        }
        let mut v: Vec<(MemoryId, f32)> = scores.into_iter().collect();
        v.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then(a.0.cmp(&b.0))
        });
        v.truncate(k);
        v
    }

    /// Saturating-IDF postings join for the lane-0 bridge (CHITTA_BRIDGE_LANE0).
    /// Same co-atom join as `bridge_candidates` but replaces the hard df <= tau
    /// gate with a smooth BM25 IDF `ln((N - df + 0.5)/(df + 0.5) + 1)`: common
    /// atoms decay to ~0 weight instead of being excluded, so a shared path with
    /// df=40 still contributes a little rather than nothing. Non-negative, bounded
    /// by ln(N+1). ZERO cosine; `exclude` (the anchor) is dropped.
    pub fn bridge_candidates_saturating(
        &self,
        anchor_paths: &[String],
        n_total: usize,
        exclude: MemoryId,
        k: usize,
    ) -> Vec<(MemoryId, f32)> {
        let n = n_total as f32;
        let mut scores: HashMap<MemoryId, f32> = HashMap::new();
        for path in anchor_paths {
            if let Some(entries) = self.by_path.get(path) {
                let dfreq = entries.len();
                if dfreq == 0 {
                    continue;
                }
                let df = dfreq as f32;
                let w = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                if w <= 0.0 {
                    continue;
                }
                for e in entries {
                    if e.memory_id == exclude {
                        continue;
                    }
                    *scores.entry(e.memory_id).or_insert(0.0) += w;
                }
            }
        }
        let mut v: Vec<(MemoryId, f32)> = scores.into_iter().collect();
        v.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then(a.0.cmp(&b.0))
        });
        v.truncate(k);
        v
    }

    pub fn len(&self) -> usize {
        self.by_path.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_associate_query() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/main.cpp", 0.9);
        idx.associate(2, 10, "src/main.cpp", 0.7);
        idx.associate(3, 20, "src/other.cpp", 1.0);
        let hits = idx.query_path("src/main.cpp", 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory_id, 1); // higher strength first
    }

    #[test]
    fn test_artifact_remove() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/main.cpp", 0.9);
        idx.remove_memory(1);
        let hits = idx.query_path("src/main.cpp", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_query_prefix() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/main.cpp", 0.9);
        idx.associate(2, 11, "src/utils.cpp", 0.8);
        idx.associate(3, 12, "include/foo.h", 0.7);
        let hits = idx.query_prefix("src/", 10);
        assert_eq!(hits.len(), 2);
        // sorted by strength desc
        assert_eq!(hits[0].normalized_path, "src/main.cpp");
    }

    #[test]
    fn test_paths_for_memory() {
        let mut idx = ArtifactIndex::new();
        idx.associate(1, 10, "src/a.cpp", 0.9);
        idx.associate(1, 11, "src/b.cpp", 0.8);
        let paths = idx.paths_for_memory(1);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains(&"src/a.cpp".to_string()));
        assert!(paths.contains(&"src/b.cpp".to_string()));
    }

    // Mirrors bench/atoms.py __main__ gate: 6 real atoms kept, 4 false dropped.
    #[test]
    fn test_extractor_gate() {
        let keep = [
            "path: /maps/projects/fernandezguerra/apps/repos/mcaf/bam2mcaf.cpp",
            "log: /finalize_tau.cpp",
            "out: /2024.11.18.624148",
            "input: /.claude/mind",
            "path: /genopack/checksum.hpp",
            "output: /projects/caeg/scratch/kbd606/tmp/pgsock",
        ];
        for s in keep {
            assert!(!extract_artifact_paths(s).is_empty(), "wrongly dropped: {s}");
        }
        // bare prose slash-words and URL boilerplate must NOT become atoms
        let drop = [
            "the model /misspecification is severe",
            "completeness /contamination duality",
            "xmlns=//www.w3.org/2000/svg viewBox",
            "see http://www.w3.org/2000/svg here",
        ];
        for s in drop {
            assert!(extract_artifact_paths(s).is_empty(), "wrongly kept atom in: {s}");
        }
    }

    // Anchor A shares a RARE path with buried partner B (df=2, fires) and a
    // COMMON path with a crowd (df>tau, gated out). B must surface; the crowd
    // must not flood. Zero cosine — pure postings join.
    #[test]
    fn test_bridge_recovers_buried_partner() {
        let mut idx = ArtifactIndex::new();
        let (a, b, tau, n) = (1u64, 2u64, 32usize, 1000usize);
        idx.associate(a, 100, "/maps/repo/rare_bridge.rs", 1.0);
        idx.associate(b, 100, "/maps/repo/rare_bridge.rs", 1.0);
        // common path shared by A and 40 unrelated memories -> df=41 > tau
        idx.associate(a, 200, "/.claude/mind", 1.0);
        for m in 1000..1040u64 {
            idx.associate(m, 200, "/.claude/mind", 1.0);
        }
        let paths = idx.paths_for_memory(a);
        let cands = idx.bridge_candidates(&paths, n, tau, a, 10);
        assert_eq!(cands.first().map(|c| c.0), Some(b), "B not surfaced first");
        // none of the df>tau crowd leaked in (common path was gated)
        assert!(cands.iter().all(|c| c.0 < 1000), "flood: gated crowd leaked");
        // silence when the anchor has no gated atom
        let mut idx2 = ArtifactIndex::new();
        for m in 0..50u64 {
            idx2.associate(m, 300, "/common/only", 1.0); // df=50 > tau
        }
        assert!(idx2.bridge_candidates(&idx2.paths_for_memory(0), n, tau, 0, 10).is_empty());
    }

    // Lane-0 saturating IDF replaces the hard df>tau gate with a down-weight:
    // the common-atom crowd is KEPT (no cliff) but ranks strictly below a
    // rare-atom partner, and the anchor itself is excluded.
    #[test]
    fn test_saturating_idf_ranks_rare_over_common() {
        let mut idx = ArtifactIndex::new();
        let (a, b, n) = (1u64, 2u64, 1000usize);
        idx.associate(a, 100, "/maps/repo/rare_bridge.rs", 1.0); // df=2
        idx.associate(b, 100, "/maps/repo/rare_bridge.rs", 1.0);
        idx.associate(a, 200, "/.claude/mind", 1.0); // common: df=41
        for m in 1000..1040u64 {
            idx.associate(m, 200, "/.claude/mind", 1.0);
        }
        let cands = idx.bridge_candidates_saturating(&idx.paths_for_memory(a), n, a, 100);
        assert_eq!(cands.first().map(|c| c.0), Some(b), "rare-atom partner not first");
        assert!(cands.iter().all(|c| c.0 != a), "anchor leaked into candidates");
        // common crowd is retained (no hard gate) but strictly below the partner
        let b_score = cands[0].1;
        let crowd: Vec<f32> = cands.iter().filter(|c| c.0 >= 1000).map(|c| c.1).collect();
        assert_eq!(crowd.len(), 40, "common crowd was gated instead of down-weighted");
        assert!(crowd.iter().all(|&s| s < b_score), "common atom outweighed rare atom");
    }
}
