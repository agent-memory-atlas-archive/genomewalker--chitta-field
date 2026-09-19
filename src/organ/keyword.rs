use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// BM25 parameters (standard defaults).
const K1: f32 = 1.2;
const B: f32 = 0.75;
const MAX_QUERY_TERMS: usize = 4;
const COMMON_TERM_RATIO: f32 = 0.20;

/// A posting: memory_id + term frequency in that document.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    memory_id: MemoryId,
    tf: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordIndex {
    /// term -> postings list
    postings: HashMap<String, Vec<Posting>>,
    /// memory_id -> document length (token count)
    doc_lengths: HashMap<MemoryId, u32>,
    /// Reverse map used to remove or reindex a single document efficiently.
    #[serde(skip)]
    doc_terms: HashMap<MemoryId, Vec<u32>>,
    #[serde(skip)]
    term_names: Vec<Option<std::sync::Arc<str>>>,
    #[serde(skip)]
    term_ids: HashMap<std::sync::Arc<str>, u32>,
    #[serde(skip)]
    free_term_ids: Vec<u32>,
    /// running total tokens across all documents
    total_tokens: u64,
    total_docs: u32,
}

/// Prepared from immutable postings; publish while retaining exclusive writer ownership.
pub(crate) struct KeywordReverseIndex {
    doc_terms: HashMap<MemoryId, Vec<u32>>,
    term_names: Vec<Option<std::sync::Arc<str>>>,
    term_ids: HashMap<std::sync::Arc<str>, u32>,
}

#[derive(Debug, Clone)]
pub struct KeywordHit {
    pub memory_id: MemoryId,
    pub bm25_score: f32,
}

impl KeywordIndex {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_terms: HashMap::new(),
            term_names: Vec::new(),
            term_ids: HashMap::new(),
            free_term_ids: Vec::new(),
            total_tokens: 0,
            total_docs: 0,
        }
    }

    /// Index a memory's text content.
    pub fn index(&mut self, memory_id: MemoryId, content: &str) {
        // Remove old postings for this memory if re-indexing.
        self.remove(memory_id);

        let tokens = tokenize(content);
        let doc_len = tokens.len() as u32;

        if doc_len == 0 {
            return;
        }

        // Count term frequencies.
        let mut tf_map: HashMap<String, u32> = HashMap::new();
        for token in tokens {
            *tf_map.entry(token).or_insert(0) += 1;
        }

        // Insert into postings lists.
        let mut reverse_terms = Vec::with_capacity(tf_map.len());
        for (term, tf) in tf_map {
            let term_id = self.intern_term(&term);
            self.postings
                .entry(term)
                .or_insert_with(Vec::new)
                .push(Posting { memory_id, tf });
            reverse_terms.push(term_id);
        }

        self.doc_terms.insert(memory_id, reverse_terms);
        self.doc_lengths.insert(memory_id, doc_len);
        self.total_tokens += doc_len as u64;
        self.total_docs += 1;
    }

    /// Remove a memory from the index (on forget).
    pub fn remove(&mut self, memory_id: MemoryId) {
        if let Some(old_len) = self.doc_lengths.remove(&memory_id) {
            self.total_tokens = self.total_tokens.saturating_sub(old_len as u64);
            self.total_docs = self.total_docs.saturating_sub(1);

            if let Some(terms) = self.doc_terms.remove(&memory_id) {
                for term_id in terms {
                    let Some(term) = self.term_names.get(term_id as usize).and_then(Option::as_ref) else { continue; };
                    let remove_term = if let Some(postings) = self.postings.get_mut(term.as_ref()) {
                        postings.retain(|p| p.memory_id != memory_id);
                        postings.is_empty()
                    } else {
                        false
                    };
                    if remove_term {
                        self.postings.remove(term.as_ref());
                        self.term_ids.remove(term.as_ref());
                        self.term_names[term_id as usize] = None;
                        self.free_term_ids.push(term_id);
                    }
                }
            } else {
                // Snapshot bodies omit the reverse map. WAL replay and writes
                // before maintenance publishes it must still remove old postings.
                self.postings.retain(|_, postings| {
                    postings.retain(|p| p.memory_id != memory_id);
                    !postings.is_empty()
                });
            }
        }
    }

    /// BM25 search. Returns hits sorted by score descending.
    pub fn search(&self, query: &str, k: usize) -> Vec<KeywordHit> {
        if self.total_docs == 0 || k == 0 {
            return Vec::new();
        }

        let query_terms = dedup_terms(tokenize(query));
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n = self.total_docs as f32;
        let avgdl = self.total_tokens as f32 / n;
        let term_infos = self.query_terms(&query_terms, n);
        if term_infos.is_empty() {
            return Vec::new();
        }

        let mut scores: HashMap<MemoryId, f32> = HashMap::new();

        for term_info in term_infos {
            for posting in term_info.postings {
                let doc_len = *self.doc_lengths.get(&posting.memory_id).unwrap_or(&1) as f32;
                let tf = posting.tf as f32;
                let tf_norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * doc_len / avgdl));
                *scores.entry(posting.memory_id).or_insert(0.0) += term_info.idf * tf_norm;
            }
        }

        let mut top_k = BinaryHeap::new();
        for (memory_id, bm25_score) in scores {
            push_top_k(&mut top_k, k, memory_id, bm25_score);
        }

        heap_to_hits(top_k)
    }

    pub fn doc_count(&self) -> usize {
        self.total_docs as usize
    }

    fn intern_term(&mut self, term: &str) -> u32 {
        if let Some(&id) = self.term_ids.get(term) { return id; }
        let name: std::sync::Arc<str> = term.into();
        let id = if let Some(id) = self.free_term_ids.pop() {
            self.term_names[id as usize] = Some(name.clone());
            id
        } else {
            let id = u32::try_from(self.term_names.len()).expect("keyword vocabulary exceeds u32");
            self.term_names.push(Some(name.clone()));
            id
        };
        self.term_ids.insert(name, id);
        id
    }

    pub fn rebuild_reverse_index(&mut self) {
        let prepared = self.prepare_reverse_index(|| false).expect("uncancelled rebuild");
        self.publish_reverse_index(prepared);
    }

    /// Readers keep using forward postings while this cancellable preparation
    /// runs under an upgradable read guard. The guard prevents intervening writes.
    pub(crate) fn prepare_reverse_index(
        &self, mut cancelled: impl FnMut() -> bool,
    ) -> Option<KeywordReverseIndex> {
        let mut result = KeywordReverseIndex {
            doc_terms: HashMap::with_capacity(self.doc_lengths.len()),
            term_names: Vec::with_capacity(self.postings.len()),
            term_ids: HashMap::with_capacity(self.postings.len()),
        };
        for (term, postings) in &self.postings {
            if cancelled() { return None; }
            let id = u32::try_from(result.term_names.len()).expect("keyword vocabulary exceeds u32");
            let name: std::sync::Arc<str> = term.as_str().into();
            result.term_names.push(Some(name.clone()));
            result.term_ids.insert(name, id);
            for (i, posting) in postings.iter().enumerate() {
                if i % 1024 == 0 && cancelled() { return None; }
                result.doc_terms.entry(posting.memory_id).or_default().push(id);
            }
        }
        for terms in result.doc_terms.values_mut() {
            if cancelled() { return None; }
            terms.shrink_to_fit();
        }
        Some(result)
    }

    pub(crate) fn publish_reverse_index(&mut self, prepared: KeywordReverseIndex) {
        self.doc_terms = prepared.doc_terms;
        self.term_names = prepared.term_names;
        self.term_ids = prepared.term_ids;
        self.free_term_ids.clear();
    }

    pub(crate) fn allocated_bytes(&self) -> (usize, usize, usize) {
        use crate::profile::map_bytes;
        let postings = map_bytes(&self.postings) + map_bytes(&self.doc_lengths)
            + self.postings.iter().map(|(term, p)| term.capacity()
                + p.capacity() * std::mem::size_of::<Posting>()).sum::<usize>();
        let reverse = map_bytes(&self.doc_terms) + self.doc_terms.values().map(|v| v.capacity() * 4).sum::<usize>()
            + map_bytes(&self.term_ids) + self.term_names.capacity() * std::mem::size_of::<Option<std::sync::Arc<str>>>()
            + self.term_names.iter().flatten().map(|s| s.len() + 16).sum::<usize>()
            + self.free_term_ids.capacity() * 4;
        // Counterfactual estimate of the pre-change reverse map on these exact
        // postings, excluding malloc headers/alignment (one allocation per term).
        let legacy = map_bytes(&self.doc_terms) + self.doc_terms.values().map(|v|
            v.len().next_power_of_two().max(4) * std::mem::size_of::<(String, u32)>()
                + v.iter().map(|&id| self.term_names[id as usize].as_ref().map_or(0, |s| s.len())).sum::<usize>()).sum::<usize>();
        (postings, reverse, legacy)
    }

    fn query_terms<'a>(&'a self, query_terms: &[String], n: f32) -> Vec<QueryTerm<'a>> {
        let mut terms: Vec<QueryTerm<'a>> = query_terms
            .iter()
            .filter_map(|term| {
                let postings = self.postings.get(term)?;
                let df = postings.len();
                let idf = ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
                Some(QueryTerm { postings, df, idf })
            })
            .collect();

        if terms.is_empty() {
            return terms;
        }

        terms.sort_unstable_by(|a, b| {
            a.df.cmp(&b.df)
                .then_with(|| b.idf.partial_cmp(&a.idf).unwrap_or(Ordering::Equal))
        });

        let min_df = terms[0].df;
        let has_selective = min_df < self.total_docs as usize;
        if has_selective {
            terms.retain(|term| term.df == min_df || (term.df as f32 / n) <= COMMON_TERM_RATIO);
        }

        terms.truncate(MAX_QUERY_TERMS);
        terms
    }

    /// IDF for a single term (BM25 variant). Returns 0.0 if term not in index.
    pub fn token_idf(&self, term: &str) -> f32 {
        if self.total_docs == 0 { return 0.0; }
        let df = self.postings.get(term).map(|p| p.len()).unwrap_or(0);
        if df == 0 { return 0.0; }
        let n = self.total_docs as f32;
        ((n - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln()
    }

    /// Max IDF across all query tokens. Useful as a rare-entity signal for scoring.
    pub fn query_max_idf(&self, query: &str) -> f32 {
        let terms = dedup_terms(tokenize(query));
        terms.iter().map(|t| self.token_idf(t)).fold(0.0f32, f32::max)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let stop_words = [
        "the", "a", "an", "is", "are", "was", "be", "to", "of", "and", "in", "it", "for", "on",
        "with", "as", "at", "by", "or", "not", "this", "that", "from", "have", "has", "had", "but",
        "its", "my", "your", "we", "i", "he", "she", "they", "do", "did", "will", "can", "would",
        "could", "should",
    ];
    text.split(|c: char| !c.is_alphanumeric())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 2 && !stop_words.contains(&t.as_str()))
        .collect()
}

fn dedup_terms(terms: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(terms.len());
    let mut out = Vec::with_capacity(terms.len());
    for term in terms {
        if seen.insert(term.clone()) {
            out.push(term);
        }
    }
    out
}

struct QueryTerm<'a> {
    postings: &'a [Posting],
    df: usize,
    idf: f32,
}

#[derive(Clone, Copy, Debug)]
struct RankedHit {
    score: f32,
    memory_id: MemoryId,
}

impl PartialEq for RankedHit {
    fn eq(&self, other: &Self) -> bool {
        self.memory_id == other.memory_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for RankedHit {}

impl Ord for RankedHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.memory_id.cmp(&other.memory_id))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn push_top_k(heap: &mut BinaryHeap<RankedHit>, k: usize, memory_id: MemoryId, score: f32) {
    let candidate = RankedHit { score, memory_id };
    if heap.len() < k {
        heap.push(candidate);
        return;
    }
    let Some(worst) = heap.peek() else {
        heap.push(candidate);
        return;
    };
    if score > worst.score || (score == worst.score && memory_id < worst.memory_id) {
        heap.pop();
        heap.push(candidate);
    }
}

fn heap_to_hits(heap: BinaryHeap<RankedHit>) -> Vec<KeywordHit> {
    let mut ranked = heap.into_vec();
    ranked.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    ranked
        .into_iter()
        .map(|hit| KeywordHit {
            memory_id: hit.memory_id,
            bm25_score: hit.score,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "the quick brown fox jumps over the lazy dog");
        idx.index(2, "rust programming language memory safety");
        idx.index(3, "fox and hound friendship story");

        let hits = idx.search("fox", 10);
        assert_eq!(hits.len(), 2);
        let ids: Vec<u64> = hits.iter().map(|h| h.memory_id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_idf_boost() {
        let mut idx = KeywordIndex::new();
        // "rust" appears in only doc 2 — should score higher for "rust" query
        idx.index(1, "programming language safety");
        idx.index(2, "rust programming language memory safety");
        idx.index(3, "programming language design");

        let hits = idx.search("rust", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].memory_id, 2);
    }

    #[test]
    fn test_remove() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "hello world foo bar");
        idx.index(2, "hello world baz");
        idx.remove(1);

        let hits = idx.search("hello", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 2);
    }

    #[test]
    fn test_multi_term() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "cognitive memory system architecture design");
        idx.index(2, "database system architecture sql");
        idx.index(3, "cognitive science brain research");

        // "cognitive architecture" — doc 1 has both, should rank highest
        let hits = idx.search("cognitive architecture", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].memory_id, 1);
    }

    #[test]
    fn test_common_terms_do_not_swamp_selective_term() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "architecture benchmark topic17 project3");
        idx.index(2, "architecture benchmark topic22 project9");
        idx.index(3, "architecture benchmark topic17 project8");

        let hits = idx.search("architecture benchmark topic17", 10);
        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .all(|hit| hit.memory_id == 1 || hit.memory_id == 3));
    }

    #[test]
    fn test_duplicate_query_terms_are_ignored() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "rust ownership");
        idx.index(2, "rust lifetimes");

        let hits_once = idx.search("rust ownership", 10);
        let hits_dup = idx.search("rust rust ownership ownership", 10);
        assert_eq!(hits_once.len(), hits_dup.len());
        assert_eq!(hits_once[0].memory_id, hits_dup[0].memory_id);
        assert!((hits_once[0].bm25_score - hits_dup[0].bm25_score).abs() < 1e-6);
    }

    #[test]
    fn test_empty_index() {
        let idx = KeywordIndex::new();
        let hits = idx.search("anything", 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_doc_count() {
        let mut idx = KeywordIndex::new();
        assert_eq!(idx.doc_count(), 0);
        idx.index(1, "hello world");
        assert_eq!(idx.doc_count(), 1);
        idx.index(2, "foo bar");
        assert_eq!(idx.doc_count(), 2);
        idx.remove(1);
        assert_eq!(idx.doc_count(), 1);
    }

    #[test]
    fn test_reindex() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "hello world");
        // Re-index same id with different content.
        idx.index(1, "rust programming");
        assert_eq!(idx.doc_count(), 1);

        // "hello" should no longer match doc 1.
        let hits = idx.search("hello", 10);
        assert!(hits.is_empty());

        // "rust" should match doc 1.
        let hits = idx.search("rust", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 1);
    }

    #[test]
    fn test_stop_words_filtered() {
        let mut idx = KeywordIndex::new();
        // "the", "is", "a" are stop words and should not be indexed.
        idx.index(1, "the sky is a deep blue");

        // Searching for stop words should return nothing.
        let hits = idx.search("the", 10);
        assert!(hits.is_empty());

        // Non-stop content still matches.
        let hits = idx.search("sky blue", 10);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_rebuild_reverse_index_supports_remove() {
        let mut idx = KeywordIndex::new();
        idx.index(1, "alpha beta gamma");
        idx.index(2, "beta delta");

        idx.doc_terms.clear();
        idx.rebuild_reverse_index();
        idx.remove(1);

        let hits = idx.search("alpha beta gamma", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 2);
    }
}

#[cfg(test)]
mod compact_reverse_tests {
    use super::*;

    #[test]
    fn mutations_before_reverse_publication_match_eager_scores() {
        let mut original = KeywordIndex::new();
        original.index(1, "shared alpha alpha");
        original.index(2, "shared beta");
        original.index(3, "shared gamma");
        let bytes = bincode::serialize(&original).unwrap();
        let mut delayed: KeywordIndex = bincode::deserialize(&bytes).unwrap();
        let mut eager: KeywordIndex = bincode::deserialize(&bytes).unwrap();
        eager.rebuild_reverse_index();
        for index in [&mut delayed, &mut eager] {
            index.index(1, "shared replacement");
            index.remove(2);
            index.index(4, "delta shared");
        }
        for publish in [false, true] {
            if publish { delayed.rebuild_reverse_index(); }
            for query in ["alpha", "beta", "gamma", "replacement", "delta"] {
                let scores = |index: &KeywordIndex| index.search(query, 10).into_iter()
                    .map(|hit| (hit.memory_id, hit.bm25_score.to_bits())).collect::<Vec<_>>();
                assert_eq!(scores(&delayed), scores(&eager), "{query}, published={publish}");
            }
        }
        delayed.remove(1);
        assert!(delayed.search("replacement", 10).is_empty());
    }

    #[test]
    fn cancelled_reverse_preparation_preserves_existing_index() {
        let mut index = KeywordIndex::new();
        for id in 1..=2048 { index.index(id, "shared alpha"); }
        let mut checks = 0;
        assert!(index.prepare_reverse_index(|| { checks += 1; checks == 3 }).is_none());
        assert_eq!(index.doc_count(), 2048);
        index.remove(1);
        assert_eq!(index.search("alpha", 3000).len(), 2047);
        assert!(!index.search("alpha", 3000).iter().any(|hit| hit.memory_id == 1));
        let prepared = index.prepare_reverse_index(|| false).unwrap();
        index.publish_reverse_index(prepared);
        index.remove(2);
        assert_eq!(index.search("alpha", 3000).len(), 2046);
    }

    #[test]
    fn term_ids_preserve_wire_format_and_survive_remove_reindex_and_rebuild() {
        let mut index = KeywordIndex::new();
        index.index(1, "shared alpha alpha");
        index.index(2, "shared beta");
        let bytes = bincode::serialize(&index).unwrap();
        // Decode the historical layout: runtime interning must add no wire fields.
        #[derive(Deserialize)]
        struct Legacy { postings: HashMap<String, Vec<Posting>>, doc_lengths: HashMap<MemoryId, u32>, total_tokens: u64, total_docs: u32 }
        let old: Legacy = bincode::deserialize(&bytes).unwrap();
        assert_eq!((old.total_tokens, old.total_docs, old.doc_lengths.len(), old.postings.len()), (5, 2, 2, 3));
        let mut restored: KeywordIndex = bincode::deserialize(&bytes).unwrap();
        restored.rebuild_reverse_index();
        for query in ["shared", "alpha", "beta"] {
            let expected = index.search(query, 5);
            let actual = restored.search(query, 5);
            assert_eq!(expected.iter().map(|h| (h.memory_id, h.bm25_score.to_bits())).collect::<Vec<_>>(),
                actual.iter().map(|h| (h.memory_id, h.bm25_score.to_bits())).collect::<Vec<_>>());
        }
        restored.remove(1);
        assert!(restored.search("alpha", 5).is_empty());
        assert_eq!(restored.search("shared", 5)[0].memory_id, 2);
        restored.index(3, "gamma shared"); // reuses freed term ID
        restored.index(2, "replacement");
        assert!(restored.search("beta", 5).is_empty());
        restored.rebuild_reverse_index();
        restored.remove(3);
        assert!(restored.search("gamma shared", 5).is_empty());
        assert_eq!(restored.search("replacement", 5)[0].memory_id, 2);
    }
}
