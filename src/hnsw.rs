use crate::ids::MemoryId;
use crate::binary::{binarize, hamming_dist, BINARY_WORDS, HAMMING_CANDIDATES};
use crate::ops::EMBED_DIM;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use turbovec::TurboQuantIndex;
use memmap2;
use rayon::prelude::*;
use smallvec::SmallVec;

const COARSE_CENTROIDS: usize = 256;
const COARSE_ASSIGNMENTS: usize = 2;
const MIN_PROBES: usize = 6;
const MAX_PROBES: usize = 24;
const MIN_CANDIDATES: usize = 1024;
const MAX_CANDIDATES: usize = 16384;
const LSH_TABLES: usize = 4;
const LSH_BITS: usize = 12;

// HNSW kicks in above this many memories — below it IVF+LSH is fast enough.
const HNSW_THRESHOLD: usize = 2000;
// Two-tier HNSW: above this count new inserts go to a small delta graph instead
// of the large base graph. Queries search both and merge. Reduces insert cost from
// O(log N_total) to O(log N_delta) at 1M+ scale.
// Lowered from 100K: most production deployments never reached 100K memories,
// so delta-merge never triggered and search latency grew unboundedly.
const HNSW_TIER2_THRESHOLD: usize = 5_000;
// Merge delta into base when delta exceeds this fraction of the base size.
const HNSW_DELTA_MERGE_RATIO: f64 = 0.10;
// HNSW build/search parameters
const HNSW_M: usize = 16;        // neighbors per non-zero layer
const HNSW_M0: usize = 32;       // neighbors at layer 0 (2×M)
const HNSW_EF_CONSTRUCTION: usize = 200; // candidates during insert
const HNSW_EF_SEARCH: usize = 64;        // candidates during search
// 1/ln(M) — controls layer probability distribution
const HNSW_ML: f64 = 0.36067; // 1/ln(16)
// Per-realm HNSW activates when a realm exceeds this count.
// Below this, binary Hamming + linear scan (search_candidates) handles it.
const PER_REALM_HNSW_THRESHOLD: usize = 500;

// Flat scan re-enabled by default 2026-07-04: the 2026-06-11 disable rested on
// LOCOMO F1 parity, but GOLDEN_SET v6 (154 realm-scoped queries) shows the
// binary-Hamming ANN path buries the true answer at rank 11-19 of 20 — the exact
// "relevant items buried" regression the constant was meant to guard. Root cause:
// hamming_candidates prefilters 108k memories to HAMMING_CANDIDATES=400 by lossy
// 1-bit sign codes, so the gold routinely misses the rescore set (measured: gold
// cosine 0.87, but sim=0.000 and rank 19 via the ANN path). mean nDCG@20 hybrid
// 0.331 → 0.712 with exact scan. The 12.5s/recall that motivated the disable was
// the SCALAR centered arm; the turbovec raw arm (default below) is exact-cosine,
// SIMD-quantized, and fast (154-query eval runs in seconds, CW-refresh ~1ms) —
// so re-enabling flat no longer reintroduces the scan-convoy. Above this cap the
// binary ANN path still serves (a separate quality problem at that scale).
// Override per-process with CHITTA_FLAT_SCAN_MAX=<n> (0 = force ANN path).
pub(crate) const FLAT_SCAN_MAX: usize = 2_000_000;

// Process-resolved flat-scan cap (env override, read once). Both the search
// short-circuit and the HNSW build gate key off this so they never diverge:
// at/below the cap the exact flat scan serves, so building/merging/snapshotting
// HNSW there is dead work (search never consults it). 0 = force the ANN path.
pub(crate) fn flat_scan_max() -> usize {
    use std::sync::OnceLock;
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("CHITTA_FLAT_SCAN_MAX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(FLAT_SCAN_MAX)
    })
}

// Skip the HNSW sidecar I/O (boot load + snapshot save). Below flat_scan_max the
// exact scan serves every query, so a corpus that never exceeds the cap keeps a
// stale on-disk graph that is loaded into RAM each boot and re-serialised each
// snapshot yet never consulted. With this set the graph is neither loaded nor
// saved; rebuild_hnsw() self-heals it lazily if a corpus ever crosses the cap.
// Default ON since the honest-gold A/B (n=161, 128k store): load vs skip left every
// recall stratum byte-identical (recall@k 0.969, single_hop 0.980, multihop 0.818) —
// the graph is never consulted at flat-scan scale. Kill switch: CHITTA_NO_HNSW_SIDECAR=0.
pub(crate) fn skip_hnsw_sidecar() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("CHITTA_NO_HNSW_SIDECAR").ok().as_deref() != Some("0")
    })
}

// .emb mmap activation threshold (field.rs): below this the embeddings stay
// in the heap, so scans never fault on a page whose backing file a
// consolidation prune (or, multi-node, a peer's sidecar rewrite over NFS)
// has unlinked. Deliberately DECOUPLED from FLAT_SCAN_MAX — flipping the
// search path must not silently arm the mmap hazard at small scale.
pub(crate) const EMB_MMAP_MIN: usize = 500_000;

/// Deterministic level assignment seeded by node ID — safe to call from parallel threads.
fn random_level_from_seed(seed: u64) -> usize {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    s ^= s >> 33;
    s = s.wrapping_mul(0xff51afd7ed558ccd);
    s ^= s >> 33;
    let f = (s >> 11) as f64 / (1u64 << 53) as f64;
    if f < 1e-15 { return 0; }
    ((-f.ln() * HNSW_ML).floor() as usize).min(16)
}

// ── Embedding lookup helper ───────────────────────────────────────────────────

/// Thin two-source embedding accessor used by HnswGraph methods.
/// Holds shared references to the heap HashMap and the mmap sidecar so callers
/// can split-borrow the struct fields (heap/offsets/mmap immutably, hnsw mutably).
///
/// `arena` is an optional fast path: a row-major `Vec<f32>` buffer (produced by
/// `backfill_hnsw_delta_parallel`) paired with an `id → row` index.  When set,
/// `get()` checks it first — O(1) with no heap allocation and perfect cache
/// locality for the bulk-build path.
pub(crate) struct EmbLookup<'a> {
    pub heap:    &'a HashMap<MemoryId, Vec<f32>>,
    pub offsets: &'a HashMap<MemoryId, u64>,
    pub mmap:    &'a Option<std::sync::Arc<memmap2::Mmap>>,
    /// Row-major flat arena: `buf[row * EMBED_DIM .. (row+1) * EMBED_DIM]`.
    /// `None` for normal (non-backfill) lookups.
    pub arena:   Option<(&'a [f32], &'a HashMap<MemoryId, usize>)>,
}

impl<'a> EmbLookup<'a> {
    #[inline]
    pub fn get(&self, id: MemoryId) -> Option<&'a [f32]> {
        if let Some((buf, id_to_row)) = self.arena {
            if let Some(&row) = id_to_row.get(&id) {
                let start = row * EMBED_DIM;
                return Some(&buf[start..start + EMBED_DIM]);
            }
        }
        if let Some(&off) = self.offsets.get(&id) {
            if let Some(ref mm) = self.mmap {
                let start = off as usize;
                let end = start + EMBED_DIM * 4;
                if end <= mm.len() {
                    let ptr = mm[start..end].as_ptr() as *const f32;
                    // SAFETY: offset was written by save_embeddings_sidecar with 4-byte alignment.
                    return Some(unsafe { std::slice::from_raw_parts(ptr, EMBED_DIM) });
                }
            }
        }
        self.heap.get(&id).map(|v| v.as_slice())
    }
}

// ── HNSW graph ────────────────────────────────────────────────────────────────

/// A layered navigable small-world graph for approximate nearest-neighbour
/// search. Purely in-RAM; rebuilt from `SemanticIndex::embeddings` after load.
///
/// All vectors stored in the parent `SemanticIndex` are unit-normalised, so
/// inner product == cosine similarity throughout.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct HnswGraph {
    /// Per-node adjacency lists, indexed by layer.
    /// `neighbors[id][layer]` = Vec of neighbour MemoryIds.
    neighbors: HashMap<MemoryId, Vec<Vec<MemoryId>>>,
    /// Entry point: (node_id, max_layer_of_that_node).
    entry_point: Option<(MemoryId, usize)>,
}

impl HnswGraph {
    fn new() -> Self {
        Self::default()
    }

    fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.neighbors.len()
    }


    pub(crate) fn contains(&self, id: MemoryId) -> bool {
        self.neighbors.contains_key(&id)
    }

    pub(crate) fn ids(&self) -> Vec<MemoryId> {
        self.neighbors.keys().copied().collect()
    }

    fn insert(&mut self, id: MemoryId, embedding: &[f32], embs: &EmbLookup<'_>) {
        let node_level = self.random_level();

        // Determine current max layer
        let current_top = self.entry_point.map(|(_, l)| l).unwrap_or(0);

        // Initialise adjacency list for new node
        self.neighbors.insert(id, vec![Vec::new(); node_level + 1]);

        let ep = match self.entry_point {
            None => {
                // First node
                self.entry_point = Some((id, node_level));
                return;
            }
            Some((ep, _)) => ep,
        };

        // Phase 1: descend from top of graph down to node_level+1
        let mut ep_set = vec![ep];
        for layer in (node_level + 1..=current_top).rev() {
            ep_set = self.search_layer(embedding, &ep_set, 1, layer, embs);
        }

        // Phase 2: insert at each layer from min(node_level, current_top) down to 0
        for layer in (0..=node_level.min(current_top)).rev() {
            let candidates = self.search_layer(embedding, &ep_set, HNSW_EF_CONSTRUCTION, layer, embs);
            let m = if layer == 0 { HNSW_M0 } else { HNSW_M };
            let selected = self.select_neighbors(embedding, &candidates, m, embs);

            // Connect new node to its selected neighbours
            if let Some(node_neighbors) = self.neighbors.get_mut(&id) {
                if layer < node_neighbors.len() {
                    node_neighbors[layer] = selected.clone();
                }
            }

            // Connect each selected neighbour back to new node (shrink if needed)
            for &neighbor_id in &selected {
                let m_max = if layer == 0 { HNSW_M0 } else { HNSW_M };
                // Push new node into neighbour's list, then check if shrink needed
                let needs_shrink = {
                    if let Some(nbr_layers) = self.neighbors.get_mut(&neighbor_id) {
                        if layer < nbr_layers.len() {
                            nbr_layers[layer].push(id);
                            nbr_layers[layer].len() > m_max
                        } else { false }
                    } else { false }
                };
                if needs_shrink {
                    // Compute shrunk list without holding mutable borrow
                    let shrunk = if let Some(nbr_layers) = self.neighbors.get(&neighbor_id) {
                        if layer < nbr_layers.len() {
                            if let Some(nbr_emb) = embs.get(neighbor_id) {
                                Some(self.select_neighbors(nbr_emb, &nbr_layers[layer].clone(), m_max, embs))
                            } else { None }
                        } else { None }
                    } else { None };
                    if let (Some(shrunk), Some(nbr_layers)) = (shrunk, self.neighbors.get_mut(&neighbor_id)) {
                        if layer < nbr_layers.len() {
                            nbr_layers[layer] = shrunk;
                        }
                    }
                }
            }

            ep_set = candidates;
        }

        // Update entry point if new node has higher layer
        if node_level > current_top {
            self.entry_point = Some((id, node_level));
        }
    }

    /// Remove a node by clearing its adjacency lists and removing it from
    /// neighbours' lists. This is O(M × L) and acceptable for soft-delete
    /// rates typical in a memory store.
    fn remove(&mut self, id: MemoryId) {
        let Some(node_neighbors) = self.neighbors.remove(&id) else { return };
        // Unlink from neighbours' adjacency lists
        for layer_neighbors in &node_neighbors {
            for &neighbor_id in layer_neighbors {
                if let Some(nbr_layers) = self.neighbors.get_mut(&neighbor_id) {
                    for layer_list in nbr_layers.iter_mut() {
                        layer_list.retain(|&n| n != id);
                    }
                }
            }
        }
        // If this was the entry point, pick any remaining node at the highest layer
        if self.entry_point.map(|(ep, _)| ep) == Some(id) {
            self.entry_point = self
                .neighbors
                .iter()
                .map(|(&nid, layers)| (nid, layers.len().saturating_sub(1)))
                .max_by_key(|&(_, l)| l);
        }
    }

    /// Greedy k-NN search. Returns up to k MemoryIds, filtered by `allowed`.
        fn search(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
        deleted: &HashSet<MemoryId>,
        embs: &EmbLookup<'_>,
    ) -> Vec<(MemoryId, f32)> {
        let Some((ep, top_layer)) = self.entry_point else { return vec![] };

        let mut ep_set = vec![ep];

        // Descend from top layer to layer 1 with ef=1
        for layer in (1..=top_layer).rev() {
            ep_set = self.search_layer(query, &ep_set, 1, layer, embs);
        }

        // Final search at layer 0 with ef_search
        let candidates = self.search_layer(query, &ep_set, HNSW_EF_SEARCH.max(k), 0, embs);

        // Filter deleted / not-allowed, compute similarities, take top k
        let mut scored: Vec<(MemoryId, f32)> = candidates
            .into_iter()
            .filter(|id| !deleted.contains(id))
            .filter(|id| allowed.map(|a| a.contains(id)).unwrap_or(true))
            .filter_map(|id| embs.get(id).map(|e| (id, dot(query, e))))
            .collect();

        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// Greedy beam search within one layer. Returns ef best candidates.
        fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[MemoryId],
        ef: usize,
        layer: usize,
        embs: &EmbLookup<'_>,
    ) -> Vec<MemoryId> {
        let mut visited: HashSet<MemoryId> = entry_points.iter().copied().collect();
        // candidates: max-heap by sim (best candidate at top for greedy expansion)
        let mut candidates: BinaryHeap<(OrderedF32, MemoryId)> = entry_points
            .iter()
            .filter_map(|&id| embs.get(id).map(|e| (OrderedF32(dot(query, e)), id)))
            .collect();
        // result: min-heap by neg-sim (worst at top for O(1) eviction when |result| > ef)
        let mut result: BinaryHeap<(OrderedF32, MemoryId)> = entry_points
            .iter()
            .filter_map(|&id| embs.get(id).map(|e| (OrderedF32(-dot(query, e)), id)))
            .collect();

        while let Some((pos_score, cid)) = candidates.pop() {
            let c_score = pos_score.0;
            // Early stop: best remaining candidate worse than worst kept result → done
            if result.len() >= ef {
                if let Some(&(OrderedF32(neg_worst_score), _)) = result.peek() {
                    if c_score < -neg_worst_score {
                        break;
                    }
                }
            }
            // Expand neighbours at this layer
            let Some(node_layers) = self.neighbors.get(&cid) else { continue };
            let Some(layer_neighbors) = node_layers.get(layer) else { continue };
            for &neighbor_id in layer_neighbors {
                if visited.insert(neighbor_id) {
                    let Some(e) = embs.get(neighbor_id) else { continue };
                    let sim = dot(query, e);
                    candidates.push((OrderedF32(sim), neighbor_id));
                    result.push((OrderedF32(-sim), neighbor_id));
                    // Trim result: pop worst (max of neg-sim)
                    while result.len() > ef {
                        result.pop();
                    }
                }
            }
        }

        result.into_iter().map(|(_, id)| id).collect()
    }

    /// Simple greedy neighbour selection (no heuristic pruning for now).
        fn select_neighbors(
        &self,
        query: &[f32],
        candidates: &[MemoryId],
        m: usize,
        embs: &EmbLookup<'_>,
    ) -> Vec<MemoryId> {
        let mut scored: Vec<(f32, MemoryId)> = candidates
            .iter()
            .filter_map(|&id| embs.get(id).map(|e| (dot(query, e), id)))
            .collect();
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(m);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    /// Draw a random layer for a new node using the standard HNSW distribution.
    /// Uses a simple LCG so there's no rand dependency.
    fn random_level(&self) -> usize {
        // Seed from number of nodes for deterministic but varied levels
        let n = self.neighbors.len() as u64;
        let mut s = n
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s ^= s >> 33;
        s = s.wrapping_mul(0xff51afd7ed558ccd);
        s ^= s >> 33;
        let f = (s >> 11) as f64 / (1u64 << 53) as f64; // uniform [0,1)
        if f < 1e-15 {
            return 0;
        }
        let level = (-f.ln() * HNSW_ML).floor() as usize;
        level.min(16) // cap at 16 layers
    }

    /// Read-only plan computation: find neighbor candidates for a new node without
    /// mutating the graph. Used by backfill_hnsw_delta_parallel to parallelize the
    /// search phase against a frozen snapshot.
    pub(crate) fn compute_insert_plan(
        &self,
        emb: &[f32],
        level: usize,
        embs: &EmbLookup<'_>,
        ef_construction: usize,
    ) -> Vec<Vec<MemoryId>> {
        let current_top = self.entry_point.map(|(_, l)| l).unwrap_or(0);
        let ep = match self.entry_point {
            None => return vec![vec![]; level + 1],
            Some((ep, _)) => ep,
        };
        let mut ep_set = vec![ep];
        for layer in (level + 1..=current_top).rev() {
            ep_set = self.search_layer(emb, &ep_set, 1, layer, embs);
        }
        let mut neighbors_per_layer = vec![vec![]; level + 1];
        for layer in (0..=level.min(current_top)).rev() {
            let candidates = self.search_layer(emb, &ep_set, ef_construction, layer, embs);
            let m = if layer == 0 { HNSW_M0 } else { HNSW_M };
            neighbors_per_layer[layer] = self.select_neighbors(emb, &candidates, m, embs);
            ep_set = candidates;
        }
        neighbors_per_layer
    }

    /// Fast apply of a pre-computed insert plan WITHOUT distance-based neighbor shrink.
    /// Wires the new node's forward edges (already ≤ M, selected in compute_insert_plan)
    /// and appends back-edges (neighbor → new), capped at 3×M so lists stay bounded.
    /// Deferring the O(M·dim) shrink keeps the write-lock hold at O(plan·M) pointer ops
    /// (microseconds, not the ~1ms/node the shrink costs) — this is what lets the daemon
    /// hold the exclusive rpc_mutex only briefly during apply. The exact shrink is
    /// reconciled at the next delta→base merge, which rebuilds via insert() (proper
    /// heuristic selection). Bounded growth keeps HNSW search valid meanwhile.
    pub(crate) fn wire_precomputed(&mut self, id: MemoryId, level: usize, per_layer: &[Vec<MemoryId>]) {
        let current_top = self.entry_point.map(|(_, l)| l).unwrap_or(0);
        self.neighbors.insert(id, vec![Vec::new(); level + 1]);
        if self.entry_point.is_none() {
            self.entry_point = Some((id, level));
            return;
        }
        for (layer, selected) in per_layer.iter().enumerate() {
            if layer >= level + 1 { break; }
            if let Some(nn) = self.neighbors.get_mut(&id) {
                if layer < nn.len() { nn[layer] = selected.clone(); }
            }
            let cap = if layer == 0 { HNSW_M0 } else { HNSW_M } * 3;
            for &nbr in selected {
                if let Some(nl) = self.neighbors.get_mut(&nbr) {
                    if layer < nl.len() && nl[layer].len() < cap {
                        nl[layer].push(id);
                    }
                }
            }
        }
        if level > current_top {
            self.entry_point = Some((id, level));
        }
    }
}

/// Newtype for f32 that implements Ord via total_cmp. Used in BinaryHeap.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.total_cmp(&other.0) }
}

/// A recall hit from semantic search.
#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub memory_id: MemoryId,
    pub cosine_similarity: f32,
}

/// Pre-computed HNSW insert plan for a deferred batch (Step-1 deferred-batched-insert).
/// Produced off-lock by `SemanticIndex::plan_delta_batch` (`&self`, under a read lock)
/// and consumed by `apply_delta_batch` (`&mut self`, under a brief write lock). `Send`
/// so it can be staged on `ChittaField` and applied on another thread.
pub struct DeltaBatchPlan {
    /// True when the global plan targets the delta tier; false when it targets base.
    tier2: bool,
    /// Global-graph plans: `(id, level, neighbors_per_layer)`.
    plans: Vec<(MemoryId, usize, Vec<Vec<MemoryId>>)>,
    /// Per-realm-graph plans: `(id, realm, level, neighbors_per_layer)`.
    realm_plans: Vec<(MemoryId, String, usize, Vec<Vec<MemoryId>>)>,
}

impl DeltaBatchPlan {
    /// Number of node-insertions this plan will wire in (global + per-realm).
    pub fn len(&self) -> usize { self.plans.len() + self.realm_plans.len() }
    pub fn is_empty(&self) -> bool { self.plans.is_empty() && self.realm_plans.is_empty() }
}

/// 4-bit TurboQuant index, warmed before open returns and refreshed by maintenance.
/// Runtime-only: published through an Arc after construction completes.
struct TurboState {
    index: crate::turbo::SearchIndex,
    /// turbovec row -> MemoryId (insertion order = add order).
    ids: Vec<MemoryId>,
    /// `mutations` value the index was built at; stale when it lags.
    built_at_mutation: u64,
    built_at: std::time::Instant,
    /// A snapshot cache with replay changes must be rebuilt on the first maintenance tick.
    startup_delta: bool,
}

/// Immutable refresh view: publishing a newer index cannot invalidate this Arc.
/// Searches run off every store guard, in a bounded pool, with the same single-
/// query arithmetic and filtering as SemanticIndex::search's raw Turbo arm.
pub(crate) struct RefreshSearch {
    turbo: Arc<TurboState>,
    deleted: HashSet<MemoryId>,
}
fn refresh_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new()
        .num_threads(std::thread::available_parallelism().map_or(1, |n| n.get().min(16)))
        .thread_name(|i| format!("cf-refresh-{i}"))
        .build().expect("competitive refresh pool"))
}

impl RefreshSearch {
    pub(crate) fn search_all(&self, candidates: &[(MemoryId, Vec<f32>)]) -> Vec<Vec<SemanticHit>> {
        use rayon::prelude::*;
        refresh_pool().install(|| candidates.par_iter().map(|(_, emb)| {
            let Some(q) = normalize(emb) else { return Vec::new(); };
            let results = self.turbo.index.search(&q, 9.min(self.turbo.ids.len().max(1)));
            let mut hits = Vec::with_capacity(9);
            for (&row, &score) in results.indices_for_query(0).iter().zip(results.scores_for_query(0)) {
                if row < 0 || !score.is_finite() { continue; }
                let Some(&id) = self.turbo.ids.get(row as usize) else { continue; };
                if self.deleted.contains(&id) { continue; }
                hits.push(SemanticHit { memory_id: id, cosine_similarity: score });
                if hits.len() == 9 { break; }
            }
            hits.sort_unstable_by(|a, b| b.cosine_similarity.total_cmp(&a.cosine_similarity).then_with(|| a.memory_id.cmp(&b.memory_id)));
            hits
        }).collect())
    }
}

/// Approximate semantic index over normalized memory embeddings.
///
/// Uses a coarse centroid assignment to generate candidate buckets, then exact
/// cosine reranking inside those candidates. This keeps query cost bounded even
/// when the sparse cortical index cannot provide candidates.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticIndex {
    /// memory_id -> embedding (flat Vec for cache efficiency)
    embeddings: HashMap<MemoryId, Vec<f32>>,
    /// soft-deleted IDs excluded from search results
    deleted: HashSet<MemoryId>,
    #[serde(default = "default_coarse_centroids")]
    coarse_centroids: Vec<Vec<f32>>,
    #[serde(default)]
    coarse_members: HashMap<u16, Vec<MemoryId>>,
    #[serde(default)]
    mem_coarse: HashMap<MemoryId, Vec<u16>>,
    /// Corpus mean of raw unit embeddings — anisotropy correction (mean-centering).
    /// NOT bincode-serialized: bincode is positional, so adding a field would corrupt
    /// existing snapshots. Persisted via the `.mu` sidecar instead. Empty = centering
    /// disabled (legacy). When set, queries, rerank candidates and binary codes are all
    /// centered against it so cosine becomes discriminative on anisotropic embeddings.
    #[serde(skip)]
    centroid: Vec<f32>,
    /// Monotone mutation counter (runtime-only). Bumped by every content
    /// mutation; save_full_snapshot skips rewriting the index sidecars when
    /// unchanged since the last save (dirty-skip; THEORY.md §8 Phase 2).
    /// Over-bumping is safe (loses the skip, never correctness).
    #[serde(skip)]
    mutations: u64,
    /// Writes/removes since the frozen centroid was last validated against the
    /// live corpus mean (runtime-only; a restart just delays one drift check —
    /// the centroid itself persists via the `.mu` sidecar).
    #[serde(skip)]
    centroid_stale_writes: u64,
    #[serde(skip)]
    lsh_planes: Vec<Vec<Vec<f32>>>,
    #[serde(skip)]
    lsh_buckets: Vec<HashMap<u16, Vec<MemoryId>>>,
    #[serde(skip)]
    mem_lsh: HashMap<MemoryId, Vec<u16>>,
    /// HNSW graph — saved to sidecar .hnsw file; rebuilt when stale.
    /// Active when `embeddings.len() >= HNSW_THRESHOLD`.
    #[serde(skip)]
    hnsw: HnswGraph,
    /// Delta tier: receives new inserts when embeddings.len() >= HNSW_TIER2_THRESHOLD.
    /// Queries merge results from both tiers. Saved to .delta.hnsw sidecar.
    /// Merged into base when delta_needs_merge() returns true (at checkpoint).
    #[serde(skip)]
    delta_hnsw: HnswGraph,
    /// Sign-bit binary codes for Hamming pre-filter (derived from embeddings; not persisted).
    #[serde(skip)]
    binary_codes: HashMap<MemoryId, Vec<u64>>,
    /// Contiguous flat scan buffer — cache-friendly alternative to HashMap iteration in hamming_candidates.
    /// Kept in sync with binary_codes by upsert/remove/load_binary_sidecar/normalize_all.
    #[serde(skip)]
    binary_vec: Vec<(MemoryId, [u64; BINARY_WORDS])>,
    /// Position index for O(1) swap-remove from binary_vec.
    #[serde(skip)]
    binary_vec_pos: HashMap<MemoryId, usize>,
    /// When true, upsert() skips HNSW inserts (set during WAL replay to avoid O(N log N) cost
    /// when binary Hamming will be the active search path after normalize_all()).
    #[serde(skip)]
    inhibit_hnsw: bool,
    /// Test-only: force `use_hnsw()` true regardless of size so graph-construction
    /// parity can be probed below `flat_scan_max()` (where production never builds it).
    #[serde(skip)]
    force_hnsw_build: bool,
    /// Per-realm HNSW graphs — built for realms exceeding PER_REALM_HNSW_THRESHOLD.
    /// Saved to .realm_hnsw sidecar. Seeded by seed_realm_map() at load time.
    #[serde(skip)]
    per_realm_hnsw: HashMap<String, HnswGraph>,
    /// Per-realm embedding counts — tracked at upsert time to decide routing.
    #[serde(skip)]
    per_realm_counts: HashMap<String, usize>,
    /// Memory-to-realm mapping — populated by upsert, consumed by remove.
    #[serde(skip)]
    per_id_realm: HashMap<MemoryId, String>,
    /// Mmap of the .emb sidecar file — populated by activate_mmap_embeddings() above 200K.
    /// Wrapped in Arc so SemanticIndex remains Clone (Arc<T>: Clone even when T: !Clone).
    #[serde(skip)]
    emb_mmap: Option<std::sync::Arc<memmap2::Mmap>>,
    /// Byte offset of the f32×EMBED_DIM data for each memory in emb_mmap.
    #[serde(skip)]
    emb_offsets: HashMap<MemoryId, u64>,
    /// Readers clone the current index under a brief read guard; builds run off-lock.
    #[serde(skip, default)]
    turbo: Arc<RwLock<Option<Arc<TurboState>>>>,
    /// Embeddings changed after the published build are scored directly.
    #[serde(skip, default)]
    turbo_changed: HashMap<MemoryId, u64>,
    #[serde(skip, default)]
    turbo_epoch: Arc<std::sync::atomic::AtomicU64>,
    /// Exact raw snapshot sidecar identity, captured BEFORE WAL replay.
    #[serde(skip)]
    startup_key: Option<[u8; 32]>,

}

impl SemanticIndex {
    pub(crate) fn allocated_bytes(&self) -> (usize, usize, usize) {
        use crate::profile::map_bytes;
        let embeddings = map_bytes(&self.embeddings)
            + self.embeddings.values().map(|v| v.capacity() * 4).sum::<usize>()
            + self.emb_mmap.as_ref().map_or(0, |m| m.len()) + map_bytes(&self.emb_offsets);
        let graph = |g: &HnswGraph| map_bytes(&g.neighbors) + g.neighbors.values().map(|layers|
            layers.capacity() * std::mem::size_of::<Vec<MemoryId>>()
                + layers.iter().map(|v| v.capacity() * 8).sum::<usize>()).sum::<usize>();
        let hnsw = graph(&self.hnsw) + graph(&self.delta_hnsw)
            + self.per_realm_hnsw.values().map(graph).sum::<usize>();
        let caches = map_bytes(&self.binary_codes)
            + self.binary_codes.values().map(|v| v.capacity() * 8).sum::<usize>()
            + self.binary_vec.capacity() * std::mem::size_of::<(MemoryId, [u64; BINARY_WORDS])>()
            + map_bytes(&self.binary_vec_pos) + map_bytes(&self.turbo_changed);
        (embeddings, hnsw, caches)
    }

    pub fn new() -> Self {
        Self {
            embeddings: HashMap::new(),
            deleted: HashSet::new(),
            coarse_centroids: default_coarse_centroids(),
            coarse_members: HashMap::new(),
            mem_coarse: HashMap::new(),
            centroid: Vec::new(),
            mutations: 0,
            centroid_stale_writes: 0,
            lsh_planes: default_lsh_planes(),
            lsh_buckets: vec![HashMap::new(); LSH_TABLES],
            mem_lsh: HashMap::new(),
            hnsw: HnswGraph::new(),
            delta_hnsw: HnswGraph::new(),
            binary_codes: HashMap::new(),
            binary_vec: Vec::new(),
            binary_vec_pos: HashMap::new(),
            inhibit_hnsw: false,
            force_hnsw_build: false,
            per_realm_hnsw: HashMap::new(),
            per_realm_counts: HashMap::new(),
            per_id_realm: HashMap::new(),
            emb_mmap: None,
            emb_offsets: HashMap::new(),
            turbo: Arc::new(RwLock::new(None)),
            turbo_changed: HashMap::new(),
            turbo_epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            startup_key: None,
        }
    }

    /// Total number of embeddings — heap + mmap-backed combined.
    #[inline]
    pub fn total_embedding_count(&self) -> usize {
        self.embeddings.len() + self.emb_offsets.len()
    }

    pub fn hnsw_len(&self) -> usize {
        self.hnsw.len() + self.delta_hnsw.len()
    }

    /// Node count of the delta tier only. Used to gate the runtime delta-sidecar
    /// persist (skip the write — and its NFS delete-on-empty branch — when empty).
    pub fn delta_len(&self) -> usize {
        self.delta_hnsw.len()
    }

    /// True when the HNSW graph is the active search path.
    /// Gated on `flat_scan_max()`, not `HNSW_THRESHOLD`: below the flat-scan cap the
    /// exact scan serves every query, so building HNSW there was pure overhead (built,
    /// merged, snapshotted, never consulted). With the ANN path forced (cap 0) we still
    /// build above the small floor so ANN always has a graph to serve.
    #[inline]
    fn use_hnsw(&self) -> bool {
        if self.force_hnsw_build { return self.total_embedding_count() >= HNSW_THRESHOLD; }
        let fmax = flat_scan_max();
        if fmax == 0 { return self.total_embedding_count() >= HNSW_THRESHOLD; }
        self.total_embedding_count() > fmax
    }

    /// True when two-tier mode is active: new inserts go to delta_hnsw, not hnsw.
    #[inline]
    fn use_tier2(&self) -> bool {
        self.total_embedding_count() >= HNSW_TIER2_THRESHOLD
    }

    /// Insert embeddings that are in the store but not yet in HNSW (WAL replay delta).
    /// O(K log N) where K = delta count. Called after WAL replay when binary_covers=true
    /// prevents a full O(N log N) rebuild but HNSW is partially stale.
    pub fn backfill_hnsw_delta(&mut self) {
        self.mutations += 1;
        if !self.use_hnsw() { return; }
        if self.hnsw.is_empty() && self.delta_hnsw.is_empty() { return; }
        let delta: Vec<MemoryId> = self.embeddings.keys()
            .chain(self.emb_offsets.keys())
            .filter(|&&id| {
                !self.deleted.contains(&id)
                    && !self.hnsw.contains(id)
                    && !self.delta_hnsw.contains(id)
            })
            .copied()
            .collect();
        if delta.is_empty() { return; }
        eprintln!("[hnsw] backfill_hnsw_delta: inserting {} delta nodes", delta.len());
        for id in delta {
            let emb_owned = {
                let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                embs.get(id).map(|s| s.to_vec())
            };
            if let Some(emb) = emb_owned {
                let tier2 = self.use_tier2();
                let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                if tier2 {
                    self.delta_hnsw.insert(id, &emb, &embs);
                } else {
                    self.hnsw.insert(id, &emb, &embs);
                }
            }
        }
    }

    /// Parallel variant of backfill_hnsw_delta.
    /// Phase 1 (parallel): clone the current graph and all embeddings into a read-only
    /// snapshot, then compute neighbor candidates for every delta node concurrently.
    /// Phase 2 (serial): apply the pre-computed plans into the live graph.
    /// Falls back to serial for small deltas (< 512 nodes) where overhead dominates.
    pub fn backfill_hnsw_delta_parallel(&mut self) {
        self.mutations += 1;
        if !self.use_hnsw() { return; }
        if self.hnsw.is_empty() && self.delta_hnsw.is_empty() { return; }

        let delta: Vec<MemoryId> = self.embeddings.keys()
            .chain(self.emb_offsets.keys())
            .filter(|&&id| {
                !self.deleted.contains(&id)
                    && !self.hnsw.contains(id)
                    && !self.delta_hnsw.contains(id)
            })
            .copied()
            .collect();

        if delta.is_empty() { return; }
        if delta.len() < 512 {
            self.backfill_hnsw_delta();
            return;
        }

        eprintln!("[hnsw] backfill_hnsw_delta_parallel: {} delta nodes", delta.len());
        let t0 = std::time::Instant::now();

        let tier2 = self.use_tier2();
        // Max HNSW layers (random_level_from_seed caps at 16, so layer 0..=16).
        const MAXL: usize = 17;
        const EPOCH_SIZE: usize = 3000;
        const EF_BULK: usize = 64;
        // Limit build parallelism to avoid saturating shared-node CPU and to
        // reduce lock contention in the parallel apply phase.
        const BUILD_THREADS: usize = 24;
        let build_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(BUILD_THREADS)
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

        // ── 1. Flat embedding store (single allocation, row-major) ────────────
        // Replaces snap_embs HashMap: eliminates per-entry pointer-chase and
        // eliminates 144k separate heap allocations scattered across 443 MB.
        let existing_graph = if tier2 { &self.delta_hnsw } else { &self.hnsw };

        // Dense row remap: all existing nodes first, then delta.
        let mut id_to_row_v: HashMap<MemoryId, u32> =
            HashMap::with_capacity(existing_graph.len() + delta.len());
        let mut row_to_id_v: Vec<MemoryId> =
            Vec::with_capacity(existing_graph.len() + delta.len());
        for id in existing_graph.ids().into_iter().chain(delta.iter().copied()) {
            if !id_to_row_v.contains_key(&id) {
                let row = row_to_id_v.len() as u32;
                id_to_row_v.insert(id, row);
                row_to_id_v.push(id);
            }
        }
        let n_rows = row_to_id_v.len();

        // Contiguous float buffer: arena[row * EMBED_DIM .. (row+1) * EMBED_DIM].
        let mut arena: Vec<f32> = vec![0.0f32; n_rows * EMBED_DIM];
        {
            let lk = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            for (row, &id) in row_to_id_v.iter().enumerate() {
                if let Some(e) = lk.get(id) {
                    arena[row * EMBED_DIM..(row + 1) * EMBED_DIM].copy_from_slice(e);
                }
            }
        }
        let id_to_row = std::sync::Arc::new(id_to_row_v);
        let row_to_id = std::sync::Arc::new(row_to_id_v);

        // ── 1b. Quantize to i8 (SQ8) — 4× smaller working set for beam search.
        // Embeddings are pre-normalized (unit sphere, values ∈ [-1,1]) so scale=127 is lossless
        // to ~0.2% cosine error, well within HNSW approximation budget. Ranking is preserved.
        // This drops the arena from 448 MB (f32) → 112 MB (i8), fitting 10M in ~7.7 GB.
        let arena_q: std::sync::Arc<Vec<i8>> = std::sync::Arc::new(
            arena.iter().map(|&x| (x * 127.0).round().clamp(-127.0, 127.0) as i8).collect()
        );
        drop(arena); // free 448 MB f32 — all build paths use arena_q

        // ── 2. Per-(node, layer) locked adjacency ─────────────────────────────
        // Flat: locked_adj[row * MAXL + layer].
        // SmallVec<[u32; 32]>: M0=32 inline — zero heap allocation for layer 0.
        let locked_adj: std::sync::Arc<Vec<parking_lot::Mutex<SmallVec<[u32; 32]>>>> =
            std::sync::Arc::new(
                (0..n_rows * MAXL).map(|_| parking_lot::Mutex::new(SmallVec::new())).collect()
            );

        // Seed from existing graph (MemoryId neighbors → u32 rows).
        for (id, nbr_layers) in &existing_graph.neighbors {
            if let Some(&row) = id_to_row.get(id) {
                for (layer, layer_nbrs) in nbr_layers.iter().enumerate() {
                    if layer >= MAXL { continue; }
                    let mut cell = locked_adj[row as usize * MAXL + layer].lock();
                    for &nbr_id in layer_nbrs {
                        if let Some(&nbr_row) = id_to_row.get(&nbr_id) {
                            cell.push(nbr_row);
                        }
                    }
                }
            }
        }

        // ── 3. Atomic entry-point tracking ───────────────────────────────────
        let ep_init = existing_graph.entry_point;
        let entry_row = std::sync::Arc::new(AtomicU32::new(
            ep_init.and_then(|(id, _)| id_to_row.get(&id).copied()).unwrap_or(u32::MAX)
        ));
        let entry_lev = std::sync::Arc::new(AtomicU32::new(
            ep_init.map_or(0, |(_, l)| l as u32)
        ));

        // ── 4. Epoch loop ─────────────────────────────────────────────────────
        // Phase A: parallel plan-compute against a frozen row-native CSR snapshot —
        //          no HashMap traversal, no per-search HashSet allocation.
        // Phase B: parallel apply into locked_adj (row-native plans, no id→row lookups).
        // Between epochs: rebuild CSR snapshot from locked_adj in O(N·M) — ~25 ms.
        let n_epochs = (delta.len() + EPOCH_SIZE - 1) / EPOCH_SIZE;

        // Initial snapshot seeded from existing graph edges already written to locked_adj.
        let mut build_snap = BuildSnapshot::from_locked_adj(
            &locked_adj, n_rows,
            entry_row.load(Ordering::Relaxed),
            entry_lev.load(Ordering::Relaxed) as u8,
            MAXL,
        );

        for (epoch_idx, chunk) in delta.chunks(EPOCH_SIZE).enumerate() {
            // Phase A: parallel plan compute against frozen CSR snapshot (i8 arena).
            let aq_a  = arena_q.clone();
            let i2r_a = id_to_row.clone();
            let plans: Vec<(u32, usize, Vec<SmallVec<[u32; 32]>>)> = build_pool.install(|| {
                chunk.par_iter()
                    .map_init(|| Scratch::new(n_rows), |scratch, &id| {
                        let row = *i2r_a.get(&id)? as u32;
                        let base = row as usize * EMBED_DIM;
                        let emb = &aq_a[base..base + EMBED_DIM];
                        let level = random_level_from_seed(id);
                        let plan = compute_insert_plan_rows(
                            &build_snap, emb, level, &aq_a, scratch, MAXL, EF_BULK,
                        );
                        Some((row, level, plan))
                    })
                    .filter_map(|x| x)
                    .collect()
            });

            // Phase B: parallel apply into locked_adj (i8 arena for shrink comparisons).
            {
                let la = locked_adj.clone();
                let aq = arena_q.clone();
                let er = entry_row.clone();
                let el = entry_lev.clone();

                build_pool.install(|| plans.par_iter().for_each(|(new_row, level, per_layer)| {
                    // Forward edges: write own adjacency layers.
                    for (layer, selected) in per_layer.iter().enumerate() {
                        if layer >= MAXL { break; }
                        *la[*new_row as usize * MAXL + layer].lock() = selected.clone();
                    }

                    // Back-edges + shrink (i8 dot for distance comparison).
                    for (layer, selected) in per_layer.iter().enumerate() {
                        if layer >= MAXL { break; }
                        let m_max = if layer == 0 { HNSW_M0 } else { HNSW_M };
                        for &nbr_row in selected.iter() {
                            let nbr_base = nbr_row as usize * EMBED_DIM;
                            let nbr_emb = &aq[nbr_base..nbr_base + EMBED_DIM];
                            let cell_idx = nbr_row as usize * MAXL + layer;
                            let mut cell = la[cell_idx].lock();
                            cell.push(*new_row);
                            if cell.len() > m_max {
                                let mut scored: Vec<(i32, u32)> = cell.iter()
                                    .filter_map(|&r| {
                                        let base = r as usize * EMBED_DIM;
                                        let e = aq.get(base..base + EMBED_DIM)?;
                                        Some((dot_i8(nbr_emb, e), r))
                                    })
                                    .collect();
                                scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                                scored.truncate(m_max);
                                *cell = scored.into_iter().map(|(_, r)| r).collect();
                            }
                        }
                    }

                    // CAS-update entry point if this node has a strictly higher level.
                    let mut cur = el.load(Ordering::Relaxed);
                    loop {
                        if (*level as u32) <= cur { break; }
                        match el.compare_exchange_weak(cur, *level as u32, Ordering::Relaxed, Ordering::Relaxed) {
                            Ok(_) => { er.store(*new_row, Ordering::Relaxed); break; }
                            Err(v) => cur = v,
                        }
                    }
                }));
            }

            // Rebuild CSR snapshot for next epoch — O(N·M), replaces HnswGraph clone + HashMap drain.
            build_snap = BuildSnapshot::from_locked_adj(
                &locked_adj, n_rows,
                entry_row.load(Ordering::Relaxed),
                entry_lev.load(Ordering::Relaxed) as u8,
                MAXL,
            );

            eprintln!(
                "[hnsw] backfill epoch {}/{} done in {:.1}s ({} nodes)",
                epoch_idx + 1, n_epochs, t0.elapsed().as_secs_f32(), chunk.len()
            );
        }

        // ── 5. Final drain: write all locked_adj rows to target (delta + existing).
        // No mid-epoch drain means all rows are written here for the first time.
        {
            let target = if tier2 { &mut self.delta_hnsw } else { &mut self.hnsw };
            for (row, &id) in row_to_id.iter().enumerate() {
                let mut layers: Vec<Vec<MemoryId>> = Vec::new();
                for layer in 0..MAXL {
                    let cell = locked_adj[row * MAXL + layer].lock();
                    if cell.is_empty() { break; }
                    layers.push(cell.iter().filter_map(|&r| row_to_id.get(r as usize).copied()).collect());
                }
                if !layers.is_empty() { target.neighbors.insert(id, layers); }
            }
            let ep_r = entry_row.load(Ordering::Relaxed);
            let ep_l = entry_lev.load(Ordering::Relaxed) as usize;
            if ep_r != u32::MAX {
                let ep_id = row_to_id[ep_r as usize];
                match target.entry_point {
                    None => target.entry_point = Some((ep_id, ep_l)),
                    Some((_, cur_l)) if ep_l > cur_l => target.entry_point = Some((ep_id, ep_l)),
                    _ => {}
                }
            }
        }

        eprintln!(
            "[hnsw] backfill_hnsw_delta_parallel: done in {:.1}s total",
            t0.elapsed().as_secs_f32()
        );
    }

    /// Returns true when the delta tier should be merged into the base HNSW.
    /// Called at checkpoint time to keep delta small and inserts fast.
    pub fn delta_needs_merge(&self) -> bool {
        self.use_tier2()
            && !self.hnsw.is_empty()
            && self.delta_hnsw.len() as f64 > self.hnsw.len() as f64 * HNSW_DELTA_MERGE_RATIO
    }

    /// Merge delta tier into base HNSW. O(delta × log N_base).
    /// Clears delta after merge. Called when delta_needs_merge() is true.
    pub fn merge_delta_into_base(&mut self) {
        self.mutations += 1;
        if self.delta_hnsw.is_empty() { return; }
        let ids = self.delta_hnsw.ids();
        eprintln!("[hnsw] merge_delta_into_base: merging {} nodes into base", ids.len());
        for id in ids {
            if self.deleted.contains(&id) { continue; }
            let emb_owned = {
                let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                embs.get(id).map(|s| s.to_vec())
            };
            if let Some(emb) = emb_owned {
                let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                self.hnsw.insert(id, &emb, &embs);
            }
        }
        self.delta_hnsw = HnswGraph::new();
    }

    /// Inhibit HNSW inserts during WAL replay when binary Hamming will be the active path.
    /// Call with `true` before replay and `false` after `normalize_all()` completes.
    /// Monotone mutation counter for dirty-skip at save (see field doc).
    pub fn mutation_count(&self) -> u64 {
        self.mutations
    }

    pub fn set_inhibit_hnsw(&mut self, v: bool) {
        self.inhibit_hnsw = v;
    }

    #[cfg(test)]
    pub fn force_hnsw_build_for_test(&mut self) {
        self.force_hnsw_build = true;
    }

    /// If the base HNSW is empty but the delta is not, promote delta→base so
    /// save_hnsw() serialises the full graph.  Called on the snapshot clone
    /// inside save_full_snapshot() — does NOT touch the live index.
    pub fn promote_delta_to_base_if_empty(&mut self) {
        if self.hnsw.is_empty() && !self.delta_hnsw.is_empty() {
            std::mem::swap(&mut self.hnsw, &mut self.delta_hnsw);
        }
    }

    /// Check whether an embedding exists without reading mmap data (no NFS I/O).
    /// Use instead of get_embedding().is_some() when only presence is needed —
    /// get_embedding() reads EMBED_DIM*4 bytes from an NFS-backed mmap, causing
    /// one page fault per call (146k × ~1ms = 146s deadlock during snapshot clone).
    pub fn has_embedding(&self, id: MemoryId) -> bool {
        self.emb_offsets.contains_key(&id) || self.embeddings.contains_key(&id)
    }

    /// Test-only: force the two-tier HNSW search path (base + delta merge), bypassing
    /// the flat-scan / binary-Hamming routing in `search`. Lets parity tests compare the
    /// GRAPH built by the per-item vs deferred-batched insert paths directly.
    #[cfg(test)]
    pub fn hnsw_search_test(&self, query: &[f32], k: usize) -> Vec<SemanticHit> {
        let Some(query_unit) = self.center_norm(query) else { return vec![] };
        let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
        let mut pairs = self.hnsw.search(&query_unit, k, None, &self.deleted, &embs);
        pairs.extend(self.delta_hnsw.search(&query_unit, k, None, &self.deleted, &embs));
        pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pairs.dedup_by_key(|p| p.0);
        pairs.truncate(k);
        pairs.into_iter().map(|(memory_id, cosine_similarity)| SemanticHit { memory_id, cosine_similarity }).collect()
    }

    /// Read-only access to an embedding by memory ID.
    pub fn get_embedding(&self, id: MemoryId) -> Option<&[f32]> {
        if let Some(&off) = self.emb_offsets.get(&id) {
            if let Some(ref mm) = self.emb_mmap {
                let start = off as usize;
                let end = start + EMBED_DIM * 4;
                if end <= mm.len() {
                    let bytes = &mm[start..end];
                    let ptr = bytes.as_ptr() as *const f32;
                    // SAFETY: offset was computed from the sidecar header; f32×EMBED_DIM data
                    // starts at a 4-byte-aligned offset (24 + n*1032, always divisible by 4).
                    return Some(unsafe { std::slice::from_raw_parts(ptr, EMBED_DIM) });
                }
            }
        }
        self.embeddings.get(&id).map(|v| v.as_slice())
    }

    /// Center a vector against the corpus mean and L2-normalize. Falls back to plain
    /// normalize when centering is disabled (empty/mismatched centroid). The returned
    /// vector lives in the same centered space as the persisted binary codes.
    #[inline]
    fn center_norm(&self, v: &[f32]) -> Option<Vec<f32>> {
        if self.centroid.len() == v.len() {
            let mut c = vec![0f32; v.len()];
            for i in 0..v.len() {
                c[i] = v[i] - self.centroid[i];
            }
            normalize(&c)
        } else {
            normalize(v)
        }
    }

    /// Recompute the corpus-mean centroid from all live (non-deleted) unit embeddings.
    /// O(N·dim) over EmbLookup — mmap-safe (read-only). Frozen after computation; only
    /// re-run via force_reindex. Persisted to the `.mu` sidecar by the snapshot writer.
    fn compute_centroid(&mut self) {
        let dim = EMBED_DIM;
        let ids: Vec<MemoryId> = self
            .embeddings
            .keys()
            .chain(self.emb_offsets.keys())
            .copied()
            .collect();
        let (sum, n) = {
            let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            let mut sum = vec![0f64; dim];
            let mut n = 0u64;
            for id in ids {
                if self.deleted.contains(&id) {
                    continue;
                }
                let Some(e) = embs.get(id) else { continue };
                if e.len() != dim {
                    continue;
                }
                let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-6 {
                    continue;
                }
                for i in 0..dim {
                    sum[i] += (e[i] / norm) as f64;
                }
                n += 1;
            }
            (sum, n)
        };
        self.centroid = if n == 0 {
            Vec::new()
        } else {
            sum.iter().map(|&s| (s / n as f64) as f32).collect()
        };
        if n > 0 {
            eprintln!(
                "[hnsw] compute_centroid: corpus mean over {} unit embeddings (|mu|={:.4})",
                n,
                l2_norm(&self.centroid)
            );
        }
    }

    /// Rebuild every sign-bit binary code (and the flat scan buffer) from the
    /// current embeddings, centered against the current centroid. O(N·dim).
    fn rebinarize_all(&mut self) {
        let centroid = self.centroid.clone();
        let all_ids: Vec<MemoryId> = self.embeddings.keys()
            .chain(self.emb_offsets.keys())
            .copied()
            .collect();
        {
            let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            self.binary_codes = all_ids.iter()
                .filter_map(|&id| embs.get(id).map(|e| (id, binarize_centered(e, &centroid))))
                .collect();
        }
        self.binary_vec.clear();
        self.binary_vec_pos.clear();
        for (id, codes) in &self.binary_codes {
            let arr: [u64; BINARY_WORDS] = codes[..BINARY_WORDS].try_into().unwrap_or([0u64; BINARY_WORDS]);
            self.binary_vec_pos.insert(*id, self.binary_vec.len());
            self.binary_vec.push((*id, arr));
        }
    }

    /// Centroid-drift maintenance for the centered binary-Hamming path.
    ///
    /// Codes and queries are centered against the SAME frozen centroid, so they
    /// never disagree with each other — but as the corpus drifts, the frozen mean
    /// loses its anisotropy-correction power and Hamming distance decorrelates
    /// from cosine (recall decay at >FLAT_SCAN_MAX scale). Called from
    /// upsert()/remove(): O(1) until ~10% of the corpus has churned (min 512;
    /// env CHITTA_CENTROID_REBUILD_WRITES), then one O(N·dim) mean recompute.
    /// Codes are re-binarized (second O(N·dim) pass) only when the fresh mean
    /// actually diverges (cos < 0.995; env CHITTA_CENTROID_DRIFT_COS) — below
    /// that the frozen centroid is kept and codes stay byte-identical.
    fn maybe_refresh_centroid(&mut self) {
        if self.centroid.len() != EMBED_DIM {
            return; // centering disabled — raw sign codes never drift
        }
        self.centroid_stale_writes += 1;
        if self.inhibit_hnsw {
            return; // WAL replay — defer to the first post-replay write
        }
        let trigger = std::env::var("CHITTA_CENTROID_REBUILD_WRITES").ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or_else(|| ((self.total_embedding_count() / 10) as u64).max(512));
        if self.centroid_stale_writes < trigger {
            return;
        }
        self.centroid_stale_writes = 0;
        let frozen = std::mem::take(&mut self.centroid);
        self.compute_centroid();
        if self.centroid.len() != frozen.len() {
            self.centroid = frozen; // degenerate corpus — keep the frozen mean
            return;
        }
        let denom = l2_norm(&frozen) * l2_norm(&self.centroid);
        let cos = if denom > 1e-12 { dot(&frozen, &self.centroid) / denom } else { 0.0 };
        let gate = std::env::var("CHITTA_CENTROID_DRIFT_COS").ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.995);
        if cos >= gate {
            eprintln!("[hnsw] centroid drift check: cos={:.4} >= {:.4} — keeping frozen mean", cos, gate);
            self.centroid = frozen; // negligible drift — existing codes stay valid
            return;
        }
        eprintln!(
            "[hnsw] centroid drift cos={:.4} < {:.4} — adopting fresh mean, re-binarizing {} codes",
            cos, gate, self.total_embedding_count()
        );
        self.rebinarize_all();
    }

    /// True if this memory has a usable (non-zero norm) embedding in the index.
    /// Returns false for missing, zero-vector, or NaN embeddings — those score 0.0
    /// against every query and need to be re-embedded.
    pub fn contains(&self, id: MemoryId) -> bool {
        match self.get_embedding(id) {
            None => false,
            Some(emb) => emb.iter().map(|&v| v * v).sum::<f32>() > 1e-6,
        }
    }

    /// Iterator over all memory IDs that have an embedding (including soft-deleted).
    /// Covers both heap-stored and mmap-backed embeddings.
    pub fn all_ids(&self) -> impl Iterator<Item = MemoryId> + '_ {
        self.embeddings.keys().copied()
            .chain(self.emb_offsets.keys().copied())
    }

    fn invalidate_turbo(&self) {
        let mut guard = self.turbo.write();
        self.turbo_epoch.fetch_add(1, Ordering::Relaxed);
        *guard = None;
    }

    pub(crate) fn prune_turbo_changes(&mut self) {
        let watermark = self.turbo.read().as_ref().map(|t| t.built_at_mutation);
        if let Some(watermark) = watermark {
            self.turbo_changed.retain(|_, mutation| *mutation > watermark);
        }
    }

    pub(crate) fn plan_refresh_searches(&self, candidates: &[(MemoryId, Vec<f32>)], realm: Option<&str>) -> Option<RefreshSearch> {
        if candidates.is_empty() || flat_scan_max() == 0 || self.total_embedding_count() > flat_scan_max() { return None; }
        if self.centroid.is_empty() && realm.and_then(|r| self.per_realm_hnsw.get(r)).is_some_and(|g| !g.is_empty()) { return None; }
        if std::env::var_os("CHITTA_FLAT_SCAN_CENTER").is_some()
            && std::env::var_os("CHITTA_FLAT_SCAN_RAW").is_none()
            && self.centroid.len() == EMBED_DIM { return None; }
        if candidates.iter().any(|(_, q)| self.center_norm(q).is_none() || normalize(q).is_none()) { return None; }
        let turbo = self.turbo.read().clone()?;
        // Dirty vectors keep the existing path, including its exact stale-vector
        // replacement semantics. Do not silently compute against older embeddings.
        if turbo.built_at_mutation != self.mutations { return None; }
        Some(RefreshSearch { turbo, deleted: self.deleted.clone() })
    }

    /// Maintenance-only: copy embeddings under the store read guard, then build
    /// and publish after releasing it. Search never calls this function.
    pub(crate) fn plan_turbo_rebuild(&self, min_mutations: u64) -> Option<TurboBuild> {
        if self.total_embedding_count() < HNSW_THRESHOLD { return None; }
        let current = self.turbo.read().clone();
        if let Some(t) = current.as_ref() {
            if self.mutations == t.built_at_mutation
                || (!t.startup_delta && self.mutations.saturating_sub(t.built_at_mutation) <= min_mutations
                    && t.built_at.elapsed() < std::time::Duration::from_secs(60)) {
                return None;
            }
        }
        let mut flat = Vec::with_capacity(self.total_embedding_count() * EMBED_DIM);
        let mut ids = Vec::with_capacity(self.total_embedding_count());
        let mut ordered_ids: Vec<_> = self.all_ids().collect();
        ordered_ids.sort_unstable();
        ordered_ids.dedup();
        for id in ordered_ids {
            if self.deleted.contains(&id) { continue; }
            let Some(emb) = self.get_embedding(id) else { continue; };
            if emb.len() != EMBED_DIM { continue; }
            let Some(unit) = normalize(emb) else { continue; };
            flat.extend_from_slice(&unit);
            ids.push(id);
        }
        Some(TurboBuild { flat, ids, mutation: self.mutations, published: self.turbo.clone(),
            epoch: self.turbo_epoch.load(Ordering::Relaxed), current_epoch: self.turbo_epoch.clone() })
    }

    /// Activate mmap mode: build offset index from .emb sidecar, then clear the heap HashMap.
    /// No-op below 200K (guard in field.rs). Safe after normalize_all() has run.
    /// After activation, get_embedding() serves from mmap; new upsert() calls repopulate the heap.
    pub fn activate_mmap_embeddings(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }?;
        if mmap.len() < 16 { return Ok(()); }
        let magic = u64::from_le_bytes(mmap[0..8].try_into().unwrap());
        if magic != Self::EMB_MAGIC { return Ok(()); }
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let record_size = 8 + EMBED_DIM * 4;
        if mmap.len() < 16 + count * record_size { return Ok(()); }
        let mut offsets = HashMap::with_capacity(count);
        let mut off = 16usize;
        for _ in 0..count {
            let id = u64::from_le_bytes(mmap[off..off+8].try_into().unwrap());
            offsets.insert(id, (off + 8) as u64);
            off += record_size;
        }
        self.emb_offsets = offsets;
        self.emb_mmap = Some(std::sync::Arc::new(mmap));
        self.embeddings.clear();
        self.embeddings.shrink_to_fit();
        eprintln!("[hnsw] activate_mmap_embeddings: {} entries mapped, heap cleared", count);
        Ok(())
    }

    /// Shared metadata bookkeeping for `upsert`/`upsert_deferred`: normalize,
    /// coarse/LSH/binary codes, embeddings-map insert, per-realm counts. Does NOT
    /// touch the global HNSW graph. Returns `(embedding_ready, realm_is_small)`.
    fn upsert_meta(&mut self, memory_id: MemoryId, mut embedding: Vec<f32>, realm: Option<&str>) -> (bool, bool) {
        self.mutations += 1;
        self.remove(memory_id);
        self.turbo_changed.insert(memory_id, self.mutations);
        normalize_in_place(&mut embedding);
        self.deleted.remove(&memory_id);
        let coarse_ids = self.assign_coarse(&embedding);
        for coarse_id in &coarse_ids {
            self.coarse_members
                .entry(*coarse_id)
                .or_default()
                .push(memory_id);
        }
        self.mem_coarse.insert(memory_id, coarse_ids);
        let lsh_ids = self.assign_lsh(&embedding);
        for (table_idx, signature) in lsh_ids.iter().enumerate() {
            self.lsh_buckets[table_idx]
                .entry(*signature)
                .or_default()
                .push(memory_id);
        }
        self.mem_lsh.insert(memory_id, lsh_ids);
        // Only binarize and index into HNSW when the embedding has the correct dimension.
        // Pending-embed memories arrive with an empty slice and must not trigger binarize.
        let embedding_ready = embedding.len() == EMBED_DIM;
        if embedding_ready {
            // Center against the frozen corpus mean (when set) so new writes share the
            // same binary space as the reindexed corpus; raw binarize otherwise.
            let codes = binarize_centered(&embedding, &self.centroid);
            let arr: [u64; BINARY_WORDS] = codes[..BINARY_WORDS].try_into().unwrap_or([0u64; BINARY_WORDS]);
            if let Some(&pos) = self.binary_vec_pos.get(&memory_id) {
                self.binary_vec[pos].1 = arr;
            } else {
                self.binary_vec_pos.insert(memory_id, self.binary_vec.len());
                self.binary_vec.push((memory_id, arr));
            }
            self.binary_codes.insert(memory_id, codes);
        }
        self.embeddings.insert(memory_id, embedding);

        // Track realm membership for per-realm HNSW routing.
        if let Some(r) = realm {
            *self.per_realm_counts.entry(r.to_string()).or_default() += 1;
            self.per_id_realm.insert(memory_id, r.to_string());
        }

        // Global HNSW: skip for small-realm writes — binary Hamming + linear scan handles those.
        let realm_is_small = realm.map(|r|
            self.per_realm_counts.get(r).copied().unwrap_or(0) < PER_REALM_HNSW_THRESHOLD
        ).unwrap_or(false);
        (embedding_ready, realm_is_small)
    }

    /// Per-realm HNSW insert — build when realm is large enough for its own graph.
    fn upsert_per_realm(&mut self, memory_id: MemoryId, realm: Option<&str>, embedding_ready: bool) {
        if embedding_ready {
            if let Some(r) = realm {
                if self.per_realm_counts.get(r).copied().unwrap_or(0) >= PER_REALM_HNSW_THRESHOLD {
                    let emb = self.embeddings[&memory_id].clone();
                    let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                    self.per_realm_hnsw.entry(r.to_string()).or_default().insert(memory_id, &emb, &embs);
                }
            }
        }
    }

    /// Add or update an embedding. Un-deletes the entry if it was soft-deleted.
    pub fn upsert(&mut self, memory_id: MemoryId, embedding: Vec<f32>, realm: Option<&str>) {
        let (embedding_ready, realm_is_small) = self.upsert_meta(memory_id, embedding, realm);
        if !self.inhibit_hnsw && self.use_hnsw() && embedding_ready && !realm_is_small {
            let emb = self.embeddings[&memory_id].clone();
            let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            if self.use_tier2() {
                self.delta_hnsw.insert(memory_id, &emb, &embs);
            } else {
                self.hnsw.insert(memory_id, &emb, &embs);
            }
        }
        self.upsert_per_realm(memory_id, realm, embedding_ready);
        self.maybe_refresh_centroid();
    }

    /// Deferred-HNSW variant of `upsert`: performs all metadata bookkeeping but SKIPS
    /// BOTH expensive graph inserts — the global delta/base HNSW AND the per-realm HNSW
    /// (they activate at the same 500-member threshold, so both must move off the write
    /// lock for the fix to land). Returns `true` when the node needs a graph plan; the
    /// caller drives `plan_delta_batch`/`apply_delta_batch` so the O(log N) neighbor
    /// search runs off the write lock.
    pub fn upsert_deferred(&mut self, memory_id: MemoryId, embedding: Vec<f32>, realm: Option<&str>) -> bool {
        let (embedding_ready, realm_is_small) = self.upsert_meta(memory_id, embedding, realm);
        self.maybe_refresh_centroid();
        if !embedding_ready { return false; }
        let global = !self.inhibit_hnsw && self.use_hnsw() && !realm_is_small;
        let per_realm = realm
            .map(|r| self.per_realm_counts.get(r).copied().unwrap_or(0) >= PER_REALM_HNSW_THRESHOLD)
            .unwrap_or(false);
        global || per_realm
    }

    /// Off-lock (`&self`) plan phase for a deferred batch insert (Step-1 of the
    /// deferred-batched-insert re-architecture). Computes HNSW neighbor plans for
    /// `ids` against the CURRENT active-tier graph WITHOUT mutating it, so it runs
    /// under `semantic_idx.read()` (concurrent with recall) instead of the write lock.
    /// Nodes already present, soft-deleted, or without a ready embedding are skipped.
    /// Levels are id-seeded (order-independent), matching `backfill_hnsw_delta_parallel`.
    pub fn plan_delta_batch(&self, ids: &[MemoryId]) -> DeltaBatchPlan {
        let tier2 = self.use_tier2();
        let global_graph = if tier2 { &self.delta_hnsw } else { &self.hnsw };
        let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
        let global_active = !self.inhibit_hnsw && self.use_hnsw();
        // Global-graph plans (parallel). Skipped for small realms, matching `upsert`.
        let plans: Vec<(MemoryId, usize, Vec<Vec<MemoryId>>)> = if global_active {
            ids.par_iter().filter_map(|&id| {
                if self.deleted.contains(&id) { return None; }
                if self.hnsw.contains(id) || self.delta_hnsw.contains(id) { return None; }
                let realm_small = self.per_id_realm.get(&id)
                    .map(|r| self.per_realm_counts.get(r).copied().unwrap_or(0) < PER_REALM_HNSW_THRESHOLD)
                    .unwrap_or(false);
                if realm_small { return None; }
                let emb = embs.get(id)?;
                if emb.len() != EMBED_DIM { return None; }
                let level = random_level_from_seed(id);
                Some((id, level, global_graph.compute_insert_plan(emb, level, &embs, HNSW_EF_CONSTRUCTION)))
            }).collect()
        } else {
            Vec::new()
        };
        // Per-realm-graph plans (parallel). A node ≥ threshold in its realm is planned
        // against that realm's graph (empty when the realm's graph does not exist yet —
        // apply creates it via or_default()).
        let realm_plans: Vec<(MemoryId, String, usize, Vec<Vec<MemoryId>>)> = ids
            .par_iter()
            .filter_map(|&id| {
                if self.deleted.contains(&id) { return None; }
                let r = self.per_id_realm.get(&id)?;
                if self.per_realm_counts.get(r).copied().unwrap_or(0) < PER_REALM_HNSW_THRESHOLD { return None; }
                if self.per_realm_hnsw.get(r).map(|g| g.contains(id)).unwrap_or(false) { return None; }
                let emb = embs.get(id)?;
                if emb.len() != EMBED_DIM { return None; }
                let level = random_level_from_seed(id);
                let empty = HnswGraph::new();
                let graph = self.per_realm_hnsw.get(r).unwrap_or(&empty);
                Some((id, r.clone(), level, graph.compute_insert_plan(emb, level, &embs, HNSW_EF_CONSTRUCTION)))
            })
            .collect();
        DeltaBatchPlan { tier2, plans, realm_plans }
    }

    /// Brief (`&mut self`) apply phase: wire a pre-computed batch plan into the live
    /// active-tier graph. O(batch × M) pointer work — no distance search — so the
    /// write lock is held for milliseconds, not the O(log N) neighbor search. Nodes
    /// a concurrent writer inserted between plan and apply are untouched (their entries
    /// are preserved; only the batch nodes are wired in), so no torn graph / lost write.
    pub fn apply_delta_batch(&mut self, plan: DeltaBatchPlan) {
        if plan.is_empty() { return; }
        self.mutations += 1;
        // Disjoint-field split borrow: mutate the graph fields, read `deleted`.
        let SemanticIndex { hnsw, delta_hnsw, per_realm_hnsw, deleted, .. } = self;
        // Global graph.
        let target = if plan.tier2 { delta_hnsw } else { hnsw };
        for (id, level, per_layer) in &plan.plans {
            if deleted.contains(id) { continue; }
            if target.contains(*id) { continue; }
            target.wire_precomputed(*id, *level, per_layer);
        }
        // Per-realm graphs (created on first insert via or_default).
        for (id, realm, level, per_layer) in &plan.realm_plans {
            if deleted.contains(id) { continue; }
            let g = per_realm_hnsw.entry(realm.clone()).or_default();
            if g.contains(*id) { continue; }
            g.wire_precomputed(*id, *level, per_layer);
        }
    }

    /// Mark a memory as deleted — excluded from future search results.
    pub fn remove(&mut self, memory_id: MemoryId) {
        self.mutations += 1;
        self.turbo_changed.insert(memory_id, self.mutations);
        if let Some(realm) = self.per_id_realm.remove(&memory_id) {
            if let Some(graph) = self.per_realm_hnsw.get_mut(&realm) {
                graph.remove(memory_id);
            }
            self.per_realm_counts.entry(realm).and_modify(|c| *c = c.saturating_sub(1));
        }
        self.hnsw.remove(memory_id);
        self.delta_hnsw.remove(memory_id);
        self.deleted.insert(memory_id);
        self.embeddings.remove(&memory_id);
        self.emb_offsets.remove(&memory_id);
        if let Some(&pos) = self.binary_vec_pos.get(&memory_id) {
            let last = self.binary_vec.len().saturating_sub(1);
            if !self.binary_vec.is_empty() {
                if pos != last {
                    self.binary_vec.swap(pos, last);
                    let swapped_id = self.binary_vec[pos].0;
                    self.binary_vec_pos.insert(swapped_id, pos);
                }
                self.binary_vec.pop();
            }
            self.binary_vec_pos.remove(&memory_id);
        }
        self.binary_codes.remove(&memory_id);
        if let Some(coarse_ids) = self.mem_coarse.remove(&memory_id) {
            for coarse_id in coarse_ids {
                let remove_bucket = if let Some(members) = self.coarse_members.get_mut(&coarse_id) {
                    members.retain(|id| *id != memory_id);
                    members.is_empty()
                } else {
                    false
                };
                if remove_bucket {
                    self.coarse_members.remove(&coarse_id);
                }
            }
        }
        if let Some(lsh_ids) = self.mem_lsh.remove(&memory_id) {
            for (table_idx, signature) in lsh_ids.into_iter().enumerate() {
                let remove_bucket =
                    if let Some(members) = self.lsh_buckets[table_idx].get_mut(&signature) {
                        members.retain(|id| *id != memory_id);
                        members.is_empty()
                    } else {
                        false
                    };
                if remove_bucket {
                    self.lsh_buckets[table_idx].remove(&signature);
                }
            }
        }

        self.maybe_refresh_centroid();
    }

    /// Search for k nearest neighbours to `query` by cosine similarity.
    ///
    /// `allowed` is an optional pre-filtered set of MemoryIds (realm filter
    /// applied by the caller). Entries absent from `allowed` are skipped.
    /// Returns hits sorted by cosine similarity descending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
        realm: Option<&str>,
    ) -> Vec<SemanticHit> {
        if query.len() != EMBED_DIM {
            return vec![];
        }
        let Some(query_unit) = self.center_norm(query) else {
            return vec![];
        };

        // Per-realm HNSW fast path: bypass global index + allowed-set filter overhead.
        // Skipped when centering is active — the HNSW graph is built in raw cosine space,
        // so a centered query would mismatch it. The centered binary path below handles
        // realm scoping via the `allowed` set instead.
        if self.centroid.is_empty() {
            if let Some(r) = realm {
                if let Some(graph) = self.per_realm_hnsw.get(r) {
                    if !graph.is_empty() {
                        let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                        let pairs = graph.search(&query_unit, k, None, &self.deleted, &embs);
                        return pairs.into_iter()
                            .map(|(memory_id, cosine_similarity)| SemanticHit { memory_id, cosine_similarity })
                            .collect();
                    }
                }
            }
        }

        if let Some(allowed_ids) = allowed {
            if allowed_ids.len() <= MIN_CANDIDATES {
                return self.search_candidates(query, allowed_ids.iter().copied(), k);
            }
        }

        // Flat-scan for sub-million-vector stores (see FLAT_SCAN_MAX): exact top-k over all
        // vectors, skipping the binary-Hamming prefilter (no recall loss at this size). Scores
        // are CENTERED cosine — each vector is centered against the corpus mean (.mu) before
        // cosine, removing bge's dominant anisotropy direction and ~2.3x'ing the
        // relevant-vs-noise margin (measured offline: raw margin 0.35 -> centered 0.79; raw
        // cosine left recall in the most-anisotropic regime). Centering is analytic with NO
        // per-candidate allocation: stored emb is unit, so |emb-mu|^2 = 1 - 2(emb·mu) + |mu|^2
        // and the centered numerator is (q_c·emb) - (q_c·mu). Falls back to raw cosine when no
        // centroid is loaded or CHITTA_FLAT_SCAN_RAW=1 (A/B + rollback escape hatch).
        let flat_max = flat_scan_max();
        if flat_max > 0 && self.total_embedding_count() <= flat_max {
            // Default to the RAW turbovec arm: exact-cosine ranking parity with the
            // centered scalar arm on GOLDEN_SET v6 (gold ranks #1 either way) but SIMD-fast
            // instead of 8.5s/query, so it is viable as the standing recall path (and keeps
            // CW-refresh's k=9 probes at ~1ms). Centering removes bge anisotropy but that
            // buys no ranking gain here while forcing the scalar path — so it is opt-in via
            // CHITTA_FLAT_SCAN_CENTER=1 for A/B. CHITTA_FLAT_SCAN_RAW retained as a no-op
            // alias that also selects raw.
            let center = std::env::var_os("CHITTA_FLAT_SCAN_CENTER").is_some()
                && std::env::var_os("CHITTA_FLAT_SCAN_RAW").is_none()
                && self.centroid.len() == EMBED_DIM;
            // Raw-cosine arm: turbovec inner-product == cosine over unit vectors.
            // (Centered arm below stays scalar — turbovec can't reproduce .mu
            // centering.) Over-fetch when an allowed/realm filter is present so
            // post-hoc filtering still yields up to k survivors.
            if !center {
                if let Some(q) = normalize(query) {
                    let guard = self.turbo.read().clone();
                    if let Some(ts) = guard.as_ref() {
                        let fetch = (if allowed.is_some() { k.saturating_mul(4).max(k) } else { k })
                            .saturating_add(self.turbo_changed.len());
                        let res = ts.index.search(&q, fetch.min(ts.ids.len().max(1)));
                        let mut hits: Vec<SemanticHit> = Vec::with_capacity(k);
                        let idxs = res.indices_for_query(0);
                        let scs = res.scores_for_query(0);
                        for (row, sc) in idxs.iter().zip(scs.iter()) {
                            if *row < 0 { continue; }
                            let Some(&id) = ts.ids.get(*row as usize) else { continue; };
                            if self.deleted.contains(&id) { continue; }
                            if self.turbo_changed.get(&id).is_some_and(|m| *m > ts.built_at_mutation) { continue; }
                            if let Some(a) = allowed { if !a.contains(&id) { continue; } }
                            if !sc.is_finite() { continue; }
                            hits.push(SemanticHit { memory_id: id, cosine_similarity: *sc });
                            if hits.len() >= k { break; }
                        }
                        // A previous index can contain old vectors and omit newly
                        // inserted ones. Replace affected scores from current embeddings.
                        for (&id, &mutation) in &self.turbo_changed {
                            if mutation <= ts.built_at_mutation { continue; }
                            hits.retain(|h| h.memory_id != id);
                            if self.deleted.contains(&id) { continue; }
                            if let Some(a) = allowed { if !a.contains(&id) { continue; } }
                            if let Some(emb) = self.get_embedding(id) {
                                let sim = dot(&q, emb);
                                if sim.is_finite() { hits.push(SemanticHit { memory_id: id, cosine_similarity: sim }); }
                            }
                        }
                        hits.sort_unstable_by(|a, b| b.cosine_similarity.total_cmp(&a.cosine_similarity).then_with(|| a.memory_id.cmp(&b.memory_id)));
                        hits.truncate(k);
                        return hits;
                    }
                }
            }
            // Centered+normalized query (query_unit) when centering; else raw-normalized.
            let q_opt = if center { Some(query_unit.clone()) } else { normalize(query) };
            if let Some(q) = q_opt {
                let q_dot_mu = if center { dot(&q, &self.centroid) } else { 0.0 };
                let mu_sq    = if center { dot(&self.centroid, &self.centroid) } else { 0.0 };
                let mut top_k = BinaryHeap::new();
                for id in self.all_ids() {
                    if self.deleted.contains(&id) {
                        continue;
                    }
                    if let Some(a) = allowed {
                        if !a.contains(&id) {
                            continue;
                        }
                    }
                    let Some(emb) = self.get_embedding(id) else { continue; };
                    if emb.len() != EMBED_DIM {
                        continue;
                    }
                    // Centered cosine via the analytic identity above (no per-candidate
                    // allocation), or raw dot when centering is off. Stored embeddings are unit
                    // (normalize_all on load + upsert). is_finite() drops any corrupt/zero vector.
                    let sim = if center {
                        let denom = (1.0 - 2.0 * dot(emb, &self.centroid) + mu_sq).max(1e-12).sqrt();
                        (dot(&q, emb) - q_dot_mu) / denom
                    } else {
                        dot(&q, emb)
                    };
                    if !sim.is_finite() {
                        continue;
                    }
                    push_top_k(&mut top_k, k, id, sim);
                }
                return heap_to_hits(top_k);
            }
        }

        // Binary Hamming pre-filter: primary path when codes are fully in sync.
        // O(N × 12 u64) scan + float rescore of top-HAMMING_CANDIDATES — replaces HNSW.
        let total = self.total_embedding_count();
        if self.binary_codes.len() == total
            && self.binary_vec.len() == self.binary_codes.len()
            && !self.binary_codes.is_empty() {
            let query_bits = binarize(&query_unit);
            let candidates = self.hamming_candidates(&query_bits, allowed);
            if !candidates.is_empty() {
                let mut top_k = BinaryHeap::new();
                for memory_id in candidates {
                    let Some(embedding) = self.get_embedding(memory_id) else { continue; };
                    let Some(cand) = self.center_norm(embedding) else { continue; };
                    let sim = dot(&query_unit, &cand);
                    push_top_k(&mut top_k, k, memory_id, sim);
                }
                return heap_to_hits(top_k);
            }
        }

        // Fallback: HNSW when active and binary codes are absent/stale.
        // In two-tier mode, search both base and delta graphs, merge by similarity.
        // Only when centering is disabled — the HNSW graph is built in raw cosine space.
        if self.centroid.is_empty() && self.use_hnsw() && (!self.hnsw.is_empty() || !self.delta_hnsw.is_empty()) {
            let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            let mut pairs: Vec<(MemoryId, f32)> = if !self.hnsw.is_empty() {
                self.hnsw.search(&query_unit, k, allowed, &self.deleted, &embs)
            } else {
                vec![]
            };
            if self.use_tier2() && !self.delta_hnsw.is_empty() {
                let delta_pairs = self.delta_hnsw.search(&query_unit, k, allowed, &self.deleted, &embs);
                pairs.extend(delta_pairs);
                pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                pairs.dedup_by_key(|p| p.0);
                pairs.truncate(k);
            }
            return pairs
                .into_iter()
                .map(|(memory_id, cosine_similarity)| SemanticHit { memory_id, cosine_similarity })
                .collect();
        }

        let mut candidates = self.collect_lsh_candidates(&query_unit, allowed, k);
        if candidates.is_empty() {
            let probes = self.choose_probe_ids(&query_unit, MIN_PROBES);
            candidates = self.collect_candidates(&probes, allowed, k);
        }
        if candidates.is_empty() {
            return vec![];
        }

        let mut top_k = BinaryHeap::new();
        for memory_id in candidates {
            let Some(embedding) = self.embeddings.get(&memory_id) else {
                continue;
            };
            let Some(cand) = self.center_norm(embedding) else { continue; };
            let sim = dot(&query_unit, &cand);
            push_top_k(&mut top_k, k, memory_id, sim);
        }

        heap_to_hits(top_k)
    }

    /// Like `search`, but applies a retrieval-signature alignment boost.
    ///
    /// For each candidate, after computing cosine similarity, adds:
    ///   `beta * max(0.0, dot(query_ctx_32, signature_32))`
    /// where `query_ctx_32` is a 32-dim projected query context sketch and
    /// `signature_32` is the cached mean retrieval signature for that memory
    /// (from `RetrievalHistory::signature`).
    ///
    /// Memories with empty signatures receive zero boost.
    pub fn search_with_signature_boost(
        &self,
        query: &[f32],
        k: usize,
        allowed: Option<&HashSet<MemoryId>>,
        query_ctx_32: &[f32],
        signatures: &HashMap<MemoryId, Vec<f32>>,
        beta: f32,
        _realm: Option<&str>,
    ) -> Vec<SemanticHit> {
        if query.len() != EMBED_DIM {
            return vec![];
        }
        let Some(query_unit) = self.center_norm(query) else {
            return vec![];
        };

        let binary_ready = self.binary_codes.len() == self.total_embedding_count() && !self.binary_codes.is_empty();
        let candidates = if let Some(allowed_ids) = allowed {
            if allowed_ids.len() <= MIN_CANDIDATES {
                allowed_ids.iter().copied().collect::<Vec<_>>()
            } else if binary_ready {
                let query_bits = binarize(&query_unit);
                self.hamming_candidates(&query_bits, allowed)
            } else {
                let mut cands = self.collect_lsh_candidates(&query_unit, allowed, k);
                if cands.is_empty() {
                    let probes = self.choose_probe_ids(&query_unit, MIN_PROBES);
                    cands = self.collect_candidates(&probes, allowed, k);
                }
                cands
            }
        } else if binary_ready {
            let query_bits = binarize(&query_unit);
            self.hamming_candidates(&query_bits, allowed)
        } else {
            let mut cands = self.collect_lsh_candidates(&query_unit, allowed, k);
            if cands.is_empty() {
                let probes = self.choose_probe_ids(&query_unit, MIN_PROBES);
                cands = self.collect_candidates(&probes, allowed, k);
            }
            cands
        };

        if candidates.is_empty() {
            return vec![];
        }

        let mut top_k = BinaryHeap::new();
        for memory_id in candidates {
            if self.deleted.contains(&memory_id) {
                continue;
            }
            let Some(embedding) = self.get_embedding(memory_id) else {
                continue;
            };
            let Some(cand) = self.center_norm(embedding) else { continue; };
            let base_sim = dot(&query_unit, &cand);
            let boost = if let Some(sig) = signatures.get(&memory_id) {
                if !sig.is_empty() && !query_ctx_32.is_empty() {
                    beta * dot_n(query_ctx_32, sig, 32).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let adjusted_sim = base_sim + boost;
            push_top_k(&mut top_k, k, memory_id, adjusted_sim);
        }

        heap_to_hits(top_k)
    }

    /// Exact cosine search over a bounded candidate set.
    pub fn search_candidates<I>(&self, query: &[f32], candidates: I, k: usize) -> Vec<SemanticHit>
    where
        I: IntoIterator<Item = MemoryId>,
    {
        if query.len() != EMBED_DIM {
            return vec![];
        }
        let Some(query_unit) = self.center_norm(query) else {
            return vec![];
        };

        let mut top_k = BinaryHeap::new();
        for memory_id in candidates {
            if self.deleted.contains(&memory_id) {
                continue;
            }
            let Some(embedding) = self.get_embedding(memory_id) else {
                continue;
            };
            let Some(cand) = self.center_norm(embedding) else { continue; };
            let sim = dot(&query_unit, &cand);
            push_top_k(&mut top_k, k, memory_id, sim);
        }

        heap_to_hits(top_k)
    }

    pub fn len(&self) -> usize {
        self.total_embedding_count()
    }

    pub fn is_empty(&self) -> bool {
        self.total_embedding_count() == 0
    }

    /// Drop all embeddings whose length doesn't match EMBED_DIM (model-swap migration).
    /// Returns the IDs that were purged so the caller can mark them embed_pending.
    pub fn purge_wrong_dim(&mut self) -> Vec<MemoryId> {
        self.mutations += 1;
        let bad: Vec<MemoryId> = self.embeddings.iter()
            .filter(|(_, v)| v.len() != EMBED_DIM)
            .map(|(&id, _)| id)
            .collect();
        for &id in &bad {
            self.embeddings.remove(&id);
            self.binary_codes.remove(&id);
            self.mem_coarse.remove(&id);
            self.mem_lsh.remove(&id);
            self.deleted.remove(&id);
        }
        if !bad.is_empty() {
            self.invalidate_turbo();
            // Reset coarse/LSH/binary_vec — they are also wrong-dim.
            self.coarse_members.clear();
            self.lsh_buckets = vec![std::collections::HashMap::new(); LSH_TABLES];
            self.binary_vec.clear();
            self.binary_vec_pos.clear();
            eprintln!(
                "[chitta-field] purged {} wrong-dim embeddings (model swap migration)",
                bad.len()
            );
        }
        bad
    }

    /// Normalize embeddings loaded from older snapshots that stored raw vectors.
    pub fn normalize_all(&mut self) { self.normalize_with_cache(None); }

    /// Cached divisors preserve the historical f32 division exactly; only the
    /// replay delta needs a norm reduction and LSH projection. The raw snapshot
    /// digest is captured while loading .emb, never from post-replay vectors.
    pub(crate) fn normalize_with_cache(&mut self, path: Option<&std::path::Path>) {
        self.mutations += 1;
        self.invalidate_turbo();
        let key = self.startup_key.filter(|_| self.emb_mmap.is_none());
        let cached: Option<HashMap<MemoryId, (f32, Vec<u16>)>> = path.zip(key.as_ref())
            .and_then(|(p, k)| crate::startup_cache::load(p, k));
        let cached = cached.filter(|m| m.values().all(|(_, sigs)|
            sigs.len() == LSH_TABLES && sigs.iter().all(|s| (*s as usize) < (1 << LSH_BITS))));
        let cache_hit = cached.is_some();
        // Save only a pristine snapshot subset, never a post-replay mixture under
        // the snapshot's key. Normal checkpoints always create both base caches.
        let can_save = !cache_hit && self.turbo_changed.is_empty() && key.is_some();
        let mut divisors = HashMap::new();
        if can_save {
            divisors = refresh_pool().install(|| self.embeddings.par_iter()
                .map(|(&id, v)| (id, l2_norm(v))).collect());
        }
        let changed = &self.turbo_changed;
        refresh_pool().install(|| self.embeddings.par_iter_mut().for_each(|(id, embedding)| {
            let norm = cached.as_ref().filter(|_| !changed.contains_key(id))
                .and_then(|m| m.get(id)).map(|r| r.0).unwrap_or_else(|| l2_norm(embedding));
            if !(norm < 1e-9 || norm == 1.0) {
                for x in embedding { *x /= norm; }
            }
        }));
        if self.lsh_planes.is_empty() { self.lsh_planes = default_lsh_planes(); }
        // Upsert's signatures precede the loader's final normalization. Recompute
        // delta signatures after it, even if the old count happens to match.
        self.mem_lsh = refresh_pool().install(|| self.all_ids().collect::<Vec<_>>().par_iter()
            .filter_map(|&id| self.get_embedding(id).map(|v| {
                let sigs = cached.as_ref().filter(|_| !changed.contains_key(&id))
                    .and_then(|m| m.get(&id)).map(|r| r.1.clone())
                    .unwrap_or_else(|| self.assign_lsh(v));
                (id, sigs)
            })).collect());
        if self.binary_codes.len() != self.total_embedding_count() {
            // Centered binary codes when a centroid is loaded (.mu sidecar); raw otherwise.
            let centroid = self.centroid.clone();
            let all_ids: Vec<MemoryId> = self.embeddings.keys()
                .chain(self.emb_offsets.keys())
                .copied()
                .collect();
            let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            self.binary_codes = all_ids.iter()
                .filter_map(|&id| embs.get(id).map(|e| (id, binarize_centered(e, &centroid))))
                .collect();
        }
        if self.binary_vec.len() != self.binary_codes.len() {
            self.binary_vec.clear();
            self.binary_vec_pos.clear();
            for (id, codes) in &self.binary_codes {
                let arr: [u64; BINARY_WORDS] = codes[..BINARY_WORDS].try_into().unwrap_or([0u64; BINARY_WORDS]);
                self.binary_vec_pos.insert(*id, self.binary_vec.len());
                self.binary_vec.push((*id, arr));
            }
        }
        if self.coarse_centroids.is_empty() {
            self.coarse_centroids = default_coarse_centroids();
        }
        if self.lsh_planes.is_empty() {
            self.lsh_planes = default_lsh_planes();
        }
        // Coarse, LSH, and HNSW are all snapshot-serialized (or sidecar) and kept in sync
        // by incremental upsert/remove during WAL replay — skip O(N) rebuilds when consistent.
        let total = self.total_embedding_count();
        let coarse_ok = self.mem_coarse.len() == total;
        let lsh_ok    = self.mem_lsh.len()    == total;
        let hnsw_ok   = self.hnsw_len() > 0 && self.hnsw_len() == total;
        if !coarse_ok {
            self.rebuild_ann();
        } else {
            if !lsh_ok {
                // mem_lsh not populated — recompute from embeddings (O(N) with math)
                self.rebuild_lsh();
            } else {
                // mem_lsh is snapshot-restored — rebuild inverted index from it (O(N), no math)
                self.rebuild_lsh_buckets_from_mem();
            }
            let binary_covers = self.binary_codes.len() == total
                && !self.binary_codes.is_empty();
            if !hnsw_ok && self.use_hnsw() {
                if binary_covers && !self.hnsw.is_empty() {
                    self.backfill_hnsw_delta_parallel();
                } else if !binary_covers {
                    eprintln!("[hnsw] rebuild_hnsw: hnsw={} total={} — rebuilding", self.hnsw.len(), total);
                    self.rebuild_hnsw();
                }
                // binary_covers && hnsw.is_empty(): binary Hamming is the active path, skip rebuild.
            }
        }
        self.trim_deleted();
        if can_save {
            if let Some((p, k)) = path.zip(key.as_ref()) {
                let rows: HashMap<_, _> = self.mem_lsh.iter().filter_map(|(&id, sigs)|
                    divisors.get(&id).map(|&norm| (id, (norm, sigs.clone())))).collect();
                if let Err(e) = crate::startup_cache::save(p, k, &rows) {
                    eprintln!("[chitta-field] optional LSH cache write failed: {e}");
                }
            }
        }
        if path.is_some() { eprintln!("[chitta-field] LSH cache hit={cache_hit}"); }
    }

    fn snapshot_vector_key(bytes: &[u8]) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"snapshot-lsh-divisors-turbovec-0.9-v2");
        h.update(crate::ops::EMBED_MODEL_ID.as_bytes());
        h.update((EMBED_DIM as u64).to_le_bytes());
        h.update((LSH_TABLES as u64).to_le_bytes());
        h.update((LSH_BITS as u64).to_le_bytes());
        h.update(bytes);
        h.finalize().into()
    }

    /// Called only on the detached snapshot clone, after saving the raw .emb.
    /// Emulate the next loader's exact normalization (including float rounding).
    /// Never invalidate/publish through Arc fields shared with the live index.
    pub(crate) fn prepare_startup_caches(&mut self, snapshot: &std::path::Path) {
        self.turbo = Arc::new(RwLock::new(None));
        self.turbo_epoch = Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Reload the bytes actually persisted: saving filters invalid/deleted
        // vectors, and mmap snapshots keep their historical raw-vector semantics.
        if !self.load_embeddings_sidecar(&snapshot.with_extension("emb")) { return; }
        self.emb_mmap = None;
        self.emb_offsets.clear();
        self.turbo_changed.clear();
        self.mem_lsh.clear();
        self.lsh_buckets.clear();
        self.normalize_with_cache(Some(&snapshot.with_extension("lsh")));
        self.warm_turbo_with_cache(Some(snapshot));
    }

    pub(crate) fn warm_turbo_with_cache(&self, snapshot: Option<&std::path::Path>) {
        let _phase = crate::profile::LoadPhase::new("turbo_startup");
        let key = self.startup_key.filter(|_| self.emb_mmap.is_none());
        if let Some((snapshot, key)) = snapshot.zip(key.as_ref()) {
            let path = snapshot.with_extension("turbo");
            let meta: Option<(Vec<MemoryId>, [u8; 32])> =
                crate::startup_cache::load(&snapshot.with_extension("turbo.meta"), key);
            if let Some((ids, checksum)) = meta {
                let valid_ids = ids.iter().copied().collect::<HashSet<_>>();
                // Snapshot rows remain usable during replay-delta maintenance.
                // Search excludes deleted/replaced rows and scores changed rows exactly.
                if valid_ids.len() == ids.len()
                    && std::fs::read(&path).ok().is_some_and(|b| crate::startup_cache::digest(&b) == checksum) {
                    if let Ok(index) = refresh_pool().install(|| crate::turbo::SearchIndex::load(&path)) {
                        if index.dim() == EMBED_DIM && index.bit_width() == 4 && index.len() == ids.len() {
                            refresh_pool().install(|| index.prepare());
                            if let Some(query) = ids.first().and_then(|id| self.get_embedding(*id)).and_then(normalize) {
                                refresh_pool().broadcast(|_| { let _ = index.search(&query, 9.min(ids.len())); });
                            }
                            *self.turbo.write() = Some(Arc::new(TurboState { index, ids,
                                built_at_mutation: self.turbo_changed.values().min().map_or(self.mutations, |m| m.saturating_sub(1)),
                                built_at: std::time::Instant::now(), startup_delta: !self.turbo_changed.is_empty() }));
                            eprintln!("[chitta-field] Turbo cache hit=true replay_delta={}", self.turbo_changed.len());
                            return;
                        }
                    }
                }
            }
        }
        if !self.turbo_changed.is_empty() && snapshot.is_some() {
            eprintln!("[chitta-field] Turbo cache hit=false reason=corpus_calibration_delta");
        }
        if let Some(plan) = self.plan_turbo_rebuild(0) { plan.build(); }
        if let Some((snapshot, key)) = snapshot.filter(|_| self.turbo_changed.is_empty()).zip(key.as_ref()) {
            if let Some(t) = self.turbo.read().as_ref() {
                let path = snapshot.with_extension("turbo");
                let tmp = snapshot.with_extension("turbo.tmp");
                let save = (|| -> std::io::Result<()> {
                    t.index.write(&tmp)?;
                    let checksum = crate::startup_cache::digest(&std::fs::read(&tmp)?);
                    std::fs::File::open(&tmp)?.sync_all()?;
                    std::fs::rename(&tmp, &path)?;
                    crate::startup_cache::save(&snapshot.with_extension("turbo.meta"), key, &(&t.ids, checksum))
                })();
                if let Err(e) = save { eprintln!("[chitta-field] optional Turbo cache write failed: {e}"); }
            }
            eprintln!("[chitta-field] Turbo cache hit=false");
        }
    }

    /// Unconditionally rebuild every derived search structure (binary codes, coarse
    /// assignments, LSH buckets, HNSW) from the current embeddings. Used after an
    /// embedding-dimension migration: the on-disk indices were built in a different
    /// vector space, so normalize_all()'s count-gated refresh cannot detect they are
    /// geometrically stale. This recomputes them from scratch.
    pub fn force_reindex(&mut self) {
        self.startup_key = None;
        self.mutations += 1;
        self.invalidate_turbo();
        for embedding in self.embeddings.values_mut() {
            normalize_in_place(embedding);
        }
        // Recompute the anisotropy-correction centroid from the (now unit) corpus, then
        // derive binary codes in the centered space so the Hamming prefilter and cosine
        // rerank agree. Cloned out to avoid borrowing self while mutating binary_codes.
        self.compute_centroid();
        self.centroid_stale_writes = 0;
        self.rebinarize_all();
        // Clear graphs so rebuild_ann's hnsw_valid check fails and forces a full rebuild.
        self.hnsw = HnswGraph::new();
        self.delta_hnsw = HnswGraph::new();
        self.mem_coarse.clear();
        self.mem_lsh.clear();
        // rebuild_ann() skips HNSW when centroid is non-empty (centered-query path gates
        // it off). Temporarily clear centroid so the HNSW IS built here, then restore.
        // Without this, save_hnsw() writes a 9-byte empty file and every daemon restart
        // triggers a full 30-60 min rebuild from scratch.
        let saved_centroid = std::mem::take(&mut self.centroid);
        self.rebuild_ann();
        self.centroid = saved_centroid;
        // rebuild_ann() inserts into delta_hnsw (tier2 mode). Merge into base so
        // save_hnsw() writes a complete graph — on next startup hnsw_valid=true → no rebuild.
        self.merge_delta_into_base();
        self.trim_deleted();
    }

    /// Remove IDs from `deleted` that are no longer reachable via either HNSW graph.
    /// Safe to call any time; called at the end of normalize_all() so it runs once per WAL replay.
    pub fn trim_deleted(&mut self) {
        self.mutations += 1;
        self.deleted.retain(|id| {
            self.hnsw.contains(*id) || self.delta_hnsw.contains(*id)
        });
        self.deleted.shrink_to_fit();
    }

    /// Rebuild lsh_buckets from existing mem_lsh (no embedding math — O(N) HashMap inserts only).
    fn rebuild_lsh_buckets_from_mem(&mut self) {
        self.lsh_buckets = vec![HashMap::new(); LSH_TABLES];
        for (&id, sigs) in &self.mem_lsh {
            for (t, &sig) in sigs.iter().enumerate() {
                self.lsh_buckets[t].entry(sig).or_default().push(id);
            }
        }
    }

    fn rebuild_lsh(&mut self) {
        self.lsh_buckets = vec![HashMap::new(); LSH_TABLES];
        self.mem_lsh.clear();
        let assignments: Vec<(MemoryId, Vec<u16>)> = self
            .embeddings
            .iter()
            .map(|(&id, emb)| (id, self.assign_lsh(emb)))
            .collect();
        for (id, sigs) in assignments {
            for (t, sig) in sigs.iter().enumerate() {
                self.lsh_buckets[t].entry(*sig).or_default().push(id);
            }
            self.mem_lsh.insert(id, sigs);
        }
    }


    /// Hamming pre-filter: returns the top-HAMMING_CANDIDATES IDs by minimum Hamming
    /// distance from the binarized query, restricted to `allowed` when provided.
        fn hamming_candidates(
        &self,
        query_bits: &[u64],
        allowed: Option<&HashSet<MemoryId>>,
    ) -> Vec<MemoryId> {
        let mut scored: Vec<(u32, MemoryId)> = self
            .binary_vec
            .iter()
            .filter_map(|(id, codes)| {
                if self.deleted.contains(id) { return None; }
                if let Some(a) = allowed { if !a.contains(id) { return None; } }
                Some((hamming_dist(query_bits, codes.as_slice()), *id))
            })
            .collect();
        scored.sort_unstable_by_key(|&(d, id)| (d, id));
        scored.truncate(HAMMING_CANDIDATES);
        scored.into_iter().map(|(_, id)| id).collect()
    }

    /// Serialize the HNSW graph to a sidecar file.
    pub fn save_hnsw(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = bincode::serialize(&self.hnsw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = path.with_extension("hnsw.tmp");
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load HNSW from sidecar file. Returns false if file is missing/stale.
    /// "Stale" means the graph node count doesn't match the embeddings count.
    pub fn load_hnsw(&mut self, path: &std::path::Path) -> bool {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("[hnsw] load_hnsw: read {:?} failed: {}", path, e);
                return false;
            }
        };
        let graph: HnswGraph = match bincode::deserialize(&bytes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[hnsw] load_hnsw: deserialize failed: {}", e);
                return false;
            }
        };
        let total = self.total_embedding_count();
        if graph.len() > total {
            // Sidecar is LARGER than the snapshot — can't happen with normal flow, reject.
            eprintln!("[hnsw] load_hnsw: sidecar ({} nodes) exceeds snapshot ({} entries) — discarding",
                graph.len(), total);
            return false;
        }
        let delta = total.saturating_sub(graph.len());
        if delta > 0 {
            eprintln!("[hnsw] load_hnsw: partial sidecar ({} nodes, {} delta will be backfilled after WAL replay)",
                graph.len(), delta);
        } else {
            eprintln!("[hnsw] load_hnsw: loaded {} nodes from {:?}", graph.len(), path);
        }
        self.hnsw = graph;
        true
    }

    // ── Delta HNSW sidecar (.delta.hnsw) ────────────────────────────────────

    /// Save delta HNSW to sidecar. Removes the file if delta is empty (after merge).
    pub fn save_delta_hnsw(&self, path: &std::path::Path) -> std::io::Result<()> {
        if self.delta_hnsw.is_empty() {
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
        let bytes = bincode::serialize(&self.delta_hnsw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load delta HNSW from sidecar. Returns false if missing or corrupt.
    pub fn load_delta_hnsw(&mut self, path: &std::path::Path) -> bool {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let graph: HnswGraph = match bincode::deserialize(&bytes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[hnsw] load_delta_hnsw: deserialize failed: {}", e);
                return false;
            }
        };
        eprintln!("[hnsw] load_delta_hnsw: loaded {} delta nodes from {:?}", graph.len(), path);
        self.delta_hnsw = graph;
        true
    }

    // ── Per-realm HNSW sidecar (.realm_hnsw) ────────────────────────────────

    /// Populate per-realm tracking from the store's realm_members map.
    /// Call after snapshot load, before WAL replay.
    pub fn seed_realm_map(&mut self, realm_members: &HashMap<String, HashSet<MemoryId>>) {
        self.per_id_realm.clear();
        self.per_realm_counts.clear();
        for (realm, members) in realm_members {
            self.per_realm_counts.insert(realm.clone(), members.len());
            for &id in members {
                self.per_id_realm.insert(id, realm.clone());
            }
        }
    }

    pub fn save_realm_hnsw(&self, path: &std::path::Path) -> std::io::Result<()> {
        if self.per_realm_hnsw.is_empty() {
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
        let bytes = bincode::serialize(&self.per_realm_hnsw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let tmp = {
            let mut s = path.as_os_str().to_owned();
            s.push(".tmp");
            std::path::PathBuf::from(s)
        };
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load_realm_hnsw(&mut self, path: &std::path::Path) -> bool {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return false,
        };
        let graphs: HashMap<String, HnswGraph> = match bincode::deserialize(&bytes) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[hnsw] load_realm_hnsw: deserialize failed: {}", e);
                return false;
            }
        };
        eprintln!("[hnsw] load_realm_hnsw: loaded {} realm graphs", graphs.len());
        self.per_realm_hnsw = graphs;
        true
    }

    // ── Embedding sidecar (.emb) ─────────────────────────────────────────────
    // Format: [magic:u64][count:u64]([id:u64][f32×EMBED_DIM])×count
    // All values little-endian. Atomic write via .tmp rename.

    const EMB_MAGIC: u64 = 0x454D4244_00000003; // "EMBD\0\0\0\x03" — bumped for EMBED_DIM 768→1536 (ssl_distiller_dpo)

    pub fn save_embeddings_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        // When mmap is active the heap holds only the post-activation delta. Write a combined
        // file: heap delta first (freshest values win), then mmap bulk for ids not in the heap.
        //
        // Integrity invariants enforced HERE, at the persistence boundary, so no corrupt vector
        // can ever reach disk (a past migration wrote misaligned/duplicate records that poisoned
        // recall — see flat-scan path):
        //   * finite + bounded — every component finite and |v| <= EMB_SANE_BOUND; a normalized
        //     embedding has |v| <= 1, so anything larger is a misaligned/garbage record. Skipped.
        //   * de-duplicated — each id written at most once (heap takes precedence over mmap).
        //   * exact count — the u64 header is back-patched to the number actually written, never
        //     an upfront guess that a skip could falsify.
        use std::io::{Seek, SeekFrom};
        // Normalized vectors live in [-1, 1]; this bound only rejects corruption, never signal.
        const EMB_SANE_BOUND: f32 = 8.0;
        let sane = |v: &[f32]| -> bool {
            v.len() == EMBED_DIM && v.iter().all(|x| x.is_finite() && x.abs() <= EMB_SANE_BOUND)
        };
        let tmp = path.with_extension("emb.tmp");
        let mut written: u64 = 0;
        let mut seen: HashSet<MemoryId> = HashSet::with_capacity(self.total_embedding_count());
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            f.write_all(&Self::EMB_MAGIC.to_le_bytes())?;
            f.write_all(&0u64.to_le_bytes())?; // count placeholder — back-patched below
            // Delta: heap embeddings (freshest). Written first so they win on id collision.
            for (&id, emb) in &self.embeddings {
                if !sane(emb) || !seen.insert(id) {
                    continue;
                }
                f.write_all(&id.to_le_bytes())?;
                for &v in emb.iter().take(EMBED_DIM) {
                    f.write_all(&v.to_le_bytes())?;
                }
                written += 1;
            }
            // Bulk: raw f32 bytes from mmap, for ids not already written from the heap.
            if let Some(ref mm) = self.emb_mmap {
                for (&id, &off) in &self.emb_offsets {
                    if seen.contains(&id) {
                        continue;
                    }
                    let start = off as usize;
                    let end = start + EMBED_DIM * 4;
                    if end > mm.len() {
                        continue;
                    }
                    let vec: Vec<f32> = mm[start..end]
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    if !sane(&vec) {
                        continue; // drop misaligned/garbage record rather than persist it
                    }
                    seen.insert(id);
                    f.write_all(&id.to_le_bytes())?;
                    f.write_all(&mm[start..end])?;
                    written += 1;
                }
            }
            let mut f = f.into_inner()?;
            f.seek(SeekFrom::Start(8))?;
            f.write_all(&written.to_le_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load embeddings from sidecar. Returns false if missing/corrupt.
    /// On success, `self.embeddings` is fully populated from the sidecar.
    /// Uses mmap for zero-copy parsing; falls back to read() on NFS/SIGBUS risk.
    pub fn load_embeddings_sidecar(&mut self, path: &std::path::Path) -> bool {
        // Try mmap first (avoids double-buffering: OS page cache + heap Vec<u8>).
        // Fall back to read() if mmap fails (e.g. NFS with mmap disabled).
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mmap = unsafe { memmap2::Mmap::map(&file) }.ok();
        let fallback;
        let bytes: &[u8] = if let Some(ref m) = mmap {
            &m[..]
        } else {
            fallback = match std::fs::read(path) { Ok(b) => b, Err(_) => return false };
            &fallback
        };
        if bytes.len() < 16 {
            eprintln!("[hnsw] load_emb: sidecar too short");
            return false;
        }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != Self::EMB_MAGIC {
            eprintln!("[hnsw] load_emb: bad magic {:016x}", magic);
            return false;
        }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let record_size = 8 + EMBED_DIM * 4;
        if bytes.len() < 16 + count * record_size {
            eprintln!("[hnsw] load_emb: truncated ({} bytes for {} entries)", bytes.len(), count);
            return false;
        }
        let mut map = HashMap::with_capacity(count);
        let mut off = 16usize;
        for _ in 0..count {
            let id = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap());
            off += 8;
            let mut emb = vec![0f32; EMBED_DIM];
            for v in emb.iter_mut() {
                *v = f32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
                off += 4;
            }
            map.insert(id, emb);
        }
        self.invalidate_turbo();
        self.embeddings = map;
        self.startup_key = Some(Self::snapshot_vector_key(bytes));
        self.turbo_changed.clear();
        // Heal absent/old/corrupt caches from the actual snapshot before replay.
        // Otherwise a busy legacy family would miss forever until checkpoint.
        if count <= EMB_MMAP_MIN {
            let cache_path = path.with_extension("lsh");
            let key = self.startup_key.as_ref().unwrap();
            let cached: Option<HashMap<MemoryId, (f32, Vec<u16>)>> =
                crate::startup_cache::load(&cache_path, key);
            let valid = cached.as_ref().is_some_and(|m| m.len() == self.embeddings.len()
                && m.iter().all(|(id, (_, sigs))| self.embeddings.contains_key(id)
                    && sigs.len() == LSH_TABLES
                    && sigs.iter().all(|s| (*s as usize) < (1 << LSH_BITS))));
            if !valid {
                let _phase = crate::profile::LoadPhase::new("lsh_snapshot_cache");
                if self.lsh_planes.is_empty() { self.lsh_planes = default_lsh_planes(); }
                // Both fields are serde-skipped. Cache regeneration activates
                // projections before WAL replay, so its bucket array must exist
                // before replay upserts try to insert the new signatures.
                self.lsh_buckets.resize_with(LSH_TABLES, HashMap::new);
                let rows: HashMap<_, _> = refresh_pool().install(|| self.embeddings.par_iter()
                    .map(|(&id, v)| {
                        let norm = l2_norm(v);
                        let mut unit = v.clone();
                        if !(norm < 1e-9 || norm == 1.0) { for x in &mut unit { *x /= norm; } }
                        (id, (norm, self.assign_lsh(&unit)))
                    }).collect());
                if let Err(e) = crate::startup_cache::save(&cache_path, key, &rows) {
                    eprintln!("[chitta-field] optional snapshot LSH cache write failed: {e}");
                }
            }
        }
        eprintln!("[hnsw] load_emb: loaded {} embeddings from sidecar", count);
        true
    }

    pub fn embeddings_count(&self) -> usize {
        self.total_embedding_count()
    }

    /// Clear embeddings (called on snapshot clone before bincode serialization in v10+).
    pub fn clear_embeddings(&mut self) {
        self.embeddings = HashMap::new();
    }

    // ── Binary codes sidecar (.bin) ──────────────────────────────────────────
    // Format: [magic:u64][count:u64]([id:u64][u64×BINARY_WORDS])×count

    const MU_MAGIC: u64 = 0x4D550000_00000001; // "MU\0..\x01" — corpus-mean centroid sidecar

    /// Persist the corpus-mean centroid to the `.mu` sidecar (magic + dim + f32×dim).
    /// No-op (removes any stale file) when centering is disabled (empty centroid).
    pub fn save_centroid_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        if self.centroid.is_empty() {
            let _ = std::fs::remove_file(path);
            return Ok(());
        }
        let tmp = path.with_extension("mu.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&Self::MU_MAGIC.to_le_bytes())?;
            f.write_all(&(self.centroid.len() as u64).to_le_bytes())?;
            for &x in &self.centroid {
                f.write_all(&x.to_le_bytes())?;
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load the corpus-mean centroid from the `.mu` sidecar. Returns false if
    /// missing/corrupt/dim-mismatched (centering then stays disabled).
    pub fn load_centroid_sidecar(&mut self, path: &std::path::Path) -> bool {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return false,
        };
        if bytes.len() < 16 {
            return false;
        }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != Self::MU_MAGIC {
            return false;
        }
        let dim = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        if dim != EMBED_DIM || bytes.len() < 16 + dim * 4 {
            return false;
        }
        let mut mu = vec![0f32; dim];
        let mut off = 16usize;
        for x in mu.iter_mut() {
            *x = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            off += 4;
        }
        self.centroid = mu;
        eprintln!("[hnsw] load_mu: centroid loaded (|mu|={:.4}) — centering active", l2_norm(&self.centroid));
        true
    }

    const BIN_MAGIC: u64 = 0x42494E41_00000003; // "BINA\0\0\0\x03" — bumped for EMBED_DIM 768→1536 (ssl_distiller_dpo)

    pub fn save_binary_sidecar(&self, path: &std::path::Path) -> std::io::Result<()> {
        let tmp = path.with_extension("bin.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&Self::BIN_MAGIC.to_le_bytes())?;
            f.write_all(&(self.binary_codes.len() as u64).to_le_bytes())?;
            for (&id, codes) in &self.binary_codes {
                f.write_all(&id.to_le_bytes())?;
                for &w in codes.iter().take(BINARY_WORDS) {
                    f.write_all(&w.to_le_bytes())?;
                }
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load binary codes from sidecar. Returns false if missing/corrupt.
    pub fn load_binary_sidecar(&mut self, path: &std::path::Path) -> bool {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return false,
        };
        if bytes.len() < 16 {
            return false;
        }
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != Self::BIN_MAGIC {
            return false;
        }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let record_size = 8 + BINARY_WORDS * 8;
        if bytes.len() < 16 + count * record_size {
            eprintln!("[hnsw] load_bin: truncated ({} bytes for {} entries)", bytes.len(), count);
            return false;
        }
        let mut map = HashMap::with_capacity(count);
        let mut off = 16usize;
        for _ in 0..count {
            let id = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap());
            off += 8;
            let mut codes = vec![0u64; BINARY_WORDS];
            for w in codes.iter_mut() {
                *w = u64::from_le_bytes(bytes[off..off+8].try_into().unwrap());
                off += 8;
            }
            map.insert(id, codes);
        }
        self.binary_codes = map;
        self.binary_vec.clear();
        self.binary_vec_pos.clear();
        for (id, codes) in &self.binary_codes {
            let arr: [u64; BINARY_WORDS] = codes[..BINARY_WORDS].try_into().unwrap_or([0u64; BINARY_WORDS]);
            self.binary_vec_pos.insert(*id, self.binary_vec.len());
            self.binary_vec.push((*id, arr));
        }
        true
    }

    fn rebuild_ann(&mut self) {
        let hnsw_valid = (!self.hnsw.is_empty() || !self.delta_hnsw.is_empty())
            && (self.hnsw.len() + self.delta_hnsw.len() == self.total_embedding_count());
        self.coarse_members.clear();
        self.mem_coarse.clear();
        self.lsh_buckets = vec![HashMap::new(); LSH_TABLES];
        self.mem_lsh.clear();
        let assignments: Vec<(MemoryId, Vec<u16>)> = self
            .embeddings
            .iter()
            .map(|(&memory_id, embedding)| (memory_id, self.assign_coarse(embedding)))
            .collect();
        for (memory_id, coarse_ids) in assignments {
            for coarse_id in &coarse_ids {
                self.coarse_members
                    .entry(*coarse_id)
                    .or_default()
                    .push(memory_id);
            }
            self.mem_coarse.insert(memory_id, coarse_ids);
        }
        let lsh_assignments: Vec<(MemoryId, Vec<u16>)> = self
            .embeddings
            .iter()
            .map(|(&memory_id, embedding)| (memory_id, self.assign_lsh(embedding)))
            .collect();
        for (memory_id, signatures) in lsh_assignments {
            for (table_idx, signature) in signatures.iter().enumerate() {
                self.lsh_buckets[table_idx]
                    .entry(*signature)
                    .or_default()
                    .push(memory_id);
            }
            self.mem_lsh.insert(memory_id, signatures);
        }

        // Rebuild HNSW if collection is large enough.
        // Prefer incremental backfill when HNSW is already loaded and only a small WAL
        // delta is missing — avoids clearing the loaded sidecar and re-inserting 130K+ nodes.
        // Skipped while centering is active: centered queries never traverse the raw-space
        // HNSW graph (search() gates it off), so the graph would be dead weight — the
        // binary-Hamming path covers the full corpus.
        if !hnsw_valid && self.use_hnsw() && self.centroid.is_empty() {
            let total = self.total_embedding_count();
            let covered = self.hnsw_len();
            if covered > 0 && total.saturating_sub(covered) < 10_000 {
                self.backfill_hnsw_delta_parallel();
            } else {
                self.rebuild_hnsw();
            }
        }
    }

    fn rebuild_hnsw(&mut self) {
        self.hnsw = HnswGraph::new();
        self.delta_hnsw = HnswGraph::new();
        // Seed with first node so backfill_hnsw_delta_parallel doesn't exit early.
        let mut ids: Vec<MemoryId> = self.embeddings.keys()
            .chain(self.emb_offsets.keys())
            .copied()
            .collect();
        ids.sort_unstable();
        let Some(&first) = ids.first() else { return };
        {
            let embs = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
            if let Some(emb) = embs.get(first).map(|s| s.to_vec()) {
                let embs2 = EmbLookup { heap: &self.embeddings, offsets: &self.emb_offsets, mmap: &self.emb_mmap, arena: None };
                self.hnsw.insert(first, &emb, &embs2);
            }
        }
        // Parallel phase: remaining nodes computed against growing snapshots.
        self.backfill_hnsw_delta_parallel();
    }

    fn assign_coarse(&self, embedding: &[f32]) -> Vec<u16> {
        self.choose_probe_ids(embedding, COARSE_ASSIGNMENTS)
    }

    fn assign_lsh(&self, embedding: &[f32]) -> Vec<u16> {
        self.lsh_planes
            .iter()
            .map(|table| lsh_signature(table, embedding))
            .collect()
    }

    fn choose_probe_ids(&self, query: &[f32], desired: usize) -> Vec<u16> {
        let take = desired.min(self.coarse_centroids.len());
        if take == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(f32, u16)> = self
            .coarse_centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| (dot(query, centroid), idx as u16))
            .collect();
        let nth = take.saturating_sub(1);
        scored.select_nth_unstable_by(nth, |a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(take);
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.into_iter().map(|(_, id)| id).collect()
    }

    fn collect_candidates(
        &self,
        probes: &[u16],
        allowed: Option<&HashSet<MemoryId>>,
        k: usize,
    ) -> Vec<MemoryId> {
        let max_candidates = (k.max(1) * 64).max(MIN_CANDIDATES).min(MAX_CANDIDATES);
        let max_probes = probes.len().min(MAX_PROBES);
        let mut seen = HashSet::with_capacity(max_candidates.min(4096));
        let mut candidates = Vec::with_capacity(max_candidates.min(4096));

        for coarse_id in probes.iter().take(max_probes) {
            let Some(members) = self.coarse_members.get(coarse_id) else {
                continue;
            };
            let mut ordered = members.clone();
            ordered.sort_unstable();
            for memory_id in ordered {
                if self.deleted.contains(&memory_id) {
                    continue;
                }
                if let Some(allowed_set) = allowed {
                    if !allowed_set.contains(&memory_id) {
                        continue;
                    }
                }
                if seen.insert(memory_id) {
                    candidates.push(memory_id);
                    if candidates.len() >= max_candidates {
                        return candidates;
                    }
                }
            }
        }
        candidates
    }

    fn collect_lsh_candidates(
        &self,
        query: &[f32],
        allowed: Option<&HashSet<MemoryId>>,
        k: usize,
    ) -> Vec<MemoryId> {
        let max_candidates = (k.max(1) * 64)
            .max(MIN_CANDIDATES / 2)
            .min(MAX_CANDIDATES / 2);
        let signatures = self.assign_lsh(query);
        let mut seen = HashSet::with_capacity(max_candidates.min(4096));
        let mut candidates = Vec::with_capacity(max_candidates.min(4096));

        for (table_idx, signature) in signatures.iter().enumerate() {
            self.extend_from_bucket(
                table_idx,
                *signature,
                allowed,
                max_candidates,
                &mut seen,
                &mut candidates,
            );
            if candidates.len() >= max_candidates {
                return candidates;
            }
        }

        for bit in 0..LSH_BITS {
            for (table_idx, signature) in signatures.iter().enumerate() {
                self.extend_from_bucket(
                    table_idx,
                    *signature ^ (1u16 << bit),
                    allowed,
                    max_candidates,
                    &mut seen,
                    &mut candidates,
                );
                if candidates.len() >= max_candidates {
                    return candidates;
                }
            }
        }

        candidates
    }

    fn extend_from_bucket(
        &self,
        table_idx: usize,
        signature: u16,
        allowed: Option<&HashSet<MemoryId>>,
        max_candidates: usize,
        seen: &mut HashSet<MemoryId>,
        candidates: &mut Vec<MemoryId>,
    ) {
        let Some(members) = self.lsh_buckets[table_idx].get(&signature) else {
            return;
        };
        let mut ordered = members.clone();
        ordered.sort_unstable();
        for memory_id in ordered {
            if self.deleted.contains(&memory_id) {
                continue;
            }
            if let Some(allowed_set) = allowed {
                if !allowed_set.contains(&memory_id) {
                    continue;
                }
            }
            if seen.insert(memory_id) {
                candidates.push(memory_id);
                if candidates.len() >= max_candidates {
                    return;
                }
            }
        }
    }
}

// ── HNSW bulk-build helpers ───────────────────────────────────────────────────

/// Flat CSR view of locked_adj — built after each Phase B for Phase A beam search.
/// No HashMap, no MemoryId: all traversal uses u32 row indices into the flat arena.
struct BuildSnapshot {
    n_rows:      usize,
    entry_row:   u32,   // u32::MAX when no entry point yet
    entry_level: u8,
    /// offsets[row * maxl + layer .. +1] indexes into nbrs
    offsets:     Vec<u32>,
    nbrs:        Vec<u32>,
}

impl BuildSnapshot {
    fn from_locked_adj(
        locked_adj: &[parking_lot::Mutex<SmallVec<[u32; 32]>>],
        n_rows: usize,
        entry_row: u32,
        entry_level: u8,
        maxl: usize,
    ) -> Self {
        let n_cells = n_rows * maxl;
        let mut offsets = vec![0u32; n_cells + 1];
        for i in 0..n_cells {
            offsets[i + 1] = offsets[i] + locked_adj[i].lock().len() as u32;
        }
        let total = offsets[n_cells] as usize;
        let mut nbrs = vec![0u32; total];
        for i in 0..n_cells {
            let cell = locked_adj[i].lock();
            let start = offsets[i] as usize;
            nbrs[start..start + cell.len()].copy_from_slice(&cell);
        }
        BuildSnapshot { n_rows, entry_row, entry_level, offsets, nbrs }
    }

    #[inline]
    fn layer_neighbors(&self, row: u32, layer: usize, maxl: usize) -> &[u32] {
        let base = row as usize * maxl + layer;
        &self.nbrs[self.offsets[base] as usize..self.offsets[base + 1] as usize]
    }
}

/// Per-thread scratch for bulk beam search — generation-mark visited set, no per-search alloc.
struct Scratch {
    visited: Vec<u32>,
    gen:     u32,
}
impl Scratch {
    fn new(n_rows: usize) -> Self { Scratch { visited: vec![0u32; n_rows], gen: 1 } }
    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 { self.visited.fill(0); self.gen = 1; }
    }
    #[inline]
    fn visit(&mut self, row: u32) -> bool {
        let slot = &mut self.visited[row as usize];
        if *slot == self.gen { return false; }
        *slot = self.gen;
        true
    }
}

/// i8 × i8 dot product — preserves cosine ranking for unit-normalized embeddings.
/// Returns i32 (no f32 needed for comparison; compiler auto-vectorizes to SIMD).
#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
}

fn search_layer_rows(
    snap: &BuildSnapshot,
    query: &[i8],
    entry_rows: &[u32],
    ef: usize,
    layer: usize,
    arena: &[i8],
    scratch: &mut Scratch,
    maxl: usize,
) -> Vec<u32> {
    use std::cmp::Reverse;
    scratch.reset();
    // candidates: max-heap by sim (pop best for greedy expansion)
    let mut candidates: BinaryHeap<(i32, u32)> = BinaryHeap::new();
    // result: min-heap by sim via Reverse (pop worst for eviction when |result| > ef)
    let mut result: BinaryHeap<(Reverse<i32>, u32)> = BinaryHeap::new();
    for &row in entry_rows {
        if (row as usize) < snap.n_rows && scratch.visit(row) {
            let base = row as usize * EMBED_DIM;
            let sim = dot_i8(query, &arena[base..base + EMBED_DIM]);
            candidates.push((sim, row));
            result.push((Reverse(sim), row));
        }
    }
    while let Some((c_score, crow)) = candidates.pop() {
        if result.len() >= ef {
            // result.peek() = worst result (smallest sim = largest Reverse)
            if let Some(&(Reverse(worst_score), _)) = result.peek() {
                if c_score < worst_score { break; }
            }
        }
        for &nrow in snap.layer_neighbors(crow, layer, maxl) {
            if (nrow as usize) < snap.n_rows && scratch.visit(nrow) {
                let nb = nrow as usize * EMBED_DIM;
                let sim = dot_i8(query, &arena[nb..nb + EMBED_DIM]);
                candidates.push((sim, nrow));
                result.push((Reverse(sim), nrow));
                while result.len() > ef { result.pop(); }
            }
        }
    }
    result.into_iter().map(|(_, r)| r).collect()
}

fn compute_insert_plan_rows(
    snap: &BuildSnapshot,
    emb: &[i8],
    level: usize,
    arena: &[i8],
    scratch: &mut Scratch,
    maxl: usize,
    ef_bulk: usize,
) -> Vec<SmallVec<[u32; 32]>> {
    if snap.entry_row == u32::MAX {
        return vec![SmallVec::new(); level + 1];
    }
    let current_top = snap.entry_level as usize;
    let mut ep_set = vec![snap.entry_row];
    // Greedy descent from top layer to level+1 (ef=1 for speed)
    for layer in (level + 1..=current_top).rev() {
        let next = search_layer_rows(snap, emb, &ep_set, 1, layer, arena, scratch, maxl);
        if !next.is_empty() { ep_set = next; }
    }
    let mut plan: Vec<SmallVec<[u32; 32]>> = vec![SmallVec::new(); level + 1];
    for layer in (0..=level.min(current_top)).rev() {
        let candidates = search_layer_rows(snap, emb, &ep_set, ef_bulk, layer, arena, scratch, maxl);
        let m = if layer == 0 { HNSW_M0 } else { HNSW_M };
        let mut scored: Vec<(i32, u32)> = candidates.iter().map(|&r| {
            let base = r as usize * EMBED_DIM;
            (dot_i8(emb, &arena[base..base + EMBED_DIM]), r)
        }).collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(m);
        plan[layer] = scored.iter().map(|(_, r)| *r).collect();
        ep_set = candidates;
    }
    plan
}

fn default_coarse_centroids() -> Vec<Vec<f32>> {
    (0..COARSE_CENTROIDS)
        .map(|idx| random_unit_vec(EMBED_DIM, 10_000 + idx as u64))
        .collect()
}



fn default_lsh_planes() -> Vec<Vec<Vec<f32>>> {
    (0..LSH_TABLES)
        .map(|table| {
            (0..LSH_BITS)
                .map(|bit| random_unit_vec(EMBED_DIM, 20_000 + (table * LSH_BITS + bit) as u64))
                .collect()
        })
        .collect()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = l2_norm(v);
    if norm < 1e-9 {
        return None;
    }
    Some(v.iter().map(|x| *x / norm).collect())
}

fn normalize_in_place(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm < 1e-9 || norm == 1.0 {
        return;
    }
    for x in v {
        *x /= norm;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dot_n(a: &[f32], b: &[f32], n: usize) -> f32 {
    a.iter().zip(b.iter()).take(n).map(|(x, y)| x * y).sum()
}

/// Sign-bit code of `v - centroid` (mean-centered). Falls back to `binarize(v)` when no
/// centroid is set. Sign bits are scale-invariant, so the difference is binarized directly
/// without renormalizing — matching `center_norm`, which only differs by a positive scale.
fn binarize_centered(v: &[f32], centroid: &[f32]) -> Vec<u64> {
    if centroid.len() == v.len() {
        let mut c = vec![0f32; v.len()];
        for i in 0..v.len() {
            c[i] = v[i] - centroid[i];
        }
        binarize(&c)
    } else {
        binarize(v)
    }
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
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.memory_id.cmp(&other.memory_id))
    }
}

impl PartialOrd for RankedHit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn push_top_k(heap: &mut BinaryHeap<RankedHit>, k: usize, memory_id: MemoryId, score: f32) {
    if k == 0 {
        return;
    }
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

fn heap_to_hits(heap: BinaryHeap<RankedHit>) -> Vec<SemanticHit> {
    let mut ranked: Vec<RankedHit> = heap.into_vec();
    ranked.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.memory_id.cmp(&b.memory_id))
    });
    ranked
        .into_iter()
        .map(|hit| SemanticHit {
            memory_id: hit.memory_id,
            cosine_similarity: hit.score,
        })
        .collect()
}

fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;
            let r = (-2.0 * (u1 + 1e-10).ln()).sqrt();
            r * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    normalize_in_place(&mut v);
    v
}

fn lsh_signature(planes: &[Vec<f32>], embedding: &[f32]) -> u16 {
    let mut sig = 0u16;
    for (bit, plane) in planes.iter().enumerate() {
        if dot(embedding, plane) >= 0.0 {
            sig |= 1u16 << bit;
        }
    }
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_returns_nearest() {
        let mut idx = SemanticIndex::new();
        let mut e1 = vec![0.0f32; EMBED_DIM];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; EMBED_DIM];
        e2[1] = 1.0;
        let mut e3 = vec![0.0f32; EMBED_DIM];
        e3[0] = 0.9;
        e3[1] = 0.1;
        idx.upsert(1, e1.clone(), None);
        idx.upsert(2, e2, None);
        idx.upsert(3, e3, None);

        let hits = idx.search(&e1, 2, None, None);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].memory_id, 1); // exact match
                                          // e3 has a larger component on dim 0 (same as query) than e2
        assert_eq!(hits[1].memory_id, 3);
    }

    #[test]
    fn test_deleted_not_returned() {
        let mut idx = SemanticIndex::new();
        let e = vec![1.0f32; EMBED_DIM];
        idx.upsert(1, e.clone(), None);
        idx.upsert(2, e.clone(), None);
        idx.remove(1);
        let hits = idx.search(&e, 10, None, None);
        assert!(hits.iter().all(|h| h.memory_id != 1));
    }

    #[test]
    fn test_allowed_filter() {
        let mut idx = SemanticIndex::new();
        let e = vec![1.0f32; EMBED_DIM];
        idx.upsert(1, e.clone(), None);
        idx.upsert(2, e.clone(), None);
        idx.upsert(3, e.clone(), None);

        let allowed: HashSet<MemoryId> = [2u64, 3u64].into_iter().collect();
        let hits = idx.search(&e, 10, Some(&allowed), None);
        assert!(hits.iter().all(|h| h.memory_id != 1));
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn test_wrong_dim_returns_empty() {
        let idx = SemanticIndex::new();
        let hits = idx.search(&[1.0f32; 64], 5, None, None);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_upsert_overwrites() {
        let mut idx = SemanticIndex::new();
        let mut e1 = vec![0.0f32; EMBED_DIM];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; EMBED_DIM];
        e2[1] = 1.0;

        idx.upsert(1, e1, None);
        idx.upsert(1, e2.clone(), None); // overwrite

        // Query aligned with dim-1; id=1 should now match well
        let hits = idx.search(&e2, 1, None, None);
        assert_eq!(hits[0].memory_id, 1);
    }

    #[test]
    fn test_candidate_search() {
        let mut idx = SemanticIndex::new();
        let mut e1 = vec![0.0f32; EMBED_DIM];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; EMBED_DIM];
        e2[1] = 1.0;
        idx.upsert(1, e1.clone(), None);
        idx.upsert(2, e2.clone(), None);

        let hits = idx.search_candidates(&e1, [2, 1], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, 1);
        assert!(hits[0].cosine_similarity > 0.99);
    }
}

/// Owns all build inputs; no SemanticIndex/store lock is held during quantization.
pub(crate) struct TurboBuild {
    flat: Vec<f32>,
    ids: Vec<MemoryId>,
    mutation: u64,
    published: Arc<RwLock<Option<Arc<TurboState>>>>,
    epoch: u64,
    current_epoch: Arc<std::sync::atomic::AtomicU64>,
}

impl TurboBuild {
    pub(crate) fn build(self) {
        let _phase = crate::profile::LoadPhase::new("turbo");
        eprintln!("[chitta-field] Turbo rebuild start rows={} mutation={}", self.ids.len(), self.mutation);
        let Ok(mut index) = TurboQuantIndex::new(EMBED_DIM, 4) else { return; };
        if self.ids.is_empty() { return; }
        refresh_pool().install(|| {
            index.add(&self.flat);
            index.prepare();
        });
        // Initialize worker threads and any per-thread search scratch before
        // publication. Use a corpus vector, never a benchmark-specific query;
        // this reads the index without updating competitive weights or scores.
        refresh_pool().broadcast(|_| {
            let _ = index.search(&self.flat[..EMBED_DIM], 9.min(self.ids.len()));
        });
        let built = Arc::new(TurboState { index: crate::turbo::SearchIndex::Native(index), ids: self.ids,
            built_at_mutation: self.mutation, built_at: std::time::Instant::now(), startup_delta: false });
        let mut guard = self.published.write();
        // Protect against an older concurrent maintenance plan winning the swap.
        if self.current_epoch.load(Ordering::Relaxed) == self.epoch
            && guard.as_ref().is_none_or(|t| t.built_at_mutation <= self.mutation) {
            *guard = Some(built);
        }
    }
}

#[cfg(test)]
mod turbo_maintenance_tests {
    use super::*;

    fn index() -> SemanticIndex {
        let mut idx = SemanticIndex::new();
        let mut v = vec![0.0; EMBED_DIM]; v[0] = 1.0;
        // Direct population avoids testing HNSW construction here.
        for id in 1..=2001 { idx.embeddings.insert(id, v.clone()); }
        idx
    }

    #[test]
    fn search_never_builds_and_stale_index_keeps_new_updated_deleted_vectors_visible() {
        let mut idx = index();
        let mut q = vec![0.0; EMBED_DIM]; q[1] = 1.0;
        assert!(!idx.search(&q, 3, None, None).is_empty());
        assert!(idx.turbo.read().is_none(), "search must not build");
        idx.plan_turbo_rebuild(64).unwrap().build();
        let old = idx.turbo.read().clone().unwrap();
        assert!(idx.plan_turbo_rebuild(64).is_none());
        let mut original = vec![0.0; EMBED_DIM]; original[0] = 1.0;
        let leading = idx.search(&original, 1, None, None)[0].memory_id;
        let mut negative = original.clone(); negative[0] = -1.0;
        idx.upsert_meta(leading, negative, None);
        let replacements = idx.search(&original, 3, None, None);
        assert_eq!(replacements.len(), 3);
        assert!(!replacements.iter().any(|h| h.memory_id == leading));
        idx.upsert_meta(3000, q.clone(), None);
        assert_eq!(idx.search(&q, 1, None, None)[0].memory_id, 3000);
        assert!(Arc::ptr_eq(idx.turbo.read().as_ref().unwrap(), &old));
        let plan = idx.plan_turbo_rebuild(0).unwrap();
        idx.upsert_meta(3001, q.clone(), None); // raced with the build snapshot
        plan.build();
        idx.prune_turbo_changes();
        idx.remove(3000);
        let hits = idx.search(&q, 3, None, None);
        assert_eq!(hits[0].memory_id, 3001);
        assert_eq!(hits.len(), 3);
        assert!(!hits.iter().any(|h| h.memory_id == 3000));
        let mut opposite = q.clone(); opposite[1] = -1.0;
        idx.upsert_meta(3001, opposite, None);
        assert_ne!(idx.search(&q, 1, None, None)[0].memory_id, 3001);
    }

    #[test]
    fn parallel_refresh_matches_serial_search_and_rejects_dirty_views() {
        let mut idx = index();
        for (id, vector) in &mut idx.embeddings {
            vector.fill(0.0);
            vector[*id as usize % EMBED_DIM] = 1.0;
            vector[(*id as usize + 17) % EMBED_DIM] = 0.3;
            *vector = normalize(vector).unwrap();
        }
        idx.deleted.insert(2);
        idx.plan_turbo_rebuild(0).unwrap().build();
        let candidates: Vec<_> = (1..=16).map(|id| (id, idx.get_embedding(id).unwrap().to_vec())).collect();
        let view = idx.plan_refresh_searches(&candidates, None).unwrap();
        let actual = view.search_all(&candidates);
        for ((_, query), hits) in candidates.iter().zip(actual) {
            let expected = idx.search(query, 9, None, None);
            assert_eq!(hits.iter().map(|h| (h.memory_id, h.cosine_similarity.to_bits())).collect::<Vec<_>>(),
                expected.iter().map(|h| (h.memory_id, h.cosine_similarity.to_bits())).collect::<Vec<_>>());
        }
        idx.upsert_meta(3000, candidates[0].1.clone(), None);
        assert!(idx.plan_refresh_searches(&candidates, None).is_none());
        assert!(!view.search_all(&candidates)[0].is_empty(), "prior view remains valid");
    }

    #[test]
    fn unchanged_turbo_does_not_rebuild_after_the_age_limit() {
        let idx = index();
        idx.plan_turbo_rebuild(0).unwrap().build();
        {
            let mut slot = idx.turbo.write();
            Arc::get_mut(slot.as_mut().unwrap()).unwrap().built_at =
                std::time::Instant::now() - std::time::Duration::from_secs(120);
        }
        assert!(idx.plan_turbo_rebuild(0).is_none(), "age alone cannot invalidate unchanged vectors");
    }

    #[test]
    fn maintenance_swap_keeps_prior_readers_alive_and_discards_invalidated_plans() {
        let idx = index();
        idx.plan_turbo_rebuild(64).unwrap().build();
        let old = idx.turbo.read().clone().unwrap();
        idx.invalidate_turbo();
        let plan = idx.plan_turbo_rebuild(64).unwrap();
        idx.invalidate_turbo();
        plan.build();
        assert!(idx.turbo.read().is_none());
        assert_eq!(old.ids.len(), 2001, "existing readers still own their index");
    }
}

#[cfg(test)]
mod startup_cache_tests {
    use super::*;

    fn raw_index(n: u64) -> SemanticIndex {
        let mut idx = SemanticIndex::new();
        for id in 0..n {
            let mut v = vec![0.0; EMBED_DIM];
            v[id as usize % EMBED_DIM] = 0.731;
            v[(id as usize + 1) % EMBED_DIM] = 0.413;
            idx.embeddings.insert(id, v);
        }
        idx
    }

    fn load_raw(raw: &SemanticIndex, path: &std::path::Path) -> SemanticIndex {
        raw.save_embeddings_sidecar(&path.with_extension("emb")).unwrap();
        let mut loaded = SemanticIndex::new();
        assert!(loaded.load_embeddings_sidecar(&path.with_extension("emb")));
        loaded
    }

    #[test]
    fn deserialized_index_with_missing_lsh_cache_accepts_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.snapshot");
        let raw = raw_index(2001);
        raw.save_embeddings_sidecar(&path.with_extension("emb")).unwrap();
        let mut replayed: SemanticIndex = bincode::deserialize(&bincode::serialize(&raw).unwrap()).unwrap();
        assert!(replayed.lsh_planes.is_empty() && replayed.lsh_buckets.is_empty());
        assert!(replayed.load_embeddings_sidecar(&path.with_extension("emb")));
        replayed.set_inhibit_hnsw(true);
        replayed.upsert(3000, vec![0.1; EMBED_DIM], None);
        replayed.normalize_with_cache(Some(&path.with_extension("lsh")));
        assert!(replayed.contains(3000));
        assert_eq!(replayed.mem_lsh[&3000].len(), LSH_TABLES);
        for (table, signature) in replayed.mem_lsh[&3000].iter().enumerate() {
            assert!(replayed.lsh_buckets[table][signature].contains(&3000));
        }
    }

    #[test]
    fn lsh_cache_plus_replayed_delta_matches_full_rebuild_bitwise() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.snapshot");
        let mut base = load_raw(&raw_index(2001), &path);
        base.normalize_with_cache(Some(&path.with_extension("lsh")));
        let mut replayed = SemanticIndex::new();
        assert!(replayed.load_embeddings_sidecar(&path.with_extension("emb")));
        replayed.set_inhibit_hnsw(true);
        for id in 1900..2201 {
            let mut v = raw_index(1).embeddings.remove(&0).unwrap();
            v[id as usize % EMBED_DIM] = 0.827;
            replayed.upsert(id, v, None);
        }
        replayed.remove(17);
        let mut full = replayed.clone();
        full.normalize_all();
        replayed.normalize_with_cache(Some(&path.with_extension("lsh")));
        assert_eq!(full.mem_lsh, replayed.mem_lsh);
        for (&id, vector) in &full.embeddings {
            assert_eq!(vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                replayed.embeddings[&id].iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let mut changed_raw = raw_index(2001);
        changed_raw.embeddings.get_mut(&0).unwrap()[10] = 0.82;
        let mut changed = load_raw(&changed_raw, &path);
        assert_ne!(base.startup_key, changed.startup_key);
        changed.normalize_with_cache(Some(&path.with_extension("lsh")));
        assert_eq!(changed.mem_lsh[&0], changed.assign_lsh(&changed.embeddings[&0]));
        std::fs::write(path.with_extension("lsh"), b"truncated").unwrap();
        let mut rebuilt = load_raw(&raw_index(2001), &path);
        rebuilt.normalize_with_cache(Some(&path.with_extension("lsh")));
        assert_eq!(base.mem_lsh, rebuilt.mem_lsh);
    }

    fn turbo_rows(index: &TurboState, path: &std::path::Path) -> (HashMap<MemoryId, Vec<u8>>, Vec<u8>) {
        index.index.write(path).unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..6], b"TVPI\x03\x04");
        let row_bytes = EMBED_DIM / 2;
        let scale_offset = 14 + index.ids.len() * row_bytes;
        let rows = index.ids.iter().enumerate().map(|(row, &id)| {
            let mut value = bytes[14 + row * row_bytes..14 + (row + 1) * row_bytes].to_vec();
            value.extend_from_slice(&bytes[scale_offset + row * 4..scale_offset + (row + 1) * 4]);
            (id, value)
        }).collect();
        (rows, bytes[scale_offset + index.ids.len() * 4..].to_vec())
    }

    #[test]
    fn turbo_replayed_delta_preserves_full_rebuild_rows_and_calibration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.snapshot");
        let mut base = load_raw(&raw_index(2001), &path);
        base.normalize_with_cache(Some(&path.with_extension("lsh")));
        base.warm_turbo_with_cache(Some(&path));
        let mut replayed = SemanticIndex::new();
        assert!(replayed.load_embeddings_sidecar(&path.with_extension("emb")));
        replayed.set_inhibit_hnsw(true);
        for id in 1900..2201 {
            let mut v = vec![0.013; EMBED_DIM];
            v[id as usize % EMBED_DIM] = 0.827;
            replayed.upsert(id, v, None);
        }
        replayed.remove(17);
        replayed.normalize_with_cache(Some(&path.with_extension("lsh")));
        let mut full = replayed.clone();
        full.turbo = Arc::new(RwLock::new(None));
        full.warm_turbo_with_cache(None);
        replayed.warm_turbo_with_cache(Some(&path));
        let snapshot = replayed.turbo.read().clone().unwrap();
        assert!(snapshot.startup_delta);
        assert_eq!(snapshot.ids, base.turbo.read().as_ref().unwrap().ids);
        // Even an arbitrarily high mutation threshold cannot delay startup work.
        let plan = replayed.plan_turbo_rebuild(u64::MAX).unwrap();
        let query = replayed.get_embedding(2200).unwrap().to_vec();
        let hits = replayed.search(&query, 10, None, None);
        assert!(hits.iter().any(|h| h.memory_id == 2200));
        assert!(!hits.iter().any(|h| h.memory_id == 17));
        // Search can complete while the maintenance plan is outstanding.
        assert!(Arc::ptr_eq(&snapshot, replayed.turbo.read().as_ref().unwrap()));
        plan.build();
        replayed.prune_turbo_changes();
        assert!(replayed.plan_turbo_rebuild(u64::MAX).is_none());
        let expected = turbo_rows(full.turbo.read().as_ref().unwrap(), &dir.path().join("full.tv"));
        assert_eq!(expected,
            turbo_rows(replayed.turbo.read().as_ref().unwrap(), &dir.path().join("cached.tv")));
        let snapshot_rows = turbo_rows(base.turbo.read().as_ref().unwrap(), &dir.path().join("base.tv"));
        assert_ne!(snapshot_rows.1, expected.1,
            "TQ+ calibration changes with the corpus: blindly appending cannot preserve rebuild bits");
    }

    #[test]
    fn turbo_cache_preserves_rows_scores_and_rejects_changed_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.snapshot");
        let first = load_raw(&raw_index(2001), &path);
        first.warm_turbo_with_cache(Some(&path));
        let mut second = SemanticIndex::new();
        assert!(second.load_embeddings_sidecar(&path.with_extension("emb")));
        second.warm_turbo_with_cache(Some(&path));
        let a = first.turbo.read(); let b = second.turbo.read();
        let a = a.as_ref().unwrap(); let b = b.as_ref().unwrap();
        assert_eq!(a.ids, b.ids);
        let q = &first.embeddings[&11];
        let x = a.index.search(q, 10); let y = b.index.search(q, 10);
        assert_eq!(x.indices_for_query(0), y.indices_for_query(0));
        assert_eq!(x.scores_for_query(0).iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                   y.scores_for_query(0).iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let mut changed = raw_index(2001);
        changed.embeddings.get_mut(&0).unwrap()[7] = 0.82;
        let changed = load_raw(&changed, &path);
        assert_ne!(first.startup_key, changed.startup_key);
        changed.warm_turbo_with_cache(Some(&path));
        assert!(changed.turbo.read().is_some());
    }
}
