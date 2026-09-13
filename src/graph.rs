use crate::organ::triplet::TripletStore;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

impl Direction {
    pub fn from_str(s: &str) -> Self {
        match s {
            "incoming" => Direction::Incoming,
            "both" => Direction::Both,
            _ => Direction::Outgoing,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TraversalHit {
    pub node: String,
    pub hops: u8,
    pub edge_weight: f32,
    pub path_triplet_ids: Vec<u64>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl TripletStore {
    /// BFS traversal over the triplet graph from `start`.
    ///
    /// Follows edges whose predicate matches `edge_types` (all if empty).
    /// Skips tombstoned and superseded entries.
    /// Returns up to `max_results` hits ordered by BFS level.
    pub fn graph_traverse(
        &self,
        start: &str,
        edge_types: &[&str],
        max_hops: usize,
        max_results: usize,
        direction: Direction,
        max_edges: usize,
    ) -> Vec<TraversalHit> {
        if max_results == 0 || max_edges == 0 { return Vec::new(); }
        let ts = now_ms();
        let mut results: Vec<TraversalHit> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        // (node, hops, cumulative_weight, path)
        let mut queue: VecDeque<(String, u8, f32, Vec<u64>)> = VecDeque::new();

        visited.insert(start.to_string());
        queue.push_back((start.to_string(), 0, 1.0, Vec::new()));
        let mut edges_inspected = 0usize;

        'traversal: while let Some((node, hops, weight, path)) = queue.pop_front() {
            if hops > 0 {
                results.push(TraversalHit {
                    node: node.clone(),
                    hops,
                    edge_weight: weight,
                    path_triplet_ids: path.clone(),
                });
                if results.len() >= max_results {
                    break;
                }
            }
            if hops as usize >= max_hops {
                continue;
            }
            let next_hops = hops + 1;

            if matches!(direction, Direction::Outgoing | Direction::Both) {
                if let Some(ids) = self.subject_ids(&node) {
                    for &id in ids {
                        if edges_inspected >= max_edges { break 'traversal; }
                        edges_inspected += 1;
                        if let Some(e) = self.entry_by_id_crate(id) {
                            if e.valid_to_ms != 0 && ts >= e.valid_to_ms { continue; }
                            if self.is_superseded_crate(id) { continue; }
                            if !edge_types.is_empty() && !edge_types.contains(&e.predicate.as_str()) { continue; }
                            if visited.insert(e.object.clone()) {
                                let mut p = path.clone();
                                p.push(id);
                                queue.push_back((e.object.clone(), next_hops, weight * e.weight, p));
                            }
                        }
                    }
                }
            }

            if matches!(direction, Direction::Incoming | Direction::Both) {
                if let Some(ids) = self.object_ids(&node) {
                    for &id in ids {
                        if edges_inspected >= max_edges { break 'traversal; }
                        edges_inspected += 1;
                        if let Some(e) = self.entry_by_id_crate(id) {
                            if e.valid_to_ms != 0 && ts >= e.valid_to_ms { continue; }
                            if self.is_superseded_crate(id) { continue; }
                            if !edge_types.is_empty() && !edge_types.contains(&e.predicate.as_str()) { continue; }
                            if visited.insert(e.subject.clone()) {
                                let mut p = path.clone();
                                p.push(id);
                                queue.push_back((e.subject.clone(), next_hops, weight * e.reverse_weight(), p));
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Personalized PageRank over the triplet graph.
    ///
    /// Seeds get constant restart probability. Scores propagate via outgoing edges
    /// weighted by `weight`. Returns top_k nodes by score descending.
    pub fn graph_pagerank(
        &self,
        seeds: &[&str],
        edge_types: &[&str],
        damping: f32,
        iterations: u8,
        top_k: usize,
        max_nodes: usize,
        max_edges: usize,
    ) -> Vec<(String, f32)> {
        let ts = now_ms();
        if seeds.is_empty() || top_k == 0 || max_nodes == 0 || max_edges == 0 {
            return Vec::new();
        }
        let bounded_seeds: Vec<&str> = seeds.iter().copied().take(max_nodes).collect();
        let seed_boost = (1.0 - damping) / bounded_seeds.len() as f32;
        let mut scores: HashMap<String, f32> = HashMap::new();
        for s in &bounded_seeds {
            scores.insert(s.to_string(), seed_boost);
        }
        let mut edges_inspected = 0usize;

        for _ in 0..iterations {
            let mut next: HashMap<String, f32> = HashMap::new();
            for s in &bounded_seeds {
                *next.entry(s.to_string()).or_insert(0.0) += seed_boost;
            }
            for (node, &score) in &scores {
                if edges_inspected >= max_edges { break; }
                if let Some(ids) = self.subject_ids(node) {
                    let remaining = max_edges - edges_inspected;
                    let inspected_here = ids.len().min(remaining);
                    edges_inspected += inspected_here;
                    let edges: Vec<_> = ids.iter().take(inspected_here)
                        .filter_map(|&id| self.entry_by_id_crate(id))
                        .filter(|e| {
                            (e.valid_to_ms == 0 || ts < e.valid_to_ms)
                                && !self.is_superseded_crate(e.id)
                                && (edge_types.is_empty() || edge_types.contains(&e.predicate.as_str()))
                        })
                        .collect();
                    let total: f32 = edges.iter().map(|e| e.weight).sum();
                    if total <= 0.0 { continue; }
                    for e in edges {
                        if next.contains_key(&e.object) || next.len() < max_nodes {
                            *next.entry(e.object.clone()).or_insert(0.0) +=
                                damping * score * (e.weight / total);
                        }
                    }
                }
            }
            let sum: f32 = next.values().sum();
            if sum > 0.0 { for v in next.values_mut() { *v /= sum; } }
            scores = next;
            if edges_inspected >= max_edges { break; }
        }

        let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }
}
