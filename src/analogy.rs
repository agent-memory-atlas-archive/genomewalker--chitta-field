//! Explicit directed relation transfer over indexed triplets; no VSA ranking.
use crate::ids::MemoryId;
use crate::organ::triplet::{TripletEntry, TripletStore};
use std::collections::{BTreeMap, HashSet};

/// Compatibility type for the separate legacy store snapshot API.
#[derive(Clone, Debug)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub memory_id: Option<MemoryId>,
}

#[derive(Debug)]
pub struct RelationTransfer {
    pub relations: Vec<TripletEntry>,
    pub results: Vec<Vec<TripletEntry>>,
    pub reason: Option<&'static str>,
    /// Total organ entries, diagnostic only; never a serving bound.
    pub indexed: usize,
}

/// Copy only the two indexed subject neighbourhoods at a single read epoch.
/// Exact symbols and forward edges match query_graph semantics. All predicates
/// explicitly linking a→b transfer; no fuzzy or reverse relation is inferred.
/// Validity includes the start time (the legacy query API only checks expiry).
pub fn proportional(
    store: &TripletStore, a: &str, b: &str, c: &str, at_ms: i64,
) -> RelationTransfer {
    let relations: Vec<_> = store.query_subject(a, at_ms).into_iter()
        .filter(|e| e.subject == a && e.object == b && e.valid_from_ms <= at_ms)
        .cloned().collect();
    let predicates: HashSet<_> = relations.iter().map(|e| e.predicate.as_str()).collect();
    let edges = if predicates.is_empty() { Vec::new() } else {
        store.query_subject(c, at_ms).into_iter()
            .filter(|e| e.subject == c && predicates.contains(e.predicate.as_str()) && e.valid_from_ms <= at_ms)
            .cloned().collect()
    };
    let reason = if relations.is_empty() { Some("no_source_relation") }
        else if edges.is_empty() { Some("no_target_relation") } else { None };
    let mut transfer = RelationTransfer {
        relations, results: Vec::new(), reason, indexed: store.triplet_count(),
    };
    // Sorting/grouping can also be called after releasing the store guard.
    transfer.results = edges.into_iter().map(|e| vec![e]).collect();
    transfer
}

fn edge_order(a: &TripletEntry, b: &TripletEntry) -> std::cmp::Ordering {
    // Corrupt/non-finite weights sort last and are never exported as NaN.
    let weight = |e: &TripletEntry| if e.weight.is_finite() { e.weight } else { f32::NEG_INFINITY };
    weight(b).total_cmp(&weight(a))
        .then_with(|| b.valid_from_ms.cmp(&a.valid_from_ms))
        .then_with(|| a.object.cmp(&b.object))
        .then_with(|| a.predicate.cmp(&b.predicate))
        .then_with(|| a.source_memory_id.cmp(&b.source_memory_id))
        .then_with(|| a.id.cmp(&b.id))
}

impl RelationTransfer {
    /// Rank outside the organ lock, keeping all citations for each neighbour.
    pub fn rank(&mut self, limit: usize) {
        self.relations.sort_by(edge_order);
        let mut grouped: BTreeMap<String, Vec<TripletEntry>> = BTreeMap::new();
        for edge in self.results.drain(..).flatten() {
            grouped.entry(edge.object.clone()).or_default().push(edge);
        }
        self.results = grouped.into_values().collect();
        for edges in &mut self.results { edges.sort_by(edge_order); }
        self.results.sort_by(|a, b| edge_order(&a[0], &b[0]));
        self.results.truncate(limit.max(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn add(s: &mut TripletStore, a: &str, p: &str, b: &str, w: f32, time: i64) -> u64 {
        s.add(a.into(), p.into(), b.into(), w, time, Some(42), None)
    }
    fn solve(s: &TripletStore, a: &str, b: &str, c: &str, limit: usize) -> RelationTransfer {
        let mut r = proportional(s, a, b, c, 100);
        r.rank(limit);
        r
    }
    #[test]
    fn directed_support_only_and_explicit_abstention() {
        let mut s = TripletStore::new();
        add(&mut s, "a", "uses", "b", 1.0, 1);
        add(&mut s, "c", "other", "noise", 1.0, 1);
        add(&mut s, "unrelated", "uses", "candidate", 1.0, 1);
        add(&mut s, "reverse", "uses", "c", 1.0, 1);
        assert_eq!(solve(&s, "a", "b", "c", 3).reason, Some("no_target_relation"));
        assert_eq!(solve(&s, "b", "a", "c", 3).reason, Some("no_source_relation"));
        add(&mut s, "c", "uses", "b", 0.8, 1);
        let r = solve(&s, "a", "b", "c", 3);
        assert_eq!(r.results.len(), 1);
        assert_eq!(r.results[0][0].object, "b"); // b is a valid answer too.
        assert_eq!(r.results[0][0].source_memory_id, Some(42));
        assert!(r.reason.is_none());
        assert_eq!(solve(&s, "A", "b", "c", 3).reason, Some("no_source_relation"));
    }
    #[test]
    fn replay_id_collisions_cannot_supply_false_source_or_target_edges() {
        let mut s = TripletStore::new();
        let source = add(&mut s, "a", "p", "b", 1.0, 1);
        let target = add(&mut s, "c", "p", "d", 1.0, 1);
        s.replay_add(target, "other".into(), "p".into(), "false".into(), 1.0, 1, None, None);
        assert_eq!(solve(&s, "a", "b", "c", 3).reason, Some("no_target_relation"));
        s.replay_add(source, "other".into(), "p".into(), "b".into(), 1.0, 1, None, None);
        assert_eq!(solve(&s, "a", "b", "other", 3).reason, Some("no_source_relation"));
    }
    #[test]
    fn complete_beyond_old_bounds_and_invalid_prefix() {
        let mut s = TripletStore::new();
        for i in 0..10_100 {
            let id = add(&mut s, "c", "irrelevant", &format!("noise{i}"), 1.0, 1);
            s.invalidate(id, 2);
        }
        add(&mut s, "a", "zzz", "b", 1.0, 1);
        add(&mut s, "c", "zzz", "answer", 0.9, 3);
        assert_eq!(solve(&s, "a", "b", "c", 1).results[0][0].object, "answer");
    }
    #[test]
    fn all_predicates_dedup_weight_recency_and_expiry() {
        let mut s = TripletStore::new();
        for p in ["p", "q"] { add(&mut s, "a", p, "b", 1.0, 1); }
        add(&mut s, "c", "p", "old", 0.9, 1);
        add(&mut s, "c", "q", "new", 0.9, 2);
        add(&mut s, "c", "p", "new", 0.8, 3);
        add(&mut s, "c", "p", "weak", 0.5, 99);
        add(&mut s, "c", "p", "future", 1.0, 101);
        let expired = add(&mut s, "c", "p", "expired", 1.0, 1);
        s.invalidate(expired, 99);
        let r = solve(&s, "a", "b", "c", 3);
        assert_eq!(r.relations.len(), 2);
        assert_eq!(r.results.iter().map(|e| e[0].object.as_str()).collect::<Vec<_>>(), vec!["new", "old", "weak"]);
        assert_eq!(r.results[0].len(), 2);
        assert_eq!(solve(&s, "a", "b", "c", 1).results.len(), 1);
        s.invalidate(r.relations[0].id, 99);
        s.invalidate(r.relations[1].id, 99);
        assert_eq!(solve(&s, "a", "b", "c", 3).reason, Some("no_source_relation"));
    }
}
