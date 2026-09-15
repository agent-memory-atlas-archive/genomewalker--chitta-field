use super::cortex::SparseCode;
use crate::ids::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type ProtoId = u32;

pub const VIGILANCE: f32 = 0.003;
pub const MAX_PROTOTYPES: usize = 2048;

#[derive(Serialize, Deserialize)]
pub struct PrototypeEntry {
    pub id: ProtoId,
    pub centroid: SparseCode,
    pub count: u32,
}

#[derive(Serialize, Deserialize)]
pub struct PrototypeIndex {
    protos: Vec<PrototypeEntry>,
    mem_to_proto: HashMap<MemoryId, ProtoId>,
    transitions: HashMap<(ProtoId, ProtoId), f32>,
    vigilance: f32,
    next_id: ProtoId,
}

impl Default for PrototypeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PrototypeIndex {
    pub fn new() -> Self {
        Self {
            protos: Vec::new(),
            mem_to_proto: HashMap::new(),
            transitions: HashMap::new(),
            vigilance: VIGILANCE,
            next_id: 0,
        }
    }

    /// Assign a memory to a prototype. Returns the ProtoId.
    pub fn assign(&mut self, memory_id: MemoryId, code: &SparseCode) -> ProtoId {
        if code.is_empty() {
            // No code to assign; create an empty prototype as fallback only if none exist
            if self.protos.is_empty() {
                return self.create_proto(code);
            }
            // Just assign to the first (best we can do with empty code)
            let pid = self.protos[0].id;
            self.mem_to_proto.insert(memory_id, pid);
            return pid;
        }

        let proto_id = if self.protos.is_empty() {
            // No prototypes yet — always create first one
            self.create_proto(code)
        } else {
            // Find best matching prototype
            let (best_idx, best_sim) = self.best_match(code);
            if best_sim >= self.vigilance {
                // Update centroid online and assign
                let entry = &mut self.protos[best_idx];
                entry.count += 1;
                Self::update_centroid_static(entry, code);
                entry.id
            } else if self.protos.len() < MAX_PROTOTYPES {
                // Vigilance not met and capacity available: create new prototype
                self.create_proto(code)
            } else {
                // Capacity reached: assign to best match regardless
                let pid = self.protos[best_idx].id;
                let entry = &mut self.protos[best_idx];
                entry.count += 1;
                Self::update_centroid_static(entry, code);
                pid
            }
        };

        self.mem_to_proto.insert(memory_id, proto_id);
        proto_id
    }

    fn create_proto(&mut self, code: &SparseCode) -> ProtoId {
        let id = self.next_id;
        self.next_id += 1;
        let mut centroid = code.clone();
        // Normalize activations to sum=1
        let sum: f32 = centroid.activations.iter().sum();
        if sum > 1e-9 {
            for a in centroid.activations.iter_mut() {
                *a /= sum;
            }
        }
        self.protos.push(PrototypeEntry {
            id,
            centroid,
            count: 1,
        });
        id
    }

    /// Find the index (into self.protos) and similarity score of the best matching prototype.
    fn best_match(&self, code: &SparseCode) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, proto) in self.protos.iter().enumerate() {
            let sim = sparse_dot(&proto.centroid, code);
            if sim.total_cmp(&best_sim).is_gt()
                || (sim.total_cmp(&best_sim).is_eq() && proto.id < self.protos[best_idx].id) {
                best_sim = sim;
                best_idx = i;
            }
        }
        (best_idx, best_sim)
    }

    /// Find the nearest prototype for any sparse code (used in search scoring).
    pub fn nearest_proto(&self, code: &SparseCode) -> Option<ProtoId> {
        if self.protos.is_empty() || code.is_empty() {
            return None;
        }
        let (best_idx, _) = self.best_match(code);
        Some(self.protos[best_idx].id)
    }

    fn update_centroid_static(entry: &mut PrototypeEntry, new_code: &SparseCode) {
        let count = entry.count as f32;
        let old_weight = (count - 1.0) / count;
        let new_weight = 1.0 / count;

        // Merge all feature activations using a temporary map
        let mut merged: HashMap<u32, f32> = HashMap::new();

        for (&fid, &act) in entry
            .centroid
            .feature_ids
            .iter()
            .zip(entry.centroid.activations.iter())
        {
            *merged.entry(fid).or_insert(0.0) += old_weight * act;
        }
        for (&fid, &act) in new_code.feature_ids.iter().zip(new_code.activations.iter()) {
            *merged.entry(fid).or_insert(0.0) += new_weight * act;
        }

        // Keep top-64 by weighted sum
        let mut features: Vec<(u32, f32)> = merged.into_iter().collect();
        const TOP_K: usize = 64;
        if features.len() > TOP_K {
            features.select_nth_unstable_by(TOP_K, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            features.truncate(TOP_K);
        }

        // Sort by feature_id for merge efficiency
        features.sort_by_key(|(fid, _)| *fid);

        // Re-normalize activations to sum=1
        let sum: f32 = features.iter().map(|(_, a)| *a).sum();
        let norm = if sum > 1e-9 { sum } else { 1.0 };

        entry.centroid.feature_ids = features.iter().map(|(fid, _)| *fid).collect();
        entry.centroid.activations = features.iter().map(|(_, a)| a / norm).collect();
    }

    /// Strengthen directed transition a→b.
    /// Asymmetric: a→b gets full delta, b→a gets attenuated (0.3×).
    /// Sequential data naturally produces asymmetric couplings (FEP §3.2).
    pub fn strengthen_transition(&mut self, a: ProtoId, b: ProtoId, delta: f32) {
        if a == b {
            return;
        }
        let key_ab = (a, b);
        let key_ba = (b, a);
        let fwd = (self.transitions.get(&key_ab).copied().unwrap_or(0.0) + delta).min(1.0);
        let rev = (self.transitions.get(&key_ba).copied().unwrap_or(0.0) + delta * 0.3).min(1.0);
        self.transitions.insert(key_ab, fwd);
        self.transitions.insert(key_ba, rev);
    }

    pub fn get_proto(&self, memory_id: MemoryId) -> Option<ProtoId> {
        self.mem_to_proto.get(&memory_id).copied()
    }

    pub fn get(&self, proto_id: ProtoId) -> Option<&PrototypeEntry> {
        self.protos.iter().find(|p| p.id == proto_id)
    }

    pub fn transition(&self, a: ProtoId, b: ProtoId) -> f32 {
        self.transitions.get(&(a, b)).copied().unwrap_or(0.0)
    }

    /// Get nearest prototype and its similarity score.
    pub fn nearest_proto_with_sim(&self, code: &SparseCode) -> Option<(ProtoId, f32)> {
        if self.protos.is_empty() || code.is_empty() {
            return None;
        }
        let (best_idx, best_sim) = self.best_match(code);
        Some((self.protos[best_idx].id, best_sim))
    }

    /// Get the top-N outgoing transitions from a prototype, sorted by weight desc.
    pub fn top_transitions(&self, from: ProtoId, n: usize) -> Vec<(ProtoId, f32)> {
        let mut out: Vec<(ProtoId, f32)> = self.transitions.iter()
            .filter(|&(&(a, _), _)| a == from)
            .map(|(&(_, b), &w)| (b, w))
            .collect();
        out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(n);
        out
    }

    /// Get centroid for a prototype.
    pub fn get_centroid(&self, proto_id: ProtoId) -> Option<&SparseCode> {
        self.protos.iter().find(|p| p.id == proto_id).map(|p| &p.centroid)
    }

    /// Adaptive vigilance: lower when prediction error is high (more prototypes),
    /// higher when model is accurate (fewer prototypes). FEP §4.1.
    pub fn set_vigilance(&mut self, v: f32) {
        self.vigilance = v.clamp(0.001, 0.01);
    }

    pub fn vigilance(&self) -> f32 {
        self.vigilance
    }

    pub fn remove_memory(&mut self, memory_id: MemoryId) {
        self.mem_to_proto.remove(&memory_id);
    }

    pub fn count(&self) -> usize {
        self.protos.len()
    }
}

/// Sparse dot product using merge-style intersection on sorted feature_ids.
fn sparse_dot(a: &SparseCode, b: &SparseCode) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    let mut j = 0;
    let a_ids = &a.feature_ids;
    let b_ids = &b.feature_ids;
    while i < a_ids.len() && j < b_ids.len() {
        match a_ids[i].cmp(&b_ids[j]) {
            std::cmp::Ordering::Equal => {
                sum += a.activations[i] * b.activations[j];
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    sum
}
