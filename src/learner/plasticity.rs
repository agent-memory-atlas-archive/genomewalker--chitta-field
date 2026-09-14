use crate::ids::MemoryId;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct AccessStats {
    /// Exponentially weighted mean inter-access interval (ms).
    ewma_interval_ms: f64,
    last_accessed_ms: i64,
    access_count: u32,
    /// Cached surprise score from last encoding (reconstruction error).
    surprise: f32,
}

pub struct PlasticityLearner {
    stats: HashMap<MemoryId, AccessStats>,
    /// EMA decay factor (0.3 = recent accesses weighted more).
    alpha: f64,
}

impl PlasticityLearner {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            alpha: 0.3,
        }
    }

    /// Record an access to a memory. Returns the new recommended decay rate.
    /// Frequently accessed memories get lower decay rates (they matter).
    /// Never-accessed memories keep the default.
    pub fn record_access(&mut self, memory_id: MemoryId, now_ms: i64) -> f32 {
        let entry = self.stats.entry(memory_id).or_insert(AccessStats {
            ewma_interval_ms: 86_400_000.0, // default: 1 day
            last_accessed_ms: now_ms,
            access_count: 0,
            surprise: 0.5,
        });

        if entry.access_count > 0 {
            let interval = (now_ms - entry.last_accessed_ms).max(1) as f64;
            entry.ewma_interval_ms =
                self.alpha * interval + (1.0 - self.alpha) * entry.ewma_interval_ms;
        }
        entry.last_accessed_ms = now_ms;
        entry.access_count += 1;

        self.recommended_decay_rate(memory_id)
    }

    /// Snapshot only affected histories. Drain computes recommendations off-lock;
    /// the live learner is updated only after its WAL append succeeds.
    pub fn access_preview(&self, ids: impl Iterator<Item = MemoryId>) -> Self {
        let mut stats = HashMap::new();
        for id in ids {
            if let Some(value) = self.stats.get(&id) { stats.insert(id, value.clone()); }
        }
        Self { stats, alpha: self.alpha }
    }

    /// Update the cached surprise score for a memory.
    /// Called after encode_memory computes reconstruction error.
    pub fn update_surprise(&mut self, memory_id: MemoryId, surprise: f32) {
        if let Some(entry) = self.stats.get_mut(&memory_id) {
            entry.surprise = surprise;
        } else {
            self.stats.insert(memory_id, AccessStats {
                ewma_interval_ms: 86_400_000.0,
                last_accessed_ms: 0,
                access_count: 0,
                surprise,
            });
        }
    }

    /// Get recommended decay rate for a memory based on access history + surprise.
    ///
    /// FEP-principled: memories decay fast when they are predictable from other
    /// memories (low surprise) and decay slowly when they carry unique information
    /// (high surprise). This balances accuracy and complexity. FEP §2.3.
    ///
    /// Base: 0.001/day. Range: [0.0001, 0.01] per day.
    pub fn recommended_decay_rate(&self, memory_id: MemoryId) -> f32 {
        let stats = self.stats.get(&memory_id);
        let ewma = stats.map(|s| s.ewma_interval_ms).unwrap_or(86_400_000.0);
        let surprise = stats.map(|s| s.surprise).unwrap_or(0.5);

        // Access-frequency component: short interval = frequent = lower decay
        let days = ewma / 86_400_000.0;
        let base_rate = 0.0001_f64 * days.powf(0.5);

        // Surprise modulation: high surprise → slower decay (resist forgetting)
        // surprise is [0, 1], alpha controls modulation strength
        let alpha = 0.5_f64;
        let surprise_factor = 1.0 - alpha * surprise as f64;

        let rate = (base_rate * surprise_factor).clamp(0.0001, 0.01);
        rate as f32
    }

    pub fn memory_count(&self) -> usize {
        self.stats.len()
    }
}
