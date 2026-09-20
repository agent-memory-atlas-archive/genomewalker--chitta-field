#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CfHandle CfHandle;

typedef struct {
    uint64_t memory_id;
    float    score;
    float    semantic_score;
    int64_t  ts_ms;
    float    strength;
    float    confidence;
    uint32_t access_count;
    float    semantic_weight;
    float    status_mul;
    float    epistemic_mul;
    float    strength_factor;
    float    affect_valence;
    float    affect_arousal;
    float    actr_activation;
    float    surprise_boost;
    float    arousal_boost;
    float    mood_congruence;
    float    frustration_boost;
    float    interference_factor;
    float    spacing_boost;
} CfRecallHit;

/* Lifecycle */
CfHandle* cf_open(const char* data_dir, const char* lock_dir);
/* Secondary indexes are published independently of recall readiness. */
bool cf_startup_indexes_ready(CfHandle *handle);
/* The triplet graph's derived indexes rebuild off the open path; the exact and
   hybrid triplet lanes and spreading activation must answer loading until this
   returns true. */
bool cf_triplets_ready(CfHandle *handle);

void      cf_close(CfHandle* h);
const char* cf_last_error(const CfHandle* h);

/* Write */
int cf_put_memory(CfHandle* h,
    const char* kind, const char* realm,
    const uint8_t* content_ptr, size_t content_len,
    const float* embedding_ptr, size_t embedding_len,
    float confidence, float decay_rate, int64_t authored_at_ms,
    uint64_t* out_memory_id);

int cf_update_state(CfHandle* h, uint64_t memory_id,
    float strength_delta, float confidence_delta, float decay_rate,
    uint8_t touch, int8_t pin);

int cf_forget(CfHandle* h, uint64_t memory_id);

int cf_add_assoc_edge(CfHandle* h,
    uint64_t src, uint64_t dst, uint8_t edge_type, float weight);

int cf_upsert_artifact(CfHandle* h,
    const char* normalized_path, uint64_t* out_artifact_id);

/* Read */
int cf_recall_semantic(CfHandle* h,
    const float* query_embedding, size_t embedding_len,
    const char* realm, size_t k,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written,
    bool no_learn);

int cf_recall_semantic_ctx(CfHandle* h,
    const float* query_embedding, size_t embedding_len,
    const char* realm, size_t k,
    float query_valence, float query_arousal,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_recall_temporal(CfHandle* h,
    int64_t start_ms, int64_t end_ms, const char* realm, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_recall_temporal_events(CfHandle* h,
    int64_t start_ms, int64_t end_ms, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_recall_artifact(CfHandle* h,
    const char* normalized_path, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_expand_associations(CfHandle* h,
    const uint64_t* seed_ids, size_t seed_count,
    size_t max_hops, size_t limit,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written);

int cf_ppr_lane(CfHandle* h,
    const uint64_t* seed_ids, const float* seed_weights, size_t seed_count,
    size_t top_g,
    uint64_t* out_ids, float* out_scores, size_t out_cap, size_t* out_written);

int cf_recall_keyword(CfHandle* h,
    const char* query, size_t k, const char* realm,
    CfRecallHit* hits_buf, size_t hits_cap, size_t* hits_written,
    bool no_learn);

int cf_get_content(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

// Read-only recall hydration: never records an access, including buffer retries.
int cf_peek_content(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Deterministic provenance lookup (keyed lane). 0=hit (fills out_id+buf),
   1=clean miss, <0=error. sha tried first, then input path; either may be NULL. */
int cf_provenance_lookup(CfHandle* h,
    const char* sha, const char* input,
    uint64_t* out_id, uint8_t* buf, size_t buf_cap, size_t* written);

/* Deterministic correction lookup (keyed lane, capability #2). 0=hit (fills
   out_id + buf with up to 3 fired [correction] records, newest first), 1=clean
   miss, <0=error. text may be NULL. Exact-key bigram probe — no fuzzy recall. */
int cf_correction_check(CfHandle* h,
    const char* text,
    uint64_t* out_id, uint8_t* buf, size_t buf_cap, size_t* written);

/* Deterministic task-state lookup (keyed lane, capability #3). 0=hit (fills
   out_id + buf with the LATEST [task] record for the slug), 1=clean miss,
   <0=error. id may be NULL. Exact-key HashMap read — no fuzzy recall. */
int cf_task_state_lookup(CfHandle* h,
    const char* id,
    uint64_t* out_id, uint8_t* buf, size_t buf_cap, size_t* written);

int cf_get_kind(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap);

int cf_get_realm(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap);

int32_t cf_set_realm(CfHandle* h, uint64_t memory_id, const char* new_realm);

/* Triplets */
int cf_add_triplet(CfHandle* h,
    const char* subject, const char* predicate, const char* object,
    float weight,
    uint64_t source_memory_id,
    uint64_t* out_triplet_id);

int cf_invalidate_triplet(CfHandle* h, uint64_t triplet_id);
int cf_forget_triplet(CfHandle* h, const char* subject, const char* predicate, const char* object);
int cf_select_route(CfHandle* h, const char* query, uint64_t* out_episode_id, uint8_t* out_route);
int cf_route_feedback(CfHandle* h, uint64_t episode_id, float reward);
int cf_backfill_embedding(CfHandle* h, uint64_t memory_id, const float* embedding_ptr, size_t embedding_len);
int cf_pending_embeddings(CfHandle* h, uint64_t* out_ids, size_t max_ids, size_t* out_count);
int64_t cf_requeue_all_embeddings(CfHandle* h, const char* model_id, size_t model_id_len);

int cf_query_subject(CfHandle* h, const char* subject,
    char* buf, size_t buf_cap, size_t* written);

int cf_query_object(CfHandle* h, const char* object,
    char* buf, size_t buf_cap, size_t* written);

int cf_query_entity(CfHandle* h, const char* entity,
    char* buf, size_t buf_cap, size_t* written);

/* Learner */
int    cf_feedback(CfHandle* h, uint64_t episode_id, float reward);
size_t cf_recommended_window(CfHandle* h, const char* session_type);

/* Maintenance */
int    cf_flush(CfHandle* h);
size_t cf_memory_count(const CfHandle* h);
size_t cf_raw_memory_count(const CfHandle* h);
size_t cf_pending_count(const CfHandle* h);
/** Ingest new ops from foreign-instance segment files on shared storage.
 *  Safe to call from any thread. Returns ops applied, -1 on error. */
int    cf_sync_foreign(CfHandle* h);
/* sync_foreign split into two phases so the disk reads never run under the daemon's rpc lock.
 * collect: does the (possibly NFS-blocking) reads, no rpc lock.  apply: in-memory, rpc lock. */
int    cf_sync_foreign_collect(CfHandle* h);
int    cf_sync_foreign_apply(CfHandle* h);

/* Code Intelligence */

typedef struct {
    uint64_t symbol_id;
    float    score;
    uint8_t  kind[64];
    uint8_t  name[256];
    uint8_t  signature[512];
    uint8_t  file_path[1024];
    uint32_t line_start;
    uint32_t line_end;
    uint64_t repo_id;
} CfSymbolHit;

int cf_upsert_symbol(CfHandle* h,
    const char* kind, const char* name,
    const char* signature, const char* file_path,
    uint32_t line_start, uint32_t line_end, uint64_t repo_id,
    const float* embedding, size_t embed_len,
    const char* description,
    uint64_t memory_id,
    uint64_t* out_id);

int cf_remove_symbol(CfHandle* h, uint64_t symbol_id);

int cf_search_symbols_by_name(CfHandle* h,
    const char* query, size_t limit,
    CfSymbolHit* buf, size_t buf_len, size_t* written);

int cf_search_symbols_semantic(CfHandle* h,
    const float* query, size_t embed_len, size_t k,
    CfSymbolHit* buf, size_t buf_len, size_t* written);

int cf_symbols_in_file(CfHandle* h, const char* file_path,
    CfSymbolHit* buf, size_t buf_len, size_t* written);

/* path_filter: substring the file_path must contain (NULL/empty = unscoped) */
int cf_search_symbols_by_name_scoped(CfHandle* h,
    const char* query, size_t limit, const char* path_filter,
    CfSymbolHit* buf, size_t buf_len, size_t* written);

/* Remove every symbol in a file (per-file invalidation before re-extract). */
int cf_remove_symbols_by_file(CfHandle* h, const char* file_path,
    size_t* out_removed);

/* GC the symbol index. path_excludes = comma-separated substrings (NULL = none).
   Writes JSON stats {total,dup,excluded_path,dead_path,would_remove,removed,
   dry_run,top_dirs} to buf. */
int cf_dedupe_symbols(CfHandle* h, int dry_run, int check_fs,
    const char* path_excludes, uint8_t* buf, size_t buf_cap, size_t* written);

int cf_add_sym_call_edge(CfHandle* h, uint64_t caller_id, uint64_t callee_id);

int cf_get_callees(CfHandle* h, uint64_t symbol_id,
    uint64_t* buf, size_t buf_len, size_t* written);

int cf_get_callers(CfHandle* h, uint64_t symbol_id,
    uint64_t* buf, size_t buf_len, size_t* written);

int cf_upsert_code_file(CfHandle* h,
    const char* path, const char* project, int64_t mtime, uint64_t* out_id);

size_t cf_symbol_count(const CfHandle* h);
size_t cf_code_file_count(const CfHandle* h);

/* Sparse Predictive Associative Field (SPAF) */
size_t cf_encode_all(CfHandle* h);
size_t cf_cortical_count(const CfHandle* h);
size_t cf_prototype_count(CfHandle* h);

/* Residual Product Quantization */
bool   cf_train_pq(CfHandle* h);
size_t cf_encode_all_pq(CfHandle* h);
size_t cf_pq_count(CfHandle* h);

/* Lite Encoder (bag-of-words → sparse code, no ONNX) */
int32_t cf_train_lite_encoder(CfHandle* h);
int32_t cf_save_lite_encoder(CfHandle* h);
uint8_t cf_lite_encoder_ready(const CfHandle* h);
int32_t cf_encode_lite(const CfHandle* h,
    const uint8_t* text_ptr, size_t text_len,
    uint32_t* out_atoms, float* out_weights);

/* Cortical snapshot */
bool cf_save_snapshot(CfHandle* h);

/* Full state snapshot */
bool cf_save_full_snapshot(CfHandle* h);

/* Tier demotion */
uint64_t cf_run_demotion(CfHandle* h, int64_t now_ms);

/* Domain Event Log */
int cf_iterate_log(CfHandle* h, uint64_t from_seqno,
    void (*callback)(const uint8_t* op_json, size_t op_len, uint64_t seqno, void* ctx),
    void* ctx);

int cf_emit_event(CfHandle* h,
    const char* domain, const char* kind, const char* entity_id,
    const uint8_t* payload_json, size_t payload_len,
    const char* realm, uint64_t fencing_token,
    uint64_t* out_event_id);

/* Supports domain="user_model" and domain="transcript".
 * Returns 0 on success, 1 if not found, -2 if buf too small, -1 on error. */
int cf_get_latest_event(CfHandle* h,
    const char* domain, const char* kind, const char* entity_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Query events by domain+kind+target (e.g. msg_inbox). Returns JSON array into buf.
 * Returns 0=ok, -2=buf too small, -1=error. */
int cf_get_events_by_target(CfHandle* h,
    const char* domain, const char* kind, const char* target,
    size_t limit,
    uint8_t* out_buf, size_t buf_cap, size_t* written);

/* Query all events matching domain+kind across all targets. Returns JSON array (newest-first).
 * Returns 0=ok, -2=buf too small, -1=error. */
int cf_get_events_by_domain_kind(CfHandle* h,
    const char* domain, const char* kind,
    size_t limit,
    uint8_t* out_buf, size_t buf_cap, size_t* written);

/* Check whether any event exists for (domain, kind, target). Returns 1 if found, 0 if not. */
int cf_has_event(CfHandle* h,
    const char* domain, const char* kind, const char* target);

/* Look up a single event by event_id. Returns JSON object, or "{}" if not found.
 * Returns 0=ok, -2=buf too small, -1=error. */
int cf_get_event_by_id(CfHandle* h,
    uint64_t event_id,
    uint8_t* out_buf, size_t buf_cap, size_t* written);

/* Session management */
int cf_session_register(CfHandle* h,
    const char* session_id, const char* kind, const char* realm, int64_t now_ms);

int cf_session_heartbeat(CfHandle* h,
    const char* session_id, int64_t now_ms);

int cf_session_deregister(CfHandle* h,
    const char* session_id, int64_t now_ms);

/* Transcript management */
int cf_transcript_register(CfHandle* h,
    const char* transcript_id, const char* session_id);

int cf_transcript_update_progress(CfHandle* h,
    const char* transcript_id, float progress_pct);

int cf_transcript_add_turn(CfHandle* h,
    const char* transcript_id, const char* role,
    const uint8_t* content_ptr, size_t content_len,
    int64_t ts_ms, uint64_t* out_turn_id);

/* Task / Sadhana / Dream management */
int cf_task_create(CfHandle* h,
    const char* task_id, const char* kind,
    const uint8_t* payload_json, size_t payload_len,
    int64_t now_ms, uint64_t fencing_token);

int cf_task_transition(CfHandle* h,
    const char* task_id, const char* new_status,
    int64_t now_ms, uint64_t fencing_token);

int cf_task_list(CfHandle* h,
    const char* kind_filter, uint8_t active_only,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* User model management */
int cf_user_model_upsert(CfHandle* h,
    const char* entity_id, const char* entity_type,
    const uint8_t* payload_json, size_t payload_len,
    int64_t now_ms);

int cf_user_model_observe(CfHandle* h,
    const char* entity_id, int64_t now_ms);

int cf_user_model_list(CfHandle* h,
    const char* entity_type_filter,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Theme management */
int cf_theme_create(CfHandle* h,
    uint64_t theme_id, const char* name);

int cf_theme_update_centroid(CfHandle* h,
    uint64_t theme_id,
    const uint8_t* centroid_json, size_t centroid_len);

int cf_theme_assign_member(CfHandle* h,
    uint64_t theme_id, uint64_t memory_id);

int cf_theme_remove_member(CfHandle* h,
    uint64_t theme_id, uint64_t memory_id);

int cf_theme_list(CfHandle* h,
    uint8_t* buf, size_t buf_cap, size_t* written);

int cf_theme_get(CfHandle* h, uint64_t theme_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

int cf_theme_stats(CfHandle* h, const char* realm,
    uint8_t* buf, size_t buf_cap, size_t* written);

int cf_theme_recall(CfHandle* h,
    const float* embedding_ptr, size_t embedding_len,
    size_t k, const char* realm,
    uint8_t* buf, size_t buf_cap, size_t* written);

int cf_theme_maintain(CfHandle* h,
    uint8_t* buf, size_t buf_cap, size_t* written);

int cf_theme_assign_orphans(CfHandle* h,
    size_t batch_size, const char* realm,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Filtered recall — returns JSON array into buf */
int cf_recall_filtered(CfHandle* h,
    const char* kind, const char* realm,
    float min_confidence, float min_strength, size_t limit,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Paginated memory listing sorted by strength/recency/confidence */
int cf_list_memories(CfHandle* h,
    const char* kind, const char* realm,
    const char* sort_by, size_t limit, size_t offset,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Aggregate stats: count_by_kind, avg_confidence, avg_strength, total */
int cf_memory_stats(CfHandle* h, const char* realm_filter,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Get single task by ID (JSON payload) */
int cf_task_get(CfHandle* h, const char* task_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Update task payload (returns 0=ok, -1=not found) */
int32_t cf_task_update_payload(CfHandle* h, const char* task_id,
    const char* payload_json, int64_t now_ms);

/* List sessions (JSON array), active_only filters by active status */
int cf_session_list(CfHandle* h, int32_t active_only,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* List transcripts (JSON array, most recent first) */
int cf_transcript_list(CfHandle* h, size_t limit,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Get memory metadata by ID (JSON) */
int cf_get_memory_metadata(CfHandle* h, uint64_t memory_id,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Update memory kind field (returns 0=ok, -1=not found) */
int32_t cf_update_memory_kind(CfHandle* h, uint64_t memory_id, const char* new_kind);

/* List all triplets where entity is subject OR object */
int cf_list_triplets_for_entity(CfHandle* h, const char* entity, size_t limit,
    char* buf, size_t buf_cap, size_t* written);

/* Code file listing (JSON array) */
int cf_list_code_files(CfHandle* h, const char* project_filter,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Clear all code files + symbols for a project */
int cf_clear_project(CfHandle* h, const char* project);

/* Update symbol description */
int cf_set_symbol_description(CfHandle* h, uint64_t symbol_id,
    const char* description, size_t description_len);

/* Update memory content + embedding */
int cf_update_memory_content(CfHandle* h, uint64_t id,
    const uint8_t* content, size_t content_len,
    const float* embedding, size_t embedding_len);

/* List distinct realm names (JSON string array) */
int cf_realm_list(CfHandle* h,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Recall memories by kind, sorted by confidence (JSON array) */
int cf_recall_by_kind(CfHandle* h, const char* kind, size_t limit,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Purge corrupt memories (empty content or non-finite affect). Returns 0 on success. */
int cf_purge_corrupt(CfHandle* h, size_t* out_purged);

/* Recall instrumentation */
int32_t cf_get_memory_embeddings_batch(const CfHandle* handle, const uint64_t* ids, size_t ids_len, char* out_buf, size_t out_buf_len, size_t* written);
int32_t cf_record_recall_batch(CfHandle* handle, const uint64_t* ids, size_t ids_len, const int8_t* centroid_q, size_t centroid_q_len, float centroid_scale, uint64_t context_hash, int64_t ts_ms, float base_assoc_delta);

/* Analytics */
int cf_analytics_append(CfHandle* h,
    const char* kind, const char* entity_id,
    const uint8_t* payload_json, size_t payload_len,
    int64_t ts_ms);

int cf_analytics_recent(CfHandle* h,
    size_t limit,
    uint8_t* buf, size_t buf_cap, size_t* written);

/* Association edge query */
int cf_get_assoc_edges(CfHandle* h, uint64_t memory_id, size_t limit,
    char* buf, size_t buf_cap, size_t* written);

int cf_set_epistemic_status(CfHandle* h, uint64_t memory_id, uint8_t epistemic_status);
int cf_set_affect(CfHandle* h, uint64_t memory_id, float valence, float arousal);

/* Contradiction engine */
int cf_get_conflicts(CfHandle* h, uint64_t memory_id,
    uint64_t* out_ids, size_t max_ids, size_t* out_count);

int cf_get_supersession_chain(CfHandle* h, uint64_t memory_id,
    uint64_t* out_ids, size_t max_ids, size_t* out_count);

int cf_get_confirmations(CfHandle* h, uint64_t memory_id,
    uint64_t* out_ids, size_t max_ids, size_t* out_count);

#ifdef __cplusplus
}
#endif
