/**
 * @file snapshot_store.c
 * @brief SQLite-backed chunked snapshot storage.
 */

#include "io/snapshot_store.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(NLO_HAVE_SQLITE3)
#include <sqlite3.h>

#ifndef NLO_STORAGE_DEFAULT_DB_MAX_BYTES
#define NLO_STORAGE_DEFAULT_DB_MAX_BYTES (50ull * 1024ull * 1024ull * 1024ull)
#endif

#ifndef NLO_STORAGE_DEFAULT_CHUNK_RECORDS
#define NLO_STORAGE_DEFAULT_CHUNK_RECORDS 16u
#endif

#ifndef NLO_STORAGE_DB_SIZE_SAFETY_BYTES
#define NLO_STORAGE_DB_SIZE_SAFETY_BYTES (1ull << 20)
#endif

struct nlo_snapshot_store {
    sqlite3* db;
    sqlite3_stmt* insert_chunk_stmt;
    sqlite3_stmt* update_run_stmt;
    sqlite3_stmt* upsert_final_output_stmt;
    sqlite3_stmt* page_count_stmt;
    sqlite3_stmt* page_size_stmt;
    nlo_complex* chunk_buffer;
    size_t num_time_samples;
    size_t chunk_records;
    size_t chunk_fill;
    size_t chunk_start_record;
    size_t chunk_index;
    size_t max_bytes;
    nlo_storage_db_cap_policy cap_policy;
    int accept_writes;
    nlo_storage_result result;
};

static int nlo_checked_mul_size_t(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return -1;
    }
    if (a == 0u || b == 0u) {
        *out = 0u;
        return 0;
    }
    if (a > (SIZE_MAX / b)) {
        return -1;
    }
    *out = a * b;
    return 0;
}

static uint64_t nlo_fnv1a_seed(void)
{
    return UINT64_C(1469598103934665603);
}

static uint64_t nlo_fnv1a_step(uint64_t hash, const void* bytes, size_t len)
{
    const unsigned char* ptr = (const unsigned char*)bytes;
    for (size_t i = 0u; i < len; ++i) {
        hash ^= (uint64_t)ptr[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void nlo_hash_cstr(uint64_t* hash, const char* text)
{
    const char* safe_text = (text != NULL) ? text : "";
    *hash = nlo_fnv1a_step(*hash, safe_text, strlen(safe_text));
}

static void nlo_hash_hex16(uint64_t hash, char out_hex[17])
{
    (void)snprintf(out_hex, 17u, "%016llx", (unsigned long long)hash);
}

static void nlo_compute_system_hash(
    const sim_config* config,
    const nlo_execution_options* exec_options,
    size_t num_time_samples,
    char out_hash[17]
)
{
    uint64_t hash = nlo_fnv1a_seed();
    hash = nlo_fnv1a_step(hash, &num_time_samples, sizeof(num_time_samples));
    if (config != NULL) {
        hash = nlo_fnv1a_step(hash, &config->time.nt, sizeof(config->time.nt));
        hash = nlo_fnv1a_step(hash, &config->spatial.nx, sizeof(config->spatial.nx));
        hash = nlo_fnv1a_step(hash, &config->spatial.ny, sizeof(config->spatial.ny));
    }
    if (exec_options != NULL) {
        hash = nlo_fnv1a_step(hash, &exec_options->backend_type, sizeof(exec_options->backend_type));
        hash = nlo_fnv1a_step(hash, &exec_options->fft_backend, sizeof(exec_options->fft_backend));
        hash = nlo_fnv1a_step(hash, &exec_options->device_heap_fraction, sizeof(exec_options->device_heap_fraction));
        hash = nlo_fnv1a_step(hash, &exec_options->record_ring_target, sizeof(exec_options->record_ring_target));
        hash = nlo_fnv1a_step(hash,
                              &exec_options->forced_device_budget_bytes,
                              sizeof(exec_options->forced_device_budget_bytes));
    }
    nlo_hash_hex16(hash, out_hash);
}

static void nlo_compute_config_hash(const sim_config* config, char out_hash[17])
{
    uint64_t hash = nlo_fnv1a_seed();
    if (config != NULL) {
        hash = nlo_fnv1a_step(hash,
                              &config->propagation.starting_step_size,
                              sizeof(config->propagation.starting_step_size));
        hash = nlo_fnv1a_step(hash,
                              &config->propagation.max_step_size,
                              sizeof(config->propagation.max_step_size));
        hash = nlo_fnv1a_step(hash,
                              &config->propagation.min_step_size,
                              sizeof(config->propagation.min_step_size));
        hash = nlo_fnv1a_step(hash,
                              &config->propagation.error_tolerance,
                              sizeof(config->propagation.error_tolerance));
        hash = nlo_fnv1a_step(hash,
                              &config->propagation.propagation_distance,
                              sizeof(config->propagation.propagation_distance));
        hash = nlo_fnv1a_step(hash, &config->time.nt, sizeof(config->time.nt));
        hash = nlo_fnv1a_step(hash, &config->time.pulse_period, sizeof(config->time.pulse_period));
        hash = nlo_fnv1a_step(hash, &config->time.delta_time, sizeof(config->time.delta_time));
        hash = nlo_fnv1a_step(hash, &config->spatial.nx, sizeof(config->spatial.nx));
        hash = nlo_fnv1a_step(hash, &config->spatial.ny, sizeof(config->spatial.ny));
        hash = nlo_fnv1a_step(hash, &config->spatial.delta_x, sizeof(config->spatial.delta_x));
        hash = nlo_fnv1a_step(hash, &config->spatial.delta_y, sizeof(config->spatial.delta_y));
        hash = nlo_fnv1a_step(hash, &config->runtime.num_constants, sizeof(config->runtime.num_constants));
        hash = nlo_fnv1a_step(hash, config->runtime.constants, sizeof(config->runtime.constants));
        nlo_hash_cstr(&hash, config->runtime.dispersion_factor_expr);
        nlo_hash_cstr(&hash, config->runtime.dispersion_expr);
        nlo_hash_cstr(&hash, config->runtime.transverse_factor_expr);
        nlo_hash_cstr(&hash, config->runtime.transverse_expr);
        nlo_hash_cstr(&hash, config->runtime.nonlinear_expr);
    }
    nlo_hash_hex16(hash, out_hash);
}

static void nlo_make_run_id(char out_run_id[NLO_STORAGE_RUN_ID_MAX], const char* explicit_id)
{
    if (out_run_id == NULL) {
        return;
    }
    if (explicit_id != NULL && explicit_id[0] != '\0') {
        (void)snprintf(out_run_id, NLO_STORAGE_RUN_ID_MAX, "%s", explicit_id);
        return;
    }

    const unsigned long long now = (unsigned long long)time(NULL);
    static unsigned long long serial = 0ull;
    serial += 1ull;
    (void)snprintf(out_run_id, NLO_STORAGE_RUN_ID_MAX, "run-%llx-%llx", now, serial);
}

static int nlo_sqlite_exec(sqlite3* db, const char* sql)
{
    char* err = NULL;
    const int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        if (err != NULL) {
            fprintf(stderr, "[nlolib][io] sqlite exec error: %s\n", err);
        }
        sqlite3_free(err);
        return -1;
    }
    return 0;
}

static int nlo_prepare(sqlite3* db, const char* sql, sqlite3_stmt** out_stmt)
{
    if (db == NULL || sql == NULL || out_stmt == NULL) {
        return -1;
    }
    *out_stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, out_stmt, NULL) != SQLITE_OK) {
        return -1;
    }
    return 0;
}

static int nlo_bind_text(sqlite3_stmt* stmt, int index, const char* text)
{
    const char* safe = (text != NULL) ? text : "";
    return (sqlite3_bind_text(stmt, index, safe, -1, SQLITE_TRANSIENT) == SQLITE_OK) ? 0 : -1;
}

static int nlo_step_done(sqlite3_stmt* stmt)
{
    const int rc = sqlite3_step(stmt);
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

static int nlo_query_db_size(sqlite3_stmt* page_count_stmt, sqlite3_stmt* page_size_stmt, size_t* out_size)
{
    if (page_count_stmt == NULL || page_size_stmt == NULL || out_size == NULL) {
        return -1;
    }

    sqlite3_reset(page_count_stmt);
    sqlite3_clear_bindings(page_count_stmt);
    if (sqlite3_step(page_count_stmt) != SQLITE_ROW) {
        sqlite3_reset(page_count_stmt);
        return -1;
    }
    const sqlite3_int64 page_count = sqlite3_column_int64(page_count_stmt, 0);
    sqlite3_reset(page_count_stmt);

    sqlite3_reset(page_size_stmt);
    sqlite3_clear_bindings(page_size_stmt);
    if (sqlite3_step(page_size_stmt) != SQLITE_ROW) {
        sqlite3_reset(page_size_stmt);
        return -1;
    }
    const sqlite3_int64 page_size = sqlite3_column_int64(page_size_stmt, 0);
    sqlite3_reset(page_size_stmt);

    if (page_count < 0 || page_size < 0) {
        return -1;
    }

    const unsigned long long bytes =
        (unsigned long long)page_count * (unsigned long long)page_size;
    if (bytes > (unsigned long long)SIZE_MAX) {
        *out_size = SIZE_MAX;
    } else {
        *out_size = (size_t)bytes;
    }
    return 0;
}

static int nlo_store_update_run(nlo_snapshot_store* store)
{
    if (store == NULL || store->update_run_stmt == NULL) {
        return -1;
    }

    if (sqlite3_bind_int64(store->update_run_stmt, 1, (sqlite3_int64)store->result.records_spilled) != SQLITE_OK) {
        return -1;
    }
    if (sqlite3_bind_int64(store->update_run_stmt, 2, (sqlite3_int64)store->result.chunks_written) != SQLITE_OK) {
        return -1;
    }
    if (sqlite3_bind_int64(store->update_run_stmt, 3, (sqlite3_int64)store->result.db_size_bytes) != SQLITE_OK) {
        return -1;
    }
    if (sqlite3_bind_int(store->update_run_stmt, 4, store->result.truncated ? 1 : 0) != SQLITE_OK) {
        return -1;
    }
    if (nlo_bind_text(store->update_run_stmt, 5, store->result.run_id) != 0) {
        return -1;
    }
    return nlo_step_done(store->update_run_stmt);
}

static nlo_snapshot_store_status nlo_store_flush_chunk(nlo_snapshot_store* store)
{
    if (store == NULL || store->chunk_fill == 0u) {
        return NLO_SNAPSHOT_STORE_STATUS_OK;
    }

    if (!store->accept_writes) {
        store->result.truncated = 1;
        store->chunk_fill = 0u;
        return NLO_SNAPSHOT_STORE_STATUS_SOFT_LIMIT;
    }

    size_t payload_bytes = 0u;
    if (nlo_checked_mul_size_t(store->chunk_fill, store->num_time_samples, &payload_bytes) != 0 ||
        nlo_checked_mul_size_t(payload_bytes, sizeof(nlo_complex), &payload_bytes) != 0 ||
        payload_bytes > (size_t)INT_MAX) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    size_t db_size = 0u;
    if (nlo_query_db_size(store->page_count_stmt, store->page_size_stmt, &db_size) == 0) {
        store->result.db_size_bytes = db_size;
    }

    if (store->max_bytes > 0u) {
        const unsigned long long projected =
            (unsigned long long)store->result.db_size_bytes +
            (unsigned long long)payload_bytes +
            (unsigned long long)NLO_STORAGE_DB_SIZE_SAFETY_BYTES;
        if (projected > (unsigned long long)store->max_bytes) {
            if (store->cap_policy == NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES) {
                store->result.truncated = 1;
                store->accept_writes = 0;
                store->chunk_fill = 0u;
                (void)nlo_store_update_run(store);
                return NLO_SNAPSHOT_STORE_STATUS_SOFT_LIMIT;
            }
            return NLO_SNAPSHOT_STORE_STATUS_ERROR;
        }
    }

    if (nlo_bind_text(store->insert_chunk_stmt, 1, store->result.run_id) != 0) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->insert_chunk_stmt, 2, (sqlite3_int64)store->chunk_index) != SQLITE_OK) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->insert_chunk_stmt, 3, (sqlite3_int64)store->chunk_start_record) != SQLITE_OK) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->insert_chunk_stmt, 4, (sqlite3_int64)store->chunk_fill) != SQLITE_OK) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_blob(store->insert_chunk_stmt,
                          5,
                          store->chunk_buffer,
                          (int)payload_bytes,
                          SQLITE_TRANSIENT) != SQLITE_OK) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (nlo_step_done(store->insert_chunk_stmt) != 0) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    store->result.records_spilled += store->chunk_fill;
    store->result.chunks_written += 1u;
    store->chunk_index += 1u;
    store->chunk_fill = 0u;

    if (nlo_query_db_size(store->page_count_stmt, store->page_size_stmt, &db_size) == 0) {
        store->result.db_size_bytes = db_size;
    }

    if (store->max_bytes > 0u &&
        store->result.db_size_bytes > store->max_bytes &&
        store->cap_policy == NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES) {
        store->result.truncated = 1;
        store->accept_writes = 0;
    }

    if (nlo_store_update_run(store) != 0) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    return NLO_SNAPSHOT_STORE_STATUS_OK;
}

static int nlo_store_initialize_schema(sqlite3* db)
{
    static const char* schema_sql =
        "PRAGMA journal_mode=WAL;"
        "PRAGMA synchronous=NORMAL;"
        "PRAGMA foreign_keys=ON;"
        "CREATE TABLE IF NOT EXISTS io_system_config ("
        "  system_hash TEXT PRIMARY KEY,"
        "  backend_type INTEGER NOT NULL,"
        "  fft_backend INTEGER NOT NULL,"
        "  num_time_samples INTEGER NOT NULL,"
        "  nt INTEGER NOT NULL,"
        "  nx INTEGER NOT NULL,"
        "  ny INTEGER NOT NULL,"
        "  device_heap_fraction REAL NOT NULL,"
        "  record_ring_target INTEGER NOT NULL,"
        "  forced_device_budget_bytes INTEGER NOT NULL,"
        "  created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE TABLE IF NOT EXISTS io_sim_config ("
        "  config_hash TEXT PRIMARY KEY,"
        "  z_end REAL NOT NULL,"
        "  h_start REAL NOT NULL,"
        "  h_max REAL NOT NULL,"
        "  h_min REAL NOT NULL,"
        "  tolerance REAL NOT NULL,"
        "  pulse_period REAL NOT NULL,"
        "  delta_time REAL NOT NULL,"
        "  disp_factor_expr TEXT,"
        "  disp_expr TEXT,"
        "  trans_factor_expr TEXT,"
        "  trans_expr TEXT,"
        "  nonlinear_expr TEXT,"
        "  constants_blob BLOB,"
        "  created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ");"
        "CREATE TABLE IF NOT EXISTS io_runs ("
        "  run_id TEXT PRIMARY KEY,"
        "  system_hash TEXT NOT NULL,"
        "  config_hash TEXT NOT NULL,"
        "  num_recorded_samples INTEGER NOT NULL,"
        "  num_time_samples INTEGER NOT NULL,"
        "  chunk_records INTEGER NOT NULL,"
        "  cap_policy INTEGER NOT NULL,"
        "  max_bytes INTEGER NOT NULL,"
        "  records_written INTEGER NOT NULL DEFAULT 0,"
        "  chunks_written INTEGER NOT NULL DEFAULT 0,"
        "  db_size_bytes INTEGER NOT NULL DEFAULT 0,"
        "  truncated INTEGER NOT NULL DEFAULT 0,"
        "  created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  FOREIGN KEY(system_hash) REFERENCES io_system_config(system_hash),"
        "  FOREIGN KEY(config_hash) REFERENCES io_sim_config(config_hash)"
        ");"
        "CREATE TABLE IF NOT EXISTS io_record_chunks ("
        "  run_id TEXT NOT NULL,"
        "  chunk_index INTEGER NOT NULL,"
        "  record_start INTEGER NOT NULL,"
        "  record_count INTEGER NOT NULL,"
        "  payload BLOB NOT NULL,"
        "  PRIMARY KEY(run_id, chunk_index),"
        "  UNIQUE(run_id, record_start),"
        "  FOREIGN KEY(run_id) REFERENCES io_runs(run_id) ON DELETE CASCADE"
        ");"
        "CREATE TABLE IF NOT EXISTS io_final_output_fields ("
        "  run_id TEXT PRIMARY KEY,"
        "  num_time_samples INTEGER NOT NULL,"
        "  payload BLOB NOT NULL,"
        "  created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "  FOREIGN KEY(run_id) REFERENCES io_runs(run_id) ON DELETE CASCADE"
        ");";

    return nlo_sqlite_exec(db, schema_sql);
}

static int nlo_store_insert_metadata(
    nlo_snapshot_store* store,
    const sim_config* config,
    const nlo_execution_options* exec_options,
    size_t num_time_samples,
    size_t num_recorded_samples
)
{
    if (store == NULL || store->db == NULL || config == NULL || exec_options == NULL) {
        return -1;
    }

    char system_hash[17];
    char config_hash[17];
    nlo_compute_system_hash(config, exec_options, num_time_samples, system_hash);
    nlo_compute_config_hash(config, config_hash);

    sqlite3_stmt* stmt = NULL;
    if (nlo_prepare(store->db,
                    "INSERT INTO io_system_config("
                    "system_hash, backend_type, fft_backend, num_time_samples, nt, nx, ny, "
                    "device_heap_fraction, record_ring_target, forced_device_budget_bytes"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(system_hash) DO NOTHING;",
                    &stmt) != 0) {
        return -1;
    }
    if (nlo_bind_text(stmt, 1, system_hash) != 0 ||
        sqlite3_bind_int(stmt, 2, (int)exec_options->backend_type) != SQLITE_OK ||
        sqlite3_bind_int(stmt, 3, (int)exec_options->fft_backend) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 4, (sqlite3_int64)num_time_samples) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 5, (sqlite3_int64)config->time.nt) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 6, (sqlite3_int64)config->spatial.nx) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 7, (sqlite3_int64)config->spatial.ny) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 8, exec_options->device_heap_fraction) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 9, (sqlite3_int64)exec_options->record_ring_target) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 10, (sqlite3_int64)exec_options->forced_device_budget_bytes) != SQLITE_OK ||
        nlo_step_done(stmt) != 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    if (nlo_prepare(store->db,
                    "INSERT INTO io_sim_config("
                    "config_hash, z_end, h_start, h_max, h_min, tolerance, pulse_period, delta_time, "
                    "disp_factor_expr, disp_expr, trans_factor_expr, trans_expr, nonlinear_expr, constants_blob"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(config_hash) DO NOTHING;",
                    &stmt) != 0) {
        return -1;
    }
    if (nlo_bind_text(stmt, 1, config_hash) != 0 ||
        sqlite3_bind_double(stmt, 2, config->propagation.propagation_distance) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 3, config->propagation.starting_step_size) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 4, config->propagation.max_step_size) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 5, config->propagation.min_step_size) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 6, config->propagation.error_tolerance) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 7, config->time.pulse_period) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 8, config->time.delta_time) != SQLITE_OK ||
        nlo_bind_text(stmt, 9, config->runtime.dispersion_factor_expr) != 0 ||
        nlo_bind_text(stmt, 10, config->runtime.dispersion_expr) != 0 ||
        nlo_bind_text(stmt, 11, config->runtime.transverse_factor_expr) != 0 ||
        nlo_bind_text(stmt, 12, config->runtime.transverse_expr) != 0 ||
        nlo_bind_text(stmt, 13, config->runtime.nonlinear_expr) != 0 ||
        sqlite3_bind_blob(stmt,
                          14,
                          config->runtime.constants,
                          (int)sizeof(config->runtime.constants),
                          SQLITE_TRANSIENT) != SQLITE_OK ||
        nlo_step_done(stmt) != 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    if (nlo_prepare(store->db,
                    "INSERT INTO io_runs("
                    "run_id, system_hash, config_hash, num_recorded_samples, num_time_samples, "
                    "chunk_records, cap_policy, max_bytes, records_written, chunks_written, db_size_bytes, truncated"
                    ") VALUES(?,?,?,?,?,?,?,?,0,0,0,0) "
                    "ON CONFLICT(run_id) DO NOTHING;",
                    &stmt) != 0) {
        return -1;
    }
    if (nlo_bind_text(stmt, 1, store->result.run_id) != 0 ||
        nlo_bind_text(stmt, 2, system_hash) != 0 ||
        nlo_bind_text(stmt, 3, config_hash) != 0 ||
        sqlite3_bind_int64(stmt, 4, (sqlite3_int64)num_recorded_samples) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 5, (sqlite3_int64)num_time_samples) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 6, (sqlite3_int64)store->chunk_records) != SQLITE_OK ||
        sqlite3_bind_int(stmt, 7, (int)store->cap_policy) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 8, (sqlite3_int64)store->max_bytes) != SQLITE_OK ||
        nlo_step_done(stmt) != 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    return 0;
}

nlo_snapshot_store* nlo_snapshot_store_open(const nlo_snapshot_store_open_params* params)
{
    if (params == NULL || params->storage_options == NULL || params->storage_options->sqlite_path == NULL) {
        return NULL;
    }

    if (params->storage_options->sqlite_path[0] == '\0') {
        return NULL;
    }

    nlo_snapshot_store* store = (nlo_snapshot_store*)calloc(1u, sizeof(nlo_snapshot_store));
    if (store == NULL) {
        return NULL;
    }

    store->num_time_samples = params->num_time_samples;
    store->chunk_records =
        (params->storage_options->chunk_records > 0u)
            ? params->storage_options->chunk_records
            : NLO_STORAGE_DEFAULT_CHUNK_RECORDS;
    if (store->chunk_records > params->num_recorded_samples) {
        store->chunk_records = params->num_recorded_samples;
    }
    if (store->chunk_records == 0u) {
        store->chunk_records = 1u;
    }

    store->max_bytes =
        (params->storage_options->sqlite_max_bytes > 0u)
            ? params->storage_options->sqlite_max_bytes
            : (size_t)NLO_STORAGE_DEFAULT_DB_MAX_BYTES;
    store->cap_policy = params->storage_options->cap_policy;
    store->accept_writes = 1;

    nlo_make_run_id(store->result.run_id, params->storage_options->run_id);

    size_t chunk_elements = 0u;
    if (nlo_checked_mul_size_t(store->chunk_records, store->num_time_samples, &chunk_elements) != 0) {
        nlo_snapshot_store_close(store);
        return NULL;
    }
    store->chunk_buffer = (nlo_complex*)calloc(chunk_elements, sizeof(nlo_complex));
    if (store->chunk_buffer == NULL) {
        nlo_snapshot_store_close(store);
        return NULL;
    }

    if (sqlite3_open_v2(params->storage_options->sqlite_path,
                        &store->db,
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                        NULL) != SQLITE_OK) {
        nlo_snapshot_store_close(store);
        return NULL;
    }

    if (nlo_store_initialize_schema(store->db) != 0 ||
        nlo_store_insert_metadata(store,
                                  params->config,
                                  params->exec_options,
                                  params->num_time_samples,
                                  params->num_recorded_samples) != 0 ||
        nlo_prepare(store->db,
                    "INSERT INTO io_record_chunks(run_id, chunk_index, record_start, record_count, payload)"
                    " VALUES(?,?,?,?,?);",
                    &store->insert_chunk_stmt) != 0 ||
        nlo_prepare(store->db,
                    "UPDATE io_runs SET records_written=?, chunks_written=?, db_size_bytes=?, truncated=? "
                    "WHERE run_id=?;",
                    &store->update_run_stmt) != 0 ||
        nlo_prepare(store->db,
                    "INSERT INTO io_final_output_fields(run_id, num_time_samples, payload) "
                    "VALUES(?,?,?) "
                    "ON CONFLICT(run_id) DO UPDATE SET "
                    "num_time_samples=excluded.num_time_samples, "
                    "payload=excluded.payload;",
                    &store->upsert_final_output_stmt) != 0 ||
        nlo_prepare(store->db, "PRAGMA page_count;", &store->page_count_stmt) != 0 ||
        nlo_prepare(store->db, "PRAGMA page_size;", &store->page_size_stmt) != 0) {
        nlo_snapshot_store_close(store);
        return NULL;
    }

    (void)nlo_query_db_size(store->page_count_stmt, store->page_size_stmt, &store->result.db_size_bytes);
    (void)nlo_store_update_run(store);
    return store;
}

nlo_snapshot_store_status nlo_snapshot_store_write_record(
    nlo_snapshot_store* store,
    size_t record_index,
    const nlo_complex* record,
    size_t num_time_samples
)
{
    if (store == NULL || record == NULL || num_time_samples != store->num_time_samples) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    if (store->chunk_fill == 0u) {
        store->chunk_start_record = record_index;
    }

    const size_t dst_index = store->chunk_fill * store->num_time_samples;
    memcpy(store->chunk_buffer + dst_index,
           record,
           store->num_time_samples * sizeof(nlo_complex));

    store->chunk_fill += 1u;
    store->result.records_captured += 1u;

    if (store->chunk_fill >= store->chunk_records) {
        return nlo_store_flush_chunk(store);
    }

    return NLO_SNAPSHOT_STORE_STATUS_OK;
}

nlo_snapshot_store_status nlo_snapshot_store_write_final_output_field(
    nlo_snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
)
{
    if (store == NULL ||
        field == NULL ||
        store->upsert_final_output_stmt == NULL ||
        num_time_samples != store->num_time_samples) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    size_t payload_bytes = 0u;
    if (nlo_checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &payload_bytes) != 0 ||
        payload_bytes > (size_t)INT_MAX) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    if (nlo_bind_text(store->upsert_final_output_stmt, 1, store->result.run_id) != 0) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->upsert_final_output_stmt, 2, (sqlite3_int64)num_time_samples) != SQLITE_OK) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_blob(store->upsert_final_output_stmt,
                          3,
                          field,
                          (int)payload_bytes,
                          SQLITE_TRANSIENT) != SQLITE_OK) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (nlo_step_done(store->upsert_final_output_stmt) != 0) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    return NLO_SNAPSHOT_STORE_STATUS_OK;
}

nlo_snapshot_store_status nlo_snapshot_store_flush(nlo_snapshot_store* store)
{
    if (store == NULL) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    const nlo_snapshot_store_status status = nlo_store_flush_chunk(store);
    if (status == NLO_SNAPSHOT_STORE_STATUS_ERROR) {
        return status;
    }

    if (nlo_store_update_run(store) != 0) {
        return NLO_SNAPSHOT_STORE_STATUS_ERROR;
    }

    return status;
}

void nlo_snapshot_store_get_result(const nlo_snapshot_store* store, nlo_storage_result* out_result)
{
    if (out_result == NULL) {
        return;
    }

    *out_result = (nlo_storage_result){0};
    if (store == NULL) {
        return;
    }

    *out_result = store->result;
}

void nlo_snapshot_store_close(nlo_snapshot_store* store)
{
    if (store == NULL) {
        return;
    }

    if (store->db != NULL) {
        (void)nlo_snapshot_store_flush(store);
    }
    if (store->insert_chunk_stmt != NULL) {
        sqlite3_finalize(store->insert_chunk_stmt);
        store->insert_chunk_stmt = NULL;
    }
    if (store->update_run_stmt != NULL) {
        sqlite3_finalize(store->update_run_stmt);
        store->update_run_stmt = NULL;
    }
    if (store->upsert_final_output_stmt != NULL) {
        sqlite3_finalize(store->upsert_final_output_stmt);
        store->upsert_final_output_stmt = NULL;
    }
    if (store->page_count_stmt != NULL) {
        sqlite3_finalize(store->page_count_stmt);
        store->page_count_stmt = NULL;
    }
    if (store->page_size_stmt != NULL) {
        sqlite3_finalize(store->page_size_stmt);
        store->page_size_stmt = NULL;
    }
    if (store->db != NULL) {
        sqlite3_close(store->db);
        store->db = NULL;
    }

    free(store->chunk_buffer);
    free(store);
}

int nlo_snapshot_store_is_available(void)
{
    return 1;
}

#else

struct nlo_snapshot_store {
    int unused;
};

nlo_snapshot_store* nlo_snapshot_store_open(const nlo_snapshot_store_open_params* params)
{
    (void)params;
    return NULL;
}

nlo_snapshot_store_status nlo_snapshot_store_write_record(
    nlo_snapshot_store* store,
    size_t record_index,
    const nlo_complex* record,
    size_t num_time_samples
)
{
    (void)store;
    (void)record_index;
    (void)record;
    (void)num_time_samples;
    return NLO_SNAPSHOT_STORE_STATUS_ERROR;
}

nlo_snapshot_store_status nlo_snapshot_store_write_final_output_field(
    nlo_snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
)
{
    (void)store;
    (void)field;
    (void)num_time_samples;
    return NLO_SNAPSHOT_STORE_STATUS_ERROR;
}

nlo_snapshot_store_status nlo_snapshot_store_flush(nlo_snapshot_store* store)
{
    (void)store;
    return NLO_SNAPSHOT_STORE_STATUS_ERROR;
}

void nlo_snapshot_store_get_result(const nlo_snapshot_store* store, nlo_storage_result* out_result)
{
    (void)store;
    if (out_result != NULL) {
        *out_result = (nlo_storage_result){0};
    }
}

void nlo_snapshot_store_close(nlo_snapshot_store* store)
{
    (void)store;
}

int nlo_snapshot_store_is_available(void)
{
    return 0;
}

#endif
