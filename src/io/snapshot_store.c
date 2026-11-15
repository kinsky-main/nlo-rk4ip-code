/**
 * @file snapshot_store.c
 * @brief SQLite-backed run storage: reduced record maps, raw chunks and the `runs` row.
 *
 * The table layout is documented in docs/nlolib_run_database.md.
 */

#include "io/snapshot_store.h"
#include "io/log_sink.h"
#include "core/sim_dimensions_internal.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(HAVE_SQLITE3)
#include <sqlite3.h>
#include <fftw3.h>
#include "io/run_store.h"

#ifndef STORAGE_DEFAULT_DB_MAX_BYTES
#define STORAGE_DEFAULT_DB_MAX_BYTES (50ull * 1024ull * 1024ull * 1024ull)
#endif

#ifndef STORAGE_DEFAULT_CHUNK_RECORDS
#define STORAGE_DEFAULT_CHUNK_RECORDS 16u
#endif

#ifndef STORAGE_DB_SIZE_SAFETY_BYTES
#define STORAGE_DB_SIZE_SAFETY_BYTES (1ull << 20)
#endif

#define SNAPSHOT_TWO_PI 6.283185307179586476925286766559

struct snapshot_store {
    sqlite3* db;
    sqlite3_stmt* insert_chunk_stmt;
    sqlite3_stmt* update_run_stmt;
    sqlite3_stmt* upsert_final_output_stmt;
    sqlite3_stmt* insert_map_stmt;
    sqlite3_stmt* page_count_stmt;
    sqlite3_stmt* page_size_stmt;
    const sim_config* config;
    nlo_complex* chunk_buffer;
    size_t num_time_samples;
    size_t num_recorded_samples;
    size_t chunk_records;
    size_t chunk_fill;
    size_t chunk_start_record;
    size_t chunk_index;
    size_t max_bytes;
    storage_db_cap_policy cap_policy;
    int accept_writes;
    int in_txn;
    size_t records_in_txn;

    storage_record_mode record_mode;
    int store_maps;
    int store_raw;
    int store_final_field;
    size_t nt;
    size_t nx;
    size_t ny;
    double dt;
    double dx;
    double dy;
    size_t stride_t;
    size_t stride_x;
    size_t stride_y;
    size_t raw_nt;
    size_t raw_nx;
    size_t raw_ny;
    size_t raw_samples;
    size_t raw_records;
    size_t records_seen;
    double* record_z;
    double* map_power;
    double* map_spectrum;
    double* map_fluence_x;
    double* map_fluence_y;
    fftw_complex* fft_in;
    fftw_complex* fft_out;
    fftw_plan fft_plan;

    storage_result result;
};

static int checked_mul_size_t(size_t a, size_t b, size_t* out)
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

static uint64_t fnv1a_seed(void)
{
    return UINT64_C(1469598103934665603);
}

static uint64_t fnv1a_step(uint64_t hash, const void* bytes, size_t len)
{
    const unsigned char* ptr = (const unsigned char*)bytes;
    for (size_t i = 0u; i < len; ++i) {
        hash ^= (uint64_t)ptr[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void hash_cstr(uint64_t* hash, const char* text)
{
    const char* safe_text = (text != NULL) ? text : "";
    *hash = fnv1a_step(*hash, safe_text, strlen(safe_text));
}

static void hash_hex16(uint64_t hash, char out_hex[17])
{
    (void)snprintf(out_hex, 17u, "%016llx", (unsigned long long)hash);
}

static void compute_system_hash(
    const sim_config* config,
    const execution_options* exec_options,
    size_t num_time_samples,
    char out_hash[17]
)
{
    uint64_t hash = fnv1a_seed();
    hash = fnv1a_step(hash, &num_time_samples, sizeof(num_time_samples));
    if (config != NULL) {
        hash = fnv1a_step(hash, &config->time.nt, sizeof(config->time.nt));
        hash = fnv1a_step(hash, &config->spatial.nx, sizeof(config->spatial.nx));
        hash = fnv1a_step(hash, &config->spatial.ny, sizeof(config->spatial.ny));
    }
    if (exec_options != NULL) {
        hash = fnv1a_step(hash, &exec_options->backend_type, sizeof(exec_options->backend_type));
        hash = fnv1a_step(hash, &exec_options->fft_backend, sizeof(exec_options->fft_backend));
        hash = fnv1a_step(hash, &exec_options->device_heap_fraction, sizeof(exec_options->device_heap_fraction));
        hash = fnv1a_step(hash, &exec_options->record_ring_target, sizeof(exec_options->record_ring_target));
        hash = fnv1a_step(hash,
                              &exec_options->forced_device_budget_bytes,
                              sizeof(exec_options->forced_device_budget_bytes));
    }
    hash_hex16(hash, out_hash);
}

static void compute_config_hash(const sim_config* config, char out_hash[17])
{
    uint64_t hash = fnv1a_seed();
    if (config != NULL) {
        hash = fnv1a_step(hash,
                              &config->propagation.starting_step_size,
                              sizeof(config->propagation.starting_step_size));
        hash = fnv1a_step(hash,
                              &config->propagation.max_step_size,
                              sizeof(config->propagation.max_step_size));
        hash = fnv1a_step(hash,
                              &config->propagation.min_step_size,
                              sizeof(config->propagation.min_step_size));
        hash = fnv1a_step(hash,
                              &config->propagation.error_tolerance,
                              sizeof(config->propagation.error_tolerance));
        hash = fnv1a_step(hash,
                              &config->propagation.propagation_distance,
                              sizeof(config->propagation.propagation_distance));
        hash = fnv1a_step(hash, &config->tensor.nt, sizeof(config->tensor.nt));
        hash = fnv1a_step(hash, &config->tensor.nx, sizeof(config->tensor.nx));
        hash = fnv1a_step(hash, &config->tensor.ny, sizeof(config->tensor.ny));
        hash = fnv1a_step(hash, &config->tensor.layout, sizeof(config->tensor.layout));
        hash = fnv1a_step(hash, &config->time.nt, sizeof(config->time.nt));
        hash = fnv1a_step(hash, &config->time.pulse_period, sizeof(config->time.pulse_period));
        hash = fnv1a_step(hash, &config->time.delta_time, sizeof(config->time.delta_time));
        hash = fnv1a_step(hash, &config->spatial.nx, sizeof(config->spatial.nx));
        hash = fnv1a_step(hash, &config->spatial.ny, sizeof(config->spatial.ny));
        hash = fnv1a_step(hash, &config->spatial.delta_x, sizeof(config->spatial.delta_x));
        hash = fnv1a_step(hash, &config->spatial.delta_y, sizeof(config->spatial.delta_y));
        hash = fnv1a_step(hash, &config->runtime.num_constants, sizeof(config->runtime.num_constants));
        hash = fnv1a_step(hash, config->runtime.constants, sizeof(config->runtime.constants));
        hash_cstr(&hash, config->runtime.linear_factor_expr);
        hash_cstr(&hash, config->runtime.linear_expr);
        hash_cstr(&hash, config->runtime.potential_expr);
        hash_cstr(&hash, config->runtime.dispersion_factor_expr);
        hash_cstr(&hash, config->runtime.dispersion_expr);
        hash_cstr(&hash, config->runtime.nonlinear_expr);
    }
    hash_hex16(hash, out_hash);
}

static void make_run_id(char out_run_id[STORAGE_RUN_ID_MAX], const char* explicit_id)
{
    if (out_run_id == NULL) {
        return;
    }
    if (explicit_id != NULL && explicit_id[0] != '\0') {
        (void)snprintf(out_run_id, STORAGE_RUN_ID_MAX, "%s", explicit_id);
        return;
    }

    const unsigned long long now = (unsigned long long)time(NULL);
    static unsigned long long serial = 0ull;
    serial += 1ull;
    (void)snprintf(out_run_id, STORAGE_RUN_ID_MAX, "run-%llx-%llx", now, serial);
}

static int sqlite_exec(sqlite3* db, const char* sql)
{
    char* err = NULL;
    const int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        if (err != NULL) {
            log_emit(LOG_LEVEL_ERROR,
                         "[nlolib][io] sqlite exec error:\n"
                         "  - detail: %s",
                         err);
        }
        sqlite3_free(err);
        return -1;
    }
    return 0;
}

static int prepare(sqlite3* db, const char* sql, sqlite3_stmt** out_stmt)
{
    if (db == NULL || sql == NULL || out_stmt == NULL) {
        return -1;
    }
    *out_stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, out_stmt, NULL) != SQLITE_OK) {
        log_emit(LOG_LEVEL_ERROR, "[nlolib][io] sqlite prepare error: %s", sqlite3_errmsg(db));
        return -1;
    }
    return 0;
}

static int bind_text(sqlite3_stmt* stmt, int index, const char* text)
{
    const char* safe = (text != NULL) ? text : "";
    return (sqlite3_bind_text(stmt, index, safe, -1, SQLITE_TRANSIENT) == SQLITE_OK) ? 0 : -1;
}

static int bind_doubles(sqlite3_stmt* stmt, int index, const double* values, size_t count)
{
    const size_t bytes = count * sizeof(double);
    if (bytes > (size_t)INT_MAX) {
        return -1;
    }
    if (count == 0u) {
        return (sqlite3_bind_zeroblob(stmt, index, 0) == SQLITE_OK) ? 0 : -1;
    }
    return (sqlite3_bind_blob(stmt, index, values, (int)bytes, SQLITE_TRANSIENT) == SQLITE_OK) ? 0 : -1;
}

static int step_done(sqlite3_stmt* stmt)
{
    const int rc = sqlite3_step(stmt);
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
    return (rc == SQLITE_DONE) ? 0 : -1;
}

static int store_txn_begin(snapshot_store* store)
{
    if (store->in_txn) {
        return 0;
    }
    if (sqlite_exec(store->db, "BEGIN;") != 0) {
        return -1;
    }
    store->in_txn = 1;
    store->records_in_txn = 0u;
    return 0;
}

static int store_txn_commit(snapshot_store* store)
{
    if (!store->in_txn) {
        return 0;
    }
    store->in_txn = 0;
    store->records_in_txn = 0u;
    return sqlite_exec(store->db, "COMMIT;");
}

static int query_db_size(sqlite3_stmt* page_count_stmt, sqlite3_stmt* page_size_stmt, size_t* out_size)
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

static int store_update_run(snapshot_store* store)
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
    if (bind_text(store->update_run_stmt, 5, store->result.run_id) != 0) {
        return -1;
    }
    return step_done(store->update_run_stmt);
}

static snapshot_store_status store_flush_chunk(snapshot_store* store)
{
    if (store == NULL || store->chunk_fill == 0u) {
        return SNAPSHOT_STORE_STATUS_OK;
    }

    if (!store->accept_writes) {
        store->result.truncated = 1;
        store->chunk_fill = 0u;
        return SNAPSHOT_STORE_STATUS_SOFT_LIMIT;
    }

    size_t payload_bytes = 0u;
    if (checked_mul_size_t(store->chunk_fill, store->raw_samples, &payload_bytes) != 0 ||
        checked_mul_size_t(payload_bytes, sizeof(nlo_complex), &payload_bytes) != 0 ||
        payload_bytes > (size_t)INT_MAX) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    size_t db_size = 0u;
    if (query_db_size(store->page_count_stmt, store->page_size_stmt, &db_size) == 0) {
        store->result.db_size_bytes = db_size;
    }

    if (store->max_bytes > 0u) {
        const unsigned long long projected =
            (unsigned long long)store->result.db_size_bytes +
            (unsigned long long)payload_bytes +
            (unsigned long long)STORAGE_DB_SIZE_SAFETY_BYTES;
        if (projected > (unsigned long long)store->max_bytes) {
            if (store->cap_policy == STORAGE_DB_CAP_POLICY_STOP_WRITES) {
                store->result.truncated = 1;
                store->accept_writes = 0;
                store->chunk_fill = 0u;
                (void)store_update_run(store);
                return SNAPSHOT_STORE_STATUS_SOFT_LIMIT;
            }
            return SNAPSHOT_STORE_STATUS_ERROR;
        }
    }

    if (store_txn_begin(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (bind_text(store->insert_chunk_stmt, 1, store->result.run_id) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->insert_chunk_stmt, 2, (sqlite3_int64)store->chunk_index) != SQLITE_OK) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->insert_chunk_stmt, 3, (sqlite3_int64)store->chunk_start_record) != SQLITE_OK) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->insert_chunk_stmt, 4, (sqlite3_int64)store->chunk_fill) != SQLITE_OK) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_blob(store->insert_chunk_stmt,
                          5,
                          store->chunk_buffer,
                          (int)payload_bytes,
                          SQLITE_TRANSIENT) != SQLITE_OK) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (step_done(store->insert_chunk_stmt) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    store->raw_records += store->chunk_fill;
    store->result.records_spilled += store->chunk_fill;
    store->result.chunks_written += 1u;
    store->chunk_index += 1u;
    store->chunk_fill = 0u;

    if (query_db_size(store->page_count_stmt, store->page_size_stmt, &db_size) == 0) {
        store->result.db_size_bytes = db_size;
    }

    if (store->max_bytes > 0u &&
        store->result.db_size_bytes > store->max_bytes &&
        store->cap_policy == STORAGE_DB_CAP_POLICY_STOP_WRITES) {
        store->result.truncated = 1;
        store->accept_writes = 0;
    }

    if (store_update_run(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    return SNAPSHOT_STORE_STATUS_OK;
}

// MARK: Schema

static const char* k_runs_param_columns[][2] = {
    {"nlolib_version", "TEXT"},
    {"nlolib_binding", "TEXT"},
    {"nlolib_backend", "TEXT"},
    {"nlolib_run_id", "TEXT"},
    {"nlolib_nt", "INTEGER"},
    {"nlolib_nx", "INTEGER"},
    {"nlolib_ny", "INTEGER"},
    {"nlolib_dt", "REAL"},
    {"nlolib_dx", "REAL"},
    {"nlolib_dy", "REAL"},
    {"nlolib_z_end", "REAL"},
    {"nlolib_h_start", "REAL"},
    {"nlolib_h_min", "REAL"},
    {"nlolib_h_max", "REAL"},
    {"nlolib_tol", "REAL"},
    {"nlolib_linear_factor_expr", "TEXT"},
    {"nlolib_linear_expr", "TEXT"},
    {"nlolib_potential_expr", "TEXT"},
    {"nlolib_dispersion_factor_expr", "TEXT"},
    {"nlolib_dispersion_expr", "TEXT"},
    {"nlolib_nonlinear_expr", "TEXT"},
    {"nlolib_constants", "TEXT"},
    {"nlolib_nonlinear_model", "TEXT"},
    {"nlolib_gamma", "REAL"},
    {"nlolib_raman_fraction", "REAL"},
    {"nlolib_raman_tau1", "REAL"},
    {"nlolib_raman_tau2", "REAL"},
    {"nlolib_shock_omega0", "REAL"},
    {"nlolib_input_sha256", "TEXT"},
    {"nlolib_potential_sha256", "TEXT"},
    {"nlolib_num_records", "INTEGER"},
    {"nlolib_records_written", "INTEGER"},
    {"nlolib_record_mode", "TEXT"},
    {"nlolib_raw_stride_t", "INTEGER"},
    {"nlolib_raw_stride_x", "INTEGER"},
    {"nlolib_raw_stride_y", "INTEGER"},
    {"nlolib_final_field", "INTEGER"},
    {"nlolib_truncated", "INTEGER"}
};

#define RUNS_PARAM_COUNT (sizeof(k_runs_param_columns) / sizeof(k_runs_param_columns[0]))

static int table_has_column(sqlite3* db, const char* table, const char* column, int* out_has)
{
    char sql[128];
    (void)snprintf(sql, sizeof(sql), "PRAGMA table_info(%s);", table);
    sqlite3_stmt* stmt = NULL;
    if (prepare(db, sql, &stmt) != 0) {
        return -1;
    }
    *out_has = 0;
    for (;;) {
        const int rc = sqlite3_step(stmt);
        if (rc == SQLITE_DONE) {
            break;
        }
        if (rc != SQLITE_ROW) {
            sqlite3_finalize(stmt);
            return -1;
        }
        const char* col_name = (const char*)sqlite3_column_text(stmt, 1);
        if (col_name != NULL && sqlite3_stricmp(col_name, column) == 0) {
            *out_has = 1;
        }
    }
    sqlite3_finalize(stmt);
    return 0;
}

static int ensure_column(sqlite3* db, const char* table, const char* column, const char* decl)
{
    int has = 0;
    if (table_has_column(db, table, column, &has) != 0) {
        return -1;
    }
    if (has) {
        return 0;
    }
    char sql[256];
    (void)snprintf(sql, sizeof(sql), "ALTER TABLE %s ADD COLUMN \"%s\" %s;", table, column, decl);
    return sqlite_exec(db, sql);
}

int snapshot_store_ensure_schema(struct sqlite3* db)
{
    static const char* schema_sql =
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
        "  linear_factor_expr TEXT,"
        "  linear_expr TEXT,"
        "  potential_expr TEXT,"
        "  disp_factor_expr TEXT,"
        "  disp_expr TEXT,"
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
        "  key TEXT,"
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
        ");"
        "CREATE TABLE IF NOT EXISTS io_record_layout ("
        "  run_id TEXT PRIMARY KEY,"
        "  record_mode TEXT NOT NULL,"
        "  nt INTEGER NOT NULL,"
        "  nx INTEGER NOT NULL,"
        "  ny INTEGER NOT NULL,"
        "  dt REAL NOT NULL,"
        "  dx REAL NOT NULL,"
        "  dy REAL NOT NULL,"
        "  num_records INTEGER NOT NULL DEFAULT 0,"
        "  raw_stride_t INTEGER NOT NULL,"
        "  raw_stride_x INTEGER NOT NULL,"
        "  raw_stride_y INTEGER NOT NULL,"
        "  raw_nt INTEGER NOT NULL,"
        "  raw_nx INTEGER NOT NULL,"
        "  raw_ny INTEGER NOT NULL,"
        "  raw_records INTEGER NOT NULL DEFAULT 0,"
        "  record_z BLOB NOT NULL,"
        "  t_axis BLOB NOT NULL,"
        "  omega_axis BLOB NOT NULL,"
        "  x_axis BLOB NOT NULL,"
        "  y_axis BLOB NOT NULL,"
        "  FOREIGN KEY(run_id) REFERENCES io_runs(run_id) ON DELETE CASCADE"
        ");"
        "CREATE TABLE IF NOT EXISTS io_record_maps ("
        "  run_id TEXT NOT NULL,"
        "  record_index INTEGER NOT NULL,"
        "  z REAL NOT NULL,"
        "  power BLOB NOT NULL,"
        "  spectrum BLOB NOT NULL,"
        "  fluence_x BLOB NOT NULL,"
        "  fluence_y BLOB NOT NULL,"
        "  PRIMARY KEY(run_id, record_index),"
        "  FOREIGN KEY(run_id) REFERENCES io_runs(run_id) ON DELETE CASCADE"
        ");"
        "CREATE TABLE IF NOT EXISTS runs ("
        "    key TEXT PRIMARY KEY,"
        "    kind TEXT NOT NULL,"
        "    file TEXT NOT NULL,"
        "    created_at TEXT,"
        "    git_commit TEXT,"
        "    julia_version TEXT"
        ");";

    if (db == NULL || sqlite_exec(db, schema_sql) != 0) {
        return -1;
    }

    /* Columns added after the first schema version; older files get them on open. */
    if (ensure_column(db, "io_sim_config", "linear_factor_expr", "TEXT") != 0 ||
        ensure_column(db, "io_sim_config", "linear_expr", "TEXT") != 0 ||
        ensure_column(db, "io_sim_config", "potential_expr", "TEXT") != 0 ||
        ensure_column(db, "io_runs", "key", "TEXT") != 0) {
        return -1;
    }
    return sqlite_exec(db, "CREATE INDEX IF NOT EXISTS io_runs_key ON io_runs(key);");
}

static int store_insert_metadata(
    snapshot_store* store,
    const sim_config* config,
    const execution_options* exec_options,
    size_t num_time_samples,
    size_t num_recorded_samples
)
{
    if (store == NULL || store->db == NULL || config == NULL || exec_options == NULL) {
        return -1;
    }

    char system_hash[17];
    char config_hash[17];
    compute_system_hash(config, exec_options, num_time_samples, system_hash);
    compute_config_hash(config, config_hash);

    sqlite3_stmt* stmt = NULL;
    if (prepare(store->db,
                    "INSERT INTO io_system_config("
                    "system_hash, backend_type, fft_backend, num_time_samples, nt, nx, ny, "
                    "device_heap_fraction, record_ring_target, forced_device_budget_bytes"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(system_hash) DO NOTHING;",
                    &stmt) != 0) {
        return -1;
    }
    if (bind_text(stmt, 1, system_hash) != 0 ||
        sqlite3_bind_int(stmt, 2, (int)exec_options->backend_type) != SQLITE_OK ||
        sqlite3_bind_int(stmt, 3, (int)exec_options->fft_backend) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 4, (sqlite3_int64)num_time_samples) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 5, (sqlite3_int64)config->time.nt) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 6, (sqlite3_int64)config->spatial.nx) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 7, (sqlite3_int64)config->spatial.ny) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 8, exec_options->device_heap_fraction) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 9, (sqlite3_int64)exec_options->record_ring_target) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 10, (sqlite3_int64)exec_options->forced_device_budget_bytes) != SQLITE_OK ||
        step_done(stmt) != 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    if (prepare(store->db,
                    "INSERT INTO io_sim_config("
                    "config_hash, z_end, h_start, h_max, h_min, tolerance, pulse_period, delta_time, "
                    "linear_factor_expr, linear_expr, potential_expr, "
                    "disp_factor_expr, disp_expr, nonlinear_expr, constants_blob"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(config_hash) DO NOTHING;",
                    &stmt) != 0) {
        return -1;
    }
    if (bind_text(stmt, 1, config_hash) != 0 ||
        sqlite3_bind_double(stmt, 2, config->propagation.propagation_distance) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 3, config->propagation.starting_step_size) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 4, config->propagation.max_step_size) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 5, config->propagation.min_step_size) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 6, config->propagation.error_tolerance) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 7, config->time.pulse_period) != SQLITE_OK ||
        sqlite3_bind_double(stmt, 8, config->time.delta_time) != SQLITE_OK ||
        bind_text(stmt, 9, config->runtime.linear_factor_expr) != 0 ||
        bind_text(stmt, 10, config->runtime.linear_expr) != 0 ||
        bind_text(stmt, 11, config->runtime.potential_expr) != 0 ||
        bind_text(stmt, 12, config->runtime.dispersion_factor_expr) != 0 ||
        bind_text(stmt, 13, config->runtime.dispersion_expr) != 0 ||
        bind_text(stmt, 14, config->runtime.nonlinear_expr) != 0 ||
        sqlite3_bind_blob(stmt,
                          15,
                          config->runtime.constants,
                          (int)sizeof(config->runtime.constants),
                          SQLITE_TRANSIENT) != SQLITE_OK ||
        step_done(stmt) != 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    if (prepare(store->db,
                    "INSERT INTO io_runs("
                    "run_id, system_hash, config_hash, num_recorded_samples, num_time_samples, "
                    "chunk_records, cap_policy, max_bytes, records_written, chunks_written, db_size_bytes, truncated"
                    ") VALUES(?,?,?,?,?,?,?,?,0,0,0,0) "
                    "ON CONFLICT(run_id) DO NOTHING;",
                    &stmt) != 0) {
        return -1;
    }
    if (bind_text(stmt, 1, store->result.run_id) != 0 ||
        bind_text(stmt, 2, system_hash) != 0 ||
        bind_text(stmt, 3, config_hash) != 0 ||
        sqlite3_bind_int64(stmt, 4, (sqlite3_int64)num_recorded_samples) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 5, (sqlite3_int64)num_time_samples) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 6, (sqlite3_int64)store->chunk_records) != SQLITE_OK ||
        sqlite3_bind_int(stmt, 7, (int)store->cap_policy) != SQLITE_OK ||
        sqlite3_bind_int64(stmt, 8, (sqlite3_int64)store->max_bytes) != SQLITE_OK ||
        step_done(stmt) != 0) {
        sqlite3_finalize(stmt);
        return -1;
    }
    sqlite3_finalize(stmt);

    return 0;
}

static const char* record_mode_name(storage_record_mode mode)
{
    switch (mode) {
        case STORAGE_RECORD_MODE_RAW:
            return "raw";
        case STORAGE_RECORD_MODE_REDUCED_AND_RAW:
            return "reduced+raw";
        case STORAGE_RECORD_MODE_REDUCED:
        default:
            return "reduced";
    }
}

/* Axes as the solver defines them: centered t, x, y and the angular frequency in ascending order. */
static int store_insert_layout(snapshot_store* store)
{
    const size_t n_axes = store->nt + store->nt + store->nx + store->ny;
    double* axes = (double*)calloc(n_axes, sizeof(double));
    if (axes == NULL) {
        return -1;
    }
    double* t_axis = axes;
    double* omega_axis = t_axis + store->nt;
    double* x_axis = omega_axis + store->nt;
    double* y_axis = x_axis + store->nx;
    const double t_center = 0.5 * (double)(store->nt - 1u);
    const double omega_step = SNAPSHOT_TWO_PI / ((double)store->nt * store->dt);
    const size_t negatives = store->nt / 2u;
    for (size_t i = 0u; i < store->nt; ++i) {
        t_axis[i] = ((double)i - t_center) * store->dt;
        omega_axis[i] = ((double)i - (double)negatives) * omega_step;
    }
    for (size_t i = 0u; i < store->nx; ++i) {
        x_axis[i] = ((double)i - 0.5 * (double)(store->nx - 1u)) * store->dx;
    }
    for (size_t i = 0u; i < store->ny; ++i) {
        y_axis[i] = ((double)i - 0.5 * (double)(store->ny - 1u)) * store->dy;
    }

    sqlite3_stmt* stmt = NULL;
    int rc = prepare(store->db,
                     "INSERT OR REPLACE INTO io_record_layout("
                     "run_id, record_mode, nt, nx, ny, dt, dx, dy, num_records, "
                     "raw_stride_t, raw_stride_x, raw_stride_y, raw_nt, raw_nx, raw_ny, raw_records, "
                     "record_z, t_axis, omega_axis, x_axis, y_axis"
                     ") VALUES(?,?,?,?,?,?,?,?,0,?,?,?,?,?,?,0,zeroblob(0),?,?,?,?);",
                     &stmt);
    if (rc == 0) {
        const int raw = store->store_raw;
        rc = (bind_text(stmt, 1, store->result.run_id) != 0 ||
              bind_text(stmt, 2, record_mode_name(store->record_mode)) != 0 ||
              sqlite3_bind_int64(stmt, 3, (sqlite3_int64)store->nt) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 4, (sqlite3_int64)store->nx) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 5, (sqlite3_int64)store->ny) != SQLITE_OK ||
              sqlite3_bind_double(stmt, 6, store->dt) != SQLITE_OK ||
              sqlite3_bind_double(stmt, 7, store->dx) != SQLITE_OK ||
              sqlite3_bind_double(stmt, 8, store->dy) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 9, (sqlite3_int64)store->stride_t) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 10, (sqlite3_int64)store->stride_x) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 11, (sqlite3_int64)store->stride_y) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 12, (sqlite3_int64)(raw ? store->raw_nt : 0u)) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 13, (sqlite3_int64)(raw ? store->raw_nx : 0u)) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 14, (sqlite3_int64)(raw ? store->raw_ny : 0u)) != SQLITE_OK ||
              bind_doubles(stmt, 15, t_axis, store->nt) != 0 ||
              bind_doubles(stmt, 16, omega_axis, store->nt) != 0 ||
              bind_doubles(stmt, 17, x_axis, store->nx) != 0 ||
              bind_doubles(stmt, 18, y_axis, store->ny) != 0 ||
              step_done(stmt) != 0) ? -1 : 0;
        sqlite3_finalize(stmt);
    }
    free(axes);
    return rc;
}

// MARK: Reduced maps

static void store_compute_maps(snapshot_store* store, const nlo_complex* record)
{
    const size_t nt = store->nt;
    const size_t nx = store->nx;
    const size_t ny = store->ny;
    memset(store->map_power, 0, nt * sizeof(double));
    memset(store->map_spectrum, 0, nt * sizeof(double));
    memset(store->map_fluence_x, 0, nx * sizeof(double));
    memset(store->map_fluence_y, 0, ny * sizeof(double));

    for (size_t x = 0u; x < nx; ++x) {
        for (size_t y = 0u; y < ny; ++y) {
            const nlo_complex* line = record + ((x * ny) + y) * nt;
            double line_sum = 0.0;
            for (size_t t = 0u; t < nt; ++t) {
                const double intensity = (line[t].re * line[t].re) + (line[t].im * line[t].im);
                store->map_power[t] += intensity;
                line_sum += intensity;
            }
            store->map_fluence_x[x] += line_sum;
            store->map_fluence_y[y] += line_sum;
        }
    }

    /* Unnormalised forward DFT along t of every (x, y) line, the solver's own transform. */
    fftw_execute_dft(store->fft_plan, (fftw_complex*)(void*)record, store->fft_out);
    const size_t shift = nt - (nt / 2u);
    const size_t lines = nx * ny;
    for (size_t l = 0u; l < lines; ++l) {
        const fftw_complex* spectrum = store->fft_out + l * nt;
        for (size_t j = 0u; j < nt; ++j) {
            const size_t k = (j + shift) % nt;
            store->map_spectrum[j] += (spectrum[k][0] * spectrum[k][0]) + (spectrum[k][1] * spectrum[k][1]);
        }
    }
}

static double store_record_z(const snapshot_store* store, size_t record_index)
{
    return (store->record_z != NULL && record_index < store->num_recorded_samples)
               ? store->record_z[record_index]
               : 0.0;
}

void snapshot_store_set_record_schedule(
    snapshot_store* store,
    const double* explicit_record_z,
    size_t explicit_record_z_count
)
{
    if (store == NULL || store->record_z == NULL) {
        return;
    }
    const double z_end = (store->config != NULL) ? store->config->propagation.propagation_distance : 0.0;
    const size_t n = store->num_recorded_samples;
    for (size_t i = 0u; i < n; ++i) {
        if (explicit_record_z != NULL && explicit_record_z_count > 0u) {
            store->record_z[i] = (i < explicit_record_z_count) ? explicit_record_z[i] : z_end;
        } else if (n == 1u) {
            store->record_z[i] = z_end;
        } else {
            store->record_z[i] = z_end * ((double)i / (double)(n - 1u));
        }
    }
}

// MARK: Open / write

snapshot_store* snapshot_store_open(const snapshot_store_open_params* params)
{
    if (params == NULL || params->storage_options == NULL || params->storage_options->sqlite_path == NULL ||
        params->config == NULL) {
        return NULL;
    }

    const storage_options* options = params->storage_options;
    if (options->sqlite_path[0] == '\0') {
        return NULL;
    }

    snapshot_store* store = (snapshot_store*)calloc(1u, sizeof(snapshot_store));
    if (store == NULL) {
        return NULL;
    }

    store->config = params->config;
    store->num_time_samples = params->num_time_samples;
    store->num_recorded_samples = params->num_recorded_samples;
    store->chunk_records =
        (options->chunk_records > 0u)
            ? options->chunk_records
            : STORAGE_DEFAULT_CHUNK_RECORDS;
    if (store->chunk_records > params->num_recorded_samples) {
        store->chunk_records = params->num_recorded_samples;
    }
    if (store->chunk_records == 0u) {
        store->chunk_records = 1u;
    }

    store->max_bytes =
        (options->sqlite_max_bytes > 0u)
            ? options->sqlite_max_bytes
            : (size_t)STORAGE_DEFAULT_DB_MAX_BYTES;
    store->cap_policy = options->cap_policy;
    store->accept_writes = 1;

    store->record_mode = options->record_mode;
    if (store->record_mode != STORAGE_RECORD_MODE_RAW &&
        store->record_mode != STORAGE_RECORD_MODE_REDUCED_AND_RAW) {
        store->record_mode = STORAGE_RECORD_MODE_REDUCED;
    }
    store->store_maps = (store->record_mode != STORAGE_RECORD_MODE_RAW);
    store->store_raw = (store->record_mode != STORAGE_RECORD_MODE_REDUCED);
    store->store_final_field = (options->log_final_output_field_to_db != 0);

    int explicit_nd = 0;
    if (resolve_sim_dimensions_internal(params->config,
                                            params->num_time_samples,
                                            &store->nt,
                                            &store->nx,
                                            &store->ny,
                                            &explicit_nd) != 0 ||
        store->nt == 0u || store->nx == 0u || store->ny == 0u ||
        store->nt * store->nx * store->ny != params->num_time_samples) {
        free(store);
        return NULL;
    }
    store->dt = (params->config->time.delta_time > 0.0)
                    ? params->config->time.delta_time
                    : ((params->config->time.pulse_period > 0.0)
                           ? params->config->time.pulse_period / (double)store->nt
                           : 1.0);
    store->dx = (params->config->spatial.delta_x > 0.0) ? params->config->spatial.delta_x : 1.0;
    store->dy = (params->config->spatial.delta_y > 0.0) ? params->config->spatial.delta_y : 1.0;
    store->stride_t = (options->raw_stride_t > 1u) ? options->raw_stride_t : 1u;
    store->stride_x = (options->raw_stride_x > 1u) ? options->raw_stride_x : 1u;
    store->stride_y = (options->raw_stride_y > 1u) ? options->raw_stride_y : 1u;
    store->raw_nt = (store->nt + store->stride_t - 1u) / store->stride_t;
    store->raw_nx = (store->nx + store->stride_x - 1u) / store->stride_x;
    store->raw_ny = (store->ny + store->stride_y - 1u) / store->stride_y;
    store->raw_samples = store->raw_nt * store->raw_nx * store->raw_ny;

    make_run_id(store->result.run_id, options->run_id);

    store->record_z = (double*)calloc(store->num_recorded_samples, sizeof(double));
    if (store->record_z == NULL) {
        snapshot_store_close(store);
        return NULL;
    }
    snapshot_store_set_record_schedule(store, NULL, 0u);

    if (store->store_raw) {
        size_t chunk_elements = 0u;
        if (checked_mul_size_t(store->chunk_records, store->raw_samples, &chunk_elements) != 0) {
            snapshot_store_close(store);
            return NULL;
        }
        store->chunk_buffer = (nlo_complex*)calloc(chunk_elements, sizeof(nlo_complex));
        if (store->chunk_buffer == NULL) {
            snapshot_store_close(store);
            return NULL;
        }
    }

    if (store->store_maps) {
        store->map_power = (double*)calloc(2u * store->nt + store->nx + store->ny, sizeof(double));
        store->fft_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * store->num_time_samples);
        store->fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * store->num_time_samples);
        if (store->map_power == NULL || store->fft_in == NULL || store->fft_out == NULL) {
            snapshot_store_close(store);
            return NULL;
        }
        store->map_spectrum = store->map_power + store->nt;
        store->map_fluence_x = store->map_spectrum + store->nt;
        store->map_fluence_y = store->map_fluence_x + store->nx;
        const int n = (int)store->nt;
        store->fft_plan = fftw_plan_many_dft(1, &n, (int)(store->nx * store->ny),
                                             store->fft_in, NULL, 1, n,
                                             store->fft_out, NULL, 1, n,
                                             FFTW_FORWARD, FFTW_ESTIMATE | FFTW_UNALIGNED);
        if (store->fft_plan == NULL) {
            snapshot_store_close(store);
            return NULL;
        }
    }

    /* Staging files live on the share too: write them with the run store's VFS (see run_store_sqlite_vfs). */
    if (sqlite3_open_v2(options->sqlite_path,
                        &store->db,
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                        run_store_sqlite_vfs()) != SQLITE_OK) {
        snapshot_store_close(store);
        return NULL;
    }
    sqlite3_busy_timeout(store->db, 60000);

    /* Rollback journal only: WAL needs shared memory that network shares do not provide. */
    if (sqlite_exec(store->db, "PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;") != 0 ||
        snapshot_store_ensure_schema(store->db) != 0 ||
        store_insert_metadata(store,
                                  params->config,
                                  params->exec_options,
                                  params->num_time_samples,
                                  params->num_recorded_samples) != 0 ||
        store_insert_layout(store) != 0 ||
        prepare(store->db,
                    "INSERT INTO io_record_chunks(run_id, chunk_index, record_start, record_count, payload)"
                    " VALUES(?,?,?,?,?);",
                    &store->insert_chunk_stmt) != 0 ||
        prepare(store->db,
                    "UPDATE io_runs SET records_written=?, chunks_written=?, db_size_bytes=?, truncated=? "
                    "WHERE run_id=?;",
                    &store->update_run_stmt) != 0 ||
        prepare(store->db,
                    "INSERT INTO io_final_output_fields(run_id, num_time_samples, payload) "
                    "VALUES(?,?,?) "
                    "ON CONFLICT(run_id) DO UPDATE SET "
                    "num_time_samples=excluded.num_time_samples, "
                    "payload=excluded.payload;",
                    &store->upsert_final_output_stmt) != 0 ||
        prepare(store->db,
                    "INSERT OR REPLACE INTO io_record_maps(run_id, record_index, z, power, spectrum, fluence_x, fluence_y) "
                    "VALUES(?,?,?,?,?,?,?);",
                    &store->insert_map_stmt) != 0 ||
        prepare(store->db, "PRAGMA page_count;", &store->page_count_stmt) != 0 ||
        prepare(store->db, "PRAGMA page_size;", &store->page_size_stmt) != 0) {
        snapshot_store_close(store);
        return NULL;
    }

    (void)query_db_size(store->page_count_stmt, store->page_size_stmt, &store->result.db_size_bytes);
    (void)store_update_run(store);
    return store;
}

static void store_downsample_raw(snapshot_store* store, const nlo_complex* record, nlo_complex* dst)
{
    size_t out = 0u;
    for (size_t x = 0u; x < store->nx; x += store->stride_x) {
        for (size_t y = 0u; y < store->ny; y += store->stride_y) {
            const nlo_complex* line = record + ((x * store->ny) + y) * store->nt;
            for (size_t t = 0u; t < store->nt; t += store->stride_t) {
                dst[out++] = line[t];
            }
        }
    }
}

static snapshot_store_status store_write_map(snapshot_store* store, size_t record_index, const nlo_complex* record)
{
    store_compute_maps(store, record);
    if (store_txn_begin(store) != 0 ||
        bind_text(store->insert_map_stmt, 1, store->result.run_id) != 0 ||
        sqlite3_bind_int64(store->insert_map_stmt, 2, (sqlite3_int64)record_index) != SQLITE_OK ||
        sqlite3_bind_double(store->insert_map_stmt, 3, store_record_z(store, record_index)) != SQLITE_OK ||
        bind_doubles(store->insert_map_stmt, 4, store->map_power, store->nt) != 0 ||
        bind_doubles(store->insert_map_stmt, 5, store->map_spectrum, store->nt) != 0 ||
        bind_doubles(store->insert_map_stmt, 6, store->map_fluence_x, store->nx) != 0 ||
        bind_doubles(store->insert_map_stmt, 7, store->map_fluence_y, store->ny) != 0 ||
        step_done(store->insert_map_stmt) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (!store->store_raw) {
        store->result.records_spilled += 1u;
    }
    return SNAPSHOT_STORE_STATUS_OK;
}

snapshot_store_status snapshot_store_write_record(
    snapshot_store* store,
    size_t record_index,
    const nlo_complex* record,
    size_t num_time_samples
)
{
    if (store == NULL || record == NULL || num_time_samples != store->num_time_samples) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    if (record_index + 1u > store->records_seen) {
        store->records_seen = record_index + 1u;
    }
    store->result.records_captured += 1u;

    snapshot_store_status status = SNAPSHOT_STORE_STATUS_OK;
    if (store->store_maps) {
        status = store_write_map(store, record_index, record);
        if (status == SNAPSHOT_STORE_STATUS_ERROR) {
            return status;
        }
    }

    if (store->store_raw) {
        if (store->chunk_fill == 0u) {
            store->chunk_start_record = record_index;
        }
        store_downsample_raw(store, record, store->chunk_buffer + store->chunk_fill * store->raw_samples);
        store->chunk_fill += 1u;
        if (store->chunk_fill >= store->chunk_records) {
            status = store_flush_chunk(store);
            if (status == SNAPSHOT_STORE_STATUS_ERROR) {
                return status;
            }
        }
    }

    store->records_in_txn += 1u;
    if (store->records_in_txn >= store->chunk_records && store_txn_commit(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    return status;
}

snapshot_store_status snapshot_store_write_final_output_field(
    snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
)
{
    if (store == NULL ||
        field == NULL ||
        store->upsert_final_output_stmt == NULL ||
        num_time_samples != store->num_time_samples) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    size_t payload_bytes = 0u;
    if (checked_mul_size_t(num_time_samples, sizeof(nlo_complex), &payload_bytes) != 0 ||
        payload_bytes > (size_t)INT_MAX) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    if (bind_text(store->upsert_final_output_stmt, 1, store->result.run_id) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_int64(store->upsert_final_output_stmt, 2, (sqlite3_int64)num_time_samples) != SQLITE_OK) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (sqlite3_bind_blob(store->upsert_final_output_stmt,
                          3,
                          field,
                          (int)payload_bytes,
                          SQLITE_TRANSIENT) != SQLITE_OK) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (step_done(store->upsert_final_output_stmt) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    return SNAPSHOT_STORE_STATUS_OK;
}

snapshot_store_status snapshot_store_flush(snapshot_store* store)
{
    if (store == NULL) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    const snapshot_store_status status = store_flush_chunk(store);
    if (status == SNAPSHOT_STORE_STATUS_ERROR) {
        return status;
    }

    if (store_update_run(store) != 0 || store_txn_commit(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    return status;
}

// MARK: Finalize

typedef struct {
    const char* name;
    int type; /* SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_TEXT or SQLITE_NULL */
    sqlite3_int64 i64;
    double f64;
    const char* text;
} runs_value;

static void set_i64(runs_value* v, sqlite3_int64 value)
{
    v->type = SQLITE_INTEGER;
    v->i64 = value;
}

static void set_f64(runs_value* v, double value)
{
    v->type = isfinite(value) ? SQLITE_FLOAT : SQLITE_NULL;
    v->f64 = value;
}

static void set_text(runs_value* v, const char* text)
{
    v->type = (text != NULL && text[0] != '\0') ? SQLITE_TEXT : SQLITE_NULL;
    v->text = text;
}

/* JSON array of the runtime constants, formatted locale-independently. */
static void format_constants(const runtime_operator_params* rt, char* out, size_t cap)
{
    size_t used = 0u;
    out[used++] = '[';
    const size_t count = (rt->num_constants <= RUNTIME_OPERATOR_CONSTANTS_MAX) ? rt->num_constants
                                                                               : RUNTIME_OPERATOR_CONSTANTS_MAX;
    for (size_t i = 0u; i < count && used + 32u < cap; ++i) {
        char number[40];
        if (isfinite(rt->constants[i])) {
            (void)snprintf(number, sizeof(number), "%.17g", rt->constants[i]);
            for (char* c = number; *c != '\0'; ++c) {
                if (*c == ',') {
                    *c = '.';
                }
            }
        } else {
            (void)snprintf(number, sizeof(number), "null");
        }
        used += (size_t)snprintf(out + used, cap - used, "%s%s", (i > 0u) ? "," : "", number);
    }
    (void)snprintf(out + used, cap - used, "]");
}

static int store_insert_runs_row(snapshot_store* store, const snapshot_store_finalize_params* params)
{
    const sim_config* cfg = store->config;
    const runtime_operator_params* rt = &cfg->runtime;
    const int raman = (rt->nonlinear_model == NONLINEAR_MODEL_KERR_RAMAN);
    char constants[RUNTIME_OPERATOR_CONSTANTS_MAX * 26u + 8u];
    format_constants(rt, constants, sizeof(constants));

    runs_value values[RUNS_PARAM_COUNT];
    memset(values, 0, sizeof(values));
    for (size_t i = 0u; i < RUNS_PARAM_COUNT; ++i) {
        values[i].name = k_runs_param_columns[i][0];
        values[i].type = SQLITE_NULL;
    }
    size_t i = 0u;
    set_text(&values[i++], params->version);
    set_text(&values[i++], (params->binding != NULL && params->binding[0] != '\0') ? params->binding : "c");
    set_text(&values[i++], params->backend);
    set_text(&values[i++], store->result.run_id);
    set_i64(&values[i++], (sqlite3_int64)store->nt);
    set_i64(&values[i++], (sqlite3_int64)store->nx);
    set_i64(&values[i++], (sqlite3_int64)store->ny);
    set_f64(&values[i++], store->dt);
    if (store->nx > 1u) {
        set_f64(&values[i], store->dx);
    }
    i += 1u;
    if (store->ny > 1u) {
        set_f64(&values[i], store->dy);
    }
    i += 1u;
    set_f64(&values[i++], cfg->propagation.propagation_distance);
    set_f64(&values[i++], cfg->propagation.starting_step_size);
    set_f64(&values[i++], cfg->propagation.min_step_size);
    set_f64(&values[i++], cfg->propagation.max_step_size);
    set_f64(&values[i++], cfg->propagation.error_tolerance);
    set_text(&values[i++], rt->linear_factor_expr);
    set_text(&values[i++], rt->linear_expr);
    set_text(&values[i++], rt->potential_expr);
    set_text(&values[i++], rt->dispersion_factor_expr);
    set_text(&values[i++], rt->dispersion_expr);
    set_text(&values[i++], rt->nonlinear_expr);
    set_text(&values[i++], constants);
    set_text(&values[i++], raman ? "kerr_raman" : "expr");
    if (raman) {
        set_f64(&values[i], rt->nonlinear_gamma);
        set_f64(&values[i + 1u], rt->raman_fraction);
        set_f64(&values[i + 2u], rt->raman_tau1);
        set_f64(&values[i + 3u], rt->raman_tau2);
        set_f64(&values[i + 4u], rt->shock_omega0);
    }
    i += 5u;
    set_text(&values[i++], params->input_sha256);
    set_text(&values[i++], params->potential_sha256);
    set_i64(&values[i++], (sqlite3_int64)store->num_recorded_samples);
    set_i64(&values[i++], (sqlite3_int64)store->records_seen);
    set_text(&values[i++], record_mode_name(store->record_mode));
    if (store->store_raw) {
        set_i64(&values[i], (sqlite3_int64)store->stride_t);
        set_i64(&values[i + 1u], (sqlite3_int64)store->stride_x);
        set_i64(&values[i + 2u], (sqlite3_int64)store->stride_y);
    }
    i += 3u;
    set_i64(&values[i++], store->store_final_field ? 1 : 0);
    set_i64(&values[i++], store->result.truncated ? 1 : 0);
    if (i != RUNS_PARAM_COUNT) {
        return -1;
    }

    for (size_t c = 0u; c < RUNS_PARAM_COUNT; ++c) {
        if (ensure_column(store->db, "runs", k_runs_param_columns[c][0], k_runs_param_columns[c][1]) != 0) {
            return -1;
        }
    }

    char sql[4096];
    size_t used = (size_t)snprintf(sql, sizeof(sql),
                                   "INSERT OR REPLACE INTO runs(key, kind, file, created_at, git_commit, julia_version");
    for (size_t c = 0u; c < RUNS_PARAM_COUNT; ++c) {
        used += (size_t)snprintf(sql + used, sizeof(sql) - used, ", %s", k_runs_param_columns[c][0]);
    }
    used += (size_t)snprintf(sql + used, sizeof(sql) - used,
                             ") VALUES(?, 'nlolib', ?, strftime('%%Y-%%m-%%dT%%H:%%M:%%f', 'now'), ?, NULL");
    for (size_t c = 0u; c < RUNS_PARAM_COUNT; ++c) {
        used += (size_t)snprintf(sql + used, sizeof(sql) - used, ", ?");
    }
    (void)snprintf(sql + used, sizeof(sql) - used, ");");

    sqlite3_stmt* stmt = NULL;
    if (prepare(store->db, sql, &stmt) != 0) {
        return -1;
    }
    int rc = (bind_text(stmt, 1, params->key) != 0 ||
              bind_text(stmt, 2, params->index_file) != 0 ||
              bind_text(stmt, 3, params->git_commit) != 0) ? -1 : 0;
    for (size_t c = 0u; rc == 0 && c < RUNS_PARAM_COUNT; ++c) {
        const int idx = (int)c + 4;
        switch (values[c].type) {
            case SQLITE_INTEGER:
                rc = (sqlite3_bind_int64(stmt, idx, values[c].i64) == SQLITE_OK) ? 0 : -1;
                break;
            case SQLITE_FLOAT:
                rc = (sqlite3_bind_double(stmt, idx, values[c].f64) == SQLITE_OK) ? 0 : -1;
                break;
            case SQLITE_TEXT:
                rc = bind_text(stmt, idx, values[c].text);
                break;
            default:
                rc = (sqlite3_bind_null(stmt, idx) == SQLITE_OK) ? 0 : -1;
                break;
        }
    }
    if (rc == 0) {
        rc = step_done(stmt);
    }
    sqlite3_finalize(stmt);
    return rc;
}

snapshot_store_status snapshot_store_finalize(snapshot_store* store, const snapshot_store_finalize_params* params)
{
    if (store == NULL || store->db == NULL || params == NULL || params->key == NULL ||
        strlen(params->key) != 16u || params->index_file == NULL) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (store_txn_begin(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    const snapshot_store_status chunk_status = store_flush_chunk(store);
    if (chunk_status == SNAPSHOT_STORE_STATUS_ERROR || store_update_run(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    sqlite3_stmt* stmt = NULL;
    int rc = prepare(store->db, "UPDATE io_runs SET key=? WHERE run_id=?;", &stmt);
    if (rc == 0) {
        rc = (bind_text(stmt, 1, params->key) != 0 || bind_text(stmt, 2, store->result.run_id) != 0 ||
              step_done(stmt) != 0) ? -1 : 0;
        sqlite3_finalize(stmt);
    }
    if (rc == 0) {
        rc = prepare(store->db,
                     "UPDATE io_record_layout SET num_records=?, raw_records=?, record_z=? WHERE run_id=?;",
                     &stmt);
    }
    if (rc == 0) {
        rc = (sqlite3_bind_int64(stmt, 1, (sqlite3_int64)store->records_seen) != SQLITE_OK ||
              sqlite3_bind_int64(stmt, 2, (sqlite3_int64)store->raw_records) != SQLITE_OK ||
              bind_doubles(stmt, 3, store->record_z, store->records_seen) != 0 ||
              bind_text(stmt, 4, store->result.run_id) != 0 ||
              step_done(stmt) != 0) ? -1 : 0;
        sqlite3_finalize(stmt);
    }
    if (rc == 0) {
        rc = store_insert_runs_row(store, params);
    }
    if (rc != 0) {
        (void)sqlite3_exec(store->db, "ROLLBACK;", NULL, NULL, NULL);
        store->in_txn = 0;
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (store_txn_commit(store) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    (void)snprintf(store->result.key, sizeof(store->result.key), "%s", params->key);
    return chunk_status;
}

snapshot_store_status snapshot_store_read_all_records(
    snapshot_store* store,
    nlo_complex* out_records,
    size_t num_recorded_samples,
    size_t num_time_samples
)
{
    if (store == NULL ||
        out_records == NULL ||
        num_recorded_samples == 0u ||
        num_time_samples != store->raw_samples ||
        !store->store_raw ||
        store->db == NULL) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    const snapshot_store_status flush_status = snapshot_store_flush(store);
    if (flush_status == SNAPSHOT_STORE_STATUS_ERROR) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    size_t total_elements = 0u;
    if (checked_mul_size_t(num_recorded_samples, num_time_samples, &total_elements) != 0) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    memset(out_records, 0, total_elements * sizeof(nlo_complex));

    unsigned char* seen_records = (unsigned char*)calloc(num_recorded_samples, sizeof(unsigned char));
    if (seen_records == NULL) {
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    sqlite3_stmt* stmt = NULL;
    if (prepare(store->db,
                    "SELECT record_start, record_count, payload "
                    "FROM io_record_chunks "
                    "WHERE run_id=? "
                    "ORDER BY chunk_index ASC;",
                    &stmt) != 0 ||
        stmt == NULL) {
        free(seen_records);
        return SNAPSHOT_STORE_STATUS_ERROR;
    }
    if (bind_text(stmt, 1, store->result.run_id) != 0) {
        sqlite3_finalize(stmt);
        free(seen_records);
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    size_t copied_records = 0u;
    int rc = SQLITE_OK;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const sqlite3_int64 record_start_i64 = sqlite3_column_int64(stmt, 0);
        const sqlite3_int64 record_count_i64 = sqlite3_column_int64(stmt, 1);
        const void* payload = sqlite3_column_blob(stmt, 2);
        const int payload_bytes = sqlite3_column_bytes(stmt, 2);

        if (record_start_i64 < 0 || record_count_i64 <= 0 || payload == NULL || payload_bytes <= 0) {
            sqlite3_finalize(stmt);
            free(seen_records);
            return SNAPSHOT_STORE_STATUS_ERROR;
        }

        const size_t record_start = (size_t)record_start_i64;
        const size_t record_count = (size_t)record_count_i64;
        if (record_start >= num_recorded_samples ||
            record_count > (num_recorded_samples - record_start)) {
            sqlite3_finalize(stmt);
            free(seen_records);
            return SNAPSHOT_STORE_STATUS_ERROR;
        }

        size_t expected_payload_bytes = 0u;
        if (checked_mul_size_t(record_count, num_time_samples, &expected_payload_bytes) != 0 ||
            checked_mul_size_t(expected_payload_bytes,
                                   sizeof(nlo_complex),
                                   &expected_payload_bytes) != 0 ||
            expected_payload_bytes != (size_t)payload_bytes) {
            sqlite3_finalize(stmt);
            free(seen_records);
            return SNAPSHOT_STORE_STATUS_ERROR;
        }

        const size_t dst_offset = record_start * num_time_samples;
        memcpy(out_records + dst_offset, payload, expected_payload_bytes);

        for (size_t i = 0u; i < record_count; ++i) {
            const size_t idx = record_start + i;
            if (seen_records[idx] != 0u) {
                sqlite3_finalize(stmt);
                free(seen_records);
                return SNAPSHOT_STORE_STATUS_ERROR;
            }
            seen_records[idx] = 1u;
            copied_records += 1u;
        }
    }

    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
        free(seen_records);
        return SNAPSHOT_STORE_STATUS_ERROR;
    }

    if (copied_records != num_recorded_samples) {
        const snapshot_store_status status =
            store->result.truncated ? SNAPSHOT_STORE_STATUS_SOFT_LIMIT : SNAPSHOT_STORE_STATUS_ERROR;
        free(seen_records);
        return status;
    }

    free(seen_records);
    return flush_status;
}

void snapshot_store_get_result(const snapshot_store* store, storage_result* out_result)
{
    if (out_result == NULL) {
        return;
    }

    *out_result = (storage_result){0};
    if (store == NULL) {
        return;
    }

    *out_result = store->result;
}

void snapshot_store_close(snapshot_store* store)
{
    if (store == NULL) {
        return;
    }

    if (store->db != NULL) {
        (void)snapshot_store_flush(store);
        if (store->in_txn) {
            (void)sqlite3_exec(store->db, "COMMIT;", NULL, NULL, NULL);
            store->in_txn = 0;
        }
    }
    sqlite3_stmt** stmts[] = {
        &store->insert_chunk_stmt,
        &store->update_run_stmt,
        &store->upsert_final_output_stmt,
        &store->insert_map_stmt,
        &store->page_count_stmt,
        &store->page_size_stmt
    };
    for (size_t i = 0u; i < sizeof(stmts) / sizeof(stmts[0]); ++i) {
        if (*stmts[i] != NULL) {
            sqlite3_finalize(*stmts[i]);
            *stmts[i] = NULL;
        }
    }
    if (store->db != NULL) {
        sqlite3_close(store->db);
        store->db = NULL;
    }

    if (store->fft_plan != NULL) {
        fftw_destroy_plan(store->fft_plan);
    }
    if (store->fft_in != NULL) {
        fftw_free(store->fft_in);
    }
    if (store->fft_out != NULL) {
        fftw_free(store->fft_out);
    }
    free(store->map_power);
    free(store->record_z);
    free(store->chunk_buffer);
    free(store);
}

int snapshot_store_is_available(void)
{
    return 1;
}

#else

struct snapshot_store {
    int unused;
};

int snapshot_store_ensure_schema(struct sqlite3* db)
{
    (void)db;
    return -1;
}

snapshot_store* snapshot_store_open(const snapshot_store_open_params* params)
{
    (void)params;
    return NULL;
}

void snapshot_store_set_record_schedule(
    snapshot_store* store,
    const double* explicit_record_z,
    size_t explicit_record_z_count
)
{
    (void)store;
    (void)explicit_record_z;
    (void)explicit_record_z_count;
}

snapshot_store_status snapshot_store_finalize(snapshot_store* store, const snapshot_store_finalize_params* params)
{
    (void)store;
    (void)params;
    return SNAPSHOT_STORE_STATUS_ERROR;
}

snapshot_store_status snapshot_store_write_record(
    snapshot_store* store,
    size_t record_index,
    const nlo_complex* record,
    size_t num_time_samples
)
{
    (void)store;
    (void)record_index;
    (void)record;
    (void)num_time_samples;
    return SNAPSHOT_STORE_STATUS_ERROR;
}

snapshot_store_status snapshot_store_write_final_output_field(
    snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
)
{
    (void)store;
    (void)field;
    (void)num_time_samples;
    return SNAPSHOT_STORE_STATUS_ERROR;
}

snapshot_store_status snapshot_store_flush(snapshot_store* store)
{
    (void)store;
    return SNAPSHOT_STORE_STATUS_ERROR;
}

snapshot_store_status snapshot_store_read_all_records(
    snapshot_store* store,
    nlo_complex* out_records,
    size_t num_recorded_samples,
    size_t num_time_samples
)
{
    (void)store;
    (void)out_records;
    (void)num_recorded_samples;
    (void)num_time_samples;
    return SNAPSHOT_STORE_STATUS_ERROR;
}

void snapshot_store_get_result(const snapshot_store* store, storage_result* out_result)
{
    (void)store;
    if (out_result != NULL) {
        *out_result = (storage_result){0};
    }
}

void snapshot_store_close(snapshot_store* store)
{
    (void)store;
}

int snapshot_store_is_available(void)
{
    return 0;
}

#endif
