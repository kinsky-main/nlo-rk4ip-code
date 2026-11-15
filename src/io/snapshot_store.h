/**
 * @file snapshot_store.h
 * @brief SQLite-backed snapshot chunk storage for large propagation runs.
 */
#pragma once

#include "core/state.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SNAPSHOT_STORE_STATUS_OK = 0,
    SNAPSHOT_STORE_STATUS_SOFT_LIMIT = 1,
    SNAPSHOT_STORE_STATUS_ERROR = 2
} snapshot_store_status;

/**
 * @brief Parameters used when opening a snapshot store.
 */
typedef struct {
    const sim_config* config;
    const execution_options* exec_options;
    const storage_options* storage_options;
    size_t num_time_samples;
    size_t num_recorded_samples;
} snapshot_store_open_params;

/**
 * @brief Identity and provenance of a completed run, written to its `runs` row.
 */
typedef struct {
    const char* key;
    const char* index_file;
    const char* git_commit;
    const char* binding;
    const char* version;
    const char* backend;
    const char* input_sha256;
    const char* potential_sha256;
} snapshot_store_finalize_params;

struct sqlite3;

/**
 * @brief Create or migrate every table of a run database on an open connection.
 *
 * Creates the io_* data tables, the record-map tables and the contract
 * `runs` table, and adds columns missing from databases written by older
 * versions (inspected with `PRAGMA table_info`).
 *
 * @param db Open SQLite connection.
 * @return int 0 on success.
 */
int snapshot_store_ensure_schema(struct sqlite3* db);

/**
 * @brief Open and initialize a snapshot store.
 *
 * Returns NULL when storage is disabled or initialization fails.
 *
 * @param params Storage-open parameters.
 * @return snapshot_store* Store handle, or NULL when unavailable/failure.
 */
snapshot_store* snapshot_store_open(const snapshot_store_open_params* params);

/**
 * @brief Append one captured record into the in-memory chunk buffer.
 *
 * Commits a chunk once capacity is reached.
 *
 * @param store Snapshot store handle.
 * @param record_index Zero-based record index.
 * @param record Record buffer pointer.
 * @param num_time_samples Number of complex samples in @p record.
 * @return snapshot_store_status Write status.
 */
snapshot_store_status snapshot_store_write_record(
    snapshot_store* store,
    size_t record_index,
    const nlo_complex* record,
    size_t num_time_samples
);

/**
 * @brief Persist the final output field for a run as a dedicated DB row.
 *
 * @param store Snapshot store handle.
 * @param field Final output field buffer.
 * @param num_time_samples Number of complex samples in @p field.
 * @return snapshot_store_status Write status.
 */
snapshot_store_status snapshot_store_write_final_output_field(
    snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
);

/**
 * @brief Set the z position of every record (explicit schedule or uniform spacing).
 *
 * @param store Snapshot store handle.
 * @param explicit_record_z Explicit record positions, or NULL for uniform spacing over [0, z_end].
 * @param explicit_record_z_count Entries in @p explicit_record_z.
 */
void snapshot_store_set_record_schedule(
    snapshot_store* store,
    const double* explicit_record_z,
    size_t explicit_record_z_count
);

/**
 * @brief Write the last data, the record layout and the `runs` row in one transaction.
 *
 * After this call the store's database holds a complete run.
 *
 * @param store Snapshot store handle.
 * @param params Run identity and provenance.
 * @return snapshot_store_status Write status.
 */
snapshot_store_status snapshot_store_finalize(snapshot_store* store, const snapshot_store_finalize_params* params);

/**
 * @brief Flush any pending chunk to persistent storage.
 *
 * @param store Snapshot store handle.
 * @return snapshot_store_status Flush status.
 */
snapshot_store_status snapshot_store_flush(snapshot_store* store);

/**
 * @brief Read all recorded snapshot chunks into a caller-provided dense buffer.
 *
 * The destination layout is record-major with contiguous records:
 * [record0 samples][record1 samples]...[recordN-1 samples].
 *
 * @param store Snapshot store handle.
 * @param out_records Destination dense record buffer.
 * @param num_recorded_samples Number of records expected in @p out_records.
 * @param num_time_samples Number of complex samples per record.
 * @return snapshot_store_status Read status.
 */
snapshot_store_status snapshot_store_read_all_records(
    snapshot_store* store,
    nlo_complex* out_records,
    size_t num_recorded_samples,
    size_t num_time_samples
);

/**
 * @brief Retrieve current storage result snapshot.
 *
 * @param store Snapshot store handle.
 * @param out_result Destination storage summary.
 */
void snapshot_store_get_result(const snapshot_store* store, storage_result* out_result);

/**
 * @brief Close storage resources and finalize run metadata.
 *
 * @param store Snapshot store handle to close.
 */
void snapshot_store_close(snapshot_store* store);

/**
 * @brief Returns nonzero when SQLite storage support is compiled in.
 *
 * @return int Nonzero when snapshot storage backend is available.
 */
int snapshot_store_is_available(void);

#ifdef __cplusplus
}
#endif

