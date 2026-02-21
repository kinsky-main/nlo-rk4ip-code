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
    NLO_SNAPSHOT_STORE_STATUS_OK = 0,
    NLO_SNAPSHOT_STORE_STATUS_SOFT_LIMIT = 1,
    NLO_SNAPSHOT_STORE_STATUS_ERROR = 2
} nlo_snapshot_store_status;

/**
 * @brief Parameters used when opening a snapshot store.
 */
typedef struct {
    const sim_config* config;
    const nlo_execution_options* exec_options;
    const nlo_storage_options* storage_options;
    size_t num_time_samples;
    size_t num_recorded_samples;
} nlo_snapshot_store_open_params;

/**
 * @brief Open and initialize a snapshot store.
 *
 * Returns NULL when storage is disabled or initialization fails.
 *
 * @param params Storage-open parameters.
 * @return nlo_snapshot_store* Store handle, or NULL when unavailable/failure.
 */
nlo_snapshot_store* nlo_snapshot_store_open(const nlo_snapshot_store_open_params* params);

/**
 * @brief Append one captured record into the in-memory chunk buffer.
 *
 * Commits a chunk once capacity is reached.
 *
 * @param store Snapshot store handle.
 * @param record_index Zero-based record index.
 * @param record Record buffer pointer.
 * @param num_time_samples Number of complex samples in @p record.
 * @return nlo_snapshot_store_status Write status.
 */
nlo_snapshot_store_status nlo_snapshot_store_write_record(
    nlo_snapshot_store* store,
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
 * @return nlo_snapshot_store_status Write status.
 */
nlo_snapshot_store_status nlo_snapshot_store_write_final_output_field(
    nlo_snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
);

/**
 * @brief Flush any pending chunk to persistent storage.
 *
 * @param store Snapshot store handle.
 * @return nlo_snapshot_store_status Flush status.
 */
nlo_snapshot_store_status nlo_snapshot_store_flush(nlo_snapshot_store* store);

/**
 * @brief Retrieve current storage result snapshot.
 *
 * @param store Snapshot store handle.
 * @param out_result Destination storage summary.
 */
void nlo_snapshot_store_get_result(const nlo_snapshot_store* store, nlo_storage_result* out_result);

/**
 * @brief Close storage resources and finalize run metadata.
 *
 * @param store Snapshot store handle to close.
 */
void nlo_snapshot_store_close(nlo_snapshot_store* store);

/**
 * @brief Returns nonzero when SQLite storage support is compiled in.
 *
 * @return int Nonzero when snapshot storage backend is available.
 */
int nlo_snapshot_store_is_available(void);

#ifdef __cplusplus
}
#endif

