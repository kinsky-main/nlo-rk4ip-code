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
 */
nlo_snapshot_store* nlo_snapshot_store_open(const nlo_snapshot_store_open_params* params);

/**
 * @brief Append one captured record into the in-memory chunk buffer.
 *
 * Commits a chunk once capacity is reached.
 */
nlo_snapshot_store_status nlo_snapshot_store_write_record(
    nlo_snapshot_store* store,
    size_t record_index,
    const nlo_complex* record,
    size_t num_time_samples
);

/**
 * @brief Persist the final output field for a run as a dedicated DB row.
 */
nlo_snapshot_store_status nlo_snapshot_store_write_final_output_field(
    nlo_snapshot_store* store,
    const nlo_complex* field,
    size_t num_time_samples
);

/**
 * @brief Flush any pending chunk to persistent storage.
 */
nlo_snapshot_store_status nlo_snapshot_store_flush(nlo_snapshot_store* store);

/**
 * @brief Retrieve current storage result snapshot.
 */
void nlo_snapshot_store_get_result(const nlo_snapshot_store* store, nlo_storage_result* out_result);

/**
 * @brief Close storage resources and finalize run metadata.
 */
void nlo_snapshot_store_close(nlo_snapshot_store* store);

/**
 * @brief Returns nonzero when SQLite storage support is compiled in.
 */
int nlo_snapshot_store_is_available(void);

#ifdef __cplusplus
}
#endif

