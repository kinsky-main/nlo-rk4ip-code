/**
 * @file init_internal.h
 * @brief Internal helpers shared across core initialization modules.
 */
#pragma once

#include "core/state.h"
#include <stddef.h>

int checked_mul_size_t(size_t a, size_t b, size_t* out);
size_t query_available_system_memory_bytes(void);
size_t apply_memory_headroom(size_t available_bytes);

/**
 * @brief Host-memory capacity for dense (in-memory) record output.
 *
 * One derivation, shared by nlolib_query_runtime_limits and the propagate
 * validation, so the two cannot disagree.
 */
typedef struct {
    size_t bytes_per_record;      /**< One record: num_time_samples complex values. */
    size_t working_set_bytes;     /**< Host reserve for the solver's working vectors. */
    size_t host_budget_bytes;     /**< Available host memory after the headroom margin. */
    size_t record_budget_bytes;   /**< Bytes the records may occupy in total. */
    size_t max_records;           /**< record_budget_bytes / bytes_per_record, capped
                                       at the storage (precision) ceiling. */
    int derivable;                /**< Zero when the sample count or the host's
                                       available memory is unknown; nothing is
                                       enforced then. */
} dense_record_limit;

/**
 * @brief Derive the dense record capacity from host memory available now.
 *
 * @param config Merged configuration; selects the working-vector count.
 * @param num_time_samples Flattened samples per record.
 * @param resident_credit_bytes Bytes the caller already holds resident for
 *        this run (see query_resident_bytes()), or 0. Such pages no longer
 *        show up as available memory, so they are added back to the
 *        available reading before the headroom is applied.
 * @param out_limit Destination; zeroed with derivable=0 when not derivable.
 * @return int 0 on success, nonzero on invalid arguments or overflow.
 */
int resolve_dense_record_limit(
    const sim_config* config,
    size_t num_time_samples,
    size_t resident_credit_bytes,
    dense_record_limit* out_limit
);

/**
 * @brief Bytes of a caller buffer that are already backed by physical memory.
 *
 * Runs once per propagate call, never per step.
 *
 * @param address Start of the buffer, or NULL.
 * @param length Buffer length in bytes.
 * @return size_t Resident bytes, at most @p length; 0 where the platform
 *         cannot tell, which only makes the limit more conservative.
 */
size_t query_resident_bytes(const void* address, size_t length);

/**
 * @brief Replace the host's available-memory reading, for deterministic tests.
 *
 * Not part of the public API and not thread-safe.
 *
 * @param bytes Value query_available_system_memory_bytes() reports from now
 *        on; zero restores the real reading. While set,
 *        query_resident_bytes() reports 0, or the whole buffer after
 *        set_resident_bytes_override_fully_resident(1).
 */
void set_available_system_memory_override_bytes(size_t bytes);

/**
 * @brief Choose what query_resident_bytes() reports under the memory override.
 *
 * Lets a test stand for a caller that has already touched its buffers. Not
 * part of the public API and not thread-safe.
 *
 * @param fully_resident Nonzero: every buffer is reported fully resident while
 *        the available-memory override is set. Zero (default): none is.
 */
void set_resident_bytes_override_fully_resident(int fully_resident);
