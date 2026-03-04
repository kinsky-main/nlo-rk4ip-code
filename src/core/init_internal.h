/**
 * @file init_internal.h
 * @brief Internal helpers shared across core initialization modules.
 */
#pragma once

#include <stddef.h>

int nlo_checked_mul_size_t(size_t a, size_t b, size_t* out);
size_t nlo_query_available_system_memory_bytes(void);
size_t nlo_apply_memory_headroom(size_t available_bytes);
size_t nlo_compute_host_record_capacity(size_t num_time_samples, size_t requested_records);
