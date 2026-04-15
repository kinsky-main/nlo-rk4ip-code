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
