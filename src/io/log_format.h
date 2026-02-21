/**
 * @file log_format.h
 * @dir src/io
 * @brief Formatting helpers for structured nlolib logs.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Format an unsigned integer with comma group separators.
 *
 * @param dst Output character buffer.
 * @param dst_size Capacity of @p dst in bytes.
 * @param value Integer value to format.
 * @return size_t Number of bytes written excluding the null terminator.
 */
size_t nlo_log_format_u64_grouped(char* dst, size_t dst_size, uint64_t value);

/**
 * @brief Format a byte-count as a human-readable IEC unit string.
 *
 * @param dst Output character buffer.
 * @param dst_size Capacity of @p dst in bytes.
 * @param bytes Raw byte count.
 * @return size_t Number of bytes written excluding the null terminator.
 */
size_t nlo_log_format_bytes_human(char* dst, size_t dst_size, size_t bytes);
