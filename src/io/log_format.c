/**
 * @file log_format.c
 * @brief Formatting helpers for structured nlolib logs.
 */

#include "io/log_format.h"
#include <stdio.h>
#include <string.h>

size_t nlo_log_format_u64_grouped(char* dst, size_t dst_size, uint64_t value)
{
    if (dst == NULL || dst_size == 0u) {
        return 0u;
    }

    char digits[32];
    size_t digit_count = 0u;
    if (value == 0u) {
        digits[digit_count++] = '0';
    } else {
        while (value > 0u && digit_count < sizeof(digits)) {
            digits[digit_count++] = (char)('0' + (value % 10u));
            value /= 10u;
        }
    }

    size_t out_index = 0u;
    for (size_t i = 0u; i < digit_count; ++i) {
        if (i > 0u && ((digit_count - i) % 3u) == 0u) {
            if (out_index + 1u >= dst_size) {
                break;
            }
            dst[out_index++] = ',';
        }

        if (out_index + 1u >= dst_size) {
            break;
        }
        dst[out_index++] = digits[digit_count - 1u - i];
    }

    dst[out_index] = '\0';
    return out_index;
}

size_t nlo_log_format_bytes_human(char* dst, size_t dst_size, size_t bytes)
{
    if (dst == NULL || dst_size == 0u) {
        return 0u;
    }

    static const char* units[] = {"B", "KB", "MB", "GB"};
    static const double scales[] = {
        1.0,
        1024.0,
        1024.0 * 1024.0,
        1024.0 * 1024.0 * 1024.0
    };

    size_t unit_index = 0u;
    if (bytes >= 1024u) {
        unit_index = 1u;
    }
    if (bytes >= (1024u * 1024u)) {
        unit_index = 2u;
    }
    if (bytes >= (1024u * 1024u * 1024u)) {
        unit_index = 3u;
    }

    if (unit_index == 0u) {
        const int written = snprintf(dst, dst_size, "%zu %s", bytes, units[unit_index]);
        if (written < 0) {
            dst[0] = '\0';
            return 0u;
        }
        return (size_t)written;
    }

    const double scaled = (double)bytes / scales[unit_index];
    const int written = snprintf(dst, dst_size, "%.1f %s", scaled, units[unit_index]);
    if (written < 0) {
        dst[0] = '\0';
        return 0u;
    }
    return (size_t)written;
}
