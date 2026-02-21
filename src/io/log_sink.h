/**
 * @file log_sink.h
 * @dir src/io
 * @brief Central logging sink and progress emitters for nlolib runtime logs.
 */

#pragma once

#include <stddef.h>

typedef enum
{
    /** Only error logs are emitted. */
    NLO_LOG_LEVEL_ERROR = 0,
    /** Warnings and errors are emitted. */
    NLO_LOG_LEVEL_WARN = 1,
    /** Informational, warning, and error logs are emitted. */
    NLO_LOG_LEVEL_INFO = 2,
    /** Verbose debug logs are emitted. */
    NLO_LOG_LEVEL_DEBUG = 3
} nlo_log_level;

/**
 * @brief Configure an optional file sink for nlolib logs.
 *
 * @param path_utf8 UTF-8 path to destination log file.
 *        Pass NULL or empty string to disable file logging.
 * @param append Nonzero appends to file; zero truncates file.
 * @return int Zero on success; nonzero on error.
 */
int nlo_log_set_file(const char* path_utf8, int append);

/**
 * @brief Configure an in-memory ring buffer sink for logs.
 *
 * @param capacity_bytes Ring capacity in bytes.
 *        Pass zero to disable buffering.
 * @return int Zero on success; nonzero on error.
 */
int nlo_log_set_buffer(size_t capacity_bytes);

/**
 * @brief Clear buffered in-memory logs.
 *
 * @return int Zero on success; nonzero on error.
 */
int nlo_log_clear_buffer(void);

/**
 * @brief Read buffered logs into caller-provided memory.
 *
 * @param dst Destination character buffer.
 * @param dst_bytes Capacity of @p dst in bytes.
 * @param out_written Number of bytes copied (excluding null terminator).
 * @param consume Nonzero consumes copied bytes from ring buffer.
 * @return int Zero on success; nonzero on error.
 */
int nlo_log_read_buffer(char* dst, size_t dst_bytes, size_t* out_written, int consume);

/**
 * @brief Set global log level threshold.
 *
 * @param level Desired level in @ref nlo_log_level range.
 * @return int Zero on success; nonzero on error.
 */
int nlo_log_set_level(int level);

/**
 * @brief Configure progress log behavior.
 *
 * @param enabled Nonzero enables progress logs.
 * @param milestone_percent Percent milestone cadence in [1, 100].
 * @param emit_on_step_adjust Nonzero emits step-size adjustment entries.
 * @return int Zero on success; nonzero on error.
 */
int nlo_log_set_progress_options(int enabled, int milestone_percent, int emit_on_step_adjust);

/**
 * @brief Emit a formatted log line through active sinks.
 *
 * @param level Severity level of this entry.
 * @param fmt Printf-style format string.
 */
void nlo_log_emit(int level, const char* fmt, ...);

/**
 * @brief Emit preformatted text through active sinks.
 *
 * @param level Severity level of this entry.
 * @param text Message bytes to emit.
 * @param text_len Length of @p text (bytes).
 */
void nlo_log_emit_raw(int level, const char* text, size_t text_len);

/**
 * @brief Begin a progress tracking span.
 *
 * @param z_start Starting z-coordinate.
 * @param z_end Final z-coordinate.
 */
void nlo_log_progress_begin(double z_start, double z_end);

/**
 * @brief Emit progress state for an accepted step.
 *
 * @param step_index Zero-based accepted-step index.
 * @param z Current accepted z-coordinate.
 * @param z_end Final z-coordinate.
 * @param step_size Accepted step size.
 * @param error Adaptive error estimate.
 * @param next_step_size Next candidate step size.
 */
void nlo_log_progress_step_accepted(
    size_t step_index,
    double z,
    double z_end,
    double step_size,
    double error,
    double next_step_size
);

/**
 * @brief Emit progress state for a rejected adaptive step.
 *
 * @param step_index Zero-based accepted-step index.
 * @param z Current z-coordinate.
 * @param z_end Final z-coordinate.
 * @param attempted_step Rejected attempted step size.
 * @param error Adaptive error estimate.
 * @param retry_step Next retry step size.
 * @param reject_attempt Rejection attempt count.
 */
void nlo_log_progress_step_rejected(
    size_t step_index,
    double z,
    double z_end,
    double attempted_step,
    double error,
    double retry_step,
    size_t reject_attempt
);

/**
 * @brief Complete progress tracking span.
 *
 * @param z Final reached z-coordinate.
 * @param z_end Final requested z-coordinate.
 * @param success Nonzero when run completed successfully.
 */
void nlo_log_progress_finish(double z, double z_end, int success);
