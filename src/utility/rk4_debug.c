/**
 * @file rk4_debug.c
 * @brief Debug diagnostics helpers for RK4 propagation isolation.
 */

#include "utility/rk4_debug.h"

#if RK4_DEBUG_ACTIVE

#include "backend/vector_backend_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double max_abs;
    double mean_abs_sq;
    size_t non_finite_count;
    size_t first_non_finite_index;
    double first_non_finite_re;
    double first_non_finite_im;
} rk4_debug_stats;

static int rk4_debug_enabled_cached = -1;
static size_t rk4_debug_every_cached = 0u;
static int rk4_debug_first_non_finite_reported = 0;
static int rk4_debug_dispersion_summary_printed = 0;

static int rk4_debug_enabled(void)
{
    if (rk4_debug_enabled_cached >= 0) {
        return rk4_debug_enabled_cached;
    }

    const char* env = getenv("RK4_DEBUG");
    if (env == NULL || *env == '\0' || *env == '0') {
        rk4_debug_enabled_cached = 0;
    } else {
        rk4_debug_enabled_cached = 1;
    }

    return rk4_debug_enabled_cached;
}

static size_t rk4_debug_every(void)
{
    if (rk4_debug_every_cached != 0u) {
        return rk4_debug_every_cached - 1u;
    }

    size_t every = 1u;
    const char* env = getenv("RK4_DEBUG_EVERY");
    if (env != NULL && *env != '\0') {
        const long parsed = strtol(env, NULL, 10);
        if (parsed > 0) {
            every = (size_t)parsed;
        }
    }

    rk4_debug_every_cached = every + 1u;
    return every;
}

static int rk4_debug_should_log_step(size_t step_index)
{
    const size_t every = rk4_debug_every();
    return (every == 0u) ? 1 : ((step_index % every) == 0u);
}

static int rk4_debug_collect_stats(
    const vector_backend* backend,
    const vec_buffer* vec,
    size_t sample_count,
    rk4_debug_stats* out_stats
)
{
    if (backend == NULL || vec == NULL || out_stats == NULL) {
        return -1;
    }
    if (vector_backend_get_type(backend) != VECTOR_BACKEND_CPU) {
        return -1;
    }

    if (sample_count == 0u || sample_count > vec->length) {
        sample_count = vec->length;
    }

    const void* host_ptr = NULL;
    if (vec_get_const_host_ptr(backend, vec, &host_ptr) != VEC_STATUS_OK || host_ptr == NULL) {
        return -1;
    }

    const nlo_complex* data = (const nlo_complex*)host_ptr;
    double max_abs = 0.0;
    double sum_abs_sq = 0.0;
    size_t finite_count = 0u;

    out_stats->non_finite_count = 0u;
    out_stats->first_non_finite_index = 0u;
    out_stats->first_non_finite_re = 0.0;
    out_stats->first_non_finite_im = 0.0;

    for (size_t i = 0; i < sample_count; ++i) {
        const double re = RE(data[i]);
        const double im = IM(data[i]);
        if (!isfinite(re) || !isfinite(im)) {
            if (out_stats->non_finite_count == 0u) {
                out_stats->first_non_finite_index = i;
                out_stats->first_non_finite_re = re;
                out_stats->first_non_finite_im = im;
            }
            out_stats->non_finite_count += 1u;
            continue;
        }

        const double abs_sq = (re * re) + (im * im);
        if (!isfinite(abs_sq)) {
            if (out_stats->non_finite_count == 0u) {
                out_stats->first_non_finite_index = i;
                out_stats->first_non_finite_re = re;
                out_stats->first_non_finite_im = im;
            }
            out_stats->non_finite_count += 1u;
            continue;
        }

        const double abs_value = sqrt(abs_sq);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
        sum_abs_sq += abs_sq;
        finite_count += 1u;
    }

    out_stats->max_abs = max_abs;
    out_stats->mean_abs_sq = (finite_count > 0u) ? (sum_abs_sq / (double)finite_count) : NAN;
    return 0;
}

void rk4_debug_reset_run(void)
{
    rk4_debug_first_non_finite_reported = 0;
    rk4_debug_dispersion_summary_printed = 0;
}

void rk4_debug_log_vec_stats(
    const simulation_state* state,
    const vec_buffer* vec,
    const char* stage,
    size_t step_index,
    double z,
    double step
)
{
    if (!rk4_debug_enabled()) {
        return;
    }
    if (state == NULL || state->backend == NULL) {
        return;
    }

    rk4_debug_stats stats;
    if (rk4_debug_collect_stats(state->backend, vec, state->num_time_samples, &stats) != 0) {
        if (rk4_debug_should_log_step(step_index)) {
            fprintf(stderr,
                    "[RK4_DEBUG] z=%.9e step=%.9e idx=%zu stage=%s stats=unavailable\n",
                    z,
                    step,
                    step_index,
                    stage);
        }
        return;
    }

    const int has_non_finite = (stats.non_finite_count > 0u);
    const int periodic_log = rk4_debug_should_log_step(step_index);
    const int first_non_finite_event = has_non_finite && !rk4_debug_first_non_finite_reported;
    if (!periodic_log && !first_non_finite_event) {
        return;
    }

    fprintf(stderr,
            "[RK4_DEBUG] z=%.9e step=%.9e idx=%zu stage=%s max_abs=%.9e mean_abs_sq=%.9e non_finite=%zu\n",
            z,
            step,
            step_index,
            stage,
            stats.max_abs,
            stats.mean_abs_sq,
            stats.non_finite_count);

    if (first_non_finite_event) {
        fprintf(stderr,
                "[RK4_DEBUG] FIRST_NONFINITE z=%.9e step=%.9e idx=%zu stage=%s sample_idx=%zu re=%.9e im=%.9e\n",
                z,
                step,
                step_index,
                stage,
                stats.first_non_finite_index,
                stats.first_non_finite_re,
                stats.first_non_finite_im);
        rk4_debug_first_non_finite_reported = 1;
    }
}

void rk4_debug_log_error_control(
    size_t step_index,
    double z,
    double step,
    double error,
    double scale,
    double next_step
)
{
    if (!rk4_debug_enabled()) {
        return;
    }
    if (!rk4_debug_should_log_step(step_index) && isfinite(error) && error >= 0.0) {
        return;
    }

    fprintf(stderr,
            "[RK4_DEBUG] z=%.9e step=%.9e idx=%zu error=%.9e scale=%.9e next_step=%.9e\n",
            z,
            step,
            step_index,
            error,
            scale,
            next_step);
}

void rk4_debug_log_dispersion_factor(
    vector_backend* backend,
    const vec_buffer* dispersion_factor,
    size_t num_dispersion_terms,
    double step_size
)
{
    if (!rk4_debug_enabled()) {
        return;
    }
    if (backend == NULL || dispersion_factor == NULL) {
        return;
    }

    rk4_debug_stats stats;
    if (rk4_debug_collect_stats(backend, dispersion_factor, dispersion_factor->length, &stats) != 0) {
        if (!rk4_debug_dispersion_summary_printed) {
            fprintf(stderr,
                    "[RK4_DEBUG] dispersion_factor terms=%zu step=%.9e stats=unavailable\n",
                    num_dispersion_terms,
                    step_size);
            rk4_debug_dispersion_summary_printed = 1;
        }
        return;
    }

    if (!rk4_debug_dispersion_summary_printed || stats.non_finite_count > 0u) {
        fprintf(stderr,
                "[RK4_DEBUG] dispersion_factor terms=%zu step=%.9e max_abs=%.9e mean_abs_sq=%.9e non_finite=%zu\n",
                num_dispersion_terms,
                step_size,
                stats.max_abs,
                stats.mean_abs_sq,
                stats.non_finite_count);
        rk4_debug_dispersion_summary_printed = 1;
    }
}

#else

void rk4_debug_reset_run(void)
{
}

void rk4_debug_log_vec_stats(
    const simulation_state* state,
    const vec_buffer* vec,
    const char* stage,
    size_t step_index,
    double z,
    double step
)
{
    (void)state;
    (void)vec;
    (void)stage;
    (void)step_index;
    (void)z;
    (void)step;
}

void rk4_debug_log_error_control(
    size_t step_index,
    double z,
    double step,
    double error,
    double scale,
    double next_step
)
{
    (void)step_index;
    (void)z;
    (void)step;
    (void)error;
    (void)scale;
    (void)next_step;
}

void rk4_debug_log_dispersion_factor(
    vector_backend* backend,
    const vec_buffer* dispersion_factor,
    size_t num_dispersion_terms,
    double step_size
)
{
    (void)backend;
    (void)dispersion_factor;
    (void)num_dispersion_terms;
    (void)step_size;
}

#endif
