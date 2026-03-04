/**
 * @file sim_dimensions_internal.h
 * @brief Internal helpers for resolving runtime simulation dimensions.
 */
#pragma once

#include "core/state.h"
#include <stddef.h>

/**
 * @brief Multiply two `size_t` values with overflow detection.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @param out Destination result on success.
 * @return int 0 on success, nonzero on overflow/invalid output pointer.
 */
static inline int nlo_sim_dimensions_checked_mul(size_t a, size_t b, size_t* out)
{
    if (out == NULL) {
        return -1;
    }

    if (a == 0u || b == 0u) {
        *out = 0u;
        return 0;
    }

    if (a > (SIZE_MAX / b)) {
        return -1;
    }

    *out = a * b;
    return 0;
}

/**
 * @brief Resolve flattened sample count into runtime (nt, nx, ny) dimensions.
 *
 * Tensor descriptors take precedence. Non-tensor runs are treated as
 * temporal 1D unless explicitly configured as a valid degenerate shape.
 *
 * @param config Simulation configuration.
 * @param total_samples Flattened sample count.
 * @param out_nt Destination temporal sample count.
 * @param out_nx Destination x-dimension sample count.
 * @param out_ny Destination y-dimension sample count.
 * @param out_explicit_nd Destination flag for explicit ND mode.
 * @return int 0 on success, nonzero when dimensions are inconsistent.
 */
static inline int nlo_resolve_sim_dimensions_internal(
    const sim_config* config,
    size_t total_samples,
    size_t* out_nt,
    size_t* out_nx,
    size_t* out_ny,
    int* out_explicit_nd
)
{
    if (config == NULL ||
        out_nt == NULL ||
        out_nx == NULL ||
        out_ny == NULL ||
        out_explicit_nd == NULL ||
        total_samples == 0u) {
        return -1;
    }

    if (config->tensor.nt > 0u) {
        if (config->tensor.nx == 0u ||
            config->tensor.ny == 0u ||
            config->tensor.layout != NLO_TENSOR_LAYOUT_XYT_T_FAST) {
            return -1;
        }

        size_t ntx = 0u;
        size_t resolved_total = 0u;
        if (nlo_sim_dimensions_checked_mul(config->tensor.nt, config->tensor.nx, &ntx) != 0 ||
            nlo_sim_dimensions_checked_mul(ntx, config->tensor.ny, &resolved_total) != 0 ||
            resolved_total != total_samples) {
            return -1;
        }

        *out_nt = config->tensor.nt;
        *out_nx = config->tensor.nx;
        *out_ny = config->tensor.ny;
        *out_explicit_nd = 1;
        return 0;
    }

    const size_t configured_nt = config->time.nt;
    size_t nx = config->spatial.nx;
    size_t ny = config->spatial.ny;

    if (configured_nt == 0u) {
        if (nx == 0u && ny == 0u) {
            *out_nt = total_samples;
            *out_nx = 1u;
            *out_ny = 1u;
            *out_explicit_nd = 0;
            return 0;
        }
        if (nx == 0u || ny == 0u) {
            return -1;
        }

        size_t total_points = 0u;
        if (nlo_sim_dimensions_checked_mul(nx, ny, &total_points) != 0 ||
            total_points != total_samples) {
            return -1;
        }

        if (ny == 1u && nx == total_samples) {
            *out_nt = total_samples;
            *out_nx = 1u;
            *out_ny = 1u;
            *out_explicit_nd = 0;
            return 0;
        }

        *out_nt = 1u;
        *out_nx = nx;
        *out_ny = ny;
        *out_explicit_nd = 0;
        return 0;
    }

    if (nx == 0u) {
        nx = 1u;
    }
    if (ny == 0u) {
        ny = 1u;
    }

    size_t ntx = 0u;
    if (nlo_sim_dimensions_checked_mul(configured_nt, nx, &ntx) != 0) {
        return -1;
    }

    size_t resolved_total = 0u;
    if (nlo_sim_dimensions_checked_mul(ntx, ny, &resolved_total) != 0 ||
        resolved_total != total_samples) {
        return -1;
    }

    *out_nt = configured_nt;
    *out_nx = nx;
    *out_ny = ny;
    *out_explicit_nd = 1;
    return 0;
}
