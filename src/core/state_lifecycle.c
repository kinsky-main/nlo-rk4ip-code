/**
 * @file state_lifecycle.c
 * @brief Simulation state teardown and resource ownership release.
 */

#include "core/state.h"
#include "fft/fft.h"
#include "io/snapshot_store.h"
#include <stdlib.h>

static void nlo_destroy_vec_if_set(nlo_vector_backend* backend, nlo_vec_buffer** vec)
{
    if (backend == NULL || vec == NULL || *vec == NULL) {
        return;
    }

    nlo_vec_destroy(backend, *vec);
    *vec = NULL;
}
void free_simulation_state(simulation_state* state)
{
    if (state == NULL) {
        return;
    }

    if (state->fft_plan != NULL) {
        nlo_fft_plan_destroy(state->fft_plan);
        state->fft_plan = NULL;
    }

    if (state->backend != NULL) {
        nlo_destroy_vec_if_set(state->backend, &state->current_field_vec);
        nlo_destroy_vec_if_set(state->backend, &state->frequency_grid_vec);

        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.ip_field_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.field_magnitude_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.field_working_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.field_freq_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.omega_power_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_1_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_2_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_3_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.k_4_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.dispersion_factor_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.dispersion_operator_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.potential_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.previous_field_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_intensity_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_delayed_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_spectrum_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_mix_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_polarization_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_derivative_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_response_fft_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.raman_derivative_factor_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.wt_axis_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.kx_axis_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.ky_axis_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.t_axis_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.x_axis_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.y_axis_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.wt_mesh_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.kx_mesh_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.ky_mesh_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.t_mesh_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.x_mesh_vec);
        nlo_destroy_vec_if_set(state->backend, &state->working_vectors.y_mesh_vec);
        for (size_t i = 0u; i < NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS; ++i) {
            nlo_destroy_vec_if_set(state->backend, &state->runtime_operator_stack_vec[i]);
        }

        if (state->record_ring_vec != NULL) {
            for (size_t i = 0; i < state->record_ring_capacity; ++i) {
                nlo_destroy_vec_if_set(state->backend, &state->record_ring_vec[i]);
            }
        }

        nlo_vector_backend_destroy(state->backend);
        state->backend = NULL;
    }

    free(state->record_ring_vec);
    nlo_snapshot_store_close(state->snapshot_store);
    state->snapshot_store = NULL;
    free(state->snapshot_scratch_record);
    free(state->field_buffer);
    free(state);
}

