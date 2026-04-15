/**
 * @file state_lifecycle.c
 * @brief Simulation state teardown and resource ownership release.
 */

#include "core/state.h"
#include "fft/fft.h"
#include "io/snapshot_store.h"
#include <stdlib.h>

static void destroy_vec_if_set(vector_backend* backend, vec_buffer** vec)
{
    if (backend == NULL || vec == NULL || *vec == NULL) {
        return;
    }

    vec_destroy(backend, *vec);
    *vec = NULL;
}
void free_simulation_state(simulation_state* state)
{
    if (state == NULL) {
        return;
    }

    if (state->fft_plan != NULL) {
        fft_plan_destroy(state->fft_plan);
        state->fft_plan = NULL;
    }

    if (state->backend != NULL) {
        destroy_vec_if_set(state->backend, &state->current_field_vec);
        destroy_vec_if_set(state->backend, &state->frequency_grid_vec);

        destroy_vec_if_set(state->backend, &state->working_vectors.ip_field_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.field_working_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.field_freq_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.k_final_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.k_temp_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.dispersion_factor_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.dispersion_operator_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.potential_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.previous_field_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_intensity_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_delayed_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_spectrum_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_mix_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_polarization_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_derivative_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_response_fft_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.raman_derivative_factor_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.wt_mesh_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.kx_mesh_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.ky_mesh_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.t_mesh_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.x_mesh_vec);
        destroy_vec_if_set(state->backend, &state->working_vectors.y_mesh_vec);
        destroy_vec_if_set(state->backend, &state->init_vectors.wt_axis_vec);
        destroy_vec_if_set(state->backend, &state->init_vectors.kx_axis_vec);
        destroy_vec_if_set(state->backend, &state->init_vectors.ky_axis_vec);
        destroy_vec_if_set(state->backend, &state->init_vectors.t_axis_vec);
        destroy_vec_if_set(state->backend, &state->init_vectors.x_axis_vec);
        destroy_vec_if_set(state->backend, &state->init_vectors.y_axis_vec);
        for (size_t i = 0u; i < OPERATOR_PROGRAM_MAX_STACK_SLOTS; ++i) {
            destroy_vec_if_set(state->backend, &state->runtime_operator_stack_vec[i]);
        }

        if (state->record_ring_vec != NULL) {
            for (size_t i = 0; i < state->record_ring_capacity; ++i) {
                destroy_vec_if_set(state->backend, &state->record_ring_vec[i]);
            }
        }

        vector_backend_destroy(state->backend);
        state->backend = NULL;
    }

    free(state->record_ring_vec);
    snapshot_store_close(state->snapshot_store);
    state->snapshot_store = NULL;
    free(state->snapshot_scratch_record);
    free(state->field_buffer);
    free(state);
}

