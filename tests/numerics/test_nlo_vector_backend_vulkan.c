/**
 * @file test_nlo_vector_backend_vulkan.c
 * @dir tests/numerics
 * @brief Smoke tests for Vulkan vector backend contract checks.
 */

#include "backend/vector_backend.h"
#include "backend/nlo_complex.h"
#include "backend/vector_backend_internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef NLO_TEST_EPS
#define NLO_TEST_EPS 1e-11
#endif

int main(void)
{
    nlo_vector_backend* auto_backend = nlo_vector_backend_create_vulkan(NULL);
    if (auto_backend != NULL) {
        assert(nlo_vector_backend_get_type(auto_backend) == NLO_VECTOR_BACKEND_VULKAN);
        assert(nlo_vec_begin_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        assert(nlo_vec_begin_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        assert(nlo_vec_end_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        assert(nlo_vec_end_simulation(auto_backend) == NLO_VEC_STATUS_OK);

        {
            nlo_vec_buffer* fine = NULL;
            nlo_vec_buffer* coarse = NULL;
            nlo_vec_buffer* base = NULL;
            nlo_vec_buffer* k4 = NULL;
            nlo_vec_buffer* k5 = NULL;
            const nlo_complex base_values[2] = {nlo_make(1.0, -2.0), nlo_make(-0.5, 0.25)};
            const nlo_complex k4_values[2] = {nlo_make(3.0, 1.0), nlo_make(-4.0, 2.0)};
            const nlo_complex k5_values[2] = {nlo_make(-1.0, 0.5), nlo_make(0.75, -1.5)};
            nlo_complex fine_out[2] = {0};
            nlo_complex coarse_out[2] = {0};

            assert(nlo_vec_create(auto_backend, NLO_VEC_KIND_COMPLEX64, 2u, &fine) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_create(auto_backend, NLO_VEC_KIND_COMPLEX64, 2u, &coarse) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_create(auto_backend, NLO_VEC_KIND_COMPLEX64, 2u, &base) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_create(auto_backend, NLO_VEC_KIND_COMPLEX64, 2u, &k4) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_create(auto_backend, NLO_VEC_KIND_COMPLEX64, 2u, &k5) == NLO_VEC_STATUS_OK);

            assert(nlo_vec_upload(auto_backend, base, base_values, sizeof(base_values)) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_upload(auto_backend, k4, k4_values, sizeof(k4_values)) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_upload(auto_backend, k5, k5_values, sizeof(k5_values)) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_complex_embedded_error_pair_real(auto_backend,
                                                            fine,
                                                            coarse,
                                                            base,
                                                            k4,
                                                            1.0 / 6.0,
                                                            1.0 / 15.0,
                                                            k5,
                                                            0.1) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_download(auto_backend, fine, fine_out, sizeof(fine_out)) == NLO_VEC_STATUS_OK);
            assert(nlo_vec_download(auto_backend, coarse, coarse_out, sizeof(coarse_out)) == NLO_VEC_STATUS_OK);
            assert(fabs(fine_out[0].re - 1.5) < NLO_TEST_EPS);
            assert(fabs(fine_out[1].im - 0.5833333333333334) < NLO_TEST_EPS);
            assert(fabs(coarse_out[0].re - 1.1) < NLO_TEST_EPS);
            assert(fabs(coarse_out[1].im - 0.23333333333333334) < NLO_TEST_EPS);

            nlo_vec_destroy(auto_backend, k5);
            nlo_vec_destroy(auto_backend, k4);
            nlo_vec_destroy(auto_backend, base);
            nlo_vec_destroy(auto_backend, coarse);
            nlo_vec_destroy(auto_backend, fine);
        }

        nlo_vector_backend_destroy(auto_backend);
    }

    nlo_vk_backend_config empty = {0};
    nlo_vector_backend* explicit_empty_backend = nlo_vector_backend_create_vulkan(&empty);
    if (explicit_empty_backend != NULL) {
        assert(nlo_vector_backend_get_type(explicit_empty_backend) == NLO_VECTOR_BACKEND_VULKAN);
        nlo_vector_backend_destroy(explicit_empty_backend);
    }

    nlo_vk_backend_config invalid = {0};
    invalid.queue_family_index = 1u;
    assert(nlo_vector_backend_create_vulkan(&invalid) == NULL);
    printf("test_nlo_vector_backend_vulkan: validates auto resolve, guards, and fused adaptive combine kernel.\n");
    return 0;
}


