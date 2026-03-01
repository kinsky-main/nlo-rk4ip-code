/**
 * @file test_nlo_vector_backend.c
 * @dir tests/numerics
 * @brief Unit tests for vector backend CPU implementation.
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "backend/vector_backend.h"
#include "backend/nlo_complex.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef NLO_TEST_EPS
#define NLO_TEST_EPS 1e-12
#endif

#ifndef NLO_TEST_TWO_PI
#define NLO_TEST_TWO_PI 6.283185307179586476925286766559
#endif

static void test_real_pipeline(void)
{
    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* a = NULL;
    nlo_vec_buffer* b = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_REAL64, 4, &a) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_REAL64, 4, &b) == NLO_VEC_STATUS_OK);

    const double init_a[4] = {1.0, 2.0, 3.0, 4.0};
    const double init_b[4] = {2.0, 0.5, -1.0, 3.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    assert(nlo_vec_upload(backend, a, init_a, sizeof(init_a)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, b, init_b, sizeof(init_b)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_real_mul_inplace(backend, a, b) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_download(backend, a, out, sizeof(out)) == NLO_VEC_STATUS_OK);

    assert(fabs(out[0] - 2.0) < NLO_TEST_EPS);
    assert(fabs(out[1] - 1.0) < NLO_TEST_EPS);
    assert(fabs(out[2] + 3.0) < NLO_TEST_EPS);
    assert(fabs(out[3] - 12.0) < NLO_TEST_EPS);

    nlo_vec_destroy(backend, a);
    nlo_vec_destroy(backend, b);
    nlo_vector_backend_destroy(backend);

    printf("test_real_pipeline: validates CPU backend real ops.\n");
}

static void test_transfer_guard(void)
{
    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* vec = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2, &vec) == NLO_VEC_STATUS_OK);

    nlo_complex values[2] = { nlo_make(1.0, 0.0), nlo_make(0.0, -2.0) };

    assert(nlo_vec_begin_simulation(backend) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, vec, values, sizeof(values)) == NLO_VEC_STATUS_TRANSFER_FORBIDDEN);
    assert(nlo_vec_end_simulation(backend) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, vec, values, sizeof(values)) == NLO_VEC_STATUS_OK);

    nlo_vec_destroy(backend, vec);
    nlo_vector_backend_destroy(backend);

    printf("test_transfer_guard: validates simulation transfer guard.\n");
}

static void test_validation_contracts(void)
{
    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* real_a = NULL;
    nlo_vec_buffer* real_b = NULL;
    nlo_vec_buffer* real_short = NULL;
    nlo_vec_buffer* complex_vec = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_REAL64, 4u, &real_a) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_REAL64, 4u, &real_b) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_REAL64, 3u, &real_short) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 4u, &complex_vec) == NLO_VEC_STATUS_OK);

    assert(nlo_vec_real_copy(backend, real_a, real_short) == NLO_VEC_STATUS_INVALID_ARGUMENT);
    assert(nlo_vec_complex_axpy_real(backend, complex_vec, complex_vec, nlo_make(1.0, 0.0)) ==
           NLO_VEC_STATUS_INVALID_ARGUMENT);
    assert(nlo_vec_complex_relative_error(backend, complex_vec, complex_vec, 1e-9, NULL) ==
           NLO_VEC_STATUS_INVALID_ARGUMENT);
    assert(nlo_vec_complex_weighted_rms_error(backend, complex_vec, complex_vec, 1e-9, 1e-9, NULL) ==
           NLO_VEC_STATUS_INVALID_ARGUMENT);

    nlo_vec_destroy(backend, complex_vec);
    nlo_vec_destroy(backend, real_short);
    nlo_vec_destroy(backend, real_b);
    nlo_vec_destroy(backend, real_a);
    nlo_vector_backend_destroy(backend);

    printf("test_validation_contracts: validates backend vector guard contracts.\n");
}

static void test_axis_and_mesh_ops(void)
{
    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* wt_axis = NULL;
    nlo_vec_buffer* centered_axis = NULL;
    nlo_vec_buffer* x_axis = NULL;
    nlo_vec_buffer* x_mesh = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 4u, &wt_axis) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 3u, &centered_axis) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 3u, &x_axis) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 12u, &x_mesh) == NLO_VEC_STATUS_OK);

    assert(nlo_vec_complex_axis_unshifted_from_delta(backend, wt_axis, 0.5) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_complex_axis_centered_from_delta(backend, centered_axis, 2.0) == NLO_VEC_STATUS_OK);

    nlo_complex wt_values[4] = {0};
    nlo_complex centered_values[3] = {0};
    assert(nlo_vec_download(backend, wt_axis, wt_values, sizeof(wt_values)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_download(backend, centered_axis, centered_values, sizeof(centered_values)) == NLO_VEC_STATUS_OK);

    assert(fabs(wt_values[0].re - 0.0) < NLO_TEST_EPS);
    assert(fabs(wt_values[1].re - (NLO_TEST_TWO_PI / 2.0)) < NLO_TEST_EPS);
    assert(fabs(wt_values[2].re + NLO_TEST_TWO_PI) < NLO_TEST_EPS);
    assert(fabs(wt_values[3].re + (NLO_TEST_TWO_PI / 2.0)) < NLO_TEST_EPS);
    assert(fabs(centered_values[0].re + 2.0) < NLO_TEST_EPS);
    assert(fabs(centered_values[1].re - 0.0) < NLO_TEST_EPS);
    assert(fabs(centered_values[2].re - 2.0) < NLO_TEST_EPS);

    const nlo_complex x_values[3] = {nlo_make(10.0, 0.0), nlo_make(20.0, 0.0), nlo_make(30.0, 0.0)};
    assert(nlo_vec_upload(backend, x_axis, x_values, sizeof(x_values)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_complex_mesh_from_axis_tfast(backend, x_mesh, x_axis, 2u, 2u, NLO_VEC_MESH_AXIS_X) ==
           NLO_VEC_STATUS_OK);

    nlo_complex x_mesh_values[12] = {0};
    assert(nlo_vec_download(backend, x_mesh, x_mesh_values, sizeof(x_mesh_values)) == NLO_VEC_STATUS_OK);
    for (size_t x = 0u; x < 3u; ++x) {
        for (size_t y = 0u; y < 2u; ++y) {
            const size_t base = ((x * 2u) + y) * 2u;
            assert(fabs(x_mesh_values[base + 0u].re - x_values[x].re) < NLO_TEST_EPS);
            assert(fabs(x_mesh_values[base + 1u].re - x_values[x].re) < NLO_TEST_EPS);
        }
    }

    nlo_vec_destroy(backend, x_mesh);
    nlo_vec_destroy(backend, x_axis);
    nlo_vec_destroy(backend, centered_axis);
    nlo_vec_destroy(backend, wt_axis);
    nlo_vector_backend_destroy(backend);

    printf("test_axis_and_mesh_ops: validates axis generation and t-fast mesh expansion.\n");
}

int main(void)
{
    test_real_pipeline();
    test_transfer_guard();
    test_validation_contracts();
    test_axis_and_mesh_ops();
    printf("test_nlo_vector_backend: all subtests completed.\n");
    return 0;
}
