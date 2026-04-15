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

#ifndef TEST_EPS
#define TEST_EPS 1e-12
#endif

#ifndef TEST_TWO_PI
#define TEST_TWO_PI 6.283185307179586476925286766559
#endif

static void test_real_pipeline(void)
{
    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    vec_buffer* a = NULL;
    vec_buffer* b = NULL;
    assert(vec_create(backend, VEC_KIND_REAL64, 4, &a) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_REAL64, 4, &b) == VEC_STATUS_OK);

    const double init_a[4] = {1.0, 2.0, 3.0, 4.0};
    const double init_b[4] = {2.0, 0.5, -1.0, 3.0};
    double out[4] = {0.0, 0.0, 0.0, 0.0};

    assert(vec_upload(backend, a, init_a, sizeof(init_a)) == VEC_STATUS_OK);
    assert(vec_upload(backend, b, init_b, sizeof(init_b)) == VEC_STATUS_OK);
    assert(vec_real_mul_inplace(backend, a, b) == VEC_STATUS_OK);
    assert(vec_download(backend, a, out, sizeof(out)) == VEC_STATUS_OK);

    assert(fabs(out[0] - 2.0) < TEST_EPS);
    assert(fabs(out[1] - 1.0) < TEST_EPS);
    assert(fabs(out[2] + 3.0) < TEST_EPS);
    assert(fabs(out[3] - 12.0) < TEST_EPS);

    vec_destroy(backend, a);
    vec_destroy(backend, b);
    vector_backend_destroy(backend);

    printf("test_real_pipeline: validates CPU backend real ops.\n");
}

static void test_transfer_guard(void)
{
    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    vec_buffer* vec = NULL;
    assert(vec_create(backend, VEC_KIND_COMPLEX64, 2, &vec) == VEC_STATUS_OK);

    nlo_complex values[2] = { make(1.0, 0.0), make(0.0, -2.0) };

    assert(vec_begin_simulation(backend) == VEC_STATUS_OK);
    assert(vec_upload(backend, vec, values, sizeof(values)) == VEC_STATUS_TRANSFER_FORBIDDEN);
    assert(vec_end_simulation(backend) == VEC_STATUS_OK);
    assert(vec_upload(backend, vec, values, sizeof(values)) == VEC_STATUS_OK);

    vec_destroy(backend, vec);
    vector_backend_destroy(backend);

    printf("test_transfer_guard: validates simulation transfer guard.\n");
}

static void test_validation_contracts(void)
{
    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    vec_buffer* real_a = NULL;
    vec_buffer* real_b = NULL;
    vec_buffer* real_short = NULL;
    vec_buffer* complex_vec = NULL;
    assert(vec_create(backend, VEC_KIND_REAL64, 4u, &real_a) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_REAL64, 4u, &real_b) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_REAL64, 3u, &real_short) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, 4u, &complex_vec) == VEC_STATUS_OK);

    assert(vec_real_copy(backend, real_a, real_short) == VEC_STATUS_INVALID_ARGUMENT);
    assert(vec_complex_axpy_real(backend, complex_vec, complex_vec, make(1.0, 0.0)) ==
           VEC_STATUS_INVALID_ARGUMENT);
    assert(vec_complex_relative_error(backend, complex_vec, complex_vec, 1e-9, NULL) ==
           VEC_STATUS_INVALID_ARGUMENT);
    assert(vec_complex_weighted_rms_error(backend, complex_vec, complex_vec, 1e-9, 1e-9, NULL) ==
           VEC_STATUS_INVALID_ARGUMENT);

    vec_destroy(backend, complex_vec);
    vec_destroy(backend, real_short);
    vec_destroy(backend, real_b);
    vec_destroy(backend, real_a);
    vector_backend_destroy(backend);

    printf("test_validation_contracts: validates backend vector guard contracts.\n");
}

static void test_axis_and_mesh_ops(void)
{
    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    vec_buffer* wt_axis = NULL;
    vec_buffer* centered_axis = NULL;
    vec_buffer* x_axis = NULL;
    vec_buffer* x_mesh = NULL;
    assert(vec_create(backend, VEC_KIND_COMPLEX64, 4u, &wt_axis) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, 3u, &centered_axis) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, 3u, &x_axis) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, 12u, &x_mesh) == VEC_STATUS_OK);

    assert(vec_complex_axis_unshifted_from_delta(backend, wt_axis, 0.5) == VEC_STATUS_OK);
    assert(vec_complex_axis_centered_from_delta(backend, centered_axis, 2.0) == VEC_STATUS_OK);

    nlo_complex wt_values[4] = {0};
    nlo_complex centered_values[3] = {0};
    assert(vec_download(backend, wt_axis, wt_values, sizeof(wt_values)) == VEC_STATUS_OK);
    assert(vec_download(backend, centered_axis, centered_values, sizeof(centered_values)) == VEC_STATUS_OK);

    assert(fabs(wt_values[0].re - 0.0) < TEST_EPS);
    assert(fabs(wt_values[1].re - (TEST_TWO_PI / 2.0)) < TEST_EPS);
    assert(fabs(wt_values[2].re + TEST_TWO_PI) < TEST_EPS);
    assert(fabs(wt_values[3].re + (TEST_TWO_PI / 2.0)) < TEST_EPS);
    assert(fabs(centered_values[0].re + 2.0) < TEST_EPS);
    assert(fabs(centered_values[1].re - 0.0) < TEST_EPS);
    assert(fabs(centered_values[2].re - 2.0) < TEST_EPS);

    const nlo_complex x_values[3] = {make(10.0, 0.0), make(20.0, 0.0), make(30.0, 0.0)};
    assert(vec_upload(backend, x_axis, x_values, sizeof(x_values)) == VEC_STATUS_OK);
    assert(vec_complex_mesh_from_axis_tfast(backend, x_mesh, x_axis, 2u, 2u, VEC_MESH_AXIS_X) ==
           VEC_STATUS_OK);

    nlo_complex x_mesh_values[12] = {0};
    assert(vec_download(backend, x_mesh, x_mesh_values, sizeof(x_mesh_values)) == VEC_STATUS_OK);
    for (size_t x = 0u; x < 3u; ++x) {
        for (size_t y = 0u; y < 2u; ++y) {
            const size_t base = ((x * 2u) + y) * 2u;
            assert(fabs(x_mesh_values[base + 0u].re - x_values[x].re) < TEST_EPS);
            assert(fabs(x_mesh_values[base + 1u].re - x_values[x].re) < TEST_EPS);
        }
    }

    vec_destroy(backend, x_mesh);
    vec_destroy(backend, x_axis);
    vec_destroy(backend, centered_axis);
    vec_destroy(backend, wt_axis);
    vector_backend_destroy(backend);

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
