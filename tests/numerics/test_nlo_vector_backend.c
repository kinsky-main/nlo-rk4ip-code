/**
 * @file test_nlo_vector_backend.c
 * @dir tests/numerics
 * @brief Unit tests for vector backend CPU implementation.
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "backend/vector_backend.h"
#include "backend/nlo_complex.h"
#include "backend/vector_backend_internal.h"
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

static void test_affine_comb4_cpu(void)
{
    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* dst = NULL;
    nlo_vec_buffer* a = NULL;
    nlo_vec_buffer* b = NULL;
    nlo_vec_buffer* c = NULL;
    nlo_vec_buffer* d = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &dst) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &a) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &b) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &c) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &d) == NLO_VEC_STATUS_OK);

    const nlo_complex av[2] = {nlo_make(1.0, 2.0), nlo_make(-1.0, 0.5)};
    const nlo_complex bv[2] = {nlo_make(2.0, -1.0), nlo_make(0.5, 0.25)};
    const nlo_complex cv[2] = {nlo_make(-3.0, 4.0), nlo_make(1.0, -2.0)};
    const nlo_complex dv[2] = {nlo_make(0.25, -0.5), nlo_make(3.0, 1.0)};
    nlo_complex out[2] = {0};

    assert(nlo_vec_upload(backend, a, av, sizeof(av)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, b, bv, sizeof(bv)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, c, cv, sizeof(cv)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, d, dv, sizeof(dv)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_complex_affine_comb4_real(backend, dst, a, 1.0, b, -2.0, c, 0.5, d, 4.0) ==
           NLO_VEC_STATUS_OK);
    assert(nlo_vec_download(backend, dst, out, sizeof(out)) == NLO_VEC_STATUS_OK);

    assert(fabs(out[0].re - (-3.5)) < NLO_TEST_EPS);
    assert(fabs(out[0].im - 4.0) < NLO_TEST_EPS);
    assert(fabs(out[1].re - 10.5) < NLO_TEST_EPS);
    assert(fabs(out[1].im - 3.0) < NLO_TEST_EPS);

    nlo_vec_destroy(backend, d);
    nlo_vec_destroy(backend, c);
    nlo_vec_destroy(backend, b);
    nlo_vec_destroy(backend, a);
    nlo_vec_destroy(backend, dst);
    nlo_vector_backend_destroy(backend);

    printf("test_affine_comb4_cpu: validates fused four-term affine combination.\n");
}

static void test_embedded_error_pair_cpu(void)
{
    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* fine = NULL;
    nlo_vec_buffer* coarse = NULL;
    nlo_vec_buffer* base = NULL;
    nlo_vec_buffer* k4 = NULL;
    nlo_vec_buffer* k5 = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &fine) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &coarse) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &base) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &k4) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, 2u, &k5) == NLO_VEC_STATUS_OK);

    const nlo_complex base_values[2] = {nlo_make(1.0, -2.0), nlo_make(-0.5, 0.25)};
    const nlo_complex k4_values[2] = {nlo_make(3.0, 1.0), nlo_make(-4.0, 2.0)};
    const nlo_complex k5_values[2] = {nlo_make(-1.0, 0.5), nlo_make(0.75, -1.5)};
    nlo_complex fine_out[2] = {0};
    nlo_complex coarse_out[2] = {0};

    assert(nlo_vec_upload(backend, base, base_values, sizeof(base_values)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, k4, k4_values, sizeof(k4_values)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_upload(backend, k5, k5_values, sizeof(k5_values)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_complex_embedded_error_pair_real(backend,
                                                    fine,
                                                    coarse,
                                                    base,
                                                    k4,
                                                    1.0 / 6.0,
                                                    1.0 / 15.0,
                                                    k5,
                                                    0.1) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_download(backend, fine, fine_out, sizeof(fine_out)) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_download(backend, coarse, coarse_out, sizeof(coarse_out)) == NLO_VEC_STATUS_OK);

    assert(fabs(fine_out[0].re - 1.5) < NLO_TEST_EPS);
    assert(fabs(fine_out[0].im - (-1.8333333333333333)) < NLO_TEST_EPS);
    assert(fabs(fine_out[1].re - (-1.1666666666666667)) < NLO_TEST_EPS);
    assert(fabs(fine_out[1].im - 0.5833333333333334) < NLO_TEST_EPS);

    assert(fabs(coarse_out[0].re - 1.1) < NLO_TEST_EPS);
    assert(fabs(coarse_out[0].im - (-1.8833333333333333)) < NLO_TEST_EPS);
    assert(fabs(coarse_out[1].re - (-0.6916666666666667)) < NLO_TEST_EPS);
    assert(fabs(coarse_out[1].im - 0.23333333333333334) < NLO_TEST_EPS);

    nlo_vec_destroy(backend, k5);
    nlo_vec_destroy(backend, k4);
    nlo_vec_destroy(backend, base);
    nlo_vec_destroy(backend, coarse);
    nlo_vec_destroy(backend, fine);
    nlo_vector_backend_destroy(backend);

    printf("test_embedded_error_pair_cpu: validates fused fine/coarse adaptive combination.\n");
}

int main(void)
{
    test_real_pipeline();
    test_transfer_guard();
    test_validation_contracts();
    test_axis_and_mesh_ops();
    test_affine_comb4_cpu();
    test_embedded_error_pair_cpu();
    printf("test_nlo_vector_backend: all subtests completed.\n");
    return 0;
}
