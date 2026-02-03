/**
 * @file test_nlo_vector_backend.c
 * @dir tests/numerics
 * @brief Unit tests for vector backend CPU implementation.
 * @author Wenzel Kinsky
 * @date 2026-02-02
 */

#include "numerics/vector_backend.h"
#include "backend/nlo_complex.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef NLO_TEST_EPS
#define NLO_TEST_EPS 1e-12
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

int main(void)
{
    test_real_pipeline();
    test_transfer_guard();
    printf("test_nlo_vector_backend: all subtests completed.\n");
    return 0;
}
