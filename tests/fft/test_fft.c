/**
 * @file test_fft.c
 * @brief Unit tests for vec-based FFT forward/inverse operations.
 */

#include "backend/nlo_complex.h"
#include "backend/vector_backend.h"
#include "fft/fft.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#ifndef NLO_TEST_EPS
#define NLO_TEST_EPS 1e-10
#endif

#if defined(NLO_FFT_BACKEND_FFTW)

#define TEST_FFT_SIZE 16

static void test_fft_round_trip(void)
{
    const size_t n = TEST_FFT_SIZE;
    nlo_complex time_domain[TEST_FFT_SIZE];
    nlo_complex round_trip[TEST_FFT_SIZE];

    for (size_t i = 0; i < n; ++i) {
        time_domain[i] = nlo_make((double)i * 0.25, (double)(n - i) * -0.1);
    }

    nlo_vector_backend* backend = nlo_vector_backend_create_cpu();
    assert(backend != NULL);

    nlo_vec_buffer* in = NULL;
    nlo_vec_buffer* freq = NULL;
    nlo_vec_buffer* out = NULL;
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, n, &in) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, n, &freq) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_create(backend, NLO_VEC_KIND_COMPLEX64, n, &out) == NLO_VEC_STATUS_OK);

    assert(nlo_vec_upload(backend, in, time_domain, sizeof(time_domain)) == NLO_VEC_STATUS_OK);

    nlo_fft_plan* plan = NULL;
    assert(nlo_fft_plan_create(backend, n, &plan) == NLO_VEC_STATUS_OK);
    assert(plan != NULL);

    assert(nlo_fft_forward_vec(plan, in, freq) == NLO_VEC_STATUS_OK);
    assert(nlo_fft_inverse_vec(plan, freq, out) == NLO_VEC_STATUS_OK);
    assert(nlo_vec_download(backend, out, round_trip, sizeof(round_trip)) == NLO_VEC_STATUS_OK);

    for (size_t i = 0; i < n; ++i) {
        assert(fabs(NLO_RE(round_trip[i]) - NLO_RE(time_domain[i])) < NLO_TEST_EPS);
        assert(fabs(NLO_IM(round_trip[i]) - NLO_IM(time_domain[i])) < NLO_TEST_EPS);
    }

    nlo_fft_plan_destroy(plan);
    nlo_vec_destroy(backend, in);
    nlo_vec_destroy(backend, freq);
    nlo_vec_destroy(backend, out);
    nlo_vector_backend_destroy(backend);

    printf("test_fft_round_trip: validates forward/inverse FFT consistency.\n");
}

int main(void)
{
    test_fft_round_trip();
    printf("test_fft: all subtests completed.\n");
    return 0;
}

#else

int main(void)
{
    printf("test_fft: FFTW backend disabled; no subtests executed.\n");
    return 0;
}

#endif
