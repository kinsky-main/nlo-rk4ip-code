/**
 * @file test_fft.c
 * @dir tests/fft
 * @brief Unit tests for FFT forward/inverse operations.
 * @author Wenzel Kinsky
 * @date 2026-01-30
 */

#include "fft/fft.h"
#include "fft/nlo_complex.h"
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
    nlo_complex freq_domain[TEST_FFT_SIZE];
    nlo_complex round_trip[TEST_FFT_SIZE];

    for (size_t i = 0; i < n; ++i) {
        time_domain[i] = nlo_make((double)i * 0.25, (double)(n - i) * -0.1);
    }

    assert(fft_init(n) == 0);
    forward_fft(time_domain, freq_domain, n);
    inverse_fft(freq_domain, round_trip, n);

    for (size_t i = 0; i < n; ++i) {
        assert(fabs(NLO_RE(round_trip[i]) - NLO_RE(time_domain[i])) < NLO_TEST_EPS);
        assert(fabs(NLO_IM(round_trip[i]) - NLO_IM(time_domain[i])) < NLO_TEST_EPS);
    }

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
