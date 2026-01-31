/**
 * @file fft.c
 * @dir src/fft
 * @brief FFTW implementation of FFT operations.
 * @author Wenzel Kinsky
 * @date 2026-01-30
 */

// MARK: Includes

#include "fft/fft.h"
#include "fft/nlo_complex.h"
#include <stddef.h>

#if defined(NLO_FFT_BACKEND_FFTW)

#include <fftw3.h>
#include <limits.h>

// MARK: Local State

static size_t fft_signal_size = 0;
static fftw_plan fft_forward_plan = NULL;
static fftw_plan fft_inverse_plan = NULL;
static fftw_complex* fft_plan_in = NULL;
static fftw_complex* fft_plan_out = NULL;
static double fft_inverse_scale = 0.0;

// MARK: Local Helpers

static void fft_cleanup_plans(void)
{
    if (fft_forward_plan != NULL) {
        fftw_destroy_plan(fft_forward_plan);
        fft_forward_plan = NULL;
    }

    if (fft_inverse_plan != NULL) {
        fftw_destroy_plan(fft_inverse_plan);
        fft_inverse_plan = NULL;
    }

    if (fft_plan_in != NULL) {
        fftw_free(fft_plan_in);
        fft_plan_in = NULL;
    }

    if (fft_plan_out != NULL) {
        fftw_free(fft_plan_out);
        fft_plan_out = NULL;
    }

    fft_signal_size = 0;
    fft_inverse_scale = 0.0;
}

// MARK: Public Definitions

int fft_init(size_t signal_size)
{
    if (signal_size == 0) {
        return -1;
    }

    if (signal_size == fft_signal_size && fft_forward_plan != NULL && fft_inverse_plan != NULL) {
        return 0;
    }

    if (signal_size > (size_t)INT_MAX) {
        return -1;
    }

    fft_cleanup_plans();

    fft_plan_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * signal_size);
    fft_plan_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * signal_size);
    if (fft_plan_in == NULL || fft_plan_out == NULL) {
        fft_cleanup_plans();
        return -1;
    }

    const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
    fft_forward_plan = fftw_plan_dft_1d((int)signal_size,
                                        fft_plan_in,
                                        fft_plan_out,
                                        FFTW_FORWARD,
                                        flags);
    fft_inverse_plan = fftw_plan_dft_1d((int)signal_size,
                                        fft_plan_in,
                                        fft_plan_out,
                                        FFTW_BACKWARD,
                                        flags);
    if (fft_forward_plan == NULL || fft_inverse_plan == NULL) {
        fft_cleanup_plans();
        return -1;
    }

    fft_signal_size = signal_size;
    fft_inverse_scale = 1.0 / (double)signal_size;
    return 0;
}

void forward_fft(const nlo_complex* time_domain_signal,
                 nlo_complex* frequency_domain_signal,
                 size_t signal_size)
{
    if (time_domain_signal == NULL || frequency_domain_signal == NULL || signal_size == 0) {
        return;
    }

    if (fft_init(signal_size) != 0) {
        return;
    }

    fftw_execute_dft(fft_forward_plan,
                     (fftw_complex*)time_domain_signal,
                     (fftw_complex*)frequency_domain_signal);
}

void inverse_fft(const nlo_complex* frequency_domain_signal,
                 nlo_complex* time_domain_signal,
                 size_t signal_size)
{
    if (frequency_domain_signal == NULL || time_domain_signal == NULL || signal_size == 0) {
        return;
    }

    if (fft_init(signal_size) != 0) {
        return;
    }

    fftw_execute_dft(fft_inverse_plan,
                     (fftw_complex*)frequency_domain_signal,
                     (fftw_complex*)time_domain_signal);

    for (size_t i = 0; i < signal_size; ++i) {
        NLO_RE(time_domain_signal[i]) *= fft_inverse_scale;
        NLO_IM(time_domain_signal[i]) *= fft_inverse_scale;
    }
}

#else

int fft_init(size_t signal_size)
{
    (void)signal_size;
    return -1;
}

void forward_fft(const nlo_complex* time_domain_signal,
                 nlo_complex* frequency_domain_signal,
                 size_t signal_size)
{
    (void)time_domain_signal;
    (void)frequency_domain_signal;
    (void)signal_size;
}

void inverse_fft(const nlo_complex* frequency_domain_signal,
                 nlo_complex* time_domain_signal,
                 size_t signal_size)
{
    (void)frequency_domain_signal;
    (void)time_domain_signal;
    (void)signal_size;
}

#endif
