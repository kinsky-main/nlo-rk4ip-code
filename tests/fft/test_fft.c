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

#ifndef TEST_EPS
#define TEST_EPS 1e-10
#endif

#define TEST_FFT_SIZE 16

static void test_fft_round_trip(void)
{
    const size_t n = TEST_FFT_SIZE;
    nlo_complex time_domain[TEST_FFT_SIZE];
    nlo_complex round_trip[TEST_FFT_SIZE];

    for (size_t i = 0; i < n; ++i) {
        time_domain[i] = make((double)i * 0.25, (double)(n - i) * -0.1);
    }

    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    vec_buffer* in = NULL;
    vec_buffer* freq = NULL;
    vec_buffer* out = NULL;
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &in) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &freq) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &out) == VEC_STATUS_OK);

    assert(vec_upload(backend, in, time_domain, sizeof(time_domain)) == VEC_STATUS_OK);

    fft_plan* plan = NULL;
    assert(fft_plan_create_with_backend(backend,
                                            n,
                                            FFT_BACKEND_FFTW,
                                            &plan) == VEC_STATUS_OK);
    assert(plan != NULL);

    assert(fft_forward_vec(plan, in, freq) == VEC_STATUS_OK);
    assert(fft_inverse_vec(plan, freq, out) == VEC_STATUS_OK);
    assert(vec_download(backend, out, round_trip, sizeof(round_trip)) == VEC_STATUS_OK);

    for (size_t i = 0; i < n; ++i) {
        assert(fabs(RE(round_trip[i]) - RE(time_domain[i])) < TEST_EPS);
        assert(fabs(IM(round_trip[i]) - IM(time_domain[i])) < TEST_EPS);
    }

    fft_plan_destroy(plan);
    vec_destroy(backend, in);
    vec_destroy(backend, freq);
    vec_destroy(backend, out);
    vector_backend_destroy(backend);

    printf("test_fft_round_trip: validates forward/inverse FFT consistency.\n");
}

static void test_fft_backend_selection_validation(void)
{
    const size_t n = TEST_FFT_SIZE;
    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    fft_plan* auto_plan = NULL;
    assert(fft_plan_create_with_backend(backend,
                                            n,
                                            FFT_BACKEND_AUTO,
                                            &auto_plan) == VEC_STATUS_OK);
    assert(auto_plan != NULL);
    fft_plan_destroy(auto_plan);

    /* Explicit VKFFT requests are a type-mismatch on CPU backends. */
    fft_plan* vk_plan = NULL;
    assert(fft_plan_create_with_backend(backend,
                                            n,
                                            FFT_BACKEND_VKFFT,
                                            &vk_plan) == VEC_STATUS_INVALID_ARGUMENT);
    assert(vk_plan == NULL);

    vector_backend_destroy(backend);
    printf("test_fft_backend_selection_validation: validates runtime FFT selection guards.\n");
}

static void test_fft_shape_and_io_validation(void)
{
    const size_t n = TEST_FFT_SIZE;
    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    fft_plan* plan = NULL;
    fft_shape invalid_rank = {
        .rank = 0u,
        .dims = {n, 1u, 1u}
    };
    assert(fft_plan_create_shaped_with_backend(backend,
                                                   &invalid_rank,
                                                   FFT_BACKEND_FFTW,
                                                   &plan) == VEC_STATUS_INVALID_ARGUMENT);
    assert(plan == NULL);

    fft_shape invalid_dim = {
        .rank = 2u,
        .dims = {n, 0u, 1u}
    };
    assert(fft_plan_create_shaped_with_backend(backend,
                                                   &invalid_dim,
                                                   FFT_BACKEND_FFTW,
                                                   &plan) == VEC_STATUS_INVALID_ARGUMENT);
    assert(plan == NULL);

    assert(fft_plan_create_with_backend(backend,
                                            n,
                                            FFT_BACKEND_FFTW,
                                            &plan) == VEC_STATUS_OK);
    assert(plan != NULL);

    vec_buffer* in = NULL;
    vec_buffer* out = NULL;
    vec_buffer* wrong_len = NULL;
    vec_buffer* wrong_kind = NULL;
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &in) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &out) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n - 1u, &wrong_len) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_REAL64, n, &wrong_kind) == VEC_STATUS_OK);

    assert(fft_forward_vec(plan, in, wrong_len) == VEC_STATUS_INVALID_ARGUMENT);
    assert(fft_forward_vec(plan, in, wrong_kind) == VEC_STATUS_INVALID_ARGUMENT);
    assert(fft_inverse_vec(plan, wrong_len, out) == VEC_STATUS_INVALID_ARGUMENT);

    vec_destroy(backend, wrong_kind);
    vec_destroy(backend, wrong_len);
    vec_destroy(backend, out);
    vec_destroy(backend, in);
    fft_plan_destroy(plan);
    vector_backend_destroy(backend);
    printf("test_fft_shape_and_io_validation: validates FFT shape and IO guards.\n");
}

static void test_fft_round_trip_3d_shape(void)
{
    const size_t nt = 4u;
    const size_t ny = 2u;
    const size_t nx = 2u;
    const size_t n = nt * ny * nx;
    nlo_complex time_domain[16];
    nlo_complex round_trip[16];

    for (size_t i = 0u; i < n; ++i) {
        time_domain[i] = make((double)i * 0.125, (double)(n - i) * 0.03125);
    }

    vector_backend* backend = vector_backend_create_cpu();
    assert(backend != NULL);

    vec_buffer* in = NULL;
    vec_buffer* freq = NULL;
    vec_buffer* out = NULL;
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &in) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &freq) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &out) == VEC_STATUS_OK);
    assert(vec_upload(backend, in, time_domain, sizeof(time_domain)) == VEC_STATUS_OK);

    const fft_shape shape = {
        .rank = 3u,
        .dims = {nt, ny, nx}
    };
    fft_plan* plan = NULL;
    assert(fft_plan_create_shaped_with_backend(backend,
                                                   &shape,
                                                   FFT_BACKEND_FFTW,
                                                   &plan) == VEC_STATUS_OK);
    assert(plan != NULL);
    assert(fft_forward_vec(plan, in, freq) == VEC_STATUS_OK);
    assert(fft_inverse_vec(plan, freq, out) == VEC_STATUS_OK);
    assert(vec_download(backend, out, round_trip, sizeof(round_trip)) == VEC_STATUS_OK);

    for (size_t i = 0u; i < n; ++i) {
        assert(fabs(RE(round_trip[i]) - RE(time_domain[i])) < TEST_EPS);
        assert(fabs(IM(round_trip[i]) - IM(time_domain[i])) < TEST_EPS);
    }

    fft_plan_destroy(plan);
    vec_destroy(backend, in);
    vec_destroy(backend, freq);
    vec_destroy(backend, out);
    vector_backend_destroy(backend);
    printf("test_fft_round_trip_3d_shape: validates shaped 3D FFT round-trip consistency.\n");
}

static void test_fft_round_trip_vulkan_if_available(void)
{
    const size_t n = TEST_FFT_SIZE;
    nlo_complex time_domain[TEST_FFT_SIZE];
    nlo_complex round_trip[TEST_FFT_SIZE];

    for (size_t i = 0u; i < n; ++i) {
        time_domain[i] = make((double)i * 0.25, (double)(n - i) * -0.1);
    }

    vector_backend* backend = vector_backend_create_vulkan(NULL);
    if (backend == NULL) {
        printf("test_fft_round_trip_vulkan_if_available: Vulkan unavailable, skipping.\n");
        return;
    }

    vec_buffer* in = NULL;
    vec_buffer* freq = NULL;
    vec_buffer* out = NULL;
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &in) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &freq) == VEC_STATUS_OK);
    assert(vec_create(backend, VEC_KIND_COMPLEX64, n, &out) == VEC_STATUS_OK);
    assert(vec_upload(backend, in, time_domain, sizeof(time_domain)) == VEC_STATUS_OK);

    fft_plan* plan = NULL;
    assert(fft_plan_create_with_backend(backend,
                                            n,
                                            FFT_BACKEND_VKFFT,
                                            &plan) == VEC_STATUS_OK);
    assert(plan != NULL);

    assert(fft_forward_vec(plan, in, freq) == VEC_STATUS_OK);
    assert(fft_inverse_vec(plan, freq, out) == VEC_STATUS_OK);
    assert(vec_download(backend, out, round_trip, sizeof(round_trip)) == VEC_STATUS_OK);

    for (size_t i = 0u; i < n; ++i) {
        assert(fabs(RE(round_trip[i]) - RE(time_domain[i])) < 1e-8);
        assert(fabs(IM(round_trip[i]) - IM(time_domain[i])) < 1e-8);
    }

    fft_plan_destroy(plan);
    vec_destroy(backend, in);
    vec_destroy(backend, freq);
    vec_destroy(backend, out);
    vector_backend_destroy(backend);
    printf("test_fft_round_trip_vulkan_if_available: validates Vulkan forward/inverse FFT consistency.\n");
}

int main(void)
{
    test_fft_round_trip();
    test_fft_backend_selection_validation();
    test_fft_shape_and_io_validation();
    test_fft_round_trip_3d_shape();
    test_fft_round_trip_vulkan_if_available();
    printf("test_fft: all subtests completed.\n");
    return 0;
}
