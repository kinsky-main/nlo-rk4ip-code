/**
 * @file test_nlo_vector_backend_vulkan.c
 * @dir tests/numerics
 * @brief Smoke tests for Vulkan vector backend contract checks.
 */

#include "backend/nlo_complex.h"
#include "backend/vector_backend.h"
#include <math.h>

/* NLOLIB_VULKAN_FP32 computes on the device in single precision, so checks of
 * Vulkan results get a tolerance floor a few orders above float epsilon. */
#if defined(NLO_VK_FP32) && NLO_VK_FP32
#define VK_TOL(tol) fmax((tol), 1e-5)
#else
#define VK_TOL(tol) (tol)
#endif
#include <stdio.h>
#include <stdlib.h>

/* Not assert(): Release builds define NDEBUG. */
#define VK_CHECK(cond)                                                        \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: VK_CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define COMB_N 300u

/* vec_complex_affine_comb_real on Vulkan against the CPU loop, for 2-4 terms,
 * including dst aliasing a source (as the fused RK step uses it). */
static void test_affine_comb_matches_cpu(vector_backend* vk)
{
    vector_backend* cpu = vector_backend_create_cpu();
    VK_CHECK(cpu != NULL);
    static nlo_complex host[4][COMB_N];
    static nlo_complex want[COMB_N];
    static nlo_complex got[COMB_N];
    vec_buffer* vk_src[4] = {NULL};
    vec_buffer* cpu_src[4] = {NULL};
    for (size_t k = 0u; k < 4u; ++k) {
        for (size_t i = 0u; i < COMB_N; ++i) {
            host[k][i] = make(sin(0.1 * (double)(i + 7u * k)), cos(0.03 * (double)(i * (k + 1u))));
        }
        VK_CHECK(vec_create(vk, VEC_KIND_COMPLEX64, COMB_N, &vk_src[k]) == VEC_STATUS_OK);
        VK_CHECK(vec_create(cpu, VEC_KIND_COMPLEX64, COMB_N, &cpu_src[k]) == VEC_STATUS_OK);
    }
    const double coeffs[4] = {1.0, -0.37, 0.125, 2.5};

    for (size_t terms = 2u; terms <= 4u; ++terms) {
        for (int alias = 0; alias <= 1; ++alias) {
            for (size_t k = 0u; k < 4u; ++k) {
                VK_CHECK(vec_upload(vk, vk_src[k], host[k], sizeof(host[k])) == VEC_STATUS_OK);
                VK_CHECK(vec_upload(cpu, cpu_src[k], host[k], sizeof(host[k])) == VEC_STATUS_OK);
            }
            /* alias: dst is the second source; otherwise the unused fourth buffer. */
            vec_buffer* vk_dst = alias ? vk_src[1] : vk_src[3];
            vec_buffer* cpu_dst = alias ? cpu_src[1] : cpu_src[3];
            if (!alias && terms == 4u) {
                continue;
            }
            VK_CHECK(vec_complex_affine_comb_real(vk, vk_dst, terms, (const vec_buffer* const*)vk_src, coeffs) ==
                     VEC_STATUS_OK);
            VK_CHECK(vec_complex_affine_comb_real(cpu, cpu_dst, terms, (const vec_buffer* const*)cpu_src, coeffs) ==
                     VEC_STATUS_OK);
            VK_CHECK(vec_download(vk, vk_dst, got, sizeof(got)) == VEC_STATUS_OK);
            VK_CHECK(vec_download(cpu, cpu_dst, want, sizeof(want)) == VEC_STATUS_OK);
            for (size_t i = 0u; i < COMB_N; ++i) {
                double ref_re = 0.0;
                for (size_t k = 0u; k < terms; ++k) {
                    ref_re += coeffs[k] * RE(host[k][i]);
                }
                VK_CHECK(fabs(RE(got[i]) - RE(want[i])) <= VK_TOL(1e-14));
                VK_CHECK(fabs(IM(got[i]) - IM(want[i])) <= VK_TOL(1e-14));
                VK_CHECK(fabs(RE(want[i]) - ref_re) <= 1e-14);
            }
        }
    }

    VK_CHECK(vec_complex_affine_comb_real(vk, vk_src[0], 5u, (const vec_buffer* const*)vk_src, coeffs) ==
             VEC_STATUS_INVALID_ARGUMENT);
    for (size_t k = 0u; k < 4u; ++k) {
        vec_destroy(vk, vk_src[k]);
        vec_destroy(cpu, cpu_src[k]);
    }
    vector_backend_destroy(cpu);
    printf("test_nlo_vector_backend_vulkan: affine combination matches the CPU for 2-4 terms.\n");
}

int main(void)
{
    vector_backend* auto_backend = vector_backend_create_vulkan(NULL);
    if (auto_backend != NULL) {
        VK_CHECK(vector_backend_get_type(auto_backend) == VECTOR_BACKEND_VULKAN);
        VK_CHECK(vec_begin_simulation(auto_backend) == VEC_STATUS_OK);
        VK_CHECK(vec_begin_simulation(auto_backend) == VEC_STATUS_OK);
        VK_CHECK(vec_end_simulation(auto_backend) == VEC_STATUS_OK);
        VK_CHECK(vec_end_simulation(auto_backend) == VEC_STATUS_OK);
        test_affine_comb_matches_cpu(auto_backend);
        vector_backend_destroy(auto_backend);
    }

    /* A zeroed config is not invalid: the handle triple may be all set or all
     * omitted, and all-omitted requests auto selection. The explicit-config
     * guards live in test_nlo_vector_backend_vulkan_config. */
    printf("test_nlo_vector_backend_vulkan: Vulkan backend smoke checks done.\n");
    return 0;
}


