/**
 * @file test_nlo_vk_instance_reuse.c
 * @dir tests/numerics
 * @brief Regression test: many Vulkan backend lifetimes in one process.
 *
 * Every nlo state creates its own vector backend, so a process that runs N
 * propagates creates N Vulkan backends.  Each backend used to create its own
 * VkInstance, and on Linux the NVIDIA ICD opens a D-Bus system-bus connection
 * inside vkCreateInstance that vkDestroyInstance does not close, which capped a
 * process at eleven: on a Quadro P4000 / 535.179 the eleventh was the last
 * connection the driver would open, and from the twelfth backend on its results
 * shifted by ~1 ULP and stayed shifted.  PR #30 recorded the same eleven-instance
 * boundary manifesting as a hang.
 *
 * The instance is now process-wide, so this drives well past that ceiling and
 * pins the two things that matter: one vkCreateInstance however many backends,
 * and results that do not drift from one backend lifetime to the next.
 *
 * Skips (exit 0 without asserting) when no Vulkan device answers the loader,
 * following test_nlo_vector_backend_vulkan.c.  CTest gives it a timeout because
 * the regression has also been seen to manifest as a hang.
 */

#include "backend/nlo_complex.h"
#include "backend/vector_backend.h"
#include "backend/vk_auto_context.h"
#include <math.h>
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

/* Comfortably past the old eleven-instance ceiling. */
#define REUSE_CYCLES 24u
#define REUSE_N 512u

/* One backend lifetime's worth of work, shaped like a propagate: create a
 * backend, upload, run a couple of kernels, download, tear the backend down. */
static void run_cycle(unsigned int cycle, const nlo_complex* input, nlo_complex* out)
{
    vector_backend* backend = vector_backend_create_vulkan(NULL);
    VK_CHECK(backend != NULL);
    VK_CHECK(vector_backend_get_type(backend) == VECTOR_BACKEND_VULKAN);

    vec_buffer* buffer = NULL;
    VK_CHECK(vec_create(backend, VEC_KIND_COMPLEX64, REUSE_N, &buffer) == VEC_STATUS_OK);

    VK_CHECK(vec_upload(backend, buffer, input, REUSE_N * sizeof(*input)) == VEC_STATUS_OK);
    VK_CHECK(vec_complex_exp_inplace(backend, buffer) == VEC_STATUS_OK);
    VK_CHECK(vec_complex_scalar_mul_inplace(backend, buffer, make(0.5, -0.25)) == VEC_STATUS_OK);
    VK_CHECK(vec_download(backend, buffer, out, REUSE_N * sizeof(*out)) == VEC_STATUS_OK);

    vec_destroy(backend, buffer);
    vector_backend_destroy(backend);

    fprintf(stderr, "cycle %2u/%u done\n", cycle, (unsigned int)REUSE_CYCLES);
    fflush(stderr);
}

int main(void)
{
    /* Probe once: no Vulkan device means skip, not fail. */
    vector_backend* probe = vector_backend_create_vulkan(NULL);
    if (probe == NULL) {
        printf("test_nlo_vk_instance_reuse: no Vulkan device available; skipped.\n");
        return 0;
    }
    vector_backend_destroy(probe);

    static nlo_complex input[REUSE_N];
    static nlo_complex want[REUSE_N];
    static nlo_complex got[REUSE_N];
    for (size_t i = 0u; i < REUSE_N; ++i) {
        input[i] = make(0.01 * (double)(i % 29u), 0.02 * (double)(i % 17u));
    }

    run_cycle(1u, input, want);
    for (unsigned int cycle = 2u; cycle <= REUSE_CYCLES; ++cycle) {
        run_cycle(cycle, input, got);
        for (size_t i = 0u; i < REUSE_N; ++i) {
            VK_CHECK(fabs(RE(got[i]) - RE(want[i])) <= 1e-15);
            VK_CHECK(fabs(IM(got[i]) - IM(want[i])) <= 1e-15);
        }
    }
    printf("test_nlo_vk_instance_reuse: %u backend lifetimes agree.\n",
           (unsigned int)REUSE_CYCLES);

    /* The whole point: one VkInstance served every backend above. */
    const unsigned int instances = vk_auto_instance_create_count();
    printf("test_nlo_vk_instance_reuse: vkCreateInstance calls = %u.\n", instances);
    VK_CHECK(instances == 1u);
    return 0;
}
