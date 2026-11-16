/**
 * @file test_nlo_vector_backend_vulkan_config.c
 * @dir tests/numerics
 * @brief Contract tests for vector_backend_create_vulkan() explicit-config handling.
 *
 * vector_backend_create_vulkan() documents two valid shapes for the handle
 * triple (physical_device/device/queue): all set, or all omitted. All-omitted
 * means "select a device automatically", so a zeroed config is a request for
 * auto selection carrying only the non-handle tuning fields -- not an invalid
 * config. Every language binding relies on this: each one zeroes the handle
 * fields and cannot construct a VkDevice at all.
 *
 * Only a partially-filled triple is invalid, and that is rejected before any
 * Vulkan call, so it is checked in both build configurations.
 */

#include "backend/nlo_complex.h"
#include "backend/vector_backend.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Not assert(): Release builds define NDEBUG. */
#define CFG_CHECK(cond)                                                       \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: CFG_CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CFG_N 64u

/* A non-null placeholder handle. The partial-config guard rejects the config
 * before any Vulkan entry point sees it, so this is never dereferenced. */
static void* const fake_handle = (void*)(uintptr_t)0x1;

/* A partially-filled handle triple is incomplete and must be rejected, whether
 * or not a device is present. */
static void test_partial_config_is_rejected(void)
{
    vk_backend_config only_physical_device = {0};
    only_physical_device.physical_device = (VkPhysicalDevice)fake_handle;
    CFG_CHECK(vector_backend_create_vulkan(&only_physical_device) == NULL);

    vk_backend_config only_device = {0};
    only_device.device = (VkDevice)fake_handle;
    CFG_CHECK(vector_backend_create_vulkan(&only_device) == NULL);

    vk_backend_config only_queue = {0};
    only_queue.queue = (VkQueue)fake_handle;
    CFG_CHECK(vector_backend_create_vulkan(&only_queue) == NULL);

    vk_backend_config missing_queue = {0};
    missing_queue.physical_device = (VkPhysicalDevice)fake_handle;
    missing_queue.device = (VkDevice)fake_handle;
    CFG_CHECK(vector_backend_create_vulkan(&missing_queue) == NULL);

    printf("test_nlo_vector_backend_vulkan_config: partial explicit config is rejected.\n");
}

/* A backend is only legitimately "accepted" if it actually works. Prove the
 * one built from a zeroed config can move data and run a kernel. */
static void test_backend_is_usable(vector_backend* backend)
{
    static nlo_complex host[CFG_N];
    static nlo_complex got[CFG_N];
    vec_buffer* buffer = NULL;

    for (size_t i = 0u; i < CFG_N; ++i) {
        host[i] = make((double)i, -(double)i);
    }

    CFG_CHECK(vec_create(backend, VEC_KIND_COMPLEX64, CFG_N, &buffer) == VEC_STATUS_OK);
    CFG_CHECK(vec_upload(backend, buffer, host, sizeof(host)) == VEC_STATUS_OK);
    CFG_CHECK(vec_complex_scalar_mul_inplace(backend, buffer, make(2.0, 0.0)) == VEC_STATUS_OK);
    CFG_CHECK(vec_download(backend, buffer, got, sizeof(got)) == VEC_STATUS_OK);

    for (size_t i = 0u; i < CFG_N; ++i) {
        CFG_CHECK(RE(got[i]) == 2.0 * (double)i);
        CFG_CHECK(IM(got[i]) == -2.0 * (double)i);
    }

    vec_destroy(backend, buffer);
}

/* A zeroed config is the documented all-omitted case: it selects a device
 * automatically and yields a working backend, exactly as a NULL config does. */
static void test_zeroed_config_selects_a_device(void)
{
    vector_backend* probe = vector_backend_create_vulkan(NULL);
    if (probe == NULL) {
        printf("test_nlo_vector_backend_vulkan_config: no Vulkan device; skipping auto-selection checks.\n");
        return;
    }
    vector_backend_destroy(probe);

    vk_backend_config zeroed = {0};
    vector_backend* backend = vector_backend_create_vulkan(&zeroed);
    CFG_CHECK(backend != NULL);
    CFG_CHECK(vector_backend_get_type(backend) == VECTOR_BACKEND_VULKAN);
    test_backend_is_usable(backend);
    vector_backend_destroy(backend);

    printf("test_nlo_vector_backend_vulkan_config: zeroed config auto-selects a usable device.\n");
}

int main(void)
{
    test_partial_config_is_rejected();
    test_zeroed_config_selects_a_device();
    return 0;
}
