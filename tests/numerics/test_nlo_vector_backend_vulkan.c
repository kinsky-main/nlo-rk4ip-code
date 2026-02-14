/**
 * @file test_nlo_vector_backend_vulkan.c
 * @dir tests/numerics
 * @brief Smoke tests for Vulkan vector backend contract checks.
 */

#include "backend/vector_backend.h"
#include <assert.h>
#include <stdio.h>

int main(void)
{
#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN
    assert(nlo_vector_backend_create_vulkan(NULL) == NULL);

    nlo_vk_backend_config invalid = {0};
    assert(nlo_vector_backend_create_vulkan(&invalid) == NULL);
    printf("test_nlo_vector_backend_vulkan: validates Vulkan config guards.\n");
#else
    printf("test_nlo_vector_backend_vulkan: Vulkan backend disabled; no subtests executed.\n");
#endif
    return 0;
}


