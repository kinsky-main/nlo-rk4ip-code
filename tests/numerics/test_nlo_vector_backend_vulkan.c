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
    vector_backend* auto_backend = vector_backend_create_vulkan(NULL);
    if (auto_backend != NULL) {
        assert(vector_backend_get_type(auto_backend) == VECTOR_BACKEND_VULKAN);
        assert(vec_begin_simulation(auto_backend) == VEC_STATUS_OK);
        assert(vec_begin_simulation(auto_backend) == VEC_STATUS_OK);
        assert(vec_end_simulation(auto_backend) == VEC_STATUS_OK);
        assert(vec_end_simulation(auto_backend) == VEC_STATUS_OK);
        vector_backend_destroy(auto_backend);
    }

    vk_backend_config invalid = {0};
    assert(vector_backend_create_vulkan(&invalid) == NULL);
    printf("test_nlo_vector_backend_vulkan: validates Vulkan explicit config guards.\n");
    return 0;
}


