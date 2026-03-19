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
    nlo_vector_backend* auto_backend = nlo_vector_backend_create_vulkan(NULL);
    if (auto_backend != NULL) {
        assert(nlo_vector_backend_get_type(auto_backend) == NLO_VECTOR_BACKEND_VULKAN);
        assert(nlo_vec_begin_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        assert(nlo_vec_begin_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        assert(nlo_vec_end_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        assert(nlo_vec_end_simulation(auto_backend) == NLO_VEC_STATUS_OK);
        nlo_vector_backend_destroy(auto_backend);
    }

    nlo_vk_backend_config empty = {0};
    nlo_vector_backend* explicit_empty_backend = nlo_vector_backend_create_vulkan(&empty);
    if (explicit_empty_backend != NULL) {
        assert(nlo_vector_backend_get_type(explicit_empty_backend) == NLO_VECTOR_BACKEND_VULKAN);
        nlo_vector_backend_destroy(explicit_empty_backend);
    }

    nlo_vk_backend_config invalid = {0};
    invalid.queue_family_index = 1u;
    assert(nlo_vector_backend_create_vulkan(&invalid) == NULL);
    printf("test_nlo_vector_backend_vulkan: validates empty-config auto resolve and partial-config guards.\n");
    return 0;
}


