/**
 * @file vk_backend_resources.c
 * @brief Vulkan backend lifecycle and resource management.
 */

#include "numerics/vk_backend_internal.h"
#include "nlo_vk_shader_paths.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

VkDeviceSize nlo_vk_min_size(VkDeviceSize a, VkDeviceSize b)
{
    return (a < b) ? a : b;
}

enum {
    NLO_VK_DESCRIPTOR_SET_BUDGET_DEFAULT_BYTES = 512u * 1024u,
    NLO_VK_DESCRIPTOR_SET_ESTIMATED_BYTES = 512u,
    NLO_VK_DESCRIPTOR_SET_MIN_COUNT = 16u,
    NLO_VK_DESCRIPTOR_SET_MAX_COUNT = 4096u
};

static uint32_t nlo_vk_clamp_descriptor_set_count(uint64_t count)
{
    if (count < (uint64_t)NLO_VK_DESCRIPTOR_SET_MIN_COUNT) {
        return NLO_VK_DESCRIPTOR_SET_MIN_COUNT;
    }
    if (count > (uint64_t)NLO_VK_DESCRIPTOR_SET_MAX_COUNT) {
        return NLO_VK_DESCRIPTOR_SET_MAX_COUNT;
    }
    return (uint32_t)count;
}

static uint32_t nlo_vk_select_descriptor_set_target(const nlo_vk_backend_config* config)
{
    if (config != NULL && config->descriptor_set_count_override > 0u) {
        return nlo_vk_clamp_descriptor_set_count((uint64_t)config->descriptor_set_count_override);
    }

    size_t budget_bytes = NLO_VK_DESCRIPTOR_SET_BUDGET_DEFAULT_BYTES;
    if (config != NULL && config->descriptor_set_budget_bytes > 0u) {
        budget_bytes = config->descriptor_set_budget_bytes;
    }

    uint64_t estimated_count = (uint64_t)(budget_bytes / (size_t)NLO_VK_DESCRIPTOR_SET_ESTIMATED_BYTES);
    if (estimated_count == 0u) {
        estimated_count = (uint64_t)NLO_VK_DESCRIPTOR_SET_MIN_COUNT;
    }

    return nlo_vk_clamp_descriptor_set_count(estimated_count);
}

static bool nlo_vk_supports_required_features(
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceLimits* out_limits
)
{
    if (physical_device == VK_NULL_HANDLE) {
        return false;
    }

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physical_device, &features);
    if (features.shaderFloat64 != VK_TRUE) {
        return false;
    }

    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    if (out_limits != NULL) {
        *out_limits = properties.limits;
    }

    return true;
}

static uint32_t nlo_vk_find_memory_type(
    nlo_vector_backend* backend,
    uint32_t type_filter,
    VkMemoryPropertyFlags properties
)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(backend->vk.physical_device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) != 0u &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    return UINT32_MAX;
}

nlo_vec_status nlo_vk_create_buffer_raw(
    nlo_vector_backend* backend,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer* out_buffer,
    VkDeviceMemory* out_memory
)
{
    if (backend == NULL || out_buffer == NULL || out_memory == NULL || size == 0u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    if (vkCreateBuffer(backend->vk.device, &buffer_info, NULL, out_buffer) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(backend->vk.device, *out_buffer, &mem_req);
    uint32_t type_index = nlo_vk_find_memory_type(backend, mem_req.memoryTypeBits, properties);
    if (type_index == UINT32_MAX) {
        vkDestroyBuffer(backend->vk.device, *out_buffer, NULL);
        *out_buffer = VK_NULL_HANDLE;
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = type_index
    };
    if (vkAllocateMemory(backend->vk.device, &alloc_info, NULL, out_memory) != VK_SUCCESS) {
        vkDestroyBuffer(backend->vk.device, *out_buffer, NULL);
        *out_buffer = VK_NULL_HANDLE;
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    if (vkBindBufferMemory(backend->vk.device, *out_buffer, *out_memory, 0u) != VK_SUCCESS) {
        vkDestroyBuffer(backend->vk.device, *out_buffer, NULL);
        vkFreeMemory(backend->vk.device, *out_memory, NULL);
        *out_buffer = VK_NULL_HANDLE;
        *out_memory = VK_NULL_HANDLE;
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

void nlo_vk_destroy_buffer_raw(nlo_vector_backend* backend, VkBuffer* buffer, VkDeviceMemory* memory)
{
    if (backend == NULL) {
        return;
    }
    if (buffer != NULL && *buffer != VK_NULL_HANDLE) {
        if (backend->vk.device != VK_NULL_HANDLE) {
            vkDestroyBuffer(backend->vk.device, *buffer, NULL);
        }
        *buffer = VK_NULL_HANDLE;
    }
    if (memory != NULL && *memory != VK_NULL_HANDLE) {
        if (backend->vk.device != VK_NULL_HANDLE) {
            vkFreeMemory(backend->vk.device, *memory, NULL);
        }
        *memory = VK_NULL_HANDLE;
    }
}

static nlo_vec_status nlo_vk_read_binary_file(const char* path, uint32_t** out_words, size_t* out_size)
{
    if (path == NULL || out_words == NULL || out_size == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    FILE* fp = fopen(path, "rb");
    if (fp == NULL) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    long file_size_long = ftell(fp);
    if (file_size_long <= 0) {
        fclose(fp);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    const size_t file_size = (size_t)file_size_long;
    if ((file_size % sizeof(uint32_t)) != 0u) {
        fclose(fp);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    uint32_t* words = (uint32_t*)malloc(file_size);
    if (words == NULL) {
        fclose(fp);
        return NLO_VEC_STATUS_ALLOCATION_FAILED;
    }

    size_t nread = fread(words, 1u, file_size, fp);
    fclose(fp);
    if (nread != file_size) {
        free(words);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    *out_words = words;
    *out_size = file_size;
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_create_compute_pipeline(
    nlo_vector_backend* backend,
    const char* shader_path,
    VkPipeline* out_pipeline
)
{
    uint32_t* shader_words = NULL;
    size_t shader_size = 0u;
    nlo_vec_status status = nlo_vk_read_binary_file(shader_path, &shader_words, &shader_size);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkShaderModuleCreateInfo module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_size,
        .pCode = shader_words
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(backend->vk.device, &module_info, NULL, &module) != VK_SUCCESS) {
        free(shader_words);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    free(shader_words);

    VkPipelineShaderStageCreateInfo stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main"
    };
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage_info,
        .layout = backend->vk.pipeline_layout
    };

    if (vkCreateComputePipelines(backend->vk.device,
                                 backend->vk.pipeline_cache,
                                 1u,
                                 &pipeline_info,
                                 NULL,
                                 out_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(backend->vk.device, module, NULL);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    vkDestroyShaderModule(backend->vk.device, module, NULL);
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_create_command_resources(nlo_vector_backend* backend, const nlo_vk_backend_config* config)
{
    if (config->command_pool != VK_NULL_HANDLE) {
        backend->vk.command_pool = config->command_pool;
        backend->vk.owns_command_pool = false;
    } else {
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = backend->vk.queue_family_index,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
        };
        if (vkCreateCommandPool(backend->vk.device, &pool_info, NULL, &backend->vk.command_pool) != VK_SUCCESS) {
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
        backend->vk.owns_command_pool = true;
    }

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = backend->vk.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1u
    };
    if (vkAllocateCommandBuffers(backend->vk.device, &alloc_info, &backend->vk.command_buffer) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    if (vkCreateFence(backend->vk.device, &fence_info, NULL, &backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_create_descriptor_resources(
    nlo_vector_backend* backend,
    const nlo_vk_backend_config* config
)
{
    VkDescriptorSetLayoutBinding bindings[3] = {0};
    bindings[0].binding = 0u;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1u;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1u;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1u;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2u;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1u;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3u,
        .pBindings = bindings
    };
    if (vkCreateDescriptorSetLayout(backend->vk.device, &layout_info, NULL, &backend->vk.descriptor_set_layout) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0u,
        .size = (uint32_t)sizeof(nlo_vk_push_constants)
    };
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1u,
        .pSetLayouts = &backend->vk.descriptor_set_layout,
        .pushConstantRangeCount = 1u,
        .pPushConstantRanges = &push_range
    };
    if (vkCreatePipelineLayout(backend->vk.device, &pipeline_layout_info, NULL, &backend->vk.pipeline_layout) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    backend->vk.descriptor_set_count = 0u;
    backend->vk.descriptor_sets = NULL;

    uint32_t target_count = nlo_vk_select_descriptor_set_target(config);
    while (target_count >= NLO_VK_DESCRIPTOR_SET_MIN_COUNT) {
        bool attempt_failed = false;
        VkDescriptorPoolSize pool_size = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3u * target_count
        };
        VkDescriptorPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .poolSizeCount = 1u,
            .pPoolSizes = &pool_size,
            .maxSets = target_count
        };
        if (vkCreateDescriptorPool(backend->vk.device, &pool_info, NULL, &backend->vk.descriptor_pool) != VK_SUCCESS) {
            attempt_failed = true;
        }

        if (!attempt_failed) {
            VkDescriptorSetLayout* set_layouts =
                (VkDescriptorSetLayout*)malloc((size_t)target_count * sizeof(*set_layouts));
            if (set_layouts == NULL) {
                attempt_failed = true;
            } else {
                for (uint32_t i = 0u; i < target_count; ++i) {
                    set_layouts[i] = backend->vk.descriptor_set_layout;
                }

                backend->vk.descriptor_sets =
                    (VkDescriptorSet*)calloc((size_t)target_count, sizeof(*backend->vk.descriptor_sets));
                if (backend->vk.descriptor_sets == NULL) {
                    attempt_failed = true;
                } else {
                    VkDescriptorSetAllocateInfo set_info = {
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = backend->vk.descriptor_pool,
                        .descriptorSetCount = target_count,
                        .pSetLayouts = set_layouts
                    };
                    VkResult set_alloc_result =
                        vkAllocateDescriptorSets(backend->vk.device, &set_info, backend->vk.descriptor_sets);
                    if (set_alloc_result == VK_SUCCESS) {
                        backend->vk.descriptor_set_count = target_count;
                    } else {
                        attempt_failed = true;
                    }
                }
            }

            free(set_layouts);
        }

        if (backend->vk.descriptor_set_count != 0u && backend->vk.descriptor_sets != NULL) {
            break;
        }
        if (backend->vk.descriptor_sets != NULL) {
            free(backend->vk.descriptor_sets);
            backend->vk.descriptor_sets = NULL;
        }
        if (backend->vk.descriptor_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(backend->vk.device, backend->vk.descriptor_pool, NULL);
            backend->vk.descriptor_pool = VK_NULL_HANDLE;
        }
        if (target_count == NLO_VK_DESCRIPTOR_SET_MIN_COUNT) {
            break;
        }
        uint32_t next_count = target_count / 2u;
        if (next_count < NLO_VK_DESCRIPTOR_SET_MIN_COUNT) {
            next_count = NLO_VK_DESCRIPTOR_SET_MIN_COUNT;
        }
        fprintf(stderr,
                "[nlo_vk] descriptor set allocation failed at %u sets; retrying at %u sets.\n",
                target_count,
                next_count);
        target_count = next_count;
    }

    if (backend->vk.descriptor_set_count == 0u || backend->vk.descriptor_sets == NULL) {
        fprintf(stderr,
                "[nlo_vk] failed to allocate descriptor sets in range [%u, %u].\n",
                NLO_VK_DESCRIPTOR_SET_MIN_COUNT,
                nlo_vk_select_descriptor_set_target(config));
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkPipelineCacheCreateInfo cache_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
    };
    if (vkCreatePipelineCache(backend->vk.device, &cache_info, NULL, &backend->vk.pipeline_cache) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_create_kernels(nlo_vector_backend* backend)
{
    const char* shader_paths[NLO_VK_KERNEL_COUNT] = {
        NLO_VK_SHADER_REAL_FILL_PATH,
        NLO_VK_SHADER_REAL_MUL_INPLACE_PATH,
        NLO_VK_SHADER_COMPLEX_FILL_PATH,
        NLO_VK_SHADER_COMPLEX_SCALAR_MUL_INPLACE_PATH,
        NLO_VK_SHADER_COMPLEX_ADD_INPLACE_PATH,
        NLO_VK_SHADER_COMPLEX_MUL_INPLACE_PATH,
        NLO_VK_SHADER_COMPLEX_MAGNITUDE_SQUARED_PATH,
        NLO_VK_SHADER_COMPLEX_EXP_INPLACE_PATH,
        NLO_VK_SHADER_COMPLEX_REAL_POW_INPLACE_PATH,
        NLO_VK_SHADER_COMPLEX_RELATIVE_ERROR_REDUCE_PATH,
        NLO_VK_SHADER_REAL_MAX_REDUCE_PATH,
        NLO_VK_SHADER_COMPLEX_WEIGHTED_RMS_REDUCE_PATH,
        NLO_VK_SHADER_PAIR_SUM_REDUCE_PATH,
        NLO_VK_SHADER_COMPLEX_AXIS_UNSHIFTED_FROM_DELTA_PATH,
        NLO_VK_SHADER_COMPLEX_AXIS_CENTERED_FROM_DELTA_PATH,
        NLO_VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_T_PATH,
        NLO_VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_Y_PATH,
        NLO_VK_SHADER_COMPLEX_MESH_FROM_AXIS_TFAST_X_PATH
    };

    for (size_t i = 0u; i < (size_t)NLO_VK_KERNEL_COUNT; ++i) {
        nlo_vec_status status = nlo_vk_create_compute_pipeline(backend, shader_paths[i], &backend->vk.kernels[i].pipeline);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    return NLO_VEC_STATUS_OK;
}

static void nlo_vk_destroy_resources(nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return;
    }

    const VkDevice device = backend->vk.device;
    if (device != VK_NULL_HANDLE &&
        backend->vk.staging_mapped_ptr != NULL &&
        backend->vk.staging_memory != VK_NULL_HANDLE) {
        vkUnmapMemory(device, backend->vk.staging_memory);
        backend->vk.staging_mapped_ptr = NULL;
    }
    nlo_vk_destroy_buffer_raw(backend, &backend->vk.staging_buffer, &backend->vk.staging_memory);
    nlo_vk_destroy_buffer_raw(backend, &backend->vk.reduction_buffer_a, &backend->vk.reduction_memory_a);
    nlo_vk_destroy_buffer_raw(backend, &backend->vk.reduction_buffer_b, &backend->vk.reduction_memory_b);
    backend->vk.reduction_capacity = 0u;

    for (size_t i = 0u; i < (size_t)NLO_VK_KERNEL_COUNT; ++i) {
        if (backend->vk.kernels[i].pipeline != VK_NULL_HANDLE) {
            if (device != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, backend->vk.kernels[i].pipeline, NULL);
            }
            backend->vk.kernels[i].pipeline = VK_NULL_HANDLE;
        }
    }

    if (backend->vk.pipeline_cache != VK_NULL_HANDLE) {
        if (device != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device, backend->vk.pipeline_cache, NULL);
        }
        backend->vk.pipeline_cache = VK_NULL_HANDLE;
    }
    if (backend->vk.descriptor_pool != VK_NULL_HANDLE) {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, backend->vk.descriptor_pool, NULL);
        }
        backend->vk.descriptor_pool = VK_NULL_HANDLE;
    }
    if (backend->vk.descriptor_sets != NULL) {
        free(backend->vk.descriptor_sets);
        backend->vk.descriptor_sets = NULL;
    }
    backend->vk.descriptor_set_count = 0u;
    if (backend->vk.pipeline_layout != VK_NULL_HANDLE) {
        if (device != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, backend->vk.pipeline_layout, NULL);
        }
        backend->vk.pipeline_layout = VK_NULL_HANDLE;
    }
    if (backend->vk.descriptor_set_layout != VK_NULL_HANDLE) {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, backend->vk.descriptor_set_layout, NULL);
        }
        backend->vk.descriptor_set_layout = VK_NULL_HANDLE;
    }
    if (device != VK_NULL_HANDLE &&
        backend->vk.command_buffer != VK_NULL_HANDLE &&
        backend->vk.command_pool != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device,
                             backend->vk.command_pool,
                             1u,
                             &backend->vk.command_buffer);
    }
    backend->vk.command_buffer = VK_NULL_HANDLE;
    if (backend->vk.submit_fence != VK_NULL_HANDLE) {
        if (device != VK_NULL_HANDLE) {
            vkDestroyFence(device, backend->vk.submit_fence, NULL);
        }
        backend->vk.submit_fence = VK_NULL_HANDLE;
    }
    if (device != VK_NULL_HANDLE &&
        backend->vk.command_pool != VK_NULL_HANDLE &&
        backend->vk.owns_command_pool) {
        vkDestroyCommandPool(device, backend->vk.command_pool, NULL);
        backend->vk.command_pool = VK_NULL_HANDLE;
    }
    if (!backend->vk.owns_command_pool) {
        backend->vk.command_pool = VK_NULL_HANDLE;
    }
    backend->vk.owns_command_pool = false;
    backend->vk.staging_mapped_ptr = NULL;
    backend->vk.staging_capacity = 0u;
    backend->vk.simulation_phase_recording = false;
    backend->vk.simulation_phase_has_commands = false;
    backend->vk.simulation_descriptor_set_cursor = 0u;
}

nlo_vec_status nlo_vk_ensure_staging_capacity(nlo_vector_backend* backend, VkDeviceSize min_bytes)
{
    if (backend->vk.staging_buffer != VK_NULL_HANDLE && backend->vk.staging_capacity >= min_bytes) {
        return NLO_VEC_STATUS_OK;
    }

    VkDeviceSize capacity = backend->vk.staging_capacity;
    if (capacity == 0u) {
        capacity = (VkDeviceSize)NLO_VK_DEFAULT_STAGING_BYTES;
    }
    while (capacity < min_bytes) {
        capacity *= 2u;
    }

    VkBuffer new_buffer = VK_NULL_HANDLE;
    VkDeviceMemory new_memory = VK_NULL_HANDLE;
    nlo_vec_status status = nlo_vk_create_buffer_raw(
        backend,
        capacity,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &new_buffer,
        &new_memory);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    void* mapped = NULL;
    if (vkMapMemory(backend->vk.device, new_memory, 0u, capacity, 0u, &mapped) != VK_SUCCESS) {
        nlo_vk_destroy_buffer_raw(backend, &new_buffer, &new_memory);
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (backend->vk.staging_mapped_ptr != NULL && backend->vk.staging_memory != VK_NULL_HANDLE) {
        vkUnmapMemory(backend->vk.device, backend->vk.staging_memory);
        backend->vk.staging_mapped_ptr = NULL;
    }
    nlo_vk_destroy_buffer_raw(backend, &backend->vk.staging_buffer, &backend->vk.staging_memory);

    backend->vk.staging_buffer = new_buffer;
    backend->vk.staging_memory = new_memory;
    backend->vk.staging_mapped_ptr = mapped;
    backend->vk.staging_capacity = capacity;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_ensure_reduction_capacity(nlo_vector_backend* backend, VkDeviceSize min_elements)
{
    if (min_elements == 0u) {
        min_elements = 1u;
    }

    if (backend->vk.reduction_buffer_a != VK_NULL_HANDLE &&
        backend->vk.reduction_buffer_b != VK_NULL_HANDLE &&
        backend->vk.reduction_capacity >= min_elements) {
        return NLO_VEC_STATUS_OK;
    }

    VkDeviceSize capacity = backend->vk.reduction_capacity;
    if (capacity == 0u) {
        capacity = 64u;
    }
    while (capacity < min_elements) {
        capacity *= 2u;
    }

    VkDeviceSize bytes = capacity * (VkDeviceSize)sizeof(double);
    VkBuffer new_a = VK_NULL_HANDLE;
    VkDeviceMemory mem_a = VK_NULL_HANDLE;
    VkBuffer new_b = VK_NULL_HANDLE;
    VkDeviceMemory mem_b = VK_NULL_HANDLE;

    nlo_vec_status status = nlo_vk_create_buffer_raw(
        backend,
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &new_a,
        &mem_a);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_create_buffer_raw(
        backend,
        bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &new_b,
        &mem_b);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_vk_destroy_buffer_raw(backend, &new_a, &mem_a);
        return status;
    }

    nlo_vk_destroy_buffer_raw(backend, &backend->vk.reduction_buffer_a, &backend->vk.reduction_memory_a);
    nlo_vk_destroy_buffer_raw(backend, &backend->vk.reduction_buffer_b, &backend->vk.reduction_memory_b);

    backend->vk.reduction_buffer_a = new_a;
    backend->vk.reduction_memory_a = mem_a;
    backend->vk.reduction_buffer_b = new_b;
    backend->vk.reduction_memory_b = mem_b;
    backend->vk.reduction_capacity = capacity;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_backend_init(nlo_vector_backend* backend, const nlo_vk_backend_config* config)
{
    if (backend == NULL || config == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (config->physical_device == VK_NULL_HANDLE ||
        config->device == VK_NULL_HANDLE ||
        config->queue == VK_NULL_HANDLE) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    VkPhysicalDeviceLimits limits;
    if (!nlo_vk_supports_required_features(config->physical_device, &limits)) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (limits.maxPushConstantsSize < sizeof(nlo_vk_push_constants)) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (limits.maxComputeWorkGroupInvocations < NLO_VK_LOCAL_SIZE_X ||
        limits.maxComputeWorkGroupSize[0] < NLO_VK_LOCAL_SIZE_X ||
        limits.maxComputeWorkGroupCount[0] == 0u ||
        limits.maxStorageBufferRange == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    backend->vk.physical_device = config->physical_device;
    backend->vk.device = config->device;
    backend->vk.queue = config->queue;
    backend->vk.queue_family_index = config->queue_family_index;
    backend->vk.instance = VK_NULL_HANDLE;
    backend->vk.owns_instance = false;
    backend->vk.owns_device = false;
    backend->vk.shader_float64_supported = true;
    backend->vk.limits = limits;
    backend->vk.device_type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    backend->vk.device_local_bytes = 0u;
    backend->vk.device_name[0] = '\0';
    backend->vk.max_kernel_chunk_bytes = (VkDeviceSize)limits.maxStorageBufferRange;
    backend->vk.descriptor_sets = NULL;
    backend->vk.descriptor_set_count = 0u;
    backend->vk.simulation_phase_recording = false;
    backend->vk.simulation_phase_has_commands = false;
    backend->vk.simulation_descriptor_set_cursor = 0u;

    nlo_vec_status status = nlo_vk_create_command_resources(backend, config);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_vk_destroy_resources(backend);
        return status;
    }

    status = nlo_vk_create_descriptor_resources(backend, config);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_vk_destroy_resources(backend);
        return status;
    }

    status = nlo_vk_create_kernels(backend);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_vk_destroy_resources(backend);
        return status;
    }

    VkDeviceSize initial_staging = nlo_vk_min_size((VkDeviceSize)NLO_VK_DEFAULT_STAGING_BYTES, backend->vk.max_kernel_chunk_bytes);
    if (initial_staging == 0u) {
        initial_staging = (VkDeviceSize)NLO_VK_DEFAULT_STAGING_BYTES;
    }
    status = nlo_vk_ensure_staging_capacity(backend, initial_staging);
    if (status != NLO_VEC_STATUS_OK) {
        nlo_vk_destroy_resources(backend);
        return status;
    }

    return NLO_VEC_STATUS_OK;
}

void nlo_vk_backend_shutdown(nlo_vector_backend* backend)
{
    if (backend == NULL) {
        return;
    }
    if (backend->vk.simulation_phase_recording) {
        (void)nlo_vk_simulation_phase_flush(backend);
    }
    if (backend->vk.queue != VK_NULL_HANDLE) {
        (void)vkQueueWaitIdle(backend->vk.queue);
    }
    nlo_vk_destroy_resources(backend);

    if (backend->vk.owns_device && backend->vk.device != VK_NULL_HANDLE) {
        vkDestroyDevice(backend->vk.device, NULL);
    }
    backend->vk.device = VK_NULL_HANDLE;
    backend->vk.queue = VK_NULL_HANDLE;
    backend->vk.physical_device = VK_NULL_HANDLE;
    backend->vk.owns_device = false;

    if (backend->vk.owns_instance && backend->vk.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(backend->vk.instance, NULL);
    }
    backend->vk.instance = VK_NULL_HANDLE;
    backend->vk.owns_instance = false;
}
