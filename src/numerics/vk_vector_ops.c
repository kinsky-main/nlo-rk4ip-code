/**
 * @file vk_vector_ops.c
 * @dir src/numerics
 * @brief Vulkan vector operation implementation details.
 */

#include "numerics/vk_vector_ops.h"

#include "nlo_vk_shader_paths.h"
#include "utility/perf_profile.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VkDeviceSize nlo_vk_min_size(VkDeviceSize a, VkDeviceSize b)
{
    return (a < b) ? a : b;
}

enum {
    NLO_VK_DESCRIPTOR_SET_BUDGET_DEFAULT_BYTES = 16u * 1024u * 1024u,
    NLO_VK_DESCRIPTOR_SET_ESTIMATED_BYTES = 512u,
    NLO_VK_DESCRIPTOR_SET_MIN_COUNT = 16u,
    NLO_VK_DESCRIPTOR_SET_MAX_COUNT = 4096u
};

static nlo_vec_status nlo_vk_begin_commands_raw(nlo_vector_backend* backend);

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

static nlo_vec_status nlo_vk_create_buffer_raw(
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

static void nlo_vk_destroy_buffer_raw(nlo_vector_backend* backend, VkBuffer* buffer, VkDeviceMemory* memory)
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
        NLO_VK_SHADER_REAL_MAX_REDUCE_PATH
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

static nlo_vec_status nlo_vk_ensure_staging_capacity(nlo_vector_backend* backend, VkDeviceSize min_bytes)
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

static nlo_vec_status nlo_vk_ensure_reduction_capacity(nlo_vector_backend* backend, VkDeviceSize min_elements)
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

static nlo_vec_status nlo_vk_begin_commands_raw(nlo_vector_backend* backend)
{
    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    if (vkResetCommandBuffer(backend->vk.command_buffer, 0u) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    if (vkBeginCommandBuffer(backend->vk.command_buffer, &begin_info) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_begin_commands(nlo_vector_backend* backend)
{
    if (backend->in_simulation) {
        return nlo_vk_simulation_phase_begin(backend);
    }
    return nlo_vk_begin_commands_raw(backend);
}

void nlo_vk_simulation_phase_mark_commands(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return;
    }
    backend->vk.simulation_phase_has_commands = true;
}

nlo_vec_status nlo_vk_simulation_phase_begin(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->vk.simulation_phase_recording) {
        return NLO_VEC_STATUS_OK;
    }

    nlo_vec_status status = nlo_vk_begin_commands_raw(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    backend->vk.simulation_phase_recording = true;
    backend->vk.simulation_phase_has_commands = false;
    backend->vk.simulation_descriptor_set_cursor = 0u;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_simulation_phase_command_buffer(
    nlo_vector_backend* backend,
    VkCommandBuffer* out_command_buffer
)
{
    if (backend == NULL || out_command_buffer == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_vec_status status = nlo_vk_simulation_phase_begin(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    *out_command_buffer = backend->vk.command_buffer;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_simulation_phase_flush(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!backend->vk.simulation_phase_recording) {
        return NLO_VEC_STATUS_OK;
    }

    if (vkEndCommandBuffer(backend->vk.command_buffer) != VK_SUCCESS) {
        backend->vk.simulation_phase_recording = false;
        backend->vk.simulation_phase_has_commands = false;
        backend->vk.simulation_descriptor_set_cursor = 0u;
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (backend->vk.simulation_phase_has_commands) {
        if (vkResetFences(backend->vk.device, 1u, &backend->vk.submit_fence) != VK_SUCCESS) {
            backend->vk.simulation_phase_recording = false;
            backend->vk.simulation_phase_has_commands = false;
            backend->vk.simulation_descriptor_set_cursor = 0u;
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }

        VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1u,
            .pCommandBuffers = &backend->vk.command_buffer
        };
        if (vkQueueSubmit(backend->vk.queue, 1u, &submit_info, backend->vk.submit_fence) != VK_SUCCESS) {
            backend->vk.simulation_phase_recording = false;
            backend->vk.simulation_phase_has_commands = false;
            backend->vk.simulation_descriptor_set_cursor = 0u;
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
        if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            backend->vk.simulation_phase_recording = false;
            backend->vk.simulation_phase_has_commands = false;
            backend->vk.simulation_descriptor_set_cursor = 0u;
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
    }

    backend->vk.simulation_phase_recording = false;
    backend->vk.simulation_phase_has_commands = false;
    backend->vk.simulation_descriptor_set_cursor = 0u;
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_submit_commands(nlo_vector_backend* backend)
{
    if (backend->in_simulation) {
        nlo_vk_simulation_phase_mark_commands(backend);
        return NLO_VEC_STATUS_OK;
    }

    if (vkEndCommandBuffer(backend->vk.command_buffer) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkResetFences(backend->vk.device, 1u, &backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1u,
        .pCommandBuffers = &backend->vk.command_buffer
    };
    if (vkQueueSubmit(backend->vk.queue, 1u, &submit_info, backend->vk.submit_fence) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (vkWaitForFences(backend->vk.device, 1u, &backend->vk.submit_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    return NLO_VEC_STATUS_OK;
}

static void nlo_vk_cmd_buffer_barrier(
    VkCommandBuffer cmd,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkPipelineStageFlags src_stage,
    VkAccessFlags src_access,
    VkPipelineStageFlags dst_stage,
    VkAccessFlags dst_access
)
{
    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = offset,
        .size = size
    };
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0u, 0u, NULL, 1u, &barrier, 0u, NULL);
}

static void nlo_vk_cmd_transfer_to_compute(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

static void nlo_vk_cmd_compute_to_compute(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

static void nlo_vk_cmd_compute_to_transfer(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
}

static void nlo_vk_cmd_transfer_to_host(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size)
{
    nlo_vk_cmd_buffer_barrier(cmd,
                              buffer,
                              offset,
                              size,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_PIPELINE_STAGE_HOST_BIT,
                              VK_ACCESS_HOST_READ_BIT);
}

static size_t nlo_vk_max_chunk_elements(const nlo_vector_backend* backend, size_t elem_size)
{
    uint64_t by_storage = (uint64_t)backend->vk.limits.maxStorageBufferRange / (uint64_t)elem_size;
    uint64_t by_dispatch = (uint64_t)backend->vk.limits.maxComputeWorkGroupCount[0] * (uint64_t)NLO_VK_LOCAL_SIZE_X;
    uint64_t max_elems = (by_storage < by_dispatch) ? by_storage : by_dispatch;
    if (max_elems > (uint64_t)UINT32_MAX) {
        max_elems = (uint64_t)UINT32_MAX;
    }
    return (size_t)max_elems;
}

static nlo_vec_status nlo_vk_acquire_descriptor_set(
    nlo_vector_backend* backend,
    VkDescriptorSet* out_descriptor_set
)
{
    if (backend == NULL || out_descriptor_set == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->vk.descriptor_sets == NULL || backend->vk.descriptor_set_count == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    if (!backend->in_simulation) {
        *out_descriptor_set = backend->vk.descriptor_sets[0];
        return NLO_VEC_STATUS_OK;
    }

    const uint32_t index = backend->vk.simulation_descriptor_set_cursor;
    if (index >= backend->vk.descriptor_set_count) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    *out_descriptor_set = backend->vk.descriptor_sets[index];
    backend->vk.simulation_descriptor_set_cursor = index + 1u;
    return NLO_VEC_STATUS_OK;
}

static void nlo_vk_update_descriptor_set(
    nlo_vector_backend* backend,
    VkDescriptorSet descriptor_set,
    VkBuffer dst_buffer,
    VkDeviceSize dst_offset,
    VkDeviceSize dst_size,
    VkBuffer src_buffer,
    VkDeviceSize src_offset,
    VkDeviceSize src_size,
    VkBuffer src2_buffer,
    VkDeviceSize src2_offset,
    VkDeviceSize src2_size
)
{
    VkDescriptorBufferInfo dst_info = {
        .buffer = dst_buffer,
        .offset = dst_offset,
        .range = dst_size
    };
    VkDescriptorBufferInfo src_info = {
        .buffer = src_buffer,
        .offset = src_offset,
        .range = src_size
    };
    VkDescriptorBufferInfo src2_info = {
        .buffer = src2_buffer,
        .offset = src2_offset,
        .range = src2_size
    };
    VkWriteDescriptorSet writes[3] = {0};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descriptor_set;
    writes[0].dstBinding = 0u;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1u;
    writes[0].pBufferInfo = &dst_info;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descriptor_set;
    writes[1].dstBinding = 1u;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1u;
    writes[1].pBufferInfo = &src_info;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = descriptor_set;
    writes[2].dstBinding = 2u;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].descriptorCount = 1u;
    writes[2].pBufferInfo = &src2_info;

    vkUpdateDescriptorSets(backend->vk.device, 3u, writes, 0u, NULL);
}

static nlo_vec_status nlo_vk_dispatch_kernel(
    nlo_vector_backend* backend,
    nlo_vk_kernel_id kernel_id,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src,
    size_t elem_size,
    size_t length,
    double scalar0,
    double scalar1
)
{
    const size_t max_chunk_elems = nlo_vk_max_chunk_elements(backend, elem_size);
    if (max_chunk_elems == 0u) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    size_t offset_elems = 0u;
    while (offset_elems < length) {
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        size_t chunk_elems = length - offset_elems;
        if (chunk_elems > max_chunk_elems) {
            chunk_elems = max_chunk_elems;
        }

        nlo_vec_status status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkBuffer src_buffer = (src != NULL) ? src->vk_buffer : dst->vk_buffer;
        VkDeviceSize byte_offset = (VkDeviceSize)(offset_elems * elem_size);
        VkDeviceSize chunk_bytes = (VkDeviceSize)(chunk_elems * elem_size);

        nlo_vk_update_descriptor_set(backend,
                                     descriptor_set,
                                     dst->vk_buffer,
                                     byte_offset,
                                     chunk_bytes,
                                     src_buffer,
                                     byte_offset,
                                     chunk_bytes,
                                     src_buffer,
                                     byte_offset,
                                     chunk_bytes);

        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        nlo_vk_cmd_transfer_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        if (src_buffer != dst->vk_buffer) {
            nlo_vk_cmd_transfer_to_compute(cmd, src_buffer, byte_offset, chunk_bytes);
            nlo_vk_cmd_compute_to_compute(cmd, src_buffer, byte_offset, chunk_bytes);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, backend->vk.kernels[kernel_id].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        nlo_vk_push_constants push = {
            .count = (uint32_t)chunk_elems,
            .pad = 0u,
            .scalar0 = scalar0,
            .scalar1 = scalar1
        };
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);

        uint32_t groups = (uint32_t)((chunk_elems + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
        vkCmdDispatch(cmd, groups, 1u, 1u);

        nlo_vk_cmd_compute_to_compute(cmd, dst->vk_buffer, byte_offset, chunk_bytes);
        nlo_vk_cmd_compute_to_transfer(cmd, dst->vk_buffer, byte_offset, chunk_bytes);

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_dispatch(1u,
                                          2u,
                                          2u * (uint64_t)chunk_bytes);

        offset_elems += chunk_elems;
    }

    return NLO_VEC_STATUS_OK;
}

static uint32_t nlo_vk_dispatch_groups_for_count(size_t count)
{
    if (count == 0u) {
        return 1u;
    }
    return (uint32_t)((count + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) / (size_t)NLO_VK_LOCAL_SIZE_X);
}

static nlo_vec_status nlo_vk_dispatch_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (backend == NULL || current == NULL || previous == NULL || out_error == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (current->length != previous->length) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (current->length > (size_t)UINT32_MAX) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    uint32_t groups = nlo_vk_dispatch_groups_for_count(current->length);
    uint32_t required_sets = 1u;
    uint32_t reduce_count = groups;
    while (reduce_count > 1u) {
        reduce_count = nlo_vk_dispatch_groups_for_count((size_t)reduce_count);
        required_sets += 1u;
    }
    uint32_t available_sets = backend->vk.descriptor_set_count;
    if (backend->in_simulation) {
        const uint32_t used = backend->vk.simulation_descriptor_set_cursor;
        if (used > available_sets) {
            return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
        }
        available_sets -= used;
    }
    if (required_sets > available_sets) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    nlo_vec_status status = nlo_vk_ensure_reduction_capacity(backend, (VkDeviceSize)groups);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    status = nlo_vk_ensure_staging_capacity(backend, (VkDeviceSize)sizeof(double));
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_begin_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkCommandBuffer cmd = backend->vk.command_buffer;
    VkDeviceSize complex_bytes = (VkDeviceSize)current->bytes;
    VkDeviceSize partial_bytes = (VkDeviceSize)groups * (VkDeviceSize)sizeof(double);
    uint64_t dispatch_count = 0u;
    uint64_t dispatch_pass_count = 0u;
    uint64_t dispatch_pass_bytes = 0u;

    nlo_vk_cmd_compute_to_compute(cmd, current->vk_buffer, 0u, complex_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, previous->vk_buffer, 0u, complex_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_b, 0u, partial_bytes);

    nlo_vk_update_descriptor_set(backend,
                                 descriptor_set,
                                 backend->vk.reduction_buffer_a,
                                 0u,
                                 partial_bytes,
                                 current->vk_buffer,
                                 0u,
                                 complex_bytes,
                                 previous->vk_buffer,
                                 0u,
                                 complex_bytes);

    vkCmdBindPipeline(cmd,
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      backend->vk.kernels[NLO_VK_KERNEL_COMPLEX_RELATIVE_ERROR_REDUCE].pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            backend->vk.pipeline_layout,
                            0u,
                            1u,
                            &descriptor_set,
                            0u,
                            NULL);

    nlo_vk_push_constants push = {
        .count = (uint32_t)current->length,
        .pad = 0u,
        .scalar0 = epsilon,
        .scalar1 = 0.0
    };
    vkCmdPushConstants(cmd,
                       backend->vk.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0u,
                       (uint32_t)sizeof(push),
                       &push);
    vkCmdDispatch(cmd, groups, 1u, 1u);
    dispatch_count += 1u;
    dispatch_pass_count += 2u;
    dispatch_pass_bytes += (uint64_t)partial_bytes * 2u;
    nlo_vk_cmd_compute_to_compute(cmd, backend->vk.reduction_buffer_a, 0u, partial_bytes);

    VkBuffer src_buffer = backend->vk.reduction_buffer_a;
    VkBuffer dst_buffer = backend->vk.reduction_buffer_b;
    uint32_t count = groups;
    while (count > 1u) {
        status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        uint32_t next_groups = nlo_vk_dispatch_groups_for_count((size_t)count);
        VkDeviceSize src_bytes = (VkDeviceSize)count * (VkDeviceSize)sizeof(double);
        VkDeviceSize dst_bytes = (VkDeviceSize)next_groups * (VkDeviceSize)sizeof(double);

        nlo_vk_update_descriptor_set(backend,
                                     descriptor_set,
                                     dst_buffer,
                                     0u,
                                     dst_bytes,
                                     src_buffer,
                                     0u,
                                     src_bytes,
                                     src_buffer,
                                     0u,
                                     src_bytes);

        vkCmdBindPipeline(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          backend->vk.kernels[NLO_VK_KERNEL_REAL_MAX_REDUCE].pipeline);
        vkCmdBindDescriptorSets(cmd,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                backend->vk.pipeline_layout,
                                0u,
                                1u,
                                &descriptor_set,
                                0u,
                                NULL);

        push.count = count;
        push.scalar0 = 0.0;
        push.scalar1 = 0.0;
        vkCmdPushConstants(cmd,
                           backend->vk.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0u,
                           (uint32_t)sizeof(push),
                           &push);
        vkCmdDispatch(cmd, next_groups, 1u, 1u);
        dispatch_count += 1u;
        dispatch_pass_count += 2u;
        dispatch_pass_bytes += (uint64_t)src_bytes + (uint64_t)dst_bytes;
        nlo_vk_cmd_compute_to_compute(cmd, dst_buffer, 0u, dst_bytes);

        VkBuffer tmp = src_buffer;
        src_buffer = dst_buffer;
        dst_buffer = tmp;
        count = next_groups;
    }

    nlo_vk_cmd_compute_to_transfer(cmd, src_buffer, 0u, (VkDeviceSize)sizeof(double));

    VkBufferCopy copy = {
        .srcOffset = 0u,
        .dstOffset = 0u,
        .size = (VkDeviceSize)sizeof(double)
    };
    vkCmdCopyBuffer(cmd, src_buffer, backend->vk.staging_buffer, 1u, &copy);
    nlo_vk_cmd_transfer_to_host(cmd, backend->vk.staging_buffer, 0u, (VkDeviceSize)sizeof(double));

    status = nlo_vk_submit_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }
    nlo_perf_profile_add_gpu_dispatch(dispatch_count,
                                      dispatch_pass_count,
                                      dispatch_pass_bytes);
    nlo_perf_profile_add_gpu_copy(1u, (uint64_t)sizeof(double));
    nlo_perf_profile_add_gpu_download(1u, (uint64_t)sizeof(double));
    if (backend->in_simulation) {
        status = nlo_vk_simulation_phase_flush(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
    }

    double ratio = 0.0;
    memcpy(&ratio, backend->vk.staging_mapped_ptr, sizeof(double));
    if (ratio < 0.0) {
        ratio = 0.0;
    }
    *out_error = sqrt(ratio);
    return NLO_VEC_STATUS_OK;
}

static nlo_vec_status nlo_vk_copy_buffer_chunked(
    nlo_vector_backend* backend,
    VkBuffer src_buffer,
    VkBuffer dst_buffer,
    size_t bytes
)
{
    VkDeviceSize remaining = (VkDeviceSize)bytes;
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = nlo_vk_min_size(remaining, backend->vk.max_kernel_chunk_bytes);
        nlo_vec_status status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkCommandBuffer cmd = backend->vk.command_buffer;
        nlo_vk_cmd_compute_to_transfer(cmd, src_buffer, offset, chunk);
        nlo_vk_cmd_compute_to_transfer(cmd, dst_buffer, offset, chunk);

        VkBufferCopy copy = {
            .srcOffset = offset,
            .dstOffset = offset,
            .size = chunk
        };
        vkCmdCopyBuffer(cmd, src_buffer, dst_buffer, 1u, &copy);
        nlo_vk_cmd_transfer_to_compute(cmd, dst_buffer, offset, chunk);

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_copy(1u, (uint64_t)chunk);

        offset += chunk;
        remaining -= chunk;
    }

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

nlo_vec_status nlo_vk_buffer_create(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    return nlo_vk_create_buffer_raw(backend,
                                    (VkDeviceSize)buffer->bytes,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                    &buffer->vk_buffer,
                                    &buffer->vk_memory);
}

void nlo_vk_buffer_destroy(nlo_vector_backend* backend, nlo_vec_buffer* buffer)
{
    nlo_vk_destroy_buffer_raw(backend, &buffer->vk_buffer, &buffer->vk_memory);
}

nlo_vec_status nlo_vk_upload(
    nlo_vector_backend* backend,
    nlo_vec_buffer* buffer,
    const void* data,
    size_t bytes
)
{
    nlo_vec_status status = nlo_vk_ensure_staging_capacity(backend, (VkDeviceSize)NLO_VK_DEFAULT_STAGING_BYTES);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    const uint8_t* src = (const uint8_t*)data;
    VkDeviceSize remaining = (VkDeviceSize)bytes;
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = nlo_vk_min_size(remaining, backend->vk.staging_capacity);
        memcpy(backend->vk.staging_mapped_ptr, src + (size_t)offset, (size_t)chunk);

        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        VkBufferCopy copy = {
            .srcOffset = 0u,
            .dstOffset = offset,
            .size = chunk
        };
        vkCmdCopyBuffer(backend->vk.command_buffer, backend->vk.staging_buffer, buffer->vk_buffer, 1u, &copy);
        nlo_vk_cmd_transfer_to_compute(backend->vk.command_buffer, buffer->vk_buffer, offset, chunk);

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_copy(1u, (uint64_t)chunk);
        nlo_perf_profile_add_gpu_upload(1u, (uint64_t)chunk);

        remaining -= chunk;
        offset += chunk;
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_download(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* buffer,
    void* data,
    size_t bytes
)
{
    nlo_vec_status status = nlo_vk_ensure_staging_capacity(backend, (VkDeviceSize)NLO_VK_DEFAULT_STAGING_BYTES);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    uint8_t* dst = (uint8_t*)data;
    VkDeviceSize remaining = (VkDeviceSize)bytes;
    VkDeviceSize offset = 0u;

    while (remaining > 0u) {
        VkDeviceSize chunk = nlo_vk_min_size(remaining, backend->vk.staging_capacity);
        status = nlo_vk_begin_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }

        nlo_vk_cmd_compute_to_transfer(backend->vk.command_buffer, buffer->vk_buffer, offset, chunk);
        VkBufferCopy copy = {
            .srcOffset = offset,
            .dstOffset = 0u,
            .size = chunk
        };
        vkCmdCopyBuffer(backend->vk.command_buffer, buffer->vk_buffer, backend->vk.staging_buffer, 1u, &copy);
        nlo_vk_cmd_transfer_to_host(backend->vk.command_buffer, backend->vk.staging_buffer, 0u, chunk);

        status = nlo_vk_submit_commands(backend);
        if (status != NLO_VEC_STATUS_OK) {
            return status;
        }
        nlo_perf_profile_add_gpu_copy(1u, (uint64_t)chunk);
        nlo_perf_profile_add_gpu_download(1u, (uint64_t)chunk);

        memcpy(dst + (size_t)offset, backend->vk.staging_mapped_ptr, (size_t)chunk);
        remaining -= chunk;
        offset += chunk;
    }

    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_vk_op_real_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, double value)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_REAL_FILL,
                                  dst,
                                  NULL,
                                  sizeof(double),
                                  dst->length,
                                  value,
                                  0.0);
}

nlo_vec_status nlo_vk_op_real_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    return nlo_vk_copy_buffer_chunked(backend, src->vk_buffer, dst->vk_buffer, dst->bytes);
}

nlo_vec_status nlo_vk_op_real_mul_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_REAL_MUL_INPLACE,
                                  dst,
                                  src,
                                  sizeof(double),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_fill(nlo_vector_backend* backend, nlo_vec_buffer* dst, nlo_complex value)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_FILL,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  NLO_RE(value),
                                  NLO_IM(value));
}

nlo_vec_status nlo_vk_op_complex_copy(nlo_vector_backend* backend, nlo_vec_buffer* dst, const nlo_vec_buffer* src)
{
    return nlo_vk_copy_buffer_chunked(backend, src->vk_buffer, dst->vk_buffer, dst->bytes);
}

nlo_vec_status nlo_vk_op_complex_magnitude_squared(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* src,
    nlo_vec_buffer* dst
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_MAGNITUDE_SQUARED,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_scalar_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    nlo_complex alpha
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_SCALAR_MUL_INPLACE,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  NLO_RE(alpha),
                                  NLO_IM(alpha));
}

nlo_vec_status nlo_vk_op_complex_mul_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_MUL_INPLACE,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_add_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    const nlo_vec_buffer* src
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_ADD_INPLACE,
                                  dst,
                                  src,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_exp_inplace(nlo_vector_backend* backend, nlo_vec_buffer* dst)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_EXP_INPLACE,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  0.0,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_real_pow_inplace(
    nlo_vector_backend* backend,
    nlo_vec_buffer* dst,
    double exponent
)
{
    return nlo_vk_dispatch_kernel(backend,
                                  NLO_VK_KERNEL_COMPLEX_REAL_POW_INPLACE,
                                  dst,
                                  NULL,
                                  sizeof(nlo_complex),
                                  dst->length,
                                  exponent,
                                  0.0);
}

nlo_vec_status nlo_vk_op_complex_relative_error(
    nlo_vector_backend* backend,
    const nlo_vec_buffer* current,
    const nlo_vec_buffer* previous,
    double epsilon,
    double* out_error
)
{
    if (epsilon <= 0.0) {
        epsilon = 1e-12;
    }
    return nlo_vk_dispatch_complex_relative_error(backend, current, previous, epsilon, out_error);
}


