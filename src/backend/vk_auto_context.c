/**
 * @file vk_auto_context.c
 * @brief Internal Vulkan auto-detection context helpers.
 */

#include "backend/vk_auto_context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void nlo_vk_auto_copy_reason(
    char* reason,
    size_t reason_capacity,
    const char* message
)
{
    if (reason == NULL || reason_capacity == 0u) {
        return;
    }

    if (message == NULL) {
        reason[0] = '\0';
        return;
    }

#if defined(_MSC_VER)
    strncpy_s(reason, reason_capacity, message, _TRUNCATE);
#else
    snprintf(reason, reason_capacity, "%s", message);
#endif
}

static void nlo_vk_auto_reset_context(nlo_vk_auto_context* context)
{
    if (context == NULL) {
        return;
    }

    memset(context, 0, sizeof(*context));
}

static uint64_t nlo_vk_auto_device_local_bytes(VkPhysicalDevice physical_device)
{
    if (physical_device == VK_NULL_HANDLE) {
        return 0u;
    }

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    uint64_t total = 0u;
    for (uint32_t i = 0u; i < memory_properties.memoryHeapCount; ++i) {
        if ((memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
            total += memory_properties.memoryHeaps[i].size;
        }
    }

    return total;
}

static int nlo_vk_auto_find_compute_queue_family(
    VkPhysicalDevice physical_device,
    uint32_t* out_queue_family_index
)
{
    if (physical_device == VK_NULL_HANDLE || out_queue_family_index == NULL) {
        return -1;
    }

    uint32_t family_count = 0u;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, NULL);
    if (family_count == 0u) {
        return -1;
    }

    VkQueueFamilyProperties* families =
        (VkQueueFamilyProperties*)calloc((size_t)family_count, sizeof(*families));
    if (families == NULL) {
        return -1;
    }

    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, families);

    int found = 0;
    for (uint32_t i = 0u; i < family_count; ++i) {
        if (families[i].queueCount > 0u &&
            (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
            *out_queue_family_index = i;
            found = 1;
            break;
        }
    }

    free(families);
    return found ? 0 : -1;
}

typedef struct {
    VkPhysicalDevice physical_device;
    uint32_t queue_family_index;
    VkPhysicalDeviceType device_type;
    uint64_t device_local_bytes;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
} nlo_vk_device_candidate;

static int nlo_vk_auto_candidate_is_better(
    const nlo_vk_device_candidate* candidate,
    const nlo_vk_device_candidate* best,
    int has_best
)
{
    if (candidate == NULL) {
        return 0;
    }
    if (!has_best || best == NULL) {
        return 1;
    }

    const int candidate_discrete =
        (candidate->device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1 : 0;
    const int best_discrete =
        (best->device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1 : 0;
    if (candidate_discrete != best_discrete) {
        return candidate_discrete > best_discrete;
    }

    if (candidate->device_local_bytes != best->device_local_bytes) {
        return candidate->device_local_bytes > best->device_local_bytes;
    }

    return 0;
}

static int nlo_vk_auto_select_device(
    VkInstance instance,
    nlo_vk_device_candidate* out_candidate
)
{
    if (instance == VK_NULL_HANDLE || out_candidate == NULL) {
        return -1;
    }

    uint32_t device_count = 0u;
    if (vkEnumeratePhysicalDevices(instance, &device_count, NULL) != VK_SUCCESS ||
        device_count == 0u) {
        return -1;
    }

    VkPhysicalDevice* devices =
        (VkPhysicalDevice*)calloc((size_t)device_count, sizeof(*devices));
    if (devices == NULL) {
        return -1;
    }

    const VkResult enumerate_result = vkEnumeratePhysicalDevices(instance, &device_count, devices);
    if (enumerate_result != VK_SUCCESS) {
        free(devices);
        return -1;
    }

    nlo_vk_device_candidate best = {0};
    int has_best = 0;
    for (uint32_t i = 0u; i < device_count; ++i) {
        VkPhysicalDeviceFeatures features = {0};
        vkGetPhysicalDeviceFeatures(devices[i], &features);
        if (features.shaderFloat64 != VK_TRUE) {
            continue;
        }

        uint32_t queue_family_index = 0u;
        if (nlo_vk_auto_find_compute_queue_family(devices[i], &queue_family_index) != 0) {
            continue;
        }

        VkPhysicalDeviceProperties properties = {0};
        vkGetPhysicalDeviceProperties(devices[i], &properties);

        nlo_vk_device_candidate candidate = {0};
        candidate.physical_device = devices[i];
        candidate.queue_family_index = queue_family_index;
        candidate.device_type = properties.deviceType;
        candidate.device_local_bytes = nlo_vk_auto_device_local_bytes(devices[i]);
#if defined(_MSC_VER)
        strncpy_s(candidate.device_name,
                  sizeof(candidate.device_name),
                  properties.deviceName,
                  _TRUNCATE);
#else
        snprintf(candidate.device_name, sizeof(candidate.device_name), "%s", properties.deviceName);
#endif

        if (nlo_vk_auto_candidate_is_better(&candidate, &best, has_best)) {
            best = candidate;
            has_best = 1;
        }
    }

    free(devices);
    if (!has_best) {
        return -1;
    }

    *out_candidate = best;
    return 0;
}

int nlo_vk_auto_context_init(
    nlo_vk_auto_context* context,
    char* reason,
    size_t reason_capacity
)
{
    if (context == NULL) {
        nlo_vk_auto_copy_reason(reason, reason_capacity, "Vulkan auto-context pointer is null.");
        return -1;
    }

    nlo_vk_auto_reset_context(context);
    nlo_vk_auto_copy_reason(reason, reason_capacity, "");

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "nlolib",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "nlolib",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2
    };

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };
    if (vkCreateInstance(&instance_info, NULL, &context->instance) != VK_SUCCESS) {
        nlo_vk_auto_copy_reason(reason, reason_capacity, "Failed to create Vulkan instance.");
        return -1;
    }

    nlo_vk_device_candidate candidate = {0};
    if (nlo_vk_auto_select_device(context->instance, &candidate) != 0) {
        nlo_vk_auto_copy_reason(reason,
                                reason_capacity,
                                "No compatible Vulkan device with shaderFloat64 and compute queue.");
        nlo_vk_auto_context_destroy(context);
        return -1;
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = candidate.queue_family_index,
        .queueCount = 1u,
        .pQueuePriorities = &queue_priority
    };

    VkPhysicalDeviceFeatures device_features = {
        .shaderFloat64 = VK_TRUE
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1u,
        .pQueueCreateInfos = &queue_info,
        .pEnabledFeatures = &device_features
    };
    if (vkCreateDevice(candidate.physical_device, &device_info, NULL, &context->device) != VK_SUCCESS) {
        nlo_vk_auto_copy_reason(reason, reason_capacity, "Failed to create Vulkan logical device.");
        nlo_vk_auto_context_destroy(context);
        return -1;
    }

    vkGetDeviceQueue(context->device, candidate.queue_family_index, 0u, &context->queue);
    if (context->queue == VK_NULL_HANDLE) {
        nlo_vk_auto_copy_reason(reason, reason_capacity, "Failed to get Vulkan compute queue.");
        nlo_vk_auto_context_destroy(context);
        return -1;
    }

    context->physical_device = candidate.physical_device;
    context->queue_family_index = candidate.queue_family_index;
    context->device_type = candidate.device_type;
    context->device_local_bytes = candidate.device_local_bytes;
#if defined(_MSC_VER)
    strncpy_s(context->device_name, sizeof(context->device_name), candidate.device_name, _TRUNCATE);
#else
    snprintf(context->device_name, sizeof(context->device_name), "%s", candidate.device_name);
#endif
    return 0;
}

void nlo_vk_auto_context_destroy(nlo_vk_auto_context* context)
{
    if (context == NULL) {
        return;
    }

    if (context->device != VK_NULL_HANDLE) {
        (void)vkDeviceWaitIdle(context->device);
        vkDestroyDevice(context->device, NULL);
        context->device = VK_NULL_HANDLE;
    }

    if (context->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context->instance, NULL);
        context->instance = VK_NULL_HANDLE;
    }

    context->physical_device = VK_NULL_HANDLE;
    context->queue = VK_NULL_HANDLE;
    context->queue_family_index = 0u;
    context->device_type = 0;
    context->device_local_bytes = 0u;
    context->device_name[0] = '\0';
}
