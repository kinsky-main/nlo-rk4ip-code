/**
 * @file vulkan_bench_context.c
 * @brief Vulkan setup helpers for benchmark executables.
 */

#include "vulkan_bench_context.h"

#ifdef NLO_ENABLE_VECTOR_BACKEND_VULKAN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void nlo_bench_copy_reason(
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

static int nlo_bench_find_compute_queue_family(
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
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
            *out_queue_family_index = i;
            found = 1;
            break;
        }
    }

    free(families);
    return found ? 0 : -1;
}

static int nlo_bench_pick_physical_device(
    VkInstance instance,
    VkPhysicalDevice* out_physical_device,
    uint32_t* out_queue_family_index
)
{
    if (instance == VK_NULL_HANDLE ||
        out_physical_device == NULL ||
        out_queue_family_index == NULL) {
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

    VkResult enumerate_result = vkEnumeratePhysicalDevices(instance, &device_count, devices);
    if (enumerate_result != VK_SUCCESS) {
        free(devices);
        return -1;
    }

    int found = 0;
    for (uint32_t i = 0u; i < device_count; ++i) {
        VkPhysicalDeviceFeatures features = {0};
        vkGetPhysicalDeviceFeatures(devices[i], &features);
        if (features.shaderFloat64 == VK_FALSE) {
            continue;
        }

        uint32_t queue_family_index = 0u;
        if (nlo_bench_find_compute_queue_family(devices[i], &queue_family_index) != 0) {
            continue;
        }

        *out_physical_device = devices[i];
        *out_queue_family_index = queue_family_index;
        found = 1;
        break;
    }

    free(devices);
    return found ? 0 : -1;
}

int nlo_bench_vk_context_init(
    nlo_bench_vk_context* context,
    char* reason,
    size_t reason_capacity
)
{
    if (context == NULL) {
        nlo_bench_copy_reason(reason, reason_capacity, "Vulkan context pointer is null.");
        return -1;
    }

    memset(context, 0, sizeof(*context));
    nlo_bench_copy_reason(reason, reason_capacity, "");

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "nlolib-bench",
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
        nlo_bench_copy_reason(reason, reason_capacity, "Failed to create Vulkan instance.");
        return -1;
    }

    if (nlo_bench_pick_physical_device(context->instance,
                                       &context->physical_device,
                                       &context->queue_family_index) != 0) {
        nlo_bench_copy_reason(reason,
                              reason_capacity,
                              "No suitable Vulkan physical device with compute + shaderFloat64.");
        nlo_bench_vk_context_destroy(context);
        return -1;
    }

    const float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = context->queue_family_index,
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

    if (vkCreateDevice(context->physical_device, &device_info, NULL, &context->device) != VK_SUCCESS) {
        nlo_bench_copy_reason(reason, reason_capacity, "Failed to create Vulkan logical device.");
        nlo_bench_vk_context_destroy(context);
        return -1;
    }

    vkGetDeviceQueue(context->device, context->queue_family_index, 0u, &context->queue);
    if (context->queue == VK_NULL_HANDLE) {
        nlo_bench_copy_reason(reason, reason_capacity, "Failed to get Vulkan queue.");
        nlo_bench_vk_context_destroy(context);
        return -1;
    }

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = context->queue_family_index
    };

    if (vkCreateCommandPool(context->device, &pool_info, NULL, &context->command_pool) != VK_SUCCESS) {
        nlo_bench_copy_reason(reason, reason_capacity, "Failed to create Vulkan command pool.");
        nlo_bench_vk_context_destroy(context);
        return -1;
    }

    nlo_bench_copy_reason(reason, reason_capacity, "");
    return 0;
}

void nlo_bench_vk_context_destroy(nlo_bench_vk_context* context)
{
    if (context == NULL) {
        return;
    }

    if (context->device != VK_NULL_HANDLE) {
        (void)vkDeviceWaitIdle(context->device);
    }

    if (context->command_pool != VK_NULL_HANDLE && context->device != VK_NULL_HANDLE) {
        vkDestroyCommandPool(context->device, context->command_pool, NULL);
        context->command_pool = VK_NULL_HANDLE;
    }

    if (context->device != VK_NULL_HANDLE) {
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
}

#endif
