/**
 * @file vk_auto_context.h
 * @brief Internal Vulkan auto-detection context helpers.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

/**
 * @brief Auto-detected Vulkan handles and device metadata.
 */
typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family_index;
    VkPhysicalDeviceType device_type;
    uint64_t device_local_bytes;
    char device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
} nlo_vk_auto_context;

/**
 * @brief Create a minimal Vulkan context by selecting a suitable device/queue.
 *
 * @param context Destination context object.
 * @param reason Optional human-readable failure reason buffer.
 * @param reason_capacity Size of @p reason in bytes.
 * @return int 0 on success, nonzero on failure.
 */
int nlo_vk_auto_context_init(
    nlo_vk_auto_context* context,
    char* reason,
    size_t reason_capacity
);

/**
 * @brief Destroy resources owned by an auto-created Vulkan context.
 *
 * @param context Context to destroy/reset.
 */
void nlo_vk_auto_context_destroy(nlo_vk_auto_context* context);
