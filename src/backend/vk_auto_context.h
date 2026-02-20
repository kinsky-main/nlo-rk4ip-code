/**
 * @file vk_auto_context.h
 * @brief Internal Vulkan auto-detection context helpers.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

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

int nlo_vk_auto_context_init(
    nlo_vk_auto_context* context,
    char* reason,
    size_t reason_capacity
);

void nlo_vk_auto_context_destroy(nlo_vk_auto_context* context);
