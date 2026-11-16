/**
 * @file vk_auto_context.h
 * @brief Internal Vulkan auto-detection context helpers.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef ENABLE_VULKAN_BACKEND
#define ENABLE_VULKAN_BACKEND 1
#endif

#if ENABLE_VULKAN_BACKEND
#include <vulkan/vulkan.h>
#else
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef int VkPhysicalDeviceType;
#ifndef VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256
#endif
#endif

/**
 * @brief Auto-detected Vulkan handles and device metadata.
 *
 * @p instance is the process-wide instance owned by this translation unit, not
 * by the context: vk_auto_context_destroy() leaves it alive for the next
 * context.  See vk_auto_instance_shutdown().
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
} vk_auto_context;

/**
 * @brief Create a minimal Vulkan context by selecting a suitable device/queue.
 *
 * @param context Destination context object.
 * @param reason Optional human-readable failure reason buffer.
 * @param reason_capacity Size of @p reason in bytes.
 * @return int 0 on success, nonzero on failure.
 */
int vk_auto_context_init(
    vk_auto_context* context,
    char* reason,
    size_t reason_capacity
);

/**
 * @brief Destroy resources owned by an auto-created Vulkan context.
 *
 * Destroys the logical device and clears the context.  The VkInstance is
 * process-wide and is deliberately left alive; it is released by
 * vk_auto_instance_shutdown() or at process exit.
 *
 * @param context Context to destroy/reset.
 */
void vk_auto_context_destroy(vk_auto_context* context);

/**
 * @brief Destroy the process-wide Vulkan instance, if one was created.
 *
 * nlolib creates one VkInstance per process and reuses it for every backend,
 * because some drivers hold per-instance resources that vkDestroyInstance does
 * not release (the NVIDIA Linux ICD opens a D-Bus system-bus connection inside
 * vkCreateInstance and keeps it, capping a process at eleven instances).
 *
 * Also registered with atexit(), so callers need not invoke it.  Call it
 * explicitly only to release the instance early, after every backend created
 * from it has been destroyed; a later backend creation makes a fresh instance.
 * Safe to call more than once, and from any thread.
 */
void vk_auto_instance_shutdown(void);

/**
 * @brief Number of VkInstance objects this process has created so far.
 *
 * Diagnostic hook: stays at 1 however many backends are created, and is the
 * regression guard against going back to one instance per backend.
 *
 * @return unsigned int Count of successful vkCreateInstance calls.
 */
unsigned int vk_auto_instance_create_count(void);
