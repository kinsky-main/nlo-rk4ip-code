/**
 * @file vk_operator_jit.c
 * @brief Runtime Vulkan JIT compilation and execution for operator programs.
 */

#include "backend/vector_backend_internal.h"
#include "io/log_sink.h"
#include "numerics/vk_backend_internal.h"
#include "physics/operator_program_jit.h"
#include "utility/perf_profile.h"
#include "vk_jit_glsl_sources.h"
#include <glslang_c_interface.h>

/* Runtime glslang gets no -D flags, so the precision switch goes in the source. */
#if defined(NLO_VK_FP32) && NLO_VK_FP32
#define VK_JIT_PRECISION_DEFINE "#define NLO_VK_FP32 1\n"
#else
#define VK_JIT_PRECISION_DEFINE ""
#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static SRWLOCK g_jit_cache_lock = SRWLOCK_INIT;
static void jit_cache_lock(void) { AcquireSRWLockExclusive(&g_jit_cache_lock); }
static void jit_cache_unlock(void) { ReleaseSRWLockExclusive(&g_jit_cache_lock); }
#else
#include <pthread.h>
static pthread_mutex_t g_jit_cache_lock = PTHREAD_MUTEX_INITIALIZER;
static void jit_cache_lock(void) { (void)pthread_mutex_lock(&g_jit_cache_lock); }
static void jit_cache_unlock(void) { (void)pthread_mutex_unlock(&g_jit_cache_lock); }
#endif

/* Upper bound on distinct programs kept in the process-wide SPIR-V cache. */
#define JIT_SPIRV_CACHE_MAX_ENTRIES 64u

struct vk_operator_jit_entry {
    uint64_t lowered_hash;
    operator_program_context context;
    uint32_t active_symbol_mask;
    size_t value_count;
    size_t root_value;
    uint32_t local_size_x;
    VkPipeline pipeline;
    operator_value values[OPERATOR_PROGRAM_MAX_VALUES];
    struct vk_operator_jit_entry* next;
};

/*
 * Device-independent SPIR-V for one lowered program, kept for the life of the
 * process so later backends (one per propagate call) skip glslang entirely.
 */
typedef struct jit_spirv_cache_entry {
    uint64_t lowered_hash;
    operator_program_context context;
    uint32_t active_symbol_mask;
    size_t value_count;
    size_t root_value;
    uint32_t local_size_x;
    operator_value values[OPERATOR_PROGRAM_MAX_VALUES];
    uint32_t* words;
    size_t word_count;
    struct jit_spirv_cache_entry* next;
} jit_spirv_cache_entry;

static jit_spirv_cache_entry* g_jit_spirv_cache = NULL;
static size_t g_jit_spirv_cache_count = 0u;

typedef struct {
    char* data;
    size_t length;
    size_t capacity;
} jit_string_builder;

static int jit_str_builder_reserve(jit_string_builder* builder, size_t extra)
{
    if (builder == NULL) {
        return -1;
    }

    size_t required = builder->length + extra + 1u;
    if (required <= builder->capacity) {
        return 0;
    }

    size_t next_capacity = (builder->capacity > 0u) ? builder->capacity : 256u;
    while (next_capacity < required) {
        next_capacity *= 2u;
    }

    char* next = (char*)realloc(builder->data, next_capacity);
    if (next == NULL) {
        return -1;
    }

    builder->data = next;
    builder->capacity = next_capacity;
    return 0;
}

static int jit_str_builder_append(jit_string_builder* builder, const char* text)
{
    if (builder == NULL || text == NULL) {
        return -1;
    }

    const size_t text_len = strlen(text);
    if (jit_str_builder_reserve(builder, text_len) != 0) {
        return -1;
    }

    memcpy(builder->data + builder->length, text, text_len);
    builder->length += text_len;
    builder->data[builder->length] = '\0';
    return 0;
}

static int jit_str_builder_appendf(jit_string_builder* builder, const char* fmt, ...)
{
    if (builder == NULL || fmt == NULL) {
        return -1;
    }

    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    const int required = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (required < 0) {
        va_end(args);
        return -1;
    }

    if (jit_str_builder_reserve(builder, (size_t)required) != 0) {
        va_end(args);
        return -1;
    }

    const int written = vsnprintf(builder->data + builder->length,
                                  builder->capacity - builder->length,
                                  fmt,
                                  args);
    va_end(args);
    if (written != required) {
        return -1;
    }

    builder->length += (size_t)written;
    return 0;
}

static void jit_str_builder_destroy(jit_string_builder* builder)
{
    if (builder == NULL) {
        return;
    }
    free(builder->data);
    builder->data = NULL;
    builder->length = 0u;
    builder->capacity = 0u;
}

static int jit_operator_programs_equal(
    const vk_operator_jit_entry* entry,
    const operator_program* program
)
{
    if (entry == NULL || program == NULL) {
        return 0;
    }
    if (entry->lowered_hash != program->lowered_hash ||
        entry->context != program->context ||
        entry->active_symbol_mask != program->active_symbol_mask ||
        entry->value_count != program->value_count ||
        entry->root_value != program->root_value ||
        entry->local_size_x != VK_LOCAL_SIZE_X) {
        return 0;
    }

    for (size_t i = 0u; i < program->value_count; ++i) {
        const operator_value* a = &entry->values[i];
        const operator_value* b = &program->values[i];
        if (a->opcode != b->opcode ||
            a->left != b->left ||
            a->right != b->right ||
            a->symbol_mask != b->symbol_mask ||
            RE(a->literal) != RE(b->literal) ||
            IM(a->literal) != IM(b->literal)) {
            return 0;
        }
    }

    return 1;
}

/* Include callback: resolve #include against the GLSL embedded at build time. */
static glsl_include_result_t* jit_shader_include_load(const char* header_name)
{
    if (header_name == NULL) {
        return NULL;
    }

    for (size_t i = 0u; i < vk_jit_glsl_source_count; ++i) {
        const vk_jit_glsl_source* source = &vk_jit_glsl_sources[i];
        if (strcmp(source->name, header_name) != 0) {
            continue;
        }
        glsl_include_result_t* result = (glsl_include_result_t*)malloc(sizeof(*result));
        if (result == NULL) {
            return NULL;
        }
        result->header_name = source->name;
        result->header_data = source->text;
        result->header_length = source->length;
        return result;
    }
    return NULL;
}

static glsl_include_result_t* jit_shader_include_local(
    void* ctx,
    const char* header_name,
    const char* includer_name,
    size_t include_depth
)
{
    (void)includer_name;
    (void)include_depth;
    (void)ctx;
    return jit_shader_include_load(header_name);
}

static glsl_include_result_t* jit_shader_include_system(
    void* ctx,
    const char* header_name,
    const char* includer_name,
    size_t include_depth
)
{
    (void)includer_name;
    (void)include_depth;
    (void)ctx;
    return jit_shader_include_load(header_name);
}

static int jit_shader_include_free(void* ctx, glsl_include_result_t* result)
{
    (void)ctx;
    if (result == NULL) {
        return 0;
    }

    /* Name and data point into the embedded table; only the result is ours. */
    free(result);
    return 0;
}

static int jit_symbol_uses_binding(uint32_t symbol_mask, vk_descriptor_binding binding)
{
    switch (binding) {
        case VK_BINDING_FIELD:
            return ((symbol_mask & (OPERATOR_SYMBOL_MASK_A | OPERATOR_SYMBOL_MASK_I)) != 0u);
        case VK_BINDING_W:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_W) != 0u);
        case VK_BINDING_WT:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_WT) != 0u);
        case VK_BINDING_KX:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_KX) != 0u);
        case VK_BINDING_KY:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_KY) != 0u);
        case VK_BINDING_T:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_T) != 0u);
        case VK_BINDING_X:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_X) != 0u);
        case VK_BINDING_Y:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_Y) != 0u);
        case VK_BINDING_D:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_D) != 0u);
        case VK_BINDING_V:
            return ((symbol_mask & OPERATOR_SYMBOL_MASK_V) != 0u);
        default:
            return 0;
    }
}

static const char* jit_symbol_value_expr(const operator_value* value)
{
    if (value == NULL) {
        return "jit_make_complex(0.0, 0.0)";
    }

    switch (value->opcode) {
        case OPERATOR_OP_PUSH_SYMBOL_W:
            return "jit_symbol_fetch_w(idx)";
        case OPERATOR_OP_PUSH_SYMBOL_A:
            return "field_vals[idx]";
        case OPERATOR_OP_PUSH_SYMBOL_I:
            return "jit_complex_magnitude_squared(field_vals[idx])";
        case OPERATOR_OP_PUSH_SYMBOL_D:
            return "dispersion_vals[idx]";
        case OPERATOR_OP_PUSH_SYMBOL_V:
            return "potential_vals[idx]";
        case OPERATOR_OP_PUSH_SYMBOL_H:
            return "jit_make_complex(pc.scalar0, 0.0)";
        case OPERATOR_OP_PUSH_SYMBOL_WT:
            return "jit_symbol_fetch_wt(idx)";
        case OPERATOR_OP_PUSH_SYMBOL_KX:
            return "jit_symbol_fetch_kx(idx)";
        case OPERATOR_OP_PUSH_SYMBOL_KY:
            return "jit_symbol_fetch_ky(idx)";
        case OPERATOR_OP_PUSH_SYMBOL_T:
            return "jit_symbol_fetch_t(idx)";
        case OPERATOR_OP_PUSH_SYMBOL_X:
            return "jit_symbol_fetch_x(idx)";
        case OPERATOR_OP_PUSH_SYMBOL_Y:
            return "jit_symbol_fetch_y(idx)";
        default:
            return "jit_make_complex(0.0, 0.0)";
    }
}

static int jit_emit_value_line(
    jit_string_builder* source,
    const operator_program* program,
    size_t index
)
{
    const operator_value* value = &program->values[index];
    const char* unary_fmt = NULL;
    const char* binary_fmt = NULL;

    if (value->opcode == OPERATOR_OP_PUSH_LITERAL) {
        return jit_str_builder_appendf(source,
                                       "    vk_complex64 v%zu = jit_make_complex(%.17g, %.17g);\n",
                                       index,
                                       RE(value->literal),
                                       IM(value->literal));
    }

    if (value->symbol_mask != OPERATOR_SYMBOL_MASK_NONE &&
        value->left == OPERATOR_VALUE_INVALID &&
        value->right == OPERATOR_VALUE_INVALID) {
        return jit_str_builder_appendf(source,
                                       "    vk_complex64 v%zu = %s;\n",
                                       index,
                                       jit_symbol_value_expr(value));
    }

    switch (value->opcode) {
        case OPERATOR_OP_NEGATE:
            unary_fmt = "jit_complex_negate(v%u)";
            break;
        case OPERATOR_OP_EXP:
            unary_fmt = "jit_complex_exp(v%u)";
            break;
        case OPERATOR_OP_LOG:
            unary_fmt = "jit_complex_log(v%u)";
            break;
        case OPERATOR_OP_SQRT:
            unary_fmt = "jit_complex_real_pow(v%u, 0.5)";
            break;
        case OPERATOR_OP_SIN:
            unary_fmt = "jit_complex_sin(v%u)";
            break;
        case OPERATOR_OP_COS:
            unary_fmt = "jit_complex_cos(v%u)";
            break;
        case OPERATOR_OP_POW_REAL_LITERAL:
            return jit_str_builder_appendf(source,
                                           "    vk_complex64 v%zu = jit_complex_real_pow(v%u, %.17g);\n",
                                           index,
                                           (unsigned int)value->left,
                                           RE(value->literal));
        case OPERATOR_OP_ADD:
            binary_fmt = "jit_complex_add(v%u, v%u)";
            break;
        case OPERATOR_OP_MUL:
            binary_fmt = "jit_complex_mul(v%u, v%u)";
            break;
        case OPERATOR_OP_DIV:
            binary_fmt = "jit_complex_div(v%u, v%u)";
            break;
        case OPERATOR_OP_POW:
            binary_fmt = "jit_complex_pow(v%u, v%u)";
            break;
        default:
            return -1;
    }

    if (unary_fmt != NULL) {
        return jit_str_builder_appendf(source, "    vk_complex64 v%zu = ", index) ||
               jit_str_builder_appendf(source, unary_fmt, (unsigned int)value->left) ||
               jit_str_builder_append(source, ";\n");
    }

    if (binary_fmt != NULL) {
        return jit_str_builder_appendf(source, "    vk_complex64 v%zu = ", index) ||
               jit_str_builder_appendf(source,
                                       binary_fmt,
                                       (unsigned int)value->left,
                                       (unsigned int)value->right) ||
               jit_str_builder_append(source, ";\n");
    }

    return -1;
}

static int jit_build_operator_source(
    const operator_program* program,
    jit_string_builder* source
)
{
    if (program == NULL || source == NULL || !program->active || program->value_count == 0u) {
        return -1;
    }

    if (jit_str_builder_append(source,
                               "#version 460\n"
                               "#extension GL_GOOGLE_include_directive : require\n"
                               VK_JIT_PRECISION_DEFINE
                               "#include \"nlo_precision.glslinc\"\n\n"
                               "#include \"complex_device.glslinc\"\n"
                               "#include \"double_math.glslinc\"\n"
                               "#include \"complex_math.glslinc\"\n\n") != 0 ||
        jit_str_builder_appendf(source,
                                "layout(local_size_x = %u) in;\n\n",
                                VK_LOCAL_SIZE_X) != 0 ||
        jit_str_builder_append(source,
                               "layout(push_constant) uniform PushConstants {\n"
                               "    uint count;\n"
                               "    uint _pad;\n"
                               "    nlo_real scalar0;\n"
                               "    nlo_real scalar1;\n"
                               "} pc;\n\n"
                               "layout(set = 0, binding = 0, std430) buffer DstBuffer { vk_complex64 dst[]; };\n"
                               "layout(set = 0, binding = 1, std430) readonly buffer FieldBuffer { vk_complex64 field_vals[]; };\n"
                               "layout(set = 0, binding = 2, std430) readonly buffer WBuffer { vk_complex64 w_vals[]; };\n"
                               "layout(set = 0, binding = 3, std430) readonly buffer WTBuffer { vk_complex64 wt_vals[]; };\n"
                               "layout(set = 0, binding = 4, std430) readonly buffer KXBuffer { vk_complex64 kx_vals[]; };\n"
                               "layout(set = 0, binding = 5, std430) readonly buffer KYBuffer { vk_complex64 ky_vals[]; };\n"
                               "layout(set = 0, binding = 6, std430) readonly buffer TBuffer { vk_complex64 t_vals[]; };\n"
                               "layout(set = 0, binding = 7, std430) readonly buffer XBuffer { vk_complex64 x_vals[]; };\n"
                               "layout(set = 0, binding = 8, std430) readonly buffer YBuffer { vk_complex64 y_vals[]; };\n"
                               "layout(set = 0, binding = 9, std430) readonly buffer DBuffer { vk_complex64 dispersion_vals[]; };\n"
                               "layout(set = 0, binding = 10, std430) readonly buffer VBuffer { vk_complex64 potential_vals[]; };\n\n"
                               "uint jit_symbol_nt()\n"
                               "{\n"
                               "    uint len = uint(wt_vals.length());\n"
                               "    if (len != 0u && len != pc.count) {\n"
                               "        return len;\n"
                               "    }\n"
                               "    len = uint(t_vals.length());\n"
                               "    if (len != 0u && len != pc.count) {\n"
                               "        return len;\n"
                               "    }\n"
                               "    len = uint(w_vals.length());\n"
                               "    if (len != 0u && len != pc.count) {\n"
                               "        return len;\n"
                               "    }\n"
                               "    return pc.count;\n"
                               "}\n\n"
                               "uint jit_symbol_ny()\n"
                               "{\n"
                               "    uint len = uint(ky_vals.length());\n"
                               "    if (len != 0u && len != pc.count) {\n"
                               "        return len;\n"
                               "    }\n"
                               "    len = uint(y_vals.length());\n"
                               "    if (len != 0u && len != pc.count) {\n"
                               "        return len;\n"
                               "    }\n"
                               "    return 1u;\n"
                               "}\n\n"
                               "uint jit_symbol_t_index(uint idx)\n"
                               "{\n"
                               "    uint nt = jit_symbol_nt();\n"
                               "    return (nt > 0u) ? (idx % nt) : 0u;\n"
                               "}\n\n"
                               "uint jit_symbol_plane_index(uint idx)\n"
                               "{\n"
                               "    uint nt = jit_symbol_nt();\n"
                               "    return (nt > 0u) ? (idx / nt) : 0u;\n"
                               "}\n\n"
                               "uint jit_symbol_y_index(uint idx)\n"
                               "{\n"
                               "    uint ny = jit_symbol_ny();\n"
                               "    return (ny > 0u) ? (jit_symbol_plane_index(idx) % ny) : 0u;\n"
                               "}\n\n"
                               "uint jit_symbol_x_index(uint idx)\n"
                               "{\n"
                               "    uint ny = jit_symbol_ny();\n"
                               "    return (ny > 0u) ? (jit_symbol_plane_index(idx) / ny) : 0u;\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_w(uint idx)\n"
                               "{\n"
                               "    uint len = uint(w_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return w_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? w_vals[jit_symbol_t_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_wt(uint idx)\n"
                               "{\n"
                               "    uint len = uint(wt_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return wt_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? wt_vals[jit_symbol_t_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_kx(uint idx)\n"
                               "{\n"
                               "    uint len = uint(kx_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return kx_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? kx_vals[jit_symbol_x_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_ky(uint idx)\n"
                               "{\n"
                               "    uint len = uint(ky_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return ky_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? ky_vals[jit_symbol_y_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_t(uint idx)\n"
                               "{\n"
                               "    uint len = uint(t_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return t_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? t_vals[jit_symbol_t_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_x(uint idx)\n"
                               "{\n"
                               "    uint len = uint(x_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return x_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? x_vals[jit_symbol_x_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "vk_complex64 jit_symbol_fetch_y(uint idx)\n"
                               "{\n"
                               "    uint len = uint(y_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return y_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? y_vals[jit_symbol_y_index(idx) % len] : jit_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "void main()\n"
                               "{\n"
                               "    uint idx = gl_GlobalInvocationID.x;\n"
                               "    if (idx >= pc.count) {\n"
                               "        return;\n"
                               "    }\n") != 0) {
        return -1;
    }

    for (size_t i = 0u; i < program->value_count; ++i) {
        if (jit_emit_value_line(source, program, i) != 0) {
            return -1;
        }
    }

    if (jit_str_builder_appendf(source,
                                "    dst[idx] = v%zu;\n}\n",
                                program->root_value) != 0) {
        return -1;
    }
    return 0;
}

/*
 * Resource limits for compiling compute shaders. Only the compute and
 * control-flow limits matter for operator kernels. Filled by hand because the
 * glslang-default-resource-limits library is not part of every glslang
 * provider.
 */
static void jit_default_resource(glslang_resource_t* resource)
{
    memset(resource, 0, sizeof(*resource));
    resource->max_compute_work_group_count_x = 65535;
    resource->max_compute_work_group_count_y = 65535;
    resource->max_compute_work_group_count_z = 65535;
    resource->max_compute_work_group_size_x = 1024;
    resource->max_compute_work_group_size_y = 1024;
    resource->max_compute_work_group_size_z = 64;
    resource->max_compute_uniform_components = 1024;
    resource->max_compute_texture_image_units = 16;
    resource->max_compute_image_uniforms = 8;
    resource->max_compute_atomic_counters = 8;
    resource->max_compute_atomic_counter_buffers = 1;
    resource->max_combined_shader_output_resources = 8;
    resource->max_combined_image_uniforms = 8;
    resource->max_image_units = 8;
    resource->limits.non_inductive_for_loops = 1;
    resource->limits.while_loops = 1;
    resource->limits.do_while_loops = 1;
    resource->limits.general_uniform_indexing = 1;
    resource->limits.general_attribute_matrix_vector_indexing = 1;
    resource->limits.general_varying_indexing = 1;
    resource->limits.general_sampler_indexing = 1;
    resource->limits.general_variable_indexing = 1;
    resource->limits.general_constant_matrix_vector_indexing = 1;
}

static vec_status jit_compile_glsl_to_spirv(
    const char* source,
    uint32_t** out_words,
    size_t* out_word_count
)
{
    if (source == NULL || out_words == NULL || out_word_count == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    /* glslang reference-counts process init under its own global lock. */
    if (glslang_initialize_process() == 0) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    glslang_resource_t resource;
    jit_default_resource(&resource);
    glslang_input_t input = {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = GLSLANG_STAGE_COMPUTE,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_2,
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_5,
        .code = source,
        .default_version = 460,
        .default_profile = GLSLANG_NO_PROFILE,
        .force_default_version_and_profile = 0,
        .forward_compatible = 0,
        .messages = GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT,
        .resource = &resource,
        .callbacks = {
            .include_system = jit_shader_include_system,
            .include_local = jit_shader_include_local,
            .free_include_result = jit_shader_include_free
        },
        .callbacks_ctx = NULL
    };

    vec_status status = VEC_STATUS_BACKEND_UNAVAILABLE;
    glslang_shader_t* shader = glslang_shader_create(&input);
    glslang_program_t* program = NULL;

    if (shader == NULL) {
        goto cleanup;
    }
    if (glslang_shader_preprocess(shader, &input) == 0) {
        log_emit(LOG_LEVEL_WARN,
                     "[nlolib] Vulkan operator JIT preprocess failed: %s",
                     glslang_shader_get_info_log(shader));
        goto cleanup;
    }
    if (glslang_shader_parse(shader, &input) == 0) {
        log_emit(LOG_LEVEL_WARN,
                     "[nlolib] Vulkan operator JIT parse failed: %s",
                     glslang_shader_get_info_log(shader));
        goto cleanup;
    }

    program = glslang_program_create();
    if (program == NULL) {
        goto cleanup;
    }
    glslang_program_add_shader(program, shader);
    if (glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT) == 0) {
        log_emit(LOG_LEVEL_WARN,
                     "[nlolib] Vulkan operator JIT link failed: %s",
                     glslang_program_get_info_log(program));
        goto cleanup;
    }

    glslang_spv_options_t spv_options = {
        .generate_debug_info = false,
        .strip_debug_info = true,
        /* main's fetched glslang is built without the SPIR-V optimizer. */
        .disable_optimizer = true,
        .optimize_size = false,
        .disassemble = false,
        .validate = false,
        .emit_nonsemantic_shader_debug_info = false,
        .emit_nonsemantic_shader_debug_source = false
    };
    glslang_program_SPIRV_generate_with_options(program,
                                                GLSLANG_STAGE_COMPUTE,
                                                &spv_options);
    const size_t word_count = glslang_program_SPIRV_get_size(program);
    if (word_count == 0u) {
        const char* messages = glslang_program_SPIRV_get_messages(program);
        if (messages != NULL && messages[0] != '\0') {
            log_emit(LOG_LEVEL_WARN,
                         "[nlolib] Vulkan operator JIT SPIR-V generation failed: %s",
                         messages);
        }
        goto cleanup;
    }

    uint32_t* words = (uint32_t*)malloc(word_count * sizeof(uint32_t));
    if (words == NULL) {
        status = VEC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    glslang_program_SPIRV_get(program, words);
    *out_words = words;
    *out_word_count = word_count;
    status = VEC_STATUS_OK;

cleanup:
    if (program != NULL) {
        glslang_program_delete(program);
    }
    if (shader != NULL) {
        glslang_shader_delete(shader);
    }
    glslang_finalize_process();
    return status;
}

static vec_status jit_create_compute_pipeline_from_words(
    vector_backend* backend,
    const uint32_t* words,
    size_t word_count,
    VkPipeline* out_pipeline
)
{
    if (backend == NULL || words == NULL || word_count == 0u || out_pipeline == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    VkShaderModuleCreateInfo module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = word_count * sizeof(uint32_t),
        .pCode = words
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(backend->vk.device, &module_info, NULL, &module) != VK_SUCCESS) {
        return VEC_STATUS_BACKEND_UNAVAILABLE;
    }

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

    vec_status status = VEC_STATUS_OK;
    if (vkCreateComputePipelines(backend->vk.device,
                                 backend->vk.pipeline_cache,
                                 1u,
                                 &pipeline_info,
                                 NULL,
                                 out_pipeline) != VK_SUCCESS) {
        status = VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    vkDestroyShaderModule(backend->vk.device, module, NULL);
    return status;
}

static vk_operator_jit_entry* jit_find_operator_jit_entry(
    vector_backend* backend,
    const operator_program* program
)
{
    if (backend == NULL || program == NULL) {
        return NULL;
    }

    vk_operator_jit_entry* entry = backend->vk.operator_jit_entries;
    while (entry != NULL) {
        if (jit_operator_programs_equal(entry, program)) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

static const vec_buffer* jit_select_symbol_buffer(
    const operator_eval_context* eval_ctx,
    vk_descriptor_binding binding
)
{
    if (eval_ctx == NULL) {
        return NULL;
    }

    switch (binding) {
        case VK_BINDING_FIELD:
            return eval_ctx->field;
        case VK_BINDING_W:
            return (eval_ctx->frequency_grid != NULL) ? eval_ctx->frequency_grid : eval_ctx->wt_grid;
        case VK_BINDING_WT:
            return (eval_ctx->wt_grid != NULL) ? eval_ctx->wt_grid : eval_ctx->frequency_grid;
        case VK_BINDING_KX:
            return eval_ctx->kx_grid;
        case VK_BINDING_KY:
            return eval_ctx->ky_grid;
        case VK_BINDING_T:
            return eval_ctx->t_grid;
        case VK_BINDING_X:
            return eval_ctx->x_grid;
        case VK_BINDING_Y:
            return eval_ctx->y_grid;
        case VK_BINDING_D:
            return eval_ctx->dispersion_factor;
        case VK_BINDING_V:
            return eval_ctx->potential;
        default:
            return NULL;
    }
}

static int jit_spirv_cache_matches(
    const jit_spirv_cache_entry* entry,
    const operator_program* program
)
{
    if (entry->lowered_hash != program->lowered_hash ||
        entry->context != program->context ||
        entry->active_symbol_mask != program->active_symbol_mask ||
        entry->value_count != program->value_count ||
        entry->root_value != program->root_value ||
        entry->local_size_x != VK_LOCAL_SIZE_X) {
        return 0;
    }
    for (size_t i = 0u; i < program->value_count; ++i) {
        const operator_value* a = &entry->values[i];
        const operator_value* b = &program->values[i];
        if (a->opcode != b->opcode || a->left != b->left || a->right != b->right ||
            a->symbol_mask != b->symbol_mask ||
            RE(a->literal) != RE(b->literal) || IM(a->literal) != IM(b->literal)) {
            return 0;
        }
    }
    return 1;
}

/* Copy cached SPIR-V for @p program into a new allocation; 0 on a miss. */
static int jit_spirv_cache_lookup(
    const operator_program* program,
    uint32_t** out_words,
    size_t* out_word_count
)
{
    int found = 0;
    jit_cache_lock();
    for (const jit_spirv_cache_entry* e = g_jit_spirv_cache; e != NULL; e = e->next) {
        if (!jit_spirv_cache_matches(e, program)) {
            continue;
        }
        uint32_t* words = (uint32_t*)malloc(e->word_count * sizeof(uint32_t));
        if (words != NULL) {
            memcpy(words, e->words, e->word_count * sizeof(uint32_t));
            *out_words = words;
            *out_word_count = e->word_count;
            found = 1;
        }
        break;
    }
    jit_cache_unlock();
    return found;
}

static void jit_spirv_cache_insert(
    const operator_program* program,
    const uint32_t* words,
    size_t word_count
)
{
    jit_spirv_cache_entry* entry = (jit_spirv_cache_entry*)calloc(1u, sizeof(*entry));
    uint32_t* copy = (uint32_t*)malloc(word_count * sizeof(uint32_t));
    if (entry == NULL || copy == NULL) {
        free(entry);
        free(copy);
        return;
    }
    memcpy(copy, words, word_count * sizeof(uint32_t));
    entry->lowered_hash = program->lowered_hash;
    entry->context = program->context;
    entry->active_symbol_mask = program->active_symbol_mask;
    entry->value_count = program->value_count;
    entry->root_value = program->root_value;
    entry->local_size_x = VK_LOCAL_SIZE_X;
    memcpy(entry->values, program->values, program->value_count * sizeof(program->values[0]));
    entry->words = copy;
    entry->word_count = word_count;

    int inserted = 0;
    jit_cache_lock();
    int present = 0;
    for (const jit_spirv_cache_entry* e = g_jit_spirv_cache; e != NULL; e = e->next) {
        if (jit_spirv_cache_matches(e, program)) {
            present = 1;
            break;
        }
    }
    if (!present && g_jit_spirv_cache_count < JIT_SPIRV_CACHE_MAX_ENTRIES) {
        entry->next = g_jit_spirv_cache;
        g_jit_spirv_cache = entry;
        g_jit_spirv_cache_count += 1u;
        inserted = 1;
    }
    jit_cache_unlock();
    if (!inserted) {
        free(copy);
        free(entry);
    }
}

size_t operator_program_jit_cache_entry_count(void)
{
    jit_cache_lock();
    const size_t count = g_jit_spirv_cache_count;
    jit_cache_unlock();
    return count;
}

/* NLO_OPERATOR_JIT=0 keeps every program on the interpreter. */
static int jit_disabled_by_environment(void)
{
    const char* value = getenv("NLO_OPERATOR_JIT");
    return (value != NULL && value[0] == '0' && value[1] == '\0') ? 1 : 0;
}

vec_status operator_program_prepare_jit(
    vector_backend* backend,
    operator_program* program
)
{
    if (backend == NULL || program == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!program->active || backend->type != VECTOR_BACKEND_VULKAN) {
        return VEC_STATUS_UNSUPPORTED;
    }
    if (!program->jit_eligible || jit_disabled_by_environment()) {
        perf_profile_add_operator_jit_fallback();
        return VEC_STATUS_UNSUPPORTED;
    }
    if (program->vk_jit_entry != NULL) {
        /* Already prepared, and already counted. */
        program->vk_jit_active = 1;
        return VEC_STATUS_OK;
    }

    vk_operator_jit_entry* cached = jit_find_operator_jit_entry(backend, program);
    if (cached != NULL) {
        program->vk_jit_entry = cached;
        program->vk_jit_active = 1;
        perf_profile_add_operator_jit_program();
        return VEC_STATUS_OK;
    }

    jit_string_builder source = {0};
    uint32_t* spirv_words = NULL;
    size_t spirv_word_count = 0u;
    vec_status status = VEC_STATUS_BACKEND_UNAVAILABLE;
    if (jit_build_operator_source(program, &source) != 0) {
        status = VEC_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    if (jit_spirv_cache_lookup(program, &spirv_words, &spirv_word_count)) {
        status = VEC_STATUS_OK;
    } else {
        status = jit_compile_glsl_to_spirv(source.data, &spirv_words, &spirv_word_count);
        if (status != VEC_STATUS_OK) {
            goto cleanup;
        }
        jit_spirv_cache_insert(program, spirv_words, spirv_word_count);
    }

    vk_operator_jit_entry* entry = (vk_operator_jit_entry*)calloc(1u, sizeof(*entry));
    if (entry == NULL) {
        status = VEC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    entry->lowered_hash = program->lowered_hash;
    entry->context = program->context;
    entry->active_symbol_mask = program->active_symbol_mask;
    entry->value_count = program->value_count;
    entry->root_value = program->root_value;
    entry->local_size_x = VK_LOCAL_SIZE_X;
    memcpy(entry->values, program->values, program->value_count * sizeof(program->values[0]));

    status = jit_create_compute_pipeline_from_words(backend,
                                                       spirv_words,
                                                       spirv_word_count,
                                                       &entry->pipeline);
    if (status != VEC_STATUS_OK) {
        free(entry);
        goto cleanup;
    }

    entry->next = backend->vk.operator_jit_entries;
    backend->vk.operator_jit_entries = entry;
    program->vk_jit_entry = entry;
    program->vk_jit_active = 1;

cleanup:
    if (status != VEC_STATUS_OK) {
        program->vk_jit_active = 0;
        perf_profile_add_operator_jit_fallback();
    } else {
        perf_profile_add_operator_jit_program();
    }
    free(spirv_words);
    jit_str_builder_destroy(&source);
    return status;
}

static void jit_update_operator_descriptor_set(
    vector_backend* backend,
    VkDescriptorSet descriptor_set,
    const VkBuffer* buffers,
    const VkDeviceSize* ranges
)
{
    VkDescriptorBufferInfo infos[VK_DESCRIPTOR_BINDING_COUNT] = {0};
    VkWriteDescriptorSet writes[VK_DESCRIPTOR_BINDING_COUNT] = {0};

    for (uint32_t i = 0u; i < VK_DESCRIPTOR_BINDING_COUNT; ++i) {
        infos[i].buffer = buffers[i];
        infos[i].offset = 0u;
        infos[i].range = ranges[i];

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1u;
        writes[i].pBufferInfo = &infos[i];
    }

    vkUpdateDescriptorSets(backend->vk.device,
                           VK_DESCRIPTOR_BINDING_COUNT,
                           writes,
                           0u,
                           NULL);
}

static size_t jit_collect_unique_buffers(
    const VkBuffer* buffers,
    const VkDeviceSize* ranges,
    size_t buffer_count,
    VkBuffer* out_unique,
    VkDeviceSize* out_unique_ranges,
    size_t unique_capacity
)
{
    size_t unique_count = 0u;
    for (size_t i = 0u; i < buffer_count; ++i) {
        if (buffers[i] == VK_NULL_HANDLE) {
            continue;
        }

        int seen = 0;
        for (size_t j = 0u; j < unique_count; ++j) {
            if (out_unique[j] == buffers[i]) {
                if (ranges != NULL && out_unique_ranges != NULL && out_unique_ranges[j] < ranges[i]) {
                    out_unique_ranges[j] = ranges[i];
                }
                seen = 1;
                break;
            }
        }
        if (seen || unique_count >= unique_capacity) {
            continue;
        }
        out_unique[unique_count++] = buffers[i];
        if (out_unique_ranges != NULL) {
            out_unique_ranges[unique_count - 1u] = (ranges != NULL) ? ranges[i] : 0u;
        }
    }
    return unique_count;
}

vec_status operator_program_execute_jit(
    vector_backend* backend,
    const operator_program* program,
    const operator_eval_context* eval_ctx,
    vec_buffer* out_vector
)
{
    if (backend == NULL || program == NULL || eval_ctx == NULL || out_vector == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->type != VECTOR_BACKEND_VULKAN ||
        !program->vk_jit_active ||
        program->vk_jit_entry == NULL ||
        out_vector->owner != backend ||
        out_vector->kind != VEC_KIND_COMPLEX64) {
        return VEC_STATUS_UNSUPPORTED;
    }
    if (out_vector->length == 0u || out_vector->length > (size_t)UINT32_MAX) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    VkBuffer bound_buffers[VK_DESCRIPTOR_BINDING_COUNT] = {0};
    VkDeviceSize bound_ranges[VK_DESCRIPTOR_BINDING_COUNT] = {0};
    bound_buffers[VK_BINDING_DST] = out_vector->vk_buffer;
    bound_ranges[VK_BINDING_DST] = (VkDeviceSize)vk_device_bytes(out_vector->bytes);

    for (uint32_t binding = VK_BINDING_FIELD; binding < VK_DESCRIPTOR_BINDING_COUNT; ++binding) {
        const vec_buffer* source = jit_select_symbol_buffer(eval_ctx, (vk_descriptor_binding)binding);
        if (source != NULL) {
            if (source->owner != backend ||
                source->kind != VEC_KIND_COMPLEX64 ||
                source->bytes == 0u) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            if ((binding == VK_BINDING_FIELD ||
                 binding == VK_BINDING_D ||
                 binding == VK_BINDING_V) &&
                source->length != out_vector->length) {
                return VEC_STATUS_INVALID_ARGUMENT;
            }
            bound_buffers[binding] = source->vk_buffer;
            bound_ranges[binding] = (VkDeviceSize)vk_device_bytes(source->bytes);
            continue;
        }

        if (jit_symbol_uses_binding(program->active_symbol_mask,
                                             (vk_descriptor_binding)binding)) {
            return VEC_STATUS_INVALID_ARGUMENT;
        }
        bound_buffers[binding] = out_vector->vk_buffer;
        bound_ranges[binding] = (VkDeviceSize)vk_device_bytes(out_vector->bytes);
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    vec_status status = vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    status = vk_begin_commands(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    VkBuffer unique_buffers[VK_DESCRIPTOR_BINDING_COUNT] = {0};
    VkDeviceSize unique_ranges[VK_DESCRIPTOR_BINDING_COUNT] = {0};
    const size_t unique_count = jit_collect_unique_buffers(bound_buffers,
                                                              bound_ranges,
                                                              VK_DESCRIPTOR_BINDING_COUNT,
                                                              unique_buffers,
                                                              unique_ranges,
                                                              VK_DESCRIPTOR_BINDING_COUNT);
    VkCommandBuffer cmd = backend->vk.command_buffer;
    for (size_t i = 0u; i < unique_count; ++i) {
        vk_cmd_transfer_to_compute(cmd,
                                       unique_buffers[i],
                                       0u,
                                       unique_ranges[i]);
        vk_cmd_compute_to_compute(cmd,
                                      unique_buffers[i],
                                      0u,
                                      unique_ranges[i]);
    }

    jit_update_operator_descriptor_set(backend,
                                          descriptor_set,
                                          bound_buffers,
                                          bound_ranges);

    vkCmdBindPipeline(cmd,
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      program->vk_jit_entry->pipeline);
    vkCmdBindDescriptorSets(cmd,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            backend->vk.pipeline_layout,
                            0u,
                            1u,
                            &descriptor_set,
                            0u,
                            NULL);

    vk_push_constants push = {
        .count = (uint32_t)out_vector->length,
        .pad = 0u,
        .scalar0 = (vk_real)eval_ctx->half_step_size,
        .scalar1 = (vk_real)0.0
    };
    vkCmdPushConstants(cmd,
                       backend->vk.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0u,
                       (uint32_t)sizeof(push),
                       &push);

    const uint32_t groups =
        (uint32_t)((out_vector->length + (size_t)VK_LOCAL_SIZE_X - 1u) /
                   (size_t)VK_LOCAL_SIZE_X);
    vkCmdDispatch(cmd, groups, 1u, 1u);
    vk_cmd_compute_to_compute(cmd,
                                  out_vector->vk_buffer,
                                  0u,
                                  (VkDeviceSize)vk_device_bytes(out_vector->bytes));
    vk_cmd_compute_to_transfer(cmd,
                                   out_vector->vk_buffer,
                                   0u,
                                   (VkDeviceSize)vk_device_bytes(out_vector->bytes));

    status = vk_submit_commands(backend);
    if (status != VEC_STATUS_OK) {
        return status;
    }

    perf_profile_add_gpu_dispatch(1u,
                                      (uint64_t)unique_count + 1u,
                                      ((uint64_t)unique_count + 1u) * (uint64_t)vk_device_bytes(out_vector->bytes));
    return VEC_STATUS_OK;
}

void vk_operator_jit_destroy_all(vector_backend* backend)
{
    if (backend == NULL || backend->type != VECTOR_BACKEND_VULKAN) {
        return;
    }

    vk_operator_jit_entry* entry = backend->vk.operator_jit_entries;
    while (entry != NULL) {
        vk_operator_jit_entry* next = entry->next;
        if (entry->pipeline != VK_NULL_HANDLE && backend->vk.device != VK_NULL_HANDLE) {
            vkDestroyPipeline(backend->vk.device, entry->pipeline, NULL);
        }
        free(entry);
        entry = next;
    }
    backend->vk.operator_jit_entries = NULL;
}
