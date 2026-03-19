/**
 * @file vk_operator_jit.c
 * @brief Runtime Vulkan JIT compilation and execution for operator programs.
 */

#include "backend/vector_backend_internal.h"
#include "io/log_sink.h"
#include "nlo_vk_shader_paths.h"
#include "physics/operator_program_jit.h"
#include "utility/perf_profile.h"
#include <glslang_c_interface.h>
#include <resource_limits_c.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct nlo_vk_operator_jit_entry {
    uint64_t lowered_hash;
    nlo_operator_program_context context;
    uint32_t active_symbol_mask;
    size_t value_count;
    size_t root_value;
    uint32_t local_size_x;
    VkPipeline pipeline;
    nlo_operator_value values[NLO_OPERATOR_PROGRAM_MAX_VALUES];
    struct nlo_vk_operator_jit_entry* next;
};

typedef struct {
    char* data;
    size_t length;
    size_t capacity;
} nlo_string_builder;

typedef struct {
    const char* source_dir;
} nlo_vk_shader_include_context;

static int nlo_vk_glslang_refcount = 0;

static int nlo_str_builder_reserve(nlo_string_builder* builder, size_t extra)
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

static int nlo_str_builder_append(nlo_string_builder* builder, const char* text)
{
    if (builder == NULL || text == NULL) {
        return -1;
    }

    const size_t text_len = strlen(text);
    if (nlo_str_builder_reserve(builder, text_len) != 0) {
        return -1;
    }

    memcpy(builder->data + builder->length, text, text_len);
    builder->length += text_len;
    builder->data[builder->length] = '\0';
    return 0;
}

static int nlo_str_builder_appendf(nlo_string_builder* builder, const char* fmt, ...)
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

    if (nlo_str_builder_reserve(builder, (size_t)required) != 0) {
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

static void nlo_str_builder_destroy(nlo_string_builder* builder)
{
    if (builder == NULL) {
        return;
    }
    free(builder->data);
    builder->data = NULL;
    builder->length = 0u;
    builder->capacity = 0u;
}

static int nlo_vk_operator_programs_equal(
    const nlo_vk_operator_jit_entry* entry,
    const nlo_operator_program* program
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
        entry->local_size_x != NLO_VK_LOCAL_SIZE_X) {
        return 0;
    }

    for (size_t i = 0u; i < program->value_count; ++i) {
        const nlo_operator_value* a = &entry->values[i];
        const nlo_operator_value* b = &program->values[i];
        if (a->opcode != b->opcode ||
            a->left != b->left ||
            a->right != b->right ||
            a->symbol_mask != b->symbol_mask ||
            NLO_RE(a->literal) != NLO_RE(b->literal) ||
            NLO_IM(a->literal) != NLO_IM(b->literal)) {
            return 0;
        }
    }

    return 1;
}

static int nlo_vk_glslang_acquire(void)
{
    if (nlo_vk_glslang_refcount == 0) {
        if (glslang_initialize_process() == 0) {
            return -1;
        }
    }
    nlo_vk_glslang_refcount += 1;
    return 0;
}

static void nlo_vk_glslang_release(void)
{
    if (nlo_vk_glslang_refcount <= 0) {
        return;
    }

    nlo_vk_glslang_refcount -= 1;
    if (nlo_vk_glslang_refcount == 0) {
        glslang_finalize_process();
    }
}

static glsl_include_result_t* nlo_vk_shader_include_load(
    const nlo_vk_shader_include_context* context,
    const char* header_name
)
{
    if (context == NULL || context->source_dir == NULL || header_name == NULL) {
        return NULL;
    }

    size_t path_len = strlen(context->source_dir) + strlen(header_name) + 2u;
    char* path = (char*)malloc(path_len);
    if (path == NULL) {
        return NULL;
    }
    (void)snprintf(path, path_len, "%s/%s", context->source_dir, header_name);

    FILE* fp = fopen(path, "rb");
    free(path);
    if (fp == NULL) {
        return NULL;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long size_long = ftell(fp);
    if (size_long < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }

    size_t size = (size_t)size_long;
    char* bytes = (char*)malloc(size + 1u);
    if (bytes == NULL) {
        fclose(fp);
        return NULL;
    }
    const size_t read = fread(bytes, 1u, size, fp);
    fclose(fp);
    if (read != size) {
        free(bytes);
        return NULL;
    }
    bytes[size] = '\0';

    char* header_name_copy = (char*)malloc(strlen(header_name) + 1u);
    if (header_name_copy == NULL) {
        free(bytes);
        return NULL;
    }
    memcpy(header_name_copy, header_name, strlen(header_name) + 1u);

    glsl_include_result_t* result = (glsl_include_result_t*)malloc(sizeof(*result));
    if (result == NULL) {
        free(header_name_copy);
        free(bytes);
        return NULL;
    }

    result->header_name = header_name_copy;
    result->header_data = bytes;
    result->header_length = size;
    return result;
}

static glsl_include_result_t* nlo_vk_shader_include_local(
    void* ctx,
    const char* header_name,
    const char* includer_name,
    size_t include_depth
)
{
    (void)includer_name;
    (void)include_depth;
    return nlo_vk_shader_include_load((const nlo_vk_shader_include_context*)ctx, header_name);
}

static glsl_include_result_t* nlo_vk_shader_include_system(
    void* ctx,
    const char* header_name,
    const char* includer_name,
    size_t include_depth
)
{
    (void)includer_name;
    (void)include_depth;
    return nlo_vk_shader_include_load((const nlo_vk_shader_include_context*)ctx, header_name);
}

static int nlo_vk_shader_include_free(void* ctx, glsl_include_result_t* result)
{
    (void)ctx;
    if (result == NULL) {
        return 0;
    }

    free((void*)result->header_name);
    free((void*)result->header_data);
    free(result);
    return 0;
}

static int nlo_operator_symbol_uses_binding(uint32_t symbol_mask, nlo_vk_descriptor_binding binding)
{
    switch (binding) {
        case NLO_VK_BINDING_FIELD:
            return ((symbol_mask & (NLO_OPERATOR_SYMBOL_MASK_A | NLO_OPERATOR_SYMBOL_MASK_I)) != 0u);
        case NLO_VK_BINDING_W:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_W) != 0u);
        case NLO_VK_BINDING_WT:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_WT) != 0u);
        case NLO_VK_BINDING_KX:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_KX) != 0u);
        case NLO_VK_BINDING_KY:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_KY) != 0u);
        case NLO_VK_BINDING_T:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_T) != 0u);
        case NLO_VK_BINDING_X:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_X) != 0u);
        case NLO_VK_BINDING_Y:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_Y) != 0u);
        case NLO_VK_BINDING_D:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_D) != 0u);
        case NLO_VK_BINDING_V:
            return ((symbol_mask & NLO_OPERATOR_SYMBOL_MASK_V) != 0u);
        default:
            return 0;
    }
}

static const char* nlo_operator_symbol_value_expr(const nlo_operator_value* value)
{
    if (value == NULL) {
        return "nlo_vk_make_complex(0.0, 0.0)";
    }

    switch (value->opcode) {
        case NLO_OPERATOR_OP_PUSH_SYMBOL_W:
            return "nlo_vk_symbol_fetch_w(idx)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_A:
            return "field_vals[idx]";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_I:
            return "nlo_vk_complex_magnitude_squared(field_vals[idx])";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_D:
            return "dispersion_vals[idx]";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_V:
            return "potential_vals[idx]";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_H:
            return "nlo_vk_make_complex(pc.scalar0, 0.0)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_WT:
            return "nlo_vk_symbol_fetch_wt(idx)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_KX:
            return "nlo_vk_symbol_fetch_kx(idx)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_KY:
            return "nlo_vk_symbol_fetch_ky(idx)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_T:
            return "nlo_vk_symbol_fetch_t(idx)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_X:
            return "nlo_vk_symbol_fetch_x(idx)";
        case NLO_OPERATOR_OP_PUSH_SYMBOL_Y:
            return "nlo_vk_symbol_fetch_y(idx)";
        default:
            return "nlo_vk_make_complex(0.0, 0.0)";
    }
}

static int nlo_vk_emit_value_line(
    nlo_string_builder* source,
    const nlo_operator_program* program,
    size_t index
)
{
    const nlo_operator_value* value = &program->values[index];
    const char* unary_fmt = NULL;
    const char* binary_fmt = NULL;

    if (value->opcode == NLO_OPERATOR_OP_PUSH_LITERAL) {
        return nlo_str_builder_appendf(source,
                                       "    nlo_vk_complex64 v%zu = nlo_vk_make_complex(%.17g, %.17g);\n",
                                       index,
                                       NLO_RE(value->literal),
                                       NLO_IM(value->literal));
    }

    if (value->symbol_mask != NLO_OPERATOR_SYMBOL_MASK_NONE &&
        value->left == NLO_OPERATOR_VALUE_INVALID &&
        value->right == NLO_OPERATOR_VALUE_INVALID) {
        return nlo_str_builder_appendf(source,
                                       "    nlo_vk_complex64 v%zu = %s;\n",
                                       index,
                                       nlo_operator_symbol_value_expr(value));
    }

    switch (value->opcode) {
        case NLO_OPERATOR_OP_NEGATE:
            unary_fmt = "nlo_vk_complex_negate(v%u)";
            break;
        case NLO_OPERATOR_OP_EXP:
            unary_fmt = "nlo_vk_complex_exp(v%u)";
            break;
        case NLO_OPERATOR_OP_LOG:
            unary_fmt = "nlo_vk_complex_log(v%u)";
            break;
        case NLO_OPERATOR_OP_SQRT:
            unary_fmt = "nlo_vk_complex_real_pow(v%u, 0.5)";
            break;
        case NLO_OPERATOR_OP_SIN:
            unary_fmt = "nlo_vk_complex_sin(v%u)";
            break;
        case NLO_OPERATOR_OP_COS:
            unary_fmt = "nlo_vk_complex_cos(v%u)";
            break;
        case NLO_OPERATOR_OP_POW_REAL_LITERAL:
            return nlo_str_builder_appendf(source,
                                           "    nlo_vk_complex64 v%zu = nlo_vk_complex_real_pow(v%u, %.17g);\n",
                                           index,
                                           (unsigned int)value->left,
                                           NLO_RE(value->literal));
        case NLO_OPERATOR_OP_ADD:
            binary_fmt = "nlo_vk_complex_add(v%u, v%u)";
            break;
        case NLO_OPERATOR_OP_MUL:
            binary_fmt = "nlo_vk_complex_mul(v%u, v%u)";
            break;
        case NLO_OPERATOR_OP_DIV:
            binary_fmt = "nlo_vk_complex_div(v%u, v%u)";
            break;
        case NLO_OPERATOR_OP_POW:
            binary_fmt = "nlo_vk_complex_pow(v%u, v%u)";
            break;
        default:
            return -1;
    }

    if (unary_fmt != NULL) {
        return nlo_str_builder_appendf(source, "    nlo_vk_complex64 v%zu = ", index) ||
               nlo_str_builder_appendf(source, unary_fmt, (unsigned int)value->left) ||
               nlo_str_builder_append(source, ";\n");
    }

    if (binary_fmt != NULL) {
        return nlo_str_builder_appendf(source, "    nlo_vk_complex64 v%zu = ", index) ||
               nlo_str_builder_appendf(source,
                                       binary_fmt,
                                       (unsigned int)value->left,
                                       (unsigned int)value->right) ||
               nlo_str_builder_append(source, ";\n");
    }

    return -1;
}

static int nlo_vk_build_operator_source(
    const nlo_operator_program* program,
    nlo_string_builder* source
)
{
    if (program == NULL || source == NULL || !program->active || program->value_count == 0u) {
        return -1;
    }

    if (nlo_str_builder_append(source,
                               "#version 460\n"
                               "#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n"
                               "#extension GL_GOOGLE_include_directive : require\n\n"
                               "#include \"nlo_complex_device.glslinc\"\n"
                               "#include \"nlo_double_math.glslinc\"\n"
                               "#include \"nlo_complex_math.glslinc\"\n\n") != 0 ||
        nlo_str_builder_appendf(source,
                                "layout(local_size_x = %u) in;\n\n",
                                NLO_VK_LOCAL_SIZE_X) != 0 ||
        nlo_str_builder_append(source,
                               "layout(push_constant) uniform PushConstants {\n"
                               "    uint count;\n"
                               "    uint _pad;\n"
                               "    double scalar0;\n"
                               "    double scalar1;\n"
                               "} pc;\n\n"
                               "layout(set = 0, binding = 0, std430) buffer DstBuffer { nlo_vk_complex64 dst[]; };\n"
                               "layout(set = 0, binding = 1, std430) readonly buffer FieldBuffer { nlo_vk_complex64 field_vals[]; };\n"
                               "layout(set = 0, binding = 2, std430) readonly buffer WBuffer { nlo_vk_complex64 w_vals[]; };\n"
                               "layout(set = 0, binding = 3, std430) readonly buffer WTBuffer { nlo_vk_complex64 wt_vals[]; };\n"
                               "layout(set = 0, binding = 4, std430) readonly buffer KXBuffer { nlo_vk_complex64 kx_vals[]; };\n"
                               "layout(set = 0, binding = 5, std430) readonly buffer KYBuffer { nlo_vk_complex64 ky_vals[]; };\n"
                               "layout(set = 0, binding = 6, std430) readonly buffer TBuffer { nlo_vk_complex64 t_vals[]; };\n"
                               "layout(set = 0, binding = 7, std430) readonly buffer XBuffer { nlo_vk_complex64 x_vals[]; };\n"
                               "layout(set = 0, binding = 8, std430) readonly buffer YBuffer { nlo_vk_complex64 y_vals[]; };\n"
                               "layout(set = 0, binding = 9, std430) readonly buffer DBuffer { nlo_vk_complex64 dispersion_vals[]; };\n"
                               "layout(set = 0, binding = 10, std430) readonly buffer VBuffer { nlo_vk_complex64 potential_vals[]; };\n\n"
                               "uint nlo_vk_symbol_nt()\n"
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
                               "uint nlo_vk_symbol_ny()\n"
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
                               "uint nlo_vk_symbol_t_index(uint idx)\n"
                               "{\n"
                               "    uint nt = nlo_vk_symbol_nt();\n"
                               "    return (nt > 0u) ? (idx % nt) : 0u;\n"
                               "}\n\n"
                               "uint nlo_vk_symbol_plane_index(uint idx)\n"
                               "{\n"
                               "    uint nt = nlo_vk_symbol_nt();\n"
                               "    return (nt > 0u) ? (idx / nt) : 0u;\n"
                               "}\n\n"
                               "uint nlo_vk_symbol_y_index(uint idx)\n"
                               "{\n"
                               "    uint ny = nlo_vk_symbol_ny();\n"
                               "    return (ny > 0u) ? (nlo_vk_symbol_plane_index(idx) % ny) : 0u;\n"
                               "}\n\n"
                               "uint nlo_vk_symbol_x_index(uint idx)\n"
                               "{\n"
                               "    uint ny = nlo_vk_symbol_ny();\n"
                               "    return (ny > 0u) ? (nlo_vk_symbol_plane_index(idx) / ny) : 0u;\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_w(uint idx)\n"
                               "{\n"
                               "    uint len = uint(w_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return w_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? w_vals[nlo_vk_symbol_t_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_wt(uint idx)\n"
                               "{\n"
                               "    uint len = uint(wt_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return wt_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? wt_vals[nlo_vk_symbol_t_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_kx(uint idx)\n"
                               "{\n"
                               "    uint len = uint(kx_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return kx_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? kx_vals[nlo_vk_symbol_x_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_ky(uint idx)\n"
                               "{\n"
                               "    uint len = uint(ky_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return ky_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? ky_vals[nlo_vk_symbol_y_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_t(uint idx)\n"
                               "{\n"
                               "    uint len = uint(t_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return t_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? t_vals[nlo_vk_symbol_t_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_x(uint idx)\n"
                               "{\n"
                               "    uint len = uint(x_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return x_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? x_vals[nlo_vk_symbol_x_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
                               "}\n\n"
                               "nlo_vk_complex64 nlo_vk_symbol_fetch_y(uint idx)\n"
                               "{\n"
                               "    uint len = uint(y_vals.length());\n"
                               "    if (len == pc.count) {\n"
                               "        return y_vals[idx];\n"
                               "    }\n"
                               "    return (len > 0u) ? y_vals[nlo_vk_symbol_y_index(idx) % len] : nlo_vk_make_complex(0.0, 0.0);\n"
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
        if (nlo_vk_emit_value_line(source, program, i) != 0) {
            return -1;
        }
    }

    if (nlo_str_builder_appendf(source,
                                "    dst[idx] = v%zu;\n}\n",
                                program->root_value) != 0) {
        return -1;
    }
    return 0;
}

static nlo_vec_status nlo_vk_compile_glsl_to_spirv(
    const char* source,
    uint32_t** out_words,
    size_t* out_word_count
)
{
    if (source == NULL || out_words == NULL || out_word_count == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (nlo_vk_glslang_acquire() != 0) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    nlo_vk_shader_include_context include_context = {
        .source_dir = NLO_VK_SHADER_SOURCE_DIR
    };
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
        .resource = glslang_default_resource(),
        .callbacks = {
            .include_system = nlo_vk_shader_include_system,
            .include_local = nlo_vk_shader_include_local,
            .free_include_result = nlo_vk_shader_include_free
        },
        .callbacks_ctx = &include_context
    };

    nlo_vec_status status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    glslang_shader_t* shader = glslang_shader_create(&input);
    glslang_program_t* program = NULL;

    if (shader == NULL) {
        goto cleanup;
    }
    if (glslang_shader_preprocess(shader, &input) == 0) {
        nlo_log_emit(NLO_LOG_LEVEL_WARN,
                     "[nlolib] Vulkan operator JIT preprocess failed: %s",
                     glslang_shader_get_info_log(shader));
        goto cleanup;
    }
    if (glslang_shader_parse(shader, &input) == 0) {
        nlo_log_emit(NLO_LOG_LEVEL_WARN,
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
        nlo_log_emit(NLO_LOG_LEVEL_WARN,
                     "[nlolib] Vulkan operator JIT link failed: %s",
                     glslang_program_get_info_log(program));
        goto cleanup;
    }

    glslang_spv_options_t spv_options = {
        .generate_debug_info = false,
        .strip_debug_info = true,
        .disable_optimizer = false,
        .optimize_size = false,
        .disassemble = false,
        .validate = false,
        .emit_nonsemantic_shader_debug_info = false,
        .emit_nonsemantic_shader_debug_source = false,
        .compile_only = false,
        .optimize_allow_expanded_id_bound = false
    };
    glslang_program_SPIRV_generate_with_options(program,
                                                GLSLANG_STAGE_COMPUTE,
                                                &spv_options);
    const size_t word_count = glslang_program_SPIRV_get_size(program);
    if (word_count == 0u) {
        const char* messages = glslang_program_SPIRV_get_messages(program);
        if (messages != NULL && messages[0] != '\0') {
            nlo_log_emit(NLO_LOG_LEVEL_WARN,
                         "[nlolib] Vulkan operator JIT SPIR-V generation failed: %s",
                         messages);
        }
        goto cleanup;
    }

    uint32_t* words = (uint32_t*)malloc(word_count * sizeof(uint32_t));
    if (words == NULL) {
        status = NLO_VEC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    glslang_program_SPIRV_get(program, words);
    *out_words = words;
    *out_word_count = word_count;
    status = NLO_VEC_STATUS_OK;

cleanup:
    if (program != NULL) {
        glslang_program_delete(program);
    }
    if (shader != NULL) {
        glslang_shader_delete(shader);
    }
    nlo_vk_glslang_release();
    return status;
}

static nlo_vec_status nlo_vk_create_compute_pipeline_from_words(
    nlo_vector_backend* backend,
    const uint32_t* words,
    size_t word_count,
    VkPipeline* out_pipeline
)
{
    if (backend == NULL || words == NULL || word_count == 0u || out_pipeline == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    VkShaderModuleCreateInfo module_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = word_count * sizeof(uint32_t),
        .pCode = words
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(backend->vk.device, &module_info, NULL, &module) != VK_SUCCESS) {
        return NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
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

    nlo_vec_status status = NLO_VEC_STATUS_OK;
    if (vkCreateComputePipelines(backend->vk.device,
                                 backend->vk.pipeline_cache,
                                 1u,
                                 &pipeline_info,
                                 NULL,
                                 out_pipeline) != VK_SUCCESS) {
        status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    }

    vkDestroyShaderModule(backend->vk.device, module, NULL);
    return status;
}

static nlo_vk_operator_jit_entry* nlo_vk_find_operator_jit_entry(
    nlo_vector_backend* backend,
    const nlo_operator_program* program
)
{
    if (backend == NULL || program == NULL) {
        return NULL;
    }

    nlo_vk_operator_jit_entry* entry = backend->vk.operator_jit_entries;
    while (entry != NULL) {
        if (nlo_vk_operator_programs_equal(entry, program)) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

static const nlo_vec_buffer* nlo_select_symbol_buffer(
    const nlo_operator_eval_context* eval_ctx,
    nlo_vk_descriptor_binding binding
)
{
    if (eval_ctx == NULL) {
        return NULL;
    }

    switch (binding) {
        case NLO_VK_BINDING_FIELD:
            return eval_ctx->field;
        case NLO_VK_BINDING_W:
            return (eval_ctx->frequency_grid != NULL) ? eval_ctx->frequency_grid : eval_ctx->wt_grid;
        case NLO_VK_BINDING_WT:
            return (eval_ctx->wt_grid != NULL) ? eval_ctx->wt_grid : eval_ctx->frequency_grid;
        case NLO_VK_BINDING_KX:
            return eval_ctx->kx_grid;
        case NLO_VK_BINDING_KY:
            return eval_ctx->ky_grid;
        case NLO_VK_BINDING_T:
            return eval_ctx->t_grid;
        case NLO_VK_BINDING_X:
            return eval_ctx->x_grid;
        case NLO_VK_BINDING_Y:
            return eval_ctx->y_grid;
        case NLO_VK_BINDING_D:
            return eval_ctx->dispersion_factor;
        case NLO_VK_BINDING_V:
            return eval_ctx->potential;
        default:
            return NULL;
    }
}

nlo_vec_status nlo_operator_program_prepare_jit(
    nlo_vector_backend* backend,
    nlo_operator_program* program
)
{
    if (backend == NULL || program == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (!program->active || backend->type != NLO_VECTOR_BACKEND_VULKAN || !program->vk_jit_eligible) {
        return NLO_VEC_STATUS_UNSUPPORTED;
    }
    if (program->vk_jit_entry != NULL) {
        program->vk_jit_active = 1;
        return NLO_VEC_STATUS_OK;
    }

    program->vk_jit_compile_attempted = 1;

    nlo_vk_operator_jit_entry* cached = nlo_vk_find_operator_jit_entry(backend, program);
    if (cached != NULL) {
        program->vk_jit_entry = cached;
        program->vk_jit_active = 1;
        return NLO_VEC_STATUS_OK;
    }

    nlo_string_builder source = {0};
    uint32_t* spirv_words = NULL;
    size_t spirv_word_count = 0u;
    nlo_vec_status status = NLO_VEC_STATUS_BACKEND_UNAVAILABLE;
    if (nlo_vk_build_operator_source(program, &source) != 0) {
        status = NLO_VEC_STATUS_INVALID_ARGUMENT;
        goto cleanup;
    }

    status = nlo_vk_compile_glsl_to_spirv(source.data, &spirv_words, &spirv_word_count);
    if (status != NLO_VEC_STATUS_OK) {
        goto cleanup;
    }

    nlo_vk_operator_jit_entry* entry = (nlo_vk_operator_jit_entry*)calloc(1u, sizeof(*entry));
    if (entry == NULL) {
        status = NLO_VEC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    entry->lowered_hash = program->lowered_hash;
    entry->context = program->context;
    entry->active_symbol_mask = program->active_symbol_mask;
    entry->value_count = program->value_count;
    entry->root_value = program->root_value;
    entry->local_size_x = NLO_VK_LOCAL_SIZE_X;
    memcpy(entry->values, program->values, program->value_count * sizeof(program->values[0]));

    status = nlo_vk_create_compute_pipeline_from_words(backend,
                                                       spirv_words,
                                                       spirv_word_count,
                                                       &entry->pipeline);
    if (status != NLO_VEC_STATUS_OK) {
        free(entry);
        goto cleanup;
    }

    entry->next = backend->vk.operator_jit_entries;
    backend->vk.operator_jit_entries = entry;
    program->vk_jit_entry = entry;
    program->vk_jit_active = 1;

cleanup:
    if (status != NLO_VEC_STATUS_OK) {
        program->vk_jit_active = 0;
    }
    free(spirv_words);
    nlo_str_builder_destroy(&source);
    return status;
}

static void nlo_vk_update_operator_descriptor_set(
    nlo_vector_backend* backend,
    VkDescriptorSet descriptor_set,
    const VkBuffer* buffers,
    const VkDeviceSize* ranges
)
{
    VkDescriptorBufferInfo infos[NLO_VK_DESCRIPTOR_BINDING_COUNT] = {0};
    VkWriteDescriptorSet writes[NLO_VK_DESCRIPTOR_BINDING_COUNT] = {0};

    for (uint32_t i = 0u; i < NLO_VK_DESCRIPTOR_BINDING_COUNT; ++i) {
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
                           NLO_VK_DESCRIPTOR_BINDING_COUNT,
                           writes,
                           0u,
                           NULL);
}

static size_t nlo_vk_collect_unique_buffers(
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

nlo_vec_status nlo_operator_program_execute_jit(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* out_vector
)
{
    if (backend == NULL || program == NULL || eval_ctx == NULL || out_vector == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    if (backend->type != NLO_VECTOR_BACKEND_VULKAN ||
        !program->vk_jit_active ||
        program->vk_jit_entry == NULL ||
        out_vector->owner != backend ||
        out_vector->kind != NLO_VEC_KIND_COMPLEX64) {
        return NLO_VEC_STATUS_UNSUPPORTED;
    }
    if (out_vector->length == 0u || out_vector->length > (size_t)UINT32_MAX) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    VkBuffer bound_buffers[NLO_VK_DESCRIPTOR_BINDING_COUNT] = {0};
    VkDeviceSize bound_ranges[NLO_VK_DESCRIPTOR_BINDING_COUNT] = {0};
    bound_buffers[NLO_VK_BINDING_DST] = out_vector->vk_buffer;
    bound_ranges[NLO_VK_BINDING_DST] = (VkDeviceSize)out_vector->bytes;

    for (uint32_t binding = NLO_VK_BINDING_FIELD; binding < NLO_VK_DESCRIPTOR_BINDING_COUNT; ++binding) {
        const nlo_vec_buffer* source = nlo_select_symbol_buffer(eval_ctx, (nlo_vk_descriptor_binding)binding);
        if (source != NULL) {
            if (source->owner != backend ||
                source->kind != NLO_VEC_KIND_COMPLEX64 ||
                source->bytes == 0u) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            if ((binding == NLO_VK_BINDING_FIELD ||
                 binding == NLO_VK_BINDING_D ||
                 binding == NLO_VK_BINDING_V) &&
                source->length != out_vector->length) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            bound_buffers[binding] = source->vk_buffer;
            bound_ranges[binding] = (VkDeviceSize)source->bytes;
            continue;
        }

        if (nlo_operator_symbol_uses_binding(program->active_symbol_mask,
                                             (nlo_vk_descriptor_binding)binding)) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }
        bound_buffers[binding] = out_vector->vk_buffer;
        bound_ranges[binding] = (VkDeviceSize)out_vector->bytes;
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    nlo_vec_status status = nlo_vk_acquire_descriptor_set(backend, &descriptor_set);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    status = nlo_vk_begin_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    VkBuffer unique_buffers[NLO_VK_DESCRIPTOR_BINDING_COUNT] = {0};
    VkDeviceSize unique_ranges[NLO_VK_DESCRIPTOR_BINDING_COUNT] = {0};
    const size_t unique_count = nlo_vk_collect_unique_buffers(bound_buffers,
                                                              bound_ranges,
                                                              NLO_VK_DESCRIPTOR_BINDING_COUNT,
                                                              unique_buffers,
                                                              unique_ranges,
                                                              NLO_VK_DESCRIPTOR_BINDING_COUNT);
    VkCommandBuffer cmd = backend->vk.command_buffer;
    for (size_t i = 0u; i < unique_count; ++i) {
        nlo_vk_cmd_transfer_to_compute(cmd,
                                       unique_buffers[i],
                                       0u,
                                       unique_ranges[i]);
        nlo_vk_cmd_compute_to_compute(cmd,
                                      unique_buffers[i],
                                      0u,
                                      unique_ranges[i]);
    }

    nlo_vk_update_operator_descriptor_set(backend,
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

    nlo_vk_push_constants push = {
        .count = (uint32_t)out_vector->length,
        .pad = 0u,
        .scalar0 = eval_ctx->half_step_size,
        .scalar1 = 0.0
    };
    vkCmdPushConstants(cmd,
                       backend->vk.pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0u,
                       (uint32_t)sizeof(push),
                       &push);

    const uint32_t groups =
        (uint32_t)((out_vector->length + (size_t)NLO_VK_LOCAL_SIZE_X - 1u) /
                   (size_t)NLO_VK_LOCAL_SIZE_X);
    vkCmdDispatch(cmd, groups, 1u, 1u);
    nlo_vk_cmd_compute_to_compute(cmd,
                                  out_vector->vk_buffer,
                                  0u,
                                  (VkDeviceSize)out_vector->bytes);
    nlo_vk_cmd_compute_to_transfer(cmd,
                                   out_vector->vk_buffer,
                                   0u,
                                   (VkDeviceSize)out_vector->bytes);

    status = nlo_vk_submit_commands(backend);
    if (status != NLO_VEC_STATUS_OK) {
        return status;
    }

    nlo_perf_profile_add_gpu_dispatch(1u,
                                      (uint64_t)unique_count + 1u,
                                      ((uint64_t)unique_count + 1u) * (uint64_t)out_vector->bytes);
    return NLO_VEC_STATUS_OK;
}

void nlo_vk_operator_jit_destroy_all(nlo_vector_backend* backend)
{
    if (backend == NULL || backend->type != NLO_VECTOR_BACKEND_VULKAN) {
        return;
    }

    nlo_vk_operator_jit_entry* entry = backend->vk.operator_jit_entries;
    while (entry != NULL) {
        nlo_vk_operator_jit_entry* next = entry->next;
        if (entry->pipeline != VK_NULL_HANDLE && backend->vk.device != VK_NULL_HANDLE) {
            vkDestroyPipeline(backend->vk.device, entry->pipeline, NULL);
        }
        free(entry);
        entry = next;
    }
    backend->vk.operator_jit_entries = NULL;
}
