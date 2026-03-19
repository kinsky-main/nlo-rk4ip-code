/**
 * @file operator_program_compile.c
 * @brief Runtime expression parser/compiler for operator programs.
 */

#include "physics/operator_program.h"
#include "backend/nlo_complex.h"
#include "numerics/vector_ops.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    NLO_TOKEN_END = 0,
    NLO_TOKEN_NUMBER = 1,
    NLO_TOKEN_IDENT = 2,
    NLO_TOKEN_PLUS = 3,
    NLO_TOKEN_MINUS = 4,
    NLO_TOKEN_STAR = 5,
    NLO_TOKEN_SLASH = 6,
    NLO_TOKEN_CARET = 7,
    NLO_TOKEN_LPAREN = 8,
    NLO_TOKEN_RPAREN = 9
} nlo_token_kind;

typedef struct {
    nlo_token_kind kind;
    double number;
    char ident[32];
} nlo_token;

typedef struct {
    const char* cursor;
    nlo_token current;
    nlo_operator_program_context context;
    size_t num_constants;
    const double* constants;
    nlo_operator_program* program;
} nlo_parser;

static int nlo_opcode_is_symbol_leaf(nlo_operator_opcode opcode)
{
    return (opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_W ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_A ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_I ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_D ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_V ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_H ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_WT ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_KX ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_KY ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_T ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_X ||
            opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_Y);
}

static int nlo_opcode_is_unary(nlo_operator_opcode opcode)
{
    return (opcode == NLO_OPERATOR_OP_NEGATE ||
            opcode == NLO_OPERATOR_OP_EXP ||
            opcode == NLO_OPERATOR_OP_LOG ||
            opcode == NLO_OPERATOR_OP_SQRT ||
            opcode == NLO_OPERATOR_OP_SIN ||
            opcode == NLO_OPERATOR_OP_COS ||
            opcode == NLO_OPERATOR_OP_POW_REAL_LITERAL);
}

static int nlo_opcode_is_binary(nlo_operator_opcode opcode)
{
    return (opcode == NLO_OPERATOR_OP_ADD ||
            opcode == NLO_OPERATOR_OP_MUL ||
            opcode == NLO_OPERATOR_OP_DIV ||
            opcode == NLO_OPERATOR_OP_POW);
}

static uint32_t nlo_symbol_mask_for_opcode(nlo_operator_opcode opcode)
{
    switch (opcode) {
        case NLO_OPERATOR_OP_PUSH_SYMBOL_W:
            return NLO_OPERATOR_SYMBOL_MASK_W;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_A:
            return NLO_OPERATOR_SYMBOL_MASK_A;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_I:
            return NLO_OPERATOR_SYMBOL_MASK_I;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_D:
            return NLO_OPERATOR_SYMBOL_MASK_D;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_V:
            return NLO_OPERATOR_SYMBOL_MASK_V;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_H:
            return NLO_OPERATOR_SYMBOL_MASK_H;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_WT:
            return NLO_OPERATOR_SYMBOL_MASK_WT;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_KX:
            return NLO_OPERATOR_SYMBOL_MASK_KX;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_KY:
            return NLO_OPERATOR_SYMBOL_MASK_KY;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_T:
            return NLO_OPERATOR_SYMBOL_MASK_T;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_X:
            return NLO_OPERATOR_SYMBOL_MASK_X;
        case NLO_OPERATOR_OP_PUSH_SYMBOL_Y:
            return NLO_OPERATOR_SYMBOL_MASK_Y;
        default:
            return NLO_OPERATOR_SYMBOL_MASK_NONE;
    }
}

static int nlo_value_is_literal(const nlo_operator_value* value)
{
    return (value != NULL && value->opcode == NLO_OPERATOR_OP_PUSH_LITERAL);
}

static int nlo_eval_unary_literal(
    nlo_operator_opcode opcode,
    nlo_complex operand,
    nlo_complex extra_literal,
    nlo_complex* out_value
)
{
    if (out_value == NULL) {
        return -1;
    }

    nlo_complex tmp[1] = {operand};
    switch (opcode) {
        case NLO_OPERATOR_OP_NEGATE:
            *out_value = nlo_make(-NLO_RE(operand), -NLO_IM(operand));
            return 0;
        case NLO_OPERATOR_OP_EXP:
            nlo_complex_exp_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case NLO_OPERATOR_OP_LOG:
            nlo_complex_log_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case NLO_OPERATOR_OP_SQRT:
            nlo_complex_real_pow_inplace(tmp, 1u, 0.5);
            *out_value = tmp[0];
            return 0;
        case NLO_OPERATOR_OP_SIN:
            nlo_complex_sin_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case NLO_OPERATOR_OP_COS:
            nlo_complex_cos_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case NLO_OPERATOR_OP_POW_REAL_LITERAL:
            nlo_complex_real_pow_inplace(tmp, 1u, NLO_RE(extra_literal));
            *out_value = tmp[0];
            return 0;
        default:
            return -1;
    }
}

static int nlo_eval_binary_literal(
    nlo_operator_opcode opcode,
    nlo_complex left,
    nlo_complex right,
    nlo_complex* out_value
)
{
    if (out_value == NULL) {
        return -1;
    }

    nlo_complex lhs[1] = {left};
    nlo_complex rhs[1] = {right};
    switch (opcode) {
        case NLO_OPERATOR_OP_ADD:
            nlo_complex_add_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        case NLO_OPERATOR_OP_MUL:
            nlo_complex_mul_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        case NLO_OPERATOR_OP_DIV:
            nlo_complex_real_pow_inplace(rhs, 1u, -1.0);
            nlo_complex_mul_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        case NLO_OPERATOR_OP_POW:
            nlo_complex_pow_elementwise_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        default:
            return -1;
    }
}

static int nlo_value_matches(
    const nlo_operator_value* value,
    nlo_operator_opcode opcode,
    uint16_t left,
    uint16_t right,
    nlo_complex literal,
    uint32_t symbol_mask
)
{
    return (value != NULL &&
            value->opcode == opcode &&
            value->left == left &&
            value->right == right &&
            value->symbol_mask == symbol_mask &&
            NLO_RE(value->literal) == NLO_RE(literal) &&
            NLO_IM(value->literal) == NLO_IM(literal));
}

static int nlo_program_intern_value(
    nlo_operator_program* program,
    nlo_operator_opcode opcode,
    uint16_t left,
    uint16_t right,
    nlo_complex literal,
    uint16_t* out_value_id
)
{
    if (program == NULL || out_value_id == NULL) {
        return -1;
    }

    uint32_t symbol_mask = NLO_OPERATOR_SYMBOL_MASK_NONE;
    if (nlo_opcode_is_symbol_leaf(opcode)) {
        symbol_mask = nlo_symbol_mask_for_opcode(opcode);
    } else if (nlo_opcode_is_unary(opcode)) {
        if (left == NLO_OPERATOR_VALUE_INVALID || left >= program->value_count) {
            return -1;
        }
        const nlo_operator_value* input = &program->values[left];
        symbol_mask = input->symbol_mask;
        if (nlo_value_is_literal(input)) {
            nlo_complex folded = nlo_make(0.0, 0.0);
            if (nlo_eval_unary_literal(opcode, input->literal, literal, &folded) != 0) {
                return -1;
            }
            opcode = NLO_OPERATOR_OP_PUSH_LITERAL;
            left = NLO_OPERATOR_VALUE_INVALID;
            right = NLO_OPERATOR_VALUE_INVALID;
            literal = folded;
            symbol_mask = NLO_OPERATOR_SYMBOL_MASK_NONE;
        }
    } else if (nlo_opcode_is_binary(opcode)) {
        if (left == NLO_OPERATOR_VALUE_INVALID || right == NLO_OPERATOR_VALUE_INVALID ||
            left >= program->value_count || right >= program->value_count) {
            return -1;
        }
        const nlo_operator_value* lhs = &program->values[left];
        const nlo_operator_value* rhs = &program->values[right];
        symbol_mask = lhs->symbol_mask | rhs->symbol_mask;
        if (nlo_value_is_literal(lhs) && nlo_value_is_literal(rhs)) {
            nlo_complex folded = nlo_make(0.0, 0.0);
            if (nlo_eval_binary_literal(opcode, lhs->literal, rhs->literal, &folded) != 0) {
                return -1;
            }
            opcode = NLO_OPERATOR_OP_PUSH_LITERAL;
            left = NLO_OPERATOR_VALUE_INVALID;
            right = NLO_OPERATOR_VALUE_INVALID;
            literal = folded;
            symbol_mask = NLO_OPERATOR_SYMBOL_MASK_NONE;
        }
    } else if (opcode != NLO_OPERATOR_OP_PUSH_LITERAL) {
        return -1;
    }

    for (size_t i = 0u; i < program->value_count; ++i) {
        if (nlo_value_matches(&program->values[i], opcode, left, right, literal, symbol_mask)) {
            *out_value_id = (uint16_t)i;
            return 0;
        }
    }

    if (program->value_count >= NLO_OPERATOR_PROGRAM_MAX_VALUES) {
        return -1;
    }

    program->values[program->value_count] = (nlo_operator_value){
        .opcode = opcode,
        .left = left,
        .right = right,
        .literal = literal,
        .symbol_mask = symbol_mask
    };
    *out_value_id = (uint16_t)program->value_count;
    program->value_count += 1u;
    return 0;
}

static uint64_t nlo_program_hash_mix(uint64_t hash, uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t nlo_program_hash_double(double value)
{
    union {
        double d;
        uint64_t u;
    } bits = {.d = value};
    return bits.u;
}

static uint64_t nlo_program_compute_lowered_hash(const nlo_operator_program* program)
{
    if (program == NULL) {
        return 0u;
    }

    uint64_t hash = 1469598103934665603ull;
    hash = nlo_program_hash_mix(hash, (uint64_t)program->context);
    hash = nlo_program_hash_mix(hash, (uint64_t)program->value_count);
    hash = nlo_program_hash_mix(hash, (uint64_t)program->root_value);
    hash = nlo_program_hash_mix(hash, (uint64_t)program->active_symbol_mask);
    for (size_t i = 0u; i < program->value_count; ++i) {
        const nlo_operator_value* value = &program->values[i];
        hash = nlo_program_hash_mix(hash, (uint64_t)value->opcode);
        hash = nlo_program_hash_mix(hash, (uint64_t)value->left);
        hash = nlo_program_hash_mix(hash, (uint64_t)value->right);
        hash = nlo_program_hash_mix(hash, (uint64_t)value->symbol_mask);
        hash = nlo_program_hash_mix(hash, nlo_program_hash_double(NLO_RE(value->literal)));
        hash = nlo_program_hash_mix(hash, nlo_program_hash_double(NLO_IM(value->literal)));
    }
    return hash;
}

static void nlo_program_mark_reachable(
    const nlo_operator_program* program,
    uint16_t value_id,
    unsigned char* marked
)
{
    if (program == NULL || marked == NULL ||
        value_id == NLO_OPERATOR_VALUE_INVALID || value_id >= program->value_count ||
        marked[value_id] != 0u) {
        return;
    }

    marked[value_id] = 1u;
    const nlo_operator_value* value = &program->values[value_id];
    nlo_program_mark_reachable(program, value->left, marked);
    nlo_program_mark_reachable(program, value->right, marked);
}

static int nlo_program_compact_values(nlo_operator_program* program)
{
    if (program == NULL || program->value_count == 0u ||
        program->root_value >= program->value_count) {
        return -1;
    }

    unsigned char marked[NLO_OPERATOR_PROGRAM_MAX_VALUES] = {0};
    uint16_t remap[NLO_OPERATOR_PROGRAM_MAX_VALUES];
    nlo_operator_value compacted[NLO_OPERATOR_PROGRAM_MAX_VALUES];
    for (size_t i = 0u; i < NLO_OPERATOR_PROGRAM_MAX_VALUES; ++i) {
        remap[i] = NLO_OPERATOR_VALUE_INVALID;
    }

    nlo_program_mark_reachable(program, (uint16_t)program->root_value, marked);

    size_t compacted_count = 0u;
    for (size_t i = 0u; i < program->value_count; ++i) {
        if (marked[i] == 0u) {
            continue;
        }
        if (compacted_count >= NLO_OPERATOR_PROGRAM_MAX_VALUES) {
            return -1;
        }
        remap[i] = (uint16_t)compacted_count;
        compacted[compacted_count++] = program->values[i];
    }

    if (compacted_count == 0u) {
        return -1;
    }

    for (size_t i = 0u; i < compacted_count; ++i) {
        nlo_operator_value* value = &compacted[i];
        if (value->left != NLO_OPERATOR_VALUE_INVALID) {
            if (value->left >= program->value_count ||
                remap[value->left] == NLO_OPERATOR_VALUE_INVALID) {
                return -1;
            }
            value->left = remap[value->left];
        }
        if (value->right != NLO_OPERATOR_VALUE_INVALID) {
            if (value->right >= program->value_count ||
                remap[value->right] == NLO_OPERATOR_VALUE_INVALID) {
                return -1;
            }
            value->right = remap[value->right];
        }
        program->values[i] = *value;
    }

    program->root_value = remap[program->root_value];
    program->value_count = compacted_count;
    return 0;
}

static int nlo_program_lower_values(nlo_operator_program* program)
{
    if (program == NULL) {
        return -1;
    }

    program->value_count = 0u;
    program->root_value = 0u;
    program->active_symbol_mask = NLO_OPERATOR_SYMBOL_MASK_NONE;
    program->lowered_hash = 0u;
    program->vk_jit_eligible = 0;
    program->vk_jit_active = 0;
    program->vk_jit_compile_attempted = 0;
    program->vk_jit_warning_emitted = 0;
    program->vk_jit_entry = NULL;

    if (!program->active) {
        return 0;
    }

    uint16_t stack[NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS] = {0};
    size_t depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const nlo_operator_instruction instruction = program->instructions[i];
        uint16_t value_id = NLO_OPERATOR_VALUE_INVALID;

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_LITERAL) {
            if (depth >= NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS ||
                nlo_program_intern_value(program,
                                         NLO_OPERATOR_OP_PUSH_LITERAL,
                                         NLO_OPERATOR_VALUE_INVALID,
                                         NLO_OPERATOR_VALUE_INVALID,
                                         instruction.literal,
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth++] = value_id;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_IMAG_UNIT) {
            if (depth >= NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS ||
                nlo_program_intern_value(program,
                                         NLO_OPERATOR_OP_PUSH_LITERAL,
                                         NLO_OPERATOR_VALUE_INVALID,
                                         NLO_OPERATOR_VALUE_INVALID,
                                         nlo_make(0.0, 1.0),
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth++] = value_id;
            continue;
        }

        if (nlo_opcode_is_symbol_leaf(instruction.opcode)) {
            if (depth >= NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS ||
                nlo_program_intern_value(program,
                                         instruction.opcode,
                                         NLO_OPERATOR_VALUE_INVALID,
                                         NLO_OPERATOR_VALUE_INVALID,
                                         nlo_make(0.0, 0.0),
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth++] = value_id;
            continue;
        }

        if (nlo_opcode_is_unary(instruction.opcode)) {
            if (depth == 0u ||
                nlo_program_intern_value(program,
                                         instruction.opcode,
                                         stack[depth - 1u],
                                         NLO_OPERATOR_VALUE_INVALID,
                                         instruction.literal,
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth - 1u] = value_id;
            continue;
        }

        if (nlo_opcode_is_binary(instruction.opcode)) {
            if (depth < 2u ||
                nlo_program_intern_value(program,
                                         instruction.opcode,
                                         stack[depth - 2u],
                                         stack[depth - 1u],
                                         nlo_make(0.0, 0.0),
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth - 2u] = value_id;
            depth -= 1u;
            continue;
        }

        return -1;
    }

    if (depth != 1u || stack[0] == NLO_OPERATOR_VALUE_INVALID || stack[0] >= program->value_count) {
        return -1;
    }

    program->root_value = (size_t)stack[0];
    if (nlo_program_compact_values(program) != 0) {
        return -1;
    }
    program->active_symbol_mask = program->values[program->root_value].symbol_mask;
    program->lowered_hash = nlo_program_compute_lowered_hash(program);
    program->vk_jit_eligible = 1;
    return 0;
}

static int nlo_is_identifier_start(char c)
{
    return (c == '_') || isalpha((unsigned char)c);
}

static int nlo_is_identifier_char(char c)
{
    return (c == '_') || isalnum((unsigned char)c);
}

static int nlo_context_allows_symbol_w(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_DISPERSION_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_DISPERSION);
}

static int nlo_context_allows_symbol_wt(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR ||
            context == NLO_OPERATOR_CONTEXT_DISPERSION_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_DISPERSION);
}

static int nlo_context_allows_symbol_kx(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR);
}

static int nlo_context_allows_symbol_ky(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR);
}

static int nlo_context_allows_symbol_t(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR ||
            context == NLO_OPERATOR_CONTEXT_NONLINEAR ||
            context == NLO_OPERATOR_CONTEXT_POTENTIAL);
}

static int nlo_context_allows_symbol_x(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR ||
            context == NLO_OPERATOR_CONTEXT_NONLINEAR ||
            context == NLO_OPERATOR_CONTEXT_POTENTIAL);
}

static int nlo_context_allows_symbol_y(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR ||
            context == NLO_OPERATOR_CONTEXT_NONLINEAR ||
            context == NLO_OPERATOR_CONTEXT_POTENTIAL);
}

static int nlo_context_allows_symbol_a(nlo_operator_program_context context)
{
    (void)context;
    return 1;
}

static int nlo_context_allows_symbol_i(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_NONLINEAR);
}

static int nlo_context_allows_symbol_d(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_DISPERSION ||
            context == NLO_OPERATOR_CONTEXT_LINEAR);
}

static int nlo_context_allows_symbol_v(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_DISPERSION ||
            context == NLO_OPERATOR_CONTEXT_NONLINEAR ||
            context == NLO_OPERATOR_CONTEXT_LINEAR);
}

static int nlo_context_allows_symbol_h(nlo_operator_program_context context)
{
    return (context == NLO_OPERATOR_CONTEXT_DISPERSION ||
            context == NLO_OPERATOR_CONTEXT_LINEAR);
}

static int nlo_emit_instruction(
    nlo_operator_program* program,
    nlo_operator_opcode opcode,
    nlo_complex literal
)
{
    if (program == NULL || program->instruction_count >= NLO_OPERATOR_PROGRAM_MAX_INSTRUCTIONS) {
        return -1;
    }

    program->instructions[program->instruction_count++] = (nlo_operator_instruction){
        .opcode = opcode,
        .literal = literal
    };
    return 0;
}

static int nlo_parser_next_token(nlo_parser* parser)
{
    if (parser == NULL || parser->cursor == NULL) {
        return -1;
    }

    while (*parser->cursor != '\0' && isspace((unsigned char)*parser->cursor)) {
        ++parser->cursor;
    }

    const char c = *parser->cursor;
    if (c == '\0') {
        parser->current = (nlo_token){.kind = NLO_TOKEN_END};
        return 0;
    }

    if (c == '+') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_PLUS};
        return 0;
    }
    if (c == '-') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_MINUS};
        return 0;
    }
    if (c == '*') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_STAR};
        return 0;
    }
    if (c == '/') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_SLASH};
        return 0;
    }
    if (c == '^') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_CARET};
        return 0;
    }
    if (c == '(') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_LPAREN};
        return 0;
    }
    if (c == ')') {
        ++parser->cursor;
        parser->current = (nlo_token){.kind = NLO_TOKEN_RPAREN};
        return 0;
    }

    if (isdigit((unsigned char)c) || c == '.') {
        char* end_ptr = NULL;
        const double value = strtod(parser->cursor, &end_ptr);
        if (end_ptr == parser->cursor) {
            return -1;
        }

        parser->cursor = end_ptr;
        parser->current = (nlo_token){
            .kind = NLO_TOKEN_NUMBER,
            .number = value
        };
        return 0;
    }

    if (nlo_is_identifier_start(c)) {
        size_t len = 0u;
        while (nlo_is_identifier_char(parser->cursor[len])) {
            if (len + 1u >= sizeof(parser->current.ident)) {
                return -1;
            }
            ++len;
        }

        memset(parser->current.ident, 0, sizeof(parser->current.ident));
        memcpy(parser->current.ident, parser->cursor, len);
        parser->cursor += len;
        parser->current.kind = NLO_TOKEN_IDENT;
        return 0;
    }

    return -1;
}

static int nlo_parse_expression(nlo_parser* parser);
static int nlo_parse_term(nlo_parser* parser);
static int nlo_parse_power(nlo_parser* parser);
static int nlo_parse_unary(nlo_parser* parser);
static int nlo_parse_primary(nlo_parser* parser);

static int nlo_parse_function_call(
    nlo_parser* parser,
    nlo_operator_opcode opcode
)
{
    if (parser == NULL || parser->program == NULL) {
        return -1;
    }

    if (nlo_parser_next_token(parser) != 0 || parser->current.kind != NLO_TOKEN_LPAREN) {
        return -1;
    }
    if (nlo_parser_next_token(parser) != 0) {
        return -1;
    }
    if (nlo_parse_expression(parser) != 0 || parser->current.kind != NLO_TOKEN_RPAREN) {
        return -1;
    }
    if (nlo_emit_instruction(parser->program, opcode, nlo_make(0.0, 0.0)) != 0) {
        return -1;
    }

    return nlo_parser_next_token(parser);
}

static int nlo_parse_primary(nlo_parser* parser)
{
    if (parser == NULL || parser->program == NULL) {
        return -1;
    }

    if (parser->current.kind == NLO_TOKEN_NUMBER) {
        const nlo_complex literal = nlo_make(parser->current.number, 0.0);
        if (nlo_emit_instruction(parser->program,
                                 NLO_OPERATOR_OP_PUSH_LITERAL,
                                 literal) != 0) {
            return -1;
        }

        return nlo_parser_next_token(parser);
    }

    if (parser->current.kind == NLO_TOKEN_IDENT) {
        if (strcmp(parser->current.ident, "exp") == 0) {
            return nlo_parse_function_call(parser, NLO_OPERATOR_OP_EXP);
        }

        if (strcmp(parser->current.ident, "log") == 0) {
            return nlo_parse_function_call(parser, NLO_OPERATOR_OP_LOG);
        }

        if (strcmp(parser->current.ident, "sqrt") == 0) {
            return nlo_parse_function_call(parser, NLO_OPERATOR_OP_SQRT);
        }

        if (strcmp(parser->current.ident, "sin") == 0) {
            return nlo_parse_function_call(parser, NLO_OPERATOR_OP_SIN);
        }

        if (strcmp(parser->current.ident, "cos") == 0) {
            return nlo_parse_function_call(parser, NLO_OPERATOR_OP_COS);
        }

        if (strcmp(parser->current.ident, "w") == 0) {
            if (nlo_context_allows_symbol_w(parser->context)) {
                if (nlo_emit_instruction(parser->program,
                                         NLO_OPERATOR_OP_PUSH_SYMBOL_W,
                                         nlo_make(0.0, 0.0)) != 0) {
                    return -1;
                }
                return nlo_parser_next_token(parser);
            }
            if (nlo_context_allows_symbol_wt(parser->context)) {
                if (nlo_emit_instruction(parser->program,
                                         NLO_OPERATOR_OP_PUSH_SYMBOL_WT,
                                         nlo_make(0.0, 0.0)) != 0) {
                    return -1;
                }
                return nlo_parser_next_token(parser);
            }
            return -1;
        }

        if (strcmp(parser->current.ident, "wt") == 0) {
            if (!nlo_context_allows_symbol_wt(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_WT,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "kx") == 0) {
            if (!nlo_context_allows_symbol_kx(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_KX,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "ky") == 0) {
            if (!nlo_context_allows_symbol_ky(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_KY,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "t") == 0) {
            if (!nlo_context_allows_symbol_t(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_T,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "x") == 0) {
            if (!nlo_context_allows_symbol_x(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_X,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "y") == 0) {
            if (!nlo_context_allows_symbol_y(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_Y,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "A") == 0) {
            if (!nlo_context_allows_symbol_a(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_A,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "I") == 0) {
            if (!nlo_context_allows_symbol_i(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_I,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "D") == 0) {
            if (!nlo_context_allows_symbol_d(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_D,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "V") == 0) {
            if (!nlo_context_allows_symbol_v(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_V,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "h") == 0) {
            if (!nlo_context_allows_symbol_h(parser->context)) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_H,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "i") == 0) {
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_IMAG_UNIT,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (parser->current.ident[0] == 'c') {
            char* end_ptr = NULL;
            const long constant_index = strtol(parser->current.ident + 1, &end_ptr, 10);
            if (end_ptr == NULL || *end_ptr != '\0' || constant_index < 0) {
                return -1;
            }
            if ((size_t)constant_index >= parser->num_constants || parser->constants == NULL) {
                return -1;
            }

            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_LITERAL,
                                     nlo_make(parser->constants[constant_index], 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        return -1;
    }

    if (parser->current.kind == NLO_TOKEN_LPAREN) {
        if (nlo_parser_next_token(parser) != 0) {
            return -1;
        }
        if (nlo_parse_expression(parser) != 0 || parser->current.kind != NLO_TOKEN_RPAREN) {
            return -1;
        }
        return nlo_parser_next_token(parser);
    }

    return -1;
}

static int nlo_parse_unary(nlo_parser* parser)
{
    if (parser == NULL || parser->program == NULL) {
        return -1;
    }

    if (parser->current.kind == NLO_TOKEN_MINUS) {
        if (nlo_parser_next_token(parser) != 0) {
            return -1;
        }
        if (nlo_parse_unary(parser) != 0) {
            return -1;
        }
        return nlo_emit_instruction(parser->program,
                                    NLO_OPERATOR_OP_NEGATE,
                                    nlo_make(0.0, 0.0));
    }

    return nlo_parse_primary(parser);
}

static int nlo_parse_power(nlo_parser* parser)
{
    if (nlo_parse_unary(parser) != 0) {
        return -1;
    }

    if (parser->current.kind == NLO_TOKEN_CARET) {
        if (nlo_parser_next_token(parser) != 0 || nlo_parse_power(parser) != 0) {
            return -1;
        }
        if (nlo_emit_instruction(parser->program,
                                 NLO_OPERATOR_OP_POW,
                                 nlo_make(0.0, 0.0)) != 0) {
            return -1;
        }
    }

    return 0;
}

static int nlo_parse_term(nlo_parser* parser)
{
    if (nlo_parse_power(parser) != 0) {
        return -1;
    }

    while (parser->current.kind == NLO_TOKEN_STAR || parser->current.kind == NLO_TOKEN_SLASH) {
        const nlo_token_kind op = parser->current.kind;
        if (nlo_parser_next_token(parser) != 0 || nlo_parse_power(parser) != 0) {
            return -1;
        }
        if (nlo_emit_instruction(parser->program,
                                 (op == NLO_TOKEN_STAR)
                                     ? NLO_OPERATOR_OP_MUL
                                     : NLO_OPERATOR_OP_DIV,
                                 nlo_make(0.0, 0.0)) != 0) {
            return -1;
        }
    }

    return 0;
}

static int nlo_parse_expression(nlo_parser* parser)
{
    if (nlo_parse_term(parser) != 0) {
        return -1;
    }

    while (parser->current.kind == NLO_TOKEN_PLUS || parser->current.kind == NLO_TOKEN_MINUS) {
        const nlo_token_kind op = parser->current.kind;
        if (nlo_parser_next_token(parser) != 0 || nlo_parse_term(parser) != 0) {
            return -1;
        }

        if (op == NLO_TOKEN_MINUS) {
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_NEGATE,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
        }

        if (nlo_emit_instruction(parser->program,
                                 NLO_OPERATOR_OP_ADD,
                                 nlo_make(0.0, 0.0)) != 0) {
            return -1;
        }
    }

    return 0;
}

static int nlo_program_compute_stack_requirements(nlo_operator_program* program)
{
    if (program == NULL) {
        return -1;
    }

    size_t depth = 0u;
    size_t max_depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const nlo_operator_opcode op = program->instructions[i].opcode;
        if (op == NLO_OPERATOR_OP_PUSH_LITERAL ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_W ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_WT ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_KX ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_KY ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_T ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_X ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_Y ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_A ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_I ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_D ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_V ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_H ||
            op == NLO_OPERATOR_OP_PUSH_IMAG_UNIT) {
            ++depth;
            if (depth > max_depth) {
                max_depth = depth;
            }
            continue;
        }

        if (op == NLO_OPERATOR_OP_NEGATE ||
            op == NLO_OPERATOR_OP_EXP ||
            op == NLO_OPERATOR_OP_LOG ||
            op == NLO_OPERATOR_OP_SQRT ||
            op == NLO_OPERATOR_OP_SIN ||
            op == NLO_OPERATOR_OP_COS ||
            op == NLO_OPERATOR_OP_POW_REAL_LITERAL) {
            if (depth < 1u) {
                return -1;
            }
            continue;
        }

        if (op == NLO_OPERATOR_OP_ADD ||
            op == NLO_OPERATOR_OP_MUL ||
            op == NLO_OPERATOR_OP_DIV ||
            op == NLO_OPERATOR_OP_POW) {
            if (depth < 2u) {
                return -1;
            }
            depth -= 1u;
            continue;
        }

        return -1;
    }

    if (depth != 1u) {
        return -1;
    }
    if (max_depth == 0u || max_depth > NLO_OPERATOR_PROGRAM_MAX_STACK_SLOTS) {
        return -1;
    }

    program->required_stack_slots = max_depth;
    return 0;
}

static void nlo_program_lower_real_literal_powers(nlo_operator_program* program)
{
    if (program == NULL || program->instruction_count == 0u) {
        return;
    }

    size_t write = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const nlo_operator_instruction instruction = program->instructions[i];
        if (instruction.opcode == NLO_OPERATOR_OP_POW &&
            write > 0u &&
            program->instructions[write - 1u].opcode == NLO_OPERATOR_OP_PUSH_LITERAL &&
            NLO_IM(program->instructions[write - 1u].literal) == 0.0) {
            const double exponent = NLO_RE(program->instructions[write - 1u].literal);
            write -= 1u;
            program->instructions[write++] = (nlo_operator_instruction){
                .opcode = NLO_OPERATOR_OP_POW_REAL_LITERAL,
                .literal = nlo_make(exponent, 0.0)
            };
            continue;
        }

        program->instructions[write++] = instruction;
    }

    program->instruction_count = write;
}

nlo_vec_status nlo_operator_program_compile(
    const char* expression,
    nlo_operator_program_context context,
    size_t num_constants,
    const double* constants,
    nlo_operator_program* out_program
)
{
    if (out_program == NULL) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    *out_program = (nlo_operator_program){0};
    out_program->context = context;
    if (expression == NULL || expression[0] == '\0') {
        return NLO_VEC_STATUS_OK;
    }

    nlo_parser parser = {
        .cursor = expression,
        .context = context,
        .num_constants = num_constants,
        .constants = constants,
        .program = out_program
    };

    if (nlo_parser_next_token(&parser) != 0 ||
        nlo_parse_expression(&parser) != 0 ||
        parser.current.kind != NLO_TOKEN_END) {
        *out_program = (nlo_operator_program){0};
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    nlo_program_lower_real_literal_powers(out_program);
    if (nlo_program_compute_stack_requirements(out_program) != 0) {
        *out_program = (nlo_operator_program){0};
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    out_program->active = 1;
    if (nlo_program_lower_values(out_program) != 0) {
        *out_program = (nlo_operator_program){0};
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }
    return NLO_VEC_STATUS_OK;
}
