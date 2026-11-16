/**
 * @file operator_program_compile.c
 * @brief Runtime expression parser/compiler for operator programs.
 */

#include "physics/operator_program.h"
#include "backend/nlo_complex.h"
#include "numerics/vector_ops.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TOKEN_END = 0,
    TOKEN_NUMBER = 1,
    TOKEN_IDENT = 2,
    TOKEN_PLUS = 3,
    TOKEN_MINUS = 4,
    TOKEN_STAR = 5,
    TOKEN_SLASH = 6,
    TOKEN_CARET = 7,
    TOKEN_LPAREN = 8,
    TOKEN_RPAREN = 9
} token_kind;

typedef struct {
    token_kind kind;
    double number;
    char ident[32];
} token;

typedef struct {
    const char* cursor;
    token current;
    operator_program_context context;
    size_t num_constants;
    const double* constants;
    operator_program* program;
} parser;

static int lowering_opcode_is_symbol_leaf(operator_opcode opcode)
{
    return (opcode == OPERATOR_OP_PUSH_SYMBOL_W ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_A ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_I ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_D ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_V ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_H ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_WT ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_KX ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_KY ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_T ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_X ||
            opcode == OPERATOR_OP_PUSH_SYMBOL_Y);
}

static int lowering_opcode_is_unary(operator_opcode opcode)
{
    return (opcode == OPERATOR_OP_NEGATE ||
            opcode == OPERATOR_OP_EXP ||
            opcode == OPERATOR_OP_LOG ||
            opcode == OPERATOR_OP_SQRT ||
            opcode == OPERATOR_OP_SIN ||
            opcode == OPERATOR_OP_COS ||
            opcode == OPERATOR_OP_POW_REAL_LITERAL);
}

static int lowering_opcode_is_binary(operator_opcode opcode)
{
    return (opcode == OPERATOR_OP_ADD ||
            opcode == OPERATOR_OP_MUL ||
            opcode == OPERATOR_OP_DIV ||
            opcode == OPERATOR_OP_POW);
}

static uint32_t lowering_symbol_mask_for_opcode(operator_opcode opcode)
{
    switch (opcode) {
        case OPERATOR_OP_PUSH_SYMBOL_W:
            return OPERATOR_SYMBOL_MASK_W;
        case OPERATOR_OP_PUSH_SYMBOL_A:
            return OPERATOR_SYMBOL_MASK_A;
        case OPERATOR_OP_PUSH_SYMBOL_I:
            return OPERATOR_SYMBOL_MASK_I;
        case OPERATOR_OP_PUSH_SYMBOL_D:
            return OPERATOR_SYMBOL_MASK_D;
        case OPERATOR_OP_PUSH_SYMBOL_V:
            return OPERATOR_SYMBOL_MASK_V;
        case OPERATOR_OP_PUSH_SYMBOL_H:
            return OPERATOR_SYMBOL_MASK_H;
        case OPERATOR_OP_PUSH_SYMBOL_WT:
            return OPERATOR_SYMBOL_MASK_WT;
        case OPERATOR_OP_PUSH_SYMBOL_KX:
            return OPERATOR_SYMBOL_MASK_KX;
        case OPERATOR_OP_PUSH_SYMBOL_KY:
            return OPERATOR_SYMBOL_MASK_KY;
        case OPERATOR_OP_PUSH_SYMBOL_T:
            return OPERATOR_SYMBOL_MASK_T;
        case OPERATOR_OP_PUSH_SYMBOL_X:
            return OPERATOR_SYMBOL_MASK_X;
        case OPERATOR_OP_PUSH_SYMBOL_Y:
            return OPERATOR_SYMBOL_MASK_Y;
        default:
            return OPERATOR_SYMBOL_MASK_NONE;
    }
}

static int lowering_value_is_literal(const operator_value* value)
{
    return (value != NULL && value->opcode == OPERATOR_OP_PUSH_LITERAL);
}

static int lowering_eval_unary_literal(
    operator_opcode opcode,
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
        case OPERATOR_OP_NEGATE:
            *out_value = make(-RE(operand), -IM(operand));
            return 0;
        case OPERATOR_OP_EXP:
            complex_exp_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case OPERATOR_OP_LOG:
            complex_log_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case OPERATOR_OP_SQRT:
            complex_real_pow_inplace(tmp, 1u, 0.5);
            *out_value = tmp[0];
            return 0;
        case OPERATOR_OP_SIN:
            complex_sin_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case OPERATOR_OP_COS:
            complex_cos_inplace(tmp, 1u);
            *out_value = tmp[0];
            return 0;
        case OPERATOR_OP_POW_REAL_LITERAL:
            complex_real_pow_inplace(tmp, 1u, RE(extra_literal));
            *out_value = tmp[0];
            return 0;
        default:
            return -1;
    }
}

static int lowering_eval_binary_literal(
    operator_opcode opcode,
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
        case OPERATOR_OP_ADD:
            complex_add_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        case OPERATOR_OP_MUL:
            complex_mul_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        case OPERATOR_OP_DIV:
            complex_real_pow_inplace(rhs, 1u, -1.0);
            complex_mul_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        case OPERATOR_OP_POW:
            complex_pow_elementwise_inplace(lhs, rhs, 1u);
            *out_value = lhs[0];
            return 0;
        default:
            return -1;
    }
}

static int lowering_value_matches(
    const operator_value* value,
    operator_opcode opcode,
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
            RE(value->literal) == RE(literal) &&
            IM(value->literal) == IM(literal));
}

static int lowering_program_intern_value(
    operator_program* program,
    operator_opcode opcode,
    uint16_t left,
    uint16_t right,
    nlo_complex literal,
    uint16_t* out_value_id
)
{
    if (program == NULL || out_value_id == NULL) {
        return -1;
    }

    uint32_t symbol_mask = OPERATOR_SYMBOL_MASK_NONE;
    if (lowering_opcode_is_symbol_leaf(opcode)) {
        symbol_mask = lowering_symbol_mask_for_opcode(opcode);
    } else if (lowering_opcode_is_unary(opcode)) {
        if (left == OPERATOR_VALUE_INVALID || left >= program->value_count) {
            return -1;
        }
        const operator_value* input = &program->values[left];
        symbol_mask = input->symbol_mask;
        if (lowering_value_is_literal(input)) {
            nlo_complex folded = make(0.0, 0.0);
            if (lowering_eval_unary_literal(opcode, input->literal, literal, &folded) != 0) {
                return -1;
            }
            opcode = OPERATOR_OP_PUSH_LITERAL;
            left = OPERATOR_VALUE_INVALID;
            right = OPERATOR_VALUE_INVALID;
            literal = folded;
            symbol_mask = OPERATOR_SYMBOL_MASK_NONE;
        }
    } else if (lowering_opcode_is_binary(opcode)) {
        if (left == OPERATOR_VALUE_INVALID || right == OPERATOR_VALUE_INVALID ||
            left >= program->value_count || right >= program->value_count) {
            return -1;
        }
        const operator_value* lhs = &program->values[left];
        const operator_value* rhs = &program->values[right];
        symbol_mask = lhs->symbol_mask | rhs->symbol_mask;
        if (lowering_value_is_literal(lhs) && lowering_value_is_literal(rhs)) {
            nlo_complex folded = make(0.0, 0.0);
            if (lowering_eval_binary_literal(opcode, lhs->literal, rhs->literal, &folded) != 0) {
                return -1;
            }
            opcode = OPERATOR_OP_PUSH_LITERAL;
            left = OPERATOR_VALUE_INVALID;
            right = OPERATOR_VALUE_INVALID;
            literal = folded;
            symbol_mask = OPERATOR_SYMBOL_MASK_NONE;
        }
    } else if (opcode != OPERATOR_OP_PUSH_LITERAL) {
        return -1;
    }

    for (size_t i = 0u; i < program->value_count; ++i) {
        if (lowering_value_matches(&program->values[i], opcode, left, right, literal, symbol_mask)) {
            *out_value_id = (uint16_t)i;
            return 0;
        }
    }

    if (program->value_count >= OPERATOR_PROGRAM_MAX_VALUES) {
        return -1;
    }

    program->values[program->value_count] = (operator_value){
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

static uint64_t lowering_program_hash_mix(uint64_t hash, uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t lowering_program_hash_double(double value)
{
    union {
        double d;
        uint64_t u;
    } bits = {.d = value};
    return bits.u;
}

static uint64_t lowering_program_compute_lowered_hash(const operator_program* program)
{
    if (program == NULL) {
        return 0u;
    }

    uint64_t hash = 1469598103934665603ull;
    hash = lowering_program_hash_mix(hash, (uint64_t)program->context);
    hash = lowering_program_hash_mix(hash, (uint64_t)program->value_count);
    hash = lowering_program_hash_mix(hash, (uint64_t)program->root_value);
    hash = lowering_program_hash_mix(hash, (uint64_t)program->active_symbol_mask);
    for (size_t i = 0u; i < program->value_count; ++i) {
        const operator_value* value = &program->values[i];
        hash = lowering_program_hash_mix(hash, (uint64_t)value->opcode);
        hash = lowering_program_hash_mix(hash, (uint64_t)value->left);
        hash = lowering_program_hash_mix(hash, (uint64_t)value->right);
        hash = lowering_program_hash_mix(hash, (uint64_t)value->symbol_mask);
        hash = lowering_program_hash_mix(hash, lowering_program_hash_double(RE(value->literal)));
        hash = lowering_program_hash_mix(hash, lowering_program_hash_double(IM(value->literal)));
    }
    return hash;
}

static void lowering_program_mark_reachable(
    const operator_program* program,
    uint16_t value_id,
    unsigned char* marked
)
{
    if (program == NULL || marked == NULL ||
        value_id == OPERATOR_VALUE_INVALID || value_id >= program->value_count ||
        marked[value_id] != 0u) {
        return;
    }

    marked[value_id] = 1u;
    const operator_value* value = &program->values[value_id];
    lowering_program_mark_reachable(program, value->left, marked);
    lowering_program_mark_reachable(program, value->right, marked);
}

static int lowering_program_compact_values(operator_program* program)
{
    if (program == NULL || program->value_count == 0u ||
        program->root_value >= program->value_count) {
        return -1;
    }

    unsigned char marked[OPERATOR_PROGRAM_MAX_VALUES] = {0};
    uint16_t remap[OPERATOR_PROGRAM_MAX_VALUES];
    operator_value compacted[OPERATOR_PROGRAM_MAX_VALUES];
    for (size_t i = 0u; i < OPERATOR_PROGRAM_MAX_VALUES; ++i) {
        remap[i] = OPERATOR_VALUE_INVALID;
    }

    lowering_program_mark_reachable(program, (uint16_t)program->root_value, marked);

    size_t compacted_count = 0u;
    for (size_t i = 0u; i < program->value_count; ++i) {
        if (marked[i] == 0u) {
            continue;
        }
        if (compacted_count >= OPERATOR_PROGRAM_MAX_VALUES) {
            return -1;
        }
        remap[i] = (uint16_t)compacted_count;
        compacted[compacted_count++] = program->values[i];
    }

    if (compacted_count == 0u) {
        return -1;
    }

    for (size_t i = 0u; i < compacted_count; ++i) {
        operator_value* value = &compacted[i];
        if (value->left != OPERATOR_VALUE_INVALID) {
            if (value->left >= program->value_count ||
                remap[value->left] == OPERATOR_VALUE_INVALID) {
                return -1;
            }
            value->left = remap[value->left];
        }
        if (value->right != OPERATOR_VALUE_INVALID) {
            if (value->right >= program->value_count ||
                remap[value->right] == OPERATOR_VALUE_INVALID) {
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

static int lowering_program_lower_values(operator_program* program)
{
    if (program == NULL) {
        return -1;
    }

    program->value_count = 0u;
    program->root_value = 0u;
    program->active_symbol_mask = OPERATOR_SYMBOL_MASK_NONE;
    program->lowered_hash = 0u;
    program->jit_eligible = 0;

    if (!program->active) {
        return 0;
    }

    uint16_t stack[OPERATOR_PROGRAM_MAX_STACK_SLOTS] = {0};
    size_t depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const operator_instruction instruction = program->instructions[i];
        uint16_t value_id = OPERATOR_VALUE_INVALID;

        if (instruction.opcode == OPERATOR_OP_PUSH_LITERAL) {
            if (depth >= OPERATOR_PROGRAM_MAX_STACK_SLOTS ||
                lowering_program_intern_value(program,
                                         OPERATOR_OP_PUSH_LITERAL,
                                         OPERATOR_VALUE_INVALID,
                                         OPERATOR_VALUE_INVALID,
                                         instruction.literal,
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth++] = value_id;
            continue;
        }

        if (instruction.opcode == OPERATOR_OP_PUSH_IMAG_UNIT) {
            if (depth >= OPERATOR_PROGRAM_MAX_STACK_SLOTS ||
                lowering_program_intern_value(program,
                                         OPERATOR_OP_PUSH_LITERAL,
                                         OPERATOR_VALUE_INVALID,
                                         OPERATOR_VALUE_INVALID,
                                         make(0.0, 1.0),
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth++] = value_id;
            continue;
        }

        if (lowering_opcode_is_symbol_leaf(instruction.opcode)) {
            if (depth >= OPERATOR_PROGRAM_MAX_STACK_SLOTS ||
                lowering_program_intern_value(program,
                                         instruction.opcode,
                                         OPERATOR_VALUE_INVALID,
                                         OPERATOR_VALUE_INVALID,
                                         make(0.0, 0.0),
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth++] = value_id;
            continue;
        }

        if (lowering_opcode_is_unary(instruction.opcode)) {
            if (depth == 0u ||
                lowering_program_intern_value(program,
                                         instruction.opcode,
                                         stack[depth - 1u],
                                         OPERATOR_VALUE_INVALID,
                                         instruction.literal,
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth - 1u] = value_id;
            continue;
        }

        if (lowering_opcode_is_binary(instruction.opcode)) {
            if (depth < 2u ||
                lowering_program_intern_value(program,
                                         instruction.opcode,
                                         stack[depth - 2u],
                                         stack[depth - 1u],
                                         make(0.0, 0.0),
                                         &value_id) != 0) {
                return -1;
            }
            stack[depth - 2u] = value_id;
            depth -= 1u;
            continue;
        }

        return -1;
    }

    if (depth != 1u || stack[0] == OPERATOR_VALUE_INVALID || stack[0] >= program->value_count) {
        return -1;
    }

    program->root_value = (size_t)stack[0];
    if (lowering_program_compact_values(program) != 0) {
        return -1;
    }
    program->active_symbol_mask = program->values[program->root_value].symbol_mask;
    program->lowered_hash = lowering_program_compute_lowered_hash(program);
    program->jit_eligible = 1;
    return 0;
}

static int is_identifier_start(char c)
{
    return (c == '_') || isalpha((unsigned char)c);
}

static int is_identifier_char(char c)
{
    return (c == '_') || isalnum((unsigned char)c);
}

static int context_allows_symbol_w(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_DISPERSION_FACTOR ||
            context == OPERATOR_CONTEXT_DISPERSION);
}

static int context_allows_symbol_wt(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == OPERATOR_CONTEXT_LINEAR ||
            context == OPERATOR_CONTEXT_DISPERSION_FACTOR ||
            context == OPERATOR_CONTEXT_DISPERSION);
}

static int context_allows_symbol_kx(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == OPERATOR_CONTEXT_LINEAR);
}

static int context_allows_symbol_ky(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == OPERATOR_CONTEXT_LINEAR);
}

static int context_allows_symbol_t(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == OPERATOR_CONTEXT_LINEAR ||
            context == OPERATOR_CONTEXT_NONLINEAR ||
            context == OPERATOR_CONTEXT_POTENTIAL);
}

static int context_allows_symbol_x(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == OPERATOR_CONTEXT_LINEAR ||
            context == OPERATOR_CONTEXT_NONLINEAR ||
            context == OPERATOR_CONTEXT_POTENTIAL);
}

static int context_allows_symbol_y(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_LINEAR_FACTOR ||
            context == OPERATOR_CONTEXT_LINEAR ||
            context == OPERATOR_CONTEXT_NONLINEAR ||
            context == OPERATOR_CONTEXT_POTENTIAL);
}

static int context_allows_symbol_a(operator_program_context context)
{
    (void)context;
    return 1;
}

static int context_allows_symbol_i(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_NONLINEAR);
}

static int context_allows_symbol_d(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_DISPERSION ||
            context == OPERATOR_CONTEXT_LINEAR);
}

static int context_allows_symbol_v(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_DISPERSION ||
            context == OPERATOR_CONTEXT_NONLINEAR ||
            context == OPERATOR_CONTEXT_LINEAR);
}

static int context_allows_symbol_h(operator_program_context context)
{
    return (context == OPERATOR_CONTEXT_DISPERSION ||
            context == OPERATOR_CONTEXT_LINEAR);
}

static int emit_instruction(
    operator_program* program,
    operator_opcode opcode,
    nlo_complex literal
)
{
    if (program == NULL || program->instruction_count >= OPERATOR_PROGRAM_MAX_INSTRUCTIONS) {
        return -1;
    }

    program->instructions[program->instruction_count++] = (operator_instruction){
        .opcode = opcode,
        .literal = literal
    };
    return 0;
}

static int parser_next_token(parser* parser)
{
    if (parser == NULL || parser->cursor == NULL) {
        return -1;
    }

    while (*parser->cursor != '\0' && isspace((unsigned char)*parser->cursor)) {
        ++parser->cursor;
    }

    const char c = *parser->cursor;
    if (c == '\0') {
        parser->current = (token){.kind = TOKEN_END};
        return 0;
    }

    if (c == '+') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_PLUS};
        return 0;
    }
    if (c == '-') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_MINUS};
        return 0;
    }
    if (c == '*') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_STAR};
        return 0;
    }
    if (c == '/') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_SLASH};
        return 0;
    }
    if (c == '^') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_CARET};
        return 0;
    }
    if (c == '(') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_LPAREN};
        return 0;
    }
    if (c == ')') {
        ++parser->cursor;
        parser->current = (token){.kind = TOKEN_RPAREN};
        return 0;
    }

    if (isdigit((unsigned char)c) || c == '.') {
        char* end_ptr = NULL;
        const double value = strtod(parser->cursor, &end_ptr);
        if (end_ptr == parser->cursor) {
            return -1;
        }

        parser->cursor = end_ptr;
        parser->current = (token){
            .kind = TOKEN_NUMBER,
            .number = value
        };
        return 0;
    }

    if (is_identifier_start(c)) {
        size_t len = 0u;
        while (is_identifier_char(parser->cursor[len])) {
            if (len + 1u >= sizeof(parser->current.ident)) {
                return -1;
            }
            ++len;
        }

        memset(parser->current.ident, 0, sizeof(parser->current.ident));
        memcpy(parser->current.ident, parser->cursor, len);
        parser->cursor += len;
        parser->current.kind = TOKEN_IDENT;
        return 0;
    }

    return -1;
}

static int parse_expression(parser* parser);
static int parse_term(parser* parser);
static int parse_power(parser* parser);
static int parse_unary(parser* parser);
static int parse_primary(parser* parser);

static int parse_function_call(
    parser* parser,
    operator_opcode opcode
)
{
    if (parser == NULL || parser->program == NULL) {
        return -1;
    }

    if (parser_next_token(parser) != 0 || parser->current.kind != TOKEN_LPAREN) {
        return -1;
    }
    if (parser_next_token(parser) != 0) {
        return -1;
    }
    if (parse_expression(parser) != 0 || parser->current.kind != TOKEN_RPAREN) {
        return -1;
    }
    if (emit_instruction(parser->program, opcode, make(0.0, 0.0)) != 0) {
        return -1;
    }

    return parser_next_token(parser);
}

static int parse_primary(parser* parser)
{
    if (parser == NULL || parser->program == NULL) {
        return -1;
    }

    if (parser->current.kind == TOKEN_NUMBER) {
        const nlo_complex literal = make(parser->current.number, 0.0);
        if (emit_instruction(parser->program,
                                 OPERATOR_OP_PUSH_LITERAL,
                                 literal) != 0) {
            return -1;
        }

        return parser_next_token(parser);
    }

    if (parser->current.kind == TOKEN_IDENT) {
        if (strcmp(parser->current.ident, "exp") == 0) {
            return parse_function_call(parser, OPERATOR_OP_EXP);
        }

        if (strcmp(parser->current.ident, "log") == 0) {
            return parse_function_call(parser, OPERATOR_OP_LOG);
        }

        if (strcmp(parser->current.ident, "sqrt") == 0) {
            return parse_function_call(parser, OPERATOR_OP_SQRT);
        }

        if (strcmp(parser->current.ident, "sin") == 0) {
            return parse_function_call(parser, OPERATOR_OP_SIN);
        }

        if (strcmp(parser->current.ident, "cos") == 0) {
            return parse_function_call(parser, OPERATOR_OP_COS);
        }

        if (strcmp(parser->current.ident, "w") == 0) {
            if (context_allows_symbol_w(parser->context)) {
                if (emit_instruction(parser->program,
                                         OPERATOR_OP_PUSH_SYMBOL_W,
                                         make(0.0, 0.0)) != 0) {
                    return -1;
                }
                return parser_next_token(parser);
            }
            if (context_allows_symbol_wt(parser->context)) {
                if (emit_instruction(parser->program,
                                         OPERATOR_OP_PUSH_SYMBOL_WT,
                                         make(0.0, 0.0)) != 0) {
                    return -1;
                }
                return parser_next_token(parser);
            }
            return -1;
        }

        if (strcmp(parser->current.ident, "wt") == 0) {
            if (!context_allows_symbol_wt(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_WT,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "kx") == 0) {
            if (!context_allows_symbol_kx(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_KX,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "ky") == 0) {
            if (!context_allows_symbol_ky(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_KY,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "t") == 0) {
            if (!context_allows_symbol_t(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_T,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "x") == 0) {
            if (!context_allows_symbol_x(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_X,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "y") == 0) {
            if (!context_allows_symbol_y(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_Y,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "A") == 0) {
            if (!context_allows_symbol_a(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_A,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "I") == 0) {
            if (!context_allows_symbol_i(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_I,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "D") == 0) {
            if (!context_allows_symbol_d(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_D,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "V") == 0) {
            if (!context_allows_symbol_v(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_V,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "h") == 0) {
            if (!context_allows_symbol_h(parser->context)) {
                return -1;
            }
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_SYMBOL_H,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "i") == 0) {
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_IMAG_UNIT,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
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

            if (emit_instruction(parser->program,
                                     OPERATOR_OP_PUSH_LITERAL,
                                     make(parser->constants[constant_index], 0.0)) != 0) {
                return -1;
            }
            return parser_next_token(parser);
        }

        return -1;
    }

    if (parser->current.kind == TOKEN_LPAREN) {
        if (parser_next_token(parser) != 0) {
            return -1;
        }
        if (parse_expression(parser) != 0 || parser->current.kind != TOKEN_RPAREN) {
            return -1;
        }
        return parser_next_token(parser);
    }

    return -1;
}

static int parse_unary(parser* parser)
{
    if (parser == NULL || parser->program == NULL) {
        return -1;
    }

    if (parser->current.kind == TOKEN_MINUS) {
        if (parser_next_token(parser) != 0) {
            return -1;
        }
        if (parse_unary(parser) != 0) {
            return -1;
        }
        return emit_instruction(parser->program,
                                    OPERATOR_OP_NEGATE,
                                    make(0.0, 0.0));
    }

    return parse_primary(parser);
}

static int parse_power(parser* parser)
{
    if (parse_unary(parser) != 0) {
        return -1;
    }

    if (parser->current.kind == TOKEN_CARET) {
        if (parser_next_token(parser) != 0 || parse_power(parser) != 0) {
            return -1;
        }
        if (emit_instruction(parser->program,
                                 OPERATOR_OP_POW,
                                 make(0.0, 0.0)) != 0) {
            return -1;
        }
    }

    return 0;
}

static int parse_term(parser* parser)
{
    if (parse_power(parser) != 0) {
        return -1;
    }

    while (parser->current.kind == TOKEN_STAR || parser->current.kind == TOKEN_SLASH) {
        const token_kind op = parser->current.kind;
        if (parser_next_token(parser) != 0 || parse_power(parser) != 0) {
            return -1;
        }
        if (emit_instruction(parser->program,
                                 (op == TOKEN_STAR)
                                     ? OPERATOR_OP_MUL
                                     : OPERATOR_OP_DIV,
                                 make(0.0, 0.0)) != 0) {
            return -1;
        }
    }

    return 0;
}

static int parse_expression(parser* parser)
{
    if (parse_term(parser) != 0) {
        return -1;
    }

    while (parser->current.kind == TOKEN_PLUS || parser->current.kind == TOKEN_MINUS) {
        const token_kind op = parser->current.kind;
        if (parser_next_token(parser) != 0 || parse_term(parser) != 0) {
            return -1;
        }

        if (op == TOKEN_MINUS) {
            if (emit_instruction(parser->program,
                                     OPERATOR_OP_NEGATE,
                                     make(0.0, 0.0)) != 0) {
                return -1;
            }
        }

        if (emit_instruction(parser->program,
                                 OPERATOR_OP_ADD,
                                 make(0.0, 0.0)) != 0) {
            return -1;
        }
    }

    return 0;
}

static int program_compute_stack_requirements(operator_program* program)
{
    if (program == NULL) {
        return -1;
    }

    size_t depth = 0u;
    size_t max_depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const operator_opcode op = program->instructions[i].opcode;
        if (op == OPERATOR_OP_PUSH_LITERAL ||
            op == OPERATOR_OP_PUSH_SYMBOL_W ||
            op == OPERATOR_OP_PUSH_SYMBOL_WT ||
            op == OPERATOR_OP_PUSH_SYMBOL_KX ||
            op == OPERATOR_OP_PUSH_SYMBOL_KY ||
            op == OPERATOR_OP_PUSH_SYMBOL_T ||
            op == OPERATOR_OP_PUSH_SYMBOL_X ||
            op == OPERATOR_OP_PUSH_SYMBOL_Y ||
            op == OPERATOR_OP_PUSH_SYMBOL_A ||
            op == OPERATOR_OP_PUSH_SYMBOL_I ||
            op == OPERATOR_OP_PUSH_SYMBOL_D ||
            op == OPERATOR_OP_PUSH_SYMBOL_V ||
            op == OPERATOR_OP_PUSH_SYMBOL_H ||
            op == OPERATOR_OP_PUSH_IMAG_UNIT) {
            ++depth;
            if (depth > max_depth) {
                max_depth = depth;
            }
            continue;
        }

        if (op == OPERATOR_OP_NEGATE ||
            op == OPERATOR_OP_EXP ||
            op == OPERATOR_OP_LOG ||
            op == OPERATOR_OP_SQRT ||
            op == OPERATOR_OP_SIN ||
            op == OPERATOR_OP_COS ||
            op == OPERATOR_OP_POW_REAL_LITERAL) {
            if (depth < 1u) {
                return -1;
            }
            continue;
        }

        if (op == OPERATOR_OP_ADD ||
            op == OPERATOR_OP_MUL ||
            op == OPERATOR_OP_DIV ||
            op == OPERATOR_OP_POW) {
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
    if (max_depth == 0u || max_depth > OPERATOR_PROGRAM_MAX_STACK_SLOTS) {
        return -1;
    }

    program->required_stack_slots = max_depth;
    return 0;
}

static void program_lower_real_literal_powers(operator_program* program)
{
    if (program == NULL || program->instruction_count == 0u) {
        return;
    }

    size_t write = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const operator_instruction instruction = program->instructions[i];
        if (instruction.opcode == OPERATOR_OP_POW &&
            write > 0u &&
            program->instructions[write - 1u].opcode == OPERATOR_OP_PUSH_LITERAL &&
            IM(program->instructions[write - 1u].literal) == 0.0) {
            const double exponent = RE(program->instructions[write - 1u].literal);
            write -= 1u;
            program->instructions[write++] = (operator_instruction){
                .opcode = OPERATOR_OP_POW_REAL_LITERAL,
                .literal = make(exponent, 0.0)
            };
            continue;
        }

        program->instructions[write++] = instruction;
    }

    program->instruction_count = write;
}

vec_status operator_program_compile(
    const char* expression,
    operator_program_context context,
    size_t num_constants,
    const double* constants,
    operator_program* out_program
)
{
    if (out_program == NULL) {
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    *out_program = (operator_program){0};
    out_program->context = context;
    if (expression == NULL || expression[0] == '\0') {
        return VEC_STATUS_OK;
    }

    parser parser = {
        .cursor = expression,
        .context = context,
        .num_constants = num_constants,
        .constants = constants,
        .program = out_program
    };

    if (parser_next_token(&parser) != 0 ||
        parse_expression(&parser) != 0 ||
        parser.current.kind != TOKEN_END) {
        *out_program = (operator_program){0};
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    program_lower_real_literal_powers(out_program);
    if (program_compute_stack_requirements(out_program) != 0) {
        *out_program = (operator_program){0};
        return VEC_STATUS_INVALID_ARGUMENT;
    }

    out_program->active = 1;
    if (lowering_program_lower_values(out_program) != 0) {
        /* Not representable as a value DAG: keep the interpreter program. */
        out_program->jit_eligible = 0;
        out_program->value_count = 0u;
        out_program->root_value = 0u;
        out_program->active_symbol_mask = OPERATOR_SYMBOL_MASK_NONE;
        out_program->lowered_hash = 0u;
    }
    return VEC_STATUS_OK;
}
