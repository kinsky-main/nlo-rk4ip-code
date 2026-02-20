/**
 * @file operator_program.c
 * @brief Runtime expression parser and executor for operator programs.
 */

#include "physics/operator_program.h"
#include "backend/nlo_complex.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    NLO_TOKEN_END = 0,
    NLO_TOKEN_NUMBER = 1,
    NLO_TOKEN_IDENT = 2,
    NLO_TOKEN_PLUS = 3,
    NLO_TOKEN_MINUS = 4,
    NLO_TOKEN_STAR = 5,
    NLO_TOKEN_LPAREN = 6,
    NLO_TOKEN_RPAREN = 7
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

static int nlo_is_identifier_start(char c)
{
    return (c == '_') || isalpha((unsigned char)c);
}

static int nlo_is_identifier_char(char c)
{
    return (c == '_') || isalnum((unsigned char)c);
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
            if (nlo_parser_next_token(parser) != 0 || parser->current.kind != NLO_TOKEN_LPAREN) {
                return -1;
            }
            if (nlo_parser_next_token(parser) != 0) {
                return -1;
            }
            if (nlo_parse_expression(parser) != 0 || parser->current.kind != NLO_TOKEN_RPAREN) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_EXP,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "w") == 0) {
            if (parser->context != NLO_OPERATOR_CONTEXT_DISPERSION) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_W,
                                     nlo_make(0.0, 0.0)) != 0) {
                return -1;
            }
            return nlo_parser_next_token(parser);
        }

        if (strcmp(parser->current.ident, "A") == 0) {
            if (parser->context != NLO_OPERATOR_CONTEXT_NONLINEAR) {
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
            if (parser->context != NLO_OPERATOR_CONTEXT_NONLINEAR) {
                return -1;
            }
            if (nlo_emit_instruction(parser->program,
                                     NLO_OPERATOR_OP_PUSH_SYMBOL_I,
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

static int nlo_parse_term(nlo_parser* parser)
{
    if (nlo_parse_unary(parser) != 0) {
        return -1;
    }

    while (parser->current.kind == NLO_TOKEN_STAR) {
        if (nlo_parser_next_token(parser) != 0 || nlo_parse_unary(parser) != 0) {
            return -1;
        }
        if (nlo_emit_instruction(parser->program,
                                 NLO_OPERATOR_OP_MUL,
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
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_A ||
            op == NLO_OPERATOR_OP_PUSH_SYMBOL_I ||
            op == NLO_OPERATOR_OP_PUSH_IMAG_UNIT) {
            ++depth;
            if (depth > max_depth) {
                max_depth = depth;
            }
            continue;
        }

        if (op == NLO_OPERATOR_OP_NEGATE || op == NLO_OPERATOR_OP_EXP) {
            if (depth < 1u) {
                return -1;
            }
            continue;
        }

        if (op == NLO_OPERATOR_OP_ADD || op == NLO_OPERATOR_OP_MUL) {
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
        parser.current.kind != NLO_TOKEN_END ||
        nlo_program_compute_stack_requirements(out_program) != 0) {
        *out_program = (nlo_operator_program){0};
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    out_program->active = 1;
    return NLO_VEC_STATUS_OK;
}

nlo_vec_status nlo_operator_program_execute(
    nlo_vector_backend* backend,
    const nlo_operator_program* program,
    const nlo_operator_eval_context* eval_ctx,
    nlo_vec_buffer* const* stack_vectors,
    size_t stack_vector_count,
    nlo_vec_buffer* out_vector
)
{
    if (backend == NULL ||
        program == NULL ||
        eval_ctx == NULL ||
        stack_vectors == NULL ||
        out_vector == NULL ||
        !program->active ||
        program->required_stack_slots == 0u ||
        stack_vector_count < program->required_stack_slots) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    size_t stack_depth = 0u;
    for (size_t i = 0u; i < program->instruction_count; ++i) {
        const nlo_operator_instruction instruction = program->instructions[i];
        nlo_vec_status status = NLO_VEC_STATUS_OK;

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_LITERAL) {
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_fill(backend, dst, instruction.literal);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_W) {
            if (eval_ctx->frequency_grid == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->frequency_grid);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_A) {
            if (eval_ctx->field == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_copy(backend, dst, eval_ctx->field);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_SYMBOL_I) {
            if (eval_ctx->field == NULL) {
                return NLO_VEC_STATUS_INVALID_ARGUMENT;
            }
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_magnitude_squared(backend, eval_ctx->field, dst);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_PUSH_IMAG_UNIT) {
            nlo_vec_buffer* dst = stack_vectors[stack_depth];
            status = nlo_vec_complex_fill(backend, dst, nlo_make(0.0, 1.0));
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            ++stack_depth;
            continue;
        }

        if (stack_depth == 0u) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_NEGATE) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_scalar_mul_inplace(backend, top, nlo_make(-1.0, 0.0));
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_EXP) {
            nlo_vec_buffer* top = stack_vectors[stack_depth - 1u];
            status = nlo_vec_complex_exp_inplace(backend, top);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            continue;
        }

        if (stack_depth < 2u) {
            return NLO_VEC_STATUS_INVALID_ARGUMENT;
        }

        nlo_vec_buffer* left = stack_vectors[stack_depth - 2u];
        nlo_vec_buffer* right = stack_vectors[stack_depth - 1u];
        if (instruction.opcode == NLO_OPERATOR_OP_ADD) {
            status = nlo_vec_complex_add_inplace(backend, left, right);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        if (instruction.opcode == NLO_OPERATOR_OP_MUL) {
            status = nlo_vec_complex_mul_inplace(backend, left, right);
            if (status != NLO_VEC_STATUS_OK) {
                return status;
            }
            --stack_depth;
            continue;
        }

        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    if (stack_depth != 1u) {
        return NLO_VEC_STATUS_INVALID_ARGUMENT;
    }

    return nlo_vec_complex_copy(backend, out_vector, stack_vectors[0]);
}
