/**
 * @file operator_program_compile.c
 * @brief Runtime expression parser/compiler for operator programs.
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
            op == NLO_OPERATOR_OP_COS) {
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
