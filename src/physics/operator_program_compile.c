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
    return VEC_STATUS_OK;
}
