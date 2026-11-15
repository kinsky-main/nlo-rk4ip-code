/**
 * @file test_operator_lowering.c
 * @brief Checks the value-DAG lowering done by operator_program_compile.
 *
 * Every expression is evaluated two ways on host arrays: by replaying the
 * instruction stream with the interpreter's semantics, and by walking the
 * lowered value DAG. Both use the same scalar helpers, so results must match
 * exactly. Structural checks cover ordering, sharing, folding, symbol masks
 * and hash stability.
 */

#include "physics/operator_program.h"
#include "numerics/vector_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_POINTS 7u

/* Not assert(): Release builds define NDEBUG. */
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static nlo_complex g_symbols[OPERATOR_OP_POW_REAL_LITERAL + 1][N_POINTS];

static int opcode_is_symbol(operator_opcode op)
{
    return op == OPERATOR_OP_PUSH_SYMBOL_W || op == OPERATOR_OP_PUSH_SYMBOL_A ||
           op == OPERATOR_OP_PUSH_SYMBOL_I || op == OPERATOR_OP_PUSH_SYMBOL_D ||
           op == OPERATOR_OP_PUSH_SYMBOL_V || op == OPERATOR_OP_PUSH_SYMBOL_H ||
           op == OPERATOR_OP_PUSH_SYMBOL_WT || op == OPERATOR_OP_PUSH_SYMBOL_KX ||
           op == OPERATOR_OP_PUSH_SYMBOL_KY || op == OPERATOR_OP_PUSH_SYMBOL_T ||
           op == OPERATOR_OP_PUSH_SYMBOL_X || op == OPERATOR_OP_PUSH_SYMBOL_Y;
}

static void init_symbols(void)
{
    for (size_t s = 0u; s <= (size_t)OPERATOR_OP_POW_REAL_LITERAL; ++s) {
        for (size_t k = 0u; k < N_POINTS; ++k) {
            const double x = 0.1 + 0.37 * (double)k + 0.05 * (double)s;
            g_symbols[s][k] = make(x, 0.25 * cos(x) - 0.1);
        }
    }
    /* h is a real scalar in the solver. */
    for (size_t k = 0u; k < N_POINTS; ++k) {
        g_symbols[OPERATOR_OP_PUSH_SYMBOL_H][k] = make(0.015, 0.0);
    }
}

static void fill(nlo_complex* dst, nlo_complex v)
{
    for (size_t k = 0u; k < N_POINTS; ++k) {
        dst[k] = v;
    }
}

static void apply_unary(operator_opcode op, nlo_complex* top, nlo_complex literal)
{
    switch (op) {
        case OPERATOR_OP_NEGATE:
            for (size_t k = 0u; k < N_POINTS; ++k) {
                top[k] = make(-RE(top[k]), -IM(top[k]));
            }
            break;
        case OPERATOR_OP_EXP: complex_exp_inplace(top, N_POINTS); break;
        case OPERATOR_OP_LOG: complex_log_inplace(top, N_POINTS); break;
        case OPERATOR_OP_SQRT: complex_real_pow_inplace(top, N_POINTS, 0.5); break;
        case OPERATOR_OP_SIN: complex_sin_inplace(top, N_POINTS); break;
        case OPERATOR_OP_COS: complex_cos_inplace(top, N_POINTS); break;
        case OPERATOR_OP_POW_REAL_LITERAL:
            complex_real_pow_inplace(top, N_POINTS, RE(literal));
            break;
        default: CHECK(0 && "not a unary opcode");
    }
}

static void apply_binary(operator_opcode op, nlo_complex* left, nlo_complex* right)
{
    switch (op) {
        case OPERATOR_OP_ADD: complex_add_inplace(left, right, N_POINTS); break;
        case OPERATOR_OP_MUL: complex_mul_inplace(left, right, N_POINTS); break;
        case OPERATOR_OP_DIV:
            complex_real_pow_inplace(right, N_POINTS, -1.0);
            complex_mul_inplace(left, right, N_POINTS);
            break;
        case OPERATOR_OP_POW: complex_pow_elementwise_inplace(left, right, N_POINTS); break;
        default: CHECK(0 && "not a binary opcode");
    }
}

/* Mirrors operator_program_execute on host arrays. */
static void eval_instructions(const operator_program* p, nlo_complex* out)
{
    static nlo_complex stack[OPERATOR_PROGRAM_MAX_STACK_SLOTS][N_POINTS];
    size_t depth = 0u;
    for (size_t i = 0u; i < p->instruction_count; ++i) {
        const operator_instruction ins = p->instructions[i];
        if (ins.opcode == OPERATOR_OP_PUSH_LITERAL) {
            fill(stack[depth++], ins.literal);
        } else if (ins.opcode == OPERATOR_OP_PUSH_IMAG_UNIT) {
            fill(stack[depth++], make(0.0, 1.0));
        } else if (opcode_is_symbol(ins.opcode)) {
            memcpy(stack[depth++], g_symbols[ins.opcode], sizeof(stack[0]));
        } else if (ins.opcode == OPERATOR_OP_ADD || ins.opcode == OPERATOR_OP_MUL ||
                   ins.opcode == OPERATOR_OP_DIV || ins.opcode == OPERATOR_OP_POW) {
            apply_binary(ins.opcode, stack[depth - 2u], stack[depth - 1u]);
            --depth;
        } else {
            apply_unary(ins.opcode, stack[depth - 1u], ins.literal);
        }
    }
    CHECK(depth == 1u);
    memcpy(out, stack[0], sizeof(stack[0]));
}

/* Walks the lowered DAG in array order. */
static void eval_values(const operator_program* p, nlo_complex* out)
{
    static nlo_complex vals[OPERATOR_PROGRAM_MAX_VALUES][N_POINTS];
    static nlo_complex rhs[N_POINTS];
    for (size_t i = 0u; i < p->value_count; ++i) {
        const operator_value* v = &p->values[i];
        if (v->opcode == OPERATOR_OP_PUSH_LITERAL) {
            fill(vals[i], v->literal);
        } else if (opcode_is_symbol(v->opcode)) {
            memcpy(vals[i], g_symbols[v->opcode], sizeof(vals[0]));
        } else if (v->right != OPERATOR_VALUE_INVALID) {
            memcpy(vals[i], vals[v->left], sizeof(vals[0]));
            memcpy(rhs, vals[v->right], sizeof(rhs));
            apply_binary(v->opcode, vals[i], rhs);
        } else {
            memcpy(vals[i], vals[v->left], sizeof(vals[0]));
            apply_unary(v->opcode, vals[i], v->literal);
        }
    }
    memcpy(out, vals[p->root_value], sizeof(vals[0]));
}

static operator_program compile_ok(const char* expr, operator_program_context ctx,
                                   size_t nc, const double* c)
{
    static operator_program p;
    const vec_status status = operator_program_compile(expr, ctx, nc, c, &p);
    if (status != VEC_STATUS_OK) {
        fprintf(stderr, "compile failed: %s\n", expr);
    }
    CHECK(status == VEC_STATUS_OK);
    CHECK(p.active);
    return p;
}

static void check_structure(const operator_program* p)
{
    CHECK(p->jit_eligible == 1);
    CHECK(p->value_count > 0u && p->value_count <= OPERATOR_PROGRAM_MAX_VALUES);
    CHECK(p->root_value == p->value_count - 1u);
    for (size_t i = 0u; i < p->value_count; ++i) {
        const operator_value* v = &p->values[i];
        CHECK(v->left == OPERATOR_VALUE_INVALID || v->left < i);
        CHECK(v->right == OPERATOR_VALUE_INVALID || v->right < i);
        uint32_t mask = 0u;
        if (v->left != OPERATOR_VALUE_INVALID) {
            mask |= p->values[v->left].symbol_mask;
        }
        if (v->right != OPERATOR_VALUE_INVALID) {
            mask |= p->values[v->right].symbol_mask;
        }
        if (!opcode_is_symbol(v->opcode)) {
            CHECK(v->symbol_mask == mask);
        }
        /* No duplicate nodes after interning. */
        for (size_t j = 0u; j < i; ++j) {
            const operator_value* w = &p->values[j];
            CHECK(!(w->opcode == v->opcode && w->left == v->left && w->right == v->right &&
                     RE(w->literal) == RE(v->literal) && IM(w->literal) == IM(v->literal)));
        }
    }
    CHECK(p->active_symbol_mask == p->values[p->root_value].symbol_mask);
}

static void check_equivalent(const char* expr, operator_program_context ctx,
                             size_t nc, const double* c)
{
    const operator_program p = compile_ok(expr, ctx, nc, c);
    check_structure(&p);
    nlo_complex a[N_POINTS];
    nlo_complex b[N_POINTS];
    eval_instructions(&p, a);
    eval_values(&p, b);
    for (size_t k = 0u; k < N_POINTS; ++k) {
        const double scale = fmax(1.0, hypot(RE(a[k]), IM(a[k])));
        const double err = hypot(RE(a[k]) - RE(b[k]), IM(a[k]) - IM(b[k])) / scale;
        if (!(err <= 1e-15)) {
            fprintf(stderr, "mismatch for %s at %zu: err=%g\n", expr, k, err);
        }
        CHECK(err <= 1e-15);
    }
}

static void test_example_expressions(void)
{
    const double c[] = {-0.04, -0.08, 1.5, 4.0, 0.5};
    check_equivalent("i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))", OPERATOR_CONTEXT_LINEAR_FACTOR, 5u, c);
    check_equivalent("exp(h*D)", OPERATOR_CONTEXT_LINEAR, 5u, c);
    check_equivalent("i*A*(c2*(I/(1.0 + I/c3)) + V)", OPERATOR_CONTEXT_NONLINEAR, 5u, c);
    check_equivalent("i*c0*w*w", OPERATOR_CONTEXT_DISPERSION_FACTOR, 5u, c);
    check_equivalent("exp(h*D + V)", OPERATOR_CONTEXT_DISPERSION, 5u, c);
    check_equivalent("i*c2*I*A", OPERATOR_CONTEXT_NONLINEAR, 5u, c);
    check_equivalent("sqrt(I) - log(1.0 + I)", OPERATOR_CONTEXT_NONLINEAR, 5u, c);
    check_equivalent("sin(t)*cos(x) + y^2", OPERATOR_CONTEXT_POTENTIAL, 5u, c);
    check_equivalent("A^c4 / (A + 2)", OPERATOR_CONTEXT_NONLINEAR, 5u, c);
    check_equivalent("-(A*A) + A*A*A", OPERATOR_CONTEXT_NONLINEAR, 5u, c);
}

static void test_symbol_masks(void)
{
    const double c[] = {1.0, 2.0};
    operator_program p = compile_ok("i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))",
                                    OPERATOR_CONTEXT_LINEAR_FACTOR, 2u, c);
    CHECK(p.active_symbol_mask ==
           (OPERATOR_SYMBOL_MASK_WT | OPERATOR_SYMBOL_MASK_KX | OPERATOR_SYMBOL_MASK_KY));
    p = compile_ok("i*A*I", OPERATOR_CONTEXT_NONLINEAR, 0u, NULL);
    CHECK(p.active_symbol_mask == (OPERATOR_SYMBOL_MASK_A | OPERATOR_SYMBOL_MASK_I));
}

static void test_sharing_and_folding(void)
{
    /* A, A*A, A*A + A*A: the repeated product is one node. */
    operator_program p = compile_ok("A*A + A*A", OPERATOR_CONTEXT_NONLINEAR, 0u, NULL);
    check_structure(&p);
    CHECK(p.value_count == 3u);

    /* 2*3 folds to the literal 6; the result is 6 + A. */
    p = compile_ok("2.0*3.0 + A", OPERATOR_CONTEXT_NONLINEAR, 0u, NULL);
    check_structure(&p);
    CHECK(p.value_count == 3u);
    CHECK(p.values[0].opcode == OPERATOR_OP_PUSH_LITERAL);
    CHECK(RE(p.values[0].literal) == 6.0 && IM(p.values[0].literal) == 0.0);

    /* A constant-only expression lowers to a single literal. */
    p = compile_ok("exp(0.0) + 1.0", OPERATOR_CONTEXT_NONLINEAR, 0u, NULL);
    check_structure(&p);
    CHECK(p.value_count == 1u);
    CHECK(p.active_symbol_mask == OPERATOR_SYMBOL_MASK_NONE);
    CHECK(RE(p.values[0].literal) == 2.0);
}

static void test_hash(void)
{
    const double c1[] = {0.5};
    const double c2[] = {0.25};
    const uint64_t h1 = compile_ok("i*c0*A*I", OPERATOR_CONTEXT_NONLINEAR, 1u, c1).lowered_hash;
    const uint64_t h1b = compile_ok("i*c0*A*I", OPERATOR_CONTEXT_NONLINEAR, 1u, c1).lowered_hash;
    const uint64_t h2 = compile_ok("i*c0*A*I", OPERATOR_CONTEXT_NONLINEAR, 1u, c2).lowered_hash;
    const uint64_t h3 = compile_ok("i*c0*A*A", OPERATOR_CONTEXT_NONLINEAR, 1u, c1).lowered_hash;
    CHECK(h1 != 0u);
    CHECK(h1 == h1b);
    CHECK(h1 != h2);
    CHECK(h1 != h3);
}

int main(void)
{
    init_symbols();
    test_example_expressions();
    test_symbol_masks();
    test_sharing_and_folding();
    test_hash();
    printf("test_core_operator_lowering: all checks passed\n");
    return 0;
}
