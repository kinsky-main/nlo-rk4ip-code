/**
 * @file test_operator_jit_vulkan.c
 * @dir tests/numerics
 * @brief Vulkan operator JIT against the CPU interpreter.
 *
 * Each expression is compiled once and run through the CPU interpreter, the
 * Vulkan interpreter (where it supports the expression) and the Vulkan JIT.
 * The JIT must match the Vulkan interpreter to 1e-12. Against the CPU the bound
 * is per case: GPU transcendentals come from double_math.glslinc, not libm, so
 * division, pow and sin/cos differ from the CPU by more. Also checks the process-wide
 * SPIR-V cache across backends and the NLO_OPERATOR_JIT=0 switch. Skips (exit
 * 0) when no Vulkan device is available.
 */

#include "backend/nlo_complex.h"
#include "backend/vector_backend.h"
#include "physics/operator_program.h"
#include "physics/operator_program_jit.h"

#include <math.h>

/* NLOLIB_VULKAN_FP32 computes on the device in single precision, so checks of
 * Vulkan results get a tolerance floor a few orders above float epsilon. */
#if defined(NLO_VK_FP32) && NLO_VK_FP32
#define VK_TOL(tol) fmax((tol), 1e-5)
#else
#define VK_TOL(tol) (tol)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Not assert(): Release builds define NDEBUG. */
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* Tensor grid used for every case; 1-D cases read only w/A/D/V. */
#define NT 16u
#define NX 5u
#define NY 3u
#define N_POINTS (NT * NX * NY)

enum { SYM_W, SYM_WT, SYM_KX, SYM_KY, SYM_T, SYM_X, SYM_Y, SYM_A, SYM_D, SYM_V, SYM_COUNT };

typedef struct {
    vector_backend* backend;
    vec_buffer* symbols[SYM_COUNT];
    vec_buffer* stack[OPERATOR_PROGRAM_MAX_STACK_SLOTS];
    vec_buffer* out;
} backend_buffers;

static nlo_complex g_host[SYM_COUNT][N_POINTS];

static void fill_host_symbols(void)
{
    for (size_t i = 0u; i < N_POINTS; ++i) {
        const size_t t = i % NT;
        const size_t plane = i / NT;
        const size_t y = plane % NY;
        const size_t x = plane / NY;
        const double ti = (double)t - 0.5 * (double)NT;
        g_host[SYM_W][i] = make(0.3 * ti, 0.0);
        g_host[SYM_WT][i] = make(0.25 * ti, 0.0);
        g_host[SYM_KX][i] = make(0.7 * ((double)x - 2.0), 0.0);
        g_host[SYM_KY][i] = make(0.9 * ((double)y - 1.0), 0.0);
        g_host[SYM_T][i] = make(0.05 * ti, 0.0);
        g_host[SYM_X][i] = make(0.2 * ((double)x - 2.0), 0.0);
        g_host[SYM_Y][i] = make(0.3 * ((double)y - 1.0), 0.0);
        g_host[SYM_A][i] = make(cos(0.3 * (double)i) + 0.2, 0.5 * sin(0.17 * (double)i));
        g_host[SYM_D][i] = make(0.0, -0.01 * ti * ti);
        g_host[SYM_V][i] = make(0.02 * (double)(x + y), 0.0);
    }
}

static void buffers_create(backend_buffers* b, vector_backend* backend)
{
    memset(b, 0, sizeof(*b));
    b->backend = backend;
    for (size_t s = 0u; s < SYM_COUNT; ++s) {
        CHECK(vec_create(backend, VEC_KIND_COMPLEX64, N_POINTS, &b->symbols[s]) == VEC_STATUS_OK);
        CHECK(vec_upload(backend, b->symbols[s], g_host[s], sizeof(g_host[s])) == VEC_STATUS_OK);
    }
    for (size_t s = 0u; s < OPERATOR_PROGRAM_MAX_STACK_SLOTS; ++s) {
        CHECK(vec_create(backend, VEC_KIND_COMPLEX64, N_POINTS, &b->stack[s]) == VEC_STATUS_OK);
    }
    CHECK(vec_create(backend, VEC_KIND_COMPLEX64, N_POINTS, &b->out) == VEC_STATUS_OK);
}

static void buffers_destroy(backend_buffers* b)
{
    for (size_t s = 0u; s < SYM_COUNT; ++s) {
        vec_destroy(b->backend, b->symbols[s]);
    }
    for (size_t s = 0u; s < OPERATOR_PROGRAM_MAX_STACK_SLOTS; ++s) {
        vec_destroy(b->backend, b->stack[s]);
    }
    vec_destroy(b->backend, b->out);
}

static operator_eval_context eval_context(const backend_buffers* b, int tensor)
{
    operator_eval_context ctx = {0};
    ctx.frequency_grid = tensor ? NULL : b->symbols[SYM_W];
    if (tensor) {
        ctx.wt_grid = b->symbols[SYM_WT];
        ctx.kx_grid = b->symbols[SYM_KX];
        ctx.ky_grid = b->symbols[SYM_KY];
        ctx.t_grid = b->symbols[SYM_T];
        ctx.x_grid = b->symbols[SYM_X];
        ctx.y_grid = b->symbols[SYM_Y];
    }
    ctx.field = b->symbols[SYM_A];
    ctx.dispersion_factor = b->symbols[SYM_D];
    ctx.potential = b->symbols[SYM_V];
    ctx.half_step_size = 0.0125;
    return ctx;
}

static vec_status run(const backend_buffers* b, const operator_program* program, int tensor,
                      nlo_complex* out)
{
    const operator_eval_context ctx = eval_context(b, tensor);
    const vec_status status = operator_program_execute(b->backend, program, &ctx, b->stack,
                                                       OPERATOR_PROGRAM_MAX_STACK_SLOTS, b->out);
    if (status == VEC_STATUS_OK) {
        CHECK(vec_download(b->backend, b->out, out, N_POINTS * sizeof(nlo_complex)) ==
              VEC_STATUS_OK);
    }
    return status;
}

static double max_rel_diff(const nlo_complex* a, const nlo_complex* ref)
{
    double scale = 0.0;
    double diff = 0.0;
    for (size_t i = 0u; i < N_POINTS; ++i) {
        scale = fmax(scale, hypot(RE(ref[i]), IM(ref[i])));
        diff = fmax(diff, hypot(RE(a[i]) - RE(ref[i]), IM(a[i]) - IM(ref[i])));
    }
    return diff / fmax(scale, 1e-300);
}

typedef struct {
    const char* expr;
    operator_program_context context;
    int tensor;
    double cpu_tolerance;
} jit_case;

static const jit_case k_cases[] = {
    {"i*c0*w*w", OPERATOR_CONTEXT_DISPERSION_FACTOR, 0, 1e-12},
    {"exp(h*D)", OPERATOR_CONTEXT_DISPERSION, 0, 1e-12},
    {"exp(h*D + V)", OPERATOR_CONTEXT_DISPERSION, 0, 1e-12},
    {"i*c2*I*A", OPERATOR_CONTEXT_NONLINEAR, 0, 1e-12},
    {"i*A*(c2*(I/(1.0 + I/c3)) + V)", OPERATOR_CONTEXT_NONLINEAR, 0, 1e-10},
    {"sqrt(I) - log(1.0 + I)", OPERATOR_CONTEXT_NONLINEAR, 0, 1e-9},
    {"A^c4 / (A + 2)", OPERATOR_CONTEXT_NONLINEAR, 0, 1e-6},
    {"sin(A)*cos(A) + A^2", OPERATOR_CONTEXT_NONLINEAR, 0, 1e-6},
    {"i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))", OPERATOR_CONTEXT_LINEAR_FACTOR, 1, 1e-12},
    {"exp(h*D)", OPERATOR_CONTEXT_LINEAR, 1, 1e-12},
    {"c0*(x*x + y*y) + t", OPERATOR_CONTEXT_POTENTIAL, 1, 1e-12},
    {"i*A*(c2*(I/(1.0 + I/c3)) + V)", OPERATOR_CONTEXT_NONLINEAR, 1, 1e-10},
};
#define N_CASES (sizeof(k_cases) / sizeof(k_cases[0]))

static void set_jit_env(const char* value)
{
#if defined(_WIN32)
    CHECK(_putenv_s("NLO_OPERATOR_JIT", value) == 0);
#else
    CHECK(setenv("NLO_OPERATOR_JIT", value, 1) == 0);
#endif
}

int main(void)
{
    vector_backend* vk = vector_backend_create_vulkan(NULL);
    if (vk == NULL) {
        printf("test_operator_jit_vulkan: no Vulkan device, skipped.\n");
        return 0;
    }
    vector_backend* cpu = vector_backend_create_cpu();
    CHECK(cpu != NULL);
    set_jit_env("1");

    fill_host_symbols();
    backend_buffers vb;
    backend_buffers cb;
    buffers_create(&vb, vk);
    buffers_create(&cb, cpu);

    const double constants[] = {-0.04, -0.08, 1.5, 4.0, 0.5};
    static operator_program programs[N_CASES];
    static nlo_complex ref[N_POINTS];
    static nlo_complex vk_ref[N_POINTS];
    static nlo_complex got[N_POINTS];
    for (size_t k = 0u; k < N_CASES; ++k) {
        operator_program* p = &programs[k];
        CHECK(operator_program_compile(k_cases[k].expr, k_cases[k].context, 5u, constants, p) ==
              VEC_STATUS_OK);
        CHECK(p->jit_eligible);

        CHECK(run(&cb, p, k_cases[k].tensor, ref) == VEC_STATUS_OK);
        /* main's Vulkan interpreter lacks complex log/sin/cos/pow. */
        const vec_status vk_status = run(&vb, p, k_cases[k].tensor, vk_ref);
        CHECK(vk_status == VEC_STATUS_OK || vk_status == VEC_STATUS_UNSUPPORTED);

        CHECK(operator_program_prepare_jit(vk, p) == VEC_STATUS_OK);
        CHECK(p->vk_jit_active && p->vk_jit_entry != NULL);
        CHECK(run(&vb, p, k_cases[k].tensor, got) == VEC_STATUS_OK);
        const double err_cpu = max_rel_diff(got, ref);
        const double err_vk = (vk_status == VEC_STATUS_OK) ? max_rel_diff(got, vk_ref) : -1.0;
        char vk_text[16] = "n/a";
        if (err_vk >= 0.0) {
            snprintf(vk_text, sizeof(vk_text), "%.2e", err_vk);
        }
        printf("  %-40s tensor=%d  JIT vs Vulkan interpreter: %-9s  vs CPU: %.2e\n",
               k_cases[k].expr, k_cases[k].tensor, vk_text, err_cpu);
        CHECK(err_vk <= VK_TOL(1e-12));
        CHECK(err_cpu <= VK_TOL(k_cases[k].cpu_tolerance));
    }

    /* A second backend reuses the cached SPIR-V: no new cache entries. */
    const size_t cached = operator_program_jit_cache_entry_count();
    CHECK(cached >= 1u && cached <= N_CASES);
    vector_backend* vk2 = vector_backend_create_vulkan(NULL);
    CHECK(vk2 != NULL);
    operator_program again;
    CHECK(operator_program_compile(k_cases[4].expr, k_cases[4].context, 5u, constants, &again) ==
          VEC_STATUS_OK);
    CHECK(operator_program_prepare_jit(vk2, &again) == VEC_STATUS_OK);
    CHECK(operator_program_jit_cache_entry_count() == cached);

    /* NLO_OPERATOR_JIT=0 leaves programs on the interpreter. */
    set_jit_env("0");
    operator_program off;
    CHECK(operator_program_compile(k_cases[3].expr, k_cases[3].context, 5u, constants, &off) ==
          VEC_STATUS_OK);
    CHECK(operator_program_prepare_jit(vk2, &off) == VEC_STATUS_UNSUPPORTED);
    CHECK(!off.vk_jit_active);
    set_jit_env("1");

    vector_backend_destroy(vk2);
    buffers_destroy(&vb);
    buffers_destroy(&cb);
    vector_backend_destroy(vk);
    vector_backend_destroy(cpu);
    printf("test_operator_jit_vulkan: %zu expressions match the interpreters.\n",
           (size_t)N_CASES);
    return 0;
}
