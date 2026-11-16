#include "tensor_scaling_plan.h"

#include <stdio.h>
#include <stdlib.h>

/* Not assert(): Release builds define NDEBUG. */
#define PLAN_CHECK(cond)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "%s:%d: PLAN_CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static void test_shape_mapping(void)
{
    bench_tensor_shape shape = {0};
    PLAN_CHECK(bench_tensor_shape_from_scale(32u, &shape) == 0);
    PLAN_CHECK(shape.scale == 32u);
    PLAN_CHECK(shape.nt == 64u);
    PLAN_CHECK(shape.nx == 32u);
    PLAN_CHECK(shape.ny == 32u);
    PLAN_CHECK(shape.total_samples == (size_t)65536u);
    printf("test_shape_mapping: 2:1:1 tensor scale mapping verified.\n");
}

static void test_region_classification(void)
{
    bench_tensor_region_inputs inputs = {
        .working_set_bytes = 256u,
        .host_budget_bytes = 1024u,
        .gpu_budget_bytes = 512u,
        .cpu_init_ok = 1,
        .gpu_init_ok = 1
    };

    PLAN_CHECK(bench_tensor_classify_fit_region(&inputs) == BENCH_TENSOR_REGION_GPU_FIT);

    inputs.gpu_budget_bytes = 128u;
    inputs.gpu_init_ok = 0;
    PLAN_CHECK(bench_tensor_classify_fit_region(&inputs) == BENCH_TENSOR_REGION_HOST_FIT_ONLY);

    inputs.working_set_bytes = 2048u;
    PLAN_CHECK(bench_tensor_classify_fit_region(&inputs) == BENCH_TENSOR_REGION_TOO_LARGE);

    inputs.cpu_init_ok = 0;
    inputs.working_set_bytes = 64u;
    PLAN_CHECK(bench_tensor_classify_fit_region(&inputs) == BENCH_TENSOR_REGION_TOO_LARGE);
    printf("test_region_classification: tensor fit regions classified as expected.\n");
}

static void test_output_record_count(void)
{
    PLAN_CHECK(bench_tensor_records_for_output_bytes(0u, 1024u) == 0u);
    PLAN_CHECK(bench_tensor_records_for_output_bytes(1024u, 1u) == 2u);
    PLAN_CHECK(bench_tensor_records_for_output_bytes(1024u, 2048u) == 2u);
    PLAN_CHECK(bench_tensor_records_for_output_bytes(1024u, 2500u) == 3u);
    printf("test_output_record_count: spill-region record planning verified.\n");
}

int main(void)
{
    test_shape_mapping();
    test_region_classification();
    test_output_record_count();
    return 0;
}
