"""The grid-dependent runtime limits must be non-zero for a prepared config.

`nlolib_query_runtime_limits` derives `estimated_required_working_set_bytes` and
`max_num_recorded_samples_in_memory` from a config that states its own sample
count. A config prepared for propagation leaves `time.nt` at zero, because the
solver is told the sample count as an argument to propagate and infers the
layout from it, so both fields came back zero for every binding caller -- which
made the documented `min(wanted, max_num_recorded_samples_in_memory)` recipe
yield zero records, and propagate then rejects zero.

The wrapper now stamps the count onto a copy of the config for this call. The
copy matters: propagate and the snapshot store both hash the caller's config, so
it must be left exactly as prepared.
"""

import math

import nlolib

N = 512
DT = 0.002


def _config(num_time_samples=N):
    two_pi = 2.0 * math.pi
    step = two_pi / (float(num_time_samples) * DT)
    grid = [
        complex(i * step if i <= num_time_samples // 2
                else -(num_time_samples - i) * step, 0.0)
        for i in range(num_time_samples)
    ]
    return nlolib.prepare_sim_config(
        num_time_samples,
        propagation_distance=0.05,
        starting_step_size=1e-3,
        max_step_size=1e-3,
        min_step_size=1e-3,
        error_tolerance=1e-10,
        pulse_period=float(num_time_samples) * DT,
        delta_time=DT,
        frequency_grid=grid,
        runtime=nlolib.RuntimeOperators(
            dispersion_factor_expr="i*c0*w*w",
            nonlinear_expr="i*c1*A*I",
            constants=[-0.01, 1.0],
        ),
    )


def test_grid_dependent_limits_are_populated() -> None:
    api = nlolib.NLolib()
    cfg = _config()
    limits = api.query_runtime_limits(cfg, nlolib.default_execution_options(
        nlolib.NLO_VECTOR_BACKEND_CPU))

    working_set = int(limits.estimated_required_working_set_bytes)
    in_memory = int(limits.max_num_recorded_samples_in_memory)

    assert working_set > 0, (
        "estimated_required_working_set_bytes is 0; the config's sample count "
        "did not reach nlolib_query_runtime_limits"
    )
    assert in_memory > 0, (
        "max_num_recorded_samples_in_memory is 0, so the documented "
        "min(wanted, max_num_recorded_samples_in_memory) recipe yields 0"
    )
    # One working vector per sample is the floor; the solver holds several.
    assert working_set >= N * 16, f"working set {working_set} is below one vector"

    # It must say something: equal to the storage cap would mean the field
    # carries no information, which is what it used to do.
    with_storage = int(limits.max_num_recorded_samples_with_storage)
    assert in_memory < with_storage, (
        f"in-memory capacity {in_memory} equals the storage cap {with_storage}, "
        "so it is not derived from memory at all"
    )


def test_in_memory_capacity_shrinks_as_records_grow() -> None:
    """Capacity is host memory over record size, so a 4x grid holds ~4x fewer."""
    api = nlolib.NLolib()
    opts = nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_CPU)
    small = int(api.query_runtime_limits(_config(N), opts)
                .max_num_recorded_samples_in_memory)
    large = int(api.query_runtime_limits(_config(16 * N), opts)
                .max_num_recorded_samples_in_memory)
    assert 0 < large < small, f"capacity did not shrink with record size: {small} -> {large}"
    ratio = small / large
    assert 8.0 <= ratio <= 32.0, f"expected roughly 16x fewer records, got {ratio:.1f}x"


def test_limits_scale_with_the_grid() -> None:
    """A field pinned to the grid, not a constant that happens to be non-zero."""
    api = nlolib.NLolib()
    opts = nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_CPU)
    small = int(api.query_runtime_limits(_config(N), opts)
                .estimated_required_working_set_bytes)
    large = int(api.query_runtime_limits(_config(4 * N), opts)
                .estimated_required_working_set_bytes)
    assert large == 4 * small, f"working set did not scale with nt: {small} -> {large}"


def test_query_does_not_mutate_the_callers_config() -> None:
    """The stamp goes on a copy: propagate and the snapshot store hash this struct."""
    api = nlolib.NLolib()
    cfg = _config()
    assert cfg.simulation_config.time.nt == 0
    api.query_runtime_limits(cfg, nlolib.default_execution_options(
        nlolib.NLO_VECTOR_BACKEND_CPU))
    assert cfg.simulation_config.time.nt == 0, (
        "query_runtime_limits wrote time.nt into the caller's config"
    )


def test_propagate_still_works_after_a_limits_query() -> None:
    api = nlolib.NLolib()
    cfg = _config()
    opts = nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_CPU)
    limits = api.query_runtime_limits(cfg, opts)
    records = min(4, int(limits.max_num_recorded_samples_in_memory))
    assert records > 0

    field = [complex(math.exp(-(((i - N / 2) * DT) / 0.05) ** 2), 0.0) for i in range(N)]
    result = api.propagate(cfg, field, records, opts)
    assert len(result.records) == records


def test_storage_ceiling_is_not_a_memory_figure() -> None:
    """The streaming ceiling must not move with the grid, or with host memory.

    max_num_recorded_samples_in_memory describes in-memory recording, where
    every record stays resident. A run streaming to the snapshot store keeps
    only a ring window resident, so its ceiling is
    max_num_recorded_samples_with_storage -- a precision ceiling. If that one
    ever started tracking memory, a streaming run would be told it can record
    far less than it can.
    """
    api = nlolib.NLolib()
    opts = nlolib.default_execution_options(nlolib.NLO_VECTOR_BACKEND_CPU)
    caps = [
        int(api.query_runtime_limits(_config(n), opts)
            .max_num_recorded_samples_with_storage)
        for n in (N, 16 * N)
    ]
    assert caps[0] == caps[1], f"storage ceiling moved with the grid: {caps}"

    in_memory = int(api.query_runtime_limits(_config(N), opts)
                    .max_num_recorded_samples_in_memory)
    assert caps[0] > in_memory, (
        "the storage ceiling is no higher than the in-memory capacity, so "
        "streaming would be constrained by host memory it does not use"
    )


def main() -> None:
    test_grid_dependent_limits_are_populated()
    test_in_memory_capacity_shrinks_as_records_grow()
    test_limits_scale_with_the_grid()
    test_storage_ceiling_is_not_a_memory_figure()
    test_query_does_not_mutate_the_callers_config()
    test_propagate_still_works_after_a_limits_query()
    print("test_python_runtime_limits_grid: grid-dependent runtime limits populated.")


if __name__ == "__main__":
    main()
