from nlolib_ctypes import (
    NLO_VECTOR_BACKEND_CPU,
    NLolib,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)


def _base_config(n: int):
    return prepare_sim_config(
        n,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * n,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )


def main():
    api = NLolib()
    print("test_python_bindings: loaded nlolib ctypes bindings.")

    n = 128
    cfg = _base_config(n)
    input_field = [0j] * n

    try:
        out = api.propagate(cfg, input_field, 1, None)
        assert len(out) == 1 and len(out[0]) == n
        print("test_python_bindings: nlolib_propagate default AUTO backend returned expected status.")
    except RuntimeError:
        print(
            "test_python_bindings: default AUTO backend unavailable on this machine; "
            "continuing with explicit CPU backend checks."
        )

    cpu_opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)
    out_records = api.propagate(cfg, input_field, 4, cpu_opts)
    assert len(out_records) == 4
    assert len(out_records[0]) == n
    print("test_python_bindings: nlolib_propagate with recorded outputs returned expected status.")

    nx = 16
    ny = 8
    nxy = nx * ny
    cfg_2d = prepare_sim_config(
        nxy,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * nxy,
        spatial_nx=nx,
        spatial_ny=ny,
        spatial_frequency_grid=[0j] * nxy,
        potential_grid=[1 + 0j] * nxy,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )
    out_2d = api.propagate(cfg_2d, [0j] * nxy, 1, cpu_opts)
    assert len(out_2d) == 1
    assert len(out_2d[0]) == nxy
    print("test_python_bindings: nlolib_propagate flattened 2D call returned expected status.")

    bad_cfg_2d = prepare_sim_config(
        nxy,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        frequency_grid=[0j] * nxy,
        spatial_nx=nx + 1,
        spatial_ny=ny,
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0]),
    )
    try:
        api.propagate(bad_cfg_2d, [0j] * nxy, 1, cpu_opts)
        raise AssertionError("expected invalid shape to fail")
    except RuntimeError as exc:
        assert "status=1" in str(exc)
    print("test_python_bindings: invalid flattened XY shape rejected as expected.")


if __name__ == "__main__":
    main()
