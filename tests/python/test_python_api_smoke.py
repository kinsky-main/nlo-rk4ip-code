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
    cpu_opts = default_execution_options(NLO_VECTOR_BACKEND_CPU)

    n = 128
    cfg = _base_config(n)
    input_field = [0j] * n

    out_records = api.propagate(cfg, input_field, 4, cpu_opts)
    assert len(out_records) == 4
    assert len(out_records[0]) == n
    print("test_python_api_smoke: CPU 1D propagation returned expected record-major shape.")

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
    print("test_python_api_smoke: flattened 2D propagation returned expected shape.")

    nt = 4
    nx3 = 4
    ny3 = 2
    n3 = nt * nx3 * ny3
    cfg_3d = prepare_sim_config(
        n3,
        propagation_distance=0.0,
        starting_step_size=0.01,
        max_step_size=0.1,
        min_step_size=0.001,
        error_tolerance=1e-6,
        pulse_period=1.0,
        delta_time=0.001,
        time_nt=nt,
        frequency_grid=[0j] * nt,
        spatial_nx=nx3,
        spatial_ny=ny3,
        spatial_frequency_grid=[0j] * (nx3 * ny3),
        potential_grid=[0j] * (nx3 * ny3),
        runtime=RuntimeOperators(constants=[0.0, 0.0, 1.0, 0.0]),
    )
    out_3d = api.propagate(cfg_3d, [0j] * n3, 1, cpu_opts)
    assert len(out_3d) == 1
    assert len(out_3d[0]) == n3
    print("test_python_api_smoke: explicit 3+1D layout propagation returned expected shape.")

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
        raise AssertionError("expected invalid flattened shape to fail")
    except RuntimeError as exc:
        assert "status=1" in str(exc)
    print("test_python_api_smoke: invalid flattened XY shape rejected as expected.")


if __name__ == "__main__":
    main()
