"""Fused Vulkan RK algebra against the unfused sequence (NLO_DISABLE_VK_FUSED_STEP=1).

Fixed step: final fields agree to 1e-12 relative L2. Adaptive ERK4(3): the
same number of accepted steps, with step sizes equal up to rounding, so the
controller reads the same error norm on both paths. Skips when no Vulkan device is available.
"""

import math
import os

from nlolib import (
    TENSOR_LAYOUT_XYT_T_FAST,
    VECTOR_BACKEND_VULKAN,
    NLolib,
    RuntimeOperators,
    default_execution_options,
    prepare_sim_config,
)


def _centered(n, d):
    return [(i - 0.5 * (n - 1)) * d for i in range(n)]


def _omega(n, dt):
    return [complex(2.0 * math.pi * (i if i < (n + 1) // 2 else i - n) / (n * dt), 0.0) for i in range(n)]


def _case_kerr1d(adaptive):
    n, dt = 1024, 0.01
    field = [complex(math.exp(-((t / 0.2) ** 2)), 0.0) for t in _centered(n, dt)]
    h = 1.0e-3
    cfg = prepare_sim_config(
        n,
        propagation_distance=0.05,
        starting_step_size=(8.0 * h) if adaptive else h,
        max_step_size=(8.0 * h) if adaptive else h,
        min_step_size=1.0e-6 if adaptive else h,
        error_tolerance=1.0e-7,
        pulse_period=n * dt,
        delta_time=dt,
        frequency_grid=_omega(n, dt),
        runtime=RuntimeOperators(
            linear_factor_expr="i*c0*(w*w)",
            linear_expr="exp(h*D)",
            nonlinear_expr="i*A*(c1*I - c2*I*I)/(1.0 + c3*I)",
            constants=[-0.005, 80.0, 20.0, 0.5],
        ),
    )
    return cfg, field


def _case_grin_tensor(adaptive):
    nt, nxy, dt, dx = 64, 16, 0.02, 0.03
    t = _centered(nt, dt)
    x = _centered(nxy, dx)
    temporal = [1.0 / math.cosh(v / 0.3) for v in t]
    field = []
    potential = []
    for yv in x:
        for xv in x:
            spatial = math.exp(-((xv / 0.25) ** 2) - ((yv / 0.3) ** 2))
            field.extend(complex(spatial * tv, 0.0) for tv in temporal)
            potential.extend([complex(1.5e-3 * (xv * xv + yv * yv), 0.0)] * nt)
    h = 2.0e-3
    cfg = prepare_sim_config(
        nt * nxy * nxy,
        propagation_distance=0.02,
        starting_step_size=(4.0 * h) if adaptive else h,
        max_step_size=(4.0 * h) if adaptive else h,
        min_step_size=1.0e-6 if adaptive else h,
        error_tolerance=1.0e-10,
        pulse_period=nt * dt,
        delta_time=dt,
        tensor_nt=nt,
        tensor_nx=nxy,
        tensor_ny=nxy,
        tensor_layout=int(TENSOR_LAYOUT_XYT_T_FAST),
        delta_x=dx,
        delta_y=dx,
        frequency_grid=_omega(nt, dt),
        potential_grid=potential,
        runtime=RuntimeOperators(
            linear_factor_expr="i*(c0*(wt*wt) + c1*(kx*kx + ky*ky))",
            linear_expr="exp(h*D)",
            nonlinear_expr="i*A*(c2*(I/(1.0 + I/c3)) + V)",
            constants=[-0.04, -0.08, 1.5, 4.0],
        ),
    )
    return cfg, field


def _run(api, opts, cfg, field, fused, adaptive):
    if fused:
        os.environ.pop("NLO_DISABLE_VK_FUSED_STEP", None)
    else:
        os.environ["NLO_DISABLE_VK_FUSED_STEP"] = "1"
    kwargs = {"capture_step_history": True, "step_history_capacity": 20000} if adaptive else {}
    result = api.propagate(cfg, field, 2, opts, **kwargs)
    os.environ.pop("NLO_DISABLE_VK_FUSED_STEP", None)
    return result


def _rel_l2(a, b):
    num = sum(abs(x - y) ** 2 for x, y in zip(a, b))
    den = sum(abs(y) ** 2 for y in b)
    return math.sqrt(num / den)


def main():
    api = NLolib()
    api.set_progress_options(enabled=False)
    opts = default_execution_options(VECTOR_BACKEND_VULKAN)
    try:
        probe_cfg, probe_field = _case_kerr1d(False)
        api.propagate(probe_cfg, probe_field, 2, opts)
    except Exception as exc:  # noqa: BLE001 - any failure means no usable Vulkan device
        print(f"test_python_fused_rk_step: Vulkan unavailable ({exc}); skipped.")
        return

    for name, make_case in (("kerr1d", _case_kerr1d), ("grin_tensor", _case_grin_tensor)):
        cfg, field = make_case(False)
        fused = _run(api, opts, cfg, field, True, False).records[-1]
        unfused = _run(api, opts, cfg, field, False, False).records[-1]
        err = _rel_l2(fused, unfused)
        print(f"test_python_fused_rk_step: {name} fixed step, fused vs unfused rel L2 {err:.2e}")
        assert err <= 1e-12, (name, err)

        cfg, field = make_case(True)
        fused_run = _run(api, opts, cfg, field, True, True)
        unfused_run = _run(api, opts, cfg, field, False, True)
        fused_steps = fused_run.meta["step_history"]["step_size"]
        unfused_steps = unfused_run.meta["step_history"]["step_size"]
        assert len(fused_steps) == len(unfused_steps) and len(fused_steps) > 3, (
            name,
            len(fused_steps),
            len(unfused_steps),
        )
        step_diff = max(abs(a - b) / b for a, b in zip(fused_steps, unfused_steps))
        field_err = _rel_l2(fused_run.records[-1], unfused_run.records[-1])
        print(
            f"test_python_fused_rk_step: {name} adaptive, {len(fused_steps)} accepted steps, "
            f"max step-size rel diff {step_diff:.2e}, final field rel L2 {field_err:.2e}"
        )
        # The error norm is a difference of near-equal fields, so rounding-level
        # field changes move it by ~eps/tol and the next step by a quarter of
        # that. A read of the norm before the GPU finishes (the archived
        # async-flush race) changes the step count or sizes by far more.
        assert step_diff <= 1e-6, (name, step_diff)
        assert field_err <= 1e-10, (name, field_err)

    print("test_python_fused_rk_step: fused RK algebra matches the unfused path.")


if __name__ == "__main__":
    main()
