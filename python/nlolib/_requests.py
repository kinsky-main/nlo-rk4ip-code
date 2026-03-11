"""
Propagation request normalization for both low-level and high-level APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ._binding import (
    NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES,
    NloExecutionOptions,
    NloPhysicsConfig,
    NloSimulationConfig,
)
from ._config import (
    PreparedSimConfig,
    _default_frequency_grid,
    _normalize_pulse_spec,
    _resolve_operator_spec,
    _solver_profile_defaults,
    _validate_coupled_pulse_spec,
    prepare_sim_config,
)
from ._models import ProgressInfo, RuntimeOperators


@dataclass
class _NormalizedPropagateRequest:
    sim_cfg: NloSimulationConfig
    phys_cfg: NloPhysicsConfig
    input_seq: list[complex]
    num_records: int
    exec_options: NloExecutionOptions | None
    sqlite_path: str | None
    run_id: str | None
    sqlite_max_bytes: int
    chunk_records: int
    cap_policy: int
    log_final_output_field_to_db: bool
    return_records: bool
    capture_step_history: bool
    step_history_capacity: int
    output_label: str
    explicit_record_z: list[float] | None
    progress_callback: Callable[[ProgressInfo], bool | int | None] | None
    meta_overrides: dict[str, Any]


class PropagateRequestBuilder:
    def from_config(
        self,
        config: PreparedSimConfig | NloSimulationConfig,
        *args: Any,
        **kwargs: Any,
    ) -> _NormalizedPropagateRequest:
        input_field = kwargs.pop("input_field", None)
        num_recorded_samples = kwargs.pop("num_recorded_samples", None)
        physics_config = kwargs.pop("physics_config", None)
        exec_options = kwargs.pop("exec_options", None)

        if len(args) > 3:
            raise TypeError("low-level propagate accepts at most three positional args after config")
        if len(args) >= 1:
            if input_field is not None:
                raise TypeError("input_field provided in both args and kwargs")
            input_field = args[0]
        if len(args) >= 2:
            if num_recorded_samples is not None:
                raise TypeError("num_recorded_samples provided in both args and kwargs")
            num_recorded_samples = args[1]
        if len(args) == 3:
            if exec_options is not None:
                raise TypeError("exec_options provided in both args and kwargs")
            exec_options = args[2]

        if input_field is None or num_recorded_samples is None:
            raise TypeError("low-level propagate requires input_field and num_recorded_samples")

        sqlite_path = kwargs.pop("sqlite_path", None)
        run_id = kwargs.pop("run_id", None)
        sqlite_max_bytes = int(kwargs.pop("sqlite_max_bytes", 0))
        chunk_records = int(kwargs.pop("chunk_records", 0))
        cap_policy = int(kwargs.pop("cap_policy", NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES))
        log_final_output_field_to_db = bool(kwargs.pop("log_final_output_field_to_db", False))
        return_records = bool(kwargs.pop("return_records", True))
        capture_step_history = bool(kwargs.pop("capture_step_history", False))
        step_history_capacity = int(kwargs.pop("step_history_capacity", (200000 if capture_step_history else 0)))
        progress_callback = kwargs.pop("progress_callback", None)
        t_eval_raw = kwargs.pop("t_eval", None)
        if kwargs:
            raise TypeError(f"unexpected propagate kwargs: {sorted(kwargs.keys())}")

        num_records = int(num_recorded_samples)
        if num_records <= 0:
            raise ValueError("num_recorded_samples must be > 0")

        input_seq = list(input_field)
        if len(input_seq) == 0:
            raise ValueError("input_field must be non-empty")
        if step_history_capacity < 0:
            raise ValueError("step_history_capacity must be >= 0")
        if capture_step_history and step_history_capacity <= 0:
            raise ValueError("step_history_capacity must be > 0 when capture_step_history=True")

        explicit_record_z: list[float] | None = None
        if t_eval_raw is not None:
            explicit_record_z = [float(v) for v in t_eval_raw]
            if len(explicit_record_z) <= 0:
                raise ValueError("t_eval must be non-empty when provided")
            num_records = len(explicit_record_z)

        if isinstance(config, PreparedSimConfig):
            sim_cfg = config.simulation_config
            phys_cfg = config.physics_config
        else:
            sim_cfg = config
            phys_cfg = NloPhysicsConfig()
        if physics_config is not None:
            phys_cfg = physics_config

        return _NormalizedPropagateRequest(
            sim_cfg=sim_cfg,
            phys_cfg=phys_cfg,
            input_seq=input_seq,
            num_records=num_records,
            exec_options=exec_options,
            sqlite_path=(str(sqlite_path) if sqlite_path is not None else None),
            run_id=(None if run_id is None else str(run_id)),
            sqlite_max_bytes=sqlite_max_bytes,
            chunk_records=chunk_records,
            cap_policy=cap_policy,
            log_final_output_field_to_db=log_final_output_field_to_db,
            return_records=return_records,
            capture_step_history=capture_step_history,
            step_history_capacity=step_history_capacity,
            output_label=("final" if num_records == 1 else "dense"),
            explicit_record_z=explicit_record_z,
            progress_callback=progress_callback,
            meta_overrides={},
        )

    def from_pulse(
        self,
        pulse: Any,
        *args: Any,
        **kwargs: Any,
    ) -> _NormalizedPropagateRequest:
        if len(args) > 2:
            raise TypeError("high-level propagate accepts at most two positional operator args")

        linear_operator = kwargs.pop("linear_operator", "gvd")
        nonlinear_operator = kwargs.pop("nonlinear_operator", "kerr")
        if len(args) >= 1:
            linear_operator = args[0]
        if len(args) >= 2:
            nonlinear_operator = args[1]

        if "transverse_operator" in kwargs:
            raise TypeError(
                "transverse_operator has been removed; encode diffraction in linear_operator "
                "with tensor descriptors (tensor_nt/tensor_nx/tensor_ny)"
            )
        propagation_distance = kwargs.pop("propagation_distance", None)
        pulse_period_override = kwargs.pop("pulse_period", None)
        frequency_grid_override = kwargs.pop("frequency_grid", None)
        tensor_nt_override = kwargs.pop("tensor_nt", None)
        tensor_nx_override = kwargs.pop("tensor_nx", None)
        tensor_ny_override = kwargs.pop("tensor_ny", None)
        tensor_layout_override = kwargs.pop("tensor_layout", None)
        delta_x_override = kwargs.pop("delta_x", None)
        delta_y_override = kwargs.pop("delta_y", None)
        spatial_frequency_grid_override = kwargs.pop("spatial_frequency_grid", None)
        potential_grid_override = kwargs.pop("potential_grid", None)
        output = kwargs.pop("output", "dense")
        preset = kwargs.pop("preset", "balanced")
        records = kwargs.pop("records", None)
        starting_step_size_override = kwargs.pop("starting_step_size", None)
        max_step_size_override = kwargs.pop("max_step_size", None)
        min_step_size_override = kwargs.pop("min_step_size", None)
        error_tolerance_override = kwargs.pop("error_tolerance", None)
        exec_options = kwargs.pop("exec_options", None)
        sqlite_path = kwargs.pop("sqlite_path", None)
        run_id = kwargs.pop("run_id", None)
        sqlite_max_bytes = int(kwargs.pop("sqlite_max_bytes", 0))
        chunk_records = int(kwargs.pop("chunk_records", 0))
        cap_policy = int(kwargs.pop("cap_policy", NLO_STORAGE_DB_CAP_POLICY_STOP_WRITES))
        log_final_output_field_to_db = bool(kwargs.pop("log_final_output_field_to_db", False))
        return_records = bool(kwargs.pop("return_records", True))
        capture_step_history = bool(kwargs.pop("capture_step_history", False))
        step_history_capacity = int(kwargs.pop("step_history_capacity", (200000 if capture_step_history else 0)))
        progress_callback = kwargs.pop("progress_callback", None)
        t_eval_raw = kwargs.pop("t_eval", None)
        nonlinear_model_override = kwargs.pop("nonlinear_model", None)
        nonlinear_gamma_override = kwargs.pop("nonlinear_gamma", None)
        raman_fraction_override = kwargs.pop("raman_fraction", None)
        raman_tau1_override = kwargs.pop("raman_tau1", None)
        raman_tau2_override = kwargs.pop("raman_tau2", None)
        shock_omega0_override = kwargs.pop("shock_omega0", None)
        raman_response_time_override = kwargs.pop("raman_response_time", None)
        constants_override = kwargs.pop("constants", None)
        constant_bindings_override = kwargs.pop("constant_bindings", None)
        auto_capture_constants_override = kwargs.pop("auto_capture_constants", None)
        if kwargs:
            raise TypeError(f"unexpected high-level propagate kwargs: {sorted(kwargs.keys())}")
        if propagation_distance is None:
            raise TypeError("high-level propagate requires propagation_distance")
        if step_history_capacity < 0:
            raise ValueError("step_history_capacity must be >= 0")
        if capture_step_history and step_history_capacity <= 0:
            raise ValueError("step_history_capacity must be > 0 when capture_step_history=True")

        pulse_spec = _normalize_pulse_spec(pulse)
        if pulse_period_override is not None:
            pulse_spec.pulse_period = float(pulse_period_override)
        if frequency_grid_override is not None:
            pulse_spec.frequency_grid = frequency_grid_override  # type: ignore[assignment]
        if tensor_nt_override is not None:
            pulse_spec.tensor_nt = int(tensor_nt_override)
        if tensor_nx_override is not None:
            pulse_spec.tensor_nx = int(tensor_nx_override)
        if tensor_ny_override is not None:
            pulse_spec.tensor_ny = int(tensor_ny_override)
        if tensor_layout_override is not None:
            pulse_spec.tensor_layout = int(tensor_layout_override)
        if delta_x_override is not None:
            pulse_spec.delta_x = float(delta_x_override)
        if delta_y_override is not None:
            pulse_spec.delta_y = float(delta_y_override)
        if spatial_frequency_grid_override is not None:
            pulse_spec.spatial_frequency_grid = spatial_frequency_grid_override  # type: ignore[assignment]
        if potential_grid_override is not None:
            pulse_spec.potential_grid = potential_grid_override  # type: ignore[assignment]

        profile = _solver_profile_defaults(preset, float(propagation_distance))
        if starting_step_size_override is not None:
            profile["starting_step_size"] = float(starting_step_size_override)
        if max_step_size_override is not None:
            profile["max_step_size"] = float(max_step_size_override)
        if min_step_size_override is not None:
            profile["min_step_size"] = float(min_step_size_override)
        if error_tolerance_override is not None:
            profile["error_tolerance"] = float(error_tolerance_override)
        num_records = int(records) if records is not None else int(profile["records"])
        if output == "final":
            num_records = 1
        elif output != "dense":
            raise ValueError("output must be 'dense' or 'final'")
        if num_records <= 0:
            raise ValueError("records must be > 0")

        explicit_record_z: list[float] | None = None
        if t_eval_raw is not None:
            explicit_record_z = [float(v) for v in t_eval_raw]
            if len(explicit_record_z) <= 0:
                raise ValueError("t_eval must be non-empty when provided")
            num_records = len(explicit_record_z)

        linear_expr, linear_fn, linear_constants, linear_bindings = _resolve_operator_spec(
            "linear",
            linear_operator,
            0,
        )
        nonlinear_expr, nonlinear_fn, nonlinear_constants, nonlinear_bindings = _resolve_operator_spec(
            "nonlinear",
            nonlinear_operator,
            len(linear_constants),
        )

        binding_map: dict[str, float] = {}
        for source in (linear_bindings, nonlinear_bindings):
            if source is None:
                continue
            for key, value in source.items():
                existing = binding_map.get(key)
                if existing is not None and existing != value:
                    raise ValueError(f"conflicting callable param '{key}' across operators")
                binding_map[key] = value

        num_time_samples = len(pulse_spec.samples)
        tensor_mode = (
            pulse_spec.tensor_nt is not None and
            pulse_spec.tensor_nx is not None and
            pulse_spec.tensor_ny is not None
        )
        if tensor_mode:
            _validate_coupled_pulse_spec(pulse_spec)
        temporal_samples = (
            int(pulse_spec.tensor_nt)
            if pulse_spec.tensor_nt is not None and int(pulse_spec.tensor_nt) > 0
            else num_time_samples
        )
        pulse_period = (
            float(pulse_spec.pulse_period)
            if pulse_spec.pulse_period is not None
            else float(pulse_spec.delta_time) * float(temporal_samples)
        )
        frequency_grid = (
            list(pulse_spec.frequency_grid)
            if pulse_spec.frequency_grid is not None
            else _default_frequency_grid(temporal_samples, pulse_spec.delta_time)
        )

        runtime = RuntimeOperators(
            linear_factor_expr=linear_expr,
            nonlinear_expr=nonlinear_expr,
            linear_factor_fn=linear_fn,
            nonlinear_fn=nonlinear_fn,
            constants=[*linear_constants, *nonlinear_constants],
            constant_bindings=binding_map if binding_map else None,
            auto_capture_constants=(not binding_map),
        )
        if nonlinear_model_override is not None:
            runtime.nonlinear_model = int(nonlinear_model_override)
        if nonlinear_gamma_override is not None:
            runtime.nonlinear_gamma = float(nonlinear_gamma_override)
        if raman_fraction_override is not None:
            runtime.raman_fraction = float(raman_fraction_override)
        if raman_tau1_override is not None:
            runtime.raman_tau1 = float(raman_tau1_override)
        if raman_tau2_override is not None:
            runtime.raman_tau2 = float(raman_tau2_override)
        if shock_omega0_override is not None:
            runtime.shock_omega0 = float(shock_omega0_override)
        if raman_response_time_override is not None:
            runtime.raman_response_time = raman_response_time_override  # type: ignore[assignment]
        if constants_override is not None:
            runtime.constants = constants_override  # type: ignore[assignment]
        if constant_bindings_override is not None:
            runtime.constant_bindings = constant_bindings_override  # type: ignore[assignment]
        if auto_capture_constants_override is not None:
            runtime.auto_capture_constants = bool(auto_capture_constants_override)

        config = prepare_sim_config(
            num_time_samples,
            propagation_distance=float(propagation_distance),
            starting_step_size=float(profile["starting_step_size"]),
            max_step_size=float(profile["max_step_size"]),
            min_step_size=float(profile["min_step_size"]),
            error_tolerance=float(profile["error_tolerance"]),
            pulse_period=float(pulse_period),
            delta_time=float(pulse_spec.delta_time),
            tensor_nt=pulse_spec.tensor_nt,
            tensor_nx=pulse_spec.tensor_nx,
            tensor_ny=pulse_spec.tensor_ny,
            tensor_layout=int(pulse_spec.tensor_layout),
            frequency_grid=frequency_grid,
            delta_x=float(pulse_spec.delta_x),
            delta_y=float(pulse_spec.delta_y),
            spatial_frequency_grid=pulse_spec.spatial_frequency_grid,
            potential_grid=pulse_spec.potential_grid,
            runtime=runtime,
        )
        sim_cfg = config.simulation_config
        phys_cfg = config.physics_config
        return _NormalizedPropagateRequest(
            sim_cfg=sim_cfg,
            phys_cfg=phys_cfg,
            input_seq=list(pulse_spec.samples),
            num_records=num_records,
            exec_options=exec_options,
            sqlite_path=(str(sqlite_path) if sqlite_path is not None else None),
            run_id=(None if run_id is None else str(run_id)),
            sqlite_max_bytes=sqlite_max_bytes,
            chunk_records=chunk_records,
            cap_policy=cap_policy,
            log_final_output_field_to_db=log_final_output_field_to_db,
            return_records=return_records,
            capture_step_history=capture_step_history,
            step_history_capacity=step_history_capacity,
            output_label=output,
            explicit_record_z=explicit_record_z,
            progress_callback=progress_callback,
            meta_overrides={
                "preset": preset,
                "output": output,
                "coupled": bool(
                    tensor_mode and ((int(pulse_spec.tensor_nx) > 1) or (int(pulse_spec.tensor_ny) > 1))
                ),
            },
        )
