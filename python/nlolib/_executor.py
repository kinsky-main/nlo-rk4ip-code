"""
Execution layer for normalized propagation requests.
"""

from __future__ import annotations

import ctypes

from ._binding import (
    PROPAGATE_OUTPUT_DENSE,
    PROPAGATE_OUTPUT_FINAL_ONLY,
    VECTOR_BACKEND_AUTO,
    NLOLIB_STATUS_ABORTED,
    NLOLIB_STATUS_OK,
    NloComplex,
    NloProgressCallback,
    NloPropagateOptions,
    NloPropagateOutput,
    NloStepEvent,
    NloStorageResult,
    complex_array_to_list,
    default_storage_options,
    make_complex_array,
)
from ._config import (
    _progress_info_from_struct,
    _step_events_to_meta,
    _storage_result_to_meta,
    _validate_explicit_record_z,
)
from ._models import PropagationAbortedError, PropagationResult
from ._requests import _NormalizedPropagateRequest


_DENSE_OUTPUT_MAX_BYTES = 2 * 1024 * 1024 * 1024
_COMPLEX_BYTES = ctypes.sizeof(NloComplex)


class PropagationExecutor:
    def __init__(self, lib: ctypes.CDLL, storage_is_available) -> None:
        self.lib = lib
        self._storage_is_available = storage_is_available

    def execute(self, request: _NormalizedPropagateRequest) -> PropagationResult:
        if request.sqlite_path is not None and not self._storage_is_available():
            raise RuntimeError("SQLite storage is not available in this nlolib build")

        sim_ptr = ctypes.pointer(request.sim_cfg)
        phys_ptr = ctypes.pointer(request.phys_cfg)
        n = len(request.input_seq)
        if request.return_records:
            dense_output_bytes = n * request.num_records * _COMPLEX_BYTES
            if dense_output_bytes > _DENSE_OUTPUT_MAX_BYTES:
                if request.sqlite_path is not None:
                    raise ValueError(
                        "dense output request requires "
                        f"{dense_output_bytes} bytes, above the {_DENSE_OUTPUT_MAX_BYTES} byte safety cap; "
                        "use return_records=False with SQLite storage for large recorded outputs"
                    )
                raise ValueError(
                    "dense output request requires "
                    f"{dense_output_bytes} bytes, above the {_DENSE_OUTPUT_MAX_BYTES} byte safety cap; "
                    "enable SQLite storage or request fewer records/samples"
                )
        in_arr = make_complex_array(request.input_seq)
        out_arr = make_complex_array(n * request.num_records) if request.return_records else None
        explicit_record_z_arr = None
        progress_callback_ref = None
        progress_callback_context = None
        progress_callback_error: list[BaseException | None] = [None]

        storage_opts = None
        storage_keepalive: list[bytes] = []
        if request.sqlite_path is not None:
            storage_opts, storage_keepalive = default_storage_options(
                sqlite_path=request.sqlite_path,
                run_id=request.run_id,
                sqlite_max_bytes=request.sqlite_max_bytes,
                chunk_records=request.chunk_records,
                cap_policy=request.cap_policy,
                log_final_output_field_to_db=request.log_final_output_field_to_db,
            )
        _ = storage_keepalive
        if request.explicit_record_z is not None:
            distance = float(request.sim_cfg.propagation.propagation_distance)
            _validate_explicit_record_z(request.explicit_record_z, distance)
            explicit_record_z_arr = (ctypes.c_double * len(request.explicit_record_z))(
                *[float(v) for v in request.explicit_record_z]
            )

        propagate_options = (
            self.lib.nlolib_propagate_options_default()
            if bool(getattr(self.lib, "_has_propagate_defaults", False))
            else NloPropagateOptions()
        )
        propagate_options.num_recorded_samples = request.num_records
        propagate_options.output_mode = (
            PROPAGATE_OUTPUT_FINAL_ONLY
            if request.num_records == 1
            else PROPAGATE_OUTPUT_DENSE
        )
        propagate_options.return_records = int(bool(request.return_records))
        propagate_options.exec_options = (
            ctypes.pointer(request.exec_options)
            if request.exec_options is not None
            else None
        )
        propagate_options.storage_options = (
            ctypes.pointer(storage_opts)
            if storage_opts is not None
            else None
        )
        propagate_options.explicit_record_z = (
            ctypes.cast(explicit_record_z_arr, ctypes.POINTER(ctypes.c_double))
            if explicit_record_z_arr is not None
            else None
        )
        propagate_options.explicit_record_z_count = (
            len(request.explicit_record_z)
            if request.explicit_record_z is not None
            else 0
        )
        if request.progress_callback is not None:
            progress_callback_context = ctypes.py_object(request.progress_callback)

            def _progress_trampoline(info_ptr, user_data):
                callback = ctypes.cast(user_data, ctypes.POINTER(ctypes.py_object)).contents.value
                assert info_ptr
                progress_info = _progress_info_from_struct(info_ptr.contents)
                try:
                    result = callback(progress_info)
                except BaseException as exc:  # pragma: no cover
                    progress_callback_error[0] = exc
                    return 0
                if result is None:
                    return 1
                return 1 if bool(result) else 0

            progress_callback_ref = NloProgressCallback(_progress_trampoline)
            propagate_options.progress_callback = progress_callback_ref
            propagate_options.progress_user_data = ctypes.cast(
                ctypes.pointer(progress_callback_context),
                ctypes.c_void_p,
            )
        else:
            propagate_options.progress_callback = NloProgressCallback()
            propagate_options.progress_user_data = None

        storage_result = NloStorageResult()
        records_written = ctypes.c_size_t(0)
        step_events_out = (
            (NloStepEvent * int(request.step_history_capacity))()
            if request.capture_step_history and request.step_history_capacity > 0
            else None
        )
        step_events_written = ctypes.c_size_t(0)
        step_events_dropped = ctypes.c_size_t(0)
        propagate_output = (
            self.lib.nlolib_propagate_output_default()
            if bool(getattr(self.lib, "_has_propagate_defaults", False))
            else NloPropagateOutput()
        )
        propagate_output.output_records = (
            ctypes.cast(out_arr, ctypes.POINTER(NloComplex))
            if out_arr is not None
            else None
        )
        propagate_output.output_record_capacity = request.num_records if out_arr is not None else 0
        propagate_output.records_written = ctypes.pointer(records_written)
        propagate_output.storage_result = ctypes.pointer(storage_result)
        propagate_output.output_step_events = (
            ctypes.cast(step_events_out, ctypes.POINTER(NloStepEvent))
            if step_events_out is not None
            else None
        )
        propagate_output.output_step_event_capacity = (
            int(request.step_history_capacity)
            if step_events_out is not None
            else 0
        )
        propagate_output.step_events_written = ctypes.pointer(step_events_written)
        propagate_output.step_events_dropped = ctypes.pointer(step_events_dropped)

        status = int(
            self.lib.nlolib_propagate(
                sim_ptr,
                phys_ptr,
                n,
                ctypes.cast(in_arr, ctypes.POINTER(NloComplex)),
                ctypes.pointer(propagate_options),
                ctypes.pointer(propagate_output),
            )
        )
        if status not in {NLOLIB_STATUS_OK, NLOLIB_STATUS_ABORTED}:
            raise RuntimeError(f"nlolib_propagate failed with status={status}")

        records_written_count = int(records_written.value)
        if out_arr is None:
            out_records: list[list[complex]] = []
        else:
            out_records = self._records_from_complex_array(out_arr, n, records_written_count)
        storage_result_meta = (
            _storage_result_to_meta(storage_result)
            if request.sqlite_path is not None
            else None
        )

        distance = float(request.sim_cfg.propagation.propagation_distance)
        if request.explicit_record_z is not None:
            z_axis = [float(v) for v in request.explicit_record_z[:records_written_count]]
        else:
            z_axis = self._build_z_axis(distance, records_written_count)
        final = list(out_records[-1]) if out_records else []
        meta: dict[str, object] = {
            "output": request.output_label,
            "records": request.num_records,
            "records_requested": request.num_records,
            "records_written": records_written_count,
            "storage_enabled": bool(request.sqlite_path is not None),
            "records_returned": bool(len(out_records) > 0),
            "backend_requested": (
                int(request.exec_options.backend_type)
                if request.exec_options is not None
                else int(VECTOR_BACKEND_AUTO)
            ),
            "coupled": bool(
                int(request.sim_cfg.spatial.nx) > 1 or int(request.sim_cfg.spatial.ny) > 1
            ),
            "status": int(status),
            "message": (
                "propagate aborted by progress callback"
                if status == NLOLIB_STATUS_ABORTED
                else "propagate completed"
            ),
        }
        if storage_result_meta is not None:
            meta["storage_result"] = storage_result_meta
        if step_events_out is not None:
            step_history = _step_events_to_meta(step_events_out, int(step_events_written.value))
            step_history["dropped"] = int(step_events_dropped.value)
            step_history["capacity"] = int(request.step_history_capacity)
            meta["step_history"] = step_history
        meta.update(request.meta_overrides)
        result = PropagationResult(records=out_records, z_axis=z_axis, final=final, meta=meta)
        if progress_callback_error[0] is not None:
            raise progress_callback_error[0]
        if status == NLOLIB_STATUS_ABORTED:
            raise PropagationAbortedError(result)
        return result

    @staticmethod
    def _records_from_complex_array(
        out_arr: ctypes.Array,
        num_time_samples: int,
        num_recorded_samples: int,
    ) -> list[list[complex]]:
        flat = complex_array_to_list(out_arr, num_time_samples * num_recorded_samples)
        records: list[list[complex]] = []
        for i in range(num_recorded_samples):
            start = i * num_time_samples
            records.append(flat[start : start + num_time_samples])
        return records

    @staticmethod
    def _build_z_axis(distance: float, num_records: int) -> list[float]:
        if num_records <= 0:
            return []
        if num_records == 1:
            return [distance]
        return [
            (distance * float(i)) / float(num_records - 1)
            for i in range(num_records)
        ]
