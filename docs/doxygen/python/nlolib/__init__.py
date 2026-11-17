"""
\file __init__.py
\brief Docs-only public Python API shim for the ``nlolib`` package.
\ingroup python_binding

This shim mirrors the public Python package surface for the Doxygen site. It is
not imported by the runtime package; the implementation lives in the
``nlolib-python`` repository (``python/nlolib/``).

The surface is the one all three bindings share, described in
``docs/binding_api_vocabulary.md`` and held as data in
``tests/api_surface.json``: module functions, no client class, the same types
and fields, and lowercase strings for every enum-style value.

\code{.py}
import nlolib

result = nlolib.propagate(
    nlolib.PulseSpec(samples=field0, delta_time=dt),
    lambda A, w: 1j * (0.5 * beta2) * w * w,
    lambda A, I: 1j * gamma * A * I,
    propagation_distance=1.0,
    num_records=2,
    preset="balanced",
    exec_options=nlolib.ExecutionOptions(backend_type="auto"),
)
final_power = sum(abs(a) ** 2 for a in result.final)
\endcode

\par Enum strings
- ``ExecutionOptions.backend_type``: ``"cpu"``, ``"vulkan"``, ``"auto"``
- ``ExecutionOptions.fft_backend``: ``"auto"``, ``"fftw"``, ``"vkfft"``
- ``nonlinear_model``: ``"expr"``, ``"kerr_raman"``
- ``StorageOptions.cap_policy``: ``"stop_writes"``, ``"fail"``
- ``set_log_level``: ``"error"``, ``"warn"``, ``"info"``, ``"debug"``
- ``set_progress_stream``: ``"stderr"``, ``"stdout"``, ``"both"``
- ``ProgressInfo.event_type``: ``"accepted"``, ``"rejected"``, ``"finish"``
- status: ``"ok"``, ``"invalid_argument"``, ``"allocation_failed"``,
  ``"not_implemented"``, ``"aborted"``

An unknown string raises ``ValueError``.

\par Errors
- operator is not a callable (strings included): ``TypeError``
- operator cannot be translated, or uses a symbol its context does not allow:
  ``ValueError``
- bad pulse or bad option value: ``ValueError``
- unknown option name: ``TypeError``
- library not found or not loadable: ``OSError``
- feature not in this build: NlolibError with ``status == "not_implemented"``
- a C call returned a non-OK status: NlolibError
- aborted by the progress callback: PropagationAbortedError
"""


class PulseSpec:
    """\ingroup python_binding
    Launch field and its grid.

    Fields, in order: ``samples``, ``delta_time``, ``pulse_period``,
    ``frequency_grid``, ``tensor_nt``, ``tensor_nx``, ``tensor_ny``,
    ``delta_x`` (1.0), ``delta_y`` (1.0), ``spatial_frequency_grid``,
    ``potential_grid``, ``wt_axis``, ``kx_axis``, ``ky_axis``.

    Only ``samples`` and ``delta_time`` are required. Setting ``tensor_nt``,
    ``tensor_nx`` and ``tensor_ny`` makes the run a tensor (t, y, x) run, with
    ``samples`` flattened t-fastest. ``propagate`` accepts only this type as
    its pulse; a dict is a ``ValueError``.
    """


class ExecutionOptions:
    """\ingroup python_binding
    Backend selection and device memory budget.

    Fields: ``backend_type`` (``"auto"``), ``fft_backend`` (``"auto"``),
    ``device_heap_fraction`` (0.70), ``record_ring_target`` (0),
    ``forced_device_budget_bytes`` (0).
    """


class StorageOptions:
    """\ingroup python_binding
    Store the run in the shared run database ``sqlite_path``.

    Fields: ``sqlite_path`` (required), ``run_id``, ``sqlite_max_bytes`` (0),
    ``chunk_records`` (0), ``cap_policy`` (``"stop_writes"``),
    ``log_final_output_field_to_db`` (True), ``record_mode`` (``"reduced"``,
    ``"raw"`` or ``"reduced_and_raw"``), ``raw_stride_t``, ``raw_stride_x``,
    ``raw_stride_y`` (1), ``defer_merge`` (False).
    """


class PropagationResult:
    """\ingroup python_binding
    Result of propagate().

    Fields: ``records`` (list of rows, one per recorded z), ``z_axis``,
    ``final``, ``step_history`` (a StepHistory, or None unless
    ``capture_step_history``), ``storage_result`` (a StorageResult, or None
    unless ``storage`` was given), ``meta``.

    ``meta`` holds ``preset``, ``records_requested``, ``records_written``,
    ``backend_requested``, ``coupled`` and ``status``.
    """


class StepHistory:
    """\ingroup python_binding
    Per-step solver telemetry.

    Fields: ``step_index``, ``z``, ``step_size``, ``next_step_size``,
    ``error``, ``dropped``, ``capacity``. A non-zero ``dropped`` means the
    history was truncated; raise ``step_history_capacity``.
    """


class StorageResult:
    """\ingroup python_binding
    Outcome of a storage-backed run.

    Fields: ``run_id``, ``records_captured``, ``records_spilled``,
    ``chunks_written``, ``db_size_bytes``, ``truncated`` (bool), ``key`` (the
    run key), ``merged`` and ``deferred`` (bool).
    """


class StagingRecovery:
    """\ingroup python_binding
    Staging files ``storage_recover_staging`` handled.

    Fields: ``merged``, ``discarded``, ``skipped``.
    """


class ProgressInfo:
    """\ingroup python_binding
    Payload passed to ``progress_callback``.

    Fields: ``event_type`` (``"accepted"``, ``"rejected"`` or ``"finish"``),
    ``step_index``, ``reject_attempt``, ``z``, ``z_end``, ``percent``,
    ``step_size``, ``next_step_size``, ``error``, ``elapsed_seconds``,
    ``eta_seconds``.
    """


class RuntimeLimits:
    """\ingroup python_binding
    Limits returned by query_runtime_limits().

    Fields: ``max_num_time_samples_runtime``,
    ``max_num_recorded_samples_in_memory``,
    ``max_num_recorded_samples_with_storage``,
    ``estimated_required_working_set_bytes``,
    ``estimated_device_budget_bytes``, ``storage_available`` (bool).
    """


class PerfProfileSnapshot:
    """\ingroup python_binding
    Performance counters returned by perf_profile_read().

    Fields: ``dispersion_ms``, ``nonlinear_ms``, ``dispersion_calls``,
    ``nonlinear_calls``, ``gpu_dispatch_count``, ``gpu_copy_count``,
    ``gpu_device_copy_count``, ``gpu_device_copy_bytes``,
    ``gpu_host_transfer_copy_count``, ``gpu_host_transfer_copy_bytes``,
    ``gpu_memory_pass_count``, ``gpu_memory_pass_bytes``, ``gpu_upload_count``,
    ``gpu_download_count``, ``gpu_upload_bytes``, ``gpu_download_bytes``,
    ``operator_jit_programs``, ``operator_jit_fallbacks``.
    """


class NlolibError(RuntimeError):
    """\ingroup python_binding
    A C call returned a non-OK status, or a feature is not in this build.

    ``status`` is the status string: ``"invalid_argument"``,
    ``"allocation_failed"``, ``"not_implemented"`` or ``"aborted"``.
    """


class PropagationAbortedError(NlolibError):
    """\ingroup python_binding
    The progress callback aborted the run.

    ``result`` is the PropagationResult holding what was recorded before the
    abort.
    """


def load(path=None):
    """\ingroup python_binding
    Load the nlolib shared library.

    \param path Library file to load. When omitted, the library shipped beside
                the package is used.
    \throws OSError the library is not found or cannot be loaded.
    """


def loaded_library_path():
    """\ingroup python_binding
    Path of the loaded shared library.
    """


def propagate(
    pulse,
    linear_operator,
    nonlinear_operator,
    *,
    propagation_distance,
    preset="balanced",
    starting_step_size=None,
    max_step_size=None,
    min_step_size=None,
    error_tolerance=None,
    num_records=None,
    record_z=None,
    return_records=True,
    potential_operator=None,
    linear_step_operator=None,
    constant_bindings=None,
    nonlinear_model="expr",
    nonlinear_gamma=0.0,
    raman_fraction=0.0,
    raman_tau1=0.0122,
    raman_tau2=0.0320,
    shock_omega0=0.0,
    raman_response_time=None,
    exec_options=None,
    storage=None,
    capture_step_history=False,
    step_history_capacity=200000,
    progress_callback=None,
):
    """\ingroup python_binding
    Propagate a pulse. The one entry point of the package.

    Operators are plain callables whose body is a single expression. Free
    names in the body are captured as real constants; ``constant_bindings``
    names values explicitly. A string is a ``TypeError``.

    \par Operator contexts
    The binding lowers each operator to the expression text the C core
    compiles, in the context the run type selects:
    - temporal run (``pulse.tensor_nt`` unset): ``linear_operator`` takes
      ``(A, w)``; ``linear_step_operator`` takes ``(A, D, h, w, V)``.
    - tensor run: ``linear_operator`` takes ``(A, wt, kx, ky, x, y, t)``;
      ``linear_step_operator`` takes ``(A, D, h, wt, kx, ky, x, y, t, V)``;
      ``potential_operator`` takes ``(x, y, t)``.
    - ``nonlinear_operator`` takes ``(A, I, V)`` in both.

    \par Parameter binding
    A parameter named after one of the context's symbols binds to it, wherever
    it stands. The remaining parameters take the next unused symbol in the
    context's order. A parameter named after a symbol the context does not
    provide, or two parameters resolving to one symbol, is a ``ValueError``.
    ``w`` and ``wt`` are the same symbol, so ``lambda A, w: ...`` works for
    both run types.

    \par Which records come back
    ``record_z`` given: those positions. ``num_records == 1`` without
    ``record_z``: the final field only. Otherwise ``num_records`` uniformly
    spaced positions from 0 to ``propagation_distance``.

    \param pulse                PulseSpec.
    \param linear_operator      Callable giving the linear factor \f$D\f$,
                                applied as \f$\exp(hD)\f$.
    \param nonlinear_operator   Callable giving the nonlinear right-hand side.
    \param propagation_distance Total distance in z. Required.
    \param preset               ``"fast"``, ``"balanced"`` or ``"accuracy"``:
                                supplies the step sizes, tolerance and record
                                count left unset.
    \param starting_step_size   First trial step, in z units.
    \param max_step_size        Upper step bound.
    \param min_step_size        Lower step bound.
    \param error_tolerance      Adaptive-step local error target.
    \param num_records          Number of recorded z positions.
    \param record_z             Explicit record positions.
    \param return_records       False to keep records out of the result, for
                                storage-only runs.
    \param potential_operator   Callable giving \f$V(x, y, t)\f$; tensor runs
                                only, and not together with
                                ``pulse.potential_grid``.
    \param linear_step_operator Callable replacing the default per-step linear
                                operator \f$\exp(hD)\f$.
    \param constant_bindings    Mapping of name to real value for names used
                                in the operators.
    \param nonlinear_model      ``"expr"`` or ``"kerr_raman"``.
    \param nonlinear_gamma      Kerr coefficient of the built-in model.
    \param raman_fraction       Raman fraction \f$f_R\f$.
    \param raman_tau1           Raman response time constant.
    \param raman_tau2           Raman response time constant.
    \param shock_omega0         Self-steepening carrier frequency; 0 disables.
    \param raman_response_time  Sampled Raman response, replacing the
                                analytic one.
    \param exec_options         ExecutionOptions.
    \param storage              StorageOptions.
    \param capture_step_history True to fill ``result.step_history``.
    \param step_history_capacity Steps kept before dropping.
    \param progress_callback    Called with a ProgressInfo; return False to
                                abort.
    \return PropagationResult.
    \throws TypeError               an operator is not a callable.
    \throws ValueError              bad pulse, bad option value, or an
                                    operator that cannot be translated.
    \throws NlolibError             the C call failed.
    \throws PropagationAbortedError the progress callback aborted the run.
    """


def query_runtime_limits(pulse=None, exec_options=None):
    """\ingroup python_binding
    Query the limits the runtime derives for a pulse and execution options.

    \param pulse        PulseSpec, or None for the limits of a default run.
    \param exec_options ExecutionOptions, or None for the defaults.
    \return RuntimeLimits.
    """


def storage_is_available():
    """\ingroup python_binding
    True when this build has SQLite snapshot storage.
    """


def storage_run_key(pulse, linear_operator, nonlinear_operator, *, propagation_distance,
                    preset="balanced", starting_step_size=None, max_step_size=None,
                    min_step_size=None, error_tolerance=None, num_records=None, record_z=None,
                    potential_operator=None, linear_step_operator=None, constant_bindings=None,
                    nonlinear_model="expr", nonlinear_gamma=0.0, raman_fraction=0.0,
                    raman_tau1=0.0122, raman_tau2=0.0320, shock_omega0=0.0,
                    raman_response_time=None, storage=None):
    """\ingroup python_binding
    Run key under which ``propagate`` with the same arguments stores the run.

    Takes ``propagate``'s arguments that enter the key; without ``storage`` the
    default ``StorageOptions`` apply. The key does not depend on the backend.

    \return 16 lowercase hex digits.
    """


def storage_has_run(sqlite_path, key):
    """\ingroup python_binding
    True when the shared run database ``sqlite_path`` has a run with ``key``.
    """


def storage_recover_staging(sqlite_path, *, discard_incomplete=False, min_age_seconds=86400.0):
    """\ingroup python_binding
    Merge the staging files left for the shared run database ``sqlite_path``.

    With ``discard_incomplete``, delete incomplete files at least
    ``min_age_seconds`` old.

    \return StagingRecovery.
    """


def tensor_record_view(record, nt, nx, ny):
    """\ingroup python_binding
    Reshape one flat tensor record to (t, y, x), t fastest.

    \param record One record of a tensor run, of length ``nt * nx * ny``.
    \param nt     Samples along t.
    \param nx     Samples along x.
    \param ny     Samples along y.
    \return Nested lists indexed ``[t][y][x]``; the package has no NumPy
            dependency.
    """


def set_log_file(path, *, append=False):
    """\ingroup python_binding
    Write the library log to a file. ``None`` stops file logging.
    """


def set_log_buffer(capacity_bytes=262144):
    """\ingroup python_binding
    Keep the library log in an in-memory ring buffer.
    """


def clear_log_buffer():
    """\ingroup python_binding
    Empty the log buffer.
    """


def read_log_buffer(consume=True, max_bytes=262144):
    """\ingroup python_binding
    Read the log buffer as text.

    \param consume   True to remove what was read.
    \param max_bytes Upper bound on the text returned.
    """


def set_log_level(level="info"):
    """\ingroup python_binding
    Set the log level: ``"error"``, ``"warn"``, ``"info"`` or ``"debug"``.
    """


def set_progress_options(enabled=True, milestone_percent=5, emit_on_step_adjust=False):
    """\ingroup python_binding
    Configure the library's own progress output.
    """


def set_progress_stream(stream_mode="stderr"):
    """\ingroup python_binding
    Send progress output to ``"stderr"``, ``"stdout"`` or ``"both"``.
    """


def perf_profile_set_enabled(enabled=True):
    """\ingroup python_binding
    Turn the performance counters on or off.
    """


def perf_profile_is_enabled():
    """\ingroup python_binding
    True when the performance counters are on.
    """


def perf_profile_reset():
    """\ingroup python_binding
    Zero the performance counters.
    """


def perf_profile_read():
    """\ingroup python_binding
    Read the performance counters.

    \return PerfProfileSnapshot.
    """
