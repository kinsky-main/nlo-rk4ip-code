classdef constants
    %CONSTANTS Named values for the nlolib C API enums and limits.
    %
    %   The MATLAB wrapper takes plain integers for backend selection, log
    %   levels, storage policies and the like. Before this class those values
    %   could only be found in doc comments, so callers wrote magic numbers
    %   that no longer matched the C header if it changed. These are the same
    %   names the Python (nlolib.*) and Julia (NLOLib.*) packages export, with
    %   the same values, so a script reads the same way in all three.
    %
    %   Enum-like values are int32 and limits are uint64, matching the field
    %   types in src/nlolib_matlab.h, so they can be assigned straight into a
    %   config or execution-options struct without a cast.
    %
    %   Example:
    %     options.exec_options = struct( ...
    %         'backend_type', nlolib.constants.VECTOR_BACKEND_VULKAN, ...
    %         'fft_backend',  nlolib.constants.FFT_BACKEND_VKFFT);
    %
    %   See also NLolib, nlolib.prepare_sim_config

    properties (Constant)
        %% Solver limits
        % Unbounded sentinel for the time-sample count. Prefer asking
        % NLolib/query_runtime_limits for the real runtime limit.
        NT_MAX = intmax('uint64')
        % Maximum length of a storage run identifier, in bytes.
        STORAGE_RUN_ID_MAX = uint64(64)

        %% Vector backend selection (execution_options.backend_type)
        VECTOR_BACKEND_CPU = int32(0)
        VECTOR_BACKEND_VULKAN = int32(1)
        VECTOR_BACKEND_AUTO = int32(2)

        %% FFT backend selection (execution_options.fft_backend)
        FFT_BACKEND_AUTO = int32(0)
        FFT_BACKEND_FFTW = int32(1)
        FFT_BACKEND_VKFFT = int32(2)

        %% Nonlinear model (runtime_operator_params.nonlinear_model)
        % Evaluate the compiled nonlinear expression.
        NONLINEAR_MODEL_EXPR = int32(0)
        % Use the built-in Kerr + Raman model and its raman_* parameters.
        NONLINEAR_MODEL_KERR_RAMAN = int32(1)

        %% Coupled-field memory layout (tensor_layout)
        % t fastest, then y, then x.
        TENSOR_LAYOUT_XYT_T_FAST = int32(0)

        %% Behaviour when the SQLite store hits its size cap
        STORAGE_DB_CAP_POLICY_STOP_WRITES = int32(0)
        STORAGE_DB_CAP_POLICY_FAIL = int32(1)

        %% Record selection (propagate_output mode)
        PROPAGATE_OUTPUT_DENSE = int32(0)
        PROPAGATE_OUTPUT_FINAL_ONLY = int32(1)

        %% Status codes returned by the C entry points
        NLOLIB_STATUS_OK = int32(0)
        NLOLIB_STATUS_INVALID_ARGUMENT = int32(1)
        NLOLIB_STATUS_ALLOCATION_FAILED = int32(2)
        NLOLIB_STATUS_NOT_IMPLEMENTED = int32(3)
        NLOLIB_STATUS_ABORTED = int32(4)

        %% Runtime log level threshold (NLolib/set_log_level)
        NLOLIB_LOG_LEVEL_ERROR = int32(0)
        NLOLIB_LOG_LEVEL_WARN = int32(1)
        NLOLIB_LOG_LEVEL_INFO = int32(2)
        NLOLIB_LOG_LEVEL_DEBUG = int32(3)

        %% Progress output stream (NLolib/set_progress_stream)
        NLOLIB_PROGRESS_STREAM_STDERR = int32(0)
        NLOLIB_PROGRESS_STREAM_STDOUT = int32(1)
        NLOLIB_PROGRESS_STREAM_BOTH = int32(2)

        %% Progress callback event kinds (ProgressInfo.event_type)
        PROGRESS_EVENT_ACCEPTED = int32(0)
        PROGRESS_EVENT_REJECTED = int32(1)
        PROGRESS_EVENT_FINISH = int32(2)
    end
end
