classdef NLolib < handle
    %NLOLIB High-level MATLAB wrapper around the nlolib C shared library.
    %   Uses loadlibrary/calllib to call nlolib_propagate() directly,
    %   with no Python dependency.
    %
    %   INSTALLATION
    %
    %   Option A — .mltbx (recommended)
    %     Download the latest nlolib.mltbx from GitHub Releases and
    %     double-click it, or run:
    %       matlab.addons.install('nlolib.mltbx');
    %
    %   Option B — Build from source
    %     1. Build the library and stage MATLAB files:
    %          cmake -S . -B build
    %          cmake --build build --config Release --target matlab_stage
    %     2. Add the staging directory to the MATLAB path:
    %          addpath('<repo>/build/matlab_toolbox');
    %          nlolib_setup();
    %
    %   The shared library (nlolib.dll / libnlolib.so) and the header
    %   nlolib_matlab.h must be reachable at runtime.  The wrapper
    %   searches common build-tree locations automatically; set the
    %   NLOLIB_LIBRARY environment variable to override.
    %
    %   PREREQUISITES
    %     - MATLAB R2019b or later.
    %     - A GPU driver with the Vulkan loader (ships with standard
    %       NVIDIA / AMD / Intel desktop drivers).
    %
    %   QUICK START
    %     api = nlolib.NLolib();
    %     cfg = struct(...);
    %     result = api.propagate(cfg, field0, numRecords);
    %
    %   See also: examples/matlab/runtime_temporal_demo.m

    properties (Constant, Access = private)
        LIBNAME = 'nlolib';
    end

    methods
        function obj = NLolib(libraryPath)
            %NLOLIB Construct wrapper and load the shared library.
            %   obj = nlolib.NLolib()          — auto-discover library
            %   obj = nlolib.NLolib(path)       — explicit library path
            if nargin < 1
                libraryPath = "";
            end

            if ~libisloaded(obj.LIBNAME)
                headerPath = nlolib.NLolib.resolve_header();
                dllCandidates = nlolib.NLolib.resolve_library_candidates(libraryPath);
                loaded = false;
                loadErrors = strings(0, 1);
                for idx = 1:numel(dllCandidates)
                    dllPath = dllCandidates{idx};
                    try
                        nlolib.NLolib.prepend_library_dir_to_path(dllPath);
                        [notfound, loadWarnings] = loadlibrary(dllPath, headerPath, ...
                                                               'alias', obj.LIBNAME);
                        nlolib.NLolib.validate_load_result(notfound, loadWarnings, ...
                                                           dllPath, headerPath);
                        loaded = true;
                        break;
                    catch ME
                        loadErrors(end + 1, 1) = string(dllPath) + " -> " + string(ME.message); %#ok<AGROW>
                        if libisloaded(obj.LIBNAME)
                            unloadlibrary(obj.LIBNAME);
                        end
                    end
                end
                if ~loaded
                    error('nlolib:libraryLoadFailed', ...
                          'Failed to load nlolib from candidates:\n  %s', ...
                          strjoin(cellstr(loadErrors), '\n  '));
                end
            end
            nlolib.NLolib.ensure_types_loaded();
        end

        function delete(~)
            % Destructor intentionally does NOT unload the library since
            % multiple NLolib instances may share the same loaded lib.
            % Call nlolib.NLolib.unload() explicitly if needed.
        end

        function result = propagate(obj, primary, varargin)
            %PROPAGATE Unified propagation entrypoint.
            %   Low-level:
            %     result = obj.propagate(cfg, field, numRecords)
            %     result = obj.propagate(cfg, field, numRecords, execOpts)
            %     result = obj.propagate(cfg, field, numRecords, execOpts, storageOpts)
            %   High-level:
            %     result = obj.propagate(pulse, linearOp, nonlinearOp, options)
            if nargin < 2
                error('nlolib:invalidPropagateCall', ...
                      'propagate requires at least a primary argument');
            end

            if nlolib.NLolib.looks_like_pulse_spec(primary)
                linearOperator = "gvd";
                nonlinearOperator = "kerr";
                options = struct();
                if numel(varargin) >= 1
                    linearOperator = varargin{1};
                end
                if numel(varargin) >= 2
                    nonlinearOperator = varargin{2};
                end
                if numel(varargin) >= 3
                    options = varargin{3};
                end
                if numel(varargin) > 3
                    error('nlolib:invalidPropagateCall', ...
                          'high-level propagate accepts at most 3 trailing arguments');
                end
                result = obj.propagate_high_level(primary, linearOperator, nonlinearOperator, options);
                return;
            end

            if numel(varargin) < 2
                error('nlolib:invalidPropagateCall', ...
                      'low-level propagate requires config, inputField, and numRecordedSamples');
            end
            config = primary;
            inputField = varargin{1};
            numRecordedSamples = varargin{2};
            execOptions = struct();
            storageOptions = struct();
            if numel(varargin) >= 3
                candidate = varargin{3};
                if isstruct(candidate) && isfield(candidate, 'sqlite_path')
                    storageOptions = candidate;
                else
                    execOptions = candidate;
                end
            end
            if numel(varargin) >= 4
                candidate = varargin{4};
                if ~(isstruct(candidate) && isfield(candidate, 'sqlite_path'))
                    error('nlolib:invalidPropagateCall', ...
                          'fifth low-level argument must be storage options');
                end
                storageOptions = candidate;
            end
            if numel(varargin) > 4
                error('nlolib:invalidPropagateCall', ...
                      'low-level propagate accepts at most 4 trailing arguments');
            end

            distance = nlolib.NLolib.resolve_propagation_distance(config);
            if ~isempty(fieldnames(storageOptions))
                [records, storageResult] = obj.propagate_low_level_with_storage( ...
                    config, inputField, numRecordedSamples, storageOptions, execOptions);
                storageEnabled = true;
            else
                records = obj.propagate_low_level_records(config, inputField, numRecordedSamples, execOptions);
                storageResult = struct();
                storageEnabled = false;
            end

            result = struct();
            result.records = records;
            if size(records, 1) > 0
                result.final = records(end, :);
            else
                result.final = [];
            end
            result.z_axis = nlolib.NLolib.make_z_axis(distance, double(numRecordedSamples));
            result.meta = struct();
            result.meta.output = char("dense");
            if double(numRecordedSamples) == 1
                result.meta.output = char("final");
            end
            result.meta.records = double(numRecordedSamples);
            result.meta.storage_enabled = logical(storageEnabled);
            result.meta.records_returned = size(records, 1) > 0;
            if isstruct(execOptions) && isfield(execOptions, 'backend_type')
                result.meta.backend_requested = double(execOptions.backend_type);
            else
                result.meta.backend_requested = 2;
            end
            result.meta.coupled = nlolib.NLolib.is_coupled_config(config);
            if storageEnabled
                result.meta.storage_result = storageResult;
            end
        end

        function records = propagate_low_level_records(obj, config, inputField, ...
                                                       numRecordedSamples, execOptions)
            %PROPAGATE_LOW_LEVEL_RECORDS Low-level propagation that returns records matrix.
            if nargin < 5
                execOptions = struct();
            end

            numTimeSamples = numel(inputField);
            numRecordedSamples = uint64(numRecordedSamples);

            % Pack sim_config and collect keepalive handles.
            [cfgPtr, keepalive] = nlolib.prepare_sim_config(config); %#ok<ASGLU>

            useInterleaved = nlolib.NLolib.has_library_function(obj.LIBNAME, ...
                                                                 'nlolib_propagate_interleaved');
            if useInterleaved
                inPtr = nlolib.pack_complex_interleaved_array(inputField);
            else
                inPtr = nlolib.pack_complex_array(inputField);
            end

            % Allocate output buffer.
            outLen = uint64(numTimeSamples) * numRecordedSamples;
            if useInterleaved
                outPtr = nlolib.pack_complex_interleaved_array(zeros(1, double(outLen)));
            else
                outPtr = nlolib.pack_complex_array(zeros(1, double(outLen)));
            end
            totalComplex = double(outLen);

            matlabDebug = false;
            if isstruct(execOptions) && isfield(execOptions, 'matlab_debug')
                rawDebug = logical(execOptions.matlab_debug);
                matlabDebug = any(rawDebug(:));
            end
            preProbe = struct();
            postProbe = struct();
            if matlabDebug
                preProbe = nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                          "pre-call", true);
            end

            % Build execution options with explicit typed defaults.
            if isempty(fieldnames(execOptions))
                execOptsPtr = nlolib.NLolib.make_exec_options(struct());
            else
                execOptsPtr = nlolib.NLolib.make_exec_options(execOptions);
            end

            streamLogs = false;
            streamLogBufferBytes = uint64(262144);
            if isstruct(execOptions) && isfield(execOptions, 'matlab_stream_logs')
                streamLogs = logical(execOptions.matlab_stream_logs);
                streamLogs = any(streamLogs(:));
            end
            if isstruct(execOptions) && isfield(execOptions, 'matlab_log_buffer_bytes')
                streamLogBufferBytes = uint64(execOptions.matlab_log_buffer_bytes);
            end
            if streamLogs && ...
                    nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_set_log_buffer') && ...
                    nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_clear_log_buffer')
                obj.set_log_buffer(streamLogBufferBytes);
                obj.clear_log_buffer();
            end

            if useInterleaved
                statusRaw = calllib(obj.LIBNAME, 'nlolib_propagate_interleaved', ...
                                    cfgPtr, ...
                                    uint64(numTimeSamples), ...
                                    inPtr, ...
                                    numRecordedSamples, ...
                                    outPtr, ...
                                    execOptsPtr);
            else
                statusRaw = calllib(obj.LIBNAME, 'nlolib_propagate', ...
                                    cfgPtr, ...
                                    uint64(numTimeSamples), ...
                                    inPtr, ...
                                    numRecordedSamples, ...
                                    outPtr, ...
                                    execOptsPtr);
            end
            if matlabDebug
                postProbe = nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                           "post-call", true);
            end

            streamedLogs = "";
            if streamLogs && nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_read_log_buffer')
                streamedLogs = obj.tail_logs(true, streamLogBufferBytes);
            end
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);

            if statusCode ~= 0
                if streamLogs && strlength(streamedLogs) > 0
                    statusDetail = statusDetail + newline + "runtime logs:" + newline + streamedLogs;
                end
                if matlabDebug
                    if isstruct(postProbe) && isfield(postProbe, 'value_size')
                        sizeText = join(string(double(postProbe.value_size(:).')), "x");
                    else
                        sizeText = "";
                    end
                    probeDetail = sprintf(['MATLAB probe post-call: ptr.class=%s ptr.datatype=%s ' ...
                                           'value.class=%s value.size=%s raw=%d re=%d im=%d expected=%g'], ...
                                          char(string(getfield_safe(postProbe, 'pointer_class', ""))), ...
                                          char(string(getfield_safe(postProbe, 'pointer_datatype', ""))), ...
                                          char(string(getfield_safe(postProbe, 'value_class', ""))), ...
                                          char(string(sizeText)), ...
                                          int64(getfield_safe(postProbe, 'raw_count', 0)), ...
                                          int64(getfield_safe(postProbe, 're_count', 0)), ...
                                          int64(getfield_safe(postProbe, 'im_count', 0)), ...
                                          double(getfield_safe(postProbe, 'expected_count', 0)));
                    if strlength(statusDetail) > 0
                        statusDetail = statusDetail + " | " + string(probeDetail);
                    else
                        statusDetail = string(probeDetail);
                    end
                end
                if strlength(statusDetail) > 0
                    error('nlolib:propagateFailed', ...
                          'nlolib_propagate failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:propagateFailed', ...
                          'nlolib_propagate failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end

            debugContext = struct('enabled', matlabDebug, ...
                                  'expected_count', totalComplex, ...
                                  'pre_probe', preProbe, ...
                                  'post_probe', postProbe);
            records = nlolib.unpack_records(outPtr, ...
                                            double(numRecordedSamples), ...
                                            numTimeSamples, ...
                                            debugContext);
        end

        function tf = storage_is_available(obj)
            %STORAGE_IS_AVAILABLE Return true when SQLite storage is compiled in.
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_storage_is_available')
                tf = false;
                return;
            end
            tf = logical(calllib(obj.LIBNAME, 'nlolib_storage_is_available'));
        end

        function set_log_file(obj, path, append)
            %SET_LOG_FILE Configure optional runtime file logging.
            if nargin < 3
                append = false;
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_set_log_file')
                error('nlolib:logUnavailable', ...
                      'nlolib_set_log_file is not available in this library build.');
            end

            if nargin < 2 || strlength(string(path)) == 0
                pathPtr = libpointer();
            else
                pathPtr = libpointer('cstring', char(string(path)));
            end

            statusRaw = calllib(obj.LIBNAME, 'nlolib_set_log_file', ...
                                pathPtr, int32(logical(append)));
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:logFileFailed', ...
                          'nlolib_set_log_file failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:logFileFailed', ...
                          'nlolib_set_log_file failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end
        end

        function set_log_buffer(obj, capacityBytes)
            %SET_LOG_BUFFER Configure in-memory runtime log ring buffer.
            if nargin < 2 || isempty(capacityBytes)
                capacityBytes = uint64(262144);
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_set_log_buffer')
                error('nlolib:logUnavailable', ...
                      'nlolib_set_log_buffer is not available in this library build.');
            end

            statusRaw = calllib(obj.LIBNAME, 'nlolib_set_log_buffer', uint64(capacityBytes));
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:logBufferFailed', ...
                          'nlolib_set_log_buffer failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:logBufferFailed', ...
                          'nlolib_set_log_buffer failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end
        end

        function clear_log_buffer(obj)
            %CLEAR_LOG_BUFFER Clear in-memory runtime log ring buffer.
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_clear_log_buffer')
                error('nlolib:logUnavailable', ...
                      'nlolib_clear_log_buffer is not available in this library build.');
            end

            statusRaw = calllib(obj.LIBNAME, 'nlolib_clear_log_buffer');
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:logBufferFailed', ...
                          'nlolib_clear_log_buffer failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:logBufferFailed', ...
                          'nlolib_clear_log_buffer failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end
        end

        function text = read_log_buffer(obj, consume, maxBytes)
            %READ_LOG_BUFFER Read buffered runtime logs.
            if nargin < 2
                consume = true;
            end
            if nargin < 3 || isempty(maxBytes)
                maxBytes = uint64(262144);
            end
            if uint64(maxBytes) < uint64(2)
                error('nlolib:invalidLogBufferRead', ...
                      'maxBytes must be >= 2');
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_read_log_buffer')
                error('nlolib:logUnavailable', ...
                      'nlolib_read_log_buffer is not available in this library build.');
            end

            outPtr = libpointer('uint8Ptr', zeros(1, double(maxBytes), 'uint8'));
            writtenPtr = libpointer('uint64Ptr', uint64(0));
            statusRaw = calllib(obj.LIBNAME, 'nlolib_read_log_buffer', ...
                                outPtr, uint64(maxBytes), writtenPtr, int32(logical(consume)));
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:logBufferReadFailed', ...
                          'nlolib_read_log_buffer failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:logBufferReadFailed', ...
                          'nlolib_read_log_buffer failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end

            byteCount = double(writtenPtr.Value);
            if byteCount <= 0
                text = "";
                return;
            end
            bytes = outPtr.Value(1:byteCount);
            text = string(native2unicode(bytes, 'UTF-8'));
        end

        function text = tail_logs(obj, consume, maxBytes)
            %TAIL_LOGS Print buffered runtime logs to MATLAB Command Window.
            if nargin < 2
                consume = true;
            end
            if nargin < 3
                maxBytes = uint64(262144);
            end
            text = obj.read_log_buffer(consume, maxBytes);
            if strlength(text) == 0
                return;
            end
            fprintf('%s', char(text));
            if ~endsWith(text, newline)
                fprintf('\n');
            end
        end

        function set_log_level(obj, level)
            %SET_LOG_LEVEL Configure runtime log level threshold.
            if nargin < 2 || isempty(level)
                level = int32(2);
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_set_log_level')
                error('nlolib:logUnavailable', ...
                      'nlolib_set_log_level is not available in this library build.');
            end
            statusRaw = calllib(obj.LIBNAME, 'nlolib_set_log_level', int32(level));
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:logLevelFailed', ...
                          'nlolib_set_log_level failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:logLevelFailed', ...
                          'nlolib_set_log_level failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end
        end

        function set_progress_options(obj, enabled, milestonePercent, emitOnStepAdjust)
            %SET_PROGRESS_OPTIONS Configure runtime progress log behavior.
            if nargin < 2 || isempty(enabled)
                enabled = true;
            end
            if nargin < 3 || isempty(milestonePercent)
                milestonePercent = int32(5);
            end
            if nargin < 4 || isempty(emitOnStepAdjust)
                emitOnStepAdjust = false;
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_set_progress_options')
                error('nlolib:logUnavailable', ...
                      'nlolib_set_progress_options is not available in this library build.');
            end
            statusRaw = calllib(obj.LIBNAME, 'nlolib_set_progress_options', ...
                                int32(logical(enabled)), ...
                                int32(milestonePercent), ...
                                int32(logical(emitOnStepAdjust)));
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:progressOptionsFailed', ...
                          'nlolib_set_progress_options failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:progressOptionsFailed', ...
                          'nlolib_set_progress_options failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end
        end

        function [records, storageResult] = propagate_low_level_with_storage( ...
                obj, config, inputField, numRecordedSamples, storageOptions, execOptions)
            %PROPAGATE_LOW_LEVEL_WITH_STORAGE Low-level propagation with SQLite storage.
            if nargin < 6
                execOptions = struct();
            end
            if nargin < 5 || isempty(storageOptions) || ~isstruct(storageOptions)
                error('nlolib:invalidStorageOptions', 'storageOptions must be a struct');
            end
            if ~isfield(storageOptions, 'sqlite_path') || strlength(string(storageOptions.sqlite_path)) == 0
                error('nlolib:invalidStorageOptions', ...
                      'storageOptions.sqlite_path is required');
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_propagate_with_storage')
                error('nlolib:storageUnavailable', ...
                      'nlolib_propagate_with_storage is unavailable in this library build.');
            end
            if ~obj.storage_is_available()
                error('nlolib:storageUnavailable', ...
                      'SQLite storage is not available in this nlolib build.');
            end

            returnRecords = true;
            if isfield(storageOptions, 'return_records') && ~isempty(storageOptions.return_records)
                returnRecords = logical(storageOptions.return_records);
            end

            numTimeSamples = numel(inputField);
            numRecordedSamples = uint64(numRecordedSamples);

            [cfgPtr, keepalive] = nlolib.prepare_sim_config(config); %#ok<ASGLU>
            inPtr = nlolib.pack_complex_array(inputField);

            if returnRecords
                outLen = uint64(numTimeSamples) * numRecordedSamples;
                outPtr = nlolib.pack_complex_array(zeros(1, double(outLen)));
            else
                outPtr = libpointer('nlo_complexPtr');
            end

            if isempty(fieldnames(execOptions))
                execOptsPtr = nlolib.NLolib.make_exec_options(struct());
            else
                execOptsPtr = nlolib.NLolib.make_exec_options(execOptions);
            end

            streamLogs = false;
            streamLogBufferBytes = uint64(262144);
            if isstruct(execOptions) && isfield(execOptions, 'matlab_stream_logs')
                streamLogs = logical(execOptions.matlab_stream_logs);
                streamLogs = any(streamLogs(:));
            end
            if isstruct(execOptions) && isfield(execOptions, 'matlab_log_buffer_bytes')
                streamLogBufferBytes = uint64(execOptions.matlab_log_buffer_bytes);
            end
            if streamLogs && ...
                    nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_set_log_buffer') && ...
                    nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_clear_log_buffer')
                obj.set_log_buffer(streamLogBufferBytes);
                obj.clear_log_buffer();
            end

            [storageOptsPtr, storageKeepalive] = ...
                nlolib.NLolib.make_storage_options(storageOptions); %#ok<ASGLU>
            storageResultRaw = libstruct('nlo_storage_result');

            statusRaw = calllib(obj.LIBNAME, 'nlolib_propagate_with_storage', ...
                                cfgPtr, ...
                                uint64(numTimeSamples), ...
                                inPtr, ...
                                numRecordedSamples, ...
                                outPtr, ...
                                execOptsPtr, ...
                                storageOptsPtr, ...
                                storageResultRaw);

            streamedLogs = "";
            if streamLogs && nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_read_log_buffer')
                streamedLogs = obj.tail_logs(true, streamLogBufferBytes);
            end
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if streamLogs && strlength(streamedLogs) > 0
                    statusDetail = statusDetail + newline + "runtime logs:" + newline + streamedLogs;
                end
                if strlength(statusDetail) > 0
                    error('nlolib:propagateWithStorageFailed', ...
                          'nlolib_propagate_with_storage failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:propagateWithStorageFailed', ...
                          'nlolib_propagate_with_storage failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end

            if returnRecords
                records = nlolib.unpack_records(outPtr, ...
                                                double(numRecordedSamples), ...
                                                numTimeSamples);
            else
                records = zeros(0, numTimeSamples);
            end
            storageResult = nlolib.NLolib.to_storage_result_struct(storageResultRaw);
        end

        function result = propagate_high_level(obj, pulse, linearOperator, nonlinearOperator, options)
            %PROPAGATE_HIGH_LEVEL High-level pulse/operator facade with balanced defaults.
            %   result = obj.propagate_high_level(pulse, linearOperator, nonlinearOperator, options)
            %
            %   pulse.samples    : complex input field (required)
            %   pulse.delta_time : temporal spacing (required)
            %   options.propagation_distance : propagation distance (required)
            %
            %   result.records : dense record-major output matrix
            %   result.z_axis  : z locations of returned records
            %   result.final   : final output field (last row of records)
            %   result.meta    : request metadata
            if nargin < 5
                options = struct();
            end

            [cfg, inputField, numRecords, execOptions, zAxis, meta] = ...
                nlolib.NLolib.build_simulation_request(pulse, linearOperator, nonlinearOperator, options);
            storageOptions = struct();
            if isfield(options, 'storage') && ~isempty(options.storage)
                if ~isstruct(options.storage)
                    error('nlolib:invalidStorageOptions', ...
                          'options.storage must be a struct');
                end
                storageOptions = options.storage;
            end

            if ~isempty(fieldnames(storageOptions))
                [records, storageResult] = obj.propagate_low_level_with_storage( ...
                    cfg, inputField, numRecords, storageOptions, execOptions);
                meta.storage_enabled = true;
                meta.storage_result = storageResult;
            else
                records = obj.propagate_low_level_records(cfg, inputField, numRecords, execOptions);
                meta.storage_enabled = false;
            end
            meta.records_returned = size(records, 1) > 0;

            result = struct();
            result.records = records;
            result.z_axis = zAxis;
            if size(records, 1) > 0
                result.final = records(end, :);
            else
                result.final = [];
            end
            result.meta = meta;
        end

        function limits = query_runtime_limits(obj, config, execOptions)
            %QUERY_RUNTIME_LIMITS Query runtime-derived solver limits.
            if nargin < 2
                config = struct( ...
                    'num_time_samples', 1, ...
                    'propagation_distance', 0.0, ...
                    'starting_step_size', 1e-3, ...
                    'max_step_size', 1e-2, ...
                    'min_step_size', 1e-5, ...
                    'error_tolerance', 1e-6, ...
                    'pulse_period', 1.0, ...
                    'delta_time', 1.0, ...
                    'frequency_grid', complex(0.0, 0.0));
            end
            if nargin < 3
                execOptions = struct();
            end
            if ~nlolib.NLolib.has_library_function(obj.LIBNAME, 'nlolib_query_runtime_limits')
                error('nlolib:runtimeLimitsUnavailable', ...
                      'nlolib_query_runtime_limits is not available in this library build.');
            end

            [cfgPtr, keepalive] = nlolib.prepare_sim_config(config); %#ok<ASGLU>
            execOptsPtr = nlolib.NLolib.make_exec_options(execOptions);
            out = libstruct('nlo_runtime_limits');
            statusRaw = calllib(obj.LIBNAME, 'nlolib_query_runtime_limits', ...
                                cfgPtr, execOptsPtr, out);
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);
            if statusCode ~= 0
                if strlength(statusDetail) > 0
                    error('nlolib:runtimeLimitsFailed', ...
                          'nlolib_query_runtime_limits failed with status=%d (%s). %s', ...
                          statusCode, statusName, statusDetail);
                else
                    error('nlolib:runtimeLimitsFailed', ...
                          'nlolib_query_runtime_limits failed with status=%d (%s)', ...
                          statusCode, statusName);
                end
            end
            limits = struct(out);
        end
    end

    methods (Static)
        function tf = looks_like_pulse_spec(candidate)
            tf = isstruct(candidate) && ...
                 isfield(candidate, 'samples') && ...
                 isfield(candidate, 'delta_time') && ...
                 ~isfield(candidate, 'num_time_samples');
        end

        function tf = is_coupled_config(config)
            tf = false;
            if ~isstruct(config)
                return;
            end
            if isfield(config, 'spatial_nx') && isfield(config, 'spatial_ny')
                tf = (double(config.spatial_nx) > 1) || (double(config.spatial_ny) > 1);
                return;
            end
            if isfield(config, 'spatial') && isstruct(config.spatial) && ...
                    isfield(config.spatial, 'nx') && isfield(config.spatial, 'ny')
                tf = (double(config.spatial.nx) > 1) || (double(config.spatial.ny) > 1);
            end
        end

        function zAxis = make_z_axis(distance, numRecords)
            if numRecords <= 1
                zAxis = distance;
                return;
            end
            zAxis = linspace(0.0, distance, numRecords);
        end

        function distance = resolve_propagation_distance(config)
            distance = 0.0;
            if ~isstruct(config)
                return;
            end
            if isfield(config, 'propagation_distance')
                distance = double(config.propagation_distance);
                return;
            end
            if isfield(config, 'propagation') && isstruct(config.propagation) && ...
                    isfield(config.propagation, 'propagation_distance')
                distance = double(config.propagation.propagation_distance);
            end
        end

        function unload()
            %UNLOAD Explicitly unload the nlolib shared library.
            if libisloaded('nlolib')
                unloadlibrary('nlolib');
            end
        end

        function ensure_types_loaded()
            %ENSURE_TYPES_LOADED Validate MATLAB can resolve required types.
            if ~libisloaded('nlolib')
                error('nlolib:notLoaded', ...
                      'Library not loaded. Create an nlolib.NLolib instance first.');
            end

            try
                s = libstruct('sim_config');
            catch ME
                error('nlolib:simConfigTypeUnavailable', ...
                      ['Failed to resolve C type ''sim_config'' via libstruct().\n' ...
                       'This usually means MATLAB loaded an incompatible header/library pair,\n' ...
                       'or multiple nlolib package copies are on the MATLAB path.\n' ...
                       'Remediation:\n' ...
                       '  1) nlolib.NLolib.unload();\n' ...
                       '  2) clear classes;\n' ...
                       '  3) verify only one nlolib package is on path;\n' ...
                       '  4) recreate api = nlolib.NLolib();\n\n' ...
                       'Original error: %s'], ME.message);
            end

            try
                names = fieldnames(s);
            catch
                names = {};
            end
            if isempty(names)
                error('nlolib:simConfigTypeUnavailable', ...
                      ['MATLAB resolved an invalid/empty ''sim_config'' definition.\n' ...
                       'This can happen after loadlibrary parser warnings or stale prototypes.\n' ...
                       'Run:\n' ...
                       '  nlolib.NLolib.unload();\n' ...
                       '  clear classes;\n' ...
                       'Then recreate nlolib.NLolib().']);
            end

            % Probe nested sub-struct assignment; some parser modes leave
            % nested fields as empty placeholders.
            try
                probe = libstruct('sim_config');
                prop = libstruct('propagation_params');
                set(prop, 'starting_step_size', 0.0);
                set(probe, 'propagation', prop); %#ok<NASGU>
            catch ME
                error('nlolib:simConfigTypeUnavailable', ...
                      ['MATLAB loaded sim_config but nested type assignment failed.\n' ...
                       'This indicates an incompatible parsed type table.\n' ...
                       'Run:\n' ...
                       '  nlolib.NLolib.unload();\n' ...
                       '  clear classes;\n' ...
                       'Then recreate nlolib.NLolib().\n\n' ...
                       'Original error: %s'], ME.message);
            end

            try
                fg = libstruct('nlo_frequency_grid', ...
                               struct('frequency_grid', struct('re', 0.0, 'im', 0.0))); %#ok<NASGU>
            catch ME
                error('nlolib:simConfigTypeUnavailable', ...
                      ['MATLAB failed to construct nlo_frequency_grid.\n' ...
                       'This points to a header parse mismatch for frequency pointer fields.\n' ...
                       'Run:\n' ...
                       '  nlolib.NLolib.unload();\n' ...
                       '  clear classes;\n' ...
                       'Then recreate nlolib.NLolib().\n\n' ...
                       'Original error: %s'], ME.message);
            end

            try
                maxConstants = 16;   % NLO_RUNTIME_OPERATOR_CONSTANTS_MAX
                rt = libstruct('runtime_operator_params', ...
                               struct('dispersion_factor_expr', 'a', ...
                                      'dispersion_expr', 'w', ...
                                      'transverse_factor_expr', 'w', ...
                                      'transverse_expr', 'a', ...
                                      'nonlinear_expr', 'a', ...
                                      'num_constants', uint64(0), ...
                                      'constants', zeros(1, maxConstants))); %#ok<NASGU>
            catch ME
                error('nlolib:simConfigTypeUnavailable', ...
                      ['MATLAB failed to construct runtime_operator_params with char fields.\n' ...
                       'This points to a header parse mismatch for const char* fields.\n' ...
                       'Run:\n' ...
                       '  nlolib.NLolib.unload();\n' ...
                       '  clear classes;\n' ...
                       'Then recreate nlolib.NLolib().\n\n' ...
                       'Original error: %s'], ME.message);
            end
        end
    end

    methods (Static, Access = private)
        function headerPath = resolve_header()
            %RESOLVE_HEADER Locate nlolib_matlab.h relative to this file.
            roots = nlolib.NLolib.resolve_roots();
            candidates = {};
            for idx = 1:numel(roots)
                root = roots{idx};
                candidates{end + 1} = fullfile(root, 'lib', 'nlolib_matlab.h'); %#ok<AGROW>
                candidates{end + 1} = fullfile(root, 'src', 'nlolib_matlab.h'); %#ok<AGROW>
                candidates{end + 1} = fullfile(root, 'build', 'matlab_toolbox', ...
                                               'lib', 'nlolib_matlab.h'); %#ok<AGROW>
            end
            candidates = unique(candidates, 'stable');

            for idx = 1:numel(candidates)
                if isfile(candidates{idx})
                    headerPath = candidates{idx};
                    return;
                end
            end
            error('nlolib:headerNotFound', ...
                  'Cannot locate nlolib_matlab.h. Searched:\n  %s', ...
                  strjoin(candidates, '\n  '));
        end

        function dllPath = resolve_library(userPath)
            %RESOLVE_LIBRARY Locate the nlolib shared library.
            candidates = nlolib.NLolib.resolve_library_candidates(userPath);
            dllPath = candidates{1};
        end

        function candidates = resolve_library_candidates(userPath)
            %RESOLVE_LIBRARY_CANDIDATES Locate candidate nlolib shared libraries.
            if strlength(string(userPath)) > 0 && isfile(userPath)
                candidates = {char(userPath)};
                return;
            end

            envPath = getenv('NLOLIB_LIBRARY');
            if ~isempty(envPath) && isfile(envPath)
                candidates = {envPath};
                return;
            end

            roots = nlolib.NLolib.resolve_roots();
            if ispc
                libFile = 'nlolib.dll';
                searchTail = { ...
                    fullfile('lib'), ...
                    fullfile('lib', 'win64'), ...
                    fullfile('build', 'matlab_toolbox', 'lib'), ...
                    fullfile('build', 'src', 'Release'), ...
                    fullfile('build', 'src', 'Debug'), ...
                    fullfile('build', 'src', 'RelWithDebInfo'), ...
                    fullfile('build', 'src'), ...
                    fullfile('src', 'Release'), ...
                    fullfile('src', 'Debug'), ...
                    fullfile('src', 'RelWithDebInfo'), ...
                    fullfile('src'), ...
                    fullfile('python', 'Release'), ...
                    fullfile('python', 'Debug'), ...
                    fullfile('python') ...
                };
            elseif ismac
                libFile = 'libnlolib.dylib';
                searchTail = { ...
                    fullfile('lib'), ...
                    fullfile('lib', 'maci64'), ...
                    fullfile('build', 'matlab_toolbox', 'lib'), ...
                    fullfile('build', 'src'), ...
                    fullfile('src'), ...
                    fullfile('python') ...
                };
            else
                libFile = 'libnlolib.so';
                searchTail = { ...
                    fullfile('lib'), ...
                    fullfile('lib', 'glnxa64'), ...
                    fullfile('build', 'matlab_toolbox', 'lib'), ...
                    fullfile('build', 'src'), ...
                    fullfile('src'), ...
                    fullfile('python') ...
                };
            end

            searchDirs = {};
            for idx = 1:numel(roots)
                root = roots{idx};
                for jdx = 1:numel(searchTail)
                    searchDirs{end + 1} = fullfile(root, searchTail{jdx}); %#ok<AGROW>
                end
            end
            searchDirs = unique(searchDirs, 'stable');

            foundPaths = {};
            foundDatenum = [];
            for idx = 1:numel(searchDirs)
                candidate = fullfile(searchDirs{idx}, libFile);
                if isfile(candidate)
                    info = dir(candidate);
                    foundPaths{end + 1} = candidate; %#ok<AGROW>
                    foundDatenum(end + 1) = info.datenum; %#ok<AGROW>
                end
            end
            if ~isempty(foundPaths)
                [~, order] = sort(foundDatenum, 'descend');
                candidates = foundPaths(order);
                return;
            end
            error('nlolib:libraryNotFound', ...
                  'Cannot locate %s. Set NLOLIB_LIBRARY or pass path.', ...
                  libFile);
        end

        function prepend_library_dir_to_path(dllPath)
            %PREPEND_LIBRARY_DIR_TO_PATH Ensure DLL folder is discoverable on Windows.
            if ~ispc
                return;
            end
            dllDir = fileparts(char(string(dllPath)));
            if strlength(string(dllDir)) == 0 || ~isfolder(dllDir)
                return;
            end
            currentPath = string(getenv('PATH'));
            parts = split(currentPath, ';');
            normParts = lower(strtrim(parts));
            if any(normParts == lower(string(dllDir)))
                return;
            end
            if strlength(currentPath) > 0
                setenv('PATH', char(string(dllDir) + ";" + currentPath));
            else
                setenv('PATH', char(string(dllDir)));
            end
        end

        function tf = has_library_function(libName, functionName)
            tf = false;
            try
                names = libfunctions(libName);
                tf = any(strcmp(names, functionName));
            catch
                tf = false;
            end
        end

        function ptr = make_exec_options(opts)
            %MAKE_EXEC_OPTIONS Build a nlo_execution_options libstruct.
            s = libstruct('nlo_execution_options');

            if isfield(opts, 'backend_type')
                s.backend_type = int32(opts.backend_type);
            else
                s.backend_type = int32(2);   % AUTO (parity with Python wrapper default)
            end
            if isfield(opts, 'fft_backend')
                s.fft_backend = int32(opts.fft_backend);
            else
                s.fft_backend = int32(0);    % AUTO
            end
            if isfield(opts, 'device_heap_fraction')
                s.device_heap_fraction = double(opts.device_heap_fraction);
            else
                s.device_heap_fraction = 0.70;
            end
            if isfield(opts, 'record_ring_target')
                s.record_ring_target = uint64(opts.record_ring_target);
            else
                s.record_ring_target = uint64(0);
            end
            if isfield(opts, 'forced_device_budget_bytes')
                s.forced_device_budget_bytes = uint64(opts.forced_device_budget_bytes);
            else
                s.forced_device_budget_bytes = uint64(0);
            end

            % Zero-initialise Vulkan sub-struct (users won't set this from
            % MATLAB; auto-detection in the C library handles it).
            s.vulkan.physical_device       = libpointer();
            s.vulkan.device                = libpointer();
            s.vulkan.queue                 = libpointer();
            s.vulkan.queue_family_index    = uint32(0);
            s.vulkan.command_pool          = libpointer();
            s.vulkan.descriptor_set_budget_bytes  = uint64(0);
            s.vulkan.descriptor_set_count_override = uint32(0);

            ptr = s;
        end

        function [opts, keepalive] = make_storage_options(storage)
            %MAKE_STORAGE_OPTIONS Build nlo_storage_options libstruct.
            keepalive = {};
            if ~isstruct(storage)
                error('nlolib:invalidStorageOptions', 'storage options must be a struct');
            end
            if ~isfield(storage, 'sqlite_path') || strlength(string(storage.sqlite_path)) == 0
                error('nlolib:invalidStorageOptions', ...
                      'storage.sqlite_path must be a non-empty string');
            end

            opts = libstruct('nlo_storage_options');
            sqlitePathPtr = libpointer('cstring', char(string(storage.sqlite_path)));
            keepalive{end + 1} = sqlitePathPtr; %#ok<AGROW>
            opts.sqlite_path = sqlitePathPtr;

            if isfield(storage, 'run_id') && strlength(string(storage.run_id)) > 0
                runIdPtr = libpointer('cstring', char(string(storage.run_id)));
                keepalive{end + 1} = runIdPtr; %#ok<AGROW>
                opts.run_id = runIdPtr;
            else
                opts.run_id = libpointer();
            end

            opts.sqlite_max_bytes = uint64(getfield_safe(storage, 'sqlite_max_bytes', 0));
            opts.chunk_records = uint64(getfield_safe(storage, 'chunk_records', 0));
            opts.cap_policy = int32(getfield_safe(storage, 'cap_policy', 0));
            opts.log_final_output_field_to_db = int32(logical(getfield_safe(storage, ...
                                                                              'log_final_output_field_to_db', ...
                                                                              false)));
        end

        function out = to_storage_result_struct(storageResultRaw)
            %TO_STORAGE_RESULT_STRUCT Convert nlo_storage_result to MATLAB struct.
            out = struct();
            runIdRaw = char(storageResultRaw.run_id(:).');
            nul = find(runIdRaw == char(0), 1);
            if ~isempty(nul)
                runIdRaw = runIdRaw(1:(nul - 1));
            end
            out.run_id = string(runIdRaw);
            out.records_captured = double(storageResultRaw.records_captured);
            out.records_spilled = double(storageResultRaw.records_spilled);
            out.chunks_written = double(storageResultRaw.chunks_written);
            out.db_size_bytes = double(storageResultRaw.db_size_bytes);
            out.truncated = logical(storageResultRaw.truncated);
        end

        function [cfg, inputField, numRecords, execOptions, zAxis, meta] = ...
                build_simulation_request(pulse, linearOperator, nonlinearOperator, options)
            if ~isstruct(options)
                error('nlolib:invalidSimulateOptions', 'options must be a struct');
            end
            if ~isfield(options, 'propagation_distance')
                error('nlolib:invalidSimulateOptions', ...
                      'options.propagation_distance is required');
            end

            zEnd = double(options.propagation_distance);
            if zEnd <= 0.0
                error('nlolib:invalidSimulateOptions', ...
                      'options.propagation_distance must be > 0');
            end

            preset = "balanced";
            if isfield(options, 'preset') && ~isempty(options.preset)
                preset = lower(string(options.preset));
            end
            defaults = nlolib.NLolib.solver_defaults(preset, zEnd);

            output = "dense";
            if isfield(options, 'output') && ~isempty(options.output)
                output = lower(string(options.output));
            end

            numRecords = defaults.records;
            if isfield(options, 'records') && ~isempty(options.records)
                numRecords = double(options.records);
            end
            if output == "final"
                numRecords = 1;
            elseif output ~= "dense"
                error('nlolib:invalidSimulateOptions', ...
                      'options.output must be ''dense'' or ''final''');
            end
            if ~isscalar(numRecords) || numRecords <= 0 || floor(numRecords) ~= numRecords
                error('nlolib:invalidSimulateOptions', ...
                      'options.records must be a positive integer');
            end
            numRecords = uint64(numRecords);

            if isfield(options, 'exec_options') && ~isempty(options.exec_options)
                execOptions = options.exec_options;
            elseif isfield(options, 'execOptions') && ~isempty(options.execOptions)
                execOptions = options.execOptions;
            else
                execOptions = struct();
            end

            pulseSpec = nlolib.NLolib.normalize_pulse_spec(pulse);
            inputField = pulseSpec.samples;
            numTimeSamples = numel(inputField);

            transverseOperator = "none";
            if isfield(options, 'transverse_operator') && ~isempty(options.transverse_operator)
                transverseOperator = options.transverse_operator;
            elseif isfield(options, 'transverseOperator') && ~isempty(options.transverseOperator)
                transverseOperator = options.transverseOperator;
            end
            transverseRequested = ~(ischar(transverseOperator) || isstring(transverseOperator)) || ...
                                  ~strcmpi(strtrim(char(string(transverseOperator))), 'none');
            if transverseRequested
                nlolib.NLolib.validate_coupled_pulse_spec(pulseSpec);
            end

            [linearExpr, linearFn, linearConstants] = ...
                nlolib.NLolib.resolve_operator_spec('linear', linearOperator, 0);
            if transverseRequested
                [transverseExpr, transverseFn, transverseConstants] = ...
                    nlolib.NLolib.resolve_operator_spec('transverse', transverseOperator, numel(linearConstants));
            else
                transverseExpr = "";
                transverseFn = [];
                transverseConstants = [];
            end
            [nonlinearExpr, nonlinearFn, nonlinearConstants] = ...
                nlolib.NLolib.resolve_operator_spec('nonlinear', nonlinearOperator, ...
                                                    numel(linearConstants) + numel(transverseConstants));

            runtime = struct();
            if strlength(linearExpr) > 0
                runtime.dispersion_factor_expr = char(linearExpr);
            end
            if ~isempty(linearFn)
                runtime.dispersion_factor_fn = linearFn;
            end
            if strlength(transverseExpr) > 0
                runtime.transverse_factor_expr = char(transverseExpr);
                runtime.transverse_expr = 'exp(h*D)';
            end
            if ~isempty(transverseFn)
                runtime.transverse_factor_fn = transverseFn;
                runtime.transverse_expr = 'exp(h*D)';
            end
            if strlength(nonlinearExpr) > 0
                runtime.nonlinear_expr = char(nonlinearExpr);
            end
            if ~isempty(nonlinearFn)
                runtime.nonlinear_fn = nonlinearFn;
            end
            runtime.constants = [linearConstants, transverseConstants, nonlinearConstants];

            cfg = struct();
            cfg.num_time_samples = numTimeSamples;
            cfg.propagation_distance = zEnd;
            cfg.starting_step_size = defaults.starting_step_size;
            cfg.max_step_size = defaults.max_step_size;
            cfg.min_step_size = defaults.min_step_size;
            cfg.error_tolerance = defaults.error_tolerance;
            cfg.delta_time = pulseSpec.delta_time;
            cfg.pulse_period = pulseSpec.pulse_period;
            cfg.frequency_grid = pulseSpec.frequency_grid;
            if ~isempty(pulseSpec.time_nt)
                cfg.time_nt = pulseSpec.time_nt;
            end
            if ~isempty(pulseSpec.spatial_nx)
                cfg.spatial_nx = pulseSpec.spatial_nx;
            end
            if ~isempty(pulseSpec.spatial_ny)
                cfg.spatial_ny = pulseSpec.spatial_ny;
            end
            if ~isempty(pulseSpec.spatial_frequency_grid)
                cfg.spatial_frequency_grid = pulseSpec.spatial_frequency_grid;
            end
            if ~isempty(pulseSpec.potential_grid)
                cfg.potential_grid = pulseSpec.potential_grid;
            end
            cfg.delta_x = pulseSpec.delta_x;
            cfg.delta_y = pulseSpec.delta_y;
            cfg.runtime = runtime;

            if double(numRecords) == 1
                zAxis = zEnd;
            else
                zAxis = linspace(0.0, zEnd, double(numRecords));
            end

            meta = struct();
            meta.preset = char(preset);
            meta.output = char(output);
            meta.records = double(numRecords);
            meta.coupled = logical(transverseRequested);
        end

        function pulse = normalize_pulse_spec(pulseInput)
            if ~isstruct(pulseInput)
                error('nlolib:invalidPulseSpec', 'pulse must be a struct');
            end
            if ~isfield(pulseInput, 'samples') || ~isfield(pulseInput, 'delta_time')
                error('nlolib:invalidPulseSpec', ...
                      'pulse must define samples and delta_time');
            end

            samples = pulseInput.samples;
            if isempty(samples)
                error('nlolib:invalidPulseSpec', 'pulse.samples must be non-empty');
            end

            deltaTime = double(pulseInput.delta_time);
            if deltaTime <= 0.0
                error('nlolib:invalidPulseSpec', 'pulse.delta_time must be > 0');
            end

            n = numel(samples);
            temporalSamples = n;
            if isfield(pulseInput, 'time_nt') && ~isempty(pulseInput.time_nt)
                temporalSamples = double(pulseInput.time_nt);
                if temporalSamples <= 0 || floor(temporalSamples) ~= temporalSamples
                    error('nlolib:invalidPulseSpec', ...
                          'pulse.time_nt must be a positive integer');
                end
            end
            pulse = struct();
            pulse.samples = reshape(samples, 1, n);
            pulse.delta_time = deltaTime;
            if isfield(pulseInput, 'pulse_period') && ~isempty(pulseInput.pulse_period)
                pulse.pulse_period = double(pulseInput.pulse_period);
            else
                pulse.pulse_period = double(temporalSamples) * deltaTime;
            end

            if isfield(pulseInput, 'frequency_grid') && ~isempty(pulseInput.frequency_grid)
                pulse.frequency_grid = pulseInput.frequency_grid;
            else
                pulse.frequency_grid = nlolib.NLolib.default_frequency_grid(temporalSamples, deltaTime);
            end

            pulse.time_nt = [];
            if isfield(pulseInput, 'time_nt') && ~isempty(pulseInput.time_nt)
                pulse.time_nt = double(pulseInput.time_nt);
            end

            pulse.spatial_nx = [];
            if isfield(pulseInput, 'spatial_nx') && ~isempty(pulseInput.spatial_nx)
                pulse.spatial_nx = double(pulseInput.spatial_nx);
            end
            pulse.spatial_ny = [];
            if isfield(pulseInput, 'spatial_ny') && ~isempty(pulseInput.spatial_ny)
                pulse.spatial_ny = double(pulseInput.spatial_ny);
            end

            pulse.delta_x = 1.0;
            if isfield(pulseInput, 'delta_x') && ~isempty(pulseInput.delta_x)
                pulse.delta_x = double(pulseInput.delta_x);
            end
            pulse.delta_y = 1.0;
            if isfield(pulseInput, 'delta_y') && ~isempty(pulseInput.delta_y)
                pulse.delta_y = double(pulseInput.delta_y);
            end

            pulse.spatial_frequency_grid = [];
            if isfield(pulseInput, 'spatial_frequency_grid') && ~isempty(pulseInput.spatial_frequency_grid)
                pulse.spatial_frequency_grid = pulseInput.spatial_frequency_grid;
            end
            pulse.potential_grid = [];
            if isfield(pulseInput, 'potential_grid') && ~isempty(pulseInput.potential_grid)
                pulse.potential_grid = pulseInput.potential_grid;
            end
        end

        function validate_coupled_pulse_spec(pulse)
            if ~isfield(pulse, 'time_nt') || isempty(pulse.time_nt) || double(pulse.time_nt) <= 0
                error('nlolib:invalidPulseSpec', ...
                      'pulse.time_nt must be > 0 for coupled transverse simulations');
            end
            if ~isfield(pulse, 'spatial_nx') || isempty(pulse.spatial_nx) || double(pulse.spatial_nx) <= 0
                error('nlolib:invalidPulseSpec', ...
                      'pulse.spatial_nx must be > 0 for coupled transverse simulations');
            end
            if ~isfield(pulse, 'spatial_ny') || isempty(pulse.spatial_ny) || double(pulse.spatial_ny) <= 0
                error('nlolib:invalidPulseSpec', ...
                      'pulse.spatial_ny must be > 0 for coupled transverse simulations');
            end

            nt = double(pulse.time_nt);
            nx = double(pulse.spatial_nx);
            ny = double(pulse.spatial_ny);
            expected = nt * nx * ny;
            if numel(pulse.samples) ~= expected
                error('nlolib:invalidPulseSpec', ...
                      'numel(pulse.samples) must equal pulse.time_nt * pulse.spatial_nx * pulse.spatial_ny');
            end

            if ~isfield(pulse, 'spatial_frequency_grid') || isempty(pulse.spatial_frequency_grid)
                error('nlolib:invalidPulseSpec', ...
                      'pulse.spatial_frequency_grid is required for coupled transverse simulations');
            end
            xy = nx * ny;
            sfLen = numel(pulse.spatial_frequency_grid);
            if sfLen ~= xy && sfLen ~= expected
                error('nlolib:invalidPulseSpec', ...
                      'spatial_frequency_grid length must equal spatial_nx*spatial_ny or full-volume size');
            end
            if isfield(pulse, 'potential_grid') && ~isempty(pulse.potential_grid)
                potentialLen = numel(pulse.potential_grid);
                if potentialLen ~= xy && potentialLen ~= expected
                    error('nlolib:invalidPulseSpec', ...
                          'potential_grid length must equal spatial_nx*spatial_ny or full-volume size');
                end
            end
        end

        function defaults = solver_defaults(preset, propagationDistance)
            switch char(lower(string(preset)))
                case 'balanced'
                    defaults = struct( ...
                        'starting_step_size', propagationDistance / 200.0, ...
                        'max_step_size', propagationDistance / 25.0, ...
                        'min_step_size', propagationDistance / 20000.0, ...
                        'error_tolerance', 1e-6, ...
                        'records', 128);
                case 'fast'
                    defaults = struct( ...
                        'starting_step_size', propagationDistance / 120.0, ...
                        'max_step_size', propagationDistance / 12.0, ...
                        'min_step_size', propagationDistance / 4000.0, ...
                        'error_tolerance', 5e-6, ...
                        'records', 64);
                case 'accuracy'
                    defaults = struct( ...
                        'starting_step_size', propagationDistance / 400.0, ...
                        'max_step_size', propagationDistance / 50.0, ...
                        'min_step_size', propagationDistance / 80000.0, ...
                        'error_tolerance', 1e-7, ...
                        'records', 192);
                otherwise
                    error('nlolib:invalidSimulatePreset', ...
                          'unsupported simulate preset ''%s''', char(string(preset)));
            end
        end

        function omega = default_frequency_grid(numTimeSamples, deltaTime)
            twoPi = 2.0 * pi;
            omegaStep = twoPi / (double(numTimeSamples) * deltaTime);
            positiveLimit = floor((double(numTimeSamples) - 1.0) / 2.0);
            omega = zeros(1, numTimeSamples);
            for idx = 1:numTimeSamples
                i = idx - 1;
                if i <= positiveLimit
                    omega(idx) = double(i) * omegaStep;
                else
                    omega(idx) = -(double(numTimeSamples - i) * omegaStep);
                end
            end
            omega = complex(omega, 0.0);
        end

        function [expr, fn, constants] = resolve_operator_spec(context, operator, offset)
            expr = "";
            fn = [];
            constants = [];

            params = [];
            if ischar(operator) || isstring(operator)
                expr = string(operator);
            elseif isstruct(operator)
                if isfield(operator, 'expr') && ~isempty(operator.expr)
                    expr = string(operator.expr);
                end
                if isfield(operator, 'fn') && ~isempty(operator.fn)
                    fn = operator.fn;
                end
                if isfield(operator, 'params') && ~isempty(operator.params)
                    params = operator.params;
                end
            else
                error('nlolib:invalidOperatorSpec', ...
                      '%s operator must be a preset string or struct', context);
            end

            if strlength(expr) > 0 && ~isempty(fn)
                error('nlolib:invalidOperatorSpec', ...
                      '%s operator cannot define both expr and fn', context);
            end

            if strlength(expr) > 0 && isempty(fn)
                [presetExpr, presetParams, isPreset] = nlolib.NLolib.operator_preset(context, expr);
                if isPreset
                    expr = presetExpr;
                    if isempty(params)
                        params = presetParams;
                    end
                end
            end

            if strlength(expr) == 0 && isempty(fn)
                error('nlolib:invalidOperatorSpec', ...
                      '%s operator must define expr/fn or a known preset', context);
            end

            if strcmpi(char(string(context)), 'transverse') && ~isempty(fn)
                error('nlolib:invalidOperatorSpec', ...
                      'transverse callable operators are not supported in MATLAB facade');
            end

            if ~isempty(fn)
                if ~isempty(params)
                    error('nlolib:invalidOperatorSpec', ...
                          '%s callable operator params are not supported in MATLAB facade', ...
                          context);
                end
                return;
            end

            [expr, constants] = nlolib.NLolib.parameterize_expression(expr, params, offset);
        end

        function [expr, params, matched] = operator_preset(context, name)
            matched = true;
            params = [];
            key = char(lower(string(name)));
            switch char(lower(string(context)))
                case 'linear'
                    switch key
                        case {'gvd', 'default'}
                            expr = "i*beta2*w*w-loss";
                            params = struct('beta2', -0.5, 'loss', 0.0);
                        case 'none'
                            expr = "0";
                            params = struct();
                        otherwise
                            expr = string(name);
                            matched = false;
                    end
                case 'nonlinear'
                    switch key
                        case {'kerr', 'default'}
                            expr = "i*gamma*I";
                            params = struct('gamma', 1.0);
                        case 'none'
                            expr = "0";
                            params = struct();
                        otherwise
                            expr = string(name);
                            matched = false;
                    end
                case 'transverse'
                    switch key
                        case {'diffraction', 'default'}
                            expr = "i*beta_t*w";
                            params = struct('beta_t', 1.0);
                        case 'none'
                            expr = "0";
                            params = struct();
                        otherwise
                            expr = string(name);
                            matched = false;
                    end
                otherwise
                    expr = string(name);
                    matched = false;
            end
        end

        function [expression, constants] = parameterize_expression(expressionInput, params, offset)
            expression = string(expressionInput);
            constants = [];
            if isempty(params)
                return;
            end

            if isnumeric(params)
                constants = double(params(:).');
                return;
            end

            if ~isstruct(params)
                error('nlolib:invalidOperatorSpec', ...
                      'operator params must be numeric or struct');
            end

            names = fieldnames(params);
            constants = zeros(1, numel(names));
            for idx = 1:numel(names)
                key = names{idx};
                constants(idx) = double(params.(key));
                token = sprintf('c%d', offset + idx - 1);
                pattern = ['(?<![A-Za-z0-9_])' regexptranslate('escape', key) '(?![A-Za-z0-9_])'];
                expression = regexprep(expression, pattern, token);
            end
        end

        function roots = resolve_roots()
            %RESOLVE_ROOTS Candidate roots for source and staged toolbox layouts.
            packageDir = fileparts(mfilename("fullpath"));     % .../+nlolib
            containerDir = fileparts(packageDir);              % .../matlab OR .../matlab_toolbox
            parentDir = fileparts(containerDir);               % repo root OR .../build
            [~, leaf] = fileparts(containerDir);

            if strcmpi(leaf, 'matlab')
                roots = unique({parentDir, containerDir}, 'stable');
                return;
            end

            if strcmpi(leaf, 'matlab_toolbox')
                grandParentDir = fileparts(parentDir);
                roots = unique({containerDir, parentDir, grandParentDir}, 'stable');
                return;
            end

            roots = unique({containerDir, parentDir}, 'stable');
        end

        function validate_load_result(notfound, loadWarnings, dllPath, headerPath)
            %VALIDATE_LOAD_RESULT Surface loadlibrary diagnostics clearly.
            if ~isempty(notfound)
                missingText = char(nlolib.NLolib.as_text(notfound));
                error('nlolib:loadlibraryNotFound', ...
                      ['loadlibrary could not resolve one or more symbols.\n' ...
                       'Library: %s\n' ...
                       'Header : %s\n' ...
                       'Missing symbols:\n%s'], ...
                      dllPath, headerPath, missingText);
            end

            warningText = strtrim(char(nlolib.NLolib.as_text(loadWarnings)));
            if ~isempty(warningText)
                warning('nlolib:loadlibraryWarnings', ...
                        ['loadlibrary produced parser warnings.\n' ...
                         'These are often non-fatal, but unresolved types will fail fast later.\n' ...
                         'Library: %s\n' ...
                         'Header : %s\n\n%s'], ...
                        dllPath, headerPath, warningText);
            end
        end

        function text = as_text(value)
            %AS_TEXT Convert MATLAB values returned by loadlibrary to text.
            if isempty(value)
                text = "";
                return;
            end
            if isstring(value)
                text = join(value, newline);
                return;
            end
            if iscell(value)
                text = join(string(value), newline);
                return;
            end
            text = string(value);
        end

        function [code, name, detail] = normalize_status(raw)
            %NORMALIZE_STATUS Convert calllib enum return into numeric code.
            detail = "";
            if isnumeric(raw) && isscalar(raw)
                code = int32(raw);
                name = nlolib.NLolib.status_name(code);
                return;
            end

            token = '';
            if isnumeric(raw) && ~isscalar(raw)
                vals = double(raw(:).');
                if ~isempty(vals) && all(isfinite(vals)) && ...
                        all(vals == floor(vals)) && ...
                        all(vals >= 0) && all(vals <= 255)
                    vals = vals(vals ~= 0);
                    if ~isempty(vals)
                        isPrintableAscii = (vals >= 32 & vals <= 126) | ...
                                           vals == 9 | vals == 10 | vals == 13;
                        if all(isPrintableAscii)
                            token = strtrim(char(vals));
                            if ~isempty(token)
                                detail = "status token decoded from numeric byte vector";
                            end
                        end
                    end
                end
            end

            if isempty(token)
                token = strtrim(char(string(raw)));
            end

            if isempty(token)
                code = int32(-1);
                name = "UNKNOWN_EMPTY";
                detail = nlolib.NLolib.describe_status_raw(raw);
                return;
            end

            switch token
                case 'NLOLIB_STATUS_OK'
                    code = int32(0);
                case 'NLOLIB_STATUS_INVALID_ARGUMENT'
                    code = int32(1);
                case 'NLOLIB_STATUS_ALLOCATION_FAILED'
                    code = int32(2);
                case 'NLOLIB_STATUS_NOT_IMPLEMENTED'
                    code = int32(3);
                otherwise
                    code = int32(-1);
                    if strlength(detail) == 0
                        detail = nlolib.NLolib.describe_status_raw(raw);
                    else
                        detail = detail + " | " + nlolib.NLolib.describe_status_raw(raw);
                    end
            end
            name = string(token);
        end

        function detail = describe_status_raw(raw)
            sz = size(raw);
            sizeParts = strings(1, numel(sz));
            for idx = 1:numel(sz)
                sizeParts(idx) = string(sz(idx));
            end
            sizeText = join(sizeParts, "x");

            detail = "raw status return class=" + string(class(raw)) + ...
                     " size=" + sizeText;

            if isnumeric(raw)
                flat = double(raw(:).');
                maxPreview = min(numel(flat), 16);
                if maxPreview > 0
                    previewParts = strings(1, maxPreview);
                    for idx = 1:maxPreview
                        previewParts(idx) = string(flat(idx));
                    end
                    previewText = join(previewParts, ",");
                    detail = detail + " preview=[" + previewText + "]";
                end
            end
        end

        function name = status_name(code)
            switch int32(code)
                case 0
                    name = "NLOLIB_STATUS_OK";
                case 1
                    name = "NLOLIB_STATUS_INVALID_ARGUMENT";
                case 2
                    name = "NLOLIB_STATUS_ALLOCATION_FAILED";
                case 3
                    name = "NLOLIB_STATUS_NOT_IMPLEMENTED";
                otherwise
                    name = "NLOLIB_STATUS_UNKNOWN";
            end
        end
    end
end

function value = getfield_safe(s, name, defaultValue)
if nargin < 3
    defaultValue = [];
end
if isstruct(s) && isfield(s, name)
    value = s.(name);
else
    value = defaultValue;
end
end
