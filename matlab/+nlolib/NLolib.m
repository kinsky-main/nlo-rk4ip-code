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
    %     records = api.propagate(cfg, field0, numRecords);
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
                dllPath    = nlolib.NLolib.resolve_library(libraryPath);
                [notfound, loadWarnings] = loadlibrary(dllPath, headerPath, ...
                                                       'alias', obj.LIBNAME);
                nlolib.NLolib.validate_load_result(notfound, loadWarnings, ...
                                                   dllPath, headerPath);
            end
            nlolib.NLolib.ensure_types_loaded();
        end

        function delete(~)
            % Destructor intentionally does NOT unload the library since
            % multiple NLolib instances may share the same loaded lib.
            % Call nlolib.NLolib.unload() explicitly if needed.
        end

        function records = propagate(obj, config, inputField, ...
                                     numRecordedSamples, execOptions)
            %PROPAGATE Run a nonlinear propagation simulation.
            %   records = obj.propagate(cfg, field, numRecords)
            %   records = obj.propagate(cfg, field, numRecords, execOpts)
            if nargin < 5
                execOptions = struct();
            end

            numTimeSamples = numel(inputField);
            numRecordedSamples = uint64(numRecordedSamples);

            % Pack sim_config and collect keepalive handles.
            [cfgPtr, keepalive] = nlolib.prepare_sim_config(config); %#ok<ASGLU>

            % Pack input field into interleaved [re im re im ...] buffer.
            inPtr = nlolib.pack_complex_array(inputField);

            % Allocate output buffer.
            outLen = uint64(numTimeSamples) * numRecordedSamples;
            outPtr = nlolib.pack_complex_array(zeros(1, double(outLen)));
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

            statusRaw = calllib(obj.LIBNAME, 'nlolib_propagate', ...
                                cfgPtr, ...
                                uint64(numTimeSamples), ...
                                inPtr, ...
                                numRecordedSamples, ...
                                outPtr, ...
                                execOptsPtr);
            if matlabDebug
                postProbe = nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                           "post-call", true);
            end
            [statusCode, statusName, statusDetail] = nlolib.NLolib.normalize_status(statusRaw);

            if statusCode ~= 0
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
    end

    methods (Static)
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
            if strlength(string(userPath)) > 0 && isfile(userPath)
                dllPath = char(userPath);
                return;
            end

            envPath = getenv('NLOLIB_LIBRARY');
            if ~isempty(envPath) && isfile(envPath)
                dllPath = envPath;
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

            for idx = 1:numel(searchDirs)
                candidate = fullfile(searchDirs{idx}, libFile);
                if isfile(candidate)
                    dllPath = candidate;
                    return;
                end
            end
            error('nlolib:libraryNotFound', ...
                  'Cannot locate %s. Set NLOLIB_LIBRARY or pass path.', ...
                  libFile);
        end

        function ptr = make_exec_options(opts)
            %MAKE_EXEC_OPTIONS Build a nlo_execution_options libstruct.
            s = libstruct('nlo_execution_options');

            if isfield(opts, 'backend_type')
                s.backend_type = int32(opts.backend_type);
            else
                s.backend_type = int32(0);   % CPU (portable default for MATLAB)
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
