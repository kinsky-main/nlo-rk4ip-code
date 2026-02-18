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
                loadlibrary(dllPath, headerPath, 'alias', obj.LIBNAME);
            end
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
            outBuf = zeros(1, double(outLen) * 2);
            outPtr = libpointer('doublePtr', outBuf);

            % Build execution options (or pass NULL).
            if isempty(fieldnames(execOptions))
                execOptsPtr = libpointer();          % NULL
            else
                execOptsPtr = nlolib.NLolib.make_exec_options(execOptions);
            end

            status = calllib(obj.LIBNAME, 'nlolib_propagate', ...
                             cfgPtr, ...
                             uint64(numTimeSamples), ...
                             inPtr, ...
                             numRecordedSamples, ...
                             outPtr, ...
                             execOptsPtr);

            if status ~= 0
                error('nlolib:propagateFailed', ...
                      'nlolib_propagate failed with status=%d', status);
            end

            records = nlolib.unpack_records(outPtr, ...
                                            double(numRecordedSamples), ...
                                            numTimeSamples);
        end
    end

    methods (Static)
        function unload()
            %UNLOAD Explicitly unload the nlolib shared library.
            if libisloaded('nlolib')
                unloadlibrary('nlolib');
            end
        end
    end

    methods (Static, Access = private)
        function headerPath = resolve_header()
            %RESOLVE_HEADER Locate nlolib_matlab.h relative to this file.
            repoRoot   = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            candidates = {
                fullfile(repoRoot, 'src', 'nlolib_matlab.h')
                fullfile(repoRoot, 'lib', 'nlolib_matlab.h')
            };
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

            repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            if ispc
                searchDirs = {
                    fullfile(repoRoot, 'build', 'src', 'Release')
                    fullfile(repoRoot, 'build', 'src', 'Debug')
                    fullfile(repoRoot, 'build', 'src', 'RelWithDebInfo')
                    fullfile(repoRoot, 'build', 'src')
                    fullfile(repoRoot, 'lib', 'win64')
                    fullfile(repoRoot, 'python', 'Release')
                    fullfile(repoRoot, 'python', 'Debug')
                    fullfile(repoRoot, 'python')
                };
                libFile = 'nlolib.dll';
            else
                searchDirs = {
                    fullfile(repoRoot, 'build', 'src')
                    fullfile(repoRoot, 'lib', 'glnxa64')
                    fullfile(repoRoot, 'python')
                };
                libFile = 'libnlolib.so';
            end

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
                s.backend_type = int32(2);   % AUTO
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
    end
end
