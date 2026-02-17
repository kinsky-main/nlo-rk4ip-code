classdef NLolib < handle
    properties (Access = private)
        pyApi
        pyBindings
    end

    methods
        function obj = NLolib(libraryPath)
            if nargin < 1
                libraryPath = "";
            end

            apiDir = fullfile(fileparts(fileparts(fileparts(mfilename("fullpath")))), "python");
            inPath = false;
            pathList = py.sys.path;
            pathCount = int64(py.len(pathList));
            for idx = 1:double(pathCount)
                entry = string(pathList{idx});
                if entry == string(apiDir)
                    inPath = true;
                    break;
                end
            end
            if ~inPath
                pathList.insert(int32(0), apiDir);
            end

            obj.pyBindings = py.importlib.import_module("nlolib_ctypes");
            if strlength(string(libraryPath)) == 0
                obj.pyApi = obj.pyBindings.NLolib();
            else
                obj.pyApi = obj.pyBindings.NLolib(pyargs("path", char(string(libraryPath))));
            end
        end

        function records = propagate(obj, config, inputField, numRecordedSamples, execOptions)
            if nargin < 5
                execOptions = struct();
            end

            prepared = nlolib.prepare_sim_config(config);
            pyInput = nlolib.matlab_complex_vector_to_py_list(inputField);

            if isempty(fieldnames(execOptions))
                pyOpts = py.None;
            else
                pyOpts = obj.pyBindings.default_execution_options();
                if isfield(execOptions, "backend_type")
                    pyOpts.backend_type = int32(execOptions.backend_type);
                end
                if isfield(execOptions, "fft_backend")
                    pyOpts.fft_backend = int32(execOptions.fft_backend);
                end
                if isfield(execOptions, "device_heap_fraction")
                    pyOpts.device_heap_fraction = double(execOptions.device_heap_fraction);
                end
                if isfield(execOptions, "record_ring_target")
                    pyOpts.record_ring_target = int64(execOptions.record_ring_target);
                end
                if isfield(execOptions, "forced_device_budget_bytes")
                    pyOpts.forced_device_budget_bytes = int64(execOptions.forced_device_budget_bytes);
                end
            end

            pyRecords = obj.pyApi.propagate(prepared, pyInput, int64(numRecordedSamples), pyOpts);
            records = nlolib.py_record_list_to_matlab(pyRecords);
        end
    end
end
