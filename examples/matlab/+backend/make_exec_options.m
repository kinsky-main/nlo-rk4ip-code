function execOptions = make_exec_options(simOptions, numRecords)
%MAKE_EXEC_OPTIONS Convert example simulation options to nlolib exec options.

if nargin < 1 || isempty(simOptions)
    simOptions = backend.default_simulation_options();
end
if nargin < 2 || isempty(numRecords)
    numRecords = 0;
end

options = simOptions;
options.backend = lower(string(options.backend));
options.fft_backend = lower(string(options.fft_backend));
if options.record_ring_target == 0 && ...
   (options.backend == "auto" || options.backend == "vulkan") && ...
   numRecords > 0
    options.record_ring_target = uint64(max(1, min(double(numRecords), 32)));
end

execOptions = struct();
execOptions.backend_type = map_backend(options.backend);
execOptions.fft_backend = map_fft_backend(options.fft_backend);
execOptions.device_heap_fraction = double(options.device_heap_fraction);
execOptions.record_ring_target = uint64(options.record_ring_target);
execOptions.forced_device_budget_bytes = uint64(options.forced_device_budget_bytes);
end

function code = map_backend(name)
switch lower(string(name))
    case "cpu"
        code = int32(0);
    case "vulkan"
        code = int32(1);
    case "auto"
        code = int32(2);
    otherwise
        error("Unsupported backend '%s'.", string(name));
end
end

function code = map_fft_backend(name)
switch lower(string(name))
    case "auto"
        code = int32(0);
    case "fftw"
        code = int32(1);
    case "vkfft"
        code = int32(2);
    otherwise
        error("Unsupported fft_backend '%s'.", string(name));
end
end
