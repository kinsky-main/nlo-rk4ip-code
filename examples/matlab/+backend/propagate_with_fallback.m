function [records, info] = propagate_with_fallback(api, cfg, inputField, numRecords, simOptions)
%PROPAGATE_WITH_FALLBACK Propagate with AUTO->CPU fallback policy.

if nargin < 5 || isempty(simOptions)
    simOptions = backend.default_simulation_options();
end

options = normalize_options(simOptions, numRecords);
primaryExec = make_exec_options(options);
primaryBackend = lower(string(options.backend));
if strlength(primaryBackend) == 0
    primaryBackend = "auto";
end

info = struct();
info.requested_backend = char(primaryBackend);
info.used_backend = char(primaryBackend);
info.used_fallback = false;
info.primary_error = "";

try
    records = api.propagate(cfg, inputField, numRecords, primaryExec);
catch primaryME
    if primaryBackend ~= "auto"
        rethrow(primaryME);
    end

    info.used_fallback = true;
    info.primary_error = string(primaryME.message);
    warnText = sprintf("AUTO backend failed; retrying with CPU backend.\nAUTO error: %s", ...
                       char(info.primary_error));
    warning("backend:autoFallback", ...
            "%s", ...
            warnText);

    fallbackOpts = options;
    fallbackOpts.backend = "cpu";
    if lower(string(fallbackOpts.fft_backend)) == "vkfft"
        fallbackOpts.fft_backend = "auto";
    end
    fallbackExec = make_exec_options(fallbackOpts);
    records = api.propagate(cfg, inputField, numRecords, fallbackExec);
    info.used_backend = "cpu";
end
end

function options = normalize_options(simOptions, numRecords)
options = simOptions;
options.backend = lower(string(options.backend));
options.fft_backend = lower(string(options.fft_backend));

if options.record_ring_target == 0 && ...
   (options.backend == "auto" || options.backend == "vulkan")
    options.record_ring_target = uint64(max(1, min(double(numRecords), 32)));
end
end

function out = make_exec_options(options)
out = struct();
out.backend_type = map_backend(options.backend);
out.fft_backend = map_fft_backend(options.fft_backend);
out.device_heap_fraction = double(options.device_heap_fraction);
out.record_ring_target = uint64(options.record_ring_target);
out.forced_device_budget_bytes = uint64(options.forced_device_budget_bytes);
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
