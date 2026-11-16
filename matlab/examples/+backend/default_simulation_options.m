function opts = default_simulation_options(varargin)
%DEFAULT_SIMULATION_OPTIONS Build simulation execution option defaults.
%
%   opts = backend.default_simulation_options()
%   opts = backend.default_simulation_options('backend', "cpu", ...)

opts = struct();
opts.backend = "auto";
opts.fft_backend = "auto";
opts.device_heap_fraction = 0.70;
opts.record_ring_target = uint64(0);
opts.forced_device_budget_bytes = uint64(0);

if mod(numel(varargin), 2) ~= 0
    error("Name-value arguments must come in pairs.");
end

for idx = 1:2:numel(varargin)
    name = string(varargin{idx});
    value = varargin{idx + 1};
    switch lower(name)
        case "backend"
            opts.backend = string(value);
        case "fft_backend"
            opts.fft_backend = string(value);
        case "device_heap_fraction"
            opts.device_heap_fraction = double(value);
        case "record_ring_target"
            opts.record_ring_target = uint64(value);
        case "forced_device_budget_bytes"
            opts.forced_device_budget_bytes = uint64(value);
        otherwise
            error("Unknown option '%s'.", name);
    end
end
end
