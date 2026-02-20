function runtime_temporal_demo()
% Just code to get the path and functions into the path, only useful for testing the runtime interface
repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
matlabCandidates = { ...
    fullfile(repoRoot, "matlab"), ...
    repoRoot ...
};
for idx = 1:numel(matlabCandidates)
    if isfolder(matlabCandidates{idx})
        addpath(matlabCandidates{idx});
    end
end
if exist("nlolib_setup", "file") == 2
    nlolib_setup();
else
    addpath(fullfile(repoRoot, "examples", "matlab"));
end

% Actually how you would use the runtime interface, this is just a simple demo of a temporal simulation with dispersion only
n = 512;
dt = 0.02;
t = ((0:(n - 1)) - 0.5 * (n - 1)) * dt;
field0 = exp(-((t / 0.25) .^ 2)) .* exp(-1i * 8.0 * t);

omega = 2.0 * pi * fftfreq_unshifted(n, dt);
beta2 = 0.05;

cfg = struct();
cfg.num_time_samples = n;
cfg.propagation_distance = 0.25;
cfg.starting_step_size = 1e-3;
cfg.max_step_size = 5e-3;
cfg.min_step_size = 1e-5;
cfg.error_tolerance = 1e-7;
cfg.pulse_period = n * dt;
cfg.delta_time = dt;
cfg.frequency_grid = complex(omega, zeros(size(omega)));

runtime = struct();
runtime.dispersion_factor_fn = @(A, w) 1i * (beta2 / 2.0) * (w .* w);
runtime.constants = [beta2 / 2.0, 0.0, 0.01];
cfg.runtime = runtime;

api = nlolib.NLolib();
records = api.propagate(cfg, field0, 2);
finalField = records(end, :).';

fprintf("runtime_temporal_demo: propagated %d samples.\n", n);
fprintf("initial power=%.6e final power=%.6e\n", ...
        sum(abs(field0) .^ 2), sum(abs(finalField) .^ 2));
end

function omega = fftfreq_unshifted(n, dt)
omega = zeros(1, n);
factor = 1.0 / (n * dt);
half = floor((n - 1) / 2);
for i = 0:(n - 1)
    if i <= half
        omega(i + 1) = i * factor;
    else
        omega(i + 1) = -(n - i) * factor;
    end
end
end
