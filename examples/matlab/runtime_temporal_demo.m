function runtime_temporal_demo()
repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
addpath(fullfile(repoRoot, "matlab"));

n = 512;
dt = 0.02;
t = ((0:(n - 1)) - 0.5 * (n - 1)) * dt;
field0 = exp(-((t / 0.25) .^ 2)) .* exp(-1i * 8.0 * t);

omega = 2.0 * pi * fftfreq_unshifted(n, dt);
beta2 = 0.05;

cfg = struct();
cfg.num_time_samples = n;
cfg.gamma = 0.0;
cfg.betas = [];
cfg.alpha = 0.0;
cfg.propagation_distance = 0.25;
cfg.starting_step_size = 1e-3;
cfg.max_step_size = 5e-3;
cfg.min_step_size = 1e-5;
cfg.error_tolerance = 1e-7;
cfg.pulse_period = n * dt;
cfg.delta_time = dt;
cfg.frequency_grid = complex(omega, zeros(size(omega)));

runtime = struct();
runtime.dispersion_fn = @(w) exp(1i * (beta2 / 2.0) * (w .* w));
runtime.constants = [];
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
