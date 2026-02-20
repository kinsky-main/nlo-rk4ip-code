function runtime_temporal_demo()
repoRoot = setup_matlab_example_environment();

n = 512;
dt = 0.02;
t = backend.centered_time_grid(n, dt);
field0 = exp(-((t / 0.25) .^ 2)) .* exp(-1i * 8.0 * t);

omega = backend.angular_frequency_grid(n, dt);
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
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "vkfft");
[records, info] = backend.propagate_with_fallback(api, cfg, field0, 2, simOptions);
finalField = records(end, :).';

fprintf("runtime_temporal_demo: propagated %d samples.\n", n);
fprintf("initial power=%.6e final power=%.6e\n", ...
        sum(abs(field0) .^ 2), sum(abs(finalField) .^ 2));
fprintf("backend requested=%s used=%s fallback=%d\n", ...
        info.requested_backend, info.used_backend, info.used_fallback);

outputDir = fullfile(repoRoot, "examples", "matlab", "output", "runtime_temporal_demo");
if ~isfolder(outputDir)
    mkdir(outputDir);
end
plotPath = backend.plot_final_intensity_comparison( ...
    t, records(1, :), records(end, :), ...
    fullfile(outputDir, "final_intensity_comparison.png"), ...
    "x_label", "Time t", ...
    "title", "Runtime Temporal Demo: Initial vs Final Intensity", ...
    "reference_label", "Initial", ...
    "final_label", "Final");
fprintf("saved plot: %s\n", plotPath);
end
