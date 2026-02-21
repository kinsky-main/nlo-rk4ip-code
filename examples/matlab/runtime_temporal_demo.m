function runtime_temporal_demo()
repoRoot = setup_matlab_example_environment();

n = 512;
dt = 0.02;
t = backend.centered_time_grid(n, dt);
field0 = exp(-((t / 0.25) .^ 2)) .* exp(-1i * 8.0 * t);

omega = backend.angular_frequency_grid(n, dt);
beta2 = 0.05;

pulse = struct();
pulse.samples = field0;
pulse.delta_time = dt;
pulse.pulse_period = n * dt;
pulse.frequency_grid = complex(omega, zeros(size(omega)));

linearOperator = struct();
linearOperator.fn = @(A, w) 1i * (beta2 / 2.0) * (w .* w);

nonlinearOperator = struct();
nonlinearOperator.expr = "i*gamma*I + i*V";
nonlinearOperator.params = struct('gamma', 0.01);

api = nlolib.NLolib();
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "vkfft");
execOptions = backend.make_exec_options(simOptions, 2);
simulateOptions = struct();
simulateOptions.propagation_distance = 0.25;
simulateOptions.records = 2;
simulateOptions.preset = "accuracy";
simulateOptions.exec_options = execOptions;
result = api.simulate(pulse, linearOperator, nonlinearOperator, simulateOptions);
records = result.records;
finalField = records(end, :).';

fprintf("runtime_temporal_demo: propagated %d samples.\n", n);
fprintf("initial power=%.6e final power=%.6e\n", ...
        sum(abs(field0) .^ 2), sum(abs(finalField) .^ 2));

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
