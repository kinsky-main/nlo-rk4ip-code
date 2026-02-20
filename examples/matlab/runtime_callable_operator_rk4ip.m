function runtime_callable_operator_rk4ip()
%RUNTIME_CALLABLE_OPERATOR_RK4IP Callable runtime-operator demo.

setup_matlab_example_environment();

n = 1024;
dt = 0.01;
beta2 = 0.05;
scale = beta2 / 2.0;
zFinal = 1.0;
t = backend.centered_time_grid(n, dt);
field0 = exp(-((t / 0.20) .^ 2)) .* exp((-1.0i) * 12.0 * t);

runtime = struct();
runtime.dispersion_factor_fn = @(A, w) i * scale .* (w .^ 2);
runtime.nonlinear_fn = @(A, I) 0 .* I;

cfg = struct();
cfg.num_time_samples = n;
cfg.propagation_distance = zFinal;
cfg.starting_step_size = 1e-3;
cfg.max_step_size = 5e-3;
cfg.min_step_size = 1e-5;
cfg.error_tolerance = 1e-7;
cfg.pulse_period = n * dt;
cfg.delta_time = dt;
cfg.frequency_grid = complex(backend.angular_frequency_grid(n, dt), zeros(1, n));
cfg.runtime = runtime;

api = nlolib.NLolib();
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "vkfft");
[records, info] = backend.propagate_with_fallback(api, cfg, field0, 2, simOptions);
zRecords = linspace(0.0, zFinal, 2);

fprintf("runtime callable example completed.\n");
fprintf("z records: [%g, %g]\n", zRecords(1), zRecords(end));
fprintf("initial power=%.6e\n", sum(abs(records(1, :)) .^ 2));
fprintf("final power=%.6e\n", sum(abs(records(end, :)) .^ 2));
fprintf("backend requested=%s used=%s fallback=%d\n", ...
        info.requested_backend, info.used_backend, info.used_fallback);
end
