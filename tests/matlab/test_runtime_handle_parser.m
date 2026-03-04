function tests = test_runtime_handle_parser()
tests = functiontests(localfunctions);
end

function testTranslateElementwiseAndImaginaryUnit(testCase)
beta2 = 0.5;
runtime = struct();
[expr, constants] = nlolib.translate_runtime_handle( ...
    @(A, w) 1i * beta2 .* (w .^ 2), ...
    "dispersion_factor", ...
    runtime);

verifyThat(testCase, expr, matlab.unittest.constraints.ContainsSubstring("i"));
verifyThat(testCase, expr, matlab.unittest.constraints.ContainsSubstring("^"));
verifyEqual(testCase, constants, beta2, "AbsTol", 0.0);
end

function testTranslateBetaSumConstants(testCase)
beta2 = 0.04;
beta3 = -0.003;
beta4 = 0.0002;
runtime = struct();
[expr, constants] = nlolib.translate_runtime_handle( ...
    @(A, w) 1i * (beta2 .* (w .^ 2) + beta3 .* (w .^ 3) + beta4 .* (w .^ 4)), ...
    "dispersion_factor", ...
    runtime);

verifyThat(testCase, expr, matlab.unittest.constraints.ContainsSubstring("^2"));
verifyThat(testCase, expr, matlab.unittest.constraints.ContainsSubstring("^3"));
verifyThat(testCase, expr, matlab.unittest.constraints.ContainsSubstring("^4"));
verifyEqual(testCase, constants, [beta2, beta3, beta4], "AbsTol", 0.0);
end

function testTranslateDiffractionAndRamanHandles(testCase)
betaT = -0.018;
gamma = 0.015;
fR = 0.18;
runtime = struct();

[exprDiff, constantsDiff] = nlolib.translate_runtime_handle( ...
    @(A, w) 1i * betaT .* w, ...
    "dispersion_factor", ...
    runtime);
verifyThat(testCase, exprDiff, matlab.unittest.constraints.ContainsSubstring("w"));
verifyEqual(testCase, constantsDiff, betaT, "AbsTol", 0.0);

[exprRaman, constantsRaman] = nlolib.translate_runtime_handle( ...
    @(A, I, V) 1i .* A .* (gamma .* (1.0 - fR) .* I + gamma .* fR .* V), ...
    "nonlinear", ...
    runtime);
verifyThat(testCase, exprRaman, matlab.unittest.constraints.ContainsSubstring("A"));
verifyThat(testCase, exprRaman, matlab.unittest.constraints.ContainsSubstring("I"));
verifyThat(testCase, exprRaman, matlab.unittest.constraints.ContainsSubstring("V"));
verifyEqual(testCase, constantsRaman, [gamma, fR], "AbsTol", 0.0);
end

function testTranslateUnknownIdentifierRejected(testCase)
beta2 = 0.5;
runtime = struct('auto_capture_constants', false);
assert_throws_contains(testCase, ...
    @() nlolib.translate_runtime_handle(@(A, w) 1i * beta2 .* w, "dispersion_factor", runtime), ...
    "unknown identifier");
end

function testTranslateComplexCaptureRejected(testCase)
badGain = 1.0 + 2.0i;
runtime = struct();
assert_throws_contains(testCase, ...
    @() nlolib.translate_runtime_handle(@(A, w) badGain .* w, "dispersion_factor", runtime), ...
    "must be a real scalar");
end

function testBetaSumPropagationParity(testCase)
api = nlolib.NLolib();
execOpts = struct('backend_type', int32(0));

n = 192;
dt = 0.02;
beta2 = 0.04;
beta3 = -0.003;
beta4 = 0.0002;
omega = omega_grid_unshifted(n, dt);
inputField = deterministic_complex_field(n);

common = base_runtime_config(n, dt, 0.08, omega);

cfgString = common;
cfgString.runtime = struct( ...
    'dispersion_factor_expr', "i*(c0*(w^2)+c1*(w^3)+c2*(w^4))", ...
    'constants', [beta2, beta3, beta4, 0.0]);

cfgCallable = common;
cfgCallable.runtime = struct( ...
    'dispersion_factor_fn', @(A, w) 1i * (beta2 * (w ^ 2) + beta3 * (w ^ 3) + beta4 * (w ^ 4)), ...
    'constants', [0.0, 0.0, 0.0, 0.0]);

resString = api.propagate(cfgString, inputField, uint64(2), execOpts);
resCallable = api.propagate(cfgCallable, inputField, uint64(2), execOpts);
err = max_abs_diff(resString.records(2, :), resCallable.records(2, :));
verifyLessThanOrEqual(testCase, err, 5e-7);
end

function testDiffractionPropagationParity(testCase)
api = nlolib.NLolib();
execOpts = struct('backend_type', int32(0));

nt = 1;
nx = 6;
ny = 4;
n = nt * nx * ny;
betaT = -0.018;
inputField = deterministic_complex_field(n);

cfgCommon = coupled_runtime_config(nt, nx, ny);
cfgCommon.potential_grid = complex(zeros(1, n), zeros(1, n));

cfgString = cfgCommon;
cfgString.runtime = struct( ...
    'linear_factor_expr', "i*c0*(kx*kx + ky*ky)", ...
    'linear_expr', "exp(h*D)", ...
    'nonlinear_expr', "0", ...
    'constants', [betaT, 0.0, 0.0, 0.0]);

cfgCallable = cfgCommon;
cfgCallable.runtime = struct( ...
    'dispersion_factor_expr', "i*c0*(kx*kx + ky*ky)", ...
    'dispersion_expr', "exp(h*D)", ...
    'nonlinear_expr', "0", ...
    'constants', [betaT, 0.0, 0.0, 0.0]);

resString = api.propagate(cfgString, inputField, uint64(2), execOpts);
resCallable = api.propagate(cfgCallable, inputField, uint64(2), execOpts);
err = max_abs_diff(resString.records(2, :), resCallable.records(2, :));
verifyLessThanOrEqual(testCase, err, 1e-7);
end

function testRamanLikePropagationParity(testCase)
api = nlolib.NLolib();
execOpts = struct('backend_type', int32(0));

nt = 4;
nx = 4;
ny = 4;
n = nt * nx * ny;
gamma = 0.015;
fR = 0.18;
potentialXY = complex(0.02 * double(1:(nx * ny)), zeros(1, nx * ny));
potential = repmat(potentialXY, 1, nt);
inputField = deterministic_complex_field(n);

cfgCommon = coupled_runtime_config(nt, nx, ny);
cfgCommon.potential_grid = potential;
cfgCommon.propagation_distance = 0.008;

cfgString = cfgCommon;
cfgString.runtime = struct( ...
    'linear_factor_expr', "0", ...
    'linear_expr', "exp(h*D)", ...
    'nonlinear_expr', "i*A*(c0*(1.0-c1)*I + c0*c1*V)", ...
    'constants', [gamma, fR, 0.0, 0.0]);

cfgCallable = cfgCommon;
cfgCallable.runtime = struct( ...
    'linear_factor_expr', "0", ...
    'linear_expr', "exp(h*D)", ...
    'nonlinear_fn', @(A, I, V) 1i .* A .* (gamma .* (1.0 - fR) .* I + gamma .* fR .* V), ...
    'constants', [0.0, 0.0, 0.0, 0.0]);

resString = api.propagate(cfgString, inputField, uint64(2), execOpts);
resCallable = api.propagate(cfgCallable, inputField, uint64(2), execOpts);
err = max_abs_diff(resString.records(2, :), resCallable.records(2, :));
verifyLessThanOrEqual(testCase, err, 1e-7);
end

function cfg = base_runtime_config(n, dt, zFinal, omega)
cfg = struct();
cfg.num_time_samples = uint64(n);
cfg.propagation_distance = zFinal;
cfg.starting_step_size = 1e-3;
cfg.max_step_size = 5e-3;
cfg.min_step_size = 1e-5;
cfg.error_tolerance = 1e-7;
cfg.pulse_period = n * dt;
cfg.delta_time = dt;
cfg.frequency_grid = complex(omega, zeros(1, n));
end

function cfg = coupled_runtime_config(nt, nx, ny)
cfg = struct();
cfg.num_time_samples = uint64(nt * nx * ny);
cfg.propagation_distance = 0.01;
cfg.starting_step_size = 1e-3;
cfg.max_step_size = 2e-3;
cfg.min_step_size = 1e-5;
cfg.error_tolerance = 1e-7;
cfg.pulse_period = nt * 0.02;
cfg.delta_time = 0.02;
cfg.tensor_nt = uint64(nt);
cfg.tensor_nx = uint64(nx);
cfg.tensor_ny = uint64(ny);
cfg.tensor_layout = int32(0);
cfg.frequency_grid = complex(zeros(1, nt), zeros(1, nt));
end

function out = omega_grid_unshifted(n, dt)
twoPi = 2.0 * pi;
out = zeros(1, n);
for idx = 1:n
    i = idx - 1;
    if i <= floor((n - 1) / 2)
        out(idx) = twoPi * (double(i) / (double(n) * dt));
    else
        out(idx) = twoPi * (-(double(n - i) / (double(n) * dt)));
    end
end
end

function field = deterministic_complex_field(n)
idx = double(0:(n - 1));
re = 0.12 * sin(0.13 * idx) + 0.07 * cos(0.23 * idx);
im = 0.11 * sin(0.19 * idx) - 0.05 * cos(0.31 * idx);
field = complex(re, im);
end

function err = max_abs_diff(a, b)
err = max(abs(a(:) - b(:)));
end

function assert_throws_contains(testCase, fn, expected)
caught = false;
try
    fn();
catch exc
    caught = true;
    verifyThat(testCase, string(exc.message), ...
        matlab.unittest.constraints.ContainsSubstring(string(expected)));
end
verifyTrue(testCase, caught, "Expected the callable to throw.");
end
