function epsilon = second_order_soliton_rk4ip()
%SECOND_ORDER_SOLITON_RK4IP Second-order soliton analytical validation.

repoRoot = setup_matlab_example_environment();

beta2 = -0.01;
gamma = 0.01;
alpha = 0.0;
tfwhm = 100e-3;
t0 = tfwhm / (2.0 * log(1.0 + sqrt(2.0)));
p0 = (2^2) * abs(beta2) / (gamma * t0 * t0);
zFinal = 0.506;

n = 2^10;
tmax = 8.0 * t0;
T = linspace(-tmax, tmax, n);
t = T / t0;
dt = T(2) - T(1);
omega = backend.angular_frequency_grid(n, dt);

u0 = 2.0 * sech_array(t);
a0 = to_physical_envelope(u0, 0.0, p0, alpha);

numRecords = 160;
pulse = struct();
pulse.samples = a0;
pulse.delta_time = dt;
pulse.pulse_period = n * dt;
pulse.frequency_grid = complex(omega, zeros(1, n));

linearOperator = struct();
linearOperator.expr = "i*beta2*w*w-loss";
linearOperator.params = struct('beta2', 0.5 * beta2, 'loss', 0.5 * alpha);

nonlinearOperator = struct();
nonlinearOperator.expr = "i*A*(gamma*I + V)";
nonlinearOperator.params = struct('gamma', gamma);

api = nlolib.NLolib();
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "auto");
execOptions = backend.make_exec_options(simOptions, numRecords);
execOptions.capture_step_history = true;
execOptions.step_history_capacity = uint64(200000);
stepTelemetry = empty_step_telemetry();

propagateOptions = struct();
propagateOptions.propagation_distance = zFinal;
propagateOptions.records = numRecords;
propagateOptions.preset = "balanced";
propagateOptions.exec_options = execOptions;
result = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);
aRecords = result.records;
zRecords = result.z_axis;
stepTelemetry = step_telemetry_from_result(result);

ensure_finite_records_or_error(aRecords, zRecords);

uNumRecords = zeros(size(aRecords));
uTrueRecords = zeros(size(aRecords));
for idx = 1:numRecords
    z = zRecords(idx);
    uNumRecords(idx, :) = to_normalized_envelope(aRecords(idx, :), z, p0, alpha);
    uTrueRecords(idx, :) = second_order_soliton_normalized_envelope(t, z, beta2, t0);
end

uNum = uNumRecords(end, :);
uTrue = uTrueRecords(end, :);
epsilon = average_relative_intensity_error(uNum, uTrue);
if ~isfinite(epsilon)
    error("final epsilon is non-finite; numerical output is invalid.");
end

errorCurve = relative_l2_error_curve(uNumRecords, uTrueRecords);
z0AnalyticError = analytical_initial_condition_error(t, beta2, t0);

lambda0Nm = 1550.0;
nFftVisual = 4 * n;
[zMap, lambdaNm, spectralMap] = compute_wavelength_spectral_map_from_records( ...
    aRecords, zRecords, dt, lambda0Nm, nFftVisual);
if any(~isfinite(spectralMap), "all")
    error("spectral map contains non-finite values.");
end

outputDir = fullfile(repoRoot, "examples", "matlab", "output", "second_order_soliton");
if ~isfolder(outputDir)
    mkdir(outputDir);
end

savedPaths = strings(0, 1);
savedPaths(end + 1, 1) = string(plot_wavelength_step_history( ...
    zMap, lambdaNm, spectralMap, stepTelemetry, ...
    fullfile(outputDir, "wavelength_intensity_colormap.png"))); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_re_im_comparison( ...
    t, uTrue, uNum, ...
    fullfile(outputDir, "final_re_im_comparison.png"), ...
    "x_label", "Dimensionless time t = T/T0", ...
    "title", sprintf("Second-Order Soliton at z = %.3f m: Re/Im Comparison", zFinal), ...
    "reference_label", "Analytical", ...
    "final_label", "Numerical")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_intensity_comparison( ...
    t, uTrue, uNum, ...
    fullfile(outputDir, "final_intensity_comparison.png"), ...
    "x_label", "Dimensionless time t = T/T0", ...
    "title", sprintf("Second-Order Soliton at z = %.3f m: Intensity Comparison", zFinal), ...
    "reference_label", "Analytical", ...
    "final_label", "Numerical")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_total_error_over_propagation( ...
    zRecords, errorCurve, ...
    fullfile(outputDir, "total_error_over_propagation.png"), ...
    "title", "Second-Order Soliton: Total Error Over Propagation", ...
    "y_label", "Relative L2 error (numerical vs analytical)")); %#ok<AGROW>

[sgnBeta2, ld, lnl] = normalized_nlse_coefficients(beta2, gamma, t0, p0);
fprintf("normalized NLSE coefficients: sgn(beta2)=%+d, 1/(2*LD)=%.6e 1/m, exp(-alpha*z_final)/LNL=%.6e 1/m.\n", ...
        int32(sgnBeta2), 0.5 / ld, exp(-alpha * zFinal) / lnl);
fprintf("analytical z=0 envelope max error = %.6e\n", z0AnalyticError);
fprintf("epsilon = %.6e\n", epsilon);
fprintf("step telemetry events: accepted=%d, next=%d, dropped=%d\n", ...
        numel(stepTelemetry.accepted_z), numel(stepTelemetry.next_z), int64(stepTelemetry.dropped));
for idx = 1:numel(savedPaths)
    fprintf("saved plot: %s\n", savedPaths(idx));
end
end

function out = sech_array(x)
out = 1.0 ./ cosh(x);
end

function out = to_normalized_envelope(a, z, p0, alpha)
out = a .* exp(0.5 * alpha * z) / sqrt(p0);
end

function out = to_physical_envelope(u, z, p0, alpha)
out = u .* (sqrt(p0) * exp(-0.5 * alpha * z));
end

function [sgnBeta2, ld, lnl] = normalized_nlse_coefficients(beta2, gamma, t0, p0)
if beta2 == 0.0
    error("beta2 must be non-zero for NLSE normalization.");
end
ld = (t0 * t0) / abs(beta2);
lnl = 1.0 / (gamma * p0);
sgnBeta2 = 1.0;
if beta2 < 0.0
    sgnBeta2 = -1.0;
end
end

function out = second_order_soliton_normalized_envelope(t, z, beta2, t0)
ld = (t0 * t0) / abs(beta2);
xi = z / ld;
numerator = 4.0 * (cosh(3.0 .* t) + 3.0 .* exp(4.0i * xi) .* cosh(t)) .* exp(0.5i * xi);
denominator = cosh(4.0 .* t) + 4.0 .* cosh(2.0 .* t) + 3.0 .* cos(4.0 * xi);
out = numerator ./ denominator;
end

function out = average_relative_intensity_error(aNum, aTrue)
intNum = abs(aNum).^2;
intTrue = abs(aTrue).^2;
finiteMask = isfinite(intNum) & isfinite(intTrue);
if ~any(finiteMask)
    out = NaN;
    return;
end
numerator = mean(abs(intNum(finiteMask) - intTrue(finiteMask)));
denominator = max(intTrue(finiteMask));
if denominator <= 0.0 || ~isfinite(denominator)
    out = NaN;
    return;
end
out = numerator / denominator;
end

function out = analytical_initial_condition_error(t, beta2, t0)
uRef = 2.0 * sech_array(t);
uAnalytic = second_order_soliton_normalized_envelope(t, 0.0, beta2, t0);
out = max(abs(uRef - uAnalytic));
end

function [zMap, lambdaNm, specMap] = compute_wavelength_spectral_map_from_records( ...
    aRecords, zSamples, dt, lambda0Nm, nFftVisual)

n = size(aRecords, 2);
nFft = max(n, round(nFftVisual));
freqShifted = fftshift(fftfreq_matlab(nFft, dt));
nu0 = 299792.458 / lambda0Nm;
nu = nu0 + freqShifted;
valid = nu > 0.0;
lambdaNm = 299792.458 ./ nu(valid);

spectra = fftshift(fft(aRecords, nFft, 2), 2);
specMap = abs(spectra(:, valid)).^2;

[lambdaNm, order] = sort(lambdaNm, "ascend");
specMap = specMap(:, order);
specMap(~isfinite(specMap)) = 0.0;
specMap(specMap < 0.0) = 0.0;
maxValue = max(specMap, [], "all");
if maxValue > 0.0
    specMap = specMap / maxValue;
end

% Keep only occupied spectral support to avoid near-zero bins stretching
% wavelength axes into an effectively blank image.
if ~isempty(specMap)
    spectralProfile = max(specMap, [], 1);
    supportThreshold = max(max(spectralProfile) * 1e-3, 1e-12);
    supportIdx = find(spectralProfile >= supportThreshold);
    if numel(supportIdx) >= 8
        supportMin = lambdaNm(supportIdx(1));
        supportMax = lambdaNm(supportIdx(end));
        halfSpan = max(abs(supportMin - lambda0Nm), abs(supportMax - lambda0Nm));
        if halfSpan > 0.0
            halfSpan = halfSpan * 1.02;
            lower = lambda0Nm - halfSpan;
            upper = lambda0Nm + halfSpan;
            bandMask = (lambdaNm >= lower) & (lambdaNm <= upper);
            if nnz(bandMask) >= 8
                lambdaNm = lambdaNm(bandMask);
                specMap = specMap(:, bandMask);
            end
        end
    end
end
zMap = zSamples;
end

function outPath = plot_wavelength_step_history(zAxis, lambdaNm, spectralMap, telemetry, outputPath)
fig = figure('Visible', 'off', 'Position', [80, 80, 980, 1040]);
tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

axMap = nexttile(1);
imagesc(axMap, zAxis, lambdaNm, spectralMap.');
axis(axMap, 'xy');
try
    colormap(axMap, "magma");
catch
    colormap(axMap, "parula");
end
axMap.Box = 'on';
xlabel(axMap, "Propagation distance z (m)");
ylabel(axMap, "Wavelength (nm)");
title(axMap, "Spectral Intensity Envelope vs Propagation Distance");
pbaspect(axMap, [1, 1, 1]);
cbar = colorbar(axMap);
cbar.Label.String = "Normalized spectral intensity";

axStep = nexttile(2);
telemetryPlot = filter_record_clipped_steps(telemetry, zAxis);
hasSeries = false;
if ~isempty(telemetryPlot.accepted_z)
    [zSorted, order] = sort(telemetryPlot.accepted_z, "ascend");
    plot(axStep, zSorted, telemetryPlot.accepted_step_sizes(order), ...
         '-', 'LineWidth', 1.2, 'Color', [0.00, 0.45, 0.74], ...
         'DisplayName', 'Accepted step\_size');
    hold(axStep, 'on');
    hasSeries = true;
end

if hasSeries
    xlabel(axStep, "Propagation distance z (m)");
    ylabel(axStep, "Step size (m)");
    title(axStep, "Adaptive RK4IP Step Sizes");
    grid(axStep, 'on');
    legend(axStep, 'Location', 'best');
else
    title(axStep, "Adaptive RK4IP Step Sizes");
    text(axStep, 0.5, 0.5, "No adaptive step-adjustment events captured", ...
         'Units', 'normalized', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle');
    set(axStep, 'XTick', [], 'YTick', []);
end

% Keep bottom panel width exactly aligned to the top map panel width.
mapPos = axMap.Position;
stepPos = axStep.Position;
stepPos(1) = mapPos(1);
stepPos(3) = mapPos(3);
axStep.Position = stepPos;

outPath = outputPath;
exportgraphics(fig, outPath, 'Resolution', 280);
close(fig);
end

function telemetry = empty_step_telemetry()
telemetry = struct();
telemetry.accepted_z = zeros(0, 1);
telemetry.accepted_step_sizes = zeros(0, 1);
telemetry.next_z = zeros(0, 1);
telemetry.next_step_sizes = zeros(0, 1);
telemetry.dropped = 0.0;
end

function telemetry = step_telemetry_from_result(result)
telemetry = empty_step_telemetry();
if ~isstruct(result) || ~isfield(result, 'step_history') || ~isstruct(result.step_history)
    return;
end
h = result.step_history;
if ~isfield(h, 'z') || ~isfield(h, 'step_size') || ~isfield(h, 'next_step_size')
    return;
end
z = double(h.z(:));
accepted = double(h.step_size(:));
nextSizes = double(h.next_step_size(:));
n = min([numel(z), numel(accepted), numel(nextSizes)]);
if n <= 0
    if isfield(h, 'dropped')
        telemetry.dropped = double(h.dropped);
    end
    return;
end
telemetry.accepted_z = z(1:n);
telemetry.accepted_step_sizes = accepted(1:n);
telemetry.next_z = z(1:n);
telemetry.next_step_sizes = nextSizes(1:n);
if isfield(h, 'dropped')
    telemetry.dropped = double(h.dropped);
end
end

function outTelemetry = filter_record_clipped_steps(telemetry, zAxis)
outTelemetry = telemetry;
if ~isstruct(telemetry) || isempty(telemetry.accepted_z) || numel(zAxis) <= 1
    return;
end
zAxis = double(zAxis(:));
spacing = (zAxis(end) - zAxis(1)) / double(numel(zAxis) - 1);
if ~isfinite(spacing) || spacing <= 0.0
    return;
end

z = double(telemetry.accepted_z(:));
accepted = double(telemetry.accepted_step_sizes(:));
proposed = double(telemetry.next_step_sizes(:));
n = min([numel(z), numel(accepted), numel(proposed)]);
if n <= 0
    return;
end
z = z(1:n);
accepted = accepted(1:n);
proposed = proposed(1:n);

z0 = double(zAxis(1));
zEnd = double(zAxis(end));
expectedBoundaries = zAxis(2:end-1);
if isempty(expectedBoundaries)
    return;
end

proximityEps = max(64.0 * eps(max(1.0, abs(zEnd))), spacing * 0.012);
clippedMask = false(size(z));
for idx = 1:numel(expectedBoundaries)
    clippedMask = clippedMask | (abs(z - expectedBoundaries(idx)) <= proximityEps);
end
if ~any(clippedMask)
    return;
end

keepMask = ~clippedMask;
outTelemetry.accepted_z = z(keepMask);
outTelemetry.accepted_step_sizes = accepted(keepMask);
outTelemetry.next_z = z(keepMask);
outTelemetry.next_step_sizes = proposed(keepMask);
end

function out = relative_l2_error_curve(records, referenceRecords)
if any(size(records) ~= size(referenceRecords))
    error("records and referenceRecords must have the same shape.");
end
refNorms = vecnorm(referenceRecords, 2, 2);
normFloor = max(max(refNorms) * 1e-12, 1e-15);
out = zeros(size(records, 1), 1);
for idx = 1:size(records, 1)
    ref = referenceRecords(idx, :);
    num = records(idx, :);
    refNorm = norm(ref, 2);
    safeNorm = max(refNorm, normFloor);
    out(idx) = norm(num - ref, 2) / safeNorm;
end
out = out(:).';
end

function ensure_finite_records_or_error(aRecords, zSamples)
maxFiniteAmplitude = 0.0;
for idx = 1:size(aRecords, 1)
    fieldZ = aRecords(idx, :);
    finiteMask = isfinite(real(fieldZ)) & isfinite(imag(fieldZ));
    if ~all(finiteMask)
        error(["numerical propagation diverged with non-finite field values near z = %.6e m; " ...
               "max finite |A| before divergence = %.6e."], ...
              zSamples(idx), maxFiniteAmplitude);
    end
    maxFiniteAmplitude = max(maxFiniteAmplitude, max(abs(fieldZ)));
end
end

function freq = fftfreq_matlab(n, dt)
freq = zeros(1, n);
half = floor((n - 1) / 2);
scale = 1.0 / (n * dt);
for idx = 0:(n - 1)
    if idx <= half
        freq(idx + 1) = idx * scale;
    else
        freq(idx + 1) = -(n - idx) * scale;
    end
end
end
