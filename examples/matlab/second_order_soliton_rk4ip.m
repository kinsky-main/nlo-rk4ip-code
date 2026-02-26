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
nonlinearOperator.expr = "i*gamma*I + i*V";
nonlinearOperator.params = struct('gamma', gamma);

api = nlolib.NLolib();
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "auto");
execOptions = backend.make_exec_options(simOptions, numRecords);
stepTelemetry = empty_step_telemetry();
logBufferBytes = uint64(1024 * 1024);
progressTelemetryEnabled = false;
try
    api.set_log_buffer(logBufferBytes);
    api.set_log_level(int32(2));
    api.set_progress_options(true, int32(100), true);
    api.clear_log_buffer();
    progressTelemetryEnabled = true;
catch err
    fprintf("warning: step telemetry logging unavailable; continuing without step subplot data (%s)\n", ...
            err.message);
end

propagateOptions = struct();
propagateOptions.propagation_distance = zFinal;
propagateOptions.records = numRecords;
propagateOptions.preset = "balanced";
propagateOptions.exec_options = execOptions;
result = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);
aRecords = result.records;
zRecords = result.z_axis;
if progressTelemetryEnabled
    try
        runtimeLogs = api.read_log_buffer(true, logBufferBytes);
        stepTelemetry = parse_step_telemetry(runtimeLogs);
    catch err
        fprintf("warning: failed to read runtime step telemetry; continuing without it (%s)\n", ...
                err.message);
    end
end

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
fprintf("step telemetry events: adjustments=%d, rejected=%d\n", ...
        numel(stepTelemetry.adjustment_z), numel(stepTelemetry.rejected_z));
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
hasSeries = false;
if ~isempty(telemetry.adjustment_z)
    [zSorted, order] = sort(telemetry.adjustment_z, "ascend");
    plot(axStep, zSorted, telemetry.adjustment_sizes(order), ...
         '-', 'LineWidth', 1.2, 'Color', [0.00, 0.45, 0.74], ...
         'DisplayName', 'Adjusted accepted step');
    hold(axStep, 'on');
    hasSeries = true;
end
if ~isempty(telemetry.rejected_z)
    [zRejectedSorted, order] = sort(telemetry.rejected_z, "ascend");
    scatter(axStep, zRejectedSorted, telemetry.rejected_attempted_sizes(order), ...
            14, [0.85, 0.33, 0.10], 'filled', ...
            'DisplayName', 'Rejected attempted step', ...
            'MarkerFaceAlpha', 0.7, ...
            'MarkerEdgeAlpha', 0.7);
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
telemetry.adjustment_z = zeros(0, 1);
telemetry.adjustment_sizes = zeros(0, 1);
telemetry.rejected_z = zeros(0, 1);
telemetry.rejected_attempted_sizes = zeros(0, 1);
end

function telemetry = parse_step_telemetry(logText)
telemetry = empty_step_telemetry();
if isempty(logText)
    return;
end

lines = splitlines(string(logText));
numLines = numel(lines);
i = 1;
while i <= numLines
    line = strtrim(lines(i));
    if line ~= "[nlolib] step_adjustment:" && line ~= "[nlolib] step_rejected:"
        i = i + 1;
        continue;
    end

isAdjustment = (line == "[nlolib] step_adjustment:");
zCurrent = NaN;
stepSize = NaN;
attemptedStep = NaN;
    i = i + 1;
    while i <= numLines
        current = strtrim(lines(i));
        if startsWith(current, "[nlolib]")
            break;
        end
        if startsWith(current, "- ")
            [key, value, hasPair] = parse_log_key_value(extractAfter(current, 2));
            if hasPair
                switch key
                    case "z_current"
                        zCurrent = parse_log_number(value);
                    case "step_size"
                        stepSize = parse_log_number(value);
                    case "attempted_step"
                        attemptedStep = parse_log_number(value);
                end
            end
        end
        i = i + 1;
    end

    if isAdjustment
        if isfinite(zCurrent) && isfinite(stepSize)
            telemetry.adjustment_z(end + 1, 1) = zCurrent; %#ok<AGROW>
            telemetry.adjustment_sizes(end + 1, 1) = stepSize; %#ok<AGROW>
        end
    else
        if isfinite(zCurrent) && isfinite(attemptedStep)
            telemetry.rejected_z(end + 1, 1) = zCurrent; %#ok<AGROW>
            telemetry.rejected_attempted_sizes(end + 1, 1) = attemptedStep; %#ok<AGROW>
        end
    end
end
end

function [key, value, hasPair] = parse_log_key_value(textLine)
lineText = string(textLine);
delimiterIndex = strfind(char(lineText), ':');
if isempty(delimiterIndex)
    key = "";
    value = "";
    hasPair = false;
    return;
end
firstColon = delimiterIndex(1);
key = strtrim(extractBefore(lineText, firstColon));
value = strtrim(extractAfter(lineText, firstColon));
hasPair = strlength(key) > 0;
end

function out = parse_log_number(textValue)
if strlength(textValue) == 0
    out = NaN;
    return;
end
cleaned = replace(string(textValue), ",", "");
cleaned = erase(cleaned, "%");
out = str2double(cleaned);
if ~isfinite(out)
    out = NaN;
end
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
