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
propagateOptions = struct();
propagateOptions.propagation_distance = zFinal;
propagateOptions.records = numRecords;
propagateOptions.preset = "balanced";
propagateOptions.exec_options = execOptions;
result = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);
aRecords = result.records;
zRecords = result.z_axis;

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
[zMap, lambdaNm, spectralMap] = compute_wavelength_spectral_map_from_records( ...
    aRecords, zRecords, dt, lambda0Nm);
if any(~isfinite(spectralMap), "all")
    error("spectral map contains non-finite values.");
end

outputDir = fullfile(repoRoot, "examples", "matlab", "output", "second_order_soliton");
if ~isfolder(outputDir)
    mkdir(outputDir);
end

savedPaths = strings(0, 1);
savedPaths(end + 1, 1) = string(backend.plot_intensity_colormap_vs_propagation( ...
    lambdaNm, zMap, spectralMap, ...
    fullfile(outputDir, "wavelength_intensity_colormap.png"), ...
    "x_label", "Wavelength (nm)", ...
    "y_label", "Propagation distance z (m)", ...
    "title", "Spectral Intensity Envelope vs Propagation Distance", ...
    "colorbar_label", "Normalized spectral intensity", ...
    "cmap", "magma")); %#ok<AGROW>

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
    aRecords, zSamples, dt, lambda0Nm)

n = size(aRecords, 2);
freqShifted = fftshift(fftfreq_matlab(n, dt));
nu0 = 299792.458 / lambda0Nm;
nu = nu0 + freqShifted;
valid = nu > 0.0;
lambdaNm = 299792.458 ./ nu(valid);

spectra = fftshift(fft(aRecords, [], 2), 2);
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
        left = max(supportIdx(1) - 2, 1);
        right = min(supportIdx(end) + 2, numel(lambdaNm));
        lambdaNm = lambdaNm(left:right);
        specMap = specMap(:, left:right);
    end
end
zMap = zSamples;
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
