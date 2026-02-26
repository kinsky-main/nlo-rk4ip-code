function slopeRelError = linear_drift_rk4ip()
%LINEAR_DRIFT_RK4IP Linear dispersive drift example with plot parity.

repoRoot = setup_matlab_example_environment();

numSamples = 1024;
dt = 0.01;
sigma = 0.20;
beta2 = 0.05;
gamma = 0.0;
chirp = 12.0;
zFinal = 1.0;
numRecords = 180;

t = backend.centered_time_grid(numSamples, dt);
field0 = gaussian_with_phase_ramp(t, sigma, chirp);

pulse = struct();
pulse.samples = field0;
pulse.delta_time = dt;
pulse.pulse_period = numSamples * dt;
pulse.frequency_grid = complex(backend.angular_frequency_grid(numSamples, dt), zeros(1, numSamples));

linearOperator = struct();
linearOperator.expr = "i*beta2*w*w-loss";
linearOperator.params = struct('beta2', 0.5 * beta2, 'loss', 0.0);

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
propagateOptions.preset = "accuracy";
propagateOptions.exec_options = execOptions;
result = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);
records = result.records;
zRecords = result.z_axis;

recordNorms = vecnorm(records, 2, 2);
centroid = centroid_curve(t, records);
centroidShift = centroid - centroid(1);
measuredSlope = polyfit(zRecords, centroidShift.', 1);
measuredSlope = measuredSlope(1);
predictedSlope = beta2 * chirp;
theoryShift = predictedSlope * zRecords;
slopeRelError = abs(measuredSlope - predictedSlope) / max(abs(predictedSlope), 1e-12);
referenceRecords = linear_reference_records(field0, zRecords, beta2, dt);
errorCurve = relative_l2_error_curve(records, referenceRecords);

outputDir = fullfile(repoRoot, "examples", "matlab", "output", "linear_drift");
if ~isfolder(outputDir)
    mkdir(outputDir);
end

savedPaths = strings(0, 1);
savedPaths(end + 1, 1) = string(backend.plot_intensity_colormap_vs_propagation( ...
    t, zRecords, abs(records).^2, ...
    fullfile(outputDir, "intensity_propagation_map.png"), ...
    "x_label", "Time t", ...
    "title", "Linear Drift: Temporal Intensity Propagation", ...
    "colorbar_label", "Normalized intensity", ...
    "cmap", "viridis")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_re_im_comparison( ...
    t, records(1, :), records(end, :), ...
    fullfile(outputDir, "final_re_im_comparison.png"), ...
    "x_label", "Time t", ...
    "title", "Linear Drift: Final Re/Im Comparison", ...
    "reference_label", "Initial", ...
    "final_label", "Final")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_intensity_comparison( ...
    t, records(1, :), records(end, :), ...
    fullfile(outputDir, "final_intensity_comparison.png"), ...
    "x_label", "Time t", ...
    "title", "Linear Drift: Final Intensity Comparison", ...
    "reference_label", "Initial", ...
    "final_label", "Final")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_total_error_over_propagation( ...
    zRecords, errorCurve, ...
    fullfile(outputDir, "total_error_over_propagation.png"), ...
    "title", "Linear Drift: Full-Window Relative L2 Error Over Propagation", ...
    "y_label", "Relative L2 error (numerical vs analytical)")); %#ok<AGROW>

fprintf("linear drift example completed.\n");
fprintf("record norms: first=%.6e, last=%.6e, min=%.6e\n", ...
        recordNorms(1), recordNorms(end), min(recordNorms));
fprintf("centroid shift: z0=%.6e, z_end=%.6e\n", centroidShift(1), centroidShift(end));
fprintf("slope (signed): measured=%.6e, predicted=%.6e\n", measuredSlope, predictedSlope);
fprintf("slope relative error: %.6e\n", slopeRelError);
fprintf("centroid theory final shift: %.6e\n", theoryShift(end));
for idx = 1:numel(savedPaths)
    fprintf("saved plot: %s\n", savedPaths(idx));
end
end

function out = gaussian_with_phase_ramp(t, sigma, d)
envelope = exp(-((t / sigma) .^ 2));
out = envelope .* exp((-1.0i) * d * t);
end

function out = centroid_curve(t, records)
intensity = abs(records).^2;
weighted = intensity * t(:);
norms = sum(intensity, 2);
safeNorms = max(norms, 1.0);
out = weighted ./ safeNorms;
out = out(:).';
end

function referenceRecords = linear_reference_records(field0, zRecords, beta2, dt)
n = numel(field0);
omega = backend.angular_frequency_grid(n, dt);
phaseCoeff = 0.5 * beta2 * (omega .^ 2);
spectrum0 = fft(field0);
numZ = numel(zRecords);
referenceRecords = complex(zeros(numZ, n), zeros(numZ, n));
phase = complex(zeros(1, n), zeros(1, n));
spectrumZ = complex(zeros(1, n), zeros(1, n));
for idx = 1:numZ
    z = zRecords(idx);
    phase(:) = exp(1.0i * (phaseCoeff * z));
    spectrumZ(:) = spectrum0 .* phase;
    referenceRecords(idx, :) = ifft(spectrumZ);
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
