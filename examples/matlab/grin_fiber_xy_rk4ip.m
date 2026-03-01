function grin_fiber_xy_rk4ip()
%GRIN_FIBER_XY_RK4IP GRIN transverse phase validations with analytical refs.

repoRoot = setup_matlab_example_environment();

api = nlolib.NLolib();
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "auto", ...
    "device_heap_fraction", 0.70);
outputRoot = fullfile(repoRoot, "examples", "matlab", "output", "grin_fiber_xy");
if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

scenarios = { ...
    struct('scenario_name', "grin_phase_validation_symmetric", ...
           'nx', 384, 'ny', 384, 'dx', 0.6, 'dy', 0.6, 'w0', 7.5, ...
           'grin_gx', 2.0e-4, 'grin_gy', 2.0e-4, 'x_offset', 0.0, 'y_offset', 0.0, ...
           'propagation_distance', 0.25, 'num_records', 8), ...
    struct('scenario_name', "grin_phase_validation_astigmatic_offset", ...
           'nx', 384, 'ny', 320, 'dx', 0.6, 'dy', 0.7, 'w0', 8.0, ...
           'grin_gx', 3.0e-4, 'grin_gy', 1.2e-4, 'x_offset', 2.5, 'y_offset', -1.8, ...
           'propagation_distance', 0.25, 'num_records', 8) ...
};

allSaved = strings(0, 1);
for idx = 1:numel(scenarios)
    [savedPaths, finalError, powerDrift] = run_phase_validation( ...
        api, scenarios{idx}, simOptions, outputRoot);
    fprintf("%s: final_error=%.6e power_drift=%.6e\n", ...
            scenarios{idx}.scenario_name, finalError, powerDrift);
    allSaved = [allSaved; savedPaths(:)]; %#ok<AGROW>
end

[proofSaved, proofError] = run_diffraction_operator_proof_check(api, simOptions, outputRoot);
fprintf("diffraction_operator_proof_check: final_error=%.6e\n", proofError);
if ~isempty(proofSaved)
    allSaved = [allSaved; string(proofSaved)]; %#ok<AGROW>
end

for idx = 1:numel(allSaved)
    fprintf("saved plot: %s\n", allSaved(idx));
end
end

function [savedPaths, finalError, powerDrift] = run_phase_validation(api, scenario, simOptions, outputRoot)
nx = scenario.nx;
ny = scenario.ny;
numRecords = scenario.num_records;
nxy = nx * ny;

x = ((0:(nx - 1)) - 0.5 * (nx - 1)) * scenario.dx;
y = ((0:(ny - 1)) - 0.5 * (ny - 1)) * scenario.dy;
[xx, yy] = meshgrid(x, y);

phaseUnit = (scenario.grin_gx * (xx .* xx)) + (scenario.grin_gy * (yy .* yy));
field0 = exp(-(((xx - scenario.x_offset) .^ 2 + (yy - scenario.y_offset) .^ 2) / (scenario.w0 ^ 2)));
field0 = complex(field0, zeros(size(field0)));
field0Flat = flatten_xy_row_major(field0);

pulse = struct();
pulse.samples = field0Flat;
pulse.delta_time = 1.0;
pulse.pulse_period = double(nx);
pulse.frequency_grid = complex(zeros(1, nxy), zeros(1, nxy));
pulse.tensor_nt = 1;
pulse.tensor_nx = nx;
pulse.tensor_ny = ny;
pulse.tensor_layout = 0;
pulse.delta_x = scenario.dx;
pulse.delta_y = scenario.dy;
pulse.potential_grid = flatten_xy_row_major(complex(phaseUnit, zeros(size(phaseUnit))));

linearOperator = struct();
linearOperator.expr = "i*beta2*w*w-loss";
linearOperator.params = struct('beta2', 0.0, 'loss', 0.0);

nonlinearOperator = struct();
nonlinearOperator.expr = "i*A*(gamma*I + V)";
nonlinearOperator.params = struct('gamma', 0.0);

execOptions = backend.make_exec_options(simOptions, numRecords);
propagateOptions = struct();
propagateOptions.propagation_distance = scenario.propagation_distance;
propagateOptions.records = numRecords;
propagateOptions.preset = "accuracy";
propagateOptions.exec_options = execOptions;
result = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);
recordsFlat = result.records;
zRecords = result.z_axis;
records = unflatten_records_row_major(recordsFlat, numRecords, ny, nx);

analyticalRecords = zeros(size(records));
for ridx = 1:numRecords
    z = zRecords(ridx);
    analyticalRecords(ridx, :, :) = field0 .* exp((1.0i) * phaseUnit * z);
end

records2d = reshape(records, [numRecords, nxy]);
analytical2d = reshape(analyticalRecords, [numRecords, nxy]);
fullError = relative_l2_error_curve(records2d, analytical2d);

centerRow = floor(ny / 2) + 1;
profileRecords = squeeze(records(:, centerRow, :));
analyticalProfiles = squeeze(analyticalRecords(:, centerRow, :));
profileError = relative_l2_error_curve(profileRecords, analyticalProfiles);

outDir = fullfile(outputRoot, char(scenario.scenario_name));
if ~isfolder(outDir)
    mkdir(outDir);
end

savedPaths = strings(0, 1);
savedPaths(end + 1, 1) = string(backend.plot_intensity_colormap_vs_propagation( ...
    x, zRecords, abs(profileRecords).^2, ...
    fullfile(outDir, "centerline_intensity_colormap.png"), ...
    "x_label", "Transverse coordinate x", ...
    "y_label", "Propagation distance z", ...
    "title", sprintf("%s: center-line intensity vs propagation", scenario.scenario_name), ...
    "colorbar_label", "Normalized center-line intensity", ...
    "cmap", "viridis")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_re_im_comparison( ...
    x, analyticalProfiles(end, :), profileRecords(end, :), ...
    fullfile(outDir, "final_re_im_profile_comparison.png"), ...
    "x_label", "Transverse coordinate x", ...
    "title", sprintf("%s: final Re/Im profile (analytical vs numerical)", scenario.scenario_name), ...
    "reference_label", "Analytical final", ...
    "final_label", "Numerical final")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_intensity_comparison( ...
    x, analyticalProfiles(end, :), profileRecords(end, :), ...
    fullfile(outDir, "final_intensity_profile_comparison.png"), ...
    "x_label", "Transverse coordinate x", ...
    "title", sprintf("%s: final intensity profile (analytical vs numerical)", scenario.scenario_name), ...
    "reference_label", "Analytical final", ...
    "final_label", "Numerical final")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_total_error_over_propagation( ...
    zRecords, fullError, ...
    fullfile(outDir, "full_field_relative_error_over_propagation.png"), ...
    "title", sprintf("%s: full-field error over propagation", scenario.scenario_name), ...
    "y_label", "Relative L2 error (full field)")); %#ok<AGROW>

saved3dNum = backend.plot_3d_intensity_scatter_propagation( ...
    x, y, zRecords, records, ...
    fullfile(outDir, "propagation_3d_numerical_scatter.png"), ...
    "intensity_cutoff", 0.08, ...
    "xy_stride", 16, ...
    "min_marker_size", 2.0, ...
    "max_marker_size", 34.0, ...
    "title", sprintf("%s: numerical propagation (3D scatter)", scenario.scenario_name));
if ~isempty(saved3dNum)
    savedPaths(end + 1, 1) = string(saved3dNum); %#ok<AGROW>
end

saved3dAna = backend.plot_3d_intensity_scatter_propagation( ...
    x, y, zRecords, analyticalRecords, ...
    fullfile(outDir, "propagation_3d_analytical_scatter.png"), ...
    "intensity_cutoff", 0.08, ...
    "xy_stride", 16, ...
    "min_marker_size", 2.0, ...
    "max_marker_size", 34.0, ...
    "title", sprintf("%s: analytical propagation (3D scatter)", scenario.scenario_name));
if ~isempty(saved3dAna)
    savedPaths(end + 1, 1) = string(saved3dAna); %#ok<AGROW>
end

inPower = sum(abs(squeeze(records(1, :, :))).^2, "all");
outPower = sum(abs(squeeze(records(end, :, :))).^2, "all");
powerDrift = abs(outPower - inPower) / max(inPower, 1e-12);
finalError = fullError(end);
finalProfileError = profileError(end);
fprintf("%s: records=%d, grid=(%d,%d), final_full_error=%.6e, final_profile_error=%.6e, power_drift=%.6e\n", ...
        scenario.scenario_name, numRecords, ny, nx, finalError, finalProfileError, powerDrift);
end

function [savedPath, finalError] = run_diffraction_operator_proof_check(api, simOptions, outputRoot)
nx = 128;
ny = 128;
dx = 0.7;
dy = 0.7;
w0 = 9.0;
grinGx = 2.5e-4;
grinGy = 1.7e-4;
diffractionCoeff = -0.02;
propagationDistance = 0.20;
numRecords = 6;
maxFinalError = 5.0e-2;

x = ((0:(nx - 1)) - 0.5 * (nx - 1)) * dx;
y = ((0:(ny - 1)) - 0.5 * (ny - 1)) * dy;
[xx, yy] = meshgrid(x, y);
field0 = exp(-((xx .* xx + yy .* yy) / (w0 ^ 2)));
field0 = complex(field0, zeros(size(field0)));
field0Flat = flatten_xy_row_major(field0);
nxy = nx * ny;

kx = backend.angular_frequency_grid(nx, dx);
ky = backend.angular_frequency_grid(ny, dy);
[kxx, kyy] = meshgrid(kx, ky);
k2 = (kxx .* kxx) + (kyy .* kyy);

execOptions = backend.make_exec_options(simOptions, numRecords);

cfgTensor = struct();
cfgTensor.num_time_samples = nxy;
cfgTensor.propagation_distance = propagationDistance;
cfgTensor.starting_step_size = 1.0e-3;
cfgTensor.max_step_size = 2.0e-3;
cfgTensor.min_step_size = 5.0e-5;
cfgTensor.error_tolerance = 1.0e-7;
cfgTensor.pulse_period = 1.0;
cfgTensor.delta_time = 1.0;
cfgTensor.tensor_nt = 1;
cfgTensor.tensor_nx = nx;
cfgTensor.tensor_ny = ny;
cfgTensor.tensor_layout = 0;
cfgTensor.delta_x = dx;
cfgTensor.delta_y = dy;
cfgTensor.frequency_grid = complex(0.0, 0.0);
cfgTensor.runtime = struct( ...
    'linear_factor_expr', "i*c0*(kx*kx + ky*ky)", ...
    'linear_expr', "exp(h*D)", ...
    'nonlinear_expr', "0", ...
    'constants', diffractionCoeff);

tensorResult = api.propagate(cfgTensor, field0Flat, numRecords, execOptions);
tensorRecords = unflatten_records_row_major(tensorResult.records, numRecords, ny, nx);
zRecords = tensorResult.z_axis;

fft0 = fft2(field0);
fftReference = complex(zeros(numRecords, ny, nx));
for ridx = 1:numRecords
    z = zRecords(ridx);
    spectralPhase = exp((1.0i) * diffractionCoeff .* k2 .* z);
    fftReference(ridx, :, :) = ifft2(fft0 .* spectralPhase);
end

reference2d = reshape(fftReference, [numRecords, nxy]);
tensor2d = reshape(tensorRecords, [numRecords, nxy]);
errorCurve = relative_l2_error_curve(tensor2d, reference2d);
finalError = errorCurve(end);
if finalError > maxFinalError
    error("Diffraction operator proof check failed (tensor vs FFT2): final error %.6e exceeds %.6e", ...
          finalError, maxFinalError);
end

outDir = fullfile(outputRoot, "diffraction_operator_proof_check");
if ~isfolder(outDir)
    mkdir(outDir);
end
savedPath = backend.plot_total_error_over_propagation( ...
    zRecords, errorCurve, ...
    fullfile(outDir, "tensor_diffraction_operator_error_vs_fft2_reference.png"), ...
    "title", "GRIN diffraction proof check: tensor operator vs FFT2 reference", ...
    "y_label", "Relative L2 error (tensor vs FFT2 reference)");
end

function flat = flatten_xy_row_major(field)
flat = reshape(field.', 1, []);
end

function records = unflatten_records_row_major(recordsFlat, numRecords, ny, nx)
records = zeros(numRecords, ny, nx);
for ridx = 1:numRecords
    records(ridx, :, :) = reshape(recordsFlat(ridx, :), [nx, ny]).';
end
end

function out = relative_l2_error_curve(records, referenceRecords)
if any(size(records) ~= size(referenceRecords))
    error("records and referenceRecords must have the same shape.");
end

out = zeros(size(records, 1), 1);
for idx = 1:size(records, 1)
    ref = referenceRecords(idx, :);
    refNorm = norm(ref, 2);
    safeRefNorm = refNorm;
    if safeRefNorm <= 0.0
        safeRefNorm = 1.0;
    end
    out(idx) = norm(records(idx, :) - ref, 2) / safeRefNorm;
end
out = out(:).';
end
