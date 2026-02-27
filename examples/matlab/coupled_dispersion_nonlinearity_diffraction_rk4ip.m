function coupled_dispersion_nonlinearity_diffraction_rk4ip()
%COUPLED_DISPERSION_NONLINEARITY_DIFFRACTION_RK4IP Coupled 3+1D demo.

repoRoot = setup_matlab_example_environment();

api = nlolib.NLolib();
simOptions = backend.default_simulation_options( ...
    "backend", "auto", ...
    "fft_backend", "auto", ...
    "device_heap_fraction", 0.70);

nt = 16;
nx = 64;
ny = 64;
dt = 0.02;
dx = 0.8;
dy = 0.8;
zFinal = 0.20;
numRecords = 10;

beta2 = 0.06;
gammaFull = 0.45;
diffractionCoeff = -0.020;
grinStrength = 1.6e-4;
temporalWidth = 0.22;
spatialWidth = 8.0;
chirp = 8.0;

totalSamples = nt * nx * ny;
limitsCfg = struct('num_time_samples', totalSamples, ...
                   'propagation_distance', zFinal, ...
                   'starting_step_size', 8e-4, ...
                   'max_step_size', 2e-3, ...
                   'min_step_size', 2e-5, ...
                   'error_tolerance', 1e-7, ...
                   'pulse_period', nt * dt, ...
                   'delta_time', dt, ...
                   'time_nt', nt, ...
                   'spatial_nx', nx, 'spatial_ny', ny, ...
                   'frequency_grid', complex(zeros(1, nt), zeros(1, nt)));
runtimeLimits = api.query_runtime_limits(limitsCfg);
if totalSamples > double(runtimeLimits.max_num_time_samples_runtime)
    error("Requested nt*nx*ny=%d exceeds runtime max_num_time_samples=%d.", ...
          totalSamples, runtimeLimits.max_num_time_samples_runtime);
end

[t, x, y, zRecords, fullRecords] = run_case( ...
    api, gammaFull, nt, nx, ny, dt, dx, dy, zFinal, numRecords, beta2, ...
    diffractionCoeff, grinStrength, temporalWidth, spatialWidth, chirp, simOptions);
[~, ~, ~, ~, linearRecords] = run_case( ...
    api, 0.0, nt, nx, ny, dt, dx, dy, zFinal, numRecords, beta2, ...
    diffractionCoeff, grinStrength, temporalWidth, spatialWidth, chirp, simOptions);

fullIntensity = abs(fullRecords).^2;
linearIntensity = abs(linearRecords).^2;
fullSpatialRecords = squeeze(sum(fullIntensity, 2));
linearSpatialRecords = squeeze(sum(linearIntensity, 2));
centerY = floor(ny / 2) + 1;
centerX = floor(nx / 2) + 1;
timeMid = floor(nt / 2) + 1;
temporalCenterFull = squeeze(fullIntensity(:, :, centerY, centerX));
xCenterTMidFull = squeeze(fullIntensity(:, timeMid, centerY, :));
errorCurve = relative_l2_error_curve( ...
    reshape(fullRecords, [numRecords, nt * ny * nx]), ...
    reshape(linearRecords, [numRecords, nt * ny * nx]));

powerFull = squeeze(sum(sum(sum(fullIntensity, 4), 3), 2));
powerLinear = squeeze(sum(sum(sum(linearIntensity, 4), 3), 2));
powerDriftFull = abs(powerFull(end) - powerFull(1)) / max(powerFull(1), 1e-12);
powerDriftLinear = abs(powerLinear(end) - powerLinear(1)) / max(powerLinear(1), 1e-12);

fprintf("coupled propagation completed: grid=(t=%d,y=%d,x=%d), records=%d, final_full_vs_linear_error=%.6e\n", ...
        nt, ny, nx, numRecords, errorCurve(end));
fprintf("power drift: full=%.6e, linear=%.6e\n", powerDriftFull, powerDriftLinear);

outputDir = fullfile(repoRoot, "examples", "matlab", "output", ...
                     "coupled_dispersion_nonlinearity_diffraction");
if ~isfolder(outputDir)
    mkdir(outputDir);
end
savedPaths = strings(0, 1);

saved3dFull = backend.plot_3d_intensity_scatter_propagation( ...
    x, y, zRecords, fullSpatialRecords, ...
    fullfile(outputDir, "full_spatial_integrated_3d_scatter.png"), ...
    "intensity_cutoff", 0.03, ...
    "adaptive_cutoff", true, ...
    "target_point_count", 24000, ...
    "xy_stride", "auto", ...
    "min_marker_size", 2.0, ...
    "max_marker_size", 36.0, ...
    "title", "Full coupled case: spatial intensity integrated over time");
if ~isempty(saved3dFull)
    savedPaths(end + 1, 1) = string(saved3dFull); %#ok<AGROW>
end

saved3dLinear = backend.plot_3d_intensity_scatter_propagation( ...
    x, y, zRecords, linearSpatialRecords, ...
    fullfile(outputDir, "linear_baseline_spatial_integrated_3d_scatter.png"), ...
    "intensity_cutoff", 0.03, ...
    "adaptive_cutoff", true, ...
    "target_point_count", 24000, ...
    "xy_stride", "auto", ...
    "min_marker_size", 2.0, ...
    "max_marker_size", 36.0, ...
    "title", "Linear baseline: spatial intensity integrated over time");
if ~isempty(saved3dLinear)
    savedPaths(end + 1, 1) = string(saved3dLinear); %#ok<AGROW>
end

savedPaths(end + 1, 1) = string(backend.plot_intensity_colormap_vs_propagation( ...
    t, zRecords, temporalCenterFull, ...
    fullfile(outputDir, "temporal_center_colormap_full.png"), ...
    "x_label", "Time t", ...
    "y_label", "Propagation distance z", ...
    "title", "Full coupled case: center-point temporal intensity vs z", ...
    "colorbar_label", "Normalized intensity", ...
    "cmap", "magma")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_intensity_colormap_vs_propagation( ...
    x, zRecords, xCenterTMidFull, ...
    fullfile(outputDir, "transverse_centerline_colormap_full.png"), ...
    "x_label", "Transverse x (t = t_mid, y = y_mid)", ...
    "y_label", "Propagation distance z", ...
    "title", "Full coupled case: transverse center-line intensity vs z", ...
    "colorbar_label", "Normalized intensity", ...
    "cmap", "viridis")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_re_im_comparison( ...
    t, squeeze(linearRecords(end, :, centerY, centerX)), ...
    squeeze(fullRecords(end, :, centerY, centerX)), ...
    fullfile(outputDir, "final_temporal_center_re_im_comparison.png"), ...
    "x_label", "Time t", ...
    "title", "Final center-point temporal field (linear baseline vs full)", ...
    "reference_label", "Linear baseline", ...
    "final_label", "Full coupled")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_final_intensity_comparison( ...
    x, squeeze(linearRecords(end, timeMid, centerY, :)), ...
    squeeze(fullRecords(end, timeMid, centerY, :)), ...
    fullfile(outputDir, "final_transverse_centerline_intensity_comparison.png"), ...
    "x_label", "Transverse coordinate x", ...
    "title", "Final transverse center-line intensity (linear baseline vs full)", ...
    "reference_label", "Linear baseline", ...
    "final_label", "Full coupled")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(backend.plot_total_error_over_propagation( ...
    zRecords, errorCurve, ...
    fullfile(outputDir, "full_vs_linear_relative_error_over_propagation.png"), ...
    "title", "Full coupled vs linear baseline: relative L2 error over z", ...
    "y_label", "Relative L2 error")); %#ok<AGROW>

savedPaths(end + 1, 1) = string(save_two_curve_plot( ...
    fullfile(outputDir, "power_over_propagation_full_vs_linear.png"), ...
    zRecords, powerFull, powerLinear, ...
    "Full coupled", "Linear baseline", ...
    "Total power sum(|A|^2)", "Power trend over propagation")); %#ok<AGROW>

for idx = 1:numel(savedPaths)
    fprintf("saved plot: %s\n", savedPaths(idx));
end
end

function [t, x, y, zRecords, records] = run_case( ...
    api, gamma, nt, nx, ny, dt, dx, dy, zFinal, numRecords, beta2, ...
    diffractionCoeff, grinStrength, temporalWidth, spatialWidth, chirp, simOptions)

t = backend.centered_time_grid(nt, dt);
x = ((0:(nx - 1)) - 0.5 * (nx - 1)) * dx;
y = ((0:(ny - 1)) - 0.5 * (ny - 1)) * dy;
[xx, yy] = meshgrid(x, y);

temporal = exp(-((t / temporalWidth) .^ 2)) .* exp((-1.0i) * chirp * t);
spatial = exp(-((xx .* xx + yy .* yy) / (spatialWidth ^ 2)));

field0 = zeros(nt, ny, nx);
for tidx = 1:nt
    field0(tidx, :, :) = temporal(tidx) .* spatial;
end

omega = backend.angular_frequency_grid(nt, dt);
k2 = k2_grid(nx, ny, dx, dy);
potential = grinStrength * (xx .* xx + yy .* yy);

field0Flat = flatten_tyx_row_major(field0);
pulse = struct();
pulse.samples = field0Flat;
pulse.delta_time = dt;
pulse.pulse_period = nt * dt;
pulse.time_nt = nt;
pulse.frequency_grid = complex(omega, zeros(1, nt));
pulse.spatial_nx = nx;
pulse.spatial_ny = ny;
pulse.delta_x = dx;
pulse.delta_y = dy;
pulse.spatial_frequency_grid = flatten_xy_row_major(k2);
pulse.potential_grid = flatten_xy_row_major(complex(potential, zeros(size(potential))));

linearOperator = struct();
linearOperator.expr = "i*beta2*w*w-loss";
linearOperator.params = struct('beta2', 0.5 * beta2, 'loss', 0.0);

transverseOperator = struct();
transverseOperator.expr = "i*beta_t*w";
transverseOperator.params = struct('beta_t', diffractionCoeff);

nonlinearOperator = struct();
nonlinearOperator.expr = "i*A*(gamma*I + V)";
nonlinearOperator.params = struct('gamma', gamma);

execOptions = backend.make_exec_options(simOptions, numRecords);
propagateOptions = struct();
propagateOptions.propagation_distance = zFinal;
propagateOptions.records = numRecords;
propagateOptions.preset = "accuracy";
propagateOptions.exec_options = execOptions;
propagateOptions.transverse_operator = transverseOperator;
result = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);

recordsFlat = result.records;
records = unflatten_tyx_records(recordsFlat, numRecords, nt, ny, nx);
zRecords = result.z_axis;
end

function k2 = k2_grid(nx, ny, dx, dy)
kx = backend.angular_frequency_grid(nx, dx);
ky = backend.angular_frequency_grid(ny, dy);
[kkx, kky] = meshgrid(kx, ky);
k2 = complex((kkx .* kkx) + (kky .* kky), zeros(ny, nx));
end

function flat = flatten_xy_row_major(matrixYX)
flat = reshape(matrixYX.', 1, []);
end

function flat = flatten_tyx_row_major(volumeTYX)
flat = reshape(permute(volumeTYX, [3, 2, 1]), 1, []);
end

function records = unflatten_tyx_records(recordsFlat, numRecords, nt, ny, nx)
records = zeros(numRecords, nt, ny, nx);
for ridx = 1:numRecords
    row = recordsFlat(ridx, :);
    records(ridx, :, :, :) = permute(reshape(row, [nx, ny, nt]), [3, 2, 1]);
end
end

function out = relative_l2_error_curve(recordsA, recordsB)
if any(size(recordsA) ~= size(recordsB))
    error("recordsA and recordsB must have the same shape.");
end
out = zeros(size(recordsA, 1), 1);
for idx = 1:size(recordsA, 1)
    a = recordsA(idx, :);
    b = recordsB(idx, :);
    denom = max(norm(b, 2), 1e-12);
    out(idx) = norm(a - b, 2) / denom;
end
out = out(:).';
end

function outPath = save_two_curve_plot(outputPath, zAxis, curveA, curveB, labelA, labelB, yLabel, titleText)
outPath = char(string(outputPath));
parent = fileparts(outPath);
if strlength(string(parent)) > 0 && ~isfolder(parent)
    mkdir(parent);
end

fig = figure("Visible", "off");
ax = axes(fig);
plot(ax, zAxis, curveA, "LineWidth", 1.9, "DisplayName", labelA);
hold(ax, "on");
plot(ax, zAxis, curveB, "--", "LineWidth", 1.8, "DisplayName", labelB);
xlabel(ax, "Propagation distance z");
ylabel(ax, yLabel);
title(ax, titleText);
grid(ax, "on");
legend(ax, "Location", "best");
print(fig, outPath, "-dpng", "-r200");
close(fig);
end
