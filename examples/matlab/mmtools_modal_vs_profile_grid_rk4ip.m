function mmtools_modal_vs_profile_grid_rk4ip(varargin)
%MMTOOLS_MODAL_VS_PROFILE_GRID_RK4IP Compare MMTools modal GPU propagation against nlolib profile-grid GPU propagation.
%   This is an apples-to-apples observable comparison on one deterministic,
%   passive GRIN/Kerr multimode case. It matches center wavelength,
%   propagation length, temporal-window order of magnitude, launch energy,
%   and broad multimode excitation intent. It does not attempt mode-by-mode
%   field identity. MMTools is run through its fixed-step zero-coupling
%   RMC path because that is the available fixed-step no-gain MPA entry
%   point; the off-diagonal coupling matrices are zero.
%
%   Optional name-value inputs:
%     'smoke'  - true uses a reduced-size manual validation case.

params = parse_inputs(varargin{:});
repoRoot = setup_matlab_example_environment();
outputDir = fullfile(repoRoot, "examples", "matlab", "output", "mmtools_modal_vs_profile_grid");
if ~isfolder(outputDir)
    mkdir(outputDir);
end

mmtoolsRoot = "C:\Users\Wenzel\Final Year Project\MMTools";
if ~isfolder(mmtoolsRoot)
    error("MMTools root not found at %s.", mmtoolsRoot);
end

if gpuDeviceCount("available") < 1
    error("This comparison requires a GPU for both MMTools and nlolib, but MATLAB reports no available GPU device.");
end
gpuDevice(1);

caseCfg = build_case_config(params.smoke);
add_mmtools_paths(mmtoolsRoot);

fprintf("Verified MATLAB GPU, constructing nlolib handle.\n");
api = nlolib.NLolib();
verify_nlolib_gpu_available(api);
fprintf("nlolib GPU smoke passed. Running MMTools case.\n");

[mmResult, mmRuntimeSeconds] = run_mmtools_case(mmtoolsRoot, caseCfg);
fprintf("MMTools case complete. Running nlolib full-profile case.\n");
[nloResult, nloRuntimeSeconds] = run_nlolib_case(api, caseCfg);
fprintf("nlolib full-profile case complete. Writing outputs.\n");

summary = table( ...
    ["MMTools"; "nlolib"], ...
    ["GPU"; "GPU"], ...
    [mmRuntimeSeconds; nloRuntimeSeconds], ...
    [mmResult.energy_drift; nloResult.energy_drift], ...
    [mmResult.rms_radius_um; nloResult.rms_radius_um], ...
    [max(mmResult.temporal_marginal); max(nloResult.temporal_marginal)], ...
    [max(mmResult.spectrum); max(nloResult.spectrum)], ...
    'VariableNames', { ...
        'solver', 'backend', 'runtime_seconds', 'energy_drift', ...
        'rms_radius_um', 'final_temporal_peak', 'final_spectrum_peak' ...
    } ...
);
summaryPath = fullfile(outputDir, "mmtools_modal_vs_profile_grid_summary.csv");
writetable(summary, summaryPath);

save_runtime_plot(summary, fullfile(outputDir, "mmtools_modal_vs_profile_grid_runtime.png"));
save_curve_comparison( ...
    mmResult.time_ps, mmResult.temporal_marginal, ...
    nloResult.time_ps, nloResult.temporal_marginal, ...
    "Time (ps)", "Final temporal marginal (a.u.)", ...
    "MMTools", "nlolib", ...
    fullfile(outputDir, "mmtools_modal_vs_profile_grid_temporal_marginal.png"));
save_curve_comparison( ...
    mmResult.frequency_thz, mmResult.spectrum, ...
    nloResult.frequency_thz, nloResult.spectrum, ...
    "Frequency (THz)", "Final spectrum (a.u.)", ...
    "MMTools", "nlolib", ...
    fullfile(outputDir, "mmtools_modal_vs_profile_grid_spectrum.png"));
save_spatial_comparison( ...
    mmResult.x_um, mmResult.y_um, mmResult.spatial_intensity, ...
    nloResult.x_um, nloResult.y_um, nloResult.spatial_intensity, ...
    fullfile(outputDir, "mmtools_modal_vs_profile_grid_spatial_intensity.png"));

fprintf("Saved summary CSV: %s\n", summaryPath);
fprintf("MMTools runtime: %.3f s, energy drift: %.3e, RMS radius: %.3f um\n", ...
        mmRuntimeSeconds, mmResult.energy_drift, mmResult.rms_radius_um);
fprintf("nlolib  runtime: %.3f s, energy drift: %.3e, RMS radius: %.3f um\n", ...
        nloRuntimeSeconds, nloResult.energy_drift, nloResult.rms_radius_um);
end

function params = parse_inputs(varargin)
params = struct('smoke', false);
if mod(numel(varargin), 2) ~= 0
    error("Name-value arguments must come in pairs.");
end
for idx = 1:2:numel(varargin)
    name = lower(string(varargin{idx}));
    value = varargin{idx + 1};
    switch name
        case "smoke"
            params.smoke = logical(value);
        otherwise
            error("Unknown option '%s'.", name);
    end
end
end

function cfg = build_case_config(smoke)
rng(17, "twister");

cfg = struct();
cfg.smoke = logical(smoke);
cfg.lambda0_m = 1030e-9;
cfg.propagation_length_m = 0.012;
cfg.time_window_ps = 1.6;
cfg.tfwhm_ps = 0.045;
cfg.total_energy_nj = 5.0;
cfg.mode_count = 3;
cfg.nt = 48;
cfg.grid_n = 16;
cfg.xy_extent_um = 64.0;
cfg.chirp = 1.5;
cfg.beta2_ps2_per_m = 0.0208;
cfg.diffraction_coeff = -0.0045;
cfg.grin_strength = 7.0e-4;
cfg.gamma_grid = 0.010;
cfg.step_count = 4;

if ~cfg.smoke
    cfg.propagation_length_m = 0.15;
    cfg.time_window_ps = 6.0;
    cfg.tfwhm_ps = 0.08;
    cfg.total_energy_nj = 17.0;
    cfg.mode_count = 21;
    cfg.nt = 512;
    cfg.grid_n = 64;
    cfg.xy_extent_um = 120.0;
    cfg.chirp = 3.0;
    cfg.grin_strength = 1.5e-3;
    cfg.gamma_grid = 0.026;
    cfg.step_count = 64;
end

coeffs = randn(1, cfg.mode_count) + 1i * randn(1, cfg.mode_count);
coeffs = coeffs .* linspace(1.0, 0.35, cfg.mode_count);
coeffs = coeffs ./ norm(coeffs, 2);
cfg.modal_coeffs = coeffs;

spatialCoeffCount = min(cfg.mode_count, 10);
spatialCoeffs = coeffs(1:spatialCoeffCount);
spatialCoeffs = spatialCoeffs ./ norm(spatialCoeffs, 2);
cfg.spatial_basis_coeffs = spatialCoeffs;
end

function add_mmtools_paths(mmtoolsRoot)
pathsToAdd = { ...
    fullfile(mmtoolsRoot, "GMMNLSE", "GMMNLSE algorithm"), ...
    fullfile(mmtoolsRoot, "GMMNLSE", "user_helpers") ...
};
for idx = 1:numel(pathsToAdd)
    if isfolder(pathsToAdd{idx})
        addpath(pathsToAdd{idx}, "-begin");
    end
end
end

function verify_nlolib_gpu_available(api)
dt = 0.02;
cfg = struct( ...
    'num_time_samples', 32, ...
    'propagation_distance', 0.01, ...
    'starting_step_size', 0.001, ...
    'max_step_size', 0.02, ...
    'min_step_size', 0.000001, ...
    'error_tolerance', 1e-9, ...
    'pulse_period', 32 * dt, ...
    'delta_time', dt, ...
    'frequency_grid', complex(zeros(1, 32), zeros(1, 32)) ...
);
field0 = complex(ones(1, 32), zeros(1, 32));
execOptions = backend.make_exec_options( ...
    backend.default_simulation_options("backend", "vulkan", "fft_backend", "auto"), ...
    1);
try
    api.propagate(cfg, field0, 1, execOptions);
catch ME
    error("nlolib GPU/Vulkan execution path is unavailable: %s", ME.message);
end
end

function [result, runtimeSeconds] = run_mmtools_case(mmtoolsRoot, cfg)
fiberDir = fullfile(mmtoolsRoot, "GMMNLSE", "Fibers", "GRIN_168_400_wavelength1030nm");
tensorPath = fullfile(fiberDir, "S_tensors_21modes.mat");
betasPath = fullfile(fiberDir, "betas.mat");
if ~isfile(tensorPath) || ~isfile(betasPath)
    error("MMTools GRIN fiber assets are incomplete under %s.", fiberDir);
end

loadedBetas = load(betasPath, "betas");
loadedTensors = load(tensorPath, "SR");
if size(loadedBetas.betas, 2) < cfg.mode_count || size(loadedTensors.SR, 1) < cfg.mode_count
    error("Requested mode_count=%d exceeds available GRIN fiber assets.", cfg.mode_count);
end

fiber = struct();
fiber.MM_folder = [fiberDir filesep];
fiber.betas = loadedBetas.betas(:, 1:cfg.mode_count);
fiber.SR = loadedTensors.SR(1:cfg.mode_count, 1:cfg.mode_count, 1:cfg.mode_count, 1:cfg.mode_count, :);
fiber.L0 = cfg.propagation_length_m;

sim = struct();
sim.lambda0 = cfg.lambda0_m;
sim.midx = 1:cfg.mode_count;
sim.gpu_yes = true;
sim.include_Raman = false;
sim.gain_model = 0;
sim.progress_bar = false;
sim.save_period = cfg.propagation_length_m;
sim.dz = cfg.propagation_length_m / cfg.step_count;
sim.adaptive_dz.threshold = 5e-2;
sim.MPA.M = 2;
sim.MPA.n_tot_max = 6;
sim.MPA.n_tot_min = 1;
sim.MPA.tol = 1e-4;
sim.rmc.model = true;
sim.rmc.matrices = zeros(cfg.mode_count, cfg.mode_count, cfg.step_count);
sim.scalar = true;
sim.pulse_centering = true;

[fiber, sim] = load_default_GMMNLSE_propagate(fiber, sim, 'multimode');
fprintf("MMTools setup complete. Building launch condition.\n");
initialCondition = build_MMgaussian( ...
    cfg.tfwhm_ps, ...
    cfg.time_window_ps, ...
    cfg.total_energy_nj, ...
    cfg.mode_count, ...
    cfg.nt, ...
    {'ifft', 0}, ...
    cfg.modal_coeffs ...
);
initialCondition.fields = apply_temporal_chirp(initialCondition.fields, cfg.time_window_ps, cfg.chirp);
fprintf("MMTools launch ready. Starting propagation.\n");

try
    tic;
    propOutput = GMMNLSE_propagate(fiber, initialCondition, sim);
    runtimeSeconds = toc;
catch ME
    error("MMTools GPU propagation failed: %s", ME.message);
end

dt = propOutput.dt;
time_ps = (-cfg.nt / 2 : cfg.nt / 2 - 1)' * dt;
frequency_thz = sim.f0 + (-cfg.nt / 2 : cfg.nt / 2 - 1)' / cfg.time_window_ps;

initialFields = gather_if_needed(initialCondition.fields);
finalFields = gather_if_needed(propOutput.fields(:, :, end));
temporalMarginal = sum(abs(finalFields).^2, 2);
temporalMarginal = normalize_curve(temporalMarginal);
finalSpectrum = sum(abs(ifftshift(ifft(finalFields, [], 1), 1)).^2, 2);
finalSpectrum = normalize_curve(finalSpectrum);

[x_um, y_um, spatialIntensity] = mmtools_spatial_intensity(fiberDir, cfg.mode_count, finalFields);
energyInitial = modal_energy_nj(initialFields, dt);
energyFinal = modal_energy_nj(finalFields, dt);
energyDrift = relative_drift(energyInitial, energyFinal);
result = struct( ...
    'time_ps', time_ps, ...
    'frequency_thz', frequency_thz, ...
    'temporal_marginal', temporalMarginal, ...
    'spectrum', finalSpectrum, ...
    'x_um', x_um, ...
    'y_um', y_um, ...
    'spatial_intensity', spatialIntensity, ...
    'rms_radius_um', rms_radius_um(x_um, y_um, spatialIntensity), ...
    'energy_drift', energyDrift ...
);
end

function [result, runtimeSeconds] = run_nlolib_case(api, cfg)
simOptions = backend.default_simulation_options( ...
    "backend", "vulkan", ...
    "fft_backend", "auto", ...
    "device_heap_fraction", 0.70);
execOptions = backend.make_exec_options(simOptions, 2);

nt = cfg.nt;
nx = cfg.grid_n;
ny = cfg.grid_n;
dt = cfg.time_window_ps / nt;
dx = cfg.xy_extent_um / nx;
dy = cfg.xy_extent_um / ny;
time_ps = backend.centered_time_grid(nt, dt).';
x_um = ((0:(nx - 1)) - 0.5 * (nx - 1)) * dx;
y_um = ((0:(ny - 1)) - 0.5 * (ny - 1)) * dy;
[xx, yy] = meshgrid(x_um, y_um);
omega = backend.angular_frequency_grid(nt, dt);

temporal = exp(-((time_ps / (0.60 * cfg.tfwhm_ps)) .^ 2)) .* exp((-1.0i) * cfg.chirp * (time_ps / cfg.time_window_ps) .^ 2);
spatial = multimode_launch_profile(xx, yy, cfg);
field0 = zeros(nt, ny, nx);
for tidx = 1:nt
    field0(tidx, :, :) = temporal(tidx) .* spatial;
end
field0 = scale_field_energy(field0, dt, dx, dy, cfg.total_energy_nj);

potentialXY = cfg.grin_strength * ((xx / max(abs(x_um))) .^ 2 + (yy / max(abs(y_um))) .^ 2);

pulse = struct();
pulse.samples = flatten_tyx_row_major(field0);
pulse.delta_time = dt;
pulse.pulse_period = cfg.time_window_ps;
pulse.tensor_nt = nt;
pulse.tensor_nx = nx;
pulse.tensor_ny = ny;
pulse.tensor_layout = 0;
pulse.frequency_grid = complex(omega, zeros(1, nt));
pulse.delta_x = dx;
pulse.delta_y = dy;
pulse.potential_grid = repmat(flatten_xy_row_major(complex(potentialXY, zeros(size(potentialXY)))), 1, nt);

linearOperator = struct();
linearOperator.expr = "i*(beta2*wt*wt + beta_t*(kx*kx + ky*ky))";
linearOperator.params = struct('beta2', 0.5 * cfg.beta2_ps2_per_m, 'beta_t', cfg.diffraction_coeff);

nonlinearOperator = struct();
nonlinearOperator.expr = "i*A*(gamma*I + V)";
nonlinearOperator.params = struct('gamma', cfg.gamma_grid);

propagateOptions = struct();
propagateOptions.propagation_distance = cfg.propagation_length_m;
propagateOptions.records = 2;
propagateOptions.exec_options = execOptions;
propagateOptions.starting_step_size = cfg.propagation_length_m / cfg.step_count;
propagateOptions.max_step_size = cfg.propagation_length_m / cfg.step_count;
propagateOptions.min_step_size = cfg.propagation_length_m / cfg.step_count;
propagateOptions.error_tolerance = 1e-9;

try
    tic;
    propagation = api.propagate(pulse, linearOperator, nonlinearOperator, propagateOptions);
    runtimeSeconds = toc;
catch ME
    error("nlolib GPU propagation failed: %s", ME.message);
end

records = unflatten_tyx_records(propagation.records, 2, nt, ny, nx);
initialField = squeeze(records(1, :, :, :));
finalField = squeeze(records(2, :, :, :));
temporalMarginal = squeeze(sum(sum(abs(finalField).^2, 3), 2));
temporalMarginal = normalize_curve(temporalMarginal);
frequency_thz = (299792.458 / (cfg.lambda0_m * 1e9)) + ((-nt / 2 : nt / 2 - 1)' / cfg.time_window_ps);
finalSpectrum = squeeze(sum(sum(abs(ifftshift(ifft(finalField, [], 1), 1)).^2, 3), 2));
finalSpectrum = normalize_curve(finalSpectrum);
spatialIntensity = squeeze(sum(abs(finalField).^2, 1)) * dt * 1e-3;
spatialIntensity = spatialIntensity ./ max(max(spatialIntensity), 1e-12);

energyInitial = field_energy_nj(initialField, dt, dx, dy);
energyFinal = field_energy_nj(finalField, dt, dx, dy);
energyDrift = relative_drift(energyInitial, energyFinal);
result = struct( ...
    'time_ps', time_ps, ...
    'frequency_thz', frequency_thz, ...
    'temporal_marginal', temporalMarginal, ...
    'spectrum', finalSpectrum, ...
    'x_um', x_um(:), ...
    'y_um', y_um(:), ...
    'spatial_intensity', spatialIntensity, ...
    'rms_radius_um', rms_radius_um(x_um(:), y_um(:), spatialIntensity), ...
    'energy_drift', energyDrift ...
);
end

function fieldsOut = apply_temporal_chirp(fieldsIn, timeWindowPs, chirp)
fieldsOut = fieldsIn;
nt = size(fieldsIn, 1);
t = (-nt / 2 : nt / 2 - 1)' * (timeWindowPs / nt);
phase = exp((-1.0i) * chirp * (t / timeWindowPs) .^ 2);
fieldsOut = fieldsOut .* phase;
end

function out = gather_if_needed(value)
if isa(value, 'gpuArray')
    out = gather(value);
else
    out = value;
end
end

function energy = modal_energy_nj(fieldsTM, dt)
energy = sum(sum(abs(fieldsTM).^2, 2), 1) * dt * 1e-3;
energy = double(energy);
end

function energy = field_energy_nj(fieldTYX, dt, dx, dy)
energy = sum(abs(fieldTYX).^2, 'all') * dt * dx * dy * 1e-3;
energy = double(energy);
end

function drift = relative_drift(initialValue, finalValue)
drift = abs(double(finalValue) - double(initialValue)) / max(abs(double(initialValue)), 1e-12);
end

function [x_um, y_um, spatialIntensity] = mmtools_spatial_intensity(fiberDir, modeCount, finalFieldsTM)
[modeProfiles, x_um] = load_mode_profiles(fiberDir, modeCount);
spatialField = recompose_into_space(false, modeProfiles, finalFieldsTM, "");
dt = 1.0;
spatialIntensity = squeeze(sum(abs(spatialField).^2, 1)) * dt;
spatialIntensity = spatialIntensity ./ max(max(spatialIntensity), 1e-12);
y_um = x_um;
end

function [modeProfilesSampled, xSampled_um] = load_mode_profiles(fiberDir, modeCount)
modeProfiles = [];
xAxis = [];
for modeIdx = 1:modeCount
    filePath = fullfile(fiberDir, sprintf("mode%dwavelength10300.mat", modeIdx));
    loaded = load(filePath, "phi", "x");
    if isempty(modeProfiles)
        nSide = size(loaded.phi, 1);
        modeProfiles = zeros(nSide, nSide, modeCount);
        xAxis = loaded.x(:);
    end
    modeProfiles(:, :, modeIdx) = loaded.phi;
end
xAxis = xAxis - mean(xAxis);
dx = xAxis(2) - xAxis(1);
modeProfiles = modeProfiles ./ sqrt(sum(sum(abs(modeProfiles).^2, 1), 2)) / dx;
factor = 8;
sampleIdx = 1:factor:size(modeProfiles, 1);
modeProfilesSampled = zeros(numel(sampleIdx), numel(sampleIdx), modeCount);
for modeIdx = 1:modeCount
    modeProfilesSampled(:, :, modeIdx) = modeProfiles(sampleIdx, sampleIdx, modeIdx);
end
xSampled_um = xAxis(sampleIdx);
end

function profile = multimode_launch_profile(xx, yy, cfg)
radius = sqrt(xx .^ 2 + yy .^ 2);
theta = atan2(yy, xx);
w0 = 0.28 * cfg.xy_extent_um;
g = exp(-(radius / w0) .^ 2);

basis = zeros([size(xx), numel(cfg.spatial_basis_coeffs)]);
basis(:, :, 1) = g;
if size(basis, 3) >= 2
    basis(:, :, 2) = (xx / w0) .* g;
end
if size(basis, 3) >= 3
    basis(:, :, 3) = (yy / w0) .* g;
end
if size(basis, 3) >= 4
    basis(:, :, 4) = ((xx .^ 2 - yy .^ 2) / (w0 ^ 2)) .* g;
end
if size(basis, 3) >= 5
    basis(:, :, 5) = ((xx .* yy) / (w0 ^ 2)) .* g;
end
if size(basis, 3) >= 6
    basis(:, :, 6) = ((2 * radius .^ 2 / (w0 ^ 2)) - 1) .* g;
end
if size(basis, 3) >= 7
    basis(:, :, 7) = ((radius / w0) .* cos(3 * theta)) .* g;
end
if size(basis, 3) >= 8
    basis(:, :, 8) = ((radius / w0) .* sin(3 * theta)) .* g;
end
if size(basis, 3) >= 9
    basis(:, :, 9) = ((radius .^ 2 / (w0 ^ 2)) .* cos(4 * theta)) .* g;
end
if size(basis, 3) >= 10
    basis(:, :, 10) = ((radius .^ 2 / (w0 ^ 2)) .* sin(4 * theta)) .* g;
end

profile = zeros(size(xx));
for idx = 1:numel(cfg.spatial_basis_coeffs)
    profile = profile + cfg.spatial_basis_coeffs(idx) .* basis(:, :, idx);
end
profile = profile ./ max(max(abs(profile)), 1e-12);
end

function fieldOut = scale_field_energy(fieldIn, dt, dx, dy, targetEnergyNj)
currentEnergyNj = field_energy_nj(fieldIn, dt, dx, dy);
fieldOut = fieldIn * sqrt(targetEnergyNj / max(currentEnergyNj, 1e-12));
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

function out = normalize_curve(values)
out = double(values(:));
peak = max(out);
if peak > 0.0
    out = out ./ peak;
end
end

function value = rms_radius_um(x_um, y_um, spatialIntensity)
x = double(x_um(:));
y = double(y_um(:));
[xx, yy] = meshgrid(x, y);
intensity = double(spatialIntensity);
weight = max(sum(intensity, 'all'), 1e-12);
value = sqrt(sum(((xx .^ 2) + (yy .^ 2)) .* intensity, 'all') / weight);
end

function save_runtime_plot(summaryTable, outputPath)
fig = figure("Visible", "off");
ax = axes(fig);
bar(ax, categorical(summaryTable.solver), summaryTable.runtime_seconds);
ylabel(ax, "Runtime (s)");
grid(ax, "on");
save_figure(fig, outputPath);
close(fig);
end

function save_curve_comparison(xA, yA, xB, yB, xLabel, yLabel, labelA, labelB, outputPath)
fig = figure("Visible", "off");
ax = axes(fig);
plot(ax, xA, yA, "LineWidth", 1.8, "DisplayName", labelA);
hold(ax, "on");
plot(ax, xB, yB, "--", "LineWidth", 1.8, "DisplayName", labelB);
xlabel(ax, xLabel);
ylabel(ax, yLabel);
grid(ax, "on");
legend(ax, "Location", "best");
save_figure(fig, outputPath);
close(fig);
end

function save_spatial_comparison(xA, yA, fieldA, xB, yB, fieldB, outputPath)
fig = figure("Visible", "off", "Position", [100 100 900 360]);
tiledlayout(fig, 1, 2, "Padding", "compact", "TileSpacing", "compact");

nexttile;
imagesc(xA, yA, fieldA);
axis image;
set(gca, "YDir", "normal");
title("MMTools");
xlabel("x (um)");
ylabel("y (um)");
colorbar;

nexttile;
imagesc(xB, yB, fieldB);
axis image;
set(gca, "YDir", "normal");
title("nlolib");
xlabel("x (um)");
ylabel("y (um)");
colorbar;

save_figure(fig, outputPath);
close(fig);
end

function save_figure(fig, outputPath)
parentDir = fileparts(outputPath);
if strlength(string(parentDir)) > 0 && ~isfolder(parentDir)
    mkdir(parentDir);
end
exportgraphics(fig, outputPath, "Resolution", 200);
end
