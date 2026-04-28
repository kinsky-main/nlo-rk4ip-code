function mmtools_tensor_scaling_for_python_benchmark(varargin)
%MMTOOLS_TENSOR_SCALING_FOR_PYTHON_BENCHMARK Time MMTools at equal point counts.
%   Writes the MMTools GPU runtime CSV consumed by
%   examples/python/tensor_backend_scaling_rk4ip.py --mmtools-summary-csv.
%
%   Equal-point pairing:
%     nlolib  -> nt * nx * ny
%     MMTools -> nt * mode_count
%     mode_count = nx * ny
%
%   Optional name-value inputs:
%     'scales' - numeric vector with nt = 2*scale and nx = ny = scale.
%                Default: [3, 4, 5].
%     'warmup' - warmup runs per scale. Default: 0.
%     'runs'   - measured runs per scale. Default: 1.

params = parse_inputs(varargin{:});
repoRoot = setup_matlab_example_environment();
outputDir = fullfile(repoRoot, "examples", "matlab", "output", "mmtools_tensor_scaling");
if ~isfolder(outputDir)
    mkdir(outputDir);
end

mmtoolsRoot = 'C:\Users\Wenzel\Final Year Project\MMTools';
if ~isfolder(mmtoolsRoot)
    error("MMTools root not found at %s.", mmtoolsRoot);
end
if gpuDeviceCount("available") < 1
    error("MMTools GPU scaling requires a MATLAB GPU device.");
end
gpuDevice(1);
add_mmtools_paths(mmtoolsRoot);

rows = cell(0, 18);
for scale = reshape(params.scales, 1, [])
    cfg = build_case_config(scale);
    fprintf("MMTools total_points=%d nt=%d mode_count=%d\n", ...
            cfg.total_points, cfg.nt, cfg.mode_count);

    timings = zeros(1, params.runs);
    setupTimings = zeros(1, params.runs);
    wallTimings = zeros(1, params.runs);
    status = "ok";
    message = "";
    measured = 0;
    try
        for runIdx = 1:(params.warmup + params.runs)
            [runtimeSeconds, setupSeconds, wallSeconds] = run_mmtools_case(cfg);
            if runIdx > params.warmup
                measured = measured + 1;
                timings(measured) = runtimeSeconds;
                setupTimings(measured) = setupSeconds;
                wallTimings(measured) = wallSeconds;
            end
        end
    catch ME
        status = "error";
        message = string(ME.message);
        timings(:) = NaN;
        setupTimings(:) = NaN;
        wallTimings(:) = NaN;
    end

    if measured > 0
        meanPropagateSeconds = mean(timings(1:measured), "omitnan");
        meanSetupSeconds = mean(setupTimings(1:measured), "omitnan");
        meanWallSeconds = mean(wallTimings(1:measured), "omitnan");
        if measured > 1
            stdPropagateSeconds = std(timings(1:measured), 0, "omitnan");
            stdSetupSeconds = std(setupTimings(1:measured), 0, "omitnan");
            stdWallSeconds = std(wallTimings(1:measured), 0, "omitnan");
        else
            stdPropagateSeconds = 0.0;
            stdSetupSeconds = 0.0;
            stdWallSeconds = 0.0;
        end
        throughput = cfg.total_points / meanWallSeconds;
    else
        meanPropagateSeconds = NaN;
        meanSetupSeconds = NaN;
        meanWallSeconds = NaN;
        stdPropagateSeconds = NaN;
        stdSetupSeconds = NaN;
        stdWallSeconds = NaN;
        throughput = NaN;
    end

    rows(end + 1, :) = { ...
        "MMTools", "GPU", cfg.nt, cfg.nx, cfg.ny, cfg.mode_count, cfg.total_points, ...
        meanWallSeconds, stdWallSeconds, throughput, ...
        meanPropagateSeconds, stdPropagateSeconds, ...
        meanSetupSeconds, stdSetupSeconds, meanWallSeconds, stdWallSeconds, ...
        status, message ...
    }; %#ok<AGROW>

    fprintf("  status=%s propagate=%.6g +/- %.6g s setup=%.6g +/- %.6g s wall=%.6g +/- %.6g s throughput=%.6g points/s\n", ...
            status, meanPropagateSeconds, stdPropagateSeconds, ...
            meanSetupSeconds, stdSetupSeconds, meanWallSeconds, stdWallSeconds, ...
            throughput);
end

summary = cell2table(rows, 'VariableNames', { ...
    'solver', 'backend', 'nt', 'nx', 'ny', 'mode_count', 'total_points', ...
    'runtime_seconds', 'runtime_seconds_std', 'throughput_points_per_second', ...
    'propagate_seconds', 'propagate_seconds_std', ...
    'setup_seconds', 'setup_seconds_std', 'wall_seconds', 'wall_seconds_std', ...
    'status', 'message' ...
});
summaryPath = fullfile(outputDir, "mmtools_tensor_scaling_results.csv");
writetable(summary, summaryPath);
fprintf("Saved MMTools scaling CSV: %s\n", summaryPath);
end

function params = parse_inputs(varargin)
params = struct('scales', [3, 4, 5], 'warmup', 0, 'runs', 1);
if mod(numel(varargin), 2) ~= 0
    error("Name-value arguments must come in pairs.");
end
for idx = 1:2:numel(varargin)
    name = lower(string(varargin{idx}));
    value = varargin{idx + 1};
    switch name
        case "scales"
            params.scales = double(value);
        case "warmup"
            params.warmup = double(value);
        case "runs"
            params.runs = double(value);
        otherwise
            error("Unknown option '%s'.", name);
    end
end
params.scales = unique(round(params.scales(:).'), "stable");
params.warmup = max(0, round(params.warmup));
params.runs = max(1, round(params.runs));
end

function cfg = build_case_config(scale)
rng(17 + scale, "twister");
cfg = struct();
cfg.scale = scale;
cfg.nt = 2 ^ 10;
cfg.nx = scale;
cfg.ny = scale;
cfg.mode_count = cfg.nx * cfg.ny;
cfg.total_points = cfg.nt * cfg.mode_count;
cfg.lambda0_m = 1030e-9;
cfg.propagation_length_m = 0.20;
cfg.time_window_ps = 2.56;
cfg.tfwhm_ps = 0.045;
cfg.total_energy_nj = 5.0;
cfg.chirp = 1.5;
cfg.starting_step_size_m = 1.0e-4;
cfg.max_step_size_m = 0.02;
cfg.adaptive_threshold = 1.0e-9;
cfg.mpa_tolerance = 1.0e-9;
cfg.beta0_base = 8.8268e6;
cfg.beta0_spacing = 25.0;
cfg.beta1 = 0.0;
cfg.beta2 = 0.0209;

coeffs = randn(1, cfg.mode_count) + 1i * randn(1, cfg.mode_count);
coeffs = coeffs .* linspace(1.0, 0.35, cfg.mode_count);
cfg.modal_coeffs = coeffs ./ norm(coeffs, 2);
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

function [runtimeSeconds, setupSeconds, wallSeconds] = run_mmtools_case(cfg)
wallTimer = tic;
device = gpuDevice();
fiber = struct();
fiber.betas = synthetic_betas(cfg);
fiber.SR = synthetic_sr_tensor(cfg.mode_count);
fiber.L0 = cfg.propagation_length_m;

sim = struct();
sim.lambda0 = cfg.lambda0_m;
sim.midx = 1:cfg.mode_count;
sim.gpu_yes = true;
sim.include_Raman = false;
sim.gain_model = 0;
sim.progress_bar = false;
sim.save_period = cfg.propagation_length_m;
sim.dz = cfg.starting_step_size_m;
sim.MPA.M = 8;
sim.MPA.n_tot_min = 1;
sim.MPA.tol = cfg.mpa_tolerance;
sim.scalar = true;
sim.pulse_centering = true;
sim.adaptive_dz.threshold = cfg.adaptive_threshold;
sim.adaptive_dz.max_dz = cfg.max_step_size_m;

[fiber, sim] = load_default_GMMNLSE_propagate(fiber, sim, 'multimode');
sim.midx = 1:cfg.mode_count;
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

wait(device);
setupSeconds = toc(wallTimer);
propagateTimer = tic;
propOutput = GMMNLSE_propagate_with_adaptive(fiber, initialCondition, sim);
wait(device);
runtimeSeconds = toc(propagateTimer);
wallSeconds = toc(wallTimer);
if isempty(propOutput.fields)
    error("MMTools propagation returned no saved fields.");
end
end

function betas = synthetic_betas(cfg)
modeAxis = double(0:(cfg.mode_count - 1));
betas = zeros(3, cfg.mode_count);
betas(1, :) = cfg.beta0_base + cfg.beta0_spacing * modeAxis;
betas(2, :) = cfg.beta1;
betas(3, :) = cfg.beta2;
end

function SR = synthetic_sr_tensor(numModes)
SR = zeros(numModes, numModes, numModes, numModes);
diagIdx = 1:numModes;
linearIdx = sub2ind(repmat(numModes, 1, 4), diagIdx, diagIdx, diagIdx, diagIdx);
SR(linearIdx) = 1.0e10;
end

function fieldsOut = apply_temporal_chirp(fieldsIn, timeWindowPs, chirp)
fieldsOut = fieldsIn;
nt = size(fieldsIn, 1);
t = (-nt / 2 : nt / 2 - 1)' * (timeWindowPs / nt);
phase = exp((-1.0i) * chirp * (t / timeWindowPs) .^ 2);
fieldsOut = fieldsOut .* phase;
end
