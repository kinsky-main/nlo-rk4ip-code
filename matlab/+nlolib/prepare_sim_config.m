function prepared = prepare_sim_config(cfg)
if ~isstruct(cfg)
    error("cfg must be a struct");
end

pyBindings = py.importlib.import_module("nlolib_ctypes");

[dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg);

if ~isfield(cfg, "num_time_samples")
    error("cfg.num_time_samples is required");
end
if ~isfield(cfg, "gamma")
    error("cfg.gamma is required");
end
if ~isfield(cfg, "betas")
    error("cfg.betas is required");
end
if ~isfield(cfg, "alpha")
    error("cfg.alpha is required");
end
if ~isfield(cfg, "propagation_distance")
    error("cfg.propagation_distance is required");
end
if ~isfield(cfg, "starting_step_size")
    error("cfg.starting_step_size is required");
end
if ~isfield(cfg, "max_step_size")
    error("cfg.max_step_size is required");
end
if ~isfield(cfg, "min_step_size")
    error("cfg.min_step_size is required");
end
if ~isfield(cfg, "error_tolerance")
    error("cfg.error_tolerance is required");
end
if ~isfield(cfg, "pulse_period")
    error("cfg.pulse_period is required");
end
if ~isfield(cfg, "delta_time")
    error("cfg.delta_time is required");
end
if ~isfield(cfg, "frequency_grid")
    error("cfg.frequency_grid is required");
end

if strlength(dispersionExpr) == 0 && strlength(nonlinearExpr) == 0 && isempty(constants)
    runtimeObj = py.None;
else
    runtimeObj = pyBindings.RuntimeOperators(pyargs( ...
        "dispersion_expr", dispersionExpr, ...
        "nonlinear_expr", nonlinearExpr, ...
        "constants", numeric_vector_to_py_list(constants) ...
    ));
end

spatialNx = py.None;
if isfield(cfg, "spatial_nx")
    spatialNx = int64(cfg.spatial_nx);
end
spatialNy = py.None;
if isfield(cfg, "spatial_ny")
    spatialNy = int64(cfg.spatial_ny);
end
deltaX = 1.0;
if isfield(cfg, "delta_x")
    deltaX = double(cfg.delta_x);
end
deltaY = 1.0;
if isfield(cfg, "delta_y")
    deltaY = double(cfg.delta_y);
end
grinGx = 0.0;
if isfield(cfg, "grin_gx")
    grinGx = double(cfg.grin_gx);
end
grinGy = 0.0;
if isfield(cfg, "grin_gy")
    grinGy = double(cfg.grin_gy);
end

spatialFreq = py.None;
if isfield(cfg, "spatial_frequency_grid") && ~isempty(cfg.spatial_frequency_grid)
    spatialFreq = nlolib.matlab_complex_vector_to_py_list(cfg.spatial_frequency_grid);
end
grinPhase = py.None;
if isfield(cfg, "grin_potential_phase_grid") && ~isempty(cfg.grin_potential_phase_grid)
    grinPhase = nlolib.matlab_complex_vector_to_py_list(cfg.grin_potential_phase_grid);
end

prepared = pyBindings.prepare_sim_config(int64(cfg.num_time_samples), ...
    pyargs( ...
        "gamma", double(cfg.gamma), ...
        "betas", numeric_vector_to_py_list(cfg.betas), ...
        "alpha", double(cfg.alpha), ...
        "propagation_distance", double(cfg.propagation_distance), ...
        "starting_step_size", double(cfg.starting_step_size), ...
        "max_step_size", double(cfg.max_step_size), ...
        "min_step_size", double(cfg.min_step_size), ...
        "error_tolerance", double(cfg.error_tolerance), ...
        "pulse_period", double(cfg.pulse_period), ...
        "delta_time", double(cfg.delta_time), ...
        "frequency_grid", nlolib.matlab_complex_vector_to_py_list(cfg.frequency_grid), ...
        "spatial_nx", spatialNx, ...
        "spatial_ny", spatialNy, ...
        "delta_x", deltaX, ...
        "delta_y", deltaY, ...
        "grin_gx", grinGx, ...
        "grin_gy", grinGy, ...
        "spatial_frequency_grid", spatialFreq, ...
        "grin_potential_phase_grid", grinPhase, ...
        "runtime", runtimeObj ...
    ) ...
);
end

function [dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg)
dispersionExpr = "";
nonlinearExpr = "";
constants = [];

if ~isfield(cfg, "runtime") || isempty(cfg.runtime)
    return;
end

runtime = cfg.runtime;
if isfield(runtime, "constants") && ~isempty(runtime.constants)
    constants = double(runtime.constants(:).');
end

if isfield(runtime, "dispersion_expr") && ~isempty(runtime.dispersion_expr)
    dispersionExpr = string(runtime.dispersion_expr);
end
if isfield(runtime, "nonlinear_expr") && ~isempty(runtime.nonlinear_expr)
    nonlinearExpr = string(runtime.nonlinear_expr);
end

if isfield(runtime, "dispersion_fn") && ~isempty(runtime.dispersion_fn)
    [translated, captured] = nlolib.translate_runtime_handle(runtime.dispersion_fn, "dispersion", runtime);
    translated = shift_constant_indices(translated, numel(constants));
    constants = [constants, captured]; %#ok<AGROW>
    dispersionExpr = translated;
end

if isfield(runtime, "nonlinear_fn") && ~isempty(runtime.nonlinear_fn)
    [translated, captured] = nlolib.translate_runtime_handle(runtime.nonlinear_fn, "nonlinear", runtime);
    translated = shift_constant_indices(translated, numel(constants));
    constants = [constants, captured]; %#ok<AGROW>
    nonlinearExpr = translated;
end
end

function shifted = shift_constant_indices(expr, offset)
shifted = char(expr);
if offset == 0
    return;
end

[starts, ends, tokens] = regexp(shifted, "c(\d+)", "start", "end", "tokens");
if isempty(starts)
    return;
end

out = strings(1, numel(starts) * 2 + 1);
cursor = 1;
outIdx = 1;
for idx = 1:numel(starts)
    out(outIdx) = string(shifted(cursor:starts(idx) - 1));
    outIdx = outIdx + 1;
    value = str2double(tokens{idx}{1}) + offset;
    out(outIdx) = "c" + string(value);
    outIdx = outIdx + 1;
    cursor = ends(idx) + 1;
end
out(outIdx) = string(shifted(cursor:end));
shifted = join(out(1:outIdx), "");
shifted = char(shifted);
end

function pyList = numeric_vector_to_py_list(values)
vals = double(values(:).');
pyList = py.list();
for idx = 1:numel(vals)
    pyList.append(vals(idx));
end
end
