function [cfgPtr, keepalive] = prepare_sim_config(cfg)
%PREPARE_SIM_CONFIG Build a sim_config libstruct from a MATLAB struct.
%   [cfgPtr, keepalive] = nlolib.prepare_sim_config(cfg)
%
%   cfg is a flat MATLAB struct with fields matching the nlolib C API.
%   Returns a libstruct('sim_config', ...) and a compatibility keepalive
%   cell array (currently empty).
if ~isstruct(cfg)
    error("cfg must be a struct");
end

% --- validate required fields -------------------------------------------
required = { ...
    "num_time_samples", ...
    "propagation_distance", "starting_step_size", "max_step_size", ...
    "min_step_size", "error_tolerance", "pulse_period", "delta_time", ...
    "frequency_grid"};
for idx = 1:numel(required)
    if ~isfield(cfg, required{idx})
        error("cfg.%s is required", required{idx});
    end
end

% The library must be loaded before libstruct can resolve type names.
if ~libisloaded('nlolib')
    error('nlolib:notLoaded', ...
          'Library not loaded. Create an nlolib.NLolib instance first.');
end

nlolib.NLolib.ensure_types_loaded();
maxConstants = 16;            % NLO_RUNTIME_OPERATOR_CONSTANTS_MAX from nlolib_matlab.h

numTs = double(cfg.num_time_samples);

[dispersionFactorExpr, dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg);
constants = double(constants(:).');
numConstants = numel(constants);
if numConstants > maxConstants
    error('nlolib:tooManyRuntimeConstants', ...
          'cfg.runtime.constants has %d values but max supported is %d.', ...
          numConstants, maxConstants);
end
constantsFixed = zeros(1, maxConstants);
if numConstants > 0
    constantsFixed(1:numConstants) = constants;
end

freqStruct = complex_to_nlo_complex_struct(cfg.frequency_grid);

spatialMl = struct();
spatialMl.nx = uint64(get_optional(cfg, "spatial_nx", numTs));
spatialMl.ny = uint64(get_optional(cfg, "spatial_ny", 1));
spatialMl.delta_x = double(get_optional(cfg, "delta_x", 1.0));
spatialMl.delta_y = double(get_optional(cfg, "delta_y", 1.0));

if isfield(cfg, "spatial_frequency_grid") && ~isempty(cfg.spatial_frequency_grid)
    spatialMl.spatial_frequency_grid = complex_to_nlo_complex_struct(cfg.spatial_frequency_grid);
end

if isfield(cfg, "potential_grid") && ~isempty(cfg.potential_grid)
    spatialMl.potential_grid = complex_to_nlo_complex_struct(cfg.potential_grid);
end

runtimeMl = struct();
if strlength(dispersionFactorExpr) > 0
    runtimeMl.dispersion_factor_expr = char(dispersionFactorExpr);
end
if strlength(dispersionExpr) > 0
    runtimeMl.dispersion_expr = char(dispersionExpr);
end
if strlength(nonlinearExpr) > 0
    runtimeMl.nonlinear_expr = char(nonlinearExpr);
end
runtimeMl.num_constants = uint64(numConstants);
runtimeMl.constants = constantsFixed;

cfgMl = struct();
cfgMl.propagation = struct( ...
    'starting_step_size', double(cfg.starting_step_size), ...
    'max_step_size', double(cfg.max_step_size), ...
    'min_step_size', double(cfg.min_step_size), ...
    'error_tolerance', double(cfg.error_tolerance), ...
    'propagation_distance', double(cfg.propagation_distance));
cfgMl.time = struct( ...
    'pulse_period', double(cfg.pulse_period), ...
    'delta_time', double(cfg.delta_time));
cfgMl.frequency = struct('frequency_grid', freqStruct);
cfgMl.spatial = spatialMl;
cfgMl.runtime = runtimeMl;

cfgPtr = libstruct('sim_config', cfgMl);
keepalive = {};
end

% ========================================================================
% Local helper functions
% ========================================================================

function val = get_optional(cfg, name, default)
if isfield(cfg, name)
    val = cfg.(name);
else
    val = default;
end
end

function out = complex_to_nlo_complex_struct(values)
vals = values(:).';
re = num2cell(real(vals));
im = num2cell(imag(vals));
out = struct('re', re, 'im', im);
end

function [dispersionFactorExpr, dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg)
dispersionFactorExpr = "";
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

if isfield(runtime, "dispersion_factor_expr") && ~isempty(runtime.dispersion_factor_expr)
    dispersionFactorExpr = string(runtime.dispersion_factor_expr);
end
if isfield(runtime, "dispersion_expr") && ~isempty(runtime.dispersion_expr)
    dispersionExpr = string(runtime.dispersion_expr);
end
if isfield(runtime, "nonlinear_expr") && ~isempty(runtime.nonlinear_expr)
    nonlinearExpr = string(runtime.nonlinear_expr);
end

if isfield(runtime, "dispersion_factor_fn") && ~isempty(runtime.dispersion_factor_fn)
    [translated, captured] = nlolib.translate_runtime_handle(runtime.dispersion_factor_fn, "dispersion_factor", runtime);
    translated = shift_constant_indices(translated, numel(constants));
    constants = [constants, captured];
    dispersionFactorExpr = translated;
end

if isfield(runtime, "dispersion_fn") && ~isempty(runtime.dispersion_fn)
    [translated, captured] = nlolib.translate_runtime_handle(runtime.dispersion_fn, "dispersion", runtime);
    translated = shift_constant_indices(translated, numel(constants));
    constants = [constants, captured];
    dispersionExpr = translated;
end

if isfield(runtime, "nonlinear_fn") && ~isempty(runtime.nonlinear_fn)
    [translated, captured] = nlolib.translate_runtime_handle(runtime.nonlinear_fn, "nonlinear", runtime);
    translated = shift_constant_indices(translated, numel(constants));
    constants = [constants, captured];
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
