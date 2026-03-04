function [simCfgPtr, physicsCfgPtr, keepalive] = prepare_sim_config(cfg)
%PREPARE_SIM_CONFIG Build a sim_config libstruct from a MATLAB struct.
%   [simCfgPtr, physicsCfgPtr, keepalive] = nlolib.prepare_sim_config(cfg)
%
%   cfg is a flat MATLAB struct with fields matching the nlolib C API.
%   Returns a libstruct('sim_config', ...) and a keepalive cell array for
%   pointer-backed fields (complex arrays and c-strings).
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
keepalive = {};
maxConstants = 16;            % NLO_RUNTIME_OPERATOR_CONSTANTS_MAX from nlolib_matlab.h

numTs = double(cfg.num_time_samples);

legacyShapeFields = {"time_nt", "spatial_nx", "spatial_ny"};
for idx = 1:numel(legacyShapeFields)
    key = legacyShapeFields{idx};
    if isfield(cfg, key) && ~isempty(cfg.(key))
        error('nlolib:legacyApiRemoved', ...
              'cfg.%s has been removed; use tensor_nt/tensor_nx/tensor_ny instead.', key);
    end
end

[linearFactorExpr, linearExpr, potentialExpr, dispersionFactorExpr, dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg);
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

freqPtr = nlolib.pack_complex_array(cfg.frequency_grid);
keepalive{end + 1} = freqPtr; %#ok<AGROW>

spatialMl = struct();
tensorNt = double(get_optional(cfg, "tensor_nt", 0));
tensorNx = double(get_optional(cfg, "tensor_nx", 0));
tensorNy = double(get_optional(cfg, "tensor_ny", 0));
if tensorNt > 0
    if tensorNx <= 0 || tensorNy <= 0
        error('nlolib:invalidTensorShape', ...
              'tensor_nt requires tensor_nx and tensor_ny to be positive.');
    end
    defaultNx = tensorNx;
    defaultNy = tensorNy;
else
    defaultNx = numTs;
    defaultNy = 1;
end
spatialMl.nx = uint64(defaultNx);
spatialMl.ny = uint64(defaultNy);
spatialMl.delta_x = double(get_optional(cfg, "delta_x", 1.0));
spatialMl.delta_y = double(get_optional(cfg, "delta_y", 1.0));

if isfield(cfg, "spatial_frequency_grid") && ~isempty(cfg.spatial_frequency_grid)
    sfLen = numel(cfg.spatial_frequency_grid);
    if tensorNt > 0
        xyCount = tensorNx * tensorNy;
        if sfLen ~= xyCount && sfLen ~= numTs
            error('nlolib:invalidSpatialFrequencyGridShape', ...
                  'cfg.spatial_frequency_grid length must equal num_time_samples or tensor_nx*tensor_ny for tensor runs.');
        end
    elseif sfLen ~= numTs
        error('nlolib:invalidSpatialFrequencyGridShape', ...
              'cfg.spatial_frequency_grid length must equal num_time_samples for temporal runs.');
    end
    spatialFreqPtr = nlolib.pack_complex_array(cfg.spatial_frequency_grid);
    spatialMl.spatial_frequency_grid = spatialFreqPtr;
    keepalive{end + 1} = spatialFreqPtr; %#ok<AGROW>
end
if isfield(cfg, "kx_axis") && ~isempty(cfg.kx_axis)
    kxAxisPtr = nlolib.pack_complex_array(cfg.kx_axis);
    spatialMl.kx_axis = kxAxisPtr;
    keepalive{end + 1} = kxAxisPtr; %#ok<AGROW>
end
if isfield(cfg, "ky_axis") && ~isempty(cfg.ky_axis)
    kyAxisPtr = nlolib.pack_complex_array(cfg.ky_axis);
    spatialMl.ky_axis = kyAxisPtr;
    keepalive{end + 1} = kyAxisPtr; %#ok<AGROW>
end

if isfield(cfg, "potential_grid") && ~isempty(cfg.potential_grid)
    potentialValues = cfg.potential_grid;
    if tensorNt > 0
        xyCount = tensorNx * tensorNy;
        potentialLen = numel(potentialValues);
        if potentialLen == xyCount
            potentialValues = kron(reshape(potentialValues, 1, []), ones(1, tensorNt));
        elseif potentialLen ~= numTs
            error('nlolib:invalidPotentialGridShape', ...
                  'cfg.potential_grid length must equal num_time_samples or tensor_nx*tensor_ny for tensor runs.');
        end
    elseif numel(potentialValues) ~= numTs
        error('nlolib:invalidPotentialGridShape', ...
              'cfg.potential_grid length must equal num_time_samples for temporal runs.');
    end
    potentialPtr = nlolib.pack_complex_array(potentialValues);
    spatialMl.potential_grid = potentialPtr;
    keepalive{end + 1} = potentialPtr; %#ok<AGROW>
end

runtimeMl = struct();
if strlength(linearFactorExpr) > 0
    linearFactorPtr = cstring_ptr(linearFactorExpr);
    runtimeMl.linear_factor_expr = linearFactorPtr;
    keepalive{end + 1} = linearFactorPtr; %#ok<AGROW>
end
if strlength(linearExpr) > 0
    linearPtr = cstring_ptr(linearExpr);
    runtimeMl.linear_expr = linearPtr;
    keepalive{end + 1} = linearPtr; %#ok<AGROW>
end
if strlength(potentialExpr) > 0
    potentialExprPtr = cstring_ptr(potentialExpr);
    runtimeMl.potential_expr = potentialExprPtr;
    keepalive{end + 1} = potentialExprPtr; %#ok<AGROW>
end
if strlength(dispersionFactorExpr) > 0
    dispersionFactorPtr = cstring_ptr(dispersionFactorExpr);
    runtimeMl.dispersion_factor_expr = dispersionFactorPtr;
    keepalive{end + 1} = dispersionFactorPtr; %#ok<AGROW>
end
if strlength(dispersionExpr) > 0
    dispersionPtr = cstring_ptr(dispersionExpr);
    runtimeMl.dispersion_expr = dispersionPtr;
    keepalive{end + 1} = dispersionPtr; %#ok<AGROW>
end
if strlength(nonlinearExpr) > 0
    nonlinearPtr = cstring_ptr(nonlinearExpr);
    runtimeMl.nonlinear_expr = nonlinearPtr;
    keepalive{end + 1} = nonlinearPtr; %#ok<AGROW>
end
runtimeCfg = struct();
if isfield(cfg, "runtime") && ~isempty(cfg.runtime)
    runtimeCfg = cfg.runtime;
end
runtimeMl.nonlinear_model = int32(get_optional(runtimeCfg, "nonlinear_model", 0));
runtimeMl.nonlinear_gamma = double(get_optional(runtimeCfg, "nonlinear_gamma", 0.0));
runtimeMl.raman_fraction = double(get_optional(runtimeCfg, "raman_fraction", 0.0));
runtimeMl.raman_tau1 = double(get_optional(runtimeCfg, "raman_tau1", 0.0122));
runtimeMl.raman_tau2 = double(get_optional(runtimeCfg, "raman_tau2", 0.0320));
runtimeMl.shock_omega0 = double(get_optional(runtimeCfg, "shock_omega0", 0.0));
if isfield(runtimeCfg, "raman_response_time") && ~isempty(runtimeCfg.raman_response_time)
    ramanPtr = nlolib.pack_complex_array(runtimeCfg.raman_response_time);
    runtimeMl.raman_response_time = ramanPtr;
    runtimeMl.raman_response_len = uint64(numel(runtimeCfg.raman_response_time));
    keepalive{end + 1} = ramanPtr; %#ok<AGROW>
else
    runtimeMl.raman_response_len = uint64(0);
end
runtimeMl.num_constants = uint64(numConstants);
runtimeMl.constants = constantsFixed;

simMl = struct();
simMl.propagation = struct( ...
    'starting_step_size', double(cfg.starting_step_size), ...
    'max_step_size', double(cfg.max_step_size), ...
    'min_step_size', double(cfg.min_step_size), ...
    'error_tolerance', double(cfg.error_tolerance), ...
    'propagation_distance', double(cfg.propagation_distance));
simMl.tensor = struct( ...
    'nt', uint64(get_optional(cfg, "tensor_nt", 0)), ...
    'nx', uint64(get_optional(cfg, "tensor_nx", 0)), ...
    'ny', uint64(get_optional(cfg, "tensor_ny", 0)), ...
    'layout', int32(get_optional(cfg, "tensor_layout", 0)));
simMl.time = struct( ...
    'nt', uint64(0), ...
    'pulse_period', double(cfg.pulse_period), ...
    'delta_time', double(cfg.delta_time), ...
    'wt_axis', []);
if isfield(cfg, "wt_axis") && ~isempty(cfg.wt_axis)
    wtAxisPtr = nlolib.pack_complex_array(cfg.wt_axis);
    simMl.time.wt_axis = wtAxisPtr;
    keepalive{end + 1} = wtAxisPtr; %#ok<AGROW>
end
simMl.frequency = struct('frequency_grid', freqPtr);
simMl.spatial = spatialMl;

simCfgPtr = libstruct('nlo_simulation_config', simMl);
physicsCfgPtr = libstruct('runtime_operator_params', runtimeMl);
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

function ptr = cstring_ptr(text)
ptr = libpointer('cstring', char(string(text)));
end

function [linearFactorExpr, linearExpr, potentialExpr, dispersionFactorExpr, dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg)
linearFactorExpr = "";
linearExpr = "";
potentialExpr = "";
dispersionFactorExpr = "";
dispersionExpr = "";
nonlinearExpr = "";
constants = [];

if ~isfield(cfg, "runtime") || isempty(cfg.runtime)
    return;
end

runtime = cfg.runtime;
legacyRuntimeFields = {"transverse_factor_expr", "transverse_expr", "transverse_factor_fn", "transverse_fn"};
for idx = 1:numel(legacyRuntimeFields)
    key = legacyRuntimeFields{idx};
    if isfield(runtime, key) && ~isempty(runtime.(key))
        error('nlolib:legacyApiRemoved', ...
              'cfg.runtime.%s has been removed; encode diffraction in linear_factor_expr/linear_expr.', key);
    end
end
if isfield(runtime, "constants") && ~isempty(runtime.constants)
    constants = double(runtime.constants(:).');
end

if isfield(runtime, "linear_factor_expr") && ~isempty(runtime.linear_factor_expr)
    linearFactorExpr = string(runtime.linear_factor_expr);
end
if isfield(runtime, "linear_expr") && ~isempty(runtime.linear_expr)
    linearExpr = string(runtime.linear_expr);
end
if isfield(runtime, "potential_expr") && ~isempty(runtime.potential_expr)
    potentialExpr = string(runtime.potential_expr);
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
