function [cfgPtr, keepalive] = prepare_sim_config(cfg)
%PREPARE_SIM_CONFIG Build a sim_config libstruct from a MATLAB struct.
%   [cfgPtr, keepalive] = nlolib.prepare_sim_config(cfg)
%
%   cfg is a flat MATLAB struct with fields matching the nlolib C API.
%   Returns a libpointer to the populated sim_config and a cell array
%   of handles that must remain alive for the duration of the calllib
%   call (they own heap memory referenced by pointer fields).
if ~isstruct(cfg)
    error("cfg must be a struct");
end

% --- validate required fields -------------------------------------------
required = { ...
    "num_time_samples", "gamma", "betas", "alpha", ...
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

s = libstruct('sim_config');
keepalive = {};

% --- nonlinear ----------------------------------------------------------
s.nonlinear.gamma = double(cfg.gamma);

% --- dispersion ---------------------------------------------------------
betas = double(cfg.betas(:).');
numBetas = numel(betas);
s.dispersion.num_dispersion_terms = uint64(numBetas);
% betas is a fixed-size C array — write into its first N elements.
for idx = 1:numBetas
    s.dispersion.betas(idx) = betas(idx);
end
s.dispersion.alpha = double(cfg.alpha);

% --- propagation --------------------------------------------------------
s.propagation.starting_step_size   = double(cfg.starting_step_size);
s.propagation.max_step_size        = double(cfg.max_step_size);
s.propagation.min_step_size        = double(cfg.min_step_size);
s.propagation.error_tolerance      = double(cfg.error_tolerance);
s.propagation.propagation_distance = double(cfg.propagation_distance);

% --- time ---------------------------------------------------------------
s.time.pulse_period = double(cfg.pulse_period);
s.time.delta_time   = double(cfg.delta_time);

% --- frequency grid (pointer field — needs keepalive) -------------------
freqPtr = nlolib.pack_complex_array(cfg.frequency_grid);
keepalive{end + 1} = freqPtr;
s.frequency.frequency_grid = freqPtr;

% --- spatial grid -------------------------------------------------------
numTs = double(cfg.num_time_samples);
s.spatial.nx = uint64(get_optional(cfg, "spatial_nx", numTs));
s.spatial.ny = uint64(get_optional(cfg, "spatial_ny", 1));
s.spatial.delta_x = double(get_optional(cfg, "delta_x", 1.0));
s.spatial.delta_y = double(get_optional(cfg, "delta_y", 1.0));
s.spatial.grin_gx = double(get_optional(cfg, "grin_gx", 0.0));
s.spatial.grin_gy = double(get_optional(cfg, "grin_gy", 0.0));

if isfield(cfg, "spatial_frequency_grid") && ~isempty(cfg.spatial_frequency_grid)
    spFreqPtr = nlolib.pack_complex_array(cfg.spatial_frequency_grid);
    keepalive{end + 1} = spFreqPtr;
    s.spatial.spatial_frequency_grid = spFreqPtr;
else
    s.spatial.spatial_frequency_grid = libpointer();
end

if isfield(cfg, "grin_potential_phase_grid") && ~isempty(cfg.grin_potential_phase_grid)
    grinPtr = nlolib.pack_complex_array(cfg.grin_potential_phase_grid);
    keepalive{end + 1} = grinPtr;
    s.spatial.grin_potential_phase_grid = grinPtr;
else
    s.spatial.grin_potential_phase_grid = libpointer();
end

% --- runtime operators --------------------------------------------------
[dispersionExpr, nonlinearExpr, constants] = resolve_runtime(cfg);

if strlength(dispersionExpr) > 0
    dispBytes = [uint8(char(dispersionExpr)), 0];   % null-terminated
    dispPtr   = libpointer('int8Ptr', int8(dispBytes));
    keepalive{end + 1} = dispPtr;
    s.runtime.dispersion_expr = dispPtr;
else
    s.runtime.dispersion_expr = libpointer();
end

if strlength(nonlinearExpr) > 0
    nlBytes = [uint8(char(nonlinearExpr)), 0];
    nlPtr   = libpointer('int8Ptr', int8(nlBytes));
    keepalive{end + 1} = nlPtr;
    s.runtime.nonlinear_expr = nlPtr;
else
    s.runtime.nonlinear_expr = libpointer();
end

s.runtime.num_constants = uint64(numel(constants));
for idx = 1:numel(constants)
    s.runtime.constants(idx) = double(constants(idx));
end

cfgPtr = s;
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
