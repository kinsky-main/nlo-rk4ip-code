function [expression, constants] = translate_runtime_handle(fn, context, runtime)
if ~isa(fn, "function_handle")
    error("runtime handle must be a function_handle");
end
context = string(context);
if context ~= "dispersion_factor" && context ~= "dispersion" && context ~= "nonlinear"
    error("context must be 'dispersion_factor', 'dispersion', or 'nonlinear'");
end
if nargin < 3
    runtime = struct();
end

handleText = char(func2str(fn));
[argNames, body] = parse_handle_text(handleText);

if context == "dispersion_factor"
    if numel(argNames) < 1 || numel(argNames) > 2
        error("dispersion_factor handle must take one or two arguments");
    end
    symbolMap = struct(argNames{1}, 'A');
    if numel(argNames) >= 2
        symbolMap.(argNames{2}) = 'w';
    end
elseif context == "dispersion"
    if numel(argNames) < 1 || numel(argNames) > 4
        error("dispersion handle must take one to four arguments");
    end
    symbolMap = struct(argNames{1}, 'A');
    if numel(argNames) >= 2
        symbolMap.(argNames{2}) = 'D';
    end
    if numel(argNames) >= 3
        symbolMap.(argNames{3}) = 'h';
    end
    if numel(argNames) >= 4
        symbolMap.(argNames{4}) = 'w';
    end
else
    if numel(argNames) < 1 || numel(argNames) > 3
        error("nonlinear handle must take one to three arguments");
    end
    symbolMap = struct(argNames{1}, 'A');
    if numel(argNames) >= 2
        symbolMap.(argNames{2}) = 'I';
    end
    if numel(argNames) >= 3
        symbolMap.(argNames{3}) = 'V';
    end
end

body = strrep(body, ".^", "^");
body = strrep(body, ".*", "*");
body = strrep(body, "./", "/");
body = regexprep(body, "\<1[iIjJ]\>", "i");
body = regexprep(body, "\<1[jJ]\>", "i");

mapFields = fieldnames(symbolMap);
for idx = 1:numel(mapFields)
    key = mapFields{idx};
    value = symbolMap.(key);
    body = regexprep(body, ['\<' key '\>'], value);
end

bindings = struct();
if isfield(runtime, "constant_bindings") && ~isempty(runtime.constant_bindings)
    bindings = runtime.constant_bindings;
end
autoCapture = true;
if isfield(runtime, "auto_capture_constants")
    autoCapture = logical(runtime.auto_capture_constants);
end
workspaceScope = resolve_workspace_scope(fn);

constants = [];
constantsByName = containers.Map("KeyType", "char", "ValueType", "double");
function token = addConstant(name, value)
    if ~isfinite(value)
        error("captured constant '%s' must be finite", name);
    end
    if isKey(constantsByName, name)
        idxLocal = constantsByName(name);
    else
        idxLocal = numel(constants);
        constantsByName(name) = idxLocal;
        constants(end + 1) = value; %#ok<AGROW>
    end
    token = sprintf('c%d', idxLocal);
end

identifiers = regexp(body, "[A-Za-z_]\w*", "match");
for idx = 1:numel(identifiers)
    name = identifiers{idx};
    if ismember(name, {'w', 'A', 'I', 'D', 'V', 'h', 'i', 'exp', 'log', 'sqrt', 'sin', 'cos'})
        continue;
    end
    if ~isempty(regexp(name, "^c\d+$", "once"))
        continue;
    end

    [hasBinding, boundValue] = lookup_binding(bindings, name);
    if hasBinding
        if ~(isnumeric(boundValue) && isscalar(boundValue) && isreal(boundValue))
            error("constant binding '%s' must be a real scalar", name);
        end
        replacement = addConstant(['binding:' name], double(boundValue));
        body = regexprep(body, ['\<' name '\>'], replacement);
        continue;
    end

    if autoCapture && isfield(workspaceScope, name)
        value = workspaceScope.(name);
        if ~(isnumeric(value) && isscalar(value) && isreal(value))
            error("captured variable '%s' must be a real scalar", name);
        end
        replacement = addConstant(name, double(value));
        body = regexprep(body, ['\<' name '\>'], replacement);
        continue;
    end

    error("unknown identifier '%s' in runtime handle", name);
end

expression = char(body);
end

function [argNames, body] = parse_handle_text(handleText)
tokenized = regexp(handleText, "^@\(([^)]*)\)\s*(.+)$", "tokens", "once");
if ~isempty(tokenized)
    argsRaw = strtrim(tokenized{1});
    body = strtrim(tokenized{2});
    if strlength(argsRaw) == 0
        argNames = {};
        return;
    end
    parts = split(string(argsRaw), ",");
    argNames = cellstr(strtrim(parts)).';
    return;
end

tokenized = regexp(handleText, "^@([A-Za-z_]\w*)\s*(.+)$", "tokens", "once");
if ~isempty(tokenized)
    argNames = {strtrim(tokenized{1})};
    body = strtrim(tokenized{2});
    return;
end

error("unable to parse runtime function handle text '%s'", handleText);
end

function scope = resolve_workspace_scope(fn)
scope = struct();
info = functions(fn);
if isfield(info, "workspace") && ~isempty(info.workspace)
    firstScope = info.workspace{1};
    if isstruct(firstScope)
        scope = firstScope;
    end
end
end

function [found, value] = lookup_binding(bindings, name)
found = false;
value = [];
if isempty(bindings)
    return;
end
if isstruct(bindings) && isfield(bindings, name)
    found = true;
    value = bindings.(name);
    return;
end
if isa(bindings, "containers.Map") && isKey(bindings, name)
    found = true;
    value = bindings(name);
end
end
