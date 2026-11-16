function tests = test_matlab_package_surface()
%TEST_MATLAB_PACKAGE_SURFACE Checks the public MATLAB wrapper surface.
%
%   The counterpart of tests/test_python_package_surface.py in nlolib-python. The MATLAB
%   wrapper drifted behind the Python and Julia packages for a long time --
%   most visibly, it exposed no named constants at all and callers passed
%   magic integers -- with nothing to catch it. These tests pin the surface
%   that parity work added.
%
%   tools/check_api_parity.py already compares the constant VALUES across all
%   three bindings by scanning their sources. What a static scan cannot check,
%   and what this covers, is that each name actually resolves as a property at
%   run time and carries the integer class the C fields expect.
%
%   Most of these need no shared library. The one that propagates is skipped
%   when no library is available; set NLOLIB_LIBRARY to select one.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
scriptDir = fileparts(mfilename('fullpath'));
bindingRoot = fileparts(scriptDir);
% tests/api_parity.json belongs to the core repository. CTest names it; failing
% that it is in ./core (tools/fetch_core.sh), or in the parent directory when
% this repository is the matlab/ submodule of the core repository.
candidates = {getenv('NLOLIB_API_PARITY_MANIFEST'), ...
              fullfile(bindingRoot, 'core', 'tests', 'api_parity.json'), ...
              fullfile(fileparts(bindingRoot), 'tests', 'api_parity.json')};
found = candidates(cellfun(@(p) ~isempty(p) && isfile(p), candidates));
if isempty(found)
    testCase.TestData.manifestPath = candidates{end};
else
    testCase.TestData.manifestPath = found{1};
end
end

function test_constants_match_the_parity_manifest(testCase)
manifestPath = testCase.TestData.manifestPath;
verifyTrue(testCase, isfile(manifestPath), ...
           sprintf('parity manifest not found: %s', manifestPath));

manifest = jsondecode(fileread(manifestPath));
names = fieldnames(manifest.constants);
verifyTrue(testCase, numel(names) > 20, ...
           'parity manifest looks empty; expected the full constant set');

% properties() lists a classdef's public properties without constructing it.
declared = properties('nlolib.constants');

for idx = 1:numel(names)
    name = names{idx};
    verifyTrue(testCase, ismember(name, declared), ...
               sprintf('nlolib.constants.%s is not defined', name));
    actual = nlolib.constants.(name);
    verifyTrue(testCase, isscalar(actual), ...
               sprintf('nlolib.constants.%s must be scalar', name));

    % NT_MAX is the uint64 maximum, which does not survive a round trip
    % through the JSON manifest's double, so compare it against the exact
    % value instead.
    if strcmp(name, 'NT_MAX')
        verifyEqual(testCase, actual, intmax('uint64'), ...
                    'NT_MAX must be the uint64 maximum');
        continue;
    end

    expected = manifest.constants.(name);
    verifyEqual(testCase, double(actual), double(expected), ...
                sprintf('nlolib.constants.%s disagrees with the manifest', name));
end
end

function test_constants_carry_the_c_field_classes(testCase)
% Enum-like values go into int32 fields and the limits into uint64 ones, so
% they must already be that class to drop into a config struct without a cast.
verifyClass(testCase, nlolib.constants.VECTOR_BACKEND_VULKAN, 'int32');
verifyClass(testCase, nlolib.constants.FFT_BACKEND_VKFFT, 'int32');
verifyClass(testCase, nlolib.constants.NONLINEAR_MODEL_KERR_RAMAN, 'int32');
verifyClass(testCase, nlolib.constants.PROPAGATE_OUTPUT_FINAL_ONLY, 'int32');
verifyClass(testCase, nlolib.constants.NLOLIB_STATUS_ABORTED, 'int32');
verifyClass(testCase, nlolib.constants.NLOLIB_LOG_LEVEL_DEBUG, 'int32');
verifyClass(testCase, nlolib.constants.PROGRESS_EVENT_FINISH, 'int32');
verifyClass(testCase, nlolib.constants.NT_MAX, 'uint64');
verifyClass(testCase, nlolib.constants.STORAGE_RUN_ID_MAX, 'uint64');
end

function test_public_helpers_exist(testCase)
expected = { ...
    'nlolib.NLolib', 'nlolib.constants', 'nlolib.tensor_record_view', ...
    'nlolib.translate_runtime_handle', 'nlolib.prepare_sim_config', ...
    'nlolib.pack_complex_array', 'nlolib.unpack_records'};
for idx = 1:numel(expected)
    name = expected{idx};
    verifyNotEmpty(testCase, which(name), ...
                   sprintf('%s is not on the path', name));
end

methodNames = methods('nlolib.NLolib');
expectedMethods = { ...
    'propagate', 'query_runtime_limits', 'storage_is_available', ...
    'set_log_level', 'set_log_file', 'set_log_buffer', 'clear_log_buffer', ...
    'read_log_buffer', 'tail_logs', 'set_progress_options', ...
    'set_progress_stream', 'perf_profile_set_enabled', ...
    'perf_profile_is_enabled', 'perf_profile_reset', 'perf_profile_read', ...
    'validate_record_z'};
for idx = 1:numel(expectedMethods)
    verifyTrue(testCase, ismember(expectedMethods{idx}, methodNames), ...
               sprintf('NLolib/%s is missing', expectedMethods{idx}));
end
end

function test_validate_record_z_accepts_valid_positions(testCase)
% Repeated values are allowed: the check is nondecreasing, not increasing.
nlolib.NLolib.validate_record_z([0.0, 0.25, 0.25, 1.0], 1.0);
nlolib.NLolib.validate_record_z(0.5, 1.0);
end

function test_validate_record_z_rejects_bad_positions(testCase)
verifyError(testCase, @() nlolib.NLolib.validate_record_z([], 1.0), ...
            'nlolib:invalidRecordZ');
verifyError(testCase, @() nlolib.NLolib.validate_record_z([0.0, 0.5, 0.25], 1.0), ...
            'nlolib:invalidRecordZ');
verifyError(testCase, @() nlolib.NLolib.validate_record_z([-0.1, 0.5], 1.0), ...
            'nlolib:invalidRecordZ');
verifyError(testCase, @() nlolib.NLolib.validate_record_z([0.0, 1.5], 1.0), ...
            'nlolib:invalidRecordZ');
verifyError(testCase, @() nlolib.NLolib.validate_record_z([0.0, NaN], 1.0), ...
            'nlolib:invalidRecordZ');
end

function test_tensor_record_view_reshapes_t_fastest(testCase)
nt = 4;
nx = 3;
ny = 2;
record = 1:(nt * nx * ny);

volume = nlolib.tensor_record_view(record, nt, nx, ny);
verifySize(testCase, volume, [nt, ny, nx]);
% t fastest, then y, then x: the first nt samples are the (iy=1, ix=1) column.
verifyEqual(testCase, squeeze(volume(:, 1, 1)).', double(1:nt));
verifyEqual(testCase, volume(1, 2, 1), double(nt + 1));
verifyEqual(testCase, volume(1, 1, 2), double(nt * ny + 1));
end

function test_tensor_record_view_indexes_the_records_matrix(testCase)
nt = 2;
nx = 2;
ny = 2;
records = [1:8; 11:18];

last = nlolib.tensor_record_view(records, 0, nt, nx, ny);
verifyEqual(testCase, last(1, 1, 1), 11.0, 'index 0 must select the last record');
verifyEqual(testCase, nlolib.tensor_record_view(records, 1, nt, nx, ny), ...
            nlolib.tensor_record_view(records(1, :), nt, nx, ny));

verifyError(testCase, @() nlolib.tensor_record_view(records, 3, nt, nx, ny), ...
            'nlolib:invalidTensorRecordView');
verifyError(testCase, @() nlolib.tensor_record_view(records(1, :), nt, nx, ny + 1), ...
            'nlolib:invalidTensorRecordView');
verifyError(testCase, @() nlolib.tensor_record_view(records(1, :), 1, nt, nx, ny), ...
            'nlolib:invalidTensorRecordView');
end

function test_t_eval_records_at_the_requested_positions(testCase)
try
    api = nlolib.NLolib();
catch ME
    assumeFail(testCase, sprintf('nlolib library unavailable: %s', ME.message));
    return;
end

n = 64;
pulse = struct('samples', sech(linspace(-10, 10, n)), 'delta_time', 0.02);
recordZ = [0.0, 0.01, 0.03, 0.05];
options = struct('propagation_distance', 0.05, ...
                 'records', 128, ...   % ignored: t_eval sets the count
                 't_eval', recordZ, ...
                 'exec_options', struct('backend_type', nlolib.constants.VECTOR_BACKEND_CPU));

result = api.propagate(pulse, 'gvd', 'kerr', options);
verifyEqual(testCase, size(result.records, 1), numel(recordZ), ...
            't_eval must set the record count');
verifyEqual(testCase, result.z_axis, recordZ, 'AbsTol', 1e-12, ...
            'z_axis must report the requested positions, not a linspace');
end
