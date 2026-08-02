function tests = test_ffi_smoke()
%TEST_FFI_SMOKE End-to-end checks of the loadlibrary/calllib binding.
%
%   These tests exercise the paths that pure-MATLAB unit tests cannot reach:
%   library load, libstruct/libpointer type resolution, complex buffer
%   packing/unpacking, and the propagate ABI. A binding regression such as a
%   renamed C type only shows up here.
%
%   Requires a built nlolib shared library. Set NLOLIB_LIBRARY to select one.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
try
    api = nlolib.NLolib();
catch ME
    assumeFail(testCase, sprintf('nlolib library unavailable: %s', ME.message));
    return;
end
testCase.TestData.api = api;
testCase.TestData.n = 64;
testCase.TestData.dt = 0.02;
end

function cfg = base_config(testCase)
n  = testCase.TestData.n;
dt = testCase.TestData.dt;

omega = zeros(1, n);
step = 2.0 * pi / (n * dt);
half = floor((n - 1) / 2);
for idx = 1:n
    i = idx - 1;
    if i <= half
        omega(idx) = i * step;
    else
        omega(idx) = -(n - i) * step;
    end
end

cfg = struct();
cfg.num_time_samples = n;
cfg.propagation_distance = 0.05;
cfg.starting_step_size = 1e-3;
cfg.max_step_size = 5e-3;
cfg.min_step_size = 1e-6;
cfg.error_tolerance = 1e-6;
cfg.pulse_period = n * dt;
cfg.delta_time = dt;
cfg.frequency_grid = complex(omega, zeros(1, n));
cfg.runtime = struct('dispersion_factor_expr', 'i*c0*w*w', ...
                     'nonlinear_expr', 'i*c1*A*I', ...
                     'constants', [0.5, 0.0]);
end

function field = gaussian_field(testCase)
n  = testCase.TestData.n;
dt = testCase.TestData.dt;
t = ((0:(n - 1)) - floor(n / 2)) * dt;
field = complex(exp(-((t / 0.25) .^ 2)), zeros(1, n));
end

% --- library plumbing ----------------------------------------------------

function testLibraryTypesResolve(testCase)
% Every libstruct type the wrapper uses must resolve. Catches header/wrapper
% type-name drift at the point of use.
names = {'sim_config', 'simulation_config', 'runtime_operator_params', ...
         'propagation_params', 'frequency_grid', 'execution_options', ...
         'storage_options', 'storage_result', 'step_event', ...
         'runtime_limits', 'propagate_options', 'propagate_output', ...
         'nlo_perf_profile_snapshot'};
for idx = 1:numel(names)
    s = libstruct(names{idx});
    verifyNotEmpty(testCase, fieldnames(s), ...
                   sprintf('libstruct(''%s'') resolved to an empty type', names{idx}));
end
end

function testComplexPointerRoundTrip(testCase)
% pack_complex_array must produce a pointer whose payload reads back intact.
values = complex([1.0, -2.5, 3.25], [0.5, 0.0, -1.75]);
ptr = nlolib.pack_complex_array(values);
back = nlolib.unpack_records(ptr, 1, numel(values));
verifyEqual(testCase, size(back), [1, numel(values)]);
verifyEqual(testCase, back, values, 'AbsTol', 0.0);
end

% --- propagate -----------------------------------------------------------

function testDensePropagateShape(testCase)
api = testCase.TestData.api;
cfg = base_config(testCase);
field = gaussian_field(testCase);

result = api.propagate(cfg, field, 4);

verifyEqual(testCase, size(result.records, 2), testCase.TestData.n);
verifySize(testCase, result.records, [4, testCase.TestData.n]);
verifyTrue(testCase, all(isfinite(result.records(:))), ...
           'propagated records contain non-finite values');
verifyEqual(testCase, numel(result.z_axis), size(result.records, 1));
verifyEqual(testCase, result.meta.records_written, 4);
end

function testFinalOnlyPropagate(testCase)
api = testCase.TestData.api;
cfg = base_config(testCase);
field = gaussian_field(testCase);

result = api.propagate(cfg, field, 1);

verifySize(testCase, result.records, [1, testCase.TestData.n]);
verifyEqual(testCase, char(result.meta.output), 'final');
verifyTrue(testCase, all(isfinite(result.final)));
end

function testShortRecordCountIsTruncatedNotAnError(testCase)
% Fixed-step runs (start == min == max) let the library reduce the record
% count below the request. unpack_records must ignore the unwritten tail
% rather than reporting a buffer-length mismatch.
api = testCase.TestData.api;
cfg = base_config(testCase);
cfg.starting_step_size = 5e-3;
cfg.min_step_size = 5e-3;
cfg.max_step_size = 5e-3;
field = gaussian_field(testCase);

result = api.propagate(cfg, field, 512);

verifyLessThanOrEqual(testCase, size(result.records, 1), 512);
verifyGreaterThan(testCase, size(result.records, 1), 0);
verifyEqual(testCase, size(result.records, 1), result.meta.records_written);
verifyTrue(testCase, all(isfinite(result.records(:))));
end

function testInputFieldLengthMismatchIsReported(testCase)
api = testCase.TestData.api;
cfg = base_config(testCase);
field = gaussian_field(testCase);

verifyError(testCase, @() api.propagate(cfg, field(1:(end - 1)), 2), ...
            'nlolib:inputFieldLengthMismatch');
end

function testStepHistoryCapture(testCase)
% Exercises the step_event[] output buffer: distinct slots, not N aliases of
% one struct.
api = testCase.TestData.api;
cfg = base_config(testCase);
field = gaussian_field(testCase);

execOptions = struct('capture_step_history', true, ...
                     'step_history_capacity', uint64(256));
result = api.propagate(cfg, field, 2, execOptions);

history = result.step_history;
verifyEqual(testCase, history.capacity, 256);
if numel(history.step_index) > 1
    verifyGreaterThan(testCase, history.step_index(end), history.step_index(1), ...
                      'step indices are identical -- step_event slots are aliased');
    verifyTrue(testCase, all(diff(history.step_index) >= 0));
    verifyTrue(testCase, all(isfinite(history.step_size)));
end
end

% --- ancillary entry points ----------------------------------------------

function testRuntimeLimitsQuery(testCase)
api = testCase.TestData.api;
limits = api.query_runtime_limits(base_config(testCase));
verifyTrue(testCase, isfield(limits, 'max_num_recorded_samples_in_memory'));
verifyGreaterThan(testCase, double(limits.max_num_recorded_samples_in_memory), 0);
end

function testLogBufferRoundTrip(testCase)
% Covers the char* / cstring marshalling of nlolib_read_log_buffer.
api = testCase.TestData.api;
api.set_log_buffer(uint64(65536));
api.clear_log_buffer();
api.set_log_level(int32(3));

cfg = base_config(testCase);
api.propagate(cfg, gaussian_field(testCase), 2);

text = api.read_log_buffer(true, uint64(65536));
verifyClass(testCase, text, 'string');
verifyEqual(testCase, api.read_log_buffer(true, uint64(65536)), "", ...
            'consuming read did not drain the ring buffer');
end

function testPerfProfileCounters(testCase)
api = testCase.TestData.api;
api.perf_profile_set_enabled(true);
api.perf_profile_reset();
api.propagate(base_config(testCase), gaussian_field(testCase), 2);
snapshot = api.perf_profile_read();
api.perf_profile_set_enabled(false);

verifyTrue(testCase, isfield(snapshot, 'dispersion_calls'));
verifyGreaterThan(testCase, double(snapshot.dispersion_calls), 0);
end
