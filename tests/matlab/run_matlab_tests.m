function run_matlab_tests()
scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));

addpath(fullfile(repoRoot, "matlab"), "-begin");
addpath(scriptDir, "-begin");

results = [ ...
    runtests(fullfile(scriptDir, "test_runtime_handle_parser.m")), ...
    runtests(fullfile(scriptDir, "test_ffi_smoke.m"))];
failed = [results.Failed];
incomplete = [results.Incomplete];
if any(failed)
    error("MATLAB tests failed.");
end

% Incomplete results are assumption failures (e.g. no built shared library),
% which are reported but not treated as failures.
skipped = nnz(incomplete);
fprintf("MATLAB tests: passed (%d of %d, %d skipped).\n", ...
        nnz([results.Passed]), numel(results), skipped);
end
