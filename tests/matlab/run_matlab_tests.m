function run_matlab_tests()
scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(scriptDir));

addpath(fullfile(repoRoot, "matlab"), "-begin");
addpath(scriptDir, "-begin");

results = runtests(fullfile(scriptDir, "test_runtime_handle_parser.m"));
failed = [results.Failed];
incomplete = [results.Incomplete];
if any(failed) || any(incomplete)
    error("MATLAB runtime-handle tests failed.");
end

fprintf("test_matlab_runtime_handle_parser: passed (%d tests).\n", numel(results));
end
