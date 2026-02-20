function repoRoot = setup_matlab_example_environment()
%SETUP_MATLAB_EXAMPLE_ENVIRONMENT Add required repo paths for MATLAB examples.

repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
matlabCandidates = { ...
    repoRoot, ...
    fullfile(repoRoot, "matlab"), ...
    fullfile(repoRoot, "examples", "matlab"), ...
    fullfile(repoRoot, "build", "matlab_toolbox") ...
};
for idx = 1:numel(matlabCandidates)
    if isfolder(matlabCandidates{idx})
        addpath(matlabCandidates{idx});
    end
end

if exist("nlolib_setup", "file") == 2
    nlolib_setup();
else
    addpath(fullfile(repoRoot, "examples", "matlab"));
end
end
