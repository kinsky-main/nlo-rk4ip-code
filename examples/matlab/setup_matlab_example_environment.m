function repoRoot = setup_matlab_example_environment()
%SETUP_MATLAB_EXAMPLE_ENVIRONMENT Add required repo paths for MATLAB examples.

scriptDir = fileparts(mfilename("fullpath"));
repoRoot = resolve_repo_root(scriptDir);
matlabCandidates = { ...
    fullfile(repoRoot, "matlab"), ...
    fullfile(repoRoot, "examples", "matlab"), ...
    repoRoot ...
};
for idx = 1:numel(matlabCandidates)
    if isfolder(matlabCandidates{idx})
        addpath(matlabCandidates{idx}, "-begin");
    end
end

if ispc
    dllDirs = { ...
        fullfile(repoRoot, "build", "matlab_toolbox", "lib"), ...
        fullfile(repoRoot, "build", "src", "Release"), ...
        fullfile(repoRoot, "build", "src", "Debug"), ...
        fullfile(repoRoot, "build", "src", "RelWithDebInfo"), ...
        fullfile(repoRoot, "python", "Release"), ...
        fullfile(repoRoot, "python", "Debug"), ...
        fullfile(repoRoot, "python") ...
    };
    prepend_dirs_to_path(dllDirs);
end

if exist("nlolib_setup", "file") == 2
    nlolib_setup();
else
    addpath(fullfile(repoRoot, "examples", "matlab"), "-begin");
end
end

function prepend_dirs_to_path(dirs)
currentPath = string(getenv("PATH"));
parts = split(currentPath, ";");
normParts = lower(strtrim(parts));

for idx = 1:numel(dirs)
    folder = dirs{idx};
    if ~isfolder(folder)
        continue;
    end
    token = lower(string(folder));
    if any(normParts == token)
        continue;
    end
    if strlength(currentPath) > 0
        currentPath = string(folder) + ";" + currentPath;
    else
        currentPath = string(folder);
    end
    parts = split(currentPath, ";");
    normParts = lower(strtrim(parts));
end

setenv("PATH", char(currentPath));
end

function repoRoot = resolve_repo_root(startDir)
repoRoot = char(startDir);
for idx = 1:10
    marker = fullfile(repoRoot, "CMakeLists.txt");
    matlabDir = fullfile(repoRoot, "matlab");
    examplesDir = fullfile(repoRoot, "examples", "matlab");
    if isfile(marker) && isfolder(matlabDir) && isfolder(examplesDir)
        return;
    end

    parentDir = fileparts(repoRoot);
    if strlength(string(parentDir)) == 0 || strcmp(parentDir, repoRoot)
        break;
    end
    repoRoot = parentDir;
end

repoRoot = fileparts(fileparts(fileparts(char(startDir))));
end
