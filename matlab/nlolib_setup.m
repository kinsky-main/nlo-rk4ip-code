function addedPaths = nlolib_setup(saveToPath)
%NLOLIB_SETUP Add nlolib MATLAB package and examples to MATLAB path.
%   nlolib_setup() adds the nlolib MATLAB package and example folder.
%   nlolib_setup(true) additionally saves the updated path.

if nargin < 1
    saveToPath = false;
end
saveToPath = logical(saveToPath);

matlabRoot = fileparts(mfilename("fullpath"));
repoRoot = fileparts(matlabRoot);

candidates = {
    matlabRoot
    fullfile(repoRoot, "examples", "matlab")
    fullfile(matlabRoot, "examples", "matlab")
};

added = strings(0, 1);
for idx = 1:numel(candidates)
    folder = candidates{idx};
    if isfolder(folder)
        addpath(folder);
        added(end + 1, 1) = string(folder); %#ok<AGROW>
    end
end

if ispc
    dllDirs = {
        fullfile(matlabRoot, "lib")
        fullfile(matlabRoot, "lib", "win64")
        fullfile(repoRoot, "build", "matlab_toolbox", "lib")
        fullfile(repoRoot, "build", "matlab_toolbox", "lib", "win64")
        fullfile(repoRoot, "build", "src", "Release")
        fullfile(repoRoot, "build", "src", "Debug")
        fullfile(repoRoot, "build", "src", "RelWithDebInfo")
        fullfile(repoRoot, "build", "src")
        fullfile(repoRoot, "python", "Release")
        fullfile(repoRoot, "python", "Debug")
        fullfile(repoRoot, "python")
    };
    prepend_dirs_to_path(dllDirs);
elseif ismac
    dyldDirs = {
        fullfile(matlabRoot, "lib")
        fullfile(matlabRoot, "lib", "maci64")
        fullfile(repoRoot, "build", "matlab_toolbox", "lib")
        fullfile(repoRoot, "build", "matlab_toolbox", "lib", "maci64")
        fullfile(repoRoot, "build", "src")
        fullfile(repoRoot, "python")
    };
    prepend_dirs_to_path(dyldDirs);
else
    soDirs = {
        fullfile(matlabRoot, "lib")
        fullfile(matlabRoot, "lib", "glnxa64")
        fullfile(repoRoot, "build", "matlab_toolbox", "lib")
        fullfile(repoRoot, "build", "matlab_toolbox", "lib", "glnxa64")
        fullfile(repoRoot, "build", "src")
        fullfile(repoRoot, "python")
    };
    prepend_dirs_to_path(soDirs);
end

if saveToPath
    savepath();
end

if nargout > 0
    addedPaths = cellstr(added);
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
