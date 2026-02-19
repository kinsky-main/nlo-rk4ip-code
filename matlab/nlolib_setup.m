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

if saveToPath
    savepath();
end

if nargout > 0
    addedPaths = cellstr(added);
end
end
