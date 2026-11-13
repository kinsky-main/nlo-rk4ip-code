function artifactPath = merge_mltbx(bundles, outputPath)
%MERGE_MLTBX Combine single-platform nlolib toolboxes into one .mltbx.
%   artifactPath = merge_mltbx(bundles)
%   artifactPath = merge_mltbx(bundles, outputPath)
%
%   bundles is a string array or cell array of .mltbx files made by
%   package_mltbx on different platforms, for example
%
%     merge_mltbx(["dist/nlolib-1.13.0-glnxa64.mltbx", ...
%                  "dist/nlolib-1.13.0-win64.mltbx"])
%
%   Each bundle keeps its binaries in lib/<arch> and its progress MEX under
%   +nlolib/private with a platform extension, so those files are simply
%   combined.  Every other file must be identical in all bundles (line
%   endings aside); a difference means the bundles were built from different
%   sources and the merge stops.  All bundles must have the same version.
%
%   outputPath defaults to dist/nlolib-<version>.mltbx.  The result supports
%   every platform found in the inputs, which makes it suitable for a single
%   File Exchange / GitHub release asset.
%
%   See also PACKAGE_MLTBX.

bundles = cellstr(string(bundles));
if isempty(bundles)
    error('nlolib:mergeMltbxArgs', 'At least one .mltbx file is required.');
end

version = '';
for idx = 1:numel(bundles)
    if ~isfile(bundles{idx})
        error('nlolib:mergeMltbxMissing', 'Toolbox file not found: %s', bundles{idx});
    end
    bundleVersion = matlab.addons.toolbox.toolboxVersion(bundles{idx});
    if isempty(version)
        version = bundleVersion;
    elseif ~strcmp(version, bundleVersion)
        error('nlolib:mergeMltbxVersion', ...
              'Version mismatch: %s is %s, expected %s.', ...
              bundles{idx}, bundleVersion, version);
    end
end

repoRoot = fileparts(fileparts(mfilename("fullpath")));
if nargin < 2 || strlength(string(outputPath)) == 0
    outputPath = fullfile(repoRoot, "dist", sprintf('nlolib-%s.mltbx', version));
end
outputPath = char(string(outputPath));

workDir = tempname;
mkdir(workDir);
cleanup = onCleanup(@() rmdir(workDir, 's'));
stageDir = fullfile(workDir, 'nlolib');
mkdir(stageDir);

platforms = {};
for idx = 1:numel(bundles)
    extractDir = fullfile(workDir, sprintf('bundle%d', idx));
    unzip(bundles{idx}, extractDir);
    fsroot = fullfile(extractDir, 'fsroot');
    if ~isfolder(fullfile(fsroot, 'lib'))
        error('nlolib:mergeMltbxLayout', ...
              '%s has no lib/ folder; repackage it with package_mltbx.', bundles{idx});
    end
    archDirs = dir(fullfile(fsroot, 'lib'));
    archDirs = archDirs([archDirs.isdir] & ~startsWith({archDirs.name}, '.'));
    archDirs = archDirs(arrayfun(@(d) ~isempty(dir(fullfile(d.folder, d.name, '*nlolib*'))), archDirs));
    if isempty(archDirs)
        error('nlolib:mergeMltbxLayout', ...
              '%s contains no lib/<arch> library folder.', bundles{idx});
    end
    platforms = [platforms, {archDirs.name}]; %#ok<AGROW>
    merge_mltbx_copy_tree(fsroot, stageDir, '', bundles{idx});
end
platforms = unique(platforms);

if isfile(outputPath)
    delete(outputPath);
end

% Same identifier as package_mltbx, so this bundle replaces an installed
% single-platform one.
opts = matlab.addons.toolbox.ToolboxOptions(stageDir, 'nlolib');
opts.ToolboxName = "nlolib";
opts.ToolboxVersion = char(version);
opts.Summary = "GPU-first nonlinear optics propagation library";
opts.Description = sprintf(['nlolib MATLAB bindings and runtime libraries for %s.\n' ...
                            'Includes +nlolib wrapper files, nlolib_setup.m, examples,\n' ...
                            'the user guide under docs/, and one lib/<arch> folder\n' ...
                            'per platform.'], strjoin(platforms, ', '));
opts.AuthorName = "nlolib";
opts.MinimumMatlabRelease = "R2019b";
opts.OutputFile = outputPath;
opts.ToolboxMatlabPath = {stageDir};
supported = opts.SupportedPlatforms;
supported.Win64 = any(strcmp(platforms, 'win64'));
supported.Glnxa64 = any(strcmp(platforms, 'glnxa64'));
supported.Maci64 = any(strcmp(platforms, 'maci64'));
supported.MatlabOnline = false;
opts.SupportedPlatforms = supported;

matlab.addons.toolbox.packageToolbox(opts);
artifactPath = outputPath;
fprintf('Created multi-platform toolbox (%s): %s\n', strjoin(platforms, ', '), artifactPath);
end

function merge_mltbx_copy_tree(srcRoot, dstRoot, relDir, bundle)
%MERGE_MLTBX_COPY_TREE Copy srcRoot/relDir into dstRoot, checking shared files.
entries = dir(fullfile(srcRoot, relDir));
entries = entries(~ismember({entries.name}, {'.', '..'}));
for idx = 1:numel(entries)
    rel = fullfile(relDir, entries(idx).name);
    src = fullfile(srcRoot, rel);
    dst = fullfile(dstRoot, rel);
    if entries(idx).isdir
        if ~isfolder(dst)
            mkdir(dst);
        end
        merge_mltbx_copy_tree(srcRoot, dstRoot, rel, bundle);
    elseif ~isfile(dst)
        copyfile(src, dst);
    elseif ~merge_mltbx_same_content(src, dst)
        error('nlolib:mergeMltbxConflict', ...
              ['%s in %s differs from the copy in an earlier bundle.\n' ...
               'Build all bundles from the same sources.'], rel, bundle);
    end
end
end

function same = merge_mltbx_same_content(a, b)
%MERGE_MLTBX_SAME_CONTENT Byte comparison that ignores CR/LF differences.
fa = fopen(a, 'r'); ca = fread(fa, Inf, '*uint8'); fclose(fa);
fb = fopen(b, 'r'); cb = fread(fb, Inf, '*uint8'); fclose(fb);
same = isequal(ca, cb) || isequal(ca(ca ~= 13), cb(cb ~= 13));
end
