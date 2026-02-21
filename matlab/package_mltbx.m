function artifactPath = package_mltbx(buildDir, config)
%PACKAGE_MLTBX Stage and package nlolib MATLAB toolbox artifact.
%   artifactPath = package_mltbx()
%   artifactPath = package_mltbx(buildDir, config)
%
% This helper stages files via CMake target `matlab_stage` and then emits
% a platform-tagged `.mltbx` bundle under `<repo>/dist`.

if nargin < 1 || strlength(string(buildDir)) == 0
    buildDir = "build";
end
if nargin < 2 || strlength(string(config)) == 0
    config = "Release";
end

repoRoot = fileparts(fileparts(mfilename("fullpath")));
buildDirAbs = fullfile(repoRoot, char(string(buildDir)));
stageDir = fullfile(buildDirAbs, "matlab_toolbox");
distDir = fullfile(repoRoot, "dist");
if ~isfolder(distDir)
    mkdir(distDir);
end

cmakeCommand = sprintf('cmake --build "%s" --config %s --target matlab_stage', ...
                       buildDirAbs, char(string(config)));
[status, output] = system(cmakeCommand);
if status ~= 0
    error('nlolib:matlabPackageBuildFailed', ...
          'Failed to build matlab_stage.\nCommand: %s\nOutput:\n%s', ...
          cmakeCommand, output);
end

if ~isfolder(stageDir)
    error('nlolib:matlabStageMissing', ...
          'Staging directory not found: %s', stageDir);
end

version = package_mltbx_read_version(fullfile(repoRoot, "CMakeLists.txt"));
platformTag = package_mltbx_platform_tag();
artifactName = sprintf('nlolib-%s-%s.mltbx', version, platformTag);
artifactPath = fullfile(distDir, artifactName);

zipStem = fullfile(tempdir, ['nlolib_mltbx_' char(java.util.UUID.randomUUID())]); %#ok<CHARTEN>
zip(zipStem, {'*'}, stageDir);
zipPath = [zipStem '.zip'];
if isfile(artifactPath)
    delete(artifactPath);
end
movefile(zipPath, artifactPath, 'f');

fprintf('Created toolbox bundle: %s\n', artifactPath);
end

function version = package_mltbx_read_version(cmakePath)
if ~isfile(cmakePath)
    version = "0.0.0";
    return;
end
contents = fileread(cmakePath);
tokens = regexp(contents, 'project\([^)]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)', ...
                'tokens', 'once');
if isempty(tokens)
    version = "0.0.0";
else
    version = string(tokens{1});
end
end

function tag = package_mltbx_platform_tag()
if ispc
    tag = "win64";
elseif ismac
    tag = "maci64";
else
    tag = "glnxa64";
end
end
