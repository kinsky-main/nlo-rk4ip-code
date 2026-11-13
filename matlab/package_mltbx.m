function artifactPath = package_mltbx(buildDir, config)
%PACKAGE_MLTBX Stage and package nlolib MATLAB toolbox artifact.
%   artifactPath = package_mltbx()
%   artifactPath = package_mltbx(buildDir, config)
%
% This helper stages files via CMake target `matlab_stage` and then emits
% a platform-tagged `.mltbx` bundle under `<repo>/dist`.  Platform binaries
% (shared library, loadlibrary prototype and thunk) live in lib/<arch>, so
% bundles from several platforms can be combined with merge_mltbx.
%
% See also MERGE_MLTBX.

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

% MATLAB's system() inherits its bundled runtime library path, which makes
% host tools such as cmake load MATLAB's older libstdc++. Clear it for cmake.
envPrefix = '';
if ismac
    envPrefix = 'env -u DYLD_LIBRARY_PATH ';
elseif isunix
    envPrefix = 'env -u LD_LIBRARY_PATH ';
end
cmakeCommand = sprintf('%scmake --build "%s" --config %s --target matlab_stage', ...
                       envPrefix, buildDirAbs, char(string(config)));
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

% Generate the loadlibrary prototype and thunk here, on the build machine,
% so the shipped toolbox never has to parse the header — and therefore never
% needs a C compiler on the client.
package_mltbx_generate_prototype(repoRoot, stageDir);

% Compile the live progress MEX here as well, so users get the Command
% Window progress bar without a compiler.
build_progress_mex(fullfile(stageDir, "+nlolib", "private"));

version = package_mltbx_read_version(fullfile(repoRoot, "CMakeLists.txt"));
platformTag = package_mltbx_platform_tag();
artifactName = sprintf('nlolib-%s-%s.mltbx', version, platformTag);
artifactPath = fullfile(distDir, artifactName);

if isfile(artifactPath)
    delete(artifactPath);
end

% One identifier for every platform, so a multi-platform bundle built with
% merge_mltbx updates an installed single-platform one in place.
opts = matlab.addons.toolbox.ToolboxOptions(stageDir, package_mltbx_identifier());
opts.ToolboxName = "nlolib";
opts.ToolboxVersion = char(version);
opts.Summary = "GPU-first nonlinear optics propagation library";
opts.Description = sprintf(['nlolib MATLAB bindings and staged runtime libraries.\n' ...
                            'Includes +nlolib wrapper files, nlolib_setup.m, examples,\n' ...
                            'the user guide under docs/, and platform-specific shared\n' ...
                            'libraries under lib/.']);
opts.AuthorName = "nlolib";
opts.MinimumMatlabRelease = "R2019b";
opts.OutputFile = artifactPath;
% Only the package root goes on the permanent path; examples (and their
% generic +backend helpers) are added per session by nlolib_setup().
opts.ToolboxMatlabPath = {char(stageDir)};
supportedPlatforms = opts.SupportedPlatforms;
supportedPlatforms.Win64 = false;
supportedPlatforms.Glnxa64 = false;
supportedPlatforms.Maci64 = false;
supportedPlatforms.MatlabOnline = false;
if ispc
    supportedPlatforms.Win64 = true;
elseif ismac
    supportedPlatforms.Maci64 = true;
else
    supportedPlatforms.Glnxa64 = true;
end
opts.SupportedPlatforms = supportedPlatforms;

matlab.addons.toolbox.packageToolbox(opts);

fprintf('Created toolbox bundle: %s\n', artifactPath);
end

function package_mltbx_generate_prototype(repoRoot, stageDir)
%PACKAGE_MLTBX_GENERATE_PROTOTYPE Emit nlolib_proto.m and its thunk into stageDir.
if ispc
    libFile = "nlolib.dll";
elseif ismac
    libFile = "libnlolib.dylib";
else
    libFile = "libnlolib.so";
end

% The prototype and thunk sit beside the library in lib/<arch>; the thunk
% name is platform-specific and the prototype refers to it.
archDir = fullfile(stageDir, "lib", package_mltbx_platform_tag());
libraryPath = fullfile(archDir, libFile);
headerPath = fullfile(stageDir, "lib", "nlolib_matlab.h");
if ~isfile(libraryPath) || ~isfile(headerPath)
    error('nlolib:matlabStageIncomplete', ...
          ['Staged library or header missing; cannot generate the ' ...
           'loadlibrary prototype.\n  %s\n  %s'], libraryPath, headerPath);
end

matlabDir = fullfile(repoRoot, "matlab");
addpath(matlabDir);
generate_library_prototype(libraryPath, headerPath, archDir);
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

function id = package_mltbx_identifier()
%PACKAGE_MLTBX_IDENTIFIER Toolbox GUID shared by all platform bundles.
id = 'nlolib';
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
