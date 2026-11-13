function mexPath = build_progress_mex(outDir)
%BUILD_PROGRESS_MEX Compile the live progress MEX file used by NLolib.propagate.
%   mexPath = build_progress_mex()        builds into matlab/+nlolib/private
%   mexPath = build_progress_mex(outDir)  builds into outDir
%
%   Needs a C compiler configured for MEX (see `mex -setup C`). Only the
%   machine that packages the toolbox needs one; users get the compiled file.
%   Without it, propagate still works but prints its logs after the run.

matlabDir = fileparts(mfilename("fullpath"));
if nargin < 1 || strlength(string(outDir)) == 0
    outDir = fullfile(matlabDir, "+nlolib", "private");
end
outDir = char(outDir);
if ~isfolder(outDir)
    mkdir(outDir);
end

source = fullfile(matlabDir, "mex", "progress_mex.c");
% libut provides utIsInterruptPending (Ctrl+C detection).
if ispc
    libArgs = {['-L' fullfile(matlabroot, 'extern', 'lib', 'win64', 'microsoft')], '-llibut'};
else
    libArgs = {'-lut'};
end
if isunix && ~ismac
    libArgs{end + 1} = '-ldl';
end

mex('-silent', '-R2018a', '-outdir', outDir, source, libArgs{:});
mexPath = fullfile(outDir, ['progress_mex.' mexext]);
fprintf('Built %s\n', mexPath);
end
