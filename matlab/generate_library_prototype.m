function protoPath = generate_library_prototype(libraryPath, headerPath, outputDir)
%GENERATE_LIBRARY_PROTOTYPE Emit a loadlibrary prototype and thunk for nlolib.
%   protoPath = generate_library_prototype(libraryPath, headerPath, outputDir)
%
%   Runs on the *build* machine, where a C compiler is available.  It parses
%   nlolib_matlab.h once and writes two files into outputDir:
%
%     nlolib_proto.m           - prototype describing every exported function
%     nlolib_thunk_<arch>.<ext> - prebuilt call thunk
%
%   Shipping both lets nlolib.NLolib load the library with
%   loadlibrary(dll, @nlolib_proto), which does no header parsing and needs
%   no compiler on the client.  The prototype resolves the thunk relative to
%   its own location, so the two files must stay in the same folder.
%
%   See also NLOLIB.NLOLIB, PACKAGE_MLTBX.

if nargin < 3 || strlength(string(outputDir)) == 0
    error('nlolib:prototypeArgs', ...
          'libraryPath, headerPath, and outputDir are all required.');
end

libraryPath = char(string(libraryPath));
headerPath = char(string(headerPath));
outputDir = char(string(outputDir));

if ~isfile(libraryPath)
    error('nlolib:prototypeLibraryMissing', ...
          'Shared library not found: %s', libraryPath);
end
if ~isfile(headerPath)
    error('nlolib:prototypeHeaderMissing', ...
          'Header not found: %s', headerPath);
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end

protoName = 'nlolib_proto';
protoPath = fullfile(outputDir, [protoName '.m']);

if libisloaded('nlolib')
    unloadlibrary('nlolib');
end

% loadlibrary writes the prototype and thunk into the current folder, so run
% the generation from outputDir and restore the caller's folder afterwards.
originalDir = pwd;
restoreDir = onCleanup(@() cd(originalDir));
cd(outputDir);

if isfile(protoPath)
    delete(protoPath);
end

[notfound, warnings] = loadlibrary(libraryPath, headerPath, ...
                                   'alias', 'nlolib', ...
                                   'mfilename', protoName);

if ~isempty(notfound)
    if libisloaded('nlolib')
        unloadlibrary('nlolib');
    end
    error('nlolib:prototypeUnresolvedSymbols', ...
          ['Prototype generation left unresolved symbols.\n' ...
           'Library: %s\nHeader : %s\nMissing:\n  %s'], ...
          libraryPath, headerPath, strjoin(cellstr(string(notfound)), '\n  '));
end
if ~isempty(warnings)
    warning('nlolib:prototypeWarnings', ...
            ['loadlibrary produced parser warnings while generating the ' ...
             'prototype. Unresolved types fail fast at call time, so review ' ...
             'these before shipping.\n\n%s'], ...
            strtrim(char(strjoin(cellstr(string(warnings)), newline))));
end

if libisloaded('nlolib')
    unloadlibrary('nlolib');
end

cd(originalDir); % back to the caller's folder; restoreDir stays as a safety net

if ~isfile(protoPath)
    error('nlolib:prototypeNotGenerated', ...
          'loadlibrary did not produce %s', protoPath);
end

% Building the thunk leaves compiler intermediates next to it.  Drop them so
% only the loadable thunk is packaged.
intermediateExts = {'.c', '.obj', '.o', '.exp', '.lib'};
for idx = 1:numel(intermediateExts)
    leftovers = dir(fullfile(outputDir, ['nlolib_thunk_*' intermediateExts{idx}]));
    for jdx = 1:numel(leftovers)
        delete(fullfile(outputDir, leftovers(jdx).name));
    end
end

thunks = dir(fullfile(outputDir, 'nlolib_thunk_*'));
thunks = thunks(~[thunks.isdir]);
if isempty(thunks)
    error('nlolib:thunkNotGenerated', ...
          ['loadlibrary produced %s but no nlolib_thunk_* library in %s.\n' ...
           'Without the thunk the prototype cannot load on a machine with ' ...
           'no compiler.'], protoPath, outputDir);
end

fprintf('Generated loadlibrary prototype: %s\n', protoPath);
for idx = 1:numel(thunks)
    fprintf('Generated loadlibrary thunk    : %s\n', ...
            fullfile(outputDir, thunks(idx).name));
end
end
