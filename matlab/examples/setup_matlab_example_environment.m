function bindingRoot = setup_matlab_example_environment()
%SETUP_MATLAB_EXAMPLE_ENVIRONMENT Add the nlolib package and examples to the path.
%   Works from a source checkout, where the examples are in <root>/examples,
%   and from a staged or installed toolbox, where they are in
%   <root>/examples/matlab. Returns <root>, the folder that holds +nlolib.

scriptDir = fileparts(mfilename("fullpath"));
bindingRoot = fileparts(scriptDir);
if ~isfolder(fullfile(bindingRoot, "+nlolib"))
    bindingRoot = fileparts(bindingRoot);
end

addpath(bindingRoot, "-begin");
addpath(scriptDir, "-begin");
if exist("nlolib_setup", "file") == 2
    nlolib_setup();
end
end
