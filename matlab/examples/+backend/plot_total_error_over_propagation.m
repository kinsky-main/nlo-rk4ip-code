function savedPath = plot_total_error_over_propagation( ...
    zAxis, errorCurve, outputPath, varargin)
%PLOT_TOTAL_ERROR_OVER_PROPAGATION Save propagation error curve.

p = inputParser;
addParameter(p, "title", "Total Error Over Propagation");
addParameter(p, "y_label", "Relative L2 error");
parse(p, varargin{:});

errors = double(errorCurve(:).');
errors(~isfinite(errors)) = 0.0;
errors(errors < 0.0) = 0.0;

savedPath = backend_save_figure_path(outputPath);
fig = figure("Visible", "off");
ax = axes(fig);
plot(ax, zAxis, errors, "LineWidth", 1.8, "Color", [0.85, 0.325, 0.098]);
xlabel(ax, "Propagation distance z");
ylabel(ax, p.Results.y_label);
title(ax, p.Results.title);
grid(ax, "on");
print(fig, savedPath, "-dpng", "-r200");
close(fig);
end

function outPath = backend_save_figure_path(pathIn)
outPath = char(string(pathIn));
parent = fileparts(outPath);
if strlength(string(parent)) > 0 && ~isfolder(parent)
    mkdir(parent);
end
end
