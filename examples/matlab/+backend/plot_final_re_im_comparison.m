function savedPath = plot_final_re_im_comparison( ...
    xAxis, referenceField, finalField, outputPath, varargin)
%PLOT_FINAL_RE_IM_COMPARISON Save Re/Im overlay comparison plot.

p = inputParser;
addParameter(p, "x_label", "x");
addParameter(p, "title", "Final Re/Im Comparison");
addParameter(p, "reference_label", "Reference");
addParameter(p, "final_label", "Final");
parse(p, varargin{:});

ref = referenceField(:).';
out = finalField(:).';

savedPath = backend_save_figure_path(outputPath);
fig = figure("Visible", "off");
ax = axes(fig);
plot(ax, xAxis, real(ref), "LineWidth", 1.8, "Color", [0.0, 0.447, 0.741], ...
     "DisplayName", sprintf("%s Re", p.Results.reference_label));
hold(ax, "on");
plot(ax, xAxis, imag(ref), "LineWidth", 1.8, "Color", [0.85, 0.325, 0.098], ...
     "DisplayName", sprintf("%s Im", p.Results.reference_label));
plot(ax, xAxis, real(out), "--", "LineWidth", 1.6, "Color", [0.0, 0.447, 0.741], ...
     "DisplayName", sprintf("%s Re", p.Results.final_label));
plot(ax, xAxis, imag(out), "--", "LineWidth", 1.6, "Color", [0.85, 0.325, 0.098], ...
     "DisplayName", sprintf("%s Im", p.Results.final_label));
xlabel(ax, p.Results.x_label);
ylabel(ax, "Field amplitude");
title(ax, p.Results.title);
grid(ax, "on");
legend(ax, "Location", "best");
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
