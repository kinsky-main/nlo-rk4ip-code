function savedPath = plot_final_intensity_comparison( ...
    xAxis, referenceField, finalField, outputPath, varargin)
%PLOT_FINAL_INTENSITY_COMPARISON Save intensity overlay comparison plot.

p = inputParser;
addParameter(p, "x_label", "x");
addParameter(p, "title", "Final Intensity Comparison");
addParameter(p, "reference_label", "Reference");
addParameter(p, "final_label", "Final");
parse(p, varargin{:});

refIntensity = abs(referenceField(:).').^2;
outIntensity = abs(finalField(:).').^2;

savedPath = backend_save_figure_path(outputPath);
fig = figure("Visible", "off");
ax = axes(fig);
plot(ax, xAxis, refIntensity, "LineWidth", 2.0, "Color", [0.0, 0.447, 0.741], ...
     "DisplayName", sprintf("%s |A|^2", p.Results.reference_label));
hold(ax, "on");
plot(ax, xAxis, outIntensity, "--", "LineWidth", 1.8, "Color", [0.85, 0.325, 0.098], ...
     "DisplayName", sprintf("%s |A|^2", p.Results.final_label));
xlabel(ax, p.Results.x_label);
ylabel(ax, "Intensity |A|^2");
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
