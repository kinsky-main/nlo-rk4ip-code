function savedPath = plot_intensity_colormap_vs_propagation( ...
    xAxis, zAxis, intensityMap, outputPath, varargin)
%PLOT_INTENSITY_COLORMAP_VS_PROPAGATION Save normalized intensity colormap.

p = inputParser;
addParameter(p, "x_label", "x");
addParameter(p, "y_label", "Propagation distance z");
addParameter(p, "title", "Intensity vs Propagation");
addParameter(p, "colorbar_label", "Normalized intensity");
addParameter(p, "cmap", "magma");
parse(p, varargin{:});

data = double(intensityMap);
data(~isfinite(data)) = 0.0;
data(data < 0.0) = 0.0;
peak = max(data, [], "all");
if peak > 0.0
    data = data / peak;
end

savedPath = backend_save_figure_path(outputPath);
fig = figure("Visible", "off");
ax = axes(fig);
imagesc(ax, xAxis, zAxis, data);
set(ax, "YDir", "normal");
colormap(ax, backend_map_colormap(string(p.Results.cmap)));
xlabel(ax, p.Results.x_label);
ylabel(ax, p.Results.y_label);
title(ax, p.Results.title);
cb = colorbar(ax);
ylabel(cb, p.Results.colorbar_label);
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

function cmap = backend_map_colormap(name)
switch lower(name)
    case "magma"
        cmap = hot(256);
    case "viridis"
        cmap = parula(256);
    case "plasma"
        cmap = turbo(256);
    otherwise
        cmap = parula(256);
end
end
