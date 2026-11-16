function savedPath = plot_3d_intensity_scatter_propagation( ...
    xAxis, yAxis, zAxis, fieldRecords, outputPath, varargin)
%PLOT_3D_INTENSITY_SCATTER_PROPAGATION Save 3D propagation intensity scatter.

p = inputParser;
addParameter(p, "intensity_cutoff", 0.05);
addParameter(p, "xy_stride", 16);
addParameter(p, "adaptive_cutoff", false);
addParameter(p, "target_point_count", 20000);
addParameter(p, "min_marker_size", 2.0);
addParameter(p, "max_marker_size", 36.0);
addParameter(p, "title", "3D Propagation Intensity Scatter");
parse(p, varargin{:});

x = double(xAxis(:).');
y = double(yAxis(:).');
z = double(zAxis(:).');
records = fieldRecords;

if ndims(records) ~= 3
    error("fieldRecords must be [record, y, x].");
end
if size(records, 1) ~= numel(z) || size(records, 2) ~= numel(y) || size(records, 3) ~= numel(x)
    error("Axis lengths must match fieldRecords shape.");
end
if p.Results.intensity_cutoff < 0.0 || p.Results.intensity_cutoff >= 1.0
    error("intensity_cutoff must be in [0, 1).");
end
if p.Results.target_point_count <= 0
    error("target_point_count must be positive.");
end
xyStrideRaw = p.Results.xy_stride;
if isnumeric(xyStrideRaw)
    if xyStrideRaw <= 0
        error("xy_stride must be positive.");
    end
elseif ~(ischar(xyStrideRaw) || isstring(xyStrideRaw))
    error("xy_stride must be numeric or ""auto"".");
elseif ~strcmpi(string(xyStrideRaw), "auto")
    error("xy_stride string value must be ""auto"".");
end

if isreal(records)
    intensity = double(records);
else
    intensity = abs(double(records)).^2;
end
intensity(~isfinite(intensity)) = 0.0;
intensity(intensity < 0.0) = 0.0;

maxIntensity = max(intensity, [], "all");
if maxIntensity <= 0.0
    warning("plot3d:zeroIntensity", "Intensity is zero everywhere; skipping 3D scatter.");
    savedPath = [];
    return;
end

xPoints = [];
yPoints = [];
zPoints = [];
cPoints = [];
sPoints = [];

normalizedIntensity = intensity ./ maxIntensity;
cutoff = double(p.Results.intensity_cutoff);
if p.Results.adaptive_cutoff
    nonZero = normalizedIntensity(normalizedIntensity > 0.0);
    if ~isempty(nonZero)
        cutoff = min(cutoff, max(0.005, prctile(nonZero, 70.0)));
    end
end

if isnumeric(xyStrideRaw)
    stride = max(1, round(double(xyStrideRaw)));
else
    totalPoints = numel(z) * numel(y) * numel(x);
    targetPoints = max(1, round(double(p.Results.target_point_count)));
    stride = max(1, floor(sqrt(double(totalPoints) / double(targetPoints))));
end
targetPoints = max(1, round(double(p.Results.target_point_count)));

for zi = 1:numel(z)
    for yi = 1:stride:numel(y)
        for xi = 1:stride:numel(x)
            normIntensity = normalizedIntensity(zi, yi, xi);
            if normIntensity < cutoff
                continue;
            end
            xPoints(end + 1) = x(xi); %#ok<AGROW>
            yPoints(end + 1) = y(yi); %#ok<AGROW>
            zPoints(end + 1) = z(zi); %#ok<AGROW>
            cPoints(end + 1) = normIntensity; %#ok<AGROW>
            sPoints(end + 1) = p.Results.min_marker_size + ...
                               (p.Results.max_marker_size - p.Results.min_marker_size) * normIntensity; %#ok<AGROW>
        end
    end
end

if isempty(xPoints)
    warning("plot3d:noPoints", "No points passed intensity cutoff; skipping 3D scatter.");
    savedPath = [];
    return;
end

if numel(xPoints) > targetPoints
    keepStride = ceil(numel(xPoints) / targetPoints);
    keepIdx = 1:keepStride:numel(xPoints);
    xPoints = xPoints(keepIdx);
    yPoints = yPoints(keepIdx);
    zPoints = zPoints(keepIdx);
    cPoints = cPoints(keepIdx);
    sPoints = sPoints(keepIdx);
end

savedPath = backend_save_figure_path(outputPath);
fig = figure("Visible", "off");
ax = axes(fig);
scatter3(ax, xPoints, yPoints, zPoints, sPoints, cPoints, "filled", ...
         "MarkerFaceAlpha", 0.70, "MarkerEdgeAlpha", 0.0);
xlabel(ax, "x");
ylabel(ax, "y");
zlabel(ax, "z");
title(ax, p.Results.title);
colormap(ax, turbo(256));
cb = colorbar(ax);
ylabel(cb, "Normalized intensity");
grid(ax, "on");
view(ax, 45, 28);
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
