%STANDALONE_TRANSVERSE_GRIN_BEAM Transverse (x-y) propagation demo, single file.
%
%   Self-contained example that uses nlolib exactly as installed (.mltbx or
%   `addpath(<repo>/build/matlab_toolbox); nlolib_setup()`).  It does not
%   depend on setup_matlab_example_environment or the +backend helpers.
%
%   Problem: a Gaussian beam launched off-axis into an ideal parabolic-index
%   (GRIN) medium.  This is a purely transverse run (tensor_nt = 1), so the
%   field is a 2D x-y sheet propagated along z by
%
%       dA/dz = i * beta_t * (kx^2 + ky^2) * A  +  i * V(x,y) * A
%
%   with beta_t = -1/(2k) (diffraction) and V = -(k*g^2/2)*(x^2 + y^2) (the
%   GRIN lens).  The linear term is applied in the transverse Fourier domain
%   by the runtime operator; V is supplied as a potential grid and consumed
%   by the nonlinear expression "i*A*V".
%
%   Launching a width-matched Gaussian at rest off-axis gives a beam that
%   oscillates rigidly about the axis with pitch period 2*pi/g, so the
%   numerical centroid can be checked against x0*cos(g*z).
%
%   Run with:  standalone_transverse_grin_beam
%
%   Lengths are in micrometres.

if exist("nlolib.NLolib", "class") ~= 8
    error("nlolib is not on the MATLAB path. Install nlolib.mltbx, or run " + ...
          "addpath('<repo>/build/matlab_toolbox'); nlolib_setup();");
end

% ---------------------------------------------------------------- parameters
lambda0 = 1.03;                    % vacuum wavelength [um]
n0      = 1.45;                    % background index
k0      = 2 * pi * n0 / lambda0;   % propagation constant [rad/um]
gGrin   = 2.0e-3;                  % GRIN gradient parameter [1/um]

nx = 128;                          % transverse samples (x)
ny = 128;                          % transverse samples (y)
dx = 1.5;                          % transverse step [um]
dy = 1.5;

wMatched  = sqrt(2.0 / (k0 * gGrin));   % width that propagates without breathing
xOffset   = 20.0;                       % launch offset from the axis [um]
pitch     = 2 * pi / gGrin;             % one full oscillation period [um]
zFinal    = pitch;
numRecords = 96;

betaT     = -1.0 / (2.0 * k0);          % diffraction coefficient
grinDepth = -0.5 * k0 * gGrin^2;        % potential prefactor, V = grinDepth * r^2

fprintf("matched waist = %.2f um, pitch period = %.1f um\n", wMatched, pitch);

% -------------------------------------------------------------- input field
x = ((0:(nx - 1)) - 0.5 * (nx - 1)) * dx;
y = ((0:(ny - 1)) - 0.5 * (ny - 1)) * dy;
[xx, yy] = meshgrid(x, y);                       % both are ny-by-nx

field0    = exp(-(((xx - xOffset).^2 + yy.^2) / wMatched^2));
field0    = complex(field0, zeros(size(field0)));
potential = grinDepth * (xx.^2 + yy.^2);

% nlolib's tensor layout (TENSOR_LAYOUT_XYT_T_FAST) is t fastest, then y, then
% x; with tensor_nt = 1 that is y fastest, which is exactly MATLAB's own
% ordering for an (ny, nx) array, so the sheet flattens with a plain reshape.
flatten = @(m) reshape(m, 1, []);

% ------------------------------------------------------------- solver inputs
pulse = struct();
pulse.samples        = flatten(field0);
pulse.tensor_nt      = 1;                        % purely transverse: no time axis
pulse.tensor_nx      = nx;
pulse.tensor_ny      = ny;
pulse.tensor_layout  = 0;                        % 0 = row-major t,y,x
pulse.delta_x        = dx;
pulse.delta_y        = dy;
pulse.delta_time     = 1.0;                      % unused when tensor_nt == 1
pulse.pulse_period   = 1.0;                      % unused when tensor_nt == 1
pulse.frequency_grid = complex(0.0, 0.0);        % one entry per tensor_nt
pulse.potential_grid = flatten(complex(potential, zeros(size(potential))));

linearOperator = struct( ...
    'expr',   "i*beta_t*(kx*kx + ky*ky)", ...
    'params', struct('beta_t', betaT));

nonlinearOperator = struct('expr', "i*A*V");     % GRIN potential only, no Kerr

options = struct();
options.propagation_distance = zFinal;
options.records              = numRecords;
options.preset               = "accuracy";       % "fast" | "balanced" | "accuracy"
% options.exec_options = struct('backend_type', 0);   % 0 = CPU, 1 = Vulkan, 2 = auto

% ----------------------------------------------------------------- propagate
api = nlolib.NLolib();

tic;
result = api.propagate(pulse, linearOperator, nonlinearOperator, options);
elapsed = toc;

recordsFlat = result.records;
zAxis       = result.z_axis(:).';
nRec        = size(recordsFlat, 1);

records = zeros(nRec, ny, nx);
for idx = 1:nRec
    records(idx, :, :) = reshape(recordsFlat(idx, :), [ny, nx]);
end
intensity = abs(records).^2;

fprintf("propagated %g um on a %dx%d grid in %.2f s (%d records)\n", ...
        zFinal, nx, ny, elapsed, nRec);

% ------------------------------------------------------------- observables
inPower    = sum(intensity(1, :), "all");
outPower   = sum(intensity(end, :), "all");
powerDrift = abs(outPower - inPower) / max(inPower, 1e-12);

centroidX = zeros(1, nRec);
for idx = 1:nRec
    plane = squeeze(intensity(idx, :, :));
    centroidX(idx) = sum(plane .* xx, "all") / max(sum(plane, "all"), 1e-30);
end
centroidAnalytic = xOffset * cos(gGrin * zAxis);
centroidError = max(abs(centroidX - centroidAnalytic));

fprintf("power drift = %.3e, max centroid deviation from x0*cos(g*z) = %.3f um\n", ...
        powerDrift, centroidError);

% ------------------------------------------------------------------- plots
centerY = floor(ny / 2) + 1;
xzMap   = squeeze(intensity(:, centerY, :));     % nRec-by-nx

figure("Name", "nlolib transverse GRIN beam", "Position", [100, 100, 1100, 800]);
tiledlayout(2, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile;
imagesc(x, y, squeeze(intensity(1, :, :)));
set(gca, "YDir", "normal");
axis image;
colorbar;
xlabel("x [\mum]");
ylabel("y [\mum]");
title(sprintf("Input intensity (z = 0), offset %.0f \\mum", xOffset));

nexttile;
imagesc(x, y, squeeze(intensity(end, :, :)));
set(gca, "YDir", "normal");
axis image;
colorbar;
xlabel("x [\mum]");
ylabel("y [\mum]");
title(sprintf("Final intensity (z = %.0f \\mum = one pitch)", zAxis(end)));

nexttile;
imagesc(x, zAxis, xzMap);
set(gca, "YDir", "normal");
colorbar;
xlabel("x [\mum]");
ylabel("z [\mum]");
title("Intensity along y = 0 vs propagation distance");

nexttile;
plot(zAxis, centroidX, "LineWidth", 1.8, "DisplayName", "nlolib centroid");
hold on;
plot(zAxis, centroidAnalytic, "--", "LineWidth", 1.6, ...
     "DisplayName", "x_0 cos(g z)");
grid on;
xlabel("z [\mum]");
ylabel("Beam centroid x [\mum]");
title("Centroid oscillation vs analytical GRIN prediction");
legend("Location", "best");
