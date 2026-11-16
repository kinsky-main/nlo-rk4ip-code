%STANDALONE_SPATIOTEMPORAL_KERR_BULLET Coupled 3+1D (t,x,y) demo, single file.
%
%   Self-contained example that uses nlolib exactly as installed (.mltbx or
%   `addpath(<repo>/build/matlab_toolbox); nlolib_setup()`).  It does not
%   depend on setup_matlab_example_environment or the +backend helpers.
%
%   Problem: a spatiotemporal Gaussian wave packet in a bulk Kerr medium
%   with anomalous dispersion,
%
%       dA/dz = i*(b2*wt^2 + bt*(kx^2 + ky^2))*A  +  i*gamma*|A|^2*A
%
%   with b2 = beta2/2 (beta2 < 0, anomalous) and bt = -1/(2k) (diffraction).
%   The waist is chosen so the Rayleigh range equals the dispersion length,
%       zR = k*w0^2/2 = L0 = T0^2/|beta2| = LD,
%   so the packet spreads at the same rate in time and in space and neither
%   dimension dominates -- the regime where a light bullet can form.
%
%   Two runs are compared over the same grid:
%     * linear   (gamma = 0)  -- spreads as the textbook Gaussian, and is
%                               checked against w0*sqrt(1+(z/zR)^2) and
%                               T0*sqrt(1+(z/LD)^2);
%     * Kerr     (gamma > 0)  -- self-focusing arrests part of the spread in
%                               both dimensions at once.
%
%   Run with:  standalone_spatiotemporal_kerr_bullet
%
%   Lengths are in micrometres, times in femtoseconds.  The envelope is
%   normalised to unit peak amplitude, so gamma is the on-axis nonlinear
%   phase rate in rad/um.

if exist("nlolib.NLolib", "class") ~= 8
    error("nlolib is not on the MATLAB path. Install nlolib.mltbx, or run " + ...
          "addpath('<repo>/build/matlab_toolbox'); nlolib_setup();");
end

% ---------------------------------------------------------------- parameters
lambda0 = 1.55;                     % vacuum wavelength [um]
n0      = 1.444;                    % bulk index (fused silica)
k0      = 2 * pi * n0 / lambda0;    % propagation constant [rad/um]
beta2   = -0.028;                   % GVD [fs^2/um] (anomalous at 1.55 um)

L0 = 1.0e5;                         % common diffraction/dispersion length [um]
w0 = sqrt(2 * L0 / k0);             % 1/e amplitude radius     -> zR = L0
T0 = sqrt(L0 * abs(beta2));         % 1/e amplitude half-width -> LD = L0

nt = 32;                            % samples along t
nx = 64;                            % samples along x
ny = 64;                            % samples along y
dt = 20.0;                          % [fs]
dx = 32.0;                          % [um]
dy = 32.0;                          % [um]

zFinal     = 1.5 * L0;              % [um]
numRecords = 48;
gammaKerr  = 0.6 / L0;              % nonlinear phase rate at peak [rad/um]

betaT = -1.0 / (2.0 * k0);          % diffraction coefficient

fprintf("w0 = %.1f um, T0 = %.1f fs, zR = LD = %.1f mm, z_end = %.1f mm\n", ...
        w0, T0, L0 * 1e-3, zFinal * 1e-3);

% -------------------------------------------------------------- input packet
t = ((0:(nt - 1)) - 0.5 * (nt - 1)) * dt;
x = ((0:(nx - 1)) - 0.5 * (nx - 1)) * dx;
y = ((0:(ny - 1)) - 0.5 * (ny - 1)) * dy;
[xx, yy] = meshgrid(x, y);                       % both ny-by-nx

temporal = exp(-(t.^2) / (2 * T0^2));            % 1-by-nt, unchirped
spatial  = exp(-((xx.^2 + yy.^2) / w0^2));       % ny-by-nx

field0 = zeros(nt, ny, nx);
for idx = 1:nt
    field0(idx, :, :) = temporal(idx) * spatial;
end
field0 = complex(field0, zeros(size(field0)));

% FFT-order angular frequency grid, one entry per t sample.
idx0  = 0:(nt - 1);
half  = floor((nt - 1) / 2);
omega = (2 * pi / (nt * dt)) * (idx0 - nt * (idx0 > half));

% ------------------------------------------------------------------ solve
api = nlolib.NLolib();

linearCase = run_case(api, 0.0, field0, omega, nt, nx, ny, dt, dx, dy, ...
                      betaT, beta2, zFinal, numRecords, t, x);
kerrCase   = run_case(api, gammaKerr, field0, omega, nt, nx, ny, dt, dx, dy, ...
                      betaT, beta2, zFinal, numRecords, t, x);

zAxis = kerrCase.z;

% ------------------------------------------------------------- observables
% Second moments are converted to 1/e amplitude widths so they can be read
% against the analytical Gaussian laws directly.
pulseWidthLinear = sqrt(2) * linearCase.sigmaT;
pulseWidthKerr   = sqrt(2) * kerrCase.sigmaT;
beamRadiusLinear = 2.0 * linearCase.sigmaX;
beamRadiusKerr   = 2.0 * kerrCase.sigmaX;

pulseWidthTheory = T0 * sqrt(1 + (zAxis / L0).^2);
beamRadiusTheory = w0 * sqrt(1 + (zAxis / L0).^2);

pulseErr = max(abs(pulseWidthLinear - pulseWidthTheory)) / T0;
beamErr  = max(abs(beamRadiusLinear - beamRadiusTheory)) / w0;

fprintf("linear run vs analytical Gaussian: pulse width %.2f%%, beam radius %.2f%%\n", ...
        100 * pulseErr, 100 * beamErr);
fprintf("energy drift: linear %.3e, Kerr %.3e\n", ...
        linearCase.energyDrift, kerrCase.energyDrift);
fprintf("final peak intensity: linear %.3f, Kerr %.3f (relative to input)\n", ...
        linearCase.peakI(end), kerrCase.peakI(end));

% ------------------------------------------------------------------- plots
zMm = zAxis * 1e-3;

figure("Name", "nlolib spatiotemporal Kerr packet", "Position", [80, 60, 1250, 820]);
tiledlayout(2, 3, "TileSpacing", "compact", "Padding", "compact");

nexttile;
imagesc(t, zMm, kerrCase.powerT);
set(gca, "YDir", "normal");
colorbar;
xlabel("t [fs]");
ylabel("z [mm]");
title("Kerr: power vs time");

nexttile;
imagesc(x, zMm, kerrCase.fluenceX);
set(gca, "YDir", "normal");
colorbar;
xlabel("x [\mum]");
ylabel("z [mm]");
title("Kerr: fluence vs x");

nexttile;
imagesc(t, x, kerrCase.finalXT);
set(gca, "YDir", "normal");
colorbar;
xlabel("t [fs]");
ylabel("x [\mum]");
title(sprintf("Kerr: final x-t slice (y = 0, z = %.0f mm)", zMm(end)));

nexttile;
plot(zMm, pulseWidthLinear, "LineWidth", 2.2, "DisplayName", "nlolib linear");
hold on;
plot(zMm, pulseWidthKerr, "LineWidth", 2.2, "DisplayName", "nlolib Kerr");
plot(zMm, pulseWidthTheory, "k--", "LineWidth", 1.4, "DisplayName", "analytic (linear)");
grid on;
xlabel("z [mm]");
ylabel("Pulse width T [fs]");
title("Dispersive spreading in time");
legend("Location", "northwest");

nexttile;
plot(zMm, beamRadiusLinear, "LineWidth", 2.2, "DisplayName", "nlolib linear");
hold on;
plot(zMm, beamRadiusKerr, "LineWidth", 2.2, "DisplayName", "nlolib Kerr");
plot(zMm, beamRadiusTheory, "k--", "LineWidth", 1.4, "DisplayName", "analytic (linear)");
grid on;
xlabel("z [mm]");
ylabel("Beam radius w [\mum]");
title("Diffractive spreading in space");
legend("Location", "northwest");

nexttile;
plot(zMm, linearCase.peakI, "LineWidth", 1.8, "DisplayName", "linear");
hold on;
plot(zMm, kerrCase.peakI, "LineWidth", 1.8, "DisplayName", "Kerr");
grid on;
xlabel("z [mm]");
ylabel("Peak intensity |A|^2");
title("On-axis peak intensity");
legend("Location", "northeast");

% ========================================================================
% Local functions
% ========================================================================

function out = run_case(api, gammaKerr, field0, omega, nt, nx, ny, dt, dx, dy, ...
                        betaT, beta2, zFinal, numRecords, t, x)
%RUN_CASE Propagate the packet once and reduce the records to curves/maps.

% nlolib's tensor layout (TENSOR_LAYOUT_XYT_T_FAST) is t fastest, then y,
% then x -- which is exactly MATLAB's own ordering for an (nt, ny, nx) array,
% so the volume flattens with a plain reshape.
pulse = struct();
pulse.samples        = reshape(field0, 1, []);
pulse.tensor_nt      = nt;
pulse.tensor_nx      = nx;
pulse.tensor_ny      = ny;
pulse.tensor_layout  = 0;                        % 0 = row-major t,y,x
pulse.delta_time     = dt;
pulse.pulse_period   = nt * dt;
pulse.delta_x        = dx;
pulse.delta_y        = dy;
pulse.frequency_grid = complex(omega, zeros(1, nt));   % one entry per tensor_nt

linearOperator = struct( ...
    'expr',   "i*(b2*wt*wt + bt*(kx*kx + ky*ky))", ...
    'params', struct('b2', 0.5 * beta2, 'bt', betaT));

nonlinearOperator = struct( ...
    'expr',   "i*gamma*A*I", ...
    'params', struct('gamma', gammaKerr));

options = struct();
options.propagation_distance = zFinal;
options.records              = numRecords;
options.preset               = "accuracy";       % "fast" | "balanced" | "accuracy"
% options.exec_options = struct('backend_type', 0);   % 0 = CPU, 1 = Vulkan, 2 = auto

tic;
result = api.propagate(pulse, linearOperator, nonlinearOperator, options);
elapsed = toc;

recordsFlat = result.records;
nRec        = size(recordsFlat, 1);
out = struct();
out.z       = result.z_axis(:).';
out.powerT   = zeros(nRec, nt);
out.fluenceX = zeros(nRec, nx);
out.sigmaT   = zeros(1, nRec);
out.sigmaX   = zeros(1, nRec);
out.peakI    = zeros(1, nRec);
energy       = zeros(1, nRec);

tCol = reshape(t, [nt, 1, 1]);
xRow = reshape(x, [1, 1, nx]);
centerY = floor(ny / 2) + 1;

for idx = 1:nRec
    % Records come back flat; restore the (t, y, x) volume.
    volume    = reshape(recordsFlat(idx, :), [nt, ny, nx]);
    intensity = abs(volume).^2;

    total = sum(intensity, "all");
    energy(idx)      = total;
    out.peakI(idx)   = max(intensity, [], "all");
    out.sigmaT(idx)  = sqrt(sum(intensity .* tCol.^2, "all") / total);
    out.sigmaX(idx)  = sqrt(sum(intensity .* xRow.^2, "all") / total);
    out.powerT(idx, :)   = sum(sum(intensity, 3), 2).';
    out.fluenceX(idx, :) = squeeze(sum(sum(intensity, 2), 1)).';

    if idx == nRec
        out.finalXT = squeeze(intensity(:, centerY, :)).';   % nx-by-nt
    end
end

out.energyDrift = abs(energy(end) - energy(1)) / max(energy(1), 1e-30);
fprintf("gamma = %.3e 1/um: %d records over %.1f mm in %.2f s\n", ...
        gammaKerr, nRec, zFinal * 1e-3, elapsed);
end
