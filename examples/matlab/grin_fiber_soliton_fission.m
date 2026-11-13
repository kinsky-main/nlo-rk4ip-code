%GRIN_FIBER_SOLITON_FISSION Higher-order soliton fission in a GRIN fibre (t, x, y).
%
%   A 4th-order soliton (in terms of launch power) is launched into the
%   fundamental mode of a graded-index (GRIN) multimode fibre. 
%   The field A(t, y, x, z), in sqrt(W/um^2), obeys
%
%       dA/dz = i*(beta2/2*w^2 + beta3/6*w^3 - (kx^2 + ky^2)/(2k))*A   linear
%             + i*(gamma*|A|^2 + V(x,y))*A                            nonlinear
%
%   with the parabolic GRIN potential V = -(k*g^2/2)*(x^2 + y^2).  The fibre
%   holds the beam in its fundamental mode while, in time, third-order
%   dispersion splits the input into separate fundamental solitons and
%   sheds a dispersive wave on the blue side of the spectrum.
%
%   Run with:  grin_fiber_soliton_fission
%   Units: micrometres, femtoseconds, watts.

% ------------------------------------------------------------------ fibre
lambda0 = 1.55;                       % wavelength [um]
n0      = 1.45;                       % core index on axis
k0      = 2*pi*n0/lambda0;            % propagation constant [rad/um]
g       = sqrt(2*0.01)/25;            % GRIN gradient: delta = 1%, core radius 25 um [1/um]
beta2   = -0.028;                     % group-velocity dispersion [fs^2/um]
beta3   = 0.3;                        % third-order dispersion [fs^3/um]
n2      = 2.6e-8;                     % Kerr index [um^2/W]
gamma   = 2*pi*n2/lambda0;            % Kerr coefficient [1/um per W/um^2]

% ------------------------------------------------------------------ pulse
N  = 4;                               % soliton order
T0 = 30;                              % sech half-width [fs]
w0 = sqrt(2/(k0*g));                  % fundamental GRIN mode radius [um]

LD     = T0^2/abs(beta2);             % dispersion length [um]
gEff   = gamma/(pi*w0^2);             % modal nonlinearity [1/(W um)]
P0     = N^2/(gEff*LD);               % peak power of an N-th order soliton [W]
zFinal = 1.2*LD;

fprintf("mode radius %.1f um, LD = %.1f mm, P0 = %.2f MW, z = %.1f mm\n", ...
        w0, LD/1e3, P0/1e6, zFinal/1e3);

% ------------------------------------------------------------------ grids
nt = 1024; dt = 2.5;                  % time samples, step [fs]
nx = 24;   dx = 2.0;                  % transverse samples (x and y), step [um]
tLaunch = -400;                       % launch delay: solitons drift to later t [fs]

t = ((0:nt-1) - nt/2)*dt;
x = ((0:nx-1) - nx/2)*dx;
[xx, yy] = meshgrid(x, x);
% Physical frequency offset of each fft bin.  nlolib's operator variable wt
% is the fft frequency (+2*pi*k/T), and the envelope convention exp(-i*w*t)
% makes w = -wt, so odd dispersion orders change sign in the operator below.
w = -2*pi/(nt*dt)*[0:nt/2-1, -nt/2:-1];   % [rad/fs]

% Field stored as (t, y, x): MATLAB's column-major order is t fastest, then
% y, then x, which is exactly nlolib's tensor layout, so reshape is enough.
modeShape = exp(-(xx.^2 + yy.^2)/w0^2);                  % ny-by-nx
peakAmp   = sqrt(2*P0/(pi*w0^2));                        % sqrt(peak intensity)
A0 = peakAmp*sech((t(:) - tLaunch)/T0).*reshape(modeShape, [1, nx, nx]);
V  = -0.5*k0*g^2*(xx.^2 + yy.^2);

% ------------------------------------------------------------------ solve
pulse = struct( ...
    'samples',        reshape(complex(A0), 1, []), ...
    'tensor_nt',      nt, 'tensor_nx', nx, 'tensor_ny', nx, 'tensor_layout', 0, ...
    'delta_time',     dt, 'pulse_period', nt*dt, ...
    'delta_x',        dx, 'delta_y', dx, ...
    'potential_grid', reshape(complex(V), 1, []));

linearOp = struct( ...
    'expr',   "i*(b2*wt*wt - b3*wt*wt*wt + bt*(kx*kx + ky*ky))", ...
    'params', struct('b2', beta2/2, 'b3', beta3/6, 'bt', -1/(2*k0)));
nonlinearOp = struct( ...
    'expr',   "i*A*(gamma*I + V)", ...
    'params', struct('gamma', gamma));

% ----------------------------------------------------------------- solver
% The adaptive RK4IP solver shrinks or grows each step so that the relative
% local error stays below error_tolerance, within [min_step_size,
% max_step_size].  Fields set here replace the preset's values; the preset
% fills in anything left out.
options = struct( ...
    'propagation_distance', zFinal, ...
    'records',              50, ...           % z snapshots returned
    'preset',               "balanced", ...   % "fast" | "balanced" | "accuracy"
    'error_tolerance',      1e-6, ...         % relative local error per step
    'starting_step_size',   zFinal/500, ...   % first trial step [um]
    'max_step_size',        zFinal/50, ...    % upper step bound [um]
    'min_step_size',        zFinal/1e5);      % lower bound; steps here are accepted anyway, with a warning
options.exec_options = struct( ...
    'backend_type',         1, ...            % 0 CPU, 1 Vulkan GPU, 2 auto
    'capture_step_history', true);            % fills result.step_history

result = nlolib.NLolib().propagate(pulse, linearOp, nonlinearOp, options);

steps = result.step_history;
fprintf("%d accepted steps, step %.2f to %.2f um, max local error %.1e\n", ...
        numel(steps.step_size), min(steps.step_size), max(steps.step_size), ...
        max(steps.error));

% ------------------------------------------------------------- observables
z    = result.z_axis(:)/1e3;          % [mm]
nRec = numel(z);
powerT   = zeros(nRec, nt);           % power vs time         [W]
spectrum = zeros(nRec, nt);           % modal energy spectrum [a.u.]
fluenceX = zeros(nRec, nx);           % fluence across y = 0  [a.u.]
for k = 1:nRec
    Ak = reshape(result.records(k, :), [nt, nx, nx]);
    Pt = squeeze(sum(abs(Ak).^2, [2, 3]))*dx^2;
    powerT(k, :)   = Pt;
    spectrum(k, :) = sum(abs(fft(Ak)).^2, [2, 3]);
    fluenceX(k, :) = squeeze(sum(abs(Ak(:, nx/2 + 1, :)).^2, 1));
end
[wSorted, order] = sort(w);
spectrum = spectrum(:, order);
dB = @(m) 10*log10(m/max(m, [], "all") + 1e-12);

fprintf("energy drift %.2e, peak power in/out %.2f / %.2f MW\n", ...
        abs(sum(powerT(end, :))/sum(powerT(1, :)) - 1), ...
        max(powerT(1, :))/1e6, max(powerT(end, :))/1e6);

% ------------------------------------------------------------------ plots
fig = figure("Name", "GRIN soliton fission", "Position", [80, 80, 1200, 760]);
tiledlayout(2, 2, "TileSpacing", "compact");

nexttile;
imagesc(t, z, dB(powerT)); clim([-30, 0]);
axis xy; colorbar; xlabel("t [fs]"); ylabel("z [mm]");
title("Power in time [dB]: soliton fission");

nexttile;
imagesc(wSorted, z, dB(spectrum)); clim([-40, 0]);
axis xy; colorbar; xlabel("\omega - \omega_0 [rad/fs]"); ylabel("z [mm]");
title("Spectrum [dB]: dispersive wave");

nexttile;
plot(t, powerT(1, :)/1e6, "k--", t, powerT(end, :)/1e6, "LineWidth", 1.5);
xlabel("t [fs]"); ylabel("P [MW]"); legend("input", "output"); grid on;
title("Input vs output pulse");

nexttile;
imagesc(x, z, fluenceX);
axis xy; colorbar; xlabel("x [\mum]"); ylabel("z [mm]");
title("Fluence across the core: GRIN keeps the mode");

outDir = fullfile(fileparts(mfilename("fullpath")), "output");
if ~isfolder(outDir), mkdir(outDir); end
exportgraphics(fig, fullfile(outDir, "grin_fiber_soliton_fission.png"));
