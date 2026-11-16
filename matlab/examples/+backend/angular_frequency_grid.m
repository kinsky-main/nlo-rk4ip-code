function omega = angular_frequency_grid(numSamples, deltaTime)
%ANGULAR_FREQUENCY_GRID FFT-order angular frequency grid (unshifted).

n = double(numSamples);
dt = double(deltaTime);
if n <= 0 || dt <= 0
    error("numSamples and deltaTime must be positive.");
end

omega = zeros(1, n);
scale = 2.0 * pi / (n * dt);
half = floor((n - 1) / 2);
for idx = 0:(n - 1)
    if idx <= half
        omega(idx + 1) = idx * scale;
    else
        omega(idx + 1) = -(n - idx) * scale;
    end
end
end
