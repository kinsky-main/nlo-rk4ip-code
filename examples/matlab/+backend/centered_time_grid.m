function t = centered_time_grid(numSamples, deltaTime)
%CENTERED_TIME_GRID Return centered time axis with spacing deltaTime.
t = ((0:(double(numSamples) - 1)) - 0.5 * (double(numSamples) - 1)) * double(deltaTime);
end
