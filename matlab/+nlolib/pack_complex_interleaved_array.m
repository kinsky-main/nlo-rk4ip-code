function ptr = pack_complex_interleaved_array(values)
%PACK_COMPLEX_INTERLEAVED_ARRAY Pack MATLAB complex values into doublePtr.
%   ptr = nlolib.pack_complex_interleaved_array(values)
%
%   The returned buffer uses [re0, im0, re1, im1, ...] layout expected by
%   nlolib_propagate_interleaved().
vals = values(:).';
n = numel(vals);
interleaved = zeros(1, 2 * n);
interleaved(1:2:end) = real(vals);
interleaved(2:2:end) = imag(vals);
ptr = libpointer('doublePtr', interleaved);
end
