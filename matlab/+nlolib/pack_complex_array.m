function ptr = pack_complex_array(values)
%PACK_COMPLEX_ARRAY Convert a MATLAB complex vector to an interleaved
%   double buffer compatible with nlo_complex* (struct {double re, im}).
%
%   ptr = nlolib.pack_complex_array(values)
%
%   Returns a libpointer('doublePtr', ...) whose memory layout matches
%   an array of nlo_complex structs: [re1 im1 re2 im2 ...].
values = values(:).';
n      = numel(values);
buf    = zeros(1, 2 * n);
buf(1:2:end) = real(values);
buf(2:2:end) = imag(values);
ptr = libpointer('doublePtr', buf);
end
