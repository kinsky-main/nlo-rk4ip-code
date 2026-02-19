function ptr = pack_complex_array(values)
%PACK_COMPLEX_ARRAY Convert a MATLAB complex vector to an interleaved
%   double buffer compatible with nlo_complex* (struct {double re, im}).
%
%   ptr = nlolib.pack_complex_array(values)
%
%   Returns a typed libpointer('nlo_complexPtr', ...) whose payload is a
%   struct array with fields .re and .im.
vals = values(:).';
re = num2cell(real(vals));
im = num2cell(imag(vals));
arr = struct('re', re, 'im', im);
ptr = libpointer('nlo_complexPtr', arr);
end
