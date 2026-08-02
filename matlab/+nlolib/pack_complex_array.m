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
% NOTE: setdatatype() cannot be used to bind an element count here --
% MATLAB rejects struct pointers with "Array must be numeric or logical or a
% pointer to one". The full array is still passed to the library correctly;
% only read-back through .Value is limited to the first element, which
% nlolib.unpack_records() works around with pointer arithmetic.
end
