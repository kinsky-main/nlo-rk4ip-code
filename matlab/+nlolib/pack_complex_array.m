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
if ~isempty(vals)
    try
        setdatatype(ptr, 'nlo_complexPtr', 1, numel(vals));
    catch
        % Some MATLAB parser modes may not support explicit size binding.
        % Keep best-effort pointer construction as fallback.
    end
end
end
