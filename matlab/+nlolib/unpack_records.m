function records = unpack_records(outPtr, numRecords, numTimeSamples)
%UNPACK_RECORDS Convert nlo_complex output records back to a MATLAB
%   complex matrix of size (numRecords x numTimeSamples).
%
%   records = nlolib.unpack_records(outPtr, numRecords, numTimeSamples)
%
%   outPtr is a libpointer('nlo_complexPtr', ...) pointing to output
%   records in record-major order.
numRecords     = double(numRecords);
numTimeSamples = double(numTimeSamples);
totalComplex   = numRecords * numTimeSamples;

raw = outPtr.Value;
if ~isstruct(raw) || ~all(isfield(raw, {'re', 'im'}))
    error('nlolib:invalidComplexBuffer', ...
          'Expected outPtr.Value to be an nlo_complex struct array.');
end

re = [raw.re];
im = [raw.im];
if numel(re) ~= totalComplex || numel(im) ~= totalComplex
    error('nlolib:invalidComplexBufferLength', ...
          'Output record length mismatch: expected %d complex values.', ...
          totalComplex);
end

cplx = complex(re, im);
records = reshape(cplx, [numTimeSamples, numRecords]).';
end
