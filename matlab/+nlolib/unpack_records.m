function records = unpack_records(outPtr, numRecords, numTimeSamples)
%UNPACK_RECORDS Convert an interleaved nlo_complex output buffer back to a
%   MATLAB complex matrix of size (numRecords x numTimeSamples).
%
%   records = nlolib.unpack_records(outPtr, numRecords, numTimeSamples)
%
%   outPtr is a libpointer('doublePtr', ...) pointing to the raw output of
%   nlolib_propagate (record-major, interleaved re/im doubles).
numRecords     = double(numRecords);
numTimeSamples = double(numTimeSamples);
totalComplex   = numRecords * numTimeSamples;

% Read the flat interleaved buffer.
outPtr.setdatatype('doublePtr', 1, totalComplex * 2);
flat = outPtr.Value;

re = flat(1:2:end);
im = flat(2:2:end);
cplx = complex(re, im);

records = reshape(cplx, [numTimeSamples, numRecords]).';
end
