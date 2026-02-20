function records = unpack_records(outPtr, numRecords, numTimeSamples, debugContext)
%UNPACK_RECORDS Convert nlo_complex output records back to a MATLAB
%   complex matrix of size (numRecords x numTimeSamples).
%
%   records = nlolib.unpack_records(outPtr, numRecords, numTimeSamples)
%
%   outPtr is a libpointer('nlo_complexPtr', ...) pointing to output
%   records in record-major order.
if nargin < 4
    debugContext = struct();
end

numRecords     = double(numRecords);
numTimeSamples = double(numTimeSamples);
totalComplex   = numRecords * numTimeSamples;

try
    raw = outPtr.Value;
catch
    detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                "unpack-value", false), ...
                                 debugContext);
    error('nlolib:invalidComplexBuffer', ...
          'Failed to access outPtr.Value for nlo_complex buffer. %s', ...
          detail);
end

if ~isstruct(raw) || ~all(isfield(raw, {'re', 'im'}))
    detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                "unpack-struct", false), ...
                                 debugContext);
    error('nlolib:invalidComplexBuffer', ...
          'Expected outPtr.Value to be an nlo_complex struct array. %s', ...
          detail);
end

re = [raw.re];
im = [raw.im];
if numel(re) ~= totalComplex || numel(im) ~= totalComplex
    detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                "unpack-length", false), ...
                                 debugContext);
    error('nlolib:invalidComplexBufferLength', ...
          ['Output record length mismatch: expected %d complex values, ' ...
           'got re=%d and im=%d. %s'], ...
          totalComplex, numel(re), numel(im), detail);
end

cplx = complex(re, im);
records = reshape(cplx, [numTimeSamples, numRecords]).';
end

function out = format_probe_report(report, debugContext)
parts = strings(0, 1);
parts(end + 1, 1) = "ptr.class=" + string(report.pointer_class); %#ok<AGROW>
parts(end + 1, 1) = "ptr.datatype=" + string(report.pointer_datatype); %#ok<AGROW>
parts(end + 1, 1) = "value.class=" + string(report.value_class); %#ok<AGROW>
parts(end + 1, 1) = "value.size=" + size_to_text(report.value_size); %#ok<AGROW>
parts(end + 1, 1) = "has.re=" + string(report.has_re); %#ok<AGROW>
parts(end + 1, 1) = "has.im=" + string(report.has_im); %#ok<AGROW>
parts(end + 1, 1) = "count.re=" + string(report.re_count); %#ok<AGROW>
parts(end + 1, 1) = "count.im=" + string(report.im_count); %#ok<AGROW>
parts(end + 1, 1) = "expected=" + string(report.expected_count); %#ok<AGROW>
if strlength(string(report.error)) > 0
    parts(end + 1, 1) = "probe.error=" + string(report.error); %#ok<AGROW>
end

if isstruct(debugContext) && isfield(debugContext, 'enabled') && logical(debugContext.enabled)
    parts(end + 1, 1) = "debug.enabled=true"; %#ok<AGROW>
    if isfield(debugContext, 'pre_probe') && isstruct(debugContext.pre_probe)
        parts(end + 1, 1) = "pre.value.size=" + size_to_text(debugContext.pre_probe.value_size); %#ok<AGROW>
    end
    if isfield(debugContext, 'post_probe') && isstruct(debugContext.post_probe)
        parts(end + 1, 1) = "post.value.size=" + size_to_text(debugContext.post_probe.value_size); %#ok<AGROW>
    end
end

out = char(join(parts, " | "));
end

function text = size_to_text(sz)
if isempty(sz)
    text = "[]";
    return;
end
text = join(string(double(sz(:).')), "x");
end
