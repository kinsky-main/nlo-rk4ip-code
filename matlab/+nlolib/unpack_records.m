function records = unpack_records(outPtr, numRecords, numTimeSamples, debugContext)
%UNPACK_RECORDS Convert nlo_complex output records back to a MATLAB
%   complex matrix of size (numRecords x numTimeSamples).
%
%   records = nlolib.unpack_records(outPtr, numRecords, numTimeSamples)
%
%   outPtr is a libpointer('nlo_complexPtr', ...) pointing to output
%   records in record-major order.
%
%   numRecords is the number of records the library actually wrote
%   (records_written), which may be fewer than the capacity the buffer was
%   allocated with: nlolib_propagate() reduces the record count for
%   fixed-step runs, explicit-z schedules, and callback-aborted runs.
%   Trailing unwritten capacity is therefore ignored rather than treated
%   as a length mismatch.
if nargin < 4
    debugContext = struct();
end

numRecords     = double(numRecords);
numTimeSamples = double(numTimeSamples);
totalComplex   = numRecords * numTimeSamples;

if isstruct(outPtr) && all(isfield(outPtr, {'re', 'im'}))
    raw = outPtr;
else
    try
        raw = outPtr.Value;
    catch
        detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                    "unpack-value", false), ...
                                     debugContext);
        error('nlolib:invalidComplexBuffer', ...
              'Failed to access outPtr.Value for output buffer. %s', ...
              detail);
    end
end

if isnumeric(raw)
    flat = double(raw(:).');
    expectedDoubles = 2 * totalComplex;
    if numel(flat) < expectedDoubles
        detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                    "unpack-numeric-length", false), ...
                                     debugContext);
        error('nlolib:invalidComplexBufferLength', ...
              ['Output buffer too short: need %d doubles (%d complex), ' ...
               'got %d doubles. %s'], ...
              expectedDoubles, totalComplex, numel(flat), detail);
    end
    flat = flat(1:expectedDoubles);
    re = flat(1:2:end);
    im = flat(2:2:end);
elseif isstruct(raw) && all(isfield(raw, {'re', 'im'}))
    re = [raw.re];
    im = [raw.im];
    if numel(re) < totalComplex || numel(im) < totalComplex
        % Reading .Value on an nlo_complex* yields only the first element:
        % MATLAB keeps no element count for struct pointers, and setdatatype()
        % rejects them outright ("Array must be numeric or logical or a
        % pointer to one"). Walk the buffer with pointer arithmetic instead.
        if isa(outPtr, 'lib.pointer')
            [re, im] = walk_complex_pointer(outPtr, totalComplex);
        end
    end
    if numel(re) < totalComplex || numel(im) < totalComplex
        detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                    "unpack-struct-length", false), ...
                                     debugContext);
        error('nlolib:invalidComplexBufferLength', ...
              ['Output record buffer too short: need %d complex values, ' ...
               'got re=%d and im=%d. %s'], ...
              totalComplex, numel(re), numel(im), detail);
    end
    re = re(1:totalComplex);
    im = im(1:totalComplex);
else
    detail = format_probe_report(nlolib.debug_probe_complex_ptr(outPtr, totalComplex, ...
                                                                "unpack-unsupported", false), ...
                                 debugContext);
    error('nlolib:invalidComplexBuffer', ...
          'Unsupported output buffer representation. %s', detail);
end

cplx = complex(re, im);
records = reshape(cplx, [numTimeSamples, numRecords]).';
end

function [re, im] = walk_complex_pointer(ptr, count)
%WALK_COMPLEX_POINTER Read `count` nlo_complex values via pointer arithmetic.
re = zeros(1, count);
im = zeros(1, count);
cursor = ptr;
for idx = 1:count
    value = cursor.Value;
    re(idx) = double(value(1).re);
    im(idx) = double(value(1).im);
    cursor = cursor + 1;
end
end

function out = format_probe_report(report, debugContext)
parts = strings(0, 1);
parts(end + 1, 1) = "ptr.class=" + string(report.pointer_class); %#ok<AGROW>
parts(end + 1, 1) = "ptr.datatype=" + string(report.pointer_datatype); %#ok<AGROW>
parts(end + 1, 1) = "value.class=" + string(report.value_class); %#ok<AGROW>
parts(end + 1, 1) = "value.size=" + size_to_text(report.value_size); %#ok<AGROW>
parts(end + 1, 1) = "is.numeric=" + string(report.is_numeric); %#ok<AGROW>
parts(end + 1, 1) = "count.raw=" + string(report.raw_count); %#ok<AGROW>
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
