function report = debug_probe_complex_ptr(ptr, expectedCount, contextLabel, verbose)
%DEBUG_PROBE_COMPLEX_PTR Inspect a MATLAB nlo_complex pointer layout.
%   report = nlolib.debug_probe_complex_ptr(ptr, expectedCount, contextLabel, verbose)
%   returns pointer/value metadata and extracted re/im element counts.

if nargin < 2 || isempty(expectedCount)
    expectedCount = NaN;
end
if nargin < 3 || strlength(string(contextLabel)) == 0
    contextLabel = "probe";
end
if nargin < 4
    verbose = false;
end

report = struct();
report.context = string(contextLabel);
report.pointer_class = class(ptr);
report.pointer_datatype = "";
report.value_class = "";
report.value_size = [0, 0];
report.is_struct = false;
report.has_re = false;
report.has_im = false;
report.re_count = 0;
report.im_count = 0;
report.expected_count = double(expectedCount);
report.error = "";

try
    report.pointer_datatype = string(ptr.DataType);
catch
    report.pointer_datatype = "<unavailable>";
end

raw = [];
try
    raw = ptr.Value;
    report.value_class = class(raw);
    report.value_size = size(raw);
    report.is_struct = isstruct(raw);

    if report.is_struct
        report.has_re = isfield(raw, 're');
        report.has_im = isfield(raw, 'im');
        if report.has_re
            re = [raw.re];
            report.re_count = numel(re);
        end
        if report.has_im
            im = [raw.im];
            report.im_count = numel(im);
        end
    end
catch ME
    report.error = string(ME.message);
end

if logical(verbose)
    fprintf(['[nlolib.matlab_debug] %s ptr.class=%s ptr.datatype=%s ' ...
             'value.class=%s value.size=%s re=%d im=%d expected=%g\n'], ...
            char(report.context), ...
            char(string(report.pointer_class)), ...
            char(string(report.pointer_datatype)), ...
            char(string(report.value_class)), ...
            char(size_to_text(report.value_size)), ...
            int64(report.re_count), ...
            int64(report.im_count), ...
            double(report.expected_count));
    if strlength(report.error) > 0
        fprintf("[nlolib.matlab_debug] %s probe.error=%s\n", ...
                char(report.context), char(report.error));
    end
end
end

function text = size_to_text(sz)
if isempty(sz)
    text = "[]";
    return;
end
text = join(string(double(sz(:).')), "x");
end
