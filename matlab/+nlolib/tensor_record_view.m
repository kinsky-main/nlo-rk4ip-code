function volume = tensor_record_view(records, varargin)
%TENSOR_RECORD_VIEW Reshape a coupled-run record into an (nt, ny, nx) volume.
%
%   volume = nlolib.tensor_record_view(record, nt, nx, ny)
%   volume = nlolib.tensor_record_view(records, index, nt, nx, ny)
%
%   A coupled (t, x, y) propagation returns each record as one flat row of
%   nt*nx*ny samples, in TENSOR_LAYOUT_XYT_T_FAST order: t fastest, then y,
%   then x. This reshapes one such row into a 3-D array indexed as
%   volume(it, iy, ix), the same view the Julia binding's tensor_record_view
%   returns.
%
%   RECORDS is either a single record (a vector of nt*nx*ny samples) or the
%   full records matrix from NLolib/propagate, whose rows are records; in the
%   latter form INDEX selects the row, and 0 or a negative index counts back
%   from the last record, so index 0 is the final one.
%
%   Example:
%     result = solver.propagate(pulse, linearOp, nonlinearOp, options);
%     final = nlolib.tensor_record_view(result.records, 0, nt, nx, ny);
%     imagesc(abs(squeeze(final(ceil(nt / 2), :, :))).^2);
%
%   See also NLolib/propagate, nlolib.unpack_records, nlolib.constants

    if numel(varargin) == 3
        index = [];
        [nt, nx, ny] = deal(varargin{1}, varargin{2}, varargin{3});
    elseif numel(varargin) == 4
        [index, nt, nx, ny] = deal(varargin{1}, varargin{2}, varargin{3}, varargin{4});
    else
        error('nlolib:invalidTensorRecordView', ...
              'expected (record, nt, nx, ny) or (records, index, nt, nx, ny)');
    end

    nt = double(nt);
    nx = double(nx);
    ny = double(ny);
    if any([nt, nx, ny] <= 0) || any(floor([nt, nx, ny]) ~= [nt, nx, ny])
        error('nlolib:invalidTensorRecordView', ...
              'nt, nx and ny must be positive integers');
    end

    if isempty(index)
        record = records;
    else
        if isvector(records)
            error('nlolib:invalidTensorRecordView', ...
                  'an index needs the full records matrix, not a single record');
        end
        numRecords = size(records, 1);
        index = double(index);
        if index <= 0
            index = numRecords + index;
        end
        if index < 1 || index > numRecords || floor(index) ~= index
            error('nlolib:invalidTensorRecordView', ...
                  'index %g is outside the %d available records', index, numRecords);
        end
        record = records(index, :);
    end

    expected = nt * ny * nx;
    if numel(record) ~= expected
        error('nlolib:invalidTensorRecordView', ...
              'record has %d samples but nt*nx*ny is %d', numel(record), expected);
    end

    % Column-major reshape puts t fastest, then y, then x, matching the layout
    % the solver wrote.
    volume = reshape(record, nt, ny, nx);
end
