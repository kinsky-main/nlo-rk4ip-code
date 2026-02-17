function records = py_record_list_to_matlab(pyRecords)
numRecords = int64(py.len(pyRecords));
if numRecords == 0
    records = complex(zeros(0, 0));
    return;
end

numSamples = int64(py.len(pyRecords{1}));
records = complex(zeros(double(numRecords), double(numSamples)));
for recordIdx = 1:double(numRecords)
    pyRecord = pyRecords{recordIdx};
    for sampleIdx = 1:double(numSamples)
        z = pyRecord{sampleIdx};
        records(recordIdx, sampleIdx) = z;
    end
end
end
