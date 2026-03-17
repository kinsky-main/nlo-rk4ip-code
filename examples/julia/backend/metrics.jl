using LinearAlgebra

function relative_l2_intensity_error(prediction, reference)
    pred = ComplexF64.(prediction)
    ref = ComplexF64.(reference)
    size(pred) == size(ref) || throw(ArgumentError("prediction and reference must have identical shape"))
    pred_intensity = abs2.(pred)
    ref_intensity = abs2.(ref)
    diff_norm = norm(pred_intensity .- ref_intensity)
    ref_norm = norm(ref_intensity)
    return ref_norm <= 0.0 ? NaN : diff_norm / ref_norm
end

function relative_l2_intensity_error_curve(prediction_records, reference_records)
    pred = ComplexF64.(prediction_records)
    ref = ComplexF64.(reference_records)
    size(pred) == size(ref) || throw(ArgumentError("prediction and reference must have identical shape"))
    ndims(pred) <= 1 && return [relative_l2_intensity_error(pred, ref)]
    values = Vector{Float64}(undef, size(pred, 1))
    for idx in axes(pred, 1)
        values[idx] = relative_l2_intensity_error(view(pred, idx, :), view(ref, idx, :))
    end
    return values
end
