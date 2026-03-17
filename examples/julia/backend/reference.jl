using FFTW

function exact_linear_tensor3d_records(field0_tyx,
                                       z_records,
                                       omega,
                                       kx,
                                       ky,
                                       beta2_scale::Real,
                                       beta_t::Real)
    field = ComplexF64.(field0_tyx)
    ndims(field) == 3 || throw(ArgumentError("field0_tyx must have shape [nt, ny, nx]"))
    nt, ny, nx = size(field)
    omega_axis = Float64.(collect(omega))
    kx_axis = Float64.(collect(kx))
    ky_axis = Float64.(collect(ky))
    length(omega_axis) == nt || throw(ArgumentError("omega length must match nt"))
    length(kx_axis) == nx || throw(ArgumentError("kx length must match nx"))
    length(ky_axis) == ny || throw(ArgumentError("ky length must match ny"))

    field_xyt = permutedims(field, (3, 2, 1))
    spectrum0 = fft(field_xyt)
    wt_grid = reshape(omega_axis, 1, 1, nt)
    kx_grid = reshape(kx_axis, nx, 1, 1)
    ky_grid = reshape(ky_axis, 1, ny, 1)
    linear_factor = (1im * Float64(beta2_scale)) .* (wt_grid .^ 2) .+
                    (1im * Float64(beta_t)) .* ((kx_grid .^ 2) .+ (ky_grid .^ 2))

    z_axis = Float64.(collect(z_records))
    out = Array{ComplexF64}(undef, length(z_axis), nt, ny, nx)
    for (idx, z_value) in pairs(z_axis)
        phase = exp.(linear_factor .* z_value)
        propagated = ifft(spectrum0 .* phase)
        out[idx, :, :, :] = permutedims(propagated, (3, 2, 1))
    end
    return out
end

function second_order_soliton_period(beta2::Real, t0::Real)
    ld = (Float64(t0)^2) / abs(Float64(beta2))
    return 0.5 * π * ld
end

function second_order_soliton_normalized_envelope(t, z::Real, beta2::Real, t0::Real)
    ld = (Float64(t0)^2) / abs(Float64(beta2))
    xi = Float64(z) / ld
    tau = Float64.(collect(t))
    numerator = 4.0 .* (cosh.(3.0 .* tau) .+ 3.0 .* exp.(4.0im .* xi) .* cosh.(tau)) .* exp(0.5im * xi)
    denominator = cosh.(4.0 .* tau) .+ 4.0 .* cosh.(2.0 .* tau) .+ 3.0 .* cos.(4.0 .* xi)
    return numerator ./ denominator
end

function second_order_soliton_normalized_records(t, z_records, beta2::Real, t0::Real)
    tau = Float64.(collect(t))
    z_axis = Float64.(collect(z_records))
    out = Array{ComplexF64}(undef, length(z_axis), length(tau))
    for (idx, z_value) in pairs(z_axis)
        out[idx, :] = second_order_soliton_normalized_envelope(tau, z_value, beta2, t0)
    end
    return out
end
