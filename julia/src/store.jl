# Run keys, the shared run database and a reader for stored runs.
# Table layout: docs/nlolib_run_database.md.

"""
    storage_run_key(pulse, linear_operator, nonlinear_operator; kwargs...) -> String
    storage_run_key(config, input_field; num_recorded_samples, kwargs...) -> String

Run key (16 lowercase hex digits) under which `propagate` with the same
arguments stores the run. It covers the physics and the stored output (grids,
steps, tolerances, operators and constants, nonlinear/Raman settings, digests
of the input field and of every given grid, the record schedule, `record_mode`,
raw strides and the final-field flag), not the execution backend. Use it with
[`storage_has_run`](@ref) to skip runs that are already stored.
"""
function storage_run_key(pulse, linear_operator = "gvd", nonlinear_operator = "kerr"; kwargs...)
    _api()
    return run_key(from_pulse(PropagateRequestBuilder(), pulse, linear_operator, nonlinear_operator; kwargs...))
end

function storage_run_key(config::Union{PreparedSimConfig, SimulationConfig}, input_field; num_recorded_samples, kwargs...)
    _api()
    return run_key(from_config(PropagateRequestBuilder(), config, input_field;
        num_recorded_samples = num_recorded_samples, kwargs...))
end

function _check_key(key::AbstractString)
    occursin(r"^[0-9a-f]{16}$", key) || throw(ArgumentError("run key must be 16 lowercase hex digits, got $(repr(key))"))
    return String(key)
end

"""
    storage_has_run(sqlite_path, key) -> Bool

Whether the shared run database at `sqlite_path` has a `runs` row for `key`.
A missing file holds no runs.
"""
function storage_has_run(sqlite_path::AbstractString, key::AbstractString)
    _api()
    present = Ref{Cint}(0)
    status = ccall(_sym(:nlolib_storage_has_run), Cint, (Cstring, Cstring, Ref{Cint}),
                   String(sqlite_path), _check_key(key), present)
    _check_status(status, "nlolib_storage_has_run")
    return present[] != 0
end

"""
    storage_recover_staging(sqlite_path; discard_incomplete = false, min_age_seconds = 86400.0)

Merge the complete staging files left next to `sqlite_path` (runs that finished
but could not take the lock) and, with `discard_incomplete = true`, delete
incomplete ones (runs that crashed or were killed) older than `min_age_seconds`.
Files of live processes on this machine are never touched. Returns
`(merged, discarded, skipped)`.
"""
function storage_recover_staging(sqlite_path::AbstractString; discard_incomplete::Bool = false,
                                 min_age_seconds::Real = 86400.0)
    _api()
    merged = Ref{Csize_t}(0)
    discarded = Ref{Csize_t}(0)
    skipped = Ref{Csize_t}(0)
    status = ccall(_sym(:nlolib_storage_recover_staging), Cint,
                   (Cstring, Cint, Cdouble, Ref{Csize_t}, Ref{Csize_t}, Ref{Csize_t}),
                   String(sqlite_path), Cint(discard_incomplete), Cdouble(min_age_seconds),
                   merged, discarded, skipped)
    _check_status(status, "nlolib_storage_recover_staging")
    return (merged = Int(merged[]), discarded = Int(discarded[]), skipped = Int(skipped[]))
end

"""
    StoredRun

A run read back from a run database with [`read_stored_run`](@ref).

Axes: `z` (record positions), `t` (centered time), `omega` (angular frequency,
ascending), `x`, `y` (centered transverse coordinates).
Maps, one column per record: `power[t, r]` (|A|² summed over x and y),
`spectrum[ω, r]` (|DFT_t A|² summed over x and y, on `omega`),
`fluence_x[x, r]` and `fluence_y[y, r]` (|A|² summed over t and the other
transverse axis). `final_field[t, y, x]` and `raw[t, y, x, r]` are `nothing`
when not stored. Units and conventions: docs/nlolib_run_database.md.
"""
struct StoredRun
    key::String
    run_id::String
    record_mode::Symbol
    nt::Int
    nx::Int
    ny::Int
    delta_time::Float64
    delta_x::Float64
    delta_y::Float64
    z::Vector{Float64}
    t::Vector{Float64}
    omega::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    power::Matrix{Float64}
    spectrum::Matrix{Float64}
    fluence_x::Matrix{Float64}
    fluence_y::Matrix{Float64}
    final_field::Union{Nothing, Array{ComplexF64, 3}}
    raw::Union{Nothing, Array{ComplexF64, 4}}
    raw_strides::NTuple{3, Int}
end

function _read_run_array(path::String, key::String, name::String)
    count = Ref{Csize_t}(0)
    status = ccall(_sym(:nlolib_storage_read_run_array), Cint,
                   (Cstring, Cstring, Cstring, Ptr{Cdouble}, Csize_t, Ref{Csize_t}),
                   path, key, name, C_NULL, 0, count)
    _check_status(status, "nlolib_storage_read_run_array($name)")
    values = Vector{Float64}(undef, Int(count[]))
    isempty(values) && return values
    status = ccall(_sym(:nlolib_storage_read_run_array), Cint,
                   (Cstring, Cstring, Cstring, Ptr{Cdouble}, Csize_t, Ref{Csize_t}),
                   path, key, name, values, Csize_t(length(values)), count)
    _check_status(status, "nlolib_storage_read_run_array($name)")
    return values
end

_as_complex(values::Vector{Float64}) = copy(reinterpret(ComplexF64, values))

"""
    read_stored_run(sqlite_path, key) -> StoredRun

Read the axes, reduced maps, final field and (if stored) raw records of the
run `key` from a run database (shared or staging).
"""
function read_stored_run(sqlite_path::AbstractString, key::AbstractString)
    _api()
    path = String(sqlite_path)
    run_key = _check_key(key)
    layout_ref = Ref(StorageRunLayout())
    status = ccall(_sym(:nlolib_storage_read_run_layout), Cint, (Cstring, Cstring, Ref{StorageRunLayout}),
                   path, run_key, layout_ref)
    _check_status(status, "nlolib_storage_read_run_layout")
    layout = layout_ref[]
    nt, nx, ny = Int(layout.nt), Int(layout.nx), Int(layout.ny)
    records = Int(layout.num_records)
    mode = layout.record_mode == STORAGE_RECORD_MODE_RAW ? :raw :
           layout.record_mode == STORAGE_RECORD_MODE_REDUCED_AND_RAW ? :reduced_and_raw : :reduced
    has_maps = mode != :raw
    map_of(name, n) = has_maps ? reshape(_read_run_array(path, run_key, name), n, records) : zeros(Float64, n, 0)

    final_field = nothing
    if layout.has_final_field != 0
        final_field = reshape(_as_complex(_read_run_array(path, run_key, "final_field")), nt, ny, nx)
    end
    raw = nothing
    raw_records = Int(layout.raw_records)
    if mode != :reduced && raw_records > 0
        raw = reshape(_as_complex(_read_run_array(path, run_key, "raw")),
                      Int(layout.raw_nt), Int(layout.raw_ny), Int(layout.raw_nx), raw_records)
    end
    strides = (Int(layout.raw_stride_t), Int(layout.raw_stride_x), Int(layout.raw_stride_y))
    return StoredRun(
        run_key,
        _decode_run_id(layout.run_id),
        mode,
        nt, nx, ny,
        Float64(layout.delta_time), Float64(layout.delta_x), Float64(layout.delta_y),
        _read_run_array(path, run_key, "z"),
        _read_run_array(path, run_key, "t"),
        _read_run_array(path, run_key, "omega"),
        _read_run_array(path, run_key, "x"),
        _read_run_array(path, run_key, "y"),
        map_of("power", nt),
        map_of("spectrum", nt),
        map_of("fluence_x", nx),
        map_of("fluence_y", ny),
        final_field,
        raw,
        mode == :reduced ? (1, 1, 1) : strides,
    )
end
