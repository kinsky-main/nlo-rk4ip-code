const _LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _LIB_PATH = Ref{String}("")

function _library_filename()
    if Sys.iswindows()
        return "nlolib.dll"
    elseif Sys.isapple()
        return "libnlolib.dylib"
    end
    return "libnlolib.so"
end

function _package_root()
    return normpath(joinpath(@__DIR__, ".."))
end

function _candidate_library_paths(path::Union{Nothing, AbstractString})
    filename = _library_filename()
    package_root = _package_root()
    repo_root = normpath(joinpath(package_root, ".."))
    candidates = String[]

    if path !== nothing
        push!(candidates, normpath(String(path)))
    end

    env_path = get(ENV, "NLOLIB_LIBRARY", "")
    if !isempty(env_path)
        push!(candidates, normpath(env_path))
    end

    append!(candidates, [
        joinpath(package_root, "lib", filename),
        joinpath(repo_root, "python", filename),
        joinpath(repo_root, "python", "Debug", filename),
        joinpath(repo_root, "python", "Release", filename),
        joinpath(repo_root, "build", "julia_package", "lib", filename)
    ])

    return unique(filter(isfile, candidates))
end

function load(path::Union{Nothing, AbstractString} = nothing)
    current = _LIB_HANDLE[]
    if current != C_NULL && (path === nothing || normpath(String(path)) == _LIB_PATH[])
        return current
    end

    if current != C_NULL
        Libdl.dlclose(current)
        _LIB_HANDLE[] = C_NULL
        _LIB_PATH[] = ""
    end

    candidates = _candidate_library_paths(path)
    isempty(candidates) && error("Cannot locate nlolib shared library. Set NLOLIB_LIBRARY or build julia_stage.")

    errors = String[]
    for candidate in candidates
        handle = Libdl.dlopen_e(candidate)
        if handle != C_NULL
            _LIB_HANDLE[] = handle
            _LIB_PATH[] = candidate
            return handle
        end
        push!(errors, candidate)
    end

    error("Failed to load nlolib shared library from: " * join(errors, ", "))
end

loaded_library_path() = _LIB_PATH[]

_handle() = _LIB_HANDLE[] == C_NULL ? load() : _LIB_HANDLE[]
_sym(name::Symbol) = Libdl.dlsym(_handle(), name)
