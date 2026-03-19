using ArgParse

function centered_time_grid(num_samples::Integer, delta_time::Real)
    return (collect(0:Int(num_samples) - 1) .- 0.5 * Float64(num_samples - 1)) .* Float64(delta_time)
end

function centered_spatial_grid(num_samples::Integer, delta::Real)
    return (collect(0:Int(num_samples) - 1) .- 0.5 * Float64(num_samples - 1)) .* Float64(delta)
end

function repo_root_from(path::AbstractString)
    current = normpath(dirname(path))
    for _ in 1:8
        if isfile(joinpath(current, "CMakeLists.txt")) && isdir(joinpath(current, "src"))
            return current
        end
        parent = dirname(current)
        parent == current && break
        current = parent
    end
    error("unable to locate repository root from $path")
end

function nlo_package_root_from(path::AbstractString)
    candidates = [
        normpath(joinpath(dirname(path), "..", "..", "julia")),
        normpath(joinpath(dirname(path), "..", "..", "..")),
    ]
    for candidate in candidates
        if isfile(joinpath(candidate, "Project.toml")) && isfile(joinpath(candidate, "src", "NLOLib.jl"))
            return candidate
        end
    end
    error("unable to locate the NLOLib Julia package root from $path")
end

function build_example_parser(example_slug::AbstractString, description::AbstractString)
    settings = ArgParseSettings(description = description, autofix_names = true)
    @add_arg_table! settings begin
        "--replot"
        help = "Reload the latest matching run from SQLite instead of rerunning the solver."
        action = :store_true
        "--run-group"
        help = "Explicit run group to replay. Numeric values select the nth-latest run."
        arg_type = String
        default = ""
        "--db-path"
        help = "SQLite path for example metadata and solver snapshots."
        arg_type = String
        default = joinpath(repo_root_from(@__FILE__), "examples", "output", "db", string(example_slug, ".sqlite3"))
        "--output-dir"
        help = "Directory for generated figures."
        arg_type = String
        default = joinpath(dirname(dirname(@__FILE__)), "output", example_slug)
    end
    return settings
end

function parse_example_args(example_slug::AbstractString, description::AbstractString, argv = ARGS)
    return parse_args(argv, build_example_parser(example_slug, description); as_symbols = true)
end
