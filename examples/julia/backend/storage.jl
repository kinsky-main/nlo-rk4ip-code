using Dates
using JSON3
using SHA
using SQLite
using DBInterface

function _emit_db_progress(label::AbstractString, current::Integer, total::Integer)
    total_i = max(Int(total), 1)
    current_i = clamp(Int(current), 0, total_i)
    width = 24
    filled = round(Int, width * (current_i / total_i))
    bar = repeat("#", filled) * repeat("-", width - filled)
    print(stderr, "[example-db] [", bar, "] ", current_i, "/", total_i, " ", label, current_i >= total_i ? "\n" : "\r")
    flush(stderr)
    return nothing
end

struct LoadedCase
    run_id::String
    case_key::String
    meta::Dict{String, Any}
    records::Matrix{ComplexF64}
    z_axis::Vector{Float64}
    requested_records::Int
    num_time_samples::Int
    z_end::Float64
end

mutable struct ExampleRunDB
    db_path::String
    function ExampleRunDB(db_path::AbstractString)
        raw = strip(String(db_path))
        isempty(raw) && error("example database path must not be empty")
        resolved = normpath(abspath(expanduser(raw)))
        isdir(resolved) && error("example database path '$resolved' points to a directory")
        parent = dirname(resolved)
        isempty(parent) || mkpath(parent)
        if !isfile(resolved)
            open(resolved, "a") do _
            end
        end
        db = new(resolved)
        ensure_schema!(db)
        return db
    end
end

function _with_db(f::Function, db::ExampleRunDB)
    conn = SQLite.DB(db.db_path)
    try
        return f(conn)
    finally
        SQLite.close(conn)
    end
end

function ensure_schema!(db::ExampleRunDB)
    _with_db(db) do conn
        SQLite.execute(conn, "PRAGMA foreign_keys=ON;")
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS ex_run_groups (
              example_name TEXT NOT NULL,
              run_group TEXT NOT NULL,
              created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(example_name, run_group)
            );
            """)
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS ex_case_runs (
              example_name TEXT NOT NULL,
              run_group TEXT NOT NULL,
              case_key TEXT NOT NULL,
              run_id TEXT NOT NULL,
              meta_json TEXT NOT NULL DEFAULT '{}',
              created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(example_name, run_group, case_key),
              UNIQUE(run_id),
              FOREIGN KEY(example_name, run_group)
                REFERENCES ex_run_groups(example_name, run_group) ON DELETE CASCADE
            );
            """)
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS ex_step_history (
              run_id TEXT PRIMARY KEY,
              event_count INTEGER NOT NULL,
              dropped INTEGER NOT NULL,
              capacity INTEGER NOT NULL,
              step_index_blob BLOB NOT NULL,
              z_blob BLOB NOT NULL,
              step_size_blob BLOB NOT NULL,
              next_step_size_blob BLOB NOT NULL,
              error_blob BLOB NOT NULL
            );
            """)
    end
    return db
end

new_run_group_id() = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS.sssZ")

function make_run_id(example_name::AbstractString, run_group::AbstractString, case_key::AbstractString)
    stamp = Dates.format(now(UTC), dateformat"yyyymmddHHMMSSsss")
    base = string(example_name, "|", run_group, "|", case_key, "|", stamp)
    digest = bytes2hex(sha1(base))[1:16]
    prefix = replace(lowercase(String(example_name)), r"[^a-z0-9]" => "")[1:min(end, 18)]
    return string(prefix, "-", stamp[max(end - 9, 1):end], "-", digest)
end

function begin_group(db::ExampleRunDB, example_name::AbstractString, run_group::Union{Nothing, AbstractString} = nothing)
    println(stderr, "beginning run group for example '", example_name, "' with group '", run_group, "' at path '", db.db_path, "'")
    resolved = run_group === nothing || isempty(run_group) ? new_run_group_id() : String(run_group)
    _with_db(db) do conn
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS ex_run_groups (
              example_name TEXT NOT NULL,
              run_group TEXT NOT NULL,
              created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(example_name, run_group)
            );
            """)
        SQLite.execute(conn,
            "INSERT OR IGNORE INTO ex_run_groups(example_name, run_group) VALUES(?, ?);",
            (String(example_name), resolved))
    end
    return resolved
end

function _single_string_query(db::ExampleRunDB, sql::AbstractString, params)
    return _with_db(db) do conn
        DBInterface.execute(conn, sql, params) do rows
            for row in rows
                value = row[1]
                ismissing(value) && return nothing
                return String(value)
            end
            return nothing
        end
    end
end

latest_run_group(db::ExampleRunDB, example_name::AbstractString) =
    _single_string_query(db,
        "SELECT run_group FROM ex_run_groups WHERE example_name=? ORDER BY created_utc DESC, run_group DESC LIMIT 1;",
        (String(example_name),))

function nth_latest_run_group(db::ExampleRunDB, example_name::AbstractString, run_number::Integer)
    Int(run_number) > 0 || return nothing
    return _single_string_query(db,
        "SELECT run_group FROM ex_run_groups WHERE example_name=? ORDER BY created_utc DESC, run_group DESC LIMIT 1 OFFSET ?;",
        (String(example_name), Int(run_number - 1)))
end

function resolve_replot_group(db::ExampleRunDB, example_name::AbstractString, run_group::Union{Nothing, AbstractString})
    if run_group !== nothing && !isempty(run_group)
        selector = String(run_group)
        if all(isdigit, selector)
            resolved = nth_latest_run_group(db, example_name, parse(Int, selector))
            resolved === nothing && error("run number $selector is not available for example '$example_name'")
            return resolved
        end
        return selector
    end
    latest = latest_run_group(db, example_name)
    latest === nothing && error("no stored run groups found for example '$example_name'")
    return latest
end

function storage_kwargs(db::ExampleRunDB;
                        example_name::AbstractString,
                        run_group::AbstractString,
                        case_key::AbstractString,
                        chunk_records::Integer = 0,
                        sqlite_max_bytes::Integer = 0,
                        log_final_output_field_to_db::Bool = false)
    return (
        sqlite_path = db.db_path,
        run_id = make_run_id(example_name, run_group, case_key),
        chunk_records = Int(chunk_records),
        sqlite_max_bytes = Int(sqlite_max_bytes),
        log_final_output_field_to_db = Bool(log_final_output_field_to_db),
    )
end

function _meta_json(meta)
    return String(JSON3.write(meta === nothing ? Dict{String, Any}() : meta))
end

function save_case!(db::ExampleRunDB;
                    example_name::AbstractString,
                    run_group::AbstractString,
                    case_key::AbstractString,
                    run_id::AbstractString,
                    meta = nothing)
    begin_group(db, example_name, run_group)
    _with_db(db) do conn
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS ex_case_runs (
              example_name TEXT NOT NULL,
              run_group TEXT NOT NULL,
              case_key TEXT NOT NULL,
              run_id TEXT NOT NULL,
              meta_json TEXT NOT NULL DEFAULT '{}',
              created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(example_name, run_group, case_key),
              UNIQUE(run_id),
              FOREIGN KEY(example_name, run_group)
                REFERENCES ex_run_groups(example_name, run_group) ON DELETE CASCADE
            );
            """)
        SQLite.execute(conn,
            "INSERT OR REPLACE INTO ex_case_runs(example_name, run_group, case_key, run_id, meta_json) VALUES(?,?,?,?,?);",
            (String(example_name), String(run_group), String(case_key), String(run_id), _meta_json(meta)))
    end
end

function save_case_from_solver_meta!(db::ExampleRunDB;
                                     example_name::AbstractString,
                                     run_group::AbstractString,
                                     case_key::AbstractString,
                                     solver_meta::Dict{String, Any},
                                     meta = nothing,
                                     save_step_history::Bool = false)
    storage_result = get(solver_meta, "storage_result", nothing)
    storage_result isa Dict || error("solver meta does not contain storage_result")
    run_id = String(get(storage_result, "run_id", ""))
    isempty(run_id) && error("solver storage_result did not provide a run_id")
    _emit_db_progress("writing metadata for $(case_key)", 0, 1)
    save_case!(db;
        example_name = example_name,
        run_group = run_group,
        case_key = case_key,
        run_id = run_id,
        meta = meta)
    save_step_history && save_step_history!(db; run_id = run_id, step_history = get(solver_meta, "step_history", nothing))
    _emit_db_progress("writing metadata for $(case_key)", 1, 1)
    return run_id
end

function _decode_meta_json(raw_meta)
    raw_meta === nothing && return Dict{String, Any}()
    text = String(raw_meta)
    isempty(text) && return Dict{String, Any}()
    decoded = JSON3.read(text)
    decoded isa JSON3.Object || return Dict{String, Any}()
    return Dict(string(k) => v for (k, v) in pairs(decoded))
end

function _blob_to_array(blob, ::Type{T}) where {T}
    bytes = Vector{UInt8}(blob)
    return copy(reinterpret(T, bytes))
end

function _load_records(conn, run_id::AbstractString, num_time_samples::Integer)
    rows = NamedTuple{(:record_start, :record_count, :payload), Tuple{Int, Int, Vector{UInt8}}}[]
    DBInterface.execute(conn,
        "SELECT record_start, record_count, payload FROM io_record_chunks WHERE run_id=? ORDER BY record_start ASC;",
        (String(run_id),)) do query
        for row in query
            push!(rows, (
                record_start = Int(row.record_start),
                record_count = Int(row.record_count),
                payload = Vector{UInt8}(row.payload),
            ))
        end
    end
    isempty(rows) && error("run_id '$run_id' has no io_record_chunks")
    max_records = maximum(row.record_start + row.record_count for row in rows)
    out = zeros(ComplexF64, max_records, Int(num_time_samples))
    for row in rows
        start = row.record_start
        count = row.record_count
        payload = _blob_to_array(row.payload, Float64)
        expected = count * Int(num_time_samples) * 2
        length(payload) == expected || error("run_id '$run_id' chunk decode mismatch")
        records = reshape(reinterpret(ComplexF64, payload), Int(num_time_samples), count)'
        out[start + 1:start + count, :] .= records
    end
    return out
end

function load_case(db::ExampleRunDB; example_name::AbstractString, run_group::AbstractString, case_key::AbstractString)
    return _with_db(db) do conn
        row_data = DBInterface.execute(conn,
            "SELECT c.run_id, c.meta_json, r.num_recorded_samples, r.num_time_samples, s.z_end "
            * "FROM ex_case_runs c "
            * "JOIN io_runs r ON r.run_id = c.run_id "
            * "JOIN io_sim_config s ON s.config_hash = r.config_hash "
            * "WHERE c.example_name=? AND c.run_group=? AND c.case_key=?;",
            (String(example_name), String(run_group), String(case_key))) do rows
            for row in rows
                return (
                    run_id = String(row.run_id),
                    meta_json = ismissing(row.meta_json) ? nothing : String(row.meta_json),
                    requested_records = Int(row.num_recorded_samples),
                    num_time_samples = Int(row.num_time_samples),
                    z_end = Float64(row.z_end),
                )
            end
            return nothing
        end
        row_data === nothing && error("missing case '$case_key' for example '$example_name' / '$run_group'")
        run_id = row_data.run_id
        num_time_samples = row_data.num_time_samples
        requested_records = row_data.requested_records
        z_end = row_data.z_end
        records = _load_records(conn, run_id, num_time_samples)
        loaded_records = size(records, 1)
        z_axis = loaded_records == 1 ? [z_end] : collect(range(0.0, z_end, length = loaded_records))
        return LoadedCase(
            run_id,
            String(case_key),
            _decode_meta_json(row_data.meta_json),
            records,
            z_axis,
            requested_records,
            num_time_samples,
            z_end,
        )
    end
end

function load_step_history(db::ExampleRunDB; run_id::AbstractString)
    return _with_db(db) do conn
        row_data = DBInterface.execute(conn,
            "SELECT event_count, dropped, capacity, step_index_blob, z_blob, step_size_blob, next_step_size_blob, error_blob "
            * "FROM ex_step_history WHERE run_id=?;",
            (String(run_id),)) do rows
            for row in rows
                return (
                    count = Int(row.event_count),
                    dropped = Int(row.dropped),
                    capacity = Int(row.capacity),
                    step_index_blob = Vector{UInt8}(row.step_index_blob),
                    z_blob = Vector{UInt8}(row.z_blob),
                    step_size_blob = Vector{UInt8}(row.step_size_blob),
                    next_step_size_blob = Vector{UInt8}(row.next_step_size_blob),
                    error_blob = Vector{UInt8}(row.error_blob),
                )
            end
            return nothing
        end
        row_data === nothing && return nothing
        count = row_data.count
        return Dict{String, Any}(
            "step_index" => _blob_to_array(row_data.step_index_blob, Int64)[1:count],
            "z" => _blob_to_array(row_data.z_blob, Float64)[1:count],
            "step_size" => _blob_to_array(row_data.step_size_blob, Float64)[1:count],
            "next_step_size" => _blob_to_array(row_data.next_step_size_blob, Float64)[1:count],
            "error" => _blob_to_array(row_data.error_blob, Float64)[1:count],
            "dropped" => row_data.dropped,
            "capacity" => row_data.capacity,
        )
    end
end

function _blob_from_array(values)
    return Vector{UInt8}(reinterpret(UInt8, values))
end

function save_step_history!(db::ExampleRunDB; run_id::AbstractString, step_history)
    step_history isa Dict || error("step_history metadata is missing or malformed")
    step_index = Int64.(collect(get(step_history, "step_index", Int[])))
    z = Float64.(collect(get(step_history, "z", Float64[])))
    step_size = Float64.(collect(get(step_history, "step_size", Float64[])))
    next_step_size = Float64.(collect(get(step_history, "next_step_size", Float64[])))
    error_values = Float64.(collect(get(step_history, "error", Float64[])))
    count = minimum((length(step_index), length(z), length(step_size), length(next_step_size), length(error_values)))
    step_index = step_index[1:count]
    z = z[1:count]
    step_size = step_size[1:count]
    next_step_size = next_step_size[1:count]
    error_values = error_values[1:count]

    _with_db(db) do conn
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS ex_step_history (
              run_id TEXT PRIMARY KEY,
              event_count INTEGER NOT NULL,
              dropped INTEGER NOT NULL,
              capacity INTEGER NOT NULL,
              step_index_blob BLOB NOT NULL,
              z_blob BLOB NOT NULL,
              step_size_blob BLOB NOT NULL,
              next_step_size_blob BLOB NOT NULL,
              error_blob BLOB NOT NULL
            );
            """)
        SQLite.execute(conn,
            "INSERT OR REPLACE INTO ex_step_history(run_id, event_count, dropped, capacity, step_index_blob, z_blob, step_size_blob, next_step_size_blob, error_blob) "
            * "VALUES(?,?,?,?,?,?,?,?,?);",
            (
                String(run_id),
                count,
                Int(get(step_history, "dropped", 0)),
                Int(get(step_history, "capacity", 0)),
                _blob_from_array(step_index),
                _blob_from_array(z),
                _blob_from_array(step_size),
                _blob_from_array(next_step_size),
                _blob_from_array(error_values),
            ))
    end
end
