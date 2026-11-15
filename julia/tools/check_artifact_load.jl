#!/usr/bin/env julia
#
# Fail unless NLOLib loaded its shared library out of the installed artifact.
#
# The release workflow runs this in a checkout with no build directory, so every
# development fallback path in julia/src/lib.jl is absent and the artifact is
# the only thing that can satisfy the load. Without this check a silent fall
# back to a stray library would still let the tests pass and would say nothing
# about whether the distribution works.
#
# Usage: julia --project=julia julia/tools/check_artifact_load.jl

using Artifacts
using NLOLib

const ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

isfile(ARTIFACTS_TOML) || error("no Artifacts.toml at $ARTIFACTS_TOML")

hash = artifact_hash("nlolib", ARTIFACTS_TOML)
hash === nothing && error("Artifacts.toml binds no nlolib artifact for $(Sys.MACHINE)")
root = artifact_path(hash)

load()
path = loaded_library_path()
println("loaded: ", path)
println("artifact: ", root)

startswith(path, root) ||
    error("the library came from $path, which is outside the artifact at $root")

println("the library came from the artifact")
