#!/usr/bin/env julia
#
# Write julia/Artifacts.toml for a published nlolib release.
#
# An Artifacts.toml entry binds a git-tree-sha1 to a download URL and a tarball
# hash, so it can only be written once the per-platform tarballs exist. The
# release workflow (.github/workflows/release-julia.yml) runs this against the
# tarballs it just built and publishes the result as a release asset for
# review; a human then runs it again with --download, which recomputes every
# hash from the published assets, and commits the file. Both runs must produce
# byte-identical output -- that equality is the check that the committed
# bindings describe the assets users will actually download.
#
# Usage:
#   julia julia/tools/bind_artifacts.jl --version 1.15.0 \
#       x86_64-linux-gnu=dist/nlolib-julia-1.15.0-x86_64-linux-gnu.tar.gz \
#       x86_64-w64-mingw32=dist/nlolib-julia-1.15.0-x86_64-w64-mingw32.tar.gz
#
#   julia julia/tools/bind_artifacts.jl --version 1.15.0 --download \
#       x86_64-linux-gnu x86_64-w64-mingw32
#
# Options:
#   --version V      release version; also names the tag the URLs point at
#   --repo O/N       GitHub repository hosting the release
#   --toml PATH      Artifacts.toml to write (default: julia/Artifacts.toml)
#   --download       fetch each tarball from its release URL instead of using
#                    a local path
#   --install        unpack each tree into the depot as well, so the artifact
#                    resolves without a download (used by the CI test job)
#   --keep           add to an existing Artifacts.toml instead of replacing it

using Pkg
using Pkg.Artifacts
using Base.BinaryPlatforms
using Downloads
using SHA

const DEFAULT_REPO = "kinsky-main/nlo-rk4ip-code"

struct Entry
    triplet::String
    platform::Platform
    tarball::Union{Nothing, String}
end

function die(msg)
    println(stderr, "bind_artifacts: ", msg)
    exit(1)
end

function parse_args(argv)
    version = nothing
    repo = DEFAULT_REPO
    toml = nothing
    download = false
    install = false
    keep = false
    entries = Entry[]

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--version"
            version = argv[i += 1]
        elseif arg == "--repo"
            repo = argv[i += 1]
        elseif arg == "--toml"
            toml = argv[i += 1]
        elseif arg == "--download"
            download = true
        elseif arg == "--install"
            install = true
        elseif arg == "--keep"
            keep = true
        elseif startswith(arg, "--")
            die("unknown option $arg")
        else
            triplet, _, tarball = partition_entry(arg)
            platform = tryparse(Platform, triplet)
            platform === nothing && die("cannot parse platform triplet $triplet")
            push!(entries, Entry(triplet, platform, isempty(tarball) ? nothing : tarball))
        end
        i += 1
    end

    version === nothing && die("--version is required")
    isempty(entries) && die("name at least one platform triplet")
    if toml === nothing
        toml = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
    end
    return (; version, repo, toml, download, install, keep, entries)
end

function partition_entry(arg)
    idx = findfirst('=', arg)
    idx === nothing && return (arg, "=", "")
    return (arg[1:prevind(arg, idx)], "=", arg[nextind(arg, idx):end])
end

release_url(repo, version, triplet) =
    "https://github.com/$repo/releases/download/$version/nlolib-julia-$version-$triplet.tar.gz"

sha256_of(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

# The hash recorded in Artifacts.toml is the git tree hash of the *unpacked*
# tarball, which is what Pkg recomputes after downloading. create_artifact
# unpacks into the depot and returns exactly that hash, so computing it and
# installing it are the same operation; with install=false the depot copy is
# removed again so a plain bind run leaves no state behind.
function tree_hash_of(tarball, install)
    hash = create_artifact() do dir
        Pkg.PlatformEngines.unpack(tarball, dir)
    end
    if !install
        try
            rm(artifact_path(hash); recursive = true, force = true)
        catch err
            @warn "could not remove the temporary artifact tree" hash exception = err
        end
    end
    return hash
end

function main(argv)
    opts = parse_args(argv)
    if !opts.keep && isfile(opts.toml)
        rm(opts.toml)
    end

    workdir = mktempdir()
    for entry in opts.entries
        url = release_url(opts.repo, opts.version, entry.triplet)
        tarball = entry.tarball
        if opts.download || tarball === nothing
            tarball = joinpath(workdir, basename(url))
            println("downloading $url")
            Downloads.download(url, tarball)
        end
        isfile(tarball) || die("no tarball at $tarball")

        sha256 = sha256_of(tarball)
        hash = tree_hash_of(tarball, opts.install)
        println("$(entry.triplet): git-tree-sha1=$(bytes2hex(hash.bytes)) sha256=$sha256")

        bind_artifact!(opts.toml, "nlolib", hash;
            platform = entry.platform,
            download_info = [(url, sha256)],
            lazy = false,
            force = true)
    end

    println("wrote $(opts.toml)")
    return 0
end

exit(main(ARGS))
