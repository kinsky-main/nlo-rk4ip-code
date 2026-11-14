#!/usr/bin/env julia
#
# Write (or merge) julia/Artifacts.toml for a published nlolib release.
#
# An Artifacts.toml entry binds a git-tree-sha1 to a download URL and a tarball
# sha256, so it can only be written once the per-platform tarballs exist. That
# ordering is why this is a script and not part of the package: the release
# workflow runs it against the tarballs it just built and publishes the result
# as a release asset, and a human then re-runs it with --download against the
# published assets and commits the file. Both runs must agree byte for byte --
# that equality is the check that the committed bindings describe the assets
# users will actually download.
#
# The git tree hash of an unpacked tarball is not platform independent (the
# executable bit is not represented the same way everywhere), so each tarball's
# hash is computed on the platform that will consume it. Hence two modes:
#
#   bind    one tarball on its own platform -> a single-entry fragment
#   --merge several fragments               -> the combined Artifacts.toml
#
# Usage:
#   # on each build platform
#   julia julia/tools/bind_artifacts.jl --version 1.16.0 \
#       --toml frag-x86_64-linux-gnu.toml \
#       x86_64-linux-gnu=dist/nlolib-julia-1.16.0-x86_64-linux-gnu.tar.gz
#
#   # anywhere, from the fragments
#   julia julia/tools/bind_artifacts.jl --merge --toml julia/Artifacts.toml frag-*.toml
#
#   # after the release exists, to reproduce the committed file from the assets
#   julia julia/tools/bind_artifacts.jl --version 1.16.0 --download \
#       --toml frag.toml x86_64-linux-gnu
#
# Options:
#   --version V      release version; names the tag the download URLs point at
#   --repo O/N       GitHub repository hosting the release
#   --toml PATH      file to write (default: julia/Artifacts.toml)
#   --download       fetch the tarball from its release URL instead of using a
#                    local path
#   --install        also unpack the tree into the depot, so the artifact
#                    resolves without a download (used by the CI test job)
#   --keep           add to an existing file instead of replacing it
#   --merge          treat the positional arguments as fragments to combine

using Pkg
using Pkg.Artifacts
using Base.BinaryPlatforms
using Downloads
using SHA
using TOML

const DEFAULT_REPO = "kinsky-main/nlo-rk4ip-code"
const ARTIFACT_NAME = "nlolib"

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
    merge = false
    positional = String[]

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
        elseif arg == "--merge"
            merge = true
        elseif startswith(arg, "--")
            die("unknown option $arg")
        else
            push!(positional, arg)
        end
        i += 1
    end

    isempty(positional) && die("nothing to do: name a platform triplet or a fragment")
    merge || version !== nothing || die("--version is required")
    if toml === nothing
        toml = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))
    end
    return (; version, repo, toml, download, install, keep, merge, positional)
end

release_url(repo, version, triplet) =
    "https://github.com/$repo/releases/download/$version/nlolib-julia-$version-$triplet.tar.gz"

sha256_of(path) = open(io -> bytes2hex(SHA.sha256(io)), path, "r")

# The hash Artifacts.toml records is the git tree hash of the *unpacked*
# tarball, which is what Pkg recomputes after downloading. create_artifact
# unpacks into the depot and returns exactly that hash, so computing the hash
# and installing the artifact are the same operation; without --install the
# depot copy is removed again, leaving no state behind.
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

# Rebuild a Platform from the keys of an Artifacts.toml entry. Everything that
# is not artifact bookkeeping is a platform tag, which is how Pkg reads these
# entries back as well.
const ENTRY_KEYS = ("arch", "os", "git-tree-sha1", "download", "lazy")

function entry_platform(entry, source)
    haskey(entry, "arch") && haskey(entry, "os") ||
        die("entry in $source has no arch/os")
    tags = Dict{Symbol, String}(Symbol(k) => string(v) for (k, v) in entry if !(k in ENTRY_KEYS))
    return Platform(entry["arch"], entry["os"]; tags...)
end

function bind_from_tarball(opts, spec)
    triplet, tarball = let idx = findfirst('=', spec)
        idx === nothing ? (spec, nothing) :
            (spec[1:prevind(spec, idx)], spec[nextind(spec, idx):end])
    end
    platform = tryparse(Platform, triplet)
    platform === nothing && die("cannot parse platform triplet $triplet")

    url = release_url(opts.repo, opts.version, triplet)
    if opts.download || tarball === nothing
        tarball = joinpath(mktempdir(), basename(url))
        println("downloading $url")
        Downloads.download(url, tarball)
    end
    isfile(tarball) || die("no tarball at $tarball")

    sha256 = sha256_of(tarball)
    hash = tree_hash_of(tarball, opts.install)
    println("$triplet: git-tree-sha1=$(bytes2hex(hash.bytes)) sha256=$sha256")

    bind_artifact!(opts.toml, ARTIFACT_NAME, hash;
        platform = platform,
        download_info = [(url, sha256)],
        lazy = false,
        force = true)
end

function bind_from_fragment(opts, fragment)
    isfile(fragment) || die("no fragment at $fragment")
    parsed = TOML.parsefile(fragment)
    entries = get(parsed, ARTIFACT_NAME, nothing)
    entries === nothing && die("$fragment has no [[$ARTIFACT_NAME]] entry")
    for entry in entries
        platform = entry_platform(entry, fragment)
        hash = Base.SHA1(entry["git-tree-sha1"])
        downloads = [(d["url"], d["sha256"]) for d in get(entry, "download", ())]
        isempty(downloads) && die("entry for $(triplet(platform)) in $fragment has no download URL")
        println("$(triplet(platform)): git-tree-sha1=$(entry["git-tree-sha1"])")
        bind_artifact!(opts.toml, ARTIFACT_NAME, hash;
            platform = platform,
            download_info = downloads,
            lazy = false,
            force = true)
    end
end

function main(argv)
    opts = parse_args(argv)
    if !opts.keep && isfile(opts.toml)
        rm(opts.toml)
    end
    for spec in opts.positional
        opts.merge ? bind_from_fragment(opts, spec) : bind_from_tarball(opts, spec)
    end
    println("wrote $(opts.toml)")
    return 0
end

exit(main(ARGS))
