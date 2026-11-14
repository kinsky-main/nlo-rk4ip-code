# Julia Release Guide

`NLOLib` is distributed as a registered Julia package whose native library comes
from a [Julia artifact](https://pkgdocs.julialang.org/v1/artifacts/). The
package itself is pure Julia; the per-platform tarballs holding `nlolib` and its
runtime dependencies are built by
[`.github/workflows/release-julia.yml`](../.github/workflows/release-julia.yml)
and attached to the same GitHub release as the MATLAB toolbox.

The Vulkan + VkFFT + statically linked SQLite toolchain is why the binary is
built here rather than through Yggdrasil/BinaryBuilder: the release matrix
already builds all of it on the images we support.

## Installing

Once the package is registered:

```julia
using Pkg
Pkg.add("NLOLib")
```

`Pkg` downloads the artifact for the host platform, so nothing needs to be
compiled and no Vulkan SDK is required. On Linux the Vulkan **loader**
(`libvulkan.so.1`, from `libvulkan1` or a GPU driver package) must be present,
because it is a normal shared-library dependency and deliberately not bundled —
it belongs to the system. Windows delay-loads `vulkan-1.dll`, so it is optional
there. Without a Vulkan driver the CPU backend runs unchanged.

### From a build tree

A built checkout keeps working with no artifact at all:

```bash
cmake --build build --config Release --target julia_stage
```

```julia
using Pkg
Pkg.activate("build/julia_package")
using NLOLib
NLOLib.load()
```

`julia/src/lib.jl` resolves the library in this order:

1. an explicit path passed to `NLOLib.load(path)`
2. `ENV["NLOLIB_LIBRARY"]`
3. the installed `nlolib` artifact, when `julia/Artifacts.toml` binds one for
   this platform and it is present in the depot
4. the development paths — `<package>/lib`, `../python[/Debug|/Release]`,
   `../build/julia_package/lib`

`julia_stage` does not copy `Artifacts.toml` into the staged package, so the
staged tree always loads the library it was just built with. If you work
directly out of `julia/` in a checkout that also has the artifact installed, set
`NLOLIB_LIBRARY` to pick the build you mean.

## Releasing

### 1. Tag

Versions come from `project(nlolib VERSION ...)` in the root `CMakeLists.txt`.
The tag is the bare version with no `v` prefix, matching `release-mltbx.yml`,
and `julia/Project.toml` must already declare the same version — the workflow's
`verify` job fails on either mismatch.

```bash
git tag 1.16.0
git push origin 1.16.0
```

The workflow then, per platform (`ubuntu-24.04` and `windows-2025`, pinned):
builds with `ENABLE_VULKAN_BACKEND=ON ENABLE_VKFFT=ON
SQLITE_USE_FETCHCONTENT=ON`, stages `julia_stage`, packs the staged `lib/` as
`nlolib-julia-<version>-<triplet>.tar.gz`, and computes that tarball's
`git-tree-sha1` and `sha256`. A separate matrix of jobs then installs the
artifact into a fresh checkout that has no build directory and runs
`Pkg.test()` against it, so the tests can only pass if the distribution works.
Finally the tarballs, a `SHA256SUMS`, and a generated `Artifacts.toml` are
uploaded to the release.

`workflow_dispatch` runs everything except the upload.

### 2. Bind the artifact

An `Artifacts.toml` entry binds a `git-tree-sha1` to a download URL and a
tarball hash, so **it cannot be written before the tarballs are published**.
That ordering is unavoidable; what is a choice is who commits the result. This
repository uses a two-step flow with a human in the middle rather than letting
the workflow push to `main`:

1. the tagged run publishes `Artifacts.toml` as a release asset and prints it in
   the job summary;
2. you reproduce it locally from the published assets and commit it.

```bash
julia julia/tools/bind_artifacts.jl --version 1.16.0 --download \
    --toml julia/Artifacts.toml \
    x86_64-linux-gnu x86_64-w64-mingw32
```

Then diff it against the asset the workflow published; they must be identical.
Commit `julia/Artifacts.toml` through a normal pull request.

`--download` matters: gzip records a timestamp, so rebuilding a tarball from the
same tree yields a different `sha256` even though the `git-tree-sha1` is
unchanged. The hash that has to be recorded is the one of the asset users will
actually fetch, which is the published file.

The alternative — generating the file in the workflow and committing it back
from CI — removes this step but gives the release workflow write access to the
default branch, and the thing it would push is exactly the thing that decides
what every user downloads. A reviewable commit is worth the extra step.

Two details the script exists to get right:

- The tree hash is the hash of the *unpacked* tarball, which is what `Pkg`
  recomputes after downloading. `create_artifact` performs that unpack, so
  computing the hash and installing the artifact are one operation.
- That hash is **not** platform independent (the executable bit is not
  represented identically everywhere), so each platform's hash is computed on
  the platform that consumes it. The build jobs emit single-entry fragments and
  the `bundle` job only merges them (`--merge`). The test jobs recompute the
  hash on a different machine and fail if it differs from the published
  fragment.

### 3. Register in the General registry

Nothing in this repository submits to General; the first registration is
deliberately manual.

1. Install the [JuliaRegistrator](https://github.com/JuliaRegistries/Registrator)
   GitHub App on `kinsky-main/nlo-rk4ip-code`.
2. Comment on the commit that added `julia/Artifacts.toml`:

   ```
   @JuliaRegistrator register subdir=julia
   ```

   `subdir=julia` is required: the package is not at the repository root.
3. Registrator opens a pull request against `JuliaRegistries/General`. Watch its
   AutoMerge checks and respond there.

Later versions repeat steps 1–2 of this guide and then the same comment on the
new binding commit.

TagBot is deliberately **not** installed. It creates `v`-prefixed tags from
merged registry pull requests, and this repository's release workflows trigger
on bare version tags that a human pushes; TagBot's tags would either do nothing
or double-publish.

#### What AutoMerge will say

- `julia/Project.toml` currently has `[compat]` for `julia`, `Artifacts` and
  `Libdl`. AutoMerge requires a compat entry for every dependency and an upper
  bound on each non-stdlib one; there are no non-stdlib dependencies, so this is
  already sufficient.
- **A new package must be registered at `0.0.1`, `0.1.0` or `1.0.0` to
  auto-merge.** The project version is well past that, so the first
  registration will not auto-merge; it needs a General maintainer to merge it by
  hand after the waiting period, or the package has to be first registered at
  `1.0.0` and moved to the real version afterwards. Decide this before step 3.
- The repository is public and GPL-2.0 licensed with a `LICENSE` file at the
  root, which satisfies the license check.
- AutoMerge installs and imports the package on its own runners. `NLOLib` does
  not open the library at precompile time — `load()` is explicit — so the import
  succeeds on platforms the artifact does not cover.

#### Version numbering

Worth settling before the first registration: `.git/hooks/pre-commit`, generated
from `cmake/pre-commit.in`, bumps the CMake **minor** version on every commit to
`main`. A registered package cannot re-register or reuse a version and must
increase monotonically, so under that scheme every merged commit burns a minor
version that can never be published, and the registered history is mostly gaps.
It is legal SemVer and General will accept it, but it also means the
artifact-binding commit from step 2 immediately moves the CMake version past the
version it just bound, so `julia/Project.toml` and `CMakeLists.txt` cannot both
stay in step. Bumping at release time instead would remove both problems.

## Files

- `.github/workflows/release-julia.yml` — build, test, publish
- `julia/tools/bind_artifacts.jl` — writes and merges `Artifacts.toml`
- `julia/tools/check_artifact_load.jl` — fails unless the library loaded from
  the artifact
- `julia/Artifacts.toml` — committed per release, generated by the above
