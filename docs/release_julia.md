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

Versions come from `project(nlolib VERSION ...)` in the root `CMakeLists.txt`,
and `julia/Project.toml` carries a copy of it. The two agree at every commit —
nothing advances the version automatically — so a tag can be cut from any
commit. The tag is the bare version with no `v` prefix, matching
`release-mltbx.yml`, and the workflow's `verify` job fails if the tag and the
copy disagree: the registry records the Julia version while the artifact
download URLs are built from the tag.

Advancing the version is a deliberate step, one command that rewrites
`CMakeLists.txt`, both literal copies and `sbom.cdx.json` together:

```bash
python tools/release/check_versions.py --bump minor
```

Commit that, then tag it:

```bash
git tag "$(python tools/release/read_version.py)"
git push origin "$(python tools/release/read_version.py)"
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
julia julia/tools/bind_artifacts.jl --version <version> --download \
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

The first version that can be registered is the first release whose tag
carries this workflow. `1.17.0` predates it — `release-julia.yml` is not in
that commit's tree, so no Julia tarballs exist for it and none can be made:
`workflow_dispatch` resolves the workflow file at the ref it is dispatched
against, and that ref has no such file. `1.17.0` is a MATLAB-only release
permanently, and the Julia package starts at the next bumped version.

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
  registration will not auto-merge. It is not rejected either: the pull
  request is labelled for a General maintainer, who merges it by hand after
  the usual waiting period for new packages. That is a one-time cost, and
  every later release passes AutoMerge normally.

  Registering `1.0.0` first does **not** avoid it. AutoMerge also checks that
  a new version follows the previous one, accepting only the next patch,
  minor or major, so the jump from `1.0.0` to the real version needs the same
  manual merge — and `1.0.0` would additionally need a tag and artifact
  tarballs of its own, which do not exist. Register at the real version and
  accept the one manual merge.
- The repository is public and GPL-2.0 licensed with a `LICENSE` file at the
  root, which satisfies the license check.
- AutoMerge installs and imports the package on its own runners. `NLOLib` does
  not open the library at precompile time — `load()` is explicit — so the import
  succeeds on platforms the artifact does not cover.

#### Version numbering

The version is frozen between releases and advances only when someone runs
`check_versions.py --bump`. **A registered Julia package depends on that, and
reintroducing a bump-on-commit hook would silently break the artifact
binding.** The reason is step 2: the binding commit records a
`git-tree-sha1` and a URL that both name the version its tag published. If a
hook bumped the version as that commit landed, the commit would declare one
version in `julia/Project.toml` while binding the artifact of another, and the
next tag would publish a package whose artifact URL points at the previous
release's assets. `Pkg` would install it and fail on a hash mismatch, or
worse, succeed with the wrong binary.

A registry version can also never be reused and must increase monotonically,
so a per-commit bump burns a version per commit and leaves the registered
history mostly gaps. Both problems are gone at the root rather than worked
around: no commit moves the version, so nothing has to be resynchronised at
release time. The invariant is enforced by the `test_version_copies_match`
and `test_sbom_pins_match` CTest cases, the pre-commit hook, and the `verify`
job of each release workflow.

## Files

- `.github/workflows/release-julia.yml` — build, test, publish
- `julia/tools/bind_artifacts.jl` — writes and merges `Artifacts.toml`
- `julia/tools/check_artifact_load.jl` — fails unless the library loaded from
  the artifact
- `julia/Artifacts.toml` — committed per release, generated by the above
