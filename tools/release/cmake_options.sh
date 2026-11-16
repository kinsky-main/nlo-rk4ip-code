# Shared CMake option handling for the release scripts. Source, do not execute.
#
# NOTE: the option names this project reads are NOT prefixed. They were renamed
# from NLO_* in commit 2669906; a -D for a name the project never reads is
# silently ignored (it lands in the cache as UNINITIALIZED), so a stale prefix
# disables the setting instead of failing. nlo_configure_verified enforces that
# each option actually reached the project with the value we asked for.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/cmake_options.sh"
#   nlo_configure_verified "$BUILD_DIR" NAME=VALUE [NAME=VALUE ...]
#
# The source directory is $NLO_SOURCE_DIR, or the current directory. Arguments
# in the NLO_EXTRA_CMAKE_ARGS array are passed through unverified; that is for
# values CMake normalises, such as paths, where a string comparison would fail.

nlo_configure_verified() {
    local build_dir="$1"
    shift
    local options=("$@")
    local args=(-S "${NLO_SOURCE_DIR:-.}" -B "$build_dir"
                ${NLO_EXTRA_CMAKE_ARGS[@]+"${NLO_EXTRA_CMAKE_ARGS[@]}"})
    local option
    for option in "${options[@]}"; do
        args+=("-D${option}")
    done

    cmake "${args[@]}"

    # Verify only the options this script passes. A blanket scan for
    # UNINITIALIZED entries would also flag stale ones left in an existing
    # cache by earlier runs, which CMake never removes.
    local cache="${build_dir}/CMakeCache.txt"
    local problems=""
    local name want entry actual
    for option in "${options[@]}"; do
        name="${option%%=*}"
        want="${option#*=}"
        entry="$(grep -m1 -E "^${name}:" "$cache" || true)"
        if [[ -z "$entry" ]]; then
            problems+=$'\n  '"${name} : not present in ${cache}"
        elif [[ "$entry" == *":UNINITIALIZED="* ]]; then
            problems+=$'\n  '"${name} : ignored by the project (renamed or misspelled) -- ${entry}"
        else
            actual="${entry#*=}"
            if [[ "$actual" != "$want" ]]; then
                problems+=$'\n  '"${name} : expected ${want}, cache has ${actual}"
            fi
        fi
    done

    if [[ -n "$problems" ]]; then
        echo "CMake options did not take effect:${problems}" >&2
        return 1
    fi
}
