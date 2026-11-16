#!/usr/bin/env bash
# Check out the core nlolib repository into ./core at the revision CORE_REF
# pins. The release and CI workflows do the same with actions/checkout; this is
# the local equivalent for a standalone clone of this repository. Not needed
# when this repository is mounted as a submodule of the core repository.
set -euo pipefail

BINDING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_REPO="${NLOLIB_CORE_REPO:-https://github.com/kinsky-main/nlo-rk4ip-code.git}"
CORE_REF="$(tr -d '[:space:]' < "$BINDING_ROOT/CORE_REF")"

if [[ ! -d "$BINDING_ROOT/core/.git" ]]; then
    git init -q "$BINDING_ROOT/core"
    git -C "$BINDING_ROOT/core" remote add origin "$CORE_REPO"
fi
git -C "$BINDING_ROOT/core" fetch -q --depth 1 origin "$CORE_REF"
git -C "$BINDING_ROOT/core" checkout -q --detach FETCH_HEAD
echo "core at $(git -C "$BINDING_ROOT/core" rev-parse HEAD)"
