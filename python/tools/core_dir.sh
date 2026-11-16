# Locate the core nlolib checkout this binding builds against. Source, do not
# execute. Sets BINDING_ROOT (this repository) and CORE_DIR.
#
# Resolution order:
#   1. $NLOLIB_CORE_DIR, if set;
#   2. ./core, the checkout tools/fetch_core.sh and the workflows create at the
#      revision CORE_REF pins;
#   3. the parent directory, when this repository is mounted as a submodule of
#      the core repository.

BINDING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${NLOLIB_CORE_DIR:-}" ]]; then
    CORE_DIR="$(cd "$NLOLIB_CORE_DIR" && pwd)"
elif [[ -f "$BINDING_ROOT/core/src/nlolib.h" ]]; then
    CORE_DIR="$BINDING_ROOT/core"
elif [[ -f "$BINDING_ROOT/../src/nlolib.h" ]]; then
    CORE_DIR="$(cd "$BINDING_ROOT/.." && pwd)"
else
    echo "No core nlolib checkout found. Run tools/fetch_core.sh, or set NLOLIB_CORE_DIR." >&2
    return 1
fi
if [[ ! -f "$CORE_DIR/src/nlolib.h" ]]; then
    echo "$CORE_DIR is not a core nlolib checkout (no src/nlolib.h)." >&2
    return 1
fi
