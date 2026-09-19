#!/usr/bin/env bash
# Build chitta-field with the correct Rust toolchain and system linker.
# Run from any directory: ./build.sh [cargo args...]
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
if [[ -f "$REPO/../scripts/build-env.sh" ]]; then
    source "$REPO/../scripts/build-env.sh"
    chitta_build_init || exit 1
    if [[ -n "${CARGO_TARGET_DIR:-}" && "$(realpath -m "$CARGO_TARGET_DIR")" != "$REPO"/* ]]; then
        echo "FAIL: CARGO_TARGET_DIR must be private to this checkout" >&2
        exit 2
    fi
    exec cargo "$@"
fi
unset CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER
if [ -z "${PYO3_PYTHON:-}" ] && [ -x /maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/bin/python3 ]; then
    export PYO3_PYTHON=/maps/projects/fernandezguerra/apps/opt/conda/envs/bioinfo/bin/python3
fi
export PATH="/usr/bin:$HOME/.rustup/toolchains/1.93.0-x86_64-unknown-linux-gnu/bin:$PATH"
exec cargo "$@"
