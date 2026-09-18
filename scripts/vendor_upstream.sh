#!/usr/bin/env bash
# Reproduce vendored/exllamav3 from the pinned upstream clone.
# Usage: scripts/vendor_upstream.sh [--clone-dir DIR]
# Stdlib only. Idempotent: re-running produces a byte-identical tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLONE_DIR="${1:-/tmp/upstream-exllamav3}"
[[ "${1:-}" == "--clone-dir" ]] && CLONE_DIR="${2:?}"

EXL3_REF="v1.5.0"
EXL3_COMMIT="0740edc2da569fb99174023c1d2988b1e98cb41e"
UPSTREAM_URL="https://github.com/turboderp-org/exllamav3"

if [[ ! -d "$CLONE_DIR/.git" ]]; then
  echo "cloning $UPSTREAM_URL @ $EXL3_REF -> $CLONE_DIR"
  git clone --depth 1 --branch "$EXL3_REF" "$UPSTREAM_URL" "$CLONE_DIR"
fi

PINNED="$(git -C "$CLONE_DIR" rev-parse HEAD)"
if [[ "$PINNED" != "$EXL3_COMMIT"* ]]; then
  echo "ERROR: clone is at $PINNED, expected $EXL3_COMMIT" >&2
  exit 1
fi

EXT="$CLONE_DIR/exllamav3/exllamav3_ext"
DEST="$REPO_ROOT/vendored/exllamav3"
rm -rf "$DEST/ext"
mkdir -p "$DEST/ext"

# quant/: full EXL3 kernel tree
cp -r "$EXT/quant" "$DEST/ext/quant"

# core kernel units required by the quant tree
for f in hgemm.cu hgemm.cuh hgemm_f16acc.cu \
         graph.cu graph.cuh routing.cu routing.cuh \
         activation.cu activation.cuh activation_kernels.cuh \
         compat.cuh ptx.cuh util.cuh util.h reduction.cuh \
         histogram.cu histogram.cuh; do
  cp "$EXT/$f" "$DEST/ext/$f"
done

cp "$CLONE_DIR/LICENSE" "$DEST/LICENSE"
echo "vendored exllamav3 $EXL3_REF ($PINNED) -> $DEST"
