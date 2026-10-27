#!/usr/bin/env bash
# Clone upstream vLLM at the pinned commit, apply the PIC patch, install it.
# Usage: ./scripts/install-vllm.sh [build_dir]   (default: build/vllm)
set -euo pipefail

VLLM_SHA=a7fdfeef72323eb3db6f0620e4ea200290d0ca5a
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/patches/vllm-pic.patch"
DEST="${1:-$ROOT/build/vllm}"

if [ ! -d "$DEST/.git" ]; then
  git init -q "$DEST"
  git -C "$DEST" remote add origin https://github.com/vllm-project/vllm.git
  git -C "$DEST" fetch -q --depth 1 origin "$VLLM_SHA"
  git -C "$DEST" checkout -q FETCH_HEAD
fi

if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "PIC patch already applied"
else
  git -C "$DEST" apply --check "$PATCH"
  git -C "$DEST" apply "$PATCH"
fi

# The patch only touches Python and Triton, so the upstream prebuilt kernels work.
cd "$DEST"
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
