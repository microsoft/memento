#!/bin/bash
# ============================================================
# install_overlay.sh - Install custom vLLM Python changes
# on top of a pre-installed vLLM wheel (no compilation needed)
#
# Strategy:
#   1. Keep the stock vLLM installation (with compiled C++/CUDA .so files)
#   2. Download the matching stock vLLM wheel for reference
#   3. rsync all .py files from our fork on top
#   4. Restore .so-interface files from stock (to match compiled extensions)
#
# The key insight: our fork's upstream base is newer than the stock image's
# vLLM. Files like flash_attn_interface.py and _custom_ops.py call compiled
# .so extensions with different argument counts than the stock .so expects.
# We overlay ALL Python files, then surgically restore these .so-facing files.
#
# Usage:
#   bash install_overlay.sh
#
# Requirements:
#   - Stock vLLM (>=0.13.0) must already be installed (pip install vllm)
#   - rsync must be available (script will install it if missing)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${SCRIPT_DIR}/vllm"

echo "=========================================="
echo "vLLM Python Overlay Installer (v2)"
echo "=========================================="

# 1. Ensure rsync is available
if ! command -v rsync &>/dev/null; then
    echo "Installing rsync..."
    apt-get update -qq && apt-get install -y -qq rsync
fi

# 2. Find the installed vllm package location
# IMPORTANT: Use sys.path filtering to skip the local vllm/ directory
# (when running from the repo root, python would otherwise find the local
# package first, making Source == Target — a no-op rsync).
VLLM_SITE=$(python3 -c "
import sys, os
# Remove cwd and empty-string entries so we find the *installed* package
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.getcwd()]
import vllm
print(os.path.dirname(vllm.__file__))
")
SITE_PACKAGES=$(dirname "$VLLM_SITE")

echo "Source:      ${VLLM_SRC}"
echo "Target:      ${VLLM_SITE}"

# Sanity check: Source and Target must differ
if [[ "$(realpath "$VLLM_SRC")" == "$(realpath "$VLLM_SITE")" ]]; then
    echo "ERROR: Source and Target resolve to the same directory!"
    echo "       This means python is importing the local vllm/ instead of the installed one."
    echo "       Check your PYTHONPATH / working directory."
    exit 1
fi
echo ""

STOCK_VERSION=$(python3 -c "
import sys, os
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.getcwd()]
import vllm
print(vllm.__version__)
" 2>/dev/null || echo "unknown")
echo "Stock vLLM version: ${STOCK_VERSION}"

# 3. Count .so files before overlay
SO_BEFORE=$(find "$VLLM_SITE" -name '*.so' | wc -l)
echo "Compiled extensions (.so): ${SO_BEFORE}"
echo ""

# 4. Download stock wheel and extract .py files for reference
echo "--- Downloading stock vLLM wheel for reference ---"
STOCK_DIR="/tmp/vllm_stock"
WHEEL_DIR="/tmp/vllm_wheel"
if [[ ! -d "$STOCK_DIR/vllm" ]]; then
    mkdir -p "$WHEEL_DIR"
    # Extract major.minor version and find matching release on PyPI
    # Stock version is like "0.13.1.dev0+g72506c983" but PyPI only has "0.13.0"
    STOCK_MAJOR_MINOR=$(echo "$STOCK_VERSION" | grep -oP '^\d+\.\d+' || echo "0.13")
    # Try exact version first, then fall back to major.minor.0
    STOCK_PIP_VERSION="${STOCK_MAJOR_MINOR}.0"
    echo "  Downloading vllm==${STOCK_PIP_VERSION} wheel..."
    pip download "vllm==${STOCK_PIP_VERSION}" --no-deps -d "$WHEEL_DIR" 2>&1 | tail -2
    if [[ ! -f "${WHEEL_DIR}"/vllm-*.whl ]]; then
        echo "  ERROR: Failed to download vllm==${STOCK_PIP_VERSION}. Trying latest ${STOCK_MAJOR_MINOR}.x..."
        pip download "vllm>=${STOCK_MAJOR_MINOR}.0,<${STOCK_MAJOR_MINOR}.99" --no-deps -d "$WHEEL_DIR" 2>&1 | tail -2
    fi
    mkdir -p "$STOCK_DIR"
    cd /tmp && unzip -oq "${WHEEL_DIR}"/vllm-*.whl '*.py' -d "$STOCK_DIR"
    echo "  Stock wheel extracted to ${STOCK_DIR}"
else
    echo "  Stock wheel already cached at ${STOCK_DIR}"
fi
echo ""

# 5. rsync all .py files from fork on top of stock install
echo "--- Syncing ALL Python files from fork ---"
rsync -a --include='*.py' --include='*/' --exclude='*' "${VLLM_SRC}/" "${VLLM_SITE}/" 2>&1 | tail -5
echo ""

# 6. Restore .so-interface files from stock (critical for .so compatibility)
echo "--- Restoring .so-interface files from stock ---"
# These files call compiled C++/CUDA extensions with specific argument counts
# that differ between our fork's upstream and the stock image:
#   - flash_attn_interface.py: varlen_fwd() arg count (num_splits)
#   - _custom_ops.py: _C.abi3.so function signatures
SO_INTERFACE_FILES=(
    "vllm_flash_attn/flash_attn_interface.py"
    "_custom_ops.py"
)
for relpath in "${SO_INTERFACE_FILES[@]}"; do
    stock_file="${STOCK_DIR}/vllm/${relpath}"
    target_file="${VLLM_SITE}/${relpath}"
    if [[ -f "$stock_file" ]]; then
        cp "$stock_file" "$target_file"
        echo "  Restored: ${relpath}"
    else
        echo "  WARNING: stock file not found: ${relpath}"
    fi
done
echo ""

# 7. Verify .so files are intact
SO_AFTER=$(find "$VLLM_SITE" -name '*.so' | wc -l)
echo "Post-overlay .so files: ${SO_AFTER} (was ${SO_BEFORE})"
if [[ "$SO_BEFORE" -ne "$SO_AFTER" ]]; then
    echo "WARNING: .so file count changed!"
fi

# 8. Verify
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
python3 -c "
import sys, os
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.getcwd()]
import vllm
print(f'vLLM version: {vllm.__version__}')
print(f'vLLM location: {vllm.__file__}')

from vllm.config.block_masking import BlockMaskingConfig
print('BlockMaskingConfig: OK')

from vllm.v1.core.block_masking.tracker import BlockMaskingState
print('BlockMaskingState: OK')

from vllm.v1.core.block_masking.processor import BlockMaskingProcessor
print('BlockMaskingProcessor: OK')

from vllm.engine.arg_utils import EngineArgs
print('EngineArgs: OK')

from vllm.entrypoints.llm import LLM
print('LLM entrypoint: OK')

from vllm._custom_ops import *
print('_custom_ops: OK')

from vllm.v1.sample.logits_processor.block_length_cap import BlockLengthCapLogitsProcessor
print('BlockLengthCapLogitsProcessor: OK')
"

echo ""
echo "=========================================="
echo "Overlay install complete!"
echo "=========================================="
