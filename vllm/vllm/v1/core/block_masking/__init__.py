# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block Masking module for Memento-style inference.

This module provides automatic block compaction during generation,
using vLLM's token span removal infrastructure.
"""

from .tracker import BlockMaskingState, BlockInfo
from .processor import BlockMaskingProcessor

__all__ = [
    "BlockMaskingState",
    "BlockInfo",
    "BlockMaskingProcessor",
]
