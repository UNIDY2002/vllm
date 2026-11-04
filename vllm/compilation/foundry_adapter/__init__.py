# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM lifecycle integration for the optional Foundry CUDA backend."""

from .config import CUDAGraphExtensionMode, get_graph_extension_mode
from .hooks import install_hooks

__all__ = [
    "CUDAGraphExtensionMode",
    "get_graph_extension_mode",
    "install_hooks",
]
