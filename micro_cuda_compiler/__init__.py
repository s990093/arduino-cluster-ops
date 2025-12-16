"""
Micro-CUDA Compiler Package

A compiler toolchain for compiling CUDA-like C/C++ code to Micro-CUDA ISA v1.5.

Components:
- mcc.py: LLVM IR to Micro-CUDA ISA backend
- compile_kernel.py: High-level compilation interface
- run_kernel.py: Kernel execution and testing framework

Author: Micro-CUDA Project
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Micro-CUDA Project"

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
MCUDA_HEADER = PROJECT_ROOT / "mcuda.h"
KERNELS_DIR = PROJECT_ROOT / "kernels"
DOCS_DIR = PROJECT_ROOT.parent / "docs"

# Import main functions (temporarily commented until implemented)
from .compile_kernel import compile_kernel
# from .run_kernel import run_kernel  # TODO: implement run_kernel function

__all__ = [
    "compile_kernel",
    # "run_kernel",
    "PROJECT_ROOT",
    "MCUDA_HEADER",
    "KERNELS_DIR",
]
