#!/usr/bin/env python3
"""
Compile Kernel Script

High-level interface for compiling CUDA-like C++ kernels to Micro-CUDA ISA.

Usage:
    python compile_kernel.py kernels/vector_add.cpp
    python compile_kernel.py kernels/vector_add.cpp --output my_kernel.asm
    python compile_kernel.py kernels/vector_add.cpp --llvm-ir
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_cuda_compiler import MCUDA_HEADER
from micro_cuda_compiler.target_config import get_target, list_targets, DEFAULT_TARGET
from esp32_tools.program_loader_v15 import InstructionV15

# ===== Configuration =====
CLANG_TARGET = "riscv32"
CLANG_OPT_LEVEL = "1"

def run_clang(input_file: Path, output_ll: Path, include_dirs: List[Path] = None) -> bool:
    """
    Run Clang to generate LLVM IR from C/C++ source
    
    Args:
        input_file: Input .cpp or .cu file
        output_ll: Output .ll file
        include_dirs: Additional include directories
    
    Returns:
        True if successful
    """
    if include_dirs is None:
        include_dirs = []
    
    # Always include mcuda.h directory
    include_dirs.append(MCUDA_HEADER.parent)
    
    cmd = [
        "clang",
        "-x", "c++",                    # Treat input as C++ (not CUDA)
        "-S",                           # Output assembly
        "-emit-llvm",                   # Emit LLVM IR
        f"-O{CLANG_OPT_LEVEL}",         # Optimization level
        f"--target={CLANG_TARGET}",     # 32-bit target
    ]
    
    # Add include directories
    for inc_dir in include_dirs:
        cmd.extend(["-I", str(inc_dir)])
    
    cmd.extend([
        str(input_file),
        "-o", str(output_ll)
    ])
    
    print(f"[Clang] Compiling {input_file.name}...")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Clang compilation failed:")
            print(result.stderr)
            return False
        
        print(f"[Clang] ✓ LLVM IR generated: {output_ll}")
        return True
        
    except FileNotFoundError:
        print("[ERROR] Clang not found! Please install LLVM/Clang:")
        print("  macOS: brew install llvm")
        print("  Linux: apt-get install clang")
        return False

def run_mcc(input_ll: Path, output_asm: Path, target_name: str = "default") -> bool:
    """
    Run Micro-CUDA Compiler backend (mcc.py)
    
    Args:
        input_ll: Input LLVM IR file
        output_asm: Output assembly file
        target_name: Target configuration name
    
    Returns:
        True if successful
    """
    mcc_script = Path(__file__).parent / "mcc.py"
    
    cmd = [
        sys.executable,
        str(mcc_script),
        str(input_ll),
        "--asm",
        "--target", target_name,
        "-o", str(output_asm)
    ]
    
    print(f"[MCC] Compiling IR to Micro-CUDA ISA...")
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] MCC compilation failed:")
        print(result.stderr)
        return False
    
    # Print MCC output
    if result.stdout:
        print(result.stdout)
    
    print(f"[MCC] ✓ Assembly generated: {output_asm}")
    return True

def compile_kernel(input_file: Path, 
                   output_file: Optional[Path] = None,
                   emit_llvm: bool = False,
                   target_name: str = "default",
                   verbose: bool = False) -> bool:
    """
    Complete compilation pipeline: C++ -> LLVM IR -> Micro-CUDA ASM
    
    Args:
        input_file: Input kernel file (.cpp or .cu)
        output_file: Output file (default: same name with .asm)
        emit_llvm: If True, stop at LLVM IR stage
        target_name: Target configuration name
        verbose: Enable verbose output
    
    Returns:
        True if compilation successful
    """
    # Get target configuration
    target = get_target(target_name)
    
    print("=" * 70)
    print("Micro-CUDA Kernel Compiler")
    print("=" * 70)
    print(f"Input:  {input_file}")
    
    # Validate input
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        return False
    
    # Determine output file
    if output_file is None:
        if emit_llvm:
            output_file = input_file.with_suffix('.ll')
        else:
            output_file = input_file.with_suffix('.asm')
    
    print(f"Output: {output_file}")
    print()
    
    # Display target configuration
    print("[Target] Hardware Configuration:")
    print(f"  Device:       {target.device_name}")
    print(f"  ISA Version:  {target.isa_version}")
    print(f"  Lanes:        {target.num_lanes}")
    print(f"  VRAM:         {target.vram_size // 1024} KB")
    print(f"  Registers:    R0-R{target.num_gpr-1}, F0-F{target.num_fpr-1}")
    print()
    
    # Step 1: C++ -> LLVM IR
    temp_ll = input_file.with_suffix('.ll')
    
    if not run_clang(input_file, temp_ll):
        return False
    
    if emit_llvm:
        # Rename to final output
        temp_ll.rename(output_file)
        print()
        print("=" * 70)
        print(f"✅ Compilation complete! LLVM IR: {output_file}")
        print("=" * 70)
        return True
    
    # Step 2: LLVM IR -> Micro-CUDA ASM
    if not run_mcc(temp_ll, output_file, target_name):
        # Cleanup
        if temp_ll.exists():
            temp_ll.unlink()
        return False
    
    # Cleanup temporary files
    if temp_ll.exists():
        temp_ll.unlink()
    
    print()
    print("=" * 70)
    print(f"✅ Compilation complete! Assembly: {output_file}")
    print("=" * 70)
    print()
    print("Assembly includes target configuration header")
    print()
    print("Next steps:")
    print(f"  1. Review assembly: cat {output_file}")
    print(f"  2. Run kernel: python run_kernel.py --demo")
    print()
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Micro-CUDA Kernel Compiler",
        epilog="Example: python compile_kernel.py kernels/vector_add.cpp"
    )
    
    parser.add_argument(
        "input",
        type=Path,
        nargs='?',
        help="Input kernel file (.cpp or .cu)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file (default: input name with .asm extension)"
    )
    
    parser.add_argument(
        "--llvm-ir",
        action="store_true",
        help="Stop at LLVM IR stage (output .ll file)"
    )
    
    parser.add_argument(
        "--target",
        default="default",
        help="Target configuration (default, esp32, esp32-psram, esp32s3)"
    )
    
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List available target configurations"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # List targets if requested
    if args.list_targets:
        list_targets()
        return 0
    
    # Require input file
    if not args.input:
        parser.print_help()
        return 1
    
    success = compile_kernel(
        args.input,
        args.output,
        emit_llvm=args.llvm_ir,
        target_name=args.target,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
