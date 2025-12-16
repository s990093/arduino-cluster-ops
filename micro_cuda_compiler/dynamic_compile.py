#!/usr/bin/env python3
"""
Dynamic Kernel Compiler API

Provides functions to compile CUDA kernels dynamically from Python code:
- Write kernel code inline in Python
- Compile to .asm and binary
- Load and execute on ESP32
"""

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from esp32_tools.program_loader_v15 import InstructionV15

class KernelCompiler:
    """
    Dynamic CUDA Kernel Compiler
    
    Usage:
        compiler = KernelCompiler()
        
        # Option 1: Compile from string
        kernel_code = '''
        #include "mcuda.h"
        
        __global__ void myKernel(int* A, int* B) {
            int idx = laneId();
            B[idx] = A[idx] * 2;
        }
        '''
        
        program = compiler.compile_from_string(
            kernel_code,
            output_asm="my_kernel.asm",
            target="esp32s3"
        )
        
        # Option 2: Compile from file
        program = compiler.compile_from_file(
            "kernels/conv1d.cu",
            output_asm="conv1d.asm"
        )
    """
    
    def __init__(self, compiler_path: str = "micro_cuda_compiler/compile_kernel.py"):
        self.compiler_path = Path(compiler_path)
        self.temp_files = []
    
    def compile_from_string(self,
                           kernel_code: str,
                           output_asm: Optional[str] = None,
                           target: str = "default",
                           verbose: bool = False) -> Tuple[Optional[List[InstructionV15]], Optional[str]]:
        """
        Compile kernel code from string
        
        Args:
            kernel_code: CUDA kernel source code
            output_asm: Output assembly file path (optional)
            target: Target configuration
            verbose: Print compilation output
        
        Returns:
            (program, asm_path) tuple
            - program: List of InstructionV15 objects (None if compilation failed)
            - asm_path: Path to generated assembly file
        """
        # Create temporary .cu file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(kernel_code)
            temp_cu = Path(f.name)
            self.temp_files.append(temp_cu)
        
        try:
            # Compile
            result = self.compile_from_file(
                temp_cu,
                output_asm=output_asm,
                target=target,
                verbose=verbose
            )
            return result
        finally:
            # Cleanup temp file
            if temp_cu.exists():
                temp_cu.unlink()
            if temp_cu in self.temp_files:
                self.temp_files.remove(temp_cu)
    
    def compile_from_file(self,
                         kernel_file: Path,
                         output_asm: Optional[str] = None,
                         target: str = "default",
                         verbose: bool = False) -> Tuple[Optional[List[InstructionV15]], Optional[str]]:
        """
        Compile kernel from .cu file
        
        Args:
            kernel_file: Path to .cu file
            output_asm: Output assembly file path (optional)
            target: Target configuration
            verbose: Print compilation output
        
        Returns:
            (program, asm_path) tuple
        """
        kernel_file = Path(kernel_file)
        
        if not kernel_file.exists():
            print(f"[ERROR] Kernel file not found: {kernel_file}")
            return None, None
        
        # Determine output assembly path
        if output_asm:
            asm_path = Path(output_asm)
        else:
            asm_path = kernel_file.with_suffix('.asm')
        
        # Build compilation command
        cmd = [
            sys.executable,
            str(self.compiler_path),
            str(kernel_file),
            "--target", target,
            "-o", str(asm_path)
        ]
        
        if verbose:
            print(f"[Compiler] Compiling {kernel_file.name}...")
            print(f"  Command: {' '.join(cmd)}")
        
        # Run compiler
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if verbose and result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ERROR] Compilation failed!")
            if result.stderr:
                print(result.stderr)
            return None, None
        
        if verbose:
            print(f"[Compiler] ✓ Assembly generated: {asm_path}")
        
        # TODO: Parse .asm and convert to InstructionV15 objects
        # For now, return None for program (user must provide manual assembly)
        
        return None, str(asm_path)
    
    def compile_and_load(self,
                        kernel_code: str,
                        manual_program: List[InstructionV15],
                        output_asm: Optional[str] = None,
                        target: str = "default") -> Tuple[List[InstructionV15], Optional[str]]:
        """
        Compile kernel and return program for loading
        
        This is a convenience function that:
        1. Compiles the kernel to .asm (for documentation)
        2. Returns the manual program (until compiler is complete)
        
        Args:
            kernel_code: CUDA kernel source
            manual_program: Hand-coded assembly equivalent
            output_asm: Output assembly file
            target: Target configuration
        
        Returns:
            (program, asm_path) tuple
        """
        # Compile to generate .asm for reference
        _, asm_path = self.compile_from_string(
            kernel_code,
            output_asm=output_asm,
            target=target,
            verbose=True
        )
        
        # Return manual program
        return manual_program, asm_path
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if temp_file.exists():
                temp_file.unlink()
        self.temp_files.clear()


# ===== Convenience Functions =====

def compile_kernel(kernel_code: str,
                   output_asm: Optional[str] = None,
                   output_binary: Optional[str] = None,
                   target: str = "default",
                   verbose: bool = True) -> Tuple[Optional[List[InstructionV15]], str]:
    """
    Quick compile function for inline kernels
    
    Example:
        kernel = '''
        #include "mcuda.h"
        
        __global__ void add(int* A, int* B, int* C) {
            int i = laneId();
            C[i] = A[i] + B[i];
        }
        '''
        
        program, asm_path = compile_kernel(kernel, output_asm="add.asm")
    
    Args:
        kernel_code: CUDA kernel source code
        output_asm: Output assembly file path
        output_binary: Output binary file path (TODO)
        target: Target configuration
        verbose: Print compilation messages
    
    Returns:
        (program, asm_path) tuple
    """
    compiler = KernelCompiler()
    
    try:
        program, asm_path = compiler.compile_from_string(
            kernel_code,
            output_asm=output_asm,
            target=target,
            verbose=verbose
        )
        
        # TODO: Support binary output
        if output_binary:
            print("[WARN] Binary output not yet implemented")
        
        return program, asm_path
        
    finally:
        compiler.cleanup()


def compile_kernel_file(kernel_file: str,
                        output_asm: Optional[str] = None,
                        output_binary: Optional[str] = None,
                        target: str = "default") -> Tuple[Optional[List[InstructionV15]], str]:
    """
    Compile kernel from .cu file
    
    Example:
        program, asm = compile_kernel_file(
            "kernels/conv1d.cu",
            output_asm="conv1d_compiled.asm",
            target="esp32s3"
        )
    """
    compiler = KernelCompiler()
    
    program, asm_path = compiler.compile_from_file(
        Path(kernel_file),
        output_asm=output_asm,
        target=target,
        verbose=True
    )
    
    # TODO: Support binary output
    if output_binary:
        print("[WARN] Binary output not yet implemented")
    
    return program, asm_path
