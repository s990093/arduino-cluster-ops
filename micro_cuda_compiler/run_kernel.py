#!/usr/bin/env python3
"""
Run Kernel Script

Execute compiled Micro-CUDA kernels on ESP32 CUDA VM.
Similar to test_enhanced_trace.py but specialized for compiled kernels.

Usage:
    python run_kernel.py kernels/vector_add.asm
    python run_kernel.py kernels/vector_add.asm --port /dev/cu.usbserial-XXX
    python run_kernel.py kernels/vector_add.asm --trace
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from esp32_tools import ESP32Connection
from esp32_tools.program_loader_v15 import InstructionV15

class KernelRunner:
    """
    Kernel execution framework
    
    Handles:
    - Loading compiled kernels
    - Setting up VRAM
    - Executing on ESP32
    - Retrieving and validating results
    """
    
    def __init__(self, port: str = "/dev/cu.usbserial-589A0095521", 
                 baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.conn = None
        self.trace_enabled = False
    
    def connect(self) -> bool:
        """Connect to ESP32"""
        try:
            print(f"[Connection] Connecting to {self.port}...")
            self.conn = ESP32Connection(self.port)
            print(f"[Connection] ✓ Connected at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"[Connection] ✗ Failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from ESP32"""
        if self.conn:
            self.conn.close()
            print("[Connection] Disconnected")
    
    def load_asm_kernel(self, asm_file: Path) -> List[InstructionV15]:
        """
        Load kernel from assembly file
        
        Note: Currently this is a placeholder. In the future, we'll parse
        the .asm file and convert to InstructionV15 objects.
        For now, we'll use manual instruction generation.
        """
        print(f"[Loader] Loading kernel from {asm_file}...")
        
        # TODO: Implement assembly parser
        # For now, return a hardcoded vector add kernel
        
        print("[Loader] ⚠ Assembly parsing not yet implemented")
        print("[Loader] Using built-in vector addition kernel")
        
        program = [
            InstructionV15.s2r(31, 2),          # R31 = lane ID
            InstructionV15.mov(0, 0),           # R0 = Base A
            InstructionV15.mov(1, 32),          # R1 = Base B  
            InstructionV15.mov(2, 64),          # R2 = Base C
            InstructionV15.ldl(10, 0),          # R10 = A[lane]
            InstructionV15.ldl(11, 1),          # R11 = B[lane]
            InstructionV15.iadd(12, 10, 11),    # R12 = A + B
            InstructionV15.stl(2, 12),          # C[lane] = result
            InstructionV15.exit_inst()
        ]
        
        print(f"[Loader] ✓ Loaded {len(program)} instructions")
        return program
    
    def setup_vram(self, data: Dict[str, List[int]]):
        """
        Initialize VRAM with test data
        
        Args:
            data: Dictionary mapping memory regions to data
                  Example: {"A": [1,2,3...], "B": [4,5,6...]}
        """
        print("[VRAM] Setting up device memory...")
        
        self.conn.send_command("reset")
        time.sleep(0.1)
        
        # Memory layout (4 bytes per element)
        # A: 0x0000-0x001F (8 elements * 4 bytes)
        # B: 0x0020-0x003F
        # C: 0x0040-0x005F (output)
        
        memory_map = {
            "A": 0,
            "B": 32,
            "C": 64
        }
        
        for region, values in data.items():
            if region not in memory_map:
                print(f"[VRAM] Warning: Unknown region '{region}'")
                continue
            
            base_addr = memory_map[region]
            
            for i, val in enumerate(values):
                addr = base_addr + i * 4
                self.conn.send_command(f"mem {addr} {val}")
            
            print(f"[VRAM] ✓ Region '{region}' written to 0x{base_addr:04X} ({len(values)} elements)")
    
    def load_program(self, program: List[InstructionV15]):
        """Load program to device"""
        print(f"[Program] Loading {len(program)} instructions...")
        
        for inst in program:
            self.conn.send_command(f"load {inst.to_hex()}")
        
        print(f"[Program] ✓ Program loaded")
    
    def execute(self, trace: bool = False) -> Dict:
        """
        Execute kernel on device
        
        Args:
            trace: Enable trace output
        
        Returns:
            Execution info dictionary
        """
        print("[Execute] Running kernel on 8-lane SIMD engine...")
        
        if trace:
            self.conn.send_command("trace:stream")
            time.sleep(0.05)
        
        self.conn.send_command("run", delay=0.3)
        output = self.conn.read_lines()
        
        # Parse execution info
        exec_info = {
            "status": "complete",
            "output": output,
            "cycles": None
        }
        
        # Try to extract cycle count
        for line in output:
            if "Cycles:" in line or "cycles" in line.lower():
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    exec_info["cycles"] = int(match.group(1))
        
        if trace:
            self.conn.send_command("trace:off")
        
        print(f"[Execute] ✓ Execution complete")
        if exec_info["cycles"]:
            print(f"[Execute] Cycles: {exec_info['cycles']}")
        
        return exec_info
    
    def read_results(self, base_addr: int = 64, count: int = 8) -> List[int]:
        """
        Read results from VRAM
        
        Args:
            base_addr: Base address to read from (default: 64 = region C)
            count: Number of elements to read
        
        Returns:
            List of result values
        """
        print(f"[Results] Reading from VRAM 0x{base_addr:04X}...")
        
        self.conn.send_command(f"dump {base_addr} {count}", delay=0.3)
        result_lines = self.conn.read_lines()
        
        # Debug: show raw output (uncomment for debugging)
        # print(f"[DEBUG] Raw dump output:")
        # for line in result_lines:
        #     print(f"  {line}")
        
        # Parse results - look for VRAM dump format
        # Expected format: "VRAM[addr]: value" or just numbers
        results = []
        import re
        
        for line in result_lines:
            # Skip empty lines and headers
            if not line.strip() or 'VRAM' in line and ':' not in line:
                continue
            
            # Try to extract value from "VRAM[addr]: value" format
            if 'VRAM' in line and ':' in line:
                match = re.search(r':\s*(\d+)', line)
                if match:
                    results.append(int(match.group(1)))
                    continue
            
            # Fallback: extract all numbers, take the last one (likely the value)
            numbers = re.findall(r'\b\d+\b', line)
            if len(numbers) >= 1:
                # If we have addr and value, take the value (last number)
                results.append(int(numbers[-1]))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for val in results:
            if val not in seen or len(unique_results) < count:
                unique_results.append(val)
                seen.add(val)
        
        print(f"[Results] ✓ Read {len(unique_results)} values: {unique_results[:count]}")
        return unique_results[:count]
    
    def verify_results(self, actual: List[int], expected: List[int]) -> bool:
        """
        Verify kernel results
        
        Args:
            actual: Actual results from device
            expected: Expected results
        
        Returns:
            True if results match
        """
        print("[Verify] Checking results...")
        
        if len(actual) != len(expected):
            print(f"[Verify] ✗ Length mismatch: {len(actual)} vs {len(expected)}")
            return False
        
        mismatches = []
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                mismatches.append((i, a, e))
        
        if mismatches:
            print(f"[Verify] ✗ Found {len(mismatches)} mismatch(es):")
            for idx, act, exp in mismatches[:5]:  # Show first 5
                print(f"  [Lane {idx}] Actual={act}, Expected={exp}")
            return False
        
        print(f"[Verify] ✅ All {len(actual)} results match!")
        return True

def run_vector_add_demo(port: str, trace: bool = False):
    """
    Demo: Run vector addition kernel
    
    This is the reference implementation showing the complete workflow.
    """
    runner = KernelRunner(port)
    
    if not runner.connect():
        return False
    
    try:
        print()
        print("=" * 70)
        print("Kernel Demo: Vector Addition (C = A + B)")
        print("=" * 70)
        print()
        
        # Step 1: Prepare test data
        A = [2, 3, 4, 5, 6, 7, 8, 9]
        B = [1, 2, 3, 4, 5, 6, 7, 8]
        expected = [a + b for a, b in zip(A, B)]
        
        print(f"Input A: {A}")
        print(f"Input B: {B}")
        print(f"Expected C: {expected}")
        print()
        
        # Step 2: Setup VRAM
        runner.setup_vram({"A": A, "B": B})
        print()
        
        # Step 3: Load kernel (hardcoded for now)
        program = runner.load_asm_kernel(Path("dummy.asm"))
        runner.load_program(program)
        print()
        
        # Step 4: Execute
        exec_info = runner.execute(trace=trace)
        print()
        
        # Step 5: Read results
        results = runner.read_results(base_addr=64, count=8)
        print()
        
        # Step 6: Verify
        success = runner.verify_results(results, expected)
        print()
        
        print("=" * 70)
        if success:
            print("✅ Kernel execution successful!")
        else:
            print("❌ Kernel execution failed!")
        print("=" * 70)
        
        return success
        
    finally:
        runner.disconnect()

def main():
    parser = argparse.ArgumentParser(
        description="Micro-CUDA Kernel Runner",
        epilog="Example: python run_kernel.py kernels/vector_add.asm"
    )
    
    parser.add_argument(
        "kernel",
        type=Path,
        nargs='?',
        help="Compiled kernel file (.asm)"
    )
    
    parser.add_argument(
        "--port",
        default="/dev/cu.usbserial-589A0095521",
        help="ESP32 serial port"
    )
    
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable execution trace"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in vector addition demo"
    )
    
    args = parser.parse_args()
    
    if args.demo or not args.kernel:
        # Run demo
        success = run_vector_add_demo(args.port, trace=args.trace)
        return 0 if success else 1
    
    # Run specified kernel
    print("[ERROR] Custom kernel execution not yet implemented")
    print("Use --demo to run the built-in demo")
    return 1

if __name__ == "__main__":
    sys.exit(main())
