#!/usr/bin/env python3
"""
MCC Run - Execute Micro-CUDA kernels end-to-end

Like nvcc + cuda-gdb, compile and run .cu files on ESP32.

Usage:
    python mcc_run.py kernel.cu
    python mcc_run.py kernel.cu --port /dev/ttyUSB0
    python mcc_run.py kernel.cu --vram-init input.json
    python mcc_run.py kernel.cu --target esp32s3 --trace
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from esp32_tools import ESP32Connection
from esp32_tools.program_loader_v15 import InstructionV15
from micro_cuda_compiler.dynamic_compile import compile_kernel_file
from micro_cuda_compiler.asm_parser import parse_asm_file

class MCCRunner:
    """
    End-to-end Micro-CUDA kernel runner
    
    Handles:
    1. Compilation (.cu -> .asm)
    2. Assembly parsing (.asm -> InstructionV15[])
    3. ESP32 connection
    4. VRAM initialization
    5. Program loading
    6. Execution
    7. Result retrieval
    """
    
    def __init__(self, 
                 port: str = "/dev/cu.usbserial-589A0095521",
                 target: str = "default",
                 verbose: bool = True):
        self.port = port
        self.target = target
        self.verbose = verbose
        self.conn = None
    
    def compile(self, cu_file: Path) -> Optional[Path]:
        """
        Step 1: Compile .cu to .asm
        
        Returns:
            Path to .asm file, or None if failed
        """
        if self.verbose:
            print("=" * 70)
            print("🔨 Step 1: Compiling Kernel")
            print("=" * 70)
        
        asm_file = cu_file.with_suffix('.asm')
        
        _, asm_path = compile_kernel_file(
            str(cu_file),
            output_asm=str(asm_file),
            target=self.target
        )
        
        if not asm_path or not Path(asm_path).exists():
            print(f"❌ Compilation failed!")
            return None
        
        if self.verbose:
            print(f"✅ Assembly generated: {asm_path}\n")
        
        return Path(asm_path)
    
    def load_program(self, asm_file: Path) -> Optional[List[InstructionV15]]:
        """
        Step 2: Parse .asm to instructions
        
        Returns:
            List of InstructionV15, or None if failed
        """
        if self.verbose:
            print("=" * 70)
            print("📜 Step 2: Parsing Assembly")
            print("=" * 70)
        
        try:
            program = parse_asm_file(asm_file)
            
            if not program:
                print("❌ No instructions parsed from assembly!")
                return None
            
            if self.verbose:
                print(f"✅ Loaded {len(program)} instructions\n")
            
            return program
            
        except Exception as e:
            print(f"❌ Assembly parsing failed: {e}")
            return None
    
    def connect(self) -> bool:
        """Step 3: Connect to ESP32"""
        if self.verbose:
            print("=" * 70)
            print("🔌 Step 3: Connecting to ESP32")
            print("=" * 70)
        
        try:
            self.conn = ESP32Connection(self.port)
            if self.verbose:
                print(f"✅ Connected to {self.port}\n")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def init_vram(self, vram_data: Optional[Dict[str, List[int]]] = None):
        """
        Step 4: Initialize VRAM
        
        Args:
            vram_data: Dictionary of {region_name: [values]}
                      Example: {"input": [1,2,3], "kernel": [2,3,4]}
        """
        if self.verbose:
            print("=" * 70)
            print("💾 Step 4: Initializing VRAM")
            print("=" * 70)
        
        # Reset VM
        self.conn.send_command("reset", delay=0.2)
        self.conn.read_lines()
        
        if not vram_data:
            if self.verbose:
                print("⚠️  No VRAM initialization data provided")
                print("   Using default test data\n")
            
            # Default test data for conv1d
            vram_data = {
                "input": list(range(1, 13)),  # [1..12]
                "kernel": [2, 3, 4],
            }
        
        # Memory layout
        memory_map = {
            "input": 0,      # 0x00
            "kernel": 64,    # 0x40
            "output": 128,   # 0x80
        }
        
        for region, values in vram_data.items():
            if region not in memory_map:
                print(f"⚠️  Unknown region '{region}', skipping")
                continue
            
            base_addr = memory_map[region]
            
            if self.verbose:
                print(f"Writing {region}: {values[:8]}{'...' if len(values) > 8 else ''}")
            
            for i, val in enumerate(values):
                self.conn.send_command(f"mem {base_addr + i * 4} {val}", delay=0.02)
        
        time.sleep(0.2)
        self.conn.read_lines()
        
        if self.verbose:
            print("✅ VRAM initialized\n")
    
    def execute(self, program: List[InstructionV15], trace: bool = False):
        """
        Step 5: Load and execute program
        
        Args:
            program: List of InstructionV15
            trace: Enable trace output
        """
        if self.verbose:
            print("=" * 70)
            print("⚡ Step 5: Executing Kernel")
            print("=" * 70)
        
        # Load program
        if self.verbose:
            print(f"Loading {len(program)} instructions...")
        
        for inst in program:
            self.conn.send_command(f"load {inst.to_hex()}", delay=0.02)
        
        time.sleep(0.2)
        self.conn.read_lines()
        
        if self.verbose:
            print("✅ Program loaded")
        
        # Execute
        if trace:
            self.conn.send_command("trace:stream", delay=0.1)
            self.conn.read_lines()
        
        if self.verbose:
            print("Running on 8-lane SIMD engine...")
        
        self.conn.send_command("run", delay=0.5 if trace else 0.3)
        output = self.conn.read_lines()
        
        if trace:
            self.conn.send_command("trace:off")
        
        if self.verbose:
            print("✅ Execution complete\n")
        
        return output
    
    def read_results(self, base_addr: int = 128, count: int = 8) -> List[int]:
        """
        Step 6: Read results from VRAM
        
        Args:
            base_addr: Base address to read from (default: 128 = output)
            count: Number of elements
        
        Returns:
            List of result values
        """
        if self.verbose:
            print("=" * 70)
            print("📊 Step 6: Reading Results")
            print("=" * 70)
        
        self.conn.send_command(f"dump {base_addr} {count}", delay=0.3)
        lines = self.conn.read_lines()
        
        # Parse results
        results = []
        import re
        for line in lines:
            if ':' in line:
                match = re.search(r':\s*(\d+)', line)
                if match:
                    results.append(int(match.group(1)))
        
        if self.verbose:
            print(f"Results: {results[:count]}\n")
        
        return results[:count]
    
    def cleanup(self):
        """Disconnect from ESP32"""
        if self.conn:
            self.conn.close()
            if self.verbose:
                print("🔌 Disconnected")
    
    def run(self, cu_file: Path, 
            vram_data: Optional[Dict] = None,
            trace: bool = False) -> bool:
        """
        Complete end-to-end execution
        
        Args:
            cu_file: Path to .cu kernel file
            vram_data: VRAM initialization data
            trace: Enable trace
        
        Returns:
            True if successful
        """
        print("\n" + "🚀 " * 35)
        print(f"MCC Run: {cu_file.name}")
        print("🚀 " * 35 + "\n")
        
        try:
            # Step 1: Compile
            asm_file = self.compile(cu_file)
            if not asm_file:
                return False
            
            # Step 2: Load program
            program = self.load_program(asm_file)
            if not program:
                return False
            
            # Step 3: Connect
            if not self.connect():
                return False
            
            # Step 4: Initialize VRAM
            self.init_vram(vram_data)
            
            # Step 5: Execute
            self.execute(program, trace=trace)
            
            # Step 6: Read results
            results = self.read_results()
            
            # Display final results
            print("=" * 70)
            print("✅ Execution Complete!")
            print("=" * 70)
            print(f"Output: {results}")
            print()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="MCC Run - Execute Micro-CUDA kernels end-to-end",
        epilog="Example: python mcc_run.py kernels/conv1d_manual.cu"
    )
    
    parser.add_argument(
        "kernel",
        type=Path,
        help="Kernel file (.cu)"
    )
    
    parser.add_argument(
        "--port",
        default="/dev/cu.usbserial-589A0095521",
        help="ESP32 serial port"
    )
    
    parser.add_argument(
        "--target",
        default="default",
        help="Target configuration (default, esp32, esp32-psram, esp32s3)"
    )
    
    parser.add_argument(
        "--vram-init",
        type=Path,
        help="VRAM initialization data (JSON file)"
    )
    
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Enable execution trace"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (less output)"
    )
    
    args = parser.parse_args()
    
    # Validate kernel file
    if not args.kernel.exists():
        print(f"❌ Kernel file not found: {args.kernel}")
        return 1
    
    # Load VRAM data if provided
    vram_data = None
    if args.vram_init:
        try:
            with open(args.vram_init, 'r') as f:
                vram_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load VRAM data: {e}")
            return 1
    
    # Run
    runner = MCCRunner(
        port=args.port,
        target=args.target,
        verbose=not args.quiet
    )
    
    success = runner.run(
        args.kernel,
        vram_data=vram_data,
        trace=args.trace
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
