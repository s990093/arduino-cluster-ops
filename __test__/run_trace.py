import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcc_run import MCCRunner

def run_trace_test():
    print("Running 1D Convolution Trace Test...")
    
    # 1. Setup
    cu_file = Path(__file__).parent / "conv1d.cu"
    runner = MCCRunner(verbose=True)
    
    # 2. Compile
    asm_file = runner.compile(cu_file)
    if not asm_file:
        return
    
    # 3. Load Program
    program = runner.load_program(asm_file)
    if not program:
        return
        
    # 4. Connect
    if not runner.connect():
        print("Could not connect to hardware. Is ESP32 plugged in?")
        return

    # 5. Init VRAM
    # Map consistent with what we expect. 
    # Note: We need to see if the compiler respects these.
    # For now we write data to where we expect the kernel might look?
    # Actually, we don't know where the kernel looks for args without checking the compiler.
    # But let's assume standard layout or that we can infer it from usage.
    # RUN_COMPLETE_CONV used 0x1000, 0x2000...
    # MCC_RUN uses 0, 64, 128.
    # Let's write to BOTH low addresses and high addresses just in case?
    # No, that's messy.
    # Let's assume the kernel arguments are pointers passed in registers, 
    # and the simple runtime might not set them up dynamically?
    # If the compiler compiles 'int* input' to 'use register R1', and R1 assumes 0?
    
    # Let's try to set the Params at a known location.
    params_addr = 64 * 4 # 256
    
    vram = {
        "input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "kernel": [1, 0, 2], # Weights: k0=1, k1=0, k2=2
        "params": [1], # Start offset
        # Output will be written
    }
    
    # Custom memory map for this test
    mem_map = {
        "input": 0,
        "kernel": 64,  # Address 64 (0x40)
        "output": 128, # Address 128 (0x80)
        "params": 192  # Address 192 (0xC0)
    }
    
    print("Initializing VRAM...")
    runner.conn.send_command("reset", delay=0.2)
    runner.conn.read_lines()
    
    for name, data in vram.items():
        base = mem_map.get(name)
        if base is not None:
             for i, val in enumerate(data):
                 runner.conn.send_command(f"mem {base + i*4} {val}", delay=0.01)
                 
    # We also need to Initialize Output to 0 (optional but good for debug)
    for i in range(16):
        runner.conn.send_command(f"mem {mem_map['output'] + i*4} 0", delay=0.01)

    # 6. Run with Trace (Robust Mode)
    print("Executing with Trace...")
    
    # Trace on
    runner.conn.send_command("trace:stream", delay=0.2)
    runner.conn.read_lines() # Flush ack
    
    runner.conn.send_command("run")
    
    # Read loop
    print("Collecting trace data (5s)...")
    output = []
    start_time = time.time()
    while time.time() - start_time < 5.0:
        lines = runner.conn.read_lines()
        if lines:
            output.extend(lines)
            # Optional: Check for completion marker if known
        time.sleep(0.1)

    runner.conn.send_command("trace:off")
    
    # 7. Dump Trace to file
    trace_path = Path(__file__).parent / "trace_output.log"
    with open(trace_path, "w") as f:
        for line in output:
            f.write(line + "\n")
            
    print(f"Trace saved to {trace_path} ({len(output)} lines).")
    print("Trace snippet:")
    for line in output[:10]:
        print(line)

    # 8. Read Result
    res = runner.read_results(base_addr=mem_map["output"], count=8)
    print(f"Result: {res}")
    
    runner.cleanup()

if __name__ == "__main__":
    run_trace_test()
