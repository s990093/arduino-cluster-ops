# ESP32 CUDA VM (Dual-Core) User Guide

This guide provides comprehensive documentation for the **Dual-Core ESP32 Micro-CUDA Virtual Machine**, including architecture overview, serial command interface (CLI), ISA reference, and Python SDK usage.

## 1. System Architecture

The VM utilizes the ESP32's dual-core architecture to function as a parallel GPU simulator:

- **Core 0 (Front-End / Control Processor)**:

  - Handles Serial Communication (CLI).
  - Fetches and Decodes Instructions.
  - Manages Program Counter (PC) and Control Flow (Branching).
  - Dispatches instructions to Core 1 via FreeRTOS Queue.
  - **Latency**: Asynchronous dispatch.

- **Core 1 (Back-End / SIMD Engine)**:
  - Executes Computations across **8 SIMD Lanes**.
  - Manages Vector Register File (R[32], F[32], P[8] per lane).
  - Handles Shared Memory (VRAM) Access.
  - Generates Execution Trace.

## 2. Serial Command Interface (CLI)

Connect to the ESP32 at `115200` baud. Commands are terminated by newline (`\n`).

| Command        | Arguments           | Description                                                                                |
| :------------- | :------------------ | :----------------------------------------------------------------------------------------- |
| `reset`        | None                | Resets VM state and registers (Hard Reset). VRAM is cleared.                               |
| `load`         | `<hex_instruction>` | Loads a 32-bit instruction (Hex) into program memory at current PC auto-increment.         |
| `mem`          | `<addr> <val>`      | Writes a 32-bit integer `val` to VRAM at byte address `addr`. Safe to use when stopped.    |
| `dump`         | `<addr> <count>`    | Dumps `count` words (32-bit) from VRAM starting at `addr`.                                 |
| `run`          | None                | Starts execution from PC=0. Uses **Soft Reset** (VRAM preserved). Trace starts if enabled. |
| `step`         | None                | Executes a single instruction (Fetch-Decode-Dispatch-Execute).                             |
| `reg`          | `[lane_id]`         | Displays registers for specific lane (0-7). Defaults to Lane 0.                            |
| `trace:stream` | None                | Enables real-time JSON trace streaming during execution.                                   |
| `trace:off`    | None                | Disables trace streaming.                                                                  |

**Example Manual Session:**

```bash
> reset
VM Reset
> load 10010005  # MOV R1, 5
Loaded: 10010005
> mem 0 100      # Write 100 to VRAM[0]
Mem Written
> run
Running...
```

## 3. Python SDK Usage

The project includes Python tools to automate interaction with the VM.

### 3.1 Setup

Ensure your environment is set up:

```python
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, "/path/to/arduino-cluster-ops")

from esp32_tools import ESP32Connection
from esp32_tools.program_loader_v15 import InstructionV15
```

### 3.2 Connecting

```python
PORT = "/dev/cu.usbserial-XXXX"
conn = ESP32Connection(PORT)
print("Connected!")
```

### 3.3 Writing Memory (VRAM)

Use the `mem` command via `send_command`:

```python
# Write integer 1234 to Address 0
addr = 0
val = 1234
conn.send_command(f"mem {addr} {val}", delay=0.05)

# Verify with dump
conn.send_command(f"dump {addr} 1", delay=0.1)
print(conn.read_lines())
```

### 3.4 Building Instructions (`InstructionV15`)

Helper class to generate 32-bit Hex codes for ISA v1.5.

**Common Operations:**

```python
# Integer Arithmetic
# IADD Dest, Src1, Src2
inst_add = InstructionV15.iadd(1, 2, 3) # R1 = R2 + R3

# Immediate Move
# MOV Dest, Immediate
inst_mov = InstructionV15.mov(1, 100)   # R1 = 100

# Memory Load (Gather)
# LDX Dest, BaseReg, OffsetReg
inst_ldx = InstructionV15.ldx(0, 15, 30) # R0 = Mem[R15 + R30]

# Memory Store (Scatter)
# STX BaseReg, OffsetReg, SrcReg
inst_stx = InstructionV15.stx(15, 25, 10) # Mem[R15 + R25] = R10

# Control Flow
# EXIT
inst_exit = InstructionV15.exit_inst()
```

### 3.5 Loading a Program

```python
program = [
    InstructionV15.mov(1, 10),
    InstructionV15.iadd(2, 1, 1),
    InstructionV15.exit_inst()
]

# Reset first
conn.send_command("reset")

# Load loop
for inst in program:
    conn.send_command(f"load {inst.to_hex()}")
```

### 3.6 Parsing JSON Trace

When `trace:stream` is enabled, the VM outputs JSON lines.

```python
import json
import re

def parse_trace(raw_output):
    # Extract JSON part
    start = raw_output.find('{')
    end = raw_output.rfind('}')
    json_str = raw_output[start:end+1]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError:
        print("Use robust parsing logic for large traces...")
        return None

# Enable Trace & Run
conn.send_command("trace:stream")
conn.send_command("run")

# Capture Output
output = ""
for _ in range(20):
    lines = conn.read_lines()
    if lines: output += "\n".join(lines)

trace_data = parse_trace(output)
```

## 4. ISA v1.5 Reference

### Registers

- **R0-R31**: General Purpose (Integer/Address)
- **F0-F31**: Floating Point
- **P0-P7**: Predicate (Condition Codes)
- **SR**: System Registers (LaneID, WarpSize)

### Instruction Format

`[OP: 8 bits] [DEST: 8 bits] [SRC1: 8 bits] [SRC2/IMM: 8 bits]`

### Opcode Summary

| Group       | Instructions                  | Description                                      |
| :---------- | :---------------------------- | :----------------------------------------------- |
| **Control** | `NOP`, `EXIT`, `BRA`, `BR.Z`  | Program flow. `BR.Z` branches if P0 is set.      |
| **Integer** | `MOV`, `IADD`, `ISUB`, `IMUL` | Standard ALU ops. `MOV` supports Immediate.      |
| **Logic**   | `AND`, `OR`, `XOR`            | Bitwise operations.                              |
| **Float**   | `FADD`, `FSUB`, `FMUL`        | IEEE-754 FP32 operations.                        |
| **Memory**  | `LDX`, `STX`, `LDG`, `STG`    | `LDX/STX` use `[Base + Offset]` addressing mode. |
| **System**  | `S2R`                         | Read System Reg (e.g., `S2R R31, SR_LANEID`).    |

## 5. Troubleshooting

- **VRAM Read=0**: Ensure you used `mem` command to initialize data _after_ `reset` and that `run` is using Soft Reset (Project v2.1+).
- **Connection Failed**: Check if `cli.py monitor` is running in another terminal. Only one connection allowed.
- **JSON Parse Error**: Large traces may be fragmented by Serial buffer limits. Use the robust chunk-parsing logic provided in `test_enhanced_trace.py`.
