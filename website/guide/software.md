# Software Implementation

## VRAM Organization

The ESP32-S3 has limited internal RAM (512KB SRAM). We allocate a 100KB static array as the Virtual VRAM. Since the ESP32 is a flat memory machine, mapping VRAM is a simple pointer offset operation.

| Address Range       | Description                 |
| :------------------ | :-------------------------- |
| **0x0000 - 0x0FFF** | Program Text (Instructions) |
| **0x1000 - 0x3FFF** | Global Data                 |
| **0x4000 - 0xDFFF** | Heap / Stack areas          |

## Firmware Architecture

The firmware is written in C++ (Arduino framework). The `backEndTask` is pinned to CPU 1 (Core 1) and optimized with `-O3` to ensure maximum throughput.

### SIMD Execution Loop

The core execution loop simulates the vector processing unit. The compiler unrolls the loop, and `lane` index handles `SR_LANEID`.

```cpp
// Core 1 Execution (Simplified)
void execute(Instruction inst) {
  // Optimization: Compiler unrolls loop
  for (int lane = 0; lane < 8; lane++) {
    LaneState& state = lanes[lane];

    // 1. Predicate Check (Masking)
    if (!state.getPredicate(inst.pred))
      continue;

    // 2. Execute Opcode
    switch (inst.opcode) {
      case IADD:
        state.R[dest] = state.R[src1] + state.R[src2];
        break;
      case LDL: // Lane-Aware Load
        // Automatic offset calculation
        uint32_t addr = state.R[src1] + lane * 4;
        state.R[dest] = VRAM[addr];
        break;
      // ... handle other opcodes
    }
  }
}
```

## Optimization Techniques

### Computed Goto Dispatch

Traditional C/C++ `switch` statements can be slow due to branch prediction penalties (~30 cycles). We use **Computed Goto** (a GNU C extension) to create a direct dispatch table, reducing dispatch overhead to ~5 cycles.

```cpp
static void* dispatch_table[256];

// Initialize once
dispatch_table[OP_IADD] = &&LABEL_OP_IADD;
dispatch_table[OP_FADD] = &&LABEL_OP_FADD;

// Fast Dispatch
goto *dispatch_table[inst.opcode];

LABEL_OP_IADD:
    asm_warp_add(dest, src1, src2, P);
    return;
```

This delivers a **6× speedup** in instruction dispatch.

### Assembly-Optimized Warp Operations

For critical arithmetic, we use inline Xtensa Assembly with hardware zero-overhead loops and manual unrolling.

```cpp
static inline void asm_warp_add(...) {
    __asm__ volatile (
        "loop %0, loop_end_add\n\t"  // Hardware zero-overhead loop
        // Lane N
        "l32i.n a8, %1, 0\n\t"       // Load src1
        "add    a8, a8, a9\n\t"      // Add
        "s32i.n a8, %3, 0\n\t"       // Store
        // ... Unrolled lanes ...
    );
}
```

## System Configuration

To ensure deterministic execution, the ESP32 is locked to specific parameters.

| Parameter           | Value   | Description                  |
| :------------------ | :------ | :--------------------------- |
| `VM_CPU_FREQ`       | 240 MHz | Max CPU Clock (Locked)       |
| `VM_BAUD_RATE`      | 460,800 | High-speed UART              |
| `VM_SERIAL_RX_SIZE` | 32 KB   | Oversized for Turbo Transfer |
| `VM_STACK_SIZE`     | 20 KB   | Per-Core Stack               |
| `VM_VRAM_SIZE`      | 64 KB   | Virtual Video Memory         |

## Memory Access Patterns

Different instructions imply different memory access patterns:

- **LDG/STG (Broadcast)**: Scalar access. Same address for all lanes.
- **LDL/STL (Strided)**: Vector access. `Address = Base + (LaneID * 4)`.
- **LDX/STX (Gather/Scatter)**: Indirect access. `Address = Base + Offset[LaneID]`.
- **LDS/STS (Shared)**: Fast scratchpad access for inter-lane communication.

## Fault Tolerance

- **Watchdogs**: If Core 1 hangs (e.g., infinite loop in kernel), Core 0 resets the SIMD engine.
- **DMA Integrity**: LZ4 compressed transfers include CRC32 checksums with automatic retransmission (ARQ).
