# Hardware Implementation

This section details the hardware pipeline, topology, and firmware implementation of the Micro-CUDA architecture.

## Hardware Pipeline and Topology

This architecture implements a strict hardware-level pipeline, physically emulating the GPU execution model (Host $\to$ GigaThread Engine $\to$ SM $\to$ CUDA Cores). To achieve the high-speed throughput required for "Memory Streaming" and the Parallel Bus, the physical planning of GPIOs is critical.

### Control Algorithms Summary

| Alg                | Layer            | Core Concepts            |
| :----------------- | :--------------- | :----------------------- |
| **Warp Scheduler** | Layer 2 (ESP32)  | Broadcast, Masking, Sync |
| **Grid Dispatch**  | Layer 1 (AMB82)  | Tiling, DMA Zero-Copy    |
| **SIMT Execution** | Layer 3 (RP2040) | LaneID Offset, PIO Fetch |

### Split-Bus Architecture

To maximize performance, the system avoids a shared bus topology in favor of a **Dual-Port Split-Bus** architecture. This design enables a true pipeline: while the AMB82-Mini (Layer 1) fills Buffer A on the ESP32-S3 (Layer 2), the ESP32-S3 can simultaneously broadcast data from Buffer B to the RP2040s (Layer 3).

1. **Global G-BUS (Upstream)**: Handles bulk tensor data transfer from AMB82-Mini to ESP32-S3 (50MB/s).
2. **Local G-BUS (Downstream)**: Handles instruction and local data broadcast from ESP32-S3 to the array of RP2040s.

## Pin Mapping Strategy

The GPIO mapping is optimized for Direct Memory Access (DMA) and Programmable I/O (PIO), ensuring that data lines are physically contiguous for single-cycle operations.

### Layer 1: AMB82-Mini (GPU Master)

The AMB82-Mini drives the Global G-BUS using its high-speed GPIOs. Efficient 8-bit parallel output requires direct register manipulation.

| Signal        | Type | Pin      | Function                   |
| :------------ | :--- | :------- | :------------------------- |
| G*DATA*[0..7] | OUT  | D0-D7    | 8-bit Parallel Data Bus    |
| G_WR          | OUT  | D8       | Write Strobe (Active Low)  |
| G_DC          | OUT  | D9       | Data/Command Select        |
| G*CS*[0..1]   | OUT  | D10, D11 | Chip Select for SM 0, SM 1 |
| G_BUSY        | IN   | D12      | Flow Control (Wait State)  |

### Layer 2: ESP32-S3 (Streaming Multiprocessor)

The ESP32-S3 acts as a router with dual separated interfaces to support full-duplex pipelining.

| Signal               | ESP32 Pin     | Description                  |
| :------------------- | :------------ | :--------------------------- |
| **Input Interface**  |               | **From AMB82**               |
| G*DATA*[0..7]        | GPIO 1-9      | Bits 0-7 (Skipping GPIO 3)   |
| G_WR                 | GPIO 10       | PCLK / Write Strobe          |
| G_DC / G_CS          | GPIO 11 / 12  | Control Signals              |
| G_BUSY               | GPIO 13       | Output to Master             |
| **Output Interface** |               | **To RP2040**                |
| L*DATA*[0..3]        | GPIO 15-18    | Low Nibble                   |
| L*DATA*[4..7]        | GPIO 39-42    | High Nibble                  |
| L_WR / L_DC          | GPIO 48 / 47  | Write Strobe / Data-Cmd      |
| L*CS*[0..3]          | 14, 21, 38, 3 | Chip Selects for active SMSP |
| SYNC_TRIG            | GPIO 46       | Global Barrier Sync          |

### Layer 3: RP2040 (SMSP Cores)

The RP2040's Programmable I/O (PIO) requires strictly contiguous pins.

- **Data [0-7]**: GP0 - GP7 (Contiguous Block)
- **WR Strobe**: GP8 (JMP Pin)
- **DC Signal**: GP9 (Side-set)
- **CS Input**: GP10 (IRQ/Enable)
- **Sync**: GP11 (Wait/Barrier)

## Firmware Implementation

### VRAM Organization

The ESP32-S3 has limited internal RAM (512KB SRAM). We allocate a 100KB static array as the Virtual VRAM.

- **0x0000 - 0x0FFF**: Program Text (Instructions)
- **0x1000 - 0x3FFF**: Global Data
- **0x4000 - 0xDFFF**: Heap / Stack areas

Since the ESP32 is a flat memory machine, mapping VRAM is a simple pointer offset operation.

### SIMD Engine Implementation (`vm_simd_v15.cpp`)

The firmware is written in C++ (Arduino framework). The `backEndTask` is pinned to CPU 1 and optimized with `-O3`.

> [!TIP] > **SIMD Execution**: The scheduler drives the 8-bit bus once. The allocation of work happens implicitly at the edge.
>
> - **Scheduler**: Broadcasts `Opcode: 0x64 (LDL), Operand: R1, [R0]`
> - **Lane i**: Executes `R1 = Mem[R0 + i * 4]`

#### Computed Goto Dispatch

Traditional switch statements incur significant overhead in embedded systems due to branch prediction penalties. The implementation uses **computed goto** (a GNU C extension) to eliminate this bottleneck.

- **Traditional Switch**: ~30 cycles (Branch prediction miss penalty)
- **Computed Goto**: 5 cycles (Direct jump)
- **Speedup**: ~6x dispatch speedup

```cpp
static void* dispatch_table[256];
if (!initialized) {
    for(int i=0; i<256; i++) dispatch_table[i] = &&LABEL_UNKNOWN;
    dispatch_table[OP_IADD] = &&LABEL_OP_IADD;
    // ...
}
goto *dispatch_table[inst.opcode];

LABEL_OP_IADD:
    asm_warp_add(dest, src1, src2, P);
    return;
```

#### ASM-Optimized Warp Operations

Critical arithmetic kernels are implemented using raw Xtensa LX6 assembly, utilizing hardware zero-overhead loops (`loop` instruction).

### System Configuration

| Parameter           | Value   | Description            |
| :------------------ | :------ | :--------------------- |
| `VM_CPU_FREQ`       | 240 MHz | Max CPU Clock (Locked) |
| `VM_BAUD_RATE`      | 460,800 | High-speed UART        |
| `VM_SERIAL_RX_SIZE` | 32,768  | 32KB RX Buffer (Turbo) |
| `VM_STACK_SIZE`     | 20,480  | Stack per Core (20KB)  |
| `VM_QUEUE_SIZE`     | 32      | Instruction Batches    |
| `VM_BATCH_SIZE`     | 32      | Instructions per Batch |
| `VM_VRAM_SIZE`      | 65,536  | 64KB Virtual VRAM      |

## System Reliability

### Failure Recovery

To maintain cluster stability, the firmware implements watchdog timers on both cores. If Core 1 hangs (e.g., infinite loop in kernel), Core 0 resets the SIMD engine state without requiring a full system reboot.

### Memory Safety

VRAM operations enforce strict bounds checking. Invalid addresses are clamped to a safe "bit bucket" region, preventing wild writes from crashing the firmware.
