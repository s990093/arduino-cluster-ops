# System Architecture

## Overview

The Micro-CUDA architecture transforms the ESP32-S3 into a customized GPU-like accelerator. It uses **Core 0** for instruction scheduling (Front-End) and **Core 1** for parallel SIMT execution (Back-End).

## Core 0: The Front-End Scheduler

Core 0 acts as the "Grid Master" or Warp Scheduler. It is responsible for:

- Fetching instructions from the Instruction Memory (IMEM).
- Decoding 32-bit opcodes.
- Handling control flow instructions like Branches (`BRA`, `BR.Z`) and Loops.
- managing synchronization barriers (`BAR.SYNC`).
- Dispatching safe, executable instruction bundles to the execution engine.

## Core 1: The SIMD Back-End

Core 1 is a heavily optimized software-defined **SIMD Engine**. It emulates 8 parallel "lanes" (Threads).

- **Lockstep Execution**: All 8 lanes execute the exact same instruction at the same time.
- **Divergence Handling**: If lanes diverge (e.g., `if (lane_id < 4)`), the hardware (emulated) handles masking automatically using Predicate registers (`P0-P7`).

## Register File Organization

Each lane has its own independent register set:

| Type     | Name   | Count | Width  | Description                               |
| :------- | :----- | :---- | :----- | :---------------------------------------- |
| **GPR**  | R0-R31 | 32    | 32-bit | General Purpose Integers / Addresses      |
| **FPR**  | F0-F31 | 32    | 32-bit | IEEE 754 Floating Point / BFloat16        |
| **PRED** | P0-P7  | 8     | 1-bit  | Predicate flags for conditional execution |

## Memory Hierarchy

1.  **Global Memory (VRAM)**: Shared 40KB - 1MB PSRAM/SRAM region accessible by all lanes.
2.  **Shared Memory (LDS/STS)**: Fast, low-latency scratchpad memory for inter-lane communication.
3.  **Instruction Memory (IMEM)**: Holds the kernel binaries uploaded from the host.
