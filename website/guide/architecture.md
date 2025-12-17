# System Architecture Deep Dive

## Cluster Hierarchy

The system defines a strict hierarchical topology physically emulating the CUDA execution model.

| Layer  | Device     | CUDA Equivalent | Role                                   |
| :----- | :--------- | :-------------- | :------------------------------------- |
| **L0** | Host PC    | Host CPU        | PyTorch Grid Tiling & compilation.     |
| **L1** | AMB82-Mini | GPU Master      | Central controller & DMA Engine.       |
| **L2** | ESP32-S3   | SM              | Warp Scheduler & Instruction Dispatch. |
| **L3** | RP2040     | CUDA Cores      | Parallel ALU & Execution Units.        |

## Layer 1: GPU Master (AMB82-Mini)

The AMB82-Mini acts as the cluster controller. It implements an **Asymmetric Multi-Processing (AMP)** model.

- **DMA as Virtual Core**: The DMA engine drives the Global G-BUS, generating `CS` and `WR` signals automatically from a RAM buffer. This frees the Cortex-M33 CPU to handle high-level logic.
- **Context Queue**: Manages task priority and "Grid Launch" events similar to a GPU command processor.

## Layer 2: Streaming Multiprocessor (ESP32-S3)

The ESP32-S3 mimics a discrete GPU Streaming Multiprocessor (SM) using its dual-core architecture.

![ESP32 Internal Arch](/images/micro_arch_diag.png)

```mermaid
graph TD
    User[Host PC / Python] -->|UART Command| Core0

    subgraph ESP32 [Mini-SM]
        subgraph FrontEnd ["Core 0: Front-End (Warp Scheduler)"]
            Fetch[Instruction Fetch]
            Decode[Decode & Issue]
            PC[PC Control]
            Trace[Trace Unit]
        end

        subgraph BackEnd ["Core 1: Back-End (SIMD Engine)"]
            SIMD["8-Lane SIMD Execution Unit"]
            RegFile["Register File (8x R[32])"]
        end

        FrontEnd -->|Instructions (Queue)| BackEnd
        BackEnd -->|Completion (Queue)| FrontEnd
    end
```

### Core 0 (Receiver / Front-End)

Dedicated to high-throughput I/O.

1.  **Listen**: Monitors Global G-BUS for incoming instruction packets.
2.  **Filter**: Uses Sideband Metadata (`MD0-MD3`) to accept/reject packets designated for this SM.
3.  **Queue**: Pushes valid tasks into a **Ring Buffer**.

### Core 1 (Scheduler / Back-End)

Implements the Warp Scheduler logic.

1.  **Fetch**: Pulls tasks from the Ring Buffer.
2.  **Reorder**: A simplified Reorder Queue hides memory latency.
3.  **Dispatch**: Broadcasts instructions to local RP2040 cores via the **Local G-BUS**.

## Layer 3: SMSP Cores (RP2040)

The RP2040 represents the "CUDA Cores".

- **PIO State Machine**: A custom `parallel_8080_rx` PIO program ingests 32-bit instructions at 50 MB/s without CPU intervention.
- **Double Buffered Execution**: While the PIO fills the RX FIFO, the Cortex-M0+ executes the previous batch of instructions.

## Kernel Launch Flow

1.  **Config**: AMB82 broadcasts kernel dimensions and parameters.
2.  **Broadcast**: Instructions are streamed to all ESP32s.
3.  **Sync**: A dedicated **Global Barrier** (`SYNC_TRIG`) line is pulled LOW.
4.  **Execute**: When all nodes release the barrier, execution starts simultaneously (<1µs jitter).

![Execution Mapping](/images/execution_mapping_placeholder.png)

## Hardware Interface

The system utilizes a **Dual-Port Split-Bus** design to enable a true pipeline, avoiding a shared bus bottleneck.

### Physical Bus Protocol

We use a custom **8-bit Parallel Low-Latency Bus** (Intel 8080-style).

- **Bandwidth**: ~50 MB/s (20ns cycle time)
- **Voltage**: 3.3V CMOS
- **Transmission**: Big-Endian, Burst Mode

| Signal   | Type    | Description                                 |
| :------- | :------ | :------------------------------------------ |
| `D[0:7]` | Data    | 8-bit bidirectional data bus                |
| `CS#`    | Control | Chip Select (Active Low)                    |
| `DC`     | Control | **Low**: Command / **High**: Data           |
| `WR#`    | Clock   | Write Strobe (Slave latches on Rising Edge) |
| `SYNC`   | Global  | **Warp Trigger** (Global Barrier Release)   |

### Pin Mapping (ESP32-S3)

The ESP32-S3 acts as a router/scheduler, managing simultaneous RX (from Host) and TX (to SMSP Cores).

#### Input Interface (Slave)

| Signal          | Pin      | Function                     |
| :-------------- | :------- | :--------------------------- |
| `G_DATA_[0..7]` | GPIO 1-9 | Data Input (Skipping GPIO 3) |
| `G_WR`          | GPIO 10  | Write Strobe                 |
| `G_DC`          | GPIO 11  | Data/Command                 |

#### Output Interface (Master)

| Signal          | Pin        | Function           |
| :-------------- | :--------- | :----------------- |
| `L_DATA_[0..3]` | GPIO 15-18 | Low Nibble         |
| `L_DATA_[4..7]` | GPIO 39-42 | High Nibble        |
| `L_WR`          | GPIO 48    | Write Strobe       |
| `SYNC_TRIG`     | GPIO 46    | **Global Barrier** |

### Timing & Integrity

- **Cycle Time**: 20 ns (50 MHz)
- **Setup Time**: Data must be stable 5ns before `WR#` rising edge.
- **Hold Time**: Data must be held 3ns after `WR#` rising edge.
- **Requirements**: Matching trace lengths (±1mm) and a solid common ground plane.

## Firmware Internals

### Core Responsibilities

- **Core 0 (Front-End)**: Handles Instruction Fetch, Decode, PC Control, and UART I/O.
- **Core 1 (Back-End)**: Dedicated to the 8-Lane SIMD Execution Loop.
- **Synchronization**: Uses FreeRTOS Queues for cycle-accurate lockstep execution (Issue -> Execute).

### LZ4 Decompression Integration

To accelerate kernel loading, the firmware integrates a lightweight LZ4 decompressor.

1.  **Library**: Ported `lz4.c` / `lz4.h` (std C).
2.  **Flow**:
    - Host sends `load_imem_lz4 <size>`.
    - Firmware allocates specialized buffer.
    - Chunks are received and decompressed in-place into I-RAM.
3.  **Performance**: Reduces kernel upload time by ~3-4x compared to raw binary transfer.
