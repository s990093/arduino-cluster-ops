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
