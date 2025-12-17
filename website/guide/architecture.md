# System Architecture

The system architecture defines a strict hierarchical topology, designed to physically emulate the CUDA execution model. Data flows from the high-level software abstraction on the host down to bit-level arithmetic operations in the distributed cores.

![Detailed System Architecture](/assets/figures/sys_arch.png)

## Execution Model (SIMT)

The core execution model follows a Single-Instruction Multiple-Thread (SIMT) paradigm, relying on a "Broadcast-and-Mask" mechanism.

![SIMT Execution Swimlane](/assets/figures/simt_swimlane.png)

## Cluster Hierarchy

### Layer 0: Host System (Grid)

The host PC utilizes PyTorch to define the computational graph. It performs **Grid Tiling**, breaking large tensors into smaller chunks that fit the SRAM constraints of the microcontroller network. These tiles are flattened into a serial stream.

### Layer 1: GPU Master (AMB82-Mini)

This layer acts as the centralized controller. It features an ARM Cortex-M33 core.

- **DMA Engine:** The defining feature is the use of the DMA hardware as a "Virtual Second Core." It drives the 8080 parallel bus, generating Write (WR) and Chip Select (CS) signals automatically.

### Layer 2: Streaming Multiprocessors (ESP32-S3)

The ESP32-S3 mimics a CUDA SM. It utilizes its dual-core architecture to decouple reception from scheduling. Core 0 fills a Ring Buffer from the Global Bus, while Core 1 reads from this buffer to schedule instructions for the downstream threads.

### Layer 3: SMSP / Threads (RP2040)

The RP2040 acts as the fundamental ALU. It uses its Programmable I/O (PIO) state machines to ingest instructions from the Local G-Bus, pushing them into a FIFO for executing.

## ESP32-S3 Micro-Architecture (Node Detail)

While the global architecture describes the cluster data flow, the specific implementation of the "Micro-CUDA" VM on the ESP32-S3 node mimics a discrete GPU's internal structure.

![ESP32-S3 Internal Micro-Architecture](/assets/figures/micro_arch_diag.png)

1. **Core 0 (Warp Scheduler)**: Fetches instructions, handles PC control (Branching), and queues batches for execution.
2. **Core 1 (SIMD Engine)**: Conceptually executes parallel lanes. Core 1 maintains 8 independent register contexts (Micro-CUDA VM mode) or drives external ALUs.

This dual-core split allows the "SM" to maintain high throughput by overlapping instruction fetch/decode (Core 0) with mathematical execution (Core 1).

## Scalability and Multi-Chip Integration

To scale beyond a single node, the architecture supports a hierarchical cluster topology.

### Inter-Node Communication

Multiple ESP32-S3 nodes (Layer 2) are connected via a high-speed SPI bus (50 MHz) to a central master (FPGA or Gateway). This allows the host to broadcast kernels to the entire cluster or address specific nodes for task parallelism.

### Global Synchronization

To support multi-chip kernels (e.g., distributed matrix multiplication), a dedicated open-drain GPIO line acts as a wired-AND "Global Barrier". When a kernel reaches a global sync point, it pulls the line low. The line only returns high when all nodes have released it, ensuring <1µs synchronization latency across the cluster.

## Layer 1 Detail: AMB82-Mini (Master Controller)

The AMB82-Mini serves as the high-level scheduler, implementing an Asymmetric Multi-Processing (AMP) model to manage the flow of data from the host to the distributed compute nodes.

### Layer 1 AMP Architecture

The AMB82-Mini effectively utilizes its DMA engine as a secondary processor, managed by an AMP-like scheduler that orchestrates context switching and priorities in external DDR memory.

![Layer 1 AMP Architecture](/assets/figures/layer1_amp.png)

### Grid Dispatch Algorithm

The following algorithm illustrates the Grid Dispatch and DMA Injection logic running on the AMB82-Mini:

![Grid Dispatch Algorithm](/assets/figures/layer1_algo.png)

## Layer 2 Detail: ESP32-S3 as Streaming Multiprocessor

The ESP32-S3 functions as the critical Layer 2 _Streaming Multiprocessor (SM)_, bridging the high-bandwidth Global Bus and the localized execution threads.

### Layer 2 Internal Architecture

The ESP32-S3 SM Architecture showing the split between Core 0 (Receiver) and Core 1 (Scheduler), connected by ring buffers and shared L1 PSRAM.

![Layer 2 Internal Architecture](/assets/figures/layer2_arch.png)

The architecture adopts a heterogeneous dual-core strategy:

- **Core 0 (Receiver)**: Dedicated to high-throughput I/O. It filters incoming packets from the Global Bus based on metadata sidebands (MD0-MD3), placing valid tasks into a FIFO Ring Buffer.
- **Core 1 (Scheduler)**: Implements complex warp scheduling logic. It pulls tasks from the buffer, reorders them to hide memory latency (via Reorder Queue), and dispatches instruction packets to the localized worker threads via the Local G-Bus.

### Warp Scheduler Algorithm

The Warp Scheduler operates on a "Broadcast-and-Mask" principle to ensure strictly synchronized execution across distributed lanes:

![Warp Scheduler Algorithm](/assets/figures/warp_sched_algo.png)

### SIMT Execution Algorithm (Layer 3)

The execution logic running on the RP2040 SMSP cores, depicting the Lane-Aware Load and Packed Arithmetic operations:

![SIMT Execution Algorithm](/assets/figures/simt_exec_algo.png)

## Hardware Specifications

### AMB82-Mini (GPU Grid Master)

The AMB82-Mini serves as the cluster controller and edge computing node, providing the necessary computational power for coordination and AI acceleration.

- **MCU**: ARMv8-M (Cortex-M33), up to 500MHz. Optimized for high-speed control and coordination tasks within the cluster.
- **NPU**: Intelligent Engine (0.4 TOPS). Supports efficient AI inference and accelerates edge neural networks.
- **Memory**: Built-in DDR2 128MB + External 16MB SPI Nor Flash. Utilized as the primary buffer for the GPU Grid and for firmware storage.
- **Peripherals Overview**:
  - **GPIO**: Up to 23 pins.
  - **PWM**: 8 channels.
  - **UART**: 3 interfaces.
  - **SPI**: 2 interfaces.
  - **I2C**: 1 interface.

### Comparison: RP2040 vs. ESP32

| Feature            | RP2040                          | ESP32                                                          |
| :----------------- | :------------------------------ | :------------------------------------------------------------- |
| **CPU**            | Dual-core Cortex-M0+ @ 133MHz   | Xtensa Dual/Single-core 32-bit LX6, up to 240MHz               |
| **SRAM / ROM**     | 264 KB, Independent Banks       | 320 KB RAM, 448 KB ROM                                         |
| **Flash Memory**   | External QSPI Flash (Max 16 MB) | Supports SD/SDIO/MMC/EMMC Host, Built-in Flash varies by board |
| **DMA Controller** | Yes                             | Yes                                                            |
| **Interconnect**   | Fully Connected AHB             | Dedicated DMA Channels                                         |
| **GPIO**           | 30 total, 4 Analog Inputs       | 34 Programmable GPIOs                                          |
| **Internal Flash** | 2 MB (Typical external)         | 4 MB (Typical)                                                 |
