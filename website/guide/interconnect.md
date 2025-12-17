# Interconnect & Synchronization

This section details the communication protocols and physical bus interfaces that connect the distributed cores in the Micro-CUDA architecture.

## Inter-Core Communication Protocol

A critical challenge in implementing a software-defined GPU on a dual-core MCU is ensuring efficient and correct synchronization between the Control Unit (Core 0) and the Execution Units (Core 1). We employ a strictly ordered producer-consumer model using FreeRTOS primitives.

### Communication Mechanism

The system relies on two primary queues:

1. **Instruction Queue (`instrQueue`)**: A deep buffer (size 64) that carries `InstrBatch` objects from Core 0 to Core 1. This allows the Front-End to "run ahead" of the Back-End, smoothing out fetch latencies.
2. **Feedback Queue (`feedbackQueue`)**: A small, high-priority queue used when Core 1 needs to report a predicate result back to Core 0 (e.g., for `BR.Z` or `OP_BAR_SYNC`).

### Instruction Batching

To reduce the overhead of context switching and queue locking, instructions are not sent individually. Instead, they are packed into **Instruction Batches**.

```cpp
struct InstrBatch {
    uint8_t count;          // 1-16 Instructions
    Instruction insts[16];  // Payload

    // Control Signals
    bool is_sync_req;       // Requires barrier?
    bool is_exit;           // End of Kernel?
};
```

Core 0 fills this batch until it encounters a control dependency (branch) or the batch is full. It then pushes the entire batch to the queue in a single operation.

### Synchronization Sequence

Normal instructions are pipelined via batches. Control flow instructions (e.g., `BR.Z`) trigger a feedback loop, stalling Core 0 until Core 1 processes the predicate.

![Sequence Diagram](/assets/figures/sequence.png)

### Handling Divergence

When Core 0 decodes a `BR.Z P0` instruction:

1. It flushes the current batch to Core 1 immediately.
2. It sends a special `SYNC_REQ` signal.
3. It blocks on the `feedbackQueue`.
4. Core 1 executes all pending instructions, then reads the value of P0 from Lane 0 (or a consensus of lanes), and sends it back.
5. Core 0 receives the value, updates the PC, and resumes fetching.

This design ensures that control flow decisions are always based on the most up-to-date GPU state, preventing hazards.

## Physical Bus Interface Specification

This section details the 8-bit parallel bus protocol used for inter-layer communication in the cluster architecture. The protocol is designed for high-bandwidth instruction and data streaming between heterogeneous microcontroller nodes.

### Interface Overview

- **Interface Type**: 8-bit Parallel (Half-Duplex / Broadcast-Optimized)
- **Target Bandwidth**: ~50 MB/s (limited by GPIO toggle speed and DMA efficiency)
- **Logic Voltage**: 3.3V CMOS
- **Bus Standard**: Intel 8080-style parallel interface

This physical layer is used for both **Layer 1 (AMB82-Mini) -> Layer 2 (ESP32-S3)** and **Layer 2 (ESP32-S3) -> Layer 3 (RP2040)** connections, creating a hierarchical data distribution network.

### Physical Layer Pinout

| Signal   | Type    | Logic       | Description                                                                               |
| :------- | :------ | :---------- | :---------------------------------------------------------------------------------------- |
| `D[0:7]` | Data    | Tri-state   | 8-bit bidirectional data bus. Primarily used for Master write to Slave.                   |
| `CS#`    | Control | Active Low  | **Chip Select**. When asserted low, Slave activates and monitors bus.                     |
| `DC`     | Control | High/Low    | **Data/Command** select. LOW: Control command. HIGH: Data payload.                        |
| `WR#`    | Control | Rising Edge | **Write Strobe**. Master prepares data on falling edge, Slave latches on **rising edge**. |
| `RD#`    | Control | Active Low  | **Read Strobe**. Used for Master to read Slave status.                                    |
| `SYNC`   | Global  | Active High | **Warp Trigger**. Global synchronization line for simultaneous execution.                 |

### Word Transmission Protocol

Since the Micro-CUDA ISA uses **32-bit fixed-length instructions** but the bus is only **8-bit wide**, a packing and reassembly protocol is required.

#### Byte Ordering (Endianness)

The system uses **Big-Endian** (network byte order) transmission.
**Example**: ISA Instruction `0x40100501` (HMMA.INT8 R10, R5, R1)

- Byte 3 (First): `0x40` (OpCode)
- Byte 2: `0x10` (Dest)
- Byte 1: `0x05` (Src1)
- Byte 0 (Last): `0x01` (Src2)

#### Burst Transmission Mode

To achieve 50 MB/s throughput, the protocol uses **Frame Burst Mode**:

1. Assert `CS#` LOW (Begin transmission)
2. Set `DC` LOW (Send header / magic word)
3. Set `DC` HIGH (Stream instruction/data payload)
4. Deassert `CS#` HIGH (End transmission)

### Timing Characteristics

**Key Timing Parameters**:

- **t_cycle**: 20 ns (50 MHz write rate)
- **t_setup**: Data stable >= 5 ns before rising edge
- **t_hold**: Data held >= 3 ns after rising edge

### Hardware Implementation Notes

#### RP2040 Reception (PIO State Machine)

The RP2040's Programmable I/O (PIO) is critical for achieving 50 MB/s throughput.

```asm
.program parallel_8080_rx
.wrap_target
    wait 0 pin 10      ; Wait for CS# low (Pin 10)
    wait 0 pin 11      ; Wait for WR# low (Pin 11)

    in pins, 8         ; Read 8-bit D0-D7 into ISR

    wait 1 pin 11      ; Wait for WR# rising edge
.wrap
```

#### Signal Integrity Considerations

- **Trace Length Matching**: All 8 data lines should be routed with ±1 mm length tolerance.
- **GPIO Drive Strength**: 20 mA recommended.
- **Termination**: 100Ω series termination for cables > 10 cm.
- **Ground Planes**: Continuous ground plane required.

### Performance Analysis

The theoretical maximum throughput is determined by the `WR#` toggle rate:
50 MB/s (400 Mbps) at 20ns cycle time.

In practice, overhead reduces effective throughput to approximately 40-45 MB/s, which is sufficient for streaming weights and instructions.
