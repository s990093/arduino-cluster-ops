# Evaluation

This section presents performance benchmarks, a real-world case study on Transformer Attention, and detailed electrical specifications of the cluster.

## Performance Benchmarks

To validate the efficiency of the Micro-CUDA architecture, we conducted comparative benchmarks against industry-standard solutions for embedded machine learning.

### Micro-CUDA vs. CMSIS-NN

We compared the execution time of a standard $32 \times 32$ Matrix Multiplication (INT8) kernel. The baseline is an optimized ARM CMSIS-NN implementation running on a single Core of the Raspberry Pi Pico (Cortex-M0+ @ 133MHz).

| Platform                 | Execution Time (ms) | Speedup  |
| :----------------------- | :------------------ | :------- |
| **CMSIS-NN (1 Core)**    | 14.5                | 1.0x     |
| **Micro-CUDA (4 Cores)** | 3.8                 | **3.8x** |

The comparison highlights the efficacy of the parallel SIMT dispatch model. While the individual Cortex-M0+ cores are identical, the split-bus architecture allows for concurrent data loading and execution, masking memory latency.

### Power Consumption Profile

Power efficiency is a critical metric for edge devices. Table below details the current draw across different operational states, measured at the 5V VSYS rail.

| State         | Description                 | Current (mA) | Power (W) |
| :------------ | :-------------------------- | :----------- | :-------- |
| **Idle**      | System on, Clock Gating     | 80           | 0.40      |
| **Broadcast** | ESP32 streaming data        | 350          | 1.75      |
| **Compute**   | SM cores executing ALU      | 850          | 4.25      |
| **Full Load** | Pipeline Active (RX+TX+ALU) | 920          | 4.60      |

The "Full Load" state demonstrates that the cluster operates within the envelope of standard USB-C power delivery (5V/3A), making it suitable for portable deployment without specialized power supplies.

## Case Study: Parallel Attention

To validate the architecture, we implemented the QK (Query-Key) multiplication step of the Transformer Self-Attention mechanism.

$$ \text{Score}\_i = Q_i \cdot K_i + V_i $$

### Setup

- **Warp Size**: 8
- **Data**: 3 arrays (Q, K, V) of size 8, stored contiguously in VRAM.
- **Objective**: Compute the dot product + bias for each element in parallel.

### Micro-CUDA Assembly Code

```asm
; 1. Initialization
S2R   R31, SR_LANEID     ; R31 = My Lane ID

; 2. Set Base Addresses
MOV   R0, 0x1000        ; R0 = Base of Q
MOV   R1, 0x2000        ; R1 = Base of K
MOV   R2, 0x3000        ; R2 = Base of V

; 3. Parallel Load (SIMT)
; Each lane loads from Base + LaneID*4
LDL   R10, [R0]         ; R10 = Q[lane]
LDL   R11, [R1]         ; R11 = K[lane]
LDL   R12, [R2]         ; R12 = V[lane]

; 4. Compute
IMUL  R20, R10, R11     ; R20 = Q * K
IADD  R21, R20, R12     ; R21 = (Q*K) + V

; 5. Writeback
MOV   R3, 0x4000        ; Result Base
STL   [R3], R21         ; Store Result[lane]

EXIT
```

### Execution Trace Analysis

A single `LDL` instruction issued by Core 0 triggers 8 distinct memory loads on Core 1. Trace logs confirmed that Lane 0 loaded from `0x1000`, Lane 1 from `0x1004`, and so on, proving the correctness of the lane-aware hardware logic.

### Extended Benchmark Suite

#### SGEMM (Matrix Multiplication)

A tiled matrix multiplication kernel achieves 70\% efficiency using 4x4 register blocking. The explicit `LDL/STL` vector instructions maximize bus utilization during tile loading.

#### Parallel Reduction

A log-step sum reduction uses shared memory (`LDS/STS`) to compute the sum of 1024 elements in 2048 cycles, demonstrating efficient inter-lane communication.

## Electrical Specifications

This section details the electrical characteristics, operating conditions, and timing requirements for the Micro-CUDA cluster hardware.

### Absolute Maximum Ratings

| Symbol      | Parameter                 | Min  | Max        | Unit |
| :---------- | :------------------------ | :--- | :--------- | :--- |
| **V_CC**    | Supply Voltage (VSYS)     | -0.3 | 5.5        | V    |
| **V_IO**    | GPIO Input Voltage        | -0.3 | V_CC + 0.3 | V    |
| **I_total** | Total Current Source/Sink | -    | 1200       | mA   |
| **T_str**   | Storage Temperature       | -40  | 125        | °C   |

### DC Characteristics

Operating conditions: V_CC = 5.0V, T_A = 25°C unless otherwise noted.

| Symbol     | Parameter           | Condition        | Min  | Max  | Unit |
| :--------- | :------------------ | :--------------- | :--- | :--- | :--- |
| **V_IH**   | Input High Voltage  | -                | 2.0  | V_CC | V    |
| **V_IL**   | Input Low Voltage   | -                | -0.3 | 0.8  | V    |
| **V_OH**   | Output High Voltage | I_OH = -12mA     | 2.4  | -    | V    |
| **V_OL**   | Output Low Voltage  | I_OL = 12mA      | -    | 0.4  | V    |
| **I_idle** | Idle Current        | No computation   | -    | 80   | mA   |
| **I_load** | Full Load Current   | All Cores Active | -    | 950  | mA   |

### AC Timing Analysis (Global G-BUS)

The Global G-BUS operates at 50MHz.

- **Setup Time (t_su)**: >= 5ns (Allows propagation delay across 10cm trace)
- **Hold Time (t_h)**: >= 2ns
