# Performance Benchmarks

## Micro-CUDA vs. CMSIS-NN

We compared the execution time of a standard **32x32 Matrix Multiplication (INT8)**.

- **Baseline**: ARM CMSIS-NN on Raspberry Pi Pico (Single Core @ 133MHz).
- **Target**: Micro-CUDA Cluster (ESP32 + 4x RP2040).

| Platform       | Cores | Execution Time | Speedup  |
| :------------- | :---- | :------------- | :------- |
| **CMSIS-NN**   | 1     | 14.5 ms        | 1.0x     |
| **Micro-CUDA** | 4     | 3.8 ms         | **3.8x** |

The 3.8x speedup on 4 cores demonstrates nearly linear scaling, proving the efficiency of the split-bus architecture in masking memory latency.

## Power Consumption

Measured at the 5V power rail.

| State         | Current (mA) | Power (W) | Description      |
| :------------ | :----------- | :-------- | :--------------- |
| **Idle**      | 80 mA        | 0.40 W    | System On        |
| **Broadcast** | 350 mA       | 1.75 W    | ESP32 Streaming  |
| **Full Load** | 920 mA       | 4.60 W    | All Cores Active |

The cluster operates within the standard USB-C power envelope (5V/3A), making it suitable for portable edge AI applications.
