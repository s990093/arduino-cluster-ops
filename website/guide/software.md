# Software Stack

This section details the software ecosystem enabling Micro-CUDA, from the LLVM-based compiler to the host-side drivers and profiling tools.

## Micro-CUDA Compiler (`ucuda-cc`)

To enable rapid kernel development, we developed `ucuda-cc`, a custom compiler that translates C-like high-level language into optimized Micro-CUDA assembly.

### Compiler Architecture

The compiler infrastructure leverages the industry-standard LLVM framework:

1.  **Frontend (Clang)**: Parses CUDA-like C++ code $\to$ LLVM IR.
2.  **Middle-end (LLVM OPT)**: Standard optimization passes (constant propagation, loop unrolling).
3.  **Backend (MCC)**: Custom Python-based backend that maps generic IR to Micro-CUDA ISA and performs linear-scan register allocation.

### Implementation Details

- **Arithmetic**: LLVM `add/sub/mul` $\to$ `IADD/ISUB/IMUL`.
- **Memory**: `getelementptr` $\to$ explicit address calculations. `LDX/STX` used for gathered access.
- **Control Flow**: `icmp` $\to$ `ISETP`, `br` $\to$ `BR.Z`.

### Usage & API

Development follows a standard workflow similar to `nvcc`.

**CLI Usage:**

```bash
# Compile for ESP32-S3 (8MB PSRAM)
python compile_kernel.py vector_add.cu --target esp32s3 --output vector_add.asm
```

**Python Dynamic API:**

```python
from micro_cuda_compiler import compile_kernel

kernel_src = """
__global__ void scale(int* data, int factor) {
    int idx = laneId();
    data[idx] *= factor;
}
"""
_, asm_path = compile_kernel(kernel_src, target="esp32s3", output_asm="temp.asm")
```

## Host Interface and CLI

The system implements a robust host-device communication protocol designed to maximize throughput over the UART interface ("Turbo Mode").

### Turbo Mode Configuration

- **Baud Rate**: 460,800 bps (4x standard)
- **RX Buffer**: 32KB Ring Buffer (prevents overflow during bursts)
- **Throughput**: ~40.8 KB/s measured application throughput.

### Communication Protocol

The interface uses a hybrid **ASCII Command + Binary Burst** protocol.

#### LZ4 Compression

To mitigate UART bottlenecks, the system uses LZ4 real-time decompression.

- **Compression Ratio**: 2.5x - 4.0x for instruction streams.
- **Effective Bandwidth**: ~120 KB/s.
- **Decompression Overhead**: <2ms per 2KB chunk.

### CLI Command Reference

| Command         | Format                 | Description             |
| :-------------- | :--------------------- | :---------------------- |
| `load_imem`     | `load_imem <bytes>`    | Load Kernel Binary      |
| `load_imem_lz4` | `load_imem_lz4 <size>` | Load Compressed Kernel  |
| `dma_h2d`       | `dma_h2d <addr> <len>` | Host-to-Device Transfer |
| `kernel_launch` | `kernel_launch`        | Trigger Execution       |
| `gpu_reset`     | `gpu_reset`            | Reset Registers         |
| `reg`           | `reg <lane_id>`        | Inspect Register File   |
| `stats`         | `stats`                | Show VM Status          |
| `trace:stream`  | `trace:stream`         | Enable Real-time Trace  |

## Profiling and Debugging

To support deep analysis, the system implements an **Enhanced Trace** mechanism that streams cycle-accurate execution data in JSON format.

### Trace Format

```json
{
  "cycle": 152,
  "pc": 12,
  "asm": "IADD R4, R2, R3",
  "hw_ctx": { "sm_id": 0, "lane_id": 0, "active_mask": "0xFF" },
  "perf": {
    "latency": 1,
    "stall_cycles": 0,
    "pipe_stage": "WRITEBACK"
  },
  "lanes": [ { "lane_id": 0, "R": [5, 3, 8, ... ] } ]
}
```

### Visualization

The host-side Python client parses this stream to generate interactive timelines similar to Nsight Compute.

- **Warp Divergence Analysis**: Highlights divergent branches (red) where `active_mask != 0xFF`.
- **Pipeline Stall Visualization**: Heatmaps correlate high latency instructions with memory bottlenecks.
