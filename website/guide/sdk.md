# Python SDK & Host Interface

The generic host-device interface utilizes a "Turbo Mode" UART protocol for high-speed data transfer.

## Turbo Mode Configuration

| Parameter           | Value       | Description              |
| :------------------ | :---------- | :----------------------- |
| `VM_BAUD_RATE`      | **460,800** | 4x Standard Speed        |
| `VM_SERIAL_RX_SIZE` | 32 KB       | Oversized Ring Buffer    |
| `VM_CPU_FREQ`       | 240 MHz     | Locked for IO Throughput |

This achieves an application throughput of **~40.8 KB/s** (raw) or **~120 KB/s** (with LZ4).

## Communication Protocol

The protocol uses a hybrid **ASCII Command + Binary Burst** format.

### Kernel Loading (`load_imem`)

1.  **Request**: `load_imem_lz4 <uncompressed_size>`
2.  **Handshake**: Device responds `ACK_LZ4_GO`
3.  **Transfer**: Host sends Compressed Chunks (2KB headers)
4.  **Completion**: `LZ4_LOAD_OK`

### Data Transfer (`dma_h2d`)

Used for sending large tensors (Weight Matrices, Input Batches) to VRAM.

- **Command**: `dma_h2d_lz4 <addr> <size>`
- **Method**: Direct LZ4 decompression into VRAM.

## LZ4 Compression

To mitigate UART bottlenecks, we use real-time LZ4 decompression.

- **Compression Ratio**: 2.5x - 4.0x (Instruction Streams)
- **Decompression Speed**: >1 GB/s (on MCU)
- **Overhead**: <2ms per chunk

## CLI Command Reference

The `cli.py` tool provides direct access to these primitives.

| Command         | Format         | Description             |
| :-------------- | :------------- | :---------------------- |
| `load_imem`     | `<bytes>`      | Load Kernel Binary      |
| `dma_h2d`       | `<addr> <len>` | Host-to-Device Transfer |
| `dma_d2h`       | `<addr> <len>` | Device-to-Host Hex Dump |
| `kernel_launch` | -              | Trigger Execution       |
| `gpu_reset`     | -              | Reset Registers/PC      |
| `reg`           | `<lane_id>`    | Inspect Registers       |
| `stats`         | -              | Show PC / VRAM Status   |

## Python API Example

```python
# High-Level Usage
from microcuda import Device, SourceModule

dev = Device(port="/dev/ttyUSB0")
mod = SourceModule("kernel.cu")

# Allocate memory on device
a_gpu = dev.malloc(1024)
b_gpu = dev.malloc(1024)

# Copy data (H2D)
dev.memcpy_htod(a_gpu, a_host)

# Launch Kernel (Grid=1, Block=4)
func = mod.get_function("vecAdd")
func(a_gpu, b_gpu, block=(4,1,1))
```
