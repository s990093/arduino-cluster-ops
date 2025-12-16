# LZ4 Integration Guide for ESP32 CUDA VM

This guide details how to integrate the LZ4 compression library into the ESP32 CUDA VM firmware to accelerate kernel uploads.

## 1. Library Installation

1.  **Clone the LZ4 Repository**:

    ```bash
    git clone https://github.com/lz4/lz4
    ```

2.  **Copy Source Files**:
    Copy `lib/lz4.c` and `lib/lz4.h` from the cloned repository to your firmware directory (`examples/esp32_cuda_vm/`).

3.  **Patch for Arduino/ESP32**:
    Ensure `lz4.c` can compile. Usually, including `Arduino.h` is not strictly necessary as LZ4 is pure C, but if you encounter type issues, add `#include <Arduino.h>` to the top.

## 2. Firmware Integration

### Include Header

In `esp32_cuda_vm.ino`:

```cpp
#include "lz4.h"
```

### Implement Decompression Logic

Add a handler function `handle_compressed_load` (or similar) that:

1.  Receives a "Block Header" (2 bytes) indicating compressed size.
2.  Reads the compressed data block.
3.  Uses `LZ4_decompress_safe` to decompress into a buffer.
4.  Writes the decompressed instruction data to I-RAM (Instruction Memory).

### Add Serial Command

Register a new command `load_imem_lz4 <uncompressed_size>` to trigger the compressed upload mode.

## 3. Host-Side Implementation (Python)

### Compression Logic

Use the `lz4` library in Python:

```python
import lz4.block
compressed_data = lz4.block.compress(chunk_data, store_size=False)
```

### Upload Protocol

1.  Send `load_imem_lz4 {total_uncompressed_size}\n`.
2.  Wait for `ACK_LZ4_GO`.
3.  Send blocks: `[2-byte Length] + [Compressed Data]`.
4.  Wait for `LZ4_LOAD_OK`.

## 4. Performance Benefits

LZ4 is extremely fast at decompression. For repetitive data like machine code kernels (with repeated opcodes), it can achieve significant compression ratios, reducing the UART transfer bottleneck significantly.
