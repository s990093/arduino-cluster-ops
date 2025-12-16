# ESP32 CUDA VM - Host Interface & Turbo Mode

本文件說明 Host (PC/Python) 與 ESP32 VM (Core 0/Front-End) 之間的高速通訊協定與 Turbo Mode 配置。

---

## 1. Turbo Mode 配置

為了達成高效的 Kernel Loading 與大量資料傳輸 (DMA)，本系統啟用了 **Turbo Mode**。

### 1.1 關鍵參數 (vm_config.h)

| 參數                    | 值          | 說明                                        |
| :---------------------- | :---------- | :------------------------------------------ |
| **`VM_BAUD_RATE`**      | **460800**  | 序列通訊速率 (原 115200 的 4 倍)            |
| **`VM_SERIAL_RX_SIZE`** | **32768**   | (32KB) RX 緩衝區，防止高速 Burst Write 溢出 |
| **`VM_CPU_FREQ`**       | **240 MHz** | 鎖定 CPU 最高頻率以確保 IO 吞吐             |

### 1.2 效能指標

- **理論頻寬**: ~46 KB/s (460800 bps / 10 bits per char)
- **實測 Throughput**: **~40.8 KB/s** (16KB Payload)
- **傳輸耗時**: 1KB kernel binary < 30ms

---

## 2. 通訊協定 (Protocol)

Host 與 Device 之間採用 **ASCII Command + Binary Burst** 的混合模式。

### 2.1 Host-to-Device DMA (`dma_h2d`)

用於將大量資料寫入 VRAM。

1. **Request**: Host 發送 `dma_h2d <hex_addr> <dec_size>\n`
2. **Handshake**: Device 檢查空間後回傳 `ACK_DMA_GO:<size>`
3. **Transmission**: Host **連續發送** (Burst) 指定長度的 Binary Data。
   - _注意_: Device 端採用 Block/Chunk 讀取模式，不應有字元間延遲。
4. **Completion**: Device 接收完畢後回傳 `DMA_OK`。

### 2.2 Kernel Loading (`load_imem`)

用於快速載入指令記憶體 (Instruction Memory)。

1. **Request**: Host 發送 `load_imem <byte_count>\n`
2. **Handshake**: Device 回傳 `ACK_KERN_GO:<byte_count>`
3. **Execution**: Host 發送 Binary Instruction Data (_Little Endian Packed_)。
4. **Completion**: Device 回傳 `KERN_OK` 並更新 `VM_PROGRAM_SIZE`。

---

## 3. CLI 指令集

系統啟動後進入 Front-End CLI 模式，支援以下指令：

| 指令                | 格式                   | 說明                                   |
| :------------------ | :--------------------- | :------------------------------------- |
| **`load_imem`**     | `load_imem <bytes>`    | 高速載入 Kernel Binary (Turbo)         |
| **`dma_h2d`**       | `dma_h2d <addr> <len>` | Host 到 Device 的 DMA 傳輸 (Turbo)     |
| **`dma_d2h`**       | `dma_d2h <addr> <len>` | Device 到 Host 的記憶體讀取 (Hex Dump) |
| **`kernel_launch`** | `kernel_launch`        | 啟動 Kernel 執行 (Blocking)            |
| **`gpu_reset`**     | `gpu_reset`            | 重置 VM 狀態與暫存器 (保留 VRAM)       |
| **`reg`**           | `reg <lane_id>`        | 顯示特定 Lane 的暫存器內容             |
| **`stats`**         | `stats`                | 顯示 VM 統計資訊 (PC, VRAM Size)       |
| **`trace:stream`**  | `trace:stream`         | 開啟即時 Trace 輸出 (會影響效能)       |

---

## 4. Python Client 範例

連接時務必指定正確的 Baud Rate 與 Buffer Size：

```python
import serial

# Turbo Configuration
BAUD_RATE = 460800
PORT = "/dev/cu.usbserial-..."

ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
# 建議加大 PC 端緩衝區
ser.set_buffer_size(rx_size=32768, tx_size=32768)

# 發送 DMA
ser.write(b"dma_h2d 400 1024\n")
while "ACK" not in ser.readline().decode(): pass
ser.write(binary_data_1024_bytes)
print(ser.readline().decode()) # DMA_OK
```
