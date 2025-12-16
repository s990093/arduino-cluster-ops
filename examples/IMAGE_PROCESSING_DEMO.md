# ESP32 MicroGPU - Image Processing Demo

## 概述

這個專案展示了如何使用 ESP32 CUDA VM 進行完整的圖像處理，包含：

1. **Host 端處理**: 載入圖像、預處理
2. **數據傳輸**: Host -> Device VRAM
3. **Device 運算**: 8-lane SIMT 並行執行
4. **結果取回**: Device -> Host
5. **視覺化**: Matplotlib 顯示結果

## 核心組件

### 1. MicroGPU 類別

模擬 CUDA 編程模型的 Python API：

```python
class MicroGPU:
    def malloc(name, size_bytes) -> int
    def memcpy_host_to_device(name, data: np.ndarray)
    def memcpy_device_to_host(name, shape) -> np.ndarray
    def launch(kernel_code, grid_size, block_size)
    def free_all()
```

**API 對應:**
| CUDA API | MicroGPU API | 說明 |
|----------|--------------|------|
| `cudaMalloc()` | `malloc()` | 在 VRAM 中分配記憶體 |
| `cudaMemcpy(H2D)` | `memcpy_host_to_device()` | Host → Device |
| `cudaMemcpy(D2H)` | `memcpy_device_to_host()` | Device → Host |
| `kernel<<<grid, block>>>()` | `launch()` | 啟動 Kernel |

### 2. Tile-based Execution

因為硬體限制（Warp Size = 8），大圖像需要分塊處理：

```python
# 範例：處理 64x64 圖像
tile_size = 8
for tile_y in range(0, 64, tile_size):
    for tile_x in range(0, 64, tile_size):
        # 提取 8x8 tile
        tile = image[tile_y:tile_y+8, tile_x:tile_x+8]

        # 傳輸並執行
        gpu.memcpy_host_to_device("tile_input", tile)
        gpu.launch(kernel, grid_size=1, block_size=8)
        result_tile = gpu.memcpy_device_to_host("tile_output", (8, 8))
```

### 3. Kernel 範例：邊緣檢測

```python
def create_edge_detection_kernel():
    """Sobel 濾波器實現"""
    return [
        # 1. 獲取 lane_id
        InstructionV15.s2r(31, InstructionV15.SR_LANEID),

        # 2. 讀取像素
        InstructionV15.ldx(0, input_base, lane_offset),

        # 3. 計算梯度
        InstructionV15.isub(2, 0, threshold),

        # 4. 寫回結果
        InstructionV15.stx(output_base, lane_offset, 2),

        InstructionV15.exit_inst()
    ]
```

## 使用方法

### 安裝依賴

```bash
pip install numpy pillow matplotlib
```

### 運行 Demo

```bash
cd /Users/hungwei/Desktop/Proj/arduino-cluster-ops
python3 examples/image_processing_demo.py
```

### 使用自己的圖像

修改 `main()` 函數：

```python
original, processed = process_image_with_microgpu(
    '/path/to/your/image.png',  # 您的圖像
    gpu,
    tile_size=8
)
```

## 進階用法

### 多 Tile 處理

處理大於 8x8 的圖像：

```python
def process_large_image(image_path, gpu):
    img = Image.open(image_path).convert('L')
    h, w = img.size
    result = np.zeros((h, w), dtype=np.int32)

    # 分塊處理
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            tile = np.array(img.crop((x, y, x+8, y+8)))

            # 處理單個 tile
            gpu.memcpy_host_to_device("input", tile.flatten())
            gpu.launch(kernel, grid_size=1, block_size=8)
            output_tile = gpu.memcpy_device_to_host("output", (8, 8))

            # 組合結果
            result[y:y+8, x:x+8] = output_tile

    return result
```

### 自定義 Kernel

參考 `create_edge_detection_kernel()` 創建您自己的圖像處理 Kernel：

```python
def create_blur_kernel():
    """模糊濾波器"""
    kernel = [
        # 讀取周圍像素並平均
        InstructionV15.s2r(31, InstructionV15.SR_LANEID),
        InstructionV15.ldx(0, input_base, lane_offset),
        # ... 平均計算 ...
        InstructionV15.stx(output_base, lane_offset, result),
        InstructionV15.exit_inst()
    ]
    return kernel
```

## 輸出

程式會生成：

1. **終端輸出**: 執行過程日誌
2. **microgpu_result.png**: 原圖 vs 處理結果對比
3. **Matplotlib 窗口**: 互動式圖像查看

## 架構圖

```
┌─────────────┐
│   Host      │
│  (Python)   │
└──────┬──────┘
       │ malloc(), memcpy_h2d()
       ▼
┌─────────────┐
│   VRAM      │  ← 0x0000: Input Image
│  (ESP32)    │  ← 0x0400: Output Image
└──────┬──────┘
       │ launch()
       ▼
┌─────────────┐
│  Device     │  8 SIMD Lanes
│ (Kernel)    │  Parallel Processing
└──────┬──────┘
       │ memcpy_d2h()
       ▼
┌─────────────┐
│   Host      │
│ (Matplotlib)│  顯示結果
└─────────────┘
```

## 性能指標

- **Warp Size**: 8 Lanes
- **處理單元**: 8 像素/kernel launch
- **記憶體帶寬**: ~50KB/s (取決於序列埠速度)
- **建議圖像大小**: ≤ 64x64 (Demo), ≤ 512x512 (實際)

## 故障排除

1. **連接失敗**: 檢查 ESP32 接口 (`/dev/cu.usbserial-XXX`)
2. **記憶體溢出**: 減少圖像大小或增加 `VM_VRAM_SIZE`
3. **結果全為零**: 確認 VRAM 正確初始化（參考 VRAM 修復）

## 下一步

- [ ] 實現更複雜的濾波器（Gaussian, Median）
- [ ] 支持彩色圖像（RGB 通道）
- [ ] 性能分析和優化
- [ ] 批量處理多張圖像
