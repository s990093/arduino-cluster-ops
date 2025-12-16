# MCC Run - 完整執行工具

## 🎯 像 CUDA 一樣直接執行 .cu 文件

`mcc_run.py` 提供端到端的 kernel 執行體驗：

```bash
# 像 nvcc 一樣簡單！
python mcc_run.py kernels/my_kernel.cu

# 完成所有步驟：
# ✅ 編譯 .cu → .asm
# ✅ 解析 .asm → Instructions
# ✅ 連接 ESP32
# ✅ 初始化 VRAM
# ✅ 載入程式
# ✅ 執行
# ✅ 讀取結果
# ✅ 顯示輸出
```

## 🚀 快速開始

### 基本用法

```bash
# 執行 kernel
python mcc_run.py micro_cuda_compiler/kernels/conv1d_manual.cu
```

**輸出**：

```
🚀 🚀 🚀 MCC Run: conv1d_manual.cu 🚀 🚀 🚀

======================================================================
🔨 Step 1: Compiling Kernel
======================================================================
✅ Assembly generated: conv1d_manual.asm

======================================================================
📜 Step 2: Parsing Assembly
======================================================================
✅ Loaded 16 instructions

======================================================================
🔌 Step 3: Connecting to ESP32
======================================================================
✅ Connected to /dev/cu.usbserial-589A0095521

======================================================================
💾 Step 4: Initializing VRAM
======================================================================
Writing input: [1, 2, 3, 4, 5, 6, 7, 8]...
Writing kernel: [2, 3, 4]
✅ VRAM initialized

======================================================================
⚡ Step 5: Executing Kernel
======================================================================
✅ Program loaded
Running on 8-lane SIMD engine...
✅ Execution complete

======================================================================
📊 Step 6: Reading Results
======================================================================
Results: [20, 29, 38, 47, 56, 65, 74, 83]

======================================================================
✅ Execution Complete!
======================================================================
Output: [20, 29, 38, 47, 56, 65, 74, 83]
```

### 指定 Target

```bash
# 為 ESP32-S3 編譯並執行
python mcc_run.py kernels/my_kernel.cu --target esp32s3
```

### 指定串口

```bash
# 使用不同的串口
python mcc_run.py kernels/my_kernel.cu --port /dev/ttyUSB0
```

### 自定義 VRAM 數據

創建 `vram_data.json`：

```json
{
  "input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  "kernel": [2, 3, 4],
  "bias": [10]
}
```

執行：

```bash
python mcc_run.py kernels/conv1d.cu --vram-init vram_data.json
```

### 啟用追蹤

```bash
# 查看詳細的執行追蹤
python mcc_run.py kernels/my_kernel.cu --trace
```

### 安靜模式

```bash
# 減少輸出
python mcc_run.py kernels/my_kernel.cu --quiet
```

## 📖 命令行參數

```
usage: mcc_run.py [-h] [--port PORT] [--target TARGET]
                  [--vram-init VRAM_INIT] [--trace] [-q]
                  kernel

positional arguments:
  kernel                Kernel file (.cu)

options:
  -h, --help            show this help message
  --port PORT           ESP32 serial port
  --target TARGET       Target configuration
                        (default, esp32, esp32-psram, esp32s3)
  --vram-init VRAM_INIT VRAM initialization data (JSON file)
  --trace               Enable execution trace
  -q, --quiet           Quiet mode (less output)
```

## 🎯 完整工作流程

```
1. 編譯
   .cu 文件 → Clang → LLVM IR → MCC → .asm

2. 解析
   .asm → Assembly Parser → InstructionV15[]

3. 連接
   建立與 ESP32 的串口連接

4. 初始化 VRAM
   寫入測試數據到 VRAM

5. 載入程式
   將指令上傳到 ESP32

6. 執行
   在 8-lane SIMD 引擎上運行

7. 讀取結果
   從 VRAM 讀取輸出

8. 顯示
   打印結果到控制台
```

## 💡 使用場景

### 1. 快速測試 Kernel

```bash
# 修改 kernel
vim kernels/my_kernel.cu

# 立即測試
python mcc_run.py kernels/my_kernel.cu
```

### 2. CI/CD Pipeline

```bash
#!/bin/bash
# test_kernels.sh

for kernel in kernels/*.cu; do
    echo "Testing $kernel..."
    python mcc_run.py "$kernel" --quiet || exit 1
done

echo "All kernels passed!"
```

### 3. 性能測試

```bash
# 使用追蹤模式測試性能
python mcc_run.py kernels/matmul.cu --trace > perf.log

# 分析週期數
grep "Cycles:" perf.log
```

### 4. 不同硬體配置測試

```bash
# 測試在不同 target 上的行為
for target in default esp32 esp32-psram esp32s3; do
    echo "Testing on $target..."
    python mcc_run.py kernels/my_kernel.cu --target $target
done
```

## 📝 範例 Kernel

### Vector Add

```cuda
// kernels/vector_add.cu
#include "../mcuda.h"

__global__ void vectorAdd(int* A, int* B, int* C) {
    int idx = laneId();
    C[idx] = A[idx] + B[idx];
}
```

執行：

```bash
python mcc_run.py kernels/vector_add.cu
```

### 1D Convolution

```cuda
// kernels/conv1d.cu
#include "../mcuda.h"

__global__ void conv1d(int* input, int* kernel, int* output) {
    int lane = laneId();

    int i0 = input[lane];
    int i1 = input[lane + 1];
    int i2 = input[lane + 2];

    int k0 = kernel[0];
    int k1 = kernel[1];
    int k2 = kernel[2];

    int result = i0*k0 + i1*k1 + i2*k2;
    output[lane] = result;
}
```

執行：

```bash
python mcc_run.py kernels/conv1d.cu
```

## 🔧 VRAM Memory Layout

默認記憶體布局：

| Region | Address | Size     | Description          |
| ------ | ------- | -------- | -------------------- |
| input  | 0x00    | 48 bytes | Input data (12 ints) |
| kernel | 0x40    | 32 bytes | Kernel weights       |
| output | 0x80    | 32 bytes | Output results       |

可以通過 `--vram-init` 自定義。

## 🎓 與 CUDA 對比

### NVIDIA CUDA:

```bash
# 編譯
nvcc my_kernel.cu -o my_kernel

# 執行
./my_kernel
```

### Micro-CUDA:

```bash
# 編譯 + 執行（一步完成）
python mcc_run.py my_kernel.cu
```

## 🚧 當前限制

1. **記憶體模型**：目前支援固定的記憶體布局
2. **Intrinsics**：部分 intrinsic 函數尚未完全實現
3. **動態記憶體**：不支援動態記憶體分配
4. **多 Kernel**：一次只能執行一個 kernel 函數

## 📈 未來改進

- [ ] 支援自定義記憶體布局配置
- [ ] 自動驗證結果（與預期值比較）
- [ ] 性能分析報告
- [ ] 批次執行多個 kernels
- [ ] 生成執行報告（JSON/HTML）

---

**版本**: 1.0.0  
**狀態**: 可用 ✅  
**更新**: 2025-12-13
