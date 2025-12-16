# Micro-CUDA 編譯器快速指南

## ✅ 已完成功能

1. **Target Configuration** - 硬體配置管理
2. **`.cu` 文件支援** - 像真正的 CUDA 一樣使用 `.cu` 擴展名
3. **編譯輸出包含硬體參數** - `.asm` 文件自動包含 ESP32 配置信息

## 🚀 快速使用

### 1. 列出可用的硬體配置

```bash
python micro_cuda_compiler/compile_kernel.py --list-targets
```

輸出：

```
Available Target Configurations:

  default         - ESP32 CUDA VM
                   VRAM: 40 KB, Lanes: 8, CPU: 240 MHz
  esp32           - ESP32 (Standard)
                   VRAM: 32 KB, Lanes: 8, CPU: 240 MHz
  esp32-psram     - ESP32 with 2MB PSRAM
                   VRAM: 100 KB, Lanes: 8, CPU: 240 MHz
  esp32s3         - ESP32-S3 with 8MB PSRAM
                   VRAM: 1024 KB, Lanes: 8, CPU: 240 MHz

Usage: --target <name>
```

### 2. 編譯 Kernel（使用 .cu 擴展名）

```bash
# 使用默認配置 (40 KB VRAM)
python micro_cuda_compiler/compile_kernel.py \
    micro_cuda_compiler/kernels/vector_add.cu

# 使用 ESP32-S3 配置 (1 MB VRAM)
python micro_cuda_compiler/compile_kernel.py \
    micro_cuda_compiler/kernels/vector_add.cu \
    --target esp32s3

# 使用標準 ESP32 配置 (32 KB VRAM)
python micro_cuda_compiler/compile_kernel.py \
    micro_cuda_compiler/kernels/vector_add.cu \
    --target esp32
```

### 3. 查看生成的 Assembly

```bash
cat micro_cuda_compiler/kernels/vector_add.asm
```

**範例輸出**（包含完整的硬體配置 information）：

```assembly
; ====================================================================
; Micro-CUDA Kernel - Compiled Assembly
; ====================================================================
;
; Target Configuration:
;   Device:        ESP32-S3 with 8MB PSRAM
;   ISA Version:   v1.5
;   Architecture:  Dual-Core SIMT
;
; SIMD Configuration:
;   Lanes:         8
;   Warp Size:     8
;
; Memory Configuration:
;   VRAM Size:     1048576 bytes (1024 KB)
;   Program Size:  1024 instructions
;   Stack Size:    8192 bytes
;
; Register Configuration (per lane):
;   GP Registers:  R0-R31 (32 × 32-bit)
;   FP Registers:  F0-F31 (32 × 32-bit)
;   Predicates:    P0-P7 (8 × 1-bit)
;   System Regs:   SR_0 - SR_9
;
; Communication:
;   Serial Baud:   115200
;   CPU Freq:      240 MHz
;
; Performance:
;   Typical Speed: ~30,000 inst/sec
;
; ====================================================================

; Source File: vector_add.cu
; Kernel Functions: vectorAdd
; Total Instructions: 12
; Registers Used: 8
;
; ====================================================================

; ===== CODE SECTION =====

S2R R31, SR_LANEID  ; laneId() -> R31
...
EXIT  ; Return from kernel

; ===== END OF KERNEL =====
```

### 4. 執行 Kernel

```bash
python micro_cuda_compiler/run_kernel.py --demo
```

## 📋 工作流程（完全像 NVCC）

```
User Code (.cu)  ──▶  Compile  ──▶  Assembly (.asm)  ──▶  Execute on ESP32
vector_add.cu        (with target)   (with hw config)      (8-lane SIMD)
```

### 完整範例

```bash
# Step 1: 撰寫 CUDA kernel
cat > my_kernel.cu << 'EOF'
#include "mcuda.h"

__global__ void myKernel(int* A, int* B, int* C) {
    int idx = laneId();
    C[idx] = A[idx] * B[idx];
}
EOF

# Step 2: 編譯 (指定硬體配置)
python micro_cuda_compiler/compile_kernel.py my_kernel.cu \
    --target esp32s3

# Step 3: 查看生成的 assembly (包含硬體參數)
cat my_kernel.asm

# Step 4: 執行
python micro_cuda_compiler/run_kernel.py --demo
```

## 🎯 Target 配置詳解

### Default Target

- Device: ESP32 CUDA VM
- VRAM: 40 KB
- Best for: 教學、基本測試

### ESP32 Standard

- Device: ESP32 (Standard)
- VRAM: 32 KB
- Best for: 標準 ESP32 無 PSRAM

### ESP32 with PSRAM

- Device: ESP32 with 2MB PSRAM
- VRAM: 100 KB
- Best for: 中型應用

### ESP32-S3

- Device: ESP32-S3 with 8MB PSRAM
- VRAM: 1024 KB (1 MB)
- Best for: 大型 AI 模型、複雜運算

## 📝 技術細節

### Kernel 文件命名

- **必須使用 `.cu` 擴展名**（像真正的 CUDA）
- 編譯器會自動處理（使用 `-x c++` 告訴 Clang）

### 生成的 Assembly 包含

1. **完整的硬體配置 header**

   - Device 型號
   - VRAM 大小
   - Lane 數量
   - 暫存器配置
   - CPU 頻率
   - 性能指標

2. **Source 資訊**

   - 原始檔案名
   - Kernel 函數列表
   - 指令數量
   - 使用的暫存器數量

3. **實際的組合語言程式碼**

### Driver 參數記錄

編譯時，target configuration 會自動記錄：

- `VM_VRAM_SIZE`: 配置的 VRAM 大小
- `VM_PROGRAM_SIZE`: Instruction memory 大小
- `num_lanes`: SIMD lane 數量
- `warp_size`: Warp 大小
- `baud_rate`: 串口波特率
- `cpu_freq_mhz`: CPU 時脈

所有這些參數都會寫入生成的 `.asm` 文件的 header 中！

## 🔧 當前狀態

- ✅ Target configuration 系統完成
- ✅ `.cu` 文件支援
- ✅ Hardware parameter header generation
- ✅ Multiple target 支援
- 🚧 完整的 IR → ISA 編譯（開發中）

## 下一步開發

1. 完善 IR Parser（處理更多 LLVM 指令）
2. 實作 load/store instruction selection
3. 自動 SIMT 模式偵測
4. Assembly parser（讀取 .asm 並執行）

---

**版本**: 0.1.0 Alpha  
**更新**: 2025-12-13  
**狀態**: Target configuration ✅ | Compiler 🚧
