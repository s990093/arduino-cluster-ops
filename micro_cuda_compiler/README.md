# Micro-CUDA Compiler

**將 CUDA-like C++ 編譯到 ESP32 CUDA VM 的編譯器工具鏈**

---

## 🎯 專案概述

Micro-CUDA Compiler (MCC) 是一個完整的編譯器工具鏈，讓你能夠：

1. 使用熟悉的 **CUDA-style C++** 撰寫並行程式
2. 自動編譯為 **Micro-CUDA ISA v1.5** 機器碼
3. 在 **ESP32 8-lane SIMD 引擎**上執行
4. 獲得真正的 **資料並行**性能

## 📂 專案結構

```
micro_cuda_compiler/
├── __init__.py           # Package 定義
├── mcuda.h               # CUDA 模擬 header (C++)
├── mcc.py                # 編譯器後端 (LLVM IR → Micro-CUDA ISA)
├── compile_kernel.py     # 編譯前端腳本
├── run_kernel.py         # Kernel 執行框架
└── kernels/              # 範例 kernel 目錄
    └── vector_add.cpp    # 向量加法範例
```

## 🚀 快速開始

### 1. 編譯 Kernel

```bash
python micro_cuda_compiler/compile_kernel.py \
    micro_cuda_compiler/kernels/vector_add.cpp
```

### 2. 執行 Kernel

```bash
# 連接 ESP32 並執行 demo
python micro_cuda_compiler/run_kernel.py --demo
```

### 3. 查看結果

```
======================================================================
Kernel Demo: Vector Addition (C = A + B)
======================================================================

Input A: [2, 3, 4, 5, 6, 7, 8, 9]
Input B: [1, 2, 3, 4, 5, 6, 7, 8]
Expected C: [3, 5, 7, 9, 11, 13, 15, 17]

[Verify] ✅ All 8 results match!

✅ Kernel execution successful!
======================================================================
```

## 📖 文檔

完整使用指南請參閱：[docs/MCC_GUIDE.md](../docs/MCC_GUIDE.md)

包含：

- 安裝與配置
- 編譯流程詳解
- Kernel 撰寫指南
- 進階主題
- 故障排除

## 🔧 工作流程

```
┌──────────────┐
│ User Code    │  vectorAdd.cpp (CUDA-like C++)
│ (.cpp / .cu) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Clang     │  C++ → LLVM IR
│   Frontend   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLVM IR     │  .ll (Intermediate Representation)
│    (.ll)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ MCC Backend  │  LLVM IR → Micro-CUDA ISA
│   (mcc.py)   │  - Instruction Selection
│              │  - Register Allocation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Micro-CUDA   │  .asm (Human-readable assembly)
│  Assembly    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ESP32 VM    │  Execute on 8-lane SIMD
│  Execution   │
└──────────────┘
```

## 💡 範例：向量加法

### C++ Code

```cpp
#include "mcuda.h"

__global__ void vectorAdd(int* A, int* B, int* C) {
    int idx = laneId();  // Get lane ID (0-7)
    C[idx] = A[idx] + B[idx];
}
```

### 生成的組合語言

```assembly
S2R R31, SR_LANEID     ; R31 = lane ID
MOV R0, 0              ; Base address of A
LDL R10, [R0]          ; R10 = A[lane] (SIMT load)
MOV R1, 32             ; Base address of B
LDL R11, [R1]          ; R11 = B[lane]
IADD R12, R10, R11     ; R12 = A + B
MOV R2, 64             ; Base address of C
STL [R2], R12          ; C[lane] = result (SIMT store)
EXIT
```

### 執行結果

8 個 SIMD lanes 並行執行，一次處理 8 個元素！

## 🎓 支援的功能

### ✅ 已實現

- [x] C++ → LLVM IR 編譯（via Clang）
- [x] 基本 IR 解析
- [x] 整數/浮點運算指令選擇
- [x] Lane ID intrinsic (`laneId()`)
- [x] 線性掃描暫存器分配
- [x] 組合語言輸出
- [x] Kernel 執行框架

### 🚧 開發中

- [ ] `load`/`store` 指令選擇
- [ ] SIMT 記憶體模式自動偵測
- [ ] `__syncthreads()` 支援
- [ ] Assembly 解析器
- [ ] 二進位 hex 輸出
- [ ] 更多 SFU 函數支援

### 🎯 未來計劃

- [ ] 迴圈展開優化
- [ ] 分支預測與 divergence 最小化
- [ ] Shared memory 支援
- [ ] 圖形著色暫存器分配
- [ ] 性能分析工具

## 🤝 貢獻

這是一個**研究級專案**（Master's thesis level）！歡迎貢獻：

- 🐛 Bug 回報
- 📝 改進文檔
- ✨ 新功能實作
- 📚 更多範例 kernel

## 📚 相關專案

- [ESP32 CUDA VM](../examples/esp32_cuda_vm/) - 執行環境
- [Micro-CUDA ISA v1.5](../docs/MICRO_CUDA_ISA_V15_SPEC.md) - 指令集規格
- [ISA 完整指南](../examples/esp32_cuda_vm/docs/ISA_GUIDE.md) - ISA 使用文檔

## 📜 授權

與 arduino-cluster-ops 專案一致

---

**狀態**: 🚧 Alpha 開發中  
**版本**: 0.1.0  
**最後更新**: 2025-12-13
