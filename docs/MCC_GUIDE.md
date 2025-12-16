# Micro-CUDA 編譯器使用指南 (MCC Guide)

**版本**: 0.1.0 Alpha  
**更新日期**: 2025-12-13  
**專案狀態**: 🚧 開發中

---

## 📋 目錄

1. [簡介](#簡介)
2. [安裝與配置](#安裝與配置)
3. [快速開始](#快速開始)
4. [編譯流程詳解](#編譯流程詳解)
5. [執行 Kernel](#執行-kernel)
6. [撰寫 Kernel 指南](#撰寫-kernel-指南)
7. [進階主題](#進階主題)
8. [故障排除](#故障排除)

---

## 簡介

Micro-CUDA 編譯器 (MCC) 是一個將 CUDA-like C/C++ 程式碼編譯為 Micro-CUDA ISA v1.5 機器碼的工具鏈。它讓你能夠：

- 使用熟悉的 C++ 語法撰寫並行 kernel
- 自動編譯到 ESP32 可執行的 ISA 指令
- 在 8-lane SIMD 引擎上執行並行運算
- 獲得類似 CUDA 的編程體驗

### 架構概覽

```
User Code (C++)  ──▶  Clang  ──▶  LLVM IR  ──▶  MCC Backend  ──▶  Micro-CUDA ISA
  (vectorAdd.cpp)              (.ll)               (mcc.py)           (.asm / .hex)
                                                                            │
                                                                            ▼
                      Host (Python)  ◀──────────────────────────  ESP32 CUDA VM
                      - 設置 VRAM                                   - 8-Lane SIMD
                      - 載入 Kernel                                 - Instruction Memory
                      - 執行                                        - VRAM (4KB-64KB)
                      - 讀取結果
```

---

## 安裝與配置

### 必要軟體

1. **LLVM/Clang** (用於產生 LLVM IR)

   ```bash
   # macOS
   brew install llvm

   # Ubuntu/Debian
   sudo apt-get install clang llvm

   # 驗證安裝
   clang --version
   ```

2. **Python 3.8+**

   ```bash
   python3 --version
   ```

3. **ESP32 CUDA VM 韌體**

   確保你的 ESP32 已燒錄 `esp32_cuda_vm` 韌體。參考：

   - [ESP32 CUDA VM 文件](../examples/esp32_cuda_vm/docs/ISA_GUIDE.md)

### 專案結構

```
arduino-cluster-ops/
├── micro_cuda_compiler/      # 編譯器核心
│   ├── __init__.py           # Package 定義
│   ├── mcuda.h               # CUDA 模擬 header (C++)
│   ├── mcc.py                # 編譯器後端 (LLVM IR → ISA)
│   ├── compile_kernel.py     # 編譯前端腳本
│   ├── run_kernel.py         # Kernel 執行框架
│   └── kernels/              # 範例 kernel 目錄
│       └── vector_add.cpp    # 向量加法範例
├── esp32_tools/              # ESP32 連接工具
└── docs/                    # 文件
    └── MCC_GUIDE.md         # 本文件
```

---

## 快速開始

### 步驟 1: 編譯第一個 Kernel

```bash
cd /path/to/arduino-cluster-ops

# 編譯範例 kernel
python micro_cuda_compiler/compile_kernel.py \
    micro_cuda_compiler/kernels/vector_add.cpp
```

**預期輸出**：

```
======================================================================
Micro-CUDA Kernel Compiler
======================================================================
Input:  micro_cuda_compiler/kernels/vector_add.cpp
Output: micro_cuda_compiler/kernels/vector_add.asm

[Clang] Compiling vector_add.cpp...
  Command: clang -S -emit-llvm -O1 --target=riscv32 ...
[Clang] ✓ LLVM IR generated: vector_add.ll

[MCC] Compiling IR to Micro-CUDA ISA...
[MCC] ✓ Assembly generated: vector_add.asm

======================================================================
✅ Compilation complete! Assembly: vector_add.asm
======================================================================

Next steps:
  1. Review assembly: cat vector_add.asm
  2. Run kernel: python run_kernel.py vector_add.asm
```

### 步驟 2: 查看生成的組合語言

```bash
cat micro_cuda_compiler/kernels/vector_add.asm
```

**範例輸出**：

```assembly
; Micro-CUDA Assembly
; Compiled from: vector_add.cpp
; ===== Code =====

S2R R31, SR_LANEID     ; laneId() -> R31
MOV R0, 0              ; Base address A
LDL R10, [R0]          ; Load A[lane]
MOV R1, 32             ; Base address B
LDL R11, [R1]          ; Load B[lane]
IADD R12, R10, R11     ; R12 = A + B
MOV R2, 64             ; Base address C
STL [R2], R12          ; Store result
EXIT                   ; Return from kernel
```

### 步驟 3: 在 ESP32 上執行

```bash
# 連接 ESP32，並執行內建的 demo
python micro_cuda_compiler/run_kernel.py --demo
```

**預期輸出**：

```
[Connection] Connecting to /dev/cu.usbserial-XXX...
[Connection] ✓ Connected at 115200 baud

======================================================================
Kernel Demo: Vector Addition (C = A + B)
======================================================================

Input A: [2, 3, 4, 5, 6, 7, 8, 9]
Input B: [1, 2, 3, 4, 5, 6, 7, 8]
Expected C: [3, 5, 7, 9, 11, 13, 15, 17]

[VRAM] Setting up device memory...
[VRAM] ✓ Region 'A' written to 0x0000 (8 elements)
[VRAM] ✓ Region 'B' written to 0x0020 (8 elements)

[Program] Loading 9 instructions...
[Program] ✓ Program loaded

[Execute] Running kernel on 8-lane SIMD engine...
[Execute] ✓ Execution complete
[Execute] Cycles: 9

[Results] Reading from VRAM 0x0040...
[Results] ✓ Read 8 values: [3, 5, 7, 9, 11, 13, 15, 17]

[Verify] Checking results...
[Verify] ✅ All 8 results match!

======================================================================
✅ Kernel execution successful!
======================================================================
```

---

## 編譯流程詳解

### 完整編譯管線

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: C++ Preprocessing                                      │
│   - Include mcuda.h                                            │
│   - Expand macros (__global__, laneId(), etc.)                 │
└──────────────────────┬─────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 2: Clang Frontend (C++ → LLVM IR)                        │
│   - Parse C++ syntax                                           │
│   - Type checking                                              │
│   - Generate LLVM Intermediate Representation                  │
│   - Output: .ll file (human-readable IR)                       │
└──────────────────────┬─────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 3: MCC Backend (LLVM IR → Micro-CUDA ISA)                │
│   ├─ IR Parser: Read .ll file                                 │
│   ├─ Instruction Selection: Map IR → ISA instructions         │
│   │   Examples:                                                │
│   │   - %add = add i32 %0, %1  →  IADD R1, R2, R3            │
│   │   - %mul = mul i32 %0, %1  →  IMUL R4, R2, R3            │
│   │   - call @__mcuda_lane_id  →  S2R R31, SR_LANEID         │
│   ├─ Register Allocation: Virtual registers → R0-R31          │
│   └─ Code Emission: Generate .asm file                        │
└──────────────────────┬─────────────────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 4: Assembly Output                                        │
│   - Human-readable assembly (.asm)                             │
│   - Future: Binary hex (.hex) for direct upload               │
└────────────────────────────────────────────────────────────────┘
```

### 使用選項

#### 只生成 LLVM IR

```bash
python micro_cuda_compiler/compile_kernel.py \
    kernels/vector_add.cpp --llvm-ir
```

這會產生 `vector_add.ll`，你可以查看中間表示：

```llvm
define void @vectorAdd(i32* %A, i32* %B, i32* %C) {
  %1 = call i32 @__mcuda_lane_id()
  %2 = getelementptr i32, i32* %A, i32 %1
  %3 = load i32, i32* %2
  ; ... more IR instructions
}
```

#### 指定輸出檔案

```bash
python micro_cuda_compiler/compile_kernel.py \
    kernels/vector_add.cpp -o my_kernel.asm
```

---

## 執行 Kernel

### 使用 `run_kernel.py`

`run_kernel.py` 提供完整的 kernel 執行框架，類似 `test_enhanced_trace.py`。

#### 基本用法

```bash
# 使用內建 demo
python micro_cuda_compiler/run_kernel.py --demo

# 指定串口
python micro_cuda_compiler/run_kernel.py --demo \
    --port /dev/cu.usbserial-YOUR_PORT

# 啟用 trace
python micro_cuda_compiler/run_kernel.py --demo --trace
```

### 執行流程

```python
# Pseudo-code showing the workflow

# 1. Connect to ESP32
conn = ESP32Connection(port)

# 2. Setup VRAM (Host → Device)
setup_vram({
    "A": [2, 3, 4, 5, 6, 7, 8, 9],
    "B": [1, 2, 3, 4, 5, 6, 7, 8]
})

# 3. Load Kernel
program = load_asm_kernel("vector_add.asm")
load_program(program)

# 4. Execute on Device
execute()

# 5. Read Results (Device → Host)
results = read_results(base_addr=0x40, count=8)

# 6. Verify
verify_results(results, expected)
```

---

## 撰寫 Kernel 指南

### Kernel 結構

```cpp
#include "../mcuda.h"

/**
 * Kernel 函數必須用 __global__ 標記
 * 參數通常是指標，指向 VRAM 中的資料
 */
__global__ void myKernel(int* input, int* output) {
    // 1. 獲取 Lane ID
    int idx = laneId();

    // 2. SIMT Load：每個 lane 讀取不同元素
    int data = input[idx];

    // 3. 計算（所有 lane 並行執行）
    int result = data * 2;

    // 4. SIMT Store：每個 lane 寫回結果
    output[idx] = result;
}
```

### 支援的語言特性

#### ✅ 已支援

| 特性         | 範例                | 說明                    |
| ------------ | ------------------- | ----------------------- |
| **整數運算** | `a + b`, `a * b`    | 加減乘除                |
| **浮點運算** | `float c = a + b`   | FP32 運算               |
| **Lane ID**  | `int id = laneId()` | 獲取當前 lane 編號      |
| **陣列索引** | `arr[laneId()]`     | SIMT 記憶體存取         |
| **函數呼叫** | `int x = myFunc(y)` | Device 函數（簡單內聯） |

#### 🚧 部分支援

| 特性         | 範例                      | 說明                           |
| ------------ | ------------------------- | ------------------------------ |
| **條件判斷** | `if (a > b) {...}`        | 編譯器會發出警告（Divergence） |
| **迴圈**     | `for (int i=0; i<8; i++)` | 小型迴圈可展開                 |
| **同步**     | `__syncthreads()`         | 尚未實作                       |

#### ❌ 不支援

| 特性           | 原因                       |
| -------------- | -------------------------- |
| 動態記憶體分配 | ESP32 VM 無 malloc         |
| 遞迴函數       | 無 stack 支援              |
| 複雜控制流     | Warp divergence 會降低效率 |
| C++ STL        | 需要大量 runtime support   |

### 記憶體模型

#### VRAM 佈局（由 `run_kernel.py` 管理）

```
0x0000 ┌─────────────────────┐
       │  Region A (Input)   │  32 bytes (8 int * 4)
0x0020 ├─────────────────────┤
       │  Region B (Input)   │  32 bytes
0x0040 ├─────────────────────┤
       │  Region C (Output)  │  32 bytes
0x0060 ├─────────────────────┤
       │  ... (User-defined) │
       └─────────────────────┘
```

#### SIMT 尋址規則

當你寫 `A[laneId()]` 時，編譯器會生成 **Lane-Based Load (LDL)**：

```
Lane 0: Load from addr = base_A + 0*4
Lane 1: Load from addr = base_A + 1*4
Lane 2: Load from addr = base_A + 2*4
...
Lane 7: Load from addr = base_A + 7*4
```

**關鍵**：所有 lane 在同一個 cycle 執行，達成真正的資料並行！

###範例 Kernel 模板

```cpp
#include "../mcuda.h"

/**
 * Template: Element-wise operation
 *
 * 適用於：map-style 運算，每個 lane 獨立處理一個元素
 */
__global__ void elementWiseOp(int* input, int* output, int scale) {
    int idx = laneId();

    int val = input[idx];        // SIMT Load
    int result = val * scale;    // Parallel Compute
    output[idx] = result;        // SIMT Store
}

/**
 * Template: Reduction (需要同步支援)
 *
 * ⚠️ 目前不支援，因為需要 __syncthreads()
 */
// __global__ void reduce(int* input, int* output) {
//     // Coming soon...
// }
```

---

## 進階主題

### 編譯器內部運作

#### IR 指令選擇範例

**LLVM IR**：

```llvm
%1 = call i32 @__mcuda_lane_id()
%2 = add i32 %0, %1
%3 = fadd float %4, %5
```

**Micro-CUDA ISA**：

```assembly
S2R R31, SR_LANEID      ; %1 = laneId()
IADD R1, R0, R31        ; %2 = %0 + %1
FADD F2, F4, F5         ; %3 = %4 + %5
```

#### 暫存器分配

MCC 使用**線性掃描分配器**（Linear Scan Register Allocator）：

1. 掃描 IR，建立虛擬暫存器列表
2. 按順序分配實體暫存器 R0-R31
3. 如果用完，報錯（未來會實作 spilling）

**範例**：

```
Virtual Regs    →    Physical Regs
%1              →    R1
%add            →    R2
%mul            →    R3
...
```

### 優化技巧

#### 1. 減少暫存器使用

**不好**：

```cpp
int a = input[idx];
int b = a * 2;
int c = b + 1;
int d = c * c;
output[idx] = d;
// 需要 5 個暫存器
```

**好**：

```cpp
int val = input[idx];
val = val * 2;
val = val + 1;
val = val * val;
output[idx] = val;
// 需要 1 個暫存器
```

#### 2. 避免分支

**不好**（Warp Divergence）：

```cpp
if (laneId() < 4) {
    output[laneId()] = input[laneId()] * 2;
} else {
    output[laneId()] = input[laneId()] * 3;
}
// 一半 lane 閒置
```

**好**（使用條件運算）：

```cpp
int multiplier = (laneId() < 4) ? 2 : 3;
output[laneId()] = input[laneId()] * multiplier;
// 所有 lane 都工作
```

---

## 故障排除

### 常見錯誤

#### 1. Clang not found

**錯誤訊息**：

```
[ERROR] Clang not found! Please install LLVM/Clang.
```

**解決方法**：

```bash
# macOS
brew install llvm
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# Linux
sudo apt-get install clang llvm
```

#### 2. Out of registers

**錯誤訊息**：

```
RuntimeError: Out of registers! Need more than 32
```

**解決方法**：

- 簡化 kernel，減少中間變數
- 重複使用暫存器（見優化技巧）

#### 3. Connection failed

**錯誤訊息**：

```
[Connection] ✗ Failed: [Errno 2] No such file or directory
```

**解決方法**：

```bash
# 列出可用串口
ls /dev/cu.* # macOS
ls /dev/ttyUSB* # Linux

# 使用正確的串口
python run_kernel.py --demo --port /dev/cu.usbserial-YOUR_PORT
```

### Debug 技巧

#### 1. 查看 LLVM IR

```bash
python compile_kernel.py kernels/vector_add.cpp --llvm-ir
cat kernels/vector_add.ll
```

這讓你看到編譯器的中間表示，有助於理解優化和指令選擇。

#### 2. 啟用 Trace

```bash
python run_kernel.py --demo --trace
```

這會顯示每條指令的執行細節。

#### 3. 手動檢查暫存器

執行後使用 `reg` 命令：

```bash
> reg 0    # 查看 Lane 0 的暫存器
> reg 1    # 查看 Lane 1 的暫存器
```

---

## 下一步

### 學習路徑

1. **初級**：

   - 編譯並執行 `vector_add.cpp`
   - 修改輸入資料，觀察結果
   - 撰寫簡單的 element-wise kernel

2. **中級**：

   - 實作 SAXPY (Scaled Vector Addition)
   - 實作簡單的 Attention 計算
   - 理解 SIMT 記憶體模型

3. **高級**：
   - 貢獻編譯器功能（見下方）
   - 優化指令選擇演算法
   - 實作進階 ISA 特性支援

### 參與貢獻

歡迎貢獻！重點開發領域：

- [ ] 完善 LLVM IR Parser（處理更多 IR 指令）
- [ ] 實作 `load`/`store` 指令選擇
- [ ] 自動偵測 SIMT 記憶體模式
- [ ] 支援 `__syncthreads()`
- [ ] Assembly 解析器（讀取 .asm 檔案）
- [ ] 更多範例 kernel

---

## 參考資料

- [Micro-CUDA ISA v1.5 規格](../examples/esp32_cuda_vm/docs/ISA_GUIDE.md)
- [LLVM IR 語言參考](https://llvm.org/docs/LangRef.html)
- [CUDA C++ 編程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

**版權聲明**：與 arduino-cluster-ops 專案一致

**最後更新**：2025-12-13
