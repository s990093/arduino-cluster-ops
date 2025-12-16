# ESP32 CUDA VM - ISA 完整指南

**版本**: Micro-CUDA ISA v1.5  
**架構**: True SIMT (Single Instruction, Multiple Threads)  
**硬體平台**: ESP32 雙核心 (Core0: 前端控制 / Core1: SIMD 執行引擎)

---

## 📋 目錄

1. [架構概覽](#架構概覽)
2. [ISA 規格](#isa-規格)
3. [指令集詳解](#指令集詳解)
4. [編程範例](#編程範例)
5. [使用指南](#使用指南)
6. [Python SDK](#python-sdk)
7. [進階主題](#進階主題)

---

## 架構概覽

### 完整系統架構圖

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     ESP32 CUDA VM - Micro-CUDA ISA v1.5                       ║
║                    True SIMT Architecture (Dual-Core)                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────────────┐
│                              HOST (Serial CLI)                                │
│  Commands: gpu_reset, load_imem, dma_h2d, dma_d2h, kernel_launch             │
└────────────────────────────────┬──────────────────────────────────────────────┘
                                 │ UART (115200 baud)
                                 ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                            ESP32 Dual-Core SoC                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐  ║
║  │                   Core 0: Front-End (Warp Scheduler)                    │  ║
║  ├─────────────────────────────────────────────────────────────────────────┤  ║
║  │                                                                         │  ║
║  │  [Serial Handler] ──▶ [CLI Parser]                                      │  ║
║  │         │                    │                                          │  ║
║  │         ▼                    ▼                                          │  ║
║  │  ┌──────────────────────────────────────┐                               │  ║
║  │  │     Instruction Memory (IMEM)        │                               │  ║
║  │  │  ┌────────────────────────────────┐  │                               │  ║
║  │  │  │ PC: 0  │ MOV R1, 5             │  │                               │  ║
║  │  │  │ PC: 1  │ MOV R2, 3             │  │                               │  ║
║  │  │  │ PC: 2  │ IADD R3, R1, R2       │  │                               │  ║
║  │  │  │ PC: 3  │ EXIT                  │  │                               │  ║
║  │  │  │  ...   │  ...                  │  │                               │  ║
║  │  │  └────────────────────────────────┘  │                               │  ║
║  │  │      Max: 1024 instructions          │                               │  ║
║  │  └──────────────────────────────────────┘                               │  ║
║  │         │                                                               │  ║
║  │         ▼                                                               │  ║
║  │  ┌──────────────────────────────────────┐                               │  ║
║  │  │   Instruction Fetch & Decode Unit    │                               │  ║
║  │  │  • Fetch: program[PC]                │                               │  ║
║  │  │  • Decode: Extract OP/DEST/SRC1/SRC2 │                               │  ║
║  │  │  • PC Control: PC++, BRA, BR.Z       │                               │  ║
║  │  └──────────────────────────────────────┘                               │  ║
║  │         │                                                               │  ║
║  │         ▼                                                               │  ║
║  │  ┌──────────────────────────────────────┐                               │  ║
║  │  │   FreeRTOS Queue (Dispatch)          │                               │  ║
║  │  │   Instruction ──▶ [Queue] ──▶        │                               │  ║
║  │  └──────────────────────────────────────┘                               │  ║
║  │         │                                                               │  ║
║  └─────────────────────────────┬───────────────────────────────────────────┘  ║
║                                │                                              ║
║  ┌─────────────────────────────▼───────────────────────────────────────────┐  ║
║  │                  Core 1: Back-End (SIMD Execution Engine)               │  ║
║  ├─────────────────────────────────────────────────────────────────────────┤  ║
║  │                                                                         │  ║
║  │  ┌───────────────────────────────────────────────────────────────────┐  │  ║
║  │  │             8-Lane SIMD Engine (Warp Size = 8)                    │  │  ║
║  │  │                                                                   │  │  ║
║  │  │  Lane 0      Lane 1      Lane 2      Lane 3      Lane 4      ...  │  │  ║
║  │  │  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐   │  │  ║
║  │  │  │ R[] │    │ R[] │    │ R[] │    │ R[] │    │ R[] │    │ R[] ││ │   ║
║  │  │  │ F[] │    │ F[] │    │ F[] │    │ F[] │    │ F[] │    │ F[] ││ │ ║
║  │  │  │ P[] │    │ P[] │    │ P[] │    │ P[] │    │ P[] │    │ P[] ││ │ ║
║  │  │  │ SR  │    │ SR  │    │ SR  │    │ SR  │    │ SR  │    │ SR  ││ │ ║
║  │  │  └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘│ │ ║
║  │  │     │          │          │          │          │          │   │ │ ║
║  │  │  SR.laneid  SR.laneid  SR.laneid  SR.laneid  SR.laneid  SR.la │ │ ║
║  │  │     = 0        = 1        = 2        = 3        = 4      = 7  │ │ ║
║  │  └───────┬──────────┬──────────┬──────────┬──────────┬────────┬───┘ │ ║
║  │          │          │          │          │          │        │     │ ║
║  │          └──────────┴──────────┴──────────┴──────────┴────────┘     │ ║
║  │                                  │                                  │ ║
║  │                                  ▼                                  │ ║
║  │  ┌───────────────────────────────────────────────────────────────┐ │ ║
║  │  │                 Shared Global Memory (VRAM)                   │ │ ║
║  │  │  ┌──────────────────────────────────────────────────────────┐ │ │ ║
║  │  │  │ Addr    │ 0x0000 │ 0x0004 │ 0x0008 │ 0x000C │  ...      │ │ │ ║
║  │  │  │ Data    │   10   │   20   │   30   │   40   │  ...      │ │ │ ║
║  │  │  └──────────────────────────────────────────────────────────┘ │ │ ║
║  │  │  Size: 4KB - 64KB (configurable via vm_config.h)             │ │ ║
║  │  │  Allocation: PSRAM (ESP32-S3) or Heap                        │ │ ║
║  │  └───────────────────────────────────────────────────────────────┘ │ ║
║  │                                  │                                  │ ║
║  │                                  ▼                                  │ ║
║  │  ┌───────────────────────────────────────────────────────────────┐ │ ║
║  │  │                     Execution Trace Logger                    │ │ ║
║  │  │  • Cycle Counter                                              │ │ ║
║  │  │  • Per-Lane Register Changes                                  │ │ ║
║  │  │  • Memory Access Log (Read/Write)                             │ │ ║
║  │  │  • JSON Trace Output ──▶ Serial                               │ │ ║
║  │  └───────────────────────────────────────────────────────────────┘ │ ║
║  │                                                                     │ ║
║  └──────────────────────────────── s─────────────────────────────────────┘ ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 單一 Lane 暫存器架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Lane N (N = 0..7)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   General Purpose Registers (R0-R31)                │   │
│  │   ┌────┬────┬────┬────┬─────┬─────┬─────┬──────┐   │   │
│  │   │ R0 │ R1 │ R2 │ R3 │ ... │ R30 │ R31 │      │   │   │
│  │   ├────┼────┼────┼────┼─────┼─────┼─────┤      │   │   │
│  │   │ 32 │ 32 │ 32 │ 32 │     │ 32  │ 32  │ bits │   │   │
│  │   └────┴────┴────┴────┴─────┴─────┴─────┴──────┘   │   │
│  │   32 Registers × 32-bit = 128 bytes                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Floating Point Registers (F0-F31)                 │   │
│  │   ┌────┬────┬────┬────┬─────┬─────┬─────┬──────┐   │   │
│  │   │ F0 │ F1 │ F2 │ F3 │ ... │ F30 │ F31 │      │   │   │
│  │   ├────┼────┼────┼────┼─────┼─────┼─────┤      │   │   │
│  │   │IEEE│IEEE│IEEE│IEEE│     │IEEE │IEEE │754   │   │   │
│  │   │754 │754 │754 │754 │     │754  │754  │FP32  │   │   │
│  │   └────┴────┴────┴────┴─────┴─────┴─────┴──────┘   │   │
│  │   32 Registers × 32-bit = 128 bytes                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Predicate Registers (P0-P7)                       │   │
│  │   ┌───┬───┬───┬───┬───┬───┬───┬───┐                │   │
│  │   │P0 │P1 │P2 │P3 │P4 │P5 │P6 │P7 │                │   │
│  │   ├───┼───┼───┼───┼───┼───┼───┼───┤                │   │
│  │   │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ bit            │   │
│  │   └───┴───┴───┴───┴───┴───┴───┴───┘                │   │
│  │   8 Registers × 1-bit = 1 byte                     │   │
│  │   Used for: ISETP, BR.Z (conditional execution)    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   System Registers (SR) - Read Only                 │   │
│  │   ┌──────────────────┬─────────────────────┐        │   │
│  │   │ SR_TID      (0)  │ Thread ID           │        │   │
│  │   │ SR_CTAID    (1)  │ Block ID            │        │   │
│  │   │ SR_LANEID   (2)  │ Lane ID (0-7) ★     │        │   │
│  │   │ SR_WARPSIZE (3)  │ Warp Size (8)       │        │   │
│  │   │ SR_GPU_UTIL (6)  │ GPU Utilization     │        │   │
│  │   │ SR_WARP_ID  (8)  │ Warp ID             │        │   │
│  │   │ SR_SM_ID    (9)  │ SM ID               │        │   │
│  │   └──────────────────┴─────────────────────┘        │   │
│  │   Accessed via: S2R Rd, SRn                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Local Shared Memory (256 bytes per lane)          │   │
│  │   ┌────────────────────────────────────┐            │   │
│  │   │ Addr: 0x00 - 0xFF                  │            │   │
│  │   │ Accessed via: LDS, STS instructions│            │   │
│  │   └────────────────────────────────────┘            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### SIMT 記憶體存取模型

#### LDL 指令執行流程圖

```
指令: LDL R10, [R0]  (假設 R0 = 0x1000)

Core 0 (Front-End):
┌──────────────────────────┐
│ 1. Fetch Instruction     │
│    Opcode: 0x65          │
│    DEST:   R10           │
│    SRC1:   R0            │
└────────┬─────────────────┘
         │
         │ Dispatch via Queue
         ▼
Core 1 (SIMD Engine):
┌──────────────────────────────────────────────────────────────────────┐
│ 2. Broadcast to All Lanes                                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Lane 0          Lane 1          Lane 2          ...    Lane 7      │
│  ┌────────┐     ┌────────┐     ┌────────┐             ┌────────┐   │
│  │ Read:  │     │ Read:  │     │ Read:  │             │ Read:  │   │
│  │ R0     │     │ R0     │     │ R0     │             │ R0     │   │
│  │= 0x1000│     │= 0x1000│     │= 0x1000│             │= 0x1000│   │
│  └───┬────┘     └───┬────┘     └───┬────┘             └───┬────┘   │
│      │              │              │                      │         │
│      ▼              ▼              ▼                      ▼         │
│  ┌────────┐     ┌────────┐     ┌────────┐             ┌────────┐   │
│  │SR.lane │     │SR.lane │     │SR.lane │             │SR.lane │   │
│  │id = 0  │     │id = 1  │     │id = 2  │             │id = 7  │   │
│  └───┬────┘     └───┬────┘     └───┬────┘             └───┬────┘   │
│      │              │              │                      │         │
│      ▼              ▼              ▼                      ▼         │
│  ┌────────┐     ┌────────┐     ┌────────┐             ┌────────┐   │
│  │Compute │     │Compute │     │Compute │             │Compute │   │
│  │Addr:   │     │Addr:   │     │Addr:   │             │Addr:   │   │
│  │0x1000+ │     │0x1000+ │     │0x1000+ │             │0x1000+ │   │
│  │0*4     │     │1*4     │     │2*4     │             │7*4     │   │
│  │=0x1000 │     │=0x1004 │     │=0x1008 │             │=0x101C │   │
│  └───┬────┘     └───┬────┘     └───┬────┘             └───┬────┘   │
│      │              │              │                      │         │
│      └──────────────┴──────────────┴──────────────────────┘         │
│                                    │                                │
│                                    ▼                                │
│         ┌─────────────────────────────────────────────┐             │
│         │   Parallel VRAM Access (Same Cycle)         │             │
│         │  ┌────────────────────────────────────────┐ │             │
│         │  │ 0x1000: Lane0 reads  ───▶ R10 = val0  │ │             │
│         │  │ 0x1004: Lane1 reads  ───▶ R10 = val1  │ │             │
│         │  │ 0x1008: Lane2 reads  ───▶ R10 = val2  │ │             │
│         │  │  ...                                   │ │             │
│         │  │ 0x101C: Lane7 reads  ───▶ R10 = val7  │ │             │
│         │  └────────────────────────────────────────┘ │             │
│         └─────────────────────────────────────────────┘             │
│                                                                      │
│ 結果: 單一指令，8 個 Lane 同時讀取 8 個不同位址的資料！              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 資料流程圖（完整執行週期）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Execution Pipeline                          │
└─────────────────────────────────────────────────────────────────────┘

Cycle N:
  Core 0                          Core 1
  ┌─────────┐                    ┌──────────────┐
  │ Fetch   │                    │ Execute      │
  │ PC: N   │                    │ Inst[N-1]    │
  └────┬────┘                    │ (8 lanes)    │
       │                         └──────────────┘
       │                               │
       ▼                               ▼
  ┌─────────┐                    ┌──────────────┐
  │ Decode  │                    │ Memory       │
  │ Inst[N] │                    │ Access       │
  └────┬────┘                    │ (if needed)  │
       │                         └──────────────┘
       │                               │
       ▼                               ▼
  ┌─────────┐                    ┌──────────────┐
  │ Queue   │                    │ Write Back   │
  │ Send    │◀───────Queue───────│ Update Regs  │
  └─────────┘                    └──────────────┘
                                       │
                                       ▼
                                 ┌──────────────┐
                                 │ Trace Log    │
                                 │ (Optional)   │
                                 └──────────────┘

Pipeline Characteristics:
• Loosely Coupled: Core 0 and Core 1 communicate via FreeRTOS Queue
• Asynchronous Dispatch: Core 0 can fetch ahead while Core 1 executes
• SIMD Parallelism: 8 lanes execute in parallel on Core 1
• Cycle Accurate: Each instruction logs execution cycle
```

### SIMT 執行模型總結

**關鍵特性**：

- ✅ **True SIMT**：所有 Lane 同時執行相同指令
- ✅ **Lane-Awareness**：每個 Lane 具有唯一 ID (`SR_LANEID`)
- ✅ **Data Parallelism**：支援 Per-Lane 記憶體存取
- ✅ **Shared VRAM**：所有 Lane 共享全域記憶體空間
- ✅ **Independent Register Files**：每個 Lane 有獨立的 R/F/P 暫存器

---

## 技術規格與配置參數

### 硬體需求

| 項目         | 最低需求          | 建議配置            |
| ------------ | ----------------- | ------------------- |
| **開發板**   | ESP32 (Dual-Core) | ESP32-S3 (帶 PSRAM) |
| **Flash**    | 4MB               | 8MB+                |
| **RAM**      | 320KB             | 512KB+              |
| **PSRAM**    | 可選              | 8MB (用於大型 VRAM) |
| **時脈**     | 80 MHz            | 240 MHz             |
| **USB**      | Micro-USB / USB-C | USB-C (更穩定)      |
| **串口晶片** | CP2102 / CH340    | CP2104 (更高速)     |

### 記憶體配置

#### 指令記憶體 (IMEM)

| 參數              | 預設值  | 可調範圍   | 說明               |
| ----------------- | ------- | ---------- | ------------------ |
| `VM_PROGRAM_SIZE` | 1024    | 256 - 4096 | 最大指令數量       |
| 單條指令大小      | 4 bytes | 固定       | 32-bit 編碼        |
| IMEM 總容量       | 4 KB    | 1KB - 16KB | 1024 × 4 bytes     |
| 實際 RAM 使用     | ~4 KB   | -          | 儲存於 Core 0 DRAM |

**配置位置**：`vm_config.h` 第 9 行

```cpp
#define VM_PROGRAM_SIZE  1024  // 可調整為 2048, 4096 等
```

#### 全域記憶體 (VRAM)

| 參數           | 預設值     | 可調範圍       | 說明                    |
| -------------- | ---------- | -------------- | ----------------------- |
| `VM_VRAM_SIZE` | 40960      | 4096 - 1048576 | 全域記憶體大小 (bytes)  |
| 預設配置       | 40 KB      | -              | 適用於標準 ESP32        |
| 小型配置       | 4 KB       | 4096           | 測試/學習用             |
| 中型配置       | 32 KB      | 32768          | 一般應用                |
| 大型配置       | 100 KB     | 102400         | 需要 PSRAM              |
| 超大配置       | 1 MB       | 1048576        | ESP32-S3 with 8MB PSRAM |
| 記憶體對齊     | 4 bytes    | -              | 32-bit 字對齊           |
| 分配位置       | PSRAM 優先 | PSRAM → Heap   | 自動 fallback           |

**配置位置**：`vm_config.h` 第 14 行

```cpp
#define VM_VRAM_SIZE     40960  // 40KB (標準配置)
// #define VM_VRAM_SIZE  102400  // 100KB (需要 PSRAM)
// #define VM_VRAM_SIZE  4096    // 4KB (測試用)
```

**VRAM 容量計算範例**：

```
8 Lanes × 1000 elements × 4 bytes = 32,000 bytes (32KB)
適用於：1000 個 FP32 向量的並行處理
```

#### 暫存器檔案（每個 Lane）

| 暫存器類型       | 數量 | 每個大小 | 總容量     | 說明                |
| ---------------- | ---- | -------- | ---------- | ------------------- |
| **R (通用)**     | 32   | 32-bit   | 128 bytes  | 整數、位址、索引    |
| **F (浮點)**     | 32   | 32-bit   | 128 bytes  | IEEE-754 FP32       |
| **P (條件)**     | 8    | 1-bit    | 1 byte     | 條件旗標            |
| **SR (系統)**    | ~10  | 32-bit   | ~40 bytes  | 唯讀系統暫存器      |
| **Shared Mem**   | 256  | byte     | 256 bytes  | Per-Lane 本地記憶體 |
| **單 Lane 總計** | -    | -        | ~553 bytes | R+F+P+SR+Shared     |
| **8 Lanes 總計** | -    | -        | ~4.3 KB    | 不含 VRAM           |

#### FreeRTOS 配置

| 參數                 | 預設值   | 可調範圍     | 說明                  |
| -------------------- | -------- | ------------ | --------------------- |
| `VM_STACK_SIZE`      | 8192     | 4096 - 16384 | Task 堆疊大小 (bytes) |
| `VM_QUEUE_SIZE`      | 16       | 8 - 64       | 指令佇列深度          |
| Core 0 Task Priority | 1        | 0 - 24       | Front-End 優先級      |
| Core 1 Task Priority | 1        | 0 - 24       | SIMD Engine 優先級    |
| Queue Item Size      | 16 bytes | -            | Instruction struct    |

**配置位置**：`vm_config.h` 第 19-22 行

```cpp
#define VM_STACK_SIZE    8192   // Task 堆疊
#define VM_QUEUE_SIZE    16     // 指令佇列
```

### 效能參數

#### 執行速度

| 指標               | 典型值         | 說明                 |
| ------------------ | -------------- | -------------------- |
| CPU 時脈           | 240 MHz        | ESP32 最大時脈       |
| 指令吞吐量         | ~20-50K inst/s | 依指令類型而定       |
| SIMD 並行度        | 8 lanes        | 同時執行 8 個 Thread |
| 記憶體頻寬（VRAM） | ~10 MB/s       | 8-bit 存取           |
| Serial 傳輸速率    | 115200 baud    | ~11.5 KB/s           |
| Queue 延遲         | ~1-5 cycles    | Core 0 → Core 1      |

#### 週期計數（Cycle Count）

不同指令類型的執行週期：

| 指令類型   | 週期數 | 範例                  |
| ---------- | ------ | --------------------- |
| 整數運算   | 1      | `IADD`, `ISUB`, `AND` |
| 整數乘法   | 1      | `IMUL`                |
| 整數除法   | 1-3    | `IDIV` (模擬實作)     |
| 浮點運算   | 1-2    | `FADD`, `FMUL`        |
| 浮點除法   | 2-4    | `FDIV`                |
| SFU 函數   | 3-10   | `SFU.EXP`, `SFU.GELU` |
| 記憶體載入 | 1-2    | `LDG`, `LDL`, `LDS`   |
| 記憶體儲存 | 1-2    | `STG`, `STL`, `STS`   |
| 控制流     | 1      | `BRA`, `BR.Z`, `EXIT` |
| 系統暫存器 | 1      | `S2R`, `R2S`          |

**注意**：以上週期數為邏輯週期，實際執行時間還包含 Queue 通訊和 FreeRTOS 排程開銷。

### 串口配置

| 參數           | 預設值    | 可選範圍      | 說明        |
| -------------- | --------- | ------------- | ----------- |
| `VM_BAUD_RATE` | 115200    | 9600 - 921600 | UART 波特率 |
| 資料位元       | 8         | -             | 固定        |
| 停止位元       | 1         | -             | 固定        |
| 同位檢查       | None      | -             | 無同位檢查  |
| 流量控制       | None      | -             | 無硬體流控  |
| RX Buffer      | 256 bytes | -             | ESP32 預設  |
| TX Buffer      | 256 bytes | -             | ESP32 預設  |

**高速配置**（實驗性）：

```cpp
#define VM_BAUD_RATE     921600  // ~92 KB/s (可能不穩定)
```

### 配置文件位置

| 檔案                 | 路徑                                        | 主要配置項目                 |
| -------------------- | ------------------------------------------- | ---------------------------- |
| `vm_config.h`        | `examples/esp32_cuda_vm/vm_config.h`        | VRAM, IMEM, Queue, Baud Rate |
| `instructions_v15.h` | `examples/esp32_cuda_vm/instructions_v15.h` | Opcode 定義                  |
| `vm_simd_v15.h`      | `examples/esp32_cuda_vm/vm_simd_v15.h`      | Lane 數量、Warp Size         |

### 記憶體使用估算

#### 靜態記憶體（編譯時分配）

```
指令記憶體:      1024 × 4 bytes     = 4 KB
暫存器檔案:      8 lanes × 553 bytes ≈ 4.3 KB
程式碼 + 資料:                        ≈ 50-80 KB
FreeRTOS 堆疊:   2 tasks × 8KB      = 16 KB
─────────────────────────────────────────────
靜態總計:                             ≈ 74-104 KB
```

#### 動態記憶體（執行時分配）

```
VRAM:            可配置              = 4KB - 1MB
FreeRTOS Queue:  16 × 16 bytes      = 256 bytes
Trace Buffer:    (可選)              ≈ 4-16 KB
─────────────────────────────────────────────
動態總計:                             ≈ 4KB - 1MB
```

#### ESP32 RAM 使用建議

| ESP32 型號           | 可用 RAM | VRAM 建議上限 | 說明            |
| -------------------- | -------- | ------------- | --------------- |
| ESP32 (無 PSRAM)     | ~280 KB  | 32 KB         | 保守配置        |
| ESP32 (2MB PSRAM)    | ~2.2 MB  | 100 KB        | PSRAM 用於 VRAM |
| ESP32-S3 (8MB PSRAM) | ~8.3 MB  | 1 MB+         | 大型應用        |

---

## ISA 規格

### 指令編碼

**固定 32-bit 格式**：

```
┌────────┬─────────┬─────────┬──────────┐
│ OPCODE │  DEST   │  SRC1   │ SRC2/IMM │
│ 8 bits │ 8 bits  │ 8 bits  │  8 bits  │
└────────┴─────────┴─────────┴──────────┘
  31:24    23:16     15:8       7:0
```

**欄位說明**：

- **OPCODE** [31:24]: 指令操作碼
- **DEST** [23:16]: 目標暫存器編號
- **SRC1** [15:8]: 第一來源暫存器
- **SRC2/IMM** [7:0]: 第二來源暫存器或立即值

### 系統暫存器 (SR)

| 索引     | 名稱          | 說明                                       |
| -------- | ------------- | ------------------------------------------ |
| SR_0     | SR_TID        | Thread ID (Physical Core ID)               |
| SR_1     | SR_CTAID      | Block ID (CTA ID)                          |
| **SR_2** | **SR_LANEID** | **Lane Index (0-7)，用於 SIMT 記憶體操作** |
| SR_3     | SR_WARPSIZE   | Warp Size (通常為 8)                       |
| SR_6     | SR_GPU_UTIL   | GPU 使用率                                 |
| SR_8     | SR_WARP_ID    | Warp ID                                    |
| SR_9     | SR_SM_ID      | Streaming Multiprocessor ID                |

---

## 指令集詳解

### Group 1: 系統控制 (0x00-0x0F)

| Opcode | 指令     | 格式    | 功能               | 範例       |
| ------ | -------- | ------- | ------------------ | ---------- |
| 0x00   | NOP      | -       | 空指令             | `NOP`      |
| 0x01   | EXIT     | -       | 終止 Kernel        | `EXIT`     |
| 0x02   | BRA      | Imm     | 無條件跳躍         | `BRA 10`   |
| 0x03   | BR.Z     | Imm, Pn | 條件跳躍 (P0=1 時) | `BR.Z 5`   |
| 0x05   | BAR.SYNC | Id      | Warp Barrier 同步  | `BAR.SYNC` |
| 0x07   | YIELD    | -       | 讓出執行時間片     | `YIELD`    |

**編碼範例**：

```assembly
EXIT        ; 0x01000000
BRA 10      ; 0x0200000A (跳到 PC+10)
```

### Group 2: 整數運算 (0x10-0x2F)

| Opcode | 指令     | 格式       | 功能         | 旗標 |
| ------ | -------- | ---------- | ------------ | ---- |
| 0x10   | MOV      | Rd, Imm    | 載入立即值   | -    |
| 0x11   | IADD     | Rd, Ra, Rb | 整數加法     | Z, C |
| 0x12   | ISUB     | Rd, Ra, Rb | 整數減法     | Z, C |
| 0x13   | IMUL     | Rd, Ra, Rb | 整數乘法     | -    |
| 0x14   | IDIV     | Rd, Ra, Rb | 整數除法     | -    |
| 0x17   | AND      | Rd, Ra, Rb | 位元 AND     | Z    |
| 0x18   | OR       | Rd, Ra, Rb | 位元 OR      | Z    |
| 0x19   | XOR      | Rd, Ra, Rb | 位元 XOR     | -    |
| 0x1A   | ISETP.EQ | Pn, Ra, Rb | 整數相等比較 | Pn   |
| 0x1B   | ISETP.NE | Pn, Ra, Rb | 整數不等比較 | Pn   |
| 0x1C   | ISETP.GT | Pn, Ra, Rb | 整數大於比較 | Pn   |
| 0x1D   | SHL      | Rd, Ra, Rb | 左移         | -    |
| 0x1E   | SHR      | Rd, Ra, Rb | 右移         | -    |

**編碼範例**：

```assembly
MOV R1, 100     ; 0x10010064 (R1 = 100)
IADD R3, R1, R2 ; 0x11030102 (R3 = R1 + R2)
IMUL R4, R3, R3 ; 0x13040303 (R4 = R3 * R3)
```

**範例程序 - 計算 R1 = 5 + 3**：

```assembly
MOV R2, 5       ; 0x10020005
MOV R3, 3       ; 0x10030003
IADD R1, R2, R3 ; 0x11010203
EXIT            ; 0x01000000
```

### Group 3: 浮點與 AI 運算 (0x30-0x5F)

| Opcode | 指令     | 格式       | 功能                 | 應用場景    |
| ------ | -------- | ---------- | -------------------- | ----------- |
| 0x30   | FADD     | Fd, Fa, Fb | FP32 加法            | 通用        |
| 0x31   | FSUB     | Fd, Fa, Fb | FP32 減法            | 通用        |
| 0x32   | FMUL     | Fd, Fa, Fb | FP32 乘法            | Scaling     |
| 0x33   | FDIV     | Fd, Fa, Fb | FP32 除法            | 通用        |
| 0x34   | FFMA     | Fd, Fa, Fb | $Fd = Fa × Fb + Fd$  | MAC         |
| 0x40   | HMMA.I8  | Rd, Ra, Rb | 4-way SIMD INT8 點積 | LLM Quant   |
| 0x50   | SFU.RCP  | Fd, Fa     | 倒數 $1.0/Fa$        | Softmax     |
| 0x51   | SFU.SQRT | Fd, Fa     | 平方根               | 正規化      |
| 0x52   | SFU.EXP  | Fd, Fa     | 指數函數             | Softmax     |
| 0x53   | SFU.GELU | Fd, Fa     | GELU Activation      | Transformer |
| 0x54   | SFU.RELU | Fd, Fa     | ReLU $\max(0, Fa)$   | CNN         |

**編碼範例**：

```assembly
FADD F1, F2, F3     ; 0x30010203 (F1 = F2 + F3)
FMUL F4, F1, F1     ; 0x32040101 (F4 = F1 * F1)
SFU.RELU F5, F4     ; 0x54050400 (F5 = max(0, F4))
```

### Group 4: 記憶體操作 (0x60-0x7F) ⭐

**重要**：v1.5 版本引入了 SIMT 記憶體操作，這是實現資料並行的關鍵！

#### Uniform Operations（所有 Lane 存取相同位址）

| Opcode | 指令 | 格式      | 行為                           |
| ------ | ---- | --------- | ------------------------------ |
| 0x60   | LDG  | Rd, [Ra]  | 全部 Lane 讀取相同位址（廣播） |
| 0x61   | STG  | [Ra], Rd  | 全部 Lane 寫入相同位址         |
| 0x62   | LDS  | Rd, [Imm] | 從 Shared Memory 讀取          |
| 0x63   | STS  | [Imm], Rd | 寫入 Shared Memory             |

#### SIMT Operations（每個 Lane 存取不同位址）**[NEW in v1.5]**

| Opcode | 指令    | 格式         | 行為邏輯                                                                                       |
| ------ | ------- | ------------ | ---------------------------------------------------------------------------------------------- |
| 0x65   | **LDL** | **Rd, [Ra]** | **Lane-Based Load**<br>每個 Lane 計算：`Addr = Ra + SR_LANEID * 4`<br>硬體自動添加 Lane Offset |
| 0x67   | **STL** | **[Ra], Rd** | **Lane-Based Store**<br>每個 Lane 寫入：`Addr = Ra + SR_LANEID * 4`                            |
| 0x64   | LDX     | Rd, [Ra+Rb]  | **Indexed SIMT Load**<br>每個 Lane 計算：`Addr = Ra + Rb`<br>(Rb 是 Lane 私有暫存器)           |
| 0x66   | STX     | [Ra+Rb], Rd  | **Indexed SIMT Store**<br>Scatter Write                                                        |

**LDL 硬體行為詳解**：

當執行 `LDL R10, [R0]` 時（假設 R0=0x1000）：

| Lane | SR_LANEID | 計算位址      | 讀取位址 |
| ---- | --------- | ------------- | -------- |
| 0    | 0         | 0x1000 + 0\*4 | 0x1000   |
| 1    | 1         | 0x1000 + 1\*4 | 0x1004   |
| 2    | 2         | 0x1000 + 2\*4 | 0x1008   |
| 3    | 3         | 0x1000 + 3\*4 | 0x100C   |
| 4    | 4         | 0x1000 + 4\*4 | 0x1010   |
| 5    | 5         | 0x1000 + 5\*4 | 0x1014   |
| 6    | 6         | 0x1000 + 6\*4 | 0x1018   |
| 7    | 7         | 0x1000 + 7\*4 | 0x101C   |

**結果**：一條指令，8 個 Lane 同時讀取 8 個不同位址的資料！

#### Atomic Operations

| Opcode | 指令     | 格式         | 功能           |
| ------ | -------- | ------------ | -------------- |
| 0x70   | ATOM.ADD | [Ra], Rd     | Atomic Add     |
| 0x71   | ATOM.CAS | [Ra], Rb, Rc | Compare & Swap |

### Group 5: 系統暫存器操作 (0xF0-0xFF)

| Opcode | 指令  | 格式    | 功能                               |
| ------ | ----- | ------- | ---------------------------------- |
| 0xF0   | S2R   | Rd, SRn | System to Register（讀取系統狀態） |
| 0xF1   | R2S   | SRn, Rd | Register to System                 |
| 0xF2   | TRACE | Imm     | 發送 Trace ID（調試用）            |

**編碼範例**：

```assembly
S2R R31, SR_LANEID  ; 0xF01F0002 (R31 = 我的 Lane ID)
S2R R30, SR_WARPSIZE ; 0xF01E0003 (R30 = Warp Size)
```

---

## 編程範例

### 範例 1: 簡單算術運算

**目標**：計算 `R1 = (5 + 3) * 2`

```assembly
MOV R2, 5           ; R2 = 5
MOV R3, 3           ; R3 = 3
IADD R4, R2, R3     ; R4 = 5 + 3 = 8
MOV R5, 2           ; R5 = 2
IMUL R1, R4, R5     ; R1 = 8 * 2 = 16
EXIT
```

**16 進位編碼**：

```
10020005  ; MOV R2, 5
10030003  ; MOV R3, 3
11040203  ; IADD R4, R2, R3
10050002  ; MOV R5, 2
13010405  ; IMUL R1, R4, R5
01000000  ; EXIT
```

### 範例 2: 條件跳躍 (迴圈)

**目標**：計算 1+2+3+...+10 的總和

```assembly
MOV R0, 0           ; sum = 0
MOV R1, 1           ; counter = 1
MOV R2, 10          ; limit = 10

loop:
IADD R0, R0, R1     ; sum += counter
IADD R1, R1, 1      ; counter++
ISETP.GT P0, R1, R2 ; P0 = (counter > 10)?
BR.Z -3             ; if (P0 == 0) goto loop

EXIT                ; sum in R0 = 55
```

### 範例 3: Parallel Attention (Q/K/V) ⭐

**場景**：8 個 Lane 並行載入並計算 Attention Score

**假設**：

- Q 陣列位址: 0x1000
- K 陣列位址: 0x2000
- V 陣列位址: 0x3000
- 結果位址: 0x4000

```assembly
; === 初始化 ===
S2R R31, SR_LANEID      ; R31 = 我的 Lane ID (0~7)

; 設定 Q/K/V 基底位址
MOV R0, 0x10            ; R0 = Base of Q (0x1000 >> 8)
MOV R1, 0x20            ; R1 = Base of K
MOV R2, 0x30            ; R2 = Base of V

; === SIMT Loading（關鍵）===
;硬體會執行: Effective_Addr = Base + LaneID * 4
LDL R10, [R0]           ; R10 = Q[lane]
LDL R11, [R1]           ; R11 = K[lane]
LDL R12, [R2]           ; R12 = V[lane]

; === Parallel Execution ===
; 每個 Lane 的 R10, R11, R12 都不同
IMUL R20, R10, R11      ; R20 = Q[i] * K[i] (Attention Score)
IADD R21, R20, R12      ; R21 = Score + V[i]

; === Write Back ===
MOV R3, 0x40            ; R3 = Result base
STL [R3], R21           ; Store Result[lane]

EXIT
```

**執行結果**（假設輸入資料如下）：

| 位址   | 資料 (Q)        | 資料 (K)         | 資料 (V)          |
| ------ | --------------- | ---------------- | ----------------- |
| 0x1000 | 2,3,4,5,6,7,8,9 | 3,4,5,6,7,8,9,10 | 4,5,6,7,8,9,10,11 |

| Lane | Q(R10) | K(R11) | V(R12) | Score(R20) | Final(R21) |
| ---- | ------ | ------ | ------ | ---------- | ---------- |
| 0    | 2      | 3      | 4      | 6          | 10         |
| 1    | 3      | 4      | 5      | 12         | 17         |
| 2    | 4      | 5      | 6      | 20         | 26         |
| 3    | 5      | 6      | 7      | 30         | 37         |
| 4    | 6      | 7      | 8      | 42         | 50         |
| 5    | 7      | 8      | 9      | 56         | 65         |
| 6    | 8      | 9      | 10     | 72         | 82         |
| 7    | 9      | 10     | 11     | 90         | 101        |

---

## 使用指南

### 硬體準備

1. **ESP32 開發板**（建議 ESP32-S3 或 TTGO T-Display）
2. **USB 連接線**
3. **Arduino IDE** 或 **PlatformIO**

### 韌體燒錄

#### 方法 1: 使用 CLI 工具

```bash
cd /path/to/arduino-cluster-ops

# 查看可用串口
python cli.py ports

# 上傳韌體
python cli.py upload examples/esp32_cuda_vm/esp32_cuda_vm.ino \
  --port /dev/cu.usbserial-XXXX \
  --fqbn esp32:esp32:esp32
```

#### 方法 2: 使用 Arduino IDE

1. 開啟 `examples/esp32_cuda_vm/esp32_cuda_vm.ino`
2. 選擇正確的開發板和串口
3. 點擊「上傳」

### 串口命令介面 (CLI)

連接串口（波特率 115200）：

```bash
python cli.py monitor --port /dev/cu.usbserial-XXXX --baudrate 115200
```

#### 可用命令

| 命令            | 參數             | 說明                                   |
| --------------- | ---------------- | -------------------------------------- |
| `gpu_reset`     | -                | **GPU Reset** (Hard Reset VM & VRAM)   |
| `load_imem`     | `<bytes>`        | **Load Module** (Binary Kernel Upload) |
| `dma_h2d`       | `<addr> <size>`  | **H2D DMA** (Binary Data Upload)       |
| `dma_d2h`       | `<addr> <count>` | **D2H DMA** (Read VRAM)                |
| `kernel_launch` | -                | **Launch Kernel** (Execute Function)   |
| `step`          | -                | 單步執行                               |
| `reg`           | `[lane_id]`      | 顯示指定 Lane 的暫存器（預設 Lane 0）  |
| `trace:stream`  | -                | 啟用 JSON Trace 串流                   |
| `trace:off`     | -                | 關閉 Trace                             |
| `help`          | -                | 顯示幫助訊息                           |

#### 互動式範例

```bash
> gpu_reset
GPU Reset Complete

> load 10010005
Loaded: 10010005

> load 10020003
Loaded: 10020003

> load 11030102
Loaded: 11030102

> load 01000000
Loaded: 01000000

> kernel_launch
Running...
Program Finished (EXIT)

> reg 0
=== Lane 0 Registers ===
R[1] = 8
R[2] = 5
R[3] = 3
```

---

## Python SDK

### 安裝

確保你的 Python 環境已安裝必要套件：

```bash
cd /path/to/arduino-cluster-ops
pip install pyserial
```

### 基本使用

#### 連接 ESP32

```python
import sys
sys.path.insert(0, "/path/to/arduino-cluster-ops")

from esp32_tools import ESP32Connection

PORT = "/dev/cu.usbserial-XXXX"
conn = ESP32Connection(PORT)
print("Connected!")
```

#### 寫入記憶體 (VRAM)

```python
# 寫入整數到 VRAM
conn.send_command("mem 0 100")    # VRAM[0] = 100
conn.send_command("mem 4 200")    # VRAM[4] = 200

# 驗證
conn.send_command("dump 0 2")
print(conn.read_lines())
```

#### 使用指令編碼器

```python
from esp32_tools.program_loader_v15 import InstructionV15

# 建立指令
inst1 = InstructionV15.mov(1, 100)      # MOV R1, 100
inst2 = InstructionV15.iadd(2, 1, 1)    # IADD R2, R1, R1
inst3 = InstructionV15.exit_inst()      # EXIT

# 載入程式
conn.send_command("gpu_reset")
for inst in [inst1, inst2, inst3]:
    conn.send_command(f"load {inst.to_hex()}")

# 執行
conn.send_command("kernel_launch")
```

### 完整範例：SIMT Parallel Load

```python
from esp32_tools import ESP32Connection
from esp32_tools.program_loader_v15 import InstructionV15

PORT = "/dev/cu.usbserial-XXXX"
conn = ESP32Connection(PORT)

# 初始化 VRAM：寫入陣列 [10, 20, 30, ..., 80]
conn.send_command("gpu_reset")
for i in range(8):
    conn.send_command(f"mem {i*4} {(i+1)*10}")

# 建立 SIMT 程式
program = [
    InstructionV15.s2r(31, 2),          # R31 = SR_LANEID
    InstructionV15.mov(0, 0),           # R0 = Base Address (0)
    InstructionV15.ldl(10, 0),          # R10 = VRAM[R0 + LaneID*4]
    InstructionV15.iadd(11, 10, 10),    # R11 = R10 * 2 (簡化)
    InstructionV15.exit_inst()
]

# 載入並執行
for inst in program:
    conn.send_command(f"load {inst.to_hex()}")

conn.send_command("kernel_launch")

# 查看各 Lane 結果
for lane in range(8):
    conn.send_command(f"reg {lane}")
    print(conn.read_lines())
```

### Trace 解析

```python
import json

# 啟用 Trace
conn.send_command("trace:stream")
conn.send_command("kernel_launch")

# 讀取輸出
output = ""
for _ in range(20):
    lines = conn.read_lines()
    if lines:
        output += "\n".join(lines)

# 解析 JSON
start = output.find('{')
end = output.rfind('}')
if start != -1 and end != -1:
    trace_data = json.loads(output[start:end+1])
    print(json.dumps(trace_data, indent=2))
```

---

## 進階主題

### SIMT vs Broadcast：何時使用？

| 操作類型         | 使用指令 | 說明                             |
| ---------------- | -------- | -------------------------------- |
| 載入常數（廣播） | LDG      | 所有 Lane 讀取相同值（如權重）   |
| 載入向量元素     | **LDL**  | 每個 Lane 讀取不同值（如 Q/K/V） |
| 載入稀疏資料     | LDX      | 每個 Lane 使用不同索引           |
| 寫入結果陣列     | **STL**  | 每個 Lane 寫入對應位置           |

### v1.0 vs v1.5：編程模型差異

**舊模型 (v1.0)**：Host 循環控制

```python
# Python Host Code
for lane_id in range(8):
    load_data(lane_id, Q[lane_id])
    load_data(lane_id, K[lane_id])
    compute(lane_id)
```

**新模型 (v1.5)**：True SIMT

```assembly
; Single instruction, parallel execution
LDL R10, [R0]           ; All lanes load different Q
LDL R11, [R1]           ; All lanes load different K
IMUL R20, R10, R11      ; All lanes compute
```

**優勢**：

- ✅ 減少 Host 開銷
- ✅ 真正的硬體並行
- ✅ 符合 GPU 編程範式

### 效能優化技巧

1. **記憶體對齊**：確保陣列起始位址是 4 的倍數
2. **批次載入**：使用 LDL 一次載入多個 Lane 的資料
3. **減少分支**：分支會導致 Warp Divergence，降低效率
4. **善用 Shared Memory**：LDS/STS 比 LDG/STG 快

### 調試技巧

1. **使用 TRACE 指令**：

   ```assembly
   TRACE 1     ; Mark checkpoint 1
   IADD R0, R1, R2
   TRACE 2     ; Mark checkpoint 2
   ```

2. **單步執行**：

   ```bash
   > step
   > reg 0
   > step
   > reg 0
   ```

3. **啟用 JSON Trace**：
   ```bash
   > trace:stream
   > run
   ```

### 記憶體配置建議

編輯 `vm_config.h`：

```cpp
// 小型測試 (4KB)
#define VM_VRAM_SIZE 4096

// 中型應用 (32KB)
#define VM_VRAM_SIZE 32768

// 大型應用 (64KB，需要 PSRAM)
#define VM_VRAM_SIZE 65536
```

---

## 常見問題

### Q1: VRAM 讀取全為 0？

**A**: 確保你在 `gpu_reset` 之後使用 `mem` 命令初始化資料，並且使用 `kernel_launch`（保留 VRAM）而非 `gpu_reset` 再次執行。

### Q2: 如何確認 LDL 正確運作？

**A**: 使用 `reg` 命令查看不同 Lane 的暫存器：

```bash
> reg 0
> reg 1
> reg 2
```

如果每個 Lane 的 R10 都不同，表示 SIMT Load 成功！

### Q3: JSON Trace 解析錯誤？

**A**: 大型 Trace 可能被 Serial Buffer 分割，使用 `test_enhanced_trace.py` 中的 robust parsing 邏輯。

### Q4: 如何擴展 VRAM 大小？

**A**:

1. 修改 `vm_config.h` 中的 `VM_VRAM_SIZE`
2. 確保 ESP32 有足夠 PSRAM（建議使用 ESP32-S3）
3. 重新編譯並上傳韌體

---

## 參考資料

- [MICRO_CUDA_ISA_V15_SPEC.md](file:///Users/hungwei/Desktop/Proj/arduino-cluster-ops/docs/MICRO_CUDA_ISA_V15_SPEC.md) - 完整 ISA 規格
- [USER_GUIDE.md](file:///Users/hungwei/Desktop/Proj/arduino-cluster-ops/examples/esp32_cuda_vm/USER_GUIDE.md) - 雙核心架構說明
- [instructions_v15.h](file:///Users/hungwei/Desktop/Proj/arduino-cluster-ops/examples/esp32_cuda_vm/instructions_v15.h) - Opcode 定義
- [vm_simd_v15.h](file:///Users/hungwei/Desktop/Proj/arduino-cluster-ops/examples/esp32_cuda_vm/vm_simd_v15.h) - SIMD 引擎介面

---

**版本歷史**：

- v1.5.2 (2024-12): 穩定性修正 (Switch-Case Dispatch), 移除實驗性 Float ASM, 確認 8-Lane 配置
- v1.5 (2024-12): 引入 Lane-Awareness 和 SIMT 記憶體操作
- v1.0 (2024-11): 初始版本，廣播式架構

**授權**：與 arduino-cluster-ops 專案一致

---

<br>

## 8. 參考實作 (C++ Implementation)

為了幫助開發者深入理解 Micro-CUDA 的硬體行為，本節提供了核心指令在 firmware 層 (`vm_simd_v15.cpp`) 的實際 C++ 實作邏輯。

### 8.1 記憶體佈局 (Structure of Arrays)

所有的運算都是基於 **SoA (Structure of Arrays)** 佈局進行的，這意味著我們存取的是「同一個暫存器在所有 Lane 的值」，而不是「一個 Lane 的所有暫存器」。

```cpp
// 存取 Lane N 的 R[dest]
uint32_t* dest_ptr = &warp_state.R[dest][0];

// dest_ptr[0] => Lane 0 的 R[dest]
// dest_ptr[1] => Lane 1 的 R[dest]
// ...
// dest_ptr[7] => Lane 7 的 R[dest]
```

### 8.2 SIMT Load (LDL) 實作

`LDL` 指令展示了硬體如何利用 `SR_LANEID` 來並行計算位址。

```cpp
case OP_LDL: {
    // 每個 Lane 獨立計算位址
    for(int i=0; i<8; i++) {
        // Effective Addr = Base(Ra) + LaneID * 4
        uint32_t addr = R_src1[i] + i * 4;

        if(addr < VM_VRAM_SIZE) {
            // 從 VRAM 讀取資料到目標暫存器
            R_dest[i] = *((uint32_t*)&vram[addr]);
        }
    }
    break;
}
```

### 8.3 整數加法 (IADD) 與 ASM 優化

`IADD` 預設使用 Xtensa 組合語言優化 (Level 4)，利用 `loop` 指令減少分支開銷。

```cpp
// XTENSA ASM Implementation (vm_simd_v15.cpp)
static inline void asm_warp_add(uint32_t* dest, const uint32_t* src1, const uint32_t* src2) {
    int loop_count = 2; // 8 lanes / 4 unroll = 2 iterations
    __asm__ volatile (
        "loop %0, loop_end_add\n\t"
        // --- Loop Body (Unrolled 4x) ---
        "l32i.n a8, %1, 0\n\t" "l32i.n a9, %2, 0\n\t" "add a8, a8, a9\n\t" "s32i.n a8, %3, 0\n\t"
        "l32i.n a8, %1, 4\n\t" "l32i.n a9, %2, 4\n\t" "add a8, a8, a9\n\t" "s32i.n a8, %3, 4\n\t"
        "l32i.n a8, %1, 8\n\t" "l32i.n a9, %2, 8\n\t" "add a8, a8, a9\n\t" "s32i.n a8, %3, 8\n\t"
        "l32i.n a8, %1, 12\n\t" "l32i.n a9, %2, 12\n\t" "add a8, a8, a9\n\t" "s32i.n a8, %3, 12\n\t"
        // --- Pointer Increment ---
        "addi %1, %1, 16\n\t" "addi %2, %2, 16\n\t" "addi %3, %3, 16\n\t"
        "loop_end_add:\n\t"
        : "+r"(loop_count), "+r"(src1), "+r"(src2), "+r"(dest) :: "a8", "a9", "memory"
    );
}
```

### 8.4 SFU 快速近似 (GELU)

特殊函數單元 (SFU) 使用快速近似演算法來達到高效能 AI 運算。

```cpp
// Fast Sigmoid Approximation
static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-1.702f * x));
}

// SFU_GELU Handler
case OP_SFU_GELU:
     for(int i=0; i<8; i++) {
         float x = F_src1[i];
         // GELU(x) = x * Sigmoid(1.702 * x)
         F_dest[i] = x * fast_sigmoid(x);
     }
    break;
```
