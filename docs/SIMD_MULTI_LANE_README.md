# ESP32 8-Lane SIMD Multi-Lane 實作

## 🎯 專案說明

本專案實作了在 ESP32 CUDA VM 上支援 **8 個 lane 不同 Q/K/V** 的 SIMD 架構。

### 核心概念

在 SIMT (Single Instruction Multiple Threads) 架構中：

- **一條指令，多個執行**：每條指令同時在 8 個 lane 上執行
- **指令不分 lane**：Instruction 本身對所有 lane 完全相同
- **差異在初始化**：不同結果來自不同的初始寄存器值

## 📦 新增檔案

### Python 端

- `esp32_tools/simd_initializer.py` - SIMD lane 初始化器
- `test_multi_lane_transformer.py` - 多 lane 硬體測試腳本
- `test_simd_functions.py` - 單元測試腳本

### 文檔

- `docs/SIMD_LANE_GUIDE.md` - 完整架構說明
- `TEST_REPORT.md` - 測試報告

### ESP32 韌體參考

- `examples/esp32_cuda_vm/lane_init_example.h` - Lane 初始化範例

## 🚀 快速開始

### 1. 測試 Python 功能（無需硬體）

```bash
# 執行單元測試
python3 test_simd_functions.py
```

### 2. 生成多 lane 程序

```python
from esp32_tools.simd_initializer import (
    SIMDInitializer,
    get_sequential_lanes
)

# 定義 8 個 lane 的 Q/K/V
lane_qkv = get_sequential_lanes()  # (2,3,4) 到 (9,10,11)

# 生成程序和預期結果
program, expected_results = SIMDInitializer.create_transformer_program_multi_lane(lane_qkv)

# 查看預期結果
for lane_id in range(8):
    print(f"Lane {lane_id}: {expected_results[lane_id]}")
```

### 3. 硬體測試（需要 ESP32）

```bash
# Sequential 配置（推薦）
python3 test_multi_lane_transformer.py /dev/cu.usbserial-589A0095521 sequential

# Uniform 配置（所有 lane 相同）
python3 test_multi_lane_transformer.py /dev/cu.usbserial-589A0095521 uniform

# Random 配置
python3 test_multi_lane_transformer.py /dev/cu.usbserial-589A0095521 random
```

## 📊 範例輸出

### Sequential 配置的執行結果

```
Lane   Q(R0)  K      V(R2)  Attn(R1)  Res(R16)  SS(R20)
----   -----  -----  -----  --------  --------  -------
0      2      3      4      6         6         36
1      3      4      5      12        8         144
2      4      5      6      20        10        400
3      5      6      7      30        12        900
4      6      7      8      42        14        1764
5      7      8      9      56        16        3136
6      8      9      10     72        18        5184
7      9      10     11     90        20        8100
```

### 計算公式

- **Attention Score**: `R1 = Q × K`
- **Residual**: `R16 = Q + V`
- **Sum of Squares**: `R20 = R1 × R1`

## 🔧 ESP32 韌體實作

### 方法 1: 預加載配置（推薦）

在 `vm_core.cpp` 中：

```cpp
#include "lane_init_example.h"

void setup() {
    // 使用預定義配置
    initializeLanes(simd_engine, SEQUENTIAL_CONFIG);
}
```

### 方法 2: 動態初始化

在 `vm_core.cpp` 中添加命令處理：

```cpp
void handleCommand(String cmd) {
    if (handleLaneInitCommand(simd_engine, cmd)) {
        return;
    }
    // ... 其他命令
}
```

Python 端使用：

```python
# 初始化每個 lane
for lane_id, (Q, K, V) in enumerate(lane_qkv):
    conn.send_command(f"init_lane {lane_id} 0 {Q}")  # R0
    conn.send_command(f"init_lane {lane_id} 1 {K}")  # R1
    conn.send_command(f"init_lane {lane_id} 2 {V}")  # R2
```

## 📚 詳細文檔

- **架構說明**: [`docs/SIMD_LANE_GUIDE.md`](docs/SIMD_LANE_GUIDE.md)
- **測試報告**: [`TEST_REPORT.md`](TEST_REPORT.md)
- **原始 Architecture**: [`docs/architecture.md`](docs/architecture.md)

## ✅ 測試狀態

**Python 單元測試**: 6/6 通過 ✅

- ✅ 模組導入
- ✅ Lane 配置
- ✅ 指令編碼
- ✅ 程序生成
- ✅ 初始化註釋
- ✅ 邊界情況

**硬體測試**: 待 ESP32 韌體實作

## 🎓 關鍵概念

### ✅ 正確做法

```python
# 1. 定義每個 lane 不同的初始值
lane_qkv = [(2,3,4), (3,4,5), ..., (9,10,11)]

# 2. 生成統一的指令（不區分 lane）
program = [
    Instruction.imul(1, 0, 1),  # R1 = R0 * R1
    Instruction.iadd(16, 0, 2), # R16 = R0 + R2
    Instruction.exit_inst()
]

# 3. 在韌體端預加載寄存器
# Lane 0: R0=2, R1=3, R2=4
# Lane 1: R0=3, R1=4, R2=5
# ...

# 4. 執行 → 自動得到每個 lane 不同結果
```

### ❌ 錯誤做法

```python
# ❌ 不要試圖在指令中編碼 lane
for lane_id in range(8):
    Instruction.mov_lane(lane_id, 0, Q[lane_id])  # 沒有這種指令！

# ❌ 不要為每個 lane 寫不同指令
for lane_id in range(8):
    program[lane_id].append(Instruction.mov(0, Q[lane_id]))  # 錯誤！
```

## 🤝 貢獻

這個實作遵循 NVIDIA GPU 的 SIMT 架構原則，適用於：

- GPU 架構學習
- 並行計算教學
- Transformer 加速器原型
- EdgeAI 硬體實驗

## 📄 授權

MIT License

---

**問題或建議？** 請查看 [`docs/SIMD_LANE_GUIDE.md`](docs/SIMD_LANE_GUIDE.md) 獲取完整說明！
