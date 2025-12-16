# 8-Lane SIMD Transformer 架構說明

## 🎯 核心概念

在 GPU 和你的 ESP32 CUDA 模擬器中，採用的是 **SIMT (Single Instruction Multiple Threads)** 架構：

### 關鍵特性

1. **一條指令，多個執行緒**

   - 每條 `Instruction` 同時在所有 8 個 lane 上執行
   - Instruction 本身**不記錄 lane 資訊**
   - 對所有 lane 來說，指令是完全相同的

2. **不同結果的來源**

   - 每個 lane 有**獨立的寄存器檔案** (R0-R31, F0-F31, P0-P7)
   - 執行相同指令時，使用各自 lane 的寄存器值
   - 因此產生不同的計算結果

3. **類比 GPU Warp**
   - 8 個 lane = 1 個 mini-warp
   - 真實 GPU 的 warp 通常是 32 個 thread
   - 執行模型完全相同

---

## 🔧 實作方法

### 問題：如何給每個 lane 不同的 Q/K/V？

你想在 8 個 lane 中執行 Transformer 計算，每個 lane 有不同的輸入：

```
Lane 0: Q=2, K=3, V=4
Lane 1: Q=3, K=4, V=5
Lane 2: Q=4, K=5, V=6
...
Lane 7: Q=9, K=10, V=11
```

### 解決方案

#### ❌ 錯誤方法：試圖在 Instruction 中編碼 lane 資訊

```python
# 這是錯的！Instruction 不應該區分 lane
for lane_id in range(8):
    program.append(Instruction.mov_lane(lane_id, 0, Q[lane_id]))  # ❌
```

#### ✅ 正確方法：在初始化階段設定不同的寄存器值

有兩種實作途徑：

---

### 方法 1：韌體端預加載（推薦）

在 ESP32 韌體中，載入程序前先初始化每個 lane 的寄存器：

**Python 端：**

```python
from esp32_tools.simd_initializer import SIMDInitializer

# 定義每個 lane 的 Q/K/V
lane_qkv = [
    (2,3,4), (3,4,5), (4,5,6), (5,6,7),
    (6,7,8), (7,8,9), (8,9,10), (9,10,11)
]

# 創建程序（指令與單 lane 時完全相同）
program, expected = SIMDInitializer.create_transformer_program_multi_lane(lane_qkv)

# 程序內容（所有 lane 執行相同指令）：
# IMUL R1, R0, R1    # Attention Score
# IADD R16, R0, R2   # Residual
# IMUL R20, R1, R1   # Sum of Squares
# EXIT
```

**ESP32 韌體端 (vm_core.cpp)：**

```cpp
void VMCore::loadProgram() {
    // 在載入程序前，預初始化每個 lane 的寄存器

    // Lane 0: Q=2, K=3, V=4
    simd_engine.lanes[0].R[0] = 2;
    simd_engine.lanes[0].R[1] = 3;
    simd_engine.lanes[0].R[2] = 4;

    // Lane 1: Q=3, K=4, V=5
    simd_engine.lanes[1].R[0] = 3;
    simd_engine.lanes[1].R[1] = 4;
    simd_engine.lanes[1].R[2] = 5;

    // ... Lane 2-7 類似
}
```

**執行流程：**

1. 韌體預加載不同 lane 的 R0/R1/R2
2. Python 發送程序（統一的指令）
3. 每條指令在 8 個 lane 上並行執行
4. 因為寄存器不同，結果也不同

---

### 方法 2：特殊初始化指令（複雜）

如果你想從 Python 端動態初始化，需要：

1. **定義新 Opcode**（例如 `OP_INIT_LANE`）
2. **編碼格式包含 lane_id 和值**
3. **韌體端解析並只更新對應 lane**

但這違反了 SIMT 原則，不推薦。

---

## 📊 執行範例

### 輸入（8 個 lane）

```
Lane 0: R0=2, R1=3, R2=4
Lane 1: R0=3, R1=4, R2=5
Lane 2: R0=4, R1=5, R2=6
Lane 3: R0=5, R1=6, R2=7
Lane 4: R0=6, R1=7, R2=8
Lane 5: R0=7, R1=8, R2=9
Lane 6: R0=8, R1=9, R2=10
Lane 7: R0=9, R1=10, R2=11
```

### 指令序列（所有 lane 相同）

```assembly
IMUL R1, R0, R1    # R1 = R0 * R1 (Attention Score)
IADD R16, R0, R2   # R16 = R0 + R2 (Residual)
IADD R17, R0, R2
IADD R18, R0, R2
IADD R19, R0, R2
IMUL R20, R1, R1   # R20 = R1 * R1 (Sum of Squares)
EXIT
```

### 輸出（每個 lane 不同結果）

| Lane | Q (R0) | K   | V (R2) | Attn (R1) | Residual (R16) | Sum of Squares (R20) |
| ---- | ------ | --- | ------ | --------- | -------------- | -------------------- |
| 0    | 2      | 3   | 4      | 6         | 6              | 36                   |
| 1    | 3      | 4   | 5      | 12        | 8              | 144                  |
| 2    | 4      | 5   | 6      | 20        | 10             | 400                  |
| 3    | 5      | 6   | 7      | 30        | 12             | 900                  |
| 4    | 6      | 7   | 8      | 42        | 14             | 1764                 |
| 5    | 7      | 8   | 9      | 56        | 16             | 3136                 |
| 6    | 8      | 9   | 10     | 72        | 18             | 5184                 |
| 7    | 9      | 10  | 11     | 90        | 20             | 8100                 |

---

## 🛠️ 使用新工具

### 安裝

新增的檔案：

- `esp32_tools/simd_initializer.py` - SIMD 初始化器
- `test_multi_lane_transformer.py` - 多 lane 測試腳本

### 執行測試

```bash
# 序列配置（每個 lane 遞增）
python test_multi_lane_transformer.py /dev/cu.usbserial-589A0095521 sequential

# 統一配置（所有 lane 相同）
python test_multi_lane_transformer.py /dev/cu.usbserial-589A0095521 uniform

# 隨機配置
python test_multi_lane_transformer.py /dev/cu.usbserial-589A0095521 random
```

### 程式碼範例

```python
from esp32_tools.simd_initializer import SIMDInitializer

# 1. 定義 8 個 lane 的 Q/K/V
lane_qkv = [
    (2,3,4), (3,4,5), (4,5,6), (5,6,7),
    (6,7,8), (7,8,9), (8,9,10), (9,10,11)
]

# 2. 創建程序（自動計算每個 lane 的預期結果）
program, expected_results = SIMDInitializer.create_transformer_program_multi_lane(lane_qkv)

# 3. 打印預期結果
for lane_id in range(8):
    print(f"Lane {lane_id}: {expected_results[lane_id]}")

# 4. 載入到 ESP32
from esp32_tools import ESP32Connection, ProgramLoader
conn = ESP32Connection('/dev/cu.usbserial-589A0095521')
ProgramLoader.load_program(conn, program)
```

---

## ⚠️ 重要提醒

### 當前限制

1. **MOV 指令是 broadcast**

   - `MOV R0, 5` 會將所有 lane 的 R0 設為 5
   - 無法通過 MOV 給不同 lane 設不同值

2. **需要韌體支持**

   - 必須在 `vm_core.cpp` 中實作 lane 預加載
   - 或者通過 UART 接收 lane 初始化資料

3. **Trace 輸出**
   - 確保 `vm_trace.cpp` 輸出所有 8 個 lane 的狀態
   - JSON 格式應包含 `"lanes": [...]` 數組

---

## 🔮 韌體端實作建議

### 新增 lane 初始化命令

在 `vm_core.cpp` 中添加：

```cpp
void VMCore::handleCommand(String cmd) {
    if (cmd.startsWith("init_lane ")) {
        // 格式: init_lane <lane_id> <reg> <value>
        // 例如: init_lane 0 0 2

        int lane_id = parse_lane_id(cmd);
        int reg = parse_reg(cmd);
        uint32_t value = parse_value(cmd);

        simd_engine.lanes[lane_id].R[reg] = value;

        Serial.println("OK lane_init");
    }
    // ... 其他命令
}
```

### Python 端使用

```python
# 初始化 lane 寄存器
for lane_id, (Q, K, V) in enumerate(lane_qkv):
    conn.send_command(f"init_lane {lane_id} 0 {Q}")  # R0 = Q
    conn.send_command(f"init_lane {lane_id} 1 {K}")  # R1 = K
    conn.send_command(f"init_lane {lane_id} 2 {V}")  # R2 = V

# 然後載入程序
ProgramLoader.load_program(conn, program)
```

---

## ✅ 總結

| 概念                      | 說明                     |
| ------------------------- | ------------------------ |
| **Instruction 不分 lane** | 所有 lane 執行相同指令   |
| **寄存器分 lane**         | 每個 lane 有獨立的 R/F/P |
| **初始化階段設定差異**    | 通過預加載或特殊命令設定 |
| **自動產生不同結果**      | SIMT 架構自然特性        |

**核心原則：Instruction 像單 lane 一樣寫，差異在初始化！**
