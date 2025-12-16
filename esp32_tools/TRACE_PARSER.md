# ESP32 Tools - Trace Parser Module

## 概述

`trace_parser.py` 提供了強大的 ESP32 CUDA VM JSON trace 解析功能，包含錯誤恢復和數據搶救邏輯。

## 功能特性

### 主要函數

#### `parse_enhanced_trace(trace_output, initial_mem=None)`

解析增強型 trace 輸出，支持錯誤處理和數據搶救。

**參數:**

- `trace_output` (str): 包含 JSON trace 的原始文本輸出
- `initial_mem` (List[Dict], optional): 初始記憶體狀態列表

**返回:**

- `Dict`: 解析後的 trace 字典，包含以下字段:
  - `trace_version`: 版本號
  - `architecture`: 架構類型 (SIMT)
  - `program`: 程式名稱
  - `warp_size`: Warp 大小
  - `total_instructions`: 總指令數
  - `records`: 執行記錄列表
  - `initial_memory`: 初始記憶體狀態

**使用範例:**

```python
from esp32_tools import parse_enhanced_trace

# 解析 trace 輸出
trace_data = parse_enhanced_trace(
    raw_output,
    initial_mem=[{"addr": 0, "val": 123}]
)
print(f"Parsed {trace_data['total_instructions']} instructions")
```

#### `verify_trace_memory_values(trace_data, expected_nonzero=True)`

驗證 trace 中的記憶體訪問值。

**參數:**

- `trace_data` (Dict): 解析後的 trace 字典
- `expected_nonzero` (bool): 是否期望有非零值

**返回:**

- `Dict`: 驗證結果
  - `total_records`: 總記錄數
  - `records_with_memory`: 包含記憶體訪問的記錄數
  - `nonzero_values`: 非零值數量
  - `passed`: 是否通過驗證

**使用範例:**

```python
from esp32_tools import verify_trace_memory_values

verification = verify_trace_memory_values(trace_data)
if verification['passed']:
    print(f"✅ Found {verification['nonzero_values']} non-zero values")
else:
    print("❌ All memory values are zero")
```

#### `save_trace_json(trace_data, filepath)`

保存 trace 數據到 JSON 文件。

**參數:**

- `trace_data` (Dict): 解析後的 trace 字典
- `filepath` (str): 輸出文件路徑

**使用範例:**

```python
from esp32_tools import save_trace_json

save_trace_json(trace_data, "output/trace.json")
```

## 錯誤處理

模組包含強大的錯誤恢復機制:

1. **全局 JSON 解析**: 首先嘗試將整個輸出解析為有效的 JSON
2. **記錄級搶救**: 如果全局解析失敗，使用正則表達式提取單個記錄
3. **字段提取**: 從損壞的記錄中提取關鍵字段（cycle, pc, memory_access）
4. **重構記錄**: 使用提取的數據重建可用的記錄

## 完整示例

```python
from esp32_tools import (
    ESP32Connection,
    parse_enhanced_trace,
    verify_trace_memory_values,
    save_trace_json
)

# 連接並執行
conn = ESP32Connection("/dev/cu.usbserial-XXX")
conn.send_command("trace:stream")
conn.send_command("run")

# 收集輸出
output = ""
for _ in range(50):
    lines = conn.read_lines()
    if lines:
        output += "\n".join(lines)

# 解析
trace_data = parse_enhanced_trace(output)

# 驗證
verification = verify_trace_memory_values(trace_data)
print(f"Memory ops: {verification['records_with_memory']}")

# 保存
save_trace_json(trace_data, "trace.json")
```

## 版本歷史

- **v2.1.0**: 初始發布，提取自 `test_enhanced_trace.py`
  - 添加 `parse_enhanced_trace`
  - 添加 `verify_trace_memory_values`
  - 添加 `save_trace_json`
