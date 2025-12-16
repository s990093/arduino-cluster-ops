# ESP32 Tools 模組

ESP32 測試工具模組，提供簡單易用的接口來測試 ESP32 Transformer。

## 📦 模組結構

```
esp32_tools/
├── __init__.py       # 模組入口
├── connection.py     # ESP32Connection - 串口連接管理
├── trace.py          # TraceCollector - Trace 收集和解析
├── analyzer.py       # ResultAnalyzer - 結果分析
└── tester.py         # TransformerTester - 主測試類
```

## 🚀 快速開始

### 方式 1: 使用 TransformerTester (推薦)

最簡單的使用方式，一行代碼完成測試：

```python
from esp32_tools import TransformerTester

tester = TransformerTester("/dev/cu.usbserial-589A0095521")
success = tester.run_test()
```

### 方式 2: 使用個別組件

更靈活的方式，可以自定義測試流程：

```python
from esp32_tools import ESP32Connection, TraceCollector, ResultAnalyzer

# 連接
conn = ESP32Connection("/dev/cu.usbserial-589A0095521")

# 自定義操作
conn.send_command("demo:transformer")
output, elapsed = TraceCollector.collect_execution_trace(conn)
trace_records = TraceCollector.parse_trace_json(output)

# 分析
success = ResultAnalyzer.analyze(registers, trace_records)

conn.close()
```

### 方式 3: 只使用連接功能

如果只需要發送命令並讀取回應：

```python
from esp32_tools import ESP32Connection

conn = ESP32Connection("/dev/cu.usbserial-589A0095521")
conn.send_command("reg")
response = conn.read_lines()
conn.close()
```

## 📚 API 文檔

### ESP32Connection

管理 ESP32 串口連接

```python
conn = ESP32Connection(port, baudrate=115200, timeout=2.0)
conn.send_command(cmd, delay=0.3)  # 發送命令
lines = conn.read_lines()           # 讀取輸出
conn.close()                        # 關閉連接
```

### TraceCollector

收集和解析執行 Trace

```python
# 收集 trace
output, elapsed = TraceCollector.collect_execution_trace(connection, max_wait=30)

# 解析 JSON trace
trace_records = TraceCollector.parse_trace_json(output)

# 解析寄存器
registers = TraceCollector.parse_registers(lines)
```

### ResultAnalyzer

分析測試結果

```python
# 分析並打印報告
success = ResultAnalyzer.analyze(registers, trace_records)

# 預期值
ResultAnalyzer.EXPECTED_VALUES  # 包含預期的寄存器值
```

### TransformerTester

完整的測試流程

```python
tester = TransformerTester(port, baudrate=115200)
success = tester.run_test()  # 執行完整測試並返回結果

# 訪問測試數據
tester.trace_records   # Trace 記錄
tester.registers       # 寄存器值
tester.elapsed_time    # 執行時間
```

## 📝 完整範例

查看 `example_usage.py` 獲取更多使用範例。

## ✨ 特性

- ✅ 清晰的模組化設計
- ✅ 類型提示支持
- ✅ 完善的錯誤處理
- ✅ 簡單易用的 API
- ✅ 靈活的使用方式
- ✅ 自動資源管理

## 🔧 依賴

- `pyserial` - 用於串口通訊
- Python 3.7+

安裝依賴：

```bash
pip install pyserial
```
