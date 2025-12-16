# ESP32 CUDA Tools - 快速參考卡

## 🚀 三秒鐘開始

```bash
# 1. 驗證
python verify_setup.py

# 2. 複製模板
cp examples_usage/template_custom_test.py my_test.py

# 3. 執行
python my_test.py
```

---

## 💡 最簡單的方式

```python
from esp32_tools import quick_run, Instruction

program = [
    Instruction.mov(0, 10),
    Instruction.exit_inst()
]

quick_run("/dev/cu.usbserial-YOUR_PORT", program)
```

---

## 📝 完整範例

```python
from esp32_tools import CUDARunner, Instruction

PORT = "/dev/cu.usbserial-YOUR_PORT"

program = [
    Instruction.mov(0, 10),
    Instruction.mov(1, 5),
    Instruction.imul(2, 0, 1),
    Instruction.exit_inst()
]

with CUDARunner(PORT) as runner:
    runner.run(program)
    runner.print_results()
    runner.verify_result({'R2': 50})
```

---

## 🔨 可用指令

```python
Instruction.mov(dest, imm)           # R[dest] = imm
Instruction.iadd(dest, src1, src2)   # R[dest] = R[src1] + R[src2]
Instruction.isub(dest, src1, src2)   # R[dest] = R[src1] - R[src2]
Instruction.imul(dest, src1, src2)   # R[dest] = R[src1] * R[src2]
Instruction.exit_inst()              # 退出
```

---

## 🎯 CUDARunner API

```python
# 基本
runner = CUDARunner(port)
runner.run(program)

# 輸出
runner.print_results()           # 顯示寄存器
runner.print_trace_summary()     # 顯示 trace
runner.verify_result(expected)   # 驗證結果

# 進階
runner.compile_and_load(program)
runner.execute(enable_trace=True)
runner.read_registers()
runner.save_trace("file.json")
```

---

## 📁 範例文件

| 文件 | 用途 | 執行 |
|------|------|------|
| `example1_basic.py` | 學習基礎 | `python examples_usage/example1_basic.py` |
| `example2_quick.py` | quick_run 用法 | `python examples_usage/example2_quick.py` |
| `example3_transformer.py` | 複雜計算 | `python examples_usage/example3_transformer.py` |
| `template_custom_test.py` | 自訂模板 | 複製後修改 |

---

## 🔍 找串口

```bash
# Mac
ls /dev/cu.usbserial-*

# Linux
ls /dev/ttyUSB*

# Windows
# 設備管理器 -> 端口(COM和LPT)
```

---

## 📚 文檔

| 文檔 | 內容 |
|------|------|
| `README_V2.md` | 快速開始 |
| `USAGE_GUIDE.md` | 完整指南 |
| `V2_MIGRATION_GUIDE.md` | v2.0 改造說明 |
| `FINAL_SUMMARY.md` | 完成報告 |

---

## ⚡ 快速測試

```bash
# 不需要硬體
python test_api_syntax.py
python test_simd_functions.py
python verify_setup.py

# 需要 ESP32
python examples_usage/example1_basic.py
```

---

## 🎓 學習路徑

1. **執行**: `python verify_setup.py`
2. **試試**: `python examples_usage/example1_basic.py`
3. **複製**: `cp examples_usage/template_custom_test.py my_test.py`
4. **修改**: 編輯 `my_test.py`
5. **執行**: `python my_test.py`

---

**就這麼簡單！開始寫你的程式吧！** 🚀
