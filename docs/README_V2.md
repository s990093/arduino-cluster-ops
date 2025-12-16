# ✅ ESP32 CUDA Tools v2.0 - 完成並驗證

## 🎉 改造完成

`esp32_tools` 已成功改造為**通用型測試框架**！

現在你可以：
✅ 直接在 Python 中寫程式
✅ 一鍵執行並查看結果  
✅ 自動收集和保存 trace

---

## 🚀 立即開始

### 1. 驗證安裝

```bash
python verify_setup.py
```

應該看到：

```
✅ PASS  導入測試
✅ PASS  程式創建
✅ PASS  Runner 創建
✅ PASS  範例文件
🎉 所有測試通過！準備就緒！
```

### 2. 最簡單的範例

創建 `my_first_test.py`:

```python
from esp32_tools import quick_run, Instruction

# 寫程式
program = [
    Instruction.mov(0, 10),
    Instruction.mov(1, 5),
    Instruction.imul(2, 0, 1),  # R2 = 10 * 5 = 50
    Instruction.exit_inst()
]

# 一行執行！
quick_run(
    "/dev/cu.usbserial-589A0095521",  # 改成你的串口
    program,
    expected={'R0': 10, 'R1': 5, 'R2': 50}
)
```

執行:

```bash
python my_first_test.py
```

### 3. 使用模板

```bash
# 複製模板
cp examples_usage/template_custom_test.py my_test.py

# 編輯 my_test.py，修改三個區域：
# 1. PORT = "你的串口"
# 2. program = [你的指令]
# 3. expected = {預期結果}

# 執行
python my_test.py
```

---

## 📚 完整文檔

| 文檔                      | 說明                         |
| ------------------------- | ---------------------------- |
| `USAGE_GUIDE.md`          | 完整使用指南（強烈推薦閱讀） |
| `V2_MIGRATION_GUIDE.md`   | v2.0 改造說明                |
| `docs/SIMD_LANE_GUIDE.md` | 8-Lane SIMD 架構說明         |
| `TEST_REPORT.md`          | SIMD 測試報告                |

---

## 🎯 提供的範例

在 `examples_usage/` 目錄：

| 範例                      | 說明             | 指令                                            |
| ------------------------- | ---------------- | ----------------------------------------------- |
| `example1_basic.py`       | 基礎數學運算     | `python examples_usage/example1_basic.py`       |
| `example2_quick.py`       | quick_run 用法   | `python examples_usage/example2_quick.py`       |
| `example3_transformer.py` | Transformer 計算 | `python examples_usage/example3_transformer.py` |
| `template_custom_test.py` | 自訂測試模板     | 複製後修改使用                                  |

---

## 🔧 三種使用方式

### 方式 1: quick_run（最簡單）

```python
from esp32_tools import quick_run, Instruction

program = [...]
quick_run(PORT, program, expected={...})
```

### 方式 2: CUDARunner（推薦）

```python
from esp32_tools import CUDARunner, Instruction

with CUDARunner(PORT) as runner:
    runner.run(program)
    runner.print_results()
    runner.verify_result(expected)
```

### 方式 3: 分步控制（進階）

```python
from esp32_tools import CUDARunner

runner = CUDARunner(PORT)
runner.connect()
runner.compile_and_load(program)
trace, elapsed = runner.execute()
registers = runner.read_registers()
runner.disconnect()
```

---

## 🛠️ 可用指令

```python
# 整數運算
Instruction.mov(dest, imm)           # MOV Rd, Imm
Instruction.iadd(dest, src1, src2)   # IADD Rd, Ra, Rb
Instruction.isub(dest, src1, src2)   # ISUB Rd, Ra, Rb
Instruction.imul(dest, src1, src2)   # IMUL Rd, Ra, Rb

# 控制
Instruction.exit_inst()              # EXIT
```

---

## 📊 完整範例

```python
#!/usr/bin/env python3
from esp32_tools import CUDARunner, Instruction

PORT = "/dev/cu.usbserial-589A0095521"

# 定義程式
program = [
    Instruction.mov(0, 10),        # R0 = 10
    Instruction.mov(1, 5),         # R1 = 5
    Instruction.iadd(2, 0, 1),     # R2 = 15
    Instruction.imul(3, 0, 1),     # R3 = 50
    Instruction.exit_inst()
]

# 預期結果
expected = {
    'R0': 10,
    'R1': 5,
    'R2': 15,
    'R3': 50
}

# 執行
with CUDARunner(PORT) as runner:
    runner.run(program, save_trace="my_trace.json")
    runner.print_results()
    passed = runner.verify_result(expected)

    if passed:
        print("✅ 測試通過！")
```

---

## 🎓 學習路徑

1. **第一步**: 執行 `python verify_setup.py` 驗證安裝
2. **第二步**: 修改並執行 `examples_usage/example1_basic.py`
3. **第三步**: 複製模板創建自己的測試
4. **第四步**: 閱讀 `USAGE_GUIDE.md` 了解進階用法

---

## ⚠️ 常見問題

### Q: ModuleNotFoundError: No module named 'esp32_tools'

**A**: 確保從專案根目錄執行，或使用範例文件（已包含路徑修復）

### Q: 如何找到串口路徑？

**A**:

- Mac: `ls /dev/cu.usbserial-*`
- Linux: `ls /dev/ttyUSB*`
- Windows: 設備管理器查看 COM 端口

### Q: 執行範例時找不到串口

**A**: 修改範例文件中的 `PORT` 變量為你的實際串口路徑

---

## 📦 項目結構

```
arduino-cluster-ops/
├── esp32_tools/              # 核心模組
│   ├── __init__.py
│   ├── runner.py            # ⭐ 通用執行器（新增）
│   ├── connection.py
│   ├── program_loader.py
│   ├── trace.py
│   └── simd_initializer.py
│
├── examples_usage/          # 使用範例
│   ├── example1_basic.py
│   ├── example2_quick.py
│   ├── example3_transformer.py
│   └── template_custom_test.py
│
├── docs/
│   ├── SIMD_LANE_GUIDE.md   # SIMD 架構說明
│   └── architecture.md
│
├── verify_setup.py          # 環境驗證腳本
├── USAGE_GUIDE.md           # ⭐ 完整使用指南
└── V2_MIGRATION_GUIDE.md    # v2.0 改造說明
```

---

## 🎯 核心改進

| 特性       | v1.0   | v2.0        |
| ---------- | ------ | ----------- |
| **代碼量** | ~50 行 | ~10 行      |
| **易用性** | ★★☆☆☆  | ★★★★★       |
| **範例**   | 2 個   | 4 個 + 模板 |
| **文檔**   | 基礎   | 完整        |

---

## 🚀 下一步

1. **連接 ESP32**
2. **執行驗證**: `python verify_setup.py`
3. **試試範例**: `python examples_usage/example1_basic.py`
4. **創建測試**: 複製模板開始寫你的程式！

---

## 📞 需要幫助？

- 🔥 **快速開始**: 看 `USAGE_GUIDE.md`
- 📚 **深入學習**: 看 `docs/SIMD_LANE_GUIDE.md`
- 💡 **範例參考**: 看 `examples_usage/`
- ✅ **驗證環境**: 執行 `python verify_setup.py`

---

**準備就緒！開始寫你的第一個 ESP32 CUDA 程式吧！** 🎊
