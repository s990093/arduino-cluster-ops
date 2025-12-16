# ✅ Micro-CUDA ISA v1.5 已烧入 ESP32

## 🎉 上传成功

ESP32 CUDA VM v1.5 已成功编译并烧入到 ESP32！

### 📦 烧入的内容

**固件版本**: Micro-CUDA ISA v1.5  
**架构**: True SIMT (8-Lane)  
**Warp Size**: 8 Lanes

**核心文件** (共 10 个):

1. `esp32_cuda_vm.ino` - 主程序
2. `vm_core.h/cpp` - VM 核心（指令调度）
3. `vm_simd_v15.h/cpp` - SIMD 引擎（8-Lane 并行执行）
4. `vm_trace.h/cpp` - Trace 单元
5. `instructions_v15.h` - ISA 定义
6. `README.md` - 文档
7. `ENHANCED_TRACE.md` - Trace 说明

---

## 🚀 快速开始

### 1. 连接串口监视器

```bash
python3 cli.py monitor -p /dev/cu.usbserial-589A0095521 -b 115200
```

### 2. 查看启动信息

应该看到：

```
========================================
 ESP32 Micro-CUDA VM v1.5
 Architecture: True SIMT
 Warp Size: 8 Lanes
========================================

✅ VM Initialized
✅ SIMD Engine Ready (8 Lanes)

Commands:
  load <hex>     - Load instruction
  run            - Execute program
  reset          - Reset VM
  reg            - Show registers (Lane 0)
  reg <lane>     - Show specific lane
  trace:stream   - Enable streaming trace
  help           - Show this help
```

### 3. 测试基本功能

```bash
# 加载简单程序
load 0xF01F0200    # S2R R31, SR_LANEID (获取 Lane ID)
load 0x10000010    # MOV R0, 0x10
load 0x01000000    # EXIT

# 执行
run

# 查看寄存器（每个 Lane 应该有不同的 SR_LANEID）
reg 0
reg 1
reg 7
```

---

## 🎯 运行 Parallel Attention 演示

### Python 端生成程序

```bash
python examples_usage/demo_parallel_attention_v15.py
```

这会生成：

- ✅ `parallel_attention_v15.hex` - 可执行程序

### 上传并执行

#### 方法 1: 手动加载

```python
# 在串口监视器中逐行粘贴：
load 0xF01F0200
load 0xF2000001
load 0x10000010
load 0x10010020
load 0x10020030
load 0x650A0000
load 0x650B0100
load 0x650C0200
load 0xF2000002
load 0x13140A0B
load 0x1115140C
load 0xF2000003
load 0x10030040
load 0x67150300
load 0x05000000
load 0x01000000
run
```

#### 方法 2: Python 自动化（TODO）

创建自动上传脚本。

---

## 📊 预期结果

### Lane-by-Lane 寄存器

每个 Lane 应该得到不同的结果：

```
Lane 0: R31=0,  R10=Q[0], R11=K[0], R20=Attention[0]
Lane 1: R31=1,  R10=Q[1], R11=K[1], R20=Attention[1]
...
Lane 7: R31=7,  R10=Q[7], R11=K[7], R20=Attention[7]
```

### 验证 SIMT

关键验证点：

1. ✅ 每个 Lane 的 `SR_LANEID` 不同（0-7）
2. ✅ 单条 `LDL` 指令加载不同数据
3. ✅ 每个 Lane 并行计算不同结果

---

## 🛠️ 可用命令

### 基本操作

| 命令         | 功能     | 示例              |
| ------------ | -------- | ----------------- |
| `load <hex>` | 加载指令 | `load 0xF01F0200` |
| `run`        | 执行程序 | `run`             |
| `reset`      | 重置 VM  | `reset`           |
| `help`       | 显示帮助 | `help`            |

### 调试命令

| 命令           | 功能               | 示例           |
| -------------- | ------------------ | -------------- |
| `reg`          | 显示 Lane 0 寄存器 | `reg`          |
| `reg <lane>`   | 显示指定 Lane      | `reg 3`        |
| `trace:stream` | 启用 trace         | `trace:stream` |
| `trace:off`    | 关闭 trace         | `trace:off`    |

---

## 📝 ISA v1.5 核心指令

### System Register

```
S2R R31, SR_LANEID    # 0xF01F0200 - 读取 Lane ID
```

### SIMT Memory

```
LDL R10, [R0]         # 0x650A0000 - Lane-Based Load
STL [R3], R21         # 0x67150300 - Lane-Based Store
LDX R10, [R0+R1]      # 0x640A0001 - Indexed Load
```

### Integer ALU

```
MOV R0, 0x10          # 0x10000010 - Move immediate
IADD R2, R0, R1       # 0x11020001 - Add
IMUL R20, R10, R11    # 0x13140A0B - Multiply
```

### Control

```
EXIT                  # 0x01000000 - Exit program
BAR.SYNC              # 0x05000000 - Barrier sync
```

---

## 🔧 重新烧录

如需重新烧录：

```bash
./upload_esp32.sh
```

或指定其他串口：

```bash
./upload_esp32.sh /dev/ttyUSB0
```

---

## 📚 相关文档

- `docs/MICRO_CUDA_ISA_V15_SPEC.md` - 完整 ISA 规格
- `ISA_V15_COMPLETED.md` - 实现总结
- `examples_usage/demo_parallel_attention_v15.py` - 演示程序

---

## 🎓 下一步

1. **测试基本功能**

   ```bash
   # 连接监视器
   python3 cli.py monitor -p /dev/cu.usbserial-589A0095521 -b 115200

   # 测试 Lane ID
   load 0xF01F0200
   load 0x01000000
   run
   reg 0
   reg 7
   ```

2. **运行 Parallel Attention**

   - 生成程序：`python examples_usage/demo_parallel_attention_v15.py`
   - 手动加载所有指令
   - 执行并查看结果

3. **开发自己的 Kernel**
   - 使用 `program_loader_v15.py`
   - 编写 SIMT 程序
   - 测试并验证

---

## ✅ 验证清单

- [x] 固件成功编译
- [x] 固件成功烧录
- [x] ESP32 正常启动
- [ ] 基本指令测试
- [ ] Lane ID 验证
- [ ] SIMT 内存操作测试
- [ ] Parallel Attention 演示

---

**Micro-CUDA ISA v1.5 已就绪！开始体验 True SIMT 架构吧！** 🎊
