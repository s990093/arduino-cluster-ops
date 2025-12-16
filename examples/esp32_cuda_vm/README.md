# ESP32 CUDA ISA Virtual Machine

## 概述

在 ESP32 上实现的 Layer 3 Thread Executor 虚拟机，支持完整的 CUDA ISA 指令集模拟。

## 硬件要求

- ESP32 开发板（推荐 ESP32-S3 或 TTGO T-Display）
- USB 连接用于串口通信

## 功能特性

### 寄存器文件

- **R0-R31**: 32 个 32-bit 通用寄存器
- **F0-F31**: 32 个 32-bit 浮点寄存器 (IEEE-754)
- **P0-P7**: 8 个 1-bit 谓词寄存器
- **SR0-SR31**: 32 个 32-bit 系统寄存器

### 支持的指令集

#### Group 1: Control Flow

- `NOP`, `EXIT`, `BRA`, `BR.Z`, `BAR.SYNC`, `WAIT.DMA`, `YIELD`

#### Group 2: Integer ALU

- `MOV`, `IADD`, `ISUB`, `IMUL`
- `AND`, `OR`, `XOR`
- `ISETP.EQ`, `ISETP.NE`, `ISETP.GT`, `ISETP.LT`

#### Group 3: Float & Deep Learning

- `FADD`, `FSUB`, `FMUL`, `FFMA`
- `HMMA.INT8` (Tensor Core 模拟)
- `SFU.RCP`, `SFU.SQRT`, `SFU.EXP`
- `SFU.GELU`, `SFU.RELU`

#### Group 4: Memory

- `LDG`, `STG` (Global Memory)
- `LDS`, `STS` (Shared Memory)
- `ATOM.ADD` (Atomic Operations)

#### Group 5: System

- `S2R`, `R2S` (System Register Access)
- `TRACE` (Debug)

#### Group 6: Vector

- `VADD.INT8`, `VMUL.INT8` (4-way SIMD)

## 使用方法

### 1. 烧录固件

```bash
python3 cli.py upload examples/esp32_cuda_vm/esp32_cuda_vm.ino \
  --port /dev/cu.usbserial-XXX \
  --board esp32
```

### 2. 连接串口

```bash
python3 cli.py monitor --port /dev/cu.usbserial-XXX --baudrate 115200
```

### 3. 串口命令

| 命令          | 说明         | 示例            |
| ------------- | ------------ | --------------- |
| `load:<hex>`  | 加载指令     | `load:11010203` |
| `run`         | 运行程序     | `run`           |
| `step`        | 单步执行     | `step`          |
| `reg`         | 显示寄存器   | `reg`           |
| `state`       | 显示 VM 状态 | `state`         |
| `list`        | 列出程序     | `list`          |
| `reset`       | 重置 VM      | `reset`         |
| `clear`       | 清空程序     | `clear`         |
| `demo:<name>` | 加载示例     | `demo:add`      |
| `help`        | 帮助信息     | `help`          |

## 示例程序

### 1. 简单加法

```
demo:add
run
```

程序：

```assembly
MOV R2, 5
MOV R3, 3
IADD R1, R2, R3  ; R1 = 5 + 3
EXIT
```

### 2. 循环累加

```
demo:loop
run
```

程序：

```assembly
MOV R0, 0       ; counter
MOV R1, 10      ; limit
MOV R2, 1       ; increment
loop:
  IADD R0, R0, R2
  ISETP.GT P3, R0, R1
  BR.Z P0, loop  ; if R0 < 10, goto loop
EXIT
```

### 3. 向量加法

```
demo:vector
run
```

程序：

```assembly
MOV R2, 0x01020304
MOV R3, 0x05060708
VADD.INT8 R1, R2, R3  ; 4x INT8 SIMD add
EXIT
```

## 手动编写程序

### 指令格式

```
32-bit = [OPCODE:8][DEST:8][SRC1:8][SRC2/IMM:8]
```

### 示例：计算 2 + 3

```
load:10020002  ; MOV R2, 2
load:10030003  ; MOV R3, 3
load:11010203  ; IADD R1, R2, R3
load:01000000  ; EXIT
run
```

## Opcode 参考

| Hex  | 指令      | 格式       |
| ---- | --------- | ---------- |
| 0x00 | NOP       | -          |
| 0x01 | EXIT      | -          |
| 0x10 | MOV       | Rd, Imm    |
| 0x11 | IADD      | Rd, Ra, Rb |
| 0x30 | FADD      | Fd, Fa, Fb |
| 0x40 | HMMA.INT8 | Rd, Ra, Rb |
| 0x54 | SFU.RELU  | Fd, Fa     |
| 0x60 | LDG       | Rd, [Ra]   |
| 0x80 | VADD.INT8 | Rd, Ra, Rb |

完整指令集请参考 `instructions.h`

## 性能

- CPU: ESP32 @ 240 MHz
- 吞吐量: ~50K instructions/sec
- 内存:
  - Program: 1024 instructions
  - Data (VRAM): 4KB
  - Shared (L1): 256B

## 故障排除

### 编译错误

如果遇到编译错误，请确保：

1. ESP32 core 已安装
2. 所有文件在同一目录

### 程序不执行

- 检查是否先 `load` 再 `run`
- 使用 `list` 查看已加载的程序
- 使用 `reset` 重置 VM

## 扩展开发

### 添加新指令

1. 在 `instructions.h` 中定义 opcode
2. 在 `vm_execute.cpp` 对应 group 中实现
3. 重新编译上传

### 自定义内存大小

修改 `vm_core.h` 中的：

```cpp
static const size_t PROGRAM_SIZE = 1024;
static const size_t DATA_SIZE = 4096;
```

## 许可证

与 arduino-cluster-ops 项目保持一致
