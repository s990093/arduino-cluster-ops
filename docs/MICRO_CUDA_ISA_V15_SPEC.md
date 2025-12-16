# Micro-CUDA ISA v1.5 正式版规格书

**架构代号**: Micro-Cluster (MC)  
**核心逻辑**: True SIMT (Single Instruction, Multiple Threads)  
**硬体分层**: Layer 1 (AMB82) → Layer 2 (ESP32-S3) → Layer 3 (RP2040)

---

## 🎯 v1.5 核心更新

### Lane-Awareness (通道感知)

从"同步广播阵列"进化为**真正的 SIMT 架构**。

**关键特性**:

- 每个 RP2040 (Lane) 具备独特身份 (`SR_LANEID`)
- 支持 Per-Lane 内存访问 (LDL/STL)
- 实现 Data Parallelism

---

## 1. 执行模型

### 1.1 集群全局视图

```
Grid / Kernel: Single Stream (Managed by AMB82 Master)
────────────────────────────────────────────────────────
SM0 (%smid=0) [Physical: ESP32-S3 Node 0]
 ├─ CTA0 (%ctaid=0)  -> Logical Block 0
 │   ├─ SMSP0 (%warpid=0) -> Warp Scheduler 0
 │   │   ├─ Lane 0 (%laneid=0)  [Physical: RP2040 Core 0]
 │   │   ├─ Lane 1 (%laneid=1)  [Physical: RP2040 Core 1]
 │   │   └─ ... (Up to Warp Size)
 │   └─ SMSP1 (%warpid=1)
 └─ ...
```

**术语**:

- **Warp**: 最小调度单位（指令由 ESP32 广播，Warp 内所有 Lane 同步接收）
- **Lane**: 最小执行单位（RP2040，通过 `SR_LANEID` 区分）

### 1.2 寄存器文件

每个 RP2040 独立拥有：

| 类型      | 前缀 | 数量 | 宽度   | 用途                           |
| --------- | ---- | ---- | ------ | ------------------------------ |
| General   | R    | 32   | 32-bit | 通用整数、地址、索引           |
| Float     | F    | 32   | 32-bit | FP32 浮点数                    |
| Predicate | P    | 8    | 1-bit  | 条件旗标（用于 Masking）       |
| System    | SR   | 32   | 32-bit | 系统状态（**包含 SR_LANEID**） |

### 1.3 指令编码（32-bit 固定）

```
[31:24 OPCODE] [23:16 DEST] [15:8 SRC1] [7:0 SRC2/IMM]
```

---

## 2. 完整指令集

### Group 1: 系统控制 (0x00-0x0F)

| Hex  | Mnemonic     | Operands | 功能              |
| ---- | ------------ | -------- | ----------------- |
| 0x00 | **NOP**      | -        | 空指令            |
| 0x01 | **EXIT**     | -        | 终止 Kernel       |
| 0x02 | **BRA**      | Imm      | 无条件跳转        |
| 0x03 | **BR.Z**     | Imm, Pn  | 条件跳转          |
| 0x05 | **BAR.SYNC** | Id       | Warp Barrier 同步 |
| 0x07 | **YIELD**    | -        | 让出时间片        |

### Group 2: 整数运算 (0x10-0x2F)

| Hex  | Mnemonic     | Operands   | Flags | 功能             |
| ---- | ------------ | ---------- | ----- | ---------------- |
| 0x10 | **MOV**      | Rd, Imm    | -     | 载入立即值       |
| 0x11 | **IADD**     | Rd, Ra, Rb | Z, C  | 整数加法         |
| 0x12 | **ISUB**     | Rd, Ra, Rb | Z, C  | 整数减法         |
| 0x13 | **IMUL**     | Rd, Ra, Rb | -     | 整数乘法         |
| 0x17 | **AND**      | Rd, Ra, Rb | Z     | 位元 AND         |
| 0x1A | **ISETP.EQ** | Pn, Ra, Rb | Pn    | 整数比较（相等） |
| 0x1C | **ISETP.GT** | Pn, Ra, Rb | Pn    | 整数比较（大于） |
| 0x1D | **SHL**      | Rd, Ra, Rb | -     | 左移             |
| 0x1E | **SHR**      | Rd, Ra, Rb | -     | 右移             |

### Group 3: 浮点与 AI (0x30-0x5F)

| Hex  | Mnemonic     | Operands   | 描述                 | 场景        |
| ---- | ------------ | ---------- | -------------------- | ----------- |
| 0x30 | **FADD**     | Fd, Fa, Fb | FP32 加法            | Bias        |
| 0x31 | **FSUB**     | Fd, Fa, Fb | FP32 减法            | -           |
| 0x32 | **FMUL**     | Fd, Fa, Fb | FP32 乘法            | Scaling     |
| 0x34 | **FFMA**     | Fd, Fa, Fb | $Fd = Fa × Fb + Fd$  | MAC         |
| 0x40 | **HMMA.I8**  | Rd, Ra, Rb | 4-way SIMD INT8 点积 | LLM Quant   |
| 0x50 | **SFU.RCP**  | Fd, Fa     | $1.0 / Fa$           | Softmax     |
| 0x53 | **SFU.GELU** | Fd, Fa     | GELU Activation      | Transformer |
| 0x54 | **SFU.RELU** | Fd, Fa     | ReLU: $\max(0, Fa)$  | CNN         |

### Group 4: 内存与 SIMT 寻址 (0x60-0x7F) ⭐

**核心更新区域**：区分"广播载入"与"SIMT 载入"

#### Uniform Operations（所有 Lane 相同地址）

| Hex  | Mnemonic | Operands  | 行为                           |
| ---- | -------- | --------- | ------------------------------ |
| 0x60 | **LDG**  | Rd, [Ra]  | 所有 Lane 读取相同地址（广播） |
| 0x61 | **STG**  | [Ra], Rd  | 所有 Lane 写入相同地址         |
| 0x62 | **LDS**  | Rd, [Imm] | 从 Shared Memory 读取          |

#### SIMT Operations（每个 Lane 不同地址）**[NEW in v1.5]**

| Hex  | Mnemonic | Operands    | 行为逻辑                                                                                       |
| ---- | -------- | ----------- | ---------------------------------------------------------------------------------------------- |
| 0x65 | **LDL**  | Rd, [Ra]    | **Lane-Based Load**<br>每个 Lane 计算：`Addr = Ra + SR_LANEID * 4`<br>硬件自动添加 Lane Offset |
| 0x67 | **STL**  | [Ra], Rd    | **Lane-Based Store**<br>每个 Lane 写入：`Addr = Ra + SR_LANEID * 4`                            |
| 0x64 | **LDX**  | Rd, [Ra+Rb] | **Indexed SIMT Load**<br>每个 Lane 计算：`Addr = Ra + Rb`<br>（Rb 是 Lane 私有寄存器）         |
| 0x66 | **STX**  | [Ra+Rb], Rd | **Indexed SIMT Store**<br>Scatter Write                                                        |

#### Atomic Operations

| Hex  | Mnemonic     | 功能       |
| ---- | ------------ | ---------- |
| 0x70 | **ATOM.ADD** | Atomic Add |

### Group 5: 系统寄存器 (0xF0-0xFF) ⭐

| Hex  | Mnemonic  | Operands | 功能                               |
| ---- | --------- | -------- | ---------------------------------- |
| 0xF0 | **S2R**   | Rd, SRn  | System to Register（读取系统状态） |
| 0xF1 | **R2S**   | SRn, Rd  | Register to System                 |
| 0xF2 | **TRACE** | Imm      | 发送 Trace ID                      |

---

## 3. 系统寄存器映射（SR）

**物理基础**：所有 RP2040 在硬件初始化时被分配固定 ID

| SR Index | 名称          | 定义与用途                                                         |
| -------- | ------------- | ------------------------------------------------------------------ |
| **SR_0** | SR_TID        | Local Thread ID (Physical Core ID)                                 |
| **SR_1** | SR_CTAID      | Block ID (Logical Job ID)                                          |
| **SR_2** | **SR_LANEID** | **[NEW] Lane Index (0..WarpSize-1)**<br>用于 `LDL` 指令计算 Offset |
| **SR_3** | SR_WARPSIZE   | Warp Size（通常为 8）                                              |
| **SR_6** | SR_GPU_UTIL   | Core 负载率                                                        |
| **SR_8** | SR_WARP_ID    | Warp ID，用于同步                                                  |
| **SR_9** | SR_SM_ID      | SM ID (ESP32 Node ID)                                              |

---

## 4. 实战范例：Parallel Attention (Q/K/V)

### 场景

- Warp Size = 8 (8 Lanes)
- Q, K, V 数组存放在 VRAM 连续地址
- **目标**：Lane $i$ 读取 $Q[i], K[i], V[i]$ 并并行计算

### 程序代码

```assembly
; === Initialization ===
; 1. 获取当前 Lane ID (0~7)
S2R   R31, SR_LANEID     ; R31 = My Lane ID

; 2. 设定 Q/K/V 的基底地址
MOV   R0, 0x10          ; R0 = Base of Q (0x1000 >> 8)
MOV   R1, 0x20          ; R1 = Base of K
MOV   R2, 0x30          ; R2 = Base of V

; === SIMT Loading（关键）===
; 硬件会执行: Effective_Addr = Base + LaneID * 4
LDL   R10, [R0]         ; R10 = Q[lane]
LDL   R11, [R1]         ; R11 = K[lane]
LDL   R12, [R2]         ; R12 = V[lane]

; === Parallel Execution ===
; 每个 Lane 的 R10, R11, R12 都不同
IMUL  R20, R10, R11     ; R20 = Q[i] * K[i] (Attention Score)
IADD  R21, R20, R12     ; R21 = Score + V[i]

; === Write Back ===
MOV   R3, 0x40          ; R3 = Result base
STL   [R3], R21         ; Store Result[lane]

EXIT
```

### 硬件行为解析

执行 `LDL R10, [R0]` 时：

1. **ESP32 (SM)**：发送指令 `0x650A0000`
2. **RP2040 (Lane 0)**：
   - 读取 `R0 (0x1000)`
   - 读取 `SR_LANEID (0)`
   - 计算地址 `0x1000 + 0*4 = 0x1000`
   - 执行 Load
3. **RP2040 (Lane 1)**：
   - 读取 `R0 (0x1000)`
   - 读取 `SR_LANEID (1)`
   - 计算地址 `0x1000 + 1*4 = 0x1004`
   - 执行 Load
4. **结果**：1 个 Cycle 内，所有 Lane 完成不同的内存访问

---

## 5. 执行结果示例

### 输入数据（VRAM）

```
0x1000: Q[0]=2, Q[1]=3, Q[2]=4, ..., Q[7]=9
0x2000: K[0]=3, K[1]=4, K[2]=5, ..., K[7]=10
0x3000: V[0]=4, V[1]=5, V[2]=6, ..., V[7]=11
```

### 执行结果（每个 Lane 不同）

| Lane | Q(R10) | K(R11) | V(R12) | Attn(R20) | Final(R21) |
| ---- | ------ | ------ | ------ | --------- | ---------- |
| 0    | 2      | 3      | 4      | 6         | 10         |
| 1    | 3      | 4      | 5      | 12        | 17         |
| 2    | 4      | 5      | 6      | 20        | 26         |
| 3    | 5      | 6      | 7      | 30        | 37         |
| 4    | 6      | 7      | 8      | 42        | 50         |
| 5    | 7      | 8      | 9      | 56        | 65         |
| 6    | 8      | 9      | 10     | 72        | 82         |
| 7    | 9      | 10     | 11     | 90        | 101        |

---

## 6. 实现文件清单

### ESP32 固件

- `instructions_v15.h` - ISA 定义
- `vm_simd_v15.h` - SIMD 引擎头文件
- `vm_simd_v15.cpp` - SIMD 执行逻辑（含 SIMT 内存操作）

### Python 工具

- `program_loader_v15.py` - 指令编码器
- `demo_parallel_attention_v15.py` - Parallel Attention 演示

---

## 7. 与 v1.0 对比

| 特性       | v1.0              | v1.5             |
| ---------- | ----------------- | ---------------- |
| 内存模型   | 广播（Broadcast） | SIMT（Per-Lane） |
| Lane 身份  | 无                | SR_LANEID        |
| Q/K/V 加载 | 需 Host 轮询      | 单条 LDL 指令    |
| 数据并行   | 不支持            | 完全支持         |
| 架构       | 同步阵列          | True SIMT        |

---

## 8. 编程模型

### 旧模型（v1.0）

```python
# Host 需要循环控制
for lane_id in range(8):
    load_q(lane_id)
    load_k(lane_id)
    compute(lane_id)
```

### 新模型（v1.5）

```assembly
; 单条指令，并行执行
LDL R10, [R0]  ; 所有 Lane 同时加载不同的 Q
LDL R11, [R1]  ; 所有 Lane 同时加载不同的 K
IMUL R20, R10, R11  ; 所有 Lane 同时计算
```

**优势**：

- ✅ 减少 Host 开销
- ✅ 真正的硬件并行
- ✅ 符合 GPU 编程范式

---

## 9. 下一步

1. **编译固件**：上传到 ESP32
2. **运行演示**：`python demo_parallel_attention_v15.py`
3. **扩展应用**：实现完整的 Multi-Head Attention

---

**Micro-CUDA ISA v1.5 正式发布！现在支持真正的 Data Parallelism！**
