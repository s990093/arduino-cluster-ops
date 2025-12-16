# ESP32 Micro-CUDA Architecture (Front-End/SIMD)

## 🏗️ 架构概览

本项目在 ESP32 上实现了一个微型 NVIDIA GPU 架构，采用了经典的 **Front-End / Back-End 分离** 设计。这使得单个 ESP32 可以模拟一个 Mini-SM (Streaming Multiprocessor)，并具备扩展到多芯片集群的能力。

### 核心组件

```mermaid
graph TD
    User[Host PC / Python] -->|UART Command| Core0

    subgraph ESP32 [Mini-SM]
        subgraph FrontEnd ["Core 0: Front-End (Warp Scheduler)"]
            Fetch[Instruction Fetch]
            Decode[Decode & Issue]
            PC[PC Control]
            Trace[Trace Unit]
        end

        subgraph BackEnd ["Core 1: Back-End (SIMD Engine)"]
            SIMD["8-Lane SIMD Execution Unit"]
            RegFile["Register File (8x R[32])"]
        end

        FrontEnd -->|Instructions (Queue)| BackEnd
        BackEnd -->|Completion (Queue)| FrontEnd
    end
```

---

## 🚀 详细设计

### 1. Front-End (Core 0)

**角色：Warp Scheduler**

Core 0 负责所有控制流和指令调度，模拟 GPU 的 Front-End。

- **Instruction Fetch:** 从指令缓冲区 (`program_buffer`) 读取下一条指令。
- **PC Management:** 管理 Program Counter (PC)，处理跳转 (BRA) 和循环。
- **Dispatch:** 将解码后的指令通过 **FreeRTOS Queue** 发送给 Back-End。
- **Tracing:** 收集架构状态并输出详细的 JSON Trace（兼容 Nsight Compute 格式）。
- **I/O:** 处理来自 Host 的 UART 命令 (`load`, `run`, `halt`)。

### 2. Back-End (Core 1)

**角色：SIMD Execution Engine**

Core 1 负责并行计算，模拟 GPU 的 CUDA Cores。

- **SIMD Architecture:** 真正的 Single Instruction Multiple Data 架构。
- **8 Lanes:** 单个指令会被并行应用此 8 个独立的 Lane 上。
- **Register File:** 每个 Lane 拥有独立的寄存器状态：
  - 32x 32-bit General Purpose Registers (R0-R31)
  - 32x 32-bit Float Registers (F0-F31)
  - 8x Predicate Registers (P0-P7)

### 3. Inter-Core Communication

**机制：阻塞式同步队列**

为了保证时序精确和稳定性，采用了 FreeRTOS Queues 进行同步：

1.  **Instruction Queue (`Core0 -> Core1`):** Core 0 发送指令字。Core 1 阻塞等待，直到收到指令。
2.  **Completion Queue (`Core1 -> Core0`):** Core 1 执行完毕后发送结果状态。Core 0 阻塞等待，直到收到完成信号。

这种机制确保了 Core 0 (Front-End) 和 Core 1 (Back-End) 的完美同步，模拟了 GPU 流水线中的 Issue -> Execute -> Writeback 阶段。

---

## 💾 寄存器模型

| 类型  | 数量/Lane | 描述                                    |
| ----- | --------- | --------------------------------------- |
| **R** | 32        | 通用整数寄存器 (32-bit)                 |
| **F** | 32        | 浮点寄存器 (IEEE 754 float)             |
| **P** | 8         | Predicate 寄存器 (1-bit，用于分支/掩码) |

**SIMD 扩展:**
整个系统的寄存器总容量 = 8 Lanes \* (32 R + 32 F + 8 P)。

---

## 🔄 执行流程示例

对于指令 `IADD R2, R0, R1` (Integer Add):

1.  **Core 0:** Fetch 指令 `0x11020001`。
2.  **Core 0:** Push 到 Queue。
3.  **Core 1:** Pop 指令，解码。
4.  **Core 1:** 执行 8 次 Loop (并行模拟):
    - Lane 0: `R[2] = R[0] + R[1]` (使用 Lane 0 的寄存器)
    - ...
    - Lane 7: `R[2] = R[0] + R[1]` (使用 Lane 7 的寄存器)
5.  **Core 1:** Push 完成信号。
6.  **Core 0:** 收到信号，PC+1，输出 Trace。

---

## 📊 Trace 输出格式

Trace 系统现在输出完整的 8-lane 状态：

```json
{
  "cycle": 100,
  "pc": 12,
  "instruction": "0x11020001",
  "perf": {
    "core_id": 0,
    "simd_width": 8
  },
  "lanes": [
    { "lane_id": 0, "R": [0, 1, 1, ...] },
    { "lane_id": 1, "R": [0, 2, 2, ...] },
    ...
    { "lane_id": 7, "R": [0, 8, 8, ...] }
  ]
}
```

---

## 🔮 未来扩展

这种架构极易扩展：

- **Multi-SM:** 4 个 ESP32 可以组成一个包含 32 Lanes 的完整 SM。
- **Pipelining:** 可以在 Core 0 和 Core 1 之间引入更深的 Queue 来模拟流水线延迟。
- **Memory Hierarchy:**可以在 Core 0 实现 L1 Cache 控制器，Core 1 只负责计算。
