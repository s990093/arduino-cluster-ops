# Micro-CUDA ISA Specification v2.0

**Status:** Release Candidate | **Architecture:** Micro-Cluster (MC) | **Target:** Deep Learning / Transformer

## Execution Model & Architecture Definition

### Hardware Layer Mapping

The system implements a three-layer distributed architecture:

1. **Layer 1 (Grid Master): AMB82-Mini**
   - _Role:_ Grid Tiling and DMA Data Injection
   - _Task:_ Responsible for overall grid-level data distribution and kernel launch management
2. **Layer 2 (SM / Scheduler): ESP32-S3**
   - _Role:_ Warp Scheduler / Instruction Dispatcher
   - _Task:_ Handles warp scheduling and instruction broadcast
3. **Layer 3 (Lane / Core): RP2040**
   - _Role:_ Arithmetic Execution Lane
   - _Task:_ Receives instructions via PIO and executes arithmetic operations

### Data Types

To enable efficient AI inference on the FPU-less RP2040, v2.0 introduces **Packed BF16**.

| Type            | Bit Width | Format           | Description                                    |
| :-------------- | :-------- | :--------------- | :--------------------------------------------- |
| **INT32**       | 32-bit    | 2's Complement   | Address calculation, loop counters, indexing   |
| **FP32**        | 32-bit    | IEEE 754         | [v1.5] High-precision weights, accumulators    |
| **BF16**        | 16-bit    | 1-8-7 (BFloat16) | **[v2.0]** Same exponent bits as FP32          |
| **Packed BF16** | 32-bit    | 2x BF16          | **[v2.0]** High: Element 1, Low: Element 0     |
| **INT8**        | 8-bit     | Signed           | [v1.5] Used for `HMMA.I8` quantized operations |

### Register File

Each Lane (RP2040) independently maintains its own register file:

- **R0 - R31 (General Purpose)**: 32x 32-bit registers. Can store INT32, FP32, or Packed BF16.
- **P0 - P7 (Predicate)**: 8x 1-bit condition flags.
- **SR (System Registers)**: Read-only status (e.g., `SR_LANEID`, `SR_LANEMASK`).

## Instruction Set Reference

All Micro-CUDA instructions use a fixed 32-bit encoding: `[OPCODE 8b | DEST 8b | SRC1 8b | SRC2/IMM 8b]`

### Group 1: System Control [0x00 - 0x0F]

| Op     | Mnemonic     | Operands | Description                        |
| :----- | :----------- | :------- | :--------------------------------- |
| `0x00` | **NOP**      | -        | No operation                       |
| `0x01` | **EXIT**     | -        | Terminate kernel execution         |
| `0x02` | **BRA**      | Imm      | Unconditional branch (`PC += Imm`) |
| `0x03` | **BR.Z**     | Imm, Pn  | Branch if Predicate `Pn` is 0      |
| `0x05` | **BAR.SYNC** | Id       | Warp Barrier                       |
| `0x07` | **YIELD**    | -        | Yield time slice                   |

### Group 2: Integer Arithmetic [0x10 - 0x1F]

| Op     | Mnemonic     | Operands    | Description                     |
| :----- | :----------- | :---------- | :------------------------------ |
| `0x10` | **MOV**      | Rd, Imm     | Load immediate value            |
| `0x11` | **IADD**     | Rd, Ra, Rb  | Integer addition                |
| `0x12` | **ISUB**     | Rd, Ra, Rb  | Integer subtraction             |
| `0x13` | **IMUL**     | Rd, Ra, Rb  | Integer multiplication (32-bit) |
| `0x17` | **AND**      | Rd, Ra, Rb  | Bitwise AND                     |
| `0x18` | **OR**       | Rd, Ra, Rb  | Bitwise OR                      |
| `0x1A` | **ISETP.EQ** | Pn, Ra, Rb  | If Ra == Rb, set Pn = 1         |
| `0x1C` | **ISETP.GT** | Pn, Ra, Rb  | If Ra > Rb, set Pn = 1          |
| `0x1D` | **SHL**      | Rd, Ra, Imm | Logical shift left              |

### Group 3: Deep Learning & Data Conversion (v2.0) [0x20 - 0x2F]

| Op     | Mnemonic     | Operands   | Description                  |
| :----- | :----------- | :--------- | :--------------------------- |
| `0x20` | **CVT.BF16** | Rd, Ra     | FP32(Ra) -> BF16(Rd.Low)     |
| `0x21` | **CVT.F32**  | Rd, Ra     | BF16(Ra.Low) -> FP32(Rd)     |
| `0x22` | **PACK2**    | Rd, Ra, Rb | Pack Ra.Low/Rb.Low -> Rd     |
| `0x25` | **BFADD2**   | Rd, Ra, Rb | Packed BF16 Add              |
| `0x26` | **BFMUL2**   | Rd, Ra, Rb | Packed BF16 Mul              |
| `0x27` | **BFMA2**    | Rd, Ra, Rb | Packed FMA (`Rd += Ra * Rb`) |
| `0x28` | **BFRELU2**  | Rd, Ra     | Packed ReLU (`max(0, x)`)    |

### Group 4: Float & SFU [0x30 - 0x5F]

| Op     | Mnemonic      | Operands   | Description             |
| :----- | :------------ | :--------- | :---------------------- |
| `0x30` | **FADD**      | Fd, Fa, Fb | FP32 Add                |
| `0x32` | **FMUL**      | Fd, Fa, Fb | FP32 Mul                |
| `0x34` | **FFMA**      | Fd, Fa, Fb | FP32 Fused Multiply-Add |
| `0x40` | **HMMA.I8**   | Rd, Ra, Rb | 4-way INT8 Dot Product  |
| `0x50` | **SFU.RCP**   | Rd, Ra     | 1/x Reciprocal          |
| `0x51` | **SFU.EXP2**  | Rd, Ra     | 2^x (for Softmax)       |
| `0x52` | **SFU.LOG2**  | Rd, Ra     | log2(x)                 |
| `0x53` | **SFU.RSQRT** | Rd, Ra     | 1/sqrt(x)               |
| `0x54` | **SFU.SIN**   | Rd, Ra     | sin(pi \* x)            |
| `0x55` | **SFU.COS**   | Rd, Ra     | cos(pi \* x)            |
| `0x56` | **SFU.GELU**  | Rd, Ra     | GELU Activation         |
| `0x57` | **SFU.TANH**  | Rd, Ra     | Tanh Activation         |

### Group 5: Memory Operations [0x60 - 0x7F]

| Op     | Mnemonic     | Operands    | Description                                |
| :----- | :----------- | :---------- | :----------------------------------------- |
| `0x60` | **LDG**      | Rd, [Ra]    | **Broadcast Load:** Uniform access         |
| `0x64` | **LDL**      | Rd, [Ra]    | **Lane Load:** `Addr = Ra + (LANEID * 4)`  |
| `0x63` | **LDX**      | Rd, [Ra+Rb] | **Gather Load:** Indirect `Addr = Ra + Rb` |
| `0x65` | **STX**      | [Ra+Rb], Rd | **Scatter Store**                          |
| `0x67` | **STL**      | [Ra], Rd    | **Lane Store:** `Addr = Ra + (LANEID * 4)` |
| `0x70` | **ATOM.ADD** | [Ra], Rb    | Atomic Add                                 |

### Group 6: System Instructions [0xF0 - 0xFF]

| Op     | Mnemonic  | Operands | Description                            |
| :----- | :-------- | :------- | :------------------------------------- |
| `0xF0` | **S2R**   | Rd, SRn  | System to Register (e.g. read Lane ID) |
| `0xF1` | **R2S**   | SRn, Rd  | Register to System                     |
| `0xF2` | **TRACE** | Imm      | Send Debug Trace Event                 |

## ISA v2.0 Extensions

### SIMD2 Packed Execution Model

To maximize 32-bit register utilization, each general-purpose register is treated as a vector containing two 16-bit BF16 values.

- **R[n].L**: Low 16-bit (Element 0)
- **R[n].H**: High 16-bit (Element 1)
  This allows a single instruction to process two floating-point numbers simultaneously, effectively doubling the throughput.

### Special Function Unit (SFU)

Implements high-precision LUT and Taylor series hybrid algorithms for transcendental functions (SIN, COS, EXP2, RSQRT) stored in Flash (XIP) to achieve high performance.

## Application Binary Interface (ABI)

To ensure interoperability between the Micro-CUDA assembler, linker, and runtime, a strict ABI is defined.

### Register Usage Convention

| Register      | Role                               | Preservation |
| :------------ | :--------------------------------- | :----------- |
| **r0 - r3**   | Function Arguments / Return Values | Caller Saved |
| **r4 - r11**  | Temporary Variables                | Caller Saved |
| **r12 - r27** | Saved Variables                    | Callee Saved |
| **r28 (SP)**  | Stack Pointer                      | Callee Saved |
| **r29 (LR)**  | Link Register                      | Callee Saved |
| **r30**       | Reserved (Frame Pointer)           | -            |
| **r31**       | Program Counter (PC)               | -            |

### Stack Frame Layout

The stack grows downwards in the private SRAM memory space of each core.

- Incoming Arguments (High Addr)
- Return Address (LR)
- Saved Regs (r12-r27)
- Local Variables
- Current SP (Low Addr)

### Memory Organization (Address Map)

| Address Range                 | Region        | Description                   |
| :---------------------------- | :------------ | :---------------------------- |
| `0x0000_0000` - `0x0003_FFFF` | I-Cache       | Instruction Cache (Flash)     |
| `0x1000_0000` - `0x1000_FFFF` | VRAM (Local)  | Private SRAM for current SMSP |
| `0x2000_0000` - `0x2FFF_FFFF` | VRAM (Global) | Global DDR / PSRAM Window     |
| `0x4000_0000` - `0x4000_00FF` | SFR (DMA)     | DMA Controller Registers      |
| `0x5000_0000` - `0x5000_00FF` | SFR (Tensor)  | Tensor Engine Command Queue   |
