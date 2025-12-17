# Micro-CUDA ISA Specification v2.0

**Target**: Deep Learning / Transformer  
**Architecture**: Micro-Cluster (MC)

## Execution Model

The system implements a three-layer distributed architecture:

1.  **Layer 1 (Grid Master): AMB82-Mini** - Grid Tiling and DMA Data Injection.
2.  **Layer 2 (SM / Scheduler): ESP32-S3** - Warp Scheduler & Instruction Dispatch.
3.  **Layer 3 (Lane / Core): RP2040** - Arithmetic Execution Lane.

### Lane Execution Model

The engine emulates **8 lanes** (Thread IDs 0-7). When a warp is scheduled, all 8 lanes execute the instruction in a loop (unrolled for performance).

## Data Types

To enable efficient AI inference on the FPU-less RP2040, v2.0 introduces **Packed BF16**.

| Type            | Bit Width | Format   | Description                          |
| :-------------- | :-------- | :------- | :----------------------------------- |
| **INT32**       | 32-bit    | 2's Comp | Address calculation, indexing        |
| **FP32**        | 32-bit    | IEEE 754 | [v1.5] High-precision accumulators   |
| **BF16**        | 16-bit    | 1-8-7    | **[v2.0]** BFloat16 for AI weights   |
| **Packed BF16** | 32-bit    | 2× BF16  | **[v2.0]** High: Elem 1, Low: Elem 0 |
| **INT8**        | 8-bit     | Signed   | [v1.5] Quantized operations          |

## Memory Model

### VRAM Organization

The ESP32-S3 allocates a 100KB static array as Virtual VRAM.

| Address Range       | Description                 |
| :------------------ | :-------------------------- |
| **0x0000 - 0x0FFF** | Program Text (Instructions) |
| **0x1000 - 0x3FFF** | Global Data                 |
| **0x4000 - 0xDFFF** | Heap / Stack areas          |

### Memory Access Patterns

- **LDG/STG (Broadcast)**: Scalar access. Same address for all lanes.
- **LDL/STL (Strided)**: Vector access. `Address = Base + (LaneID * 4)`.
- **LDX/STX (Gather/Scatter)**: Indirect access. `Address = Base + Offset[LaneID]`.
- **LDS/STS (Shared)**: Fast scratchpad access.

## Register File

Each Lane (RP2040) independently maintains:

- **R0 - R31**: 32× 32-bit General Purpose Registers.
- **P0 - P7**: 8× 1-bit Predicate Flags.
- **SR**: System Registers.
  - `SR.LANEMASK`: Bitmask of active lanes (usually `0xFF`).
  - `SR.LANEID`: Current lane ID (0-7).

## Instruction Encoding

All instructions use a fixed 32-bit encoding:

| [31:24]    | [23:16]  | [15:8]   | [7:0]          |
| :--------- | :------- | :------- | :------------- |
| **OPCODE** | **DEST** | **SRC1** | **SRC2 / IMM** |

## Instruction Set Reference

### Group 1: System Control [0x00 - 0x0F]

| Op     | Mnemonic     | Operands | Description                        |
| :----- | :----------- | :------- | :--------------------------------- |
| `0x00` | **NOP**      | -        | No Operation                       |
| `0x01` | **EXIT**     | -        | Terminate Kernel                   |
| `0x02` | **BRA**      | Imm      | Unconditional Branch (`PC += Imm`) |
| `0x03` | **BR.Z**     | Imm, Pn  | Branch if Predicate `Pn` is 0      |
| `0x05` | **BAR.SYNC** | Id       | Warp Barrier                       |

### Group 2: Integer Arithmetic [0x10 - 0x1F]

| Op     | Mnemonic     | Operands   | Description          |
| :----- | :----------- | :--------- | :------------------- |
| `0x10` | **MOV**      | Rd, Imm    | Load Immediate       |
| `0x11` | **IADD**     | Rd, Ra, Rb | Integer Add          |
| `0x12` | **ISUB**     | Rd, Ra, Rb | Integer Sub          |
| `0x13` | **IMUL**     | Rd, Ra, Rb | Integer Mul (32-bit) |
| `0x1A` | **ISETP.EQ** | Pn, Ra, Rb | Set Pn if Ra == Rb   |
| `0x1C` | **ISETP.GT** | Pn, Ra, Rb | Set Pn if Ra > Rb    |

### Group 3: AI & Data Conversion (v2.0) [0x20 - 0x2F]

| Op     | Mnemonic     | Operands   | Description                  |
| :----- | :----------- | :--------- | :--------------------------- |
| `0x20` | **CVT.BF16** | Rd, Ra     | FP32(Ra) $\to$ BF16(Rd.Low)  |
| `0x21` | **CVT.F32**  | Rd, Ra     | BF16(Ra.Low) $\to$ FP32(Rd)  |
| `0x22` | **PACK2**    | Rd, Ra, Rb | Pack Ra.Low/Rb.Low $\to$ Rd  |
| `0x25` | **BFADD2**   | Rd, Ra, Rb | Packed BF16 Add              |
| `0x26` | **BFMUL2**   | Rd, Ra, Rb | Packed BF16 Mul              |
| `0x27` | **BFMA2**    | Rd, Ra, Rb | Packed FMA (`Rd += Ra * Rb`) |
| `0x28` | **BFRELU2**  | Rd, Ra     | Packed ReLU (`max(0, x)`)    |

### Group 4: Float & SFU [0x30 - 0x5F]

| Op     | Mnemonic      | Operands   | Description                      |
| :----- | :------------ | :--------- | :------------------------------- |
| `0x30` | **FADD**      | Fd, Fa, Fb | FP32 Add                         |
| `0x32` | **FMUL**      | Fd, Fa, Fb | FP32 Mul                         |
| `0x50` | **SFU.RCP**   | Rd, Ra     | $1/x$ Reciprocal                 |
| `0x51` | **SFU.EXP2**  | Rd, Ra     | $2^x$ (for Softmax)              |
| `0x53` | **SFU.RSQRT** | Rd, Ra     | $1/\sqrt{x}$ (Attention Scaling) |
| `0x54` | **SFU.SIN**   | Rd, Ra     | $\sin(\pi x)$ (RoPE)             |
| `0x55` | **SFU.COS**   | Rd, Ra     | $\cos(\pi x)$ (RoPE)             |
| `0x56` | **SFU.GELU**  | Rd, Ra     | GELU Activation                  |

### Group 5: Memory Operations [0x60 - 0x7F]

| Op     | Mnemonic | Operands    | Description                             |
| :----- | :------- | :---------- | :-------------------------------------- |
| `0x60` | **LDG**  | Rd, [Ra]    | **Broadcast Load** (Uniform)            |
| `0x64` | **LDL**  | Rd, [Ra]    | **Lane Load** (`Addr = Ra + LaneID*4`)  |
| `0x63` | **LDX**  | Rd, [Ra+Rb] | **Gather Load** (Indirect)              |
| `0x65` | **STX**  | [Ra+Rb], Rd | **Scatter Store** (v2.0)                |
| `0x67` | **STL**  | [Ra], Rd    | **Lane Store** (`Addr = Ra + LaneID*4`) |

## Implementation Details

### Packed BF16 Emulation

Since the RP2040 lacks an FPU, BF16 operations are emulated in firmware. Packed instructions process the High (16-bit) and Low (16-bit) halves of a 32-bit register sequentially, effectively doubling throughput for vector operations.

### Special Function Unit (SFU)

Transcendental functions use lookup tables stored in Flash (XIP) to achieve high performance:

- **SIN/COS**: 2KB Lookup Table (1024 entries). Linear interpolation.
- **RSQRT**: Fast Inverse Square Root (Quake III algorithm adapted for BF16).

## Code Example: Softmax

```asm
; Kernel: Softmax (Exp(x) / Sum(Exp(x)))
; Input: R0 (Base Address, Packed BF16)

S2R     R31, SR_LANEID      ; Get Lane ID
LDL     R1, [R0]            ; Load Packed BF16 [x1, x2]

; Compute 2^(x * log2e)
MOV     R2, 0x3FB80000      ; log2(e)
BFMUL2  R1, R1, R2          ; Scale
SFU.EXP2 R3, R1             ; Exp2 approximation

; Accumulate (simplified)
CVT.F32 R4, R3              ; Unpack Low
SHL     R5, R3, 16
CVT.F32 R5, R5              ; Unpack High
FADD    R6, R4, R5          ; Pairwise Sum

; ... (Warp Reduction & Normalization omitted) ...
EXIT
```
