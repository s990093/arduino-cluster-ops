# Micro-CUDA ISA Reference

## Instruction Format

All instructions are 32-bit fixed width.

![Encoding](/images/enc_diag.png)

## Opcode Table

### Group 0: System

| Opcode | Mnemonic | Description                |
| :----- | :------- | :------------------------- |
| `0x00` | `NOP`    | No Operation               |
| `0x01` | `EXIT`   | Terminate Thread           |
| `0x02` | `BRA`    | Unconditional Branch       |
| `0x03` | `BR.Z`   | Branch if Zero (Predicate) |

### Group 1: Integer ALU

| Opcode | Mnemonic | Description        |
| :----- | :------- | :----------------- |
| `0x10` | `MOV`    | Move Immediate/Reg |
| `0x11` | `IADD`   | Integer Add        |
| `0x12` | `ISUB`   | Integer Subtract   |
| `0x13` | `IMUL`   | Integer Multiply   |

### Group 2: Tensor & Matrix

| Opcode | Mnemonic   | Description                        |
| :----- | :--------- | :--------------------------------- |
| `0x20` | `CVT.BF16` | Convert FP32 to BF16               |
| `0x22` | `BFMA2`    | Packed BFloat16 Fused Multiply-Add |

### Group 3: Memory

| Opcode | Mnemonic | Description              |
| :----- | :------- | :----------------------- |
| `0x50` | `LDG`    | Load Global (Coalesced)  |
| `0x51` | `STG`    | Store Global (Coalesced) |
