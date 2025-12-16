; ====================================================================
; Micro-CUDA Kernel - Compiled Assembly
; ====================================================================
;
; Target Configuration:
;   Device:        ESP32 CUDA VM
;   ISA Version:   v1.5
;   Architecture:  Dual-Core SIMT
;
; SIMD Configuration:
;   Lanes:         8
;   Warp Size:     8
;
; Memory Configuration:
;   VRAM Size:     40960 bytes (40 KB)
;   Program Size:  1024 instructions
;   Stack Size:    8192 bytes
;
; Register Configuration (per lane):
;   GP Registers:  R0-R31 (32 × 32-bit)
;   FP Registers:  F0-F31 (32 × 32-bit)
;   Predicates:    P0-P7 (8 × 1-bit)
;   System Regs:   SR_0 - SR_9
;
; Communication:
;   Serial Baud:   115200
;   CPU Freq:      240 MHz
;
; Performance:
;   Typical Speed: ~30,000 inst/sec
;
; ====================================================================

; Source File: conv1d.ll
; Kernel Functions: _Z6conv1dPiS_S_S_
; Total Instructions: 55
; Registers Used: 20
;
; ====================================================================

; ===== CODE SECTION =====

MOV R0, 0                 ; param 0 @ 0x1000 (Lo8)
MOV R1, 16
MOV R2, 8
SHL R1, R1, R2
OR R0, R0, R1
MOV R3, 0                 ; param 1 @ 0x2000 (Lo8)
MOV R4, 32
MOV R5, 8
SHL R4, R4, R5
OR R3, R3, R4
MOV R6, 0                 ; param 2 @ 0x3000 (Lo8)
MOV R7, 48
MOV R8, 8
SHL R7, R7, R8
OR R6, R6, R7
MOV R9, 0                 ; param 3 @ 0x4000 (Lo8)
MOV R10, 64
MOV R11, 8
SHL R10, R10, R11
OR R9, R9, R10
MOV R13, 0                ; Zero offset
LDX R12, R9, R13          ; R12 = Mem[R9]
IADD R9, R12, R14         ; R9 = R12 + R14
MOV R13, 0                ; Zero offset
LDX R12, R3, R13          ; R12 = Mem[R3]
MOV R15, 4                ; R15 = 4 * 1 (Lo8)
IADD R14, R3, R15         ; R14 = base + offset
MOV R13, 0                ; Zero offset
LDX R16, R14, R13         ; R16 = Mem[R14]
MOV R15, 8                ; R15 = 8 * 1 (Lo8)
IADD R14, R3, R15         ; R14 = base + offset
MOV R13, 0                ; Zero offset
LDX R3, R14, R13          ; R3 = Mem[R14]
MOV R17, 4                ; R17 = 4
IMUL R15, R9, R17         ; R15 = index * 4
IADD R14, R0, R15         ; R14 = base + offset
MOV R13, 0                ; Zero offset
LDX R0, R18, R13          ; R0 = Mem[R18]
MOV R13, 0                ; Zero offset
LDX R18, R14, R13         ; R18 = Mem[R14]
MOV R15, 4                ; R15 = 4 * 1 (Lo8)
IADD R19, R14, R15        ; R19 = base + offset
MOV R13, 0                ; Zero offset
LDX R14, R19, R13         ; R14 = Mem[R19]
IMUL R19, R0, R12         ; R19 = R0 * R12
IMUL R0, R18, R16         ; R0 = R18 * R16
IADD R12, R0, R19         ; R12 = R0 + R19
IMUL R0, R14, R3          ; R0 = R14 * R3
IADD R3, R12, R0          ; R3 = R12 + R0
MOV R17, 4                ; R17 = 4
IMUL R15, R9, R17         ; R15 = index * 4
IADD R0, R6, R15          ; R0 = base + offset
MOV R13, 0                ; Zero offset
STX R0, R13, R3           ; Mem[R0] = R3
EXIT                      ; Return from kernel

; ===== END OF KERNEL =====
