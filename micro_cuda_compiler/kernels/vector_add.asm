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

; Source File: vector_add.ll
; Kernel Functions: _Z9vectorAddPiS_S_, _Z15vectorAddManualPiS_S_
; Total Instructions: 27
; Registers Used: 14
;
; ====================================================================

; ===== CODE SECTION =====

MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
MOV R2, 64                ; param 2 @ VRAM[0x40]
S2R R3, SR_2              ; laneId() -> R3
MOV R5, 4                 ; R5 = 4
IMUL R6, R3, R5           ; R6 = index * 4
IADD R4, R0, R6           ; R4 = base + offset
MOV R8, 0                 ; Zero offset
LDX R7, R4, R8            ; R7 = Mem[R4]
MOV R5, 4                 ; R5 = 4
IMUL R6, R3, R5           ; R6 = index * 4
IADD R9, R1, R6           ; R9 = base + offset
MOV R8, 0                 ; Zero offset
LDX R10, R9, R8           ; R10 = Mem[R9]
IADD R11, R10, R7         ; R11 = R10 + R7
MOV R5, 4                 ; R5 = 4
IMUL R6, R3, R5           ; R6 = index * 4
IADD R12, R2, R6          ; R12 = base + offset
MOV R13, 0                ; Zero offset
STX R12, R13, R11         ; Mem[R12] = R11
EXIT                      ; Return from kernel
MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
MOV R2, 64                ; param 2 @ VRAM[0x40]
S2R R3, SR_2              ; laneId() -> R3
IADD R12, R11, R10        ; R12 = R11 + R10
EXIT                      ; Return from kernel

; ===== END OF KERNEL =====
