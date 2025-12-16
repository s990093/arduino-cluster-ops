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

; Source File: my_kernel.ll
; Kernel Functions: _Z14simple_processPiS_, _Z19accumulator_patternPiS_, _Z11multi_paramPiS_S_S_, _Z12neighborhoodPiS_i, _Z16your_kernel_herePiS_
; Total Instructions: 97
; Registers Used: 19
;
; ====================================================================

; ===== CODE SECTION =====

MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
S2R R2, SR_2              ; laneId() -> R2
MOV R4, 4                 ; R4 = 4
IMUL R5, R2, R4           ; R5 = index * 4
IADD R3, R0, R5           ; R3 = base + offset
MOV R7, 0                 ; Zero offset
LDX R6, R3, R7            ; R6 = Mem[R3]
MOV R4, 4                 ; R4 = 4
IMUL R5, R2, R4           ; R5 = index * 4
IADD R8, R1, R5           ; R8 = base + offset
MOV R10, 0                ; Zero offset
STX R8, R10, R9           ; Mem[R8] = R9
EXIT                      ; Return from kernel
MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
S2R R2, SR_2              ; laneId() -> R2
MOV R4, 4                 ; R4 = 4
IMUL R5, R2, R4           ; R5 = index * 4
IADD R3, R0, R5           ; R3 = base + offset
MOV R7, 0                 ; Zero offset
LDX R6, R3, R7            ; R6 = Mem[R3]
MOV R7, 0                 ; Zero offset
LDX R8, R9, R7            ; R8 = Mem[R9]
IADD R11, R8, R6          ; R11 = R8 + R6
MOV R7, 0                 ; Zero offset
LDX R12, R13, R7          ; R12 = Mem[R13]
IADD R14, R11, R12        ; R14 = R11 + R12
MOV R4, 4                 ; R4 = 4
IMUL R5, R2, R4           ; R5 = index * 4
IADD R15, R1, R5          ; R15 = base + offset
MOV R10, 0                ; Zero offset
STX R15, R10, R14         ; Mem[R15] = R14
EXIT                      ; Return from kernel
MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
MOV R16, 64               ; param 2 @ VRAM[0x40]
MOV R2, 96                ; param 3 @ VRAM[0x60]
S2R R6, SR_2              ; laneId() -> R6
MOV R4, 4                 ; R4 = 4
IMUL R5, R6, R4           ; R5 = index * 4
IADD R9, R0, R5           ; R9 = base + offset
MOV R7, 0                 ; Zero offset
LDX R8, R9, R7            ; R8 = Mem[R9]
MOV R4, 4                 ; R4 = 4
IMUL R5, R6, R4           ; R5 = index * 4
IADD R11, R1, R5          ; R11 = base + offset
MOV R7, 0                 ; Zero offset
LDX R13, R11, R7          ; R13 = Mem[R11]
MOV R4, 4                 ; R4 = 4
IMUL R5, R6, R4           ; R5 = index * 4
IADD R12, R16, R5         ; R12 = base + offset
MOV R7, 0                 ; Zero offset
LDX R14, R12, R7          ; R14 = Mem[R12]
IADD R15, R13, R8         ; R15 = R13 + R8
IMUL R17, R15, R14        ; R17 = R15 * R14
MOV R4, 4                 ; R4 = 4
IMUL R5, R6, R4           ; R5 = index * 4
IADD R18, R2, R5          ; R18 = base + offset
MOV R10, 0                ; Zero offset
STX R18, R10, R17         ; Mem[R18] = R17
EXIT                      ; Return from kernel
MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
S2R R3, SR_2              ; laneId() -> R3
IMUL R6, R3, R16          ; R6 = R3 * R16
MOV R4, 4                 ; R4 = 4
IMUL R5, R6, R4           ; R5 = index * 4
IADD R9, R0, R5           ; R9 = base + offset
MOV R7, 0                 ; Zero offset
LDX R11, R8, R7           ; R11 = Mem[R8]
MOV R7, 0                 ; Zero offset
LDX R13, R9, R7           ; R13 = Mem[R9]
IADD R12, R13, R11        ; R12 = R13 + R11
MOV R7, 0                 ; Zero offset
LDX R15, R14, R7          ; R15 = Mem[R14]
IADD R17, R12, R15        ; R17 = R12 + R15
MOV R4, 4                 ; R4 = 4
IMUL R5, R3, R4           ; R5 = index * 4
IADD R18, R1, R5          ; R18 = base + offset
MOV R10, 0                ; Zero offset
STX R18, R10, R17         ; Mem[R18] = R17
EXIT                      ; Return from kernel
MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 32                ; param 1 @ VRAM[0x20]
S2R R2, SR_2              ; laneId() -> R2
MOV R4, 4                 ; R4 = 4
IMUL R5, R2, R4           ; R5 = index * 4
IADD R3, R0, R5           ; R3 = base + offset
MOV R7, 0                 ; Zero offset
LDX R6, R3, R7            ; R6 = Mem[R3]
MOV R4, 4                 ; R4 = 4
IMUL R5, R2, R4           ; R5 = index * 4
IADD R9, R1, R5           ; R9 = base + offset
MOV R10, 0                ; Zero offset
STX R9, R10, R6           ; Mem[R9] = R6
EXIT                      ; Return from kernel

; ===== END OF KERNEL =====
