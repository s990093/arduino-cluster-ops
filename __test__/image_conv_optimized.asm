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

; Source File: image_conv_kernel.ll
; Kernel Functions: _Z15image_conv_testPiS_S_i
; Total Instructions: 20
; Registers Used: 8
;
; ====================================================================

; ===== CODE SECTION =====

MOV R0, 0                 ; param 0 @ VRAM[0x00]
MOV R1, 64                ; param 1 @ VRAM[0x40]
MOV R2, 128               ; param 2 @ VRAM[0x80]
S2R R3, SR_2              ; laneId() -> R3
MOV R5, 16                ; R5 = 16 * 1
IADD R4, R1, R5           ; R4 = base + offset
MOV R6, 0                 ; Zero offset
LDX R1, R4, R6            ; R1 = Mem[R4]
MOV R7, 4                 ; R7 = 4
IMUL R5, R3, R7           ; R5 = index * 4
IADD R4, R0, R5           ; R4 = base + offset
MOV R6, 0                 ; Zero offset
LDX R0, R4, R6            ; R0 = Mem[R4]
IMUL R4, R0, R1           ; R4 = R0 * R1
MOV R7, 4                 ; R7 = 4
IMUL R5, R3, R7           ; R5 = index * 4
IADD R0, R2, R5           ; R0 = base + offset
MOV R1, 0                 ; Zero offset
STX R0, R1, R4            ; Mem[R0] = R4
EXIT                      ; Return from kernel

; ===== END OF KERNEL =====
