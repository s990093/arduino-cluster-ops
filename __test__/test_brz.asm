
; Test BRZ polarity with Memory Store
; R0 = 10
; R10 = 0 (Addr A)
; R11 = 4 (Addr B)

MOV R0, 10
MOV R10, 0
MOV R11, 4

; Case A: P0 = 1 (True)
ISETP.EQ P0, R0, R0  ; P0 = 1
BRZ P0, label_a_jump
MOV R1, 0            ; Fallthrough (False path)
STG R10, R1          ; Store [0] = 0
BRA label_b_start
label_a_jump:
MOV R1, 1            ; Jump (True path)
STG R10, R1          ; Store [0] = 1

label_b_start:
; Case B: P0 = 0 (False)
MOV R2, 0
MOV R3, 1
ISETP.EQ P0, R2, R3  ; P0 = 0
BRZ P0, label_b_jump
MOV R4, 0            ; Fallthrough (False path)
STG R11, R4          ; Store [4] = 0
EXIT
label_b_jump:
MOV R4, 1            ; Jump (True path)
STG R11, R4          ; Store [4] = 1
EXIT
