"""
ESP32 Assembly Builder for Transformer Attention Kernels
Generates instruction sequences for matrix multiplication and softmax
"""

from pathlib import Path
import struct

class Instruction:
    """Micro-CUDA ISA v1.5 Instruction"""
    def __init__(self, opcode, dest=0, src1=0, src2=0):
        self.opcode = opcode
        self.dest = dest
        self.src1 = src1
        self.src2_imm = src2
    
    def to_hex(self):
        """Convert to 32-bit hex instruction"""
        packed = struct.pack('BBBB', self.opcode, self.dest, self.src1, self.src2_imm)
        return hex(struct.unpack('<I', packed)[0])
    
    def to_int(self):
        """Convert to 32-bit integer"""
        packed = struct.pack('BBBB', self.opcode, self.dest, self.src1, self.src2_imm)
        return struct.unpack('<I', packed)[0]

# Opcodes (from instructions_v15.h)
OP_NOP = 0x00
OP_EXIT = 0x01
OP_MOV = 0x10
OP_IADD = 0x11
OP_IMUL = 0x13
OP_FADD = 0x30
OP_FMUL = 0x32
OP_FFMA = 0x34
OP_LDG = 0x60
OP_STG = 0x61
OP_LDX = 0x64
OP_STX = 0x66
OP_S2R = 0xF0

# System registers
SR_LANEID = 2

def build_matmul_kernel():
    """
    Simplified test kernel: Just copy 8x8 floats from input to output
    This tests basic VRAM access before implementing full matmul
    
    Memory layout:
    - Input at VRAM 0x0000 (Q matrix)
    - Output at VRAM 0x4000 (result)
    
    Each lane (0-7) will:
    1. Read 8 floats from input (addresses 0-63)
    2. Write them to output
    """
    program = []
    
    # For each of 8 rows
    for i in range(8):
        # Each lane loads one element: input[i*8 + lane_id]
        # Get lane ID
        program.append(Instruction(OP_S2R, dest=0, src1=SR_LANEID))  # R0 = lane_id
        
        # Compute address: i*8 + lane_id (in words, *4 for bytes)
        offset = i * 8
        program.append(Instruction(OP_MOV, dest=1, src1=0, src2=offset))  # R1 = i*8
        program.append(Instruction(OP_IADD, dest=1, src1=1, src2=0))  # R1 = i*8 + lane_id
        program.append(Instruction(OP_MOV, dest=2, src1=0, src2=4))  # R2 = 4 (bytes per word)
        program.append(Instruction(OP_IMUL, dest=1, src1=1, src2=2))  # R1 = addr in bytes
        
        # Load from input (address in R1)
        program.append(Instruction(OP_LDX, dest=3, src1=1))  # R3 = input[addr]
        
        # Compute output address (same offset but at 0x4000 base)
        # Output starts at 0x4000 = 16384 bytes
        program.append(Instruction(OP_MOV, dest=4, src1=0, src2=64))  # R4 = 64 (0x4000/256 as immediate)
        program.append(Instruction(OP_MOV, dest=5, src1=0, src2=256))  # R5 = 256
        program.append(Instruction(OP_IMUL, dest=4, src1=4, src2=5))  # R4 = 0x4000
        program.append(Instruction(OP_IADD, dest=4, src1=4, src2=1))  # R4 = 0x4000 + offset
        
        # Store to output
        program.append(Instruction(OP_STX, dest=3, src1=4))  # output[addr] = R3
    
    program.append(Instruction(OP_EXIT))
    return program

def build_softmax_kernel():
    """
    Softmax kernel for one row: y = exp(x - max(x)) / sum(exp(x - max(x)))
    Processes 8 elements using SIMD
    
    Registers:
    R0: max value
    R1: sum
    R2: temp
    F0-F7: data (8 elements in SIMD)
    """
    program = []
    
    # Load 8 elements using lane ID
    program.append(Instruction(OP_S2R, dest=0, src1=SR_LANEID))  # R0 = lane_id
    program.append(Instruction(OP_LDX, dest=1, src1=0))  # F1 = data[lane_id]
    
    # TODO: Find max (need inter-lane communication - simplified)
    # For now, assume pre-computed max in R2
    
    # Subtract max: x = x - max
    program.append(Instruction(OP_MOV, dest=2, src1=0, src2=0))  # R2 = max (placeholder)
    # program.append(Instruction(OP_FSUB, dest=1, src1=1, src2=2))  # x -= max
    
    # TODO: Implement full softmax with EXP and sum reduction
    
    # Store result
    program.append(Instruction(OP_S2R, dest=0, src1=SR_LANEID))
    program.append(Instruction(OP_STX, dest=1, src1=0))
    
    program.append(Instruction(OP_EXIT))
    return program

def save_kernel(program, filename):
    """Save compiled kernel to binary file"""
    with open(filename, 'wb') as f:
        for inst in program:
            f.write(struct.pack('<I', inst.to_int()))
    print(f"✅ Kernel saved: {filename} ({len(program)} instructions)")

if __name__ == "__main__":
    print("🔧 Building Transformer Attention Kernels...")
    
    # Build kernels
    matmul = build_matmul_kernel()
    print(f"   MatMul kernel: {len(matmul)} instructions")
    
    softmax = build_softmax_kernel()  
    print(f"   Softmax kernel: {len(softmax)} instructions")
    
    # Save binaries
    save_kernel(matmul, "kernel_matmul.bin")
    save_kernel(softmax, "kernel_softmax.bin")
    
    print("✨ Done!")
