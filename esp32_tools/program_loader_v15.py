"""
Micro-CUDA ISA v1.5 Program Loader
支持 True SIMT 和 Lane-Awareness
"""

from typing import List


class InstructionV15:
    """
    Micro-CUDA ISA v1.5 Instruction Encoder
    
    Encoding: [31:24 OPCODE] [23:16 DEST] [15:8 SRC1] [7:0 SRC2/IMM]
    """
    
    # ===== Group 1: Control =====
    OP_NOP = 0x00
    OP_EXIT = 0x01
    OP_BRA = 0x02
    OP_BRZ = 0x03
    OP_BAR_SYNC = 0x05
    
    # ===== Group 2: Integer ALU =====
    OP_MOV = 0x10
    OP_IADD = 0x11
    OP_ISUB = 0x12
    OP_IMUL = 0x13
    OP_AND = 0x17
    OP_OR = 0x18
    OP_XOR = 0x19
    OP_ISETP_EQ = 0x1A
    OP_ISETP_GT = 0x1C
    OP_SHL = 0x1D
    OP_SHR = 0x1E
    
    # ===== Group 3: Float & AI =====
    OP_FADD = 0x30
    OP_FSUB = 0x31
    OP_FMUL = 0x32
    OP_FDIV = 0x33
    OP_FFMA = 0x34
    OP_HMMA_I8 = 0x40
    OP_SFU_RCP = 0x50
    OP_SFU_SQRT = 0x51
    OP_SFU_EXP = 0x52
    OP_SFU_GELU = 0x53
    OP_SFU_RELU = 0x54
    
    # ===== Group 4: Memory (SIMT) =====
    OP_LDG = 0x60   # Uniform load (broadcast)
    OP_STG = 0x61   # Uniform store
    OP_LDS = 0x62   # Shared load
    OP_STS = 0x63   # Shared store
    OP_LDX = 0x64   # Indexed SIMT load
    OP_LDL = 0x65   # Lane-based load (NEW)
    OP_STX = 0x66   # Indexed SIMT store
    OP_STL = 0x67   # Lane-based store (NEW)
    OP_ATOM_ADD = 0x70
    
    # ===== Group 5: System =====
    OP_S2R = 0xF0   # System to Register
    OP_R2S = 0xF1   # Register to System
    OP_TRACE = 0xF2
    
    # ===== System Register Indices =====
    SR_TID = 0
    SR_CTAID = 1
    SR_LANEID = 2  # NEW in v1.5
    SR_WARPSIZE = 3
    SR_WARP_ID = 8
    SR_SM_ID = 9
    
    def __init__(self, opcode: int, dest: int = 0, src1: int = 0, src2: int = 0):
        self.opcode = opcode & 0xFF
        self.dest = dest & 0xFF
        self.src1 = src1 & 0xFF
        self.src2 = src2 & 0xFF
    
    def encode(self) -> int:
        """编码为 32-bit word"""
        return (self.opcode << 24) | (self.dest << 16) | (self.src1 << 8) | self.src2
    
    def to_hex(self) -> str:
        """转换为十六进制字符串"""
        return f"0x{self.encode():08X}"
    
    # ===== Control Instructions =====
    @classmethod
    def nop(cls):
        return cls(cls.OP_NOP)
    
    @classmethod
    def exit_inst(cls):
        return cls(cls.OP_EXIT)
    
    @classmethod
    def bar_sync(cls, barrier_id: int = 0):
        return cls(cls.OP_BAR_SYNC, 0, 0, barrier_id)

    @classmethod
    def bra(cls, offset: int):
        return cls(cls.OP_BRA, offset, 0, 0) # Back to Dest (Absolute)
    
    @classmethod
    def brz(cls, offset: int):
        return cls(cls.OP_BRZ, offset, 0, 0)
    
    # ===== Integer ALU =====
    @classmethod
    def mov(cls, dest: int, imm: int):
        return cls(cls.OP_MOV, dest, 0, imm)
    
    @classmethod
    def iadd(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_IADD, dest, src1, src2)
    
    @classmethod
    def isub(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_ISUB, dest, src1, src2)
    
    @classmethod
    def imul(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_IMUL, dest, src1, src2)
    
    @classmethod
    def and_op(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_AND, dest, src1, src2)

    @classmethod
    def isetp_eq(cls, pred_dest: int, src1: int, src2: int):
        return cls(cls.OP_ISETP_EQ, pred_dest, src1, src2)

    @classmethod
    def isetp_gt(cls, pred_dest: int, src1: int, src2: int):
        return cls(cls.OP_ISETP_GT, pred_dest, src1, src2)
    
    # ===== Float ALU =====
    @classmethod
    def fadd(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_FADD, dest, src1, src2)
    
    @classmethod
    def fmul(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_FMUL, dest, src1, src2)
    
    @classmethod
    def ffma(cls, dest: int, src1: int, src2: int):
        """FMA: Fd = Fa * Fb + Fd"""
        return cls(cls.OP_FFMA, dest, src1, src2)
    
    # ===== Special Functions =====
    @classmethod
    def sfu_gelu(cls, dest: int, src: int):
        return cls(cls.OP_SFU_GELU, dest, src, 0)
    
    @classmethod
    def sfu_relu(cls, dest: int, src: int):
        return cls(cls.OP_SFU_RELU, dest, src, 0)
        
    @classmethod
    def or_op(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_OR, dest, src1, src2)

    @classmethod
    def shl(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_SHL, dest, src1, src2)

    @classmethod
    def shr(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_SHR, dest, src1, src2)

    @classmethod
    def xor_op(cls, dest: int, src1: int, src2: int):
        return cls(cls.OP_XOR, dest, src1, src2)
    
    # ===== SIMT Memory Operations (NEW in v1.5) =====
    @classmethod
    def ldg(cls, dest: int, addr_reg: int):
        """Uniform Load: All lanes read from same address"""
        return cls(cls.OP_LDG, dest, addr_reg, 0)
    
    @classmethod
    def stg(cls, addr_reg: int, src: int):
        """Uniform Store: All lanes write to same address"""
        return cls(cls.OP_STG, src, addr_reg, 0)
    
    @classmethod
    def ldl(cls, dest: int, base_addr_reg: int):
        """
        Lane-Based Load (NEW in v1.5)
        Each lane reads: [base + SR_LANEID * 4]
        """
        return cls(cls.OP_LDL, dest, base_addr_reg, 0)
    
    @classmethod
    def stl(cls, base_addr_reg: int, src: int):
        """
        Lane-Based Store (NEW in v1.5)
        Each lane writes: [base + SR_LANEID * 4]
        """
        return cls(cls.OP_STL, src, base_addr_reg, 0)
    
    @classmethod
    def ldx(cls, dest: int, base_reg: int, offset_reg: int):
        """
        Indexed SIMT Load (NEW in v1.5)
        Each lane reads: [Rbase + Roffset]
        """
        return cls(cls.OP_LDX, dest, base_reg, offset_reg)
    
    @classmethod
    def stx(cls, base_reg: int, offset_reg: int, src: int):
        """Indexed SIMT Store"""
        return cls(cls.OP_STX, src, base_reg, offset_reg)
    
    # ===== System Register Operations (NEW in v1.5) =====
    @classmethod
    def s2r(cls, dest: int, sr_index: int):
        """
        System to Register
        Read system register (e.g., SR_LANEID) into general register
        """
        return cls(cls.OP_S2R, dest, sr_index, 0)
    
    @classmethod
    def trace(cls, marker_id: int):
        """Emit trace marker"""
        return cls(cls.OP_TRACE, 0, 0, marker_id)


class ProgramLoaderV15:
    """Program loader for ISA v1.5"""
    
    @staticmethod
    def create_parallel_attention_program() -> List[InstructionV15]:
        """
        Create Parallel Attention Loading Program
        
        Demonstrates SIMT: Each lane loads different Q/K/V values
        
        Memory Layout (VRAM):
        0x1000: Q[0], Q[1], Q[2], ... Q[7]
        0x2000: K[0], K[1], K[2], ... K[7]
        0x3000: V[0], V[1], V[2], ... V[7]
        0x4000: Result[0], Result[1], ... Result[7]
        """
        program = []
        
        # === Step 1: Get Lane ID ===
        program.append(InstructionV15.s2r(31, InstructionV15.SR_LANEID))  # R31 = My Lane ID
        program.append(InstructionV15.trace(0x01))  # Trace marker
        
        # === Step 2: Set Base Addresses ===
        program.append(InstructionV15.mov(0, 0x10))  # R0 = Base of Q (0x1000 >> 8)
        program.append(InstructionV15.mov(1, 0x20))  # R1 = Base of K
        program.append(InstructionV15.mov(2, 0x30))  # R2 = Base of V
        
        # === Step 3: SIMT Loading (Each lane gets different data) ===
        program.append(InstructionV15.ldl(10, 0))  # R10 = Q[lane]
        program.append(InstructionV15.ldl(11, 1))  # R11 = K[lane]
        program.append(InstructionV15.ldl(12, 2))  # R12 = V[lane]
        program.append(InstructionV15.trace(0x02))
        
        # === Step 4: Parallel Computation ===
        program.append(InstructionV15.imul(20, 10, 11))  # R20 = Q * K (Attention Score)
        program.append(InstructionV15.iadd(21, 20, 12))  # R21 = Score + V
        program.append(InstructionV15.trace(0x03))
        
        # === Step 5: Write Back (SIMT Store) ===
        program.append(InstructionV15.mov(3, 0x40))  # R3 = Result base
        program.append(InstructionV15.stl(3, 21))    # Store Result[lane]
        
        # === Step 6: Synchronize and Exit ===
        program.append(InstructionV15.bar_sync(0))
        program.append(InstructionV15.exit_inst())
        
        return program
    
    @staticmethod
    def create_simple_simt_test() -> List[InstructionV15]:
        """Simple SIMT test: Each lane computes lane_id * 10"""
        program = []
        
        # Get lane ID
        program.append(InstructionV15.s2r(0, InstructionV15.SR_LANEID))  # R0 = lane_id
        
        # Compute: R1 = lane_id * 10
        program.append(InstructionV15.mov(1, 10))
        program.append(InstructionV15.imul(2, 0, 1))  # R2 = lane_id * 10
        
        program.append(InstructionV15.exit_inst())
        
        return program
  
