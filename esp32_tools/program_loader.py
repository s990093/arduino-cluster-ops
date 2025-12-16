"""
ESP32 CUDA 程序加載器
支持動態生成和加載指令到 ESP32
"""

from typing import List


class Instruction:
    """CUDA 指令編碼器"""
    
    # Opcode 定義
    OP_NOP = 0x00
    OP_EXIT = 0x01
    OP_MOV = 0x10
    OP_IADD = 0x11
    OP_ISUB = 0x12
    OP_IMUL = 0x13
    OP_FADD = 0x30
    OP_FSUB = 0x31
    OP_FMUL = 0x32
    
    def __init__(self, opcode: int, dest: int = 0, src1: int = 0, src2: int = 0):
        """
        創建指令
        
        格式: [opcode(8)] [dest(8)] [src1(8)] [src2/imm(8)]
        """
        self.opcode = opcode & 0xFF
        self.dest = dest & 0xFF
        self.src1 = src1 & 0xFF
        self.src2 = src2 & 0xFF
    
    def encode(self) -> int:
        """編碼為 32-bit word"""
        word = (self.opcode << 24) | (self.dest << 16) | (self.src1 << 8) | self.src2
        return word
    
    def to_hex(self) -> str:
        """轉換為十六進制字符串"""
        return f"0x{self.encode():08X}"
    
    @classmethod
    def mov(cls, dest: int, imm: int):
        """MOV Rd, Imm"""
        return cls(cls.OP_MOV, dest, 0, imm)
    
    @classmethod
    def iadd(cls, dest: int, src1: int, src2: int):
        """IADD Rd, Ra, Rb"""
        return cls(cls.OP_IADD, dest, src1, src2)
    
    @classmethod
    def isub(cls, dest: int, src1: int, src2: int):
        """ISUB Rd, Ra, Rb"""
        return cls(cls.OP_ISUB, dest, src1, src2)
    
    @classmethod
    def imul(cls, dest: int, src1: int, src2: int):
        """IMUL Rd, Ra, Rb"""
        return cls(cls.OP_IMUL, dest, src1, src2)
    
    @classmethod
    def exit_inst(cls):
        """EXIT"""
        return cls(cls.OP_EXIT)


class ProgramLoader:
    """程序加載器"""
    
    @staticmethod
    def create_transformer_program() -> List[Instruction]:
        """
        創建 Transformer 計算程序
        
        注意：由於 SIMD 架構，MOV 指令會在所有 lane 執行
        所以我們只能使用統一的初始值，然後讓每個 lane 自己計算
        """
        program = []
        
        # 1. 初始化 Q, K, V (使用簡單值)
        # 所有 lane 執行相同的指令，但可以通過後續運算產生不同結果
        program.append(Instruction.mov(0, 2))   # R0 = 2 (Q base)
        program.append(Instruction.mov(1, 3))   # R1 = 3 (K base)
        program.append(Instruction.mov(2, 4))   # R2 = 4 (V base)
        
        # 2. Attention Score = Q * K
        program.append(Instruction.imul(1, 0, 1))  # R1 = R0 * R1 = 2*3 = 6
        
        # 3. Residual = Q + V (多個副本用於測試)
        program.append(Instruction.iadd(16, 0, 2))  # R16 = R0 + R2 = 2+4 = 6
        program.append(Instruction.iadd(17, 0, 2))  # R17 = 6
        program.append(Instruction.iadd(18, 0, 2))  # R18 = 6
        program.append(Instruction.iadd(19, 0, 2))  # R19 = 6
        
        # 4. Sum of Squares = Score^2
        program.append(Instruction.imul(20, 1, 1))  # R20 = R1 * R1 = 6*6 = 36
        
        # 5. 退出
        program.append(Instruction.exit_inst())
        
        return program
    
    @staticmethod
    def load_program(connection, program: List[Instruction]):
        """
        加載程序到 ESP32
        
        Args:
            connection: ESP32Connection 實例
            program: 指令列表
        """
        print(f"\n📋 Loading {len(program)} instructions...")
        
        for i, inst in enumerate(program):
            hex_str = inst.to_hex()
            connection.send_command(f"load {hex_str}", delay=0.1)
            # 讀取確認信息
            response = connection.read_lines()
            if response:
                for line in response:
                    if "Loaded" in line:
                        print(f"  [{i}] {line}")
        
        print(f"✅ Program loaded ({len(program)} instructions)\n")
    
    @staticmethod
    def get_expected_results() -> dict:
        """
        獲取預期的計算結果
        
        基於程序：
        R0 = 2, R1 = 3, R2 = 4
        R1 = 2*3 = 6 (Attention)
        R16-R19 = 2+4 = 6 (Residual)
        R20 = 6*6 = 36 (Sum of Squares)
        """
        return {
            'R0': 2,
            'R1': 6,   # Attention Score
            'R2': 4,
            'R16': 6,  # Residual
            'R17': 6,
            'R18': 6,
            'R19': 6,
            'R20': 36  # Sum of Squares
        }
