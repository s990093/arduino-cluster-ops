#!/usr/bin/env python3
"""
Assembly Parser - Parse .asm files to InstructionV15 objects

Converts Micro-CUDA assembly files to executable instructions.
"""

import re
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from esp32_tools.program_loader_v15 import InstructionV15

class AssemblyParser:
    """
    Parse Micro-CUDA assembly files
    
    Supported formats:
        MOV R0, 5           ; Comment
        IADD R1, R2, R3
        S2R R31, SR_LANEID
        EXIT
    """
    
    def __init__(self):
        self.instructions = []
    
    def parse_file(self, asm_file: Path) -> List[InstructionV15]:
        """
        Parse assembly file
        
        Args:
            asm_file: Path to .asm file
        
        Returns:
            List of InstructionV15 objects
        """
        if not asm_file.exists():
            raise FileNotFoundError(f"Assembly file not found: {asm_file}")
        
        with open(asm_file, 'r') as f:
            asm_text = f.read()
        
        return self.parse(asm_text)
    
    def parse(self, asm_text: str) -> List[InstructionV15]:
        """
        Parse assembly text
        
        Args:
            asm_text: Assembly code as string
        
        Returns:
            List of InstructionV15 objects
        """
        self.instructions = []
        labels = {}
        lines = []
        
        # Pre-process lines (remove comments, whitespace)
        raw_lines = asm_text.split('\n')
        instruction_count = 0
        
        # Pass 1: Collect labels and valid lines
        for line in raw_lines:
            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]
            
            line = line.strip()
            if not line:
                continue
            
            # Label definition: "Label:"
            if line.endswith(':'):
                label = line[:-1].strip()
                labels[label] = instruction_count
                continue
            
            # Store valid instruction line
            lines.append(line)
            instruction_count += 1
            
        # Pass 2: Generate instructions
        for i, line in enumerate(lines):
            inst = self.parse_instruction(line, labels, current_addr=i)
            if inst:
                self.instructions.append(inst)
        
        return self.instructions
    
    def parse_instruction(self, line: str, labels: dict = None, current_addr: int = 0) -> Optional[InstructionV15]:
        """Parse a single instruction line"""
        
        # Remove extra whitespace
        line = ' '.join(line.split())
        
        # EXIT
        if line.upper() == 'EXIT':
            return InstructionV15.exit_inst()
            
        # BRA label / BRA offset
        match = re.match(r'BRA\s+(\w+)', line, re.IGNORECASE)
        if match:
            target = match.group(1)
            # Try label
            if labels and target in labels:
                target_idx = labels[target]
                # Return Absolute Index
                return InstructionV15.bra(target_idx)
            
            # Try int (raw offset)
            try:
                val = int(target)
                return InstructionV15.bra(val)
            except ValueError:
                pass 
        
        # BRZ label / BRZ offset
        match = re.match(r'BRZ\s+(\w+)', line, re.IGNORECASE) # BRZ dest
        if match:
            target = match.group(1)
            if labels and target in labels:
                target_idx = labels[target]
                return InstructionV15.brz(target_idx)

            try:
                val = int(target)
                return InstructionV15.brz(val)
            except ValueError:
                pass
        
        # BRZ P0, label
        match = re.match(r'BRZ\s+P(\d+),\s*(\w+)', line, re.IGNORECASE)
        if match:
            target = match.group(2)
            if labels and target in labels:
                 target_idx = labels[target]
                 return InstructionV15.brz(target_idx)
        
        # ISETP.EQ P0, R1, R2
        match = re.match(r'ISETP\.EQ\s+P(\d+),\s*R(\d+),\s*R(\d+)', line, re.IGNORECASE)
        if match:
            return InstructionV15.isetp_eq(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            
        # ISETP.GT P0, R1, R2
        match = re.match(r'ISETP\.GT\s+P(\d+),\s*R(\d+),\s*R(\d+)', line, re.IGNORECASE)
        if match:
            return InstructionV15.isetp_gt(int(match.group(1)), int(match.group(2)), int(match.group(3)))

        # MOV Rd, imm
        match = re.match(r'MOV\s+R(\d+),\s*(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            imm = int(match.group(2))
            return InstructionV15.mov(rd, imm)
        
        # S2R Rd, SR_X
        match = re.match(r'S2R\s+R(\d+),\s*SR[_\s]*(\w+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            sr_name = match.group(2).upper()
            
            # Map SR names to indices
            sr_map = {
                'LANEID': InstructionV15.SR_LANEID,
                'WARPSIZE': InstructionV15.SR_WARPSIZE,
                '2': InstructionV15.SR_LANEID,  # Direct index
            }
            
            sr = sr_map.get(sr_name, int(sr_name) if sr_name.isdigit() else 2)
            return InstructionV15.s2r(rd, sr)
        
        # IADD Rd, Rs1, Rs2
        match = re.match(r'IADD\s+R(\d+),\s*R(\d+),\s*R(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs1 = int(match.group(2))
            rs2 = int(match.group(3))
            return InstructionV15.iadd(rd, rs1, rs2)
        
        # IMUL Rd, Rs1, Rs2
        match = re.match(r'IMUL\s+R(\d+),\s*R(\d+),\s*R(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs1 = int(match.group(2))
            rs2 = int(match.group(3))
            return InstructionV15.imul(rd, rs1, rs2)
        
        # ISUB Rd, Rs1, Rs2
        match = re.match(r'ISUB\s+R(\d+),\s*R(\d+),\s*R(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs1 = int(match.group(2))
            rs2 = int(match.group(3))
            return InstructionV15.isub(rd, rs1, rs2)

        # FADD Fd, Fs1, Fs2
        match = re.match(r'FADD\s+[FR](\d+),\s*[FR](\d+),\s*[FR](\d+)', line, re.IGNORECASE)
        if match:
            fd = int(match.group(1))
            fs1 = int(match.group(2))
            fs2 = int(match.group(3))
            return InstructionV15.fadd(fd, fs1, fs2)
        
        # FMUL Fd, Fs1, Fs2
        match = re.match(r'FMUL\s+[FR](\d+),\s*[FR](\d+),\s*[FR](\d+)', line, re.IGNORECASE)
        if match:
            fd = int(match.group(1))
            fs1 = int(match.group(2))
            fs2 = int(match.group(3))
            return InstructionV15.fmul(fd, fs1, fs2)

        # Bitwise Operations: AND, OR, XOR, SHL, SHR
        # Format: OP Rd, Rs1, Rs2
        for op_name, op_func in [
            ('AND', InstructionV15.and_op),
            ('OR',  InstructionV15.or_op),
            ('XOR', InstructionV15.xor_op),
            ('SHL', InstructionV15.shl),
            ('SHR', InstructionV15.shr)
        ]:
            match = re.match(rf'{op_name}\s+R(\d+),\s*R(\d+),\s*R(\d+)', line, re.IGNORECASE)
            if match:
                return op_func(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # LDX Rd, [Rs1 + Rs2]  or  LDX Rd, Rs1, Rs2
        match = re.match(r'LDX\s+R(\d+),\s*(?:\[)?R(\d+)(?:\s*\+\s*|\s*,\s*)R(\d+)\]?', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs1 = int(match.group(2))
            rs2 = int(match.group(3))
            return InstructionV15.ldx(rd, rs1, rs2)
        
        # STX [Rd + Roffset], Rs  or  STX Rd, Roffset, Rs
        match = re.match(r'STX\s+(?:\[)?R(\d+)(?:\s*\+\s*|\s*,\s*)R(\d+)\]?,\s*R(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            roffset = int(match.group(2))
            rs = int(match.group(3))
            return InstructionV15.stx(rd, roffset, rs)
        
        # LDL Rd, [Rs]  or  LDL Rd, Rs
        match = re.match(r'LDL\s+R(\d+),\s*(?:\[)?R(\d+)\]?', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs = int(match.group(2))
            return InstructionV15.ldl(rd, rs)
        
        # LDG Rd, [Rs]  or  LDG Rd, Rs (Global Load)
        match = re.match(r'LDG\s+R(\d+),\s*(?:\[)?R(\d+)\]?', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs = int(match.group(2))
            return InstructionV15.ldg(rd, rs)

        # STG [Rd], Rs  or  STG Rd, Rs (Global Store)
        match = re.match(r'STG\s+(?:\[)?R(\d+)\]?,\s*R(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1)) # Address
            rs = int(match.group(2)) # Value
            return InstructionV15.stg(rd, rs)
        
        # STL [Rd], Rs  or  STL Rd, Rs
        match = re.match(r'STL\s+(?:\[)?R(\d+)\]?,\s*R(\d+)', line, re.IGNORECASE)
        if match:
            rd = int(match.group(1))
            rs = int(match.group(2))
            return InstructionV15.stl(rd, rs)
        
        # If we reach here, instruction not recognized
        # print(f"[WARN] Unrecognized instruction: {line}")
        return None


def parse_asm_file(asm_file: Path) -> List[InstructionV15]:
    """
    Convenience function to parse assembly file
    
    Args:
        asm_file: Path to .asm file
    
    Returns:
        List of InstructionV15 objects
    """
    parser = AssemblyParser()
    return parser.parse_file(asm_file)


def parse_asm(asm_text: str) -> List[InstructionV15]:
    """
    Convenience function to parse assembly text
    
    Args:
        asm_text: Assembly code as string
    
    Returns:
        List of InstructionV15 objects
    """
    parser = AssemblyParser()
    return parser.parse(asm_text)
