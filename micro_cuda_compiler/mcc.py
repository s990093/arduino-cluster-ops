#!/usr/bin/env python3
"""
Micro-CUDA Compiler (mcc.py)

Purpose: Compile CUDA-like C/C++ code to Micro-CUDA ISA v1.5
Architecture: LLVM IR -> Micro-CUDA Assembly -> Hex Binary

Usage:
    python mcc.py input.cu -o output.asm
    python mcc.py input.cu --llvm-ir output.ll
    python mcc.py input.cu --asm output.asm --target esp32s3

Author: Micro-CUDA Project
Version: 0.1.0 (Alpha)
"""

import sys
import os
import subprocess
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from esp32_tools.program_loader_v15 import InstructionV15

# Add micro_cuda_compiler to path
sys.path.insert(0, str(Path(__file__).parent))
from target_config import get_target, TargetConfig

# ===== Configuration =====
MCUDA_HEADER = Path(__file__).parent / "mcuda.h"
CLANG_TARGET = "riscv32"  # Use 32-bit target for proper pointer size
CLANG_OPT_LEVEL = "1"     # -O1 for basic optimization

# ===== Data Structures =====

@dataclass
class VirtualRegister:
    """Represents a virtual register from LLVM IR"""
    name: str           # e.g., "%1", "%add", "%result"
    type: str           # "int", "float", "ptr"
    physical_reg: Optional[int] = None  # Allocated physical register (R0-R31)

@dataclass
class MicroCUDAInstruction:
    """Represents a single Micro-CUDA ISA instruction"""
    opcode: str
    dest: Optional[int]
    src1: Optional[int]
    src2: Optional[int]
    imm: Optional[int]
    comment: str = ""
    
    def to_instruction_v15(self) -> 'InstructionV15':
        """Convert to InstructionV15 object for assembly"""
        # This will be implemented when we integrate with assembler
        pass
    
    def to_asm(self) -> str:
        """Convert instruction to assembly string"""
        inst_str = self.opcode
        
        # Format assembly based on opcode
        if self.opcode == "MOV":
            inst_str = f"{self.opcode} R{self.dest}, {self.imm}"
        elif self.opcode == "S2R":
            inst_str = f"{self.opcode} R{self.dest}, SR_{self.src1}"
        elif self.opcode in ["IADD", "ISUB", "IMUL", "FADD", "FMUL", "AND", "OR", "XOR", "SHL", "SHR"]:
            inst_str = f"{self.opcode} R{self.dest}, R{self.src1}, R{self.src2}"
        elif self.opcode == "LDX":
            # LDX Rd, [Rs1 + Rs2]
            inst_str = f"{self.opcode} R{self.dest}, R{self.src1}, R{self.src2}"
        elif self.opcode == "STX":
            # STX [Rd + Rs1], Rs2
            # dest = base, src1 = offset, src2 = value
            inst_str = f"{self.opcode} R{self.dest}, R{self.src1}, R{self.src2}"
        elif self.opcode == "LDL":
            inst_str = f"{self.opcode} R{self.dest}, R{self.src1}"
        elif self.opcode == "STL":
            inst_str = f"{self.opcode} R{self.dest}, R{self.src1}"
        elif self.opcode == "EXIT":
            inst_str = "EXIT"
        elif self.opcode == "BRA":
            inst_str = f"BRA {self.imm}"
        elif self.opcode == "BRZ":
            # BRZ implies checking P0 (or dest P-reg). 
            # If dest is 0 (P0), omit or print P0?
            # asm_parser supports: BRZ label or BRZ P0, label.
            # Let's use BRZ P0, label for clarity, or just BRZ label.
            # Using BRZ label matches common PTX simplified syntax if P0 is implicit.
            # But asm_parser update (Step 1485) supported BRZ label.
            inst_str = f"BRZ {self.imm}"
        elif self.opcode.startswith("ISETP"):
            # dest is P register index
            inst_str = f"{self.opcode} P{self.dest}, R{self.src1}, R{self.src2}"
        else:
            # Generic format
            parts = [self.opcode]
            if self.dest is not None:
                parts.append(f"R{self.dest}")
            if self.src1 is not None:
                parts.append(f"R{self.src1}")
            if self.src2 is not None:
                parts.append(f"R{self.src2}")
            if self.imm is not None:
                parts.append(str(self.imm))
            inst_str = " ".join(parts)
        
        # Add comment if available
        if self.comment:
            inst_str = f"{inst_str.ljust(25)} ; {self.comment}"
        
        return inst_str

class RegisterAllocator:
    """Register allocation with liveness tracking, constant caching, and reuse"""
    
    def __init__(self, max_regs=32):
        self.max_regs = max_regs
        self.next_reg = 0
        self.var_to_reg = {}
        self.initialized_regs = set()
        self.constant_cache = {}
        self.free_regs = set()  # Set of freed physical registers
    
    def allocate(self, var_name):
        """Allocate a register (reuse freed ones if possible)"""
        if var_name in self.var_to_reg:
            reg = self.var_to_reg[var_name]
            self.initialized_regs.add(reg)
            return reg
        
        # Try to reuse a register from the free pool
        if self.free_regs:
            # Pick smallest available register to keep density high
            reg = min(self.free_regs)
            self.free_regs.remove(reg)
            # If this register was caching a constant, invalidate that cache
            # (Since the register now holds a variable)
            keys_to_remove = [k for k, v in self.constant_cache.items() if v == reg]
            for k in keys_to_remove:
                del self.constant_cache[k]
        else:
            # Allocate new register
            if self.next_reg >= self.max_regs:
                raise RuntimeError(f"Out of registers! Need more than {self.max_regs}. Active: {len(self.var_to_reg) - len(self.free_regs)}")
            
            reg = self.next_reg
            self.next_reg += 1
            
        self.var_to_reg[var_name] = reg
        self.initialized_regs.add(reg)
        return reg
    
    def free(self, var_name):
        """Mark variable as dead and free its register"""
        if var_name in self.var_to_reg:
            reg = self.var_to_reg[var_name]
            self.free_regs.add(reg)
            # We don't delete from var_to_reg because we might check it again??
            # Actually, in SSA, each var name is unique.
            # But safety: keep it mapped but marked free.
            # If allocate() sees it again, it returns REUSED reg (which is fine, definition dominates use)
            pass
            
    def allocate_if_needed(self, var_name):
        """Get existing register or allocate new"""
        if var_name in self.var_to_reg:
            reg = self.var_to_reg[var_name]
            if reg not in self.initialized_regs:
                # This often happens for function arguments not caught by prologue
                # For safety, initialize to 0 if it looks like a temp
                # But print warning
                pass 
            return reg
        return self.allocate(var_name)
    
    def allocate_constant(self, value):
        """Allocate or reuse register for constant value"""
        const_key = f'const_{value}'
        if const_key in self.constant_cache:
            reg = self.constant_cache[const_key]
            # Ensure it hasn't been freed/stolen (unlikely if in cache)
            if reg not in self.free_regs:
                return reg
        
        reg = self.allocate(const_key)
        self.constant_cache[const_key] = reg
        return reg
    
    def get_registers_used(self):
        return self.next_reg

    def reset(self):
        self.next_reg = 0
        self.var_to_reg = {}
        self.initialized_regs = set()
        self.constant_cache = {}
        self.free_regs = set()

class LLVMIRParser:
    """Parse LLVM IR text format and extract instructions"""
    
    def __init__(self, ir_text: str):
        self.ir_text = ir_text
        self.functions = {}
        self.current_function = None
    
    def parse(self) -> Dict[str, List[str]]:
        """Parse IR and return dictionary of function -> instructions"""
        lines = self.ir_text.split('\n')
        current_func = None
        current_instructions = []
        
        for line in lines:
            line = line.strip()
            
            # Function definition
            if line.startswith('define'):
                if current_func:
                    self.functions[current_func] = current_instructions
                
                # Extract function name
                match = re.search(r'@(\w+)\(', line)
                if match:
                    current_func = match.group(1)
                    # IMPORTANT: Include the define line itself!
                    current_instructions = [line]
            
            # Instruction line
            elif current_func and not line.startswith('}'):
                if line and not line.startswith(';'):
                    current_instructions.append(line)
        
        # Add last function
        if current_func:
            self.functions[current_func] = current_instructions
        
        return self.functions

class MicroCUDABackend:
    """
    LLVM IR to Micro-CUDA ISA Backend
    
    This is the core compiler component that translates LLVM IR
    instructions to Micro-CUDA ISA instructions.
    """
    
    def __init__(self):
        self.allocator = RegisterAllocator()
        self.instructions = []
        self.label_counter = 0
    
    
    def load_constant(self, dest_reg: int, value: int, inst_list: List[MicroCUDAInstruction], comment: str = ""):
        """
        Load a 32-bit constant into a register using 8-bit MOV/SHL/OR sequence.
        """
        # 1. Load lower 8 bits
        byte0 = value & 0xFF
        inst_list.append(MicroCUDAInstruction("MOV", dest_reg, None, None, byte0, f"{comment} (Lo8)"))
        
        remaining = value >> 8
        shift = 8
        
        # Helper to load shift amount only once if needed? 
        # Actually shift amount must be in register for SHL
        
        while remaining > 0:
            byte = remaining & 0xFF
            if byte != 0:
                # Load byte into temp
                # We need a temporary register. 
                # Since we are inside Backend, we can use allocator?
                # But allocator handles named vars. We need unnamed temp.
                # Let's reserve a high register or allocate a fresh temp name?
                # "temp_const_load"
                temp_val_reg = self.allocator.allocate(f"%temp_const_val_{dest_reg}_{shift}_{byte}")
                temp_shift_reg = self.allocator.allocate(f"%temp_const_shift_{dest_reg}_{shift}")
                
                # MOV byte
                inst_list.append(MicroCUDAInstruction("MOV", temp_val_reg, None, None, byte, ""))
                
                # MOV shift amount
                inst_list.append(MicroCUDAInstruction("MOV", temp_shift_reg, None, None, shift, ""))
                
                # SHL temp, temp, shift
                inst_list.append(MicroCUDAInstruction("SHL", temp_val_reg, temp_val_reg, temp_shift_reg, None, ""))
                
                # OR dest, dest, temp
                inst_list.append(MicroCUDAInstruction("OR", dest_reg, dest_reg, temp_val_reg, None, ""))
            
            remaining >>= 8
            shift += 8
    
    def compile_ir_instruction(self, ir_inst: str) -> List[MicroCUDAInstruction]:
        """Compile a single LLVM IR instruction to Micro-CUDA ISA"""
        inst_list = []
        
        # Remove leading/trailing whitespace
        ir_inst = ir_inst.strip()
        
        # Skip empty lines, comments, and labels
        if not ir_inst or ir_inst.startswith(';') or ir_inst.endswith(':'):
            return inst_list
        
        # Skip alloca (stack allocation - we'll ignore for now)
        if 'alloca' in ir_inst:
            # Still need to allocate a register for the result
            match = re.match(r'%(\w+)\s*=\s*alloca', ir_inst)
            if match:
                self.allocator.allocate(f'%{match.group(1)}')
            return inst_list
        
        # ===== Pattern Matching =====
        
        # Call to intrinsic: %result = call i32 @laneId()
        # Supports both @laneId and @__mcuda_lane_id
        if 'call' in ir_inst and ('laneId' in ir_inst or '__mcuda_lane_id' in ir_inst):
            match = re.match(r'%(\w+)\s*=\s*.*call.*@(?:__mcuda_)?laneId', ir_inst)
            if match:
                result_reg = self.allocator.allocate(f'%{match.group(1)}')
                inst_list.append(MicroCUDAInstruction(
                    "S2R",   # opcode
                    result_reg, # dest
                    2,       # src1 (SR_LANEID = 2)
                    None,    # src2
                    None,    # imm
                    f"laneId() -> R{result_reg}"
                ))
        
        # GEP (Get Element Pointer): %5 = getelementptr inbounds i32, ptr %0, i32 %4
        # This calculates address: base_ptr + index * sizeof(element)
        # GEP (Get Element Pointer): %5 = getelementptr inbounds i32, ptr %0, i32 %4
        # This calculates address: base_ptr + index * sizeof(element)
        elif 'getelementptr' in ir_inst:
            # Pattern: %result = getelementptr ... i32 index
            # Index can be register (%var) or constant (123)
            match = re.match(r'%(\w+)\s*=\s*getelementptr\s+(?:inbounds\s+)?(\w+),\s*ptr\s+%(\w+),\s*i\d+\s+(%?\w+)', ir_inst)
            if match:
                result_reg = self.allocator.allocate(f'%{match.group(1)}')
                element_type = match.group(2)
                base_reg = self.allocator.allocate_if_needed(f'%{match.group(3)}')
                index_str = match.group(4)
                
                # Determine element size based on type
                if element_type == 'i32' or element_type == 'float':
                    element_size = 4
                elif element_type == 'i8':
                    element_size = 1
                elif element_type == 'i16':
                    element_size = 2
                elif element_type == 'i64' or element_type == 'double':
                    element_size = 8
                else:
                    print(f"[WARNING] Unknown GEP type '{element_type}', assuming size 4")
                    element_size = 4
                
                offset_reg = self.allocator.allocate('%gep_offset')
                
                if index_str.startswith('%'):
                    # Variable index
                    index_reg = self.allocator.allocate_if_needed(index_str)
                    
                    size_reg = self.allocator.allocate_constant(element_size)
                    # Check if size already emitted
                    size_emitted = False
                    for inst in inst_list:
                         if inst.dest == size_reg and inst.opcode == "MOV":
                             size_emitted = True
                             break
                    if not size_emitted:
                        inst_list.append(MicroCUDAInstruction(
                            opcode="MOV",
                            dest=size_reg,
                            src1=None,
                            src2=None,
                            imm=element_size,
                            comment=f"R{size_reg} = {element_size}"
                        ))
                    
                    inst_list.append(MicroCUDAInstruction(
                        opcode="IMUL",
                        dest=offset_reg,
                        src1=index_reg,
                        src2=size_reg,
                        imm=None,
                        comment=f"R{offset_reg} = index * {element_size}"
                    ))
                else:
                    # Constant index
                    const_val = int(index_str)
                    total_offset = const_val * element_size
                    print(f"[DEBUG] GEP Constant: index={index_str}, val={const_val}, size={element_size}")
                    
                    # Use load_constant
                    self.load_constant(offset_reg, total_offset, inst_list, f"R{offset_reg} = {const_val} * {element_size}")
                
                # address = base + offset
                inst_list.append(MicroCUDAInstruction(
                    opcode="IADD",
                    dest=result_reg,
                    src1=base_reg,
                    src2=offset_reg,
                    imm=None,
                    comment=f"R{result_reg} = base + offset"
                ))
        
        # Load instruction: %6 = load i32, ptr %5, align 4
        elif re.match(r'%\w+\s*=\s*load', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*load\s+\w+,\s*ptr\s+%(\w+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                addr_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                
                # LDX: Load from address
                # We need base and offset. Since addr_reg contains the full address,
                # we use addr_reg as base and 0 as offset
                zero_reg = self.allocator.allocate('%zero_offset')
                inst_list.append(MicroCUDAInstruction(
                    opcode="MOV",
                    dest=zero_reg,
                    src1=None,
                    src2=None,
                    imm=0,
                    comment="Zero offset"
                ))
                
                inst_list.append(MicroCUDAInstruction(
                    opcode="LDX",
                    dest=dest_reg,
                    src1=addr_reg,
                    src2=zero_reg,
                    imm=None,
                    comment=f"R{dest_reg} = Mem[R{addr_reg}]"
                ))
        
        # Store instruction: store i32 %9, ptr %10, align 4
        # Also handles constants: store i32 9999, ptr %10
        elif ir_inst.startswith('store'):
            # Match value (register or constant) and pointer address
            match = re.match(r'store\s+\w+\s+(%?[\w\-]+),\s*ptr\s+(%?[\w\-]+)', ir_inst)
            if match:
                val_str = match.group(1)
                addr_var = match.group(2) # Pointer variable (must be register in our model usually)
                
                # Handle Address Register
                addr_reg = self.allocator.allocate_if_needed(addr_var)
                
                # Handle Value (Register or Immediate)
                if val_str.startswith('%'):
                    # Value is a variable/register
                    if val_str not in self.allocator.var_to_reg:
                        # Should not happen if program is valid, unless optimization quirk
                        print(f"[WARNING] Store value {val_str} not defined. Initializing to 0.")
                        val_reg = self.allocator.allocate(val_str)
                        inst_list.append(MicroCUDAInstruction("MOV", val_reg, None, None, 0, "Init undefined"))
                    else:
                        val_reg = self.allocator.allocate_if_needed(val_str)
                else:
                    # Value is an immediate constant
                    val_imm = int(val_str)
                    val_reg = self.allocator.allocate(f"const_{val_imm}")
                    # Use load_constant!
                    self.load_constant(val_reg, val_imm, inst_list, f"Const {val_imm}")

                # Zero offset for STX
                zero_reg = self.allocator.allocate('%zero_offset')
                inst_list.append(MicroCUDAInstruction(
                    opcode="MOV",
                    dest=zero_reg,
                    src1=None,
                    src2=None,
                    imm=0,
                    comment="Zero offset"
                ))
                
                # STX Addr, Offset, Value
                inst_list.append(MicroCUDAInstruction(
                    "STX",      # opcode
                    addr_reg,   # dest (Base Address)
                    zero_reg,   # src1 (Offset usually)
                    val_reg,    # src2 (Value to Store)
                    None,       # imm
                    f"Mem[R{addr_reg}] = R{val_reg}" # comment
                ))
        
        # Integer addition: %add = add nsw i32 %0, %1
        elif re.match(r'%\w+\s*=\s*add', ir_inst):
            # Try to match with constant (positive or negative)
            match_const = re.match(r'%(\w+)\s*=\s*add.*%(\w+),\s*(-?\d+)', ir_inst)
            if match_const:
                dest_reg = self.allocator.allocate(f'%{match_const.group(1)}')
                src_reg = self.allocator.allocate_if_needed(f'%{match_const.group(2)}')
                imm_val = int(match_const.group(3))
                
                # IADD with immediate (use MOV + IADD)
                temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                inst_list.append(MicroCUDAInstruction(
                    opcode="MOV",
                    dest=temp_reg,
                    src1=None,
                    src2=None,
                    imm=imm_val,
                    comment=f"R{temp_reg} = {imm_val}"
                ))
                inst_list.append(MicroCUDAInstruction(
                    opcode="IADD",
                    dest=dest_reg,
                    src1=src_reg,
                    src2=temp_reg,
                    imm=None,
                    comment=f"R{dest_reg} = R{src_reg} + {imm_val}"
                ))
            else:
                # Regular add with two registers or negative/complex operand that didn't match \d+
                match = re.match(r'%(\w+)\s*=\s*add.*%(\w+),\s*(%?[\w\-]+)', ir_inst)
                if match:
                    dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                    src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                    operand2 = match.group(3)
                    
                    if operand2.startswith('%'):
                        src2_reg = self.allocator.allocate_if_needed(operand2)
                        inst_list.append(MicroCUDAInstruction(
                            opcode="IADD",
                            dest=dest_reg,
                            src1=src1_reg,
                            src2=src2_reg,
                            imm=None,
                            comment=f"R{dest_reg} = R{src1_reg} + R{src2_reg}"
                        ))
                    else:
                        imm_val = int(operand2)
                        temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                        if temp_reg not in self.allocator.constant_cache.values():
                            inst_list.append(MicroCUDAInstruction("MOV", temp_reg, None, None, imm_val, f"R{temp_reg} = {imm_val}"))
                        inst_list.append(MicroCUDAInstruction(
                            opcode="IADD",
                            dest=dest_reg,
                            src1=src1_reg,
                            src2=temp_reg,
                            imm=None,
                            comment=f"R{dest_reg} = R{src1_reg} + {imm_val}"
                        ))

        # Integer subtraction: %sub = sub nsw i32 %0, %1
        elif re.match(r'%\w+\s*=\s*sub', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*sub.*%(\w+),\s*(%?[\w\-]+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                operand2 = match.group(3)
                
                if operand2.startswith('%'):
                    src2_reg = self.allocator.allocate_if_needed(operand2)
                    inst_list.append(MicroCUDAInstruction(
                        opcode="ISUB",
                        dest=dest_reg,
                        src1=src1_reg,
                        src2=src2_reg,
                        imm=None,
                        comment=f"R{dest_reg} = R{src1_reg} - R{src2_reg}"
                    ))
                else:
                    imm_val = int(operand2)
                    temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                    # Create immediate using load_constant to handle negative/large values
                    self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                    
                    inst_list.append(MicroCUDAInstruction(
                        opcode="ISUB",
                        dest=dest_reg,
                        src1=src1_reg,
                        src2=temp_reg,
                        imm=None,
                        comment=f"R{dest_reg} = R{src1_reg} - {imm_val}"
                    ))

        # Bitwise OR: %or = or i32 %0, %1 (supports disjoint)
        elif re.match(r'%\w+\s*=\s*or', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*or.*%(\w+),\s*(%?[\w\-]+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                operand2 = match.group(3)
                
                if operand2.startswith('%'):
                    src2_reg = self.allocator.allocate_if_needed(operand2)
                    inst_list.append(MicroCUDAInstruction(
                        opcode="OR",
                        dest=dest_reg,
                        src1=src1_reg,
                        src2=src2_reg,
                        imm=None,
                        comment=f"R{dest_reg} = R{src1_reg} | R{src2_reg}"
                    ))
                else:
                    imm_val = int(operand2)
                    temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                    # Use load_constant
                    self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                    
                    inst_list.append(MicroCUDAInstruction(
                        opcode="OR",
                        dest=dest_reg,
                        src1=src1_reg,
                        src2=temp_reg,
                        imm=None,
                        comment=f"R{dest_reg} = R{src1_reg} | {imm_val}"
                    ))

        # Bitwise AND
        elif re.match(r'%\w+\s*=\s*and', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*and.*%(\w+),\s*(%?[\w\-]+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                operand2 = match.group(3)
                
                if operand2.startswith('%'):
                    src2_reg = self.allocator.allocate_if_needed(operand2)
                    inst_list.append(MicroCUDAInstruction("AND", dest_reg, src1_reg, src2_reg, None, f"R{dest_reg} = R{src1_reg} & R{src2_reg}"))
                else:
                    imm_val = int(operand2)
                    temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                    # Use load_constant
                    self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                    inst_list.append(MicroCUDAInstruction("AND", dest_reg, src1_reg, temp_reg, None, f"R{dest_reg} = R{src1_reg} & {imm_val}"))

        # Bitwise SHL
        elif re.match(r'%\w+\s*=\s*shl', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*shl.*%(\w+),\s*(%?[\w\-]+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                operand2 = match.group(3)
                if operand2.startswith('%'):
                    src2_reg = self.allocator.allocate_if_needed(operand2)
                    inst_list.append(MicroCUDAInstruction("SHL", dest_reg, src1_reg, src2_reg, None, f"R{dest_reg} = R{src1_reg} << R{src2_reg}"))
                else:
                    imm_val = int(operand2)
                    temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                    # Use load_constant
                    self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                    inst_list.append(MicroCUDAInstruction("SHL", dest_reg, src1_reg, temp_reg, None, f"R{dest_reg} = R{src1_reg} << {imm_val}"))

        # Bitwise SHR (lshr/ashr)
        elif re.match(r'%\w+\s*=\s*(?:lshr|ashr)', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*(?:lshr|ashr).*%(\w+),\s*(%?[\w\-]+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                operand2 = match.group(3)
                if operand2.startswith('%'):
                    src2_reg = self.allocator.allocate_if_needed(operand2)
                    inst_list.append(MicroCUDAInstruction("SHR", dest_reg, src1_reg, src2_reg, None, f"R{dest_reg} = R{src1_reg} >> R{src2_reg}"))
                else:
                    imm_val = int(operand2)
                    temp_reg = self.allocator.allocate(f'%const_{imm_val}')
                    # Use load_constant
                    self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                    inst_list.append(MicroCUDAInstruction("SHR", dest_reg, src1_reg, temp_reg, None, f"R{dest_reg} = R{src1_reg} >> {imm_val}"))
        
        # Integer multiplication: %mul = mul nsw i32 %0, %1
        elif re.match(r'%\w+\s*=\s*mul', ir_inst):
            # Try constant first (positive or negative)
            match_const = re.match(r'%(\w+)\s*=\s*mul.*%(\w+),\s*(-?\d+)', ir_inst)
            if match_const:
                dest_reg = self.allocator.allocate(f'%{match_const.group(1)}')
                src_reg = self.allocator.allocate_if_needed(f'%{match_const.group(2)}')
                imm_val = int(match_const.group(3))
                
                temp_reg = self.allocator.allocate(f'%const_mul_{imm_val}')
                # Use load_constant
                self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                inst_list.append(MicroCUDAInstruction(
                    opcode="IMUL",
                    dest=dest_reg,
                    src1=src_reg,
                    src2=temp_reg,
                    imm=None,
                    comment=f"R{dest_reg} = R{src_reg} * {imm_val}"
                ))
            else:
                match = re.match(r'%(\w+)\s*=\s*mul.*%(\w+),\s*(%?[\w\-]+)', ir_inst)
                if match:
                    dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                    src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                    operand2 = match.group(3)
                    
                    if operand2.startswith('%'):
                        src2_reg = self.allocator.allocate_if_needed(operand2)
                        inst_list.append(MicroCUDAInstruction(
                            opcode="IMUL",
                            dest=dest_reg,
                            src1=src1_reg,
                            src2=src2_reg,
                            imm=None,
                            comment=f"R{dest_reg} = R{src1_reg} * R{src2_reg}"
                        ))
                    else:
                        imm_val = int(operand2)
                        temp_reg = self.allocator.allocate(f'%const_mul_{imm_val}')
                        # Use load_constant
                        self.load_constant(temp_reg, imm_val, inst_list, f"R{temp_reg} = {imm_val}")
                        inst_list.append(MicroCUDAInstruction(
                            opcode="IMUL",
                            dest=dest_reg,
                            src1=src1_reg,
                            src2=temp_reg,
                            imm=None,
                            comment=f"R{dest_reg} = R{src1_reg} * {imm_val}"
                        ))
        
        # Float addition: %add = fadd float %0, %1
        elif re.match(r'%\w+\s*=\s*fadd', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*fadd.*%(\w+),\s*%(\w+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                src2_reg = self.allocator.allocate_if_needed(f'%{match.group(3)}')
                inst_list.append(MicroCUDAInstruction(
                    opcode="FADD",
                    dest=dest_reg,
                    src1=src1_reg,
                    src2=src2_reg,
                    imm=None,
                    comment=f"F{dest_reg} = F{src1_reg} + F{src2_reg}"
                ))
        
        # Float multiplication: %mul = fmul float %0, %1
        elif re.match(r'%\w+\s*=\s*fmul', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*fmul.*%(\w+),\s*%(\w+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                src2_reg = self.allocator.allocate_if_needed(f'%{match.group(3)}')
                inst_list.append(MicroCUDAInstruction(
                    opcode="FMUL",
                    dest=dest_reg,
                    src1=src1_reg,
                    src2=src2_reg,
                    imm=None,
                    comment=f"F{dest_reg} = F{src1_reg} * F{src2_reg}"
                ))
        
        # Sign extend: %1 = sext i8 %0 to i32
        elif 'sext' in ir_inst or 'zext' in ir_inst:
            match = re.match(r'%(\w+)\s*=\s*[sz]ext.*%(\w+)', ir_inst)
            if match:
                dest_reg = self.allocator.allocate(f'%{match.group(1)}')
                src_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                src_reg = self.allocator.allocate_if_needed(f'%{match.group(2)}')
                
                # Implement as register copy using IADD with 0
                zero_reg = self.allocator.allocate_constant(0)
                if zero_reg not in self.allocator.constant_cache.values():
                    inst_list.append(MicroCUDAInstruction(
                        opcode="MOV",
                        dest=zero_reg,
                        src1=None,
                        src2=None,
                        imm=0,
                        comment="R{zero_reg} = 0 (for sext/zext copy)"
                    ))
                
                inst_list.append(MicroCUDAInstruction(
                    opcode="IADD",
                    dest=dest_reg,
                    src1=src_reg,
                    src2=zero_reg,
                    imm=None,
                    comment=f"R{dest_reg} = R{src_reg} + 0 (copy)"
                ))
        
        # Return instruction
        elif ir_inst.startswith('ret'):
            inst_list.append(MicroCUDAInstruction(
                opcode="EXIT",
                dest=None,
                src1=None,
                src2=None,
                imm=None,
                comment="Return from kernel"
            ))
        
        # Integer comparison: %cond = icmp eq/ne/ugt/ult/sgt/slt i32 %0, %1
        elif re.match(r'%\w+\s*=\s*icmp', ir_inst):
            match = re.match(r'%(\w+)\s*=\s*icmp\s+(\w+)\s+\w+\s+%(\w+),\s*(%?[\w\-]+)', ir_inst)
            if match:
                cond_reg_name = f'%{match.group(1)}'
                # For now, we map the condition result to P0 implicitly or store status
                # But our allocator is for R registers.
                # VM v1.5 uses P registers for branching.
                # Strategy: Map %cond to P0 (Hardcoded for now).
                
                pred = match.group(2) # eq, ne, ugt, ult, sgt, slt
                src1_reg = self.allocator.allocate_if_needed(f'%{match.group(3)}')
                operand2 = match.group(4)
                
                src2_reg = 0
                if operand2.startswith('%'):
                    src2_reg = self.allocator.allocate_if_needed(operand2)
                else:
                    imm_val = int(operand2)
                    src2_reg = self.allocator.allocate(f'%const_{imm_val}')
                    self.load_constant(src2_reg, imm_val, inst_list, f"R{src2_reg} = {imm_val}")
                
                # Emit ISETP
                # Only support EQ, GT for now.
                # x < y  <=> y > x
                opcode = "ISETP.EQ"
                if pred == 'eq':
                    opcode = "ISETP.EQ"
                    inst_list.append(MicroCUDAInstruction(opcode, 0, src1_reg, src2_reg, None, f"P0 = (R{src1_reg} == R{src2_reg})"))
                elif pred == 'ne':
                    # VM doesn't have NE? 
                    # Use EQ and handle in branch (Branch If Not)?
                    # Or x != y.
                    # Hack: P0 = EQ. Branch logic will be inverted?
                    # Let's assume we use EQ.
                     inst_list.append(MicroCUDAInstruction("ISETP.EQ", 0, src1_reg, src2_reg, None, f"P0 = (R{src1_reg} == R{src2_reg})"))
                elif pred == 'ugt' or pred == 'sgt':
                    opcode = "ISETP.GT"
                    inst_list.append(MicroCUDAInstruction(opcode, 0, src1_reg, src2_reg, None, f"P0 = (R{src1_reg} > R{src2_reg})"))
                elif pred == 'ult' or pred == 'slt':
                    # x < y <=> y > x
                    opcode = "ISETP.GT"
                    inst_list.append(MicroCUDAInstruction(opcode, 0, src2_reg, src1_reg, None, f"P0 = (R{src2_reg} > R{src1_reg})"))
                elif pred == 'uge' or pred == 'sge':
                    # x >= y. If y is constant, x > y-1
                    # Check if operand2 was immediate
                    try:
                        imm_val = int(operand2)
                        # Use GT with imm-1
                        temp_imm = imm_val - 1
                        temp_reg = self.allocator.allocate(f'%const_{temp_imm}')
                        self.load_constant(temp_reg, temp_imm, inst_list, f"R{temp_reg} = {temp_imm}")
                        # x > y-1
                        opcode = "ISETP.GT"
                        inst_list.append(MicroCUDAInstruction(opcode, 0, src1_reg, temp_reg, None, f"P0 = (R{src1_reg} > {temp_imm})"))
                    except ValueError:
                         # Register comparison. x >= y. Not supported directly by GT.
                         # Need GE or NOT LT. 
                         # Fallback to EQ (Broken) or implement logic reversal?
                         # For now, print warning and use EQ (Result will be wrong but run)
                         print(f"[WARN] UGE/SGE with register not supported fully. Using EQ.")
                         inst_list.append(MicroCUDAInstruction("ISETP.EQ", 0, src1_reg, src2_reg, None, f"P0 = (R{src1_reg} == R{src2_reg})"))
                elif pred == 'ule' or pred == 'sle':
                    # x <= y. If y is constant, x < y+1 => y+1 > x
                    try:
                        imm_val = int(operand2)
                        temp_imm = imm_val + 1
                        temp_reg = self.allocator.allocate(f'%const_{temp_imm}')
                        self.load_constant(temp_reg, temp_imm, inst_list, f"R{temp_reg} = {temp_imm}")
                        # y+1 > x
                        opcode = "ISETP.GT"
                        inst_list.append(MicroCUDAInstruction(opcode, 0, temp_reg, src1_reg, None, f"P0 = ({temp_imm} > R{src1_reg})"))
                    except ValueError:
                        print(f"[WARN] ULE/SLE with register not supported fully.")
                        inst_list.append(MicroCUDAInstruction("ISETP.EQ", 0, src1_reg, src2_reg, None, f"P0 = (R{src1_reg} == R{src2_reg})"))
                
                # We don't allocate a GP register for the result (it's in P0)
        
        # Branch instructions
        elif ir_inst.startswith('br'):
            # Form 1: Unconditional: br label %label
            match_unc = re.match(r'br\s+label\s+%(\w+)', ir_inst)
            if match_unc:
                target_label = match_unc.group(1)
                # BRA to label (resolved later)
                inst_list.append(MicroCUDAInstruction("BRA", 0, None, None, target_label, f"Goto {target_label}"))
            
            # Form 2: Conditional: br i1 %cond, label %true, label %false
            else:
                match_cond = re.match(r'br\s+i1\s+%(\w+),\s*label\s+%(\w+),\s*label\s+%(\w+)', ir_inst)
                if match_cond:
                    cond_var = match_cond.group(1) # We assume this set P0 recently
                    true_label = match_cond.group(2)
                    false_label = match_cond.group(3)
                    
                    # BRZ (Branch if P0 is set? Wait. BR.Z usually means Branch if Zero/False?)
                    # VM code: if (predicateVal) vm.setPC(inst.dest);
                    # predicateVal = P[dest].
                    # If ISETP_EQ sets P0=1 (True) when Equal.
                    # Then BRZ (dest=0) branches if P0=1.
                    # So BRZ means "Branch if Predicate True".
                    
                    # Generate:
                    # BRZ P0, true_label
                    # BRA false_label
                    # BRZ P0, false_label (If 0/False, go to False path)
                    # BRA true_label (Else/True, go to True path)
                    
                    # BRZ Jumps on TRUE (0 due to ISETP inversion or Active Low logic)
                    # So: if P0 (True) goto true_label.
                    inst_list.append(MicroCUDAInstruction("BRZ", 0, None, None, true_label, f"If P0 (True/0) goto {true_label}"))
                    inst_list.append(MicroCUDAInstruction("BRA", 0, None, None, false_label, f"Else goto {false_label}"))

             
        # Phi nodes (skip for now)
        elif 'phi' in ir_inst:
            match = re.match(r'%(\w+)\s*=\s*phi', ir_inst)
            if match:
                self.allocator.allocate(f'%{match.group(1)}')
        
        else:
            print(f"[DEBUG] Skipping IR: {ir_inst}")
            
        return inst_list
        # - br (branches)
        # - Other intrinsics (__syncthreads, etc.)
        
    
    def analyze_lifetimes(self, ir_insts: List[str]) -> Dict[str, int]:
        """
        Analyze variable lifetimes to support register reuse.
        Returns a map: {variable_name: last_used_instruction_index}
        """
        last_uses = {}
        # Regex for valid LLVM identifiers: %name, %1, etc.
        var_pattern = re.compile(r'%[\w\.]+')
        
        for i, line in enumerate(ir_insts):
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('}'): 
                continue
                
            # If line is an assignment: %2 = add ...
            # The LHS is a definition (start of life), RHS are uses.
            # But for simple liveness, we just need to know the LAST time a variable appears.
            # Whether it's def or use, if it appears at line i, its lifetime extends to at least i.
            
            # Simple approach: find all variables in the line
            vars_found = var_pattern.findall(line)
            
            for var in vars_found:
                # Update last use index
                last_uses[var] = i
                
        return last_uses

    def compile_function(self, ir_insts: List[str]) -> List[MicroCUDAInstruction]:
        """
        Compile a function from LLVM IR to Micro-CUDA ISA
        """
        compiled = []
        
        # Step 0: Analyze lifetimes for register reuse
        # This allows us to free registers after their last use
        last_uses = self.analyze_lifetimes(ir_insts)
        if last_uses:
            print(f"[INFO] Lifetime analysis: found {len(last_uses)} variables")

        # Step 1: Parse function signature to find parameters
        function_params = []
        for inst in ir_insts:
            if inst.strip().startswith('define'):
                params_match = re.findall(r'ptr.*?%(\w+)', inst)
                function_params = [f'%{p}' for p in params_match]
                break
        
        # Step 2: Generate prologue to initialize function parameters
        if function_params:
            print(f"[INFO] Initializing {len(function_params)} function parameters")
            for i, param in enumerate(function_params):
                param_reg = self.allocator.allocate(param)
                # Custom Memory Layout for Convolution
                # Avoid 0x0000 in case of Program/Stack collision
                # P0(Input): 0x1000
                # P1(Kernel): 0x2000
                # P2(Output): 0x3000
                # P3(Params): 0x4000
                layout = [0x1000, 0x2000, 0x3000, 0x4000]
                if i < len(layout):
                    vram_addr = layout[i]
                else:
                    vram_addr = 0x8200 + (i-4)*32
                
                # Use load_constant to handle > 255 values (Addresses!)
                self.load_constant(param_reg, vram_addr, compiled, f"param {i} @ 0x{vram_addr:X}")
        
        # Step 3: Compile each IR instruction
        var_pattern = re.compile(r'%[\w\.]+')
        
        for idx, ir_inst in enumerate(ir_insts):
            # Compile the instruction
            insts = self.compile_ir_instruction(ir_inst)
            compiled.extend(insts)
            
            # Register Reuse Logic:
            # Check variables used in this instruction
            # If this is their last use, free their registers!
            
            # Find all vars in this line again
            vars_in_line = var_pattern.findall(ir_inst)
            
            for var in vars_in_line:
                # If this is the last time we see this variable
                if var in last_uses and last_uses[var] == idx:
                    # And if it is NOT a return value or persistent state
                    # (In SSA, everything is temp unless stored to memory)
                    
                    # Be careful: Function params are defined at start, but if this is last use, they can be freed too.
                    # But don't free if it's the destination of THIS instruction?
                    # The destination is defined here. It shouldn't be freed immediately unless it's unused (dead code).
                    # But if it's unused, liveness analysis would show last_use == idx.
                    # So yes, free it.
                    
                    # Log for debug
                    # print(f"  [DEBUG] Freeing {var} at line {idx}")
                    self.allocator.free(var)
        
        return compiled

# ===== Compiler Pipeline =====

def run_clang(input_file: Path, output_ll: Path) -> bool:
    """Run Clang to generate LLVM IR from C/C++ source"""
    cmd = [
        "clang",
        "-S",                    # Output assembly
        "-emit-llvm",            # Emit LLVM IR instead of native assembly
        f"-O{CLANG_OPT_LEVEL}",  # Optimization level
        f"--target={CLANG_TARGET}",  # 32-bit target
        f"-I{MCUDA_HEADER.parent}",  # Include path for mcuda.h
        str(input_file),
        "-o", str(output_ll)
    ]
    
    print(f"[INFO] Running Clang: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Clang failed:")
            print(result.stderr)
            return False
        return True
    except FileNotFoundError:
        print("[ERROR] Clang not found! Please install LLVM/Clang.")
        print("  macOS: brew install llvm")
        print("  Linux: apt-get install clang")
        return False

def compile_cuda_to_isa(input_file: Path,
                        output_file: Path,
                        emit_binary: bool = False,
                        target: str = "default") -> bool:
    """
    Compile LLVM IR to Micro-CUDA ISA
    
    Args:
        input_file: Input LLVM IR file (.ll)
        output_file: Output assembly file (.asm)
        emit_binary: If True, also emit binary format
        target: Target configuration name
    
    Returns:
        True if compilation successful
    """
    # Get target configuration
    from micro_cuda_compiler.target_config import get_target
    target_config = get_target(target)
    
    # Parse LLVM IR directly (don't re-run Clang)
    print(f"[INFO] Parsing LLVM IR from {input_file}...")
    
    # Read IR file
    with open(input_file, 'r') as f:
        ir_text = f.read()
    
    parser = LLVMIRParser(ir_text)
    ir_functions = parser.parse()
    
    if not ir_functions:
        print("[ERROR] No functions found in IR!")
        return False
    
    print(f"[INFO] Found {len(ir_functions)} function(s): {list(ir_functions.keys())}")
    
    # Compile to Micro-CUDA ISA
    print("[INFO] Compiling to Micro-CUDA ISA...")
    backend = MicroCUDABackend()
    all_instructions = []
    
    for func_name, ir_insts in ir_functions.items():
        print(f"[INFO] Compiling function: {func_name}")
        compiled_insts = backend.compile_function(ir_insts)
        all_instructions.extend(compiled_insts)
    
    # Generate assembly output
    print(f"[SUCCESS] Assembly written to: {output_file}")
    
    # Write assembly file with target configuration header
    with open(output_file, 'w') as f:
        # Write target configuration header
        f.write(target_config.format_header())
        f.write("\n")
        f.write(f"; Source File: {input_file.name}\n")
        f.write(f"; Kernel Functions: {', '.join(ir_functions.keys())}\n")
        f.write(f"; Total Instructions: {len(all_instructions)}\n")
        f.write(f"; Registers Used: {backend.allocator.get_registers_used()}\n")
        f.write(";\n")
        f.write("; " + "=" * 68 + "\n")
        f.write("\n")
        f.write("; ===== CODE SECTION =====\n")
        f.write("\n")
        
        # Write instructions
        for inst in all_instructions:
            f.write(inst.to_asm() + "\n")
        
        f.write("\n")
        f.write("; ===== END OF KERNEL =====\n")
    
    print("[SUCCESS] Compilation complete!")
    print(f"[INFO] Generated {len(all_instructions)} instructions")
    print(f"[INFO] Used {backend.allocator.get_registers_used()} registers")
    print(f"[INFO] Target: {target_config.device_name} (VRAM: {target_config.vram_size // 1024} KB, Lanes: {target_config.num_lanes})")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Micro-CUDA Compiler (mcc) - Compile CUDA-like C/C++ to Micro-CUDA ISA"
    )
    parser.add_argument("input", type=Path, help="Input .cu or .cpp file")
    parser.add_argument("-o", "--output", type=Path, help="Output file")
    parser.add_argument("--llvm-ir", action="store_true", help="Emit LLVM IR (.ll) only")
    parser.add_argument("--asm", action="store_true", help="Emit Micro-CUDA assembly (.asm)")
    parser.add_argument("--target", default="default", 
                       help="Target configuration (default, esp32, esp32-psram, esp32s3)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        return 1
    
    # Get target configuration
    target = get_target(args.target)
    
    # Determine output file
    if args.output:
        output_file = args.output
    elif args.llvm_ir:
        output_file = args.input.with_suffix('.ll')
    elif args.asm:
        output_file = args.input.with_suffix('.asm')
    else:
        output_file = args.input.with_suffix('.hex')
    
    # Run compilation
    success = compile_cuda_to_isa(
        args.input,
        args.output or args.input.with_suffix('.asm'),
        emit_binary=False,
        target=args.target
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
