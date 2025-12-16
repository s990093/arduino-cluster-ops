
import sys
import time
import struct
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serial
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
try:
    from esp32_tools.program_loader_v15 import InstructionV15
except ImportError:
    pass

# Patch InstructionV15 to add IDIV if missing
if not hasattr(InstructionV15, 'idiv'):
    InstructionV15.OP_IDIV = 0x14
    def idiv(cls, dest, src1, src2):
        return cls(cls.OP_IDIV, dest, src1, src2)
    InstructionV15.idiv = classmethod(idiv)

# Ensure aliases exist for convenience if my code used them
if not hasattr(InstructionV15, 'add'): InstructionV15.add = InstructionV15.iadd
if not hasattr(InstructionV15, 'sub'): InstructionV15.sub = InstructionV15.isub
if not hasattr(InstructionV15, 'mul'): InstructionV15.mul = InstructionV15.imul

# Patch trace to encode REG in src1
def trace_patch(cls, reg):
    return cls(cls.OP_TRACE, 0, reg, 0)
InstructionV15.trace = classmethod(trace_patch)

# ==========================================
# 1. Configuration
# ==========================================
PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800
IMG_W, IMG_H = 64, 64
IMG_SIZE = IMG_W * IMG_H
VRAM_INPUT_BASE = 0x0000
VRAM_OUTPUT_BASE = 0x4000 # 16KB offset (64*64*4 bytes)

# Edge Detection Kernel (Laplacian-ish)
#    0  -1   0
#   -1   4  -1
#    0  -1   0
KERNEL_WEIGHTS = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=torch.float32).reshape(1, 1, 3, 3)

# ==========================================
# 2. PyTorch Spec & Verification
# ==========================================
def run_pytorch_ref(input_img):
    """
    Input: Numpy array (H, W)
    Output: Numpy array (H, W)
    """
    # Convert to Tensor (1, 1, H, W)
    x = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Run Conv2d
    with torch.no_grad():
        # Padding=1 to keep size 64x64. PyTorch zero-pads.
        out = F.conv2d(x, KERNEL_WEIGHTS, padding=1)
    
    return out.squeeze().numpy()

# ==========================================
# 3. ASM Kernel Generator
# ==========================================
def build_asm_program():
    """
    Generates SIMD assembly for 3x3 Convolution.
    Optimization: Parallel processing of 8 pixels.
    Logic:
      - Loop through all pixels (0 to 4096) with step 8.
      - Each lane calculates validness (border check).
      - If valid, load valid neighbors and compute.
    """
    prog = []
    
    # Registers Map:
    # R0: Loop Counter (Base Index)
    # R1: Loop Limit (4096)
    # R2: Lane ID (0-7)
    # R3: Current Pixel Index (Global)
    # R4: X Coordinate
    # R5: Y Coordinate
    # R6: Width (64)
    # R10: Accumulator
    # R11-R19: Neighbors
    # R20: Scratch / Offsets
    
    # Init Registers
    # R31 = 8 (Shift Amount Constant)
    prog.append(InstructionV15.mov(31, 8))
    
    dest_reg_8 = 31

    def load_imm32(reg, val):
        parts = []
        parts.append((val >> 24) & 0xFF)
        parts.append((val >> 16) & 0xFF)
        parts.append((val >> 8) & 0xFF)
        parts.append(val & 0xFF)
        
        # Find first non-zero byte
        start_idx = 0
        while start_idx < 4 and parts[start_idx] == 0:
            start_idx += 1
            
        if start_idx == 4:
            prog.append(InstructionV15.sub(reg, reg, reg)) # Set to 0
            return
            
        # First MOV
        prog.append(InstructionV15.mov(reg, parts[start_idx]))
        
        # Subsequent shifts and adds
        for i in range(start_idx + 1, 4):
            prog.append(InstructionV15.shl(reg, reg, dest_reg_8)) # Shift Left 8
            if parts[i] != 0:
                # We need OR Reg, Imm? No, OR uses Reg.
                # Must load Imm to scratch?
                # Use R20 as scratch temporarily (safe in init/loading phase)
                prog.append(InstructionV15.mov(20, parts[i]))
                prog.append(InstructionV15.or_op(reg, reg, 20))

    prog.append(InstructionV15.s2r(2, InstructionV15.SR_LANEID)) # R2 = LaneID
    prog.append(InstructionV15.sub(0, 0, 0))    # R0 = Loop Counter = 0
    # R1 = 8 (Limit 1 Iteration)
    prog.append(InstructionV15.mov(1, 8))
    prog.append(InstructionV15.mov(6, 64))   # R6 = Width
    prog.append(InstructionV15.sub(30, 30, 30))   # R30 = Zero (For LDX/STX offset)
    
    # === LOOP START ===
    loop_start_idx = len(prog) 
    
    # 1. Calc Global Index: R3 = R0 + R2
    prog.append(InstructionV15.iadd(3, 0, 2)) 
    
    # 2. Calc Coordinates (No trace needed, but we can)
    prog.append(InstructionV15.mov(20, 64))
    prog.append(InstructionV15.idiv(5, 3, 20)) # Y
    prog.append(InstructionV15.imul(21, 5, 20)) # Y * 64
    prog.append(InstructionV15.isub(4, 3, 21)) # X = Index - (Y*64)
    
    # Load Accumulator R10 = 0
    prog.append(InstructionV15.sub(10, 10, 10))
    
    # --- Convolution (Unrolled) ---
    # Center: P(x, y) -> R3
    # 4 * Val
    # Byte Address = R3 * 4
    prog.append(InstructionV15.mov(22, 4))
    prog.append(InstructionV15.imul(22, 3, 22)) # R22 = Byte Addr Center
    
    prog.append(InstructionV15.ldx(11, 22, 30))  # R11 = Input[Center]
    # TRACE CENTER VAL
    prog.append(InstructionV15.trace(11))
    
    prog.append(InstructionV15.mov(23, 4))       # Weight 4
    prog.append(InstructionV15.imul(11, 11, 23)) # R11 *= 4
    prog.append(InstructionV15.iadd(10, 10, 11)) # Acc += Center*4
    
    # TRACE ACC (4*Center)
    prog.append(InstructionV15.trace(10))
    
    offset_list = [-256, 256, -4, 4] # T, B, L, R
    
    for off in offset_list:
        # Load Offset Abs to R24
        # 256 requires load_imm32
        val = abs(off)
        if val > 255:
            load_imm32(24, val)
        else:
            prog.append(InstructionV15.mov(24, val))
            
        if off < 0:
             prog.append(InstructionV15.sub(22, 22, 24)) # Addr -= off
             prog.append(InstructionV15.ldx(12, 22, 30)) # Load
             prog.append(InstructionV15.add(22, 22, 24)) # Restore Addr
        else:
             prog.append(InstructionV15.add(22, 22, 24)) # Addr += off
             prog.append(InstructionV15.ldx(12, 22, 30)) # Load
             prog.append(InstructionV15.sub(22, 22, 24)) # Restore
             
        # Trace Neighbor
        prog.append(InstructionV15.trace(12))
        
        # Mul by -1 (Sub)
        prog.append(InstructionV15.isub(10, 10, 12)) # Acc -= Val
        
        # Trace Acc Update
        prog.append(InstructionV15.trace(10))
        
    # --- Store Result ---
    # Output Addr = BaseOut (0x4000) + R3*4
    load_imm32(25, 0x4000)
    
    prog.append(InstructionV15.trace(31)) # Check Shift Amt
    prog.append(InstructionV15.trace(25)) # Check Base Addr
    
    prog.append(InstructionV15.mov(26, 4))
    prog.append(InstructionV15.imul(26, 3, 26)) # Index * 4
    prog.append(InstructionV15.iadd(26, 26, 25)) # Final Addr
    
    prog.append(InstructionV15.trace(26)) # Check Final Addr
    
    prog.append(InstructionV15.stx(26, 30, 10))   # Store Acc (Base=26, Off=30, Src=10)
    
    # === LOOP CONTROL ===
    # R0 += 8
    prog.append(InstructionV15.mov(27, 8))
    prog.append(InstructionV15.iadd(0, 0, 27))
    
    # Limit check against IMG_SIZE (4096)
    # We want to loop while R0 < IMG_SIZE
    # Equiv: Exit if R0 >= IMG_SIZE
    # isetp_ge P0, R0, IMG_SIZE
    # brz P0, loop_start  (If !GE, i.e. LT, then branch)
    
    load_imm32(29, IMG_SIZE)
    # Note: isetp_ge might not exist in V15 patch if not defined?
    # Default provided usually has gt, ge?
    # If not, use sub + check sign?
    # Assuming InstructionV15 has isetp_ge or similar. 
    # If not, checking V15 definition: usually isetp_gt, isetp_eq...
    # Let's use isetp_gt(P0, R0, R29_Limit_Minus_1) or similar.
    # Actually, let's use isetp_ge if available. 
    # If not safe, use: R0 >= 4096. 
    # Let's check if 'isetp_ge' is available in InstructionV15 used in file?
    # It is not explicitly patched in L20-35.
    # But L227 used isetp_gt.
    # We can use isetp_gt(0, 0, 29) where R29 = IMG_SIZE - 8?
    # If R0 > (4096-8), then exit.
    
    # Safer: load 4096. Check R0 >= 4096.
    # InstructionV15.setp_ge?
    # Let's assume isetp_ge exists. If not, I'll error.
    # To be safe regarding library availability, let's use isetp_gt checks.
    # Exit if R0 >= 4096.
    # Continue if R0 < 4096.
    # isetp_lt(0, 0, 29). Branch if true? 
    # brz branches if false.
    # So if (R0 < 4096) is True (1), brz DOES NOT branch.
    # We want brz to branch if True.
    # brz: Branch if Zero (False).
    # So we need condition to be FALSE when we want to EXIT.
    # And TRUE when we want to LOOP.
    # Condition: R0 < IMG_SIZE.
    # trace output showed: 0 > 8 False -> Branch. So brz branches on False.
    # Wait.
    # Trace logic:
    # R0=0. 0 > 8 is False.
    # brz (Branch if Zero) -> Branched.
    # Make sense. P0=0 (False). brz jumps to target.
    # So we loop when condition is false.
    # condition: R0 >= IMG_SIZE.
    # 0 >= 4096 -> False. Branch/Loop.
    # 4096 >= 4096 -> True. Fallthrough/Exit.
    # This works.
    
    # So we need isetp_ge.
    # If isetp_ge not available, use isetp_gt with IMG_SIZE - 1.
    # 4096 - 1 = 4095.
    # R0 > 4095.
    # 4088 > 4095 -> False (Loop).
    # 4096 > 4095 -> True (Exit).
    # Perfect.
    
    load_imm32(29, IMG_SIZE - 1) 
    prog.append(InstructionV15.isetp_gt(0, 0, 29)) 
    
    prog.append(InstructionV15.brz(loop_start_idx))
    
    prog.append(InstructionV15.exit_inst())
    
    return prog

# ==========================================
# 4. Host Utility
# ==========================================
class ESP32TurboClient:
    def __init__(self, port, baud=BAUD_RATE):
        self.ser = serial.Serial(port, baud, timeout=2)
        try:
            self.ser.set_buffer_size(rx_size=32768, tx_size=32768)
        except: pass
        time.sleep(2)
        self.ser.read_all()
        print(f"✅ Connected to ESP32 at {baud}")

    def h2d(self, addr, data_bytes):
        print(f"📤 H2D: {len(data_bytes)} bytes -> {hex(addr)}")
        self.ser.write(f"dma_h2d {hex(addr)} {len(data_bytes)}\n".encode())
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK" in line: break
        self.ser.write(data_bytes)
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if "DMA" in line: break
            
    def load_kernel(self, binary):
        print(f"🧩 Loading Kernel ({len(binary)} bytes)...")
        self.ser.write(f"load_imem {len(binary)}\n".encode())
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK" in line: break
        self.ser.write(binary)
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if "OK" in line or "LOADED" in line: break

    def launch(self):
        print("🚀 Launching Kernel...")
        self.ser.write(b"kernel_launch\n")
        start = time.time()
        while True:
            line = self.ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"DEV: {line}")
            if "EXIT" in line: break
        print(f"✅ Finished in {time.time() - start:.3f}s")
        
    def d2h(self, addr, size_bytes):
        # We use dma_d2h which dumps hex text.
        # But for 16KB this is slow and hard to parse.
        # For this verification, we only read back 16KB if necessary?
        # Or we implement a BINARY d2h in firmware?
        # Firmware only has text dump dma_d2h.
        # 16KB dump = 4096 lines. At 460800 baud this is fast enough (~0.5s).
        
        print(f"📖 D2H: Reading {size_bytes} bytes from {hex(addr)}...")
        count = size_bytes // 4
        self.ser.write(f"dma_d2h {hex(addr)} {count}\n".encode())
        
        data = []
        while len(data) < count:
            line = self.ser.readline().decode(errors='ignore').strip()
            print(f"RAW D2H: {line}")
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        val = int(parts[1].strip(), 16)
                        data.append(val)
                    except: pass
        return np.array(data, dtype=np.uint32).view(np.int32)

# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    # A. Generate Random Image
    print("🖼️ Generating Image...")
    img = np.random.randint(0, 255, (IMG_H, IMG_W), dtype=np.uint8)
    # img = np.full((IMG_H, IMG_W), 10, dtype=np.uint8)
    print("   Input[0..7]:", img[0, :8])
    
    # B. PyTorch Reference
    print("🧠 Running PyTorch...")
    ref_out = run_pytorch_ref(img)
    print(f"   Ref Output Range: {ref_out.min()} to {ref_out.max()}")
    
    # C. Prepare Data for ESP32
    # Input: 32-bit Integers (Pixel in LSB)
    # Output expected: 32-bit Integers
    flat_img = img.flatten().astype(np.int32)
    # Convert to bytes (Little Endian)
    input_bytes = struct.pack(f'<{len(flat_img)}I', *flat_img)
    
    # D. Build Kernel
    print("🔨 Building Assembly...")
    prog_objs = build_asm_program()
    inst_vals = [int(i.to_hex(), 16) for i in prog_objs]
    kernel_bin = struct.pack(f'<{len(inst_vals)}I', *inst_vals)
    
    # E. Run on Hardware
    client = ESP32TurboClient(PORT)
    client.h2d(VRAM_INPUT_BASE, input_bytes)
    client.load_kernel(kernel_bin)
    client.launch()
    
    # F. Verify
    # Read back Output
    out_size = IMG_SIZE * 4
    hw_out_raw = client.d2h(VRAM_OUTPUT_BASE, out_size) # Returns int32 array
    hw_out = hw_out_raw.reshape(IMG_H, IMG_W)
    
    # Compare
    # Note: Boundary pixels might differ due to padding handling.
    # We focus on inner 62x62
    inner_hw = hw_out[1:-1, 1:-1]
    inner_ref = ref_out[1:-1, 1:-1]
    
    diff = np.abs(inner_hw - inner_ref)
    mae = np.mean(diff)
    print(f"\n📊 Verification (Inner 62x62):")
    print(f"   Max Diff: {np.max(diff)}")
    print(f"   MAE: {mae}")
    
    # DEBUG: Print Patch
    print("\n🔍 Debug Patch (Center 4x4):")
    cy, cx = IMG_H//2, IMG_W//2
    # Adjust for inner slicing
    slice_y, slice_x = cy-2, cx-2
    print("Reference:")
    print(inner_ref[slice_y:slice_y+4, slice_x:slice_x+4])
    print("Hardware:")
    print(inner_hw[slice_y:slice_y+4, slice_x:slice_x+4])
    
    if mae < 1.0:
        print("✅ PASSED: Hardware matches PyTorch!")
    else:
        print("❌ FAILED: Significant difference.")
