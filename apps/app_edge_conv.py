
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
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("⚠️  lz4 not installed. Falling back to standard upload.")

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


# ==========================================
# 1. Configuration
# ==========================================
# Edge Detection Kernel (Laplacian-ish)
#    0  -1   0
#   -1   4  -1
#    0  -1   0
KERNEL_WEIGHTS = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=torch.float32).reshape(1, 1, 3, 3)

PORT = "/dev/cu.usbserial-2130"
BAUD_RATE = 460800  # Stable speed (921600 unstable)
IMG_W, IMG_H = 1024, 1024  # Testing 1K resolution
IMG_SIZE = IMG_W * IMG_H
VRAM_INPUT_BASE = 0x0000
# Input 130*34*4 = 17680 bytes. Output starts at higher address.
VRAM_OUTPUT_BASE = 0x8000 
TILE_W = 128
TILE_H = 32 # Reduced for Halo support (VRAM limit)
def run_pytorch_ref(input_img):
    """
    Input: Numpy array (H, W, 3)
    Output: Numpy array (H, W, 3)
    """
    # Create output container
    out_rgb = np.zeros_like(input_img, dtype=np.float32)
    
    # Process each channel independently
    for c in range(3):
        # (H, W) -> (1, 1, H, W)
        x = torch.tensor(input_img[:, :, c], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = F.conv2d(x, KERNEL_WEIGHTS, padding=1)
        out_rgb[:, :, c] = out.squeeze().numpy()
        
    return out_rgb

# ==========================================
# 3. ASM Kernel Generator
# ==========================================

# ==========================================
# 3. ASM Kernel Generator
# ==========================================
def build_asm_program(tile_w, tile_h):
    """
    Generates SIMD assembly for 3x3 Convolution.
    Optimization: Parallel processing of 8 pixels.
    Logic:
      - Loop through all pixels (0 to tile_w*tile_h) with step 8.
      - Each lane calculates validness (border check).
      - If valid, load valid neighbors and compute.
    """
    prog = []
    
    loop_limit = tile_w * tile_h
    width = tile_w
    out_base = VRAM_OUTPUT_BASE
    
    # Registers Map:
    # R0: Loop Counter (Base Index)
    # R1: Loop Limit
    # R2: Lane ID (0-7)
    # R3: Current Pixel Index (Global)
    # R4: X Coordinate
    # R5: Y Coordinate
    # R6: Width
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
                prog.append(InstructionV15.mov(20, parts[i]))
                prog.append(InstructionV15.or_op(reg, reg, 20))

    prog.append(InstructionV15.s2r(2, InstructionV15.SR_LANEID)) # R2 = LaneID
    prog.append(InstructionV15.sub(0, 0, 0))    # R0 = Loop Counter = 0
    # R1 = Limit
    load_imm32(1, loop_limit)
    
    # R6 = Width
    if width > 255:
        load_imm32(6, width)
    else:
        prog.append(InstructionV15.mov(6, width))
        
    prog.append(InstructionV15.sub(30, 30, 30))   # R30 = Zero (For LDX/STX offset)
    
    # === LOOP START ===
    loop_start_idx = len(prog) 
    
    # 1. Calc Global Index: R3 = R0 + R2
    prog.append(InstructionV15.iadd(3, 0, 2)) 
    
    # 2. Calc Coordinates
    # Y = Index / Width
    # X = Index % Width = Index - (Y * Width)
    
    # Load width into R20 for DIV/MUL (if width fits in imm8, use it, else use R6)
    # But IDIV destination cannot be src.
    # Safe to use R6 as width source. 
    prog.append(InstructionV15.idiv(5, 3, 6)) # Y = R3 / R6
    prog.append(InstructionV15.imul(21, 5, 6)) # Y * Width
    prog.append(InstructionV15.isub(4, 3, 21)) # X = Index - (Y*Width)
    
    # Load Accumulator R10 = 0
    prog.append(InstructionV15.sub(10, 10, 10))
    
    # --- Convolution (Unrolled) ---
    # Center: P(x, y) -> R3
    # 4 * Val
    # Byte Address = R3 * 4
    prog.append(InstructionV15.mov(22, 4))
    prog.append(InstructionV15.imul(22, 3, 22)) # R22 = Byte Addr Center
    prog.append(InstructionV15.ldx(11, 22, 30))  # R11 = Input[Center]
    prog.append(InstructionV15.mov(23, 4))       # Weight 4
    prog.append(InstructionV15.imul(11, 11, 23)) # R11 *= 4
    prog.append(InstructionV15.iadd(10, 10, 11)) # Acc += Center*4
    
    # Calculate Offsets: 
    # Top/Bottom: +/- Width * 4 (bytes)
    # Left/Right: +/- 4 (bytes)
    stride_byte = width * 4
    offset_list = [-stride_byte, stride_byte, -4, 4] # T, B, L, R
    
    for off in offset_list:
        # Load Offset Abs to R24
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
             
        # Mul by -1 (Sub)
        prog.append(InstructionV15.isub(10, 10, 12)) # Acc -= Val
        
    # --- Store Result ---
    # Output Addr = BaseOut (0x8000) + R3*4
    load_imm32(25, out_base)
    
    prog.append(InstructionV15.mov(26, 4))
    prog.append(InstructionV15.imul(26, 3, 26)) # Index * 4
    prog.append(InstructionV15.iadd(26, 26, 25)) # Final Addr
    
    prog.append(InstructionV15.stx(26, 30, 10))   # Store Acc (Base=26, Off=30, Src=10)
    
    # === LOOP CONTROL ===
    # R0 += 8
    prog.append(InstructionV15.mov(27, 8))
    prog.append(InstructionV15.iadd(0, 0, 27))
    
    # Loop Limit - 1
    load_imm32(29, loop_limit - 1) 
    prog.append(InstructionV15.isetp_gt(0, 0, 29)) # P0 = (R0 > Limit-1) => True if Done.
    
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
        if HAS_LZ4:
            return self.h2d_lz4(addr, data_bytes)
        else:
            self.h2d_standard(addr, data_bytes)
            return len(data_bytes), 100  # No compression
    
    def h2d_lz4(self, addr, data_bytes):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        total_size = len(data_bytes)
        
        # Chunk and Compress
        CHUNK_SIZE = 2048  # Match firmware
        offset = 0
        total_compressed = 0
        
        # Send command
        cmd = f"dma_h2d_lz4 {hex(addr)} {total_size}\n"
        # print(f"   [TX] {cmd.strip()}")  # DEBUG
        self.ser.write(cmd.encode())
        self.ser.flush()
        
        # Wait for ACK with timeout
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for ACK_LZ4_GO")
            line = self.ser.readline().decode(errors='ignore').strip()
            # Removed debug logging for speed
            if "ACK_LZ4_GO" in line: 
                break
            if line: continue
            time.sleep(0.001)
        
        # Process chunks
        while offset < total_size:
            chunk = data_bytes[offset : offset + CHUNK_SIZE]
            compressed = lz4.block.compress(chunk, store_size=False)
            total_compressed += len(compressed)
            
            # Send: [2B Len] + [Compressed Data]
            header = struct.pack('<H', len(compressed))
            self.ser.write(header + compressed)
            
            offset += len(chunk)
            time.sleep(0.0005)  # 0.5ms - minimal delay for flow control
        
        # Wait for OK
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for LZ4_LOAD_OK")
            line = self.ser.readline().decode(errors='ignore').strip()
            # Removed debug logging for speed
            if "LZ4_LOAD_OK" in line: 
                break
            if line: continue
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()
        
        # Calculate and return stats
        ratio = (total_compressed / total_size) * 100 if total_size > 0 else 100
        return total_compressed, ratio
    
    def h2d_standard(self, addr, data_bytes):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        print(f"📤 H2D: {len(data_bytes)} bytes -> {hex(addr)}")
        self.ser.write(f"dma_h2d {hex(addr)} {len(data_bytes)}\n".encode())
        self.ser.flush()
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for ACK")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK" in line: break
            if line: continue
            time.sleep(0.001)
            
        self.ser.write(data_bytes)
        self.ser.flush()
        print(f"   Waiting for DMA...")
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for DMA")
            line = self.ser.readline().decode(errors='ignore').strip()
            if line: print(f"   [RX] {line}")
            if "DMA" in line: break
            if line: continue
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()
            
    def load_kernel(self, binary):
        if HAS_LZ4:
            self.load_kernel_lz4(binary)
        else:
            self.load_kernel_standard(binary)

    def load_kernel_lz4(self, binary):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        total_size = len(binary)
        print(f"🧩 Loading Kernel (LZ4 Mode, {total_size} bytes)...")
        self.ser.write(f"load_imem_lz4 {total_size}\n".encode())
        self.ser.flush()
        
        # Wait for ACK
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for ACK_LZ4_GO")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK_LZ4_GO" in line: break
            if line: print(f"   [RX] {line}")
            time.sleep(0.001)

        # Chunk and Compress
        CHUNK_SIZE = 2048  # Match firmware
        offset = 0
        total_compressed = 0
        
        while offset < total_size:
            chunk = binary[offset : offset + CHUNK_SIZE]
            compressed = lz4.block.compress(chunk, store_size=False)
            total_compressed += len(compressed)
            
            # Send: [2B Len] + [Compressed Data]
            header = struct.pack('<H', len(compressed))
            self.ser.write(header + compressed)
            
            offset += len(chunk)
            time.sleep(0.0005)  # 0.5ms - minimal delay

        # Wait for OK
        print(f"   Waiting for LZ4_LOAD_OK...")
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for LZ4_LOAD_OK")
            line = self.ser.readline().decode(errors='ignore').strip()
            if line: print(f"   [RX] {line}")
            if "LZ4_LOAD_OK" in line: break
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()
        
        # Report compression stats
        ratio = (total_compressed / total_size) * 100 if total_size > 0 else 100
        print(f"   📦 Compression: {total_size} -> {total_compressed} bytes ({ratio:.1f}%)")

    def load_kernel_standard(self, binary):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        print(f"🧩 Loading Kernel (Standard, {len(binary)} bytes)...")
        self.ser.write(f"load_imem {len(binary)}\n".encode())
        self.ser.flush()
        print(f"   Waiting for ACK/LOADED...")
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for ACK")
            line = self.ser.readline().decode(errors='ignore').strip()
            if line: print(f"   [RX] {line}")
            if "ACK" in line: break
            time.sleep(0.001)
        
        print(f"   Sending Binary ({len(binary)} bytes)...")
        self.ser.write(binary)
        self.ser.flush()
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for OK/LOADED")
            line = self.ser.readline().decode(errors='ignore').strip()
            if line: print(f"   [RX] {line}")
            if "OK" in line or "LOADED" in line: break
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()

    def launch(self):
        self.ser.reset_input_buffer()
        print("🚀 Launching Kernel...")
        self.ser.write(b"kernel_launch\n")
        self.ser.flush()
        
        start = time.time()
        timeout_start = time.time()
        while True:
            if time.time() - timeout_start > 30:
                self.ser.reset_input_buffer()
                raise TimeoutError("Kernel launch timeout")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "EXIT" in line: break
            if line: continue
            time.sleep(0.001)
        
        exec_time = time.time() - start
        self.ser.reset_input_buffer()
        print(f"✅ Finished in {exec_time:.3f}s")
        
    def d2h(self, addr, size_bytes):
        self.ser.reset_input_buffer()
        
        # Optimized Binary D2H
        # print(f"📖 D2H: Reading {size_bytes} bytes from {hex(addr)}...") # Optional logging (can be verbose)
        count = size_bytes // 4
        self.ser.write(f"dma_d2h_binary {hex(addr)} {count}\n".encode())
        self.ser.flush()
        
        # 1. Wait for Header ACK_D2H_BIN:<bytes>
        actual_bytes = 0
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for ACK_D2H_BIN")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK_D2H_BIN" in line:
                try:
                    actual_bytes = int(line.split(":")[1])
                except:
                    print(f"❌ Error parsing D2H Header: {line}")
                    self.ser.reset_input_buffer()
                    return np.zeros(count, dtype=np.int32)
                break
            if line: continue
            time.sleep(0.001)
        
        # 2. Read Binary Data
        data_bytes = self.ser.read(actual_bytes)
        
        # 3. Wait for Footer (Optional synchronization)
        # Often it comes immediately, but buffering might delay it.
        # Just to be safe and clear the buffer for next command.
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for D2H_OK")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "D2H_OK" in line: break
            if line: continue
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()
        
        return np.frombuffer(data_bytes, dtype=np.int32)

# ==========================================
# 5. Main
# ==========================================
if __name__ == "__main__":
    # A. Load and Resize Image
    # IMG_W, IMG_H defined globally
    img_path = str(Path(__file__).parent / "IMG_6257.JPG")
    print(f"🖼️ Loading Image: {img_path}")
    
    try:
        original_img = Image.open(img_path)
        original_img = original_img.resize((IMG_W, IMG_H))
        rgb_img = original_img.convert('RGB') # RGB
        img = np.array(rgb_img, dtype=np.uint8) 
        print(f"   Loaded and resized to {img.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # B. PyTorch Reference
    print("🧠 Running PyTorch (Full Image)...")
    ref_out = run_pytorch_ref(img)
    print(f"   Ref Output Range: {ref_out.min()} to {ref_out.max()}")
    
    # C. Build Kernel
    # Padded dimensions for Halo (1px on each side)
    PAD_W = TILE_W + 2
    PAD_H = TILE_H + 2
    print(f"🔨 Building Assembly ({PAD_W}x{PAD_H} Physical Tile)...")
    prog_objs = build_asm_program(PAD_W, PAD_H)
    inst_vals = [int(i.to_hex(), 16) for i in prog_objs]
    kernel_bin = struct.pack(f'<{len(inst_vals)}I', *inst_vals)
    
    # D. Connect and Load Kernel
    try:
        client = ESP32TurboClient(PORT)
        client.load_kernel(kernel_bin) # Load once
        
        # E. Process Tiles
        # Output is RGB (3 channels)
        hw_out = np.zeros((IMG_H, IMG_W, 3), dtype=np.int32)
        
        # Calculate Tile Counts
        tiles_y = IMG_H // TILE_H
        tiles_x = IMG_W // TILE_W
        # Total tasks = Tiles * Channels
        total_tasks = tiles_y * tiles_x * 3
        
        print(f"🧩 Split Image into {tiles_x} x {tiles_y} Tiles x 3 Channels = {total_tasks} Tasks")
        print("🚀 Processing Tiles (Halo Enabled)...")
        
        # Pad the entire image first (Zero Padding to match PyTorch)
        # Pad (Y, X, C) -> only Y and X need padding
        img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        
        from tqdm import tqdm
        
        # Track compression stats
        total_original_bytes = 0
        total_compressed_bytes = 0
        
        with tqdm(total=total_tasks, unit="task") as pbar:
            for y in range(0, IMG_H, TILE_H):
                for x in range(0, IMG_W, TILE_W):
                    for c in range(3): # RGB Channels
                        # Extract Padded Tile
                        tile_padded = img_padded[y:y+PAD_H, x:x+PAD_W, c]
                        
                        # Prepare Input
                        flat_tile = tile_padded.flatten().astype(np.int32)
                        input_bytes = struct.pack(f'<{len(flat_tile)}I', *flat_tile)
                        
                        # Send -> Launch -> Read
                        if HAS_LZ4:
                            compressed_size, ratio = client.h2d(VRAM_INPUT_BASE, input_bytes)
                            total_original_bytes += len(input_bytes)
                            total_compressed_bytes += compressed_size
                        else:
                            client.h2d(VRAM_INPUT_BASE, input_bytes)
                        
                        client.launch()
                        
                        # Read Result (Padded Size)
                        out_bytes = PAD_W * PAD_H * 4
                        tile_out_raw = client.d2h(VRAM_OUTPUT_BASE, out_bytes)
                        tile_out_padded = tile_out_raw.reshape(PAD_H, PAD_W)
                        
                        # Crop Center (Remove Halo)
                        tile_out = tile_out_padded[1:-1, 1:-1]
                        
                        # Store
                        hw_out[y:y+TILE_H, x:x+TILE_W, c] = tile_out
                        
                        pbar.update(1)
        
        # Report compression stats
        if HAS_LZ4 and total_original_bytes > 0:
            overall_ratio = (total_compressed_bytes / total_original_bytes) * 100
            print(f"\n📦 Image Compression Stats:")
            print(f"   Original:   {total_original_bytes:,} bytes")
            print(f"   Compressed: {total_compressed_bytes:,} bytes")
            print(f"   Ratio:      {overall_ratio:.1f}%")
            print(f"   Savings:    {total_original_bytes - total_compressed_bytes:,} bytes ({100-overall_ratio:.1f}%)")
                
    except Exception as e:
         print(f"⚠️ Hardware execution failed or skipped: {e}")
         hw_out = np.zeros_like(ref_out) # Placeholder if HW fail
            
    # F. Verify & Visualization
    print("\n📊 Verification:")
    
    diff = np.abs(hw_out - ref_out)
    mae = np.mean(diff)
    print(f"   Max Diff: {np.max(diff)}")
    print(f"   MAE: {mae}")
    
    # Plotting
    print("🎨 Displaying Results...")
    plt.figure(figsize=(15, 5))
    
    # Normalize for display (Edges can be negative, so abs() or shift?)
    # Usually edge detection is visualized as abs value or clamped.
    # Ref output is float, HW output is int32.
    
    def normalize_for_display(arr):
        # Initial idea: ABS and Clip
        return np.clip(np.abs(arr), 0, 255).astype(np.uint8)

    plt.subplot(1, 4, 1)
    plt.title("Input Image (RGB 512x512)")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("PyTorch Reference")
    plt.imshow(normalize_for_display(ref_out))
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title(f"ESP32 SIMD Output\nMAE: {mae:.2f}")
    plt.imshow(normalize_for_display(hw_out))
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Difference Map (Abs)")
    # Boost contrast of diff map
    diff_disp = np.clip(diff * 5, 0, 255).astype(np.uint8)
    plt.imshow(diff_disp)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
