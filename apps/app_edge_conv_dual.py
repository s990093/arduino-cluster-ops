"""
Dual-Device Accelerated Edge Convolution
Uses two ESP32 devices in parallel for 16-lane SIMD processing
"""

import sys
import time
import struct
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serial
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("⚠️  lz4 not installed. Falling back to standard upload.")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from esp32_tools.program_loader_v15 import InstructionV15
except ImportError as e:
    print(f"❌ Failed to import InstructionV15: {e}")
    sys.exit(1)

# Patch InstructionV15 to add IDIV if missing
if not hasattr(InstructionV15, 'idiv'):
    InstructionV15.OP_IDIV = 0x14
    def idiv(cls, dest, src1, src2):
        return cls(cls.OP_IDIV, dest, src1, src2)
    InstructionV15.idiv = classmethod(idiv)

# Ensure aliases exist for convenience
if not hasattr(InstructionV15, 'add'): InstructionV15.add = InstructionV15.iadd
if not hasattr(InstructionV15, 'sub'): InstructionV15.sub = InstructionV15.isub
if not hasattr(InstructionV15, 'mul'): InstructionV15.mul = InstructionV15.imul


# ==========================================
# 1. Configuration - DUAL DEVICE
# ==========================================
KERNEL_WEIGHTS = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=torch.float32).reshape(1, 1, 3, 3)

# Two ESP32 Devices
PORT_DEVICE_0 = "/dev/cu.usbserial-589A0095521"
PORT_DEVICE_1 = "/dev/cu.usbserial-2130"
BAUD_RATE = 460800

IMG_W, IMG_H = 256, 256
IMG_SIZE = IMG_W * IMG_H
VRAM_INPUT_BASE = 0x0000
VRAM_OUTPUT_BASE = 0x8000
TILE_W = 128
TILE_H = 32

def run_pytorch_ref(input_img):
    """PyTorch reference implementation"""
    out_rgb = np.zeros_like(input_img, dtype=np.float32)
    
    for c in range(3):
        x = torch.tensor(input_img[:, :, c], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = F.conv2d(x, KERNEL_WEIGHTS, padding=1)
        out_rgb[:, :, c] = out.squeeze().numpy()
        
    return out_rgb


# ==========================================
# 2. ASM Kernel Generator (Same as original)
# ==========================================
def build_asm_program(tile_w, tile_h):
    """Generates SIMD assembly for 3x3 Convolution"""
    prog = []
    
    loop_limit = tile_w * tile_h
    width = tile_w
    out_base = VRAM_OUTPUT_BASE
    
    # Init Registers
    prog.append(InstructionV15.mov(31, 8))
    dest_reg_8 = 31

    def load_imm32(reg, val):
        parts = [
            (val >> 24) & 0xFF,
            (val >> 16) & 0xFF,
            (val >> 8) & 0xFF,
            val & 0xFF
        ]
        
        start_idx = 0
        while start_idx < 4 and parts[start_idx] == 0:
            start_idx += 1
            
        if start_idx == 4:
            prog.append(InstructionV15.sub(reg, reg, reg))
            return
            
        prog.append(InstructionV15.mov(reg, parts[start_idx]))
        
        for i in range(start_idx + 1, 4):
            prog.append(InstructionV15.shl(reg, reg, dest_reg_8))
            if parts[i] != 0:
                prog.append(InstructionV15.mov(20, parts[i]))
                prog.append(InstructionV15.or_op(reg, reg, 20))

    prog.append(InstructionV15.s2r(2, InstructionV15.SR_LANEID))
    prog.append(InstructionV15.sub(0, 0, 0))
    load_imm32(1, loop_limit)
    
    if width > 255:
        load_imm32(6, width)
    else:
        prog.append(InstructionV15.mov(6, width))
        
    prog.append(InstructionV15.sub(30, 30, 30))
    
    # === LOOP START ===
    loop_start_idx = len(prog) 
    
    prog.append(InstructionV15.iadd(3, 0, 2)) 
    prog.append(InstructionV15.idiv(5, 3, 6))
    prog.append(InstructionV15.imul(21, 5, 6))
    prog.append(InstructionV15.isub(4, 3, 21))
    prog.append(InstructionV15.sub(10, 10, 10))
    
    # --- Convolution (Unrolled) ---
    prog.append(InstructionV15.mov(22, 4))
    prog.append(InstructionV15.imul(22, 3, 22))
    prog.append(InstructionV15.ldx(11, 22, 30))
    prog.append(InstructionV15.mov(23, 4))
    prog.append(InstructionV15.imul(11, 11, 23))
    prog.append(InstructionV15.iadd(10, 10, 11))
    
    stride_byte = width * 4
    offset_list = [-stride_byte, stride_byte, -4, 4]
    
    for off in offset_list:
        val = abs(off)
        if val > 255:
            load_imm32(24, val)
        else:
            prog.append(InstructionV15.mov(24, val))
            
        if off < 0:
             prog.append(InstructionV15.sub(22, 22, 24))
             prog.append(InstructionV15.ldx(12, 22, 30))
             prog.append(InstructionV15.add(22, 22, 24))
        else:
             prog.append(InstructionV15.add(22, 22, 24))
             prog.append(InstructionV15.ldx(12, 22, 30))
             prog.append(InstructionV15.sub(22, 22, 24))
             
        prog.append(InstructionV15.isub(10, 10, 12))
        
    # --- Store Result ---
    load_imm32(25, out_base)
    prog.append(InstructionV15.mov(26, 4))
    prog.append(InstructionV15.imul(26, 3, 26))
    prog.append(InstructionV15.iadd(26, 26, 25))
    prog.append(InstructionV15.stx(26, 30, 10))
    
    # === LOOP CONTROL ===
    prog.append(InstructionV15.mov(27, 8))
    prog.append(InstructionV15.iadd(0, 0, 27))
    load_imm32(29, loop_limit - 1) 
    prog.append(InstructionV15.isetp_gt(0, 0, 29))
    prog.append(InstructionV15.brz(loop_start_idx))
    prog.append(InstructionV15.exit_inst())
    
    return prog


# ==========================================
# 3. ESP32 Client (Same as original)
# ==========================================
class ESP32TurboClient:
    def __init__(self, port, baud=BAUD_RATE, device_id=0):
        self.device_id = device_id
        self.port = port
        self.baud = baud
        self.lock = threading.Lock()  # Thread-safe serial access
        self.ser = serial.Serial(port, baud, timeout=2)
        try:
            self.ser.set_buffer_size(rx_size=32768, tx_size=32768)
        except: pass
        time.sleep(2)
        self.ser.read_all()
        print(f"✅ Device {device_id}: Connected to {port} at {baud}")

    def h2d(self, addr, data_bytes):
        with self.lock:
            if HAS_LZ4:
                return self.h2d_lz4(addr, data_bytes)
            else:
                self.h2d_standard(addr, data_bytes)
                return len(data_bytes), 100
    
    def h2d_lz4(self, addr, data_bytes):
        # Note: lock already acquired by h2d()
        # Clear any stale data in buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        total_size = len(data_bytes)
        CHUNK_SIZE = 2048
        offset = 0
        total_compressed = 0
        
        cmd = f"dma_h2d_lz4 {hex(addr)} {total_size}\n"
        self.ser.write(cmd.encode())
        self.ser.flush()  # Ensure command is sent
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()  # Clear on timeout
                raise TimeoutError("Timeout waiting for ACK_LZ4_GO")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK_LZ4_GO" in line: 
                break
            if line:  # Only sleep if we got data
                continue
            time.sleep(0.001)  # Small delay if no data
        
        while offset < total_size:
            chunk = data_bytes[offset : offset + CHUNK_SIZE]
            compressed = lz4.block.compress(chunk, store_size=False)
            total_compressed += len(compressed)
            
            header = struct.pack('<H', len(compressed))
            self.ser.write(header + compressed)
            
            offset += len(chunk)
            time.sleep(0.0005)
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()  # Clear on timeout
                raise TimeoutError("Timeout waiting for LZ4_LOAD_OK")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "LZ4_LOAD_OK" in line: 
                break
            if line:
                continue
            time.sleep(0.001)
        
        # Clear any residual data after completion
        self.ser.reset_input_buffer()
        
        ratio = (total_compressed / total_size) * 100 if total_size > 0 else 100
        return total_compressed, ratio
    
    def h2d_standard(self, addr, data_bytes):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
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
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for DMA")
            line = self.ser.readline().decode(errors='ignore').strip()
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
        self.ser.write(f"load_imem_lz4 {total_size}\n".encode())
        self.ser.flush()
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 5:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for ACK_LZ4_GO")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "ACK_LZ4_GO" in line: break
            if line: continue
            time.sleep(0.001)

        CHUNK_SIZE = 2048
        offset = 0
        total_compressed = 0
        
        while offset < total_size:
            chunk = binary[offset : offset + CHUNK_SIZE]
            compressed = lz4.block.compress(chunk, store_size=False)
            total_compressed += len(compressed)
            
            header = struct.pack('<H', len(compressed))
            self.ser.write(header + compressed)
            
            offset += len(chunk)
            time.sleep(0.0005)

        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for LZ4_LOAD_OK")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "LZ4_LOAD_OK" in line: break
            if line: continue
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()
        
        ratio = (total_compressed / total_size) * 100 if total_size > 0 else 100
        print(f"   Device {self.device_id}: Compression {total_size} -> {total_compressed} bytes ({ratio:.1f}%)")

    def load_kernel_standard(self, binary):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        self.ser.write(f"load_imem {len(binary)}\n".encode())
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
        
        self.ser.write(binary)
        self.ser.flush()
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 10:
                self.ser.reset_input_buffer()
                raise TimeoutError("Timeout waiting for OK/LOADED")
            line = self.ser.readline().decode(errors='ignore').strip()
            if "OK" in line or "LOADED" in line: break
            if line: continue
            time.sleep(0.001)
        
        self.ser.reset_input_buffer()

    def launch(self):
        with self.lock:
            self.ser.reset_input_buffer()  # Clear before launch
            self.ser.write(b"kernel_launch\n")
            self.ser.flush()
            
            start = time.time()
            timeout_start = time.time()
            while True:
                if time.time() - timeout_start > 30:  # 30s timeout for kernel execution
                    self.ser.reset_input_buffer()
                    raise TimeoutError("Kernel launch timeout")
                line = self.ser.readline().decode(errors='ignore').strip()
                if "EXIT" in line: break
                if line: continue
                time.sleep(0.001)
            
            exec_time = time.time() - start
            self.ser.reset_input_buffer()  # Clear after execution
            return exec_time
        
    def d2h(self, addr, size_bytes):
        with self.lock:
            self.ser.reset_input_buffer()  # Clear before D2H
            
            count = size_bytes // 4
            self.ser.write(f"dma_d2h_binary {hex(addr)} {count}\n".encode())
            self.ser.flush()
            
            actual_bytes = 0
            start_time = time.time()
            while True:
                if time.time() - start_time > 15:
                    self.ser.reset_input_buffer()
                    raise TimeoutError("Timeout waiting for ACK_D2H_BIN")
                line = self.ser.readline().decode(errors='ignore').strip()
                if "ACK_D2H_BIN" in line:
                    try:
                        actual_bytes = int(line.split(":")[1])
                    except:
                        self.ser.reset_input_buffer()
                        return np.zeros(count, dtype=np.int32)
                    break
                if line: continue
                time.sleep(0.001)
            
            data_bytes = self.ser.read(actual_bytes)
            time.sleep(0.01)  # Give ESP32 time to prepare D2H_OK response
            
            start_time = time.time()
            while True:
                if time.time() - start_time > 20:
                    self.ser.reset_input_buffer()
                    raise TimeoutError("Timeout waiting for D2H_OK")
                line = self.ser.readline().decode(errors='ignore').strip()
                if "D2H_OK" in line: break
                if line: continue
                time.sleep(0.002)
            
            self.ser.reset_input_buffer()  # Clear after D2H
            
            return np.frombuffer(data_bytes, dtype=np.int32)


# ==========================================
# 4. Parallel Task Manager
# ==========================================
class DualDeviceManager:
    """Manages two ESP32 devices for parallel processing"""
    
    def __init__(self, port0, port1, kernel_bin):
        self.clients = [
            ESP32TurboClient(port0, device_id=0),
            ESP32TurboClient(port1, device_id=1)
        ]
        
        # Load kernel to both devices
        print("🧩 Loading Kernel to both devices...")
        for client in self.clients:
            client.load_kernel(kernel_bin)
        print("✅ Both devices ready")
        
        # Stats
        self.total_original_bytes = [0, 0]
        self.total_compressed_bytes = [0, 0]
        self.task_count = [0, 0]
    
    def process_task(self, device_id, tile_data, max_retries=3):
        """Process a single tile on specified device with retry logic"""
        client = self.clients[device_id]
        
        for attempt in range(max_retries):
            try:
                # H2D
                if HAS_LZ4:
                    compressed_size, ratio = client.h2d(VRAM_INPUT_BASE, tile_data)
                    self.total_original_bytes[device_id] += len(tile_data)
                    self.total_compressed_bytes[device_id] += compressed_size
                else:
                    client.h2d(VRAM_INPUT_BASE, tile_data)
                
                # Launch
                exec_time = client.launch()
                
                # D2H
                PAD_W = TILE_W + 2
                PAD_H = TILE_H + 2
                out_bytes = PAD_W * PAD_H * 4
                tile_out_raw = client.d2h(VRAM_OUTPUT_BASE, out_bytes)
                tile_out_padded = tile_out_raw.reshape(PAD_H, PAD_W)
                tile_out = tile_out_padded[1:-1, 1:-1]
                
                self.task_count[device_id] += 1
                return tile_out, exec_time
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Device {device_id} error (attempt {attempt+1}/{max_retries}): {e}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Device {device_id} failed after {max_retries} attempts: {e}")
    
    def get_stats(self):
        """Return statistics for both devices"""
        stats = []
        for i in range(2):
            if HAS_LZ4 and self.total_original_bytes[i] > 0:
                ratio = (self.total_compressed_bytes[i] / self.total_original_bytes[i]) * 100
                stats.append({
                    'device_id': i,
                    'tasks': self.task_count[i],
                    'original': self.total_original_bytes[i],
                    'compressed': self.total_compressed_bytes[i],
                    'ratio': ratio
                })
            else:
                stats.append({
                    'device_id': i,
                    'tasks': self.task_count[i]
                })
        return stats


# ==========================================
# 5. Parallel Worker Thread
# ==========================================
def worker_thread(device_id, manager, task_queue, result_queue):
    """Worker thread for processing tasks on one device"""
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            task_queue.task_done()
            break
        
        y, x, c, tile_data = task
        
        try:
            tile_out, exec_time = manager.process_task(device_id, tile_data)
            result_queue.put((y, x, c, tile_out, exec_time))
        except Exception as e:
            print(f"❌ Device {device_id} error: {e}")
            result_queue.put((y, x, c, None, 0))
        
        task_queue.task_done()


# ==========================================
# 6. Main
# ==========================================
if __name__ == "__main__":
    # A. Load and Resize Image
    img_path = str(Path(__file__).parent.parent / "data" / "images" / "IMG_6257.JPG")
    print(f"🖼️ Loading Image: {img_path}")

    
    try:
        original_img = Image.open(img_path)
        original_img = original_img.resize((IMG_W, IMG_H))
        rgb_img = original_img.convert('RGB')
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
    PAD_W = TILE_W + 2
    PAD_H = TILE_H + 2
    print(f"🔨 Building Assembly ({PAD_W}x{PAD_H} Physical Tile)...")
    prog_objs = build_asm_program(PAD_W, PAD_H)
    inst_vals = [int(i.to_hex(), 16) for i in prog_objs]
    kernel_bin = struct.pack(f'<{len(inst_vals)}I', *inst_vals)
    
    # D. Create Dual Device Manager
    try:
        manager = DualDeviceManager(PORT_DEVICE_0, PORT_DEVICE_1, kernel_bin)
        
        # E. Prepare Task Queue
        hw_out = np.zeros((IMG_H, IMG_W, 3), dtype=np.int32)
        
        tiles_y = IMG_H // TILE_H
        tiles_x = IMG_W // TILE_W
        total_tasks = tiles_y * tiles_x * 3
        
        print(f"🧩 Split Image into {tiles_x} x {tiles_y} Tiles x 3 Channels = {total_tasks} Tasks")
        print(f"🚀 Processing with 2 Devices (16 lanes total)...")
        
        # Pad the entire image
        img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        
        # Create queues
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Start worker threads
        threads = []
        for device_id in range(2):
            t = threading.Thread(
                target=worker_thread,
                args=(device_id, manager, task_queue, result_queue)
            )
            t.start()
            threads.append(t)
        
        # Enqueue all tasks
        task_list = []
        for y in range(0, IMG_H, TILE_H):
            for x in range(0, IMG_W, TILE_W):
                for c in range(3):
                    tile_padded = img_padded[y:y+PAD_H, x:x+PAD_W, c]
                    flat_tile = tile_padded.flatten().astype(np.int32)
                    input_bytes = struct.pack(f'<{len(flat_tile)}I', *flat_tile)
                    task_list.append((y, x, c, input_bytes))
        
        # Distribute tasks (round-robin for load balancing)
        for task in task_list:
            task_queue.put(task)
        
        # Add poison pills
        for _ in range(2):
            task_queue.put(None)
        
        # Collect results with progress bar
        from tqdm import tqdm
        exec_times = []
        
        with tqdm(total=total_tasks, unit="task") as pbar:
            for _ in range(total_tasks):
                y, x, c, tile_out, exec_time = result_queue.get()
                if tile_out is not None:
                    hw_out[y:y+TILE_H, x:x+TILE_W, c] = tile_out
                    exec_times.append(exec_time)
                pbar.update(1)
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Print statistics
        print("\n📊 Performance Statistics:")
        stats = manager.get_stats()
        for s in stats:
            print(f"\n   Device {s['device_id']}:")
            print(f"      Tasks: {s['tasks']}")
            if 'ratio' in s:
                print(f"      Original:   {s['original']:,} bytes")
                print(f"      Compressed: {s['compressed']:,} bytes")
                print(f"      Ratio:      {s['ratio']:.1f}%")
        
        if exec_times:
            avg_time = np.mean(exec_times)
            total_time = sum(exec_times)
            print(f"\n   Average kernel execution: {avg_time*1000:.2f}ms")
            print(f"   Total execution time: {total_time:.2f}s")
                
    except Exception as e:
         print(f"⚠️ Hardware execution failed: {e}")
         import traceback
         traceback.print_exc()
         hw_out = np.zeros_like(ref_out)
            
    # F. Verify & Visualization
    print("\n📊 Verification:")
    
    diff = np.abs(hw_out - ref_out)
    mae = np.mean(diff)
    print(f"   Max Diff: {np.max(diff)}")
    print(f"   MAE: {mae}")
    
    # Plotting
    print("🎨 Displaying Results...")
    plt.figure(figsize=(15, 5))
    
    def normalize_for_display(arr):
        return np.clip(np.abs(arr), 0, 255).astype(np.uint8)

    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("PyTorch Reference")
    plt.imshow(normalize_for_display(ref_out))
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.title(f"Dual ESP32 Output\nMAE: {mae:.2f}")
    plt.imshow(normalize_for_display(hw_out))
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("Difference Map")
    diff_disp = np.clip(diff * 5, 0, 255).astype(np.uint8)
    plt.imshow(diff_disp)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
