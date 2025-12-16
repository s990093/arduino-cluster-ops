#!/usr/bin/env python3
"""
ESP32 MicroGPU Image Processing Demo

完整的圖像處理流程範例：
輸入圖片 -> Host 處理 -> 傳入 VRAM -> Device 運算 -> 取回結果 -> 顯示

Features:
- CUDA-style API (malloc, memcpy, launch)
- Tile-based execution for 8-lane SIMT
- Real-time visualization with Matplotlib
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
import torch.nn.functional as F
import re
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from esp32_tools import ESP32Connection
from esp32_tools.program_loader_v15 import InstructionV15


class MicroGPU:
    """
    MicroGPU: CUDA-style API for ESP32 CUDA VM
    
    模擬 CUDA 編程模型:
    - cudaMalloc() -> malloc()
    - cudaMemcpy() -> memcpy()
    - kernel<<<grid, block>>>() -> launch()
    """
    
    def __init__(self):
        # Configuration
        PORT = "/dev/cu.usbserial-589A0095521"
        # BAUD_RATE = 921600
        BAUD_RATE = 115200 # Downgraded for stability
        TIMEOUT = 120 # Increased timeout for slow baud
        VM_VRAM_SIZE = 40960 # Matches Firmware Config

        # Execution Params
        # BYTES_PER_UPLOAD = 2048
        BYTES_PER_UPLOAD = 256 # Safe chunk size
        """初始化 MicroGPU 設備"""
        self.conn = ESP32Connection(PORT, baudrate=BAUD_RATE)
        self.vram_allocator = 0  # VRAM 分配器（字節地址）
        self.allocations = {}     # 記錄已分配的記憶體
        self.program_loaded = False  # 追蹤程序是否已加載
        
        # Reset device
        self.conn.send_command("reset", delay=0.5)
        print("🎮 MicroGPU Device Initialized")
    
    def malloc(self, name: str, size_bytes: int) -> int:
        """
        在 VRAM 中分配記憶體
        
        Args:
            name: 記憶體區域名稱（用於追蹤）
            size_bytes: 需要的字節數
            
        Returns:
            分配的起始地址
        """
        addr = self.vram_allocator
        self.allocations[name] = {
            'addr': addr,
            'size': size_bytes
        }
        self.vram_allocator += size_bytes
        
        # 對齊到 4 字節
        if self.vram_allocator % 4 != 0:
            self.vram_allocator += (4 - self.vram_allocator % 4)
        
        print(f"  📦 malloc('{name}'): {size_bytes} bytes @ 0x{addr:04X}")
        return addr
    
    def memcpy_host_to_device(self, name: str, data: np.ndarray) -> None:
        """
        從 Host 複製數據到 Device VRAM
        
        Args:
            name: 目標記憶體區域名稱
            data: NumPy 數組（uint8 或 int32）
        """
        if name not in self.allocations:
            raise ValueError(f"Memory '{name}' not allocated")
        
        addr = self.allocations[name]['addr']
        
        # 確保數據是 int32
        if data.dtype != np.int32:
            data = data.astype(np.int32)
        
        # 寫入 VRAM
        for i, val in enumerate(data.flat):
            # 增加 delay 以確保 Arduino 有足夠時間處理
            self.conn.send_command(f"mem {addr + i * 4} {int(val)}", delay=0.05)
        
        print(f"  ⬇️  memcpy H->D: '{name}' ({len(data.flat)} elements)")
    
    def memcpy_device_to_host(self, name: str, shape: Tuple[int, ...]) -> np.ndarray:
        """
        從 Device VRAM 讀取數據到 Host
        
        完整的 VRAM 流程：從指定 VRAM 地址讀取計算結果
        
        Args:
            name: 源記憶體區域名稱  
            shape: 輸出數組形狀
            
        Returns:
            NumPy 數組
        """
        if name not in self.allocations:
            raise ValueError(f"Memory '{name}' not allocated")
        
        addr = self.allocations[name]['addr']
        size = np.prod(shape)
        
        # 清除之前的輸出 (如 Mem Written, Loaded 等)
        # 避免 dump 結果被淹沒或讀不到
        _ = self.conn.read_lines()
        
        # 使用 dump 從 VRAM 讀取結果
        import time
        import re
        
        # 發送命令 (delay 稍微保留一點，但主要靠 polling)
        self.conn.send_command(f"dump {addr} {size}", delay=0.1)
        
        result = []
        start_time = time.time()
        timeout = 5.0  # 5秒超時
        
        print(f"     ⏳ Polling for result ({size} items)...")
        
        raw_log = [] # 記錄所有收到的行以便調試
        
        # Clear buffer first
        self.conn.ser.reset_input_buffer()
        
        while len(result) < size and (time.time() - start_time < timeout):
            lines = self.conn.read_lines()
            for line in lines:
                clean_line = line.strip()
                raw_log.append(clean_line)
                
                # 只接受 4位16進制地址 + 冒號 + 數字 的格式
                match = re.match(r'^([0-9a-fA-F]{4}):\s+(\d+)$', clean_line)
                if match:
                    val = int(match.group(2))
                    result.append(val)
            
            if len(result) < size:
                time.sleep(0.1)
        
        # 如果結果數量不對，打印原始輸出以便調試
        if len(result) < size:
            print(f"  ⚠️  Warning: Expected {size} values, got {len(result)}")
            print(f"  ⚠️  Last 20 raw lines captured:")
            for l in raw_log[-20:]:
                print(f"      {l}")
        
        # 填充到所需大小
        while len(result) < size:
            result.append(0)
        
        # 處理溢出：將值限制在 0-255 範圍
        clamped_result = []
        for val in result[:size]:
            # 處理無符號 32 位轉有符號
            if val > 2147483647:  # 大於 int32 最大值，說明是負數
                val = val - 4294967296  # 轉換為有符號負數
            # 取絕對值 (因為 kernel 沒有 ABS 指令)
            val = abs(val)
            # Clamp 到 0-255
            clamped_result.append(max(0, min(255, val)))
        
        if len(clamped_result) != size:
            print(f"  ⚠️  Size Mismatch in cudaMemcpy! Expected {size}, Got {len(clamped_result)}. Truncating/Padding.")
            if len(clamped_result) > size:
                clamped_result = clamped_result[:size]
            else:
                while len(clamped_result) < size:
                     clamped_result.append(0)

        data = np.array(clamped_result, dtype=np.int32).reshape(shape)
        print(f"  ⬆️  memcpy D->H: '{name}' @ 0x{addr:04X} ({len(result)} checked -> {len(data.flatten())} kept)")
        return data
    
    def launch(self, kernel_code: List, grid_size: int = 1, block_size: int = 8) -> None:
        """
        啟動 Kernel 執行
        
        Args:
            kernel_code: 指令列表（InstructionV15）
            grid_size: Grid 大小（模擬多次執行）
            block_size: Block 大小（Warp Size，固定為 8）
        """
        # 只在第一次加載程序
        if not self.program_loaded:
            for inst in kernel_code:
                self.conn.send_command(f"load {inst.to_hex()}", delay=0.01)
            self.program_loaded = True
            print(f"  📝 Loaded {len(kernel_code)} instructions")
        
        print(f"  🚀 launch<<<{grid_size}, {block_size}>>>: execute")
        
        # 執行（對於 grid_size > 1，需要多次執行並調整 offset）
        for grid_idx in range(grid_size):
            self.conn.send_command("run", delay=0.5)
            print(f"     Grid[{grid_idx}/{grid_size}] executed")
    
    def free_all(self) -> None:
        """釋放所有分配的記憶體"""
        self.allocations.clear()
        self.vram_allocator = 0
        self.program_loaded = False
        print("  🗑️  All memory freed")


def create_test_kernel() -> List:
    """
    Simple Test Kernel: Write Lane ID to Output
    
    This verifies that:
    1. s2r(SR_LANEID) works correctly  
    2. Each lane has unique ID (0-7)
    3. stx writes to correct addresses
    
    Expected Output @ 0x4000: [0, 1, 2, 3, 4, 5, 6, 7]
    
    Structure matches edge_detection_kernel for dynamic patching:
    [0]: s2r
    [1]: mov(10, 0)  - Input base (will be replaced)
    [2]: mov(11, 0)  - Output base (will be replaced)
    [3:]: Core logic
    """
    kernel = [
        # [0] R31 = lane_id
        InstructionV15.s2r(31, InstructionV15.SR_LANEID),
        
        # [1] R10 = Input Base (not used, but needed for patching structure)
        InstructionV15.mov(10, 0),
        
        # [2] R11 = Output Base (will be replaced by load_register_32bit)
        InstructionV15.mov(11, 0),
        
        # [3:] Core logic starts here
        # R20 = 4 (word size)
        InstructionV15.mov(20, 4),
        
        # R21 = lane_id * 4 (byte offset)
        InstructionV15.imul(21, 31, 20),
        
        # Write lane_id to output[lane_id]
        # stx(base_reg, offset_reg, src_data)
        InstructionV15.stx(11, 21, 31),  # [R11 + R21] = R31
        
        # Exit
        InstructionV15.exit_inst()
    ]
    return kernel


def create_edge_detection_kernel() -> List:
    """
    修正後的邊緣檢測 Kernel (v2)
    
    Features:
    1. Lane 0 Guard: 使用條件運算防止越界
    2. Absolute Value: 計算 |curr - prev|
    
    Memory Layout:
    - 0x0000: 輸入圖像數據 (8 pixels)
    - 0x0020: 輸出邊緣數據 (8 pixels)
    """
    kernel = [
        # ===== 初始化 =====
        InstructionV15.s2r(31, InstructionV15.SR_LANEID),  # R31 = lane_id
        InstructionV15.mov(10, 0),      # R10 = Input Base
        InstructionV15.mov(11, 8192),   # R11 = Output Base (Offset for 64x128 chunk)
        InstructionV15.mov(20, 4),      # R20 = 4
        
        # ===== 計算當前像素地址 =====
        InstructionV15.imul(21, 31, 20),  # R21 = lane_id * 4
        InstructionV15.ldx(0, 10, 21),    # R0 = current pixel
        
        # ===== Lane 0 Guard: 如果 lane_id == 0，設 previous = current =====
        InstructionV15.mov(1, 0),         # R1 = 0 (默認 previous)
        
        # 如果 lane_id > 0，R1 = input[lane_id-1]
        # 計算前一個地址
        InstructionV15.mov(22, 1),
        InstructionV15.isub(23, 31, 22),  # R23 = lane_id - 1
        InstructionV15.imul(24, 23, 20),  # R24 = (lane_id-1) * 4
        
        # 讀取前一個值 (即使是 Lane 0 也讀，這是冒險的，但之前邏輯是這樣)
        # 注意：Lane 0 讀取 -4 可能崩潰或讀到垃圾？
        # 如果 VM 沒有保護，這很危險。
        # 原邏輯直接讀取 R24。如果是 -4 (FFFFFFFC)，可能讀到非法地址。
        # 讓我修改邏輯避免讀取非法地址。
        
        # 安全讀取邏輯:
        # 如果 lane_id == 0，R24 = 0 (讀取自己)
        # 用乘法模擬條件: R24 = R24 * (lane_id != 0) ??? 難以實現
        
        # 簡單方法: 既然 R1 初始化為 0。
        # 只有當 lane_id > 0 時才執行 ldx? 不支持條件執行。
        
        # 讓我們恢復原邏輯並觀察。原邏輯:
        InstructionV15.ldx(1, 10, 24),    # R1 = input[lane_id-1]
        
        # 如果 Lane 0，這會讀取 Mem[-4]。
        # 如果這是問題所在？
        # 在 debug_vram 測試中，Lane 0 讀取 Mem[0]。
        # 這裡 R23 = 0 - 1 = -1. R24 = -4.
        
        # 如果 Lane 0 讀了垃圾，且後面邏輯試圖修正：
        # InstructionV15.mov(25, 0), isub, ...
        # 這只是為了選擇 R1 的值。
        
        # 修改：確保 Lane 0 不讀取越界。
        # 將 R24 限制為 >= 0?
        # 無法簡單做到。
        
        # 但是，如果 Lane 0 讀取 -4 沒崩潰，只是讀了垃圾。
        # 然後我們覆蓋 R1。
        
        # 這裡我們嘗試用 R0 覆蓋 R1 (如果是 Lane 0)
        # 我們之前確實有這個邏輯嗎？
        # 原代碼：
        # if lane_id == 0, R1 = R0 (用當前覆蓋)
        # 但如何實現？
        # 之前的代碼沒有顯示具體的 "if lane_id == 0" 實現，只有注釋。
        # "算術技巧... 簡化... 接受誤差".
        
        # 讓我嘗試一個更安全的邏輯:
        # 總是讀取 R0 (當前)。
        # 只有當 lane_id > 0 時，讀取 R1 (前一個)。
        # ===== 計算梯度 diff =====
        # R2 = current - previous
        # 注意：對於 Lane 0，previous (R1) 為 0，因此 R2 = current
        InstructionV15.isub(2, 0, 1),
        
        # ===== 增強對比 (Scale * 3) =====
        # 使用 R25 存儲常數 3，避免使用 R3
        InstructionV15.mov(25, 3),        # R25 = 3
        InstructionV15.imul(2, 2, 25),    # R2 = R2 * 3
        
        # ===== 寫回VRAM =====
        # Python 端會處理符號 (signed/unsigned) 和絕對值
        InstructionV15.stx(11, 21, 2),    # output[lane_id] = R2
        
        InstructionV15.exit_inst()
    ]
    return kernel



def process_image_with_microgpu(
    image_path: str,
    gpu: MicroGPU,
    tile_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 MicroGPU 處理圖像 (Tile-based Grid Execution)
    
    因為硬體限制 (Warp Size = 8)，必須由 Host 手動切分圖像，
    每次處理 8 個像素 (一個 tile)。
    
    Args:
        image_path: 輸入圖像路徑
        gpu: MicroGPU 實例
        tile_size: 分塊大小（固定為 8，對應 Warp Size）
        
    Returns:
        (原圖, 處理後的圖像)
    """
    print("\n" + "=" * 70)
    print("🖼️  Image Processing with MicroGPU (128x128 Optimized)")
    print("=" * 70)
    
    # 1. 載入圖像 -> Use Synthetic Gradient for Debugging
    # img = Image.open(image_path).convert('L')
    # img = img.resize((target_size, target_size), Image.LANCZOS)
    
    print(f"🧪 Generating Synthetic Gradient Image for Debugging...")
    target_size = 32 # User Request: 32x32 First
    # Gradient from 0 to 255 horizontally
    img_array = np.zeros((target_size, target_size), dtype=np.int32)
    for y in range(target_size):
        for x in range(target_size):
            img_array[y, x] = (x * 255) // target_size
            
    print(f"📐 Synthetic Image {target_size}x{target_size} created")
    
    h, w = img_array.shape
    
    print(f"📥 Loaded image: {img_array.shape}")
    print(f"🔢 Total pixels: {h * w}")
    
    # 2. 分配 VRAM (每行一個緩衝區)
    # 為了優化，我們處理一整行 (128 pixels)
    # VRAM Layout:
    # 0x0000: Input Row Buffer (128 * 4 = 512 bytes)
    # 0x0200: Output Row Buffer (128 * 4 = 512 bytes)
    
    # 清除之前的分配 (如果有)
    gpu.allocations.clear()
    gpu.vram_allocator = 0
    
    print("\n💾 Allocating VRAM buffers...")
    input_base = 0x0000
    output_base = 0x0200
    
    # 不需要真正 malloc，直接使用固定地址以配合 patched kernel
    gpu.allocations["row_input"] = {'addr': input_base, 'size': w * 4}
    gpu.allocations["row_output"] = {'addr': output_base, 'size': w * 4}
    
    # 3. 創建 Kernel
    print("\n⚙️  Compiling kernel...")
    kernel = create_edge_detection_kernel()
    
    # 4. 準備輸出圖像
    output_array = np.zeros_like(img_array)
    
    # 2. 分配 VRAM
    CHUNK_ROWS = 32
    CHUNK_SIZE_PIXELS = CHUNK_ROWS * w
    CHUNK_SIZE_BYTES = CHUNK_SIZE_PIXELS * 4
    
    input_base = 0x0000
    output_base = 0x4000 # 16KB offset
    
    print(f"\n💾 Allocating VRAM buffers (Chunk-based)...")
    print(f"  Chunk Size: {CHUNK_SIZE_BYTES} bytes ({CHUNK_ROWS} rows)")
    print(f"  Input Base: 0x{input_base:X}")
    print(f"  Output Base: 0x{output_base:X}")
    
    # helper to print progress bar
    def print_progress(current, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
        filledLength = int(length * current // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        if current == total: 
            print()
            
    # helper to load 32-bit immediate into register (using 8-bit MOVs)
    def load_register_32bit(reg, val):
        # Algorithm:
        # R = (B3 << 24) | (B2 << 16) | (B1 << 8) | B0
        # 1. MOV R, B3
        # 2. MOV T, 8 (Shift)
        # 3. SHL R, R, T
        # 4. MOV T, B2
        # 5. OR R, R, T
        # 6. SHL R, R, Tm (using 8) ...
        # Optimization: Use fixed temp reg R26 for value, R27 for shift=8
        insts = []
        b0 = val & 0xFF
        b1 = (val >> 8) & 0xFF
        b2 = (val >> 16) & 0xFF
        b3 = (val >> 24) & 0xFF
        
        # Load High Byte
        insts.append(InstructionV15.mov(reg, b3))
        
        # Setup Shift Reg (R27 = 8)
        insts.append(InstructionV15.mov(27, 8))
        
        # Shift & Add B2
        insts.append(InstructionV15.shl(reg, reg, 27))
        insts.append(InstructionV15.mov(26, b2))
        insts.append(InstructionV15.or_op(reg, reg, 26))
        
        # Shift & Add B1
        insts.append(InstructionV15.shl(reg, reg, 27))
        insts.append(InstructionV15.mov(26, b1))
        insts.append(InstructionV15.or_op(reg, reg, 26))
        
        # Shift & Add B0
        insts.append(InstructionV15.shl(reg, reg, 27))
        insts.append(InstructionV15.mov(26, b0))
        insts.append(InstructionV15.or_op(reg, reg, 26))
        
        return insts

    output_array = np.zeros_like(img_array)
    import time
    start_time = time.time()
    
    # Chunk Processing
    # Chunk Processing
    num_chunks = h // CHUNK_ROWS
    for chunk_idx in range(num_chunks):
        print(f"\n📦 Processing Chunk {chunk_idx+1}/{num_chunks}...")
        
        row_start = chunk_idx * CHUNK_ROWS
        row_end = row_start + CHUNK_ROWS
        chunk_data = img_array[row_start:row_end, :].flatten()
        
        # Clear VRAM before first chunk to remove stale data
        if chunk_idx == 0:
            print("  🧹 Clearing VRAM...")
            gpu.conn.send_command("reset", delay=0.5)
        
        # 1. Bulk Upload
        print("  ⬆️  Uploading data...")
        BYTES_PER_UPLOAD = 128 # Reduced to fit safe serial buffer
        total_bytes = len(chunk_data) * 4
        uploaded_bytes = 0
        
        while uploaded_bytes < total_bytes:
            pixels_per_batch = BYTES_PER_UPLOAD // 4
            batch_start_idx = uploaded_bytes // 4
            batch_end_idx = min(len(chunk_data), batch_start_idx + pixels_per_batch)
            batch_pixels = chunk_data[batch_start_idx:batch_end_idx]
            if len(batch_pixels) == 0: break
            
            import struct
            hex_data = ""
            for val in batch_pixels:
                packed = struct.pack('<I', int(val))
                hex_data += packed.hex()
                
            # Debug: Print first batch data sample
            if uploaded_bytes == 0:
                 print(f"      [Debug] Host Data Sample: {batch_pixels[:4]}")

            curr_addr = input_base + uploaded_bytes
            gpu.conn.send_command(f"wbulk {curr_addr} {len(hex_data)//2} {hex_data}", delay=0.01)
            uploaded_bytes += len(hex_data) // 2
            
            # Progress Bar
            print_progress(uploaded_bytes, total_bytes, prefix='Upload:', length=30)
            
        print("    ✅ Complete")
        
        # Verify Input Upload
        print("    🔍 Verifying Input VRAM (First 4 pixels)...")
        # Reuse dump logic manual
        gpu.conn.ser.reset_input_buffer()
        check_count = 4
        gpu.conn.send_command(f"dump {input_base} {check_count}", delay=0.5)
        check_res = []
        raw_debug = []
        st = time.time()
        while len(check_res) < check_count and (time.time() - st < 5.0):
             ls = gpu.conn.read_lines()
             for l in ls:
                 raw_debug.append(l)
                 match = re.match(r'^([0-9a-fA-F]{4,8}):\s+(\d+)$', l.strip())
                 if match: check_res.append(int(match.group(2)))
        print(f"      [Debug] VRAM Input Sample: {check_res}")
        if not check_res:
            print(f"      [Debug] Raw Dump Output: {raw_debug}")

        # 2. Executing Kernels
        print("  🚀 Executing kernels...")
        total_blocks = CHUNK_ROWS * (w // 8)
        processed_blocks = 0
        
        for r in range(CHUNK_ROWS):
            row_offset = r * w * 4 
            for block in range(w // 8):
                block_offset = block * 32
                curr_input = input_base + row_offset + block_offset
                curr_output = output_base + row_offset + block_offset
                
                # Dynamic Kernel Generation (with 32-bit address)
                # 重新生成 Kernel，但是將開頭的 Base Address 設置部分替換
                # 原始 create_edge_detection_kernel 返回:
                # [0]: s2r(31, laneid)
                # [1]: mov(10, 0)  <-- Replace
                # [2]: mov(11, 32) <-- Replace
                # ...
                
                # Back to edge detection kernel
                base_kernel = create_edge_detection_kernel()
                # Remove [1] and [2]
                core_logic = [base_kernel[0]] + base_kernel[3:]
                
                # Insert Address Loaders
                load_in  = load_register_32bit(10, curr_input)
                load_out = load_register_32bit(11, curr_output)
                
                # Combine: [S2R] + [Load In] + [Load Out] + [Core Logic]
                full_kernel = [base_kernel[0]] + load_in + load_out + base_kernel[3:]
                
                # Send command bundle
                # Use softreset to reset VM logic but KEEP VRAM Data!
                cmd_chunk = "softreset\n" 
                kernel_hex_list = []
                for inst in full_kernel:
                    encoded = inst.encode()
                    hex_str = f"{encoded:x}"
                    kernel_hex_list.append(hex_str)
                    cmd_chunk += f"load {hex_str}\n"
                
                # Debug: log kernel
                if False and r == 0 and block == 0:  # Disabled debug output
                    print(f"      [Debug] curr_input=0x{curr_input:X}, curr_output=0x{curr_output:X}")
                    print(f"      [Debug] Full kernel ({len(kernel_hex_list)} instructions):")
                    for i, hex_str in enumerate(kernel_hex_list):
                        print(f"        [{i:2d}] {hex_str}")
                
                cmd_chunk += f"run\n"
                gpu.conn.ser.write(cmd_chunk.encode())
                
                # Debug: register verification (DISABLED for clean output)
                if False and r == 0 and block == 0:
                    # The kernel was just loaded and run command sent
                    # Wait for execution to complete
                    time.sleep(0.1)
                    # Now dump registers to see what R11 was during execution  
                    for lane_idx in range(8):
                        gpu.conn.send_command(f"reg {lane_idx}", delay=0.15)
                        reg_lines = gpu.conn.read_lines()
                        has_nonzero = False
                        lane_regs = []
                        for line in reg_lines:
                            if "R2 " in line or "R11" in line or "R21" in line or "R31" in line:
                                lane_regs.append(line.strip())
                                if " = " in line and line.split("=")[1].strip() != "0":
                                    has_nonzero = True
                        if has_nonzero:
                            print(f"      [Debug Lane {lane_idx}]")
                            for reg in lane_regs:
                                print(f"        {reg}")
                
                # Need delay? 
                # 30 lines * 10 bytes = 300 bytes. 921600bps -> 3ms.
                # Processing 8 pixels -> very fast.
                time.sleep(0.005) 
                
                processed_blocks += 1
                if processed_blocks % 8 == 0:
                     print_progress(processed_blocks, total_blocks, prefix='Exec:  ', length=30)
                     
        print_progress(total_blocks, total_blocks, prefix='Exec:  ', length=30)
        print("    ✅ Complete")
        
        # Debug: Verify Output Buffer (DISABLED for clean output)
        if False:
            print("    🔍 Verifying Output VRAM (First 4 pixels after execution)...")
            gpu.conn.ser.reset_input_buffer()
            gpu.conn.send_command(f"dump {output_base} 4", delay=0.5)
            check_out = []
            raw_out = []
            st = time.time()
            while len(check_out) < 4 and (time.time() - st < 5.0):
                 ls = gpu.conn.read_lines()
                 for l in ls:
                     raw_out.append(l)
                     match = re.match(r'^([0-9a-fA-F]{4,8}):\s+(\d+)$', l.strip())
                     if match: check_out.append(int(match.group(2)))
            print(f"      [Debug] VRAM Output Sample (@ 0x{output_base:X}): {check_out}")
            if not check_out:
                print(f"      [Debug] Raw Output Dump: {raw_out}")
        
        # Input verification (DISABLED)
        if False:
            print("    🔍 Re-checking Input VRAM after execution...")
            gpu.conn.ser.reset_input_buffer()
            gpu.conn.send_command(f"dump {input_base} 4", delay=0.5)
            check_in2 = []
            st = time.time()
            while len(check_in2) < 4 and (time.time() - st < 3.0):
                 ls = gpu.conn.read_lines()
                 for l in ls:
                     match = re.match(r'^([0-9a-fA-F]{4,8}):\s+(\d+)$', l.strip())
                     if match: check_in2.append(int(match.group(2)))
            print(f"      [Debug] Input still @ 0x{input_base:X}: {check_in2}")
        
        # 3. Download Results
        print("  ⬇️  Downloading results...")
        chunk_result = []
        downloaded = 0
        
        for i in range(0, total_bytes, BYTES_PER_UPLOAD):
            curr_addr = output_base + i
            size = min(BYTES_PER_UPLOAD, total_bytes - i)
            count_needed = size // 4
            gpu.conn.send_command(f"dump {curr_addr} {count_needed}", delay=0.05)
            
            curr_chunk_res = []
            read_start = time.time()
            while len(curr_chunk_res) < count_needed and (time.time() - read_start < 2.0):
                lines = gpu.conn.read_lines()
                for line in lines:
                    match = re.match(r'^([0-9a-fA-F]{4,8}):\s+(\d+)$', line.strip())
                    if match:
                        val = int(match.group(2))
                        if val > 2147483647: val -= 4294967296
                        val = abs(val)
                        val = max(0, min(255, val))
                        curr_chunk_res.append(val)
                        if len(curr_chunk_res) >= count_needed:
                            break
                if len(curr_chunk_res) >= count_needed:
                    break
                        
            while len(curr_chunk_res) < count_needed:
                curr_chunk_res.append(0)
            
            chunk_result.extend(curr_chunk_res)
            downloaded += size
            print_progress(downloaded, total_bytes, prefix='Downl: ', length=30)
            
        print("    ✅ Complete")

        chunk_img = np.array(chunk_result).reshape((CHUNK_ROWS, w))
        output_array[row_start:row_end, :] = chunk_img

    total_time = time.time() - start_time
    print(f"\n⏱️  Total Processing Time: {total_time:.2f}s (FPS: {1/total_time:.2f})")
    
    gpu.free_all()
    return img_array, output_array


def pytorch_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    使用 PyTorch 實現相同的邊緣檢測算法（作為 Ground Truth）
    
    算法：水平梯度 gradient = ABS(current - previous) * 3
    """
    # 轉換為 tensor
    img_tensor = torch.from_numpy(image.astype(np.float32))
    
    result = np.zeros_like(image, dtype=np.float32)
    
    # 逐行處理（模擬 ESP32 的 tile-based 執行）
    for row_idx in range(img_tensor.shape[0]):
        row = img_tensor[row_idx, :]
        
        # 計算梯度 (使用絕對值)
        gradients = torch.zeros_like(row)
        for i in range(1, len(row)):
            # 修正：使用 abs() 取絕對值
            gradient = abs(row[i] - row[i-1]) * 3
            gradients[i] = gradient
        
        # Clamp 到 0-255
        gradients = torch.clamp(gradients, 0, 255)
        result[row_idx, :] = gradients.numpy()
    
    return result.astype(np.int32)


def  visualize_results(original: np.ndarray, esp32_result: np.ndarray, torch_result: np.ndarray) -> None:
    """使用 Matplotlib 可視化三張圖片並計算誤差指標"""
    
    # 計算誤差指標
    mse = np.mean((esp32_result - torch_result) ** 2)
    mae = np.mean(np.abs(esp32_result - torch_result))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Image (Host)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(esp32_result, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'ESP32 MicroGPU Result\n(VRAM-based)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(torch_result, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'PyTorch Reference\n(Ground Truth)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'ESP32 MicroGPU Edge Detection Demo\nMSE: {mse:.2f} | MAE: {mae:.2f}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('microgpu_result.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved visualization: microgpu_result.png")
    print(f"📊 Error Metrics:")
    print(f"   - MSE (Mean Squared Error): {mse:.2f}")
    print(f"   - MAE (Mean Absolute Error): {mae:.2f}")
    # plt.show()


def main():
    """主程序"""
    print("\n" + "🎮 " * 35)
    print("ESP32 MicroGPU - Image Processing Demo")
    print("🎮 " * 35)
    
    # 初始化 GPU
    gpu = MicroGPU()
    
    # 處理圖像 (使用真實圖片並縮放)
    image_path = '/Users/hungwei/Downloads/IMG_8152.JPG'
    print(f"\n🎨 Using image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    original, esp32_processed = process_image_with_microgpu(
        image_path,
        gpu,
        tile_size=128 # Actual size handling is internal
    )
    
    # 使用 PyTorch 計算參考結果
    print("\n" + "=" * 70)
    print("🔥 PyTorch Reference Calculation")
    print("=" * 70)
    torch_processed = pytorch_edge_detection(original)
    print(f"✅ PyTorch edge detection complete")
    
    # 顯示結果
    print("\n" + "=" * 70)
    print("📊 Results Comparison")
    print("=" * 70)
    print(f"\nOriginal (sample):")
    print(original[:3, :3])
    print(f"\nESP32 Result (sample):")
    print(esp32_processed[:3, :3])
    print(f"\nPyTorch Result (sample):")
    print(torch_processed[:3, :3])
    
    # 可視化（3張圖）
    visualize_results(original, esp32_processed, torch_processed)
    
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
