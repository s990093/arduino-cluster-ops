#!/usr/bin/env python3
"""
調試 VRAM 讀寫和 Kernel 執行 (Batching Optimized + Turbo H2D)
"""

import sys
import time
import re
import struct
import serial # Added import
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from esp32_tools.program_loader_v15 import InstructionV15

# Custom Turbo Connection Class (Overrides esp32_tools one for this script)
class ESP32Connection:
    def __init__(self, port, baud=460800):  # Updated to Turbo
        print(f"🔌 Connecting to ESP32 on {port} at {baud}...")
        try:
            self.ser = serial.Serial(port, baud, timeout=2) # 2s timeout for safety
            self.ser.set_buffer_size(rx_size=32768, tx_size=32768) # Try to request large buffer on host too
        except AttributeError:
             pass # set_buffer_size might not be available on all pySerial versions/platforms
             
        time.sleep(2) # Wait for reset
        self.read_lines() # Flush startup
        print("✅ Connected!")

    def send_command(self, command, delay=0.1):
        self.ser.write(f"{command}\n".encode())
        time.sleep(delay)

    def read_lines(self):
        lines = []
        # Wait a bit if nothing is waiting, to be safe? No, relying on caller delay or timeout.
        if self.ser.in_waiting:
            while self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if line:
                        lines.append(line)
                except Exception as e:
                    print(f"Read error: {e}")
        return lines

    def close(self):
        self.ser.close()
        print("🔌 Connection closed.")

def dma_throughput_test(conn):
    print("\n" + "="*70)
    print("🚀 Turbo DMA Throughput Test (16KB)")
    print("="*70)
    
    # 1. Generate 16KB of random data
    data_size = 16384
    test_data = bytes([i % 256 for i in range(data_size)])
    target_addr = 0x4000 # Choose a safe VRAM area (16KB offset)
    
    print(f"📦 Payload: {data_size} bytes -> VRAM[{hex(target_addr)}]")
    
    cmd = f"dma_h2d {hex(target_addr)} {data_size}"
    conn.ser.write(f"{cmd}\n".encode())
    
    # Wait for Handshake
    start_wait = time.time()
    ack = False
    while time.time() - start_wait < 2.0:
        if conn.ser.in_waiting:
            line = conn.ser.readline().decode(errors='ignore').strip()
            if "ACK_DMA_GO" in line:
                print(f"  [Device]: {line}")
                ack = True
                break
            if "ERR" in line:
                print(f"❌ Handshake Error: {line}")
                return
    
    if not ack:
        print("❌ DMA Handshake Timeout")
        return

    # Burst Write
    start_time = time.time()
    conn.ser.write(test_data)
    
    # Wait for Completion
    resp = ""
    while time.time() - start_time < 5.0: # 5s max for 16KB
        if conn.ser.in_waiting:
            line = conn.ser.readline().decode(errors='ignore').strip()
            if "DMA" in line:
                resp = line
                break
            
    end_time = time.time()
    duration = end_time - start_time
    # Avoid div by zero
    if duration < 0.001: duration = 0.001
    
    speed_kb = (data_size / 1024) / duration
    
    if "DMA_OK" in resp:
        print(f"✅ Transfer Complete!")
        print(f"⏱️ Time: {duration:.4f}s")
        print(f"⚡ Speed: {speed_kb:.2f} KB/s")
    else:
        print(f"❌ Transfer Failed: {resp}")

def debug_rw(conn):
    print("\n" + "="*70)
    print("🔍 VRAM 讀寫調試")
    print("="*70)

    # 測試數據：簡單的序列
    test_data = [10, 20, 30, 40, 50, 60, 70, 80]
    binary_data = struct.pack('<' + 'I' * len(test_data), *test_data)
    
    print(f"\n📝 H2D: 寫入測試數據到 0x0400 (DMA Transfer): {test_data}")
    conn.ser.reset_input_buffer()
    
    cmd = f"dma_h2d 400 {len(binary_data)}" 
    conn.ser.write(f"{cmd}\n".encode())
    
    ack_received = False
    start_wait = time.time()
    while time.time() - start_wait < 2.0:
        if conn.ser.in_waiting:
            line = conn.ser.readline().decode(errors='ignore').strip()
            if "ACK" in line: # ACK_DMA_GO or ACK_READY_FOR_DMA
                print(f"  [Device]: {line}") 
                ack_received = True
                break
                
    if ack_received:
        conn.ser.write(binary_data)
        print(f"  📤 Sent {len(binary_data)} bytes raw binary...")
        
        start_wait = time.time()
        while time.time() - start_wait < 2.0:
             if conn.ser.in_waiting:
                line = conn.ser.readline().decode(errors='ignore').strip()
                print(f"  [Device]: {line}")
                if "DMA_COMPLETE" in line or "DMA_OK" in line:
                    print("  ✅ H2D Copy Successful!")
                    break
    else:
        print("  ❌ DMA Handshake Failed (No ACK)")

    # 讀回驗證 (D2H)
    print(f"\n📖 D2H: 讀回 0x0400 驗證:")
    # dma_d2h <hex_addr> <dec_count>
    conn.send_command("dma_d2h 400 8", delay=0.8)
    time.sleep(0.5)
    lines = conn.read_lines()
    
    print(f"收到 {len(lines)} 行:")
    for line in lines[:12]:
        print(f"  {line}")

    # 創建簡單的 kernel：複製數據並乘以 2
    print(f"\n⚙️  Kernel Launch (output = input * 2):")
    kernel = [
        InstructionV15.s2r(31, InstructionV15.SR_LANEID),  # R31 = lane_id
        InstructionV15.mov(10, 1),      # R10 = 1
        InstructionV15.mov(1, 10),      # R1 = 10 (shift amount)
        InstructionV15.shl(10, 10, 1),  # R10 = 1 << 10 = 1024 (input_base 0x0400)
        
        InstructionV15.mov(11, 32),     # R11 = output_base (0x0020)
        InstructionV15.mov(20, 4),
        InstructionV15.imul(21, 31, 20),  # R21 = lane_id * 4
        InstructionV15.ldx(0, 10, 21),    # R0 = input[lane_id + 1024]
        InstructionV15.mov(1, 2),
        InstructionV15.imul(2, 0, 1),     # R2 = R0 * 2
        InstructionV15.stx(11, 21, 2),    # output[lane_id] = R2
        InstructionV15.exit_inst()
    ]
    
    expected = [x * 2 for x in test_data]

    # --- Use load_imem for High-Speed Kernel Upload ---
    print(f"\n🚀 Uploading Kernel (Binary Mode)...")
    
    inst_binary_list = []
    for inst in kernel:
        val = int(inst.to_hex(), 16)
        inst_binary_list.append(val)
        
    binary_blob = struct.pack('<' + 'I' * len(inst_binary_list), *inst_binary_list)
    
    conn.ser.write(f"load_imem {len(binary_blob)}\n".encode())
    
    # Handshake
    start_wait = time.time()
    ack = False
    while time.time() - start_wait < 2.0:
        if conn.ser.in_waiting:
             line = conn.ser.readline().decode(errors='ignore').strip()
             print(f"  [Device]: {line}")
             if "ACK_KERN_GO" in line or "READY_FOR_KERNEL" in line:
                 ack = True
                 break
    
    if ack:
        conn.ser.write(binary_blob)
        print(f"  📤 Sent {len(binary_blob)} bytes kernel binary...")
        
        while time.time() - start_wait < 3.0:
            if conn.ser.in_waiting:
                line = conn.ser.readline().decode(errors='ignore').strip()
                print(f"  [Device]: {line}")
                if "KERN_OK" in line or "KERNEL_LOADED" in line:
                     print("  ✅ Kernel Upload Complete!")
                     break
    else:
        print("  ❌ Kernel Upload Handshake Failed")

    # 執行
    print(f"\n🚀 Launching Kernel...")
    conn.send_command("kernel_launch", delay=1.0)
    time.sleep(1.0)
    
    lines = conn.read_lines()
    for line in lines:
        print(f"  {line}")

    # 驗證輸出 (D2H)
    print(f"\n📖 D2H: 讀取輸出 (0x0020 - 0x0040):")
    conn.send_command("dma_d2h 20 8", delay=0.5)
    time.sleep(0.5)
    lines = conn.read_lines()

    print(f"收到 {len(lines)} 行:")
    for i, line in enumerate(lines):
        print(f"  [{i}] |{line}|")

    # 解析結果
    result = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^([0-9a-fA-F]{1,8}):\s+([0-9a-fA-F]+)$', line)
        if match:
            addr = match.group(1)
            val = int(match.group(2), 16) # Parse as Hex
            result.append((addr, val))

    print(f"\n📊 解析結果:")
    print(f"輸入:  {test_data}")
    print(f"預期:  {expected}")
    
    if result:
        actual = [v for a, v in result]
        print(f"實際:  {actual}")
        
        if len(actual) >= 8:
            match = all(actual[i] == expected[i] for i in range(8))
            if match:
                print(f"\n✅ 結果完全正確！")
            else:
                print(f"\n❌ 結果不匹配！")
        else:
            print(f"\n⚠️  只收到 {len(actual)} 個值，預期 8 個")
    else:
        print(f"❌ 無法解析結果")

def stress_test(conn):
    print("\n" + "="*70)
    print("🚀 Stress Test: Batch Execution (Filling Batches)")
    print("="*70)
    
    conn.ser.reset_input_buffer()
    conn.send_command("trace:off", delay=0.1) 
    conn.send_command("gpu_reset")
    time.sleep(1.0) 
    
    prog = []
    prog.append(InstructionV15.mov(1, 0))
    prog.append(InstructionV15.mov(2, 1))
    
    for _ in range(60):
        prog.append(InstructionV15.iadd(1, 1, 2)) 
        
    prog.append(InstructionV15.exit_inst())
    
    print(f"📝 Loading {len(prog)} instructions...")
    
    inst_binary_list = []
    for inst in prog:
        val = int(inst.to_hex(), 16)
        inst_binary_list.append(val)
        
    binary_blob = struct.pack('<' + 'I' * len(inst_binary_list), *inst_binary_list)
    
    start_time = time.time()
    conn.ser.write(f"load_imem {len(binary_blob)}\n".encode())
    
    ack = False
    wait_start = time.time()
    while time.time() - wait_start < 2.0:
        if conn.ser.in_waiting:
             line = conn.ser.readline().decode(errors='ignore').strip()
             if "ACK" in line or "READY" in line:
                 ack = True
                 break
                 
    if ack:
        conn.ser.write(binary_blob)
        while time.time() - wait_start < 3.0:
             if conn.ser.in_waiting:
                 line = conn.ser.readline().decode(errors='ignore').strip()
                 if "KERN_OK" in line or "KERNEL_LOADED" in line:
                     break
    else:
        print("❌ Handshake failed")
        return

    print(f"✅ Load Time: {time.time() - start_time:.3f}s")
    
    print("Running...")
    conn.send_command("kernel_launch")
    time.sleep(1.5) 
    
    conn.send_command("reg 0")
    lines = conn.read_lines()
    val = -1
    for line in lines:
        match = re.search(r'R\s*\[?\s*1\s*\]?\s*=\s*(\d+)', line)
        if match:
            try:
                val = int(match.group(1))
            except:
                pass
            print(f"🧐 R1 Final Value: {val}")
            
    if val == 60:
        print("✅ Stress Test Passed (Batching worked!)")
    else:
        print(f"❌ Stress Test Failed (Expected 60, got {val})")

def test_stats(conn):
    print("\n" + "="*70)
    print("📊 Testing 'stats' command")
    print("="*70)
    conn.ser.reset_input_buffer()
    conn.send_command("stats")
    time.sleep(0.5)
    lines = conn.read_lines()
    for line in lines:
        print(f"  {line}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = "/dev/cu.usbserial-589A0095521"
        
    conn = ESP32Connection(port)
    try:
        debug_rw(conn)
        test_stats(conn)
        dma_throughput_test(conn) # Turbo Test
        stress_test(conn)
    finally:
        conn.close()
