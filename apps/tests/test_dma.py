
import sys
import time
import struct
import numpy as np
import serial
from pathlib import Path

PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800

def test_dma():
    ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
    # Reset
    ser.dtr = False; ser.rts = False; time.sleep(0.1)
    ser.dtr = True; ser.rts = True; time.sleep(1.0)
    ser.read_all()
    
    print("Testing H2D -> D2H Loopback...")
    
    # Generate Pattern: 0..1023 (4KB)
    # Int32
    print("Generating Data...")
    arr = np.arange(1024, dtype=np.uint32)
    data = struct.pack(f'<{len(arr)}I', *arr)
    
    # H2D
    print(f"H2D: {len(data)} bytes -> 0x8000")
    ser.write(f"dma_h2d 0x8000 {len(data)}\n".encode())
    while "ACK" not in ser.readline().decode(errors='ignore'): pass
    ser.write(data)
    while "DMA" not in ser.readline().decode(errors='ignore'): pass
    
    # D2H
    print("D2H: Reading back...")
    count = 1024
    ser.write(f"dma_d2h 0x8000 {count}\n".encode())
    
    rx_data = []
    start = time.time()
    while len(rx_data) < count:
        line = ser.readline().decode(errors='ignore').strip()
        if ":" in line:
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    val = int(parts[1].strip(), 16)
                    rx_data.append(val)
                except: pass
        if time.time() - start > 5:
            print("Timeout reading D2H")
            break
            
    rx_arr = np.array(rx_data, dtype=np.uint32)
    
    print(f"Sent: {arr[:8]}")
    print(f"Recv: {rx_arr[:8]}")
    
    if np.array_equal(arr, rx_arr):
        print("✅ DMA Loopback Passed!")
    else:
        print("❌ DMA Loopback Failed!")
        diff = np.where(arr != rx_arr)[0]
        print(f"First diff at index {diff[0]}: Sent {arr[diff[0]]} Recv {rx_arr[diff[0]]}")

if __name__ == "__main__":
    test_dma()
