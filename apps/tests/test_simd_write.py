
import sys
import time
import struct
import numpy as np
import serial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from esp32_tools.program_loader_v15 import InstructionV15
except ImportError:
    pass

PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800

def test_write():
    prog = []
    
    # R0 = LaneID
    prog.append(InstructionV15.s2r(0, InstructionV15.SR_LANEID))
    
    # R1 = 0x1000 (Start Offset)
    # Use MOV for now (4096 = 0x1000). Needs 16-bit mov? 
    # MOV supports 8-bit.
    # Hack: R1 = 16. SHL 8.
    prog.append(InstructionV15.mov(1, 16))
    prog.append(InstructionV15.mov(31, 8)) # R31=8
    prog.append(InstructionV15.shl(1, 1, 31)) # R1 = 0x1000
    
    # Addr = R1 + LaneID*4
    # R2 = LaneID * 4
    prog.append(InstructionV15.mov(2, 4))
    prog.append(InstructionV15.imul(2, 0, 2))
    
    # Store: STX R0 to [R1 + R2]
    # R0 contains LaneID.
    prog.append(InstructionV15.stx(0, 1, 2))
    
    prog.append(InstructionV15.exit_inst())
    
    inst_vals = [int(i.to_hex(), 16) for i in prog]
    binary = struct.pack(f'<{len(inst_vals)}I', *inst_vals)
    
    ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
    # Reset
    ser.dtr = False; ser.rts = False; time.sleep(0.1)
    ser.dtr = True; ser.rts = True; time.sleep(1.0)
    ser.read_all()
    
    print("Loading Write Test...")
    ser.write(f"load_imem {len(binary)}\n".encode())
    while "ACK" not in ser.readline().decode(errors='ignore'): pass
    ser.write(binary)
    while "OK" not in ser.readline().decode(errors='ignore'): pass
    
    print("Launching...")
    ser.write(b"kernel_launch\n")
    while "EXIT" not in ser.readline().decode(errors='ignore'): pass
    
    print("Reading Memory 0x1000...")
    # Read 32 bytes (8 words)
    ser.write(f"dma_d2h {hex(4096)} 8\n".encode())
    
    # Parse d2h output
    # Expected: 0, 1, 2, 3, 4, 5, 6, 7
    lines = ser.read(1000).decode(errors='ignore')
    print("Raw Lines:", lines)
    
if __name__ == "__main__":
    test_write()
