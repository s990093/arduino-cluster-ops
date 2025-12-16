import sys
import time
import struct
import serial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from esp32_tools.program_loader_v15 import InstructionV15
except ImportError:
    pass

# Patch InstructionV15
if not hasattr(InstructionV15, 'idiv'):
    InstructionV15.OP_IDIV = 0x14
    def idiv(cls, dest, src1, src2):
        return cls(cls.OP_IDIV, dest, src1, src2)
    InstructionV15.idiv = classmethod(idiv)

PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800

def test_brz():
    prog = []
    
    # R0 = 0
    prog.append(InstructionV15.mov(0, 0))
    # R1 = 5
    prog.append(InstructionV15.mov(1, 5))
    
    # LOOP START (Idx 2)
    loop_start_idx = 2
    
    # R0 = R0 + 1
    prog.append(InstructionV15.mov(2, 1))
    prog.append(InstructionV15.iadd(0, 0, 2))
    
    # Trace R0
    prog.append(InstructionV15.trace(0)) # Trace R0? No trace takes immediate.
    # We can't trace reg value easily without D2H.
    # Just run loop.
    
    # P0 = (R0 > R1) (5)
    prog.append(InstructionV15.isetp_gt(0, 0, 1))
    
    # BRZ P0, LOOP_START
    # If P0 (R0>5) is False, Jump.
    prog.append(InstructionV15.brz(loop_start_idx))
    
    prog.append(InstructionV15.exit_inst())
    
    inst_vals = [int(i.to_hex(), 16) for i in prog]
    binary = struct.pack(f'<{len(inst_vals)}I', *inst_vals)
    
    ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
    # Reset ESP32
    ser.dtr = False
    ser.rts = False
    time.sleep(0.1)
    ser.dtr = True
    ser.rts = True
    time.sleep(1.0)
    
    ser.read_all()
    
    print("Loading BRZ Test...")
    ser.write(f"load_imem {len(binary)}\n".encode())
    while "ACK" not in ser.readline().decode(errors='ignore'): pass
    ser.write(binary)
    while "OK" not in ser.readline().decode(errors='ignore'): pass
    
    print("Launching...")
    ser.write(b"kernel_launch\n")
    
    start = time.time()
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        print(f"DEV: {line}")
        if "EXIT" in line: break
        if time.time() - start > 5:
            print("TIMEOUT")
            break
            
    print("Checking R0...")
    ser.write(b"reg 0\n")
    lines = ser.read(1000).decode(errors='ignore')
    print(lines)
    
if __name__ == "__main__":
    test_brz()
