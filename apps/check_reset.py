
import serial
import time

PORT = "/dev/cu.usbserial-589A0095521"
BAUD_RATE = 460800

ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
ser.dtr = False; ser.rts = False; time.sleep(0.1)
ser.dtr = True; ser.rts = True; time.sleep(1.0)
ser.read_all()

print("Sending gpu_reset...")
ser.write(b"gpu_reset\n")

while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line:
        print(f"DEV: {line}")
    if "ACK" in line or "Ready" in line: # gpu_reset doesn't print ACK?
        # Check firmware `processCommand`.
        # `gpu_reset`: calls `vm.init()`, `simd_engine.reset()`, `Serial.println("OK")`.
        # So wait for OK.
        pass
    if "OK" in line:
        break
    if time.time() - start > 5:
        break
