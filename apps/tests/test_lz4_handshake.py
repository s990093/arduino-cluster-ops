
import serial
import time
import struct

PORT = "/dev/cu.usbserial-589A0095521"
BAUD = 460800

def test_handshake():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=2)
        print(f"Opened {PORT} at {BAUD}")
    except Exception as e:
        print(f"Failed to open port: {e}")
        return

    time.sleep(2)
    ser.reset_input_buffer()
    
    # Send Command
    size = 244
    msg = f"load_imem_lz4 {size}\n"
    print(f"Sending: {msg.strip()}")
    ser.write(msg.encode())
    
    # Read response
    print("Waiting for response...")
    start = time.time()
    while time.time() - start < 5:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[RX] {line}")
            if "ACK_LZ4_GO" in line:
                print("✅ Handshake Success!")
                return
    
    print("❌ Handshake Timeout or Failure")
    ser.close()

if __name__ == "__main__":
    test_handshake()
