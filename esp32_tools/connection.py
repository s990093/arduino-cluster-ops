"""
ESP32 串口連接管理
"""

import serial
import time
from typing import List


class ESP32Connection:
    """管理 ESP32 串口連接"""
    
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self._connect(timeout)
    
    def _connect(self, timeout: float):
        """建立串口連接"""
        print(f"🔌 Connecting to ESP32 on {self.port}...")
        self.ser = serial.Serial(self.port, self.baudrate, timeout=timeout)
        time.sleep(2)  # 等待 ESP32 初始化
        self.ser.reset_input_buffer()
        print("✅ Connected!\n")
    
    def send_command(self, cmd: str, delay: float = 0.3):
        """發送命令到 ESP32"""
        self.ser.write(f"{cmd}\n".encode())
        time.sleep(delay)
    
    def read_lines(self) -> List[str]:
        """讀取所有可用的輸出行 (Limit 1000 to prevent hang)"""
        lines = []
        count = 0
        while self.ser.in_waiting and count < 1000:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    lines.append(line)
                count += 1
            except Exception:
                continue
        return lines
    
    def close(self):
        """關閉連接"""
        if self.ser and self.ser.is_open:
            self.ser.close()
