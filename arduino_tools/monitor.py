"""
Arduino 序列埠監測模組（美化版）
即時顯示 Arduino 的序列輸出，支援雙向通訊，使用 Rich 美化顯示
"""

import serial
import sys
import threading
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich import print as rprint

console = Console()


class SerialMonitor:
    """序列埠監測器類別（美化版）"""
    
    def __init__(self, port, baudrate=9600, timeout=1, data_callback=None):
        """
        初始化序列埠監測器
        
        Args:
            port (str): 序列埠位置
            baudrate (int): 鮑率 (預設 9600)
            timeout (float): 讀取超時時間
            data_callback (callable): 資料回調函式 func(message)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.data_callback = data_callback
        self.serial_conn = None
        self.running = False
        self.log_file = None
        self.message_count = 0
    
    def connect(self):
        """建立序列埠連線"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            console.print(f"[green]✓[/green] 已連接到 [magenta]{self.port}[/magenta] (鮑率: [yellow]{self.baudrate}[/yellow])")
            # 等待 Arduino 重啟
            time.sleep(2)
            return True
        except serial.SerialException as e:
            console.print(f"[red]✗[/red] 無法連接到 {self.port}: {e}")
            return False
    
    def disconnect(self):
        """關閉序列埠連線"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            console.print()
            console.print("[green]✓[/green] 已關閉序列埠連線")
        
        if self.log_file:
            self.log_file.close()
            console.print("[green]✓[/green] 已關閉記錄檔")
    
    def enable_logging(self, log_filename=None):
        """
        啟用資料記錄到檔案
        
        Args:
            log_filename (str): 記錄檔名稱，預設使用時間戳記
        """
        if not log_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"serial_log_{timestamp}.txt"
        
        try:
            self.log_file = open(log_filename, 'w', encoding='utf-8')
            console.print(f"[green]✓[/green] 記錄檔已啟用: [cyan]{log_filename}[/cyan]")
        except IOError as e:
            console.print(f"[red]✗[/red] 無法建立記錄檔: {e}")
    
    def read_serial(self):
        """讀取序列埠資料的執行緒"""
        while self.running:
            try:
                if self.serial_conn.in_waiting > 0:
                    data = self.serial_conn.readline()
                    try:
                        message = data.decode('utf-8').rstrip()
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        self.message_count += 1
                        
                        # 呼叫回調函式（這對 Web 介面很重要）
                        if self.data_callback:
                            try:
                                self.data_callback(message)
                            except Exception as e:
                                print(f"Callback Error: {e}")

                        # 美化輸出
                        output_text = Text()
                        output_text.append(f"[{timestamp}] ", style="dim cyan")
                        output_text.append(message, style="white")
                        
                        console.print(output_text)
                        
                        # 寫入記錄檔
                        if self.log_file:
                            self.log_file.write(f"[{timestamp}] {message}\n")
                            self.log_file.flush()
                    
                    except UnicodeDecodeError:
                        # 處理非 UTF-8 資料
                        console.print(f"[dim yellow][RAW] {data.hex()}[/dim yellow]")
            
            except serial.SerialException as e:
                console.print(f"\n[red]✗[/red] 序列埠錯誤: {e}")
                self.running = False
                break
    
    def write_serial(self, data):
        """
        寫入資料到序列埠
        
        Args:
            data (str): 要傳送的資料
        """
        try:
            self.serial_conn.write(data.encode('utf-8'))
            self.serial_conn.flush()
            
            # 顯示已傳送的訊息
            send_text = Text()
            send_text.append("➤ ", style="bold green")
            send_text.append(f"已傳送: {data.strip()}", style="green")
            console.print(send_text)
        except serial.SerialException as e:
            console.print(f"[red]✗[/red] 寫入失敗: {e}")
    
    def start(self, enable_input=True, log_to_file=False):
        """
        開始監測序列埠
        
        Args:
            enable_input (bool): 是否啟用使用者輸入
            log_to_file (bool): 是否記錄到檔案
        """
        if not self.connect():
            return
        
        if log_to_file:
            self.enable_logging()
        
        self.running = True
        
        # 啟動讀取執行緒
        read_thread = threading.Thread(target=self.read_serial, daemon=True)
        read_thread.start()
        
        # 顯示監測資訊面板
        console.print()
        info_text = Text()
        info_text.append("序列埠: ", style="cyan")
        info_text.append(f"{self.port}\n", style="magenta")
        info_text.append("鮑率: ", style="cyan")
        info_text.append(f"{self.baudrate}\n", style="yellow")
        
        if enable_input:
            info_text.append("\n", style="white")
            info_text.append("💡 ", style="yellow")
            info_text.append("輸入訊息並按 Enter 傳送\n", style="white")
        
        info_text.append("⚠️  ", style="red")
        info_text.append("按 Ctrl+C 結束監測", style="white dim")
        
        console.print(Panel(info_text, title="[bold green]🔍 序列埠監測中[/bold green]", border_style="green"))
        console.print()
        
        try:
            if enable_input:
                # 主執行緒處理使用者輸入
                while self.running:
                    user_input = input()
                    if user_input:
                        self.write_serial(user_input + '\n')
            else:
                # 等待直到被中斷
                while self.running:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            console.print("\n")
            console.print("[yellow]⚠[/yellow] 收到中斷訊號，正在關閉...")
        
        finally:
            self.running = False
            read_thread.join(timeout=2)
            
            # 顯示統計
            stats = Panel(
                f"[cyan]共接收 [bold]{self.message_count}[/bold] 條訊息[/cyan]",
                border_style="cyan",
                padding=(0, 2)
            )
            console.print(stats)
            
            self.disconnect()


def monitor_serial(port, baudrate=9600, enable_input=True, log_to_file=False):
    """
    便利函式：開始監測序列埠
    
    Args:
        port (str): 序列埠位置
        baudrate (int): 鮑率
        enable_input (bool): 是否啟用使用者輸入
        log_to_file (bool): 是否記錄到檔案
    """
    monitor = SerialMonitor(port, baudrate)
    monitor.start(enable_input, log_to_file)


if __name__ == '__main__':
    # 測試模組
    if len(sys.argv) < 2:
        console.print("[yellow]使用方式:[/yellow] python monitor.py <port> [baudrate]")
        sys.exit(1)
    
    port = sys.argv[1]
    baudrate = int(sys.argv[2]) if len(sys.argv) > 2 else 9600
    
    monitor_serial(port, baudrate, enable_input=True, log_to_file=False)
