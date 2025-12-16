"""
Arduino 板子偵測模組（美化版）
自動掃描並識別連接的 Arduino 裝置，使用 Rich 美化輸出
"""

import subprocess
import json
import serial.tools.list_ports
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()


def get_boards_via_cli():
    """
    使用 Arduino CLI 偵測連接的板子
    
    Returns:
        list: 包含板子資訊的字典列表
    """
    try:
        result = subprocess.run(
            ['arduino-cli', 'board', 'list', '--format', 'json'],
            capture_output=True,
            text=True,
            check=True
        )
        
        data = json.loads(result.stdout)
        boards = []
        
        for board in data.get('detected_ports', []):
            board_info = {
                'port': board.get('port', {}).get('address', 'Unknown'),
                'protocol': board.get('port', {}).get('protocol', 'serial'),
                'boards': []
            }
            
            # 取得可能的板子類型
            matching_boards = board.get('matching_boards', [])
            for mb in matching_boards:
                board_info['boards'].append({
                    'name': mb.get('name', 'Unknown'),
                    'fqbn': mb.get('fqbn', '')
                })
            
            boards.append(board_info)
        
        return boards
    
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗[/red] Arduino CLI 執行失敗: {e}")
        return []
    except json.JSONDecodeError as e:
        console.print(f"[red]✗[/red] 解析 JSON 失敗: {e}")
        return []


def get_serial_ports():
    """
    使用 pyserial 列出所有序列埠
    
    Returns:
        list: 序列埠資訊列表
    """
    ports = serial.tools.list_ports.comports()
    port_list = []
    
    for port in ports:
        port_info = {
            'device': port.device,
            'name': port.name,
            'description': port.description,
            'hwid': port.hwid,
            'manufacturer': port.manufacturer or 'Unknown'
        }
        port_list.append(port_info)
    
    return port_list


def detect_arduino_boards():
    """
    綜合偵測 Arduino 板子
    結合 Arduino CLI 和 pyserial 的結果
    
    Returns:
        dict: 包含 boards 和 serial_ports 的字典
    """
    boards = get_boards_via_cli()
    serial_ports = get_serial_ports()
    
    return {
        'boards': boards,
        'serial_ports': serial_ports
    }


def print_board_info(boards_info):
    """
    格式化輸出板子資訊（美化版）
    
    Args:
        boards_info (dict): detect_arduino_boards() 的回傳值
    """
    console.print()
    
    # 主標題
    title = Text("Arduino 裝置偵測", style="bold white")
    console.print(Panel(title, border_style="bold cyan", padding=(0, 2)))
    console.print()
    
    boards = boards_info.get('boards', [])
    
    if boards:
        # 建立 Arduino 板子表格
        arduino_table = Table(title="[bold green]✓ 偵測到的 Arduino 板子[/bold green]", 
                             border_style="green", 
                             show_header=True,
                             header_style="bold cyan")
        
        arduino_table.add_column("#", style="dim", width=4)
        arduino_table.add_column("序列埠", style="magenta", no_wrap=True)
        arduino_table.add_column("板子類型", style="yellow")
        arduino_table.add_column("FQBN", style="white dim")
        
        arduino_count = 0
        for i, board in enumerate(boards, 1):
            port = board['port']
            
            if board['boards']:
                # 有識別到 Arduino
                for b in board['boards']:
                    arduino_count += 1
                    arduino_table.add_row(
                        str(arduino_count),
                        port,
                        f"[green]{b['name']}[/green]",
                        b['fqbn']
                    )
            else:
                # 序列埠但未識別為 Arduino - 不顯示在 Arduino 表格中
                pass
        
        if arduino_count > 0:
            console.print(arduino_table)
            console.print()
    else:
        console.print(Panel("[yellow]⚠ 未偵測到 Arduino 板子[/yellow]", border_style="yellow"))
        console.print()
    
    # 所有序列埠表格（簡化版）
    serial_ports = boards_info.get('serial_ports', [])
    if serial_ports:
        # 過濾出真正的 USB/Arduino 裝置
        usb_ports = [p for p in serial_ports if 'usb' in p['device'].lower() or 'arduino' in p['description'].lower()]
        
        if usb_ports:
            port_table = Table(title="[bold blue]📌 USB 序列埠[/bold blue]", 
                             border_style="blue",
                             show_header=True,
                             header_style="bold cyan")
            
            port_table.add_column("#", style="dim", width=4)
            port_table.add_column("裝置", style="cyan", no_wrap=True)
            port_table.add_column("描述", style="white")
            port_table.add_column("製造商", style="yellow")
            
            for i, port in enumerate(usb_ports, 1):
                # 根據描述判斷是否為 Arduino
                if 'arduino' in port['description'].lower():
                    desc_style = "[green]" + port['description'] + "[/green]"
                else:
                    desc_style = port['description']
                
                port_table.add_row(
                    str(i),
                    port['device'],
                    desc_style,
                    port['manufacturer']
                )
            
            console.print(port_table)
            console.print()
    
    # 底部提示
    if boards:
        first_board = boards[0]
        # 找第一個真正的 Arduino
        arduino_port = None
        for board in boards:
            if board['boards']:
                arduino_port = board['port']
                break
        
        if arduino_port:
            hint = Text()
            hint.append("💡 下一步: ", style="bold yellow")
            hint.append(f"python3 cli.py upload examples/blink/blink.ino --port {arduino_port}", style="cyan")
            console.print(Panel(hint, border_style="yellow"))
            console.print()


if __name__ == '__main__':
    # 測試模組
    info = detect_arduino_boards()
    print_board_info(info)
