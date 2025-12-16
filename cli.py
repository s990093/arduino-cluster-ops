#!/usr/bin/env python3
"""
Arduino 開發整合式 CLI 工具（美化版）
提供裝置偵測、程式燒入、序列埠監測等功能，使用 Rich 美化終端輸出
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

from arduino_tools.board_detector import detect_arduino_boards, print_board_info
from arduino_tools.uploader import compile_and_upload, get_fqbn_from_board_name
from arduino_tools.monitor import monitor_serial

console = Console()


@click.group()
@click.version_option(version='1.0.0', prog_name='Arduino CLI Tools')
def cli():
    """
    🚀 Arduino 開發工具集
    
    提供 Arduino 裝置偵測、程式編譯燒入、序列埠監測等功能
    """
    pass


@cli.command()
def detect():
    """偵測連接的 Arduino 裝置"""
    boards_info = detect_arduino_boards()
    print_board_info(boards_info)


@cli.command()
@click.argument('sketch_path', type=click.Path(exists=True))
@click.option('--port', '-p', required=True, help='序列埠位置 (例如: /dev/cu.usbserial-xxx)')
@click.option('--board', '-b', default='uno', help='板子類型 (uno, nano, mega 等)')
@click.option('--fqbn', '-f', default=None, help='完整的 FQBN (覆寫 --board 選項)')
@click.option('--verbose', '-v', is_flag=True, help='顯示詳細輸出')
def upload(sketch_path, port, board, fqbn, verbose):
    """
    編譯並上傳 Arduino 程式
    
    範例:
        python3 cli.py upload examples/blink/blink.ino --port /dev/cu.usbserial-1234
    """
    # 決定使用的 FQBN
    if fqbn is None:
        fqbn = get_fqbn_from_board_name(board)
    
    success = compile_and_upload(sketch_path, port, fqbn, verbose)
    
    if success:
        console.print()
        hint = Text()
        hint.append("💡 提示: ", style="bold yellow")
        hint.append("使用以下指令監測序列埠輸出:\n", style="white")
        hint.append(f"   python3 cli.py monitor --port {port}", style="cyan")
        console.print(Panel(hint, border_style="yellow"))
        console.print()
        sys.exit(0)
    else:
        sys.exit(1)


@cli.command()
@click.option('--port', '-p', required=True, help='序列埠位置')
@click.option('--baudrate', '-b', default=9600, help='鮑率 (預設: 9600)')
@click.option('--no-input', is_flag=True, help='只顯示輸出，不接受輸入')
@click.option('--log', '-l', is_flag=True, help='記錄資料到檔案')
def monitor(port, baudrate, no_input, log):
    """
    監測 Arduino 序列埠輸出
    
    範例:
        python3 cli.py monitor --port /dev/cu.usbserial-1234 --baudrate 9600
    """
    enable_input = not no_input
    monitor_serial(port, baudrate, enable_input, log)


@cli.command()
@click.argument('sketch_path', type=click.Path(exists=True))
@click.option('--port', '-p', default=None, help='序列埠位置 (不指定則自動偵測)')
@click.option('--board', '-b', default='uno', help='板子類型 (uno, nano, mega 等)')
@click.option('--baudrate', '-r', default=9600, help='監測鮑率 (預設: 9600)')
@click.option('--fqbn', '-f', default=None, help='完整的 FQBN')
@click.option('--verbose', '-v', is_flag=True, help='顯示詳細輸出')
def flash_and_monitor(sketch_path, port, board, baudrate, fqbn, verbose):
    """
    一鍵燒入並監測 (編譯 → 上傳 → 監測)
    
    範例:
        python3 cli.py flash-and-monitor examples/blink/blink.ino
    """
    # 如果沒有指定 port，嘗試自動偵測
    if port is None:
        console.print("[cyan]🔍 未指定序列埠，正在自動偵測...[/cyan]")
        boards_info = detect_arduino_boards()
        
        # 找到第一個真正的 Arduino
        arduino_port = None
        for board_info in boards_info.get('boards', []):
            if board_info['boards']:
                arduino_port = board_info['port']
                break
        
        if not arduino_port:
            console.print("[red]✗[/red] 未偵測到 Arduino 裝置")
            sys.exit(1)
        
        port = arduino_port
        console.print(f"[green]✓[/green] 使用偵測到的序列埠: [magenta]{port}[/magenta]")
        console.print()
    
    # 決定使用的 FQBN
    if fqbn is None:
        fqbn = get_fqbn_from_board_name(board)
    
    # 燒入程式
    success = compile_and_upload(sketch_path, port, fqbn, verbose)
    
    if not success:
        console.print("[red]✗[/red] 燒入失敗，取消監測")
        sys.exit(1)
    
    # 稍微等待 Arduino 重啟
    import time
    console.print("[cyan]⏳ 等待 Arduino 重啟...[/cyan]")
    time.sleep(2)
    console.print()
    
    # 開始監測
    switch_panel = Panel(
        "[bold cyan]🔄 自動切換到序列埠監測模式[/bold cyan]",
        border_style="cyan",
        padding=(0, 2)
    )
    console.print(switch_panel)
    console.print()
    
    monitor_serial(port, baudrate, enable_input=True, log_to_file=False)


@cli.command()
def list_boards():
    """列出支援的板子類型"""
    from arduino_tools.uploader import BOARD_FQBN
    
    console.print()
    
    # 建立表格
    table = Table(title="[bold cyan]📋 支援的 Arduino 板子類型[/bold cyan]",
                 border_style="cyan",
                 show_header=True,
                 header_style="bold yellow")
    
    table.add_column("板子名稱", style="green", no_wrap=True)
    table.add_column("FQBN", style="white")
    table.add_column("說明", style="cyan dim")
    
    board_descriptions = {
        'uno': 'Arduino Uno',
        'nano': 'Arduino Nano (新版)',
        'nano_old': 'Arduino Nano (舊版 bootloader)',
        'mega': 'Arduino Mega',
        'mega2560': 'Arduino Mega 2560',
        'leonardo': 'Arduino Leonardo',
        'micro': 'Arduino Micro',
        'mini': 'Arduino Mini',
        'esp32': 'ESP32 Dev Module',
        'ttgo': 'TTGO T-Display ESP32',
        'ttgo_tdisplay': 'TTGO T-Display (1.14" LCD)',
    }
    
    for board_name, fqbn in BOARD_FQBN.items():
        desc = board_descriptions.get(board_name, '')
        table.add_row(board_name, fqbn, desc)
    
    console.print(table)
    console.print()
    
    # 使用提示
    hint = Text()
    hint.append("💡 使用方式:\n", style="bold yellow")
    hint.append("  --board ", style="cyan")
    hint.append("<板子名稱>", style="green")
    hint.append(" 或 ", style="white")
    hint.append("--fqbn ", style="cyan")
    hint.append("<完整FQBN>", style="white")
    
    console.print(Panel(hint, border_style="yellow"))
    console.print()


@cli.command()
@click.option('--port', '-p', default=None, help='序列埠位置 (不指定則自動偵測)')
@click.option('--baudrate', '-b', default=1000000, help='鮑率 (預設: 1000000)')
@click.option('--test', '-t', type=click.Choice(['basic', 'conv2d', 'both']), default='both', 
              help='測試類型: basic (基本讀寫), conv2d (卷積), both (兩者)')
def test_serial(port, baudrate, test):
    """
    測試序列通訊 (讀寫 VRAM)
    
    範例:
        python3 cli.py test-serial --port /dev/cu.usbmodem11401
        python3 cli.py test-serial --test basic
    """
    from simple_serial_api import SimpleSerialAPI
    import numpy as np
    import time
    
    # 如果沒有指定 port，嘗試自動偵測
    if port is None:
        console.print("[cyan]🔍 未指定序列埠，正在自動偵測...[/cyan]")
        boards_info = detect_arduino_boards()
        
        # 找到第一個真正的 Arduino
        arduino_port = None
        for board_info in boards_info.get('boards', []):
            if board_info['boards']:
                arduino_port = board_info['port']
                break
        
        if not arduino_port:
            console.print("[red]✗[/red] 未偵測到 Arduino 裝置")
            sys.exit(1)
        
        port = arduino_port
        console.print(f"[green]✓[/green] 使用偵測到的序列埠: [magenta]{port}[/magenta]")
        console.print()
    
    # 建立 API 連線
    console.print(f"[cyan]📡 連接到 {port} @ {baudrate} baud...[/cyan]")
    api = SimpleSerialAPI(port=port, baudrate=baudrate)
    
    try:
        api.connect()
        console.print("[green]✓[/green] 連線成功!\n")
        
        # 基本測試
        if test in ['basic', 'both']:
            console.print(Panel("[bold cyan]測試 1: 基本讀寫測試[/bold cyan]", border_style="cyan"))
            
            test_data = np.array([1, 2, 3, 4, 5, 10, 20, 30], dtype=np.uint8)
            console.print(f"  寫入資料: {test_data.tolist()}")
            
            if api.write_vram(0, 0, test_data):
                console.print("  [green]✓[/green] 寫入成功")
                
                # 讀取回來驗證
                time.sleep(0.01)
                read_data = api.read_vram(0, 0, len(test_data))
                console.print(f"  讀取資料: {read_data.tolist()}")
                
                if np.array_equal(test_data, read_data):
                    console.print("  [green]✓ 驗證成功！資料完全一致[/green]\n")
                else:
                    console.print("  [red]✗ 驗證失敗！資料不一致[/red]")
                    console.print(f"    期望: {test_data.tolist()}")
                    console.print(f"    實際: {read_data.tolist()}\n")
            else:
                console.print("  [red]✗[/red] 寫入失敗\n")
        
        # Conv2D 測試
        if test in ['conv2d', 'both']:
            console.print(Panel("[bold cyan]測試 2: Conv2D 完整流程[/bold cyan]", border_style="cyan"))
            
            # 8x8 測試影像
            input_img = np.array([
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 200, 200, 200, 200, 100, 100],
                [100, 100, 200, 200, 200, 200, 100, 100],
                [100, 100, 200, 200, 200, 200, 100, 100],
                [100, 100, 200, 200, 200, 200, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
                [100, 100, 100, 100, 100, 100, 100, 100],
            ], dtype=np.uint8)
            
            # 3x3 Sobel 垂直邊緣偵測
            kernel = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ], dtype=np.int8)
            
            console.print("  輸入: 8x8 影像 (中間有亮區)")
            console.print("  卷積核: Sobel 垂直邊緣偵測 (3x3)")
            console.print("  執行中...\n")
            
            result = api.full_conv2d(input_img, kernel, slave_id=0)
            
            if result is not None:
                console.print("\n  [green]✓ Conv2D 測試成功！[/green]")
                console.print("\n  輸出結果 (6x6):")
                
                # 創建結果表格
                result_table = Table(show_header=False, border_style="green", padding=0)
                for i in range(6):
                    result_table.add_column(justify="right")
                
                for row in result:
                    result_table.add_row(*[str(val) for val in row])
                
                console.print(result_table)
                console.print()
            else:
                console.print("\n  [red]✗ Conv2D 測試失敗！[/red]\n")
        
        console.print("[bold green]🎉 所有測試完成！[/bold green]")
        
    except Exception as e:
        console.print(f"[red]✗ 測試過程發生錯誤: {e}[/red]")
        sys.exit(1)
    finally:
        api.disconnect()



if __name__ == '__main__':
    cli()

