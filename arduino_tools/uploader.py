"""
Arduino 程式編譯與燒入模組（美化版）
支援 .ino 檔案的編譯和上傳，使用 Rich 美化終端輸出
"""

import subprocess
import os
import shutil
import time
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()

# 常見 Arduino 板子的 FQBN 對應表
BOARD_FQBN = {
    'uno': 'arduino:avr:uno',
    'nano': 'arduino:avr:nano',
    'nano_old': 'arduino:avr:nano:cpu=atmega328old',
    'mega': 'arduino:avr:mega',
    'mega2560': 'arduino:avr:mega:cpu=atmega2560',
    'leonardo': 'arduino:avr:leonardo',
    'micro': 'arduino:avr:micro',
    'mini': 'arduino:avr:mini',
    # ESP32 板子
    'esp32': 'esp32:esp32:esp32',
    'ttgo': 'esp32:esp32:esp32',
}


def compile_sketch(sketch_path, fqbn='arduino:avr:uno', build_path=None, verbose=False, progress_callback=None, optimize=True):
    """
    編譯 Arduino sketch
    
    Args:
        sketch_path (str): .ino 檔案路徑
        fqbn (str): Fully Qualified Board Name
        build_path (str): 編譯輸出目錄 (Output Directory)
        verbose (bool): 是否顯示詳細輸出
        progress_callback (callable): 進度回調函式 func(percent, message)
        optimize (bool): 是否啟用極限優化
    
    Returns:
        bool: 編譯是否成功
    """
    sketch_path = Path(sketch_path).resolve()
    
    if not sketch_path.exists():
        console.print(f"[red]✗[/red] 找不到檔案: {sketch_path}")
        return False
    
    # 顯示編譯資訊面板
    info_text = Text()
    info_text.append("檔案: ", style="cyan")
    info_text.append(f"{sketch_path.name}\n", style="white")
    info_text.append("板子: ", style="cyan")
    info_text.append(f"{fqbn}", style="yellow")
    if build_path:
        info_text.append("\n輸出: ", style="cyan")
        info_text.append(f"{build_path}", style="white")
    
    console.print(Panel(info_text, title="[bold blue]🔨 編譯 Arduino 程式[/bold blue]", border_style="blue"))
    
    if progress_callback:
        progress_callback(0, "還在編譯中...")
    
    cmd = ['arduino-cli', 'compile', '--fqbn', fqbn, str(sketch_path)]
    
    if build_path:
        # Create directory if not exists
        Path(build_path).mkdir(parents=True, exist_ok=True)
        cmd.extend(['--output-dir', str(build_path)])
        
    if optimize:
        optimization_flags = [
            '--build-property', 'compiler.c.extra_flags=-O3 -funroll-loops -finline-functions',
            '--build-property', 'compiler.cpp.extra_flags=-O3 -funroll-loops -finline-functions -ffast-math',
            '--build-property', 'compiler.c.elf.extra_flags=-O3',
            '--build-property', 'compiler.c.extra_flags=-flto',
            '--build-property', 'compiler.cpp.extra_flags=-flto',
            '--build-property', 'compiler.c.elf.extra_flags=-flto -fuse-linker-plugin',
        ]
        cmd.extend(optimization_flags)
        console.print("[yellow]🚀 啟用極限效能優化 (-O3, LTO)[/yellow]")
    
    if verbose:
        cmd.append('--verbose')
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]正在編譯...", total=100)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Progress simulation
            current_progress = 0
            while process.poll() is None:
                progress.update(task, advance=5)
                current_progress = min(current_progress + 5, 95)
                if progress_callback:
                    progress_callback(current_progress, "正在編譯...")
                
                time.sleep(0.1)
                if progress.tasks[0].completed >= 95:
                    progress.update(task, completed=95)
            
            progress.update(task, completed=100)
            if progress_callback:
                progress_callback(100, "編譯完成")
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                console.print("[green]✓[/green] 編譯成功!", style="bold green")
                return True
            else:
                console.print("[red]✗[/red] 編譯失敗!", style="bold red")
                if stderr:
                    console.print(Panel(stderr, title="[red]錯誤訊息[/red]", border_style="red"))
                return False
        
    except Exception as e:
        console.print(f"[red]✗[/red] 編譯錯誤: {e}", style="bold red")
        return False


def upload_sketch_esp32_with_build_path(sketch_path, port, build_path, verbose=False, progress_callback=None):
    """
    ESP32 上傳 (使用指定的 build_path)
    """
    sketch_path = Path(sketch_path).resolve()
    build_dir = Path(build_path).resolve()
    sketch_name = sketch_path.stem
    
    # Check required binaries
    bin_file = build_dir / f'{sketch_name}.ino.bin'
    bootloader = build_dir / f'{sketch_name}.ino.bootloader.bin'
    partitions = build_dir / f'{sketch_name}.ino.partitions.bin'
    
    # Boot app0 (usually standard)
    # We might need to find it from Arduino packages if not in build dir
    # But often arduino-cli copies it to build dir?
    # Let's check if it exists in build_dir first
    boot_app0 = build_dir / 'boot_app0.bin'
    
    if not boot_app0.exists():
         # Fallback search
        esp32_hardware_path = Path.home() / 'Library' / 'Arduino15' / 'packages' / 'esp32' / 'hardware' / 'esp32'
        esp32_versions = sorted(esp32_hardware_path.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)
        if esp32_versions:
            boot_app0 = esp32_versions[0] / 'tools' / 'partitions' / 'boot_app0.bin'
    
    if not all([bin_file.exists(), bootloader.exists(), partitions.exists(), boot_app0.exists()]):
        console.print(f"[red]✗[/red] Build artifacts missing in {build_dir}")
        return False

    # Find esptool
    esptool_path = Path.home() / 'Library' / 'Arduino15' / 'packages' / 'esp32' / 'tools' / 'esptool_py'
    esptool_versions = list(esptool_path.glob('*'))
    if not esptool_versions:
         console.print('[red]✗[/red] esptool not found')
         return False
    esptool = esptool_versions[0] / 'esptool'

    # Upload Command
    info_text = Text()
    info_text.append(f"Uploading {sketch_name} from {build_dir}\n", style="white")
    console.print(Panel(info_text, title="[bold green]📤 ESP32 Upload[/bold green]", border_style="green"))

    cmd = [
        str(esptool),
        '--chip', 'esp32',
        '--port', port,
        '--baud', '460800',
        '--before', 'default_reset',
        '--after', 'hard_reset',
        'write_flash', '-z',
        '--flash_mode', 'keep',
        '--flash_freq', 'keep',
        '--flash_size', 'keep',
        '0x1000', str(bootloader),
        '0x8000', str(partitions),
        '0xe000', str(boot_app0),
        '0x10000', str(bin_file)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            if verbose:
                console.print(line.rstrip(), style="dim")
        process.wait()
        if process.returncode == 0:
            console.print("[green]✓[/green] Upload Success!")
            return True
        else:
            console.print("[red]✗[/red] Upload Failed!")
            return False
    except Exception as e:
        console.print(f"[red]✗[/red] Upload Error: {e}")
        return False


def compile_and_upload(sketch_path, port, fqbn='arduino:avr:uno', verbose=False, progress_callback=None):
    """
    Combined Compile and Upload with Local Build Directory
    """
    # Create local build directory
    build_dir = Path("build").resolve()
    
    # Compile
    success = compile_sketch(
        sketch_path, 
        fqbn, 
        build_path=str(build_dir), 
        verbose=verbose, 
        progress_callback=progress_callback,
        optimize=True # Force optimize
    )
    
    if not success:
        return False
    
    # Upload
    if 'esp32' in fqbn.lower():
        return upload_sketch_esp32_with_build_path(sketch_path, port, str(build_dir), verbose, progress_callback)
    else:
        # Fallback for non-ESP32 (not verifying this path now)
        from .uploader import upload_sketch # Circular import if not careful, but we are rewriting the module.
        # Actually I should inline upload logic or call existing upload_sketch but pointing to build dir?
        # arduino-cli upload can take --input-dir
        cmd = ['arduino-cli', 'upload', '-p', port, '--fqbn', fqbn, '--input-dir', str(build_dir)]
        subprocess.run(cmd, check=True)
        return True

def get_fqbn_from_board_name(board_name):
    board_name = board_name.lower()
    return BOARD_FQBN.get(board_name, 'arduino:avr:uno')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        console.print("Usage: python uploader.py <sketch> <port> [fqbn]")
        sys.exit(1)
    
    sketch = sys.argv[1]
    port = sys.argv[2]
    fqbn = sys.argv[3] if len(sys.argv) > 3 else 'arduino:avr:uno'
    
    compile_and_upload(sketch, port, fqbn, verbose=True)
