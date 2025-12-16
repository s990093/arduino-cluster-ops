"""
ESP32 CUDA 通用執行器
提供簡潔的 API，直接寫程式碼 → 編譯 → 執行 → 查看 trace
"""

import time
import json
from typing import List, Dict, Optional, Tuple
from .connection import ESP32Connection
from .program_loader import Instruction
from .trace import TraceCollector


class CUDARunner:
    """
    ESP32 CUDA 通用執行器
    
    使用範例:
        runner = CUDARunner("/dev/cu.usbserial-589A0095521")
        
        program = [
            Instruction.mov(0, 5),
            Instruction.imul(1, 0, 0),
            Instruction.exit_inst()
        ]
        
        trace = runner.run(program)
        runner.print_results()
    """
    
    def __init__(self, port: str, baudrate: int = 115200):
        """
        初始化執行器
        
        Args:
            port: ESP32 串口路徑
            baudrate: 波特率
        """
        self.port = port
        self.baudrate = baudrate
        self.conn = None
        self.trace_records = []
        self.registers = {}
        self.elapsed_time = 0.0
        self._auto_close = True
    
    def __enter__(self):
        """Context manager 支持"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 支持"""
        self.disconnect()
    
    def connect(self):
        """連接到 ESP32"""
        if self.conn is None:
            print(f"🔌 Connecting to {self.port}...")
            self.conn = ESP32Connection(self.port, self.baudrate)
            print("✅ Connected")
    
    def disconnect(self):
        """斷開連接"""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            print("🔌 Disconnected")
    
    def compile_and_load(self, program: List[Instruction]) -> None:
        """
        編譯並加載程序到 ESP32
        
        Args:
            program: 指令列表
        """
        self.connect()
        
        print(f"\n📋 Loading {len(program)} instructions...")
        
        for i, inst in enumerate(program):
            hex_str = inst.to_hex()
            self.conn.send_command(f"load {hex_str}", delay=0.05)
            response = self.conn.read_lines()
            
            # 簡化輸出
            if i == 0 or i == len(program) - 1:
                print(f"  [{i}] {hex_str}")
        
        print(f"✅ Program loaded\n")
    
    def execute(self, enable_trace: bool = True) -> Tuple[List[dict], float]:
        """
        執行程序
        
        Args:
            enable_trace: 是否啟用 trace
            
        Returns:
            (trace_records, elapsed_time)
        """
        self.connect()
        
        if enable_trace:
            print("✨ Enabling trace...")
            self.conn.send_command("trace:stream")
            self.conn.read_lines()
        
        print("🔄 Running program...\n")
        
        # 執行並收集 trace
        output, elapsed = TraceCollector.collect_execution_trace(self.conn)
        
        # 解析 trace
        trace_records = TraceCollector.parse_trace_json(output) if enable_trace else []
        
        print(f"✅ Execution completed in {elapsed:.2f}s")
        if trace_records:
            print(f"📊 Collected {len(trace_records)} instruction traces\n")
        
        return trace_records, elapsed
    
    def read_registers(self) -> Dict[str, int]:
        """
        讀取最終寄存器值
        
        Returns:
            寄存器字典
        """
        self.connect()
        
        time.sleep(0.3)
        self.conn.send_command("reg")
        output = self.conn.read_lines()
        return TraceCollector.parse_registers(output)
    
    def run(self, 
            program: List[Instruction], 
            enable_trace: bool = True,
            save_trace: Optional[str] = None) -> List[dict]:
        """
        一鍵執行：編譯 → 加載 → 執行 → 讀取結果
        
        Args:
            program: 指令列表
            enable_trace: 是否啟用 trace
            save_trace: trace 保存文件名（可選）
            
        Returns:
            trace_records
        """
        print("=" * 70)
        print("🚀 ESP32 CUDA Program Execution")
        print("=" * 70)
        
        # 1. 編譯並加載
        self.compile_and_load(program)
        
        # 2. 執行
        self.trace_records, self.elapsed_time = self.execute(enable_trace)
        
        # 3. 讀取寄存器
        self.registers = self.read_registers()
        
        # 4. 保存 trace（如果需要）
        if save_trace and self.trace_records:
            self.save_trace(save_trace)
        
        return self.trace_records
    
    def save_trace(self, filename: str = "trace.json") -> None:
        """
        保存 trace 到文件
        
        Args:
            filename: 文件名
        """
        data = {
            "trace_version": "2.2",
            "total_instructions": len(self.trace_records),
            "elapsed_time": self.elapsed_time,
            "records": self.trace_records
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Trace saved: {filename}")
        print(f"   Instructions: {len(self.trace_records)}")
        print(f"   File size: {len(json.dumps(data)):,} bytes\n")
    
    def print_results(self, show_all: bool = False) -> None:
        """
        打印執行結果
        
        Args:
            show_all: 是否顯示所有寄存器（默認只顯示非零）
        """
        print("=" * 70)
        print("📊 Execution Results")
        print("=" * 70)
        
        if not self.registers:
            print("⚠️  No register data available")
            return
        
        # 找出非零寄存器
        non_zero = {k: v for k, v in self.registers.items() if v != 0}
        
        if show_all:
            regs_to_show = self.registers
            print("All Registers:")
        else:
            regs_to_show = non_zero
            print("Non-Zero Registers:")
        
        if not regs_to_show:
            print("  (all registers are zero)")
        else:
            # 按寄存器編號排序
            sorted_regs = sorted(regs_to_show.items(), 
                               key=lambda x: int(x[0][1:]) if x[0].startswith('R') else 999)
            
            for reg, val in sorted_regs:
                print(f"  {reg:<6} = {val:>10}")
        
        print(f"\nElapsed Time: {self.elapsed_time:.2f}s")
        print("=" * 70 + "\n")
    
    def print_trace_summary(self, max_lines: int = 10) -> None:
        """
        打印 trace 摘要
        
        Args:
            max_lines: 最多顯示多少條 trace
        """
        if not self.trace_records:
            print("⚠️  No trace data available")
            return
        
        print("=" * 70)
        print(f"📋 Trace Summary (showing {min(max_lines, len(self.trace_records))} of {len(self.trace_records)})")
        print("=" * 70)
        
        for i, record in enumerate(self.trace_records[:max_lines]):
            cycle = record.get('cycle', '?')
            pc = record.get('pc', '?')
            inst = record.get('instruction', '?')
            print(f"[{i}] Cycle {cycle:>4}, PC {pc:>3}, Inst: {inst}")
        
        if len(self.trace_records) > max_lines:
            print(f"... and {len(self.trace_records) - max_lines} more")
        
        print("=" * 70 + "\n")
    
    def verify_result(self, expected: Dict[str, int]) -> bool:
        """
        驗證執行結果
        
        Args:
            expected: 預期的寄存器值
            
        Returns:
            是否通過驗證
        """
        print("=" * 70)
        print("🔍 Verifying Results")
        print("=" * 70)
        
        all_passed = True
        
        for reg, expected_val in expected.items():
            actual_val = self.registers.get(reg, 0)
            passed = (actual_val == expected_val)
            
            status = "✅" if passed else "❌"
            print(f"{status} {reg:<6} Expected: {expected_val:>6}, Actual: {actual_val:>6}")
            
            if not passed:
                all_passed = False
        
        print("=" * 70)
        if all_passed:
            print("🎉 All verifications passed!")
        else:
            print("⚠️  Some verifications failed")
        print("=" * 70 + "\n")
        
        return all_passed


# ===== 便捷函數 =====

def quick_run(port: str, 
              program: List[Instruction],
              expected: Optional[Dict[str, int]] = None,
              save_trace: Optional[str] = None) -> bool:
    """
    快速執行：一個函數搞定所有
    
    Args:
        port: ESP32 串口
        program: 指令列表
        expected: 預期結果（可選）
        save_trace: trace 文件名（可選）
        
    Returns:
        是否通過驗證（如果提供了 expected）
    """
    with CUDARunner(port) as runner:
        # 執行
        runner.run(program, save_trace=save_trace)
        
        # 顯示結果
        runner.print_results()
        
        # 驗證（如果提供）
        if expected:
            return runner.verify_result(expected)
        
        return True
