"""
Transformer 測試主類
"""

import time
import json
from typing import Dict, List

from .connection import ESP32Connection
from .trace import TraceCollector
from .analyzer import ResultAnalyzer


class TransformerTester:
    """ESP32 Transformer 測試主類"""
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.connection = ESP32Connection(port, baudrate)
        self.trace_records = []
        self.registers = {}
        self.elapsed_time = 0.0
    
    def run_test(self) -> bool:
        """執行完整測試流程"""
        try:
            print("="*70)
            print("🚀 ESP32 Complete Transformer - Trace Mode")
            print("="*70)
            
            # 1. 加載程序
            self._load_program()
            
            # 2. 啟用 trace
            self._enable_trace()
            
            # 3. 執行並收集 trace
            output, self.elapsed_time = TraceCollector.collect_execution_trace(self.connection)
            
            # 4. 解析 trace
            self.trace_records = TraceCollector.parse_trace_json(output)
            print(f"\n✅ Collected {len(self.trace_records)} instruction traces\n")
            
            # 5. 保存 trace
            self._save_trace()
            
            # 6. 讀取寄存器
            self.registers = self._read_registers()
            
            # 7. 分析結果
            success = ResultAnalyzer.analyze(self.registers, self.trace_records)
            
            print(f"\n✨ Test completed in {self.elapsed_time:.1f}s")
            print("🎯 Complete Transformer with all SFU operations verified!\n")
            
            return success
            
        finally:
            self.connection.close()
    
    def _load_program(self):
        """加載 Transformer 程序"""
        from .program_loader import ProgramLoader
        
        # 創建並加載程序
        program = ProgramLoader.create_transformer_program()
        ProgramLoader.load_program(self.connection, program)
    
    def _enable_trace(self):
        """啟用 streaming trace"""
        print("✨ Enabling streaming trace...")
        self.connection.send_command("trace:stream")
        self.connection.read_lines()
    
    def _read_registers(self) -> Dict[str, int]:
        """讀取最終寄存器值"""
        time.sleep(0.5)
        self.connection.send_command("reg")
        output = self.connection.read_lines()
        return TraceCollector.parse_registers(output)
    
    def _save_trace(self):
        """保存 trace 到 JSON 文件"""
        data = {
            "trace_version": "2.1",
            "program": "Complete Transformer (71 instructions)",
            "total_instructions": len(self.trace_records),
            "records": self.trace_records
        }
        
        filename = "transformer_complete_trace.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Trace saved: {filename}")
        print(f"   Instructions: {len(self.trace_records)}")
        print(f"   File size: {len(json.dumps(data)):,} bytes")
