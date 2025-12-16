"""
ESP32 Trace 收集和解析工具
"""

import time
import json
from typing import List, Dict, Tuple


class TraceCollector:
    """收集和解析 ESP32 執行 Trace"""
    
    SKIP_PATTERNS = ['>', '===', 'VM', 'Loaded', '✅', 'Total']
    
    @staticmethod
    def collect_execution_trace(connection, max_wait: float = 30) -> Tuple[List[str], float]:
        """收集執行 trace 輸出"""
        print("⚡ Executing on Core1...\n")
        start_time = time.time()
        connection.ser.write("run\n".encode())
        
        all_output = []
        execution_complete = False
        
        while not execution_complete and (time.time() - start_time) < max_wait:
            time.sleep(0.3)
            
            new_lines = connection.read_lines()
            if new_lines:
                all_output.extend(new_lines)
                print(f"  Collected {len(new_lines)} lines... (total: {len(all_output)})")
                
                # 檢測完成標記
                if any("VM_EXECUTION_COMPLETE" in line for line in new_lines):
                    execution_complete = True
                    print("\n✅ VM execution completed")
        
        if not execution_complete:
            print("\n⚠️  Warning: Did not receive completion marker")
        
        elapsed = time.time() - start_time
        return all_output, elapsed
    
    @staticmethod
    def parse_trace_json(output: List[str]) -> List[Dict]:
        """解析 streaming trace JSON"""
        records = []
        current_lines = []
        brace_count = 0
        
        for line in output:
            stripped = line.strip()
            
            # 跳過非 JSON 行
            if any(stripped.startswith(pattern) for pattern in TraceCollector.SKIP_PATTERNS):
                continue
            
            # JSON 對象開始
            if stripped == '{' and brace_count == 0:
                current_lines = [line]
                brace_count = 1
            elif brace_count > 0:
                current_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                # JSON 對象結束
                if brace_count == 0 and current_lines:
                    try:
                        json_str = '\n'.join(current_lines)
                        obj = json.loads(json_str)
                        # 只保留包含 cycle 的執行記錄
                        if 'cycle' in obj:
                            records.append(obj)
                    except json.JSONDecodeError:
                        pass  # 靜默處理解析錯誤
                    finally:
                        current_lines = []
        
        return records
    
    @staticmethod
    def parse_registers(lines: List[str]) -> Dict[str, int]:
        """解析寄存器值"""
        registers = {}
        for line in lines:
            if line.startswith('R') and '=' in line:
                try:
                    name, value_part = line.split('=', 1)
                    value = int(value_part.split()[0])  # Parse as decimal
                    registers[name.strip()] = value
                except (ValueError, IndexError):
                    continue
        return registers
