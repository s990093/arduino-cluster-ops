"""
測試結果分析工具
"""

from typing import Dict, List


class ResultAnalyzer:
    """分析測試結果"""
    
    EXPECTED_VALUES = {
        'R0': 2,   # Q
        'R1': 6,   # Attention Score (Q * K = 2 * 3)
        'R2': 4,   # V
        'R16': 6,  # Residual (Q + V = 2 + 4)
        'R17': 6,
        'R18': 6,
        'R19': 6,
        'R20': 36  # Sum of Squares (Score^2 = 6 * 6)
    }
    
    @staticmethod
    def analyze(registers: Dict[str, int], trace_records: List[Dict]) -> bool:
        """分析結果並打印報告"""
        print("\n" + "="*70)
        print("📊 Results Analysis")
        print("="*70)
        
        # 顯示關鍵寄存器
        ResultAnalyzer._print_registers(registers)
        
        # 執行時間分析
        ResultAnalyzer._print_execution_time(trace_records)
        
        # 驗證結果
        success = ResultAnalyzer._verify_results(registers)
        
        print("="*70)
        return success
    
    @staticmethod
    def _print_registers(registers: Dict[str, int]):
        """打印關鍵寄存器值"""
        print("\n✓ Key Registers:")
        print(f"  R1  (Attention Score) = {registers.get('R1', 0)}")
        
        residual = [registers.get(f'R{i}', 0) for i in range(16, 20)]
        print(f"  R16-R19 (Q+V Residual) = {residual}")
        print(f"  R20 (Sum of Squares) = {registers.get('R20', 0)}")
    
    @staticmethod
    def _print_execution_time(records: List[Dict]):
        """打印執行時間統計"""
        exec_times = [r.get('exec_time_us', 0) for r in records if 'exec_time_us' in r]
        
        if exec_times:
            print(f"\n✓ Execution Time Analysis:")
            print(f"  Total: {sum(exec_times)} µs")
            print(f"  Average: {sum(exec_times)/len(exec_times):.1f} µs/instruction")
            print(f"  Min: {min(exec_times)} µs")
            print(f"  Max: {max(exec_times)} µs")
    
    @staticmethod
    def _verify_results(registers: Dict[str, int]) -> bool:
        """驗證結果是否符合預期"""
        print(f"\n✓ Verification:")
        
        checks = [
            ("Attention Score (R1)", 
             registers.get('R1', 0) == ResultAnalyzer.EXPECTED_VALUES['R1']),
            ("Residual Q+V (R16-R19)", 
             all(registers.get(f'R{i}', 0) == ResultAnalyzer.EXPECTED_VALUES[f'R{i}'] 
                 for i in range(16, 20))),
            ("Sum of Squares (R20)", 
             registers.get('R20', 0) == ResultAnalyzer.EXPECTED_VALUES['R20'])
        ]
        
        passed = 0
        for name, is_correct in checks:
            status = '✅' if is_correct else '❌'
            print(f"  {status} {name}")
            if is_correct:
                passed += 1
        
        print(f"\n📈 Score: {passed}/{len(checks)} passed")
        return passed == len(checks)
