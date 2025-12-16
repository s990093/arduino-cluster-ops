"""
SIMD Lane 初始化器
支持為 8 個 lane 設置不同的初始寄存器值
"""

from typing import List, Tuple
from .program_loader import Instruction


class SIMDInitializer:
    """
    SIMD 8-Lane 初始化器
    
    核心概念：
    - Instruction 本身不區分 lane
    - 每個 lane 執行相同的指令
    - 不同結果來自於不同的初始寄存器值
    """
    
    @staticmethod
    def create_lane_data(lane_qkv: List[Tuple[int, int, int]]) -> List[Instruction]:
        """
        創建 lane 初始化指令序列
        
        Args:
            lane_qkv: 8個 (Q, K, V) tuple，每個 lane 一個
                     例如: [(2,3,4), (3,4,5), (4,5,6), ...]
        
        Returns:
            List[Instruction]: 初始化指令列表
            
        注意：
            由於 MOV 是 broadcast，無法直接為每個 lane 設不同值。
            此函數生成的是"統一指令"，實際的 lane 初始化需要在
            ESP32 韌體端完成（例如通過 custom opcode 或預加載）。
        """
        if len(lane_qkv) != 8:
            raise ValueError("Must provide exactly 8 (Q,K,V) tuples for 8 lanes")
        
        # 檢查是否所有 lane 的 QKV 都相同
        all_same = all(qkv == lane_qkv[0] for qkv in lane_qkv)
        
        if all_same:
            # 如果所有 lane 相同，使用普通 MOV
            Q, K, V = lane_qkv[0]
            return [
                Instruction.mov(0, Q),  # R0 = Q
                Instruction.mov(1, K),  # R1 = K
                Instruction.mov(2, V),  # R2 = V
            ]
        else:
            # 如果不同，返回空列表（需要韌體端支持）
            # 或者可以通過特殊編碼實現
            print("⚠️  Warning: Different lane values require firmware-side initialization")
            print("    MOV instruction broadcasts to all lanes")
            return []
    
    @staticmethod
    def get_initialization_comment(lane_qkv: List[Tuple[int, int, int]]) -> str:
        """
        生成初始化註釋，用於文檔和調試
        
        Returns:
            多行字符串，描述每個 lane 的初始值
        """
        comment = "SIMD 8-Lane Initialization:\n"
        comment += "=" * 50 + "\n"
        for lane_id, (Q, K, V) in enumerate(lane_qkv):
            comment += f"Lane {lane_id}: R0={Q:2d} (Q), R1={K:2d} (K), R2={V:2d} (V)\n"
        comment += "=" * 50
        return comment
    
    @staticmethod
    def create_transformer_program_multi_lane(
        lane_qkv: List[Tuple[int, int, int]]
    ) -> Tuple[List[Instruction], dict]:
        """
        創建支持多 lane 的 Transformer 程序
        
        Args:
            lane_qkv: 8個 (Q,K,V) tuple
        
        Returns:
            (instructions, expected_results_per_lane)
            
        範例：
            lane_qkv = [
                (2,3,4), (3,4,5), (4,5,6), (5,6,7),
                (6,7,8), (7,8,9), (8,9,10), (9,10,11)
            ]
            
            執行結果（每個 lane 不同）：
            Lane 0: R1=6,  R16=6,  R20=36
            Lane 1: R1=12, R16=8,  R20=144
            ...
        """
        if len(lane_qkv) != 8:
            raise ValueError("Must provide exactly 8 (Q,K,V) tuples")
        
        print("\n" + "="*70)
        print("🎯 Creating Multi-Lane Transformer Program")
        print("="*70)
        print(SIMDInitializer.get_initialization_comment(lane_qkv))
        print()
        
        # ===== 指令列表（所有 lane 執行相同指令）=====
        program = []
        
        # 注意：這些 MOV 會 broadcast 到所有 lane
        # 如果需要不同值，必須由韌體預先設置
        Q0, K0, V0 = lane_qkv[0]
        
        # 檢查是否所有 lane 相同
        all_same = all(qkv == lane_qkv[0] for qkv in lane_qkv)
        
        if all_same:
            # 1. 初始化（broadcast）
            program.append(Instruction.mov(0, Q0))  # R0 = Q
            program.append(Instruction.mov(1, K0))  # R1 = K
            program.append(Instruction.mov(2, V0))  # R2 = V
        else:
            # 不同 lane 需要通過其他方式初始化
            # 這裡假設韌體已經預加載了寄存器
            print("⚠️  Assuming firmware pre-initialized lanes with different Q/K/V")
            print("    Skipping MOV instructions\n")
        
        # 2. Attention Score = Q * K
        program.append(Instruction.imul(1, 0, 1))  # R1 = R0 * R1
        
        # 3. Residual = Q + V
        program.append(Instruction.iadd(16, 0, 2))  # R16 = R0 + R2
        program.append(Instruction.iadd(17, 0, 2))  # R17 = R0 + R2
        program.append(Instruction.iadd(18, 0, 2))  # R18 = R0 + R2
        program.append(Instruction.iadd(19, 0, 2))  # R19 = R0 + R2
        
        # 4. Sum of Squares = Score^2
        program.append(Instruction.imul(20, 1, 1))  # R20 = R1 * R1
        
        # 5. 退出
        program.append(Instruction.exit_inst())
        
        # ===== 計算每個 lane 的預期結果 =====
        expected_results = {}
        for lane_id, (Q, K, V) in enumerate(lane_qkv):
            attention = Q * K
            residual = Q + V
            sum_of_squares = attention * attention
            
            expected_results[lane_id] = {
                'R0': Q,
                'R1': attention,
                'R2': V,
                'R16': residual,
                'R17': residual,
                'R18': residual,
                'R19': residual,
                'R20': sum_of_squares
            }
        
        # 打印預期結果
        print("📊 Expected Results per Lane:")
        print("-" * 70)
        print(f"{'Lane':<6} {'Q(R0)':<8} {'K':<8} {'V(R2)':<8} {'Attn(R1)':<10} {'Res(R16)':<10} {'SS(R20)':<10}")
        print("-" * 70)
        for lane_id in range(8):
            Q, K, V = lane_qkv[lane_id]
            res = expected_results[lane_id]
            print(f"{lane_id:<6} {res['R0']:<8} {K:<8} {res['R2']:<8} "
                  f"{res['R1']:<10} {res['R16']:<10} {res['R20']:<10}")
        print("=" * 70 + "\n")
        
        return program, expected_results


# ===== 預定義配置 =====

def get_uniform_lanes() -> List[Tuple[int, int, int]]:
    """所有 lane 相同的配置（用於測試）"""
    return [(2, 3, 4)] * 8


def get_sequential_lanes() -> List[Tuple[int, int, int]]:
    """序列增長的配置（每個 lane 遞增）"""
    return [
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
        (5, 6, 7),
        (6, 7, 8),
        (7, 8, 9),
        (8, 9, 10),
        (9, 10, 11)
    ]


def get_random_lanes() -> List[Tuple[int, int, int]]:
    """隨機配置（用於壓力測試）"""
    import random
    random.seed(42)
    return [
        (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        for _ in range(8)
    ]
