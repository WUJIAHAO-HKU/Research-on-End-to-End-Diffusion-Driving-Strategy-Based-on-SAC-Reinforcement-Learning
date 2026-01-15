"""
场景感知路径生成器用于MPC数据采集

基于室内6房间场景，生成避开墙体和家具的安全路径。
"""

import numpy as np
from typing import List, Optional
from indoor_scene_aware_path_generator import SceneAwarePathGenerator as BaseGenerator


class PathGenerator:
    """场景感知路径生成器（兼容原接口）"""
    
    def __init__(self, difficulty: str = "easy", seed: int = None):
        """
        Args:
            difficulty: 路径难度
                - easy: 同房间内短距离导航
                - medium: 跨房间导航（穿过1个门）
                - hard: 多房间导航（穿过2个门）
        """
        self.difficulty = difficulty
        self.base_gen = BaseGenerator(difficulty=difficulty, seed=seed)
    
    def generate_random_path(self) -> np.ndarray:
        """
        生成场景感知的随机路径（避开墙体和家具）
        
        Returns:
            path_points: (N, 2) 路径航点坐标，保证在安全区域内
        """
        # 使用场景感知生成器
        path = self.base_gen.generate_random_path(max_attempts=100)
        
        if path is None:
            # 极端情况：生成失败，使用安全的后备路径
            print(f"⚠️ 警告: 场景感知路径生成失败，使用后备路径")
            path = self.base_gen._generate_simple_fallback_path()
        
        return path


if __name__ == "__main__":
    """测试简化路径生成器"""
    print("\n" + "="*60)
    print("  简化路径生成器测试")
    print("="*60)
    
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n难度: {difficulty}")
        gen = PathGenerator(difficulty=difficulty)
        
        for i in range(3):
            path = gen.generate_random_path()
            print(f"  路径 {i+1}: {len(path)} 个航点, "
                  f"起点=({path[0,0]:.2f}, {path[0,1]:.2f}), "
                  f"终点=({path[-1,0]:.2f}, {path[-1,1]:.2f})")
    
    print("\n✓ 测试完成\n")
