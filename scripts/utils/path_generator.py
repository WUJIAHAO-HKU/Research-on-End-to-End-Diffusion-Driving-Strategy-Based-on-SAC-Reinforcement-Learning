"""
Path Generator for ROSOrin Driving Task

生成各种类型的驾驶路径用于训练和测试。
"""

import numpy as np
import torch
from typing import Tuple, List
from enum import Enum


class PathType(Enum):
    """路径类型"""
    STRAIGHT = "straight"
    CURVE = "curve"
    S_CURVE = "s_curve"
    CIRCLE = "circle"
    BEZIER = "bezier"
    RANDOM_WAYPOINTS = "random_waypoints"


class PathGenerator:
    """路径生成器"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_straight_line(
        self,
        start: np.ndarray = np.array([0.0, 0.0]),
        end: np.ndarray = np.array([3.0, 0.0]),
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成直线路径
        
        Returns:
            positions: (N, 2) 位置序列
            headings: (N,) 朝向序列
        """
        t = np.linspace(0, 1, num_points)
        positions = start[None, :] + t[:, None] * (end - start)[None, :]
        
        # 朝向始终指向终点
        direction = end - start
        heading = np.arctan2(direction[1], direction[0])
        headings = np.full(num_points, heading)
        
        return positions, headings
    
    def generate_curve(
        self,
        start: np.ndarray = np.array([0.0, 0.0]),
        radius: float = 2.0,
        angle_range: Tuple[float, float] = (0.0, np.pi/2),
        num_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成圆弧路径
        
        Args:
            start: 起点
            radius: 半径
            angle_range: 角度范围 (start_angle, end_angle)
            num_points: 采样点数
        """
        angles = np.linspace(angle_range[0], angle_range[1], num_points)
        
        # 圆心位置
        center_x = start[0] + radius * np.sin(angle_range[0])
        center_y = start[1] - radius * np.cos(angle_range[0])
        
        # 生成路径点
        x = center_x + radius * np.sin(angles)
        y = center_y + radius * np.cos(angles)
        positions = np.stack([x, y], axis=1)
        
        # 朝向沿着圆弧切线方向
        headings = angles + np.pi/2
        
        return positions, headings
    
    def generate_s_curve(
        self,
        start: np.ndarray = np.array([0.0, 0.0]),
        length: float = 4.0,
        amplitude: float = 1.0,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成S型曲线路径
        """
        t = np.linspace(0, 1, num_points)
        
        x = start[0] + t * length
        y = start[1] + amplitude * np.sin(2 * np.pi * t)
        
        positions = np.stack([x, y], axis=1)
        
        # 计算切线方向
        dx = np.gradient(x)
        dy = np.gradient(y)
        headings = np.arctan2(dy, dx)
        
        return positions, headings
    
    def generate_circle(
        self,
        center: np.ndarray = np.array([2.0, 2.0]),
        radius: float = 1.5,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成圆形路径"""
        angles = np.linspace(0, 2*np.pi, num_points)
        
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        positions = np.stack([x, y], axis=1)
        
        # 朝向切线方向
        headings = angles + np.pi/2
        
        return positions, headings
    
    def generate_bezier_curve(
        self,
        control_points: np.ndarray,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成贝塞尔曲线路径
        
        Args:
            control_points: (n_control, 2) 控制点
            num_points: 采样点数
        """
        t = np.linspace(0, 1, num_points)
        n = len(control_points) - 1
        
        positions = np.zeros((num_points, 2))
        
        # 计算贝塞尔曲线
        for i, ti in enumerate(t):
            point = np.zeros(2)
            for j in range(n + 1):
                # 伯恩斯坦多项式
                bernstein = self._bernstein_poly(j, n, ti)
                point += bernstein * control_points[j]
            positions[i] = point
        
        # 计算朝向
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        headings = np.arctan2(dy, dx)
        
        return positions, headings
    
    def _bernstein_poly(self, i: int, n: int, t: float) -> float:
        """伯恩斯坦多项式"""
        from math import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    def generate_random_waypoints(
        self,
        num_waypoints: int = 5,
        area_size: float = 4.0,
        min_distance: float = 0.5,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成随机路径点并插值成平滑路径
        
        Args:
            num_waypoints: 路径点数量
            area_size: 区域大小
            min_distance: 路径点间最小距离
            num_points: 插值后的点数
        """
        waypoints = [np.array([0.0, 0.0])]  # 起点
        
        for _ in range(num_waypoints - 1):
            while True:
                # 生成候选点
                candidate = self.rng.uniform(-area_size/2, area_size/2, size=2)
                
                # 检查与已有点的距离
                distances = [np.linalg.norm(candidate - wp) for wp in waypoints]
                if all(d > min_distance for d in distances):
                    waypoints.append(candidate)
                    break
        
        waypoints = np.array(waypoints)
        
        # 使用贝塞尔曲线插值
        return self.generate_bezier_curve(waypoints, num_points)
    
    def generate_batch(
        self,
        path_type: PathType,
        batch_size: int,
        **kwargs
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        批量生成路径
        
        Returns:
            List of (positions, headings) tuples
        """
        paths = []
        
        for i in range(batch_size):
            # 为每条路径使用不同的随机参数
            seed_offset = i * 1000
            
            if path_type == PathType.STRAIGHT:
                # 随机起点和终点
                start = self.rng.uniform(-1, 1, size=2)
                angle = self.rng.uniform(0, 2*np.pi)
                length = self.rng.uniform(2, 4)
                end = start + length * np.array([np.cos(angle), np.sin(angle)])
                path = self.generate_straight_line(start, end, **kwargs)
                
            elif path_type == PathType.CURVE:
                start = self.rng.uniform(-1, 1, size=2)
                radius = self.rng.uniform(1.5, 3.0)
                angle_start = self.rng.uniform(0, 2*np.pi)
                angle_range = self.rng.uniform(np.pi/4, np.pi)
                path = self.generate_curve(
                    start, radius, 
                    (angle_start, angle_start + angle_range),
                    **kwargs
                )
                
            elif path_type == PathType.S_CURVE:
                start = self.rng.uniform(-1, 1, size=2)
                length = self.rng.uniform(3, 5)
                amplitude = self.rng.uniform(0.5, 1.5)
                path = self.generate_s_curve(start, length, amplitude, **kwargs)
                
            elif path_type == PathType.CIRCLE:
                center = self.rng.uniform(-2, 2, size=2)
                radius = self.rng.uniform(1.0, 2.0)
                path = self.generate_circle(center, radius, **kwargs)
                
            elif path_type == PathType.BEZIER:
                num_control = self.rng.randint(4, 7)
                control_points = self.rng.uniform(-2, 2, size=(num_control, 2))
                control_points[0] = np.array([0.0, 0.0])  # 固定起点
                path = self.generate_bezier_curve(control_points, **kwargs)
                
            elif path_type == PathType.RANDOM_WAYPOINTS:
                path = self.generate_random_waypoints(**kwargs)
            
            else:
                raise ValueError(f"Unknown path type: {path_type}")
            
            paths.append(path)
        
        return paths


if __name__ == "__main__":
    # 测试路径生成器
    print("="*70)
    print("  路径生成器测试")
    print("="*70)
    
    generator = PathGenerator(seed=42)
    
    # 测试不同类型的路径
    test_cases = [
        (PathType.STRAIGHT, "直线"),
        (PathType.CURVE, "圆弧"),
        (PathType.S_CURVE, "S型曲线"),
        (PathType.CIRCLE, "圆形"),
        (PathType.BEZIER, "贝塞尔曲线"),
        (PathType.RANDOM_WAYPOINTS, "随机路径点"),
    ]
    
    for path_type, name in test_cases:
        print(f"\n测试 {name}:")
        
        if path_type == PathType.STRAIGHT:
            positions, headings = generator.generate_straight_line()
        elif path_type == PathType.CURVE:
            positions, headings = generator.generate_curve()
        elif path_type == PathType.S_CURVE:
            positions, headings = generator.generate_s_curve()
        elif path_type == PathType.CIRCLE:
            positions, headings = generator.generate_circle()
        elif path_type == PathType.BEZIER:
            control_points = np.array([
                [0, 0], [1, 2], [3, 2], [4, 0]
            ])
            positions, headings = generator.generate_bezier_curve(control_points)
        elif path_type == PathType.RANDOM_WAYPOINTS:
            positions, headings = generator.generate_random_waypoints()
        
        print(f"  - 生成 {len(positions)} 个路径点")
        print(f"  - 起点: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f})")
        print(f"  - 终点: ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f})")
        print(f"  - 路径长度: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f}m")
    
    # 测试批量生成
    print(f"\n\n批量生成测试:")
    batch_paths = generator.generate_batch(
        PathType.RANDOM_WAYPOINTS,
        batch_size=5,
        num_waypoints=4,
        num_points=50
    )
    print(f"  - 生成 {len(batch_paths)} 条路径")
    print(f"  - 每条路径 {len(batch_paths[0][0])} 个点")
    
    print("\n✓ 路径生成器测试完成")
