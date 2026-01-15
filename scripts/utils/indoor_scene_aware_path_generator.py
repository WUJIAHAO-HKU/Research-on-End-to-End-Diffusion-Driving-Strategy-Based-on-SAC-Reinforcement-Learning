"""
室内场景感知路径生成器 - 6房间版本

严格在10m×10m六房间室内场景内生成可行路径，避开所有障碍物（墙体、家具）

场景布局（与rosorin_env_cfg.py完全一致）:
+-------+-------+-------+
|  R1   |  R2   |  R3   |  (上排: y=0→5)
| 客厅  | 书房  | 卧室  |
+-------+-------+-------+
|  R4   |  R5   |  R6   |  (下排: y=-5→0)
| 餐厅  | 厨房  | 储藏  |
+-------+-------+-------+
x轴: -5.0 → -1.67 → 1.67 → 5.0
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """障碍物定义"""
    name: str
    pos: Tuple[float, float]  # (x, y)
    size: Tuple[float, float]  # (width, height)
    
    def contains_point(self, point: np.ndarray, safety_margin: float = 0.3) -> bool:
        """检查点是否在障碍物内（包含安全边距）"""
        x, y = point
        cx, cy = self.pos
        w, h = self.size
        return (abs(x - cx) < (w/2 + safety_margin) and 
                abs(y - cy) < (h/2 + safety_margin))


class IndoorSceneMap:
    """
    6房间室内场景地图 - 与rosorin_env_cfg.py完全一致
    
    场景结构:
    - 外墙: 10m × 10m完全封闭空间
    - 水平隔断: y=0 (分隔上下两排房间，3个门洞)
    - 垂直隔断: x=-1.67, x=1.67 (分隔左中右列，各2个门洞)
    - 6个房间: R1客厅, R2书房, R3卧室, R4餐厅, R5厨房, R6储藏
    """
    
    def __init__(self):
        # 房间边界 (10m × 10m，预留0.2m安全边距)
        self.room_bounds = {
            'x_min': -4.9, 'x_max': 4.9,
            'y_min': -4.9, 'y_max': 4.9
        }
        
        # ========== 水平隔断墙配置（y=0.0）==========
        # 注意：范围根据环境配置计算：中心位置 ± (宽度/2)
        # divider_h_seg1: pos=(-3.8, 0.0), size=(2.26, 0.15) → x范围[-4.93, -2.67]
        # divider_h_seg2: pos=(0.0, 0.0), size=(1.26, 0.15) → x范围[-0.63, 0.63]
        # divider_h_seg3: pos=(3.8, 0.0), size=(2.26, 0.15) → x范围[2.67, 4.93]
        self.horizontal_walls = [
            # 第1段: x=-4.93到-2.67（R1-R4之间左侧）
            {'x_min': -4.93, 'x_max': -2.67, 'y': 0.0, 'thickness': 0.15},
            # 门洞1: x=-2.67到-1.67 (R1↔R4)
            # 第2段: x=-0.63到0.63（R2-R5之间中间）  
            {'x_min': -0.63, 'x_max': 0.63, 'y': 0.0, 'thickness': 0.15},
            # 门洞2: x=0.63到1.67 (R2↔R5)
            # 第3段: x=2.67到4.93（R3-R6之间右侧）
            {'x_min': 2.67, 'x_max': 4.93, 'y': 0.0, 'thickness': 0.15},
            # 门洞3: x=1.67到2.67 (R3↔R6)
        ]
        
        # ========== 垂直隔断墙配置 ==========
        # 第一道垂直墙 x=-1.67 (分隔R1-R2 和 R4-R5)
        self.vertical_walls_left = [
            # 上排R1-R2段1: y=0→1.5
            {'x': -1.67, 'y_min': 0.0, 'y_max': 1.5, 'thickness': 0.15},
            # 门洞: y=1.5→2.5
            # 上排R1-R2段2: y=2.5→5.0
            {'x': -1.67, 'y_min': 2.5, 'y_max': 5.0, 'thickness': 0.15},
            # 下排R4-R5段1: y=-5.0→-2.5
            {'x': -1.67, 'y_min': -5.0, 'y_max': -2.5, 'thickness': 0.15},
            # 门洞: y=-2.5→-1.5
            # 下排R4-R5段2: y=-1.5→0
            {'x': -1.67, 'y_min': -1.5, 'y_max': 0.0, 'thickness': 0.15},
        ]
        
        # 第二道垂直墙 x=1.67 (分隔R2-R3 和 R5-R6)
        self.vertical_walls_right = [
            # 上排R2-R3段1: y=0→1.5
            {'x': 1.67, 'y_min': 0.0, 'y_max': 1.5, 'thickness': 0.15},
            # 门洞: y=1.5→2.5
            # 上排R2-R3段2: y=2.5→5.0
            {'x': 1.67, 'y_min': 2.5, 'y_max': 5.0, 'thickness': 0.15},
            # 下排R5-R6段1: y=-5.0→-2.5
            {'x': 1.67, 'y_min': -5.0, 'y_max': -2.5, 'thickness': 0.15},
            # 门洞: y=-2.5→-1.5
            # 下排R5-R6段2: y=-1.5→0
            {'x': 1.67, 'y_min': -1.5, 'y_max': 0.0, 'thickness': 0.15},
        ]
        
        # ========== 家具障碍物（与rosorin_env_cfg.py一致）==========
        self.furniture = [
            # R1-客厅（x: -5.0→-1.67, y: 0→5）
            Obstacle('沙发_R1', (-3.5, 3.5), (2.0, 0.9)),
            Obstacle('电视柜_R1', (-4.0, 1.2), (1.5, 0.4)),
            
            # R2-书房（x: -1.67→1.67, y: 0→5）
            Obstacle('书桌_R2', (-0.5, 3.5), (1.4, 0.7)),
            Obstacle('书架_R2', (1.2, 1.5), (0.4, 2.2)),
            
            # R3-卧室（x: 1.67→5.0, y: 0→5）
            Obstacle('床_R3', (3.5, 3.5), (2.0, 1.5)),
            Obstacle('衣柜_R3', (4.4, 1.2), (0.6, 1.8)),
            
            # R4-餐厅（x: -5.0→-1.67, y: -5→0）
            Obstacle('餐桌_R4', (-3.5, -2.5), (1.6, 1.0)),
            Obstacle('餐边柜_R4', (-4.2, -4.2), (0.5, 1.6)),
            
            # R5-厨房（x: -1.67→1.67, y: -5→0）
            Obstacle('操作台_R5', (0.0, -3.8), (2.5, 0.6)),
            Obstacle('冰箱_R5', (1.0, -1.5), (0.7, 0.7)),
            
            # R6-储藏室（x: 1.67→5.0, y: -5→0）
            Obstacle('货架1_R6', (4.3, -3.5), (0.5, 1.5)),
            Obstacle('货架2_R6', (2.5, -4.0), (0.5, 1.5)),
        ]
        
        # 定义6个房间的安全区域中心（用于路径采样）
        self.room_centers = {
            'R1_living': np.array([-3.3, 2.5]),    # 客厅
            'R2_study': np.array([0.0, 2.5]),      # 书房
            'R3_bedroom': np.array([3.3, 2.5]),    # 卧室
            'R4_dining': np.array([-3.3, -2.5]),   # 餐厅
            'R5_kitchen': np.array([0.0, -2.5]),   # 厨房
            'R6_storage': np.array([3.3, -2.5]),   # 储藏室
        }
        
        self.room_list = list(self.room_centers.keys())
    
    def is_point_valid(self, point: np.ndarray, safety_margin: float = 0.3) -> bool:
        """
        检查点是否在安全区域（6房间场景版本）
        
        Args:
            point: (x, y) 坐标
            safety_margin: 安全边距（米）
        """
        x, y = point
        
        # 1. 检查是否在房间边界内
        if not (self.room_bounds['x_min'] <= x <= self.room_bounds['x_max'] and
                self.room_bounds['y_min'] <= y <= self.room_bounds['y_max']):
            return False
        
        # 2. 检查水平隔断墙碰撞（y=0处的3段墙）
        for wall in self.horizontal_walls:
            if (wall['x_min'] <= x <= wall['x_max'] and
                abs(y - wall['y']) < (wall['thickness']/2 + safety_margin)):
                return False  # 撞墙
        
        # 3. 检查左侧垂直墙碰撞（x=-1.67处的墙段）
        for wall in self.vertical_walls_left:
            if (abs(x - wall['x']) < (wall['thickness']/2 + safety_margin) and
                wall['y_min'] <= y <= wall['y_max']):
                return False  # 撞墙
        
        # 4. 检查右侧垂直墙碰撞（x=1.67处的墙段）
        for wall in self.vertical_walls_right:
            if (abs(x - wall['x']) < (wall['thickness']/2 + safety_margin) and
                wall['y_min'] <= y <= wall['y_max']):
                return False  # 撞墙
        
        # 5. 检查家具碰撞
        for obstacle in self.furniture:
            if obstacle.contains_point(point, safety_margin):
                return False
        
        return True
    
    def sample_safe_point(self, max_attempts: int = 100, target_room: Optional[str] = None) -> Optional[np.ndarray]:
        """
        随机采样一个安全点
        
        Args:
            max_attempts: 最大尝试次数
            target_room: 指定房间（如'R1_living'），None则随机选择
        
        Returns:
            安全点坐标，如果失败返回None
        """
        for _ in range(max_attempts):
            # 选择房间
            if target_room is None:
                room = np.random.choice(self.room_list)
            else:
                room = target_room
            
            center = self.room_centers[room]
            
            # 在房间中心附近采样（高斯分布，标准差0.8m）
            point = center + np.random.randn(2) * 0.8
            
            if self.is_point_valid(point):
                return point
        
        return None
    
    def is_path_collision_free(self, path_points: np.ndarray, resolution: float = 0.1) -> bool:
        """
        检查路径是否无碰撞（线性插值检查）
        
        Args:
            path_points: (N, 2) 路径点
            resolution: 检查分辨率（米）
        """
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i + 1]
            
            # 计算需要检查的点数
            distance = np.linalg.norm(end - start)
            num_checks = int(distance / resolution) + 1
            
            # 线性插值检查
            for t in np.linspace(0, 1, num_checks):
                point = start + t * (end - start)
                if not self.is_point_valid(point):
                    return False
        
        return True


class SceneAwarePathGenerator:
    """6房间场景感知路径生成器 - 基于粒子群优化（PSO）"""
    
    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        """
        Args:
            difficulty: 路径难度
                - easy: 短距离，跨越2个房间
                - medium: 中等距离，跨越3-4个房间
                - hard: 长距离，跨越5-6个房间
        """
        self.difficulty = difficulty
        self.scene = IndoorSceneMap()
        
        if seed is not None:
            np.random.seed(seed)
        
        # 难度配置（固定起点在餐厅，终点根据难度限制在不同区域）
        self.configs = {
            "easy": {
                "start_room": "R4_dining",    # 起点固定在餐厅
                "goal_rooms": ["R5_kitchen", "R1_living"],  # 终点在厨房或客厅
                "num_rooms": 2,               # 必须跨越2个房间
            },
            "medium": {
                "start_room": "R4_dining",
                "goal_rooms": ["R2_study", "R6_storage"],  # 终点在书房或储藏
                "num_rooms": (3, 4),          # 跨越3-4个房间
            },
            "hard": {
                "start_room": "R4_dining",
                "goal_rooms": ["R3_bedroom", "R1_living"],  # 终点在卧室或客厅对角
                "num_rooms": (4, 5),          # 跨越4-5个房间（降低难度）
            },
        }
        
        # A*算法配置
        self.astar_config = {
            "grid_resolution": 0.15,   # 网格分辨率（米），提高精度
            "diagonal_cost": 1.414,   # 对角线移动代价（sqrt(2)）
            "straight_cost": 1.0,     # 直线移动代价
        }
    
    def generate_random_path(self, max_attempts: int = 50) -> Optional[np.ndarray]:
        """
        生成随机安全路径（基于A*的起点-终点路径规划）
        
        Returns:
            path_points: (N, 2) 路径航点，如果失败返回None
        """
        config = self.configs[self.difficulty]
        
        for attempt in range(max_attempts):
            # 1. 采样起点和终点
            start, end = self._sample_start_end_points(config)
            if start is None or end is None:
                continue
            
            # 2. 使用A*算法生成路径
            path = self._astar_path_planning(start, end)
            
            if path is not None and len(path) >= 2:
                # 验证路径跨越的房间数
                visited_rooms = set()
                for point in path:
                    visited_rooms.add(self._get_room(point))
                
                target_rooms = config["num_rooms"] if isinstance(config["num_rooms"], int) else config["num_rooms"][0]
                
                # 验证路径完全无碰撞（使用严格的安全边距）
                path_valid = all(self.scene.is_point_valid(p, safety_margin=0.25) for p in path)
                path_collision_free = self.scene.is_path_collision_free(path, resolution=0.05)
                
                if len(visited_rooms) >= target_rooms and path_valid and path_collision_free:
                    return path
        
        # 失败后返回简单的同房间路径
        print(f"警告: {max_attempts}次尝试后未能生成{self.difficulty}难度路径，降级为简单路径")
        return self._generate_simple_fallback_path()
    
    def _sample_start_end_points(self, config: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        采样起点和终点（使用固定的房间约束）
        
        Returns:
            (start, end) 或 (None, None)
        """
        # 在起始房间内采样起点
        start_room = config["start_room"]
        start = self.scene.sample_safe_point(target_room=start_room, max_attempts=100)
        if start is None:
            return None, None
        
        # 随机选择一个目标房间
        goal_room = np.random.choice(config["goal_rooms"])
        end = self.scene.sample_safe_point(target_room=goal_room, max_attempts=100)
        if end is None:
            return None, None
        
        return start, end
    
    def _astar_path_planning(self, start: np.ndarray, end: np.ndarray) -> Optional[np.ndarray]:
        """
        使用A*算法生成避障路径
        
        Args:
            start: 起点
            end: 终点
        
        Returns:
            完整路径（包含起点和终点），如果失败返回None
        """
        resolution = self.astar_config["grid_resolution"]
        
        # 将连续坐标转换为网格坐标
        def world_to_grid(point):
            x = int((point[0] - self.scene.room_bounds['x_min']) / resolution)
            y = int((point[1] - self.scene.room_bounds['y_min']) / resolution)
            return (x, y)
        
        def grid_to_world(grid_point):
            x = grid_point[0] * resolution + self.scene.room_bounds['x_min']
            y = grid_point[1] * resolution + self.scene.room_bounds['y_min']
            return np.array([x, y])
        
        # 启发式函数（欧几里得距离）
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))
        
        # 获取邻居（8方向）
        def get_neighbors(node):
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),  # 4方向
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # 对角线
                nx, ny = node[0] + dx, node[1] + dy
                
                # 检查是否在网格范围内
                world_point = grid_to_world((nx, ny))
                if (self.scene.room_bounds['x_min'] <= world_point[0] <= self.scene.room_bounds['x_max'] and
                    self.scene.room_bounds['y_min'] <= world_point[1] <= self.scene.room_bounds['y_max']):
                    
                    # 检查是否无碰撞（使用更大的安全边距避免路径太靠近障碍物）
                    if self.scene.is_point_valid(world_point, safety_margin=0.35):
                        # 计算移动代价（对角线代价更高）
                        cost = self.astar_config["diagonal_cost"] if abs(dx) + abs(dy) == 2 else self.astar_config["straight_cost"]
                        neighbors.append(((nx, ny), cost))
            
            return neighbors
        
        # A*主算法
        start_grid = world_to_grid(start)
        end_grid = world_to_grid(end)
        
        # 开放列表和关闭列表
        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: heuristic(start_grid, end_grid)}
        
        closed_set = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # 到达目标
            if current == end_grid:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(grid_to_world(current))
                    current = came_from[current]
                path.append(grid_to_world(start_grid))
                path.reverse()
                
                # 路径简化（更保守的策略，确保不穿过障碍物）
                simplified_path = self._simplify_path_safe(np.array(path), epsilon=0.15)
                
                return simplified_path
            
            closed_set.add(current)
            
            # 探索邻居
            for neighbor, move_cost in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end_grid)
                    
                    # 添加到开放列表
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 未找到路径
        return None
    
    def _simplify_path_safe(self, path: np.ndarray, epsilon: float = 0.15, max_depth: int = 50) -> np.ndarray:
        """
        安全的路径简化 - 确保简化后的路径不穿过障碍物
        
        采用分段简化策略：
        1. 尝试直接连接起点和终点
        2. 如果碰撞，则在中间找分割点递归简化
        3. 确保每条线段都无碰撞
        """
        if len(path) <= 2:
            return path
        
        def can_connect_directly(p1, p2):
            """检查两点是否可以直接连接（不碰撞）"""
            dist = np.linalg.norm(p2 - p1)
            if dist < 1e-6:
                return True
            # 检查线段是否穿过障碍物
            num_checks = int(dist / 0.05) + 1  # 每5cm检查一次
            for t in np.linspace(0, 1, num_checks):
                point = p1 + t * (p2 - p1)
                if not self.scene.is_point_valid(point, safety_margin=0.25):
                    return False
            return True
        
        def simplify_recursive(points, depth=0):
            """递归简化路径"""
            # 递归深度限制
            if depth > max_depth or len(points) <= 2:
                return points
            
            # 尝试直接连接起点和终点
            if can_connect_directly(points[0], points[-1]):
                return np.array([points[0], points[-1]])
            
            # 找到距离直线最远的点作为分割点
            max_dist = 0
            max_idx = 1  # 默认使用第二个点，避免max_idx=0导致无限递归
            
            line_vec = points[-1] - points[0]
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                return points  # 起点终点重合，返回原路径
            
            line_unitvec = line_vec / line_len
            
            for i in range(1, len(points) - 1):
                point_vec = points[i] - points[0]
                proj_length = np.dot(point_vec, line_unitvec)
                proj_length = np.clip(proj_length, 0, line_len)
                nearest = points[0] + proj_length * line_unitvec
                dist = np.linalg.norm(points[i] - nearest)
                
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            # 防御性检查：如果max_idx没有被更新或在边界上，使用中点
            if max_idx <= 0 or max_idx >= len(points) - 1:
                max_idx = len(points) // 2
            
            # 如果最大距离很小且可以直接连接，则简化
            if max_dist < epsilon and can_connect_directly(points[0], points[-1]):
                return np.array([points[0], points[-1]])
            
            # 递归简化两段（增加深度计数）
            left = simplify_recursive(points[:max_idx + 1], depth + 1)
            right = simplify_recursive(points[max_idx:], depth + 1)
            
            # 合并结果（去掉重复的中间点）
            return np.vstack([left[:-1], right])
        
        try:
            simplified = simplify_recursive(path, depth=0)
            
            # 最后再验证一次完整路径
            if not self.scene.is_path_collision_free(simplified, resolution=0.05):
                # 如果简化后仍有碰撞，返回原始路径
                return path
            
            return simplified
        except RecursionError:
            # 如果仍然递归过深，直接返回原路径
            return path
    
    def _simplify_path(self, path: np.ndarray, epsilon: float = 0.3) -> np.ndarray:
        """
        使用Douglas-Peucker算法简化路径
        
        Args:
            path: 原始路径点
            epsilon: 简化阈值，越大路径越简单
        
        Returns:
            简化后的路径
        """
        if len(path) <= 2:
            return path
        
        def perpendicular_distance(point, line_start, line_end):
            """计算点到线段的垂直距离"""
            if np.allclose(line_start, line_end):
                return np.linalg.norm(point - line_start)
            
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            
            # 投影到线段上
            proj_length = np.dot(point_vec, line_unitvec)
            proj_length = np.clip(proj_length, 0, line_len)
            
            # 最近点
            nearest = line_start + proj_length * line_unitvec
            return np.linalg.norm(point - nearest)
        
        def douglas_peucker(points, epsilon):
            """递归Douglas-Peucker算法"""
            dmax = 0
            index = 0
            
            for i in range(1, len(points) - 1):
                d = perpendicular_distance(points[i], points[0], points[-1])
                if d > dmax:
                    index = i
                    dmax = d
            
            if dmax > epsilon:
                # 递归简化
                rec_results1 = douglas_peucker(points[:index+1], epsilon)
                rec_results2 = douglas_peucker(points[index:], epsilon)
                
                # 合并结果
                result = np.vstack([rec_results1[:-1], rec_results2])
            else:
                # 保留起点和终点
                result = np.array([points[0], points[-1]])
            
            return result
        
        simplified = douglas_peucker(path, epsilon)
        
        # 确保路径仍然无碰撞
        if self.scene.is_path_collision_free(simplified, resolution=0.1):
            return simplified
        else:
            # 如果简化后碰撞，返回原始路径
            return path
    
    def _get_room(self, point: np.ndarray) -> str:
        """判断点在哪个房间（6房间版本）"""
        x, y = point
        
        # 根据 x, y 坐标判断所在房间
        if y > 0:  # 上排 (R1, R2, R3)
            if x < -1.67:
                return 'R1_living'
            elif x < 1.67:
                return 'R2_study'
            else:
                return 'R3_bedroom'
        else:  # 下排 (R4, R5, R6)
            if x < -1.67:
                return 'R4_dining'
            elif x < 1.67:
                return 'R5_kitchen'
            else:
                return 'R6_storage'
    
    def _are_rooms_adjacent(self, room1: str, room2: str) -> bool:
        """判断两个房间是否相邻（有门洞连通）"""
        # 定义相邻关系（有门洞的房间对）
        adjacent_pairs = {
            ('R1_living', 'R2_study'),
            ('R2_study', 'R3_bedroom'),
            ('R4_dining', 'R5_kitchen'),
            ('R5_kitchen', 'R6_storage'),
            ('R1_living', 'R4_dining'),
            ('R2_study', 'R5_kitchen'),
            ('R3_bedroom', 'R6_storage'),
        }
        
        return ((room1, room2) in adjacent_pairs or 
                (room2, room1) in adjacent_pairs)
    
    def _are_rooms_diagonal(self, room1: str, room2: str) -> bool:
        """判断两个房间是否对角（最远距离）"""
        diagonal_pairs = {
            ('R1_living', 'R6_storage'),
            ('R3_bedroom', 'R4_dining'),
        }
        
        return ((room1, room2) in diagonal_pairs or 
                (room2, room1) in diagonal_pairs)
    
    def _generate_simple_fallback_path(self) -> np.ndarray:
        """生成简单降级路径（6房间版本）- 单房间内安全路径"""
        # 为了保证100%成功，在单个房间内生成短路径
        # 选择R2_study（中间上方），空间相对开阔
        center = self.scene.room_centers['R2_study']
        
        # 在房间中心附近生成安全的短距离路径（避免靠近墙壁）
        start = center + np.array([-0.6, -0.6])  # 左下
        end = center + np.array([0.6, 0.6])      # 右上
        mid = (start + end) / 2                   # 中点
        
        return np.array([start, mid, end])


def test_generator():
    """测试场景感知路径生成器（A*版本）"""
    print("\n" + "="*80)
    print("  场景感知路径生成器测试（基于A*算法）")
    print("="*80)
    
    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n难度: {difficulty}")
        gen = SceneAwarePathGenerator(difficulty=difficulty, seed=42)
        
        success_count = 0
        total_time = 0
        
        for i in range(10):
            import time
            start_time = time.time()
            
            path = gen.generate_random_path()
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if path is not None:
                success_count += 1
                
                # 验证路径安全性
                all_valid = all(gen.scene.is_point_valid(p) for p in path)
                
                # 统计访问的房间数
                visited_rooms = set()
                for p in path:
                    visited_rooms.add(gen._get_room(p))
                
                # 计算路径总长度
                total_distance = sum(np.linalg.norm(path[j+1] - path[j]) 
                                   for j in range(len(path)-1))
                
                # 计算起点-终点直线距离
                straight_dist = np.linalg.norm(path[-1] - path[0])
                
                # 计算曲折度
                detour_ratio = total_distance / (straight_dist + 1e-6)
                
                status = "✅" if all_valid else "❌"
                
                print(f"  路径 {i+1}: {len(path)}航点, {len(visited_rooms)}房间, "
                      f"长度={total_distance:.1f}m (直线={straight_dist:.1f}m, "
                      f"曲折度={detour_ratio:.2f}), 用时={elapsed:.3f}s {status}")
            else:
                print(f"  路径 {i+1}: 生成失败")
        
        avg_time = total_time / 10
        print(f"  成功率: {success_count}/10, 平均用时: {avg_time:.3f}s")
    
    print("\n" + "="*80)
    print("✓ 测试完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_generator()
