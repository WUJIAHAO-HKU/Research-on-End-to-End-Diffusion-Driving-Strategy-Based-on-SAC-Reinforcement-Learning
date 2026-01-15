"""
Custom MDP (Markov Decision Process) functions for ROSOrin navigation task.

包含奖励函数、终止条件、观测函数等。
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# 奖励函数 (Reward Functions)
##

def progress_reward(env: "ManagerBasedRLEnv", threshold: float = 0.001) -> torch.Tensor:
    """
    奖励机器人向目标点前进的进度（密集奖励）
    
    改进:
    - 降低阈值到0.001米（确保密集反馈）
    - 使用指数缩放增强远距离时的奖励梯度
    
    Args:
        env: RL环境
        threshold: 距离减少的最小阈值（默认0.001米，约为每步位移的1/6）
        
    Returns:
        奖励值 (num_envs,)
    """
    # 获取当前位置
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # 获取目标位置
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device)
    
    goal_pos = env.goal_positions[:, :2]  # (num_envs, 2)
    
    # 计算当前距离
    current_dist = torch.norm(robot_pos - goal_pos, dim=-1)
    
    # 计算距离变化（减少为正奖励，增加为负惩罚）
    if not hasattr(env, 'previous_goal_distance'):
        env.previous_goal_distance = current_dist.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    distance_change = env.previous_goal_distance - current_dist
    env.previous_goal_distance = current_dist.clone()
    
    # 密集奖励：即使很小的进步也给奖励
    # 使用指数缩放：远距离时放大奖励信号
    scale = torch.exp(-current_dist / 5.0)  # 距离5m时缩放到0.37
    reward = distance_change * (1.0 + 2.0 * scale)  # 近距离时奖励放大3倍
    
    # 只过滤噪声级别的抖动
    # NOTE: 在PPO里我们会把threshold调到0或更小，以确保progress信号足够密集可学。
    reward = torch.where(
        torch.abs(distance_change) > threshold,
        reward,
        torch.zeros_like(reward)
    )
    
    return reward


def goal_reached_reward(
    env: "ManagerBasedRLEnv",
    distance_threshold: float = 0.5,
    reached_bonus: float = 1.0,
    milestone_1m_bonus: float = 5.0,
    milestone_2m_bonus: float = 2.0,
) -> torch.Tensor:
    """
    到达目标点的稀疏奖励 + 分段密集奖励
    
    改进:
    - 完全到达时给大额稀疏奖励（100×1=100分）
    - 接近目标时给分段奖励，提供中间里程碑
    
    Args:
        env: RL环境
        distance_threshold: 认为到达目标的距离阈值 (米)
        
    Returns:
        奖励值 (num_envs,)
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device)
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    
    # 完全到达：稀疏奖励（最终尺度由RewTerm.weight控制；这里提供可配置bonus以便不同算法降方差）
    reached = (distance < distance_threshold).float() * float(reached_bonus)
    
    # 里程碑系统（只在第一次进入阈值时给奖励）
    if not hasattr(env, 'milestone_reached_1m'):
        env.milestone_reached_1m = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, 'milestone_reached_2m'):
        env.milestone_reached_2m = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    milestone_reward = torch.zeros(env.num_envs, device=env.device)
    
    # 1米里程碑
    new_1m = (distance < 1.0) & (~env.milestone_reached_1m)
    milestone_reward += new_1m.float() * float(milestone_1m_bonus)  # 首次进入1m范围
    env.milestone_reached_1m |= (distance < 1.0)
    
    # 2米里程碑
    new_2m = (distance < 2.0) & (~env.milestone_reached_2m)
    milestone_reward += new_2m.float() * float(milestone_2m_bonus)  # 首次进入2m范围
    env.milestone_reached_2m |= (distance < 2.0)
    
    return reached + milestone_reward


def distance_to_goal_reward(env: "ManagerBasedRLEnv", scale: float = 2.0) -> torch.Tensor:
    """
    基于距离目标的直接奖励（更简单、更稳定）
    
    设计理念：
    - 奖励与"距离目标有多近"成反比
    - 距离 0m → 奖励 +scale
    - 距离 5m → 奖励 0
    - 距离 >5m → 负奖励
    
    相比 progress_reward 的优势：
    - 不依赖速度方向，只看结果
    - 避免"原地抖动"策略（距离不变 → 奖励不变）
    - 数值稳定（不需要记录历史距离）
    
    Args:
        env: RL环境
        scale: 奖励缩放系数（距离 0m 时的最大奖励）
        
    Returns:
        奖励值 (num_envs,) 范围约 [-scale, +scale]
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device)
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    
    # 距离奖励：distance=0 → reward=+scale, distance=5 → reward=0, distance>5 → 负
    # 使用指数衰减：reward = scale * (1 - dist/5)，线性衰减更稳定
    reward = scale * (1.0 - torch.clamp(distance / 5.0, 0.0, 2.0))
    
    return reward


def sprint_bonus_reward(env: "ManagerBasedRLEnv", 
                       distance_threshold: float = 2.0,
                       min_speed: float = 0.15) -> torch.Tensor:
    """
    冲刺奖励：当距离目标很近(<2m)且速度足够快(>0.15m/s)时，给额外奖励
    
    目的：打破"接近目标2m后停滞"的僵局，鼓励机器人"最后1米冲刺"
    
    Args:
        env: RL环境
        distance_threshold: 触发冲刺的距离阈值（米）
        min_speed: 最小速度要求（m/s）
        
    Returns:
        奖励值 (num_envs,) - 满足条件时为 +1.0，否则为 0
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device)
    
    # 计算距离目标的距离
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    
    # 计算朝向目标的速度
    lin_vel_w = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]
    to_goal = goal_pos - robot_pos
    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)
    to_goal_dir = to_goal / (to_goal_norm + 1e-6)
    vel_toward_goal = torch.sum(lin_vel_w * to_goal_dir, dim=-1)
    
    # 冲刺条件：距离 < 阈值 且 速度 > 最小值
    sprint_mask = (distance < distance_threshold) & (vel_toward_goal > min_speed)
    
    reward = torch.zeros(env.num_envs, device=env.device)
    reward[sprint_mask] = 1.0
    
    return reward


def velocity_tracking_reward(env: "ManagerBasedRLEnv", target_vel: float = 0.5) -> torch.Tensor:
    """
    奖励朝向目标方向的速度（有方向性的速度奖励）
    
    改进:
    - 不仅看速度大小，还看速度方向是否朝向目标
    - 倒车或横向移动会得到惩罚
    
    Args:
        env: RL环境  
        target_vel: 目标速度 (m/s)
        
    Returns:
        奖励值 (num_envs,)
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device)
    
    # 获取世界坐标系下的速度
    lin_vel_w = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]  # (N, 2) xy速度
    
    # 计算朝向目标的方向
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    to_goal = goal_pos - robot_pos
    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)
    to_goal_dir = to_goal / (to_goal_norm + 1e-6)  # 归一化方向向量
    
    # 速度在目标方向上的投影（标量）
    vel_toward_goal = torch.sum(lin_vel_w * to_goal_dir, dim=-1)  # (N,)
    
    # 奖励接近目标速度的前进（倒车给负奖励）
    vel_error = torch.abs(vel_toward_goal - target_vel)
    reward = torch.exp(-2.0 * vel_error) - 0.1  # 基线降低，避免原地不动得高分
    
    # 如果在倒车（速度投影为负），给惩罚（降低惩罚从-2.0到-1.0）
    reward = torch.where(vel_toward_goal < 0, reward - 1.0, reward)
    
    return reward


def movement_penalty(env: "ManagerBasedRLEnv", min_speed: float = 0.15) -> torch.Tensor:
    """
    惩罚停滞不动（强制机器人移动）
    
    设计理念:
    - 当速度低于 min_speed 时给予惩罚，避免"原地不动"策略
    - 惩罚力度与速度不足成正比
    
    Args:
        env: RL环境
        min_speed: 最小速度阈值 (m/s)，低于此值开始惩罚
        
    Returns:
        惩罚值 (num_envs,)，速度越低惩罚越大
    """
    # 获取当前速度（世界坐标系下的 xy 平面速度）
    lin_vel_w = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]  # (N, 2)
    speed = torch.norm(lin_vel_w, dim=-1)  # (N,) 速度大小
    
    # 计算速度不足量
    speed_deficit = torch.clamp(min_speed - speed, min=0.0)  # 低于阈值的部分
    
    # 惩罚: -speed_deficit（速度越低惩罚越大）
    # 例如：speed=0.05 → deficit=0.10 → penalty=-0.10
    #      speed=0.20 → deficit=0.00 → penalty=0.00
    penalty = -speed_deficit
    
    return penalty


def orientation_alignment_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    奖励机器人朝向目标点（密集奖励）
    
    改进:
    - 使用指数函数增强对齐奖励的梯度
    - 背向目标时给明确惩罚
    
    Returns:
        奖励值 (num_envs,)
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device)
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    # 计算目标方向向量
    to_goal = goal_pos - robot_pos
    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)
    to_goal = to_goal / (to_goal_norm + 1e-6)
    
    # 获取机器人当前朝向 (从四元数提取yaw)
    quat = env.scene.articulations["robot"].data.root_quat_w  # (num_envs, 4) [w,x,y,z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # 机器人朝向向量
    robot_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    
    # 计算方向对齐度 (cos相似度) 范围[-1, 1]
    alignment = torch.sum(robot_dir * to_goal, dim=-1)
    
    # 指数奖励：完全对齐时=1，垂直时≈0.37，背向时≈0.14
    reward = torch.exp(alignment - 1.0)  # alignment=1时reward=1, alignment=-1时reward≈0.14
    
    # 背向目标（alignment < 0）给额外惩罚
    reward = torch.where(alignment < 0, reward - 0.5, reward)
    
    return reward


def smooth_action_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """
    惩罚动作突变，鼓励平滑控制
    
    Returns:
        惩罚值 (num_envs,) - 负值
    """
    if not hasattr(env, 'previous_actions'):
        env.previous_actions = env.action_manager.action.clone()
        return torch.zeros(env.num_envs, device=env.device)
    
    action_diff = env.action_manager.action - env.previous_actions
    env.previous_actions = env.action_manager.action.clone()
    
    # L2范数惩罚
    penalty = -torch.norm(action_diff, dim=-1)
    
    return penalty


def collision_penalty(env: "ManagerBasedRLEnv", threshold: float = 50.0) -> torch.Tensor:
    """
    碰撞惩罚 (基于接触力)
    
    Args:
        env: RL环境
        threshold: 认为发生碰撞的力阈值 (N)
        
    Returns:
        惩罚值 (num_envs,) - 负值
    """
    # TODO: 需要添加ContactSensor才能使用
    # 当前环境没有配置contact sensor
    # 暂时返回0
    return torch.zeros(env.num_envs, device=env.device)


def obstacle_avoidance_penalty(env: "ManagerBasedRLEnv", 
                               safe_distance: float = 0.3,
                               danger_distance: float = 0.15) -> torch.Tensor:
    """
    LiDAR + Depth 混合的避障惩罚
    
    策略:
    - 检测前方中心区域的最小深度
    - 距离 < danger_distance: 严重惩罚（即将碰撞）
    - 距离 < safe_distance: 轻微惩罚（鼓励远离）
    
    Args:
        env: RL环境
        safe_distance: 安全距离阈值 (米)
        danger_distance: 危险距离阈值 (米)
        
    Returns:
        惩罚值 (num_envs,) - 负值
    """
    # ------------------------
    # Depth 通道：前方中心区域“鲁棒最近距离”
    # ------------------------
    depth = env.scene.sensors["camera"].data.output["distance_to_image_plane"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    
    # ⚠️ 数值稳定性：深度图可能包含Inf（障碍物太远）
    # 将Inf/NaN替换为安全距离上限（10米）
    depth = torch.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=0.0)
    depth = torch.clamp(depth, min=0.0, max=10.0)
    
    # 提取前方中心区域（40%中心区域）
    H, W = depth.shape[1], depth.shape[2]
    h_start, h_end = int(H * 0.3), int(H * 0.7)
    w_start, w_end = int(W * 0.3), int(W * 0.7)
    front_center = depth[:, h_start:h_end, w_start:w_end]

    # 计算中心区域“鲁棒最近距离”
    # 说明：深度图里经常会出现 0（无效/未命中）或极小值，直接取 min 会把避障惩罚拉爆。
    # 这里做法：
    # - 将 <= near_clip 的像素视为无效，替换成远处(10m)
    # - 取 5% 分位(kth)近似“最近障碍”而不被单个坏点支配
    front_flat = front_center.reshape(env.num_envs, -1)
    near_clip = 0.1
    front_flat = torch.where(front_flat > (near_clip + 1e-3), front_flat, torch.full_like(front_flat, 10.0))
    k = max(1, int(front_flat.shape[1] * 0.05))
    min_depth = torch.kthvalue(front_flat, k, dim=1).values
    min_depth = torch.clamp(min_depth, min=0.0, max=10.0)
    
    # 分段惩罚
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    # 极度危险：< 0.15m，大幅惩罚
    danger_mask = min_depth < danger_distance
    penalty[danger_mask] = -10.0 * (danger_distance - min_depth[danger_mask])
    
    # 警告距离：0.15~0.3m，轻微惩罚
    warning_mask = (min_depth >= danger_distance) & (min_depth < safe_distance)
    penalty[warning_mask] = -2.0 * (safe_distance - min_depth[warning_mask])
    
    # 最终安全检查：确保没有NaN/Inf
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-10.0)
    penalty = torch.clamp(penalty, min=-50.0, max=0.0)

    # ------------------------
    # LiDAR 通道：360° ranges 的“鲁棒最近距离”（优先使用）
    # ------------------------
    # 说明：我们默认 ray index 0 对应机器人前方（水平角 0°），因此前方扇区可取 [0..k]∪[N-k..N)
    # 若你的 LiDAR 角度约定不同，可再调整扇区索引逻辑。
    lidar_min_range_front = torch.full((env.num_envs,), 10.0, device=env.device)
    lidar_min_range_360 = torch.full((env.num_envs,), 10.0, device=env.device)
    try:
        if "lidar" in env.scene.sensors:
            lidar = env.scene.sensors["lidar"]
            hits = lidar.data.ray_hits_w  # (num_envs, N, 3)
            pos = lidar.data.pos_w  # (num_envs, 3)
            ranges = torch.linalg.norm(hits - pos.unsqueeze(1), dim=-1)  # (num_envs, N)

            # 数值清理
            ranges = torch.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.0)
            ranges = torch.clamp(ranges, min=0.0, max=10.0)

            # 鲁棒 min：过滤极近(<=near_clip)的坏点，再取 1% 分位
            ranges_flat = ranges
            ranges_flat = torch.where(
                ranges_flat > (near_clip + 1e-3),
                ranges_flat,
                torch.full_like(ranges_flat, 10.0),
            )
            n_rays = int(ranges_flat.shape[1])
            k_lidar = max(1, int(n_rays * 0.01))
            lidar_min_range_360 = torch.kthvalue(ranges_flat, k_lidar, dim=1).values

            # 前方扇区（默认 ±60°）
            half = max(1, int(n_rays * (60.0 / 360.0)))
            front = torch.cat([ranges_flat[:, : half + 1], ranges_flat[:, -half:]], dim=1)
            k_front = max(1, int(front.shape[1] * 0.01))
            lidar_min_range_front = torch.kthvalue(front, k_front, dim=1).values
    except Exception:
        # 如果 LiDAR 不可用，保持默认的 10m
        pass

    # ------------------------
    # 融合：取“更危险”的那个（更小距离）
    # ------------------------
    fused_min_dist = torch.minimum(min_depth, lidar_min_range_front)

    # 分段惩罚（基于融合后的最近距离）
    penalty = torch.zeros(env.num_envs, device=env.device)
    danger_mask = fused_min_dist < danger_distance
    penalty[danger_mask] = -10.0 * (danger_distance - fused_min_dist[danger_mask])
    warning_mask = (fused_min_dist >= danger_distance) & (fused_min_dist < safe_distance)
    penalty[warning_mask] = -2.0 * (safe_distance - fused_min_dist[warning_mask])
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-10.0)
    penalty = torch.clamp(penalty, min=-50.0, max=0.0)

    # 记录调试指标（供训练脚本/日志读取，确认惩罚信号是否在生效）
    try:
        env.last_front_min_depth = min_depth.detach()
        env.last_lidar_min_range_front = lidar_min_range_front.detach()
        env.last_lidar_min_range_360 = lidar_min_range_360.detach()
        env.last_obstacle_min_dist = fused_min_dist.detach()
        env.last_obstacle_penalty = penalty.detach()
    except Exception:
        pass
    
    return penalty

def stability_penalty(env: "ManagerBasedRLEnv", roll_threshold: float = 0.3, 
                      pitch_threshold: float = 0.3) -> torch.Tensor:
    """
    惩罚机器人倾斜（翻车风险）
    
    Args:
        env: RL环境
        roll_threshold: roll角阈值 (弧度)
        pitch_threshold: pitch角阈值 (弧度)
        
    Returns:
        惩罚值 (num_envs,) - 负值
    """
    quat = env.scene.articulations["robot"].data.root_quat_w  # (num_envs, 4) [w,x,y,z]
    
    # 计算roll和pitch角
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll = atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Pitch = asin(2*(w*y - z*x))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    
    # 计算超出阈值的角度
    roll_penalty = torch.clamp(torch.abs(roll) - roll_threshold, min=0.0)
    pitch_penalty = torch.clamp(torch.abs(pitch) - pitch_threshold, min=0.0)
    
    penalty = -(roll_penalty + pitch_penalty)
    
    return penalty


def height_penalty(env: "ManagerBasedRLEnv", 
                   min_height: float = 0.05, 
                   max_height: float = 0.3) -> torch.Tensor:
    """
    惩罚高度异常（离地或跳起）
    
    Args:
        env: RL环境
        min_height: 最小允许高度
        max_height: 最大允许高度
        
    Returns:
        惩罚值 (num_envs,) - 负值
    """
    height = env.scene.articulations["robot"].data.root_pos_w[:, 2]
    
    # 低于最小高度或高于最大高度都惩罚
    below_min = torch.clamp(min_height - height, min=0.0)
    above_max = torch.clamp(height - max_height, min=0.0)
    
    penalty = -(below_min + above_max) * 10.0  # 放大惩罚
    
    return penalty


##
# 终止条件 (Termination Functions)
##

def goal_reached_termination(env: "ManagerBasedRLEnv", 
                             distance_threshold: float = 0.5) -> torch.Tensor:
    """
    到达目标点终止episode
    
    Args:
        env: RL环境
        distance_threshold: 到达阈值 (米)
        
    Returns:
        布尔张量 (num_envs,)
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    reached = distance < distance_threshold
    
    return reached


def robot_fallen_termination(env: "ManagerBasedRLEnv",
                             roll_threshold: float = 0.5,
                             pitch_threshold: float = 0.5) -> torch.Tensor:
    """
    机器人倾覆终止episode
    
    Args:
        env: RL环境
        roll_threshold: roll角阈值 (弧度)
        pitch_threshold: pitch角阈值 (弧度)
        
    Returns:
        布尔张量 (num_envs,)
    """
    quat = env.scene.articulations["robot"].data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    
    fallen = (torch.abs(roll) > roll_threshold) | (torch.abs(pitch) > pitch_threshold)
    
    return fallen


def backward_termination(env: "ManagerBasedRLEnv",
                        backward_threshold: float = -0.1,
                        duration_steps: int = 50) -> torch.Tensor:
    """
    持续倒退终止条件（防止机器人倒退逃避任务）
    
    如果机器人朝向目标的速度持续为负（倒退），则终止 episode。
    这避免了机器人通过"倒退逃离目标"来避免惩罚的策略。
    
    Args:
        env: RL环境
        backward_threshold: 倒退速度阈值 (m/s)，低于此值算倒退
        duration_steps: 持续倒退的步数阈值
        
    Returns:
        布尔张量 (num_envs,)
    """
    if not hasattr(env, 'goal_positions'):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
    # 计算朝向目标的速度
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    to_goal = goal_pos - robot_pos
    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)
    to_goal_dir = to_goal / (to_goal_norm + 1e-6)
    
    lin_vel_w = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]
    vel_toward_goal = torch.sum(lin_vel_w * to_goal_dir, dim=-1)
    
    # 初始化倒退计数器（如果不存在）
    if not hasattr(env, 'backward_count'):
        env.backward_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    
    # 更新倒退计数
    is_backward = vel_toward_goal < backward_threshold
    env.backward_count = torch.where(is_backward, env.backward_count + 1, torch.zeros_like(env.backward_count))
    
    # 如果持续倒退超过阈值步数，则终止
    should_terminate = env.backward_count >= duration_steps
    
    # 重置已终止环境的计数器（在下次 reset 时会自动重置）
    return should_terminate


##
# 事件函数 (Event Functions)
##

def reset_robot_to_random_position(env: "ManagerBasedRLEnv",
                                   env_ids: torch.Tensor,
                                   x_range: tuple = (-4.0, 4.0),
                                   y_range: tuple = (-4.0, 4.0),
                                   yaw_range: tuple = (-3.14, 3.14)):
    """
    重置机器人到随机位置和朝向（避开墙壁和障碍物）
    
    改进:
    - 适应10x10m室内场景
    - 避开墙壁边界（±5m）和已知障碍物区域
    
    Args:
        env: RL环境
        env_ids: 需要重置的环境ID
        x_range: x坐标范围（默认-4到4，留出1m墙壁缓冲）
        y_range: y坐标范围
        yaw_range: yaw角范围 (弧度)
    """
    num_resets = len(env_ids)
    
    # 定义障碍物禁区（避开家具）
    # 格式：(x_min, x_max, y_min, y_max)
    obstacle_zones = [
        (0.25, 1.75, 0.85, 2.15),    # 大桌子1
        (-2.5, -1.5, -2.5, -1.5),    # 小桌子2
        (2.8, 3.2, -1.7, -1.3),      # 柱子1
        (-3.2, -2.8, 2.3, 2.7),      # 柱子2
        (4.25, 4.75, 2.0, 4.0),      # 柜子1
        (-4.75, -4.25, -4.0, -3.0),  # 柜子2
    ]
    
    # 生成随机位置直到找到有效位置
    valid_positions = torch.zeros(num_resets, dtype=torch.bool, device=env.device)
    x = torch.zeros(num_resets, device=env.device)
    y = torch.zeros(num_resets, device=env.device)
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # 生成候选位置
        x_candidates = torch.rand(num_resets, device=env.device) * (x_range[1] - x_range[0]) + x_range[0]
        y_candidates = torch.rand(num_resets, device=env.device) * (y_range[1] - y_range[0]) + y_range[0]
        
        # 检查是否在障碍物区域内
        is_valid = torch.ones(num_resets, dtype=torch.bool, device=env.device)
        for zone in obstacle_zones:
            in_zone = (x_candidates >= zone[0]) & (x_candidates <= zone[1]) & \
                     (y_candidates >= zone[2]) & (y_candidates <= zone[3])
            is_valid = is_valid & (~in_zone)
        
        # 更新未找到有效位置的环境
        need_position = ~valid_positions
        x[need_position] = torch.where(is_valid[need_position], x_candidates[need_position], x[need_position])
        y[need_position] = torch.where(is_valid[need_position], y_candidates[need_position], y[need_position])
        valid_positions = valid_positions | is_valid
        
        if valid_positions.all():
            break
    
    z = torch.ones(num_resets, device=env.device) * 0.10  # 固定高度
    
    # 随机生成朝向
    yaw = torch.rand(num_resets, device=env.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    
    # 转换yaw到四元数 [w, x, y, z]
    quat_w = torch.cos(yaw / 2)
    quat_x = torch.zeros(num_resets, device=env.device)
    quat_y = torch.zeros(num_resets, device=env.device)
    quat_z = torch.sin(yaw / 2)
    
    # 设置机器人位置和朝向
    robot = env.scene.articulations["robot"]
    robot.write_root_pose_to_sim(
        root_pose=torch.stack([x, y, z, quat_w, quat_x, quat_y, quat_z], dim=-1),
        env_ids=env_ids
    )
    
    # 重置速度
    robot.write_root_velocity_to_sim(
        root_velocity=torch.zeros(num_resets, 6, device=env.device),
        env_ids=env_ids
    )
    
    # 重置倒退计数器（如果存在）
    if hasattr(env, 'backward_count'):
        env.backward_count[env_ids] = 0


def reset_goal_position(env: "ManagerBasedRLEnv",
                       env_ids: torch.Tensor,
                       min_distance: float = 2.0,
                       max_distance: float = 6.0):
    """
    重置目标位置（室内场景，避开障碍物）
    
    改进:
    - 目标点必须在可导航区域（避开墙壁和家具）
    - 确保与机器人有合理距离
    - 适应10x10m室内布局
    
    Args:
        env: RL环境
        env_ids: 需要重置的环境ID
        min_distance: 最小距离（降低到2m，适应室内）
        max_distance: 最大距离（降低到6m）
    """
    num_resets = len(env_ids)
    
    # 初始化goal_positions如果不存在
    if not hasattr(env, 'goal_positions'):
        env.goal_positions = torch.zeros(env.num_envs, 3, device=env.device)
    
    # 定义障碍物禁区（与reset_robot_to_random_position一致）
    obstacle_zones = [
        (0.25, 1.75, 0.85, 2.15),
        (-2.5, -1.5, -2.5, -1.5),
        (2.8, 3.2, -1.7, -1.3),
        (-3.2, -2.8, 2.3, 2.7),
        (4.25, 4.75, 2.0, 4.0),
        (-4.75, -4.25, -4.0, -3.0),
    ]
    
    # 室内边界（留出0.5m缓冲）
    x_min, x_max = -4.5, 4.5
    y_min, y_max = -4.5, 4.5
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[env_ids, :2]
    
    valid_goals = torch.zeros(num_resets, dtype=torch.bool, device=env.device)
    goal_x = torch.zeros(num_resets, device=env.device)
    goal_y = torch.zeros(num_resets, device=env.device)
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # 随机距离和角度
        distance = torch.rand(num_resets, device=env.device) * (max_distance - min_distance) + min_distance
        angle = torch.rand(num_resets, device=env.device) * 2 * 3.14159
        
        # 计算候选目标位置
        goal_x_candidates = robot_pos[:, 0] + distance * torch.cos(angle)
        goal_y_candidates = robot_pos[:, 1] + distance * torch.sin(angle)
        
        # 检查是否在边界内
        in_bounds = (goal_x_candidates >= x_min) & (goal_x_candidates <= x_max) & \
                    (goal_y_candidates >= y_min) & (goal_y_candidates <= y_max)
        
        # 检查是否在障碍物区域
        is_valid = in_bounds.clone()
        for zone in obstacle_zones:
            in_zone = (goal_x_candidates >= zone[0]) & (goal_x_candidates <= zone[1]) & \
                     (goal_y_candidates >= zone[2]) & (goal_y_candidates <= zone[3])
            is_valid = is_valid & (~in_zone)
        
        # 更新未找到有效目标的环境
        need_goal = ~valid_goals
        goal_x[need_goal] = torch.where(is_valid[need_goal], goal_x_candidates[need_goal], goal_x[need_goal])
        goal_y[need_goal] = torch.where(is_valid[need_goal], goal_y_candidates[need_goal], goal_y[need_goal])
        valid_goals = valid_goals | is_valid
        
        if valid_goals.all():
            break
    
    goal_z = torch.zeros(num_resets, device=env.device)
    env.goal_positions[env_ids] = torch.stack([goal_x, goal_y, goal_z], dim=-1)
    
    # 重置距离追踪和里程碑
    if hasattr(env, 'previous_goal_distance'):
        robot_pos_full = env.scene.articulations["robot"].data.root_pos_w[env_ids, :2]
        env.previous_goal_distance[env_ids] = torch.norm(
            robot_pos_full - env.goal_positions[env_ids, :2], 
            dim=-1
        )
    
    # 重置里程碑标记
    if hasattr(env, 'milestone_reached_1m'):
        env.milestone_reached_1m[env_ids] = False
    if hasattr(env, 'milestone_reached_2m'):
        env.milestone_reached_2m[env_ids] = False

