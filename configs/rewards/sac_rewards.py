"""
SAC算法专用奖励配置

该模块定义SAC（Soft Actor-Critic）算法的奖励权重配置。
SAC是off-policy算法，使用经验回放，可以容忍更高的探索噪声。

优化策略:
- 平衡探索与利用
- 适度的惩罚项，避免过度保守
- 利用熵正则化鼓励探索
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

# 导入自定义MDP函数
import sys
import os
mdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mdp'))
if mdp_path not in sys.path:
    sys.path.insert(0, mdp_path)
import rosorin_mdp


@configclass
class SACRewardsCfg:
    """
    SAC算法奖励配置（重新平衡：强化导航信号）
    
    权重平衡（V2 - 导航主导）:
    - 主导航奖励: progress(80) + orientation(10) + velocity(8) = 98
    - 稀疏奖励: goal_reached(800)
    - 惩罚项: obstacle(1.5) + smooth(0.2) + stability(2) + height(0.5) = 4.2
    
    设计理念:
    1. 导航奖励主导（占比 ~95%）：强迫策略主动接近目标
    2. 惩罚项降权（占比 ~5%）：避免过度保守，鼓励探索
    3. 大额到达奖励（800）：提供明确的终极目标
    """
    
    # ========== 主要密集奖励 (Dense Rewards) ==========
    # 改为距离奖励：直接奖励"距离目标更近"，而不依赖速度
    progress = RewTerm(
        func=rosorin_mdp.distance_to_goal_reward,  # ↓ 改用距离奖励（更直接）
        weight=100.0,  # ↑ 提升到 100（主导信号）
        params={"scale": 2.0}  # 距离每减少 1m → 奖励 +2.0
    )
    
    orientation = RewTerm(
        func=rosorin_mdp.orientation_alignment_reward,
        weight=10.0
    )
    
    velocity_tracking = RewTerm(
        func=rosorin_mdp.velocity_tracking_reward,
        weight=12.0,  # ↑ 从 8 提升到 12（鼓励快速移动）
        params={"target_vel": 0.4}  # ↓ 降速从 0.5 到 0.4（更实际）
    )
    
    # ========== 稀疏奖励 (Sparse Rewards) ==========
    goal_reached = RewTerm(
        func=rosorin_mdp.goal_reached_reward,
        weight=1200.0,  # ↑↑↑ 从800提升到1200（强化目标激励）
        params={"distance_threshold": 0.5}
    )
    
    # ========== 辅助奖励 (Auxiliary) ==========
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.0  # ↓ 从0.01降到0（避免"原地不动"策略）
    )
    
    # 惩罚停滞（强制机器人移动）
    movement_penalty = RewTerm(
        func=rosorin_mdp.movement_penalty,
        weight=5.0,  # ↓ 从 15.0 降到 5.0（避免过度惩罚）
        params={"min_speed": 0.1}  # ↓ 从 0.2 降到 0.1 m/s（放宽要求）
    )
    
    # ========== 惩罚项 (Penalties - 降权避免过度保守) ==========
    action_smoothness = RewTerm(
        func=rosorin_mdp.smooth_action_penalty,
        weight=0.2  # ↓ 从0.3降到0.2
    )
    
    obstacle_avoidance = RewTerm(
        func=rosorin_mdp.obstacle_avoidance_penalty,
        weight=1.0,  # ↓ 从 1.5 降到 1.0（降低避障保守性）
        params={
            "safe_distance": 0.3,   # ↓ 从 0.4 降到 0.3（允许更近距离通过）
            "danger_distance": 0.15 # 保持 0.15m（极近才重罚）
        }
    )
    
    # 冲刺奖励：当距离<2m且速度>0.15m/s时，额外奖励
    sprint_bonus = RewTerm(
        func=rosorin_mdp.sprint_bonus_reward,
        weight=20.0,  # 强激励
        params={
            "distance_threshold": 2.0,
            "min_speed": 0.15
        }
    )
    
    stability = RewTerm(
        func=rosorin_mdp.stability_penalty,
        weight=2.0,  # ↓ 从4.0降到2.0
        params={
            "roll_threshold": 0.3,   # ↑ 从0.25放宽到0.3
            "pitch_threshold": 0.3   # ↑ 从0.25放宽到0.3
        }
    )
    
    height = RewTerm(
        func=rosorin_mdp.height_penalty,
        weight=0.5,  # ↓ 从1.0降到0.5
        params={
            "min_height": 0.03,  # ↓ 从0.04放宽到0.03
            "max_height": 0.4    # ↑ 从0.35放宽到0.4
        }
    )
