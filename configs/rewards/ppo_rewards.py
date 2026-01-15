"""
PPO算法专用奖励配置

该模块定义PPO（Proximal Policy Optimization）算法的奖励权重配置。
PPO是on-policy算法，需要较高的密集奖励引导以快速收敛。

优化策略:
- 增强主导航奖励（progress, orientation, velocity）
- 降低惩罚项权重，允许更多探索
- 放宽阈值，避免过早终止
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
class PPORewardsCfg:
    """
    PPO算法奖励配置 - 优化版本
    
    权重平衡:
    - 主导航奖励: progress(20) + orientation(5) + velocity(3) = 28
    - 稀疏奖励: goal_reached(100)
    - 惩罚项: obstacle(1) + smooth(0.1) + stability(3) + height(0.5) = 4.6
    
    设计思路: 主导航>>惩罚，鼓励探索和前进
    """
    
    # ========== 主要密集奖励 (Dense Rewards) ==========
    # 向目标前进的进度奖励 (最重要)
    progress = RewTerm(
        func=rosorin_mdp.progress_reward,
        weight=25.0,  # ↑ 强化主要目标（让“靠近目标”更主导）
        params={"threshold": 0.0}  # ✅ 不做deadzone过滤，最大化密集学习信号
    )
    
    # 朝向对齐奖励 (引导方向)
    orientation = RewTerm(
        func=rosorin_mdp.orientation_alignment_reward,
        weight=5.0  # ↑ 帮助快速对准目标
    )
    
    # 速度跟踪奖励 (鼓励移动)
    velocity_tracking = RewTerm(
        func=rosorin_mdp.velocity_tracking_reward,
        weight=3.0,
        params={"target_vel": 0.3}  # 保守速度，避免碰撞
    )
    
    # ========== 稀疏奖励 (Sparse Rewards) ==========
    # NOTE: 旧版 goal_reached_reward 内含里程碑(2m/1m)且此处权重=100，会导致单回合回报出现
    # 200~500 级别跳变，方差很大、解释困难。这里改为：
    # - weight 下调
    # - reached/milestone bonus 下调
    # 让“到达目标”仍明显，但不会淹没其它项。
    goal_reached = RewTerm(
        func=rosorin_mdp.goal_reached_reward,
        weight=20.0,
        params={
            "distance_threshold": 0.5,
            "reached_bonus": 1.0,
            "milestone_1m_bonus": 0.1,
            "milestone_2m_bonus": 0.05,
        }
    )
    
    # ========== 辅助奖励 (Auxiliary) ==========
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.01  # 极低权重，避免"原地不动"策略
    )
    
    # ========== 惩罚项 (Penalties) ==========
    action_smoothness = RewTerm(
        func=rosorin_mdp.smooth_action_penalty,
        weight=0.1  # ↓ 降低，允许更多探索
    )
    
    obstacle_avoidance = RewTerm(
        func=rosorin_mdp.obstacle_avoidance_penalty,
        weight=1.0,  # ↓ 降低，减少过度惩罚
        params={
            "safe_distance": 0.5,    # ↑ 增加安全距离
            "danger_distance": 0.25   # ↑ 增加危险距离
        }
    )
    
    stability = RewTerm(
        func=rosorin_mdp.stability_penalty,
        weight=3.0,  # ↓ 降低
        params={
            "roll_threshold": 0.3,   # ↑ 放宽
            "pitch_threshold": 0.3
        }
    )
    
    height = RewTerm(
        func=rosorin_mdp.height_penalty,
        weight=0.5,  # ↓ 大幅降低
        params={
            "min_height": 0.03,  # ↓ 放宽
            "max_height": 0.4
        }
    )
