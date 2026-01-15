"""
TD3算法专用奖励配置

该模块定义TD3（Twin Delayed DDPG）算法的奖励权重配置。
TD3是off-policy确定性策略算法，类似SAC但不使用熵正则化。

优化策略:
- 类似SAC但略微保守
- 更注重动作平滑性（确定性策略）
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
class TD3RewardsCfg:
    """
    TD3算法奖励配置
    
    权重平衡:
    - 主导航奖励: progress(17) + orientation(4) + velocity(2.5) = 23.5
    - 惩罚项略高于SAC（确定性策略需要更平滑）
    """
    
    # ========== 主要密集奖励 (Dense Rewards) ==========
    progress = RewTerm(
        func=rosorin_mdp.progress_reward,
        weight=17.0,
        params={"threshold": 0.001}
    )
    
    orientation = RewTerm(
        func=rosorin_mdp.orientation_alignment_reward,
        weight=4.0
    )
    
    velocity_tracking = RewTerm(
        func=rosorin_mdp.velocity_tracking_reward,
        weight=2.5,
        params={"target_vel": 0.35}
    )
    
    # ========== 稀疏奖励 (Sparse Rewards) ==========
    goal_reached = RewTerm(
        func=rosorin_mdp.goal_reached_reward,
        weight=100.0,
        params={"distance_threshold": 0.5}
    )
    
    # ========== 辅助奖励 (Auxiliary) ==========
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.01
    )
    
    # ========== 惩罚项 (Penalties) ==========
    action_smoothness = RewTerm(
        func=rosorin_mdp.smooth_action_penalty,
        weight=0.5  # ↑ 确定性策略更注重平滑
    )
    
    obstacle_avoidance = RewTerm(
        func=rosorin_mdp.obstacle_avoidance_penalty,
        weight=2.5,
        params={
            "safe_distance": 0.45,
            "danger_distance": 0.22
        }
    )
    
    stability = RewTerm(
        func=rosorin_mdp.stability_penalty,
        weight=4.5,
        params={
            "roll_threshold": 0.25,
            "pitch_threshold": 0.25
        }
    )
    
    height = RewTerm(
        func=rosorin_mdp.height_penalty,
        weight=1.5,
        params={
            "min_height": 0.04,
            "max_height": 0.35
        }
    )
