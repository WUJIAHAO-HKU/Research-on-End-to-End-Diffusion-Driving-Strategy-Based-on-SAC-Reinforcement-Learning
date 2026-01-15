"""
基础奖励配置类

该模块定义了所有奖励函数的基础配置类，供各算法继承和定制。
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
class BaseRewardsCfg:
    """
    基础奖励配置类
    
    所有奖励函数的默认配置，各算法可以继承并覆盖特定参数。
    
    设计原则:
    1. 密集奖励: progress + orientation + velocity
    2. 稀疏奖励: goal_reached
    3. 平滑惩罚: action_smoothness + stability + height + obstacle_avoidance
    """
    
    # ========== 主要密集奖励 (Dense Rewards) ==========
    progress = RewTerm(
        func=rosorin_mdp.progress_reward,
        weight=15.0,
        params={"threshold": 0.001}
    )
    
    orientation = RewTerm(
        func=rosorin_mdp.orientation_alignment_reward,
        weight=3.0
    )
    
    velocity_tracking = RewTerm(
        func=rosorin_mdp.velocity_tracking_reward,
        weight=2.0,
        params={"target_vel": 0.3}
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
        weight=0.5
    )
    
    obstacle_avoidance = RewTerm(
        func=rosorin_mdp.obstacle_avoidance_penalty,
        weight=3.0,
        params={
            "safe_distance": 0.4,
            "danger_distance": 0.2
        }
    )
    
    stability = RewTerm(
        func=rosorin_mdp.stability_penalty,
        weight=5.0,
        params={
            "roll_threshold": 0.2,
            "pitch_threshold": 0.2
        }
    )
    
    height = RewTerm(
        func=rosorin_mdp.height_penalty,
        weight=2.0,
        params={
            "min_height": 0.05,
            "max_height": 0.3
        }
    )
