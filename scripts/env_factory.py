"""
环境配置工厂函数

该模块提供创建不同算法环境配置的工厂函数。
每个算法可以使用不同的奖励权重配置。

注意: 工厂函数必须在 AppLauncher 启动后调用！
"""

import sys
import os

# 添加项目根目录到Python path以导入配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def create_ppo_env_cfg(num_envs=8, env_spacing=5.0):
    """
    创建PPO算法专用的环境配置
    
    Args:
        num_envs: 并行环境数量
        env_spacing: 环境间距
    
    Returns:
        ROSOrinEnvCfg: 配置了PPO奖励权重的环境
    """
    # 在函数内导入，避免在AppLauncher启动前导入Isaac Lab模块
    from configs.rewards.ppo_rewards import PPORewardsCfg
    from rosorin_env_cfg import ROSOrinEnvCfg
    
    # 创建环境配置实例
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    
    # 替换为PPO专用奖励配置
    env_cfg.rewards = PPORewardsCfg()
    
    return env_cfg


def create_sac_env_cfg(num_envs=8, env_spacing=5.0):
    """创建SAC算法专用的环境配置"""
    from configs.rewards.sac_rewards import SACRewardsCfg
    from rosorin_env_cfg import ROSOrinEnvCfg
    
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    env_cfg.rewards = SACRewardsCfg()
    
    return env_cfg


def create_bc_env_cfg(num_envs=8, env_spacing=5.0):
    """创建BC算法专用的环境配置"""
    from configs.rewards.bc_rewards import BCRewardsCfg
    from rosorin_env_cfg import ROSOrinEnvCfg
    
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    env_cfg.rewards = BCRewardsCfg()
    
    return env_cfg


def create_td3_env_cfg(num_envs=8, env_spacing=5.0):
    """创建TD3算法专用的环境配置"""
    from configs.rewards.td3_rewards import TD3RewardsCfg
    from rosorin_env_cfg import ROSOrinEnvCfg
    
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    env_cfg.rewards = TD3RewardsCfg()
    
    return env_cfg


def create_dagger_env_cfg(num_envs=8, env_spacing=5.0):
    """创建DAgger算法专用的环境配置"""
    from configs.rewards.dagger_rewards import DAggerRewardsCfg
    from rosorin_env_cfg import ROSOrinEnvCfg
    
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    env_cfg.rewards = DAggerRewardsCfg()
    
    return env_cfg
