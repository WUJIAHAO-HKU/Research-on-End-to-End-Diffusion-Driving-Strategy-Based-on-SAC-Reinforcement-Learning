#!/usr/bin/env python3
"""
简单的ROSOrin环境测试 - 验证环境是否正常工作
"""

import sys
sys.path.insert(0, "/home/wujiahao/IsaacLab/_build/linux-x86_64/release")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "enable_cameras": True})

import torch
import rosorin_env_cfg
from isaaclab.envs import ManagerBasedRLEnv

print("\n" + "="*70)
print("  ROSOrin环境快速测试")
print("="*70)

# 创建环境
env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = 2  # 只用2个环境快速测试
env = ManagerBasedRLEnv(cfg=env_cfg)

print(f"\n✓ 环境创建成功")
print(f"  - 环境数量: {env.num_envs}")
print(f"  - 观测维度: {env.observation_manager.group_obs_dim['policy']}")
print(f"  - 动作维度: {env.action_manager.total_action_dim}")

# 重置环境
obs, _ = env.reset()
print(f"\n✓ 环境重置成功")
print(f"  - 观测形状: {obs['policy'].shape}")

# 运行10步
print(f"\n运行10步测试...")
for i in range(10):
    actions = 0.1 * torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    obs, rewards, terminated, truncated, info = env.step(actions)
    if i % 5 == 0:
        print(f"  步骤 {i}: 平均奖励={rewards.mean():.3f}")

print(f"\n" + "="*70)
print("✅ 测试通过！ROSOrin环境运行正常")
print("="*70)

env.close()
simulation_app.close()
