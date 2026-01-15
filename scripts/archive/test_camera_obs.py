#!/usr/bin/env python3
"""测试相机集成到观测空间"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosorin_env_cfg

print("\n" + "="*70)
print("  测试相机观测集成")
print("="*70)

print("\n创建环境...")
env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)
print("✓ 环境创建成功")

print("\n观测空间信息:")
obs, _ = env.reset()
for key, value in obs.items():
    print(f"  {key}: {value.shape}, dtype={value.dtype}")
    if 'camera' in key:
        print(f"    范围: [{value.min():.3f}, {value.max():.3f}]")

print(f"\n总观测维度: {sum(v.numel() for v in obs.values())} elements")

# 测试5步
print("\n运行5步测试...")
for i in range(5):
    actions = torch.zeros(1, 4, device=env.device)
    obs, rewards, dones, truncated, info = env.step(actions)
    
    rgb_data = obs['policy'][0, 10:10+160*120*3]  # 跳过前10个(vel)
    depth_data = obs['policy'][0, 10+160*120*3:]
    
    print(f"  Step {i}: rgb_range=[{rgb_data.min():.2f}, {rgb_data.max():.2f}], "
          f"depth_mean={depth_data.mean():.2f}m")

print("\n✓ 相机观测集成成功!")

env.close()
simulation_app.close()
