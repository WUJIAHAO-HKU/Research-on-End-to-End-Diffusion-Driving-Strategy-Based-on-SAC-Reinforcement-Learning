#!/usr/bin/env python3
"""详细调试物理问题"""

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.envs import ManagerBasedRLEnv
import rosorin_env_cfg

env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)

robot = env.scene.articulations["robot"]

print("\n" + "="*70)
print("  物理系统调试")
print("="*70)

print(f"\n1. 重力检查:")
print(f"  Gravity enabled: {not env_cfg.scene.robot.spawn.rigid_props.disable_gravity}")

print(f"\n2. 初始位置:")
obs, _ = env.reset()
print(f"  Base位置: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"  Base速度: {robot.data.root_lin_vel_w[0].cpu().numpy()}")

# 让重力作用
print(f"\n3. 重力下降测试（零动作，50步）:")
for i in range(50):
    actions = torch.zeros(1, 4, device=env.device)
    obs, _, _, _, _ = env.step(actions)
    
    if i % 10 == 9:
        pos = robot.data.root_pos_w[0].cpu().numpy()
        vel = robot.data.root_lin_vel_w[0].cpu().numpy()
        print(f"  Step {i+1}: pos_z={pos[2]:.4f}, vel_z={vel[2]:.4f}")

print(f"\n4. 最终状态:")
pos = robot.data.root_pos_w[0].cpu().numpy()
print(f"  位置: {pos}")
print(f"  地面: z=-0.05 (顶面 z=0)")

if pos[2] < 0.01:
    print("  ✓ 机器人已下降")
else:
    print(f"  ✗ 机器人未下降 (z={pos[2]})")

print("\n" + "="*70)
simulation_app.close()
