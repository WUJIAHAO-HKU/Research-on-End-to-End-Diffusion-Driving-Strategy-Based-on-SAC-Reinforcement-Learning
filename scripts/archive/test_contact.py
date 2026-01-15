#!/usr/bin/env python3
"""检查机器人与地面接触情况"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.envs import ManagerBasedRLEnv
import rosorin_env_cfg

print("\n" + "="*70)
print("  检查机器人接触情况")
print("="*70)

env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = ManagerBasedRLEnv(cfg=env_cfg)

robot = env.scene.articulations["robot"]
contact_sensor = env.scene.sensors["contact_sensor"]

print(f"\n初始状态:")
print(f"  机器人位置: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"  地面位置: [0.0, 0.0, -0.05]")
print(f"  机器人质量: {robot.root_physx_view.get_masses().sum().item():.3f} kg")

# 检查初始接触
obs, _ = env.reset()
env.sim.step()  # 让物理稳定

print(f"\n接触传感器数据:")
print(f"  接触力范数: {contact_sensor.data.net_forces_w_history[0, :, :].cpu().numpy()}")
print(f"  是否在空中: {contact_sensor.data.current_air_time[0].item()}")
print(f"  接触时间: {contact_sensor.data.current_contact_time[0].item()}")

# 给一个大动作，观察10步
print(f"\n施加大动作并观察接触:")
for i in range(10):
    actions = torch.ones(env.num_envs, 4, device=env.device) * 5.0
    obs, rewards, dones, truncated, info = env.step(actions)
    
    if i % 2 == 1:
        pos = robot.data.root_pos_w[0].cpu().numpy()
        vel = robot.data.root_lin_vel_w[0, :2].cpu().numpy()
        contact_force = contact_sensor.data.net_forces_w_history[0, 0, :].cpu().numpy()
        air_time = contact_sensor.data.current_air_time[0].item()
        
        print(f"\nStep {i+1}:")
        print(f"  位置: {pos}")
        print(f"  速度: {vel}")
        print(f"  接触力: {contact_force}")
        print(f"  空中时间: {air_time:.3f}s")

print("\n" + "="*70)
simulation_app.close()
