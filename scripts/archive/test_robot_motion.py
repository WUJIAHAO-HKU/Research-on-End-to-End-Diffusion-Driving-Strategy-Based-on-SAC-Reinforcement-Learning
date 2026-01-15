#!/usr/bin/env python3
"""
Debug Robot Motion - 检查机器人为什么不移动

测试不同的动作值和actuator配置
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
import rosorin_env_cfg

print("\n" + "="*70)
print("  机器人运动调试")
print("="*70)

# 创建环境
env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = ManagerBasedRLEnv(cfg=env_cfg)

robot = env.scene.articulations["robot"]
print(f"\n机器人信息:")
print(f"  关节数: {robot.num_joints}")
print(f"  关节名: {robot.joint_names}")
print(f"  DOF数量: {robot.num_bodies} bodies, {robot.num_joints} joints")
print(f"  质量: {robot.root_physx_view.get_masses().cpu().numpy()}")

# 测试1: 检查零动作
print(f"\n[Test 1] 零动作 - 机器人应保持静止")
obs, _ = env.reset()
for i in range(10):
    actions = torch.zeros(env.num_envs, 4, device=env.device)
    obs, rewards, dones, truncated, info = env.step(actions)

pos = robot.data.root_pos_w[0].cpu().numpy()
vel = robot.data.root_lin_vel_w[0].cpu().numpy()
joint_vel = robot.data.joint_vel[0].cpu().numpy()
print(f"  位置: {pos}")
print(f"  速度: {vel}")
print(f"  轮速: {joint_vel}")

# 测试2: 最大动作
print(f"\n[Test 2] 最大正向动作 (+1.0)")
obs, _ = env.reset()
for i in range(20):
    # 所有轮子最大正向
    actions = torch.ones(env.num_envs, 4, device=env.device)
    obs, rewards, dones, truncated, info = env.step(actions)
    
    if i % 5 == 4:
        pos = robot.data.root_pos_w[0].cpu().numpy()
        vel = robot.data.root_lin_vel_w[0].cpu().numpy()
        joint_vel = robot.data.joint_vel[0].cpu().numpy()
        joint_effort = robot.data.applied_torque[0].cpu().numpy()
        print(f"  Step {i+1}: pos={pos}, vel={vel}")
        print(f"    轮速: {joint_vel}")
        print(f"    轮扭矩: {joint_effort}")

# 测试3: 仅前进
print(f"\n[Test 3] 仅前进 (所有轮子同向)")
obs, _ = env.reset()
for i in range(20):
    # 麦克纳姆轮前进：所有轮子同向旋转
    actions = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=env.device)
    obs, rewards, dones, truncated, info = env.step(actions)
    
    if i % 5 == 4:
        pos = robot.data.root_pos_w[0].cpu().numpy()
        vel = robot.data.root_lin_vel_w[0].cpu().numpy()
        print(f"  Step {i+1}: pos=[{pos[0]:.3f}, {pos[1]:.3f}], vel=[{vel[0]:.3f}, {vel[1]:.3f}]")

# 测试4: 检查applied torque
print(f"\n[Test 4] 检查力矩是否应用")
obs, _ = env.reset()
actions = torch.tensor([[5.0, 5.0, 5.0, 5.0]], device=env.device)  # 大动作
obs, rewards, dones, truncated, info = env.step(actions)

print(f"  动作指令: {actions}")
print(f"  关节目标速度: {robot.data.joint_vel_target[0]}")
print(f"  关节实际速度: {robot.data.joint_vel[0]}")
print(f"  应用扭矩: {robot.data.applied_torque[0]}")

print("\n" + "="*70)
simulation_app.close()
