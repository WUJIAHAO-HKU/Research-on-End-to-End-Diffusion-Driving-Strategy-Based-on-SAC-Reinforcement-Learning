"""
快速测试MPC数据收集流程

收集少量演示数据以验证整个pipeline。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
import rosorin_env_cfg
from mpc_controller import MPCController, MPCConfig
from path_generator import PathGenerator, PathType

print("="*70)
print("  MPC数据收集快速测试")
print("="*70)

# 创建环境
print("\n[1/3] 创建环境...")
env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = 10.0  # 短轨迹用于测试
env = ManagerBasedRLEnv(cfg=env_cfg)
print(f"✓ 环境创建成功")

# 创建MPC和路径生成器
print("\n[2/3] 创建MPC控制器...")
mpc = MPCController(MPCConfig(max_linear_vel=0.5))
path_gen = PathGenerator(seed=42)
print("✓ 控制器就绪")

# 测试收集一条轨迹
print("\n[3/3] 测试收集轨迹...")
print("-" * 70)

# 生成简单直线路径
target_positions, target_headings = path_gen.generate_straight_line(
    start=np.array([0.0, 0.0]),
    end=np.array([2.0, 0.0]),
    num_points=50
)

# 重置环境
obs, info = env.reset()
mpc.reset()

# 收集数据
observations = []
actions = []
rewards = []

max_steps = 500
for step in range(max_steps):
    # 获取机器人状态
    obs_tensor = obs['policy'][0].cpu().numpy()  # (10,)
    
    # 获取位置和朝向
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[0].cpu().numpy()
    robot_quat = env.scene.articulations["robot"].data.root_quat_w[0].cpu().numpy()
    
    # 提取yaw角
    w, x, y, z = robot_quat
    theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    # 提取速度
    lin_vel = obs_tensor[:3]
    ang_vel = obs_tensor[3:6]
    
    # 构造状态
    robot_state = np.array([
        robot_pos[0], robot_pos[1], theta,
        lin_vel[0], lin_vel[1], ang_vel[2]
    ])
    
    # MPC控制
    control, mpc_info = mpc.compute_control(
        robot_state, target_positions, target_headings
    )
    wheel_vel = mpc.robot_velocity_to_wheel_velocity(control)
    
    # 执行动作
    action = torch.from_numpy(wheel_vel).float().unsqueeze(0).to(env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 记录数据
    observations.append(obs_tensor)
    actions.append(wheel_vel)
    rewards.append(reward[0].cpu().item())
    
    # 打印进度
    if step % 100 == 0:
        distance = np.linalg.norm(robot_pos[:2] - target_positions[-1])
        print(f"  步数: {step:3d} | 位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) | "
              f"距离目标: {distance:.3f}m | 奖励: {reward[0].cpu().item():.3f}")
    
    # 检查是否到达
    distance_to_goal = np.linalg.norm(robot_pos[:2] - target_positions[-1])
    if distance_to_goal < 0.2:
        print(f"\n  ✓ 到达目标! (步数: {step}, 距离: {distance_to_goal:.3f}m)")
        break
    
    if terminated[0] or truncated[0]:
        print(f"\n  ✗ 轨迹终止 (步数: {step})")
        break

# 统计
observations = np.array(observations)
actions = np.array(actions)
rewards = np.array(rewards)

print("\n" + "="*70)
print("  收集结果")
print("="*70)
print(f"收集步数: {len(observations)}")
print(f"观测形状: {observations.shape}")
print(f"动作形状: {actions.shape}")
print(f"总奖励: {np.sum(rewards):.3f}")
print(f"平均奖励: {np.mean(rewards):.3f}")
print(f"最终距离: {distance_to_goal:.3f}m")

print("\n动作统计:")
print(f"  轮速范围: [{actions.min():.2f}, {actions.max():.2f}] rad/s")
print(f"  平均轮速: {np.abs(actions).mean():.2f} rad/s")

print("\n观测统计:")
print(f"  线速度范围: [{observations[:, :3].min():.3f}, {observations[:, :3].max():.3f}] m/s")
print(f"  角速度范围: [{observations[:, 3:6].min():.3f}, {observations[:, 3:6].max():.3f}] rad/s")

# 清理
env.close()
simulation_app.close()

print("\n" + "="*70)
print("  ✓ 快速测试完成!")
print("="*70)
print("\n如果测试成功，可以运行完整数据收集:")
print("  ./isaaclab_runner.sh scripts/collect_mpc_demonstrations.py")
