"""
测试ROSOrin环境的基本动作控制

验证轮速命令是否能正确控制机器人运动。
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.envs import ManagerBasedRLEnv
import rosorin_env_cfg

print("="*70)
print("  ROSOrin基础动作测试")
print("="*70)

# 创建环境
print("\n创建环境...")
env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)

# 重置
obs, info = env.reset()
print(f"✓ 环境就绪")

print("\n测试1: 零动作（保持静止）")
print("-" * 70)
action = torch.zeros(1, 4, device=env.device)  # [0, 0, 0, 0]

for step in range(5):
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 读取状态
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[0].cpu().numpy()
    robot_vel = env.scene.articulations["robot"].data.root_lin_vel_w[0].cpu().numpy()
    joint_vel = env.scene.articulations["robot"].data.joint_vel[0].cpu().numpy()
    
    print(f"步数 {step}: 位置=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
          f"速度=({robot_vel[0]:.3f}, {robot_vel[1]:.3f}), "
          f"轮速={joint_vel}")

print("\n测试2: 全速前进（所有轮子10 rad/s）")
print("-" * 70)
action = torch.ones(1, 4, device=env.device) * 10.0  # [10, 10, 10, 10]

for step in range(10):
    obs, reward, terminated, truncated, info = env.step(action)
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[0].cpu().numpy()
    robot_vel = env.scene.articulations["robot"].data.root_lin_vel_w[0].cpu().numpy()
    joint_vel = env.scene.articulations["robot"].data.joint_vel[0].cpu().numpy()
    
    if step % 2 == 0:
        print(f"步数 {step}: 位置=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
              f"速度=({robot_vel[0]:.3f}, {robot_vel[1]:.3f}), "
              f"轮速平均={joint_vel.mean():.2f}")

print("\n测试3: 原地旋转（左轮-10, 右轮+10）")
print("-" * 70)
# [左前, 右前, 左后, 右后]
action = torch.tensor([[-10.0, 10.0, -10.0, 10.0]], device=env.device)

env.reset()
for step in range(10):
    obs, reward, terminated, truncated, info = env.step(action)
    
    robot_pos = env.scene.articulations["robot"].data.root_pos_w[0].cpu().numpy()
    robot_quat = env.scene.articulations["robot"].data.root_quat_w[0].cpu().numpy()
    robot_ang_vel = env.scene.articulations["robot"].data.root_ang_vel_w[0].cpu().numpy()
    
    # 计算yaw角
    w, x, y, z = robot_quat
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)).item()
    
    if step % 2 == 0:
        print(f"步数 {step}: 位置=({robot_pos[0]:.3f}, {robot_pos[1]:.3f}), "
              f"yaw={yaw:.3f} rad, ang_vel_z={robot_ang_vel[2]:.3f}")

print("\n测试4: 检查动作管理器配置")
print("-" * 70)
print(f"动作空间维度: {env.num_actions}")
print(f"动作管理器: {env.action_manager}")

# 检查关节名称
robot = env.scene.articulations["robot"]
print(f"\n机器人关节信息:")
print(f"  关节名称: {robot.joint_names}")
print(f"  关节数量: {robot.num_joints}")
print(f"  可动关节数量: {robot.num_actuated_joints}")

env.close()
simulation_app.close()

print("\n" + "="*70)
print("  测试完成")
print("="*70)
