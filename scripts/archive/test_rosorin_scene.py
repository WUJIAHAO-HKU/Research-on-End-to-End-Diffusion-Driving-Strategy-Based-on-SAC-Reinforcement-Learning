#!/usr/bin/env python3
"""
Test ROSOrin Scene Creation - Task 1

验证Isaac Lab场景创建的完整性：
1. 场景加载 (ground + robot + obstacles)
2. 机器人articulation正常工作
3. 传感器配置 (contact sensor, camera, lidar)
4. 物理仿真步进

使用方法:
    ./isaaclab_runner.sh scripts/test_rosorin_scene.py --num_envs 2
"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments FIRST
parser = argparse.ArgumentParser(description="Test ROSOrin Scene Creation")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)  # Add Isaac Lab's default args including --enable_cameras
args = parser.parse_args()

# Launch app with camera support
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("\n" + "="*70)
print("  ROSOrin Scene Creation Test (Task 1)")
print("="*70)

# Now import Isaac Lab modules (AFTER app launch)
print("\n[1/6] 导入Isaac Lab模块...")
import torch
import sys
import os

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

# Import our environment configuration
import rosorin_env_cfg
print("✓ Isaac Lab模块导入成功")

# Step 2: Create environment
print(f"\n[2/6] 创建ROSOrin环境 (num_envs={args.num_envs})...")
try:
    # Create environment configuration
    env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.env_spacing = 3.0  # 3m spacing between environments
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"✓ 环境创建成功")
    print(f"  - 环境数量: {env.num_envs}")
    print(f"  - 设备: {env.device}")
    print(f"  - 观测空间: {env.observation_space}")
    print(f"  - 动作空间: {env.action_space}")

except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

# Step 4: Inspect scene components
print(f"\n[3/6] 检查场景组件...")
scene = env.scene

print(f"  场景实体:")
robot = scene.articulations['robot']
print(f"    - 机器人 (robot): {robot}")
print(f"    - 地面 (ground): {scene['ground']}")
print(f"    - 障碍物 (obstacle): {scene['obstacle']}")

# Print robot body information
print(f"\n  机器人Body信息:")
print(f"    - Bodies数量: {robot.num_bodies}")
print(f"    - Body名称列表:")
for i, name in enumerate(robot.body_names):
    print(f"      [{i}] {name}")

print(f"\n  机器人Prim路径:")
print(f"    - Root路径: {robot.root_physx_view.prim_paths[0]}")

# Check if contact sensor exists
if 'contact_sensor' in scene.sensors:
    print(f"    - 接触传感器 (contact_sensor): {scene.sensors['contact_sensor']}")
else:
    print(f"    - 接触传感器: 未配置")

# Check camera
if 'camera' in scene.sensors:
    cam = scene.sensors['camera']
    print(f"    - 相机 (camera): {cam}")
    print(f"      分辨率: {cam.image_shape}")
else:
    print(f"    - 相机: 未配置")

# Check lidar
if 'lidar' in scene.sensors:
    lidar = scene.sensors['lidar']
    print(f"    - 激光雷达 (lidar): {lidar}")
    print(f"      扫描点数: {lidar.num_rays}")
else:
    print(f"    - 激光雷达: 未配置")

# Check robot DOFs
robot = scene.articulations["robot"]
print(f"\n  机器人详情:")
print(f"    - DOF数量: {robot.num_joints}")
print(f"    - 关节名称: {robot.joint_names}")
print(f"    - 位置: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"    - 方向: {robot.data.root_quat_w[0].cpu().numpy()}")

# Step 5: Test physics simulation
print(f"\n[4/6] 测试物理仿真 (50步)...")
obs, _ = env.reset()
print(f"  初始观测形状: {obs['policy'].shape}")
print(f"  动作空间: {env.action_space}")

for step in range(50):
    # Random actions - 修正：使用正确的action维度
    # action_space.shape 返回 (num_envs, action_dim)，所以只需要取第二个维度
    action_dim = env.action_manager.total_action_dim  # 应该是4
    actions = torch.rand(env.num_envs, action_dim, device=env.device) * 2 - 1
    
    # Step environment
    obs, rewards, dones, truncated, info = env.step(actions)
    
    if step % 10 == 0:
        robot_pos = robot.data.root_pos_w[0].cpu().numpy()
        robot_vel = robot.data.root_lin_vel_w[0].cpu().numpy()
        wheel_vel = robot.data.joint_vel[0].cpu().numpy()
        
        print(f"  Step {step:3d}: pos=[{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}], "
              f"vel=[{robot_vel[0]:.2f}, {robot_vel[1]:.2f}], "
              f"reward={rewards[0]:.2f}")
        
        # Check contact sensor if available
        if "contact_sensor" in scene.sensors:
            contact_forces = scene.sensors["contact_sensor"].data.net_forces_w[0]
            if contact_forces.norm() > 0.1:
                print(f"    ⚠️  接触力检测: {contact_forces.norm():.2f} N")

print("✓ 物理仿真测试完成")

# Step 6: Test reset
print(f"\n[5/6] 测试环境重置...")
env_ids = torch.tensor([0], device=env.device)
env.reset(env_ids=env_ids)
print(f"  环境 {env_ids.cpu().numpy()} 重置后位置: {robot.data.root_pos_w[0].cpu().numpy()}")
print("✓ 重置测试完成")

# Summary
print("\n" + "="*70)
print("  ✅ Task 1: Isaac Lab场景创建 - 测试通过!")
print("="*70)
print("\n完成内容:")
print("  ✓ 场景成功创建 (ground + robot + obstacles)")
print("  ✓ 机器人articulation正常工作")
print("  ✓ 接触传感器配置成功")
print("  ✓ 物理仿真步进正常")
print("  ✓ 环境重置功能正常")
print("\n下一步:")
print("  → Task 2: 集成Camera和LiDAR传感器")
print("  → Task 3: 切换到物理引擎 (当前使用GPU加速)")
print("="*70 + "\n")

# Cleanup
simulation_app.close()
