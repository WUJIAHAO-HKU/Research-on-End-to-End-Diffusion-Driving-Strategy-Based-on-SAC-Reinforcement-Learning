#!/usr/bin/env python3
"""
Test Camera and LiDAR Sensors - Task 2

验证相机和激光雷达传感器集成：
1. 相机 (RGB + Depth)
2. 激光雷达 (360° 2D scan)

使用方法:
    ./isaaclab_runner.sh scripts/test_camera_lidar.py --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments FIRST - AppLauncher will add --enable_cameras flag
parser = argparse.ArgumentParser(description="Test Camera and LiDAR Sensors")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app with camera support ENABLED
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("\n" + "="*70)
print("  Camera & LiDAR Sensor Test (Task 2)")
print("="*70)

# Import Isaac Lab modules AFTER app launch
print("\n[1/4] 导入Isaac Lab模块...")
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

# Import environment configuration
import rosorin_env_cfg
print("✓ 模块导入成功")

# Create environment
print(f"\n[2/4] 创建环境 (num_envs={args.num_envs})...")
try:
    # Modify config to enable camera and lidar
    env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("✓ 环境创建成功")
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    simulation_app.close()
    exit(1)

# Check sensors
print(f"\n[3/4] 检查传感器状态...")
scene = env.scene

# Contact sensor
if 'contact_sensor' in scene.sensors:
    contact = scene.sensors['contact_sensor']
    print(f"  ✓ Contact Sensor: {contact.num_bodies} bodies")
else:
    print(f"  ✗ Contact Sensor: 未配置")

# Camera
if 'camera' in scene.sensors:
    camera = scene.sensors['camera']
    print(f"  ✓ Camera: {camera.image_shape}")
    print(f"    数据类型: {list(camera.data.output.keys())}")
else:
    print(f"  ✗ Camera: 未配置")

# LiDAR
if 'lidar' in scene.sensors:
    lidar = scene.sensors['lidar']
    print(f"  ✓ LiDAR: {lidar.num_rays} rays, max_distance={lidar.cfg.max_distance}m")
else:
    print(f"  ✗ LiDAR: 未配置")

# Test sensor data collection
print(f"\n[4/4] 测试传感器数据采集 (10步)...")
obs, _ = env.reset()

for step in range(10):
    # Random actions
    actions = torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 0.5
    obs, rewards, dones, truncated, info = env.step(actions)
    
    if step % 5 == 0:
        print(f"\n  Step {step}:")
        
        # Contact sensor data
        if 'contact_sensor' in scene.sensors:
            contact_forces = scene.sensors['contact_sensor'].data.net_forces_w[0]
            print(f"    Contact Force: {torch.norm(contact_forces).item():.2f} N")
        
        # Camera data
        if 'camera' in scene.sensors:
            cam_data = scene.sensors['camera'].data
            if 'rgb' in cam_data.output:
                rgb_shape = cam_data.output['rgb'].shape
                print(f"    Camera RGB: {rgb_shape}, range=[{cam_data.output['rgb'].min():.2f}, {cam_data.output['rgb'].max():.2f}]")
            if 'distance_to_image_plane' in cam_data.output:
                depth_shape = cam_data.output['distance_to_image_plane'].shape
                depth = cam_data.output['distance_to_image_plane'][0]
                print(f"    Camera Depth: {depth_shape}, mean={depth.mean():.2f}m")
        
        # LiDAR data
        if 'lidar' in scene.sensors:
            lidar_data = scene.sensors['lidar'].data.ray_hits_w[0]
            distances = torch.norm(lidar_data, dim=-1)
            print(f"    LiDAR: min={distances.min():.2f}m, mean={distances.mean():.2f}m, max={distances.max():.2f}m")

print("\n✓ 传感器数据采集测试完成")

print("\n" + "="*70)
print("  ✅ Task 2: 传感器集成测试 - 完成!")
print("="*70)

# Cleanup
env.close()
simulation_app.close()
