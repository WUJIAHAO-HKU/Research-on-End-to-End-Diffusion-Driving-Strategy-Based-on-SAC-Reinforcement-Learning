#!/usr/bin/env python3
"""简单相机测试 - 快速验证"""

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

print("\n创建环境...")
env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)
print("✓ 环境创建成功")

print("\n传感器状态:")
print(f"  Contact: {'✓' if 'contact_sensor' in env.scene.sensors else '✗'}")
print(f"  Camera: {'✓' if 'camera' in env.scene.sensors else '✗'}")
print(f"  LiDAR: {'✓' if 'lidar' in env.scene.sensors else '✗'}")

if 'camera' in env.scene.sensors:
    print(f"\n相机分辨率: {env.scene.sensors['camera'].image_shape}")

print("\n采集1帧数据...")
obs, _ = env.reset()
obs, _, _, _, _ = env.step(torch.zeros(1, 4, device=env.device))
print("✓ 完成")

env.close()
simulation_app.close()
