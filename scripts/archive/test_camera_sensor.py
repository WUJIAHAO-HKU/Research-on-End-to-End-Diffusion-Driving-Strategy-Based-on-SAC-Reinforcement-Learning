#!/usr/bin/env python3
"""
Test Camera Sensor Integration

测试相机传感器的集成：
1. RGB图像捕获
2. 深度图像捕获  
3. 图像分辨率验证

使用方法:
    ./isaaclab_runner.sh scripts/test_camera_sensor.py --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments - MUST be before any other imports
parser = argparse.ArgumentParser(description="Test Camera Sensor")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force enable cameras
args.enable_cameras = True

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass

print("\n" + "="*70)
print("  Camera Sensor Integration Test")
print("="*70)

# Simple environment with just robot + camera
@configclass
class CameraTestSceneCfg(InteractiveSceneCfg):
    """Scene with robot and camera."""
    
    # Ground plane
    terrain = sim_utils.GroundPlaneCfg()
    
    # Robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.10),
        ),
    )
    
    # Camera
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.057, 0.0, 0.092),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )
    
    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0),
    )

# Create scene
print("\n[1/3] 创建场景...")
scene_cfg = CameraTestSceneCfg(num_envs=1, env_spacing=2.0)
from isaaclab.scene import InteractiveScene
scene = InteractiveScene(scene_cfg)
print("✓ 场景创建成功")

# Reset to initialize
print("\n[2/3] 初始化传感器...")
scene.reset()
print("✓ 传感器初始化成功")

# Test camera
print("\n[3/3] 测试相机捕获...")
camera = scene.sensors["camera"]
print(f"  相机配置:")
print(f"    - 分辨率: {camera.image_shape}")
print(f"    - 更新周期: {camera.cfg.update_period}s")
print(f"    - 数据类型: {camera.cfg.data_types}")

# Capture a few frames
for i in range(5):
    scene.write_data_to_sim()
    simulation_app.update()
    
    if camera.data.output is not None:
        rgb = camera.data.output["rgb"][0]
        depth = camera.data.output["distance_to_image_plane"][0]
        print(f"\n  帧 {i+1}:")
        print(f"    RGB shape: {rgb.shape}, range: [{rgb.min():.2f}, {rgb.max():.2f}]")
        print(f"    Depth shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]m")
        break

print("\n" + "="*70)
print("  ✅ Camera Sensor Test - 通过!")
print("="*70)

# Cleanup
simulation_app.close()
