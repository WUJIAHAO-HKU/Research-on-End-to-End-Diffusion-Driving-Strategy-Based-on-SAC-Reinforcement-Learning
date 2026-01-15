#!/usr/bin/env python3
"""修改USD：移除root_joint但保留ArticulationRootAPI"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, PhysxSchema
import os

usd_path = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd"

print(f"\n正在打开: {usd_path}")
stage = Usd.Stage.Open(usd_path)

# 1. 检查articulation根
car_prim = stage.GetPrimAtPath("/car")
if car_prim and car_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
    print("✓ /car 已有 ArticulationRootAPI")
else:
    print("× /car 缺少 ArticulationRootAPI，添加中...")
    UsdPhysics.ArticulationRootAPI.Apply(car_prim)

# 2. 移除root_joint（固定关节）
root_joint_path = "/car/root_joint"
root_joint = stage.GetPrimAtPath(root_joint_path)
if root_joint:
    print(f"\n正在移除固定关节: {root_joint_path}")
    stage.RemovePrim(root_joint_path)
    print("  ✓ 已移除")

# 3. 确保base_footprint没有被固定
base_footprint_joint_path = "/car/joints/base_joint"
base_joint = stage.GetPrimAtPath(base_footprint_joint_path)
if base_joint:
    print(f"\n检查 {base_footprint_joint_path}")
    if base_joint.GetTypeName() == "PhysicsFixedJoint":
        print("  这也是固定关节，移除中...")
        stage.RemovePrim(base_footprint_joint_path)
        print("  ✓ 已移除")

# 保存
print(f"\n保存修改...")
stage.Save()
print("✓ 完成！机器人现在应该可以自由移动了")

simulation_app.close()
