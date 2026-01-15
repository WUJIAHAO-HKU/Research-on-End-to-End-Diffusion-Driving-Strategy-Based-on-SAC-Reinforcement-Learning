#!/usr/bin/env python3
"""移除USD中的root_joint固定关节，让机器人可以自由移动"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd
import os

usd_path = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd"

print(f"\n正在打开: {usd_path}")
stage = Usd.Stage.Open(usd_path)

# 查找并移除root_joint
root_joint_path = "/car/root_joint"
if stage.GetPrimAtPath(root_joint_path):
    print(f"找到 {root_joint_path}，正在移除...")
    stage.RemovePrim(root_joint_path)
    print("  ✓ 已移除")
else:
    print(f"未找到 {root_joint_path}")

# 保存
print(f"\n保存修改...")
stage.Save()
print("✓ 完成！")

simulation_app.close()
