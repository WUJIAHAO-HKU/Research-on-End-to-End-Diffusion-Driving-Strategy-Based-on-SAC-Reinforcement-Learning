"""
最基础的Isaac Sim + Isaac Lab测试
使用isaacsim.core的World类
"""

import argparse

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# 启动Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# 导入模块
from isaacsim.core.api.scenes import Scene
from isaacsim.core.api.prims import GeometryPrim, XFormPrim
from isaacsim.core.utils.prims import create_prim
import numpy as np

print(f"\n{'='*70}")
print(f"  Isaac Sim基础测试 - 场景创建")
print(f"{'='*70}\n")

# 创建场景
print("[1/4] 创建场景...")
scene = Scene()
print("✓ Scene创建成功")

# 添加地面
print("\n[2/4] 添加地面...")
scene.add_default_ground_plane()
print("✓ 地面添加成功")

# 添加立方体
print("\n[3/4] 添加立方体...")
from pxr import UsdGeom, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# 创建立方体
cube_prim = stage.DefinePrim("/World/Cube", "Cube")
cube_prim.GetAttribute("size").Set(0.2)

# 设置位置
xform_api = UsdGeom.XformCommonAPI(cube_prim)
xform_api.SetTranslate(Gf.Vec3d(0, 0, 2.0))

print("✓ 立方体添加成功")

# 运行仿真
print("\n[4/4] 运行仿真 (50步)...")

import omni.timeline
timeline = omni.timeline.get_timeline_interface()
timeline.play()

for i in range(50):
    # 更新仿真
    simulation_app.update()
    
    if i % 10 == 9:
        # 获取立方体位置
        translate = xform_api.GetTranslateOp().Get()
        print(f"  Step {i}: 立方体位置 = [{translate[0]:.3f}, {translate[1]:.3f}, {translate[2]:.3f}]")

timeline.stop()

print(f"\n✓ 测试完成!\n")
print(f"{'='*70}\n")

# 清理
simulation_app.close()
