#!/usr/bin/env python3
"""检查USD文件结构"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, default="data/assets/rosorin/rosorin.usd")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics
import os

usd_path = args.usd_path
if not os.path.isabs(usd_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    usd_path = os.path.join(project_root, usd_path)

print(f"\n正在检查USD文件: {usd_path}")
stage = Usd.Stage.Open(usd_path)

print("\n=== USD 层次结构 ===")
def print_prim_tree(prim, depth=0):
    indent = "  " * depth
    prim_type = prim.GetTypeName()
    has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
    is_mesh = prim.IsA(UsdGeom.Mesh)
    
    info = f"{indent}- {prim.GetName()} ({prim_type})"
    if is_mesh:
        info += " [Mesh]"
    if has_collision:
        info += " [Collision]"
    print(info)
    
    if depth < 4:  # 限制深度
        for child in prim.GetChildren():
            print_prim_tree(child, depth + 1)

root = stage.GetPseudoRoot()
for prim in root.GetChildren():
    print_prim_tree(prim)

print("\n=== 搜索 'wheel' 关键字 ===")
for prim in stage.Traverse():
    if "wheel" in prim.GetName().lower():
        print(f"  路径: {prim.GetPath()}")
        print(f"  类型: {prim.GetTypeName()}")
        print(f"  是Mesh: {prim.IsA(UsdGeom.Mesh)}")
        print(f"  有Collision: {prim.HasAPI(UsdPhysics.CollisionAPI)}")
        print()

simulation_app.close()
