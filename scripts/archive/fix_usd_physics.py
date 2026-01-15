#!/usr/bin/env python3
"""
Fix USD Physics Material - 为ROSOrin USD添加物理材质和摩擦力

这个脚本会：
1. 打开rosorin.usd文件
2. 为每个轮子添加高摩擦力的物理材质
3. 保存修改后的USD文件
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, default="data/assets/rosorin/rosorin.usd")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, PhysxSchema, Sdf, UsdGeom
import os

# USD文件路径
usd_path = args.usd_path
if not os.path.isabs(usd_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    usd_path = os.path.join(project_root, usd_path)

print(f"\n正在打开USD文件: {usd_path}")
stage = Usd.Stage.Open(usd_path)

# 创建物理材质
print("\n创建高摩擦力物理材质...")
material_path = "/World/PhysicsMaterials/HighFriction"
if not stage.GetPrimAtPath(material_path):
    material_prim = stage.DefinePrim(material_path, "Material")
    
    # 应用UsdPhysics MaterialAPI
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    
    # PhysX特定材质属性 - 使用正确的属性名
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    # 查看可用属性
    print(f"  PhysxMaterialAPI 可用方法: {[m for m in dir(physx_material_api) if not m.startswith('_')][:10]}")
    
    # 尝试设置摩擦力（使用prim的属性）
    material_prim.CreateAttribute("physxMaterial:staticFriction", Sdf.ValueTypeNames.Float).Set(1.5)
    material_prim.CreateAttribute("physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(1.2)
    material_prim.CreateAttribute("physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(0.0)
    
    print(f"  创建材质: {material_path}")
    print(f"    静摩擦: 1.5, 动摩擦: 1.2, 弹性: 0.0")

# 查找所有轮子碰撞体
print("\n查找轮子碰撞体并应用物理材质...")
wheel_count = 0
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    prim_name = prim.GetName().lower()
    
    # 查找colliders下的wheel碰撞体
    if "wheel" in prim_path.lower() and "collision" in prim_path.lower():
        # 查找有CollisionAPI的prim
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"  轮子碰撞: {prim.GetPath()}")
            
            # 使用UsdShade.MaterialBindingAPI来绑定物理材质
            from pxr import UsdShade
            binding_api = UsdShade.MaterialBindingAPI(prim)
            if not binding_api:
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            
            # 绑定物理材质 - 使用physics purpose
            binding_api.Bind(
                UsdShade.Material(stage.GetPrimAtPath(material_path)),
                materialPurpose="physics"
            )
            print(f"    绑定材质: {material_path}")
            wheel_count += 1

print(f"\n总共处理了 {wheel_count} 个轮子碰撞体")

# 保存修改
backup_path = usd_path.replace(".usd", "_backup.usd")
print(f"\n保存备份到: {backup_path}")
stage.Export(backup_path)

print(f"保存修改到: {usd_path}")
stage.Save()

print("\n✓ USD物理材质修复完成！")
simulation_app.close()
