"""
最小化的传感器测试 - 检查USD场景结构
基于rosorin_env_cfg.py，移除传感器以检查mesh路径
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from isaaclab.envs import ManagerBasedRLEnv
import omni.usd
from pxr import Usd

# 导入ROSOrin环境配置（已经是工作的配置）
import rosorin_env_cfg

print("\n" + "="*70)
print("  创建ROSOrin场景并检查USD结构")
print("="*70)

try:
    # 使用现有的ROSOrin配置（没有传感器）
    print("\n创建场景...")
    env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
    env_cfg.scene.num_envs = 1  # 只需要1个环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取USD stage并打印结构
    print("\n" + "="*70)
    print("  检查USD场景结构:")
    print("="*70)
    stage = omni.usd.get_context().get_stage()
    
    def print_prims(prim, depth=0, max_depth=5):
        """递归打印prim层次结构"""
        if depth > max_depth:
            return
        indent = "  " * depth
        prim_type = prim.GetTypeName()
        is_mesh = " [MESH]" if prim_type == "Mesh" else ""
        print(f"{indent}{prim.GetPath()}{is_mesh} ({prim_type})")
        
        if depth < max_depth:
            for child in prim.GetChildren():
                print_prims(child, depth + 1, max_depth)
    
    # 打印/World下的结构
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim:
        print("\n完整场景结构:")
        print_prims(world_prim, max_depth=3)
    
    # 特别检查ground的详细结构
    print("\n" + "="*70)
    print("  /World/ground 详细结构:")
    print("="*70)
    ground_prim = stage.GetPrimAtPath("/World/ground")
    if ground_prim:
        print_prims(ground_prim, max_depth=5)
    else:
        print("✗ /World/ground 不存在")
    
    # 检查obstacle
    print("\n" + "="*70)
    print("  /World/obstacle 详细结构:")
    print("="*70)
    obstacle_prim = stage.GetPrimAtPath("/World/obstacle")
    if obstacle_prim:
        print_prims(obstacle_prim, max_depth=5)
    else:
        print("✗ /World/obstacle 不存在")
    
    # 查找所有Mesh类型的prims
    print("\n" + "="*70)
    print("  所有Mesh类型的prims:")
    print("="*70)
    
    def find_meshes(prim):
        """递归查找所有mesh"""
        meshes = []
        if prim.GetTypeName() == "Mesh":
            meshes.append(prim.GetPath())
        for child in prim.GetChildren():
            meshes.extend(find_meshes(child))
        return meshes
    
    all_meshes = find_meshes(world_prim)
    for mesh_path in all_meshes:
        print(f"  - {mesh_path}")
    
    print(f"\n找到 {len(all_meshes)} 个mesh prims")
    
    print("\n✓ 场景检查完成")
    print("\n建议的mesh_prim_paths配置:")
    if all_meshes:
        print("  mesh_prim_paths = [")
        for mesh_path in all_meshes[:5]:  # 只显示前5个
            print(f"      \"{mesh_path}\",")
        if len(all_meshes) > 5:
            print(f"      # ... 还有 {len(all_meshes) - 5} 个mesh")
        print("  ]")
    
    env.close()
    
except Exception as e:
    print(f"\n✗ 场景创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("  测试完成")
print("="*70)

simulation_app.close()
