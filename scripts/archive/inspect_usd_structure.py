#!/usr/bin/env python3
"""检查rosorin.usd的层级结构"""

from pxr import Usd, UsdGeom
import os

# USD文件路径
usd_path = "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd"

# 打开USD stage
stage = Usd.Stage.Open(usd_path)

def print_hierarchy(prim, indent=0):
    """递归打印USD层级结构"""
    prefix = "  " * indent
    
    # 获取prim类型
    prim_type = prim.GetTypeName()
    
    # 检查是否是Xform或其他重要类型
    marker = ""
    if UsdGeom.Xform(prim):
        marker = " [Xform]"
    elif prim.IsA(UsdGeom.Mesh):
        marker = " [Mesh]"
    
    print(f"{prefix}{prim.GetName()}{marker} ({prim.GetPath()})")
    
    # 递归打印子节点 (只打印前3层)
    if indent < 3:
        for child in prim.GetChildren():
            print_hierarchy(child, indent + 1)

print("=" * 60)
print("USD 结构:")
print("=" * 60)

# 从根节点开始
root = stage.GetPseudoRoot()
for prim in root.GetChildren():
    print_hierarchy(prim)

print("\n" + "=" * 60)
print("查找 base_link:")
print("=" * 60)

# 搜索包含"base_link"的所有prim
for prim in stage.Traverse():
    if "base_link" in prim.GetPath().pathString.lower():
        print(f"  {prim.GetPath()} (类型: {prim.GetTypeName()})")

print("\n" + "=" * 60)
print("查找所有bodies (Xform类型):")
print("=" * 60)

# 查找所有Xform (通常是刚体)
for prim in stage.Traverse():
    if UsdGeom.Xform(prim) and "/car/" in prim.GetPath().pathString:
        print(f"  {prim.GetPath()}")
