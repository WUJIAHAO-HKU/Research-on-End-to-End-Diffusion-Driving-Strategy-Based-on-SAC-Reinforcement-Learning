#!/usr/bin/env python3
"""Quick script to check ROSOrin USD joints"""

import sys
sys.path.insert(0, "/home/wujiahao/IsaacLab/_build/linux-x86_64/release")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics

# Load USD file
stage = Usd.Stage.Open("/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd")

# Find all joints
joints = []
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Joint):
        joints.append(prim.GetPath())
        print(f"Joint: {prim.GetPath()}")
        # Check joint type
        if hasattr(prim, 'GetTypeName'):
            print(f"  Type: {prim.GetTypeName()}")

print(f"\nTotal joints found: {len(joints)}")

simulation_app.close()
