"""
极简Isaac Sim测试 - 只测试导入和World创建
参考之前成功的test_isaaclab_basic.py
"""

import argparse

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

# 启动Isaac Sim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# 导入omni.isaac.core (之前测试通过的)
from omni.isaac.core import World

print(f"\n{'='*70}")
print(f"  Isaac Sim基础功能验证")
print(f"{'='*70}\n")

# 创建World
print("[1/2] 创建World...")
world = World()
print("✓ World创建成功")

# 添加地面
print("\n[2/2] 添加地面...")
world.scene.add_default_ground_plane()
print("✓ 地面添加成功")

# 重置
print("\n[3/3] 重置World并运行5步...")
world.reset()
print("  - World已重置")

for i in range(5):
    world.step(render=False)  # 强制不渲染
    print(f"  - Step {i+1}/5")

print(f"  - 仿真运行完成")

print(f"\n✓ 测试成功!\n")
print(f"{'='*70}\n")

# 清理
world.stop()
simulation_app.close()
