#!/usr/bin/env python
"""打印机器人的body路径信息"""

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omni.isaac.lab.envs import ManagerBasedRLEnv
from scripts.rosorin_env_cfg import ROSORinEnvCfg

def main():
    # 创建环境
    env_cfg = ROSORinEnvCfg()
    env_cfg.scene.num_envs = 1  # 只创建1个环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("\n" + "="*70)
    print("机器人Body信息")
    print("="*70)
    
    robot = env.scene["robot"]
    
    print(f"\n根路径: {robot.root_physx_view.prim_paths[0]}")
    print(f"\nBody数量: {robot.num_bodies}")
    print(f"\nBody名称:")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i}] {name}")
    
    # 打印完整路径
    print(f"\n完整Body路径 (用于ContactSensor):")
    root_path = robot.root_physx_view.prim_paths[0]
    for name in robot.body_names:
        full_path = f"{root_path}/{name}"
        print(f"  {full_path}")
    
    # 使用USD API检查
    from pxr import Usd
    stage = simulation_app.context.get_stage()
    robot_prim = stage.GetPrimAtPath(root_path)
    
    print(f"\n子Prim列表:")
    for child in robot_prim.GetChildren():
        print(f"  {child.GetPath()} (类型: {child.GetTypeName()})")
    
    print("\n" + "="*70)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
