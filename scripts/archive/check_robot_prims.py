#!/usr/bin/env python3
"""在Isaac Lab环境中检查机器人的prim路径"""

import argparse
from omni.isaac.lab.app import AppLauncher

# 添加参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="无头模式")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动模拟器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

def main():
    """主函数：加载机器人并打印所有prim路径"""
    
    # 场景配置
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    
    # 添加地面
    scene_cfg.terrain = sim_utils.GroundPlaneCfg()
    
    # 添加机器人
    rosorin_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.10),
            joint_pos={".*": 0.0},
        ),
    )
    scene_cfg.articulation = rosorin_cfg
    
    # 创建场景
    scene = InteractiveScene(scene_cfg)
    
    print("\n" + "="*70)
    print("机器人Prim路径检查")
    print("="*70)
    
    # 获取机器人articulation
    robot = scene["articulation"]
    print(f"\n机器人根路径: {robot.root_physx_view.prim_paths[0]}")
    print(f"Bodies数量: {robot.num_bodies}")
    print(f"Joints数量: {robot.num_joints}")
    
    # 打印body names
    print(f"\nBody名称列表:")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i}] {name}")
    
    # 打印joint names
    print(f"\nJoint名称列表:")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i}] {name}")
    
    # 使用USD API遍历所有子prim
    from pxr import Usd
    stage = simulation_app.context.get_stage()
    
    robot_prim_path = robot.root_physx_view.prim_paths[0]
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    
    print(f"\n完整USD层级结构 (从 {robot_prim_path}):")
    print("-" * 70)
    
    def print_hierarchy(prim, indent=0):
        """递归打印层级"""
        prefix = "  " * indent
        prim_type = prim.GetTypeName()
        print(f"{prefix}├─ {prim.GetName()} [{prim_type}]")
        print(f"{prefix}   路径: {prim.GetPath()}")
        
        # 只打印前3层
        if indent < 3:
            for child in prim.GetChildren():
                print_hierarchy(child, indent + 1)
    
    print_hierarchy(robot_prim)
    
    print("\n" + "="*70)
    print("可用于ContactSensor的候选路径:")
    print("="*70)
    
    # 搜索可能的body路径
    for prim in stage.Traverse():
        path_str = prim.GetPath().pathString
        if robot_prim_path in path_str and prim.GetTypeName() in ["Xform", ""]:
            if any(keyword in path_str.lower() for keyword in ["base", "link", "body"]):
                print(f"  {path_str}")
    
    print("\n测试完成!")
    
    # 关闭
    simulation_app.close()

if __name__ == "__main__":
    main()
