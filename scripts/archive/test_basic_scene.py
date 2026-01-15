"""
最简单的Isaac Lab场景测试
直接创建地面和简单物体,不依赖Nucleus资源
"""

import sys
import argparse
import torch

print("[DEBUG] Script started", flush=True)

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

print(f"[DEBUG] Args parsed: num_envs={args.num_envs}, headless={args.headless}", flush=True)

# 启动Isaac Sim (直接使用SimulationApp)
print("[DEBUG] Importing SimulationApp...", flush=True)
from isaacsim import SimulationApp
print("[DEBUG] Creating SimulationApp...", flush=True)
simulation_app = SimulationApp({"headless": args.headless})
print("[DEBUG] SimulationApp created", flush=True)

# 导入Isaac Lab模块
print("[INFO] Importing Isaac Lab modules...")
sys.stdout.flush()

try:
    import isaaclab.sim as sim_utils
    print("[DEBUG] isaaclab.sim imported", flush=True)
    
    from isaaclab.assets import RigidObjectCfg
    print("[DEBUG] RigidObjectCfg imported", flush=True)
    
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    print("[DEBUG] InteractiveScene imported", flush=True)
    
    from isaaclab.utils import configclass
    print("[DEBUG] configclass imported", flush=True)
    
except Exception as e:
    print(f"[ERROR] Failed to import Isaac Lab modules: {e}", flush=True)
    import traceback
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)

print("[DEBUG] Defining scene configuration...", flush=True)

@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """最简单的场景 - 只有地面"""
    
    # 地面
    ground = sim_utils.GroundPlaneCfg()
    
    # 简单的立方体
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

print("[DEBUG] Scene configuration defined", flush=True)


def main():
    """主函数"""
    
    print(f"\n{'='*70}")
    print(f"  最简单的Isaac Lab场景测试")
    print(f"{'='*70}\n")
    
    # 导入SimulationContext
    from isaaclab.sim import SimulationContext
    
    # 创建仿真上下文
    print(f"[1/4] 创建仿真上下文...")
    sim_context = SimulationContext(sim_params=sim_utils.SimulationCfg(dt=0.01))
    
    # 设置主相机视角
    sim_context.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    
    # 创建场景配置
    print(f"[2/4] 创建场景配置 (num_envs={args.num_envs})...")
    scene_cfg = SimpleSceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    
    # 创建场景
    print(f"[3/4] 创建场景...")
    scene = InteractiveScene(scene_cfg)
    print(f"✓ 场景创建成功")
    print(f"  - 环境数量: {scene.num_envs}")
    print(f"  - 设备: {scene.device}")
    
    # 运行仿真
    print(f"\n[4/4] 运行仿真 (100步)...")
    count = 0
    max_steps = 100
    
    while simulation_app.is_running() and count < max_steps:
        # 重置场景
        if count == 0:
            scene.reset()
            print(f"  Step {count}: 场景重置")
        
        # 场景更新
        scene.write_data_to_sim()
        simulation_app.update()
        scene.update(dt=0.01)
        
        if count % 20 == 19:
            # 获取立方体位置
            cube_pos = scene["cube"].data.root_pos_w[0]
            print(f"  Step {count}: 立方体位置 = [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
        
        count += 1
    
    print(f"\n✓ 测试完成! 总步数: {count}")
    print(f"✓ 立方体最终位置: {scene['cube'].data.root_pos_w[0].cpu().numpy()}\n")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("[DEBUG] Entering main block", flush=True)
    try:
        main()
        print("[DEBUG] main() completed successfully", flush=True)
    except Exception as e:
        print(f"[ERROR] Exception in main: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] Closing simulation_app", flush=True)
        simulation_app.close()
        print("[DEBUG] simulation_app closed", flush=True)
