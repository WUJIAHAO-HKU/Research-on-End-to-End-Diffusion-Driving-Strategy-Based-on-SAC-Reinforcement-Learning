"""
简化的ROSOrin环境测试 - 验证Isaac Lab基础功能
"""

import argparse
import torch

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# 启动Isaac Sim (直接使用SimulationApp)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# 导入Isaac Lab模块
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg  # 不用RL版本
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationManager, ActionManager
import isaaclab.envs.mdp as mdp

##
# 场景配置
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """简化的场景配置"""
    
    # 地面
    ground = sim_utils.GroundPlaneCfg()
    
    # 机器人 (使用Carter作为占位符)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Carter/carter_v1.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
        ),
    )

##
# 环境配置
##

@configclass
class ObservationsCfg:
    """观测配置（空）"""
    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """动作配置（空）"""
    pass

@configclass
class EventCfg:
    """事件配置（空）"""
    pass

@configclass
class MyEnvCfg(ManagerBasedEnvCfg):
    """简化的环境配置"""
    
    # 场景
    scene: MySceneCfg = MySceneCfg(num_envs=4, env_spacing=2.0)
    
    # 观察配置 - 必须有，即使是空的
    observations = ObservationsCfg()
    
    # 动作配置 - 必须有，即使是空的
    actions = ActionsCfg()
    
    # 事件 (重置时的随机化)
    events = EventCfg()
    
    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.01


def main():
    """主函数"""
    
    print(f"\n{'='*70}")
    print(f"  简化Isaac Lab环境测试")
    print(f"{'='*70}\n")
    
    # 创建环境
    print(f"[1/2] 创建环境 (num_envs={args.num_envs})...")
    env_cfg = MyEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
    print(f"✓ 环境创建成功\n")
    
    # 运行仿真
    print(f"[2/2] 运行仿真...")
    count = 0
    max_steps = 500
    
    while simulation_app.is_running() and count < max_steps:
        with torch.inference_mode():
            # 每100步重置一次
            if count % 100 == 0:
                env.reset()
                print(f"  Step {count}: 环境重置")
            
            # 空动作 (因为没有定义动作)
            obs, _ = env.step(torch.empty(args.num_envs, 0, device=env.device))
            
            if count % 100 == 99:
                print(f"  Step {count}: OK")
            
            count += 1
    
    print(f"\n✓ 测试完成! 总步数: {count}\n")
    
    # 性能测试
    if count >= max_steps:
        print(f"{'='*70}")
        print(f"性能基准测试")
        print(f"{'='*70}")
        
        import time
        test_steps = 1000
        start = time.time()
        
        test_count = 0
        while test_count < test_steps and simulation_app.is_running():
            with torch.inference_mode():
                obs, _ = env.step(torch.empty(args.num_envs, 0, device=env.device))
                test_count += 1
        
        elapsed = time.time() - start
        fps = test_steps / elapsed
        
        print(f"仿真FPS: {fps:.1f}")
        print(f"平均步时间: {elapsed/test_steps*1000:.2f}ms")
        print(f"{'='*70}\n")
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
