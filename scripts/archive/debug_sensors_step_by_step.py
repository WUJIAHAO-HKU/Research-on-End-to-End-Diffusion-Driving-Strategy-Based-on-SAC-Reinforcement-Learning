#!/usr/bin/env python3
"""
传感器调试 - 分步测试
逐个启用传感器，找出正确的配置方法
"""

import sys
sys.path.insert(0, "/home/wujiahao/IsaacLab/_build/linux-x86_64/release")

# 第一步：尝试不同的SimulationApp配置
print("\n" + "="*70)
print("  第1步：测试SimulationApp相机配置")
print("="*70)

from isaacsim import SimulationApp

# 尝试多种配置
configs_to_test = [
    {"headless": True},
    {"headless": True, "enable_cameras": True},
    {"headless": True, "width": 640, "height": 480},
    {"headless": False, "width": 640, "height": 480},  # 非headless模式
]

print("\n测试配置1: 纯headless（无相机参数）")
simulation_app = SimulationApp({"headless": True})

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCfg, ContactSensorCfg, patterns
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp

print("✓ Isaac Lab模块导入成功")

##
# 第2步：测试LiDAR（不需要enable_cameras）
##

print("\n" + "="*70)
print("  第2步：测试LiDAR传感器（无mesh碰撞）")
print("="*70)

@configclass
class LidarTestSceneCfg(InteractiveSceneCfg):
    """只有LiDAR的测试场景"""
    
    # 使用GroundPlaneCfg - 会创建真正的Mesh类型
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # 障碍物 - 为LiDAR提供额外的碰撞目标
    obstacle = AssetBaseCfg(
        prim_path="/World/obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0, 0.0, 0.25)),
    )

    # ROSOrin robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            joint_pos={".*wheel.*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    
    # LiDAR - GroundPlaneCfg应该会在/World/ground路径创建Mesh
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_frame",
        update_period=0.1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=5.0,
        ),
        max_distance=5.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],  # GroundPlane的根路径
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    pass

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.1)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class EventCfg:
    pass

@configclass
class LidarEnvCfg(ManagerBasedRLEnvCfg):
    scene: LidarTestSceneCfg = LidarTestSceneCfg(num_envs=1, env_spacing=5.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    
    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 0.01
        self.episode_length_s = 10.0

try:
    print("\n创建LiDAR环境...")
    lidar_env = ManagerBasedRLEnv(cfg=LidarEnvCfg())
    print("✓ LiDAR环境创建成功！")
    
    print("\n重置环境...")
    lidar_env.reset()
    print("✓ LiDAR环境重置成功！")
    
    print("\n运行5步...")
    for i in range(5):
        actions = torch.zeros(1, 0, device=lidar_env.device)  # 空动作
        obs, rewards, terminated, truncated, info = lidar_env.step(actions)
        print(f"  步骤 {i}: 奖励={rewards.mean():.3f}")
    
    print("✓ LiDAR测试通过！")
    lidar_env.close()
    
except Exception as e:
    print(f"✗ LiDAR测试失败: {e}")
    import traceback
    traceback.print_exc()

##
# 第3步：测试相机（在新的SimulationApp实例中）
##

print("\n" + "="*70)
print("  第3步：相机测试需要重新启动SimulationApp")
print("="*70)

simulation_app.close()

print("\n重启SimulationApp（尝试启用渲染）...")
# 尝试非headless模式
simulation_app2 = SimulationApp({
    "headless": False,  # 非headless模式
    "width": 640,
    "height": 480,
})

print("✓ SimulationApp重启成功（非headless模式）")

# 重新导入（确保使用新的app）
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

@configclass
class CameraTestSceneCfg(InteractiveSceneCfg):
    """相机测试场景"""
    
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            joint_pos={".*wheel.*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    
    # 相机 - 独立的相机prim
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 1.0),  # 相机在上方
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

@configclass
class CameraEnvCfg(ManagerBasedRLEnvCfg):
    scene: CameraTestSceneCfg = CameraTestSceneCfg(num_envs=1, env_spacing=5.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    
    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 0.01
        self.episode_length_s = 10.0

try:
    print("\n创建相机环境...")
    camera_env = ManagerBasedRLEnv(cfg=CameraEnvCfg())
    print("✓ 相机环境创建成功！")
    
    print("\n重置环境...")
    camera_env.reset()
    print("✓ 相机环境重置成功！")
    
    print("\n运行3步...")
    for i in range(3):
        actions = torch.zeros(1, 0, device=camera_env.device)
        obs, rewards, terminated, truncated, info = camera_env.step(actions)
        print(f"  步骤 {i}: 奖励={rewards.mean():.3f}")
    
    print("✓ 相机测试通过！")
    camera_env.close()
    
except Exception as e:
    print(f"✗ 相机测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("  调试完成")
print("="*70)

simulation_app2.close()
