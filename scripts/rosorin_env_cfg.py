"""
ROSOrin Environment Configuration for Isaac Lab

This module defines the scene, robot, sensors, and task configuration
for the ROSOrin mecanum wheel robot driving environment.

NOTE: This file should only be imported AFTER AppLauncher has been instantiated.
"""

import math
import torch
from typing import Literal

# Isaac Lab imports (must be after AppLauncher)
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg

# MDP functions
import isaaclab.envs.mdp as mdp

# Custom MDP functions for ROSOrin navigation
import rosorin_mdp

##
# Scene Configuration
##

@configclass
class ROSOrinSceneCfg(InteractiveSceneCfg):
    """
    六房间互通室内导航场景 - 适合复杂导航训练
    
    场景组成:
    - 10m × 10m 完全封闭的外墙空间
    - 6个房间（3列×2行布局），每个房间之间都有门洞连通
    - 每个房间都有独特的家具布置
    
    房间布局（俯视图）:
    +-------+-------+-------+
    |  R1   |  R2   |  R3   |  (上排: y=0→5)
    | 客厅  | 书房  | 卧室  |
    +-------+-------+-------+
    |  R4   |  R5   |  R6   |  (下排: y=-5→0)
    | 餐厅  | 厨房  | 储藏  |
    +-------+-------+-------+
    x轴划分: -5.0 → -1.67 → 1.67 → 5.0 (每列约3.34m宽)
    """
    
    # NOTE: Isaac Lab会创建env_0作为源场景进行复制
    # replicate_physics=True时，env_0会被自动隐藏(disable physics)
    # 但仍会在场景树中可见。这是Isaac Lab的正常行为。

    # ========== 地面 ==========
    # NOTE: 地面是全局共享的，不使用env_.* 通配符
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(50.0, 50.0),  # 足够大以覆盖所有环境
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
        ),
    )
    
    # ========== 外墙（完全封闭）==========
    wall_north = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wall_north",
        spawn=sim_utils.CuboidCfg(
            size=(10.2, 0.2, 2.5),  # 10.2m长以覆盖整个宽度
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.85),
                roughness=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 5.1, 1.25)),  # 外移到y=5.1
    )
    
    wall_south = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wall_south",
        spawn=sim_utils.CuboidCfg(
            size=(10.2, 0.2, 2.5),  # 10.2m长以覆盖整个宽度
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -5.1, 1.25)),  # 外移到y=-5.1
    )
    
    wall_east = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wall_east",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 10.2, 2.5),  # 10.2m长以覆盖整个长度
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.1, 0.0, 1.25)),  # 外移到x=5.1
    )
    
    wall_west = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wall_west",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 10.2, 2.5),  # 10.2m长以覆盖整个长度
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.1, 0.0, 1.25)),  # 外移到x=-5.1
    )
    
    # ========== 水平隔断墙（y=0.0，分隔上下两排）==========
    # 第1段: x=-5.0到-2.67 (列1左半部分)
    divider_h_seg1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_h_seg1",
        spawn=sim_utils.CuboidCfg(
            size=(2.26, 0.15, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.8, 0.0, 1.25)),
    )
    # 门洞1: x=-2.67到-1.67 (1.0m宽，R1↔R4)
    
    # 第2段: x=-0.67到0.67 (列2中间部分)
    divider_h_seg2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_h_seg2",
        spawn=sim_utils.CuboidCfg(
            size=(1.26, 0.15, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.25)),
    )
    # 门洞2: x=0.67到1.67 (1.0m宽，R2↔R5)
    
    # 第3段: x=2.67到5.0 (列3右半部分)
    divider_h_seg3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_h_seg3",
        spawn=sim_utils.CuboidCfg(
            size=(2.26, 0.15, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.8, 0.0, 1.25)),
    )
    # 门洞3: x=1.67到2.67 (1.0m宽，R3↔R6)
    
    # ========== 垂直隔断墙（分隔左右列）==========
    # 第一道垂直墙 x=-1.67 (分隔R1-R2 和 R4-R5)
    
    # 上排 R1-R2 段1: y=0→1.5
    divider_v1_top_seg1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v1_top_seg1",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 1.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.67, 0.75, 1.25)),
    )
    
    # 门洞: y=1.5→2.5 (1.0m宽)
    
    # 上排 R1-R2 段2: y=2.5→5.0
    divider_v1_top_seg2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v1_top_seg2",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 2.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.67, 3.75, 1.25)),
    )
    
    # 下排 R4-R5 段1: y=-5.0→-2.5
    divider_v1_bottom_seg1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v1_bottom_seg1",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 2.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.67, -3.75, 1.25)),
    )
    
    # 门洞: y=-2.5→-1.5 (1.0m宽)
    
    # 下排 R4-R5 段2: y=-1.5→0
    divider_v1_bottom_seg2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v1_bottom_seg2",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 1.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.67, -0.75, 1.25)),
    )
    
    # 第二道垂直墙 x=1.67 (分隔R2-R3 和 R5-R6)
    
    # 上排 R2-R3 段1: y=0→1.5
    divider_v2_top_seg1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v2_top_seg1",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 1.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.67, 0.75, 1.25)),
    )
    
    # 门洞: y=1.5→2.5 (1.0m宽)
    
    # 上排 R2-R3 段2: y=2.5→5.0
    divider_v2_top_seg2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v2_top_seg2",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 2.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.67, 3.75, 1.25)),
    )
    
    # 下排 R5-R6 段1: y=-5.0→-2.5
    divider_v2_bottom_seg1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v2_bottom_seg1",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 2.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.67, -3.75, 1.25)),
    )
    
    # 门洞: y=-2.5→-1.5 (1.0m宽)
    
    # 下排 R5-R6 段2: y=-1.5→0
    divider_v2_bottom_seg2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/divider_v2_bottom_seg2",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 1.5, 2.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.88, 0.88, 0.82),
                roughness=0.85,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.67, -0.75, 1.25)),
    )
    
    # ========== 家具布置（每个房间独特）==========
    
    # R1-客厅（x: -5.0→-1.67, y: 0→5）
    sofa_r1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sofa_r1",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 0.9, 0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.3, 0.5),
                roughness=0.9,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.5, 3.5, 0.35)),
    )
    
    tv_stand_r1 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/tv_stand_r1",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 0.4, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.2, 0.1),
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.0, 1.2, 0.25)),
    )
    
    # R2-书房（x: -1.67→1.67, y: 0→5）
    desk_r2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/desk_r2",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 0.7, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.38, 0.22),
                roughness=0.65,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.5, 3.5, 0.375)),
    )
    
    bookshelf_r2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bookshelf_r2",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 2.2, 1.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.3, 0.15),
                roughness=0.7,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 1.5, 0.95)),
    )
    
    # R3-卧室（x: 1.67→5.0, y: 0→5）
    bed_r3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bed_r3",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.5, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.6, 0.5),
                roughness=0.8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, 3.5, 0.3)),
    )
    
    wardrobe_r3 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wardrobe_r3",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.8, 2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.35, 0.25),
                roughness=0.65,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.4, 1.2, 1.0)),
    )
    
    # R4-餐厅（x: -5.0→-1.67, y: -5→0）
    dining_table_r4 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/dining_table_r4",
        spawn=sim_utils.CuboidCfg(
            size=(1.6, 1.0, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.45, 0.25),
                roughness=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.5, -2.5, 0.375)),
    )
    
    sideboard_r4 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sideboard_r4",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.6, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.35, 0.2),
                roughness=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.2, -4.2, 0.5)),
    )
    
    # R5-厨房（x: -1.67→1.67, y: -5→0）
    counter_r5 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/counter_r5",
        spawn=sim_utils.CuboidCfg(
            size=(2.5, 0.6, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.6),
                metallic=0.3,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -3.8, 0.45)),
    )
    
    fridge_r5 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/fridge_r5",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.7, 1.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.9),
                metallic=0.5,
                roughness=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, -1.5, 0.9)),
    )
    
    # R6-储藏室（x: 1.67→5.0, y: -5→0）
    shelf1_r6 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/shelf1_r6",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.35),
                roughness=0.7,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.3, -3.5, 0.75)),
    )
    
    shelf2_r6 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/shelf2_r6",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.35),
                roughness=0.7,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, -4.0, 0.75)),
    )

    # ROSOrin robot (mecanum wheel robot)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",  # Spawn位置
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/data/assets/rosorin/rosorin.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=2.0,
                max_angular_velocity=4.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.10),  # 高度设置
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*wheel.*": 0.0},
            joint_vel={".*wheel.*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*wheel.*"],
                effort_limit=50.0,
                velocity_limit=20.0,
                stiffness=0.0,  # 速度控制不需要stiffness
                damping=1e3,    # 高阻尼用于velocity tracking (原来是10.0太小)
            ),
        },
    )

    # RGB-D Camera (Aurora 930 or AScamera)
    # Real specs: 640x480, but using 96x80 for memory efficiency (further reduced)
    # NOTE: Requires --enable_cameras flag when launching
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/camera",
        update_period=0.4,  # 2.5 Hz (降低更新频率以节省显存)
        height=80,  # ↓ 从 120 降到 80
        width=96,   # ↓ 从 160 降到 96
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.057, 0.0, 0.092),  # Real position from URDF relative to base_link
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",  # ROS convention: x-forward, y-left, z-up
        ),
    )

    # LiDAR (MS200/LD19/A1 - 2D planar scan, 360°, 12m range)
    # Real position: (0.011, 0.0, 0.136) from base_link
    # NOTE:
    # - RayCasterCfg 只能对单一 mesh root 生效，且更偏向 Mesh/Plane；本场景墙体/家具大量由 CuboidCfg 生成 primitive（Cube）。
    # - MultiMeshRayCasterCfg 支持 regex + primitive shapes + 多目标，更适配当前六房间场景。
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lidar_frame",
        update_period=0.2,  # ↓ 降低到 5 Hz (从 10 Hz) 节省显存
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),  # 2D scan
            # ↓ 降低分辨率：从 360 射线降到 180 射线 (2° 间隔)，节省 50% 显存
            horizontal_fov_range=(0.0, 359.999),
            horizontal_res=2.0,  # 2° = 180 rays (从 1° = 360 rays)
        ),
        max_distance=12.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=[
            # ground: shared across envs
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/defaultGroundPlane",
                is_shared=True,
                track_mesh_transforms=False,
            ),
            # static obstacles in each env: walls/dividers/furniture
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/(wall_.*|divider_.*|.*_r[1-6])",
                is_shared=False,
                track_mesh_transforms=False,
            ),
        ],
    )

    # Contact sensor for collision detection
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=2,
        track_air_time=True,
    )

    # ========== Lighting System ==========
    # 主光源 (dome light - 环境光)
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1200.0,
            color=(1.0, 0.98, 0.95),  # 暖白光
        ),
    )
    
    # 顶部射灯 (distant light - 方向光)
    ceiling_light = AssetBaseCfg(
        prim_path="/World/ceiling_light",
        spawn=sim_utils.DistantLightCfg(
            intensity=1500.0,
            color=(1.0, 0.99, 0.97),
            angle=0.5,
        ),
    )


##
# MDP Settings (Observations, Actions, Rewards, Terminations)
##

@configclass
class ActionsCfg:
    """Action specifications for ROSOrin robot."""
    
    # Joint velocity commands for 4 wheels
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*wheel.*"],
        scale=10.0,  # 提高到10.0以增加移动速度
    )


@configclass
class ObservationsCfg:
    """Observation specifications for ROSOrin robot."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot proprioception
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        
        # Joint states (wheel velocities)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # 目标点信息 (相对位置和距离)
        goal_relative_position = ObsTerm(
            func=lambda env: env.goal_positions[:, :2] - env.scene.articulations["robot"].data.root_pos_w[:, :2]
            if hasattr(env, 'goal_positions') 
            else torch.zeros(env.num_envs, 2, device=env.device)
        )
        
        goal_distance = ObsTerm(
            func=lambda env: torch.norm(
                env.goal_positions[:, :2] - env.scene.articulations["robot"].data.root_pos_w[:, :2],
                dim=-1, keepdim=True
            ) if hasattr(env, 'goal_positions')
            else torch.zeros(env.num_envs, 1, device=env.device)
        )

        # LiDAR scan (360 ranges in meters)
        lidar_scan = ObsTerm(
            func=lambda env: torch.linalg.norm(
                env.scene.sensors["lidar"].data.ray_hits_w
                - env.scene.sensors["lidar"].data.pos_w.unsqueeze(1),
                dim=-1,
            ),
            noise=GaussianNoiseCfg(mean=0.0, std=0.02),
        )
        
        # Camera RGB image (160x120x3) - flattened
        # NOTE: Requires --enable_cameras flag
        camera_rgb = ObsTerm(
            func=lambda env: env.scene.sensors["camera"].data.output["rgb"].reshape(env.num_envs, -1),
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        
        # Camera depth image (160x120) - flattened  
        camera_depth = ObsTerm(
            func=lambda env: env.scene.sensors["camera"].data.output["distance_to_image_plane"].reshape(env.num_envs, -1),
            noise=GaussianNoiseCfg(mean=0.0, std=0.02),
        )
        
        # 注意：如果需要降采样，可在此对 lidar_scan 做切片（例如每 2 个取 1 个）。
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Define observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    改进后的奖励函数体系 - 密集+稀疏混合设计
    
    设计原则:
    1. 密集奖励: progress(每步反馈) + orientation(方向引导) + velocity(速度控制)
    2. 稀疏奖励: goal_reached(完全到达) + milestones(里程碑)
    3. 平滑惩罚: action_smoothness + stability + height
    4. 权重平衡: 主导航>>辅助>>惩罚 = 15:3:7.6
    """
    
    # ========== 主要密集奖励 (Dense Rewards) ==========
    # 向目标前进的进度奖励 (最重要，密集反馈)
    progress = RewTerm(
        func=rosorin_mdp.progress_reward,
        weight=20.0,  # ↑ 提高权重（从15.0→20.0），强化主要目标
        params={"threshold": 0.0005}  # ↓ 进一步降低阈值（确保密集）
    )
    
    # 朝向对齐奖励 (密集引导方向)
    orientation = RewTerm(
        func=rosorin_mdp.orientation_alignment_reward,
        weight=5.0  # ↑ 提高权重（从3.0→5.0），帮助导航
    )
    
    # 速度跟踪奖励 (有方向性的速度控制)
    velocity_tracking = RewTerm(
        func=rosorin_mdp.velocity_tracking_reward,
        weight=3.0,  # ↑ 提高权重（从2.0→3.0），鼓励移动
        params={"target_vel": 0.3}  # ↓ 降低目标速度，更安全
    )
    
    # ========== 稀疏奖励 (Sparse Rewards) ==========
    # 到达目标点的大额奖励 + 里程碑
    goal_reached = RewTerm(
        func=rosorin_mdp.goal_reached_reward,
        weight=100.0,  # 保持大额稀疏奖励
        params={"distance_threshold": 0.5}
    )
    
    # ========== 辅助奖励 (Auxiliary) ==========
    # 基础存活奖励 (降低权重，避免主导)
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.01  # ↓↓ 大幅降低，避免"原地不动"策略
    )
    
    # ========== 惩罚项 (Penalties) ==========
    # 动作平滑惩罚 (避免抖动)
    action_smoothness = RewTerm(
        func=rosorin_mdp.smooth_action_penalty,
        weight=0.1  # ↓ 大幅降低（从0.5→0.1），允许更多探索
    )
    
    # 🆕 避障惩罚（基于深度图像）
    obstacle_avoidance = RewTerm(
        func=rosorin_mdp.obstacle_avoidance_penalty,
        weight=1.0,  # ↓ 降低权重（从3.0→1.0），减少过度惩罚
        params={
            "safe_distance": 0.5,    # ↑ 增加安全距离（从0.4→0.5）
            "danger_distance": 0.25   # ↑ 增加危险距离（从0.2→0.25）
        }
    )
    
    # 姿态稳定惩罚 (避免倾覆)
    stability = RewTerm(
        func=rosorin_mdp.stability_penalty,
        weight=3.0,  # ↓ 降低（从5.0→3.0）
        params={
            "roll_threshold": 0.3,   # ↑ 放宽（从0.2→0.3）
            "pitch_threshold": 0.3   # ↑ 放宽（从0.2→0.3）
        }
    )
    
    # 高度惩罚 (保持合理高度)
    height = RewTerm(
        func=rosorin_mdp.height_penalty,
        weight=0.5,  # ↓ 大幅降低（从2.0→0.5）
        params={
            "min_height": 0.03,  # ↓ 放宽（从0.05→0.03）
            "max_height": 0.4    # ↑ 放宽（从0.3→0.4）
        }
    )


@configclass
class TerminationsCfg:
    """
    终止条件配置
    
    Episode终止情况:
    1. 成功到达目标点
    2. 机器人倾覆
    3. 超时
    """
    
    # 成功到达目标 (SUCCESS)
    goal_reached = DoneTerm(
        func=rosorin_mdp.goal_reached_termination,
        params={"distance_threshold": 0.5}
    )
    
    # 机器人倾覆 (FAILURE)
    robot_fallen = DoneTerm(
        func=rosorin_mdp.robot_fallen_termination,
        params={
            "roll_threshold": 0.5,
            "pitch_threshold": 0.5
        }
    )
    
    # 持续倒退终止（新增：如果机器人持续倒退超过阈值则终止）
    backward_termination = DoneTerm(
        func=rosorin_mdp.backward_termination,
        params={"backward_threshold": -0.1, "duration_steps": 50}  # 倒退速度<-0.1m/s持续50步(1秒)则终止
    )
    
    # 超时 (TIMEOUT)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """
    环境随机化配置
    
    每次reset时:
    1. 随机化机器人起始位置和朝向
    2. 随机生成新的目标点
    """
    
    # 随机化机器人起始位置
    reset_robot_position = EventTerm(
        func=rosorin_mdp.reset_robot_to_random_position,
        mode="reset",
        params={
            "x_range": (-3.0, 3.0),
            "y_range": (-3.0, 3.0),
            "yaw_range": (-3.14, 3.14)
        }
    )
    
    # 随机生成目标点位置
    reset_goal_position = EventTerm(
        func=rosorin_mdp.reset_goal_position,
        mode="reset",
        params={
            "min_distance": 1.5,  # 降低到1.5m（从3.0），更容易成功
            "max_distance": 4.0   # 降低到4.0m（从8.0），适应室内短距离导航
        }
    )


##
# Environment Configuration
##

@configclass
class ROSOrinEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ROSOrin driving RL environment."""

    # Scene settings
    scene: ROSOrinSceneCfg = ROSOrinSceneCfg(num_envs=4, env_spacing=5.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.decimation = 2  # 50Hz control @ 100Hz physics
        self.episode_length_s = 50.0  # ↓ 从100秒缩短到50秒 = 2500步（强迫策略学会快速到达）
        
        # Simulation settings
        self.sim.dt = 0.01  # 100Hz physics
