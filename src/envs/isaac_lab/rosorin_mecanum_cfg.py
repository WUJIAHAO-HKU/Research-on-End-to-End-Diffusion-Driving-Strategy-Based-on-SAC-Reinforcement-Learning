"""
ROSOrin麦克纳姆轮机器人 - 简化配置

使用Isaac Lab的基础组件构建ROSOrin机器人
参数基于真实ROSOrin规格
"""

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

##
# ROSOrin物理参数 (基于真实机器人)
##

# 底盘尺寸
CHASSIS_LENGTH = 0.206  # 20.6cm
CHASSIS_WIDTH = 0.194   # 19.4cm  
CHASSIS_HEIGHT = 0.08   # 8cm

# 麦克纳姆轮参数
WHEEL_RADIUS = 0.0325   # 32.5mm
WHEEL_WIDTH = 0.025     # 25mm
WHEEL_MASS = 0.05       # 50g

# 轮距参数
WHEELBASE = CHASSIS_LENGTH  # 前后轮距
TRACK = CHASSIS_WIDTH       # 左右轮距

# 机器人总质量
ROBOT_MASS = 2.0  # 约2kg


@configclass
class ROSOrinMecanumCfg(ArticulationCfg):
    """
    ROSOrin麦克纳姆轮机器人配置
    
    使用基础几何体构建，避免URDF依赖
    """
    
    @configclass
    class InitialStateCfg(ArticulationCfg.InitialStateCfg):
        pos = (0.0, 0.0, 0.1)  # 初始位置
        rot = (1.0, 0.0, 0.0, 0.0)  # 四元数 (w, x, y, z)
        
        # 关节初始状态 (4个轮子)
        joint_pos = {
            ".*_wheel_joint": 0.0,
        }
        joint_vel = {
            ".*_wheel_joint": 0.0,
        }
    
    ##
    # Spawn配置 - 使用几何体构建机器人
    ##
    
    @configclass  
    class SpawnCfg:
        """生成ROSOrin机器人的几何体配置"""
        
        func = sim_utils.spawn_from_usd
        
        # 使用基础立方体作为底盘
        usd_path = None  # 稍后我们用代码生成
        
        # 物理材质
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=2.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        )
        
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        )
        
        # 碰撞属性
        collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.001,
        )
    
    ##
    # 执行器配置
    ##
    
    @configclass
    class ActuatorsCfg:
        """4个麦克纳姆轮的执行器"""
        
        # 轮子速度控制
        wheels = sim_utils.ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=10.0,  # 10N·m 最大扭矩
            velocity_limit=20.0,  # 20 rad/s 最大角速度  
            stiffness=0.0,
            damping=0.1,
        )
    
    # 基础配置
    prim_path = "{ENV_REGEX_NS}/Robot"
    spawn = SpawnCfg()
    init_state = InitialStateCfg()
    actuators = ActuatorsCfg()


def create_rosorin_scene_cfg(num_envs: int = 1, env_spacing: float = 2.5) -> InteractiveSceneCfg:
    """
    创建包含ROSOrin机器人的场景配置
    
    Args:
        num_envs: 并行环境数量
        env_spacing: 环境间距
        
    Returns:
        场景配置
    """
    
    @configclass
    class ROSOrinSceneCfg(InteractiveSceneCfg):
        """ROSOrin驾驶场景"""
        
        # 地面
        ground = sim_utils.GroundPlaneCfg()
        
        # ROSOrin机器人
        robot = ROSOrinMecanumCfg()
        
        # 激光雷达 (360度扫描)
        lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lidar",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),  # 激光雷达高度
            attach_yaw_only=True,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(0.0, 360.0),
                horizontal_res=1.0,  # 1度分辨率，360个点
            ),
            max_distance=10.0,  # 10米最大距离
            drift_range=(-0.0, 0.0),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        
        # TODO: 添加相机配置
        # camera = CameraCfg(...)
        
    scene_cfg = ROSOrinSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    return scene_cfg


# 麦克纳姆轮运动学
class MecanumKinematics:
    """
    麦克纳姆轮运动学模型
    
    将底盘速度 (vx, vy, omega) 转换为4个轮子的角速度
    """
    
    def __init__(self, wheelbase: float = WHEELBASE, track: float = TRACK, radius: float = WHEEL_RADIUS):
        """
        Args:
            wheelbase: 前后轮距 (m)
            track: 左右轮距 (m)  
            radius: 轮子半径 (m)
        """
        self.L = wheelbase / 2.0
        self.W = track / 2.0
        self.R = radius
        
        # 逆运动学矩阵
        # wheel_speeds = K @ [vx, vy, omega]^T
        # 轮子顺序: 左前, 右前, 左后, 右后
        self.K = torch.tensor([
            [1/self.R, -1/self.R, -(self.L + self.W)/self.R],  # 左前
            [1/self.R,  1/self.R,  (self.L + self.W)/self.R],  # 右前
            [1/self.R,  1/self.R, -(self.L + self.W)/self.R],  # 左后
            [1/self.R, -1/self.R,  (self.L + self.W)/self.R],  # 右后
        ])
    
    def inverse_kinematics(self, cmd_vel: torch.Tensor) -> torch.Tensor:
        """
        逆运动学: 底盘速度 → 轮速
        
        Args:
            cmd_vel: (batch, 3) - [vx, vy, omega]
                vx: 前向速度 (m/s)
                vy: 侧向速度 (m/s)
                omega: 角速度 (rad/s)
                
        Returns:
            wheel_speeds: (batch, 4) - [左前, 右前, 左后, 右后] (rad/s)
        """
        return cmd_vel @ self.K.T.to(cmd_vel.device)
    
    def forward_kinematics(self, wheel_speeds: torch.Tensor) -> torch.Tensor:
        """
        正运动学: 轮速 → 底盘速度
        
        Args:
            wheel_speeds: (batch, 4) - [左前, 右前, 左后, 右后] (rad/s)
            
        Returns:
            cmd_vel: (batch, 3) - [vx, vy, omega]
        """
        # 伪逆求解
        K_pinv = torch.pinverse(self.K).to(wheel_speeds.device)
        return wheel_speeds @ K_pinv.T


if __name__ == "__main__":
    # 测试运动学
    print("\n=== ROSOrin麦克纳姆轮运动学测试 ===\n")
    
    kinematics = MecanumKinematics()
    
    # 测试用例
    test_cases = [
        ("前进", torch.tensor([[0.5, 0.0, 0.0]])),
        ("后退", torch.tensor([[-0.5, 0.0, 0.0]])),
        ("左平移", torch.tensor([[0.0, 0.5, 0.0]])),
        ("右平移", torch.tensor([[0.0, -0.5, 0.0]])),
        ("原地左转", torch.tensor([[0.0, 0.0, 1.0]])),
        ("前进+左转", torch.tensor([[0.5, 0.0, 0.5]])),
    ]
    
    for name, cmd in test_cases:
        wheels = kinematics.inverse_kinematics(cmd)
        print(f"{name:12s}: cmd={cmd[0].tolist()}")
        print(f"{'':12s}  wheels=[LF:{wheels[0,0]:.2f}, RF:{wheels[0,1]:.2f}, LB:{wheels[0,2]:.2f}, RB:{wheels[0,3]:.2f}]")
        print()
    
    print("\n=== 物理参数 ===")
    print(f"底盘: {CHASSIS_LENGTH*1000:.1f}mm × {CHASSIS_WIDTH*1000:.1f}mm × {CHASSIS_HEIGHT*1000:.1f}mm")
    print(f"轮距: 前后={WHEELBASE*1000:.1f}mm, 左右={TRACK*1000:.1f}mm")
    print(f"轮子: 半径={WHEEL_RADIUS*1000:.1f}mm, 质量={WHEEL_MASS*1000:.0f}g")
    print(f"总质量: {ROBOT_MASS:.1f}kg\n")
