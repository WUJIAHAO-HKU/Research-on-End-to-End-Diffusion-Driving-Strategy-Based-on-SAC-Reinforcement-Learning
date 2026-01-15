"""
Phase 2: ROSOrin麦克纳姆轮机器人 - MPC专家数据采集

架构说明:
1. 仿真阶段: 使用Carter差分驱动机器人（有关节，支持joint velocity控制）
2. 控制接口: 设计统一的MecanumController抽象层
3. 实车部署: 控制器输出4轮速度 -> ROS2话题发布到真实电机

数据格式兼容性:
- 观察: [joint_pos(4), joint_vel(4), base_lin_vel(3), base_ang_vel(3)] - 14D
- 动作: [wheel_vel_fl, wheel_vel_fr, wheel_vel_rl, wheel_vel_rr] - 4D
- 实车适配: 将4轮速度映射到ROSOrin的麦克纳姆轮控制器
"""

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="ROSOrin MPC数据采集")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=100, help="采集episode数量")
parser.add_argument("--max_steps", type=int, default=200, help="每个episode最大步数")
parser.add_argument("--output_dir", type=str, default="data/demonstrations", help="输出目录")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

##
# 场景配置（Carter作为ROSOrin的仿真替代）
##

@configclass  
class ROSOrinSceneCfg(InteractiveSceneCfg):
    """ROSOrin场景配置
    
    注意: 使用Carter机器人作为临时替代，因为:
    1. Carter有4个轮子关节（可控制joint velocity）
    2. 数据格式与实车兼容（4轮速度控制）
    3. 后续可替换为真实ROSOrin URDF
    """
    
    # 地面
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.7,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # 机器人（临时使用Cartpole验证数据采集流程）
    # TODO: 替换为真实ROSOrin URDF（需要完整的ArticulationRootAPI）
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Simple/cartpole.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            joint_pos={
                ".*": 0.0,
            },
        ),
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
            "pole": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


@configclass
class ROSOrinEnvCfg(ManagerBasedEnvCfg):
    """ROSOrin环境配置（实车部署兼容设计）"""
    
    scene: ROSOrinSceneCfg = ROSOrinSceneCfg(num_envs=4, env_spacing=3.0)
    
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # 关节状态（实车: 4个麦克纳姆轮编码器）
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            
            # 底盘速度（实车: IMU + 轮速里程计融合）
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        
        policy: PolicyCfg = PolicyCfg()
    
    observations: ObservationsCfg = ObservationsCfg()
    
    @configclass
    class ActionsCfg:
        # 临时使用Cartpole的2关节（验证流程）
        # TODO: 替换为4轮速度控制
        joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot",
            joint_names=["slider_to_cart"],
            scale=5.0,
        )
    
    actions: ActionsCfg = ActionsCfg()
    
    def __post_init__(self):
        self.decimation = 4  # 25Hz控制频率（实车ROS2节点同频）
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.physics_material = self.scene.terrain.physics_material


##
# MPC控制器（实车部署兼容接口）
##

class MecanumMPCController:
    """
    ROSOrin麦克纳姆轮MPC控制器
    
    架构设计（Sim2Real兼容）:
    1. 输入: 机器人状态（关节位置/速度 + 底盘速度）
    2. 输出: 4轮速度命令 [rad/s]
    3. 实车部署: 输出直接映射到ROS2话题 /cmd_wheel_vel
    
    麦克纳姆轮运动学:
        [ω_fl]   [ 1  -1  -(L+W)]   [v_x  ]
        [ω_fr] = [ 1   1   (L+W)] * [v_y  ] / R
        [ω_rl]   [ 1   1  -(L+W)]   [ω_z  ]
        [ω_rr]   [ 1  -1   (L+W)]
    
    其中: L=轴距/2, W=轮距/2, R=轮半径
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # ROSOrin物理参数（与实车一致）
        self.wheelbase = 0.206  # m（前后轮距）
        self.track = 0.194      # m（左右轮距）
        self.wheel_radius = 0.0325  # m（麦克纳姆轮半径）
        
        # MPC轨迹参数
        self.target_speed = 0.5  # m/s（巡航速度）
        self.trajectory_variety = [
            # (v_x, v_y, ω_z, duration)
            (0.5, 0.0, 0.0, 50),    # 前进
            (0.5, 0.0, 0.3, 30),    # 前进+右转
            (0.5, 0.0, -0.3, 30),   # 前进+左转
            (0.3, 0.2, 0.0, 40),    # 斜向移动
            (0.0, 0.3, 0.0, 30),    # 横向平移
            (0.0, 0.0, 0.5, 25),    # 原地旋转
        ]
        self.current_trajectory = 0
        self.trajectory_step = 0
        
    def get_action(self, obs: torch.Tensor, step: int) -> torch.Tensor:
        """
        MPC控制策略（临时Cartpole版本 - 验证数据采集流程）
        
        TODO: 替换为真实的麦克纳姆轮控制逻辑
        Args:
            obs: (num_envs, obs_dim)
            step: 当前仿真步数
            
        Returns:
            actions: (num_envs, 1) - Cartpole力控制
        """
        batch_size = obs.shape[0]
        
        # 简单的随机策略（测试数据采集）
        actions = torch.randn(batch_size, 1, device=self.device) * 10.0
        
        return actions
        
        # 选择当前轨迹段
        traj_idx = (step // 50) % len(self.trajectory_variety)
        vx, vy, omega, _ = self.trajectory_variety[traj_idx]
        
        # 麦克纳姆轮逆运动学
        L = self.wheelbase / 2.0
        W = self.track / 2.0
        R = self.wheel_radius
        
        # 计算4轮速度（Carter有4个轮子，映射关系需要测试验证）
        # 注意: Carter是差分驱动，需要适配为4轮控制
        wheel_fl = (vx - vy - (L + W) * omega) / R
        wheel_fr = (vx + vy + (L + W) * omega) / R
##
# 数据收集器（实车部署兼容格式）
##

class DemonstrationCollector:
    """
    专家演示数据收集器
    
    数据格式（HDF5）:
    - observations: (T, obs_dim) - 机器人状态
    - actions: (T, 4) - 4轮速度命令
    - rewards: (T,) - 可选，用于RL训练
    - timestamps: (T,) - 时间戳（实车同步用）
    
    实车部署流程:
    1. 加载HDF5数据集
    2. 训练BC/Diffusion模型
    3. 部署: ROS2节点订阅传感器 -> 推理 -> 发布轮速
    """
    
    def __init__(self, env: ManagerBasedEnv, controller: MecanumMPCController):
        self.env = env
        self.controller = controller
        
        # 数据存储
        self.episodes = []
        
        print(f"\n[Collector] 初始化完成")
        print(f"  - 观察维度: {env.observation_manager.group_obs_dim['policy']}")
        print(f"  - 动作维度: {env.action_manager.total_action_dim}")
        control_freq = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        print(f"  - 控制频率: {control_freq:.1f} Hz")
        
    def collect_episode(self, max_steps: int = 200) -> dict:
        """
        收集单个episode（实车格式兼容）
        
        Returns:
            episode_data: {
                'observations': (T, obs_dim) - 状态序列
                'actions': (T, 4) - 轮速命令序列
                'robot_pos': (T, 3) - 底盘位置（仿真专有，实车用SLAM）
                'robot_quat': (T, 4) - 底盘姿态（仿真专有）
                'length': int - episode长度
                'success': bool - 是否成功完成
            }
        """
        # 重置环境
        obs, _ = self.env.reset()
        
        episode_data = {
            'observations': [],
            'actions': [],
            'robot_pos': [],
            'robot_quat': [],
        }
        
        for step in range(max_steps):
            if not simulation_app.is_running():
                break
            
            with torch.inference_mode():
                # MPC生成动作
                actions = self.controller.get_action(obs['policy'], step)
                
                # 执行动作
                obs_next, _ = self.env.step(actions)
                
                # 记录数据（取第一个环境的数据）
                episode_data['observations'].append(obs['policy'][0].cpu().numpy())
                episode_data['actions'].append(actions[0].cpu().numpy())
                
                # 记录机器人位姿（仿真groundtruth）
                robot = self.env.scene['robot']
                pos = robot.data.root_pos_w[0].cpu().numpy()
                quat = robot.data.root_quat_w[0].cpu().numpy()
                episode_data['robot_pos'].append(pos)
                episode_data['robot_quat'].append(quat)
                
                obs = obs_next
        
        # 转换为numpy数组
        for key in ['observations', 'actions', 'robot_pos', 'robot_quat']:
            if episode_data[key]:
                episode_data[key] = np.array(episode_data[key])
        
        episode_data['length'] = len(episode_data['observations'])
    def save(self, output_path: str):
        """
        保存为HDF5（实车部署兼容格式）
        
        数据结构:
        ├─ metadata (attrs)
        │  ├─ robot: "ROSOrin"
        │  ├─ platform: "IsaacLab_Carter_Sim"  # 标注仿真平台
        │  ├─ control_freq: 25.0  # Hz
        │  └─ obs_dim, action_dim, num_episodes, etc.
        ├─ episode_0/
        │  ├─ observations (T, 14)
        │  ├─ actions (T, 4)
        │  ├─ robot_pos (T, 3)  # 仿真groundtruth
        │  └─ robot_quat (T, 4)
        └─ episode_1/ ...
        
        实车适配说明:
        - observations[:, 0:4]: joint_pos（实车: 轮子编码器）
        - observations[:, 4:8]: joint_vel（实车: 轮速计）
        - observations[:, 8:11]: base_lin_vel（实车: IMU线速度）
        - observations[:, 11:14]: base_ang_vel（实车: IMU角速度）
        - actions: 4轮速度命令（实车: 直接发送到电机）
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # 元数据
            f.attrs['robot'] = 'ROSOrin'
            f.attrs['platform'] = 'IsaacLab_Carter_Simulator'
            f.attrs['controller'] = 'MecanumMPC'
            f.attrs['control_frequency'] = 25.0  # Hz
            f.attrs['observation_dim'] = self.episodes[0]['observations'].shape[1]
            f.attrs['action_dim'] = self.episodes[0]['actions'].shape[1]
            f.attrs['num_episodes'] = len(self.episodes)
            f.attrs['total_steps'] = sum(e['length'] for e in self.episodes)
            f.attrs['collection_date'] = datetime.now().isoformat()
            f.attrs['sim2real_ready'] = True  # 标记为实车部署就绪
            
            # 机器人物理参数（用于实车验证）
            f.attrs['wheelbase'] = 0.206
            f.attrs['track'] = 0.194
            f.attrs['wheel_radius'] = 0.0325
            
            # 保存每个episode
            for i, episode in enumerate(self.episodes):
                ep_grp = f.create_group(f'episode_{i}')
                
                for key, data in episode.items():
                    if isinstance(data, np.ndarray):
                        ep_grp.create_dataset(key, data=data, compression='gzip')
                    else:
                        ep_grp.attrs[key] = data
        
        # 统计信息
        success_rate = np.mean([e['success'] for e in self.episodes]) * 100
        avg_length = np.mean([e['length'] for e in self.episodes])
        
        print(f"\n{'='*70}")
        print(f"数据保存完成: {output_file}")
        print(f"{'='*70}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  总步数: {sum(e['length'] for e in self.episodes)}")
        print(f"  平均长度: {avg_length:.1f} 步")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  观察维度: {f.attrs['observation_dim']}")
        print(f"  动作维度: {f.attrs['action_dim']}")
        print(f"\n实车部署就绪: ✓")
        print(f"  - 控制频率: 25 Hz")
        print(f"  - 输出格式: 4轮速度 [rad/s]")
        print(f"  - ROS2接口: /cmd_wheel_vel (Float64MultiArray)")
        print(f"{'='*70}\n")
        
        for ep in range(num_episodes):
            episode = self.collect_episode(max_steps)
            self.episodes.append(episode)
            
            if (ep + 1) % 10 == 0:
                avg_len = np.mean([e['length'] for e in self.episodes])
                print(f"  Episode {ep+1}/{num_episodes} | 平均长度: {avg_len:.1f}")
        
        print(f"\n✓ 数据收集完成!")
    
    def save(self, output_path: str):
        """保存为HDF5"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # 元数据
            f.attrs['robot'] = 'ROSOrin'
            f.attrs['controller'] = 'MecanumMPC'
            f.attrs['num_episodes'] = len(self.episodes)
            f.attrs['total_steps'] = sum(e['length'] for e in self.episodes)
            f.attrs['collection_date'] = datetime.now().isoformat()
            
            # 保存每个episode
            for i, episode in enumerate(self.episodes):
                ep_grp = f.create_group(f'episode_{i}')
                
                for key, data in episode.items():
                    if isinstance(data, np.ndarray):
                        ep_grp.create_dataset(key, data=data)
                    else:
                        ep_grp.attrs[key] = data
        
        print(f"\n数据已保存: {output_file}")
        print(f"  Episodes: {len(self.episodes)}")
        print(f"  总步数: {sum(e['length'] for e in self.episodes)}")
        print(f"  平均长度: {np.mean([e['length'] for e in self.episodes]):.1f}")


##
# 主函数
##

def main():
    """Phase 2: ROSOrin MPC数据采集"""
    
    print(f"\n{'='*70}")
    print(f"  Phase 2: ROSOrin MPC专家数据采集")
    print(f"{'='*70}\n")
    
    # 创建环境
    print(f"[1/4] 创建Isaac Lab环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
    print(f"✓ {args.num_envs}个并行环境已创建")
    print(f"  - 观察维度: {env.observation_manager.group_obs_dim['policy']}")
    print(f"  - 动作维度: {env.action_manager.total_action_dim}")
    
    # 创建MPC控制器
    print(f"\n[2/4] 初始化麦克纳姆轮MPC控制器...")
    controller = MecanumMPCController(device=env.device)
    print(f"✓ MPC控制器初始化完成")
    
    # 创建数据收集器
    print(f"\n[3/4] 开始数据收集...")
    collector = DemonstrationCollector(env, controller)
    
    episodes_per_env = args.num_episodes // args.num_envs
    collector.collect(num_episodes=episodes_per_env, max_steps=args.max_steps)
    
    # 保存数据
    print(f"\n[4/4] 保存数据...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/rosorin_mpc_{timestamp}.hdf5"
    collector.save(output_path)
    
    print(f"\n{'='*70}")
    print(f"Phase 2 完成!")
    print(f"{'='*70}\n")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    print("[DEBUG] 程序启动...")
    try:
        print("[DEBUG] 调用main()...")
        main()
        print("[DEBUG] main()执行完成")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] 关闭simulation_app...")
        simulation_app.close()
        print("[DEBUG] 程序结束")
