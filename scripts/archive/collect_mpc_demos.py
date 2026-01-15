"""
Phase 2: 专家数据采集 - 简化MPC控制器

基于Isaac Lab环境收集专家演示数据
"""

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

# 解析参数
parser = argparse.ArgumentParser(description="收集MPC专家演示数据")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=100, help="采集episode数量")
parser.add_argument("--output_dir", type=str, default="data/demonstrations", help="输出目录")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# 启动Isaac Sim (直接使用SimulationApp)
print(f"[DEBUG] 启动Isaac Sim...")
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})
print(f"[DEBUG] Isaac Sim已启动")

# 导入Isaac Lab
print(f"[DEBUG] 导入Isaac Lab模块...")
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
print(f"[DEBUG] Isaac Lab模块导入完成")


##
# 场景配置
##

print(f"[DEBUG] 定义场景配置...")

@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """演示收集场景配置"""
    
    # 地面
    ground = sim_utils.GroundPlaneCfg()
    
    # 机器人 (Carter作为ROSOrin替代)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Carter/carter_v1.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2),
        ),
        actuators={
            "base": mdp.ImplicitActuatorCfg(
                joint_names_expr=["joint_wheel_.*"],
                effort_limit=100.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=1e5,
            ),
        },
    )


@configclass
class DemoEnvCfg(ManagerBasedEnvCfg):
    """演示收集环境配置"""
    
    scene: DemoSceneCfg = DemoSceneCfg(num_envs=4, env_spacing=3.0)
    
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # 关节状态
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            # 基座速度
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        
        policy: PolicyCfg = PolicyCfg()
    
    observations: ObservationsCfg = ObservationsCfg()
    
    @configclass
    class ActionsCfg:
        joint_vel = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=2.0,
        )
    
    actions: ActionsCfg = ActionsCfg()
    
    @configclass
    class EventCfg:
        reset_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "position_range": (-0.1, 0.1),
                "velocity_range": (0.0, 0.0),
            },
        )
    
    events: EventCfg = EventCfg()
    
    def __post_init__(self):
        self.decimation = 4  # 25Hz控制
        self.sim.dt = 0.01   # 100Hz物理


class SimpleMPCController:
    """简化的MPC控制器 - 用于演示数据收集"""
    
    def __init__(self, action_dim: int, device: str = "cuda"):
        self.action_dim = action_dim
        self.device = device
        
        # 控制参数
        self.target_distance = 5.0  # 目标前进距离
        self.max_speed = 0.5
        
    def get_action(self, obs: torch.Tensor, step: int) -> torch.Tensor:
        """
        生成控制动作
        
        Args:
            obs: 观察 (num_envs, obs_dim)
            step: 当前步数
            
        Returns:
            actions: (num_envs, action_dim)
        """
        batch_size = obs.shape[0]
        
        # 简单的正弦波轨迹控制
        t = step * 0.01  # 时间
        
        # 生成动作: 前进 + 轻微转向
        actions = torch.zeros(batch_size, self.action_dim, device=self.device)
        
        # 主要是前轮转向
        if self.action_dim >= 2:
            actions[:, 0] = 0.3 * torch.cos(torch.tensor(t * 0.5))  # 左轮
            actions[:, 1] = 0.3 * torch.cos(torch.tensor(t * 0.5))  # 右轮
        
        return actions


class DemonstrationCollector:
    """演示数据收集器"""
    
    def __init__(self, env: ManagerBasedEnv, controller: SimpleMPCController):
        self.env = env
        self.controller = controller
        
        # 数据缓存
        self.observations = []
        self.actions = []
        self.rewards = []
        self.episode_lengths = []
        
    def collect_episode(self, max_steps: int = 200) -> dict:
        """收集单个episode"""
        
        # 重置环境
        obs, _ = self.env.reset()
        
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        
        step = 0
        while step < max_steps and simulation_app.is_running():
            with torch.inference_mode():
                # MPC生成动作
                actions = self.controller.get_action(obs['policy'], step)
                
                # 环境步进
                obs, _ = self.env.step(actions)
                
                # 记录
                episode_obs.append(obs['policy'].cpu().numpy())
                episode_actions.append(actions.cpu().numpy())
                
                step += 1
        
        return {
            'observations': np.array(episode_obs),
            'actions': np.array(episode_actions),
            'length': step,
        }
    
    def collect(self, num_episodes: int, max_steps: int = 200):
        """收集多个episodes"""
        
        print(f"\n{'='*70}")
        print(f"开始收集 {num_episodes} 个演示episodes")
        print(f"{'='*70}\n")
        
        for ep in range(num_episodes):
            # 收集episode
            episode_data = self.collect_episode(max_steps)
            
            # 保存
            self.observations.append(episode_data['observations'])
            self.actions.append(episode_data['actions'])
            self.episode_lengths.append(episode_data['length'])
            
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{num_episodes} | 平均长度: {np.mean(self.episode_lengths):.1f}")
        
        print(f"\n✓ 收集完成!")
    
    def save(self, output_path: str):
        """保存为HDF5格式"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # 元数据
            f.attrs['num_episodes'] = len(self.observations)
            f.attrs['total_steps'] = sum(self.episode_lengths)
            f.attrs['collection_date'] = datetime.now().isoformat()
            
            # 每个episode的数据
            for i, (obs, act, length) in enumerate(zip(self.observations, self.actions, self.episode_lengths)):
                ep_grp = f.create_group(f'episode_{i}')
                ep_grp.create_dataset('observations', data=obs)
                ep_grp.create_dataset('actions', data=act)
                ep_grp.attrs['length'] = length
        
        print(f"\n数据已保存到: {output_file}")
        print(f"  - Episodes: {len(self.observations)}")
        print(f"  - 总步数: {sum(self.episode_lengths)}")
        print(f"  - 平均长度: {np.mean(self.episode_lengths):.1f}")


def main():
    """主函数"""
    
    print(f"\n{'='*70}")
    print(f"  Phase 2: MPC专家演示数据收集")
    print(f"{'='*70}\n")
    
    # 创建环境
    print(f"[1/4] 创建环境...")
    env_cfg = DemoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
    print(f"✓ {args.num_envs}个并行环境创建成功")
    
    # 创建MPC控制器
    print(f"\n[2/4] 初始化MPC控制器...")
    controller = SimpleMPCController(
        action_dim=env.action_manager.total_action_dim,
        device=env.device,
    )
    print(f"✓ MPC控制器初始化完成")
    
    # 创建数据收集器
    print(f"\n[3/4] 准备数据收集...")
    collector = DemonstrationCollector(env, controller)
    
    # 收集数据
    episodes_per_env = args.num_episodes // args.num_envs
    collector.collect(num_episodes=episodes_per_env, max_steps=200)
    
    # 保存数据
    print(f"\n[4/4] 保存数据...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{args.output_dir}/mpc_expert_{timestamp}.hdf5"
    collector.save(output_path)
    
    print(f"\n{'='*70}")
    print(f"Phase 2 完成!")
    print(f"{'='*70}\n")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    print(f"[DEBUG] 进入main函数...")
    try:
        main()
    except Exception as e:
        print(f"[ERROR] 程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[DEBUG] 关闭simulation_app...")
        simulation_app.close()
        print(f"[DEBUG] 程序结束")
