"""
ROSOrin MPC专家数据采集

使用MPC控制器生成专家轨迹，收集observation-action pairs用于训练。
"""

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="ROSOrin MPC专家数据采集")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=100, help="采集episode数量")
parser.add_argument("--max_steps", type=int, default=500, help="每个episode最大步数")
parser.add_argument("--output_dir", type=str, default="data/demonstrations", help="输出目录")
parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"], 
                    help="路径难度: easy(直线+简单曲线), medium(复杂曲线), hard(障碍物避障)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 强制启用相机（环境配置需要）
args.enable_cameras = True

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入Isaac Lab和自定义模块
from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
from mpc_controller import MPCController, MPCConfig, MecanumKinematics
from simple_path_generator import PathGenerator


class MPCDataCollector:
    """MPC数据收集器"""
    
    def __init__(self, env: ManagerBasedRLEnv, output_dir: str, difficulty: str = "easy"):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MPC控制器配置
        mpc_config = MPCConfig(
            horizon=10,
            dt=0.02,  # 50Hz控制频率
            Q_pos=10.0,
            Q_vel=1.0,
            Q_heading=5.0,
            R=0.1,
            max_linear_vel=0.5,  # 保守速度限制
            max_angular_vel=1.5,
            max_wheel_vel=15.0,
        )
        
        self.mpc = MPCController(mpc_config)
        self.kinematics = MecanumKinematics()
        self.path_gen = PathGenerator(difficulty=difficulty)
        
        # 数据缓冲区
        self.reset_buffers()
        
    def reset_buffers(self):
        """重置数据缓冲区"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        
    def collect_episode(self, env_id: int = 0) -> dict:
        """
        收集单个episode数据
        
        Args:
            env_id: 环境索引
            
        Returns:
            episode数据字典
        """
        # 重置环境
        obs_dict, _ = self.env.reset()
        obs = obs_dict["policy"]
        
        # 生成随机目标路径
        path_points = self.path_gen.generate_random_path()
        current_waypoint_idx = 0
        
        # 重置MPC
        self.mpc.reset()
        
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "path_points": path_points,
        }
        
        for step in range(args.max_steps):
            # 获取当前状态
            robot_pos = self.env.scene["robot"].data.root_pos_w[env_id, :2].cpu().numpy()
            robot_quat = self.env.scene["robot"].data.root_quat_w[env_id].cpu().numpy()
            robot_vel = self.env.scene["robot"].data.root_lin_vel_w[env_id, :2].cpu().numpy()
            robot_ang_vel = self.env.scene["robot"].data.root_ang_vel_w[env_id, 2].cpu().numpy()
            
            # 四元数转欧拉角（提取yaw）
            qw, qx, qy, qz = robot_quat
            theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            # 当前状态 [x, y, theta, vx, vy, omega]
            current_state = np.array([
                robot_pos[0], robot_pos[1], theta,
                robot_vel[0], robot_vel[1], robot_ang_vel
            ])
            
            # 更新目标航点（到达当前航点后切换到下一个）
            target_pos = path_points[current_waypoint_idx]
            distance_to_waypoint = np.linalg.norm(robot_pos - target_pos)
            
            if distance_to_waypoint < 0.3 and current_waypoint_idx < len(path_points) - 1:
                current_waypoint_idx += 1
                target_pos = path_points[current_waypoint_idx]
            
            # MPC计算最优控制（使用路径序列）
            # 从当前航点到路径末尾
            remaining_path = path_points[current_waypoint_idx:]
            if len(remaining_path) < 2:
                remaining_path = np.vstack([remaining_path, path_points[-1:]])
            
            optimal_control, _ = self.mpc.compute_control(
                current_state, 
                remaining_path
            )
            
            # 速度指令 -> 轮速
            vx_cmd, vy_cmd, omega_cmd = optimal_control
            robot_velocity = np.array([vx_cmd, vy_cmd, omega_cmd])
            wheel_velocities = self.kinematics.inverse_kinematics(robot_velocity)
            
            # 转换为tensor动作（Isaac Lab格式）
            action = torch.tensor(wheel_velocities, dtype=torch.float32, device=self.env.device)
            action = action.unsqueeze(0)  # [1, 4] for single env
            
            # 执行动作
            obs_dict, rewards, dones, truncated, infos = self.env.step(action)
            obs_next = obs_dict["policy"]
            
            # 保存数据
            episode_data["observations"].append(obs[env_id].cpu().numpy())
            episode_data["actions"].append(wheel_velocities)
            episode_data["rewards"].append(rewards[env_id].cpu().item())
            
            # 更新观测
            obs = obs_next
            
            # 检查终止条件
            if dones[env_id] or current_waypoint_idx >= len(path_points) - 1:
                break
        
        # 转换为numpy数组
        episode_data["observations"] = np.array(episode_data["observations"])
        episode_data["actions"] = np.array(episode_data["actions"])
        episode_data["rewards"] = np.array(episode_data["rewards"])
        
        return episode_data
    
    def save_dataset(self, episodes: list, filename: str):
        """
        保存数据集到HDF5文件
        
        Args:
            episodes: episode数据列表
            filename: 输出文件名
        """
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # 元数据
            f.attrs['num_episodes'] = len(episodes)
            f.attrs['difficulty'] = args.difficulty
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['obs_dim'] = episodes[0]['observations'].shape[1]
            f.attrs['action_dim'] = episodes[0]['actions'].shape[1]
            
            # 每个episode的数据
            for i, episode in enumerate(episodes):
                grp = f.create_group(f'episode_{i}')
                grp.create_dataset('observations', data=episode['observations'])
                grp.create_dataset('actions', data=episode['actions'])
                grp.create_dataset('rewards', data=episode['rewards'])
                grp.create_dataset('path_points', data=episode['path_points'])
                grp.attrs['length'] = len(episode['observations'])
                grp.attrs['total_reward'] = np.sum(episode['rewards'])
        
        print(f"\n✓ 数据集已保存: {filepath}")
        print(f"  - Episodes: {len(episodes)}")
        print(f"  - 总样本数: {sum(len(ep['observations']) for ep in episodes)}")
        print(f"  - 文件大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("  ROSOrin MPC专家数据采集")
    print("="*80)
    print(f"  - 并行环境: {args.num_envs}")
    print(f"  - Episode数: {args.num_episodes}")
    print(f"  - 最大步数: {args.max_steps}")
    print(f"  - 路径难度: {args.difficulty}")
    print(f"  - 输出目录: {args.output_dir}")
    print("="*80 + "\n")
    
    # 创建环境
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 创建数据收集器
    collector = MPCDataCollector(env, args.output_dir, args.difficulty)
    
    # 收集数据
    episodes = []
    total_steps = 0
    
    print("[1/2] 开始收集数据...\n")
    pbar = tqdm(total=args.num_episodes, desc="采集进度")
    
    episode_count = 0
    while episode_count < args.num_episodes:
        # 每个环境轮流收集episode
        for env_id in range(args.num_envs):
            if episode_count >= args.num_episodes:
                break
                
            try:
                episode_data = collector.collect_episode(env_id=env_id)
                episodes.append(episode_data)
                total_steps += len(episode_data["observations"])
                episode_count += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'steps': len(episode_data["observations"]),
                    'reward': f"{np.sum(episode_data['rewards']):.2f}",
                    'total_samples': total_steps
                })
                
            except Exception as e:
                print(f"\n⚠ Episode {episode_count} 失败: {e}")
                continue
    
    pbar.close()
    
    # 保存数据集
    print("\n[2/2] 保存数据集...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rosorin_mpc_demos_{args.difficulty}_{timestamp}.h5"
    collector.save_dataset(episodes, filename)
    
    # 统计信息
    print("\n" + "="*80)
    print("  ✅ 数据采集完成!")
    print("="*80)
    print(f"  - 成功Episodes: {len(episodes)}")
    print(f"  - 总样本数: {total_steps}")
    print(f"  - 平均Episode长度: {total_steps / len(episodes):.1f}")
    print(f"  - 平均奖励: {np.mean([np.sum(ep['rewards']) for ep in episodes]):.2f}")
    print("="*80 + "\n")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
