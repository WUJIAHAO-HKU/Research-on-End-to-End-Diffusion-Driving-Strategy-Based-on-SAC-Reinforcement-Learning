"""
MPC Expert Demonstration Collector for ROSOrin

使用MPC控制器收集专家驾驶演示数据，用于行为克隆和SAC-Diffusion训练。
"""

import h5py
import numpy as np
import torch
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加scripts目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpc_controller import MPCController, MPCConfig
from path_generator import PathGenerator, PathType


class DemonstrationCollector:
    """演示数据收集器"""
    
    def __init__(
        self,
        env,
        mpc_controller: MPCController,
        path_generator: PathGenerator,
        save_dir: str = "data/demonstrations",
    ):
        self.env = env
        self.mpc = mpc_controller
        self.path_gen = path_generator
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集的数据
        self.episodes = []
        
    def collect_episode(
        self,
        target_positions: np.ndarray,
        target_headings: np.ndarray,
        max_steps: int = 500,
        success_threshold: float = 0.2,
    ) -> Dict:
        """
        收集一条轨迹
        
        Returns:
            episode_data: 包含observations, actions, rewards等的字典
        """
        # 重置环境
        obs, info = self.env.reset()
        self.mpc.reset()
        
        # 数据存储
        observations = []
        actions = []
        rewards = []
        robot_states = []  # [x, y, theta, vx, vy, omega]
        
        done = False
        step = 0
        total_reward = 0.0
        
        while not done and step < max_steps:
            # 获取当前机器人状态（从观测中提取）
            # obs['policy'] = [base_lin_vel(3), base_ang_vel(3), joint_vel(4)] = 10D
            obs_tensor = obs['policy']  # (num_envs, 10)
            
            # 取第一个环境的观测
            lin_vel = obs_tensor[0, :3].cpu().numpy()  # [vx, vy, vz]
            ang_vel = obs_tensor[0, 3:6].cpu().numpy()  # [wx, wy, wz]
            joint_vel = obs_tensor[0, 6:].cpu().numpy()  # [w_fl, w_fr, w_bl, w_br]
            
            # 获取机器人位置（从环境中）
            # 注意：实际位置需要从env.scene.robot获取
            robot_pos = self.env.scene.articulations["robot"].data.root_pos_w[0].cpu().numpy()
            robot_quat = self.env.scene.articulations["robot"].data.root_quat_w[0].cpu().numpy()
            
            # 从四元数提取yaw角
            # quat = [w, x, y, z]
            w, x, y, z = robot_quat
            theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            
            # 构造完整状态 [x, y, theta, vx, vy, omega]
            robot_state = np.array([
                robot_pos[0], robot_pos[1], theta,
                lin_vel[0], lin_vel[1], ang_vel[2]
            ])
            
            # MPC计算控制
            control, mpc_info = self.mpc.compute_control(
                robot_state,
                target_positions,
                target_headings
            )
            
            # 转换为轮速命令
            wheel_velocities = self.mpc.robot_velocity_to_wheel_velocity(control)
            
            # 转换为Isaac Lab的action格式（Tensor）
            action = torch.from_numpy(wheel_velocities).float().unsqueeze(0).to(self.env.device)
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated[0] or truncated[0]
            
            # 记录数据
            observations.append(obs_tensor[0].cpu().numpy())
            actions.append(wheel_velocities)
            rewards.append(reward[0].cpu().item())
            robot_states.append(robot_state)
            
            total_reward += reward[0].cpu().item()
            step += 1
            
            # 检查是否成功到达目标
            distance_to_goal = np.linalg.norm(
                robot_pos[:2] - target_positions[-1]
            )
            if distance_to_goal < success_threshold:
                print(f"  ✓ 成功到达目标! (步数: {step}, 距离: {distance_to_goal:.3f}m)")
                break
        
        # 判断成功
        final_distance = np.linalg.norm(
            robot_states[-1][:2] - target_positions[-1]
        )
        success = final_distance < success_threshold
        
        episode_data = {
            'observations': np.array(observations),  # (T, 10)
            'actions': np.array(actions),  # (T, 4)
            'rewards': np.array(rewards),  # (T,)
            'robot_states': np.array(robot_states),  # (T, 6)
            'target_path': target_positions,  # (N, 2)
            'target_headings': target_headings,  # (N,)
            'total_reward': total_reward,
            'steps': step,
            'success': success,
            'final_distance': final_distance,
        }
        
        return episode_data
    
    def collect_batch(
        self,
        num_episodes: int,
        path_type: PathType = PathType.RANDOM_WAYPOINTS,
        max_steps: int = 500,
        **path_kwargs
    ) -> List[Dict]:
        """
        批量收集演示数据
        
        Args:
            num_episodes: 收集的轨迹数量
            path_type: 路径类型
            max_steps: 每条轨迹最大步数
            **path_kwargs: 传递给路径生成器的参数
        """
        print(f"\n开始收集 {num_episodes} 条演示轨迹...")
        print(f"路径类型: {path_type.value}")
        
        episodes = []
        success_count = 0
        
        # 生成路径
        print("生成训练路径...")
        paths = self.path_gen.generate_batch(
            path_type,
            batch_size=num_episodes,
            **path_kwargs
        )
        
        # 收集数据
        for i in tqdm(range(num_episodes), desc="收集演示"):
            target_positions, target_headings = paths[i]
            
            try:
                episode_data = self.collect_episode(
                    target_positions,
                    target_headings,
                    max_steps=max_steps
                )
                
                episodes.append(episode_data)
                
                if episode_data['success']:
                    success_count += 1
                    
            except Exception as e:
                print(f"\n警告: 第 {i+1} 条轨迹收集失败: {e}")
                continue
        
        success_rate = success_count / len(episodes) if episodes else 0.0
        avg_reward = np.mean([ep['total_reward'] for ep in episodes]) if episodes else 0.0
        avg_steps = np.mean([ep['steps'] for ep in episodes]) if episodes else 0.0
        
        print(f"\n收集完成!")
        print(f"  - 成功轨迹: {success_count}/{len(episodes)} ({success_rate*100:.1f}%)")
        print(f"  - 平均奖励: {avg_reward:.3f}")
        print(f"  - 平均步数: {avg_steps:.1f}")
        
        self.episodes.extend(episodes)
        return episodes
    
    def save_to_hdf5(self, filename: str = None):
        """
        保存演示数据到HDF5文件
        
        HDF5文件结构:
        /episode_0/
            observations: (T, 10)
            actions: (T, 4)
            rewards: (T,)
            robot_states: (T, 6)
            target_path: (N, 2)
            target_headings: (N,)
        /episode_1/
            ...
        /metadata/
            num_episodes
            total_transitions
            success_rate
            ...
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rosorin_mpc_demos_{timestamp}.h5"
        
        filepath = self.save_dir / filename
        
        print(f"\n保存演示数据到: {filepath}")
        
        with h5py.File(filepath, 'w') as f:
            # 保存每条轨迹
            for i, episode in enumerate(self.episodes):
                ep_group = f.create_group(f'episode_{i}')
                
                ep_group.create_dataset('observations', data=episode['observations'])
                ep_group.create_dataset('actions', data=episode['actions'])
                ep_group.create_dataset('rewards', data=episode['rewards'])
                ep_group.create_dataset('robot_states', data=episode['robot_states'])
                ep_group.create_dataset('target_path', data=episode['target_path'])
                ep_group.create_dataset('target_headings', data=episode['target_headings'])
                
                # 标量属性
                ep_group.attrs['total_reward'] = episode['total_reward']
                ep_group.attrs['steps'] = episode['steps']
                ep_group.attrs['success'] = episode['success']
                ep_group.attrs['final_distance'] = episode['final_distance']
            
            # 保存元数据
            meta_group = f.create_group('metadata')
            meta_group.attrs['num_episodes'] = len(self.episodes)
            meta_group.attrs['total_transitions'] = sum(ep['steps'] for ep in self.episodes)
            meta_group.attrs['success_rate'] = np.mean([ep['success'] for ep in self.episodes])
            meta_group.attrs['avg_reward'] = np.mean([ep['total_reward'] for ep in self.episodes])
            meta_group.attrs['avg_steps'] = np.mean([ep['steps'] for ep in self.episodes])
            meta_group.attrs['timestamp'] = datetime.now().isoformat()
        
        print(f"✓ 保存完成! 文件大小: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        return filepath
    
    def get_statistics(self) -> Dict:
        """获取收集数据的统计信息"""
        if not self.episodes:
            return {}
        
        return {
            'num_episodes': len(self.episodes),
            'total_transitions': sum(ep['steps'] for ep in self.episodes),
            'success_rate': np.mean([ep['success'] for ep in self.episodes]),
            'avg_reward': np.mean([ep['total_reward'] for ep in self.episodes]),
            'std_reward': np.std([ep['total_reward'] for ep in self.episodes]),
            'avg_steps': np.mean([ep['steps'] for ep in self.episodes]),
            'avg_final_distance': np.mean([ep['final_distance'] for ep in self.episodes]),
        }


def main():
    """主函数：运行演示收集"""
    
    print("="*70)
    print("  ROSOrin MPC专家演示收集")
    print("="*70)
    
    # 启动Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    
    # 导入Isaac Lab环境
    from isaaclab.envs import ManagerBasedRLEnv
    import rosorin_env_cfg
    
    # 创建环境
    print("\n[1/4] 创建环境...")
    env_cfg = rosorin_env_cfg.ROSOrinEnvCfg()
    env_cfg.scene.num_envs = 1  # 单环境收集
    env_cfg.episode_length_s = 30.0  # 30秒每条轨迹
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"✓ 环境创建成功 (设备: {env.device})")
    
    # 创建MPC控制器
    print("\n[2/4] 创建MPC控制器...")
    mpc_config = MPCConfig(
        horizon=10,
        dt=0.02,
        Q_pos=10.0,
        Q_heading=5.0,
        max_linear_vel=0.8,
        max_angular_vel=2.0,
    )
    mpc = MPCController(config=mpc_config)
    print("✓ MPC控制器就绪")
    
    # 创建路径生成器
    print("\n[3/4] 创建路径生成器...")
    path_gen = PathGenerator(seed=42)
    print("✓ 路径生成器就绪")
    
    # 创建数据收集器
    print("\n[4/4] 开始数据收集...")
    collector = DemonstrationCollector(
        env=env,
        mpc_controller=mpc,
        path_generator=path_gen,
    )
    
    # 收集演示（混合不同类型的路径）
    path_types_to_collect = [
        (PathType.STRAIGHT, 20, {'num_points': 100}),
        (PathType.CURVE, 20, {'num_points': 100}),
        (PathType.S_CURVE, 20, {'num_points': 100}),
        (PathType.RANDOM_WAYPOINTS, 40, {'num_waypoints': 5, 'num_points': 100}),
    ]
    
    for path_type, num_eps, kwargs in path_types_to_collect:
        print(f"\n{'='*70}")
        print(f"  收集 {path_type.value} 路径")
        print(f"{'='*70}")
        
        collector.collect_batch(
            num_episodes=num_eps,
            path_type=path_type,
            max_steps=1500,  # 30s @ 50Hz = 1500 steps
            **kwargs
        )
    
    # 打印统计信息
    print("\n" + "="*70)
    print("  收集统计")
    print("="*70)
    stats = collector.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # 保存数据
    print("\n" + "="*70)
    print("  保存数据")
    print("="*70)
    filepath = collector.save_to_hdf5()
    
    # 清理
    env.close()
    simulation_app.close()
    
    print("\n" + "="*70)
    print("  ✓ 演示收集完成!")
    print("="*70)
    print(f"\n数据文件: {filepath}")
    print(f"轨迹数量: {stats['num_episodes']}")
    print(f"总步数: {stats['total_transitions']}")
    print(f"成功率: {stats['success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
