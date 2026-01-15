#!/usr/bin/env python3
"""
可视化BC模型策略 - 在Isaac Sim中观察BC模型的驾驶行为

功能：
1. 加载训练好的BC模型
2. 在室内场景中运行BC策略
3. 可视化机器人导航过程
4. 记录成功率、碰撞率等指标
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import time

# IsaacLab imports
from isaaclab.app import AppLauncher

# 解析命令行参数
parser = argparse.ArgumentParser(description="可视化BC模型策略")
parser.add_argument('--checkpoint', type=str, required=True,
                   help='BC模型检查点路径')
parser.add_argument('--num_envs', type=int, default=2,
                   help='并行环境数量')
parser.add_argument('--num_episodes', type=int, default=10,
                   help='测试轨迹数量')
parser.add_argument('--enable_cameras', action='store_true',
                   help='启用相机渲染')
parser.add_argument('--headless', action='store_true',
                   help='无头模式（不显示GUI）')

args_cli = parser.parse_args()

# 配置AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入环境配置
from rosorin_env_cfg import ROSOrinEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv

# 导入BC模型架构（与train_bc_simple.py保持一致）
class BCPolicy(torch.nn.Module):
    """BC策略网络"""
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, obs):
        features = self.encoder(obs)
        action = self.policy_head(features)
        return action


def visualize_bc_policy(checkpoint_path, num_envs=2, num_episodes=10):
    """可视化BC策略"""
    
    print(f"\n{'='*80}")
    print(f"  BC模型策略可视化")
    print(f"{'='*80}")
    print(f"  模型路径: {checkpoint_path}")
    print(f"  环境数量: {num_envs}")
    print(f"  测试轨迹: {num_episodes}")
    print(f"{'='*80}\n")
    
    # 创建环境
    print("创建Isaac Sim环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.action_manager.total_action_dim
    
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}\n")
    
    # 加载BC模型
    print(f"加载BC模型: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bc_policy = BCPolicy(obs_dim, action_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bc_policy.load_state_dict(checkpoint['model_state_dict'])
    bc_policy.eval()
    
    print(f"  ✓ BC模型已加载")
    print(f"  - 训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - 训练损失: {checkpoint.get('train_loss', 'N/A'):.4f}\n")
    
    # 运行可视化
    print("开始可视化BC策略...\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    current_episode_reward = torch.zeros(num_envs, device=device)
    current_episode_length = torch.zeros(num_envs, device=device, dtype=torch.int32)
    
    step_count = 0
    
    while episode_count < num_episodes:
        # BC策略预测动作
        with torch.no_grad():
            action = bc_policy(obs)
        
        # 执行动作
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]
        
        current_episode_reward += reward
        current_episode_length += 1
        step_count += 1
        
        # 检查是否有episode结束
        done = terminated | truncated
        
        if done.any():
            for env_idx in range(num_envs):
                if done[env_idx] and episode_count < num_episodes:
                    ep_reward = current_episode_reward[env_idx].item()
                    ep_length = current_episode_length[env_idx].item()
                    
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    episode_count += 1
                    
                    # 判断是否成功（可以根据reward或info判断）
                    is_success = ep_reward > 50  # 假设reward>50为成功
                    if is_success:
                        success_count += 1
                    
                    status = "✅ 成功" if is_success else "❌ 失败"
                    print(f"  轨迹 {episode_count:2d}/{num_episodes}: "
                          f"奖励={ep_reward:7.2f}, 长度={ep_length:3d}, {status}")
                    
                    # 重置计数器
                    current_episode_reward[env_idx] = 0
                    current_episode_length[env_idx] = 0
        
        # 可视化延迟
        if not args_cli.headless:
            time.sleep(0.02)  # 20ms延迟，方便观察
    
    # 统计结果
    print(f"\n{'='*80}")
    print(f"  BC模型评估结果")
    print(f"{'='*80}")
    print(f"  总轨迹数: {num_episodes}")
    print(f"  成功率: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  奖励范围: [{min(episode_rewards):.2f}, {max(episode_rewards):.2f}]")
    print(f"{'='*80}\n")
    
    # 关闭环境
    env.close()


def main():
    checkpoint_path = Path(args_cli.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ 错误: 未找到模型文件 {checkpoint_path}")
        simulation_app.close()
        return
    
    visualize_bc_policy(
        checkpoint_path=str(checkpoint_path),
        num_envs=args_cli.num_envs,
        num_episodes=args_cli.num_episodes
    )
    
    simulation_app.close()


if __name__ == "__main__":
    main()
