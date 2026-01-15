#!/usr/bin/env python3
"""
SAC成功/失败案例分析脚本

在环境中运行SAC策略，记录和可视化成功与失败的轨迹
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="SAC成功/失败案例分析")
parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=50, help="分析episode数量")
parser.add_argument("--max_steps", type=int, default=500, help="每个episode最大步数")
parser.add_argument("--success_threshold", type=float, default=0.0, 
                    help="成功判定阈值（奖励>threshold为成功）")
parser.add_argument("--output_dir", type=str, default="experiments/sac_analysis", 
                    help="分析结果输出目录")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 设置headless
args.headless = True
args.enable_cameras = True  # 启用相机（即使headless也需要）

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入Isaac Lab
from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 网络定义（与evaluate_sac.py相同）
# ============================================================================

class SimpleDiffusionPolicy(nn.Module):
    """简化的扩散策略"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        latent = self.obs_encoder(obs)
        action = self.action_head(latent)
        return action
    
    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.forward(obs)
            if not deterministic:
                noise = torch.randn_like(action) * 0.1
                action = action + noise
                action = torch.clamp(action, -1, 1)
        return action


# ============================================================================
# 案例记录与分析
# ============================================================================

class EpisodeRecorder:
    """记录episode轨迹数据"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.rewards = []
        self.actions = []
        self.episode_length = 0
        self.total_reward = 0
    
    def step(self, reward, action):
        self.rewards.append(reward)
        self.actions.append(action.cpu().numpy() if isinstance(action, torch.Tensor) else action)
        self.episode_length += 1
        self.total_reward += reward
    
    def get_summary(self):
        return {
            'total_reward': self.total_reward,
            'episode_length': self.episode_length,
            'avg_reward_per_step': self.total_reward / max(self.episode_length, 1),
            'rewards': self.rewards,
            'actions': self.actions
        }


def analyze_cases():
    """分析成功和失败案例"""
    
    print("="*80)
    print("  SAC成功/失败案例分析")
    print("="*80)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  并行环境: {args.num_envs}")
    print(f"  分析Episodes: {args.num_episodes}")
    print(f"  成功阈值: {args.success_threshold}")
    print("="*80)
    
    # 创建环境
    print("\n创建分析环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取维度
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"观察维度: {obs_dim}, 动作维度: {action_dim}")
    
    # 加载模型
    print(f"\n加载SAC模型...")
    checkpoint_path = Path(args.checkpoint)
    policy = SimpleDiffusionPolicy(obs_dim, action_dim).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'actor' in checkpoint:
        policy.load_state_dict(checkpoint['actor'])
        print(f"  ✓ 已加载SAC Actor权重")
    else:
        print(f"  ❌ Checkpoint格式错误")
        simulation_app.close()
        return
    
    policy.eval()
    
    # 记录器
    success_cases = []
    failure_cases = []
    recorders = [EpisodeRecorder() for _ in range(args.num_envs)]
    
    # 环境状态
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
    
    total_episodes = 0
    step = 0
    max_total_steps = args.num_episodes * args.max_steps
    
    print(f"\n{'='*80}")
    print("开始分析...")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=args.num_episodes, desc="分析进度")
    
    while total_episodes < args.num_episodes and step < max_total_steps:
        # 选择动作（确定性）
        with torch.no_grad():
            actions = policy.get_action(obs, deterministic=True)
        
        # 执行动作
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        next_obs = next_obs_dict["policy"]
        next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        dones = terminated | truncated
        
        # 记录数据
        for i in range(args.num_envs):
            recorders[i].step(rewards[i].item(), actions[i])
            
            if dones[i] and total_episodes < args.num_episodes:
                summary = recorders[i].get_summary()
                
                # 判断成功/失败
                if summary['total_reward'] > args.success_threshold:
                    success_cases.append(summary)
                else:
                    failure_cases.append(summary)
                
                total_episodes += 1
                pbar.update(1)
                recorders[i].reset()
        
        obs = next_obs
        step += 1
    
    pbar.close()
    
    # 分析结果
    print(f"\n{'='*80}")
    print("分析结果")
    print(f"{'='*80}")
    
    num_success = len(success_cases)
    num_failure = len(failure_cases)
    total = num_success + num_failure
    
    print(f"总Episodes: {total}")
    print(f"成功: {num_success} ({num_success/total*100:.1f}%)")
    print(f"失败: {num_failure} ({num_failure/total*100:.1f}%)")
    
    # 成功案例统计
    if success_cases:
        success_rewards = [c['total_reward'] for c in success_cases]
        success_lengths = [c['episode_length'] for c in success_cases]
        print(f"\n成功案例统计:")
        print(f"  平均奖励: {np.mean(success_rewards):.2f} ± {np.std(success_rewards):.2f}")
        print(f"  奖励范围: [{min(success_rewards):.2f}, {max(success_rewards):.2f}]")
        print(f"  平均长度: {np.mean(success_lengths):.1f} ± {np.std(success_lengths):.1f} 步")
    
    # 失败案例统计
    if failure_cases:
        failure_rewards = [c['total_reward'] for c in failure_cases]
        failure_lengths = [c['episode_length'] for c in failure_cases]
        print(f"\n失败案例统计:")
        print(f"  平均奖励: {np.mean(failure_rewards):.2f} ± {np.std(failure_rewards):.2f}")
        print(f"  奖励范围: [{min(failure_rewards):.2f}, {max(failure_rewards):.2f}]")
        print(f"  平均长度: {np.mean(failure_lengths):.1f} ± {np.std(failure_lengths):.1f} 步")
    
    print(f"{'='*80}\n")
    
    # 可视化对比
    visualize_comparison(success_cases, failure_cases, args.output_dir)
    
    # 保存详细数据
    save_analysis_results(success_cases, failure_cases, args.output_dir)
    
    env.close()
    simulation_app.close()


def visualize_comparison(success_cases, failure_cases, output_dir):
    """可视化成功与失败案例对比"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 奖励分布对比
    ax = axes[0, 0]
    if success_cases:
        success_rewards = [c['total_reward'] for c in success_cases]
        ax.hist(success_rewards, bins=20, alpha=0.6, color='green', label='Success', edgecolor='black')
    if failure_cases:
        failure_rewards = [c['total_reward'] for c in failure_cases]
        ax.hist(failure_rewards, bins=20, alpha=0.6, color='red', label='Failure', edgecolor='black')
    ax.set_xlabel('Total Episode Reward', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Reward Distribution Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode长度对比
    ax = axes[0, 1]
    if success_cases:
        success_lengths = [c['episode_length'] for c in success_cases]
        ax.hist(success_lengths, bins=20, alpha=0.6, color='green', label='Success', edgecolor='black')
    if failure_cases:
        failure_lengths = [c['episode_length'] for c in failure_cases]
        ax.hist(failure_lengths, bins=20, alpha=0.6, color='red', label='Failure', edgecolor='black')
    ax.set_xlabel('Episode Length (steps)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Episode Length Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 典型成功轨迹奖励曲线
    ax = axes[1, 0]
    if success_cases:
        # 选择3个代表性成功案例
        top_cases = sorted(success_cases, key=lambda x: x['total_reward'], reverse=True)[:3]
        for i, case in enumerate(top_cases):
            ax.plot(case['rewards'], alpha=0.7, label=f'Success {i+1} (R={case["total_reward"]:.1f})')
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title('Typical Success Trajectories', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 典型失败轨迹奖励曲线
    ax = axes[1, 1]
    if failure_cases:
        # 选择3个代表性失败案例
        worst_cases = sorted(failure_cases, key=lambda x: x['total_reward'])[:3]
        for i, case in enumerate(worst_cases):
            ax.plot(case['rewards'], alpha=0.7, label=f'Failure {i+1} (R={case["total_reward"]:.1f})')
    ax.set_xlabel('Step', fontweight='bold')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title('Typical Failure Trajectories', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = output_dir / 'success_failure_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比分析图已保存到: {save_path}")
    
    plt.close()


def save_analysis_results(success_cases, failure_cases, output_dir):
    """保存详细分析结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON格式摘要
    summary = {
        'total_episodes': len(success_cases) + len(failure_cases),
        'num_success': len(success_cases),
        'num_failure': len(failure_cases),
        'success_rate': len(success_cases) / (len(success_cases) + len(failure_cases)) * 100,
        'success_stats': {
            'avg_reward': np.mean([c['total_reward'] for c in success_cases]) if success_cases else 0,
            'std_reward': np.std([c['total_reward'] for c in success_cases]) if success_cases else 0,
            'avg_length': np.mean([c['episode_length'] for c in success_cases]) if success_cases else 0,
        },
        'failure_stats': {
            'avg_reward': np.mean([c['total_reward'] for c in failure_cases]) if failure_cases else 0,
            'std_reward': np.std([c['total_reward'] for c in failure_cases]) if failure_cases else 0,
            'avg_length': np.mean([c['episode_length'] for c in failure_cases]) if failure_cases else 0,
        }
    }
    
    json_path = output_dir / 'analysis_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ 分析摘要已保存到: {json_path}")
    
    # 保存文本报告
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAC成功/失败案例分析报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"总Episodes: {summary['total_episodes']}\n")
        f.write(f"成功率: {summary['success_rate']:.1f}%\n\n")
        
        f.write("成功案例统计:\n")
        f.write(f"  数量: {summary['num_success']}\n")
        f.write(f"  平均奖励: {summary['success_stats']['avg_reward']:.2f} ± {summary['success_stats']['std_reward']:.2f}\n")
        f.write(f"  平均长度: {summary['success_stats']['avg_length']:.1f} 步\n\n")
        
        f.write("失败案例统计:\n")
        f.write(f"  数量: {summary['num_failure']}\n")
        f.write(f"  平均奖励: {summary['failure_stats']['avg_reward']:.2f} ± {summary['failure_stats']['std_reward']:.2f}\n")
        f.write(f"  平均长度: {summary['failure_stats']['avg_length']:.1f} 步\n\n")
        
        # 关键发现
        f.write("关键发现:\n")
        if success_cases and failure_cases:
            reward_gap = summary['success_stats']['avg_reward'] - summary['failure_stats']['avg_reward']
            f.write(f"  - 成功与失败奖励差距: {reward_gap:.2f}\n")
            
            length_ratio = summary['success_stats']['avg_length'] / max(summary['failure_stats']['avg_length'], 1)
            if length_ratio > 1.2:
                f.write(f"  - 成功案例平均更长 ({length_ratio:.1f}x)，说明能持续执行任务\n")
            elif length_ratio < 0.8:
                f.write(f"  - 失败案例平均更长，可能陷入困境\n")
        
        if summary['success_rate'] < 30:
            f.write(f"  ⚠️ 成功率较低 ({summary['success_rate']:.1f}%)，建议:\n")
            f.write(f"     1. 延长训练时间\n")
            f.write(f"     2. 调整奖励函数\n")
            f.write(f"     3. 检查BC预训练质量\n")
        elif summary['success_rate'] > 70:
            f.write(f"  ✅ 成功率良好 ({summary['success_rate']:.1f}%)，策略鲁棒\n")
    
    print(f"✓ 详细报告已保存到: {report_path}")


if __name__ == "__main__":
    analyze_cases()
