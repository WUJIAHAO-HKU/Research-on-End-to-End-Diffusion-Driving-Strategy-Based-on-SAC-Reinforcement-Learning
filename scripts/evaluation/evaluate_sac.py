#!/usr/bin/env python3
"""
SAC模型评估脚本

在Isaac Lab环境中评估训练好的SAC-Diffusion策略
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="SAC模型评估")
parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=20, help="评估episode数量")
parser.add_argument("--max_steps", type=int, default=500, help="每个episode最大步数")
parser.add_argument("--deterministic", action="store_true", help="是否使用确定性策略")
parser.add_argument("--render", action="store_true", help="是否渲染")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 如果不渲染则headless
if not args.render:
    args.headless = True
    args.enable_cameras = True

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入Isaac Lab
from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 定义网络结构（与训练时一致）
# ============================================================================

class SimpleDiffusionPolicy(nn.Module):
    """简化的扩散策略（用于RL微调）"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 观测编码器（降低hidden_dim以节省显存）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
        )
        
        # 动作头（简化为直接预测，不使用完整扩散）
        self.action_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 归一化到[-1, 1]
        )
    
    def forward(self, obs):
        latent = self.obs_encoder(obs)
        action = self.action_head(latent)
        return action
    
    def get_action(self, obs, deterministic=False):
        """获取动作（兼容SAC接口）"""
        with torch.no_grad():
            action = self.forward(obs)
            if not deterministic:
                # 添加探索噪声
                noise = torch.randn_like(action) * 0.1
                action = action + noise
                action = torch.clamp(action, -1, 1)
        return action


# ============================================================================
# 评估函数
# ============================================================================

def evaluate():
    """评估SAC策略"""
    
    print("="*80)
    print("  SAC策略评估")
    print("="*80)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  并行环境: {args.num_envs}")
    print(f"  评估Episodes: {args.num_episodes}")
    print(f"  确定性策略: {args.deterministic}")
    print("="*80)
    
    # 创建环境
    print("\n创建评估环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取维度
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"观察维度: {obs_dim}, 动作维度: {action_dim}")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载SAC模型: {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 错误: Checkpoint不存在: {checkpoint_path}")
        simulation_app.close()
        sys.exit(1)
    
    # 创建策略网络
    policy = SimpleDiffusionPolicy(obs_dim, action_dim).to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的checkpoint格式
    if 'actor' in checkpoint:
        # SAC checkpoint格式（完整保存了actor模型）
        actor_state = checkpoint['actor']
        policy.load_state_dict(actor_state)
        print(f"  ✓ 已加载SAC Actor权重")
        if 'step' in checkpoint:
            print(f"  训练步数: {checkpoint['step']:,}")
        if 'best_reward' in checkpoint:
            print(f"  最佳奖励: {checkpoint['best_reward']:.2f}")
    elif 'actor_state_dict' in checkpoint:
        # 备用格式
        policy.load_state_dict(checkpoint['actor_state_dict'])
        print(f"  ✓ 已加载SAC Actor权重")
    elif 'model_state_dict' in checkpoint:
        # BC checkpoint格式
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ 已加载BC预训练权重")
    else:
        # 直接是state_dict
        policy.load_state_dict(checkpoint)
        print(f"  ✓ 已加载模型权重")
    
    policy.eval()
    
    # 评估统计
    all_rewards = []
    all_lengths = []
    all_successes = []
    
    # 环境状态
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
    
    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, device=device, dtype=torch.int)
    
    total_episodes = 0
    step = 0
    max_total_steps = args.num_episodes * args.max_steps
    
    print(f"\n{'='*80}")
    print("开始评估...")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=args.num_episodes, desc="评估进度")
    
    while total_episodes < args.num_episodes and step < max_total_steps:
        # 使用策略选择动作
        with torch.no_grad():
            actions = policy.get_action(obs, deterministic=args.deterministic)
        
        # 执行动作
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        next_obs = next_obs_dict["policy"]
        next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        dones = terminated | truncated
        
        # 更新统计
        episode_rewards += rewards
        episode_lengths += 1
        
        # 检查是否有episode结束
        for i in range(args.num_envs):
            if dones[i] and total_episodes < args.num_episodes:
                all_rewards.append(episode_rewards[i].item())
                all_lengths.append(episode_lengths[i].item())
                
                # 判断成功（这里简单用奖励>0作为成功标准，可根据实际任务调整）
                success = episode_rewards[i].item() > 0
                all_successes.append(success)
                
                total_episodes += 1
                pbar.update(1)
                
                # 重置该环境的统计
                episode_rewards[i] = 0
                episode_lengths[i] = 0
        
        obs = next_obs
        step += 1
    
    pbar.close()
    
    # 计算最终统计
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    avg_length = np.mean(all_lengths)
    success_rate = np.mean(all_successes) * 100
    
    print(f"\n{'='*80}")
    print("评估结果")
    print(f"{'='*80}")
    print(f"总Episodes: {len(all_rewards)}")
    print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"奖励范围: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]")
    print(f"平均长度: {avg_length:.1f} 步")
    print(f"成功率: {success_rate:.1f}% ({sum(all_successes)}/{len(all_successes)})")
    print(f"{'='*80}\n")
    
    # 保存评估结果
    result_dir = checkpoint_path.parent.parent / "evaluation_results"
    result_dir.mkdir(exist_ok=True)
    
    result_file = result_dir / f"eval_{checkpoint_path.stem}.txt"
    with open(result_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAC策略评估结果\n")
        f.write("="*80 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"确定性策略: {args.deterministic}\n")
        f.write(f"\n总Episodes: {len(all_rewards)}\n")
        f.write(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"奖励范围: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]\n")
        f.write(f"平均长度: {avg_length:.1f} 步\n")
        f.write(f"成功率: {success_rate:.1f}% ({sum(all_successes)}/{len(all_successes)})\n")
        f.write("\n详细结果:\n")
        for i, (r, l, s) in enumerate(zip(all_rewards, all_lengths, all_successes)):
            f.write(f"Episode {i+1}: Reward={r:.2f}, Length={l}, Success={s}\n")
    
    print(f"✓ 结果已保存到: {result_file}")
    
    # 关闭
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    evaluate()
