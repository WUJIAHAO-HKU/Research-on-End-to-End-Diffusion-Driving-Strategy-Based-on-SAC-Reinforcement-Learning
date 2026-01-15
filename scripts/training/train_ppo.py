#!/usr/bin/env python3
"""
PPO训练脚本 (Proximal Policy Optimization)

该模块实现PPO算法的训练流程，使用PPO专用的奖励配置。

主要功能:
- 标准PPO算法训练
- 观察值归一化（CPU统计，GPU小批量归一化）
- 奖励组件提取和日志记录
- 模型检查点保存

使用方法:
  ./isaaclab_runner.sh scripts/training/train_ppo.py --num_envs 8 --total_steps 100000
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import json
import sys
import pickle

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="PPO训练")
parser.add_argument("--num_envs", type=int, default=8, help="并行环境数量（PPO适合更多环境）")
parser.add_argument("--total_steps", type=int, default=100000, help="总训练步数")
parser.add_argument("--pretrain_checkpoint", type=str, default=None, help="BC预训练模型路径（用于消融实验）")
parser.add_argument("--output_dir", type=str, default="experiments/baselines/ppo", help="输出目录")
parser.add_argument("--batch_size", type=int, default=512, help="批次大小")
parser.add_argument("--n_steps", type=int, default=1024, help="每次rollout步数（降低到1024节省显存）")
parser.add_argument("--n_epochs", type=int, default=10, help="每次更新的epoch数")
parser.add_argument("--save_freq", type=int, default=10000, help="保存频率")
parser.add_argument("--log_freq", type=int, default=100, help="日志频率")
parser.add_argument("--lr", type=float, default=3e-5, help="学习率（降低到3e-5避免NaN）")
parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip范围")
parser.add_argument("--vf_coef", type=float, default=0.5, help="价值函数系数")
parser.add_argument("--ent_coef", type=float, default=0.01, help="熵系数")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 设置headless和相机
args.headless = True
args.enable_cameras = True

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入Isaac Lab和自定义模块
# 添加父目录到path以导入env_factory
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env_factory import create_ppo_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


print("="*80)
print("  PPO驾驶策略训练")
print("="*80)
print(f"  并行环境: {args.num_envs}")
print(f"  总步数: {args.total_steps:,}")
print(f"  Rollout步数: {args.n_steps}")
print(f"  批次大小: {args.batch_size}")
if args.pretrain_checkpoint:
    print(f"  BC预训练: {args.pretrain_checkpoint}")
print("="*80)


# ============================================================================
# PPO网络定义
# ============================================================================

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class _ImageEncoder(nn.Module):
    """轻量CNN编码器：把(H,W,C)压成固定维度embedding。"""

    def __init__(self, in_channels: int, height: int, width: int, embed_dim: int = 64):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.in_channels = int(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 动态推断flatten维度，避免硬编码
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, self.height, self.width)
            out = self.conv(dummy)
            flat_dim = int(out.reshape(1, -1).shape[1])

        self.head = nn.Sequential(
            nn.Linear(flat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


class ActorCritic(nn.Module):
    """Actor-Critic网络（PPO）- Depth/RGB CNN encoder + 低维融合"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        low_dim: int,
        rgb_shape: tuple[int, int, int],
        depth_shape: tuple[int, int, int],
        hidden_dims=(256, 256),
        rgb_embed_dim: int = 64,
        depth_embed_dim: int = 64,
    ):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.low_dim = int(low_dim)

        rgb_h, rgb_w, rgb_c = rgb_shape
        dep_h, dep_w, dep_c = depth_shape
        self.rgb_h, self.rgb_w, self.rgb_c = int(rgb_h), int(rgb_w), int(rgb_c)
        self.dep_h, self.dep_w, self.dep_c = int(dep_h), int(dep_w), int(dep_c)

        self.rgb_dim = self.rgb_h * self.rgb_w * self.rgb_c
        self.depth_dim = self.dep_h * self.dep_w * self.dep_c

        if self.low_dim + self.rgb_dim + self.depth_dim != self.obs_dim:
            raise ValueError(
                f"obs split mismatch: low({self.low_dim}) + rgb({self.rgb_dim}) + depth({self.depth_dim}) != obs_dim({self.obs_dim})"
            )

        # 编码器（至少depth做CNN编码；这里也同时对RGB做轻量CNN，避免丢信息）
        self.rgb_encoder = _ImageEncoder(self.rgb_c, self.rgb_h, self.rgb_w, embed_dim=rgb_embed_dim)
        self.depth_encoder = _ImageEncoder(self.dep_c, self.dep_h, self.dep_w, embed_dim=depth_embed_dim)

        fusion_in_dim = self.low_dim + rgb_embed_dim + depth_embed_dim

        layers = []
        in_dim = fusion_in_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Actor头（策略）
        self.actor_mean = nn.Linear(in_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic头（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs):
        """前向传播"""
        low = obs[:, : self.low_dim]
        rgb_flat = obs[:, self.low_dim : self.low_dim + self.rgb_dim]
        depth_flat = obs[:, self.low_dim + self.rgb_dim :]

        rgb = rgb_flat.reshape(obs.shape[0], self.rgb_h, self.rgb_w, self.rgb_c).permute(0, 3, 1, 2).contiguous()
        depth = depth_flat.reshape(obs.shape[0], self.dep_h, self.dep_w, self.dep_c).permute(0, 3, 1, 2).contiguous()

        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)

        fused = torch.cat([low, rgb_feat, depth_feat], dim=-1)
        features = self.shared(fused)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_log_std = torch.clamp(action_log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        action_std = action_log_std.exp()
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """获取动作"""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = torch.tanh(action_mean)
        else:
            dist = Normal(action_mean, action_std)
            action_pre_tanh = dist.sample()
            action = torch.tanh(action_pre_tanh)
        
        return action, value
    
    def evaluate_actions(self, obs, actions):
        """评估动作（用于更新）"""
        action_mean, action_std, value = self.forward(obs)
        
        # 数值稳定性检查
        if torch.isnan(action_mean).any() or torch.isinf(action_mean).any():
            print(f"⚠️ 警告: action_mean包含NaN/Inf")
            action_mean = torch.nan_to_num(action_mean, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(action_std).any() or torch.isinf(action_std).any():
            print(f"⚠️ 警告: action_std包含NaN/Inf")
            action_std = torch.nan_to_num(action_std, nan=0.1, posinf=1.0, neginf=0.01)
        
        # 限制action_std最小值避免数值问题
        action_std = torch.clamp(action_std, min=1e-3, max=2.0)
        
        dist = Normal(action_mean, action_std)
        
        # 逆tanh
        actions_pre_tanh = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        
        # Log概率
        log_prob = dist.log_prob(actions_pre_tanh)
        # Tanh修正
        log_prob = log_prob - torch.log(1 - actions.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # 熵
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_prob, entropy


class RolloutBuffer:
    """PPO Rollout Buffer"""
    
    def __init__(self, buffer_size, obs_dim, action_dim, num_envs, device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.device = device
        
        self.reset()
    
    def reset(self):
        """重置buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.pos = 0
    
    def add(self, obs, action, reward, value, log_prob, done):
        """添加经验"""
        self.observations.append(obs.cpu())
        self.actions.append(action.cpu())
        self.rewards.append(reward.cpu())
        self.values.append(value.cpu())
        self.log_probs.append(log_prob.cpu())
        self.dones.append(done.cpu())
        self.pos += 1
    
    def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
        """计算GAE优势和回报"""
        # 转换为tensor
        rewards = torch.stack(self.rewards)  # [n_steps, n_envs, 1]
        values = torch.stack(self.values)    # [n_steps, n_envs, 1]
        dones = torch.stack(self.dones)      # [n_steps, n_envs, 1]
        # 确保dones是float mask（0.0/1.0），避免bool参与算术导致RuntimeError
        if dones.dtype == torch.bool:
            dones = dones.float()
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_values = last_values.cpu()
            else:
                next_values = values[t + 1]
            
            delta = rewards[t] + gamma * next_values * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lam
        
        returns = advantages + values
        
        return returns, advantages
    
    def get(self, returns, advantages):
        """获取所有数据"""
        # 展平batch
        observations = torch.stack(self.observations).reshape(-1, self.obs_dim)
        actions = torch.stack(self.actions).reshape(-1, self.action_dim)
        values = torch.stack(self.values).reshape(-1, 1)
        log_probs = torch.stack(self.log_probs).reshape(-1, 1)
        returns = returns.reshape(-1, 1)
        advantages = advantages.reshape(-1, 1)
        
        return observations, actions, values, log_probs, returns, advantages


class PPOAgent:
    """PPO智能体"""
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        low_dim=None,
        rgb_dim=None,
        depth_dim=None,
        rgb_shape=None,
        depth_shape=None,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.3,  # 降低到0.3
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # 观测结构（用于只归一化低维部分，避免把像素一起做running mean/var导致信号被扭曲）
        self.low_dim = int(low_dim) if low_dim is not None else int(obs_dim)
        self.rgb_dim = int(rgb_dim) if rgb_dim is not None else 0
        self.depth_dim = int(depth_dim) if depth_dim is not None else 0
        
        # 🆕 观测归一化统计量（保持在CPU上节省显存，每次使用时移到GPU）
        self.obs_mean = torch.zeros(self.low_dim)
        self.obs_var = torch.ones(self.low_dim)
        self.obs_count = 1e-4
        
        # Actor-Critic网络
        if rgb_shape is None or depth_shape is None:
            raise ValueError("rgb_shape/depth_shape is required for CNN encoder policy")

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            low_dim=self.low_dim,
            rgb_shape=rgb_shape,
            depth_shape=depth_shape,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def normalize_obs(self, obs):
        """归一化观测（在线更新统计量）"""
        if self.rgb_dim + self.depth_dim == 0 or self.low_dim == obs.shape[-1]:
            # 兼容老路径：没有图像切分信息时，对全维做归一化
            obs_cpu = obs.cpu() if obs.is_cuda else obs

            batch_mean = obs_cpu.mean(dim=0)
            batch_var = obs_cpu.var(dim=0)
            batch_count = obs_cpu.shape[0]

            delta = batch_mean - self.obs_mean
            total_count = self.obs_count + batch_count

            self.obs_mean = self.obs_mean + delta * batch_count / total_count
            self.obs_var = (
                (self.obs_var * self.obs_count + batch_var * batch_count) / total_count +
                (delta ** 2) * self.obs_count * batch_count / (total_count ** 2)
            )
            self.obs_count = total_count

            obs_mean = self.obs_mean.to(obs.device)
            obs_std = torch.sqrt(self.obs_var).to(obs.device)

            obs_normalized = (obs - obs_mean) / (obs_std + 1e-8)
            obs_normalized = torch.clamp(obs_normalized, -10.0, 10.0)
            return obs_normalized

        # 新路径：只对低维状态做running mean/var；图像保持原尺度（仅做clip/scale）
        low = obs[:, : self.low_dim]
        rgb = obs[:, self.low_dim : self.low_dim + self.rgb_dim]
        depth = obs[:, self.low_dim + self.rgb_dim :]

        low_cpu = low.cpu() if low.is_cuda else low
        
        batch_mean = low_cpu.mean(dim=0)
        batch_var = low_cpu.var(dim=0)
        batch_count = low_cpu.shape[0]
        
        # 增量更新
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        
        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        self.obs_var = (
            (self.obs_var * self.obs_count + batch_var * batch_count) / total_count +
            (delta ** 2) * self.obs_count * batch_count / (total_count ** 2)
        )
        self.obs_count = total_count
        
        # 归一化（将统计量临时移到obs所在设备）
        obs_mean = self.obs_mean.to(obs.device)
        obs_std = torch.sqrt(self.obs_var).to(obs.device)
        
        low_norm = (low - obs_mean) / (obs_std + 1e-8)
        low_norm = torch.clamp(low_norm, -10.0, 10.0)

        # RGB: 通常是0..255（或已归一化到0..1），做一个保守的自适应scale
        if rgb.numel() > 0:
            rgb_max = rgb.detach().max().item()
            if rgb_max > 1.5:
                rgb = rgb / 255.0
            rgb = torch.clamp(rgb, 0.0, 1.0)

        # Depth: 裁剪到[0,10]并归一化到[0,1]
        if depth.numel() > 0:
            depth = torch.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=0.0)
            depth = torch.clamp(depth, 0.0, 10.0) / 10.0

        return torch.cat([low_norm, rgb, depth], dim=-1)
    
    def select_action(self, obs, deterministic=False):
        """选择动作"""
        # 🆕 归一化观测
        obs = self.normalize_obs(obs)
        
        with torch.no_grad():
            action, value = self.policy.get_action(obs, deterministic)
            
            if not deterministic:
                # 计算log_prob用于训练
                action_mean, action_std, _ = self.policy(obs)
                dist = Normal(action_mean, action_std)
                action_pre_tanh = torch.atanh(torch.clamp(action, -0.999, 0.999))
                log_prob = dist.log_prob(action_pre_tanh)
                log_prob = log_prob - torch.log(1 - action.pow(2) + EPSILON)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = None
        
        return action, value, log_prob
    
    def update(self, rollout_buffer, n_epochs, batch_size):
        """更新策略"""
        # 获取最后的value用于GAE
        with torch.no_grad():
            last_obs = rollout_buffer.observations[-1].to(self.device)
            last_obs = self.normalize_obs(last_obs)  # 🆕 归一化
            _, _, last_values = self.policy(last_obs)
        
        # 计算returns和advantages
        returns, advantages = rollout_buffer.compute_returns_and_advantages(
            last_values, self.gamma, self.gae_lambda
        )
        
        # 获取数据
        observations, actions, old_values, old_log_probs, returns, advantages = rollout_buffer.get(returns, advantages)
        
        # 🆕 分批归一化观测（避免显存溢出）
        # 只在CPU上更新低维统计量；图像不参与running mean/var
        obs_cpu = observations.cpu() if observations.is_cuda else observations
        low_cpu = obs_cpu[:, : self.low_dim]
        batch_mean = low_cpu.mean(dim=0)
        batch_var = low_cpu.var(dim=0)
        batch_count = low_cpu.shape[0]
        
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        self.obs_var = (
            (self.obs_var * self.obs_count + batch_var * batch_count) / total_count +
            (delta ** 2) * self.obs_count * batch_count / (total_count ** 2)
        )
        self.obs_count = total_count
        
        # 归一化将在每个mini-batch时进行，避免一次性处理所有数据
        
        # 标准化advantages（数值稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -10.0, 10.0)  # 限制范围
        
        # 准备数据
        dataset_size = observations.shape[0]
        
        # 训练统计
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            # 随机打乱
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = observations[batch_indices].to(self.device)
                
                # 🆕 在mini-batch层面归一化（避免显存溢出）
                # NOTE: 仅归一化低维状态；图像保持原尺度（在normalize_obs中做clip/scale）
                batch_obs = self.normalize_obs(batch_obs)
                
                batch_actions = actions[batch_indices].to(self.device)
                batch_old_log_probs = old_log_probs[batch_indices].to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                batch_returns = returns[batch_indices].to(self.device)
                
                # 数值稳定性：检查输入
                if torch.isnan(batch_obs).any():
                    print(f"⚠️ 警告: batch_obs包含NaN，跳过此batch")
                    continue
                
                # 评估当前策略
                values, log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                
                # 检查输出
                if torch.isnan(values).any() or torch.isnan(log_probs).any():
                    print(f"⚠️ 警告: values或log_probs包含NaN，跳过此batch")
                    continue
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 数值稳定性检查：检测参数是否有NaN
                has_nan = False
                for name, param in self.policy.named_parameters():
                    if torch.isnan(param).any():
                        print(f"⚠️ 严重错误: 参数 {name} 包含NaN！")
                        has_nan = True
                
                if has_nan:
                    print("🛑 训练中止：参数出现NaN，请降低学习率或检查数据")
                    raise ValueError("参数包含NaN，训练失败")
                
                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # 清空buffer
        rollout_buffer.reset()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    def load_bc_pretrain(self, checkpoint_path):
        """从BC预训练加载actor参数（带numpy兼容性处理）"""
        print(f"\n加载BC预训练模型: {checkpoint_path}")
        
        # 修复numpy兼容性
        if not hasattr(np, '_core'):
            import numpy.core as _core
            sys.modules['numpy._core'] = _core
            sys.modules['numpy._core.multiarray'] = _core.multiarray
            sys.modules['numpy._core.umath'] = _core.umath
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e):
                import pickle
                class NumpyCompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith('numpy._core'):
                            module = module.replace('numpy._core', 'numpy.core')
                        return super().find_class(module, name)
                
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = NumpyCompatUnpickler(f).load()
            else:
                raise
        
        try:
            # 尝试加载匹配的权重
            self.policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("  ✓ BC预训练权重已加载")
        except Exception as e:
            print(f"  ⚠ 无法加载BC权重: {e}")
            print("  → 从头开始训练")


# ============================================================================
# 主训练循环
# ============================================================================

def extract_reward_components(env):
    """
    从Isaac Lab环境的reward_manager中提取各个奖励项的值
    
    Returns:
        dict: 奖励项名称 -> 平均值的字典
    """
    reward_dict = {}
    
    # 通过reward_manager获取各个奖励项
    try:
        # ManagerBasedRLEnv没有unwrapped，直接访问
        if hasattr(env, 'reward_manager'):
            manager = env.reward_manager
            # 使用_episode_sums而不是_term_buffers
            if hasattr(manager, '_episode_sums'):
                for term_name, term_buffer in manager._episode_sums.items():
                    if isinstance(term_buffer, torch.Tensor):
                        # 取当前步的值（不是episode累积和）
                        reward_dict[term_name] = term_buffer.mean().item()
    except Exception as e:
        print(f"⚠️ extract_reward_components错误: {e}")
    
    return reward_dict


def train():
    """主训练函数"""

    # 成功判定阈值（与环境终止条件 goal_reached_termination 保持一致）
    # 目前在 scripts/rosorin_env_cfg.py / configs/mdp/rosorin_mdp.py 中默认也是 0.5m。
    success_distance_threshold = 0.5
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 根据是否使用预训练创建不同的目录
    if args.pretrain_checkpoint:
        run_dir = output_dir / f"ppo_with_bc_{timestamp}"
    else:
        run_dir = output_dir / f"ppo_scratch_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 保存配置
    config = vars(args)
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # 创建环境
    print("\n创建训练环境...")
    print("  使用PPO专用奖励配置...")
    env_cfg = create_ppo_env_cfg(num_envs=args.num_envs)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取维度
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    print(f"  观察维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")

    # 解析相机形状，并据此切分obs为：低维状态 + RGB(flat) + Depth(flat)
    try:
        rgb_out = env.scene.sensors["camera"].data.output["rgb"]
        depth_out = env.scene.sensors["camera"].data.output["distance_to_image_plane"]
        rgb_h, rgb_w, rgb_c = int(rgb_out.shape[1]), int(rgb_out.shape[2]), int(rgb_out.shape[3])
        if depth_out.ndim == 4:
            dep_h, dep_w, dep_c = int(depth_out.shape[1]), int(depth_out.shape[2]), int(depth_out.shape[3])
        else:
            dep_h, dep_w, dep_c = int(depth_out.shape[1]), int(depth_out.shape[2]), 1
        rgb_dim = rgb_h * rgb_w * rgb_c
        depth_dim = dep_h * dep_w * dep_c
        low_dim = int(obs_dim - rgb_dim - depth_dim)
        if low_dim <= 0:
            raise ValueError(f"invalid low_dim={low_dim} (obs_dim={obs_dim}, rgb_dim={rgb_dim}, depth_dim={depth_dim})")
    except Exception as e:
        raise RuntimeError(f"Failed to infer camera shapes for CNN policy: {e}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建PPO智能体
    print("\n初始化PPO智能体...")
    agent = PPOAgent(
        obs_dim,
        action_dim,
        device,
        low_dim=low_dim,
        rgb_dim=rgb_dim,
        depth_dim=depth_dim,
        rgb_shape=(rgb_h, rgb_w, rgb_c),
        depth_shape=(dep_h, dep_w, dep_c),
        lr=args.lr,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
    )
    
    # ⚠️ BC预训练与PPO不兼容：BC输出tanh后的动作，PPO需要原始高斯动作
    # 强制从头训练以避免NaN崩溃
    if args.pretrain_checkpoint:
        print(f"\n⚠️ 警告: BC预训练与PPO不兼容（动作分布不同），将从头训练")
        print("  BC输出: tanh后的动作 ∈ [-1,1]")
        print("  PPO需要: 高斯分布原始动作 ∈ ℝ")
        print("  → 从头训练PPO，不加载BC权重\n")
    # if args.pretrain_checkpoint and Path(args.pretrain_checkpoint).exists():
    #     agent.load_bc_pretrain(args.pretrain_checkpoint)
    
    # 🆕 学习率调度器（线性衰减，避免后期不稳定）
    initial_lr = args.lr
    def lr_schedule(step):
        """线性衰减到初始学习率的10%"""
        progress = step / args.total_steps
        return max(0.1, 1.0 - 0.9 * progress)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(agent.optimizer, lr_schedule)
    
    # 创建Rollout Buffer
    print("\n创建Rollout Buffer...")
    rollout_buffer = RolloutBuffer(
        buffer_size=args.n_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_envs=args.num_envs,
        device=device
    )
    
    # 训练统计
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_successes = []  # 记录成功/失败
    best_moving_avg_reward = 0.0
    max_episode_reward = -float('inf')
    max_episode_length = 0
    
    # 环境状态
    obs = obs_dict["policy"]
    obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
    
    episode_reward = torch.zeros(args.num_envs, device=device)
    episode_length = torch.zeros(args.num_envs, device=device, dtype=torch.int)
    
    # 奖励项统计
    reward_components = {
        'progress': [],
        'goal_reached': [],
        'velocity': [],
        'orientation': [],
        'obstacle_avoidance': [],  # 🆕 避障
        'smooth_action': [],
        'collision': [],
        'stability': [],
        'height': [],
    }
    
    # 日志数据
    log_data = {
        'steps': [],
        'rewards': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'reward_components': [],
        # Debug metric from depth camera (set inside obstacle_avoidance_penalty)
        'front_min_depth': [],
        # Navigation diagnostics
        'goal_distance_mean': [],
        'vel_toward_goal_mean': [],
    }
    
    print(f"\n{'='*80}")
    if args.pretrain_checkpoint:
        print("开始训练PPO (使用BC预训练)...")
    else:
        print("开始训练PPO (从头开始)...")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=args.total_steps, desc="训练进度")
    
    while total_steps < args.total_steps:
        # Rollout阶段
        for _ in range(args.n_steps):
            # 选择动作
            with torch.no_grad():
                actions, values, log_probs = agent.select_action(obs, deterministic=False)
            
            # 执行动作
            next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            next_obs = next_obs_dict["policy"]
            
            # ⚠️ 数值稳定性：检查观测和奖励
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=100.0, neginf=-100.0)
            rewards = torch.clamp(rewards, min=-100.0, max=200.0)
            
            # 检测是否有异常奖励
            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                print(f"⚠️ 警告: 第{total_steps}步奖励包含NaN/Inf，已修复")
                rewards = torch.nan_to_num(rewards, nan=0.0, posinf=100.0, neginf=-100.0)
            

            # 提取当前步的奖励细节（在step之后立即提取）
            current_reward_components = extract_reward_components(env)

            # 🆕 深度相机调试指标：前方中心区域最小深度（米）
            front_min_depth_mean = None
            try:
                if hasattr(env, 'last_front_min_depth') and isinstance(env.last_front_min_depth, torch.Tensor):
                    d = torch.nan_to_num(env.last_front_min_depth.float(), nan=10.0, posinf=10.0, neginf=0.0)
                    front_min_depth_mean = d.mean().item()
            except Exception:
                front_min_depth_mean = None

            # 🆕 导航诊断：到目标距离均值、朝目标速度投影均值
            goal_distance_mean = None
            vel_toward_goal_mean = None
            try:
                if hasattr(env, 'goal_positions'):
                    robot_pos = env.scene.articulations["robot"].data.root_pos_w[:, :2]
                    goal_pos = env.goal_positions[:, :2]
                    to_goal = goal_pos - robot_pos
                    goal_distance = torch.norm(to_goal, dim=-1)
                    goal_distance_mean = goal_distance.mean().item()

                    lin_vel_w = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]
                    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True)
                    to_goal_dir = to_goal / (to_goal_norm + 1e-6)
                    vel_toward_goal = torch.sum(lin_vel_w * to_goal_dir, dim=-1)
                    vel_toward_goal_mean = vel_toward_goal.mean().item()
            except Exception:
                goal_distance_mean = None
                vel_toward_goal_mean = None
            
            # 🆕 调试：每1000步打印一次奖励细节
            if total_steps % 1000 == 0 and total_steps > 0:
                print(f"\n[调试 Step {total_steps}]")
                print(f"  当前步奖励: {rewards.mean().item():.3f} (范围: [{rewards.min().item():.3f}, {rewards.max().item():.3f}])")
                print(f"  Episode: {len(episode_rewards)}个完成, 当前长度: {episode_length.float().mean().item():.0f}")
                if front_min_depth_mean is not None:
                    print(f"  前方最小深度(来自Depth相机): {front_min_depth_mean:.3f} m")
                if current_reward_components:
                    # 计算单步奖励（除以当前episode长度）
                    avg_ep_len = max(episode_length.float().mean().item(), 1)
                    print(f"  单步奖励组件 (累积值/{avg_ep_len:.0f}步):")
                    for key, val in sorted(current_reward_components.items()):
                        step_reward = val / avg_ep_len
                        print(f"    {key:20s}: {step_reward:+.4f}")
                else:
                    print(f"  ⚠️ 奖励组件提取失败")


            
            done_flags = (terminated | truncated)
            dones = done_flags.float().unsqueeze(-1)

            # 严格成功判定：episode结束时若距离目标 < 阈值，则记为success
            success_flags = None
            try:
                if hasattr(env, 'goal_positions'):
                    robot_pos_xy = env.scene.articulations["robot"].data.root_pos_w[:, :2]
                    goal_pos_xy = env.goal_positions[:, :2]
                    goal_dist = torch.norm(robot_pos_xy - goal_pos_xy, dim=-1)
                    success_flags = goal_dist < success_distance_threshold
            except Exception:
                success_flags = None
            
            # 存储到buffer
            rollout_buffer.add(obs, actions, rewards.unsqueeze(-1), values, log_probs, dones)
            
            # 更新统计
            for i in range(args.num_envs):
                episode_reward[i] += rewards[i]
                episode_length[i] += 1

                if bool(done_flags[i].item()):
                    episode_rewards.append(episode_reward[i].item())
                    episode_lengths.append(episode_length[i].item())

                    # 记录全局最大episode信息（便于解释“平均奖励/最佳奖励”口径差异）
                    if episode_reward[i].item() > max_episode_reward:
                        max_episode_reward = episode_reward[i].item()
                    if episode_length[i].item() > max_episode_length:
                        max_episode_length = int(episode_length[i].item())
                    
                    # 保存当前episode的奖励组件（除以episode长度获得平均值）
                    ep_len = episode_length[i].item()
                    for key, value in current_reward_components.items():
                        if key in reward_components and ep_len > 0:
                            reward_components[key].append(value / ep_len)  # 平均单步奖励
                    
                    # 记录成功率（严格：终止时距离 < distance_threshold）
                    if success_flags is not None and bool(success_flags[i].item()):
                        episode_successes.append(1)
                    else:
                        episode_successes.append(0)
                    
                    # 重置episode统计
                    episode_reward[i] = 0
                    episode_length[i] = 0
            
            obs = next_obs
            total_steps += args.num_envs
            pbar.update(args.num_envs)
            
            if total_steps >= args.total_steps:
                break
        
        # 更新策略
        metrics = agent.update(rollout_buffer, args.n_epochs, args.batch_size)
        
        # 🆕 更新学习率
        scheduler.step()
        current_lr = agent.optimizer.param_groups[0]['lr']
        
        # 记录日志
        if total_steps % args.log_freq == 0:
            avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths[-20:]) if episode_lengths else 0.0

            # 当前正在进行的episode统计（跨env平均），用于短跑/episode很少完成时的可解释性
            running_reward_mean = float(episode_reward.mean().item())
            running_length_mean = float(episode_length.float().mean().item())
            
            # 计算平均奖励组件
            comp_str = {}
            for key, values in reward_components.items():
                if values:
                    avg_val = np.mean(values[-20:])
                    comp_str[key[:3]] = f"{avg_val:.2f}"
            
            pbar.set_postfix({
                'reward': f"{avg_reward:.2f}",
                'len': f"{avg_length:.0f}",
                'runR': f"{running_reward_mean:.2f}",
                'runL': f"{running_length_mean:.0f}",
                'policy': f"{metrics['policy_loss']:.3f}",
                'value': f"{metrics['value_loss']:.3f}",
                'lr': f"{current_lr:.2e}",  # 🆕 显示当前学习率
                'dmin': f"{front_min_depth_mean:.2f}" if front_min_depth_mean is not None else "NA",
                'gdist': f"{goal_distance_mean:.2f}" if goal_distance_mean is not None else "NA",
                'vtg': f"{vel_toward_goal_mean:.2f}" if vel_toward_goal_mean is not None else "NA",
                **comp_str,
            })
            
            log_data['steps'].append(total_steps)
            log_data['rewards'].append(avg_reward)
            log_data['policy_loss'].append(metrics['policy_loss'])
            log_data['value_loss'].append(metrics['value_loss'])
            log_data['entropy'].append(metrics['entropy'])

            # 记录深度调试指标
            log_data['front_min_depth'].append(float(front_min_depth_mean) if front_min_depth_mean is not None else None)

            # 记录导航诊断指标
            log_data['goal_distance_mean'].append(float(goal_distance_mean) if goal_distance_mean is not None else None)
            log_data['vel_toward_goal_mean'].append(float(vel_toward_goal_mean) if vel_toward_goal_mean is not None else None)
            
            # 保存奖励组件
            comp_avg = {k: float(np.mean(v[-20:])) if v else 0.0 
                       for k, v in reward_components.items()}
            log_data['reward_components'].append(comp_avg)
            
            # 每1000步打印详细信息
            if total_steps % 1000 == 0:
                print(f"\n[Step {total_steps:,}] 奖励细节:")
                print(f"  总奖励: {avg_reward:.2f} | Episode长度: {avg_length:.0f}")
                print(f"  进度: {comp_avg.get('progress', 0):.4f} | 到达: {comp_avg.get('goal_reached', 0):.4f}")
                print(f"  速度: {comp_avg.get('velocity', 0):.4f} | 朝向: {comp_avg.get('orientation', 0):.4f}")
                print(f"  平滑: {comp_avg.get('smooth_action', 0):.4f} | 碰撞: {comp_avg.get('collision', 0):.4f}")
                if goal_distance_mean is not None and vel_toward_goal_mean is not None:
                    print(f"  目标距离均值: {goal_distance_mean:.3f} m | 朝目标速度: {vel_toward_goal_mean:.3f} m/s")
        
        # 保存checkpoint
        if total_steps % args.save_freq == 0 and total_steps > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_steps}.pt"
            agent.save(checkpoint_path)
            
            # 保存训练日志
            with open(run_dir / "training_log.json", 'w') as f:
                json.dump(log_data, f, indent=2)
            
            # 保存最佳模型
            if episode_rewards and np.mean(episode_rewards[-20:]) > best_moving_avg_reward:
                best_moving_avg_reward = np.mean(episode_rewards[-20:])
                best_path = checkpoint_dir / "best_model.pt"
                agent.save(best_path)
                print(f"\n✓ 新的最佳模型 (最近20回合平均奖励: {best_moving_avg_reward:.2f})")
    
    pbar.close()
    
    # 保存最终模型
    final_path = checkpoint_dir / "final_model.pt"
    agent.save(final_path)
    
    # 保存训练摘要
    summary = {
        'total_steps': total_steps,
        'total_episodes': len(episode_rewards),
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
        # 说明：这是“最近20回合均值”的最佳值，不等同于单回合最大值
        'best_moving_avg_reward_20': float(best_moving_avg_reward),
        'max_episode_reward': float(max_episode_reward) if episode_rewards else 0.0,
        'max_episode_length': int(max_episode_length),
        'mean_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'success_rate': float(np.mean(episode_successes)) if episode_successes else 0.0,
        'success_distance_threshold': float(success_distance_threshold),
        'with_bc_pretrain': args.pretrain_checkpoint is not None,
    }
    
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("训练完成！")
    print(f"{'='*80}")
    print(f"总步数: {total_steps:,}")
    print(f"总Episodes: {len(episode_rewards)}")
    print(f"平均奖励: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"最佳(最近20回合均值): {best_moving_avg_reward:.2f}")
    print(f"单回合最高奖励: {summary['max_episode_reward']:.2f} | 单回合最长长度: {summary['max_episode_length']}")
    print(f"模型保存位置: {run_dir}")
    print(f"{'='*80}\n")
    
    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train()
