#!/usr/bin/env python3
"""
SAC-Diffusion训练启动脚本（简化版）

使用BC预训练模型初始化，在Isaac Lab环境中进行强化学习训练
"""

import argparse
import csv
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import sys
import pickle

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="SAC-Diffusion训练")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量（降低以节省显存）")
parser.add_argument("--total_steps", type=int, default=1000000, help="总训练步数")
parser.add_argument("--pretrain_checkpoint", type=str, 
                    default="experiments/bc_training/bc_training_20251228_052241/best_model.pt",
                    help="BC预训练模型路径")
parser.add_argument("--output_dir", type=str, default="experiments/sac_training", help="输出目录")
parser.add_argument("--batch_size", type=int, default=32, help="批次大小（降低到32以节省显存）")
parser.add_argument("--save_freq", type=int, default=10000, help="保存频率")
parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
parser.add_argument("--log_freq", type=int, default=100, help="日志频率")
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

from env_factory import create_sac_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
import torch.nn as nn
import torch.nn.functional as F


print("="*80)
print("  SAC-Diffusion驾驶策略训练")
print("="*80)
print(f"  预训练模型: {args.pretrain_checkpoint}")
print(f"  并行环境: {args.num_envs}")
print(f"  总步数: {args.total_steps:,}")
print(f"  批次大小: {args.batch_size}")
print("="*80)


# ============================================================================
# 定义网络结构（添加残差连接 + CNN encoder 支持）
# ============================================================================

# ===== 分析：是否需要采用 PPO 的 CNN 处理？=====
# 答案：**强烈建议采用！**
# 理由：
# 1. 观测空间维度：当前 obs_dim=76813（低维13 + RGB 57600 + Depth 19200）
#    - 直接用 MLP 处理 76k 维 flatten 向量会导致：
#      a) 参数量爆炸（第一层线性层就有 76813*512 ≈ 39M 参数）
#      b) 训练不稳定（梯度在超高维空间容易消失/爆炸）
#      c) 过拟合风险高（视觉特征未经归纳偏置约束）
# 2. PPO 的 CNN encoder 优势：
#    - 3 层卷积提取空间特征（16→32→64 channels）
#    - 压缩为 64-dim embedding（降低 900 倍维度！）
#    - 与低维状态融合后再进 MLP（总输入仅 13+64+64=141 维）
# 3. SAC 的离策略学习更需要稳定的特征表示
# 建议：立即将 SAC 改为 CNN encoder 版本，与 PPO 保持一致的架构

class _ImageEncoder(nn.Module):
    """轻量 CNN 编码器（从 PPO 复用）"""
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
        
        # 动态推断 flatten 维度
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


# ============================================================================
# 定义简化的网络结构
# ============================================================================

class ResidualBlock(nn.Module):
    """残差块：如果维度匹配则跳跃连接，否则用线性层投影"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 如果输入输出维度不同，需要投影层
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.fc(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out + identity  # 残差连接


class SimpleDiffusionPolicy(nn.Module):
    """简化的扩散策略（用于RL微调）- 添加残差连接提升训练稳定性"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128], use_residual=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_residual = use_residual
        
        if use_residual:
            # 使用残差块
            self.blocks = nn.ModuleList()
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                self.blocks.append(ResidualBlock(prev_dim, hidden_dim, dropout=0.1))
                prev_dim = hidden_dim
            self.output_layer = nn.Linear(prev_dim, action_dim)
        else:
            # 原始 MLP 结构（兼容 BC 预训练加载）
            layers = []
            prev_dim = obs_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, action_dim))
            self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        if self.use_residual:
            x = obs
            for block in self.blocks:
                x = block(x)
            return self.output_layer(x)
        else:
            return self.network(obs)
    
    def get_action(self, obs, deterministic=False, exploration_noise=0.3):
        """获取动作（兼容SAC接口）"""
        with torch.no_grad():
            action = self.forward(obs)
            if not deterministic:
                # 添加探索噪声（从0.1增加到0.3提高探索）
                noise = torch.randn_like(action) * exploration_noise
                action = action + noise
            return torch.clamp(action, -1, 1)


class CNNDiffusionPolicy(nn.Module):
    """CNN 版本的扩散策略（与 PPO 一致的架构）"""
    
    def __init__(self, low_dim, rgb_shape, depth_shape, action_dim, 
                 rgb_embed_dim=64, depth_embed_dim=64, hidden_dims=[256, 256], use_residual=True):
        super().__init__()
        self.low_dim = low_dim
        self.rgb_h, self.rgb_w, self.rgb_c = rgb_shape
        self.dep_h, self.dep_w, self.dep_c = depth_shape
        self.action_dim = action_dim
        self.use_residual = use_residual
        
        # 视觉编码器
        self.rgb_encoder = _ImageEncoder(self.rgb_c, self.rgb_h, self.rgb_w, embed_dim=rgb_embed_dim)
        self.depth_encoder = _ImageEncoder(self.dep_c, self.dep_h, self.dep_w, embed_dim=depth_embed_dim)
        
        fusion_in_dim = low_dim + rgb_embed_dim + depth_embed_dim
        
        # 融合 MLP
        if use_residual:
            self.blocks = nn.ModuleList()
            prev_dim = fusion_in_dim
            for hidden_dim in hidden_dims:
                self.blocks.append(ResidualBlock(prev_dim, hidden_dim, dropout=0.1))
                prev_dim = hidden_dim
            self.output_layer = nn.Linear(prev_dim, action_dim)
        else:
            layers = []
            prev_dim = fusion_in_dim
            for hidden_dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, action_dim))
            self.shared = nn.Sequential(*layers)
    
    def forward(self, obs):
        # 切分观测
        low = obs[:, :self.low_dim]
        rgb_flat = obs[:, self.low_dim : self.low_dim + self.rgb_h*self.rgb_w*self.rgb_c]
        depth_flat = obs[:, self.low_dim + self.rgb_h*self.rgb_w*self.rgb_c :]
        
        # 重塑为图像
        rgb = rgb_flat.reshape(obs.shape[0], self.rgb_h, self.rgb_w, self.rgb_c).permute(0, 3, 1, 2).contiguous()
        depth = depth_flat.reshape(obs.shape[0], self.dep_h, self.dep_w, self.dep_c).permute(0, 3, 1, 2).contiguous()
        
        # 编码
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        fused = torch.cat([low, rgb_feat, depth_feat], dim=-1)
        
        # 策略头
        if self.use_residual:
            x = fused
            for block in self.blocks:
                x = block(x)
            return self.output_layer(x)
        else:
            return self.shared(fused)
    
    def get_action(self, obs, deterministic=False, exploration_noise=0.3):
        with torch.no_grad():
            action = self.forward(obs)
            if not deterministic:
                noise = torch.randn_like(action) * exploration_noise
                action = action + noise
            return torch.clamp(action, -1, 1)


class QNetwork(nn.Module):
    """Q网络（价值估计）- 添加残差连接提升训练稳定性"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 512, 256], use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        if use_residual:
            # 使用残差块
            self.input_layer = nn.Linear(obs_dim + action_dim, hidden_dims[0])
            self.blocks = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                self.blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=0.0))  # Critic 不用 dropout
            self.output_layer = nn.Linear(hidden_dims[-1], 1)
        else:
            # 原始结构
            hidden_dim = hidden_dims[0]
            self.network = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        if self.use_residual:
            x = self.input_layer(x)
            x = F.relu(x)
            for block in self.blocks:
                x = block(x)
            return self.output_layer(x)
        else:
            return self.network(x)


class CNNQNetwork(nn.Module):
    """CNN 版本的 Q 网络"""
    
    def __init__(self, low_dim, rgb_shape, depth_shape, action_dim,
                 rgb_embed_dim=64, depth_embed_dim=64, hidden_dims=[512, 512, 256], use_residual=True):
        super().__init__()
        self.low_dim = low_dim
        self.rgb_h, self.rgb_w, self.rgb_c = rgb_shape
        self.dep_h, self.dep_w, self.dep_c = depth_shape
        self.use_residual = use_residual
        
        # 视觉编码器（与 policy 共享相同的架构）
        self.rgb_encoder = _ImageEncoder(self.rgb_c, self.rgb_h, self.rgb_w, embed_dim=rgb_embed_dim)
        self.depth_encoder = _ImageEncoder(self.dep_c, self.dep_h, self.dep_w, embed_dim=depth_embed_dim)
        
        fusion_in_dim = low_dim + rgb_embed_dim + depth_embed_dim + action_dim
        
        if use_residual:
            self.input_layer = nn.Linear(fusion_in_dim, hidden_dims[0])
            self.blocks = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                self.blocks.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=0.0))
            self.output_layer = nn.Linear(hidden_dims[-1], 1)
        else:
            layers = []
            prev_dim = fusion_in_dim
            for hidden_dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
    
    def forward(self, obs, action):
        # 切分观测
        low = obs[:, :self.low_dim]
        rgb_flat = obs[:, self.low_dim : self.low_dim + self.rgb_h*self.rgb_w*self.rgb_c]
        depth_flat = obs[:, self.low_dim + self.rgb_h*self.rgb_w*self.rgb_c :]
        
        # 重塑为图像
        rgb = rgb_flat.reshape(obs.shape[0], self.rgb_h, self.rgb_w, self.rgb_c).permute(0, 3, 1, 2).contiguous()
        depth = depth_flat.reshape(obs.shape[0], self.dep_h, self.dep_w, self.dep_c).permute(0, 3, 1, 2).contiguous()
        
        # 编码
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)
        fused = torch.cat([low, rgb_feat, depth_feat, action], dim=-1)
        
        if self.use_residual:
            x = self.input_layer(fused)
            x = F.relu(x)
            for block in self.blocks:
                x = block(x)
            return self.output_layer(x)
        else:
            return self.network(fused)


class SimpleSACAgent:
    """简化的SAC智能体 - 支持 CNN encoder + 残差连接 + 奖励归一化"""
    
    def __init__(self, obs_dim, action_dim, device, lr=1e-4, use_cnn=False, 
                 low_dim=13, rgb_shape=(160,120,3), depth_shape=(160,120,1), use_residual=True):
        self.device = device
        self.action_dim = action_dim
        self.use_cnn = use_cnn
        self.use_residual = use_residual
        
        print(f"\n[SAC Agent] 初始化配置:")
        print(f"  使用 CNN: {use_cnn}")
        print(f"  使用残差连接: {use_residual}")
        print(f"  学习率: {lr:.0e}")
        
        if use_cnn:
            # CNN 版本（与 PPO 一致的架构）
            print(f"  观测切分: low_dim={low_dim}, RGB={rgb_shape}, Depth={depth_shape}")
            self.actor = CNNDiffusionPolicy(
                low_dim, rgb_shape, depth_shape, action_dim, 
                use_residual=use_residual
            ).to(device)
            self.critic1 = CNNQNetwork(
                low_dim, rgb_shape, depth_shape, action_dim,
                use_residual=use_residual
            ).to(device)
            self.critic2 = CNNQNetwork(
                low_dim, rgb_shape, depth_shape, action_dim,
                use_residual=use_residual
            ).to(device)
            self.critic1_target = CNNQNetwork(
                low_dim, rgb_shape, depth_shape, action_dim,
                use_residual=use_residual
            ).to(device)
            self.critic2_target = CNNQNetwork(
                low_dim, rgb_shape, depth_shape, action_dim,
                use_residual=use_residual
            ).to(device)
        else:
            # MLP 版本（兼容旧代码）
            print(f"  MLP 观测维度: {obs_dim}")
            self.actor = SimpleDiffusionPolicy(obs_dim, action_dim, use_residual=use_residual).to(device)
            self.critic1 = QNetwork(obs_dim, action_dim, use_residual=use_residual).to(device)
            self.critic2 = QNetwork(obs_dim, action_dim, use_residual=use_residual).to(device)
            self.critic1_target = QNetwork(obs_dim, action_dim, use_residual=use_residual).to(device)
            self.critic2_target = QNetwork(obs_dim, action_dim, use_residual=use_residual).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy temperature
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim * 0.6  # ↓ 调整为 -action_dim * 0.6（防止熄值过小）
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        
        # 奖励归一化（Running Mean/Std）
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_m2 = 0.0
        self.reward_count = 0
        
        print(f"  Actor 参数量: {sum(p.numel() for p in self.actor.parameters())/1e6:.2f}M")
        print(f"  Critic 参数量: {sum(p.numel() for p in self.critic1.parameters())/1e6:.2f}M (x2)")
    
    def normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """
        在线奖励归一化（Welford 算法）
        把奖励 scale 到均值 0、标准差 1 附近
        """
        # 更新 running statistics
        for r in reward.flatten():
            self.reward_count += 1
            delta = r.item() - self.reward_mean
            self.reward_mean += delta / self.reward_count
            delta2 = r.item() - self.reward_mean
            self.reward_m2 += delta * delta2
        
        # 计算标准差
        if self.reward_count > 1:
            self.reward_std = max((self.reward_m2 / self.reward_count) ** 0.5, 1e-6)
        
        # 归一化
        normalized = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        # Clip 到 [-10, 10] 防止极端值
        return torch.clamp(normalized, -10.0, 10.0)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs, deterministic=False):
        """选择动作"""
        return self.actor.get_action(obs, deterministic)
    
    def update(self, batch):
        """更新网络"""
        obs, actions, rewards, next_obs, dones = batch
        
        # ★ 关键：奖励归一化
        rewards = self.normalize_reward(rewards)
        
        # ======== 更新Critic ========
        with torch.no_grad():
            next_actions = self.actor.get_action(next_obs)
            
            # 计算目标Q值
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # TD target
            target_q = rewards + (1 - dones) * self.gamma * target_q
            # 稳定性：避免偶发尖峰把Critic打爆（不会改变正常量级下的学习）
            target_q = torch.clamp(target_q, -1_000.0, 1_000.0)
        
        # 当前Q值
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        
        # Critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)  # ↓ 从 10.0 降到 1.0
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)  # ↓ 从 10.0 降到 1.0
        self.critic2_optimizer.step()
        
        # ======== 更新Actor ========
        # 关键修复：actor(obs) 的 raw 输出未做[-1,1]约束，会导致 actor 通过放大动作幅值
        # “投机性”把Q推到极大值 -> Q爆炸。这里用tanh做可微分的动作边界。
        raw_new_actions = self.actor(obs)
        new_actions = torch.tanh(raw_new_actions)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = -q_new.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # ↓ 从 10.0 降到 1.0
        self.actor_optimizer.step()
        
        # ======== 更新Alpha ========
        # 简化：固定alpha
        alpha_loss = torch.tensor(0.0)
        
        # ======== 软更新目标网络 ========
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # 清理显存碎片（防止OOM）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'q_value': q_new.mean().item(),
        }
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
    
    def load_bc_pretrain(self, checkpoint_path):
        """从BC预训练模型加载Actor（严格匹配结构）"""
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
                class NumpyCompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith('numpy._core'):
                            module = module.replace('numpy._core', 'numpy.core')
                        return super().find_class(module, name)
                
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = NumpyCompatUnpickler(f).load()
            else:
                raise
        
        # 读取BC模型的hidden_dims配置
        hidden_dims = checkpoint.get('hidden_dims', [512, 256, 128])
        print(f"  BC模型结构: hidden_dims={hidden_dims}")
        
        # 重新创建Actor以匹配BC结构
        print(f"  重建Actor网络以匹配BC结构...")
        self.actor = SimpleDiffusionPolicy(
            self.actor.obs_dim, 
            self.actor.action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # 严格加载权重（必须完全匹配）
        try:
            self.actor.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("  ✓ BC预训练权重已成功加载 (strict=True)")
            
            # 加载归一化参数（处理numpy数组）
            obs_mean = checkpoint.get('obs_mean', torch.zeros(self.actor.obs_dim))
            obs_std = checkpoint.get('obs_std', torch.ones(self.actor.obs_dim))
            action_mean = checkpoint.get('action_mean', torch.zeros(self.actor.action_dim))
            action_std = checkpoint.get('action_std', torch.ones(self.actor.action_dim))
            
            # 转换为tensor（如果是numpy数组）
            if isinstance(obs_mean, np.ndarray):
                self.obs_mean = torch.from_numpy(obs_mean).float().to(self.device)
                self.obs_std = torch.from_numpy(obs_std).float().to(self.device)
                self.action_mean = torch.from_numpy(action_mean).float().to(self.device)
                self.action_std = torch.from_numpy(action_std).float().to(self.device)
            else:
                self.obs_mean = obs_mean.to(self.device)
                self.obs_std = obs_std.to(self.device)
                self.action_mean = action_mean.to(self.device)
                self.action_std = action_std.to(self.device)
            
            print(f"  ✓ 归一化参数已加载 (obs: {self.obs_mean.shape}, action: {self.action_mean.shape})")
            
            # 重新初始化优化器（因为actor重建了）
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            print("  ✓ Actor优化器已重新初始化")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            print(f"  → 从头开始训练Actor")
            self.obs_mean = None
            self.obs_std = None
            self.action_mean = None
            self.action_std = None
            return False


# ============================================================================
# 经验回放池
# ============================================================================

class ReplayBuffer:
    """简单的经验回放池（使用列表存储以节省内存）"""
    
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 使用列表存储（节省内存，观测维度太大）
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
    
    def add(self, obs, action, reward, next_obs, done):
        """添加经验"""
        # 转换为CPU tensor并添加到列表
        obs_cpu = obs.cpu() if isinstance(obs, torch.Tensor) else torch.from_numpy(obs)
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else torch.from_numpy(action)
        next_obs_cpu = next_obs.cpu() if isinstance(next_obs, torch.Tensor) else torch.from_numpy(next_obs)
        
        if len(self.obs) < self.capacity:
            # 还没满，直接添加
            self.obs.append(obs_cpu)
            self.actions.append(action_cpu)
            self.rewards.append(reward)
            self.next_obs.append(next_obs_cpu)
            self.dones.append(done)
        else:
            # 已满，覆盖最旧的
            self.obs[self.ptr] = obs_cpu
            self.actions[self.ptr] = action_cpu
            self.rewards[self.ptr] = reward
            self.next_obs[self.ptr] = next_obs_cpu
            self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """采样batch"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # 从列表中采样并堆叠
        batch_obs = torch.stack([self.obs[i] for i in indices]).to(self.device)
        batch_actions = torch.stack([self.actions[i] for i in indices]).to(self.device)
        batch_rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float32).unsqueeze(1).to(self.device)
        batch_next_obs = torch.stack([self.next_obs[i] for i in indices]).to(self.device)
        batch_dones = torch.tensor([self.dones[i] for i in indices], dtype=torch.float32).unsqueeze(1).to(self.device)
        
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    
    def __len__(self):
        return self.size


# ============================================================================
# 主训练循环
# ============================================================================

def train():
    """主训练函数"""
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"sac_training_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 结构化日志（用于画RA-L级别结果图）
    metrics_csv_path = run_dir / "metrics.csv"
    episodes_csv_path = run_dir / "episodes.csv"
    config_path = run_dir / "config.yaml"

    # 保存配置（确保可复现）
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "args": vars(args),
                "notes": {
                    "obs_layout": "[low(13), rgb(160*120*3), depth(160*120*1)]",
                    "reward": "create_sac_env_cfg()",
                },
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    
    # 创建环境
    print("\n创建训练环境...")
    print("  使用SAC专用奖励配置...")
    env_cfg = create_sac_env_cfg(num_envs=args.num_envs)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取维度（动态从环境中获取，避免硬编码）
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    print(f"  观察维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检测是否应该使用 CNN（根据 obs 维度判断）
    # 典型图像观测维度: rgb(160*120*3) + depth(160*120*1) = 76800
    # 原始低维为 13；启用 LiDAR 后低维会变为 13+360=373，因此 obs_dim 会相应增加。
    use_cnn = (obs_dim > 1000)  # 如果 obs 维度很大，就使用 CNN

    # 动态推断 low_dim（避免硬编码 13；兼容 LiDAR 拼接到 low_dim）
    # ↓ 更新为新的相机分辨率 96x80 (从 160x120)
    rgb_h, rgb_w, rgb_c = (80, 96, 3)  
    dep_h, dep_w, dep_c = (80, 96, 1)
    img_dim = rgb_h * rgb_w * rgb_c + dep_h * dep_w * dep_c
    low_dim = obs_dim - img_dim if use_cnn else obs_dim
    if use_cnn and low_dim <= 0:
        raise RuntimeError(f"Invalid low_dim={low_dim}. obs_dim={obs_dim}, img_dim={img_dim}.")
    print(f"  推断 low_dim: {low_dim} (img_dim={img_dim})")
    
    # 创建SAC智能体（默认启用 CNN + 残差）
    print("\n初始化SAC智能体...")
    agent = SimpleSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        use_cnn=use_cnn,  # ✅ 自动检测是否使用 CNN
        low_dim=low_dim,
        rgb_shape=(80, 96, 3),   # ↓ 更新为新分辨率
        depth_shape=(80, 96, 1), # ↓ 更新为新分辨率
        use_residual=True  # ✅ 默认启用残差连接
    )
    
    # 加载BC预训练（仅支持 MLP 版本；CNN 版本会导致结构不匹配）
    if Path(args.pretrain_checkpoint).exists():
        if use_cnn:
            print("⚠  当前为 CNN 版本（含图像/可含LiDAR），跳过 BC 预训练加载以避免结构不匹配。")
        else:
            agent.load_bc_pretrain(args.pretrain_checkpoint)
    else:
        print(f"⚠ 预训练模型不存在: {args.pretrain_checkpoint}")
    
    # 创建经验回放池（增大到 50K 提升样本效率）
    print("\n创建经验回放池...")
    replay_buffer = ReplayBuffer(
        capacity=20000,  # ↑ 从 10K 提升到 20K（SAC 需要较大 buffer）
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    print(f"  Buffer容量: 20,000 (预计占用: ~4.8GB RAM)")
    
    # 训练统计
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    goal_reached_count = 0  # 成功到达目标的次数
    best_reward = -float('inf')

    # 打开CSV写入器
    metrics_f = open(metrics_csv_path, "w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_f,
        fieldnames=[
            "wall_time_sec",
            "step",
            "episodes",
            "success_rate",
            "time_out_rate",
            "fallen_or_other_rate",
            "avg_reward_10",
            "avg_length_10",
            "buffer_size",
            "critic1_loss",
            "critic2_loss",
            "actor_loss",
            "q_value",
            "alpha",
            "goal_distance_mean",
            "goal_distance_min",
            "vel_toward_goal_mean",
            "front_min_depth_mean",
            "lidar_min_range_front_mean",
            "lidar_min_range_360_mean",
            "obstacle_min_dist_mean",
            "obstacle_penalty_mean",
        ],
    )
    metrics_writer.writeheader()
    metrics_f.flush()

    episodes_f = open(episodes_csv_path, "w", newline="", encoding="utf-8")
    episodes_writer = csv.DictWriter(
        episodes_f,
        fieldnames=[
            "episode_id",
            "end_step",
            "env_id",
            "reward",
            "length",
            "success",
            "termination",
            "goal_distance_end",
        ],
    )
    episodes_writer.writeheader()
    episodes_f.flush()

    start_time = time.time()
    
    # ★ Warm-up 阶段：随机探索填充 buffer（5000 步）
    warmup_steps = 5000
    use_warmup = True  # 强制启用 warm-up，不依赖 BC
    
    # 环境状态（已在获取维度时reset过）
    obs = obs_dict["policy"]
    obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
    
    # BC模型已经内部归一化，不需要外部再归一化
    # 移除了错误的归一化代码
    
    episode_reward = torch.zeros(args.num_envs, device=device)
    episode_length = torch.zeros(args.num_envs, device=device, dtype=torch.int)

    time_out_count = 0
    fallen_or_other_count = 0
    
    print(f"\n{'='*80}")
    print("开始训练...")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=args.total_steps, desc="训练进度")

    # 最近一次 update() 的指标（用于在 buffer 尚未足够时也能写日志）
    last_update_metrics = {}

    try:
        while total_steps < args.total_steps:
            # ★ Warm-up 阶段切换
            if total_steps >= warmup_steps and use_warmup:
                use_warmup = False
                print(f"\n{'='*80}")
                print(f"  ✓ Warm-up完成! 切换到SAC策略...")
                print(f"  已收集 {warmup_steps} 步随机经验")
                print(f"  Buffer大小: {len(replay_buffer)}/{replay_buffer.capacity}")
                print(f"{'='*80}\n")

            # 选择动作
            with torch.no_grad():
                if use_warmup:
                    # ★ Warm-up：随机动作（均匀分布于 [-1, 1]）
                    actions = torch.rand(args.num_envs, action_dim, device=device) * 2 - 1
                else:
                    # SAC探索：使用SAC策略（高噪声）
                    actions = agent.select_action(obs, deterministic=False)

            # 执行动作
            next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            next_obs = next_obs_dict["policy"]
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
            dones = terminated | truncated

            # 在线计算目标距离/朝向目标速度（用于诊断“为什么成功率为0”）
            goal_distance = None
            vel_toward_goal = None
            try:
                if hasattr(env, "goal_positions") and "robot" in env.scene.articulations:
                    robot_pos_xy = env.scene.articulations["robot"].data.root_pos_w[:, :2]
                    goal_pos_xy = env.goal_positions[:, :2]
                    to_goal = goal_pos_xy - robot_pos_xy
                    goal_distance = torch.norm(to_goal, dim=-1)

                    to_goal_dir = to_goal / (goal_distance.unsqueeze(-1) + 1e-6)
                    lin_vel_xy = env.scene.articulations["robot"].data.root_lin_vel_w[:, :2]
                    vel_toward_goal = torch.sum(lin_vel_xy * to_goal_dir, dim=-1)
            except Exception:
                goal_distance = None
                vel_toward_goal = None

            # 添加到回放池（每个环境）
            for i in range(args.num_envs):
                replay_buffer.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i].float())

                episode_reward[i] += rewards[i]
                episode_length[i] += 1

                if dones[i]:
                    episode_rewards.append(episode_reward[i].item())
                    episode_lengths.append(episode_length[i].item())

                    # 检测是否到达目标：优先用“距离阈值”判定，避免 infos['final_info'] 缺失导致误判
                    success = False
                    goal_distance_end = None
                    if goal_distance is not None:
                        goal_distance_end = float(goal_distance[i].item())
                        success = bool(goal_distance_end < 0.5)
                    else:
                        if 'final_info' in infos and i < len(infos['final_info']):
                            success = bool(infos['final_info'][i].get('goal_reached', False))

                    if success:
                        goal_reached_count += 1

                    if bool(truncated[i].item()) if hasattr(truncated[i], "item") else bool(truncated[i]):
                        termination = "time_out"
                        time_out_count += 1
                    else:
                        termination = "goal_reached" if success else "fallen_or_other"
                        if not success:
                            fallen_or_other_count += 1

                    # 保存逐episode数据
                    episodes_writer.writerow(
                        {
                            "episode_id": len(episode_rewards),
                            "end_step": total_steps,
                            "env_id": i,
                            "reward": float(episode_rewards[-1]),
                            "length": int(episode_lengths[-1]),
                            "success": int(success),
                            "termination": termination,
                            "goal_distance_end": goal_distance_end,
                        }
                    )
                    if len(episode_rewards) % 10 == 0:
                        episodes_f.flush()

                    # 打印前10个episode的详细信息
                    if len(episode_rewards) <= 10:
                        status = "✓成功" if success else "✗超时/失败"
                        phase = "[Warm-up]" if use_warmup else "[SAC训练]"
                        print(
                            f"\n{phase} Episode {len(episode_rewards)} (环境{i}): {status}, 奖励={episode_reward[i].item():.2f}, 步数={episode_length[i].item()}"
                        )

                    episode_reward[i] = 0
                    episode_length[i] = 0

            obs = next_obs
            total_steps += args.num_envs
            pbar.update(args.num_envs)

            # 更新策略
            if len(replay_buffer) >= args.batch_size * 10:
                batch = replay_buffer.sample(args.batch_size)
                metrics = agent.update(batch)
                last_update_metrics = metrics or {}

            # 记录日志（不依赖于是否进行了 update；便于尽早确认 LiDAR/Depth 惩罚信号生效）
            if total_steps % args.log_freq == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0.0
                success_rate = (goal_reached_count / len(episode_rewards) * 100) if episode_rewards else 0.0
                time_out_rate = (time_out_count / len(episode_rewards) * 100) if episode_rewards else 0.0
                fallen_or_other_rate = (fallen_or_other_count / len(episode_rewards) * 100) if episode_rewards else 0.0

                goal_distance_mean = None
                goal_distance_min = None
                vel_toward_goal_mean = None
                if goal_distance is not None:
                    goal_distance_mean = float(goal_distance.mean().item())
                    goal_distance_min = float(goal_distance.min().item())
                if vel_toward_goal is not None:
                    vel_toward_goal_mean = float(vel_toward_goal.mean().item())

                # 避障惩罚诊断（来自 rosorin_mdp.obstacle_avoidance_penalty 内部写入的 env.last_*）
                front_min_depth_mean = None
                lidar_min_range_front_mean = None
                lidar_min_range_360_mean = None
                obstacle_min_dist_mean = None
                obstacle_penalty_mean = None
                try:
                    if hasattr(env, "last_front_min_depth"):
                        front_min_depth_mean = float(env.last_front_min_depth.mean().item())
                    if hasattr(env, "last_lidar_min_range_front"):
                        lidar_min_range_front_mean = float(env.last_lidar_min_range_front.mean().item())
                    if hasattr(env, "last_lidar_min_range_360"):
                        lidar_min_range_360_mean = float(env.last_lidar_min_range_360.mean().item())
                    if hasattr(env, "last_obstacle_min_dist"):
                        obstacle_min_dist_mean = float(env.last_obstacle_min_dist.mean().item())
                    if hasattr(env, "last_obstacle_penalty"):
                        obstacle_penalty_mean = float(env.last_obstacle_penalty.mean().item())
                except Exception:
                    pass

                # tqdm 显示：优先展示最近一次 update 的 q
                if last_update_metrics:
                    pbar.set_postfix({
                        'reward': f"{avg_reward:.2f}",
                        'len': f"{avg_length:.0f}",
                        'q': f"{last_update_metrics.get('q_value', 0.0):.2f}",
                        'episodes': len(episode_rewards),
                    })
                else:
                    pbar.set_postfix({
                        'reward': f"{avg_reward:.2f}",
                        'len': f"{avg_length:.0f}",
                        'episodes': len(episode_rewards),
                    })

                # 保存逐步训练指标（用于画loss/成功率曲线）
                metrics_writer.writerow(
                    {
                        "wall_time_sec": float(time.time() - start_time),
                        "step": int(total_steps),
                        "episodes": int(len(episode_rewards)),
                        "success_rate": float(success_rate),
                        "time_out_rate": float(time_out_rate),
                        "fallen_or_other_rate": float(fallen_or_other_rate),
                        "avg_reward_10": float(avg_reward),
                        "avg_length_10": float(avg_length),
                        "buffer_size": int(len(replay_buffer)),
                        "critic1_loss": float(last_update_metrics.get("critic1_loss", 0.0)) if last_update_metrics else None,
                        "critic2_loss": float(last_update_metrics.get("critic2_loss", 0.0)) if last_update_metrics else None,
                        "actor_loss": float(last_update_metrics.get("actor_loss", 0.0)) if last_update_metrics else None,
                        "q_value": float(last_update_metrics.get("q_value", 0.0)) if last_update_metrics else None,
                        "alpha": float(last_update_metrics.get("alpha", 0.0)) if last_update_metrics else None,
                        "goal_distance_mean": float(goal_distance_mean) if goal_distance_mean is not None else None,
                        "goal_distance_min": float(goal_distance_min) if goal_distance_min is not None else None,
                        "vel_toward_goal_mean": float(vel_toward_goal_mean) if vel_toward_goal_mean is not None else None,
                        "front_min_depth_mean": float(front_min_depth_mean) if front_min_depth_mean is not None else None,
                        "lidar_min_range_front_mean": float(lidar_min_range_front_mean) if lidar_min_range_front_mean is not None else None,
                        "lidar_min_range_360_mean": float(lidar_min_range_360_mean) if lidar_min_range_360_mean is not None else None,
                        "obstacle_min_dist_mean": float(obstacle_min_dist_mean) if obstacle_min_dist_mean is not None else None,
                        "obstacle_penalty_mean": float(obstacle_penalty_mean) if obstacle_penalty_mean is not None else None,
                    }
                )
                if total_steps % (args.log_freq * 10) == 0:
                    metrics_f.flush()

                # 每1000步打印详细信息
                if total_steps % 1000 == 0:
                    phase_tag = "[Warm-up]" if use_warmup else "[SAC训练]"
                    print(f"\n[Step {total_steps:,}] {phase_tag}")
                    print(f"  完成Episodes: {len(episode_rewards)} | 成功: {goal_reached_count} ({success_rate:.1f}%)")
                    print(f"  终止分布: 超时 {time_out_count} ({time_out_rate:.1f}%) | 其它失败 {fallen_or_other_count} ({fallen_or_other_rate:.1f}%)")
                    print(f"  平均奖励(最近10个): {avg_reward:.2f}")
                    print(f"  平均步数(最近10个): {avg_length:.0f}")
                    print(f"  Buffer大小: {len(replay_buffer)}/{replay_buffer.capacity}")
                    if last_update_metrics:
                        print(f"  Q值: {last_update_metrics.get('q_value', 0):.2f} | Actor Loss: {last_update_metrics.get('actor_loss', 0):.4f}")

                        # 警告：检测异常Q值
                        if abs(last_update_metrics.get('q_value', 0)) > 100:
                            print(f"  ⚠️ 警告: Q值异常 ({last_update_metrics.get('q_value', 0):.2f})，Critic可能不稳定")
                    print(f"  平均奖励: {avg_reward:.2f}")
                    print(f"  当前累积奖励(所有环境平均): {episode_reward.mean().item():.2f}")
                    print(f"  当前步数(所有环境平均): {episode_length.float().mean().item():.0f}")
                    print(f"  [DEBUG] 最近一步reward范围: [{rewards.min().item():.4f}, {rewards.max().item():.4f}], 均值: {rewards.mean().item():.4f}")
                    if goal_distance_mean is not None and vel_toward_goal_mean is not None:
                        print(f"  目标距离: mean={goal_distance_mean:.2f}m | min={goal_distance_min:.2f}m | 朝目标速度均值={vel_toward_goal_mean:.3f}m/s")

                    if obstacle_min_dist_mean is not None:
                        print(
                            f"  避障诊断: depth_front={front_min_depth_mean if front_min_depth_mean is not None else float('nan'):.2f}m | "
                            f"lidar_front={lidar_min_range_front_mean if lidar_min_range_front_mean is not None else float('nan'):.2f}m | "
                            f"fused_min={obstacle_min_dist_mean:.2f}m | penalty_mean={obstacle_penalty_mean if obstacle_penalty_mean is not None else float('nan'):.3f}"
                        )

            # 保存checkpoint
            if total_steps % args.save_freq == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{total_steps}.pt"
                agent.save(checkpoint_path)
                print(f"\n✓ 已保存checkpoint: {checkpoint_path}")

                # 保存最佳模型
                if episode_rewards and np.mean(episode_rewards[-20:]) > best_reward:
                    best_reward = np.mean(episode_rewards[-20:])
                    best_path = checkpoint_dir / "best_model.pt"
                    agent.save(best_path)
                    print(f"✓ 新的最佳模型 (平均奖励: {best_reward:.2f}): {best_path}")

                    # 同步一次结构化日志（方便中途kill也不丢图）
                    metrics_f.flush()
                    episodes_f.flush()

        pbar.close()

        # 保存最终模型
        final_path = checkpoint_dir / "final_model.pt"
        agent.save(final_path)

        print(f"\n{'='*80}")
        print("训练完成！")
        print(f"{'='*80}")
        print(f"总步数: {total_steps:,}")
        print(f"总Episodes: {len(episode_rewards)}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"最佳奖励: {best_reward:.2f}")
        print(f"模型保存位置: {run_dir}")
        print(f"{'='*80}\n")

    finally:
        # 关闭文件句柄
        try:
            metrics_f.flush()
            metrics_f.close()
        except Exception:
            pass
        try:
            episodes_f.flush()
            episodes_f.close()
        except Exception:
            pass

        # 关闭环境
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    train()
