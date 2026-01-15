#!/usr/bin/env python3
"""
纯SAC训练脚本 (无Diffusion Policy)

Baseline: 标准SAC算法，不使用扩散策略
用于对比验证Diffusion Policy的优势
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import sys
import pickle
import json

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="纯SAC训练（无Diffusion）")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--total_steps", type=int, default=10000, help="总训练步数")
parser.add_argument("--pretrain_checkpoint", type=str, default=None, help="BC预训练模型路径（可选）")
parser.add_argument("--output_dir", type=str, default="experiments/baselines/sac_pure", help="输出目录")
parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
parser.add_argument("--buffer_size", type=int, default=50000, help="回放池大小")
parser.add_argument("--save_freq", type=int, default=10000, help="保存频率")
parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
parser.add_argument("--log_freq", type=int, default=100, help="日志频率")
parser.add_argument("--warmup_steps", type=int, default=1000, help="预热步数（随机探索）")
parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 设置headless和相机
args.headless = True
args.enable_cameras = True

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入Isaac Lab和自定义模块
from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


print("="*80)
print("  纯SAC驾驶策略训练 (无Diffusion Policy)")
print("="*80)
print(f"  并行环境: {args.num_envs}")
print(f"  总步数: {args.total_steps:,}")
print(f"  批次大小: {args.batch_size}")
print(f"  回放池: {args.buffer_size:,}")
print("="*80)


# ============================================================================
# SAC网络定义
# ============================================================================

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


class GaussianActor(nn.Module):
    """高斯策略网络（标准SAC Actor）"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        
        # 共享层
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Mean和log_std头
        self.mean = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)
    
    def forward(self, obs, deterministic=False, return_log_prob=True):
        """
        前向传播
        
        Args:
            obs: 观测
            deterministic: 是否使用确定性策略（mean）
            return_log_prob: 是否返回log概率
        
        Returns:
            action: 动作
            log_prob: log概率（可选）
        """
        x = self.shared(obs)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        
        # 创建高斯分布
        normal = Normal(mean, std)
        
        if deterministic:
            action_pre_tanh = mean
        else:
            # 重参数化采样
            action_pre_tanh = normal.rsample()
        
        # 应用tanh并归一化到[-1, 1]
        action = torch.tanh(action_pre_tanh)
        
        if return_log_prob:
            # 计算log概率并应用tanh修正
            log_prob = normal.log_prob(action_pre_tanh)
            # Tanh修正: log_prob - log(1 - tanh^2(x))
            log_prob = log_prob - torch.log(1 - action.pow(2) + EPSILON)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            return action, log_prob
        
        return action, None
    
    def get_action(self, obs, deterministic=False):
        """兼容接口"""
        with torch.no_grad():
            action, _ = self.forward(obs, deterministic, return_log_prob=False)
            return action


class QNetwork(nn.Module):
    """Q网络（价值估计）"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        
        layers = []
        in_dim = obs_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)


class PureSACAgent:
    """纯SAC智能体（无Diffusion）"""
    
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_tune_alpha=True,
        target_entropy=None,
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Actor (Gaussian Policy)
        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        
        # Twin Q-networks
        self.critic1 = QNetwork(obs_dim, action_dim).to(device)
        self.critic2 = QNetwork(obs_dim, action_dim).to(device)
        
        # Target critics
        self.critic1_target = QNetwork(obs_dim, action_dim).to(device)
        self.critic2_target = QNetwork(obs_dim, action_dim).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy temperature
        self.auto_tune_alpha = auto_tune_alpha
        if auto_tune_alpha:
            self.target_entropy = target_entropy if target_entropy else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor([0.2], device=device).log()
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, obs, deterministic=False):
        """选择动作"""
        return self.actor.get_action(obs, deterministic)
    
    def update(self, batch):
        """更新网络"""
        obs, actions, rewards, next_obs, dones = batch
        
        # ======== 更新Critic ========
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_obs, deterministic=False, return_log_prob=True)
            
            # 计算目标Q值
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # 加入熵项
            target_q = target_q - self.alpha * next_log_probs
            
            # TD target
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 当前Q值
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        
        # Critic loss
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # ======== 更新Actor ========
        new_actions, new_log_probs = self.actor(obs, deterministic=False, return_log_prob=True)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss (最大化 Q - alpha * log_pi)
        actor_loss = (self.alpha.detach() * new_log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ======== 更新Alpha ========
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (new_log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)
        
        # ======== 软更新目标网络 ========
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item(),
            'alpha_loss': alpha_loss.item(),
            'q_value': q_new.mean().item(),
            'log_prob': new_log_probs.mean().item(),
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
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_tune_alpha else None,
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])


# ============================================================================
# 经验回放池
# ============================================================================

class ReplayBuffer:
    """经验回放池"""
    
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # 使用列表存储
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
    
    def add(self, obs, action, reward, next_obs, done):
        """添加经验"""
        obs_cpu = obs.cpu() if isinstance(obs, torch.Tensor) else torch.from_numpy(obs)
        action_cpu = action.cpu() if isinstance(action, torch.Tensor) else torch.from_numpy(action)
        next_obs_cpu = next_obs.cpu() if isinstance(next_obs, torch.Tensor) else torch.from_numpy(next_obs)
        
        if len(self.obs) < self.capacity:
            self.obs.append(obs_cpu)
            self.actions.append(action_cpu)
            self.rewards.append(reward)
            self.next_obs.append(next_obs_cpu)
            self.dones.append(done)
        else:
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

def extract_reward_components(env):
    """
    从Isaac Lab环境的reward_manager中提取各个奖励项的值
    
    Returns:
        dict: 奖励项名称 -> 平均值的字典
    """
    reward_dict = {}
    
    # 通过reward_manager获取各个奖励项
    try:
        if hasattr(env.unwrapped, 'reward_manager'):
            manager = env.unwrapped.reward_manager
            if hasattr(manager, '_term_buffers'):
                for term_name, term_buffer in manager._term_buffers.items():
                    if isinstance(term_buffer, torch.Tensor):
                        reward_dict[term_name] = term_buffer.mean().item()
    except Exception as e:
        pass  # 静默失败，返回空字典
    
    return reward_dict


def train():
    """主训练函数"""
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"sac_pure_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 保存配置
    config = vars(args)
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # 创建环境
    print("\n创建训练环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取维度
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    print(f"  观察维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建SAC智能体
    print("\n初始化纯SAC智能体...")
    agent = PureSACAgent(
        obs_dim, 
        action_dim, 
        device,
        lr=args.lr,
        auto_tune_alpha=True,
    )
    
    # 创建经验回放池
    print("\n创建经验回放池...")
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device
    )
    
    # 训练统计
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    best_reward = -float('inf')
    
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
        'smooth_action': [],
        'collision': [],
        'stability': [],
        'height': [],
    }
    
    # 日志文件
    log_data = {
        'steps': [],
        'rewards': [],
        'losses': [],
        'q_values': [],
        'alpha': [],
        'reward_components': [],
    }
    
    print(f"\n{'='*80}")
    print("开始训练纯SAC...")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=args.total_steps, desc="训练进度")
    
    while total_steps < args.total_steps:
        # 选择动作
        if total_steps < args.warmup_steps:
            # 预热阶段：随机探索
            actions = torch.rand(args.num_envs, action_dim, device=device) * 2 - 1
        else:
            with torch.no_grad():
                actions = agent.select_action(obs, deterministic=False)
        
        # 执行动作
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        next_obs = next_obs_dict["policy"]
        next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        dones = terminated | truncated
        
        # 提取当前步的奖励细节（在step之后立即提取）
        current_reward_components = extract_reward_components(env)
        
        # 添加到回放池
        for i in range(args.num_envs):
            replay_buffer.add(
                obs[i], actions[i], rewards[i], next_obs[i], dones[i].float()
            )
            
            episode_reward[i] += rewards[i]
            episode_length[i] += 1
            
            if dones[i]:
                episode_rewards.append(episode_reward[i].item())
                episode_lengths.append(episode_length[i].item())
                
                # 保存当前episode的奖励组件
                for key, value in current_reward_components.items():
                    if key in reward_components:
                        reward_components[key].append(value)
                
                # 记录成功率（基于是否到达目标）
                if episode_reward[i] > 50:  # 如果总奖励超过50（说明获得了goal_reached奖励）
                    episode_successes.append(1)
                else:
                    episode_successes.append(0)
                
                episode_reward[i] = 0
                episode_length[i] = 0
        
        obs = next_obs
        total_steps += args.num_envs
        pbar.update(args.num_envs)
        
        # 更新策略
        if len(replay_buffer) >= args.batch_size and total_steps >= args.warmup_steps:
            batch = replay_buffer.sample(args.batch_size)
            metrics = agent.update(batch)
            
            # 记录日志
            if total_steps % args.log_freq == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                avg_length = np.mean(episode_lengths[-10:]) if episode_lengths else 0.0
                
                # 计算平均奖励组件
                comp_str = {}
                for key, values in reward_components.items():
                    if values:
                        avg_val = np.mean(values[-10:])
                        comp_str[key[:3]] = f"{avg_val:.2f}"
                
                pbar.set_postfix({
                    'reward': f"{avg_reward:.2f}",
                    'len': f"{avg_length:.0f}",
                    'q': f"{metrics['q_value']:.2f}",
                    'α': f"{metrics['alpha']:.3f}",
                    **comp_str,
                })
                
                # 保存日志数据
                log_data['steps'].append(total_steps)
                log_data['rewards'].append(avg_reward)
                log_data['losses'].append(metrics['actor_loss'])
                log_data['q_values'].append(metrics['q_value'])
                log_data['alpha'].append(metrics['alpha'])
                
                # 保存奖励组件
                comp_avg = {k: float(np.mean(v[-10:])) if v else 0.0 
                           for k, v in reward_components.items()}
                log_data['reward_components'].append(comp_avg)
                
                # 每1000步打印详细信息
                if total_steps % 1000 == 0:
                    print(f"\n[Step {total_steps:,}] 奖励细节:")
                    print(f"  总奖励: {avg_reward:.2f} | Episode长度: {avg_length:.0f}")
                    
                    # 显示当前步的即时奖励细节（不是历史平均）
                    if current_reward_components:
                        print(f"  [即时奖励] 进度: {current_reward_components.get('progress', 0):.3f} | "
                              f"到达: {current_reward_components.get('goal_reached', 0):.3f}")
                        print(f"  [即时奖励] 速度: {current_reward_components.get('velocity_tracking', 0):.3f} | "
                              f"朝向: {current_reward_components.get('orientation', 0):.3f}")
                        print(f"  [即时惩罚] 平滑: {current_reward_components.get('action_smoothness', 0):.3f} | "
                              f"稳定: {current_reward_components.get('stability', 0):.3f}")
                        print(f"  [即时惩罚] 高度: {current_reward_components.get('height', 0):.3f}")
                    
                    # 显示历史平均（如果有episode完成）
                    if comp_avg and any(v != 0 for v in comp_avg.values()):
                        print(f"  [历史平均] 进度: {comp_avg.get('progress', 0):.3f} | "
                              f"到达: {comp_avg.get('goal_reached', 0):.3f}")
                        print(f"  [历史平均] 速度: {comp_avg.get('velocity', 0):.3f} | "
                              f"朝向: {comp_avg.get('orientation', 0):.3f}")
        
        # 保存checkpoint
        if total_steps % args.save_freq == 0 and total_steps > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_steps}.pt"
            agent.save(checkpoint_path)
            
            # 保存训练日志
            with open(run_dir / "training_log.json", 'w') as f:
                json.dump(log_data, f, indent=2)
            
            # 保存最佳模型
            if episode_rewards and np.mean(episode_rewards[-20:]) > best_reward:
                best_reward = np.mean(episode_rewards[-20:])
                best_path = checkpoint_dir / "best_model.pt"
                agent.save(best_path)
                print(f"\n✓ 新的最佳模型 (平均奖励: {best_reward:.2f})")
    
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
        'best_reward': float(best_reward),
        'mean_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        'success_rate': float(np.mean(episode_successes)) if episode_successes else 0.0,
    }
    
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("训练完成！")
    print(f"{'='*80}")
    print(f"总步数: {total_steps:,}")
    print(f"总Episodes: {len(episode_rewards)}")
    print(f"平均奖励: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"模型保存位置: {run_dir}")
    print(f"{'='*80}\n")
    
    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train()
