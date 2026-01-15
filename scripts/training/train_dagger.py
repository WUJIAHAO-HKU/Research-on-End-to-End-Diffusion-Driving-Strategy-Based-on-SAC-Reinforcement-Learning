#!/usr/bin/env python3
"""
DAgger训练脚本 (Dataset Aggregation)

Baseline: DAgger算法 - 介于BC和RL之间的方法
迭代地收集专家数据并训练策略，逐步减少对专家的依赖
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import json
import h5py

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="DAgger训练")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--n_iterations", type=int, default=20, help="DAgger迭代次数")
parser.add_argument("--steps_per_iteration", type=int, default=50000, help="每次迭代的步数")
parser.add_argument("--expert_data_path", type=str, 
                    default="data/demonstrations/rosorin_mpc_demos_medium_20251229_093253.h5",
                    help="初始专家数据路径")
parser.add_argument("--output_dir", type=str, default="experiments/baselines/dagger", help="输出目录")
parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
parser.add_argument("--n_epochs", type=int, default=50, help="每次迭代的训练epochs")
parser.add_argument("--save_freq", type=int, default=5, help="保存频率（迭代）")
parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
parser.add_argument("--beta_schedule", type=str, default="linear", 
                    choices=["constant", "linear", "exponential"], help="专家混合率衰减策略")
parser.add_argument("--initial_beta", type=float, default=1.0, help="初始专家混合率")
parser.add_argument("--final_beta", type=float, default=0.1, help="最终专家混合率")
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


print("="*80)
print("  DAgger驾驶策略训练")
print("="*80)
print(f"  并行环境: {args.num_envs}")
print(f"  DAgger迭代: {args.n_iterations}")
print(f"  每迭代步数: {args.steps_per_iteration:,}")
print(f"  专家数据: {args.expert_data_path}")
print(f"  Beta调度: {args.beta_schedule} ({args.initial_beta} → {args.final_beta})")
print("="*80)


# ============================================================================
# 策略网络（与BC相同的结构）
# ============================================================================

class DAggerPolicy(nn.Module):
    """DAgger策略网络"""
    
    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 512, 256)):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.network(obs)
    
    def get_action(self, obs, deterministic=True):
        """获取动作"""
        with torch.no_grad():
            return self.forward(obs)


# ============================================================================
# MPC专家（简化版，用于提供专家标签）
# ============================================================================

class MPCExpert:
    """简化的MPC专家控制器"""
    
    def __init__(self, horizon=10, dt=0.05):
        self.horizon = horizon
        self.dt = dt
    
    def get_action(self, obs):
        """
        根据观测计算专家动作
        这里使用简化的启发式规则模拟MPC
        
        实际应用中应该使用真实的MPC求解器
        """
        # 简化版：使用基于观测的启发式策略
        # 在真实实现中，这里应该调用MPC优化求解器
        
        # 假设obs包含目标距离、角度等信息
        # 这里使用简单的比例控制作为占位符
        
        # 提取相关观测（根据实际观测空间调整）
        # 注意：这是简化版本，实际应该使用真实MPC
        
        # 生成随机专家动作作为占位符
        # TODO: 替换为真实的MPC求解
        action = torch.zeros(obs.shape[0], 3, device=obs.device)
        
        # 简单的启发式：朝向目标移动
        # 实际应该使用scripts/mpc_controller.py中的MPCController
        action[:, 0] = 0.3  # 前进速度
        action[:, 1] = 0.0  # 侧向速度
        action[:, 2] = 0.0  # 旋转速度
        
        return torch.clamp(action, -1, 1)


# ============================================================================
# DAgger数据集
# ============================================================================

class DAggerDataset:
    """DAgger聚合数据集"""
    
    def __init__(self, device):
        self.device = device
        self.observations = []
        self.actions = []
    
    def add_batch(self, obs, actions):
        """添加一批数据"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu()
        
        self.observations.append(obs)
        self.actions.append(actions)
    
    def load_expert_data(self, h5_path):
        """加载初始专家数据"""
        print(f"\n加载专家数据: {h5_path}")
        
        try:
            with h5py.File(h5_path, 'r') as f:
                obs = torch.from_numpy(f['observations'][:]).float()
                actions = torch.from_numpy(f['actions'][:]).float()
                
                self.observations.append(obs)
                self.actions.append(actions)
                
                print(f"  ✓ 加载 {len(obs)} 条专家轨迹")
        except Exception as e:
            print(f"  ⚠ 加载失败: {e}")
            print("  → 从空数据集开始")
    
    def get_all_data(self):
        """获取所有数据"""
        if not self.observations:
            return None, None
        
        all_obs = torch.cat(self.observations, dim=0)
        all_actions = torch.cat(self.actions, dim=0)
        
        return all_obs, all_actions
    
    def __len__(self):
        if not self.observations:
            return 0
        return sum(len(obs) for obs in self.observations)


# ============================================================================
# DAgger训练器
# ============================================================================

class DAggerTrainer:
    """DAgger训练器"""
    
    def __init__(
        self,
        policy,
        expert,
        dataset,
        optimizer,
        device,
        batch_size=256,
    ):
        self.policy = policy
        self.expert = expert
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
    
    def train_epoch(self):
        """训练一个epoch"""
        all_obs, all_actions = self.dataset.get_all_data()
        
        if all_obs is None:
            return {'loss': 0.0}
        
        dataset_size = len(all_obs)
        indices = torch.randperm(dataset_size)
        
        total_loss = 0
        n_batches = 0
        
        for start_idx in range(0, dataset_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_obs = all_obs[batch_indices].to(self.device)
            batch_actions = all_actions[batch_indices].to(self.device)
            
            # 前向传播
            pred_actions = self.policy(batch_obs)
            
            # 计算损失（MSE）
            loss = F.mse_loss(pred_actions, batch_actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return {'loss': total_loss / n_batches if n_batches > 0 else 0.0}


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


def get_beta(iteration, n_iterations, schedule, initial_beta, final_beta):
    """计算当前迭代的专家混合率"""
    if schedule == "constant":
        return initial_beta
    elif schedule == "linear":
        alpha = iteration / n_iterations
        return initial_beta * (1 - alpha) + final_beta * alpha
    elif schedule == "exponential":
        return initial_beta * (final_beta / initial_beta) ** (iteration / n_iterations)
    else:
        return initial_beta


def train():
    """主训练函数"""
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"dagger_{timestamp}"
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
    
    # 创建策略
    print("\n初始化DAgger策略...")
    policy = DAggerPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    
    # 创建专家
    print("初始化MPC专家...")
    expert = MPCExpert()
    
    # 创建数据集
    print("初始化DAgger数据集...")
    dataset = DAggerDataset(device)
    
    # 加载初始专家数据
    if Path(args.expert_data_path).exists():
        dataset.load_expert_data(args.expert_data_path)
    
    # 创建训练器
    trainer = DAggerTrainer(
        policy=policy,
        expert=expert,
        dataset=dataset,
        optimizer=optimizer,
        device=device,
        batch_size=args.batch_size,
    )
    
    # 训练统计
    iteration_stats = []
    
    print(f"\n{'='*80}")
    print("开始DAgger训练...")
    print(f"{'='*80}\n")
    
    for iteration in range(args.n_iterations):
        print(f"\n{'='*80}")
        print(f"DAgger迭代 {iteration + 1}/{args.n_iterations}")
        print(f"{'='*80}")
        
        # 计算当前专家混合率
        beta = get_beta(iteration, args.n_iterations, args.beta_schedule, 
                       args.initial_beta, args.final_beta)
        print(f"专家混合率 (beta): {beta:.3f}")
        print(f"当前数据集大小: {len(dataset):,}")
        
        # ======== 阶段1: 使用当前策略收集数据 ========
        print(f"\n收集数据 ({args.steps_per_iteration:,} 步)...")
        
        obs = obs_dict["policy"]
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        collected_obs = []
        collected_actions = []
        
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
        episode_rewards = []
        episode_lengths = []
        current_reward = torch.zeros(args.num_envs, device=device)
        current_length = torch.zeros(args.num_envs, device=device, dtype=torch.int)
        
        steps = 0
        pbar = tqdm(total=args.steps_per_iteration, desc="收集数据")
        
        while steps < args.steps_per_iteration:
            # 根据beta混合策略和专家动作
            with torch.no_grad():
                policy_action = policy.get_action(obs)
                expert_action = expert.get_action(obs)
                
                # 按概率混合
                use_expert = torch.rand(args.num_envs, 1, device=device) < beta
                action = torch.where(use_expert, expert_action, policy_action)
            
            # 收集专家标签（用于训练）
            collected_obs.append(obs.cpu())
            collected_actions.append(expert_action.cpu())
            
            # 执行动作
            next_obs_dict, rewards, terminated, truncated, infos = env.step(action)
            next_obs = next_obs_dict["policy"]
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
            
            # 提取当前步的奖励细节（在step之后立即提取）
            current_reward_components = extract_reward_components(env)
            
            # 统计奖励
            for i in range(args.num_envs):
                current_reward[i] += rewards[i]
                current_length[i] += 1
                
                dones = (terminated | truncated)
                if dones[i]:
                    episode_rewards.append(current_reward[i].item())
                    episode_lengths.append(current_length[i].item())
                    
                    # 保存当前episode的奖励组件
                    for key, value in current_reward_components.items():
                        if key in reward_components:
                            reward_components[key].append(value)
                    
                    current_reward[i] = 0
                    current_length[i] = 0
            
            obs = next_obs
            steps += args.num_envs
            pbar.update(args.num_envs)
        
        pbar.close()
        
        # 显示收集阶段统计
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(f"  收集阶段统计:")
            print(f"    平均奖励: {avg_reward:.2f} | 平均长度: {avg_length:.0f}")
            
            # 计算平均奖励组件
            for key, values in reward_components.items():
                if values:
                    avg_val = np.mean(values)
                    print(f"    {key}: {avg_val:.2f}", end="  ")
            print()
        
        # 将新数据添加到数据集
        if collected_obs:
            batch_obs = torch.cat(collected_obs, dim=0)
            batch_actions = torch.cat(collected_actions, dim=0)
            dataset.add_batch(batch_obs, batch_actions)
            print(f"  ✓ 添加 {len(batch_obs):,} 条新数据")
        
        # ======== 阶段2: 在聚合数据集上训练 ========
        print(f"\n训练策略 ({args.n_epochs} epochs)...")
        
        epoch_losses = []
        for epoch in range(args.n_epochs):
            metrics = trainer.train_epoch()
            epoch_losses.append(metrics['loss'])
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{args.n_epochs}: Loss = {metrics['loss']:.4f}")
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        print(f"  平均损失: {avg_loss:.4f}")
        
        # 记录统计
        iteration_stats.append({
            'iteration': iteration + 1,
            'beta': beta,
            'dataset_size': len(dataset),
            'avg_loss': float(avg_loss),
            'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            'avg_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            'reward_components': {k: float(np.mean(v)) if v else 0.0 
                                 for k, v in reward_components.items()},
        })
        
        # 保存checkpoint
        if (iteration + 1) % args.save_freq == 0 or iteration == args.n_iterations - 1:
            checkpoint_path = checkpoint_dir / f"iteration_{iteration + 1}.pt"
            torch.save({
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration + 1,
                'dataset_size': len(dataset),
            }, checkpoint_path)
            print(f"\n✓ 已保存checkpoint: {checkpoint_path}")
    
    # 保存最终模型
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, final_path)
    
    # 保存训练日志
    with open(run_dir / "training_log.json", 'w') as f:
        json.dump(iteration_stats, f, indent=2)
    
    # 保存训练摘要
    summary = {
        'n_iterations': args.n_iterations,
        'final_dataset_size': len(dataset),
        'final_loss': iteration_stats[-1]['avg_loss'] if iteration_stats else 0.0,
        'beta_schedule': args.beta_schedule,
        'initial_beta': args.initial_beta,
        'final_beta': args.final_beta,
    }
    
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("DAgger训练完成！")
    print(f"{'='*80}")
    print(f"总迭代: {args.n_iterations}")
    print(f"最终数据集大小: {len(dataset):,}")
    print(f"最终损失: {summary['final_loss']:.4f}")
    print(f"模型保存位置: {run_dir}")
    print(f"{'='*80}\n")
    
    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train()
