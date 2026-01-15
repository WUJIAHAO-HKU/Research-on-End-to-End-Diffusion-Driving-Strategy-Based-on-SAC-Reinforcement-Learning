#!/usr/bin/env python3
"""
TD3训练脚本 (Twin Delayed Deep Deterministic Policy Gradient)

Baseline: TD3算法
用于对比SAC-Diffusion的性能
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

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="TD3训练")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--total_steps", type=int, default=1000000, help="总训练步数")
parser.add_argument("--output_dir", type=str, default="experiments/baselines/td3", help="输出目录")
parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
parser.add_argument("--buffer_size", type=int, default=50000, help="回放池大小")
parser.add_argument("--save_freq", type=int, default=10000, help="保存频率")
parser.add_argument("--log_freq", type=int, default=100, help="日志频率")
parser.add_argument("--warmup_steps", type=int, default=1000, help="预热步数")
parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
parser.add_argument("--policy_noise", type=float, default=0.2, help="目标策略平滑噪声")
parser.add_argument("--noise_clip", type=float, default=0.5, help="噪声裁剪范围")
parser.add_argument("--policy_delay", type=int, default=2, help="策略延迟更新频率")
parser.add_argument("--expl_noise", type=float, default=0.1, help="探索噪声")
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
from baselines.td3_agent import TD3Agent


print("="*80)
print("  TD3驾驶策略训练")
print("="*80)
print(f"  并行环境: {args.num_envs}")
print(f"  总步数: {args.total_steps:,}")
print(f"  批次大小: {args.batch_size}")
print(f"  策略延迟: {args.policy_delay}")
print("="*80)


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
        
        return {
            'states': batch_obs,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'next_states': batch_next_obs,
            'dones': batch_dones,
        }
    
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
    run_dir = output_dir / f"td3_{timestamp}"
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
    
    # 创建TD3智能体
    print("\n初始化TD3智能体...")
    agent = TD3Agent(
        state_dim=obs_dim,
        action_dim=action_dim,
        max_action=1.0,
        actor_lr=args.lr,
        critic_lr=args.lr,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        device=device,
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
    
    # 日志数据
    log_data = {
        'steps': [],
        'rewards': [],
        'actor_loss': [],
        'critic1_loss': [],
        'critic2_loss': [],
        'reward_components': [],
    }
    
    print(f"\n{'='*80}")
    print("开始训练TD3...")
    print(f"{'='*80}\n")
    
    pbar = tqdm(total=args.total_steps, desc="训练进度")
    
    while total_steps < args.total_steps:
        # 选择动作
        if total_steps < args.warmup_steps:
            # 预热阶段：随机探索
            actions = torch.rand(args.num_envs, action_dim, device=device) * 2 - 1
        else:
            # 使用TD3策略
            with torch.no_grad():
                obs_np = obs.cpu().numpy()
                actions_list = []
                for i in range(args.num_envs):
                    action = agent.select_action(obs_np[i], noise=args.expl_noise, eval_mode=False)
                    actions_list.append(action)
                actions = torch.from_numpy(np.stack(actions_list)).float().to(device)
        
        # 执行动作
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        next_obs = next_obs_dict["policy"]
        next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
        

        # 提取当前步的奖励细节（在step之后立即提取）
        current_reward_components = extract_reward_components(env)
        dones = terminated | truncated
        
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
                    'c1': f"{metrics['critic1_loss']:.3f}",
                    'c2': f"{metrics['critic2_loss']:.3f}",
                    **comp_str,
                })
                
                log_data['steps'].append(total_steps)
                log_data['rewards'].append(avg_reward)
                log_data['actor_loss'].append(metrics['actor_loss'])
                log_data['critic1_loss'].append(metrics['critic1_loss'])
                log_data['critic2_loss'].append(metrics['critic2_loss'])
                
                # 保存奖励组件
                comp_avg = {k: float(np.mean(v[-10:])) if v else 0.0 
                           for k, v in reward_components.items()}
                log_data['reward_components'].append(comp_avg)
                
                # 每1000步打印详细信息
                if total_steps % 1000 == 0:
                    print(f"\n[Step {total_steps:,}] 奖励细节:")
                    print(f"  总奖励: {avg_reward:.2f} | Episode长度: {avg_length:.0f}")
                    print(f"  进度: {comp_avg.get('progress', 0):.2f} | 到达: {comp_avg.get('goal_reached', 0):.2f}")
                    print(f"  速度: {comp_avg.get('velocity', 0):.2f} | 朝向: {comp_avg.get('orientation', 0):.2f}")
        
        # 保存checkpoint
        if total_steps % args.save_freq == 0 and total_steps > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_steps}.pt"
            agent.save(str(checkpoint_path))
            
            # 保存训练日志
            with open(run_dir / "training_log.json", 'w') as f:
                json.dump(log_data, f, indent=2)
            
            # 保存最佳模型
            if episode_rewards and np.mean(episode_rewards[-20:]) > best_reward:
                best_reward = np.mean(episode_rewards[-20:])
                best_path = checkpoint_dir / "best_model.pt"
                agent.save(str(best_path))
                print(f"\n✓ 新的最佳模型 (平均奖励: {best_reward:.2f})")
    
    pbar.close()
    
    # 保存最终模型
    final_path = checkpoint_dir / "final_model.pt"
    agent.save(str(final_path))
    
    # 保存训练摘要
    summary = {
        'total_steps': total_steps,
        'total_episodes': len(episode_rewards),
        'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
        'best_reward': float(best_reward),
        'mean_length': float(np.mean(episode_lengths)) if episode_lengths else 0.0,
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
