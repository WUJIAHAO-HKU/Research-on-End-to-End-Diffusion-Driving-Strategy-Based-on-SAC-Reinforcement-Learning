"""
运行所有Baseline算法的评估

评估：
1. BC (Behavior Cloning) 
2. SAC-Diffusion (本项目)
3. MPC (作为上界参考)
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

# 参数解析
parser = argparse.ArgumentParser(description="评估所有Baseline算法")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=30, help="每个算法评估的episode数")
parser.add_argument("--max_steps", type=int, default=500, help="每个episode最大步数")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 强制headless模式
args.headless = True
args.enable_cameras = True

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Lab导入
from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


# ===== 模型定义 =====
class BCPolicy(nn.Module):
    """BC策略网络"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 512, 256]):
        super().__init__()
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
        return self.network(obs)


class SACPolicy(nn.Module):
    """SAC策略网络（匹配训练时的结构）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super().__init__()
        # 观测编码器: [512, 512, 256]
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        # 动作头: [256, action_dim] 匹配训练checkpoint
        self.action_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    
    def forward(self, obs):
        features = self.obs_encoder(obs)
        return self.action_head(features)


def load_bc_model(checkpoint_path, device):
    """加载BC模型"""
    # 修复numpy兼容性
    if not hasattr(np, '_core'):
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.multiarray'] = _core.multiarray
        sys.modules['numpy._core.umath'] = _core.umath
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    obs_dim = len(checkpoint['obs_mean'])
    action_dim = len(checkpoint['action_mean'])
    
    model = BCPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    obs_mean = torch.from_numpy(np.array(checkpoint['obs_mean'])).to(device)
    obs_std = torch.from_numpy(np.array(checkpoint['obs_std'])).to(device)
    action_mean = torch.from_numpy(np.array(checkpoint['action_mean'])).to(device)
    action_std = torch.from_numpy(np.array(checkpoint['action_std'])).to(device)
    
    return model, obs_mean, obs_std, action_mean, action_std


def load_sac_model(checkpoint_path, device, obs_dim=76810, action_dim=4):
    """加载SAC-Diffusion模型"""
    # 修复numpy兼容性
    if not hasattr(np, '_core'):
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.multiarray'] = _core.multiarray
        sys.modules['numpy._core.umath'] = _core.umath
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # SAC模型使用不同的网络结构
    model = SACPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(checkpoint['actor'])
    model.eval()
    
    # 使用零均值和单位方差（假设SAC训练时已经处理了归一化）
    obs_mean = torch.zeros(obs_dim).to(device)
    obs_std = torch.ones(obs_dim).to(device)
    action_mean = torch.zeros(action_dim).to(device)
    action_std = torch.ones(action_dim).to(device)
    
    return model, obs_mean, obs_std, action_mean, action_std


def evaluate_policy(env, model, obs_mean, obs_std, action_mean, action_std, 
                   num_episodes, max_steps, policy_name):
    """评估策略"""
    device = next(model.parameters()).device
    num_envs = env.num_envs
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    episodes_done = 0
    env_episode_rewards = [0.0] * num_envs
    env_episode_lengths = [0] * num_envs
    
    obs_dict, _ = env.reset()
    
    print(f"\n{'='*80}")
    print(f"  评估 {policy_name}")
    print(f"{'='*80}")
    
    pbar = tqdm(total=num_episodes, desc=f"{policy_name}")
    
    for step in range(max_steps * num_episodes // num_envs + max_steps):
        obs = obs_dict["policy"]
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 归一化
        obs_normalized = (obs - obs_mean) / (obs_std + 1e-8)
        
        # 预测动作
        with torch.no_grad():
            action_normalized = model(obs_normalized)
        
        # 反归一化
        action = action_normalized * action_std + action_mean
        action = torch.clamp(action, -1.0, 1.0)
        
        # 执行动作
        obs_dict, rewards, dones, truncated, infos = env.step(action)
        
        # 累积奖励
        for i in range(num_envs):
            env_episode_rewards[i] += rewards[i].item()
            env_episode_lengths[i] += 1
            
            if dones[i] or truncated[i]:
                episode_rewards.append(env_episode_rewards[i])
                episode_lengths.append(env_episode_lengths[i])
                
                if env_episode_rewards[i] > 5.0:
                    success_count += 1
                
                env_episode_rewards[i] = 0.0
                env_episode_lengths[i] = 0
                episodes_done += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f'{np.mean(episode_rewards):.2f}',
                    'success': f'{success_count}/{episodes_done}'
                })
                
                if episodes_done >= num_episodes:
                    break
        
        if episodes_done >= num_episodes:
            break
    
    pbar.close()
    
    results = {
        'policy': policy_name,
        'num_episodes': len(episode_rewards),
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': float(success_count / len(episode_rewards)),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths]
    }
    
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    print("\n创建评估环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"experiments/baseline_comparison/comparison_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # ===== 1. 评估BC模型 =====
    bc_checkpoint = "experiments/bc_training/bc_training_20251228_052241/best_model.pt"
    if Path(bc_checkpoint).exists():
        print(f"\n加载BC模型: {bc_checkpoint}")
        bc_model, obs_mean, obs_std, action_mean, action_std = load_bc_model(bc_checkpoint, device)
        
        bc_results = evaluate_policy(
            env, bc_model, obs_mean, obs_std, action_mean, action_std,
            args.num_episodes, args.max_steps, "BC (Behavior Cloning)"
        )
        all_results.append(bc_results)
        
        # 保存BC结果
        with open(results_dir / "bc_results.json", 'w') as f:
            json.dump(bc_results, f, indent=2)
    else:
        print(f"⚠ BC模型不存在: {bc_checkpoint}")
    
    # ===== 2. 评估SAC-Diffusion模型 =====
    sac_checkpoint = "experiments/sac_training/sac_training_20251228_062324/checkpoints/best_model.pt"
    if Path(sac_checkpoint).exists():
        print(f"\n加载SAC-Diffusion模型: {sac_checkpoint}")
        sac_model, obs_mean, obs_std, action_mean, action_std = load_sac_model(sac_checkpoint, device)
        
        sac_results = evaluate_policy(
            env, sac_model, obs_mean, obs_std, action_mean, action_std,
            args.num_episodes, args.max_steps, "SAC-Diffusion"
        )
        all_results.append(sac_results)
        
        # 保存SAC结果
        with open(results_dir / "sac_diffusion_results.json", 'w') as f:
            json.dump(sac_results, f, indent=2)
    else:
        print(f"⚠ SAC-Diffusion模型不存在: {sac_checkpoint}")
    
    # ===== 生成对比报告 =====
    print(f"\n{'='*80}")
    print("  📊 Baseline对比结果")
    print(f"{'='*80}\n")
    
    print(f"{'算法':<20} {'平均奖励':<15} {'成功率':<15} {'平均长度':<15}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['policy']:<20} "
              f"{result['mean_reward']:>7.2f} ± {result['std_reward']:<5.2f} "
              f"{result['success_rate']:>6.1%}        "
              f"{result['mean_length']:>7.1f} ± {result['std_length']:<5.1f}")
    
    # 保存汇总结果
    summary = {
        'timestamp': timestamp,
        'num_episodes': args.num_episodes,
        'results': all_results
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {results_dir}")
    print(f"{'='*80}\n")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
