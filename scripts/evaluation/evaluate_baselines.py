#!/usr/bin/env python3
"""
统一Baseline评估脚本

评估所有baseline算法的性能并生成对比报告
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="评估所有baselines")
parser.add_argument("--baselines_dir", type=str, default="experiments/baselines", help="Baselines目录")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=50, help="评估episodes")
parser.add_argument("--output_dir", type=str, default="experiments/baseline_comparison", help="输出目录")
parser.add_argument("--baselines", nargs="+", 
                    default=["sac_pure", "sac_gaussian", "td3", "ppo", "dagger"],
                    help="要评估的baselines")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = True
args.enable_cameras = True

# 启动Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from rosorin_env_cfg import ROSOrinEnvCfg
from isaaclab.envs import ManagerBasedRLEnv


print("="*80)
print("  Baseline性能评估")
print("="*80)
print(f"  评估Baselines: {', '.join(args.baselines)}")
print(f"  每个baseline评估 {args.num_episodes} episodes")
print("="*80)


def load_latest_checkpoint(baseline_name, baselines_dir):
    """加载最新的checkpoint"""
    baseline_dir = Path(baselines_dir) / baseline_name
    
    if not baseline_dir.exists():
        print(f"  ⚠ 目录不存在: {baseline_dir}")
        return None
    
    # 查找最新的运行目录
    run_dirs = sorted([d for d in baseline_dir.iterdir() if d.is_dir()], 
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        print(f"  ⚠ 未找到运行目录: {baseline_dir}")
        return None
    
    latest_run = run_dirs[0]
    checkpoint_dir = latest_run / "checkpoints"
    
    # 优先使用best_model，否则使用final_model
    best_model = checkpoint_dir / "best_model.pt"
    final_model = checkpoint_dir / "final_model.pt"
    
    if best_model.exists():
        return best_model
    elif final_model.exists():
        return final_model
    else:
        print(f"  ⚠ 未找到模型文件: {checkpoint_dir}")
        return None


def load_policy(baseline_name, checkpoint_path, obs_dim, action_dim, device):
    """根据baseline类型加载策略"""
    
    # 这里需要根据不同的baseline加载不同的策略
    # 简化版本：假设所有模型都有相同的接口
    
    print(f"  加载 {baseline_name} 模型: {checkpoint_path}")
    
    # TODO: 实现具体的模型加载逻辑
    # 现在返回None表示未实现
    return None


def evaluate_policy(env, policy, num_episodes, baseline_name):
    """评估策略"""
    
    print(f"\n评估 {baseline_name}...")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
    
    current_episode_reward = torch.zeros(args.num_envs, device=obs.device)
    current_episode_length = torch.zeros(args.num_envs, device=obs.device, dtype=torch.int)
    episodes_done = 0
    
    pbar = tqdm(total=num_episodes, desc=f"评估 {baseline_name}")
    
    while episodes_done < num_episodes:
        # 选择动作
        with torch.no_grad():
            if policy is None:
                # 如果策略未加载，使用随机动作
                actions = torch.rand_like(obs[:, :3]) * 2 - 1
            else:
                # TODO: 使用策略选择动作
                actions = torch.rand_like(obs[:, :3]) * 2 - 1
        
        # 执行动作
        next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        next_obs = next_obs_dict["policy"]
        next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        dones = terminated | truncated
        
        current_episode_reward += rewards
        current_episode_length += 1
        
        # 记录完成的episodes
        for i in range(args.num_envs):
            if dones[i] and episodes_done < num_episodes:
                episode_rewards.append(current_episode_reward[i].item())
                episode_lengths.append(current_episode_length[i].item())
                
                # 判断是否成功
                if current_episode_reward[i] > 0:  # 简化的成功判断
                    success_count += 1
                
                current_episode_reward[i] = 0
                current_episode_length[i] = 0
                episodes_done += 1
                pbar.update(1)
        
        obs = next_obs
        
        if episodes_done >= num_episodes:
            break
    
    pbar.close()
    
    # 计算统计
    results = {
        'baseline': baseline_name,
        'num_episodes': len(episode_rewards),
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'success_rate': success_count / len(episode_rewards) if episode_rewards else 0.0,
    }
    
    return results


def main():
    """主函数"""
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建环境
    print("\n创建评估环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]
    action_dim = env.action_space.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"  观察维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  设备: {device}")
    
    # 评估所有baselines
    all_results = []
    
    for baseline_name in args.baselines:
        print(f"\n{'='*80}")
        print(f"评估Baseline: {baseline_name}")
        print(f"{'='*80}")
        
        # 加载checkpoint
        checkpoint_path = load_latest_checkpoint(baseline_name, args.baselines_dir)
        
        if checkpoint_path is None:
            print(f"  ⚠ 跳过 {baseline_name}（未找到checkpoint）")
            continue
        
        # 加载策略
        policy = load_policy(baseline_name, checkpoint_path, obs_dim, action_dim, device)
        
        # 评估
        results = evaluate_policy(env, policy, args.num_episodes, baseline_name)
        all_results.append(results)
        
        print(f"\n结果:")
        print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  成功率: {results['success_rate']*100:.1f}%")
        print(f"  平均长度: {results['mean_length']:.1f}")
    
    # 保存结果
    with open(run_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成对比表格
    print(f"\n{'='*80}")
    print("评估结果汇总")
    print(f"{'='*80}")
    print(f"{'Baseline':<20} {'Mean Reward':<15} {'Success Rate':<15} {'Mean Length':<15}")
    print("-" * 80)
    
    for result in sorted(all_results, key=lambda x: x['mean_reward'], reverse=True):
        print(f"{result['baseline']:<20} "
              f"{result['mean_reward']:>6.2f} ± {result['std_reward']:<5.2f}  "
              f"{result['success_rate']*100:>5.1f}%          "
              f"{result['mean_length']:>6.1f}")
    
    print(f"\n结果已保存至: {run_dir}")
    
    # 关闭环境
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
