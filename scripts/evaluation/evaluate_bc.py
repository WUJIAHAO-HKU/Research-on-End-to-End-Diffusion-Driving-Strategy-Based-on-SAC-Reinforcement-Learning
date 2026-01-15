"""
BC模型评估脚本

在Isaac Lab环境中评估训练好的BC策略
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import pickle

from isaaclab.app import AppLauncher

# 解析参数
parser = argparse.ArgumentParser(description="BC模型评估")
parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--num_episodes", type=int, default=20, help="评估episode数量")
parser.add_argument("--max_steps", type=int, default=500, help="每个episode最大步数")
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


class BCPolicy(nn.Module):
    """BC策略网络（需要与训练时一致）"""
    
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


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"\n加载模型: {checkpoint_path}")
    
    # 修复numpy兼容性问题
    # numpy 2.x中使用numpy._core，但在某些环境中可能不存在
    # 创建兼容层
    if not hasattr(np, '_core'):
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.multiarray'] = _core.multiarray
        sys.modules['numpy._core.umath'] = _core.umath
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            # 如果仍然失败，使用pickle直接加载并手动重映射
            print("  检测到numpy版本兼容性问题，使用兼容模式加载...")
            
            # 创建自定义unpickler来重映射numpy模块
            class NumpyCompatUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module.startswith('numpy._core'):
                        module = module.replace('numpy._core', 'numpy.core')
                    return super().find_class(module, name)
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint = NumpyCompatUnpickler(f).load()
        else:
            raise
    
    # 获取模型维度（从归一化参数推断）
    obs_mean_data = checkpoint['obs_mean']
    action_mean_data = checkpoint['action_mean']
    
    # 确保数据是numpy数组
    if isinstance(obs_mean_data, torch.Tensor):
        obs_mean_data = obs_mean_data.cpu().numpy()
    if isinstance(action_mean_data, torch.Tensor):
        action_mean_data = action_mean_data.cpu().numpy()
    
    obs_dim = len(obs_mean_data)
    action_dim = len(action_mean_data)
    
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  训练Epoch: {checkpoint['epoch']}")
    print(f"  验证损失: {checkpoint['val_loss']:.6f}")
    
    # 从checkpoint读取hidden_dims配置（如果存在）
    hidden_dims = checkpoint.get('hidden_dims', [512, 512, 256])
    print(f"  隐藏层维度: {hidden_dims}")
    
    # 创建模型
    model = BCPolicy(obs_dim, action_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 归一化参数 - 确保转换为numpy数组
    obs_mean_np = checkpoint['obs_mean']
    obs_std_np = checkpoint['obs_std']
    action_mean_np = checkpoint['action_mean']
    action_std_np = checkpoint['action_std']
    
    # 如果是tensor，转换为numpy
    if isinstance(obs_mean_np, torch.Tensor):
        obs_mean_np = obs_mean_np.cpu().numpy()
    if isinstance(obs_std_np, torch.Tensor):
        obs_std_np = obs_std_np.cpu().numpy()
    if isinstance(action_mean_np, torch.Tensor):
        action_mean_np = action_mean_np.cpu().numpy()
    if isinstance(action_std_np, torch.Tensor):
        action_std_np = action_std_np.cpu().numpy()
    
    obs_mean = torch.from_numpy(obs_mean_np).to(device)
    obs_std = torch.from_numpy(obs_std_np).to(device)
    action_mean = torch.from_numpy(action_mean_np).to(device)
    action_std = torch.from_numpy(action_std_np).to(device)
    
    return model, obs_mean, obs_std, action_mean, action_std


def evaluate(env, model, obs_mean, obs_std, action_mean, action_std, num_episodes, max_steps):
    """评估模型"""
    
    device = next(model.parameters()).device
    num_envs = env.num_envs
    
    # 统计信息
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    episodes_done = 0
    env_episode_rewards = [0.0] * num_envs
    env_episode_lengths = [0] * num_envs
    
    # 重置环境
    obs_dict, _ = env.reset()
    
    print("\n开始评估...")
    pbar = tqdm(total=num_episodes, desc="评估进度")
    
    for step in range(max_steps * num_episodes // num_envs):
        # 获取观测
        obs = obs_dict["policy"]
        
        # 处理inf值
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        # 归一化
        obs_normalized = (obs - obs_mean) / obs_std
        
        # 预测动作
        with torch.no_grad():
            actions_normalized = model(obs_normalized)
        
        # 反归一化
        actions = actions_normalized * action_std + action_mean
        
        # 执行动作
        obs_dict, rewards, dones, truncated, infos = env.step(actions)
        
        # 累积统计
        for i in range(num_envs):
            env_episode_rewards[i] += rewards[i].item()
            env_episode_lengths[i] += 1
            
            if dones[i] or truncated[i]:
                episode_rewards.append(env_episode_rewards[i])
                episode_lengths.append(env_episode_lengths[i])
                
                # 判断成功（简单判断：episode长度接近最大步数）
                if env_episode_lengths[i] >= max_steps * 0.8:
                    success_count += 1
                
                episodes_done += 1
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f"{env_episode_rewards[i]:.2f}",
                    'length': env_episode_lengths[i]
                })
                
                env_episode_rewards[i] = 0.0
                env_episode_lengths[i] = 0
                
                if episodes_done >= num_episodes:
                    break
        
        if episodes_done >= num_episodes:
            break
    
    pbar.close()
    
    # 计算统计
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes,
    }


def main():
    """主函数"""
    
    print("\n" + "="*80)
    print("  BC模型评估")
    print("="*80)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  并行环境: {args.num_envs}")
    print(f"  评估Episodes: {args.num_episodes}")
    print("="*80)
    
    # 检查checkpoint是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载模型
    model, obs_mean, obs_std, action_mean, action_std = load_model(
        checkpoint_path, device
    )
    
    # 创建环境
    print("\n创建评估环境...")
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 评估
    results = evaluate(
        env, model, obs_mean, obs_std, action_mean, action_std,
        args.num_episodes, args.max_steps
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("  📊 评估结果")
    print("="*80)
    print(f"  Episodes: {results['num_episodes']}")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  平均长度: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  成功率: {results['success_rate']*100:.1f}%")
    print("="*80 + "\n")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
