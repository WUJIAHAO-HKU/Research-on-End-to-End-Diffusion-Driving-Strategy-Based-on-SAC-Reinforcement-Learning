"""
简化的BC训练脚本

直接使用MPC专家数据训练策略网络
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json


class MPCDataset(Dataset):
    """MPC专家演示数据集"""
    
    def __init__(self, h5_files: list, normalize=True):
        """
        Args:
            h5_files: HDF5文件路径列表
            normalize: 是否归一化数据
        """
        self.data = []
        self.normalize = normalize
        
        # 加载所有数据
        print("\n加载数据集...")
        for h5_file in h5_files:
            print(f"  - {h5_file}")
            with h5py.File(h5_file, 'r') as f:
                num_episodes = f.attrs['num_episodes']
                difficulty = f.attrs['difficulty']
                
                for i in range(num_episodes):
                    episode = f[f'episode_{i}']
                    obs = episode['observations'][:]
                    actions = episode['actions'][:]
                    
                    # 每个样本: (observation, action)
                    for t in range(len(obs)):
                        self.data.append({
                            'obs': obs[t],
                            'action': actions[t],
                            'difficulty': difficulty
                        })
        
        print(f"✓ 总样本数: {len(self.data):,}")
        
        # 计算归一化参数
        if self.normalize:
            self._compute_normalization()
    
    def _compute_normalization(self):
        """计算归一化参数"""
        print("\n计算归一化参数...")
        
        # 采样部分数据计算统计量
        sample_size = min(10000, len(self.data))
        indices = np.random.choice(len(self.data), sample_size, replace=False)
        
        obs_samples = np.stack([self.data[i]['obs'] for i in indices])
        action_samples = np.stack([self.data[i]['action'] for i in indices])
        
        # 处理inf值（深度图像中的inf）
        obs_samples = np.nan_to_num(obs_samples, nan=0.0, posinf=10.0, neginf=0.0)
        
        self.obs_mean = obs_samples.mean(axis=0).astype(np.float32)
        self.obs_std = obs_samples.std(axis=0).astype(np.float32) + 1e-8
        
        self.action_mean = action_samples.mean(axis=0).astype(np.float32)
        self.action_std = action_samples.std(axis=0).astype(np.float32) + 1e-8
        
        print(f"  观测维度: {len(self.obs_mean)}")
        print(f"  动作维度: {len(self.action_mean)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        obs = item['obs'].astype(np.float32)
        action = item['action'].astype(np.float32)
        
        # 处理inf值
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=0.0)
        
        if self.normalize:
            obs = (obs - self.obs_mean) / self.obs_std
            action = (action - self.action_mean) / self.action_std
        
        return torch.from_numpy(obs), torch.from_numpy(action)


class BCPolicy(nn.Module):
    """简单的BC策略网络 - MLP"""
    
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


class BCTrainer:
    """BC训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dims = args.hidden_dims  # 保存hidden_dims配置
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"bc_training_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        print("\n" + "="*80)
        print("  BC训练配置")
        print("="*80)
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  Batch大小: {args.batch_size}")
        print(f"  学习率: {args.lr}")
        print(f"  Epochs: {args.epochs}")
        print("="*80)
    
    def train(self, train_loader, val_loader, model, optimizer, scheduler):
        """训练循环"""
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            for obs, actions in pbar:
                obs = obs.to(self.device)
                actions = actions.to(self.device)
                
                # 前向传播
                pred_actions = model(obs)
                loss = criterion(pred_actions, actions)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for obs, actions in val_loader:
                    obs = obs.to(self.device)
                    actions = actions.to(self.device)
                    
                    pred_actions = model(obs)
                    loss = criterion(pred_actions, actions)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # 日志
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # 获取原始数据集（处理Subset）
                original_dataset = train_loader.dataset
                if hasattr(original_dataset, 'dataset'):
                    original_dataset = original_dataset.dataset
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'obs_mean': original_dataset.obs_mean,
                    'obs_std': original_dataset.obs_std,
                    'action_mean': original_dataset.action_mean,
                    'action_std': original_dataset.action_std,
                    'hidden_dims': self.hidden_dims,  # 保存网络架构配置
                }
                torch.save(checkpoint, self.output_dir / "best_model.pt")
                print(f"  ✓ 已保存最佳模型 (val_loss: {val_loss:.6f})")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        # 训练结束后绘制曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制并保存训练曲线"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 无GUI后端
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. Loss Curve
        axes[0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('BC Training Loss Curve', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Log scale
        
        # 2. Learning Rate Curve
        axes[1].plot(epochs, self.learning_rates, 'g-', label='Learning Rate', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curves saved: {save_path}")
        plt.close()
        
        # Save training history data (for future analysis)
        import json
        history = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'learning_rates': [float(x) for x in self.learning_rates],
            'best_epoch': int(self.val_losses.index(min(self.val_losses)) + 1),
            'best_val_loss': float(min(self.val_losses)),
            'final_train_loss': float(self.train_losses[-1]),
            'final_val_loss': float(self.val_losses[-1]),
        }
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Training history saved: {self.output_dir / 'training_history.json'}")


def main():
    parser = argparse.ArgumentParser(description="BC训练")
    
    # 数据路径
    parser.add_argument("--easy_data", type=str, 
                       default="data/demonstrations/rosorin_mpc_demos_easy_20251226_152714.h5")
    parser.add_argument("--medium_data", type=str,
                       default="data/demonstrations/rosorin_mpc_demos_medium_20251228_032231.h5")
    parser.add_argument("--hard_data", type=str,
                       default="data/demonstrations/rosorin_mpc_demos_hard_20251228_044534.h5")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    
    # 模型参数
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[512, 512, 256])
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="experiments/bc_training")
    
    args = parser.parse_args()
    
    # 收集数据文件
    data_files = []
    for data_path in [args.easy_data, args.medium_data, args.hard_data]:
        if Path(data_path).exists():
            data_files.append(data_path)
        else:
            print(f"⚠ 警告: 文件不存在 {data_path}")
    
    if not data_files:
        raise ValueError("没有找到任何数据文件！")
    
    print(f"\n使用数据文件: {len(data_files)}个")
    
    # 创建数据集
    full_dataset = MPCDataset(data_files, normalize=True)
    
    # 划分训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset):,} 样本")
    print(f"  验证集: {len(val_dataset):,} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    obs_dim = len(full_dataset.obs_mean)
    action_dim = len(full_dataset.action_mean)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCPolicy(obs_dim, action_dim, hidden_dims=args.hidden_dims).to(device)
    
    print(f"\n模型参数:")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 开始训练
    trainer = BCTrainer(args)
    trainer.train(train_loader, val_loader, model, optimizer, scheduler)
    
    print("\n" + "="*80)
    print("  ✅ BC训练完成!")
    print("="*80)
    print(f"  最佳模型: {trainer.output_dir / 'best_model.pt'}")
    print("="*80)


if __name__ == "__main__":
    main()
