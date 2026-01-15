#!/usr/bin/env python3
"""
SAC训练曲线可视化脚本

从checkpoint中提取训练历史数据，生成可视化曲线图
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_checkpoint_history(checkpoint_dir):
    """从所有checkpoint中提取训练历史"""
    checkpoint_dir = Path(checkpoint_dir)
    
    # 找到所有checkpoint文件（按步数排序）
    checkpoint_files = sorted(
        checkpoint_dir.glob('checkpoint_*.pt'),
        key=lambda x: int(x.stem.split('_')[1])
    )
    
    if not checkpoint_files:
        print(f"❌ 在 {checkpoint_dir} 中未找到checkpoint文件")
        return None
    
    print(f"找到 {len(checkpoint_files)} 个checkpoint文件")
    
    # 提取数据
    steps = []
    avg_rewards = []
    best_rewards = []
    
    for ckpt_file in checkpoint_files:
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu')
            step = ckpt.get('step', 0)
            avg_reward = ckpt.get('avg_reward', 0)
            best_reward = ckpt.get('best_reward', float('-inf'))
            
            steps.append(step)
            avg_rewards.append(avg_reward)
            best_rewards.append(best_reward)
            
            print(f"  ✓ {ckpt_file.name}: step={step:,}, avg_reward={avg_reward:.2f}")
        except Exception as e:
            print(f"  ⚠ 跳过 {ckpt_file.name}: {e}")
            continue
    
    # 加载best_model和final_model
    best_model_path = checkpoint_dir / 'best_model.pt'
    final_model_path = checkpoint_dir / 'final_model.pt'
    
    best_step = None
    final_step = None
    
    if best_model_path.exists():
        try:
            best_ckpt = torch.load(best_model_path, map_location='cpu')
            best_step = best_ckpt.get('step', None)
            print(f"  ✓ Best model at step: {best_step:,}")
        except:
            pass
    
    if final_model_path.exists():
        try:
            final_ckpt = torch.load(final_model_path, map_location='cpu')
            final_step = final_ckpt.get('step', None)
            print(f"  ✓ Final model at step: {final_step:,}")
        except:
            pass
    
    return {
        'steps': steps,
        'avg_rewards': avg_rewards,
        'best_rewards': best_rewards,
        'best_step': best_step,
        'final_step': final_step
    }


def plot_training_curves(history, save_dir):
    """绘制训练曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    steps = history['steps']
    avg_rewards = history['avg_rewards']
    best_rewards = history['best_rewards']
    best_step = history['best_step']
    final_step = history['final_step']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制平均奖励曲线
    ax.plot(steps, avg_rewards, 'b-', linewidth=2, label='Average Reward', alpha=0.8)
    
    # 绘制最佳奖励水平线
    max_reward = max(best_rewards)
    ax.axhline(y=max_reward, color='g', linestyle='--', linewidth=1.5, 
               label=f'Best Reward: {max_reward:.2f}', alpha=0.7)
    
    # 标记best model位置
    if best_step and best_step in steps:
        idx = steps.index(best_step)
        ax.plot(best_step, avg_rewards[idx], 'go', markersize=12, 
                label=f'Best Model (step {best_step:,})', zorder=5)
    
    # 标记final model位置
    if final_step and final_step in steps:
        idx = steps.index(final_step)
        ax.plot(final_step, avg_rewards[idx], 'ro', markersize=12, 
                label=f'Final Model (step {final_step:,})', zorder=5)
    
    # 设置标题和标签
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title('SAC Training Progress', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    # 格式化x轴（显示千位分隔符）
    ax.ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = save_dir / 'sac_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 训练曲线已保存到: {save_path}")
    
    # 保存JSON数据
    json_path = save_dir / 'sac_training_history.json'
    with open(json_path, 'w') as f:
        json.dump({
            'steps': steps,
            'avg_rewards': avg_rewards,
            'best_rewards': best_rewards,
            'best_step': best_step,
            'final_step': final_step,
            'max_avg_reward': max(avg_rewards),
            'final_avg_reward': avg_rewards[-1] if avg_rewards else 0,
            'improvement': max(avg_rewards) - avg_rewards[0] if len(avg_rewards) > 0 else 0
        }, f, indent=2)
    print(f"✓ 训练历史数据已保存到: {json_path}")
    
    plt.show()


def generate_summary(history):
    """生成训练总结报告"""
    steps = history['steps']
    avg_rewards = history['avg_rewards']
    best_rewards = history['best_rewards']
    
    if not steps:
        print("❌ 没有数据可分析")
        return
    
    print("\n" + "="*80)
    print("  SAC训练总结报告")
    print("="*80)
    print(f"总训练步数: {steps[-1]:,}")
    print(f"Checkpoint数量: {len(steps)}")
    print(f"\n奖励统计:")
    print(f"  初始平均奖励: {avg_rewards[0]:.2f}")
    print(f"  最终平均奖励: {avg_rewards[-1]:.2f}")
    print(f"  最佳平均奖励: {max(avg_rewards):.2f} (step {steps[avg_rewards.index(max(avg_rewards))]:,})")
    print(f"  性能提升: {max(avg_rewards) - avg_rewards[0]:.2f} (+{(max(avg_rewards) - avg_rewards[0]) / abs(avg_rewards[0]) * 100:.1f}%)")
    
    # 分析趋势
    print(f"\n训练趋势:")
    first_half_avg = np.mean(avg_rewards[:len(avg_rewards)//2])
    second_half_avg = np.mean(avg_rewards[len(avg_rewards)//2:])
    
    if second_half_avg > first_half_avg:
        print(f"  ✅ 持续改进 (前半段: {first_half_avg:.2f} → 后半段: {second_half_avg:.2f})")
    else:
        print(f"  ⚠️ 性能下降 (前半段: {first_half_avg:.2f} → 后半段: {second_half_avg:.2f})")
    
    # 最佳模型位置
    best_idx = avg_rewards.index(max(avg_rewards))
    if best_idx == len(avg_rewards) - 1:
        print(f"  ✅ 最佳模型在训练终点")
    else:
        print(f"  ⚠️ 最佳模型在中途 (step {steps[best_idx]:,}/{steps[-1]:,})")
        print(f"     建议使用checkpoint_{steps[best_idx]}.pt而非final_model.pt")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="可视化SAC训练曲线")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                        help="Checkpoint目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="图表输出目录（默认与checkpoint_dir相同）")
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.checkpoint_dir
    
    # 加载训练历史
    print("加载训练历史...")
    history = load_checkpoint_history(args.checkpoint_dir)
    
    if history is None or not history['steps']:
        print("❌ 无法加载训练历史数据")
        return
    
    # 生成总结报告
    generate_summary(history)
    
    # 绘制训练曲线
    print("\n生成训练曲线图...")
    plot_training_curves(history, args.output_dir)
    
    print("\n✅ 可视化完成！")


if __name__ == "__main__":
    main()
