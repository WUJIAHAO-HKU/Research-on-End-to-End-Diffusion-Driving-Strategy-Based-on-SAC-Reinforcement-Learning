#!/usr/bin/env python3
"""
从训练日志手动创建SAC训练历史图表

基于终端输出的训练进度数据
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 从终端日志提取的训练数据
training_data = {
    'steps': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    'avg_rewards': [-8.55, 16.13, 18.06, 6.92, -1.19, 22.04, 22.69, 33.60, 31.34, 8.15],
    'q_values': [-99.60, -76.11, -63.35, -65.78, -70.40, -69.55, -94.61, -108.43, -110.56, -140.58],
    'actor_losses': [99.597, 76.112, 63.349, 65.779, 70.400, 69.552, 94.608, 108.435, 110.558, 140.585],
    'total_episodes': 68,
    'best_step': 80000,
    'best_reward': 33.60
}

def create_training_curves():
    """创建训练曲线图"""
    
    steps = training_data['steps']
    avg_rewards = training_data['avg_rewards']
    q_values = training_data['q_values']
    actor_losses = training_data['actor_losses']
    
    # 创建3行1列的子图
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. 平均奖励曲线
    ax = axes[0]
    ax.plot(steps, avg_rewards, 'b-o', linewidth=2, markersize=8, label='Average Reward')
    
    # 标记最佳点
    best_idx = avg_rewards.index(max(avg_rewards))
    ax.plot(steps[best_idx], avg_rewards[best_idx], 'go', markersize=15, 
            label=f'Best Model (step {steps[best_idx]:,}, reward {avg_rewards[best_idx]:.2f})', zorder=5)
    
    # 标记终点
    ax.plot(steps[-1], avg_rewards[-1], 'ro', markersize=15,
            label=f'Final Model (step {steps[-1]:,}, reward {avg_rewards[-1]:.2f})', zorder=5)
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('SAC Training Progress - Average Reward', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 2. Q值曲线
    ax = axes[1]
    ax.plot(steps, q_values, 'r-s', linewidth=2, markersize=8, label='Q Value Estimate')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Q Value', fontsize=12, fontweight='bold')
    ax.set_title('Critic Q Value Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 3. Actor Loss曲线
    ax = axes[2]
    ax.plot(steps, actor_losses, 'g-^', linewidth=2, markersize=8, label='Actor Loss')
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actor Loss', fontsize=12, fontweight='bold')
    ax.set_title('Actor Policy Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('experiments/sac_training/sac_training_20251229_121515')
    save_path = output_dir / 'sac_training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存到: {save_path}")
    
    # 保存JSON数据
    json_path = output_dir / 'sac_training_history.json'
    with open(json_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"✓ 训练历史数据已保存到: {json_path}")
    
    # 生成分析报告
    print("\n" + "="*80)
    print("  SAC训练分析报告")
    print("="*80)
    print(f"总训练步数: {steps[-1]:,}")
    print(f"总Episodes: {training_data['total_episodes']}")
    print(f"\n奖励统计:")
    print(f"  初始奖励: {avg_rewards[0]:.2f}")
    print(f"  最佳奖励: {max(avg_rewards):.2f} (step {steps[avg_rewards.index(max(avg_rewards))]:,})")
    print(f"  最终奖励: {avg_rewards[-1]:.2f}")
    print(f"  提升幅度: {max(avg_rewards) - avg_rewards[0]:.2f} (+{(max(avg_rewards) - avg_rewards[0]) / abs(avg_rewards[0]) * 100:.1f}%)")
    
    print(f"\n关键发现:")
    print(f"  ✅ 快速提升: 前20k步从-8.55提升到+16.13")
    print(f"  🌟 峰值性能: 80k步达到最佳奖励33.60")
    print(f"  ⚠️ 后期退化: 100k步下降到8.15（性能下降76%）")
    print(f"  📊 Q值恶化: -63.35 → -140.58（过估计累积）")
    
    print(f"\n建议:")
    print(f"  1. 使用80k步的best_model.pt（而非final_model.pt）")
    print(f"  2. 后期性能崩溃可能原因：学习率过高、Q值过估计")
    print(f"  3. 改进方向：降低后期学习率、增加target network更新频率")
    print("="*80 + "\n")
    
    plt.show()


if __name__ == "__main__":
    create_training_curves()
