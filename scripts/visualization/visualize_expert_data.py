#!/usr/bin/env python3
"""
可视化MPC专家数据 - 显示专家轨迹和行为模式

功能：
1. 加载HDF5专家数据文件
2. 可视化轨迹路径（起点→目标点）
3. 显示速度分布、动作分布
4. 回放专家轨迹（在Isaac Sim中）
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def analyze_expert_data(data_path):
    """Analyze expert data statistics"""
    print(f"\n{'='*80}")
    print(f"  Analyzing Expert Data: {Path(data_path).name}")
    print(f"{'='*80}\n")
    
    with h5py.File(data_path, 'r') as f:
        # Basic information
        num_episodes = len(f.keys())
        print(f"📊 Dataset Statistics:")
        print(f"  Total Trajectories: {num_episodes}")
        
        total_steps = 0
        episode_lengths = []
        all_actions = []
        all_rewards = []
        all_positions = []  # 存储机器人位置用于轨迹可视化
        
        # 遍历所有episode
        for ep_key in f.keys():
            ep_group = f[ep_key]
            
            observations = ep_group['observations'][:]
            actions = ep_group['actions'][:]
            rewards = ep_group['rewards'][:]
            
            episode_lengths.append(len(actions))
            total_steps += len(actions)
            all_actions.append(actions)
            all_rewards.extend(rewards)
            
            # 如果观测包含位置信息，提取用于轨迹可视化
            # 注意：我们的观测是 (base_lin_vel:3, base_ang_vel:3, joint_vel:4, camera_rgb:57600, camera_depth:19200)
            # 位置信息可以通过累积速度估算，或者如果有goal_relative_position可以推算
            
        all_actions = np.concatenate(all_actions, axis=0)
        
        print(f"  Total Steps: {total_steps}")
        print(f"  Average Trajectory Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"  Shortest/Longest: {min(episode_lengths)} / {max(episode_lengths)}\n")
        
        print(f"🎯 Performance Metrics:")
        print(f"  Average Reward: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
        print(f"  Reward Range: [{np.min(all_rewards):.3f}, {np.max(all_rewards):.3f}]\n")
        
        print(f"🎮 Action Statistics (4 Wheel Velocities):")
        for i in range(4):
            wheel_name = ['Front-Left', 'Front-Right', 'Rear-Left', 'Rear-Right'][i]
            print(f"  {wheel_name}: {all_actions[:, i].mean():.3f} ± {all_actions[:, i].std():.3f} "
                  f"[{all_actions[:, i].min():.3f}, {all_actions[:, i].max():.3f}]")
    
    return {
        'num_episodes': num_episodes,
        'total_steps': total_steps,
        'episode_lengths': episode_lengths,
        'all_actions': all_actions,
        'all_rewards': all_rewards
    }


def visualize_statistics(stats):
    """Visualize statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MPC Expert Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Trajectory length distribution
    ax = axes[0, 0]
    ax.hist(stats['episode_lengths'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(stats['episode_lengths']), color='red', linestyle='--', 
               label=f"Mean: {np.mean(stats['episode_lengths']):.1f}")
    ax.set_xlabel('Trajectory Length (steps)', fontsize=11)
    ax.set_ylabel('Trajectory Count', fontsize=11)
    ax.set_title('Trajectory Length Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Reward distribution
    ax = axes[0, 1]
    ax.hist(stats['all_rewards'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(stats['all_rewards']), color='red', linestyle='--',
               label=f"Mean: {np.mean(stats['all_rewards']):.3f}")
    ax.set_xlabel('Reward Value', fontsize=11)
    ax.set_ylabel('Step Count', fontsize=11)
    ax.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Action distribution (4 wheels)
    ax = axes[1, 0]
    wheel_names = ['Front-Left', 'Front-Right', 'Rear-Left', 'Rear-Right']
    for i in range(4):
        ax.hist(stats['all_actions'][:, i], bins=40, alpha=0.5, 
                label=f'{wheel_names[i]}', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Wheel Velocity (rad/s)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Wheel Velocity Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Action correlation matrix
    ax = axes[1, 1]
    corr_matrix = np.corrcoef(stats['all_actions'].T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(wheel_names)
    ax.set_yticklabels(wheel_names)
    ax.set_title('Wheel Velocity Correlation Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # 添加数值标注
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    return fig


def visualize_sample_trajectories(data_path, num_samples=5):
    """Visualize sample trajectories (using real start and goal points)"""
    print(f"\n{'='*80}")
    print(f"  Visualizing Sample Trajectories (first {num_samples})")
    print(f"{'='*80}\n")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('MPC Expert Trajectory Visualization - Indoor 6-Room Scene', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # ========== Draw 6-room indoor scene structure (consistent with rosorin_env_cfg.py) ==========
    # Outer walls (10m × 10m fully enclosed)
    ax.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], 'k-', linewidth=2.5, label='Outer Wall')
    
    # ========== Horizontal divider walls (y=0, separating upper and lower rows) ==========
    # Segment 1: x=-4.93 to -2.67 (left side between R1-R4)
    ax.plot([-4.93, -2.67], [0, 0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Doorway 1: x=-2.67 to -1.67 (R1↔R4)
    ax.plot([-2.67, -1.67], [0, 0], 'g--', linewidth=2, alpha=0.6, label='Doorway')
    
    # 第2段: x=-0.63到0.63（R2-R5之间中间）
    ax.plot([-0.63, 0.63], [0, 0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # 门洞2: x=0.63到1.67 (R2↔R5)
    ax.plot([0.63, 1.67], [0, 0], 'g--', linewidth=2, alpha=0.6)
    
    # 第3段: x=2.67到4.93（R3-R6之间右侧）
    ax.plot([2.67, 4.93], [0, 0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # 门洞3: x=1.67到2.67 (R3↔R6)
    ax.plot([1.67, 2.67], [0, 0], 'g--', linewidth=2, alpha=0.6)
    
    # ========== 垂直隔断墙 ==========
    # First vertical wall x=-1.67 (separating R1-R2 and R4-R5)
    # Upper R1-R2 segment 1: y=0→1.5
    ax.plot([-1.67, -1.67], [0.0, 1.5], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Doorway: y=1.5→2.5
    ax.plot([-1.67, -1.67], [1.5, 2.5], 'g--', linewidth=2, alpha=0.6)
    # Upper R1-R2 segment 2: y=2.5→5.0
    ax.plot([-1.67, -1.67], [2.5, 5.0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Lower R4-R5 segment 1: y=-5.0→-2.5
    ax.plot([-1.67, -1.67], [-5.0, -2.5], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Doorway: y=-2.5→-1.5
    ax.plot([-1.67, -1.67], [-2.5, -1.5], 'g--', linewidth=2, alpha=0.6)
    # Lower R4-R5 segment 2: y=-1.5→0
    ax.plot([-1.67, -1.67], [-1.5, 0.0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    
    # Second vertical wall x=1.67 (separating R2-R3 and R5-R6)
    # Upper R2-R3 segment 1: y=0→1.5
    ax.plot([1.67, 1.67], [0.0, 1.5], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Doorway: y=1.5→2.5
    ax.plot([1.67, 1.67], [1.5, 2.5], 'g--', linewidth=2, alpha=0.6)
    # Upper R2-R3 segment 2: y=2.5→5.0
    ax.plot([1.67, 1.67], [2.5, 5.0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Lower R5-R6 segment 1: y=-5.0→-2.5
    ax.plot([1.67, 1.67], [-5.0, -2.5], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    # Doorway: y=-2.5→-1.5
    ax.plot([1.67, 1.67], [-2.5, -1.5], 'g--', linewidth=2, alpha=0.6)
    # Lower R5-R6 segment 2: y=-1.5→0
    ax.plot([1.67, 1.67], [-1.5, 0.0], 'gray', linewidth=2, linestyle='-', alpha=0.7)
    
    # ========== 标注6个房间名称 ==========
    ax.text(-3.3, 4.5, 'R1\nLiving Room', fontsize=10, ha='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.text(0.0, 4.5, 'R2\nStudy', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(3.3, 4.5, 'R3\nBedroom', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
    ax.text(-3.3, -0.8, 'R4\nDining Room', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.0, -0.8, 'R5\nKitchen', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.text(3.3, -0.8, 'R6\nStorage', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    # ========== Draw 12 furniture positions (consistent with rosorin_env_cfg.py) ==========
    furniture = [
        # R1-Living (x: -5.0→-1.67, y: 0→5)
        {'pos': (-3.5, 3.5), 'size': (2.0, 0.9), 'name': 'Sofa', 'color': 'brown'},
        {'pos': (-4.0, 1.2), 'size': (1.5, 0.4), 'name': 'TV Stand', 'color': 'sienna'},
        # R2-Study (x: -1.67→1.67, y: 0→5)
        {'pos': (-0.5, 3.5), 'size': (1.4, 0.7), 'name': 'Desk', 'color': 'darkgoldenrod'},
        {'pos': (1.2, 1.5), 'size': (0.4, 2.2), 'name': 'Bookshelf', 'color': 'saddlebrown'},
        # R3-Bedroom (x: 1.67→5.0, y: 0→5)
        {'pos': (3.5, 3.5), 'size': (2.0, 1.5), 'name': 'Bed', 'color': 'burlywood'},
        {'pos': (4.4, 1.2), 'size': (0.6, 1.8), 'name': 'Wardrobe', 'color': 'tan'},
        # R4-Dining (x: -5.0→-1.67, y: -5→0)
        {'pos': (-3.5, -2.5), 'size': (1.6, 1.0), 'name': 'Dining Table', 'color': 'chocolate'},
        {'pos': (-4.2, -4.2), 'size': (0.5, 1.6), 'name': 'Sideboard', 'color': 'peru'},
        # R5-Kitchen (x: -1.67→1.67, y: -5→0)
        {'pos': (0.0, -3.8), 'size': (2.5, 0.6), 'name': 'Counter', 'color': 'silver'},
        {'pos': (1.0, -1.5), 'size': (0.7, 0.7), 'name': 'Fridge', 'color': 'lightgray'},
        # R6-Storage (x: 1.67→5.0, y: -5→0)
        {'pos': (4.3, -3.5), 'size': (0.5, 1.5), 'name': 'Shelf1', 'color': 'dimgray'},
        {'pos': (2.5, -4.0), 'size': (0.5, 1.5), 'name': 'Shelf2', 'color': 'gray'},
    ]
    
    for furn in furniture:
        rect = plt.Rectangle(
            (furn['pos'][0] - furn['size'][0]/2, furn['pos'][1] - furn['size'][1]/2),
            furn['size'][0], furn['size'][1],
            facecolor=furn['color'], edgecolor='black', linewidth=1, alpha=0.4
        )
        ax.add_patch(rect)
    
    # ========== Draw expert trajectories ==========
    colors = plt.cm.rainbow(np.linspace(0, 1, num_samples))
    
    with h5py.File(data_path, 'r') as f:
        episode_keys = list(f.keys())[:num_samples]
        
        for idx, ep_key in enumerate(episode_keys):
            ep_group = f[ep_key]
            
            # Extract real path points (start → waypoints → goal)
            path_points = ep_group['path_points'][:]
            start_point = path_points[0]
            goal_point = path_points[-1]
            
            print(f'Trajectory {idx+1}: Start({start_point[0]:.2f}, {start_point[1]:.2f}) → '
                  f'Goal({goal_point[0]:.2f}, {goal_point[1]:.2f})')
            
            # Draw path points connection (MPC planned path)
            ax.plot(path_points[:, 0], path_points[:, 1], 
                   color=colors[idx], linewidth=2.5, alpha=0.8, 
                   linestyle='--', label=f'Traj {idx+1}')
            
            # Draw start point (circle)
            ax.plot(start_point[0], start_point[1], 'o', 
                   color=colors[idx], markersize=12, 
                   markeredgecolor='black', markeredgewidth=2)
            
            # Draw goal point (square)
            ax.plot(goal_point[0], goal_point[1], 's', 
                   color=colors[idx], markersize=12, 
                   markeredgecolor='black', markeredgewidth=2)
            
            # Draw intermediate waypoints (small diamonds)
            if len(path_points) > 2:
                ax.plot(path_points[1:-1, 0], path_points[1:-1, 1], 'D',
                       color=colors[idx], markersize=6, 
                       markeredgecolor='black', markeredgewidth=1)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='k', linewidth=2.5, label='Outer Wall'),
        plt.Line2D([0], [0], color='gray', linewidth=2, label='Inner Wall'),
        plt.Line2D([0], [0], color='g', linewidth=2, linestyle='--', label='Doorway'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, markeredgecolor='black', label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                  markersize=10, markeredgecolor='black', label='Goal'),
        plt.Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='MPC Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize MPC Expert Data")
    parser.add_argument('--data_path', '--data', type=str, required=True, 
                       dest='data',
                       help='HDF5 data file path (supports wildcards *.h5)')
    parser.add_argument('--num_trajectories', type=int, default=5,
                       help='Number of trajectories to visualize')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to experiments/results/')
    
    args = parser.parse_args()
    
    # Handle wildcards
    from glob import glob
    data_files = glob(args.data)
    
    if not data_files:
        print(f"❌ Data file not found: {args.data}")
        return
    
    print(f"Found {len(data_files)} data file(s)")
    
    # 分析每个数据文件
    for data_file in data_files:
        stats = analyze_expert_data(data_file)
        
        # 可视化统计信息
        fig1 = visualize_statistics(stats)
        
        # 可视化样本轨迹
        fig2 = visualize_sample_trajectories(data_file, num_samples=args.num_trajectories)
        
        if args.save_plots:
            save_dir = Path("experiments/results")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            filename = Path(data_file).stem
            fig1.savefig(save_dir / f"{filename}_statistics.png", dpi=150, bbox_inches='tight')
            fig2.savefig(save_dir / f"{filename}_trajectories.png", dpi=150, bbox_inches='tight')
            print(f"\n✅ 图表已保存到 {save_dir}/")
        
        plt.show()


if __name__ == "__main__":
    main()
