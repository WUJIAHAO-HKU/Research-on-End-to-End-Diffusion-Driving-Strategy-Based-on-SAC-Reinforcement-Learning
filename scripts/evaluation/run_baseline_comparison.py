"""
Baseline算法对比实验

对比以下算法：
1. MPC (Model Predictive Control) - 专家策略
2. BC (Behavior Cloning) - 预训练策略
3. TD3 (Twin Delayed DDPG) - 确定性策略
4. SAC-Gaussian (标准SAC) - 高斯随机策略
5. SAC-Diffusion (本项目) - 扩散随机策略
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠ matplotlib/seaborn未安装，将跳过图表生成")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠ pandas未安装，将使用简化输出")

def load_results(results_dir: Path) -> Dict:
    """加载各个算法的评估结果"""
    results = {}
    
    # 查找所有评估结果文件
    for method_dir in results_dir.iterdir():
        if method_dir.is_dir():
            result_file = method_dir / "evaluation_results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results[method_dir.name] = json.load(f)
    
    return results


def create_comparison_table(results: Dict):
    """创建对比表格"""
    data = []
    
    for method, result in results.items():
        data.append({
            'Method': method,
            'Mean Reward': result.get('mean_reward', 0),
            'Std Reward': result.get('std_reward', 0),
            'Success Rate (%)': result.get('success_rate', 0) * 100,
            'Mean Episode Length': result.get('mean_length', 0),
            'Training Time (min)': result.get('training_time', 0) / 60,
        })
    
    if PANDAS_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame(data)
        df = df.sort_values('Mean Reward', ascending=False)
        return df
    else:
        # 简单排序
        data.sort(key=lambda x: x['Mean Reward'], reverse=True)
        return data


def plot_reward_comparison(results: Dict, save_path: Path):
    """绘制奖励对比图"""
    if not PLOTTING_AVAILABLE:
        print("⚠ 跳过绘图（matplotlib未安装）")
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = list(results.keys())
    rewards = [results[m].get('mean_reward', 0) for m in methods]
    stds = [results[m].get('std_reward', 0) for m in methods]
    
    colors = sns.color_palette("husl", len(methods))
    bars = ax.bar(methods, rewards, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    ax.set_ylabel('Average Episode Reward', fontsize=14)
    ax.set_title('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / 'reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存奖励对比图: {save_path / 'reward_comparison.png'}")


def plot_success_rate(results: Dict, save_path: Path):
    """绘制成功率对比图"""
    if not PLOTTING_AVAILABLE:
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = list(results.keys())
    success_rates = [results[m].get('success_rate', 0) * 100 for m in methods]
    
    colors = sns.color_palette("coolwarm", len(methods))
    bars = ax.barh(methods, success_rates, color=colors, alpha=0.8)
    
    ax.set_xlabel('Success Rate (%)', fontsize=14)
    ax.set_title('Success Rate Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 100])
    
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存成功率对比图: {save_path / 'success_rate_comparison.png'}")


def plot_training_efficiency(results: Dict, save_path: Path):
    """绘制训练效率对比图"""
    if not PLOTTING_AVAILABLE:
        return
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 过滤掉MPC（不需要训练）
    trainable_methods = {k: v for k, v in results.items() 
                         if k != 'MPC' and 'training_time' in v}
    
    if not trainable_methods:
        print("⚠ 没有训练时间数据，跳过训练效率图")
        return
    
    methods = list(trainable_methods.keys())
    times = [trainable_methods[m]['training_time'] / 60 for m in methods]
    rewards = [trainable_methods[m].get('mean_reward', 0) for m in methods]
    
    colors = sns.color_palette("Set2", len(methods))
    
    for i, method in enumerate(methods):
        ax.scatter(times[i], rewards[i], s=200, c=[colors[i]], 
                  alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(method, (times[i], rewards[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Training Time (minutes)', fontsize=14)
    ax.set_ylabel('Final Reward', fontsize=14)
    ax.set_title('Training Efficiency: Reward vs Time', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存训练效率图: {save_path / 'training_efficiency.png'}")


def generate_latex_table(df, save_path: Path):
    """生成LaTeX格式的表格"""
    if not PANDAS_AVAILABLE:
        print("⚠ pandas未安装，跳过LaTeX表格生成")
        return
    
    latex_str = df.to_latex(
        index=False,
        float_format="%.2f",
        caption="Algorithm Performance Comparison",
        label="tab:comparison"
    )
    
    latex_file = save_path / 'comparison_table.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_str)
    
    print(f"✓ 保存LaTeX表格: {latex_file}")


def generate_report(results: Dict, save_path: Path):
    """生成实验报告"""
    report = []
    report.append("=" * 80)
    report.append("  Baseline Algorithm Comparison Report")
    report.append("=" * 80)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 创建对比表格
    df = create_comparison_table(results)
    report.append("\n## 性能对比表格\n")
    
    if PANDAS_AVAILABLE:
        report.append(df.to_string(index=False))
        best_method = df.iloc[0]['Method']
        best_reward = df.iloc[0]['Mean Reward']
        df_success = df.sort_values('Success Rate (%)', ascending=False)
        best_success = df_success.iloc[0]
    else:
        # 简单格式化输出
        for row in df:
            report.append(f"{row['Method']:20s} | "
                         f"Reward: {row['Mean Reward']:6.2f} ± {row['Std Reward']:5.2f} | "
                         f"Success: {row['Success Rate (%)']:5.1f}% | "
                         f"Length: {row['Mean Episode Length']:7.1f} | "
                         f"Time: {row['Training Time (min)']:6.1f}min")
        best_method = df[0]['Method']
        best_reward = df[0]['Mean Reward']
        best_success = max(df, key=lambda x: x['Success Rate (%)'])
    
    report.append(f"\n\n## 关键发现\n")
    report.append(f"🏆 最佳算法: {best_method} (平均奖励: {best_reward:.2f})")
    
    # 成功率排名
    report.append(f"✅ 最高成功率: {best_success['Method']} ({best_success['Success Rate (%)']:.1f}%)")
    
    # 训练效率
    if PANDAS_AVAILABLE:
        df_efficient = df[df['Training Time (min)'] > 0].copy()
    else:
        df_efficient = [x for x in df if x['Training Time (min)'] > 0]
    
    if df_efficient is not None and len(df_efficient) > 0:
        if PANDAS_AVAILABLE:
            df_efficient['Efficiency'] = df_efficient['Mean Reward'] / df_efficient['Training Time (min)']
            df_efficient = df_efficient.sort_values('Efficiency', ascending=False)
            most_efficient = df_efficient.iloc[0]
            report.append(f"⚡ 训练效率最高: {most_efficient['Method']} "
                         f"(奖励/分钟: {most_efficient['Efficiency']:.3f})")
        else:
            for row in df_efficient:
                row['Efficiency'] = row['Mean Reward'] / row['Training Time (min)']
            most_efficient = max(df_efficient, key=lambda x: x['Efficiency'])
            report.append(f"⚡ 训练效率最高: {most_efficient['Method']} "
                         f"(奖励/分钟: {most_efficient['Efficiency']:.3f})")
    
    report.append("\n" + "=" * 80)
    
    # 保存报告
    report_file = save_path / 'comparison_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✓ 保存实验报告: {report_file}\n")
    
    # 打印到控制台
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description="Baseline算法对比分析")
    parser.add_argument("--results_dir", type=str, 
                       default="experiments/baseline_comparison",
                       help="结果目录")
    parser.add_argument("--output_dir", type=str,
                       default="experiments/comparison_results",
                       help="输出目录")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  Baseline算法对比分析")
    print("=" * 80)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}\n")
    
    # 如果没有结果，创建示例数据
    if not results_dir.exists() or not any(results_dir.iterdir()):
        print("⚠ 未找到评估结果，创建示例数据用于测试...")
        create_example_results(results_dir)
    
    # 加载结果
    print("加载评估结果...")
    results = load_results(results_dir)
    
    if not results:
        print("❌ 错误: 未找到任何评估结果")
        print(f"\n请确保在 {results_dir} 目录下有以下结构:")
        print("  baseline_comparison/")
        print("    ├── MPC/evaluation_results.json")
        print("    ├── BC/evaluation_results.json")
        print("    ├── TD3/evaluation_results.json")
        print("    ├── SAC-Gaussian/evaluation_results.json")
        print("    └── SAC-Diffusion/evaluation_results.json")
        return
    
    print(f"✓ 加载了 {len(results)} 个算法的结果\n")
    
    # 创建对比表格
    df = create_comparison_table(results)
    
    # 保存表格
    csv_file = output_dir / 'comparison_results.csv'
    if PANDAS_AVAILABLE:
        df.to_csv(csv_file, index=False)
        print(f"✓ 保存CSV表格: {csv_file}")
    else:
        # 手动保存CSV
        with open(csv_file, 'w') as f:
            if df:
                keys = df[0].keys()
                f.write(','.join(keys) + '\n')
                for row in df:
                    f.write(','.join(str(row[k]) for k in keys) + '\n')
        print(f"✓ 保存CSV表格: {csv_file}")
    
    # 生成图表
    print("\n生成对比图表...")
    plot_reward_comparison(results, output_dir)
    plot_success_rate(results, output_dir)
    plot_training_efficiency(results, output_dir)
    
    # 生成LaTeX表格
    generate_latex_table(df, output_dir)
    
    # 生成报告
    generate_report(results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 对比分析完成！")
    print("=" * 80)
    print(f"\n查看结果: {output_dir}")
    print("  - comparison_results.csv: 数据表格")
    print("  - reward_comparison.png: 奖励对比图")
    print("  - success_rate_comparison.png: 成功率对比图")
    print("  - training_efficiency.png: 训练效率图")
    print("  - comparison_report.txt: 实验报告")


def create_example_results(results_dir: Path):
    """创建示例结果数据（用于测试）"""
    example_results = {
        "MPC": {
            "mean_reward": 8.5,
            "std_reward": 0.3,
            "success_rate": 0.95,
            "mean_length": 450,
        },
        "BC": {
            "mean_reward": 3.0,
            "std_reward": 0.5,
            "success_rate": 0.20,
            "mean_length": 1500,
            "training_time": 1200,  # 20分钟
        },
        "TD3": {
            "mean_reward": 5.2,
            "std_reward": 1.2,
            "success_rate": 0.45,
            "mean_length": 850,
            "training_time": 3600,  # 60分钟
        },
        "SAC-Gaussian": {
            "mean_reward": 6.1,
            "std_reward": 0.9,
            "success_rate": 0.55,
            "mean_length": 720,
            "training_time": 3300,  # 55分钟
        },
        "SAC-Diffusion": {
            "mean_reward": 3.0,
            "std_reward": 0.0,
            "success_rate": 0.20,
            "mean_length": 1500,
            "training_time": 832,  # 14分钟
        }
    }
    
    for method, data in example_results.items():
        method_dir = results_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = method_dir / "evaluation_results.json"
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"✓ 创建示例数据: {results_dir}\n")


if __name__ == "__main__":
    main()
