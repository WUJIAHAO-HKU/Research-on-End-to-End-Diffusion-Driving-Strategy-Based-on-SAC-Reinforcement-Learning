# 项目结构说明

本文档描述重构后的项目目录结构和模块组织。

## 📁 目录结构概览

```
Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning/
├── configs/                          # 配置文件目录
│   ├── rewards/                      # 🆕 算法专用奖励配置
│   │   ├── base_rewards.py          # 基础奖励配置类
│   │   ├── ppo_rewards.py           # PPO算法奖励配置
│   │   ├── sac_rewards.py           # SAC算法奖励配置
│   │   ├── bc_rewards.py            # BC算法奖励配置
│   │   ├── td3_rewards.py           # TD3算法奖励配置
│   │   └── dagger_rewards.py        # DAgger算法奖励配置
│   ├── mdp/                          # 🆕 MDP函数定义
│   │   └── rosorin_mdp.py           # 自定义奖励/终止/事件函数
│   ├── env/                          # 环境配置
│   ├── model/                        # 模型架构配置
│   └── training/                     # 训练超参数配置
│
├── scripts/                          # 🔄 重组后的脚本目录
│   ├── training/                     # 🆕 训练脚本
│   │   ├── train_ppo.py             # PPO训练
│   │   ├── train_sac_gaussian.py    # SAC训练（高斯策略）
│   │   ├── train_sac_diffusion.py   # SAC训练（扩散策略）
│   │   ├── train_bc.py              # BC训练
│   │   ├── train_td3.py             # TD3训练
│   │   └── train_dagger.py          # DAgger训练
│   │
│   ├── evaluation/                   # 🆕 评估脚本
│   │   ├── evaluate_ppo.py          # 评估PPO
│   │   ├── evaluate_sac.py          # 评估SAC
│   │   ├── evaluate_bc.py           # 评估BC
│   │   ├── evaluate_baselines.py    # 评估单个基线
│   │   ├── evaluate_all_baselines.py # 批量评估
│   │   └── run_baseline_comparison.py # 基线对比
│   │
│   ├── testing/                      # 🆕 测试脚本
│   │   ├── test_reward_system.py    # 测试奖励系统
│   │   ├── test_reward_extraction.py # 测试奖励提取
│   │   ├── run_rosorin_env.py       # 测试环境
│   │   └── verify_indoor_scene.py   # 验证场景配置
│   │
│   ├── data_collection/              # 🆕 数据收集脚本
│   │   └── collect_mpc_expert_data.py # MPC专家数据收集
│   │
│   ├── visualization/                # 🆕 可视化脚本
│   │   ├── visualize_training.py    # 可视化训练过程
│   │   ├── visualize_sac_training.py # 可视化SAC训练
│   │   ├── visualize_bc_policy.py   # 可视化BC策略
│   │   ├── visualize_expert_data.py # 可视化专家数据
│   │   └── plot_sac_training.py     # 绘制SAC曲线
│   │
│   ├── analysis/                     # 🆕 分析脚本
│   │   └── analyze_sac_cases.py     # 分析SAC案例
│   │
│   ├── utils/                        # 🆕 工具脚本
│   │   ├── path_generator.py        # 路径生成器
│   │   ├── simple_path_generator.py # 简单路径生成
│   │   ├── indoor_scene_aware_path_generator.py # 室内路径生成
│   │   ├── mpc_controller.py        # MPC控制器
│   │   ├── fix_reward_extraction.py # 修复脚本
│   │   ├── urdf_to_usd.py          # 格式转换
│   │   └── ...
│   │
│   ├── deployment/                   # 🆕 部署脚本
│   │   └── deploy_to_robot.py       # 真机部署
│   │
│   ├── env_factory.py               # 🆕 环境配置工厂
│   └── rosorin_env_cfg.py           # 基础环境配置（不含奖励）
│
├── src/                              # 源代码模块
│   ├── algorithms/                   # 算法实现
│   ├── baselines/                    # 基线算法
│   ├── models/                       # 神经网络模型
│   └── envs/                         # 环境封装
│
├── experiments/                      # 实验结果
│   ├── baseline_comparison/          # 基线对比结果
│   ├── checkpoints/                  # 模型检查点
│   ├── logs/                         # 训练日志
│   └── tensorboard/                  # TensorBoard日志
│
└── docs/                             # 文档
    ├── TRAINING_WORKFLOW.md          # 训练工作流
    ├── PROJECT_STRUCTURE.md          # 本文件
    └── ...

```

## 🎯 设计理念

### 1. **奖励配置解耦** 🆕

**问题**: 之前所有算法共用一个奖励配置（`rosorin_env_cfg.py`），不同算法无法优化各自的奖励权重。

**解决方案**: 
- 创建 `configs/rewards/` 目录
- 每个算法有独立的奖励配置文件
- 通过 `env_factory.py` 工厂函数动态创建环境

**示例**:
```python
# PPO训练脚本
from env_factory import create_ppo_env_cfg

env_cfg = create_ppo_env_cfg(num_envs=8)
# 自动使用PPO专用的奖励权重配置
```

### 2. **脚本功能分类** 🔄

**问题**: 之前所有脚本混在 `scripts/` 根目录，难以定位和维护。

**解决方案**: 按功能分类到子目录：
- `training/` - 训练脚本（执行频率高）
- `evaluation/` - 评估脚本（对比不同模型）
- `testing/` - 测试脚本（调试环境和系统）
- `visualization/` - 可视化脚本（绘图和分析）
- `data_collection/` - 数据收集脚本（生成专家数据）
- `utils/` - 工具脚本（辅助功能）
- `deployment/` - 部署脚本（真机运行）

### 3. **MDP函数集中管理** 🆕

**问题**: `rosorin_mdp.py` 混在scripts目录，不易于配置引用。

**解决方案**:
- 移动到 `configs/mdp/rosorin_mdp.py`
- 所有奖励配置统一从这里导入MDP函数

## 📖 使用指南

### 训练模型

```bash
# PPO训练（使用PPO专用奖励配置）
./isaaclab_runner.sh scripts/training/train_ppo.py --num_envs 8 --total_steps 100000

# SAC训练（使用SAC专用奖励配置）
./isaaclab_runner.sh scripts/training/train_sac_gaussian.py --num_envs 8

# BC训练（使用BC专用奖励配置）
./isaaclab_runner.sh scripts/training/train_bc.py --demo_path data/demonstrations/mpc_expert.pkl
```

### 评估模型

```bash
# 评估PPO模型
./isaaclab_runner.sh scripts/evaluation/evaluate_ppo.py --checkpoint experiments/baselines/ppo/model.pth

# 批量评估所有基线
./isaaclab_runner.sh scripts/evaluation/evaluate_all_baselines.py
```

### 测试环境

```bash
# 测试奖励系统
./isaaclab_runner.sh scripts/testing/test_reward_system.py

# 验证场景配置
./isaaclab_runner.sh scripts/testing/verify_indoor_scene.py
```

### 可视化结果

```bash
# 绘制SAC训练曲线
python scripts/visualization/plot_sac_training.py --log_dir experiments/sac_training/logs

# 可视化BC策略
./isaaclab_runner.sh scripts/visualization/visualize_bc_policy.py --checkpoint model.pth
```

## 🔧 自定义奖励配置

如果需要为新算法创建专用奖励配置：

1. **创建新的奖励配置文件**:
```python
# configs/rewards/my_algorithm_rewards.py
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
import sys, os

mdp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../mdp'))
if mdp_path not in sys.path:
    sys.path.insert(0, mdp_path)
import rosorin_mdp

@configclass
class MyAlgorithmRewardsCfg:
    progress = RewTerm(func=rosorin_mdp.progress_reward, weight=25.0)
    # ... 其他奖励配置
```

2. **在env_factory.py中添加工厂函数**:
```python
def create_my_algorithm_env_cfg(num_envs=8, env_spacing=5.0):
    from configs.rewards.my_algorithm_rewards import MyAlgorithmRewardsCfg
    from rosorin_env_cfg import ROSOrinEnvCfg
    
    env_cfg = ROSOrinEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.env_spacing = env_spacing
    env_cfg.rewards = MyAlgorithmRewardsCfg()
    
    return env_cfg
```

3. **在训练脚本中使用**:
```python
from env_factory import create_my_algorithm_env_cfg

env_cfg = create_my_algorithm_env_cfg(num_envs=8)
env = ManagerBasedRLEnv(cfg=env_cfg)
```

## 🎓 算法专用奖励配置对比

| 算法 | 主导航权重 (progress+orientation+velocity) | 惩罚权重 | 设计思路 |
|------|-------------------------------------------|---------|----------|
| **PPO** | 28.0 (20+5+3) | 4.6 | 高密集奖励，鼓励探索 |
| **SAC** | 24.5 (18+4+2.5) | 7.3 | 平衡探索利用，利用经验回放 |
| **TD3** | 23.5 (17+4+2.5) | 8.5 | 更保守，注重动作平滑 |
| **BC** | 20.0 (15+3+2) | 10.5 | 基础配置，用于评估 |
| **DAgger** | 20.0 (15+3+2) | 10.5 | 与BC一致，迭代学习 |

## 📝 重要变更

### ✅ 已完成

1. ✅ 创建 `configs/rewards/` 目录和各算法奖励配置
2. ✅ 移动 `rosorin_mdp.py` 到 `configs/mdp/`
3. ✅ 重组 `scripts/` 为功能目录
4. ✅ 创建 `env_factory.py` 工厂模块
5. ✅ 更新 `train_ppo.py` 使用新配置
6. ✅ 解决循环导入问题
7. ✅ 测试训练脚本正常运行

### ⚠️ 需要注意

1. **导入顺序**: 奖励配置必须在 `AppLauncher` 启动后导入
2. **路径更新**: 所有训练脚本路径从 `scripts/train_*.py` 改为 `scripts/training/train_*.py`
3. **向后兼容**: `scripts/rosorin_env_cfg.py` 和 `scripts/rosorin_mdp.py` 仍保留，可用于旧脚本

## 🚀 后续优化建议

1. **环境配置模块化**: 将场景、传感器、机器人配置进一步分离
2. **超参数配置文件**: 为每个算法创建YAML配置文件
3. **实验管理系统**: 使用MLflow或Weights&Biases追踪实验
4. **单元测试**: 为关键模块添加测试用例
5. **CI/CD**: 自动化测试和部署流程

---

**更新日期**: 2025年12月30日  
**重构版本**: v2.0  
**维护者**: ROSOrin项目团队
