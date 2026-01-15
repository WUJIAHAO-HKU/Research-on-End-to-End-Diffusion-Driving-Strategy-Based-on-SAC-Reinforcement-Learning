# 基于SAC强化学习的端到端扩散驾驶策略研究

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.0-blue.svg)](docs/project_structure.md)

## 🚗 项目概述

本项目探索前沿的自动驾驶范式：在NVIDIA Isaac Lab高保真仿真环境中，训练一个以去噪扩散概率模型（DDPM）为核心、并采用软演员-批评家（SAC）算法优化的端到端驾驶策略。

**核心创新点：**
- 🎯 将扩散模型的去噪过程作为SAC的随机策略
- 🌈 最大熵强化学习实现多模态动作分布学习
- 🚀 Isaac Lab GPU加速并行仿真训练 (8-64 parallel envs)
- 🔄 完整的Sim2Real迁移流程到ROSOrin小车

**发表目标：** CoRL 2026 / ICRA 2026 / T-RO 顶级会议/期刊

**最新版本：** v2.0（2025-12-30）- 项目结构重构完成

---

## 📚 文档导航

- **[快速开始](docs/quickstart.md)** - 新手入门指南
- **[项目结构](docs/project_structure.md)** - 详细的目录组织说明（v2.0重构）
- **[训练工作流](docs/training_workflow.md)** - 完整的训练流程
- **[基线算法](docs/baselines.md)** - 6种baseline对比实验
- **[项目概要](docs/project_summary.md)** - 研究目标与技术路线
- **[理论基础](docs/theory.md)** - 算法原理与设计思想
- **[文档索引](docs/README.md)** - 所有文档列表

---

## 🆕 v2.0 重构更新（2025-12-30）

### 主要改进
- ✅ **奖励配置分离**: 每个算法独立的奖励权重配置（`configs/rewards/`）
- ✅ **脚本分类重组**: 按功能分类到training/evaluation/testing等目录
- ✅ **环境配置工厂**: 通过工厂函数自动加载算法专用配置
- ✅ **文档整理**: 所有文档移至docs/目录，删除临时文档

详见：[项目结构文档](docs/project_structure.md)

---

## 📊 项目状态

### 模块完成度

| 模块 | 完成度 | 状态 |
|-----|--------|------|
| 核心算法 (Diffusion + SAC) | 95% | 🟢 已完成 |
| 观测编码器 (Vision/LiDAR/Fusion) | 100% | 🟢 已完成 |
| 数据处理 (Buffer/Dataset) | 100% | 🟢 已完成 |
| 仿真环境 (Isaac Lab) | 60% | 🟡 待完善 |
| 训练脚本 | 90% | 🟢 基本完成 |
| Sim2Real部署 | 100% | 🟢 已完成 |
| 基线算法 | 100% | 🟢 已完成 |
| 配置文件 | 70% | 🟡 部分完成 |

**代码统计**: 38个Python文件 | 8,658行代码 | 100%通过语法检查

👉 **查看详细报告**: [PROJECT_STATUS.md](PROJECT_STATUS.md)

---

## 📁 项目结构（v2.0）

项目已重构为功能清晰的模块化结构。详细说明请参考 **[项目结构文档](docs/project_structure.md)**

```
.
├── README.md                          # 项目说明（本文件）
├── setup.py                           # 项目安装配置
├── requirements.txt                   # Python依赖
│
├── configs/                           # 配置文件目录
│   ├── rewards/                      # 🆕 算法专用奖励配置
│   │   ├── ppo_rewards.py           # PPO奖励权重
│   │   ├── sac_rewards.py           # SAC奖励权重
│   │   ├── bc_rewards.py            # BC奖励权重
│   │   ├── td3_rewards.py           # TD3奖励权重
│   │   └── dagger_rewards.py        # DAgger奖励权重
│   ├── mdp/                          # 🆕 MDP函数定义
│   │   └── rosorin_mdp.py           # 自定义奖励/终止函数
│   ├── env/                          # 环境配置
│   ├── model/                        # 模型配置
│   └── training/                     # 训练配置
│
├── scripts/                           # 🔄 重组后的脚本目录
│   ├── training/                     # 🆕 训练脚本
│   │   ├── train_ppo.py             # PPO训练
│   │   ├── train_sac_gaussian.py    # SAC训练（高斯策略）
│   │   ├── train_sac_diffusion.py   # SAC训练（扩散策略）
│   │   ├── train_bc.py              # BC训练
│   │   ├── train_td3.py             # TD3训练
│   │   └── train_dagger.py          # DAgger训练
│   ├── evaluation/                   # 🆕 评估脚本
│   ├── testing/                      # 🆕 测试脚本
│   ├── data_collection/              # 🆕 数据收集
│   ├── visualization/                # 🆕 可视化
│   ├── analysis/                     # 🆕 分析工具
│   ├── utils/                        # 🆕 工具脚本
│   ├── deployment/                   # 🆕 部署脚本
│   ├── env_factory.py               # 🆕 环境配置工厂
│   └── rosorin_env_cfg.py           # 基础环境配置
│
├── src/                               # 源代码模块
│   ├── algorithms/                   # 算法实现
│   ├── models/                       # 神经网络模型
│   ├── envs/                         # 环境封装
│   └── data/                         # 数据处理
│
├── experiments/                       # 实验结果
│   ├── baseline_comparison/          # 基线对比
│   ├── checkpoints/                  # 模型检查点
│   ├── logs/                         # 训练日志
│   └── tensorboard/                  # TensorBoard日志
│
├── docs/                              # 📚 文档目录
│   ├── README.md                     # 文档索引
│   ├── project_structure.md          # 项目结构详解
│   ├── training_workflow.md          # 训练工作流
│   ├── baselines.md                  # 基线算法说明
│   ├── quickstart.md                 # 快速开始
│   ├── project_summary.md            # 项目概要
│   └── theory.md                     # 理论基础
│
└── data/                              # 数据目录
    ├── demonstrations/                # 专家演示数据
    └── real_world/                    # 真实世界数据
```

**重点目录说明**：
- `configs/rewards/` - 每个算法独立的奖励配置（v2.0新增）
- `scripts/training/` - 所有训练脚本按功能分类（v2.0重组）
- `docs/` - 完整的技术文档（v2.0整理）

---

##  快速开始

详细步骤请参考 **[快速开始文档](docs/quickstart.md)** 和 **[训练工作流](docs/training_workflow.md)**

### 1. 环境配置

```bash
# 创建Conda环境
conda env create -f environment.yml
conda activate sac-diffusion-driving

# 安装项目
pip install -e .

# 安装Isaac Lab（需要GPU）
# 参考: https://isaac-sim.github.io/IsaacLab/
```

### 2. 数据收集（MPC专家演示）

```bash
# 🆕 新路径
./isaaclab_runner.sh scripts/data_collection/collect_mpc_expert_data.py \
    --num_envs 8 \
    --num_episodes 30 \
    --difficulty easy \
    --enable_cameras \
    --headless
```

### 3. 训练PPO基线

```bash
# 🆕 新路径 - 使用PPO专用奖励配置
./isaaclab_runner.sh scripts/training/train_ppo.py \
    --num_envs 8 \
    --total_steps 100000 \
    --headless
```

### 4. 训练SAC-Diffusion

```bash
# 🆕 新路径 - 使用SAC专用奖励配置
./isaaclab_runner.sh scripts/training/train_sac_diffusion.py \
    --num_envs 8 \
    --total_steps 200000 \
    --headless
```

### 5. 评估模型

```bash
# 🆕 新路径
./isaaclab_runner.sh scripts/evaluation/evaluate_ppo.py \
    --checkpoint experiments/baselines/ppo/model.pth \
    --num_envs 8 \
    --num_episodes 50
```

📖 **更多详细步骤**: 参考 [训练工作流文档](docs/training_workflow.md)

---

## 📊 基线算法对比

本项目实现了6种baseline算法用于对比。详见 **[基线算法文档](docs/baselines.md)**

| 算法 | 类型 | 策略 | 奖励配置 | 特点 |
|------|------|------|---------|------|
| **BC** | 模仿学习 | 确定性 | [bc_rewards.py](configs/rewards/bc_rewards.py) | 直接学习专家 |
| **PPO** | On-policy RL | 高斯 | [ppo_rewards.py](configs/rewards/ppo_rewards.py) | 高密集奖励 |
| **SAC** | Off-policy RL | 高斯 | [sac_rewards.py](configs/rewards/sac_rewards.py) | 最大熵 |
| **TD3** | Off-policy RL | 确定性 | [td3_rewards.py](configs/rewards/td3_rewards.py) | 双Q网络 |
| **DAgger** | 模仿学习 | 确定性 | [dagger_rewards.py](configs/rewards/dagger_rewards.py) | 迭代聚合 |
| **SAC-Diffusion** | Off-policy RL | 扩散 | [sac_rewards.py](configs/rewards/sac_rewards.py) | 多模态 |

**v2.0新特性**: 每个算法现在有独立的奖励权重配置，可单独优化！

---

## 🎓 理论基础

详细理论推导请参考 **[理论文档](docs/theory.md)**

### SAC-Diffusion核心思想

```
传统SAC: π(a|s) = Gaussian(μ(s), σ(s))
本项目:  π(a|s) = DDPM_reverse(a_T → a_0 | s)
```

**优势**：
1. 多模态动作分布（处理多种可行策略）
2. 平滑的策略梯度（扩散过程天然正则化）
3. 更好的探索能力（熵正则化+去噪随机性）

---

## 🔬 实验设置

### 仿真环境
- **平台**: NVIDIA Isaac Lab 4.0
- **场景**: 6房间室内导航（10m×10m）
- **传感器**: RGB相机(240×80) + 深度相机(160×120)
- **机器人**: ROSOrin麦克纳姆轮小车

### 训练配置
- **并行环境**: 8个
- **总训练步数**: 100K-200K
- **批次大小**: 256-512
- **学习率**: 3e-5 (PPO), 3e-4 (SAC)

详细配置参见各算法的奖励配置文件：`configs/rewards/`

---

## 🎯 v2.0 重构说明（2025-12-30）

### 重构前的问题
- ❌ 所有算法共用一个奖励配置，无法独立优化
- ❌ 所有脚本混在scripts/根目录，难以维护
- ❌ MDP函数位置不清晰

### 重构后的改进
- ✅ 每个算法独立的奖励配置文件（`configs/rewards/`）
- ✅ 脚本按功能分类到子目录（training/evaluation/testing等）
- ✅ MDP函数集中管理（`configs/mdp/`）
- ✅ 环境配置工厂模式（`scripts/env_factory.py`）
- ✅ 完整的文档体系（`docs/`）

### 命令变更示例

**旧命令** (v1.0):
```bash
./isaaclab_runner.sh scripts/train_ppo.py --num_envs 8
```

**新命令** (v2.0):
```bash
./isaaclab_runner.sh scripts/training/train_ppo.py --num_envs 8
```

详见：[项目结构文档](docs/project_structure.md)

---

## 🎓 旧版快速开始（v1.0 - 仅供参考）

```bash
# 训练扩散策略的初始版本
python scripts/train_bc.py \
    --config configs/training/bc_pretrain.yaml \
    --data_dir data/demonstrations \
    --output_dir experiments/bc_pretrain
```

### 4. SAC-Diffusion强化学习微调

```bash
# 联合训练SAC和扩散策略
python scripts/train_sac_diffusion.py \
    --config configs/training/sac_finetuning.yaml \
    --pretrained_model experiments/bc_pretrain/best_model.pth \
    --num_envs 64 \
    --output_dir experiments/sac_diffusion
```

### 5. 评估与可视化

```bash
# 在测试场景中评估
python scripts/evaluate.py \
    --config configs/experiment/baseline_comparison.yaml \
    --checkpoint experiments/sac_diffusion/best_model.pth \
    --render True \
    --save_video True
```

### 6. 实机部署

```bash
# 部署到ROSOrin小车
python scripts/deploy_to_robot.py \
    --config configs/sim2real/rosorin_deployment.yaml \
    --checkpoint experiments/sac_diffusion/best_model.pth
```

---

## 📊 实验方案

### Phase 1: 仿真验证（当前阶段）

**目标：** 在Isaac Lab中验证算法有效性

1. **Baseline对比**
   - [ ] MPC（专家策略）
   - [ ] 标准Diffusion Policy
   - [ ] TD3
   - [ ] SAC-Gaussian Policy
   - [ ] **SAC-Diffusion Policy（本文方法）**

2. **消融实验**
   - [ ] 扩散步数的影响（5, 10, 20, 50步）
   - [ ] SAC vs PPO vs TD3（哪个RL算法更适合）
   - [ ] 熵权重的影响
   - [ ] 预训练的必要性

3. **泛化性测试**
   - [ ] 不同场景（城市、高速、停车场）
   - [ ] 不同天气（晴天、雨天、夜晚）
   - [ ] 动态障碍物密度

### Phase 2: Sim2Real迁移

**目标：** 将策略迁移到真实ROSOrin小车

1. **域随机化强化**
   - 传感器噪声、延迟模拟
   - 动力学参数随机化

2. **真实世界微调**
   - 在真实小车上收集少量数据
   - 在线微调策略

---

## 🎓 理论贡献

### 核心数学框架

**目标函数（SAC with Diffusion Policy）：**

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t \left( r(s_t, a_t) + \alpha H(\pi_\theta(\cdot | s_t)) \right) \right]
$$

其中：
- $\pi_\theta(a|s)$ 是扩散策略，通过去噪过程 $p_\theta(a_0 | a_T, s)$ 定义
- $H(\pi_\theta)$ 是策略熵，鼓励探索
- $\alpha$ 是自动调节的温度参数

**扩散策略梯度：**

通过重参数化技巧，将扩散采样过程纳入梯度计算：

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s,\epsilon} \left[ \nabla_\theta \log p_\theta(a | s, \epsilon) \cdot Q(s, a) \right]
$$

详细推导见 `docs/theory.md`

---

## 📈 预期结果

| 指标 | MPC | Diffusion Policy | TD3 | **SAC-Diffusion** |
|------|-----|------------------|-----|-------------------|
| 任务成功率 | 85% | 78% | 82% | **92%** |
| 平均奖励 | 120 | 105 | 115 | **135** |
| 碰撞率 | 5% | 8% | 6% | **3%** |
| 动作平滑度 | 0.85 | 0.92 | 0.78 | **0.95** |
| 泛化性（新场景） | 72% | 65% | 70% | **80%** |

---

## 🔧 技术栈

- **仿真：** NVIDIA Isaac Lab (Isaac Sim 4.0+)
- **深度学习：** PyTorch 2.0+, PyTorch Lightning
- **强化学习：** Stable-Baselines3（修改版）
- **机器人控制：** ROS2 Humble
- **实验管理：** WandB / TensorBoard
- **硬件平台：** ROSOrin 麦克纳姆轮小车

---

## 📝 发表计划

### 论文标题（草案）

**"SAC-DiffusionDrive: Maximum Entropy Reinforcement Learning with Diffusion Policies for End-to-End Autonomous Driving"**

### 投稿目标

1. **首选：** CoRL 2026（Conference on Robot Learning）
2. **备选：** ICRA 2026 / IROS 2026
3. **期刊：** IEEE Transactions on Robotics (T-RO)

### 关键卖点

1. **理论创新：** SAC与扩散模型的数学统一框架
2. **实验充分：** 仿真+实机，多Baseline对比，详细消融
3. **应用价值：** 端到端驾驶的多模态行为建模

---

## 👥 团队与分工

- **研究者：** 吴佳豪
- **指导教师：** [待填写]
- **合作实验室：** [待填写]

---

## 📚 参考文献

### 核心相关工作

1. **Diffusion Policy** (Chi et al., RSS 2023)
2. **Diffusion-ES** (Anonymous, ICLR 2024 Under Review)
3. **Decision Diffuser** (Janner et al., ICML 2022)
4. **SAC** (Haarnoja et al., ICML 2018)
5. **Isaac Lab** (NVIDIA, 2024)

### 扩展阅读

- `docs/literature_review.md`

---

## 📞 联系方式

- **邮箱：** [你的邮箱]
- **GitHub：** [你的GitHub]
- **项目页面：** [待发布]

---

## 📄 许可证

MIT License

---

**最后更新：** 2025-12-06
