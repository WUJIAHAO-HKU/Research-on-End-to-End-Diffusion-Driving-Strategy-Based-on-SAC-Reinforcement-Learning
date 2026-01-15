# Baseline实验说明文档

**创建日期**: 2025年12月29日  
**项目**: 基于SAC强化学习的端到端扩散驾驶策略研究

---

## 📋 概述

本文档说明了为验证SAC-Diffusion策略优势而添加的所有baseline实验。

---

## 🎯 Baseline列表

### 1. 纯SAC (SAC Pure)
**文件**: `scripts/train_sac_pure.py`

**目的**: 验证扩散策略的必要性

**特点**:
- ✅ 标准SAC算法，完全移除Diffusion Policy
- ✅ 使用高斯策略网络（与SAC-Gaussian相同的actor）
- ✅ Twin Q-networks + 自动熵调节
- ✅ 可选BC预训练初始化

**运行命令**:
```bash
./isaaclab_runner.sh scripts/train_sac_pure.py \
  --num_envs 4 \
  --total_steps 1000000 \
  --batch_size 256 \
  --buffer_size 50000
```

**关键区别**: 
- ❌ 无扩散模型
- ✅ 简单高斯策略，直接输出mean和std

---

### 2. PPO (Proximal Policy Optimization)
**文件**: `scripts/train_ppo.py`

**目的**: 对比on-policy vs off-policy，验证BC预训练的作用

**特点**:
- ✅ 标准PPO算法（on-policy）
- ✅ Actor-Critic架构
- ✅ GAE优势估计
- ✅ 支持两种模式：
  - 从头训练（`ppo_scratch`）
  - BC预训练初始化（`ppo_with_bc`）

**运行命令**:
```bash
# 从头训练
./isaaclab_runner.sh scripts/train_ppo.py \
  --num_envs 16 \
  --total_steps 1000000

# 使用BC预训练（消融实验）
./isaaclab_runner.sh scripts/train_ppo.py \
  --num_envs 16 \
  --total_steps 1000000 \
  --pretrain_checkpoint experiments/bc_training/bc_training_XXX/best_model.pt
```

**关键区别**: 
- ✅ On-policy学习
- ✅ 需要更多并行环境
- ✅ 用于BC预训练消融研究

---

### 3. TD3 (Twin Delayed DDPG)
**文件**: `scripts/train_td3.py`

**目的**: 对比确定性策略 vs 随机策略

**特点**:
- ✅ 确定性Actor
- ✅ Twin Q-networks
- ✅ Delayed policy updates
- ✅ Target policy smoothing
- ✅ 使用现有的`src/baselines/td3_agent.py`

**运行命令**:
```bash
./isaaclab_runner.sh scripts/train_td3.py \
  --num_envs 4 \
  --total_steps 1000000 \
  --policy_delay 2 \
  --expl_noise 0.1
```

**关键区别**: 
- ✅ 确定性策略（加探索噪声）
- ✅ 延迟策略更新
- ✅ 目标策略平滑

---

### 4. SAC-Gaussian (标准高斯策略)
**文件**: `scripts/train_sac_gaussian.py`

**目的**: **证明扩散策略优势** - 最重要的对比

**特点**:
- ✅ 标准SAC + 高斯策略
- ✅ Squashed Gaussian (tanh变换)
- ✅ 自动熵调节
- ✅ 使用现有的`src/baselines/sac_gaussian.py`

**运行命令**:
```bash
./isaaclab_runner.sh scripts/train_sac_gaussian.py \
  --num_envs 4 \
  --total_steps 1000000 \
  --auto_tune_alpha
```

**关键区别**: 
- ❌ **不使用扩散策略**
- ✅ **标准高斯分布**（mean + std）
- 🎯 **与SAC-Diffusion直接对比**

---

### 5. DAgger (Dataset Aggregation)
**文件**: `scripts/train_dagger.py`

**目的**: 介于BC和RL之间的方法，验证迭代学习的效果

**特点**:
- ✅ 迭代收集专家数据
- ✅ 逐步减少对专家的依赖
- ✅ Beta schedule控制专家混合率
- ✅ 支持3种衰减策略：constant, linear, exponential

**运行命令**:
```bash
./isaaclab_runner.sh scripts/train_dagger.py \
  --num_envs 4 \
  --n_iterations 20 \
  --steps_per_iteration 50000 \
  --beta_schedule linear \
  --initial_beta 1.0 \
  --final_beta 0.1
```

**关键区别**: 
- ✅ 混合BC和RL
- ✅ 需要专家（MPC）在线提供标签
- ✅ 逐步降低专家依赖

---

## 📊 实验对比矩阵

| Baseline | 策略类型 | BC预训练 | 在线学习 | 专家需求 | 主要优势 |
|----------|---------|---------|---------|---------|---------|
| **SAC-Diffusion** | 扩散策略 | ✅ | ✅ | 离线 | 多模态、鲁棒 |
| **SAC Pure** | 高斯 | ❌ | ✅ | 无 | 简单快速 |
| **SAC-Gaussian** | 高斯 | ❌ | ✅ | 无 | 标准SAC |
| **TD3** | 确定性 | ❌ | ✅ | 无 | 稳定、简单 |
| **PPO** | 高斯 | 可选 | ✅ | 无 | On-policy |
| **DAgger** | 行为克隆 | ✅ | ✅ | **在线** | 迭代改进 |

---

## 🔬 实验设计

### A. 扩散策略优势验证
**对比**: SAC-Diffusion vs SAC-Gaussian

**度量指标**:
- 平均奖励
- 成功率
- 动作平滑性
- 多模态能力

**预期结果**: SAC-Diffusion在复杂场景中表现更好

---

### B. BC预训练消融实验
**对比**: PPO (scratch) vs PPO (with BC)

**度量指标**:
- 训练速度（达到阈值的步数）
- 最终性能
- 样本效率

**预期结果**: BC预训练显著加速训练

---

### C. 强化学习算法对比
**对比**: SAC vs TD3 vs PPO

**度量指标**:
- 样本效率
- 最终性能
- 训练稳定性

**预期结果**: SAC类方法样本效率更高

---

### D. 监督vs强化学习
**对比**: BC → DAgger → SAC-Diffusion

**度量指标**:
- 泛化能力
- 对新场景的适应性

**预期结果**: RL方法泛化能力更强

---

## 📁 输出目录结构

```
experiments/baselines/
├── sac_pure/
│   └── sac_pure_YYYYMMDD_HHMMSS/
│       ├── checkpoints/
│       │   ├── best_model.pt
│       │   ├── final_model.pt
│       │   └── checkpoint_*.pt
│       ├── config.yaml
│       ├── training_log.json
│       └── summary.json
│
├── sac_gaussian/
│   └── sac_gaussian_YYYYMMDD_HHMMSS/
│       └── ...
│
├── td3/
│   └── td3_YYYYMMDD_HHMMSS/
│       └── ...
│
├── ppo/
│   ├── ppo_scratch_YYYYMMDD_HHMMSS/
│   └── ppo_with_bc_YYYYMMDD_HHMMSS/
│       └── ...
│
└── dagger/
    └── dagger_YYYYMMDD_HHMMSS/
        └── ...
```

---

## 🚀 快速开始

### 1. 训练所有baselines

```bash
# 1. SAC Pure
./isaaclab_runner.sh scripts/train_sac_pure.py --num_envs 4 --total_steps 1000000

# 2. SAC-Gaussian
./isaaclab_runner.sh scripts/train_sac_gaussian.py --num_envs 4 --total_steps 1000000

# 3. TD3
./isaaclab_runner.sh scripts/train_td3.py --num_envs 4 --total_steps 1000000

# 4. PPO (从头)
./isaaclab_runner.sh scripts/train_ppo.py --num_envs 16 --total_steps 1000000

# 5. PPO (BC预训练)
./isaaclab_runner.sh scripts/train_ppo.py --num_envs 16 --total_steps 1000000 \
  --pretrain_checkpoint experiments/bc_training/bc_training_XXX/best_model.pt

# 6. DAgger
./isaaclab_runner.sh scripts/train_dagger.py --num_envs 4 --n_iterations 20
```

### 2. 评估所有baselines

可以使用修改后的评估脚本：

```bash
# TODO: 创建统一的baseline评估脚本
./isaaclab_runner.sh scripts/evaluate_baselines.py \
  --baselines sac_pure sac_gaussian td3 ppo dagger \
  --num_episodes 50
```

---

## 📈 预期结果

### 性能排序（预测）
1. **SAC-Diffusion** - 最佳（多模态、鲁棒）
2. **SAC-Gaussian** - 次佳（标准SAC）
3. **PPO (with BC)** - 良好（有预训练）
4. **TD3** - 中等（确定性策略）
5. **PPO (scratch)** - 中等（需要更多样本）
6. **DAgger** - 取决于专家质量
7. **SAC Pure** - 基线

### 关键发现（预期）
- ✅ 扩散策略在复杂场景中优于高斯策略
- ✅ BC预训练显著加速训练
- ✅ SAC类算法样本效率高于PPO
- ✅ DAgger性能介于BC和SAC之间

---

## ⚠️ 注意事项

### DAgger的MPC专家
当前DAgger使用**简化的启发式专家**。为获得最佳结果，应该：

```python
# 在train_dagger.py中替换MPCExpert
from scripts.mpc_controller import NonlinearMPCController

class MPCExpert:
    def __init__(self):
        self.mpc = NonlinearMPCController(horizon=10)
    
    def get_action(self, obs):
        # 使用真实MPC求解
        return self.mpc.compute_control(obs)
```

### 计算资源
- PPO需要更多并行环境（推荐16+）
- DAgger需要在线专家，计算开销较大
- 其他baseline可以在4个环境上训练

---

## 📚 参考文献

1. **SAC**: Haarnoja et al., "Soft Actor-Critic", ICML 2018
2. **TD3**: Fujimoto et al., "Addressing Function Approximation Error", ICML 2018
3. **PPO**: Schulman et al., "Proximal Policy Optimization", arXiv 2017
4. **DAgger**: Ross et al., "A Reduction of Imitation Learning", AISTATS 2011
5. **Diffusion Policy**: Chi et al., "Diffusion Policy", RSS 2023

---

## ✅ 完成状态

- [x] SAC Pure (无Diffusion)
- [x] SAC-Gaussian (标准高斯策略)
- [x] TD3
- [x] PPO (支持BC预训练消融)
- [x] DAgger
- [ ] 统一评估脚本
- [ ] 结果可视化脚本
- [ ] 性能对比表格生成

---

**最后更新**: 2025年12月29日
