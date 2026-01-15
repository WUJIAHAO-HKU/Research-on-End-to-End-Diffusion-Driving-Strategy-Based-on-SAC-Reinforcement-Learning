# 快速开始指南

## 🚀 环境配置

### 1. 系统要求

- **操作系统**：Ubuntu 20.04/22.04
- **GPU**：NVIDIA RTX 3080 或更高（建议 RTX 4090 / A100）
- **CUDA**：12.1+
- **内存**：32GB+ RAM
- **存储**：100GB+ 可用空间

### 2. 安装 Isaac Lab

```bash
# 安装 NVIDIA Isaac Sim
# 访问：https://developer.nvidia.com/isaac-sim
# 下载并安装 Isaac Sim 4.0+

# 克隆 Isaac Lab
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 安装 Isaac Lab
./isaaclab.sh --install

# 验证安装
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
```

### 3. 创建项目环境

```bash
# 进入项目目录
cd "/home/wujiahao/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning"

# 创建 Conda 环境
conda env create -f environment.yml
conda activate sac-diffusion-driving

# 安装项目
pip install -e .

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 4. 配置 Isaac Lab 集成

```bash
# 链接 Isaac Lab 到项目
export ISAACLAB_PATH=~/IsaacLab
export PYTHONPATH=$PYTHONPATH:$ISAACLAB_PATH/source

# 添加到 ~/.bashrc
echo 'export ISAACLAB_PATH=~/IsaacLab' >> ~/.bashrc
echo 'export PYTHONPATH=$PYTHONPATH:$ISAACLAB_PATH/source' >> ~/.bashrc
```

---

## 📊 数据准备

### 方案 A：使用专家演示数据（推荐）

```bash
# Step 1: 收集 MPC 专家数据
python scripts/collect_demonstrations.py \
    --config configs/env/isaac_lab_base.yaml \
    --robot_config configs/env/rosorin_mecanum.yaml \
    --num_episodes 1000 \
    --output_dir data/demonstrations/mpc_expert \
    --expert_type mpc

# 预期输出：
# - data/demonstrations/mpc_expert/
#   ├── episode_0000.hdf5
#   ├── episode_0001.hdf5
#   ├── ...
#   └── metadata.json (约 5-10GB)
```

### 方案 B：使用预训练模型

```bash
# 下载预训练的 Diffusion Policy（行为克隆）
wget https://your-server.com/pretrained_bc_model.pth -O checkpoints/bc_pretrain.pth
```

---

## 🎓 训练流程

### Phase 1: 行为克隆预训练（2-3天）

```bash
# 使用专家数据预训练 Diffusion Policy
python scripts/train_bc.py \
    --config configs/training/bc_pretrain.yaml \
    --data_dir data/demonstrations/mpc_expert \
    --output_dir experiments/bc_pretrain_run1 \
    --num_epochs 200 \
    --batch_size 256 \
    --gpus 1

# 监控训练
tensorboard --logdir experiments/bc_pretrain_run1/tensorboard
```

**预期结果**：
- 训练损失：< 0.01
- 验证损失：< 0.02
- 模仿成功率：> 85%

### Phase 2: SAC-Diffusion 强化学习微调（5-7天）

```bash
# 使用预训练模型初始化，进行 RL 微调
python scripts/train_sac_diffusion.py \
    --config configs/training/sac_finetuning.yaml \
    --pretrained_model experiments/bc_pretrain_run1/best_model.pth \
    --num_envs 64 \
    --max_training_steps 1000000 \
    --output_dir experiments/sac_diffusion_run1 \
    --wandb true

# 使用 WandB 监控
# 访问：https://wandb.ai/your-username/sac-diffusion-driving
```

**训练曲线检查点**：
- 10K steps：策略开始产生合理动作
- 100K steps：路径跟踪性能提升
- 500K steps：开始超越 MPC 专家
- 1M steps：收敛到最优策略

### Phase 3: 超参数搜索（可选）

```bash
# 使用 Optuna 进行超参数优化
python scripts/hyperparameter_search.py \
    --config configs/training/hyperparameters.yaml \
    --num_trials 50 \
    --output_dir experiments/hyperparam_search

# 最优超参数会自动保存
```

---

## 📈 评估

### 单次评估

```bash
# 评估训练好的模型
python scripts/evaluate.py \
    --config configs/experiment/baseline_comparison.yaml \
    --checkpoint experiments/sac_diffusion_run1/checkpoints/step_1000000.pth \
    --num_episodes 100 \
    --render true \
    --save_video true \
    --output_dir results/evaluation_run1
```

### Baseline 对比实验

```bash
# 运行完整对比实验（需要先训练所有 baseline）
bash experiments/run_baseline_comparison.sh

# 包含的 baseline：
# 1. MPC (专家)
# 2. 标准 Diffusion Policy (无 RL)
# 3. SAC-Gaussian Policy
# 4. TD3
# 5. SAC-Diffusion (本文方法)
```

### 生成结果表格和可视化

```bash
# 分析结果并生成论文图表
python scripts/visualize_results.py \
    --results_dir results/ \
    --output_dir paper_figures/

# 输出：
# - paper_figures/comparison_table.tex
# - paper_figures/learning_curves.pdf
# - paper_figures/success_rate_bar.pdf
# - paper_figures/trajectory_visualization.pdf
```

---

## 🤖 实机部署

### Step 1: 在仿真中测试部署流程

```bash
# 模拟实机延迟和噪声
python scripts/evaluate.py \
    --config configs/sim2real/rosorin_deployment.yaml \
    --checkpoint experiments/sac_diffusion_run1/best_model.pth \
    --add_latency true \
    --add_sensor_noise true
```

### Step 2: 连接真实 ROSOrin 小车

```bash
# 启动 ROS2 节点
ros2 launch rosorin_bringup bringup.launch.py

# 在另一个终端部署策略
python scripts/deploy_to_robot.py \
    --config configs/sim2real/rosorin_deployment.yaml \
    --checkpoint experiments/sac_diffusion_run1/best_model.pth \
    --ros2_namespace /rosorin \
    --safety_monitor true \
    --max_speed 0.5
```

### Step 3: 实机数据收集与微调

```bash
# 收集真实世界数据
python scripts/collect_real_world_data.py \
    --num_episodes 50 \
    --output_dir data/real_world

# 在真实数据上微调
python scripts/finetune_on_real_data.py \
    --config configs/training/real_world_finetuning.yaml \
    --pretrained_model experiments/sac_diffusion_run1/best_model.pth \
    --real_data_dir data/real_world
```

---

## 🔧 常见问题

### Q1: Isaac Lab 安装失败

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version

# 确保 Isaac Sim 正确安装
cd ~/IsaacLab
./isaaclab.sh --help
```

### Q2: 训练过程中 GPU 内存不足

**解决方案**：
```yaml
# 修改 configs/training/sac_finetuning.yaml
num_envs: 32  # 从 64 降到 32
batch_size: 128  # 从 256 降到 128
```

### Q3: 扩散模型采样太慢

**解决方案**：
```yaml
# 修改 configs/model/diffusion_policy.yaml
diffusion:
  num_diffusion_steps: 10  # 从 20 降到 10
  sampling:
    method: "ddim"  # 使用 DDIM 加速
    ddim:
      num_inference_steps: 5  # 推理时仅用 5 步
```

### Q4: 策略在仿真中表现好，但实机失败

**解决方案**：
1. 增强域随机化：
```yaml
# configs/env/isaac_lab_base.yaml
domain_randomization:
  enabled: true
  mass_scale: [0.7, 1.3]  # 增大范围
  friction_scale: [0.5, 1.5]
  camera_noise_std: 0.05  # 增加噪声
```

2. 收集更多真实世界数据进行微调

---

## 📝 实验检查清单

### 训练阶段
- [ ] 行为克隆预训练完成（> 85% 模仿成功率）
- [ ] SAC-Diffusion 微调完成（> 90% 任务成功率）
- [ ] 训练曲线平滑收敛（无异常波动）
- [ ] 模型检查点已保存

### 评估阶段
- [ ] 在 5 种 baseline 上进行对比
- [ ] 至少 100 episodes 评估数据
- [ ] 消融实验完成（扩散步数、熵权重等）
- [ ] 泛化性测试（新场景、新天气）

### 实机阶段
- [ ] 仿真-实机延迟/噪声测试通过
- [ ] 安全监控系统已部署
- [ ] 在真实小车上成功运行 20+ episodes
- [ ] 记录完整视频和数据

### 论文准备
- [ ] 所有实验数据已整理
- [ ] 图表和表格已生成
- [ ] 消融研究结果分析
- [ ] 理论推导已验证
- [ ] 相关工作对比充分

---

## 🎯 预期时间线

| 阶段 | 持续时间 | 关键里程碑 |
|------|---------|-----------|
| 环境配置 | 1-2天 | Isaac Lab 运行，GPU 正常 |
| 数据收集 | 2-3天 | 1000 episodes 专家数据 |
| BC 预训练 | 2-3天 | 模仿成功率 > 85% |
| SAC 微调 | 5-7天 | 任务成功率 > 90% |
| Baseline 训练 | 3-5天 | 所有对比方法完成 |
| 消融实验 | 2-3天 | 理解各组件贡献 |
| 实机测试 | 3-5天 | 真实小车运行成功 |
| 论文撰写 | 2-3周 | 初稿完成 |
| **总计** | **约 6-8周** | 完整研究项目 |

---

## 📞 技术支持

遇到问题？
1. 查看 `docs/implementation_details.md`
2. 检查 GitHub Issues
3. 联系项目维护者

祝研究顺利！🚗💨
