# Scripts目录说明

本目录包含项目的核心可执行脚本。

## 📁 目录结构

```
scripts/
├── archive/                    # 归档的测试和调试脚本
├── rosorin_env_cfg.py         # 🔧 ROSOrin环境配置（核心）
├── collect_mpc_expert_data.py # 📊 MPC专家数据采集
├── train_bc.py                # 🎓 行为克隆预训练
├── train_sac_diffusion.py     # 🚀 SAC-Diffusion主训练
├── evaluate.py                # 📈 模型评估
├── deploy_to_robot.py         # 🤖 真机部署
├── mpc_controller.py          # 🎯 MPC控制器实现
├── path_generator.py          # 🛤️ 路径生成器
├── simple_path_generator.py   # 🛤️ 简化路径生成器
└── [工具脚本]                  # URDF/USD转换等工具
```

---

## 🔧 核心脚本

### 1. rosorin_env_cfg.py
**ROSOrin仿真环境配置**

- **功能**：定义完整的Isaac Lab强化学习环境
- **包含**：
  - 场景配置（地面、机器人、障碍物）
  - 传感器配置（Contact、Camera、LiDAR）
  - 观测空间（76,810维）
  - 动作空间（4维轮速）
  - 奖励函数和终止条件
- **用途**：被训练和评估脚本导入使用

### 2. collect_mpc_expert_data.py
**MPC专家数据采集**

- **功能**：使用MPC控制器采集专家演示数据
- **输出**：HDF5格式数据集
- **用法**：
  ```bash
  ./isaaclab_runner.sh scripts/collect_mpc_expert_data.py \
    --num_envs 8 \
    --num_episodes 100 \
    --difficulty easy \
    --enable_cameras --headless
  ```
- **难度级别**：
  - `easy`: 直线和简单曲线（100 episodes）
  - `medium`: 复杂曲线（50 episodes）
  - `hard`: 随机航点避障（30 episodes）

### 3. train_bc.py
**行为克隆预训练**

- **功能**：使用MPC数据预训练策略网络
- **输入**：MPC专家数据（HDF5）
- **输出**：预训练checkpoint
- **用法**：
  ```bash
  ./isaaclab_runner.sh scripts/train_bc.py \
    data.demonstration_path=data/demonstrations/mpc_expert.hdf5 \
    training.batch_size=256 \
    training.epochs=100
  ```

### 4. train_sac_diffusion.py
**SAC-Diffusion主训练**

- **功能**：使用SAC算法训练扩散策略模型
- **输入**：BC预训练checkpoint（可选）
- **输出**：训练好的模型checkpoint
- **用法**：
  ```bash
  ./isaaclab_runner.sh scripts/train_sac_diffusion.py \
    num_envs=64 \
    agent.training.total_steps=1000000 \
    checkpoint_pretrain=experiments/checkpoints/bc_pretrain.pt \
    logging.wandb.enabled=true
  ```

### 5. evaluate.py
**模型评估**

- **功能**：在仿真环境中评估训练好的模型
- **输出**：成功率、平均奖励、轨迹可视化
- **用法**：
  ```bash
  ./isaaclab_runner.sh scripts/evaluate.py \
    checkpoint=experiments/checkpoints/sac_diffusion_best.pt \
    num_episodes=50
  ```

### 6. deploy_to_robot.py
**真机部署**

- **功能**：将训练好的策略部署到ROSOrin真实机器人
- **依赖**：ROS2 Humble
- **用法**：在ROSOrin小车上运行

---

## 🛠️ 辅助脚本

### MPC相关
- **mpc_controller.py**: MPC控制器类实现
- **path_generator.py**: 完整路径生成器（多种轨迹类型）
- **simple_path_generator.py**: 简化路径生成器（难度级别接口）

### 转换工具
- **urdf_to_usd.py**: URDF转USD（Python脚本）
- **convert_urdf_to_usd.sh**: URDF转USD（Shell脚本）
- **mjcf_to_usd.py**: MJCF转USD
- **urdf_to_mjcf.py**: URDF转MJCF

---

## 📂 Archive目录

包含已归档的测试和调试脚本（35个文件）：

**测试脚本** (`test_*.py`):
- test_env_integration.py - 环境集成测试
- test_camera_obs.py - 相机观测测试
- test_rosorin_scene.py - 场景测试
- 等20+个测试脚本

**调试脚本** (`debug_*.py`, `check_*.py`):
- debug_sensors_step_by_step.py - 传感器调试
- check_rosorin_joints.py - 关节检查
- 等

**转换工具旧版本**:
- fix_fixed_base.py - 修复固定关节（已完成）
- convert_rosorin_urdf.py - URDF转换旧版
- 等

**旧版数据采集脚本**:
- collect_demonstrations.py
- collect_mpc_demos.py
- collect_rosorin_mpc_demos.py

这些脚本已完成历史任务，归档保留以便参考。

---

## 📝 使用流程

### 1️⃣ 数据采集阶段
```bash
# 采集MPC专家数据（不同难度）
./isaaclab_runner.sh scripts/collect_mpc_expert_data.py \
  --num_envs 8 --num_episodes 100 --difficulty easy \
  --enable_cameras --headless
```

### 2️⃣ BC预训练阶段
```bash
# 使用MPC数据预训练
./isaaclab_runner.sh scripts/train_bc.py \
  data.demonstration_path=data/demonstrations/mpc_demos_*.h5
```

### 3️⃣ SAC-Diffusion训练
```bash
# 主训练
./isaaclab_runner.sh scripts/train_sac_diffusion.py \
  num_envs=64 \
  checkpoint_pretrain=experiments/checkpoints/bc_pretrain.pt
```

### 4️⃣ 评估
```bash
# 评估模型
./isaaclab_runner.sh scripts/evaluate.py \
  checkpoint=experiments/checkpoints/best_model.pt
```

### 5️⃣ 真机部署
```bash
# 在ROSOrin小车上
python scripts/deploy_to_robot.py \
  --checkpoint experiments/checkpoints/best_model.pt
```

---

## 🚀 快速开始

1. **环境测试**：
   ```bash
   # 已有测试脚本在archive中，环境已验证工作正常
   ```

2. **开始数据采集**：
   ```bash
   ./isaaclab_runner.sh scripts/collect_mpc_expert_data.py \
     --num_envs 4 --num_episodes 5 --difficulty easy \
     --enable_cameras --headless
   ```

3. **检查数据**：
   ```bash
   python -c "import h5py; f=h5py.File('data/demonstrations/rosorin_mpc_demos_*.h5'); print(f.keys())"
   ```

---

## 📌 注意事项

1. **相机flag**：涉及相机的脚本必须加 `--enable_cameras` 标志
2. **Headless模式**：服务器上运行加 `--headless` 标志
3. **统一启动器**：建议使用 `./isaaclab_runner.sh` 而非直接 `python`
4. **GPU内存**：相机分辨率已优化为160x120以适配8GB VRAM

---

**最后更新**: 2025-12-26
**核心脚本数**: 14个
**归档脚本数**: 35个
