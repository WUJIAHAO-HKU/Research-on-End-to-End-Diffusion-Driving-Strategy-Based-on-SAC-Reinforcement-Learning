# Archive目录

本目录包含已完成历史任务的测试、调试和旧版脚本，归档保留以便参考。

## 📋 脚本分类

### 🧪 测试脚本 (25个)

#### 环境集成测试
- **test_env_integration.py** ✅ - 完整环境集成测试（传感器+观测+动作）
- **test_rosorin_scene.py** ✅ - 场景创建5步测试
- **test_rosorin_quick.py** - 快速场景测试

#### 传感器测试
- **test_camera_obs.py** ✅ - 相机观测空间集成测试
- **test_camera_simple.py** ✅ - 简化相机测试
- **test_camera_lidar.py** - 相机和LiDAR联合测试
- **test_camera_sensor.py** - 相机传感器测试
- **test_sensors_minimal.py** - 最小传感器测试
- **test_contact.py** - 接触传感器测试

#### 机器人运动测试
- **test_robot_motion.py** ✅ - 机器人运动测试（发现固定关节问题）
- **test_basic_actions.py** - 基础动作测试
- **test_gravity.py** - 重力测试

#### 场景测试
- **test_basic_scene.py** - 基础场景测试
- **test_world_basic.py** - 基础世界测试
- **test_simple.py** - 简单测试
- **test_simple_env.py** - 简单环境测试
- **test_minimal.py** - 最小测试

#### 环境逻辑测试
- **test_rosorin_env_logic.py** - 环境逻辑测试
- **test_mpc_collection.py** - MPC数据采集测试

### 🔍 调试和检查脚本 (10个)

#### 传感器调试
- **debug_sensors_step_by_step.py** ✅ - 6步传感器调试流程

#### 机器人检查
- **check_rosorin_joints.py** - 检查关节配置
- **check_robot_prims.py** - 检查机器人primitives

#### USD结构检查
- **inspect_usd.py** - 检查USD文件
- **inspect_usd_structure.py** - 检查USD结构
- **print_body_paths.py** - 打印body路径

### 🔧 修复和转换工具 (旧版本)

#### USD修复工具（已完成任务）
- **fix_fixed_base.py** ✅ - 移除固定关节（关键修复！）
- **fix_usd_physics.py** - 修复USD物理配置
- **remove_root_joint.py** - 移除root joint

#### URDF/USD转换（旧版本）
- **convert_rosorin_urdf.py** - ROSOrin URDF转换（旧版）
- **reconvert_urdf.py** - 重新转换URDF

### 📊 数据采集（旧版本）

已被新版 `collect_mpc_expert_data.py` 替代：
- **collect_demonstrations.py** - 旧版数据采集
- **collect_mpc_demonstrations.py** - 旧版MPC演示采集
- **collect_mpc_demos.py** - 旧版MPC数据采集
- **collect_rosorin_mpc_demos.py** - 旧版ROSOrin MPC采集

---

## ✅ 关键成果

### 1. 固定关节问题解决
**脚本**: fix_fixed_base.py

**问题**：机器人加载后无法移动，轮子旋转但车体静止

**解决**：移除USD中的fixed joints
```python
# 移除的关节
- /car/root_joint (PhysicsFixedJoint)
- /car/joints/base_joint (PhysicsFixedJoint)
```

**结果**：机器人质量从1.385kg → 0.055kg，成功实现运动（~0.16 m/s）

### 2. 传感器集成成功
**脚本**: debug_sensors_step_by_step.py, test_camera_obs.py

**成果**：
- ✅ Contact sensor: 路径修正为 `/Robot/base_link`
- ✅ Camera RGB+Depth: 160x120分辨率
- ✅ 观测空间: 76,810维
- ⚠️ LiDAR: 跳过（mesh配置问题）

### 3. GPU内存优化
**脚本**: test_camera_simple.py

**问题**：640x480相机导致GPU内存溢出（8GB VRAM）

**解决**：降低分辨率至160x120（1/16像素）

**结果**：成功运行，内存充足

### 4. 环境完整验证
**脚本**: test_env_integration.py

**验证内容**：
- ✅ 2个并行环境创建
- ✅ 传感器状态检查（Contact + Camera）
- ✅ 观测空间76,810维
- ✅ 动作空间4维
- ✅ 20步仿真测试
- ✅ 数据一致性验证

---

## 📚 参考价值

这些脚本虽已归档，但包含重要的调试经验和解决方案：

### 传感器集成经验
- Camera需要 `--enable_cameras` flag
- AppLauncher vs SimulationApp选择
- 传感器路径配置（USD层级结构）
- GPU内存限制和优化策略

### 物理仿真调试
- Fixed joint检测和移除
- ArticulationRootAPI配置
- 物理材料和摩擦力设置
- 速度跟踪控制器调优

### 环境集成流程
- 6步传感器调试流程
- 5步场景创建测试
- 观测空间数据流验证
- 多环境并行测试

---

## 🔄 如果需要重新使用

虽然这些脚本已归档，但在以下情况可能需要重新运行：

1. **调试新传感器**：参考 `debug_sensors_step_by_step.py`
2. **USD问题排查**：使用 `inspect_usd_structure.py`
3. **关节配置检查**：运行 `check_rosorin_joints.py`
4. **快速环境测试**：使用 `test_rosorin_quick.py`

### 运行方式
```bash
# 归档脚本需要返回上层目录运行
cd ..
./isaaclab_runner.sh scripts/archive/[script_name].py
```

---

## 📊 统计

- **总脚本数**: 35个
- **测试脚本**: 25个
- **调试脚本**: 5个
- **修复工具**: 3个
- **转换工具**: 2个
- **旧版采集脚本**: 4个

**归档日期**: 2025-12-26
**归档原因**: 核心功能已验证完成，保留作为参考文档

---

**注意**：这些脚本代表了项目开发过程中的重要里程碑，记录了从环境搭建到传感器集成的完整调试过程。建议保留不删除。
