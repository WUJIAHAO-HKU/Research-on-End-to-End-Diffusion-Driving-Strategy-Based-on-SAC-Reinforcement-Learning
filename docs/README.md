# 文档索引

本目录包含ROSOrin驾驶策略项目的所有技术文档。

## 📚 文档列表

### 核心文档

- **[project_structure.md](project_structure.md)** - 项目结构说明
  - 目录组织
  - 模块功能
  - 设计理念
  - 使用指南

- **[training_workflow.md](training_workflow.md)** - 完整训练工作流
  - 数据采集流程
  - 各算法训练步骤
  - 评估方法
  - 问题排查

- **[baselines.md](baselines.md)** - 基线算法说明
  - 6种baseline算法对比
  - 训练配置
  - 性能指标

### 快速入门

- **[quickstart.md](quickstart.md)** - 快速开始指南
  - 环境配置
  - 第一次训练
  - 常见问题

- **[project_summary.md](project_summary.md)** - 项目概要
  - 研究目标
  - 技术路线
  - 主要成果

### 理论基础

- **[theory.md](theory.md)** - 理论与算法原理
  - SAC-Diffusion理论
  - 奖励系统设计
  - 环境建模

## 🗂️ 文档组织

```
docs/
├── README.md                 # 本文件 - 文档索引
├── project_structure.md      # 项目结构（重构v2.0）
├── training_workflow.md      # 训练工作流（完整流程）
├── baselines.md              # 基线算法说明
├── quickstart.md             # 快速入门
├── project_summary.md        # 项目概要
└── theory.md                 # 理论基础
```

## 📖 阅读顺序推荐

### 新手入门
1. [quickstart.md](quickstart.md) - 快速上手
2. [project_structure.md](project_structure.md) - 了解项目组织
3. [training_workflow.md](training_workflow.md) - 学习训练流程

### 开发者
1. [project_structure.md](project_structure.md) - 理解架构
2. [baselines.md](baselines.md) - 了解算法对比
3. [training_workflow.md](training_workflow.md) - 掌握完整流程
4. [theory.md](theory.md) - 深入理论

### 研究者
1. [project_summary.md](project_summary.md) - 研究背景
2. [theory.md](theory.md) - 算法原理
3. [baselines.md](baselines.md) - 实验设计
4. [training_workflow.md](training_workflow.md) - 实验复现

## 🔄 最近更新

- **2025-12-30**: 项目重构v2.0完成
  - 奖励配置分离（configs/rewards/）
  - 脚本分类重组（scripts/training等）
  - 更新所有文档路径

## 📝 文档维护

如需更新文档，请遵循以下规范：
- 使用Markdown格式
- 包含代码示例和使用说明
- 及时更新命令路径（重构后）
- 添加表格和图表增强可读性

---

**项目主页**: [../README.md](../README.md)
