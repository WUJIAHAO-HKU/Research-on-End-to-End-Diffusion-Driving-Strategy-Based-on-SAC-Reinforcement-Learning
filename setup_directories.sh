#!/bin/bash
# 项目结构生成脚本

echo "生成项目目录结构..."

# 创建所有必需的目录
mkdir -p src/envs/isaac_lab
mkdir -p src/envs/wrappers
mkdir -p src/models/encoders
mkdir -p src/models/diffusion
mkdir -p src/models/sac
mkdir -p src/models/baselines
mkdir -p src/algorithms
mkdir -p src/data
mkdir -p src/utils
mkdir -p src/sim2real
mkdir -p scripts
mkdir -p configs/env
mkdir -p configs/model
mkdir -p configs/training
mkdir -p configs/experiment
mkdir -p configs/sim2real
mkdir -p experiments/logs
mkdir -p experiments/checkpoints
mkdir -p experiments/tensorboard
mkdir -p experiments/videos
mkdir -p experiments/results
mkdir -p notebooks
mkdir -p tests
mkdir -p docs
mkdir -p data/demonstrations
mkdir -p data/real_world

# 创建 __init__.py 文件
touch src/__init__.py
touch src/envs/__init__.py
touch src/envs/isaac_lab/__init__.py
touch src/envs/wrappers/__init__.py
touch src/models/__init__.py
touch src/models/encoders/__init__.py
touch src/models/diffusion/__init__.py
touch src/models/sac/__init__.py
touch src/models/baselines/__init__.py
touch src/algorithms/__init__.py
touch src/data/__init__.py
touch src/utils/__init__.py
touch src/sim2real/__init__.py

echo "✓ 目录结构创建完成！"
echo "请查看 PROJECT_SUMMARY.md 了解下一步行动"
