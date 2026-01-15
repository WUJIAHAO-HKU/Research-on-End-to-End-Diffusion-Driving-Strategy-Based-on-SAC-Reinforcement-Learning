# GitHub 上传流程指南

本文档提供完整的项目上传到 GitHub 的步骤说明。

---

## 📋 项目信息

- **仓库地址**: `git@github.com:WUJIAHAO-HKU/Research-on-End-to-End-Diffusion-Driving-Strategy-Based-on-SAC-Reinforcement-Learning.git`
- **用户名**: `WUJIAHAO-HKU`
- **邮箱**: `u3661739@connect.hku.hk`
- **项目路径**: `~/ROSORIN_CAR and Reasearch/Research on End-to-End Diffusion Driving Strategy Based on SAC Reinforcement Learning`

---

## 🚀 快速开始（首次上传）

### 步骤 1: 配置 Git 用户信息

```bash
cd ~/ROSORIN_CAR\ and\ Reasearch/Research\ on\ End-to-End\ Diffusion\ Driving\ Strategy\ Based\ on\ SAC\ Reinforcement\ Learning

# 配置用户名和邮箱
git config user.name "WUJIAHAO-HKU"
git config user.email "u3661739@connect.hku.hk"

# 验证配置
git config --list | grep user
```

### 步骤 2: 初始化 Git 仓库

```bash
# 初始化本地仓库
git init

# 添加远程仓库
git remote add origin git@github.com:WUJIAHAO-HKU/Research-on-End-to-End-Diffusion-Driving-Strategy-Based-on-SAC-Reinforcement-Learning.git

# 验证远程仓库
git remote -v
```

### 步骤 3: 配置 SSH 密钥（如果尚未配置）

```bash
# 检查是否已有 SSH 密钥
ls -la ~/.ssh

# 如果没有，生成新的 SSH 密钥
ssh-keygen -t ed25519 -C "u3661739@connect.hku.hk"
# 按 Enter 使用默认路径，可选择设置密码

# 启动 SSH 代理
eval "$(ssh-agent -s)"

# 添加私钥到 SSH 代理
ssh-add ~/.ssh/id_ed25519

# 复制公钥到剪贴板
cat ~/.ssh/id_ed25519.pub
# 手动复制输出内容，然后到 GitHub → Settings → SSH and GPG keys → New SSH key 添加
```

### 步骤 4: 创建 .gitignore 文件

```bash
# 创建 .gitignore 文件（排除大文件和敏感数据）
cat > .gitignore << 'EOF'
# ========================================
# Python
# ========================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# ========================================
# 虚拟环境
# ========================================
venv/
env/
ENV/
miniconda3/
Miniconda3-latest-Linux-x86_64.sh

# ========================================
# IDE / 编辑器
# ========================================
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# ========================================
# 机器学习 / 深度学习
# ========================================
# 训练数据和数据集
data/demonstrations/*.pkl
data/demonstrations/*.hdf5
data/demonstrations/*.npz
data/real_world/
*.bag
*.mcap

# 模型权重和检查点
experiments/checkpoints/*.pt
experiments/checkpoints/*.pth
experiments/checkpoints/*.ckpt
experiments/sac_training/*/checkpoints/*.pt
experiments/bc_training/*/checkpoints/*.pt
experiments/baselines/*/checkpoints/*.pt

# TensorBoard 日志（过大）
experiments/tensorboard/
experiments/logs/*.tfevents.*
tensorboard_logs/

# 训练输出视频（可选择性上传）
experiments/videos/*.mp4
experiments/videos/*.avi

# 训练日志（CSV文件可以保留，但过大可排除）
# experiments/*/metrics.csv
# experiments/*/episodes.csv

# ========================================
# Isaac Lab / Simulation
# ========================================
# Isaac Sim 缓存
_isaac_sim/
.isaac_sim/
logs/
*.log

# USD 缓存文件
*.usd
*.usda
*.usdc
data/assets/*.usd

# URDF/MJCF 转换输出
rosorin_ws/build/
rosorin_ws/install/
rosorin_ws/log/

# ========================================
# ROS
# ========================================
rosorin_ws/build/
rosorin_ws/install/
rosorin_ws/log/
*.bag

# ========================================
# 系统文件
# ========================================
*.pyc
*.pyo
*.tmp
*.bak
*~
.cache/
.pytest_cache/

# ========================================
# 大文件 / 临时文件
# ========================================
# 图表输出
figures/*.png
figures/*.pdf

# Jupyter Notebook 检查点
.ipynb_checkpoints/

# 其他临时文件
tmp/
temp/
*.tmp

# ========================================
# 特殊排除（根据需要调整）
# ========================================
# 如果六房间场景 URDF 文件过大，也可排除
# rosorin_ws/rosorin_full.urdf
# rosorin_ws/rosorin_full_backup.urdf

# 如果教程资料过大，排除
ROSOrin智能视觉小车/

# 如果 figures 已生成，保留一份最新的即可
# figures/sac_latest/*.png
EOF

echo ".gitignore 文件已创建"
```

### 步骤 5: 添加文件到 Git

```bash
# 查看将要添加的文件
git status

# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 查看暂存区文件
git status

# 如果发现有不应该添加的大文件，可以取消暂存
# git reset HEAD <file>
```

### 步骤 6: 首次提交

```bash
# 提交到本地仓库
git commit -m "Initial commit: SAC-Diffusion driving strategy project

- Add complete project structure
- Add SAC training scripts with reward normalization
- Add baseline comparison (BC, DAgger, PPO, TD3)
- Add six-room navigation environment
- Add LiDAR + Depth camera integration
- Add visualization and analysis tools
- Add comprehensive documentation"

# 查看提交历史
git log --oneline
```

### 步骤 7: 推送到 GitHub

```bash
# 创建主分支（如果还没有）
git branch -M main

# 首次推送（设置上游分支）
git push -u origin main

# 如果遇到 "rejected" 错误（远程仓库已有内容），执行：
# git pull origin main --allow-unrelated-histories
# 然后再次推送
```

---

## 🔄 日常更新流程

### 修改代码后上传

```bash
# 1. 查看修改内容
git status
git diff

# 2. 添加修改的文件
git add <文件名>
# 或添加所有修改
git add .

# 3. 提交修改
git commit -m "描述本次修改的内容"

# 示例：
# git commit -m "Fix: 降低避障保守性，添加冲刺奖励"
# git commit -m "Feat: 添加距离奖励替代进度奖励"
# git commit -m "Update: 优化 SAC 学习率和梯度裁剪"

# 4. 推送到 GitHub
git push origin main
```

### 拉取远程更新

```bash
# 拉取远程仓库的最新代码
git pull origin main

# 如果有冲突，手动解决后：
git add <冲突文件>
git commit -m "Merge: 解决冲突"
git push origin main
```

---

## 📦 处理大文件（如果需要）

### 使用 Git LFS（Large File Storage）

如果项目中有无法避免的大文件（如模型权重、数据集），可使用 Git LFS：

```bash
# 安装 Git LFS
sudo apt-get install git-lfs  # Ubuntu/Debian
# 或
brew install git-lfs  # macOS

# 初始化 Git LFS
git lfs install

# 追踪大文件类型
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.hdf5"
git lfs track "*.npz"

# 添加 .gitattributes
git add .gitattributes

# 提交和推送
git add <大文件>
git commit -m "Add large model files via Git LFS"
git push origin main
```

---

## 🛠️ 常用命令速查

### 查看状态

```bash
git status              # 查看工作区状态
git log --oneline       # 查看提交历史
git log --graph         # 图形化显示分支
git diff                # 查看未暂存的修改
git diff --staged       # 查看已暂存的修改
```

### 撤销操作

```bash
# 撤销工作区修改（危险操作）
git checkout -- <文件名>

# 撤销暂存区的文件
git reset HEAD <文件名>

# 撤销最近一次提交（保留修改）
git reset --soft HEAD^

# 撤销最近一次提交（丢弃修改，危险操作）
git reset --hard HEAD^
```

### 分支操作

```bash
# 查看所有分支
git branch -a

# 创建新分支
git branch <分支名>

# 切换分支
git checkout <分支名>

# 创建并切换分支
git checkout -b <分支名>

# 合并分支
git merge <分支名>

# 删除本地分支
git branch -d <分支名>
```

### 远程仓库

```bash
# 查看远程仓库
git remote -v

# 添加远程仓库
git remote add <名称> <URL>

# 修改远程仓库 URL
git remote set-url origin <新URL>

# 删除远程仓库
git remote remove <名称>
```

---

## 🔧 故障排查

### 问题 1: Permission denied (publickey)

**原因**: SSH 密钥未配置或未添加到 GitHub

**解决**:
```bash
# 1. 检查 SSH 密钥
ls -la ~/.ssh

# 2. 测试 SSH 连接
ssh -T git@github.com

# 3. 如果失败，重新配置 SSH（见步骤 3）
```

### 问题 2: fatal: refusing to merge unrelated histories

**原因**: 本地仓库和远程仓库没有共同的提交历史

**解决**:
```bash
git pull origin main --allow-unrelated-histories
# 解决冲突后
git push origin main
```

### 问题 3: remote: fatal: pack exceeds maximum allowed size

**原因**: 单次推送文件过大（GitHub 限制单个文件 100MB）

**解决**:
```bash
# 1. 使用 Git LFS（见上文）
# 2. 或从提交中移除大文件
git rm --cached <大文件>
git commit --amend
git push origin main
```

### 问题 4: 推送速度很慢

**原因**: 文件过多或网络问题

**解决**:
```bash
# 1. 压缩仓库
git gc --aggressive --prune=now

# 2. 使用 HTTPS 代替 SSH（如果需要）
git remote set-url origin https://github.com/WUJIAHAO-HKU/Research-on-End-to-End-Diffusion-Driving-Strategy-Based-on-SAC-Reinforcement-Learning.git

# 3. 增加缓冲区大小
git config http.postBuffer 524288000  # 500MB
```

---

## 📝 提交信息规范（推荐）

遵循语义化提交信息规范：

```bash
# 格式: <类型>: <简短描述>

# 类型:
# - Feat: 新功能
# - Fix: 修复bug
# - Update: 更新现有功能
# - Docs: 文档修改
# - Style: 代码格式调整（不影响功能）
# - Refactor: 代码重构
# - Test: 测试相关
# - Chore: 构建过程或辅助工具变动

# 示例:
git commit -m "Feat: 添加冲刺奖励机制"
git commit -m "Fix: 修复避障惩罚权重过大问题"
git commit -m "Update: 优化 SAC 学习率为 1e-4"
git commit -m "Docs: 更新 README 添加训练结果"
```

---

## 📊 建议上传的文件结构

```
✅ 应该上传:
├── configs/                  # 配置文件
├── scripts/                  # 训练脚本
├── src/                      # 源代码
├── notebooks/                # Jupyter notebooks
├── docs/                     # 文档
├── requirements.txt          # 依赖列表
├── setup.py                  # 安装脚本
├── README.md                 # 项目说明
├── *.md                      # 其他文档
└── experiments/
    ├── results/              # 结果摘要（小文件）
    └── baseline_comparison/  # 基准测试结果

❌ 不应上传（已在 .gitignore）:
├── experiments/checkpoints/  # 模型权重（过大）
├── experiments/tensorboard/  # TensorBoard 日志
├── experiments/videos/       # 训练视频
├── data/demonstrations/      # 训练数据
├── rosorin_ws/build/         # ROS 编译文件
├── rosorin_ws/install/       # ROS 安装文件
├── __pycache__/              # Python 缓存
└── ROSOrin智能视觉小车/      # 教程资料（过大）
```

---

## ✅ 检查清单

首次上传前请确认：

- [ ] 已配置 Git 用户信息（用户名和邮箱）
- [ ] 已添加 SSH 密钥到 GitHub
- [ ] 已创建 `.gitignore` 文件
- [ ] 已检查暂存区文件（`git status`），确认无大文件
- [ ] 已编写清晰的提交信息
- [ ] 已测试 SSH 连接（`ssh -T git@github.com`）
- [ ] README.md 已更新项目说明
- [ ] 敏感信息（API密钥、密码等）已排除

---

## 📚 相关资源

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub 帮助文档](https://docs.github.com)
- [Git LFS 文档](https://git-lfs.github.com/)
- [语义化版本控制](https://semver.org/lang/zh-CN/)
- [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)

---

## 🎯 快速执行脚本

将以下内容保存为 `quick_upload.sh`，首次上传时一键执行：

```bash
#!/bin/bash

echo "=========================================="
echo "  GitHub 快速上传脚本"
echo "=========================================="

# 进入项目目录
cd ~/ROSORIN_CAR\ and\ Reasearch/Research\ on\ End-to-End\ Diffusion\ Driving\ Strategy\ Based\ on\ SAC\ Reinforcement\ Learning

# 配置用户信息
git config user.name "WUJIAHAO-HKU"
git config user.email "u3661739@connect.hku.hk"

# 初始化仓库
if [ ! -d ".git" ]; then
    git init
    git remote add origin git@github.com:WUJIAHAO-HKU/Research-on-End-to-End-Diffusion-Driving-Strategy-Based-on-SAC-Reinforcement-Learning.git
fi

# 添加文件
git add .

# 提交
echo "请输入提交信息 (或按 Enter 使用默认信息):"
read commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="Update: 项目更新 $(date +'%Y-%m-%d %H:%M:%S')"
fi

git commit -m "$commit_msg"

# 推送
git branch -M main
git push -u origin main

echo "=========================================="
echo "  上传完成！"
echo "=========================================="
```

使用方法：
```bash
chmod +x quick_upload.sh
./quick_upload.sh
```

---

**最后更新**: 2026年1月15日  
**维护者**: WUJIAHAO-HKU (u3661739@connect.hku.hk)
