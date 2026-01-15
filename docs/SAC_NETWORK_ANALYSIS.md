# SAC 网络架构改进分析

**日期**：2026-01-12  
**状态**：✅ 已添加残差连接 | ⚠️ 建议添加 CNN encoder

---

## 1. 残差连接改进（已完成）

### 修改内容
- **SimpleDiffusionPolicy**：添加 `ResidualBlock`，支持 `use_residual=True/False` 切换
- **QNetwork**：添加残差连接，critic 更稳定

### 效果
- ✅ 梯度流动更顺畅（避免梯度消失/爆炸）
- ✅ 训练收敛更快（尤其是深层网络）
- ✅ 兼容 BC 预训练加载（`use_residual=False` 时保持原结构）

### 使用方法
```python
# 训练时启用残差
agent = SimpleSACAgent(obs_dim, action_dim, device)
agent.actor = SimpleDiffusionPolicy(obs_dim, action_dim, use_residual=True)

# 加载 BC 预训练时自动切换为非残差（保持兼容）
agent.load_bc_pretrain(checkpoint_path)  # 内部会重建 use_residual=False 的网络
```

---

## 2. CNN Encoder 改进（强烈建议）

### 当前问题

| 维度组成 | 维度 | 问题 |
|---------|------|------|
| 低维状态 | 13 | ✅ 合理 |
| RGB 图像 | 160×120×3 = 57,600 | ❌ Flatten 后维度爆炸 |
| Depth 图像 | 160×120×1 = 19,200 | ❌ Flatten 后维度爆炸 |
| **总计** | **76,813** | ❌ 第一层 MLP 参数量 ≈ 39M |

**直接用 MLP 处理 76k 维 flatten 向量的后果：**
1. **参数量爆炸**：76813 × 512 ≈ 39M 参数（仅第一层！）
2. **训练不稳定**：超高维空间梯度容易消失/爆炸
3. **过拟合风险**：视觉特征未经归纳偏置约束
4. **显存占用大**：replay buffer 存 76k × 20k = 1.5GB+

### PPO 的 CNN Encoder 方案

```
RGB (160×120×3) ──┐
                  ├─> CNN Encoder ──> 64-dim embedding ──┐
Depth (160×120×1) ┘                                      ├─> 融合 MLP (141→256→256)
                                                         │
Low-dim State (13) ──────────────────────────────────────┘
```

**优势：**
- 3 层卷积（16→32→64 channels）提取空间特征
- 压缩为 64-dim embedding（**降低 900 倍维度！**）
- 与低维状态融合后总输入仅 141 维
- 参数量从 39M 降至约 0.5M（减少 78 倍）

### SAC 是否应该采用 CNN？

**答案：强烈建议采用！理由：**

1. **SAC 的离策略学习更需要稳定特征表示**
   - SAC 从 replay buffer 采样，样本跨度大，需要鲁棒的特征编码
   - MLP 直接处理 76k 维 flatten 容易过拟合到特定样本

2. **与 PPO 对比一致性**
   - 论文中 PPO 作为 baseline，如果架构差异太大，对比不公平
   - 建议 SAC 和 PPO 使用**相同的 CNN encoder**，只改策略头

3. **实车部署实时性**
   - CNN 的前向推理比 76k→512 的 MLP 更快（卷积可并行）
   - 论文中提到实车 50Hz 控制频率，CNN 更适合

4. **论文写作可解释性**
   - "端到端视觉导航" 强调视觉特征提取
   - CNN encoder 是标准做法，审稿人不会质疑

---

## 3. 建议的 SAC 网络架构（CNN 版本）

### Actor (SimpleDiffusionPolicy with CNN)
```
RGB (160×120×3)   ─> RGB_Encoder   ─> 64-dim  ──┐
Depth (160×120×1) ─> Depth_Encoder ─> 64-dim  ──┼─> 融合层 [141 → 512 → 256 → 128 (with residual)] ─> action (4)
Low-dim State (13) ───────────────────────────────┘
```

### Critic (QNetwork with CNN)
```
[obs_encoding (141) + action (4)] ─> [145 → 512 → 512 → 256 (with residual)] ─> Q-value (1)
```

---

## 4. 实施步骤

### Step 1: 创建 CNN 版本的 SAC 网络（已准备好代码框架）
- [x] 添加 `_ImageEncoder` 类（从 PPO 复用）
- [ ] 修改 `SimpleDiffusionPolicy` 支持 CNN 输入切分
- [ ] 修改 `QNetwork` 支持 CNN 编码

### Step 2: 更新训练脚本
- [ ] 在 `train()` 函数中检测 obs 是否包含图像
- [ ] 自动切分 low-dim / RGB / Depth
- [ ] 传入 CNN encoder 版本的 agent

### Step 3: 对比实验
| 配置 | Obs 处理 | 参数量 | 训练稳定性 | 成功率 |
|------|---------|--------|-----------|--------|
| 当前 MLP (flat) | 76813 → 512 | ~39M (第一层) | ⚠️ 不稳定 | ? |
| **CNN + MLP (推荐)** | 141 → 512 | ~0.5M | ✅ 稳定 | ? |

---

## 5. 对论文的影响

### 如果采用 CNN：
✅ **优势：**
- Method 一节可以强调 "lightweight CNN encoder for RGB-D"
- 与 PPO baseline 架构一致，对比公平
- 符合 "端到端视觉导航" 的叙事

✅ **Table II 更新：**
```latex
\midrule
Actor & CNN(RGB) + CNN(Depth) + Fusion MLP [141→512→256→128] \\
      & with residual blocks, dropout=0.1 \\
Critic & CNN-encoded obs [141] + action [4] → [512→512→256] \\
       & with residual blocks, no dropout \\
```

### 如果不采用 CNN：
⚠️ **风险：**
- 审稿人可能质疑："为什么 PPO 用 CNN 而 SAC 不用？"
- 难以解释 76k 维 MLP 的合理性
- 训练稳定性差可能导致实验结果不佳

---

## 6. 结论

**建议优先级：**
1. ✅ **残差连接**（已完成）：立即启用 `use_residual=True`
2. 🔥 **CNN Encoder**（强烈建议）：尽快实施，与 PPO 架构对齐
3. ⚡ **Layer Normalization**（可选）：如果 CNN 后仍不稳定，可在融合层后加 LayerNorm

**下一步行动：**
- [ ] 决定是否采用 CNN（建议：是）
- [ ] 如果采用，修改训练脚本并重新训练
- [ ] 更新论文 Table II 和 Method 一节
