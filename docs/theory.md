# 理论推导：SAC + Diffusion Policy

## 1. 问题定义

我们考虑一个标准的强化学习问题，定义为马尔可夫决策过程（MDP）：
- 状态空间：$\mathcal{S}$ (多模态观测：RGB图像、深度图、激光雷达点云、机器人状态)
- 动作空间：$\mathcal{A} \subset \mathbb{R}^d$ (连续动作：$[v_x, v_y, \omega_z]$)
- 转移概率：$P(s'|s, a)$
- 奖励函数：$r(s, a)$
- 折扣因子：$\gamma \in [0, 1)$

## 2. 最大熵强化学习 (Maximum Entropy RL)

### 2.1 最大熵目标

标准强化学习目标：
$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right]
$$

最大熵扩展（SAC的核心思想）：
$$
J_{\text{MaxEnt}}(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t \left( r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right) \right]
$$

其中：
- $H(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi(\cdot|s)}[\log \pi(a|s)]$ 是策略的熵
- $\alpha > 0$ 是温度参数，控制探索程度

**直觉解释**：
- 鼓励策略不仅要最大化奖励，还要保持高熵（多样性）
- 有助于探索，避免过早收敛到次优策略
- 对扩散模型特别适合，因为扩散模型本身就建模概率分布

### 2.2 软Q函数

最大熵框架下的软Q函数：
$$
Q^{\pi}(s, a) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{k=0}^\infty \gamma^k \left( r(s_{t+k}, a_{t+k}) + \alpha H(\pi(\cdot|s_{t+k})) \right) \Bigg| s_t=s, a_t=a \right]
$$

软Bellman方程：
$$
Q^{\pi}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P} \left[ V^{\pi}(s') \right]
$$

其中软状态值函数：
$$
V^{\pi}(s) = \mathbb{E}_{a \sim \pi} \left[ Q^{\pi}(s, a) - \alpha \log \pi(a|s) \right]
$$

## 3. 扩散模型作为策略

### 3.1 去噪扩散概率模型 (DDPM)

扩散模型定义一个马尔可夫链，从数据分布 $q(a_0)$ 逐步添加噪声到纯高斯噪声 $p(a_T) = \mathcal{N}(0, I)$。

**前向过程**（固定）：
$$
q(a_t | a_{t-1}) = \mathcal{N}(a_t; \sqrt{1-\beta_t} a_{t-1}, \beta_t I)
$$

其中 $\beta_t$ 是噪声调度表。

利用重参数化技巧：
$$
a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$。

**反向过程**（学习）：
$$
p_\theta(a_{t-1} | a_t, c) = \mathcal{N}(a_{t-1}; \mu_\theta(a_t, t, c), \Sigma_\theta(a_t, t, c))
$$

其中 $c$ 是条件信息（观测编码）。

**训练目标**：
$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{a_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(a_t, t, c) \|^2 \right]
$$

即预测添加的噪声。

### 3.2 扩散策略

我们将扩散模型 $p_\theta(a|s)$ 定义为策略：

1. **输入**：状态 $s$（多模态观测）
2. **编码**：$c = f_\phi(s)$（观测编码器）
3. **采样**：
   - 初始化：$a_T \sim \mathcal{N}(0, I)$
   - 迭代去噪：$a_{t-1} = \mu_\theta(a_t, t, c) + \sigma_t z$，$z \sim \mathcal{N}(0, I)$
   - 输出：$a_0 \sim p_\theta(\cdot | s)$

**关键优势**：
- 自然建模多模态动作分布（不同驾驶策略，如避让、超车等）
- 生成高质量、平滑的动作序列
- 条件生成适合端到端学习

## 4. SAC + Diffusion Policy 的结合

### 4.1 Actor-Critic 架构

**Actor（扩散策略）**：
$$
\pi_\theta(a|s) = p_\theta(a|s) \quad \text{(扩散模型)}
$$

**Critic（双Q网络）**：
$$
Q_{\psi_i}(s, a), \quad i=1, 2
$$

**目标网络**（软更新）：
$$
Q_{\bar{\psi}_i}(s, a), \quad \bar{\psi}_i \leftarrow \tau \psi_i + (1-\tau) \bar{\psi}_i
$$

### 4.2 Critic 更新

Critic 的目标是最小化软Bellman误差。

**目标Q值**（使用目标网络）：
$$
y(s, a, r, s') = r + \gamma \left( \min_{i=1,2} Q_{\bar{\psi}_i}(s', a') - \alpha \log \pi_\theta(a'|s') \right)
$$

其中 $a' \sim \pi_\theta(\cdot | s')$（从扩散策略采样）。

**Critic 损失**：
$$
\mathcal{L}_Q(\psi_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( Q_{\psi_i}(s, a) - y(s, a, r, s') \right)^2 \right]
$$

更新：
$$
\psi_i \leftarrow \psi_i - \eta_Q \nabla_{\psi_i} \mathcal{L}_Q(\psi_i)
$$

### 4.3 Actor 更新（关键创新）

Actor 的目标是最大化期望Q值减去策略熵的权重：

$$
J_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} \left[ \min_{i=1,2} Q_{\psi_i}(s, a) - \alpha \log \pi_\theta(a|s) \right] \right]
$$

**挑战**：扩散模型的 $\log \pi_\theta(a|s)$ 难以精确计算。

**解决方案1（蒙特卡洛估计）**：
$$
\log \pi_\theta(a|s) \approx -\mathcal{L}_{\text{DDPM}}(a, s) = -\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} a + \sqrt{1-\bar{\alpha}_t} \epsilon, t, c)\|^2
$$

直觉：噪声预测误差越小，动作概率越高。

**解决方案2（分析近似）**：
对于高斯扩散，可以推导出对数似然的下界：
$$
\log \pi_\theta(a|s) \geq -\frac{1}{2\sigma^2} \|a - \mu_\theta(s)\|^2 + \text{const}
$$

**策略梯度**（重参数化）：
$$
\nabla_\theta J_\pi(\theta) = \mathbb{E}_{s, \epsilon} \left[ \nabla_\theta \left( \min_{i=1,2} Q_{\psi_i}(s, a_\theta(s, \epsilon)) - \alpha \log \pi_\theta(a_\theta(s, \epsilon)|s) \right) \right]
$$

其中 $a_\theta(s, \epsilon)$ 是扩散采样过程（可微）。

### 4.4 温度参数自动调节

自动调节 $\alpha$ 以匹配目标熵 $\bar{H}$：

$$
\mathcal{L}_\alpha = \mathbb{E}_{a \sim \pi_\theta} \left[ -\alpha \left( \log \pi_\theta(a|s) + \bar{H} \right) \right]
$$

通常设置 $\bar{H} = -\text{dim}(\mathcal{A})$（启发式）。

更新：
$$
\alpha \leftarrow \alpha - \eta_\alpha \nabla_\alpha \mathcal{L}_\alpha
$$

实践中，优化 $\log \alpha$ 以保证 $\alpha > 0$。

## 5. 完整算法流程

### 算法：SAC with Diffusion Policy

**输入**：环境 $\mathcal{E}$，预训练扩散策略 $\pi_\theta$，Q网络 $Q_{\psi_1}, Q_{\psi_2}$

**输出**：训练好的策略 $\pi_\theta^*$

1. 初始化：
   - 经验回放池 $\mathcal{D} \leftarrow \emptyset$
   - 目标网络：$\bar{\psi}_i \leftarrow \psi_i$
   - 温度参数：$\alpha \leftarrow \alpha_0$ 或 $\log \alpha \leftarrow 0$

2. **For** $t = 1$ to $T_{\max}$ **do**:

   a. **数据收集**：
      - 观测 $s_t$ from环境
      - 采样动作：$a_t \sim \pi_\theta(\cdot | s_t)$（扩散采样）
      - 执行 $a_t$，得到 $(r_t, s_{t+1})$
      - 存储 $(s_t, a_t, r_t, s_{t+1})$ 到 $\mathcal{D}$

   b. **Critic 更新**（if $t > T_{\text{warmup}}$）：
      - 采样 mini-batch：$(s, a, r, s') \sim \mathcal{D}$
      - 计算目标：
        $$
        y = r + \gamma \left( \min_i Q_{\bar{\psi}_i}(s', a') - \alpha \log \pi_\theta(a'|s') \right), \quad a' \sim \pi_\theta(\cdot|s')
        $$
      - 更新：$\psi_i \leftarrow \psi_i - \eta_Q \nabla_{\psi_i} \mathcal{L}_Q(\psi_i)$

   c. **Actor 更新**（if $t \mod d_{\pi} = 0$）：
      - 采样 mini-batch：$s \sim \mathcal{D}$
      - 采样动作：$a \sim \pi_\theta(\cdot|s)$
      - 计算策略梯度：
        $$
        \nabla_\theta J \approx \nabla_\theta \left( \min_i Q_{\psi_i}(s, a) - \alpha \log \pi_\theta(a|s) \right)
        $$
      - 更新：$\theta \leftarrow \theta + \eta_\pi \nabla_\theta J$

   d. **温度更新**（if auto-tune）：
      - 更新：$\alpha \leftarrow \alpha - \eta_\alpha \nabla_\alpha \mathcal{L}_\alpha$

   e. **目标网络软更新**：
      - $\bar{\psi}_i \leftarrow \tau \psi_i + (1-\tau) \bar{\psi}_i$

3. **Return** $\pi_\theta$

## 6. 理论性质

### 6.1 收敛性

在表格设定下，SAC保证收敛到最优策略（Haarnoja et al., 2018）。对于扩散策略，我们假设：

1. **假设1（表达能力）**：扩散模型 $p_\theta$ 可以逼近任意条件分布
2. **假设2（梯度估计）**：重参数化技巧提供无偏梯度估计

在这些假设下，SAC-Diffusion 继承SAC的收敛保证。

### 6.2 样本复杂度

扩散模型的多模态建模能力可能**减少**样本复杂度：
- 更高效的探索（熵正则化 + 多模态分布）
- 更好的泛化（平滑动作序列）

但迭代采样增加**计算复杂度**：
- 每次动作采样需要 $T_{\text{diff}}$ 次网络前向传播
- 缓解方法：一致性模型、蒸馏

## 7. 与相关工作的对比

| 方法 | 策略类型 | 探索机制 | 多模态 |
|------|---------|---------|--------|
| **SAC (Gaussian)** | 高斯策略 | 熵正则化 | ❌ |
| **Diffusion Policy** | 扩散模型 | 随机采样 | ✅ |
| **Decision Diffuser** | 扩散模型 | 离线RL | ✅ |
| **SAC-Diffusion (本文)** | 扩散模型 | 熵正则化 + 多模态 | ✅ |

**核心区别**：
- 相比标准SAC：支持多模态动作分布
- 相比Diffusion Policy：在线RL + 最大熵优化
- 相比Decision Diffuser：Actor-Critic架构，更高效

## 8. 驾驶任务中的应用

### 8.1 多模态行为

自动驾驶中的多模态场景示例：
1. **交叉路口**：左转、直行、右转（多个合理选择）
2. **障碍物避让**：左侧超车 vs 右侧超车
3. **车道保持**：激进（速度快）vs 保守（速度慢）

扩散策略自然建模这些多模态分布。

### 8.2 奖励函数设计

$$
r(s, a) = r_{\text{progress}} + r_{\text{tracking}} + r_{\text{safety}} + r_{\text{smooth}}
$$

- $r_{\text{progress}} = \lambda_p \cdot \Delta d_{\text{goal}}$（接近目标）
- $r_{\text{tracking}} = -\lambda_t \cdot d_{\text{path}}^2$（路径跟踪）
- $r_{\text{safety}} = -\lambda_s \cdot \mathbb{1}_{\text{collision}}$（碰撞惩罚）
- $r_{\text{smooth}} = -\lambda_m \cdot \|a_t - a_{t-1}\|^2$（动作平滑）

熵项 $\alpha H(\pi)$ 额外鼓励探索。

## 9. 开放问题与未来方向

1. **理论分析**：扩散策略的收敛速度分析
2. **快速采样**：一致性模型、蒸馏技术
3. **长期规划**：分层RL + 扩散模型
4. **Sim2Real**：如何保持多模态性质在真实世界

---

**参考文献**：
1. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications", 2019
2. Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
3. Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", RSS 2023
4. Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022
