# Stable Diffusion 从零实现 - 教学计划

## 概述

本项目将带你从零实现一个简化版的 Stable Diffusion，重点在于**理解原理**而非追求生产级性能。

---

## 一、Stable Diffusion 原理详解

### 1.1 什么是扩散模型？

扩散模型的核心思想非常优雅：

```
原始图像 ──加噪声──> 纯噪声 ──去噪声──> 生成图像
          (前向过程)        (逆向过程)
```

想象一滴墨水滴入清水：
- **前向过程**：墨水慢慢扩散，最终均匀分布（变成噪声）
- **逆向过程**：如果我们能"倒放"这个过程，就能从均匀分布恢复出墨水滴

### 1.2 数学原理

#### 前向过程 (Forward Process)

给定原始图像 $x_0$，我们逐步添加高斯噪声：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

其中：
- $t$ ：时间步（0 到 T，通常 T=1000）
- $\epsilon$ ：标准高斯噪声 $\mathcal{N}(0, I)$
- $\bar{\alpha}_t$ ：噪声调度参数，控制每步加多少噪声

#### 逆向过程 (Reverse Process)

训练一个神经网络 $\epsilon_\theta(x_t, t)$ 来预测噪声，然后逐步去噪：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

#### 训练目标

$$\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

简单说：**让网络学会预测我们加进去的噪声**。

### 1.3 Stable Diffusion 架构

```mermaid
graph TB
    subgraph "Stable Diffusion 架构"
        A[文本提示] --> B[CLIP 文本编码器]
        B --> C[文本嵌入]
        
        D[随机噪声] --> E[U-Net]
        C --> E
        F[时间步嵌入] --> E
        
        E --> G[预测噪声]
        G --> H[去噪采样器]
        H --> I[潜在空间图像]
        I --> J[VAE 解码器]
        J --> K[生成图像]
    end
```

核心组件：
1. **VAE (变分自编码器)**：压缩图像到潜在空间
2. **U-Net**：预测噪声的主力网络
3. **CLIP**：将文本编码为向量
4. **Scheduler**：控制噪声添加和去除的策略

---

## 二、实现计划

### 2.1 项目结构

```
f:\Projects\llm\src\stable_diffusion\
├── __init__.py
├── theory.md                 # 理论文档
├── scheduler.py              # 噪声调度器
├── unet.py                   # U-Net 网络
│   ├── ResBlock             # 残差块
│   ├── AttentionBlock       # 注意力块
│   ├── TimeEmbedding        # 时间嵌入
│   └── UNet                  # 完整 U-Net
├── diffusion.py              # 扩散过程
│   ├── forward_diffusion    # 前向扩散
│   └── reverse_diffusion    # 逆向扩散
├── text_encoder.py           # 文本编码器（简化版）
├── vae.py                    # VAE（可选，简化版可跳过）
├── train.py                  # 训练脚本
├── sample.py                 # 采样/生成脚本
└── visualize.py              # 可视化工具
```

### 2.2 分阶段实现

---

#### **阶段 1：噪声调度器** `scheduler.py`

实现 DDPM 调度器，管理噪声的添加和去除。

```python
class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        # 计算 alpha, alpha_bar 等参数
        pass
    
    def add_noise(self, x_0, t, noise=None):
        # 前向扩散：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        pass
    
    def step(self, model_output, t, x_t):
        # 逆向一步：从 x_t 得到 x_{t-1}
        pass
```

---

#### **阶段 2：U-Net 组件** `unet.py`

##### 2.1 时间嵌入 (Time Embedding)

将时间步 t 编码为向量，使用正弦位置编码：

```python
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        # 类似 Transformer 的位置编码
        pass
    
    def forward(self, t):
        # 输入: t [batch_size]
        # 输出: [batch_size, dim]
        pass
```

##### 2.2 残差块 (ResBlock)

带时间条件的残差块：

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
    def forward(self, x, t_emb):
        # h = conv1(x) + time_mlp(t_emb)
        # return conv2(h) + skip_connection
        pass
```

##### 2.3 注意力块 (Self-Attention)

```python
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        self.norm = nn.GroupNorm(...)
        self.qkv = nn.Conv2d(...)
        self.proj = nn.Conv2d(...)
```

##### 2.4 完整 U-Net

```
输入 x_t ────┬──────────────────────────────────────┐
             │                                       │
          ┌──▼──┐                                 ┌──▼──┐
          │Down1│──────────────────────────────── │ Up1 │
          └──┬──┘                                 └──▲──┘
             │                                       │
          ┌──▼──┐                                 ┌──▼──┐
          │Down2│──────────────────────────────── │ Up2 │
          └──┬──┘                                 └──▲──┘
             │                                       │
          ┌──▼──────────────────────────────────────▲──┐
          │              Middle Block                  │
          └────────────────────────────────────────────┘
```

---

#### **阶段 3：扩散过程** `diffusion.py`

```python
class GaussianDiffusion:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
    
    def training_loss(self, x_0):
        # 1. 随机采样时间步 t
        # 2. 生成噪声 epsilon
        # 3. 前向扩散得到 x_t
        # 4. 用模型预测噪声
        # 5. 计算 MSE loss
        pass
    
    @torch.no_grad()
    def sample(self, shape):
        # 从纯噪声开始，逐步去噪生成图像
        pass
```

---

#### **阶段 4：训练与测试** `train.py`

我们将使用 **MNIST** 作为简单数据集进行验证：

```python
def train():
    # 使用 MNIST 数据集
    # 训练 U-Net 预测噪声
    # 定期保存 checkpoint 和生成样本
    pass
```

---

## 三、验证计划

### 3.1 单元测试

| 测试项 | 验证方法 |
|--------|----------|
| Scheduler | 验证 `add_noise` 然后 `step` 能近似恢复原图 |
| TimeEmbedding | 检查输出形状正确 |
| ResBlock | 输入输出形状一致性 |
| U-Net | 完整前向传播形状检查 |

### 3.2 集成测试

在 MNIST 上训练 50-100 epochs，验证生成样本质量。

### 3.3 运行命令

```bash
# 运行所有测试
pytest f:\Projects\llm\src\stable_diffusion\tests\ -v

# 训练模型
python f:\Projects\llm\src\stable_diffusion\train.py --epochs 50 --dataset mnist

# 生成样本
python f:\Projects\llm\src\stable_diffusion\sample.py --checkpoint model.pt --num_samples 16
```

---

## 四、参考资源

- [DDPM 原论文](https://arxiv.org/abs/2006.11239)
- [Stable Diffusion 论文](https://arxiv.org/abs/2112.10752)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

---

## 实现顺序建议

1. **先看原理** → 确保理解前向/逆向过程
2. **Scheduler** → 最简单，便于理解噪声控制
3. **U-Net 组件** → 从小到大：TimeEmb → ResBlock → Attention → 完整 UNet
4. **扩散过程** → 整合 Scheduler 和 U-Net
5. **训练** → 在 MNIST 上验证
6. **条件生成** → 添加文本条件（可选进阶）

---

> [!NOTE]
> 这是一个**教学项目**，实现会做适当简化。真实的 Stable Diffusion 还包括 VAE、更大的 U-Net、FP16 优化等。

---

请确认这个计划是否符合你的学习目标？我们可以按照这个顺序逐步实现，每一步我都会详细解释原理。
