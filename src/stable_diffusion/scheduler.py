"""
DDPM 噪声调度器 (Noise Scheduler)

=================================
核心原理
=================================

扩散模型的核心是两个过程：

1. **前向过程 (Forward Process)**: 逐步向图像添加高斯噪声
2. **逆向过程 (Reverse Process)**: 逐步从噪声中恢复图像

调度器的作用是管理这两个过程中噪声的强度。

=================================
数学公式
=================================

前向过程可以写成：
    x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

其中：
    - x_0: 原始图像
    - x_t: 第 t 步加噪后的图像
    - ε: 标准高斯噪声 ~ N(0, I)
    - ᾱ_t (alpha_bar): 累积噪声参数

关键参数：
    - β_t (beta): 每步添加的噪声方差，从 β_start 线性增长到 β_end
    - α_t = 1 - β_t
    - ᾱ_t = α_1 * α_2 * ... * α_t (累积乘积)

为什么用累积乘积？
    - 这允许我们直接从 x_0 计算任意时刻的 x_t
    - 不需要一步步迭代，大大加速训练

=================================
代码实现
=================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) 调度器

    控制扩散过程中噪声的添加和去除。

    Args:
        num_timesteps: 总时间步数，默认 1000
        beta_start: 初始 β 值
        beta_end: 最终 β 值
        beta_schedule: β 的调度方式，'linear' 或 'cosine'

    Example:
        >>> scheduler = DDPMScheduler(num_timesteps=1000)
        >>> x_0 = torch.randn(4, 3, 32, 32)  # 一批图像
        >>> t = torch.tensor([100, 200, 300, 400])  # 不同时间步
        >>> x_t, noise = scheduler.add_noise(x_0, t)  # 添加噪声
        >>> # x_t 是加噪后的图像，noise 是添加的噪声
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps

        # =========================================
        # Step 1: 计算 beta 序列
        # =========================================
        # beta 控制每步添加多少噪声
        # 从小到大线性增长：开始时噪声少，后期噪声多
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            # Cosine 调度在实践中效果更好
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # =========================================
        # Step 2: 计算 alpha 和 alpha_bar
        # =========================================
        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = α_1 * α_2 * ... * α_t
        # 使用累积乘积 (cumprod)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 为了计算方便，存储 alpha_bar 的前一个值
        # alpha_bar_{t-1}，用于逆向过程
        self.alphas_cumprod_prev = torch.cat(
            [
                torch.tensor([1.0]),  # t=0 时，前一个值设为 1
                self.alphas_cumprod[:-1],
            ]
        )

        # =========================================
        # Step 3: 预计算常用值（加速计算）
        # =========================================
        # 前向过程需要的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 逆向过程需要的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # 后验方差 (posterior variance)
        # 用于逆向采样时添加随机性
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(
        self, num_timesteps: int, s: float = 0.008
    ) -> torch.Tensor:
        """
        Cosine 调度，来自 "Improved DDPM" 论文

        相比线性调度，cosine 调度在早期噪声增加更慢，
        保留更多图像信息，通常效果更好。
        """
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散：向原始图像添加噪声

        公式: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε

        Args:
            x_0: 原始图像 [batch_size, channels, height, width]
            t: 时间步 [batch_size]，每个样本可以有不同的时间步
            noise: 可选，指定噪声。如果不提供则随机生成

        Returns:
            x_t: 加噪后的图像
            noise: 添加的噪声（用于计算损失）

        Example:
            >>> scheduler = DDPMScheduler()
            >>> x_0 = torch.randn(4, 3, 32, 32)
            >>> t = torch.randint(0, 1000, (4,))
            >>> x_t, noise = scheduler.add_noise(x_0, t)
        """
        # 如果没有提供噪声，随机生成
        if noise is None:
            noise = torch.randn_like(x_0)

        # 获取对应时间步的系数
        # 注意：需要将系数 reshape 为可广播的形状
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # 应用前向扩散公式
        # x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        逆向扩散：根据模型预测的噪声，从 x_t 计算 x_{t-1}

        公式（简化版）:
        x_{t-1} = 1/√(α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ) + σ_t * z

        其中 ε_θ 是模型预测的噪声，z 是随机噪声

        Args:
            model_output: 模型预测的噪声 ε_θ
            t: 当前时间步（整数）
            x_t: 当前带噪图像
            generator: 可选的随机数生成器，用于可复现性

        Returns:
            x_{t-1}: 去噪一步后的图像
        """
        # 获取当前时间步的参数
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]

        # 计算 x_0 的预测值（用于理解，实际用预测噪声）
        # x_0_pred = (x_t - √(1-ᾱ_t) * ε_θ) / √(ᾱ_t)

        # 计算均值
        # μ = 1/√(α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # 预测的均值
        pred_mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * model_output
        )

        # 如果是最后一步（t=0），不添加噪声
        if t == 0:
            return pred_mean

        # 添加噪声（重参数化技巧）
        variance = self.posterior_variance[t]
        noise = torch.randn(x_t.shape, generator=generator, device=x_t.device)

        # x_{t-1} = μ + σ * z
        x_prev = pred_mean + torch.sqrt(variance) * noise

        return x_prev

    def _extract(
        self, tensor: torch.Tensor, t: torch.Tensor, shape: Tuple
    ) -> torch.Tensor:
        """
        从 tensor 中提取 t 对应的值，并 reshape 为可广播的形状

        这是一个辅助函数，用于处理批量操作。

        Args:
            tensor: 包含所有时间步参数的 1D tensor [num_timesteps]
            t: 时间步索引 [batch_size]
            shape: 目标 shape，通常是 [batch_size, channels, height, width]

        Returns:
            提取的值，shape 为 [batch_size, 1, 1, 1]，可与图像广播
        """
        batch_size = t.shape[0]
        # 确保 tensor 在正确的设备上
        tensor = tensor.to(t.device)
        # 使用 gather 提取值
        out = tensor.gather(-1, t)
        # reshape 为 [batch_size, 1, 1, ...] 以便广播
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


# =========================================
# 测试代码
# =========================================
if __name__ == "__main__":
    print("=" * 60)
    print("DDPM Scheduler 测试")
    print("=" * 60)

    # 创建调度器
    scheduler = DDPMScheduler(num_timesteps=1000)

    # 查看参数形状
    print(f"\n参数形状:")
    print(f"  betas: {scheduler.betas.shape}")
    print(f"  alphas_cumprod: {scheduler.alphas_cumprod.shape}")

    # 可视化 alpha_bar 的变化
    print(f"\nAlpha_bar 在不同时间步的值:")
    print(f"  t=0:   {scheduler.alphas_cumprod[0]:.4f} (几乎没有噪声)")
    print(f"  t=250: {scheduler.alphas_cumprod[250]:.4f}")
    print(f"  t=500: {scheduler.alphas_cumprod[500]:.4f}")
    print(f"  t=750: {scheduler.alphas_cumprod[750]:.4f}")
    print(f"  t=999: {scheduler.alphas_cumprod[999]:.4f} (几乎全是噪声)")

    # 测试前向扩散
    print(f"\n测试前向扩散:")
    x_0 = torch.randn(4, 3, 32, 32)  # 4张 32x32 的 RGB 图像
    t = torch.tensor([0, 250, 500, 999])  # 4个不同时间步

    x_t, noise = scheduler.add_noise(x_0, t)
    print(f"  输入 x_0 shape: {x_0.shape}")
    print(f"  时间步 t: {t.tolist()}")
    print(f"  输出 x_t shape: {x_t.shape}")

    # 验证：t=0 时，x_t 应该接近 x_0
    print(f"\n验证 t=0 时 x_t ≈ x_0:")
    print(f"  x_0[0] 的部分值: {x_0[0, 0, 0, :5]}")
    print(f"  x_t[0] 的部分值: {x_t[0, 0, 0, :5]}")
    print(f"  差异: {(x_0[0] - x_t[0]).abs().mean():.6f} (应该很小)")

    # 验证：t=999 时，x_t 应该接近纯噪声
    print(f"\n验证 t=999 时 x_t ≈ 噪声:")
    print(f"  x_t[3] 的均值: {x_t[3].mean():.4f} (应该接近 0)")
    print(f"  x_t[3] 的标准差: {x_t[3].std():.4f} (应该接近 1)")

    print("\n" + "=" * 60)
    print("测试通过！")
    print("=" * 60)
