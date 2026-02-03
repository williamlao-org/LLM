"""
扩散过程实现

=================================
核心概念
=================================

扩散模型训练的核心非常简单：

1. **前向过程**: 给图像加噪声
2. **训练目标**: 让网络学会预测加的噪声
3. **采样过程**: 从纯噪声开始，逐步去噪

=================================
训练流程
=================================

for epoch in epochs:
    for x_0 in dataloader:  # x_0 是原始图像
        # 1. 随机选择时间步
        t = random.randint(0, T)

        # 2. 生成随机噪声
        noise = torch.randn_like(x_0)

        # 3. 前向扩散 - 加噪声
        x_t = scheduler.add_noise(x_0, t, noise)

        # 4. 预测噪声
        predicted_noise = model(x_t, t)

        # 5. 计算损失 (真实噪声 vs 预测噪声)
        loss = MSE(noise, predicted_noise)

        # 6. 反向传播
        loss.backward()
        optimizer.step()

=================================
采样流程 (从噪声生成图像)
=================================

x_T = torch.randn(...)  # 从纯噪声开始

for t in reversed(range(T)):  # T-1, T-2, ..., 0
    # 1. 预测噪声
    predicted_noise = model(x_t, t)

    # 2. 去噪一步
    x_{t-1} = scheduler.step(predicted_noise, t, x_t)

return x_0  # 生成的图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

from .scheduler import DDPMScheduler
from .unet import UNet


class GaussianDiffusion:
    """
    高斯扩散过程

    将 Scheduler（控制噪声）和 UNet（预测噪声）结合起来，
    实现完整的训练和采样流程。

    Args:
        model: 噪声预测网络 (UNet)
        scheduler: 噪声调度器 (DDPMScheduler)
        device: 计算设备

    Example:
        >>> unet = UNet(...)
        >>> scheduler = DDPMScheduler(num_timesteps=1000)
        >>> diffusion = GaussianDiffusion(unet, scheduler)
        >>>
        >>> # 训练
        >>> loss = diffusion.training_loss(x_0)
        >>> loss.backward()
        >>>
        >>> # 采样
        >>> samples = diffusion.sample(shape=(4, 1, 32, 32))
    """

    def __init__(self, model: nn.Module, scheduler: DDPMScheduler, device: str = "cpu"):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.num_timesteps = scheduler.num_timesteps

    def training_loss(
        self, x_0: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算训练损失

        流程：
        1. 随机采样时间步 t
        2. 生成噪声 ε
        3. 加噪得到 x_t
        4. 用模型预测噪声 ε_θ
        5. 计算 MSE(ε, ε_θ)

        Args:
            x_0: 原始图像 [B, C, H, W]
            noise: 可选指定噪声，用于调试

        Returns:
            loss: 标量损失值

        数学原理：
            Loss = E_{x_0, ε, t} [ ||ε - ε_θ(x_t, t)||² ]

        这个简单的损失函数背后有严格的数学推导，
        它等价于最大化数据的对数似然（变分下界）。
        """
        batch_size = x_0.shape[0]

        # Step 1: 随机采样时间步
        # 对于批次中的每个样本，随机选择一个时间步
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long
        )

        # Step 2: 生成噪声
        if noise is None:
            noise = torch.randn_like(x_0)

        # Step 3: 前向扩散 - 向原始图像添加噪声
        # x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        x_t, _ = self.scheduler.add_noise(x_0, t, noise)

        # Step 4: 模型预测噪声
        # ε_θ(x_t, t) - 模型尝试预测我们添加的噪声
        predicted_noise = self.model(x_t, t)

        # Step 5: 计算损失
        # L = MSE(ε, ε_θ)
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(
        self, shape: tuple, return_all_steps: bool = False, show_progress: bool = True
    ) -> torch.Tensor:
        """
        从噪声生成图像（采样/推理）

        从纯高斯噪声开始，逐步去噪，生成清晰图像。

        Args:
            shape: 生成图像的形状 [B, C, H, W]
            return_all_steps: 是否返回所有中间步骤
            show_progress: 是否显示进度条

        Returns:
            如果 return_all_steps=False: 最终生成的图像 [B, C, H, W]
            如果 return_all_steps=True: 所有步骤的图像列表

        采样过程可视化：

        t=T (纯噪声)    t=T/2 (模糊)    t=0 (清晰)
        ░░░░░░░░░       ▓▓░░░░▓▓        ████████
        ░░░░░░░░░  →    ▓▓▓░░▓▓▓   →    ████████
        ░░░░░░░░░       ▓▓░░░░▓▓        ████████
        """
        self.model.eval()

        # 从纯噪声开始
        x_t = torch.randn(shape, device=self.device)

        all_steps = [x_t] if return_all_steps else None

        # 时间步从 T-1 倒数到 0
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            # 创建时间步张量 [batch_size]
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.model(x_t, t_tensor)

            # 去噪一步
            x_t = self.scheduler.step(predicted_noise, t, x_t)

            if return_all_steps:
                all_steps.append(x_t)

        if return_all_steps:
            return all_steps
        return x_t

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: tuple,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        DDIM 采样 - 更快速的采样方法

        DDIM (Denoising Diffusion Implicit Models) 允许跳过时间步，
        大大加速采样过程。

        Args:
            shape: 生成图像形状
            num_inference_steps: 实际推理步数（远小于训练时的 1000 步）
            eta: 随机性控制 (0=确定性, 1=DDPM)
            show_progress: 显示进度条

        Returns:
            生成的图像

        Example:
            # 原本需要 1000 步，现在只需要 50 步
            samples = diffusion.sample_ddim(shape, num_inference_steps=50)
        """
        self.model.eval()

        # 创建跳跃的时间步序列
        # 例如: 1000 步 → 50 步，每 20 步采样一次
        step_ratio = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_ratio))[::-1]

        # 从噪声开始
        x_t = torch.randn(shape, device=self.device)

        # 使用 enumerate 并可选地用 tqdm 包装
        iterator = enumerate(timesteps)
        if show_progress:
            iterator = tqdm(iterator, total=len(timesteps), desc="DDIM Sampling")

        for i, t in iterator:
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.model(x_t, t_tensor)

            # 获取 alpha 值
            alpha_cumprod_t = self.scheduler.alphas_cumprod[t]

            # 计算 t_prev
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0)

            # 预测 x_0
            pred_x0 = (
                x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
            ) / torch.sqrt(alpha_cumprod_t)

            # DDIM 公式
            # σ_t 控制随机性
            sigma_t = eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev)
                / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # 方向指向 x_t
            pred_dir = (
                torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2) * predicted_noise
            )

            # 更新 x_t
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir + sigma_t * noise
            )

        return x_t


# =========================================
# 测试
# =========================================
if __name__ == "__main__":
    print("=" * 60)
    print("扩散过程测试")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")

    # 创建模型和调度器
    unet = UNet(
        in_channels=1, out_channels=1, base_dim=32, dim_mults=(1, 2, 4), time_dim=64
    ).to(device)

    scheduler = DDPMScheduler(num_timesteps=100)  # 使用较少步数测试

    diffusion = GaussianDiffusion(unet, scheduler, device)

    # 测试训练损失
    print("\n--- 测试训练损失 ---")
    x_0 = torch.randn(4, 1, 32, 32, device=device)
    loss = diffusion.training_loss(x_0)
    print(f"随机图像的损失: {loss.item():.4f}")

    # 测试采样（使用少量步数）
    print("\n--- 测试采样 ---")
    print("采样中... (100 步)")
    samples = diffusion.sample(shape=(2, 1, 32, 32), show_progress=True)
    print(f"生成样本形状: {samples.shape}")
    print(f"样本均值: {samples.mean():.4f}, 标准差: {samples.std():.4f}")

    # 测试 DDIM 采样
    print("\n--- 测试 DDIM 采样 ---")
    print("DDIM 采样中... (20 步)")
    samples_ddim = diffusion.sample_ddim(
        shape=(2, 1, 32, 32), num_inference_steps=20, show_progress=True
    )
    print(f"DDIM 样本形状: {samples_ddim.shape}")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过!")
    print("=" * 60)
