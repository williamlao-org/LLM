# Stable Diffusion 从零实现
"""
这个包实现了一个简化版的 Stable Diffusion，
重点在于理解扩散模型的核心原理。

核心组件：
- DDPMScheduler: 噪声调度器，控制前向/逆向扩散
- UNet: 噪声预测网络
- GaussianDiffusion: 整合训练和采样的扩散过程

使用示例：
    from stable_diffusion import DDPMScheduler, UNet, GaussianDiffusion

    # 创建组件
    unet = UNet(in_channels=1, out_channels=1, base_dim=64)
    scheduler = DDPMScheduler(num_timesteps=1000)
    diffusion = GaussianDiffusion(unet, scheduler, device='cuda')

    # 训练
    loss = diffusion.training_loss(images)

    # 采样
    samples = diffusion.sample(shape=(16, 1, 32, 32))
"""

from .scheduler import DDPMScheduler
from .unet import UNet
from .diffusion import GaussianDiffusion

__all__ = ["DDPMScheduler", "UNet", "GaussianDiffusion"]
