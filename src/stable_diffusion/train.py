"""
训练脚本 - 在 MNIST 上训练扩散模型

=================================
使用方法
=================================

# 基本训练
python -m stable_diffusion.train

# 自定义参数
python -m stable_diffusion.train --epochs 100 --batch_size 128 --lr 1e-4

# 使用 GPU
python -m stable_diffusion.train --device cuda

=================================
训练流程
=================================

1. 加载 MNIST 数据集
2. 创建模型和优化器
3. 训练循环:
   - 随机采样时间步
   - 加噪声到图像
   - 预测噪声
   - 计算 MSE 损失
   - 反向传播
4. 定期采样和保存 checkpoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import argparse
from tqdm import tqdm

from .scheduler import DDPMScheduler
from .unet import UNet
from .diffusion import GaussianDiffusion


def get_mnist_dataloader(batch_size: int = 64, img_size: int = 32) -> DataLoader:
    """
    获取 MNIST 数据加载器

    MNIST 原始尺寸是 28x28，我们 resize 到 32x32 以便下采样
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # 归一化到 [-1, 1]
        ]
    )

    try:
        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"MNIST 下载失败: {e}")
        print("使用合成数据进行测试...")
        return get_synthetic_dataloader(batch_size, img_size)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )


def get_synthetic_dataloader(
    batch_size: int = 64, img_size: int = 32, num_samples: int = 1000
) -> DataLoader:
    """
    创建合成数据用于测试

    生成简单的几何图形（圆、方块等）作为训练数据
    """
    from torch.utils.data import TensorDataset

    images = []
    for i in range(num_samples):
        img = torch.zeros(1, img_size, img_size)

        # 随机生成圆形或方形
        shape_type = i % 3
        cx, cy = torch.randint(8, img_size - 8, (2,)).tolist()
        size = torch.randint(4, 10, (1,)).item()

        if shape_type == 0:  # 圆形
            y, x = torch.meshgrid(
                torch.arange(img_size), torch.arange(img_size), indexing="ij"
            )
            mask = (x - cx) ** 2 + (y - cy) ** 2 < size**2
            img[0, mask] = 1.0
        elif shape_type == 1:  # 方形
            x1, y1 = max(0, cx - size), max(0, cy - size)
            x2, y2 = min(img_size, cx + size), min(img_size, cy + size)
            img[0, y1:y2, x1:x2] = 1.0
        else:  # 线条
            for j in range(-size, size + 1):
                if 0 <= cy + j < img_size and 0 <= cx + j < img_size:
                    img[0, cy + j, cx + j] = 1.0

        # 归一化到 [-1, 1]
        img = img * 2 - 1
        images.append(img)

    images = torch.stack(images)
    dataset = TensorDataset(images, torch.zeros(num_samples))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_samples(samples: torch.Tensor, path: str, nrow: int = 8):
    """保存生成的样本图像"""
    from torchvision.utils import make_grid, save_image

    # 反归一化: [-1, 1] -> [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    grid = make_grid(samples, nrow=nrow, padding=2)
    save_image(grid, path)
    print(f"样本保存到: {path}")


def train(args):
    """主训练函数"""

    # 设置设备
    device = args.device
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # 1. 数据加载
    # =========================================
    print("\n加载 MNIST 数据集...")
    dataloader = get_mnist_dataloader(args.batch_size, args.img_size)
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"批次数: {len(dataloader)}")

    # =========================================
    # 2. 创建模型
    # =========================================
    print("\n创建模型...")

    unet = UNet(
        in_channels=1,
        out_channels=1,
        base_dim=args.base_dim,
        dim_mults=tuple(args.dim_mults),
        time_dim=args.time_dim,
    ).to(device)

    scheduler = DDPMScheduler(
        num_timesteps=args.num_timesteps, beta_schedule=args.beta_schedule
    )

    diffusion = GaussianDiffusion(unet, scheduler, device)

    # 统计参数量
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"模型参数量: {num_params:,} ({num_params / 1e6:.2f}M)")

    # =========================================
    # 3. 优化器和学习率调度
    # =========================================
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)

    # 余弦退火学习率
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # =========================================
    # 4. 训练循环
    # =========================================
    print("\n开始训练...")
    print("=" * 60)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        unet.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            # 计算损失
            loss = diffusion.training_loss(images)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        # 更新学习率
        scheduler_lr.step()

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 完成, 平均损失: {avg_loss:.4f}")

        # =========================================
        # 5. 定期采样和保存
        # =========================================
        if (epoch + 1) % args.sample_every == 0:
            print("生成样本中...")
            unet.eval()

            with torch.no_grad():
                samples = diffusion.sample(
                    shape=(16, 1, args.img_size, args.img_size), show_progress=False
                )

            save_samples(samples, save_dir / f"samples_epoch_{epoch + 1:04d}.png")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_dir / "best_model.pt",
            )
            print(f"最佳模型已保存 (loss: {best_loss:.4f})")

        # 定期保存 checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
            )

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练扩散模型")

    # 数据相关
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=32)

    # 模型相关
    parser.add_argument("--base_dim", type=int, default=64, help="U-Net 基础通道数")
    parser.add_argument(
        "--dim_mults", type=int, nargs="+", default=[1, 2, 4], help="通道数倍数"
    )
    parser.add_argument("--time_dim", type=int, default=128, help="时间嵌入维度")

    # 扩散相关
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "cosine"]
    )

    # 训练相关
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 保存相关
    parser.add_argument("--save_dir", type=str, default="./outputs/diffusion")
    parser.add_argument(
        "--sample_every", type=int, default=5, help="每隔多少个 epoch 生成样本"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="每隔多少个 epoch 保存 checkpoint"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
