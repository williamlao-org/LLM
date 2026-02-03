"""
采样脚本 - 从预训练模型生成图像

=================================
使用方法
=================================

# 基本采样
python -m stable_diffusion.sample --checkpoint outputs/diffusion/best_model.pt

# 自定义参数
python -m stable_diffusion.sample --checkpoint model.pt --num_samples 16 --method ddim

# 使用 DDIM 快速采样
python -m stable_diffusion.sample --checkpoint model.pt --method ddim --ddim_steps 50
"""

import torch
import argparse
from pathlib import Path

from .scheduler import DDPMScheduler
from .unet import UNet
from .diffusion import GaussianDiffusion


def save_samples(samples: torch.Tensor, path: str, nrow: int = 4):
    """保存生成的样本"""
    from torchvision.utils import make_grid, save_image

    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    grid = make_grid(samples, nrow=nrow, padding=2)
    save_image(grid, path)
    print(f"样本保存到: {path}")


def sample(args):
    """主采样函数"""

    device = args.device
    print(f"设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================
    # 1. 创建模型
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

    # =========================================
    # 2. 加载 checkpoint
    # =========================================
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\n加载 checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        unet.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"加载成功 (epoch {checkpoint.get('epoch', 'unknown')}, loss {checkpoint.get('loss', 'unknown'):.4f})"
        )
    else:
        print("\n⚠ 未提供 checkpoint，使用随机初始化的模型（生成结果将是噪声）")

    # =========================================
    # 3. 生成样本
    # =========================================
    print(f"\n生成 {args.num_samples} 个样本...")
    unet.eval()

    shape = (args.num_samples, 1, args.img_size, args.img_size)

    with torch.no_grad():
        if args.method == "ddpm":
            print(f"使用 DDPM 采样 ({args.num_timesteps} 步)...")
            samples = diffusion.sample(shape, show_progress=True)
        else:
            print(f"使用 DDIM 采样 ({args.ddim_steps} 步)...")
            samples = diffusion.sample_ddim(
                shape,
                num_inference_steps=args.ddim_steps,
                eta=args.ddim_eta,
                show_progress=True,
            )

    # =========================================
    # 4. 保存结果
    # =========================================
    output_path = output_dir / f"samples_{args.method}.png"
    save_samples(samples, output_path, nrow=int(args.num_samples**0.5))

    # 也保存单独的图像
    if args.save_individual:
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
        from torchvision.utils import save_image

        for i, sample in enumerate(samples):
            sample = (sample + 1) / 2
            save_image(sample, individual_dir / f"sample_{i:03d}.png")
        print(f"单独图像保存到: {individual_dir}")

    print("\n采样完成!")


def parse_args():
    parser = argparse.ArgumentParser(description="从扩散模型生成样本")

    # Checkpoint
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="模型 checkpoint 路径"
    )

    # 模型配置（需要与训练时一致）
    parser.add_argument("--base_dim", type=int, default=64)
    parser.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear")

    # 采样配置
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--method", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)

    # 输出
    parser.add_argument("--output_dir", type=str, default="./outputs/samples")
    parser.add_argument("--save_individual", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args)
