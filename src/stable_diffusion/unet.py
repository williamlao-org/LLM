"""
U-Net 架构实现 (简化版)

=================================
U-Net 是什么？
=================================

U-Net 是扩散模型的核心组件，负责预测噪声。
它的名字来源于其 U 形结构：

    输入 ──► 下采样 ──► 瓶颈层 ──► 上采样 ──► 输出
         ↘         ↘         ↙         ↙
              Skip Connections (跳跃连接)

=================================
设计原则
=================================

1. **对称结构**: 编码器和解码器层数相同
2. **明确的跳跃连接**: 每个 DownBlock 对应一个 UpBlock
3. **空间尺寸匹配**: 上采样后与对应层的跳跃连接尺寸一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码 - 将时间步编码为向量

    与 Transformer 的位置编码原理相同：
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """时间嵌入: 正弦编码 → MLP"""

    def __init__(self, time_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


class ResBlock(nn.Module):
    """
    残差块 - 带时间条件

    结构:
        x → GroupNorm → GELU → Conv → (+time_emb) → GroupNorm → GELU → Conv → (+residual)
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_groups: int = 8):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.gelu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """自注意力块"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        h, _ = self.mha(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]
        return x + h


class Downsample(nn.Module):
    """下采样: 空间尺寸减半"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """上采样: 空间尺寸翻倍"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    简化版 U-Net

    架构 (以 channel_mults=(1,2,4) 为例):

        输入 [B, 1, 32, 32]
          │
          ▼
        conv_in → [B, 64, 32, 32]
          │
          ├──────────────────────────────────── skip1 ────┐
          ▼                                               │
        ResBlock(64→64) → Downsample                      │
          │                                               │
          ├──────────────────────────────────── skip2 ──┐ │
          ▼                                             │ │
        ResBlock(64→128) → Downsample → [B, 128, 8, 8]  │ │
          │                                             │ │
          ├──────────────────────────────────── skip3 ┐ │ │
          ▼                                           │ │ │
        ResBlock(128→256) [B, 256, 8, 8]              │ │ │
          │                                           │ │ │
          ▼                                           │ │ │
        Middle: ResBlock → Attention → ResBlock       │ │ │
          │                                           │ │ │
          ▼                                           │ │ │
        ResBlock(256+256→128) ←─────── concat ────────┘ │ │
          │                                             │ │
          ▼                                             │ │
        Upsample → ResBlock(128+128→64) ← concat ───────┘ │
          │                                               │
          ▼                                               │
        Upsample → ResBlock(64+64→64) ← concat ───────────┘
          │
          ▼
        conv_out → [B, 1, 32, 32]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_dim: int = 64,
        dim_mults: tuple = (1, 2, 4),
        use_attention: bool = True,
        time_dim: int = 256,
    ):
        super().__init__()

        self.time_dim = time_dim * 4  # 时间嵌入的最终维度
        dims = [base_dim * m for m in dim_mults]  # [64, 128, 256]

        # 时间嵌入
        self.time_emb = TimeEmbedding(time_dim)

        # 输入层
        self.conv_in = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # =========================================
        # 编码器
        # =========================================
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        in_dim = base_dim
        for i, out_dim in enumerate(dims):
            self.down_blocks.append(ResBlock(in_dim, out_dim, self.time_dim))
            # 最后一层不下采样
            if i < len(dims) - 1:
                self.down_samples.append(Downsample(out_dim))
            else:
                self.down_samples.append(nn.Identity())
            in_dim = out_dim

        # =========================================
        # 瓶颈层
        # =========================================
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, self.time_dim)
        self.mid_attn = SelfAttention(mid_dim) if use_attention else nn.Identity()
        self.mid_block2 = ResBlock(mid_dim, mid_dim, self.time_dim)

        # =========================================
        # 解码器
        # =========================================
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        # 逆序处理: [256, 128, 64]
        reversed_dims = list(reversed(dims))
        for i, out_dim in enumerate(reversed_dims):
            # 输入 = 当前维度 + 跳跃连接维度
            if i == 0:
                in_dim = mid_dim + reversed_dims[i]  # 256 + 256
            else:
                in_dim = reversed_dims[i - 1] + reversed_dims[i]  # 128+128, 64+64

            self.up_blocks.append(ResBlock(in_dim, out_dim, self.time_dim))
            # 第一层不上采样（已经在瓶颈层后）
            if i < len(dims) - 1:
                self.up_samples.append(Upsample(out_dim))
            else:
                self.up_samples.append(nn.Identity())

        # 输出层
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 带噪图像 [B, C, H, W]
            t: 时间步 [B]
        Returns:
            预测噪声 [B, C, H, W]
        """
        # 时间嵌入
        t_emb = self.time_emb(t)  # [B, time_dim * 4]

        # 输入
        x = self.conv_in(x)

        # 编码器 (保存跳跃连接)
        skips = []
        for block, downsample in zip(self.down_blocks, self.down_samples):
            x = block(x, t_emb)
            skips.append(x)
            x = downsample(x)

        # 瓶颈
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # 解码器 (使用跳跃连接，逆序)
        for block, upsample in zip(self.up_blocks, self.up_samples):
            skip = skips.pop()  # 从后往前取
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)
            x = upsample(x)

        return self.conv_out(x)


# =========================================
# 测试
# =========================================
if __name__ == "__main__":
    print("=" * 60)
    print("U-Net 测试")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")

    # 创建模型
    unet = UNet(
        in_channels=1, out_channels=1, base_dim=32, dim_mults=(1, 2, 4), time_dim=64
    ).to(device)

    # 统计参数
    params = sum(p.numel() for p in unet.parameters())
    print(f"参数量: {params:,} ({params / 1e6:.2f}M)")

    # 测试 32x32 输入
    x = torch.randn(2, 1, 32, 32, device=device)
    t = torch.randint(0, 1000, (2,), device=device)

    print(f"\n输入: {x.shape}, 时间步: {t.tolist()}")

    out = unet(x, t)
    print(f"输出: {out.shape}")

    assert x.shape == out.shape, "形状不匹配!"
    print("\n✓ 测试通过!")
    print("=" * 60)
