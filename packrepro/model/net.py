"""潜在特征引导的非对称卷积自编码器（论文 §2.2 + Table 1）。

编码器 —— **严格按论文 Table 1 原样**
--------------------------------------
输入 ``(B, 1, 256, 20)`` 逐层推算（无 padding）：

    输入                  256 × 20
    Conv  16 × [5×2]  s[1,2]    →  16 × 252 × 10
    MaxPool    [3×1]  s[3,1]    →  16 ×  84 × 10
    Conv  32 × [1×10] s[1,1]    →  32 ×  84 ×  1
    MaxPool    [3×1]  s[3,1]    →  32 ×  28 ×  1
    Conv 256 × [28×1] s[1,1]    → 256 ×   1 ×  1   ← 瓶颈 256 维

与论文「compressed into a latent representation of 256 features」一致。

解码器 —— **与 Table 1 有差异，必须声明**
-------------------------------------------
论文 Table 1 第 8–10 层（Transposed conv `[1×8] s[3,1]` → `[3×1] s[3,1]` → `[5×1] s[1,1]`）
会把时间维继续放大：1→28→84→250→750→746，**得到 746×8，而不是 Table 2 声明的 256×8×1**。
即论文自己的 Table 1 与 Table 2 相互矛盾（推算过程见 ``docs/DEVIATIONS.md`` 第 5 条）。

本实现的处理：
* 前两级转置卷积沿用论文的 ``[28×1] s[2,1]`` 与 ``[3×1] s[3,1]``（1→28→84，维数正确）；
* 之后用 ``Upsample`` 直接到 ``(256, 1)``，再用两层普通卷积出 8 个单体通道。

这样输出维度严格等于 Table 2 声明的 ``256 × 8``，且"解码器只重建电压"这一非对称性不变。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """前向输出。"""

    recon: torch.Tensor       # (B, T, 8) 只重建单体电压
    mu: torch.Tensor | None   # (B, L) 变分均值，未启用变分头时为 None
    logvar: torch.Tensor | None
    z: torch.Tensor           # (B, L) 潜变量


class AsymmetricAE(nn.Module):
    """编码器吃全部 20 通道，解码器只出 8 路单体电压。"""

    def __init__(
        self,
        *,
        time_len: int = 256,
        in_channels: int = 20,
        out_channels: int = 8,
        latent_dim: int = 256,
        variational: bool = True,
        decoder_channels: tuple[int, ...] = (32, 32, 16, 16),
    ) -> None:
        super().__init__()
        self.time_len = time_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.variational = variational

        # ---- 编码器：Table 1 原样 ----
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (5, 2), stride=(1, 2)),      # → 16 × 252 × 10
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(3, 1)),          # → 16 ×  84 × 10
            nn.Conv2d(16, 32, (1, 10), stride=(1, 1)),    # → 32 ×  84 ×  1
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(3, 1)),          # → 32 ×  28 ×  1
            nn.Conv2d(32, latent_dim, (28, 1), stride=(1, 1)),  # → L × 1 × 1
            nn.ReLU(inplace=True),
        )

        # ---- 变分头：论文 Table 1 没有它，但 L_reg 是 KL(q(z|x)‖p(z))，必须有 ----
        if variational:
            self.to_mu = nn.Linear(latent_dim, latent_dim)
            self.to_logvar = nn.Linear(latent_dim, latent_dim)

        # ---- 解码器：前两级沿用论文，之后上采样到精确尺寸 ----
        self.dec_up = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, decoder_channels[0], (28, 1), stride=(2, 1)),  # 1 → 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], (3, 1), stride=(3, 1)),  # 28 → 84
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(size=(time_len, 1), mode="bilinear", align_corners=False)
        self.dec_head = nn.Sequential(
            nn.Conv2d(decoder_channels[1], decoder_channels[2], (3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[2], out_channels, (1, 1)),
        )

    # ------------------------------------------------------------------ 前向
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """``(B, T, C)`` → ``(z, mu, logvar)``。"""
        if x.ndim != 3:
            raise ValueError(f"输入应为 (B, T, C)，实际 {tuple(x.shape)}")
        if (x.shape[1], x.shape[2]) != (self.time_len, self.in_channels):
            raise ValueError(
                f"输入尺寸应为 (B, {self.time_len}, {self.in_channels})，实际 {tuple(x.shape)}"
            )
        h = self.encoder(x.unsqueeze(1))            # (B, L, 1, 1)
        h = h.flatten(1)                            # (B, L)

        if not self.variational:
            return h, None, None

        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(-30.0, 20.0)   # 数值稳定
        return self.reparameterize(mu, logvar), mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` → ``(B, T, 8)``。"""
        h = z.unsqueeze(-1).unsqueeze(-1)           # (B, L, 1, 1)
        h = self.dec_up(h)                          # (B, C, 84, 1)
        h = self.upsample(h)                        # (B, C, T, 1)
        h = self.dec_head(h)                        # (B, 8, T, 1)
        return h.squeeze(-1).transpose(1, 2)        # (B, T, 8)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        z, mu, logvar = self.encode(x)
        return ModelOutput(recon=self.decode(z), mu=mu, logvar=logvar, z=z)
