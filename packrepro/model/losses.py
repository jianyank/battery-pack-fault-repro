"""复合损失 ``L_total = λ₁L_recon + λ₂L_late + λ₃L_reg``（论文式 (1)–(4)）。

``L_recon`` 只算 8 路单体电压（非对称解码器）；``L_late`` 把一部分潜变量对齐到
[SOC, 内阻]；``L_reg`` 为 KL 正则 ``D_KL(q(z|x) ‖ N(0,I))``。

``L_recon`` 用 ``F.mse_loss``（对全部元素求平均），与式 (2) 的按样本平均相差因子 ``T×C``。
该因子被 λ₁ 吸收，不改变优化方向，但 λ₁ 的绝对值与论文不可直接比较（``docs/DEVIATIONS.md`` C2）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .net import ModelOutput


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``D_KL(N(μ,σ²) ‖ N(0,I))``，按样本求均值，略去不依赖参数的常数项。"""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


@dataclass
class LossParts:
    """分项损失，供 history.csv 逐项记录（选权判据见 ``docs/DEVIATIONS.md`` A11）。"""

    total: torch.Tensor
    recon: torch.Tensor
    late: torch.Tensor
    reg: torch.Tensor


class CompositeLoss:
    """按 λ 加权。对齐目标须由调用方先在训练折上标准化再传入。"""

    def __init__(
        self,
        *,
        lambda1: float = 1.0,
        lambda2: float = 0.1,
        lambda3: float = 1e-3,
        soc_slots: int = 1,
        resistance_slots: int = 1,
    ) -> None:
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.soc_slots = int(soc_slots)
        self.resistance_slots = int(resistance_slots)

    @property
    def align_dim(self) -> int:
        """对齐目标向量的长度 = SOC 槽数 + 内阻槽数。"""
        return self.soc_slots + self.resistance_slots

    def __call__(
        self,
        out: ModelOutput,
        x_volt: torch.Tensor,
        align_target: torch.Tensor | None,
    ) -> LossParts:
        """``x_volt`` 为 ``(B, T, 8)`` **已归一化**的实测电压；``align_target`` 为
        ``(B, align_dim)`` **已标准化**的目标，为 None 时 ``L_late = 0``。
        """
        recon = F.mse_loss(out.recon, x_volt)

        late = torch.zeros((), device=x_volt.device, dtype=x_volt.dtype)
        if align_target is not None and self.lambda2 != 0.0:
            if out.z.shape[1] < self.align_dim:
                raise ValueError(
                    f"潜变量维度 {out.z.shape[1]} 小于对齐所需槽位 {self.align_dim}"
                )
            z_slots = out.z[:, : self.align_dim]
            late = F.mse_loss(z_slots, align_target)

        reg = torch.zeros((), device=x_volt.device, dtype=x_volt.dtype)
        if self.lambda3 != 0.0 and out.mu is not None and out.logvar is not None:
            reg = kl_divergence(out.mu, out.logvar)

        total = self.lambda1 * recon + self.lambda2 * late + self.lambda3 * reg
        return LossParts(total=total, recon=recon, late=late, reg=reg)
