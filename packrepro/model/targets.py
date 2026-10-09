"""构造 ``L_late`` 的对齐目标向量。

论文 §2.2 只说对齐到 "state of charge and internal resistance"，未说内阻是逐单体还是包级。
默认 ``pack_mean``（1 个内阻槽对 8 个单体的均值），两种读法的依据见 ``docs/DEVIATIONS.md`` A10。

``Hiddall`` 与对齐目标同源，不是独立真值，不得用于定位类结论（``docs/DEVIATIONS.md`` G）。
"""

from __future__ import annotations

import numpy as np


def build_align_target(
    hidden: np.ndarray,
    *,
    soc_channel: int,
    resistance_channels: list[int],
    resistance_target: str = "pack_mean",
) -> np.ndarray:
    """``hidden (N, 9)`` → 对齐目标 ``(N, D)``（**未标准化**）。

    ``hidden`` 的 ``soc_channel`` 列为 SOC，``resistance_channels`` 为 8 个单体内阻
    （列号均 1-based，按单体 1..8 顺序）。
    ``pack_mean`` → ``(N, 2)`` = [SOC, mean(R)]；``per_cell`` → ``(N, 9)`` = [SOC, R₁…R₈]。
    """
    if hidden.ndim != 2:
        raise ValueError(f"hidden 应为 (N, 9)，实际 {hidden.shape}")

    soc = hidden[:, soc_channel - 1 : soc_channel]
    res = hidden[:, [c - 1 for c in resistance_channels]]

    if resistance_target == "pack_mean":
        r = res.mean(axis=1, keepdims=True)
    elif resistance_target == "per_cell":
        r = res
    else:
        raise ValueError(f"未知 resistance_target：{resistance_target!r}")

    return np.concatenate([soc, r], axis=1)


def align_dim_for(resistance_target: str, resistance_channels: list[int]) -> int:
    """目标向量长度：1(SOC) + 1 或 8。"""
    n_res = 1 if resistance_target == "pack_mean" else len(resistance_channels)
    return 1 + n_res
