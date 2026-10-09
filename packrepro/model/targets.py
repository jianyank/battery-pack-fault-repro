"""构造 ``L_late`` 的对齐目标向量。

论文 §2.2 只说把「一部分潜变量」对齐到「state of charge and internal resistance」，
**没有说明内阻是逐单体还是包级**。两种读法都成立：

* ``pack_mean``  —— 1 个内阻槽，对齐 8 个单体内阻的均值。
  与论文 §4.2 的自述吻合：Fig.7(b) 画的是**单条** "internal-resistance-related latent
  feature"，而作者抱怨它"fails to capture the inter-cell discrepancies"——
  正是"一个槽对多个单体"才会有的失败模式。
* ``per_cell``   —— 8 个内阻槽，逐个单体对齐。

默认取 ``pack_mean``；本项为 ``ASSUMED``（见 docs/DEVIATIONS.md）。

⚠ ``Hiddall`` 与模型的对齐目标**同源**，不是独立真值，不得用于定位类结论。
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

    Parameters
    ----------
    hidden
        ``(N, 9)``，第 ``soc_channel`` 列（1-based）为 SOC，其余为 8 个单体内阻。
    soc_channel
        1-based 列号。
    resistance_channels
        1-based 列号列表，按单体 1..8 顺序。
    resistance_target
        ``"pack_mean"`` 或 ``"per_cell"``。

    Returns
    -------
    np.ndarray
        ``pack_mean`` → ``(N, 2)`` = [SOC, mean(R)]；
        ``per_cell``  → ``(N, 9)`` = [SOC, R₁…R₈]。
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
