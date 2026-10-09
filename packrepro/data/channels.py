"""把 20 通道切成电压 / 电流 / 温度三族。

列布局的来源与可信度
--------------------
论文 §4.1 只说输入是「八路电压 + 八路派生电流 + 四路温度」（合计 20），
**没有给出列序**。本地实测的奇偶列结构与其吻合，故按下表推断：

    奇数列 1,3,…,15  ->  8 路单体电压
    偶数列 2,4,…,16  ->  8 路电流
    列  17,18,19,20  ->  4 路温度

这属于 ``INFERRED``（数值吻合推出），**不是文档事实**。

⚠ 物理量纲换算
--------------
数据里的奇数列均值约为 0.346，``+3.0`` 后落在 2.94~3.65 V，与论文「单体电压 3.0~3.6 V」吻合；
偶数列最小约 ``-1.886``，``×100`` 后为 ``-188.6 A``，与论文「放电 0~-200 A」吻合。
但这只是**数值吻合**，没有字段字典佐证。而且换算会**直接改变余弦相似度**的数值。
故默认 ``apply_physical_scale=False``，保持缓存原值。
"""

from __future__ import annotations

import numpy as np


def _to_index(cols_1based: list[int]) -> list[int]:
    """1-based 列号（MATLAB 习惯）→ 0-based 下标。"""
    return [c - 1 for c in cols_1based]


def split_channels(
    signals: np.ndarray,
    *,
    voltage_cols: list[int],
    current_cols: list[int],
    temperature_cols: list[int],
    apply_physical_scale: bool = False,
    voltage_offset: float = 3.0,
    current_scale: float = 100.0,
) -> dict[str, np.ndarray]:
    """``(N, T, 20)`` → 三族通道。

    Parameters
    ----------
    signals
        形状 ``(N, T, 20)``。
    voltage_cols, current_cols, temperature_cols
        **1-based** 列号，与 MATLAB 习惯一致，便于与数据说明对照。
    apply_physical_scale
        是否施加未经证实的 ``+3.0`` / ``×100`` 换算。默认 False。

    Returns
    -------
    dict
        ``{"voltage": (N,T,8), "current": (N,T,8), "temperature": (N,T,4), "full": (N,T,20)}``
    """
    if signals.ndim != 3:
        raise ValueError(f"signals 应为 3 维 (N, T, C)，实际 {signals.ndim} 维：{signals.shape}")

    n_cols = signals.shape[2]
    idx_v = _to_index(voltage_cols)
    idx_i = _to_index(current_cols)
    idx_t = _to_index(temperature_cols)

    if max(max(idx_v), max(idx_i), max(idx_t)) >= n_cols:
        raise ValueError(f"列号越界：最大列号超过通道数 {n_cols}")

    voltage = signals[:, :, idx_v]
    current = signals[:, :, idx_i]
    temperature = signals[:, :, idx_t]

    if apply_physical_scale:
        voltage = voltage + voltage_offset
        current = current * current_scale

    return {
        "voltage": np.ascontiguousarray(voltage),
        "current": np.ascontiguousarray(current),
        "temperature": np.ascontiguousarray(temperature),
        "full": signals,
    }
