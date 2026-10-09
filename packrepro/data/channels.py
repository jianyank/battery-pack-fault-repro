"""把 20 通道切成电压 / 电流 / 温度三族。

论文 §4.1 只说 8 电压 + 8 电流 + 4 温度，未给列序。本模块按本地奇偶列结构推断：
奇数列 1,3,…,15 为电压，偶数列 2,4,…,16 为电流，列 17–20 为温度。属 ``INFERRED``，非文档事实。

``+3.0`` / ``×100`` 量纲换算与论文量程数值吻合，但无字段字典佐证，且会直接改变余弦
相似度，故默认关闭。两处依据见 ``docs/DEVIATIONS.md`` C3。
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

    三个列参数均为 **1-based**（与 MATLAB 习惯一致，便于与数据说明对照）。
    返回 ``{"voltage": (N,T,8), "current": (N,T,8), "temperature": (N,T,4), "full": (N,T,20)}``。
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
