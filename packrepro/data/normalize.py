"""按通道的稳健标准化。

论文完全未提输入归一化，本模块及其参数均属 ``ASSUMED``（``docs/DEVIATIONS.md`` A13）。

取中位数 / MAD 而非均值 / 标准差：电池原始数据存在明显离群段（静置、异常脉冲），
均值与标准差会被它们带偏。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 拟合统计量时最多使用的行数（确定性等距抽样）。
#: 中位数/MAD 是稳定统计量，抽样不改变其量级；而全量拟合会额外分配 GB 级临时数组。
MAX_FIT_ROWS = 500_000


@dataclass(frozen=True)
class ChannelScaler:
    """按通道的 (中位数, 尺度) 标准化：``(x - center) / scale``。"""

    center: np.ndarray  # (C,)
    scale: np.ndarray   # (C,)
    method: str

    @classmethod
    def fit(cls, signals: np.ndarray, *, method: str = "robust", floor: float = 1e-3) -> "ChannelScaler":
        """在 ``(N, T, C)`` 上拟合。统计量只应在训练折上拟合，调用方负责这一点。"""
        if signals.ndim != 3:
            raise ValueError(f"signals 应为 3 维 (N,T,C)，实际 {signals.shape}")

        flat = signals.reshape(-1, signals.shape[-1])
        if flat.shape[0] > MAX_FIT_ROWS:
            idx = np.linspace(0, flat.shape[0] - 1, MAX_FIT_ROWS).astype(np.int64)
            flat = flat[idx]

        if method == "robust":
            center = np.median(flat, axis=0)
            scale = 1.4826 * np.median(np.abs(flat - center), axis=0)
            # 近零方差通道的 MAD 可能为 0，退回标准差；仍为 0 时用下限兜底。
            std = flat.std(axis=0)
            degenerate = scale < 1e-7
            scale = np.where(degenerate & (std >= 1e-7), std, scale)
            scale = np.maximum(scale, floor)
        elif method == "zscore":
            center = flat.mean(axis=0)
            scale = np.maximum(flat.std(axis=0), floor)
        elif method == "none":
            center = np.zeros(flat.shape[1])
            scale = np.ones(flat.shape[1])
        else:
            raise ValueError(f"未知归一化方法：{method!r}")

        return cls(center=center, scale=scale, method=method)

    def transform(self, signals: np.ndarray) -> np.ndarray:
        return (signals - self.center) / self.scale

    def inverse_transform(self, signals: np.ndarray) -> np.ndarray:
        return signals * self.scale + self.center

    def subset(self, indices: list[int]) -> "ChannelScaler":
        """取列子集对应的 scaler。

        例：拟合在 20 通道上的 scaler，用 ``subset(电压列的 0-based 下标)`` 得到电压专用
        scaler，保证 ``transform`` 前后通道一致。
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.center.size):
            raise ValueError(f"列下标越界：{indices}（共 {self.center.size} 通道）")
        return ChannelScaler(center=self.center[idx], scale=self.scale[idx], method=self.method)
