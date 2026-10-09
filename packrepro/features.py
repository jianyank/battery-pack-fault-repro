"""24 维故障指标（论文 §2.3）。

每个单体 3 个指标 × 8 单体 = 24 维包级特征向量：

1. **电压偏差标准差** ``σ_V,i``（§2.3.1，式 (5)–(7)）—— 单体相对包均值的波动。
2. **实测/重建电压余弦相似度** ``CS_i``（§2.3.2，式 (8)）—— 波形方向一致性。
3. **重建误差第三四分位数** ``Q3_i``（§2.3.3）—— 误差上部的中心趋势。

⚠ 两处必须继承的口径
--------------------
* 重建误差与余弦相似度都应在**物理电压域**计算，因此调用方须先用通道 scaler
  把重建结果 ``inverse_transform`` 回原始域，再传给本模块。
* 论文用 Q3 的理由是"对尾部极值敏感"—— 这个说法在统计上站不住
  （四分位数恰恰是对极值**不**敏感的量）。本模块**照论文实现 Q3**，
  但该理由已在 ``docs/DEVIATIONS.md`` 记为"论文论证有误、实现照旧"，
  并另行提供高分位（p95/p99）作为对照，不做默认。
"""

from __future__ import annotations

import numpy as np

FAMILIES = ("sigma_v", "cosine", "q3")


def voltage_deviation_std(volt: np.ndarray) -> np.ndarray:
    """``σ_V,i``：``(N, L, C)`` → ``(N, C)``。式 (5)–(7)。

    ``V_mean(k) = mean_i V_i(k)``，``V_E,i(k) = V_i(k) − V_mean(k)``，
    ``σ_V,i = sqrt( Σ_k (V_E,i(k) − mean_k V_E,i)² / (L−1) )``。
    """
    if volt.ndim != 3:
        raise ValueError(f"volt 应为 (N, L, C)，实际 {volt.shape}")
    l = volt.shape[1]
    if l < 2:
        raise ValueError("窗口长度至少为 2 才能算样本标准差")

    v_mean = volt.mean(axis=2, keepdims=True)
    dev = volt - v_mean
    return dev.std(axis=1, ddof=1)


def cosine_similarity(meas: np.ndarray, recon: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """``CS_i``：``(N, L, C) × (N, L, C)`` → ``(N, C)``。式 (8)。"""
    if meas.shape != recon.shape:
        raise ValueError(f"实测与重建形状不一致：{meas.shape} vs {recon.shape}")
    num = (meas * recon).sum(axis=1)
    den = np.linalg.norm(meas, axis=1) * np.linalg.norm(recon, axis=1)
    return num / np.maximum(den, eps)


def reconstruction_q3(recon_err_abs: np.ndarray, *, percentile: float = 75.0) -> np.ndarray:
    """``Q3_i``：``(N, L, C)`` 的绝对重建误差 → ``(N, C)``。

    论文 §2.3.3：对长度为 L 的误差序列升序排序后取第 75 个样本百分位。
    """
    if recon_err_abs.ndim != 3:
        raise ValueError(f"recon_err_abs 应为 (N, L, C)，实际 {recon_err_abs.shape}")
    return np.percentile(recon_err_abs, percentile, axis=1)


def compute_features(
    volt_meas: np.ndarray,
    volt_recon: np.ndarray,
    *,
    q3_percentile: float = 75.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """三族拼接 → ``(N, 24)``。

    Parameters
    ----------
    volt_meas, volt_recon
        ``(N, L, 8)``，**同一物理电压域**（原始域，非归一化域）。

    Returns
    -------
    np.ndarray
        ``(N, 24)``：列序为 ``[σ_V(8), CS(8), Q3(8)]``。
    """
    sigma_v = voltage_deviation_std(volt_meas)
    cos = cosine_similarity(volt_meas, volt_recon, eps=eps)
    q3 = reconstruction_q3(np.abs(volt_meas - volt_recon), percentile=q3_percentile)
    return np.concatenate([sigma_v, cos, q3], axis=1)


def feature_names(n_cells: int = 8) -> list[str]:
    """列名，便于落盘与人工核对。"""
    return (
        [f"sigma_v_cell{i}" for i in range(1, n_cells + 1)]
        + [f"cosine_cell{i}" for i in range(1, n_cells + 1)]
        + [f"q3_cell{i}" for i in range(1, n_cells + 1)]
    )
