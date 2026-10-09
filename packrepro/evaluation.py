"""评价：算特征 → 拟合 LOF → 检测 → 汇总。

核心口径
--------
* 特征在**物理电压域**计算：模型输出的是归一化电压，必须先 ``inverse_transform``
  回原始域，否则 §2.3 的 Q3（单位为电压）与余弦相似度都失去意义。
* LOF 的拟合集（校准集）= **正常训练数据**，与论文 §2.4「阈值取正常训练集 LOF
  分数的 99 分位」一致。
* ``persistence`` 会同时报 1 与 5 两版，因为论文**没有**去抖规则（见 docs/DEVIATIONS.md）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Config
from .data.dataset import PackData
from .data.normalize import ChannelScaler
from .detect import LOFDetector
from .features import compute_features
from .model.net import AsymmetricAE


@dataclass
class PackEval:
    """单个包的评价结果。"""

    pack_id: int
    n_windows: int
    exceedance_rate: float
    confirmed_rate: float
    first_alarm_frac: float | None   # 首次确认报警的窗口序号 / 总窗数
    threshold: float
    k_used: int
    calib: dict[str, float]
    scores: np.ndarray = field(repr=False)
    exceeded: np.ndarray = field(repr=False)
    confirmed: np.ndarray = field(repr=False)


@torch.no_grad()
def reconstruct(
    model: AsymmetricAE,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """对 ``(N, T, 20)`` 归一化输入做重建，返回归一化域的 ``(N, T, 8)``。"""
    model.eval()
    out = np.empty((x.shape[0], x.shape[1], model.out_channels), dtype=np.float32)
    for s in range(0, x.shape[0], batch_size):
        e = min(s + batch_size, x.shape[0])
        xb = torch.from_numpy(np.ascontiguousarray(x[s:e], dtype=np.float32)).to(device)
        out[s:e] = model(xb).recon.float().cpu().numpy()
    return out


def compute_pack_features(
    model: AsymmetricAE,
    pack: PackData,
    *,
    ch_scaler: ChannelScaler,
    volt_scaler: ChannelScaler,
    device: torch.device,
    q3_percentile: float = 75.0,
    eps: float = 1e-8,
    batch_size: int = 512,
) -> np.ndarray:
    """``(N, 24)`` 特征，在**物理电压域**上算。"""
    x_norm = ch_scaler.transform(pack.full)
    recon_norm = reconstruct(model, x_norm, device=device, batch_size=batch_size)

    # 回到物理电压域 —— 两个指标都依赖这一步
    meas_v = volt_scaler.inverse_transform(pack.voltage.astype(np.float64))
    recon_v = volt_scaler.inverse_transform(recon_norm.astype(np.float64))

    return compute_features(meas_v, recon_v, q3_percentile=q3_percentile, eps=eps)


def evaluate_pack(
    *,
    cfg: Config,
    calib_features: np.ndarray,
    target_features: np.ndarray,
    pack_id: int,
) -> tuple[PackEval, LOFDetector]:
    """用 ``calib_features`` 拟合 LOF，在 ``target_features`` 上检测。"""
    k = cfg.get("detect.k")
    q = cfg.get("detect.threshold_quantile")
    std = cfg.get("detect.standardize")

    detector = LOFDetector(k=k, threshold_quantile=q, persistence=1, standardize=std)
    detector.fit(calib_features)
    res = detector.detect(target_features)
    calib = detector.calib_score_summary()

    n = res.scores.size
    first = int(np.argmax(res.confirmed)) if res.confirmed.any() else None

    return (
        PackEval(
            pack_id=int(pack_id),
            n_windows=n,
            exceedance_rate=float(res.exceeded.mean()),
            confirmed_rate=float(res.confirmed.mean()),
            first_alarm_frac=None if first is None else first / n,
            threshold=res.threshold,
            k_used=res.k_used,
            calib=calib,
            scores=res.scores,
            exceeded=res.exceeded,
            confirmed=res.confirmed,
        ),
        detector,
    )


def persistence_variant(
    scores: np.ndarray, threshold: float, persistence: int
) -> tuple[np.ndarray, np.ndarray]:
    """对同一组分数套用不同的持久性窗数，用于 K=1 与 K=5 对照。"""
    exceeded = scores > threshold
    confirmed = np.zeros_like(exceeded, dtype=bool)
    run = 0
    for i, flag in enumerate(exceeded):
        run = run + 1 if flag else 0
        confirmed[i] = run >= persistence
    return exceeded, confirmed
