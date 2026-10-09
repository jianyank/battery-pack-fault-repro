"""LOF 异常检测（论文 §2.4）。

在正常特征上拟合 LOF，取分数的 ``quantile`` 分位为阈值 θ（论文：99 分位）；
``score > θ`` 记为越限；连续 ``persistence`` 个窗口越限才从第 persistence 窗起确认报警，
不回填先前窗口。

三项论文未给、已在 ``configs/base.yaml`` 的 ``assumed:`` 段登记的取值：
``k``（A4）、``persistence``（A5）、``standardize``（A12）。

在域内取 99 分位意味着校准集自身必然约 1% 越限；跨域评分时越限比例远高于 1%，
是校准域与评分域不一致的必然结果，不是 LOF 算错（``docs/DEVIATIONS.md`` F）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class DetectionResult:
    scores: np.ndarray          # (N,) LOF 分数，正常 ≈ 1，越大越异常
    threshold: float            # θ
    exceeded: np.ndarray        # (N,) bool 单窗越限
    confirmed: np.ndarray       # (N,) bool 连续 persistence 窗确认
    alarm_started: np.ndarray   # (N,) bool 报警起始沿
    persistence: int
    k_used: int


class LOFDetector:
    """LOF + 分位阈值 + 持久性。"""

    def __init__(
        self,
        *,
        k: int = 20,
        threshold_quantile: float = 0.99,
        persistence: int = 1,
        standardize: bool = True,
        floor: float = 1e-7,
    ) -> None:
        if k < 1:
            raise ValueError("k 必须 >= 1")
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile 必须在 (0,1) 内")
        if persistence < 1:
            raise ValueError("persistence 必须 >= 1")
        self.k = int(k)
        self.quantile = float(threshold_quantile)
        self.persistence = int(persistence)
        self.standardize = standardize
        self.floor = float(floor)

        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._model: LocalOutlierFactor | None = None
        self.threshold_: float | None = None
        self.k_used_: int | None = None
        self.calib_scores_: np.ndarray | None = None

    # ---------------------------------------------------------------- 内部
    def _z(self, x: np.ndarray, *, fit: bool) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"特征应为 (N, D)，实际 {x.shape}")
        if not self.standardize:
            return x
        if fit:
            self._center = np.median(x, axis=0)
            mad = 1.4826 * np.median(np.abs(x - self._center), axis=0)
            std = x.std(axis=0)
            degenerate = mad < 1e-12
            scale = np.where(degenerate & (std >= 1e-12), std, mad)
            self._scale = np.maximum(scale, self.floor)
        if self._center is None or self._scale is None:
            raise RuntimeError("先调用 fit 再 score")
        return (x - self._center) / self._scale

    def _score(self, z: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("先调用 fit 再 score")
        # sklearn 的 LOF 在 novelty=True 下 score_samples 返回 **负** LOF，
        # 取负号后才是经典 LOF（正常 ≈ 1，越大越异常）。
        return -self._model.score_samples(z)

    # ---------------------------------------------------------------- 接口
    def fit(self, calib_features: np.ndarray) -> "LOFDetector":
        """在正常特征上拟合并冻结阈值 θ。"""
        z = self._z(calib_features, fit=True)
        n = z.shape[0]
        if n < 3:
            raise ValueError("至少需要 3 个校准窗口")
        self.k_used_ = int(min(self.k, n - 1))
        self._model = LocalOutlierFactor(n_neighbors=self.k_used_, novelty=True, n_jobs=1).fit(z)
        calib_scores = self._score(z)
        self.calib_scores_ = calib_scores
        self.threshold_ = float(np.quantile(calib_scores, self.quantile))
        return self

    def calib_score_summary(self) -> dict[str, float]:
        """校准集自身的分数分布。

        用于核对"正常 ≈ 1"这一先验与 θ 的实际量级：θ 远大于 1（如 4~7）说明正常工况
        本身就散布在多个密度区域，阈值跨域迁移能力很弱。
        """
        if self.calib_scores_ is None:
            raise RuntimeError("先调用 fit")
        s = self.calib_scores_
        return {
            "n": float(s.size),
            "median": float(np.median(s)),
            "p90": float(np.quantile(s, 0.90)),
            "p99": float(np.quantile(s, 0.99)),
            "max": float(s.max()),
            "threshold": float(self.threshold_ or np.nan),
        }

    def detect(self, features: np.ndarray) -> DetectionResult:
        """对新窗口评分并做持久性判定。"""
        if self.threshold_ is None:
            raise RuntimeError("先调用 fit 再 detect")
        z = self._z(features, fit=False)
        scores = self._score(z)
        exceeded = scores > self.threshold_

        confirmed = np.zeros_like(exceeded, dtype=bool)
        run = 0
        for i, flag in enumerate(exceeded):
            run = run + 1 if flag else 0
            confirmed[i] = run >= self.persistence

        previous = np.concatenate([[False], confirmed[:-1]])
        return DetectionResult(
            scores=scores,
            threshold=float(self.threshold_),
            exceeded=exceeded,
            confirmed=confirmed,
            alarm_started=confirmed & ~previous,
            persistence=self.persistence,
            k_used=int(self.k_used_ or self.k),
        )
