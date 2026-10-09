"""出图。图注一律用英文，避免中文字体缺失导致方块字。

每张图都把口径写进标题/图注：协议（paper / lopo）、越限率不是准确率、纵轴是否为对数、
横轴是窗口序号而非时间。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_history(history_csv: Path, out: Path, title: str = "") -> Path:
    """训练历史：总损失与重建损失、分项损失。"""
    rows = list(csv.DictReader(Path(history_csv).open(encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for key, style in (("total", "-"), ("recon", "--")):
        axes[0].plot(ep, [float(r[f"train_{key}"]) for r in rows], style, label=f"train {key}")
        axes[0].plot(ep, [float(r[f"val_{key}"]) for r in rows], style, alpha=0.7, label=f"val {key}")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("total / reconstruction")

    for key in ("recon", "late", "reg"):
        axes[1].plot(ep, [float(r[f"val_{key}"]) for r in rows], label=f"val {key}")
    axes[1].set_xlabel("epoch")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].set_title("loss components (val)")

    fig.suptitle(title or "training history")
    return _save(fig, out)


def plot_lof_scores(
    scores: np.ndarray,
    threshold: float,
    *,
    pack_id: int,
    out: Path,
    protocol: str,
    log_y: bool = True,
) -> Path:
    """单包的 LOF 分数曲线与阈值线。横轴是窗口序号，不是时间。"""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(scores.size)
    ax.plot(x, scores, lw=0.6, color="#1f77b4", label="LOF score")
    ax.axhline(threshold, color="#d62728", ls="--", lw=1.2, label=f"threshold = {threshold:.3f}")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("window index (NOT time)")
    ax.set_ylabel("LOF score")
    ax.legend(fontsize=8)
    frac = float((scores > threshold).mean())
    ax.set_title(
        f"pack {pack_id} - LOF score (protocol={protocol}); "
        f"exceedance = {frac:.2%}\n"
        "Exceedance rate is a DESCRIPTIVE proportion, not a false-alarm rate."
    )
    return _save(fig, out)


def plot_exceedance_bars(
    rows: list[dict],
    *,
    out: Path,
    protocol: str,
    confirm_key: str = "confirmed_rate_K1",
    title: str = "exceedance by pack",
) -> Path:
    """各包越限 / 确认比例柱状图。

    确认列名由 ``confirm_key`` 给出，因为 ``rows`` 的列名由调用方决定
    （本项目为 ``confirmed_rate_K<K>``），不假设列名。
    """
    if not rows:
        raise ValueError("rows 为空")
    for r in rows:
        if confirm_key not in r:
            raise KeyError(f"rows 缺少确认列 {confirm_key!r}；实有：{sorted(r)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(rows))
    w = 0.38
    ax.bar(idx - w / 2, [r["exceedance_rate"] * 100 for r in rows], w,
           label="single-window exceedance", color="#1f77b4")
    ax.bar(idx + w / 2, [r[confirm_key] * 100 for r in rows], w,
           label=f"persistence-confirmed ({confirm_key.rsplit('_', 1)[-1]})", color="#ff7f0e")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"pack {r['pack_id']}" for r in rows])
    ax.set_ylabel("percent of windows")
    ax.legend(fontsize=8)
    ax.set_title(f"{title} (protocol={protocol})\nproportions only - no ground truth, no accuracy")
    return _save(fig, out)


def plot_voltage_evolution(pack_voltage: np.ndarray, *, pack_id: int, out: Path) -> Path:
    """各单体电压随窗口的演化（论文 Fig.5 形态）。

    纵轴取每个窗口的均值电压，横轴为窗口序号。
    """
    mean_v = pack_voltage.mean(axis=1)  # (N, 8)
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(mean_v.shape[1]):
        ax.plot(mean_v[:, c], lw=0.7, label=f"cell {c + 1}")
    ax.set_xlabel("window index (NOT time)")
    ax.set_ylabel("mean cell voltage (raw cache domain)")
    ax.legend(fontsize=7, ncol=4)
    ax.set_title(f"pack {pack_id} - cell voltage evolution (raw cache domain, no unverified offset applied)")
    return _save(fig, out)


def plot_resistance_and_latent(
    hidden: np.ndarray,
    latent: np.ndarray,
    *,
    pack_id: int,
    resistance_channels: list[int],
    out: Path,
    latent_slot: int = 0,
) -> Path:
    """内阻轨迹 vs 潜特征（论文 Fig.7 形态）。

    ``hidden`` 与对齐目标同源，不是独立真值；本图只展示对齐是否发生，不构成定位准确率证据。
    """
    res = hidden[:, [c - 1 for c in resistance_channels]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c in range(res.shape[1]):
        axes[0].plot(res[:, c], lw=0.7, label=f"cell {c + 1}")
    axes[0].set_xlabel("window index")
    axes[0].set_ylabel("resistance proxy (unknown units)")
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].set_title("(a) model-derived resistance proxy")

    axes[1].plot(latent[:, latent_slot], lw=0.7, color="#2ca02c")
    axes[1].set_xlabel("window index")
    axes[1].set_ylabel(f"latent slot {latent_slot}")
    axes[1].set_title("(b) aligned latent feature")

    fig.suptitle(
        f"pack {pack_id} - resistance proxy vs aligned latent "
        "(SAME-SOURCE, not independent ground truth)"
    )
    return _save(fig, out)
