"""读取 ``Stand*.mat``。

实测事实（见 ``docs/REPRODUCTION_NOTES.md``）：四个文件均为 MAT v5（非 v7.3），
``h5py`` 读不了，只能用 ``scipy.io``；``Xall`` 形状 ``(256, 20, N)``，样本本身就是
256×20 的窗，滑窗步长不可知（``docs/DEVIATIONS.md`` B1）；ID 变量名在文件间不统一
（``ThisTrainID`` / ``ThisID1`` / ``ThisID``）；``Hiddall`` 仅 train 与 test1 有，
``ThisTimeSlot`` 仅 test1/2/3 有。

已知 scipy 缺陷：``whosmat`` 在 ``StandTestData1.mat`` 上抛 ``TypeError`` —— 该文件本身
是合法的 v5 文件。故失败时回退 ``loadmat`` 并记录实际路径，供审计留痕，不静默掩盖。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

#: 上一次 whosmat 走的是哪条路径（'whosmat' 或 'loadmat-fallback(...)'）
_last_whosmat_method: str = ""


def last_whosmat_method() -> str:
    """返回上一次 :func:`whosmat_safe` 实际使用的读取路径，供审计留痕。"""
    return _last_whosmat_method


def whosmat_safe(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    """列变量清单；whosmat 失败时回退 loadmat 并记录路径。

    回退的代价是要把整个文件读进内存，所以只在必要时发生。
    """
    global _last_whosmat_method
    try:
        out = [(n, tuple(s), c) for n, s, c in sio.whosmat(str(path))]
        _last_whosmat_method = "whosmat"
        return out
    except Exception as exc:  # noqa: BLE001 —— 需捕获 scipy 的具体缺陷
        raw = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
        out = [
            (k, tuple(np.shape(v)), str(np.asarray(v).dtype))
            for k, v in raw.items()
            if not k.startswith("__")
        ]
        _last_whosmat_method = f"loadmat-fallback ({type(exc).__name__}: {exc})"
        return out


@dataclass(frozen=True)
class Bundle:
    """一个 ``Stand*.mat`` 读出来的内容。样本在前：``(N, 256, 20)``。"""

    name: str
    signals: np.ndarray          # (N, 256, C) float64
    pack_ids: np.ndarray         # (N,) int
    time_slot: np.ndarray | None # (N,) 或 None —— 实测无法读出，见 time_note
    hidden: np.ndarray | None    # (N, 9) 或 None；推断为 [SOC, R₁…R₈]，未经核实
    loader: str                  # 实际读取路径，审计用
    time_note: str = "absent"    # 该文件时间变量的实测状态：absent / numeric / 不可解码

    @property
    def n_windows(self) -> int:
        return int(self.signals.shape[0])

    def pack_counts(self) -> dict[int, int]:
        """每个包各有多少窗，按包号排序。"""
        ids, counts = np.unique(self.pack_ids, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}


def _first_present(raw: dict, names: list[str]) -> str | None:
    for n in names:
        if n in raw:
            return n
    return None


def load_bundle(
    path: Path,
    *,
    name: str = "",
    signal_var: str = "Xall",
    id_candidates: list[str] | None = None,
    time_var: str = "ThisTimeSlot",
    hidden_var: str = "Hiddall",
    expected_rows: int = 256,
    expected_cols: int = 20,
    hidden_channels: int = 9,
) -> Bundle:
    """读一个 ``Stand*.mat`` 并做形状断言。形状不符立即抛错，不静默降级。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    id_candidates = id_candidates or ["ThisTrainID", "ThisID1", "ThisID"]

    # 先列变量（顺带触发并记录 whosmat 缺陷路径）
    listing = whosmat_safe(path)
    loader = _last_whosmat_method
    available = {n for n, _, _ in listing}

    raw = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)

    if signal_var not in raw:
        raise KeyError(f"{path.name} 中没有信号变量 {signal_var!r}；实有：{sorted(available)}")

    x = np.asarray(raw[signal_var], dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"{signal_var} 应为 3 维，实际 {x.ndim} 维：{x.shape}")
    rows, cols, n = x.shape
    if (rows, cols) != (expected_rows, expected_cols):
        raise ValueError(
            f"{path.name} 的 {signal_var} 形状应为 ({expected_rows}, {expected_cols}, N)，"
            f"实际 {x.shape}"
        )

    id_name = _first_present(raw, id_candidates)
    if id_name is None:
        raise KeyError(
            f"{path.name} 中找不到 ID 变量，试过 {id_candidates}；实有：{sorted(available)}"
        )
    pack_ids = np.asarray(raw[id_name]).reshape(-1).astype(np.int64)
    if pack_ids.size != n:
        raise ValueError(f"ID 长度 {pack_ids.size} 与窗数 {n} 不一致")

    # 实测 test1/2/3 的 ``ThisTimeSlot`` 是 MATLAB MCOS datetime 对象，scipy 无法解码其内容，
    # 故时间轴在本项目里实际不可用 —— 这独立印证了"窗口序号不是时间轴"。
    # 如实记录读到了什么，不假装读到了。
    time_slot = None
    time_note = "absent"
    if time_var in raw:
        t = np.asarray(raw[time_var])
        if t.dtype.names is not None or t.dtype == object:
            time_note = f"present but unreadable MCOS object (dtype fields={t.dtype.names})"
        elif t.size == n:
            time_slot = t.reshape(-1)
            time_note = f"numeric, size={t.size}"
        else:
            time_note = f"present but shape {t.shape} != ({n},)"

    hidden = None
    if hidden_var in raw:
        h = np.asarray(raw[hidden_var], dtype=np.float64)
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != hidden_channels:
            raise ValueError(
                f"{hidden_var} 形状应为 (1, {hidden_channels}, N)，实际 {h.shape}"
            )
        hidden = h.reshape(hidden_channels, -1).T  # (N, 9)
        if hidden.shape[0] != n:
            raise ValueError(f"{hidden_var} 窗数 {hidden.shape[0]} 与信号 {n} 不一致")

    # (256, 20, N) -> (N, 256, 20)：样本在前
    signals = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))

    if not np.isfinite(signals).all():
        raise ValueError(f"{path.name} 的信号含非有限值")

    # loadmat 会把整个文件读进内存；本函数的产出已全部复制出来，及时释放峰值。
    # 不这么做时，train（26508 窗）加载峰值可达约 3 GB。
    del raw, x
    gc.collect()

    return Bundle(
        name=name or path.stem,
        signals=signals,
        pack_ids=pack_ids,
        time_slot=time_slot,
        hidden=hidden,
        loader=loader,
        time_note=time_note,
    )
