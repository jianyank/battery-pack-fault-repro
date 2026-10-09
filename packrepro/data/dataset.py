"""把 :class:`~packrepro.data.mat_io.Bundle` 组装成按包切分的数据集。

数据已是切好的窗（``Xall (256, 20, N)``），本模块不再切窗，只做按包分组、通道切分、按需归一化。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from .channels import split_channels
from .mat_io import Bundle, load_bundle


@dataclass
class PackData:
    """单个电池包的窗口集合。数组一律 ``(N, 256, k)``。"""

    pack_id: int
    voltage: np.ndarray          # (N, 256, 8)
    current: np.ndarray          # (N, 256, 8)
    temperature: np.ndarray      # (N, 256, 4)
    full: np.ndarray             # (N, 256, 20)
    hidden: np.ndarray | None    # (N, 9) 或 None
    time_slot: np.ndarray | None # (N,) 或 None

    @property
    def n_windows(self) -> int:
        return int(self.full.shape[0])

    def apply_scaler(self, scaler) -> "PackData":
        """返回施加了同一套归一化的新 :class:`PackData`（不原地修改）。"""
        return PackData(
            pack_id=self.pack_id,
            voltage=scaler.transform(self.voltage),
            current=scaler.transform(self.current),
            temperature=scaler.transform(self.temperature),
            full=scaler.transform(self.full),
            hidden=self.hidden,
            time_slot=self.time_slot,
        )


def load_experiment(cfg: Config, name: str) -> Bundle:
    """按配置读一个 ``Stand*.mat``（train / test1 / test2 / test3）。"""
    return load_bundle(
        cfg.data_file(name),
        name=name,
        signal_var=cfg.get("data.var.signal"),
        id_candidates=cfg.get("data.var.id_candidates"),
        time_var=cfg.get("data.var.time"),
        hidden_var=cfg.get("data.var.hidden"),
        expected_rows=cfg.get("data.expected.rows"),
        expected_cols=cfg.get("data.expected.cols"),
        hidden_channels=cfg.get("data.expected.hidden_channels"),
    )


def build_packs(bundle: Bundle, cfg: Config) -> dict[int, PackData]:
    """把 Bundle 按包号拆成 ``{pack_id: PackData}``。

    全部窗口（含论文所称的"故障段"）都会被保留，调用方须自行按 §4.1 选取窗口，
    不要把返回值当作纯正常数据。
    """
    ch = split_channels(
        bundle.signals,
        voltage_cols=cfg.get("channels.cell_voltage"),
        current_cols=cfg.get("channels.current"),
        temperature_cols=cfg.get("channels.temperature"),
        apply_physical_scale=cfg.get("channels.apply_physical_scale"),
        voltage_offset=cfg.get("channels.voltage_offset"),
        current_scale=cfg.get("channels.current_scale"),
    )

    # 时间变量在本数据上不可用（MCOS 对象，见 mat_io）；长度对不上时一律丢弃，
    # 不静默错位 —— 错位的时间轴比没有时间轴更危险。
    ts = bundle.time_slot
    if ts is not None and np.asarray(ts).reshape(-1).size != bundle.pack_ids.size:
        ts = None

    packs: dict[int, PackData] = {}
    for pid in np.unique(bundle.pack_ids):
        m = bundle.pack_ids == pid
        p = int(pid)
        packs[p] = PackData(
            pack_id=p,
            voltage=ch["voltage"][m],
            current=ch["current"][m],
            temperature=ch["temperature"][m],
            full=ch["full"][m],
            hidden=None if bundle.hidden is None else bundle.hidden[m],
            time_slot=None if ts is None else np.asarray(ts).reshape(-1)[m],
        )
    return packs


def train_val_split(
    n: int, *, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """把 ``n`` 个窗口切成 (train_idx, val_idx)。

    train 文件没有时间轴，无法按时间切分，故用固定种子的无放回随机切分 ——
    属 ``ASSUMED``（``docs/DEVIATIONS.md`` A6）。
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio 应在 [0, 1) 内")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    return perm[n_val:], perm[:n_val]
