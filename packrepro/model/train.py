"""单个 LOPO 折的训练。

四个包 {6,8,9,10} 做跨包留出：每次留出一个包，用另外三个训练（论文 §4.1）。
标准化统计量与对齐目标统计量只在训练折上拟合，不碰留出包。

选权判据为 ``val_recon`` 而非含 λ₂ 项的 ``val_loss`` —— 各配置 λ 不同，按 ``val_loss``
选会变成"谁加权项多谁大"，而不是"谁重建得准"。本项为 ``ASSUMED``（``docs/DEVIATIONS.md`` A11）。
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.dataset import PackData, train_val_split
from ..data.normalize import ChannelScaler
from .losses import CompositeLoss
from .net import AsymmetricAE
from .targets import align_dim_for, build_align_target


@dataclass
class FoldTensors:
    """一个折的张量集合（CPU，float32）。"""

    train_x: torch.Tensor       # (n, T, 20) 归一化后的全通道输入
    train_v: torch.Tensor       # (n, T, 8)  归一化后的电压（重建目标）
    train_a: torch.Tensor | None
    val_x: torch.Tensor
    val_v: torch.Tensor
    val_a: torch.Tensor | None
    ch_scaler: ChannelScaler      # 20 通道
    volt_scaler: ChannelScaler    # 电压列子集，推理时用于反归一化回物理域
    align_scaler: ChannelScaler | None
    n_train: int
    n_val: int


@dataclass
class FoldResult:
    fold_pack: int
    best_epoch: int
    best_val_recon: float
    epochs_run: int
    elapsed_s: float
    history_path: Path
    ckpt_path: Path
    history: list[dict] = field(default_factory=list)


def _concat_packs(packs: list[PackData], key: str) -> np.ndarray:
    return np.concatenate([getattr(p, key) for p in packs], axis=0)


def prepare_fold(cfg: Config, train_packs: list[PackData]) -> FoldTensors:
    """拼接训练折的窗、切分、拟合统计量、转张量。"""
    full = _concat_packs(train_packs, "full")
    volt = _concat_packs(train_packs, "voltage")
    hidden = None
    if all(p.hidden is not None for p in train_packs):
        hidden = _concat_packs(train_packs, "hidden")

    itr, iva = train_val_split(
        full.shape[0], val_ratio=cfg.get("train.val_ratio"), seed=cfg.get("train.seed")
    )

    norm_method = cfg.get("data.normalize.method")
    norm_floor = cfg.get("data.normalize.floor")
    ch = ChannelScaler.fit(full[itr], method=norm_method, floor=norm_floor)
    # 全通道 scaler 拟合在 20 通道上；重建目标只有 8 路电压，需取对应的列子集。
    volt_idx = [c - 1 for c in cfg.get("channels.cell_voltage")]
    ch_volt = ch.subset(volt_idx)

    align_scaler = None
    a_raw = None
    if hidden is not None and cfg.get("train.loss.lambda2") != 0.0:
        a_raw = build_align_target(
            hidden,
            soc_channel=cfg.get("hidden.soc_channel"),
            resistance_channels=cfg.get("hidden.resistance_channels"),
            resistance_target=cfg.get("train.loss.latent_align.resistance_target"),
        )
        expect = align_dim_for(
            cfg.get("train.loss.latent_align.resistance_target"),
            cfg.get("hidden.resistance_channels"),
        )
        if a_raw.shape[1] != expect:
            raise ValueError(f"对齐目标维度 {a_raw.shape[1]} 与预期 {expect} 不符")
        align_scaler = ChannelScaler.fit(
            a_raw[itr].reshape(-1, 1, a_raw.shape[1]), method="robust", floor=norm_floor
        )

    def to_t(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def align_t(idx: np.ndarray) -> torch.Tensor | None:
        if a_raw is None or align_scaler is None:
            return None
        return to_t(align_scaler.transform(a_raw[idx]))

    return FoldTensors(
        train_x=to_t(ch.transform(full[itr])),
        train_v=to_t(ch_volt.transform(volt[itr])),
        train_a=align_t(itr),
        val_x=to_t(ch.transform(full[iva])),
        val_v=to_t(ch_volt.transform(volt[iva])),
        val_a=align_t(iva),
        ch_scaler=ch,
        volt_scaler=ch_volt,
        align_scaler=align_scaler,
        n_train=int(itr.size),
        n_val=int(iva.size),
    )


def _run_epoch(
    model: AsymmetricAE,
    loss_fn: CompositeLoss,
    x: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    """跑一个 epoch；``optimizer=None`` 时为评估（no_grad）。"""
    train_mode = optimizer is not None
    model.train(train_mode)
    n = x.shape[0]
    sums = {"total": 0.0, "recon": 0.0, "late": 0.0, "reg": 0.0}

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            xb = x[s:e].to(device, non_blocking=True)
            vb = v[s:e].to(device, non_blocking=True)
            ab = None if a is None else a[s:e].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(xb)
                parts = loss_fn(out, vb, ab)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                parts.total.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = e - s
            for k, t in (("total", parts.total), ("recon", parts.recon),
                         ("late", parts.late), ("reg", parts.reg)):
                sums[k] += float(t.detach()) * bs

    return {k: val / n for k, val in sums.items()}


def train_fold(
    cfg: Config,
    *,
    fold_pack: int,
    train_packs: list[PackData],
    out_dir: Path,
    device: torch.device,
    max_epochs: int | None = None,
    seed: int | None = None,
) -> FoldResult:
    """训练一个折，返回最佳 checkpoint 与逐 epoch 历史。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("train.seed") if seed is None else seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = prepare_fold(cfg, train_packs)

    n_in = (
        len(cfg.get("channels.cell_voltage"))
        + len(cfg.get("channels.current"))
        + len(cfg.get("channels.temperature"))
    )
    model = AsymmetricAE(
        time_len=cfg.get("windows.length"),
        in_channels=n_in,
        out_channels=len(cfg.get("channels.cell_voltage")),
        latent_dim=cfg.get("model.latent_dim"),
        variational=cfg.get("model.variational"),
    ).to(device)

    loss_fn = CompositeLoss(
        lambda1=cfg.get("train.loss.lambda1"),
        lambda2=cfg.get("train.loss.lambda2"),
        lambda3=cfg.get("train.loss.lambda3"),
        soc_slots=cfg.get("train.loss.latent_align.soc_slots"),
        resistance_slots=cfg.get("train.loss.latent_align.resistance_slots"),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("train.lr"),
        weight_decay=cfg.get("train.weight_decay"),
    )

    epochs = int(max_epochs if max_epochs is not None else cfg.get("train.max_epochs"))
    patience = cfg.get("train.early_stop_patience")
    batch_size = cfg.get("train.batch_size")
    amp = bool(cfg.get("train.amp"))
    grad_clip = cfg.get("train.grad_clip")
    select_on = cfg.get("train.select_on")

    history: list[dict] = []
    best = {"epoch": -1, "val_recon": float("inf"), "state": None}
    bad = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        tr = _run_epoch(model, loss_fn, data.train_x, data.train_v, data.train_a,
                        batch_size=batch_size, device=device, optimizer=optimizer,
                        amp=amp, grad_clip=grad_clip)
        va = _run_epoch(model, loss_fn, data.val_x, data.val_v, data.val_a,
                        batch_size=batch_size, device=device, optimizer=None,
                        amp=amp, grad_clip=grad_clip)

        row = {"epoch": ep, "lr": optimizer.param_groups[0]["lr"]}
        for k, v in tr.items():
            row[f"train_{k}"] = v
        for k, v in va.items():
            row[f"val_{k}"] = v
        history.append(row)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"    ep {ep:4d}  train {tr['total']:.5f} (rec {tr['recon']:.5f})"
                  f"  val {va['total']:.5f} (rec {va['recon']:.5f})", flush=True)

        score = va[select_on.replace("val_", "")] if select_on.startswith("val_") else va["recon"]
        if score < best["val_recon"]:
            best = {
                "epoch": ep,
                "val_recon": score,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    早停：连续 {patience} 个 epoch 未改善（best ep={best['epoch']}）", flush=True)
                break

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt_path = out_dir / "best.pt"
    torch.save(
        {
            "fold_pack": fold_pack,
            "epoch": best["epoch"],
            "val_recon": best["val_recon"],
            "model_state": best["state"],
            "ch_center": data.ch_scaler.center,
            "ch_scale": data.ch_scaler.scale,
            # 电压列专用 scaler：推理时把重建结果反归一化回物理电压域（§2.3 的三个指标都在原始域上算）
            "volt_center": data.volt_scaler.center,
            "volt_scale": data.volt_scaler.scale,
            "align_center": None if data.align_scaler is None else data.align_scaler.center,
            "align_scale": None if data.align_scaler is None else data.align_scaler.scale,
            "n_train": data.n_train,
            "n_val": data.n_val,
        },
        ckpt_path,
    )

    return FoldResult(
        fold_pack=fold_pack,
        best_epoch=int(best["epoch"]),
        best_val_recon=float(best["val_recon"]),
        epochs_run=len(history),
        elapsed_s=time.time() - t0,
        history_path=history_path,
        ckpt_path=ckpt_path,
        history=history,
    )


def pick_device(pref: str = "auto") -> torch.device:
    """``auto`` → 有 CUDA 用 CUDA。"""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
