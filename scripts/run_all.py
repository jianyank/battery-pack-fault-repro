"""端到端复现驱动：数据 → 训练 → 指标 → LOF → 图表。

用法::

    python scripts/run_all.py                       # 用 configs/base.yaml，协议取其中的 protocol
    python scripts/run_all.py --protocol paper      # 论文 §4.1 原样
    python scripts/run_all.py --protocol lopo       # 跨包留出（更严格的泛化测试）
    python scripts/run_all.py --max-epochs 5        # 冒烟用，快速跑通全链路

产物落在 ``outputs/<timestamp>_<protocol>/``：history、checkpoint、特征、LOF 分数、
检测对照表、图表、summary.json。

口径边界（必须随结果一并声明）
------------------------------
* 不报提前量天数（本地无投运日期，窗口序号不是时间轴）。
* 不报真实虚警率 / 召回率 / F1（无独立健康段与逐窗故障真值）。
* 不报单体定位准确率（Hiddall 与对齐目标同源，非独立真值）。
* 论文 Test dataset 2 的 pack 12、14 **本地无数据**，表中显式标注为 missing。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from packrepro import paths, viz  # noqa: E402
from packrepro.config import Config  # noqa: E402
from packrepro.data import build_packs, load_experiment, train_val_split  # noqa: E402
from packrepro.data.normalize import ChannelScaler  # noqa: E402
from packrepro.detect import LOFDetector  # noqa: E402
from packrepro.evaluation import (  # noqa: E402
    compute_pack_features,
    persistence_variant,
    reconstruct,
)
from packrepro.model import AsymmetricAE  # noqa: E402
from packrepro.model.train import pick_device, train_fold  # noqa: E402


def _load_model(cfg: Config, ckpt: Path, device: torch.device) -> tuple[AsymmetricAE, dict]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = AsymmetricAE(
        time_len=cfg.get("windows.length"),
        in_channels=sum(
            len(cfg.get(f"channels.{k}"))
            for k in ("cell_voltage", "current", "temperature")
        ),
        out_channels=len(cfg.get("channels.cell_voltage")),
        latent_dim=cfg.get("model.latent_dim"),
        variational=cfg.get("model.variational"),
    )
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, blob


def _volt_scaler_from_ckpt(blob: dict) -> ChannelScaler:
    return ChannelScaler(center=blob["volt_center"], scale=blob["volt_scale"], method="robust")


def _ch_scaler_from_ckpt(blob: dict) -> ChannelScaler:
    return ChannelScaler(center=blob["ch_center"], scale=blob["ch_scale"], method="robust")


def _write_scores_csv(path: Path, pack_id: int, scores, threshold, confirmed) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pack_id", "window_index_not_time", "lof_score", "threshold", "confirmed"])
        for i, s in enumerate(scores):
            w.writerow([pack_id, i, f"{s:.10g}", f"{threshold:.10g}", int(bool(confirmed[i]))])


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Battery_Pack_Fault_Manuscript 复现")
    ap.add_argument("--config", default=None)
    ap.add_argument("--protocol", choices=["paper", "lopo"], default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    protocol = args.protocol or cfg.get("experiment.protocol")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or f"{protocol}"
    run_dir = paths.make_run_dir(tag, timestamp=stamp)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg.get("train.device"))
    print(f"run_dir : {run_dir.relative_to(paths.project_root())}")
    print(f"device  : {device}")
    print(f"protocol: {protocol}\n")

    q3p = cfg.get("features.q3_percentile")
    eps = cfg.get("features.eps")
    pv = cfg.get("detect.persistence_variants")

    rows: list[dict] = []
    feature_store: dict[str, np.ndarray] = {}
    summary: dict = {
        "protocol": protocol,
        "run_dir": run_dir.name,
        "device": str(device),
        "assumed_items": cfg.assumed_items(),
        "test2_packs_missing": cfg.get("experiment.test2_packs_missing"),
        "packs": {},
    }

    # ---------------------------------------------------------------- 训练
    print("[1/3] 读取训练数据并训练")
    t0 = time.time()
    train_bundle = load_experiment(cfg, "train")
    train_packs = build_packs(train_bundle, cfg)
    print(f"  train 包: {sorted(train_packs)}  共 {train_bundle.n_windows} 窗"
          f"  ({time.time() - t0:.1f}s)")
    summary["time_axis"] = {
        "train": train_bundle.time_note,
        "note": "窗口序号不是时间轴；所有以天数为单位的结论均不可复现",
    }

    if protocol == "paper":
        folds = [None]  # 单一模型，用全部训练包
    else:
        folds = list(cfg.get("experiment.folds"))

    trained: dict[int, dict] = {}
    for fold in folds:
        key = fold if fold is not None else 0
        if fold is None:
            use_packs = [train_packs[p] for p in cfg.get("experiment.train_packs")]
            label = "all"
        else:
            use_packs = [train_packs[p] for p in sorted(train_packs) if p != fold]
            label = f"holdout{fold}"
        print(f"  训练 [{label}] 用包 {[p.pack_id for p in use_packs]}")
        res = train_fold(
            cfg, fold_pack=key, train_packs=use_packs,
            out_dir=run_dir / f"model_{label}", device=device,
            max_epochs=args.max_epochs,
        )
        model, blob = _load_model(cfg, res.ckpt_path, device)
        trained[key] = {
            "model": model, "blob": blob, "result": res, "label": label,
            "train_pack_ids": [p.pack_id for p in use_packs],
        }
        print(f"    best_epoch={res.best_epoch}  val_recon={res.best_val_recon:.6f}"
              f"  epochs={res.epochs_run}  {res.elapsed_s:.1f}s")
        viz.plot_history(res.history_path, fig_dir / f"history_{label}.png",
                         title=f"training history [{label}] protocol={protocol}")

    summary["train_seconds"] = time.time() - t0

    # ---------------------------------------------------------------- 校准 + 检测
    print("\n[2/3] 计算特征、拟合 LOF、检测")

    def detect_with(model_key: int, calib_features, target_pack, tag_name: str):
        d = trained[model_key]
        model, blob = d["model"], d["blob"]
        feats = compute_pack_features(
            model, target_pack,
            ch_scaler=_ch_scaler_from_ckpt(blob),
            volt_scaler=_volt_scaler_from_ckpt(blob),
            device=device, q3_percentile=q3p, eps=eps,
        )
        det = LOFDetector(
            k=cfg.get("detect.k"),
            threshold_quantile=cfg.get("detect.threshold_quantile"),
            persistence=1,
            standardize=cfg.get("detect.standardize"),
        ).fit(calib_features)
        res = det.detect(feats)
        return feats, res, det

    # 记录每个模型在其训练包上的特征（作为 LOF 校准集）
    calib_cache: dict[int, np.ndarray] = {}

    def calib_features_for(model_key: int, pack_ids: list[int]) -> np.ndarray:
        if model_key in calib_cache:
            return calib_cache[model_key]
        d = trained[model_key]
        blob = d["blob"]
        chs = _ch_scaler_from_ckpt(blob)
        vsc = _volt_scaler_from_ckpt(blob)
        parts = []
        for pid in pack_ids:
            parts.append(compute_pack_features(
                d["model"], train_packs[pid],
                ch_scaler=chs, volt_scaler=vsc, device=device,
                q3_percentile=q3p, eps=eps,
            ))
        arr = np.concatenate(parts, axis=0)
        calib_cache[model_key] = arr
        return arr

    def evaluate_targets(calib: np.ndarray, packs_dict: dict, dataset_name: str,
                         model_key: int, paper_packs: set[int] | None = None):
        """对数据集内**全部**包评分。

        ``paper_packs`` 标出其中哪些属于论文的对应测试集；其余包本地有、论文未用，
        仍然评分并出表，但用 ``in_paper_test_set`` 列区分，避免被误当论文结果引用。
        """
        for pid in sorted(packs_dict):
            pack = packs_dict[pid]
            feats, res, det = detect_with(model_key, calib, pack, f"{dataset_name}_p{pid}")
            feature_store[f"{dataset_name}:{pid}"] = feats
            _write_scores_csv(run_dir / f"lof_scores_{dataset_name}_pack{pid}.csv",
                              pid, res.scores, res.threshold, res.confirmed)
            viz.plot_lof_scores(res.scores, res.threshold, pack_id=pid,
                                out=fig_dir / f"lof_{dataset_name}_pack{pid}.png",
                                protocol=f"{protocol}/{dataset_name}")
            cal = det.calib_score_summary()
            row = {
                "dataset": dataset_name, "pack_id": pid, "n_windows": int(res.scores.size),
                "in_paper_test_set": None if paper_packs is None else (pid in paper_packs),
                "threshold": float(res.threshold), "k_used": int(res.k_used),
                "exceedance_rate": float(res.exceeded.mean()),
                "calib_median_score": cal["median"], "calib_p99_score": cal["p99"],
            }
            for k in pv:
                _, conf = persistence_variant(res.scores, res.threshold, k)
                row[f"confirmed_rate_K{k}"] = float(conf.mean())
                first = int(np.argmax(conf)) if conf.any() else None
                row[f"first_alarm_frac_K{k}"] = None if first is None else first / conf.size
            rows.append(row)
            summary["packs"][f"{dataset_name}:{pid}"] = {
                "n_windows": row["n_windows"], "threshold": row["threshold"],
                "exceedance_rate": row["exceedance_rate"],
                "calib_median_score": cal["median"],
            }
            print(f"  {dataset_name} pack {pid:2d}: {row['n_windows']:6d} 窗  θ={row['threshold']:.4f}"
                  f"  越限={row['exceedance_rate']:.2%}"
                  f"  确认(K=1)={row['confirmed_rate_K1']:.2%}"
                  f"  校准中位数={cal['median']:.3f}")

    if protocol == "paper":
        mkey = 0
        calib = calib_features_for(mkey, cfg.get("experiment.train_packs"))
        # train 数据集本身就是校准集，不属于任何"测试集"
        evaluate_targets(calib, train_packs, "train", mkey, paper_packs=None)
        test1 = build_packs(load_experiment(cfg, "test1"), cfg)
        evaluate_targets(calib, test1, "test1", mkey,
                         paper_packs=set(cfg.get("experiment.test1_packs")))
        test2 = build_packs(load_experiment(cfg, "test2"), cfg)
        evaluate_targets(calib, test2, "test2", mkey,
                         paper_packs=set(cfg.get("experiment.test2_packs")))
    else:
        for fold in sorted(trained):
            d = trained[fold]
            others = [p for p in sorted(train_packs) if p != fold]
            calib = calib_features_for(fold, others)
            evaluate_targets(calib, {fold: train_packs[fold]}, "holdout", fold)

    # ---------------------------------------------------------------- 落盘
    print("\n[3/3] 汇总与出图")
    if feature_store:
        from packrepro.features import feature_names

        np.savez_compressed(
            run_dir / "features.npz",
            feature_names=np.array(feature_names(len(cfg.get("channels.cell_voltage")))),
            **feature_store,
        )

    cols = list(rows[0].keys()) if rows else []
    with (run_dir / "detection_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    if rows:
        viz.plot_exceedance_bars(
            rows, out=fig_dir / "exceedance_by_pack.png", protocol=protocol,
            confirm_key=f"confirmed_rate_K{pv[0]}",
        )
    summary["rows"] = rows
    summary["elapsed_total_s"] = time.time() - t0
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n完成。产物：{run_dir.relative_to(paths.project_root())}")
    print("  口径提醒：越限比例是描述性比例，**不是**虚警率；窗口序号不是时间。")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
