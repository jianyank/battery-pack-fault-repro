"""确定性基线重算：从已存 checkpoint 重算特征与 LOF，不训练。

存在的原因：推理期曾做变分采样，同一批窗口每次前向都得到不同特征（``docs/DEVIATIONS.md``
D4，已修复）。本脚本建立可复算的基线，并复用 ``run_all.py`` 的 ``_load_model`` 与
``_write_scores_csv``，保证与生产路径同源。

    python scripts/rebuild_baseline.py
    python scripts/rebuild_baseline.py --ckpt outputs/<ts>_paper_final/model_all/best.pt

产物落在 ``outputs/<timestamp>_paper_deterministic/``，结构与 ``run_all.py`` 一致。

口径边界：越限比例是描述性比例而非虚警率；窗口序号不是时间轴；本脚本只把"不可复算"
变成"可复算"，不主张检测性能提高。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402

from packrepro import paths, viz  # noqa: E402
from packrepro.config import Config  # noqa: E402
from packrepro.data import build_packs, load_experiment  # noqa: E402
from packrepro.detect import LOFDetector  # noqa: E402
from packrepro.evaluation import (  # noqa: E402
    compute_pack_features,
    persistence_variant,
    scalers_from_ckpt,
)
from packrepro.features import feature_names  # noqa: E402
from packrepro.model.train import pick_device  # noqa: E402

# 复用 run_all 的模型装配与 CSV 写出：宁可 import 私有名，也不要拷贝一份会漂移的实现
from run_all import _load_model, _write_scores_csv  # noqa: E402

DEFAULT_CKPT = "outputs/20261009T073945Z_paper_final/model_all/best.pt"
DEFAULT_REF = "outputs/20261009T073945Z_paper_final/detection_table.csv"


def _read_ref_table(path: Path) -> dict[tuple[str, int], dict]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {(r["dataset"], int(r["pack_id"])): r for r in csv.DictReader(fh)}


def _delta_vs_ref(rows: list[dict], ref: dict[tuple[str, int], dict]) -> dict:
    """逐行对比新表与旧表，量化缺陷的影响面。"""
    keys = ("threshold", "exceedance_rate", "calib_median_score", "confirmed_rate_K5")
    per_row: list[dict] = []
    worst = {k: 0.0 for k in keys}
    flipped = 0
    for row in rows:
        key = (row["dataset"], int(row["pack_id"]))
        old = ref.get(key)
        if old is None:
            continue
        deltas: dict[str, float] = {}
        for k in keys:
            a, b = row.get(k), old.get(k)
            if a is None or b in (None, ""):
                continue
            d = abs(float(a) - float(b))
            deltas[k] = d
            worst[k] = max(worst[k], d)
        old_conf = float(old["confirmed_rate_K5"] or 0.0)
        new_conf = float(row.get("confirmed_rate_K5") or 0.0)
        if (old_conf > 0.0) != (new_conf > 0.0):
            flipped += 1
        per_row.append({"dataset": row["dataset"], "pack_id": row["pack_id"], **deltas})
    return {"rows": per_row, "max_abs_delta": worst, "k5_zero_nonzero_flips": flipped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 checkpoint 重算确定性基线（不训练）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint 路径（相对项目根）")
    ap.add_argument("--ref-table", default=DEFAULT_REF, help="对比用的旧检测表")
    ap.add_argument("--tag", default="paper_deterministic")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config.load()
    root = paths.project_root()
    ckpt = (root / args.ckpt).resolve()
    if not ckpt.is_file():
        print(f"[FAIL] 找不到 checkpoint：{ckpt}")
        return 1

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths.make_run_dir(args.tag, timestamp=ts)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg.get("train.device"))
    q3p = cfg.get("features.q3_percentile")
    eps = cfg.get("features.eps")
    pv = cfg.get("detect.persistence_variants")
    q = cfg.get("detect.threshold_quantile")
    k = cfg.get("detect.k")
    std = cfg.get("detect.standardize")

    print(f"run_dir : {run_dir.relative_to(root)}")
    print(f"device  : {device}")
    print(f"ckpt    : {ckpt.relative_to(root)}")
    print(f"detect  : k={k} q={q} standardize={std}\n")

    t0 = time.time()
    model, blob = _load_model(cfg, ckpt, device)
    chs, vsc = scalers_from_ckpt(blob)

    def feats_of(pack) -> np.ndarray:
        return compute_pack_features(
            model, pack, ch_scaler=chs, volt_scaler=vsc,
            device=device, q3_percentile=q3p, eps=eps, deterministic=True,
        )

    # ------------------------------------------------------------ 读数据
    print("[1/3] 读取数据")
    train_packs = build_packs(load_experiment(cfg, "train"), cfg)
    print(f"  train 包 {sorted(train_packs)}")
    datasets: dict[str, dict] = {"train": train_packs}
    for name in ("test1", "test2"):
        datasets[name] = build_packs(load_experiment(cfg, name), cfg)
        print(f"  {name}  包 {sorted(datasets[name])}")

    # ------------------------------------------------------------ 确定性自检
    print("\n[2/3] 确定性自检 + 校准 + 检测")
    probe_pid = min(train_packs, key=lambda p: train_packs[p].n_windows)
    a = feats_of(train_packs[probe_pid])
    b = feats_of(train_packs[probe_pid])
    deterministic = bool(np.array_equal(a, b))
    max_probe_delta = float(np.abs(a - b).max())
    print(f"  同一包(pack {probe_pid})两次算特征：逐位相同 = {deterministic}"
          f"  max|Δ| = {max_probe_delta:.3e}")
    if not deterministic:
        print("[FAIL] 特征仍不可复算 —— 推理期抽样未真正关闭，终止。")
        return 1

    calib = np.concatenate(
        [feats_of(train_packs[p]) for p in cfg.get("experiment.train_packs")], axis=0
    )
    det = LOFDetector(
        k=k, threshold_quantile=q,
        persistence=cfg.get("detect.persistence_windows"), standardize=std,
    ).fit(calib)
    calib_summary = det.calib_score_summary()
    print(f"  校准集 {calib.shape[0]} 窗  θ={det.threshold_:.6f}"
          f"  中位数={calib_summary['median']:.6f}  p99={calib_summary['p99']:.6f}")

    rows: list[dict] = []
    feature_store: dict[str, np.ndarray] = {}
    paper_sets = {
        "train": None,
        "test1": set(cfg.get("experiment.test1_packs")),
        "test2": set(cfg.get("experiment.test2_packs")),
    }
    for name in ("train", "test1", "test2"):
        for pid in sorted(datasets[name]):
            feats = feats_of(datasets[name][pid])
            feature_store[f"{name}:{pid}"] = feats
            res = det.detect(feats)
            _write_scores_csv(run_dir / f"lof_scores_{name}_pack{pid}.csv",
                              pid, res.scores, res.threshold, res.confirmed)
            if not args.no_figures:
                viz.plot_lof_scores(res.scores, res.threshold, pack_id=pid,
                                    out=fig_dir / f"lof_{name}_pack{pid}.png",
                                    protocol=f"paper/{name}")
            paper_packs = paper_sets[name]
            row = {
                "dataset": name, "pack_id": pid, "n_windows": int(res.scores.size),
                "in_paper_test_set": None if paper_packs is None else (pid in paper_packs),
                "threshold": float(res.threshold), "k_used": int(res.k_used),
                "exceedance_rate": float(res.exceeded.mean()),
                "calib_median_score": calib_summary["median"],
                "calib_p99_score": calib_summary["p99"],
            }
            for kk in pv:
                _, conf = persistence_variant(res.scores, res.threshold, kk)
                row[f"confirmed_rate_K{kk}"] = float(conf.mean())
                first = int(np.argmax(conf)) if conf.any() else None
                row[f"first_alarm_frac_K{kk}"] = None if first is None else first / conf.size
            rows.append(row)
            print(f"  {name} pack {pid:2d}: {row['n_windows']:6d} 窗"
                  f"  越限={row['exceedance_rate']:7.2%}"
                  f"  确认K1={row['confirmed_rate_K1']:7.2%}"
                  f"  确认K5={row['confirmed_rate_K5']:7.2%}")

    # ------------------------------------------------------------ 落盘
    print("\n[3/3] 落盘")
    np.savez_compressed(
        run_dir / "features.npz",
        feature_names=np.array(feature_names(len(cfg.get("channels.cell_voltage")))),
        **feature_store,
    )
    cols = list(rows[0].keys())
    with (run_dir / "detection_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    if not args.no_figures:
        viz.plot_exceedance_bars(
            rows, out=fig_dir / "exceedance_by_pack.png", protocol="paper",
            confirm_key=f"confirmed_rate_K{pv[0]}",
        )

    delta = _delta_vs_ref(rows, _read_ref_table((root / args.ref_table).resolve()))
    if delta["rows"]:
        print(f"  与 {args.ref_table} 的偏差：")
        for kk, v in delta["max_abs_delta"].items():
            print(f"    max|Δ {kk}| = {v:.3e}")
        print(f"    K=5 由零变非零（或反之）的行数 = {delta['k5_zero_nonzero_flips']}")

    summary = {
        "protocol": "paper",
        "run_dir": run_dir.name,
        "device": str(device),
        "checkpoint": str(ckpt.relative_to(root)),
        "reference_table": args.ref_table,
        "deterministic_inference": True,
        "determinism_probe": {
            "pack_id": int(probe_pid),
            "bitwise_identical": deterministic,
            "max_abs_delta": max_probe_delta,
        },
        "assumed_items": cfg.assumed_items(),
        "calib": calib_summary,
        "test2_packs_missing": cfg.get("experiment.test2_packs_missing"),
        "time_axis": {"train": "absent",
                      "note": "窗口序号不是时间轴；所有以天数为单位的结论均不可复现"},
        "delta_vs_reference": delta,
        "rows": rows,
        "elapsed_total_s": time.time() - t0,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n完成。产物：{run_dir.relative_to(root)}")
    print("  口径提醒：越限比例是描述性比例，**不是**虚警率；窗口序号不是时间。")
    print("  本脚本只把「不可复算」变成「可复算」，不主张检测性能提高。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
