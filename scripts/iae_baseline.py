"""IAE 基线（论文 §4.3 的对照方法之一）—— 复用同一模型，换判据，不训练。

论文正文对 IAE 的限定
--------------------
* "a method utilizing an **IAE** in conjunction with a **3σ threshold criterion**"
* "The autoencoder structure used in IAE method was **kept identical to that of the
  proposed method**" —— 故本脚本复用同一个 ``best.pt``，不另训模型
* 阈值："a **reconstruction error of 0.212 V** for IAE method"

源方法（Zhang et al., *Applied Energy* 330:120312, 2023）以**重构电压与原电压的 RMSE**
判定异常。故本脚本的统计量取物理电压域的 RMSE，阈值按论文的 3σ 判据在正常训练数据上拟合。

与论文的不可比之处（必须随结果声明）
------------------------------------
* 论文 Table 3/4 比的是**报警天数**（投运日起算）。本地无投运日期、时间轴不可解码，
  **本脚本不产生报警天数**，只给出逐包越限率与确认率。
* 论文的 0.212 V 是在**它的**模型上拟合的；本项目模型的误差尺度不同，故阈值按本项目的
  正常训练数据自行拟合 3σ，并把 0.212 V 仅作量级参照一并印出。
* 越限比例是**描述性比例，不是虚警率**。

    python scripts/iae_baseline.py
    python scripts/iae_baseline.py --ckpt outputs/<ts>/model_all/best.pt
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

from packrepro import paths  # noqa: E402
from packrepro.config import Config  # noqa: E402
from packrepro.data import build_packs, load_experiment  # noqa: E402
from packrepro.evaluation import persistence_variant, reconstruct, scalers_from_ckpt  # noqa: E402
from packrepro.model.train import pick_device  # noqa: E402

from run_all import _load_model  # noqa: E402  复用同一份模型装配，避免两处漂移

DEFAULT_CKPT = "outputs/20261009T073945Z_paper_final/model_all/best.pt"
#: 论文 §4.3 给出的 IAE 阈值，仅作量级参照（见模块 docstring）
PAPER_IAE_THRESHOLD_V = 0.212


def _voltage_rmse(model, pack, chs, vsc, device) -> dict[str, np.ndarray]:
    """逐窗的重构误差（物理电压域），两种统计量。"""
    x_norm = chs.transform(pack.full)
    recon_norm = reconstruct(model, x_norm, device=device, deterministic=True)
    meas_v = vsc.inverse_transform(pack.voltage.astype(np.float64))
    recon_v = vsc.inverse_transform(recon_norm.astype(np.float64))
    err = recon_v - meas_v                                   # (N, T, C)

    rmse_pack = np.sqrt(np.mean(err ** 2, axis=(1, 2)))       # 全部 (T, C) 上的 RMSE
    rmse_cell = np.sqrt(np.mean(err ** 2, axis=1))            # (N, C) 逐单体 RMSE
    return {"rmse_pack": rmse_pack, "rmse_max_cell": rmse_cell.max(axis=1)}


def _threesigma(x: np.ndarray) -> tuple[float, float, float]:
    """论文所述的 3σ 判据：阈值 = 正常数据的 mean + 3·std。"""
    mu, sd = float(np.mean(x)), float(np.std(x))
    return mu, sd, mu + 3.0 * sd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="IAE 基线（复用同一模型，不训练）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--ref-lof", default=None, help="本项目 LOF 分支的 detection_table.csv")
    ap.add_argument("--tag", default="iae_baseline")
    args = ap.parse_args(argv)

    cfg = Config.load()
    root = paths.project_root()
    ckpt = (root / args.ckpt).resolve()
    if not ckpt.is_file():
        print(f"[FAIL] 找不到 checkpoint：{ckpt}")
        return 1

    ref_lof = (root / args.ref_lof).resolve() if args.ref_lof else None
    if ref_lof is None:
        cands = sorted(root.glob("outputs/*_paper_deterministic"))
        ref_lof = (cands[-1] / "detection_table.csv") if cands else None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = paths.make_run_dir(args.tag, timestamp=ts)
    device = pick_device(cfg.get("train.device"))
    pv = cfg.get("detect.persistence_variants")
    print(f"run_dir : {run_dir.relative_to(root)}")
    print(f"device  : {device}")
    print(f"ckpt    : {ckpt.relative_to(root)}\n")

    t0 = time.time()
    model, blob = _load_model(cfg, ckpt, device)
    chs, vsc = scalers_from_ckpt(blob)

    print("[1/3] 读取数据")
    train_packs = build_packs(load_experiment(cfg, "train"), cfg)
    datasets = {"train": train_packs}
    for name in ("test1", "test2"):
        datasets[name] = build_packs(load_experiment(cfg, name), cfg)
        print(f"  {name} 包 {sorted(datasets[name])}")

    # ------------------------------------------------------------ 确定性自检
    print("\n[2/3] 阈值拟合（3σ，正常训练数据）")
    probe = min(train_packs, key=lambda p: train_packs[p].n_windows)
    a = _voltage_rmse(model, train_packs[probe], chs, vsc, device)["rmse_pack"]
    b = _voltage_rmse(model, train_packs[probe], chs, vsc, device)["rmse_pack"]
    identical = bool(np.array_equal(a, b))
    print(f"  两次算同一包：逐位相同 = {identical}")
    if not identical:
        print("[FAIL] 结果不可复算，终止。")
        return 1

    # 校准：全部训练包（论文的"正常训练数据"）
    calib = {k: [] for k in ("rmse_pack", "rmse_max_cell")}
    for pid in sorted(train_packs):
        for k, v in _voltage_rmse(model, train_packs[pid], chs, vsc, device).items():
            calib[k].append(v)
    calib = {k: np.concatenate(v) for k, v in calib.items()}
    thresh = {}
    for k, v in calib.items():
        mu, sd, thr = _threesigma(v)
        thresh[k] = thr
        print(f"  {k:<14} n={v.size}  μ={mu:.4f} σ={sd:.4f}  3σ阈值={thr:.4f} V"
              f"  中位数={np.median(v):.4f}")
    print(f"  参照：论文 §4.3 的 IAE 阈值 = {PAPER_IAE_THRESHOLD_V} V（在它的模型上拟合）")

    # ------------------------------------------------------------ 逐包检测
    rows: list[dict] = []
    paper_sets = {"train": None, "test1": set(cfg.get("experiment.test1_packs")),
                  "test2": set(cfg.get("experiment.test2_packs"))}
    for name in ("train", "test1", "test2"):
        for pid in sorted(datasets[name]):
            st = _voltage_rmse(model, datasets[name][pid], chs, vsc, device)
            row = {"dataset": name, "pack_id": pid, "n_windows": int(st["rmse_pack"].size),
                   "in_paper_test_set": None if paper_sets[name] is None
                   else (pid in paper_sets[name])}
            for k in ("rmse_pack", "rmse_max_cell"):
                sc = st[k]
                row[f"{k}_median"] = float(np.median(sc))
                # 两套阈值都报：本项目按 3σ 拟合的，以及论文自己的 0.212 V。
                # 后者用于回答"IAE 的沉默是不是本项目阈值偏严造成的"。
                for label, thr in (("", thresh[k]), ("_at_paper_thr", PAPER_IAE_THRESHOLD_V)):
                    row[f"{k}_exceedance_rate{label}"] = float((sc > thr).mean())
                    _, conf5 = persistence_variant(sc, thr, 5)
                    row[f"{k}_confirmed_rate_K5{label}"] = float(conf5.mean())
            rows.append(row)
            print(f"  {name} pack {pid:2d}: {row['n_windows']:6d} 窗"
                  f"  rmse_pack 越限={row['rmse_pack_exceedance_rate']:7.2%}"
                  f"  确认K5={row['rmse_pack_confirmed_rate_K5']:7.2%}")

    # ------------------------------------------------------------ 与本项目 LOF 分支对照
    comparison = None
    if ref_lof and ref_lof.is_file():
        with ref_lof.open(encoding="utf-8") as fh:
            lof = {(r["dataset"], int(r["pack_id"])): r for r in csv.DictReader(fh)}
        comparison = []
        for row in rows:
            old = lof.get((row["dataset"], row["pack_id"]))
            if old is None:
                continue
            comparison.append({
                "dataset": row["dataset"], "pack_id": row["pack_id"],
                "lof_exceedance": float(old["exceedance_rate"]),
                "lof_confirmed_K5": float(old["confirmed_rate_K5"] or 0.0),
                "iae_exceedance": row["rmse_pack_exceedance_rate"],
                "iae_confirmed_K5": row["rmse_pack_confirmed_rate_K5"],
            })

    # ------------------------------------------------------------ 落盘
    print("\n[3/3] 落盘")
    cols = list(rows[0].keys())
    with (run_dir / "iae_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "baseline": "IAE",
        "run_dir": run_dir.name,
        "checkpoint": str(ckpt.relative_to(root)),
        "device": str(device),
        "statistic": "物理电压域 RMSE（论文：重构电压与原电压的 RMSE）",
        "threshold_rule": "正常训练数据的 mean + 3σ（论文：3σ threshold criterion）",
        "fitted_thresholds_V": thresh,
        "paper_iae_threshold_V": PAPER_IAE_THRESHOLD_V,
        "calib_summary": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                              "median": float(np.median(v))} for k, v in calib.items()},
        "reference_lof_table": None if ref_lof is None else str(ref_lof.relative_to(root)),
        "vs_lof": comparison,
        "rows": rows,
        "caveats": [
            "越限比例是描述性比例，不是虚警率（无独立健康段与逐窗故障真值）",
            "无投运日期与可解码时间轴，**不产生报警天数**，与论文 Table 3/4 不可逐日对照",
            "0.212 V 是论文模型的阈值；本项目阈值在自身正常数据上重新拟合，两者不可直接比数值",
            "rmse_max_cell 是本项目的逐单体变体，**不是**论文的 IAE 统计量",
        ],
        "elapsed_total_s": time.time() - t0,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if comparison:
        le = np.mean([c["lof_exceedance"] for c in comparison])
        ie = np.mean([c["iae_exceedance"] for c in comparison])
        lc = np.mean([c["lof_confirmed_K5"] for c in comparison])
        ic = np.mean([c["iae_confirmed_K5"] for c in comparison])
        print(f"\n与本项目 LOF 分支对照（13 包宏平均）：")
        print(f"  越限率   LOF {le:.2%}  vs  IAE {ie:.2%}")
        print(f"  K5 确认  LOF {lc:.2%}  vs  IAE {ic:.2%}")

    print(f"\n完成。产物：{run_dir.relative_to(root)}")
    print("  口径提醒：越限比例是描述性比例，不是虚警率；本脚本不产生报警天数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
