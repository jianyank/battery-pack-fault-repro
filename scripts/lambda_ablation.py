"""λ 消融：三项损失的权重各关一次，看对重建与检出的影响。

变体（其余权重同 ``base.yaml``：λ₁=1.0）
----------------------------------------
* ``no_late``        λ₂=0   —— 去掉物理对齐项（潜变量不再约束到内阻/SOC）
* ``no_kl``          λ₃=0   —— 去掉 KL 正则
* ``no_late_no_kl``  λ₂=λ₃=0 —— 退化为纯重建自编码器

对照组是 ``full``（λ₂=0.1、λ₃=1e-3）。**不需要重跑**：本机训练逐位可复现
（同 seed 两次训练的权重与 history 完全相同，已实测），故现有的 full-λ 模型与
一次全新的 full 训练等价；其确定性评分见 ``outputs/*_paper_deterministic/``。

口径边界：各变体的检出差异是**描述性比例**，不构成"哪个 λ 更好"的结论 ——
无独立健康段与故障真值，报警变多既可能是检出变多、也可能是误报变多。

    python scripts/lambda_ablation.py
"""

from __future__ import annotations

import argparse
import copy
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
import yaml  # noqa: E402

from packrepro import paths  # noqa: E402

import run_all  # noqa: E402  直接调用其 run()，避免重复 CLI 拼装

VARIANTS = [
    ("no_late", {"lambda2": 0.0}),
    ("no_kl", {"lambda3": 0.0}),
    ("no_late_no_kl", {"lambda2": 0.0, "lambda3": 0.0}),
]


def _best_from_history(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    best = min(rows, key=lambda r: float(r["val_recon"]))
    return {"epochs_run": len(rows), "best_epoch": int(best["epoch"]),
            "best_val_recon": float(best["val_recon"])}


def _detect_summary(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: dict = {}
    for ds in ("train", "test1"):
        sub = [r for r in rows if r["dataset"] == ds]
        out[f"{ds}_exceedance"] = float(np.mean([float(r["exceedance_rate"]) for r in sub]))
        out[f"{ds}_K5"] = float(np.mean([float(r["confirmed_rate_K5"] or 0.0) for r in sub]))
    for pid in (2, 5):
        sub = [r for r in rows if r["dataset"] == "test2" and int(r["pack_id"]) == pid]
        out[f"test2:{pid}_K5"] = float(sub[0]["confirmed_rate_K5"] or 0.0) if sub else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="λ 消融（3 个变体，重训）")
    ap.add_argument("--max-epochs", type=int, default=None)
    args = ap.parse_args(argv)

    root = paths.project_root()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = paths.make_run_dir("lambda_ablation", timestamp=ts)
    (out_root / "configs").mkdir()

    base = yaml.safe_load((root / "configs" / "base.yaml").open(encoding="utf-8"))
    lam = base["train"]["loss"]
    print(f"对照 full: λ2={lam['lambda2']} λ3={lam['lambda3']}\n")

    results: dict[str, dict] = {}
    t0 = time.time()
    for name, over in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["train"]["loss"].update(over)
        cfg["experiment"]["protocol"] = "paper"
        cpath = out_root / "configs" / f"{name}.yaml"
        with cpath.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)

        print(f"\n{'=' * 70}\n[{name}] λ2={cfg['train']['loss']['lambda2']} "
              f"λ3={cfg['train']['loss']['lambda3']}\n{'=' * 70}")
        rc = run_all.run(["--config", str(cpath), "--protocol", "paper",
                          "--tag", f"abl_{name}",
                          *(["--max-epochs", str(args.max_epochs)] if args.max_epochs else [])])
        if rc != 0:
            print(f"[FAIL] 变体 {name} 返回 {rc}，终止。")
            return rc

        # 找到刚写出的运行目录（按 tag 匹配，取最新）
        cands = sorted(root.glob(f"outputs/*_abl_{name}"))
        run_dir = cands[-1]
        results[name] = {
            "run_dir": run_dir.name,
            "lambda2": cfg["train"]["loss"]["lambda2"],
            "lambda3": cfg["train"]["loss"]["lambda3"],
            **_best_from_history(run_dir / "model_all" / "history.csv"),
            **_detect_summary(run_dir / "detection_table.csv"),
        }
        print(f"  -> {run_dir.name}")

    # 对照：现有 full-λ 模型（训练可复现，故等价于重跑一次 full）
    det_dirs = sorted(root.glob("outputs/*_paper_deterministic"))
    full_row = None
    if det_dirs:
        full_row = {
            "run_dir": det_dirs[-1].name,
            "lambda2": lam["lambda2"], "lambda3": lam["lambda3"],
            **_best_from_history(sorted(root.glob("outputs/*_paper_final/model_all/history.csv"))[-1]),
            **_detect_summary(det_dirs[-1] / "detection_table.csv"),
        }

    print(f"\n{'=' * 70}\n汇总\n{'=' * 70}")
    cols = ["run_dir", "lambda2", "lambda3", "best_epoch", "best_val_recon",
            "train_exceedance", "train_K5", "test1_exceedance", "test1_K5",
            "test2:2_K5", "test2:5_K5"]
    print(f"{'variant':<16}{'best_ep':>8}{'val_recon':>12}{'trainK5':>10}"
          f"{'test1越限':>11}{'test1K5':>10}{'p2K5':>8}{'p5K5':>8}")
    if full_row:
        print(f"{'full(对照)':<16}{full_row['best_epoch']:>8}{full_row['best_val_recon']:>12.6f}"
              f"{full_row['train_K5']:>10.2%}{full_row['test1_exceedance']:>11.2%}"
              f"{full_row['test1_K5']:>10.2%}"
              f"{(full_row['test2:2_K5'] or 0):>8.2%}{(full_row['test2:5_K5'] or 0):>8.2%}")
    for name, _ in VARIANTS:
        r = results[name]
        print(f"{name:<16}{r['best_epoch']:>8}{r['best_val_recon']:>12.6f}"
              f"{r['train_K5']:>10.2%}{r['test1_exceedance']:>11.2%}{r['test1_K5']:>10.2%}"
              f"{(r['test2:2_K5'] or 0):>8.2%}{(r['test2:5_K5'] or 0):>8.2%}")

    with (out_root / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        if full_row:
            w.writerow({c: full_row.get(c) for c in cols})
        for name, _ in VARIANTS:
            w.writerow({c: results[name].get(c) for c in cols})

    (out_root / "summary.json").write_text(json.dumps({
        "control_full": full_row,
        "variants": results,
        "note": "对照 full 未重训：本机训练逐位可复现，故现有 full-λ 模型等价于重跑一次 full",
        "caveats": [
            "越限比例是描述性比例，不是虚警率（无独立健康段与逐窗故障真值）",
            "本表不构成『哪个 λ 更好』的结论：报警变多既可能是检出变多、也可能是误报变多",
            "λ 的绝对值与论文不可比（论文未给 λ；式 (2) 的求和/平均约定也不同，见 DEVIATIONS C2）",
        ],
        "elapsed_total_s": time.time() - t0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成。产物：{out_root.relative_to(root)}  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
