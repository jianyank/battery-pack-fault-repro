"""检测侧单因素敏感性扫描 —— 只读，不重训。

从 ``rebuild_baseline.py`` 冻结的 ``features.npz`` 重算 LOF，逐次只改一个因素，
刻画结果对论文未给项的依赖程度（默认 k=20、q=0.99、standardize=true、全 24 维）。

四组因素：
F1 ``standardize``（A12）—— 开启后各维同权。逐维 std：``sigma_v`` ≈2e-4（最小）、
``cosine`` ≈3.7e-2、``q3`` ≈8e-2（最大）；不标准化时欧氏距离被 ``q3`` 主导，
而 ``q3`` 恰是最不敏感的一族（test1 K5 仅 2.13%），故关掉标准化会显著变差；
F2 ``k``（A4）—— base.yaml 注释承诺过扫描但从未执行；
F3 特征族子集 —— §2.3 给了三族但未说明各自贡献，D3 已论证 cosine 族判别力弱；
F4 ``threshold_quantile`` —— 验证"99 分位 ⇒ 校准集自身必然约 1% 越限"。

这是敏感性刻画，不是调参择优：不得据"哪个设置能检出 pack 2 / pack 5"来选默认值 ——
那是在无标签数据上择优，为本项目明令禁止的做法（``docs/REPRODUCTION_NOTES.md`` §5）。
本脚本不产生"某设置更好"的结论，只回答"结果对哪一项最敏感"。越限率是描述性比例，
不是虚警率。

    python scripts/lof_scan.py --run-dir outputs/<ts>_paper_deterministic
    python scripts/lof_scan.py --only default        # 只跑一个组合（用于计时）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from packrepro import paths, viz  # noqa: E402
from packrepro.config import Config  # noqa: E402
from packrepro.detect import LOFDetector  # noqa: E402

#: (标签, 覆盖项)。覆盖项只含要改的键；未列出的键取 base.yaml 默认。
COMBOS: list[tuple[str, dict]] = [
    ("default", {}),
    ("standardize=False", {"standardize": False}),
    ("k=5", {"k": 5}),
    ("k=10", {"k": 10}),
    ("k=50", {"k": 50}),
    ("k=100", {"k": 100}),
    ("family=sigma_v", {"families": ("sigma_v",)}),
    ("family=cosine", {"families": ("cosine",)}),
    ("family=q3", {"families": ("q3",)}),
    ("family=sigma_v+q3", {"families": ("sigma_v", "q3")}),
    ("q=0.999", {"q": 0.999}),
]


def _family_mask(names: list[str], families: tuple[str, ...] | None) -> np.ndarray:
    """按 ``sigma_v_*`` / ``cosine_*`` / ``q3_*`` 前缀取列（不硬编码下标）。"""
    if families is None:
        return np.ones(len(names), dtype=bool)
    return np.array([any(n.startswith(f + "_") for f in families) for n in names])


def _confirmed_k(scores: np.ndarray, thr: float, k: int) -> np.ndarray:
    ex = scores > thr
    out = np.zeros_like(ex, dtype=bool)
    run = 0
    for i, f in enumerate(ex):
        run = run + 1 if f else 0
        out[i] = run >= k
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LOF 单因素敏感性扫描（只读）")
    ap.add_argument("--run-dir", default=None,
                    help="含 features.npz 的运行目录；默认取最新的 *_paper_deterministic")
    ap.add_argument("--ref", default=None, help="对齐用的 detection_table.csv；默认同目录")
    ap.add_argument("--only", default=None, help="只跑指定标签的组合")
    ap.add_argument("--tag", default="lof_scan")
    args = ap.parse_args(argv)

    cfg = Config.load()
    root = paths.project_root()

    if args.run_dir:
        src = (root / args.run_dir).resolve()
    else:
        cands = sorted(root.glob("outputs/*_paper_deterministic"))
        if not cands:
            print("[FAIL] 找不到 *_paper_deterministic 目录，先跑 rebuild_baseline.py")
            return 1
        src = cands[-1]
    npz_path = src / "features.npz"
    if not npz_path.is_file():
        print(f"[FAIL] 找不到 {npz_path}")
        return 1

    d = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    keys = [k for k in d.keys() if k != "feature_names"]
    by_ds: dict[str, list[tuple[int, np.ndarray]]] = {}
    for key in keys:
        ds, pid = key.split(":")
        by_ds.setdefault(ds, []).append((int(pid), d[key]))
    for ds in by_ds:
        by_ds[ds].sort()

    train_packs = cfg.get("experiment.train_packs")
    calib_all = np.concatenate([d[f"train:{p}"] for p in train_packs], axis=0)

    k_def = cfg.get("detect.k")
    q_def = cfg.get("detect.threshold_quantile")
    std_def = cfg.get("detect.standardize")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = paths.make_run_dir(args.tag, timestamp=ts)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"源特征 : {src.relative_to(root)}/features.npz  ({calib_all.shape[0]} 校准窗)")
    print(f"产物   : {run_dir.relative_to(root)}")
    print(f"默认   : k={k_def} q={q_def} standardize={std_def} families=all24\n")

    combos = COMBOS
    if args.only:
        combos = [c for c in COMBOS if c[0] == args.only] or []
        if not combos:
            print(f"[FAIL] 未知组合标签 {args.only!r}；可选：{[c[0] for c in COMBOS]}")
            return 1

    rows: list[dict] = []
    for label, ov in combos:
        t0 = time.time()
        k = ov.get("k", k_def)
        q = ov.get("q", q_def)
        std = ov.get("standardize", std_def)
        fams = ov.get("families")
        mask = _family_mask(names, fams)
        fam_label = "all24" if fams is None else "+".join(fams)

        calib = calib_all[:, mask]
        det = LOFDetector(
            k=k, threshold_quantile=q,
            persistence=cfg.get("detect.persistence_windows"), standardize=std,
        ).fit(calib)
        cal = det.calib_score_summary()

        for ds, packs in by_ds.items():
            for pid, X in packs:
                res = det.detect(X[:, mask])
                conf5 = _confirmed_k(res.scores, res.threshold, 5)
                rows.append({
                    "combo": label, "standardize": std, "k": k, "q": q,
                    "families": fam_label, "n_features": int(mask.sum()),
                    "dataset": ds, "pack_id": pid, "n_windows": int(res.scores.size),
                    "threshold": float(res.threshold),
                    "calib_median": cal["median"], "calib_p99": cal["p99"],
                    "exceedance_rate": float(res.exceeded.mean()),
                    "confirmed_rate_K5": float(conf5.mean()),
                })
        macro = [r for r in rows if r["combo"] == label]
        ex = np.mean([r["exceedance_rate"] for r in macro])
        c5 = np.mean([r["confirmed_rate_K5"] for r in macro])
        print(f"  [{label:<18}] D={int(mask.sum()):2d}  θ={det.threshold_:7.4f}"
              f"  宏平均越限={ex:7.2%}  宏平均K5={c5:7.2%}  ({time.time() - t0:.1f}s)")

    # ---------------------------------------------------------------- 自检
    ref_path = (root / args.ref).resolve() if args.ref else src / "detection_table.csv"
    if ref_path.is_file():
        with ref_path.open(encoding="utf-8") as fh:
            ref = {(r["dataset"], int(r["pack_id"])): r for r in csv.DictReader(fh)}
        worst = 0.0
        for r in rows:
            if r["combo"] != "default":
                continue
            old = ref.get((r["dataset"], r["pack_id"]))
            if old is None:
                continue
            for col in ("threshold", "exceedance_rate", "confirmed_rate_K5"):
                worst = max(worst, abs(float(r[col]) - float(old[col])))
        print(f"\n自检：default 组合 与 {ref_path.name} 的 max|Δ| = {worst:.3e}"
              f"  ({'逐值一致' if worst == 0.0 else '不一致 —— 需排查'})")

    # ---------------------------------------------------------------- 落盘
    cols = list(rows[0].keys())
    with (run_dir / "scan.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    _plot(rows, fig_dir)

    summary = {
        "source_features": str(npz_path.relative_to(root)),
        "run_dir": run_dir.name,
        "defaults": {"k": k_def, "threshold_quantile": q_def, "standardize": std_def},
        "combos": [c[0] for c in combos],
        "sensitivity_of_exceedance_rate": _sensitivity(rows, "exceedance_rate"),
        "sensitivity_of_confirmed_rate_K5": _sensitivity(rows, "confirmed_rate_K5"),
        # 论文 test dataset 2 里本地有数据的两个包 —— 单独列出，因为论文对它们报了检出
        "pack2_pack5_K5": {
            combo: {f"{x['dataset']}:{x['pack_id']}": x["confirmed_rate_K5"]
                    for x in rows if x["combo"] == combo and x["pack_id"] in (2, 5)
                    and x["dataset"] == "test2"}
            for combo in dict.fromkeys(r["combo"] for r in rows)
        },
        "caveats": [
            "越限比例是描述性比例，不是虚警率（无独立健康段与逐窗故障真值）",
            "本表刻画敏感性，不构成『某设置更好』的结论",
            "不得据『哪设置能检出 pack 2/5』择优默认值",
            "窗口非独立重复，不做显著性检验、不报 p 值或置信区间",
        ],
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n完成。产物：{run_dir.relative_to(root)}")
    print("  口径提醒：越限比例是描述性比例，不是虚警率；本表只刻画敏感性，不排名。")
    return 0


def _sensitivity(rows: list[dict], col: str) -> dict[str, float]:
    """每个因素内，该指标（跨全部包取宏平均后）的极差。

    每组都必须**把 default 计进来** —— 扫描表里 default 是单独一行的标签，
    不带 ``k=``／``q=`` 之类前缀；漏掉它会让 F1/F4 这种只有一个非默认取值的
    因素只剩一个样本，极差算不出（曾因此把 F1/F4 静默丢掉）。
    """
    combos = list(dict.fromkeys(r["combo"] for r in rows))
    out: dict[str, float] = {}
    for prefix, label in (
        ("standardize=", "F1 standardize"),
        ("k=", "F2 k"),
        ("family=", "F3 特征族"),
        ("q=", "F4 threshold_quantile"),
    ):
        sel = [c for c in combos if c.startswith(prefix) or c == "default"]
        vals = [
            float(np.mean([r[col] for r in rows if r["combo"] == c])) for c in sel
        ]
        if len(vals) >= 2:
            out[label] = float(max(vals) - min(vals))
    d = [r[col] for r in rows if r["combo"] == "default"]
    if d:
        out["default"] = float(np.mean(d))
    return out


def _plot(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combos = list(dict.fromkeys(r["combo"] for r in rows))
    theta = [np.mean([r["threshold"] for r in rows if r["combo"] == c]) for c in combos]
    k5 = [np.mean([r["confirmed_rate_K5"] for r in rows if r["combo"] == c]) for c in combos]

    # 标签一律 ASCII —— viz 的 DejaVu Sans 无中日韩字形，中文会渲染成方框
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.55 * len(combos) + 2.5))
    for ax, vals, title, xlabel in (
        (axes[0], theta, "LOF threshold theta", "theta (q-quantile of calibration LOF)"),
        (axes[1], k5, "K=5 confirmed rate (macro over packs)", "confirmed rate"),
    ):
        y = np.arange(len(combos))[::-1]
        ax.barh(y, vals, color="#4C78A8")
        ax.set_yticks(y)
        ax.set_yticklabels(combos, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "LOF sensitivity to paper-unspecified choices  |  "
        "descriptive proportions, NOT false-alarm rates  |  one factor changed at a time",
        fontsize=9,
    )
    fig.tight_layout()
    viz._save(fig, fig_dir / "scan_sensitivity.png")  # viz 的落盘助手（项目内部）


if __name__ == "__main__":
    raise SystemExit(main())
