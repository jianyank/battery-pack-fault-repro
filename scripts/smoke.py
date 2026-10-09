"""快速自检：不训练，逐项验证数据层、模型层、指标层、检测层、出图层。

    python scripts/smoke.py

全部通过打印 ``SMOKE OK`` 并以 0 退出，任一项失败以 1 退出。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from packrepro import paths  # noqa: E402
from packrepro.config import Config  # noqa: E402
from packrepro.data import (  # noqa: E402
    build_packs,
    load_experiment,
    train_val_split,
)
from packrepro.data.normalize import ChannelScaler  # noqa: E402
from packrepro.detect import LOFDetector  # noqa: E402
from packrepro.features import compute_features, feature_names  # noqa: E402
from packrepro.model import AsymmetricAE, CompositeLoss  # noqa: E402
from packrepro.model.targets import build_align_target  # noqa: E402

FAILED: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"  [PASS] {name}" + (f"  — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001
        FAILED.append(name)
        print(f"  [FAIL] {name}  — {type(exc).__name__}: {exc}")


def main() -> int:
    print(f"project root: {paths.project_root()}")
    cfg = Config.load()

    state: dict = {}

    def c1_config():
        items = cfg.assumed_items()
        missing = cfg.check_assumed_keys_exist()
        assert not missing, f"assumed 登记的 key 在配置里不存在：{missing}"
        return f"{len(items)} 项论文未给参数已登记"

    def c2_data():
        b = load_experiment(cfg, "train")
        state["bundle"] = b
        assert b.n_windows == 26508, f"train 窗数应 26508，实际 {b.n_windows}"
        assert b.signals.shape[1:] == (256, 20), b.signals.shape
        return f"train {b.n_windows} 窗, loader={b.loader}"

    def c3_packs():
        packs = build_packs(state["bundle"], cfg)
        state["packs"] = packs
        assert sorted(packs) == [6, 8, 9, 10], sorted(packs)
        exp = {6: 9127, 8: 11147, 9: 5093, 10: 1141}
        got = {p: packs[p].n_windows for p in sorted(packs)}
        assert got == exp, f"包分布不符：{got} != {exp}"
        return f"包分布 {got}"

    def c4_split_scaler():
        p = state["packs"][6]
        itr, iva = train_val_split(p.n_windows, val_ratio=0.15, seed=42)
        sc = ChannelScaler.fit(p.full[itr], method=cfg.get("data.normalize.method"))
        vsc = sc.subset([c - 1 for c in cfg.get("channels.cell_voltage")])
        z = vsc.transform(p.voltage[iva])
        assert z.shape == p.voltage[iva].shape
        assert abs(float(np.median(z))) < 0.2, f"标准化后中位数偏离 0：{np.median(z)}"
        state["volt_scaler"] = vsc
        state["ch_scaler"] = sc
        return f"train {itr.size} / val {iva.size}"

    def c5_targets():
        p = state["packs"][6]
        t = build_align_target(
            p.hidden,
            soc_channel=cfg.get("hidden.soc_channel"),
            resistance_channels=cfg.get("hidden.resistance_channels"),
            resistance_target=cfg.get("train.loss.latent_align.resistance_target"),
        )
        assert t.shape == (p.n_windows, 2), t.shape
        return f"对齐目标 {t.shape}"

    def c6_model_forward():
        m = AsymmetricAE(
            time_len=cfg.get("windows.length"),
            latent_dim=cfg.get("model.latent_dim"),
            variational=cfg.get("model.variational"),
        )
        o = m(torch.randn(2, 256, 20))
        assert tuple(o.recon.shape) == (2, 256, 8), o.recon.shape
        assert tuple(o.z.shape) == (2, 256), o.z.shape
        n = sum(p.numel() for p in m.parameters())
        return f"recon {tuple(o.recon.shape)}, z {tuple(o.z.shape)}, {n:,} 参数"

    def c7_loss():
        m = AsymmetricAE()
        o = m(torch.randn(2, 256, 20))
        fn = CompositeLoss(lambda1=1.0, lambda2=0.1, lambda3=1e-3)
        parts = fn(o, torch.randn(2, 256, 8), torch.randn(2, 2))
        fn0 = CompositeLoss(lambda1=1.0, lambda2=0.0, lambda3=0.0)
        parts0 = fn0(o, torch.randn(2, 256, 8), None)
        assert torch.allclose(parts0.total, parts0.recon), "λ₂=λ₃=0 时 total 应等于 recon"
        return (f"total={float(parts.total.detach()):.4f} "
                f"recon={float(parts.recon.detach()):.4f}")

    def c8_features():
        rng = np.random.default_rng(0)
        v = rng.normal(size=(16, 256, 8))
        f = compute_features(v, v + 0.01, q3_percentile=75.0)
        assert f.shape == (16, 24), f.shape
        assert np.allclose(f[:, 8:16], 1.0, atol=1e-3), "自身与自身的余弦应≈1"
        assert len(feature_names()) == 24
        return f"特征 {f.shape}"

    def c9_lof():
        rng = np.random.default_rng(0)
        calib = rng.normal(size=(500, 24))
        det = LOFDetector(k=20, threshold_quantile=0.99, persistence=1).fit(calib)
        res = det.detect(rng.normal(size=(200, 24)))
        assert res.scores.shape == (200,)
        assert 0.0 <= res.exceeded.mean() <= 1.0
        s = det.calib_score_summary()
        return f"θ={res.threshold:.3f}, 校准中位数={s['median']:.3f}(应≈1)"

    def c11_deterministic_inference():
        """推理期必须不抽样，否则同一批窗口每次评分都得到不同特征（``docs/DEVIATIONS.md`` D4）。"""
        m = AsymmetricAE().eval()
        x = torch.randn(2, 256, 20)
        with torch.no_grad():
            a = m(x, sample=False).recon
            b = m(x, sample=False).recon
            c = m(x, sample=True).recon
            d = m(x, sample=True).recon
        assert torch.equal(a, b), "sample=False 两次前向应逐位相同"
        assert not torch.equal(c, d), "sample=True 两次前向应不同（变分采样）"
        return f"推理不抽样；采样 max|Δ|={float((c - d).abs().max()):.4f}"

    def c10_figure():
        from packrepro.viz import plot_lof_scores

        with tempfile.TemporaryDirectory() as td:
            out = plot_lof_scores(
                np.abs(np.random.default_rng(0).normal(size=300)) + 1.0,
                2.0, pack_id=6, out=Path(td) / "x.png", protocol="smoke",
            )
            assert out.is_file() and out.stat().st_size > 0
        return "出图正常"

    print("\nchecks:")
    check("1  配置加载与 assumed 登记", c1_config)
    check("2  读取 train", c2_data)
    check("3  按包组装与计数", c3_packs)
    check("4  切分与归一化", c4_split_scaler)
    check("5  对齐目标构造", c5_targets)
    check("6  模型前向", c6_model_forward)
    check("7  复合损失", c7_loss)
    check("8  24 维特征", c8_features)
    check("9  LOF 检测", c9_lof)
    check("10 出图", c10_figure)
    check("11 推理期确定性（不抽样）", c11_deterministic_inference)

    if FAILED:
        print(f"\nSMOKE FAILED: {len(FAILED)} 项未通过 -> {FAILED}")
        return 1
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
