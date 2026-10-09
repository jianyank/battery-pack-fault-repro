"""packrepro —— Battery_Pack_Fault_Manuscript.pdf 的复现实现。

论文：Latent Feature Informed Asymmetric Autoencoder for Fault Detection in
Lithium-Ion Battery Packs（Preprint submitted to Elsevier, 2026-08-13）。

设计约定
--------
1. **全部路径相对项目根**，本包内不得出现绝对路径。项目根由
   :mod:`packrepro.paths` 定位，整个目录可整体搬迁后继续运行。
2. 论文未给出的超参数（λ、LOF 的 k、滑窗步长、持久性规则等）一律在
   ``configs/base.yaml`` 中显式写出并标注 ``ASSUMED``，依据见 ``docs/DEVIATIONS.md``。
3. 结果口径边界见 ``docs/REPRODUCTION_NOTES.md``。
"""

__version__ = "0.1.0"
