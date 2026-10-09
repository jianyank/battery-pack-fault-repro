"""packrepro —— Battery_Pack_Fault_Manuscript.pdf 的复现实现。

论文：Latent Feature Informed Asymmetric Autoencoder for Fault Detection in
Lithium-Ion Battery Packs（Preprint submitted to Elsevier, 2026-08-13）。

约定：全部路径相对项目根，本包内不得出现绝对路径（项目根由 :mod:`packrepro.paths` 定位）；
论文未给的超参数一律写进 ``configs/base.yaml`` 的 ``assumed:`` 段，依据见 ``docs/DEVIATIONS.md``；
结果口径边界见 ``docs/REPRODUCTION_NOTES.md``。
"""

__version__ = "0.1.0"
