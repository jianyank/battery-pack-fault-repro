# Battery Pack Fault Manuscript —— 复现

对论文 **"Latent Feature Informed Asymmetric Autoencoder for Fault Detection in
Lithium-Ion Battery Packs"**（Preprint submitted to Elsevier, 2026-08-13）的独立复现。

方法链：电池包多通道时序 → 非对称卷积自编码器（部分潜变量对齐到电池模型状态）→
24 维故障指标 → 局部离群因子（LOF）→ 异常报警。

> **先读这个**：本项目对论文有 13 项假定、2 项不可恢复的缺口、3 处与论文不一致的实现选择，
> 以及 3 处复现中发现的论文自身问题。全部逐条登记在 **[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)**。
> 任何引用本复现数字的场合，都应同时引用该文件。

---

## 1. 环境

已验证可用的环境是本机 conda 环境 `batfd`：

```bash
conda activate batfd          # Python 3.12.14 / torch 2.14.0+cu126 / CUDA 可用
# 或按 requirements.txt 自行安装
```

关键依赖：`torch`、`numpy`、`scipy`、`scikit-learn`、`matplotlib`、`pandas`、`PyYAML`。
**注意**：系统默认 python 没有 torch，必须用装了 torch 的环境。

## 2. 数据

四个 `Stand*.mat` 放在 `data/raw/`，共 **828,213,891 字节**（已 gitignore，不入库）。

| 文件 | 大小 | 内容 |
|---|---:|---|
| `StandTrainData.mat` | 279 MB | 26,508 窗；包 `{6,8,9,10}` 的**正常段**；含 `Hiddall` |
| `StandTestData1.mat` | 103 MB | 8,110 窗；**同样四包**的故障段；含 `Hiddall` |
| `StandTestData2.mat` | 82 MB | 21,013 窗；包 `1–5` |
| `StandTestData3.mat` | 364 MB | 40,093 窗；包 `17–21` |

格式为 **MAT v5**（`h5py` 读不了，只能用 `scipy.io`）。
`Xall` 形状 `(256, 20, N)` —— **每个样本本身就是切好的窗**，本项目不再切分。

> 已知 scipy 缺陷：`scipy.io.whosmat` 会在部分文件上抛 `TypeError`，
> 读取层会自动回退 `loadmat` 并记录实际路径（见 `packrepro/data/mat_io.py`）。

## 3. 运行

```bash
# 快速自检（不训练，约 1 分钟）
python scripts/smoke.py

# 端到端：论文协议
python scripts/run_all.py --protocol paper

# 端到端：跨包留出（更严格的泛化测试，不是论文协议）
python scripts/run_all.py --protocol lopo

# 冒烟：3 个 epoch 跑通全链路
python scripts/run_all.py --max-epochs 3
```

产物落在 `outputs/<timestamp>_<tag>/`：训练历史、checkpoint、特征、LOF 分数、
`detection_table.csv`、`figures/*.png`、`summary.json`。

训练耗时参考：RTX 4060 Laptop 上约 **1.0 秒/epoch**（17k 窗、batch 512），
故 1500 epochs 单折约 0.4 小时。

## 4. 目录结构

```
configs/base.yaml          # 单一事实来源；`assumed:` 段是论文未给项的机器可读清单
configs/fault_table.yaml   # 28 包故障表（取自论文 PDF 第 20 页原文）
data/raw/                  # 四个 .mat（不入库）
packrepro/
  paths.py                 # ★ 相对根定位，全库无绝对路径
  config.py                # 配置加载与校验
  data/                    # mat_io / channels / windows / normalize / dataset
  model/                   # net（非对称 AE）/ losses / targets / train
  features.py              # 24 维故障指标（§2.3）
  detect.py                # LOF + 分位阈值 + 持久性（§2.4）
  evaluation.py            # 重建、特征、检测的编排
  viz.py                   # 出图
scripts/
  smoke.py                 # 10 项自检
  run_all.py               # 端到端驱动
docs/
  DEVIATIONS.md            # ★ 与论文的全部偏差登记
  REPRODUCTION_NOTES.md    # 论文给了什么 / 不可复现什么
  Battery_Pack_Fault_Manuscript.pdf
```

## 5. 可移植性

**整个目录可任意搬迁、改名、换机器。** 项目根由 `packrepro/paths.py` 从文件位置反推，
代码中不含绝对路径，`configs/base.yaml` 中的路径一律相对项目根书写。
若需显式指定根，可设环境变量 `PACKREPRO_ROOT`。

**实测已验证**：把整个目录改名为 `_portability_probe_renamed`、不设任何环境变量，
在其下运行 `python scripts/smoke.py` —— 10 项自检全部通过，项目根被正确解析到新路径。
代码与配置中的绝对路径扫描为**零命中**。

复现该验证：复制目录（或仅复制代码并把 `data/raw/*.mat` 硬链接过去），改名后跑 `smoke.py` 即可。

## 6. 结果口径 —— 不报什么

| 项 | 可否 | 原因 |
|---|---|---|
| 训练侧 `val_recon`、24 维特征分布、LOF 分数与越限比例 | ✅ | 直接可算 |
| 提前量天数 | ❌ | 无投运日期；窗口序号不是时间轴 |
| 真实虚警率 / 召回率 / F1 | ❌ | 无独立健康段与逐窗故障真值 |
| 单体定位准确率 | ❌ | `Hiddall` 与对齐目标同源，非独立真值 |
| 与论文 Table 3/4 逐日对照 | ❌ | 天数零点不同；pack 12、14 本地无数据 |

**越限比例是描述性比例，不是虚警率。** 论文取训练集 LOF 分数的 99 分位作阈值，
即训练数据本身就有约 1% 越限；当被评窗口来自另一分布时，越限比例必然远高于 1%。
这不是 LOF 算错，而是阈值校准域与评分域不一致的必然结果。
