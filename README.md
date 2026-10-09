# Battery Pack Fault Manuscript —— 复现

对论文 **"Latent Feature Informed Asymmetric Autoencoder for Fault Detection in
Lithium-Ion Battery Packs"**（Preprint submitted to Elsevier, 2026-08-13）的独立复现。

方法链：电池包多通道时序 → 非对称卷积自编码器（部分潜变量对齐到电池模型状态）→
24 维故障指标 → 局部离群因子（LOF）→ 异常报警。

> **先读这个**：本项目对论文有 13 项假定、3 项不可恢复的缺口、3 处与论文不一致的实现选择，
> 以及 5 处复现中发现的问题（其中 D1–D3、D5 属**论文自身**，D4 是**本项目的实现缺陷**、已修复）。
> 全部逐条登记在 **[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)**。
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

**数据不随版本控制走**（`data/raw/` 已 gitignore，828 MB）。把本目录搬到别处时
必须一并带上这四个文件，否则无法运行。校验和与来源见
[`data/README.md`](data/README.md)，用 `python run.py check-data` 核对。

格式为 **MAT v5**（`h5py` 读不了，只能用 `scipy.io`）。
`Xall` 形状 `(256, 20, N)` —— **每个样本本身就是切好的窗**，本项目不再切分。

> 已知 scipy 缺陷：`scipy.io.whosmat` 会在部分文件上抛 `TypeError`，
> 读取层会自动回退 `loadmat` 并记录实际路径（见 `packrepro/data/mat_io.py`）。

## 3. 运行

统一入口 `run.py`，**与当前工作目录无关**，可从任意位置调用：

```bash
python run.py check-data    # 核对 data/raw 四个 .mat 的大小与 SHA256（搬到别处后先跑这个）
python run.py smoke         # 11 项自检，不训练，约 1 分钟
python run.py paper         # 端到端，论文 §4.1 协议
python run.py lopo          # 端到端，跨包留出（更严格的泛化测试，不是论文协议）
python run.py all           # 先 paper 再 lopo
```

命令后可透传参数，例如 `python run.py paper --max-epochs 3` 做冒烟。

`scripts/` 下的脚本也可直接运行（等效）：`python scripts/smoke.py`、
`python scripts/run_all.py --protocol paper`、`python scripts/check_data.py`。

两个**诊断**脚本（不训练，秒级／约 1 分钟）：

```bash
python scripts/rebuild_baseline.py   # 从已存 checkpoint 重算确定性基线（约 1 分钟）
python scripts/lof_scan.py           # LOF 侧单因素敏感性扫描（秒级）
```

> `rebuild_baseline.py` 的存在是因为一处已修复的缺陷：推理期曾做变分采样，导致
> 同一批窗口每次前向都得到不同特征（见 [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) D4）。

产物落在 `outputs/<timestamp>_<tag>/`：训练历史、checkpoint、特征、LOF 分数、
`detection_table.csv`、`figures/*.png`、`summary.json`。

训练耗时参考：RTX 4060 Laptop 上约 **1.0 秒/epoch**（17k 窗、batch 512），
故 1500 epochs 单折约 0.4 小时。

## 4. 目录结构

```
run.py                     # ★ cwd 无关的统一入口（check-data / smoke / paper / lopo / all）
configs/base.yaml          # 单一事实来源；`assumed:` 段是论文未给项的机器可读清单
configs/fault_table.yaml   # 28 包故障表（取自论文 PDF 第 20 页原文）
data/raw/                  # 四个 .mat（不入库；校验和见 data/README.md）
packrepro/
  paths.py                 # ★ 相对根定位，全库无绝对路径
  config.py                # 配置加载与校验
  data/                    # mat_io / channels / normalize / dataset
  model/                   # net（非对称 AE）/ losses / targets / train
  features.py              # 24 维故障指标（§2.3）
  detect.py                # LOF + 分位阈值 + 持久性（§2.4）
  evaluation.py            # 重建、特征、检测的编排
  viz.py                   # 出图
scripts/
  smoke.py                 # 11 项自检
  run_all.py               # 端到端驱动
  check_data.py            # 数据大小与 SHA256 核对
  rebuild_baseline.py      # 从 checkpoint 重算确定性基线（不训练）
  lof_scan.py              # LOF 单因素敏感性扫描（只读）
docs/
  DEVIATIONS.md            # ★ 与论文的全部偏差登记
  REPRODUCTION_NOTES.md    # 论文给了什么 / 不可复现什么
  Battery_Pack_Fault_Manuscript.pdf
  Battery_Pack_Fault_Reproduction_Report.md   # ⛔ 已弃用：pypack 时代产物，数字不可引用
```

## 5. 别处运行

项目根由 `packrepro/paths.py` 从文件位置反推，代码与配置中**不含任何绝对路径**，
`configs/base.yaml` 里的路径一律相对项目根书写。若需显式指定根，可设 `PACKREPRO_ROOT`。

### 已实测验证的场景

| 场景 | 结果 |
|---|---|
| 目录改名 / 换路径 | ✅ 在 `_portability_probe_renamed` 下、不设环境变量，smoke 全项通过 |
| 从任意工作目录启动 | ✅ 在 `/tmp` 下 `python <路径>/run.py check-data` 与 `run.py smoke` 均通过 |
| 无 GPU 的机器 | ✅ `CUDA_VISIBLE_DEVICES=""` 跑通全链路，自动回退 CPU |
| 无需安装 | ✅ 无 `pyproject`/`setup.py`，靠 `sys.path` 插入，不需要 `pip install -e .` |

### 搬到别处的检查清单

1. **带上数据。** `data/raw/` 下的四个 `.mat` 不随版本控制走。跑
   `python run.py check-data` 核对（大小 + SHA256，期望值见 [`data/README.md`](data/README.md)）。
2. **选对解释器。** 必须是装了 `torch` 的环境（见 `requirements.txt`）。
   IDE 默认解释器往往不是它 —— 症状是 `ModuleNotFoundError: No module named 'torch'`。
3. **跑一遍自检。** `python run.py smoke`，11 项应全过。

### 未实测的场景

* **跨操作系统**：只在 Windows 上验证过。代码用 `pathlib`，理论上跨平台，但**未实测**。
* **Python 版本**：只在 3.12.14 上验证过。
* **torch 版本**：只在 2.14.0 上验证过。

## 6. 结果口径 —— 不报什么

可报 / 不可报的完整清单见 [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) G 节。要点：
不报提前量天数（无投运日期，窗口序号不是时间轴）、真实虚警率 / 召回率 / F1（无独立健康段
与逐窗故障真值）、单体定位准确率（`Hiddall` 与对齐目标同源）、与论文 Table 3/4 的逐日对照。

**越限比例是描述性比例，不是虚警率。** 论文取训练集 LOF 分数的 99 分位作阈值，
即训练数据本身就有约 1% 越限；被评窗口来自另一分布时，越限比例必然远高于 1%。
这不是 LOF 算错，而是阈值校准域与评分域不一致的必然结果。
