# 数据目录

`raw/` 下的四个 `Stand*.mat` 是复现的**唯一硬依赖**。
它们合计 **828,213,891 字节**，因体积原因**不纳入版本控制**（见根目录 `.gitignore`）。
因此：**把本目录复制/克隆到别处时，必须一并带上 `raw/` 下的四个文件**，否则无法运行。

## 需要的文件与校验和

运行 `python run.py check-data` 可自动核对下表（大小 + SHA256）。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `StandTrainData.mat` | 279,241,224 | `2f0d40c4bf99f1efa473ac52044e3cfb98b796dac25a730977049666cef051e0` |
| `StandTestData1.mat` | 102,858,505 | `d499b0685c2fcea9f7f16a254322fc11195673546e8bf2a1631d3e393b145628` |
| `StandTestData2.mat` | 82,457,959 | `2ff1842861f41a830ad11a4854ccd75ef193b8f7360f9d36443b936941a87e4a` |
| `StandTestData3.mat` | 363,656,203 | `d1e04e97ba3ce533678db399a8b8efdc54022e3d114994c6ce321968ad3be52f` |

## 来源

本项目的数据**复制**自本地既有目录：

```
D:\Matlab\project\pypack\datasets\battery\source\
```

原位置**未被移动或修改**，其他项目（bs / chronoguard / pypack）对它的引用不受影响。

> 数据本身来自论文所述的商用电池包车队数据集（论文参考文献 [25]）。
> 论文声明为 28 个 8 串 160 Ah 磷酸铁锂包；本地可得 **14 个包**：
> `[1, 2, 3, 4, 5, 6, 8, 9, 10, 17, 18, 19, 20, 21]`。
> **论文 Test dataset 2 的 pack 12、14 不在其中**，对应的 Table 4 两列不可复现。

## 格式要点（实测）

* 均为 **MAT v5**（非 v7.3）→ `h5py` 读不了，只能用 `scipy.io`。
* `Xall` 形状 `(256, 20, N)` —— **窗已经切好了**，本项目不做任何再切分；
  原始滑窗步长**不可恢复**（详见 `docs/DEVIATIONS.md` B1）。
* ID 变量名在各文件间不统一：`ThisTrainID` / `ThisID1` / `ThisID`（`uint8`）。
* `Hiddall` 形状 `(1, 9, N)`，仅 train 与 test1 有；推断为 `[SOC, R₁…R₈]`，
  **单位未知**（实测值域 0.03–2.12，与 mΩ 量级不符），详见 `DEVIATIONS.md` B2。
* `ThisTimeSlot` 是 MATLAB **MCOS datetime 对象**，`scipy.io` **无法解码** ——
  时间轴实际不可用，详见 `DEVIATIONS.md` B3。
