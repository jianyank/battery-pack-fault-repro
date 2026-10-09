"""项目根定位 —— 全库唯一的路径来源，项目中不得出现绝对路径。

按优先级：① 环境变量 ``PACKREPRO_ROOT``（须指向含 ``configs/base.yaml`` 的目录）；
② 从本文件向上回溯，取第一个含 ``configs/base.yaml`` 的目录。

找不到就抛错，不回退到 cwd —— 静默回退会让"跑错目录"看起来像"跑通了"。
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_ROOT = "PACKREPRO_ROOT"

#: 用作项目根判定的标记文件（相对项目根书写）
_MARKER = Path("configs") / "base.yaml"

#: 项目根下的固定目录
CONFIGS_DIRNAME = "configs"
DATA_RAW_DIRNAME = Path("data") / "raw"
OUTPUTS_DIRNAME = "outputs"
DOCS_DIRNAME = "docs"


def project_root() -> Path:
    """返回项目根目录（绝对路径，仅在本函数内构造，不外泄到配置）。"""
    override = os.environ.get(ENV_ROOT)
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / _MARKER).is_file():
            raise FileNotFoundError(
                f"{ENV_ROOT}={override!r} 未指向有效项目根："
                f"该目录下找不到 {_MARKER}"
            )
        return root

    for parent in Path(__file__).resolve().parents:
        if (parent / _MARKER).is_file():
            return parent

    raise FileNotFoundError(
        f"向上回溯未找到含 {_MARKER} 的目录。"
        f"请确认目录结构完整，或设置环境变量 {ENV_ROOT} 指向项目根。"
    )


def configs_dir() -> Path:
    """``<root>/configs``"""
    return project_root() / CONFIGS_DIRNAME


def data_raw_dir() -> Path:
    """``<root>/data/raw`` —— 四个 Stand*.mat 所在目录。"""
    return project_root() / DATA_RAW_DIRNAME


def outputs_dir() -> Path:
    """``<root>/outputs`` —— 运行产物根目录（不入库）。"""
    return project_root() / OUTPUTS_DIRNAME


def docs_dir() -> Path:
    """``<root>/docs``"""
    return project_root() / DOCS_DIRNAME


def resolve_config_path(value: str | Path) -> Path:
    """把配置里的相对路径解析成绝对路径。绝对路径一律拒绝，以免把本机路径写进配置。"""
    p = Path(value)
    if p.is_absolute():
        raise ValueError(
            f"配置中的路径必须相对项目根，收到绝对路径：{value!r}"
        )
    return project_root() / p


def make_run_dir(tag: str, *, timestamp: str) -> Path:
    """创建并返回 ``<root>/outputs/<timestamp>_<tag>/``。"""
    run_dir = outputs_dir() / f"{timestamp}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
