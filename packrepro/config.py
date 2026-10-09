"""配置加载与校验。对外只暴露 :class:`Config`。

配置里的路径均相对项目根，由 :mod:`packrepro.paths` 解析成绝对路径；绝对路径只在
本模块内部短暂存在，不写回配置、不外泄。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import paths

DEFAULT_CONFIG_REL = Path("configs") / "base.yaml"


class ConfigError(ValueError):
    """配置不合规（缺键、形状对不上、路径非法等）。"""


class Config:
    """``configs/base.yaml`` 的只读视图。"""

    def __init__(self, raw: dict[str, Any], source: Path) -> None:
        self._raw = raw
        self.source = source
        self._validate()

    # ---------------------------------------------------------------- 构造
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """读配置。

        ``path`` 为 None 时取 ``<root>/configs/base.yaml``；给相对路径时相对项目根解析。
        """
        if path is None:
            resolved = paths.project_root() / DEFAULT_CONFIG_REL
        else:
            p = Path(path)
            resolved = p if p.is_absolute() else paths.project_root() / p

        if not resolved.is_file():
            raise ConfigError(f"配置文件不存在：{resolved}")

        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ConfigError(f"配置根节点应为映射，实际为 {type(raw).__name__}")

        return cls(raw, resolved)

    # ---------------------------------------------------------------- 访问
    def get(self, dotted: str, default: Any = None) -> Any:
        """按点号取嵌套值，例如 ``cfg.get("train.lr")``。"""
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise ConfigError(f"缺少配置项：{dotted}")
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    # ---------------------------------------------------------------- 路径
    def data_file(self, name: str) -> Path:
        """返回 ``<root>/data/raw/<文件名>`` 的绝对路径（不检查是否存在）。"""
        filename = self.get(f"data.files.{name}")
        rel = self.get("paths.data_raw")
        return paths.resolve_config_path(Path(rel) / filename)

    def outputs_root(self) -> Path:
        return paths.resolve_config_path(self.get("paths.outputs"))

    # ---------------------------------------------------------------- 校验
    def _validate(self) -> None:
        n_v = len(self.get("channels.cell_voltage"))
        n_i = len(self.get("channels.current"))
        n_t = len(self.get("channels.temperature"))
        if (n_v, n_i, n_t) != (8, 8, 4):
            raise ConfigError(
                f"通道数应为 8 电压 / 8 电流 / 4 温度（论文 §4.1），"
                f"实际 {n_v}/{n_i}/{n_t}"
            )
        if n_v + n_i + n_t != self.get("data.expected.cols"):
            raise ConfigError("通道总数与 data.expected.cols 不一致")

        # 列号不得重叠（1-based）
        cols = (
            self.get("channels.cell_voltage")
            + self.get("channels.current")
            + self.get("channels.temperature")
        )
        if len(set(cols)) != len(cols):
            raise ConfigError("channels 各族的列号存在重叠")
        if min(cols) < 1 or max(cols) > self.get("data.expected.cols"):
            raise ConfigError(f"列号越界：应在 1..{self.get('data.expected.cols')}")

        length = self.get("windows.length")
        if length != self.get("data.expected.rows"):
            raise ConfigError("windows.length 必须等于 data.expected.rows")

        if self.get("model.latent_dim") <= 0:
            raise ConfigError("model.latent_dim 必须为正")

        q = self.get("detect.threshold_quantile")
        if not 0.0 < float(q) < 1.0:
            raise ConfigError("detect.threshold_quantile 必须在 (0,1) 内")

        if int(self.get("detect.k")) < 1:
            raise ConfigError("detect.k 必须 >= 1")

    # ---------------------------------------------------------------- 展示
    def assumed_items(self) -> list[dict[str, Any]]:
        """返回论文未给、由本项目假定的参数清单（配置里 ``assumed:`` 段的原样内容）。

        这些项在论文里无数，引用时不得当作论文事实。
        """
        items = self._raw.get("assumed")
        if not isinstance(items, list):
            raise ConfigError("配置缺少 `assumed:` 清单（论文未给项的机器可读登记）")
        for item in items:
            if not isinstance(item, dict) or not {"key", "what", "value", "why"} <= item.keys():
                raise ConfigError(f"assumed 条目字段不全（需 key/what/value/why）：{item!r}")
        return items

    def check_assumed_keys_exist(self) -> list[str]:
        """核对 ``assumed`` 里登记的 key 在配置中确实存在，返回缺失项。"""
        missing: list[str] = []
        for item in self.assumed_items():
            try:
                self.get(item["key"])
            except ConfigError:
                missing.append(item["key"])
        return missing
