"""核对 `data/raw/` 下的四个数据文件：大小 + SHA256。

    python run.py check-data
    python scripts/check_data.py

全部齐备且校验和一致时打印 OK 并以 0 退出，否则列出缺失/不符的文件并以 1 退出。
搬到别处后应先跑这个 —— 数据不随版本控制走，是"别处无改动运行"唯一真正的硬依赖。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packrepro import paths  # noqa: E402
from packrepro.config import Config  # noqa: E402

#: 期望的 (字节数, SHA256)。与本目录 data/README.md 中的表一致。
EXPECTED: dict[str, tuple[int, str]] = {
    "StandTrainData.mat": (
        279_241_224,
        "2f0d40c4bf99f1efa473ac52044e3cfb98b796dac25a730977049666cef051e0",
    ),
    "StandTestData1.mat": (
        102_858_505,
        "d499b0685c2fcea9f7f16a254322fc11195673546e8bf2a1631d3e393b145628",
    ),
    "StandTestData2.mat": (
        82_457_959,
        "2ff1842861f41a830ad11a4854ccd75ef193b8f7360f9d36443b936941a87e4a",
    ),
    "StandTestData3.mat": (
        363_656_203,
        "d1e04e97ba3ce533678db399a8b8efdc54022e3d114994c6ce321968ad3be52f",
    ),
}


def sha256_of(path: Path, *, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    cfg = Config.load()
    raw = paths.data_raw_dir()

    # 配置里声明的文件名必须都在 EXPECTED 里，反之亦然 —— 防止改了配置忘了改校验表
    declared = set(cfg["data"]["files"].values())
    if declared != set(EXPECTED):
        print(f"[FAIL] 配置声明的文件与校验表不一致\n  配置: {sorted(declared)}\n  校验表: {sorted(EXPECTED)}")
        return 1

    print(f"数据目录：{raw}")
    problems: list[str] = []
    for name, (size, digest) in EXPECTED.items():
        p = raw / name
        if not p.is_file():
            problems.append(f"{name}: 缺失")
            print(f"  [缺失] {name}")
            continue
        actual_size = p.stat().st_size
        if actual_size != size:
            problems.append(f"{name}: 大小不符 {actual_size} != {size}")
            print(f"  [大小不符] {name}: {actual_size:,} != {size:,}")
            continue
        actual = sha256_of(p)
        if actual != digest:
            problems.append(f"{name}: SHA256 不符")
            print(f"  [校验和不符] {name}\n    实际 {actual}\n    期望 {digest}")
            continue
        print(f"  [OK] {name}  {size:,} B")

    if problems:
        print(f"\nDATA CHECK FAILED: {len(problems)} 项 -> {problems}")
        print("\n数据不随版本控制走。请把四个 .mat 放到 data/raw/ 下，")
        print("或阅读 data/README.md 的「来源」一节。")
        return 1

    total = sum(s for s, _ in EXPECTED.values())
    print(f"\nDATA OK —— 四个文件齐备，合计 {total:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
