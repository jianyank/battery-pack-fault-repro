#!/usr/bin/env python
"""统一入口 —— **与当前工作目录无关**，可从任意位置调用。

用法::

    python /path/to/Battery_Pack_Fault_Manuscript/run.py check-data
    python /path/to/Battery_Pack_Fault_Manuscript/run.py smoke
    python /path/to/Battery_Pack_Fault_Manuscript/run.py paper
    python /path/to/Battery_Pack_Fault_Manuscript/run.py lopo
    python /path/to/Battery_Pack_Fault_Manuscript/run.py all

命令之后可跟透传参数，例如::

    python run.py paper --max-epochs 3

命令一览
--------
``check-data``  核对 data/raw 下四个 .mat 的大小与 SHA256（搬到别处后先跑这个）
``smoke``       10 项自检：配置 / 数据 / 切分 / 模型 / 损失 / 特征 / LOF / 出图
``paper``       端到端，论文 §4.1 协议
``lopo``        端到端，跨包留出（更严格的泛化测试，**不是**论文协议）
``all``         先 paper 再 lopo

本文件不依赖任何绝对路径：项目根由本文件位置反推，故目录整体搬迁后仍可用。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

USAGE = """用法: python run.py <命令> [透传参数]

命令:
  check-data   核对 data/raw 下四个 .mat 的大小与 SHA256
  smoke        10 项自检（不训练，约 1 分钟）
  paper        端到端 —— 论文 §4.1 协议
  lopo         端到端 —— 跨包留出（不是论文协议）
  all          先 paper 再 lopo
  help         显示本帮助
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]

    if command == "check-data":
        import check_data

        return check_data.main()

    if command == "smoke":
        import smoke

        return smoke.main()

    if command in ("paper", "lopo"):
        import run_all

        return run_all.run(["--protocol", command, *rest])

    if command == "all":
        import run_all

        rc = run_all.run(["--protocol", "paper", *rest])
        if rc != 0:
            return rc
        return run_all.run(["--protocol", "lopo", *rest])

    print(f"未知命令：{command!r}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
