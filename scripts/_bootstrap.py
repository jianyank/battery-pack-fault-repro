"""把项目根加进 sys.path，使 scripts/ 下的脚本**从任意位置**都能运行。

这里刻意不写任何绝对路径：项目根由本文件位置反推（scripts/ 的上一级），
因此整个目录改名为 ``_moved_test`` 后仍可运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
