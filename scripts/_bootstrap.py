"""把项目根加进 sys.path，使 scripts/ 下的脚本从任意位置都能运行。

项目根由本文件位置反推（scripts/ 的上一级），不写绝对路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
