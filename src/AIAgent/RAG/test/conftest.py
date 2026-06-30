"""让测试可以直接导入 RAG 目录下的学习模块。"""

import sys
from pathlib import Path


RAG_DIR = Path(__file__).resolve().parent.parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))
