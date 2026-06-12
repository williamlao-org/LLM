# Python 导入机制：彻底搞清楚

## 一、两个核心变量

Python 的导入系统由**两个变量**驱动，理解了它们就理解了一切：

| 变量 | 是什么 | 谁来设置 |
|------|--------|---------|
| `sys.path` | 一个目录列表，Python 去这些目录里**找包** | Python 启动时自动设置 + `pip install` 可扩展 |
| `__package__` | 当前模块**属于哪个包**（点分路径字符串） | 由**启动方式**决定（`python xxx.py` vs `python -m`） |

---

## 二、两种启动方式

### `python xxx.py`（脚本模式）

```
- 该文件被当作独立脚本
- __name__    = "__main__"
- __package__ = None（Python 不知道它属于哪个包）
- 相对导入（from .xxx）一定会失败
- sys.path 会加入该文件所在的目录
```

### `python -m package.module`（模块模式）

```
- Python 通过包的层级去找这个模块
- __name__    = "__main__"
- __package__ = "package"（Python 知道它属于哪个包）
- 相对导入（from .xxx）可以正常工作
- sys.path 会加入当前工作目录（cwd）
```

### 对比示例

```
python rewrite_main.py
  → Python: "这是个独立脚本，我不知道它有没有兄弟模块"
  → from .rewrite_graph import ...  ← 💥 你谁？我没有父包！

python -m src.Agent.state_machine_graphflow.rewrite_main
  → Python: "哦，rewrite_main 属于 src.Agent.state_machine_graphflow 包"
  → from .rewrite_graph import ...  ← ✅ 去同包里找 rewrite_graph，找到了！
```

---

## 三、两种导入方式

### 绝对导入：从 `sys.path` 开始找

```python
from src.Agent.ReAct.main import something
```

Python 的行为：

```
遍历 sys.path 里的每个目录：
  D:\Projects\LLM\           ← 有 src/ 吗？有！
    → src/ 有 Agent/ 吗？有！
      → Agent/ 有 ReAct/ 吗？有！
        → ReAct/ 有 main.py 吗？有！→ ✅ 找到了
```

**不依赖 `__package__`**，所以 `python xxx.py` 也能用（只要 `sys.path` 对）。

### 相对导入：从 `__package__` 开始找

```python
from .rewrite_graph import GraphFlow     # 一个点 = 当前包
from ..ReAct.main import something       # 两个点 = 父包
```

Python 的行为：

```
先看 __package__ 是什么：
  __package__ = "src.Agent.state_machine_graphflow"

from .rewrite_graph
  → "." = 当前包 = src.Agent.state_machine_graphflow
  → 找 src.Agent.state_machine_graphflow.rewrite_graph → ✅

from ..ReAct.main
  → ".." = 上一级 = src.Agent
  → 找 src.Agent.ReAct.main → ✅
```

**如果 `__package__` 是 `None`**（即 `python xxx.py` 脚本模式），Python 连起点都没有，直接报错。

### 对比表

| | 绝对导入 | 相对导入 |
|---|---|---|
| 语法 | `from src.Agent.xxx import ...` | `from .xxx import ...` |
| 查找起点 | `sys.path` 里的目录 | `__package__`（当前包） |
| 需要 `__package__`？ | ❌ 不需要 | ✅ 必须有 |
| `python xxx.py` 能用？ | ✅ 能（只要 sys.path 对） | ❌ 不能 |
| `python -m` 能用？ | ✅ 能 | ✅ 能 |
| 适合场景 | 跨包导入 | 同包内导入 |

### 直觉类比

```
绝对导入 = 写完整地址寄快递
  "中国 北京市 海淀区 xx路 xx号"
  → 从全局出发，一级一级找

相对导入 = 跟邻居说方位
  "隔壁老王"（.xxx）
  "楼上的张三"（..xxx）
  → 必须先知道"我在哪"（__package__），否则"隔壁"相对于谁？
```

---

## 四、`pip install -e .` 的作用

### 它做了什么？

在虚拟环境的 `site-packages` 里放了一个 `.pth` 文件，指向项目根目录，等价于：

```python
sys.path.append("D:\\Projects\\LLM")  # 让 Python 永久知道这个路径
```

### 它不做什么？

**不影响 `__package__`。** `__package__` 完全由启动方式决定。

### 有和没有的差别

```powershell
# 没有 pip install -e . 时：
# python -m 只在项目根目录下能用（因为 cwd 会被加入 sys.path）
D:\Projects\LLM> python -m src.Agent.state_machine_graphflow.rewrite_main  ✅
D:\Desktop>      python -m src.Agent.state_machine_graphflow.rewrite_main  ❌ 找不到包

# 有 pip install -e . 后：
# python -m 在任何地方都能用（因为项目根目录已经在 sys.path 里了）
D:\Desktop>      python -m src.Agent.state_machine_graphflow.rewrite_main  ✅
D:\anywhere>     python -m src.Agent.state_machine_graphflow.rewrite_main  ✅
```

### 完整链条

```
python -m src.Agent.state_machine_graphflow.rewrite_main

  1. Python 要找 src.Agent.state_machine_graphflow 这个包
     → 去 sys.path 里的每个目录找
     → 如果没 pip install -e . 又不在项目根目录 → 找不到 → ModuleNotFoundError
     → 如果 pip install -e . 了，sys.path 里有项目根目录 → 找到了 ✅

  2. -m 参数设置 __package__ = "src.Agent.state_machine_graphflow"
     → 这一步完全是 -m 触发的，跟 pip install 无关

  3. from .rewrite_graph import ...
     → Python 看 __package__，知道要去同包里找 → 成功 ✅
```

---

## 五、包名的规则

Python 的包名（即文件夹名）**必须是合法的 Python 标识符**：

- ✅ `state_machine_graphflow`（字母、下划线）
- ✅ `ReAct`（字母）
- ❌ `StateMachine & GraphFlow`（含空格和 `&`）

即使放了 `__init__.py`，如果目录名不合法，Python 也**不认**它是一个包。

同时，沿途每一层目录都需要有 `__init__.py`：

```
D:\Projects\LLM\
└── src/
    ├── __init__.py          ← 必须有
    └── Agent/
        ├── __init__.py      ← 必须有
        └── state_machine_graphflow/
            ├── __init__.py  ← 必须有
            ├── rewrite_main.py
            ├── rewrite_graph.py
            └── rewrite_node.py
```

---

## 六、实际建议

```python
# 同包内的模块 → 用相对导入（简洁，重命名包时不用改）
from .rewrite_graph import GraphFlow

# 跨包的模块 → 用绝对导入（清晰明确）
from src.Agent.ReAct.main import something
```

运行方式：

```powershell
# 先装包（只需一次）
uv pip install -e .

# 之后在任意目录运行
python -m src.Agent.state_machine_graphflow.rewrite_main
```

---

## 七、一句话总结

> **`python -m` 设 `__package__`（让相对导入能用），`pip install -e .` 扩展 `sys.path`（让包从任何地方都能被找到）。两者配合，才能在任意位置运行包内任意模块。**
