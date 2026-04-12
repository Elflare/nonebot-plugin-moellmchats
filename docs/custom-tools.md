# 自定义工具开发

`custom_tools/` 目录允许你直接编写原生 Python 函数，供大模型通过 Function Calling 原生调用，**无需模拟 NoneBot 消息事件**。适合编写计算器、爬虫、系统查询等轻量级扩展工具。

首次运行后会自动生成该目录及示例模板 `custom_tools/example.py`。

---

## 编写规范

### 基本示例

```python
from typing import Annotated

async def get_weather(
    city: Annotated[str, "需要查询天气的城市名称，如：北京市、上海市"]
) -> str:
    """
    获取指定城市的实时天气情况。当用户询问天气时调用此工具。
    """
    # 实际使用时替换为真实 API 调用
    return f"{city}今天天气晴朗，气温25度。"
```

**规范说明：**

- 函数名即为工具名，建议使用英文下划线命名
- 参数类型用 `Annotated[类型, "参数说明"]` 标注，说明会直接传给 LLM
- 函数 docstring 即为工具描述，告诉 LLM 何时调用此工具
- 支持 `async def`（推荐）和普通 `def`
- 返回值为 `str`，直接作为工具执行结果返回给 LLM

### 多参数示例

```python
from typing import Annotated

async def calculate(
    expression: Annotated[str, "数学表达式，如：(1+2)*3/4"],
    precision: Annotated[int, "结果保留小数位数，默认2"] = 2
) -> str:
    """
    计算数学表达式的结果。当用户需要精确计算时调用。
    """
    try:
        result = eval(expression)
        return f"计算结果：{round(result, precision)}"
    except Exception as e:
        return f"计算失败：{e}"
```

---

## 工具依赖拓扑（TOOL_DEPENDENCIES）

当一个工具的执行依赖另一个工具时，可以声明 `TOOL_DEPENDENCIES`，让分类模型分配主工具时自动将依赖工具也注入给 LLM。

```python
from typing import Annotated

# 声明：当 LLM 被分配了 get_weather 工具时，强制同时注入 extract_webpage 工具
TOOL_DEPENDENCIES = {
    "get_weather": ["extract_webpage"]
}

async def get_weather(
    city: Annotated[str, "城市名称"]
) -> str:
    """获取城市天气。"""
    ...

async def extract_webpage(
    url: Annotated[str, "要抓取内容的网页 URL"]
) -> str:
    """抓取指定网页的文本内容。"""
    ...
```

**格式**：`{ "触发工具名": ["要同时注入的工具名1", "工具名2"] }`

---

## 热重载

编写完成后，使用 Bot 指令 `刷新工具` 即可热重载，**无需重启 NoneBot**。

---

## 注意事项

- 文件名任意（`.py` 后缀），但函数名不能与系统内置工具冲突
- 同一文件中可定义多个工具函数
- 不支持从工具函数内主动发送 QQ 消息（如需与 QQ 交互请使用 NoneBot 插件集成）
- 工具执行异常时建议 `try/except` 后返回错误描述字符串，而非抛出异常

---

## 相关页面

- [NoneBot 插件集成](./plugin-integration.md) — 让 LLM 调用现有 Bot 插件
- [配置参考 → model_config.json](./configuration.md#model_configjson--智能调度配置) — `use_tools`、`tool_blacklist`、`resident_plugins` 配置项
