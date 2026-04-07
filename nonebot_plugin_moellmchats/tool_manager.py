import nonebot
from nonebot.log import logger
import ujson as json
import importlib.util
from pathlib import Path
from .model_selector import model_selector, config_path
from .utils import build_schema_from_func
import inspect


class ToolManager:
    def __init__(self):
        self.plugin_info = {}
        self.custom_tools = {}  # 存储自定义普通函数: name -> dict

        # 初始化自定义插件说明的配置路径
        self.custom_info_file = Path(config_path / "custom_plugin_info.json")
        # 初始化自定义函数的文件夹路径
        self.custom_tools_dir = Path(config_path / "custom_tools")

        self._init_files()
        self.load_custom_tools()
        # 用于全局存储所有自定义工具上报的依赖拓扑
        self.tool_dependencies = {}

    def _init_files(self):
        """初始化配置文件和文件夹，并生成模板以供用户参考"""
        # 1. 生成自定义插件描述模板
        if not self.custom_info_file.exists():
            default_info = {
                "_comment": "键名必须是你想修改的 nonebot 插件的真实包名（比如 nonebot_plugin_tarot）",
                "nonebot_plugin_example": {
                    "name": "示例插件名称",
                    "description": "详细描述该插件的功能，告诉大模型在什么场景下应该调用它。例如：用于抽取今天的塔罗牌运势。",
                    "usage": "严格写明该插件的触发指令格式。例如：发送'塔罗牌'或'抽牌'",
                },
            }
            with open(self.custom_info_file, "w", encoding="utf-8") as f:
                json.dump(default_info, f, ensure_ascii=False, indent=4)

        # 2. 生成自定义函数文件夹及代码模板
        is_first_time_dir = not self.custom_tools_dir.exists()
        self.custom_tools_dir.mkdir(parents=True, exist_ok=True)

        template_file = self.custom_tools_dir / "example.py"
        # 如果是首次创建文件夹，且模板文件不存在，则生成网页提取模板
        if is_first_time_dir or not list(self.custom_tools_dir.glob("*.py")):
            template_content = '''
"""
这是一个自定义大模型工具（Function Calling）的示例文件。
你可以参考此模板，在此目录下编写自己的原生 Python 函数。

【编写规范】
1. 零依赖：不需要导入 nonebot 或任何插件依赖，纯 Python 原生写法。
2. 异步函数：工具函数必须是 `async def` 定义的异步函数。
3. 工具描述：将函数的主要用途写在 `docstring`（三重引号注释）中，大模型会据此判断何时调用该工具。
4. 参数描述：引入 Python 原生的 `typing.Annotated`，格式为 `参数名: Annotated[类型, "参数说明"]`，以便大模型准确提取参数。
5. 返回值：最好返回字符串（str），大模型会直接读取此返回结果。
6. 多工具支持：你可以在同一个 .py 脚本中编写多个异步函数，插件会自动扫描并全部加载为独立工具，无需分拆文件。

【生效方式】
编写或修改完后，在群聊中发送管理员指令：`/刷新工具` 或 `/重载工具` 即可即时生效！
"""

import re
import datetime
import aiohttp
from typing import Annotated

# ==========================================
# 示例 1：无参数的工具
# ==========================================
async def get_current_datetime() -> str:
    """
    获取当前的系统日期、时间和星期几。
    当用户询问现在几点、今天几号、今天星期几等与当前时间相关的问题时，调用此工具获取准确时间。
    """
    try:
        now = datetime.datetime.now()
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        weekday_str = weekdays[now.weekday()]
        
        return f"当前系统时间是: {now.strftime(f'%Y年%m月%d日 %H:%M:%S {weekday_str}')}"
    except Exception as e:
        return f"获取时间失败: {str(e)}"

# ==========================================
# 示例 2：带参数的工具
# ==========================================
# 【依赖拓扑声明】
# 键为“触发条件”，值为“需要一并注入的工具列表”
# 表示：当大模型被分配了 web_search 工具时，强制将本脚本中的 extract_webpage 工具也提供给它。
TOOL_DEPENDENCIES = {
    "web_search": ["extract_webpage"]
}
async def extract_webpage(
    url: Annotated[str, "需要提取的完整网页链接，必须包含 http:// 或 https://"]
) -> str:
    """
    读取并提取指定URL网页的正文内容。
    当需要深入了解搜索结果中的链接，或用户要求分析某个网页时调用。
    """
    if not url.startswith(("http://", "https://")):
        return "提取失败：请提供有效的URL（以http://或https://开头）"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    timeout = aiohttp.ClientTimeout(total=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    return f"提取失败：网页返回状态码 {response.status}"
                html = await response.text()
                
        # 使用正则移除 script 和 style 标签及其内容
        text = re.sub(r'<(script|style).*?>.*?</\1>', '', html, flags=re.IGNORECASE | re.DOTALL)
        # 移除所有剩余的 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 清理多余的空白符和空行
        text = re.sub(r'\\n\\s*\\n', '\\n', text).strip()
        text = re.sub(r' {2,}', ' ', text)
        
        max_length = 4000
        if len(text) > max_length:
            text = text[:max_length] + "\\n\\n...[由于内容过长，为防止上下文超出限制，已自动截断]"
            
        return f"网页提取成功，以下是内容摘要：\\n{text}"
    except Exception as e:
        return f"提取网页失败，发生错误：{str(e)}"
'''
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template_content)

    def load_custom_tools(self):
        """遍历并加载 custom_tools 文件夹下的所有自定义函数"""
        self.custom_tools.clear()
        # 每次重载前清空旧的依赖，防止热重载时叠加死循环
        self.tool_dependencies = {}
        for file in self.custom_tools_dir.glob("*.py"):
            if file.name.startswith("__"):
                continue
            module_name = f"custom_tools_{file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # 1. 扫描并聚合该脚本中声明的依赖拓扑
                    if hasattr(module, "TOOL_DEPENDENCIES") and isinstance(
                        module.TOOL_DEPENDENCIES, dict
                    ):
                        for trigger_tool, deps in module.TOOL_DEPENDENCIES.items():
                            # 使用 set 防止多个脚本对同一个主工具注入了重复的依赖
                            self.tool_dependencies.setdefault(
                                trigger_tool, set()
                            ).update(deps)
                            logger.debug(f"文件 {file.name} 注入了依赖: {trigger_tool} -> {deps}")
                    # 规范：要求用户在自定义脚本中定义一个 TOOLS_REGISTRY 列表
                    if hasattr(module, "TOOLS_REGISTRY"):
                        # 兼容老写法：直接读取
                        for schema in module.TOOLS_REGISTRY:
                            self.custom_tools[schema["name"]] = schema
                    else:
                        # 新写法：全自动动态扫描并注入
                        for name, obj in inspect.getmembers(
                            module, inspect.iscoroutinefunction
                        ):
                            # 过滤掉私有函数 (以 _ 开头) 以及导入的其他模块的函数
                            if (
                                not name.startswith("_")
                                and obj.__module__ == module.__name__
                            ):
                                schema = build_schema_from_func(obj)
                                self.custom_tools[schema["name"]] = schema
                except Exception as e:
                    logger.error(f"加载自定义工具文件 {file.name} 失败: {e}")
        logger.debug(f"最终的工具依赖拓扑: {self.tool_dependencies}")
        logger.debug(f"最终加载的自定义工具: {list(self.custom_tools.keys())}")

    def expand_dependencies(self, plugins: set) -> set:
        """
        展开工具依赖关系，确保多步任务所需的伴生工具被一并注入。
        """
        expanded = set(plugins)
        queue = list(plugins)

        while queue:
            current = queue.pop(0)
            if current in self.tool_dependencies:
                for dep in self.tool_dependencies[current]:
                    if dep not in expanded:
                        logger.debug(f"尝试注入依赖 [{dep}]。存在性检查 custom_tools: {dep in getattr(self, 'custom_tools', {})}, plugin_info: {dep in getattr(self, 'plugin_info', {})}")
                        # 确保依赖的工具确实施加在了已加载的工具列表中
                        if dep in getattr(self, "custom_tools", {}) or dep in getattr(
                            self, "plugin_info", {}
                        ):
                            expanded.add(dep)
                            queue.append(dep)
        logger.debug(f"收到初始插件集合: {plugins}")
        return expanded

    def refresh_plugins(self):
        self.plugin_info.clear()
        blacklist = model_selector.get_tool_blacklist()

        # 读取自定义插件描述
        custom_info = {}
        try:
            with open(self.custom_info_file, "r", encoding="utf-8") as f:
                custom_info = json.load(f)
        except Exception as e:
            logger.error(f"读取自定义插件描述文件失败: {e}")

        for plugin in nonebot.plugin.get_loaded_plugins():
            if plugin.name in blacklist or "saa" in plugin.name:
                continue

            info = None
            # 优先使用用户的自定义配置
            if plugin.name in custom_info:
                info = custom_info[plugin.name]
            elif plugin.metadata:
                info = {
                    "name": plugin.metadata.name,
                    "description": plugin.metadata.description,
                    "usage": plugin.metadata.usage,
                }

            if info:
                self.plugin_info[plugin.name] = info

    def get_brief_catalog(self) -> str:
        if not self.plugin_info:
            self.refresh_plugins()
            self.load_custom_tools()  # 确保每次获取目录时自定义工具是最新的

        catalog = []
        if model_selector.get_use_tools():
            # 原有的 Nonebot 插件
            for name, info in self.plugin_info.items():
                display_name = info.get("name") or name
                catalog.append(
                    f"- 插件标识: {name} | 插件名称: {display_name} | 描述: {info['description']}"
                )

            # 追加自定义普通函数
            for name, info in self.custom_tools.items():
                catalog.append(
                    f"- 插件标识: {name} | 插件名称: 自定义函数 | 描述: {info['description']} (此为原生函数调用)"
                )

        if model_selector.get_web_search():
            catalog.append(
                "- 插件标识: web_search | 插件名称: 联网搜索 | 描述: 回答实时问题（如时间、新闻、天气等可以使用，进行联网搜索）"
            )

        return (
            "\n".join(catalog)
            if catalog
            else "当前工具调用与联网功能均已关闭，无需返回任何插件。"
        )

    def get_tool_schema(self, plugin_names: list, include_search: bool = False) -> list:
        tools = []
        for name in plugin_names:
            if name in self.plugin_info:
                info = self.plugin_info[name]
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": f"插件名称：{info.get('name') or name}。功能描述：{info['description']}。原始用法说明：{info['usage']}",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "严格根据该插件的'原始用法说明'，生成可以直接触发该插件的机器人指令字符串。",
                                    }
                                },
                                "required": ["command"],
                            },
                        },
                    }
                )
            # 处理被选中的自定义函数
            elif name in self.custom_tools:
                info = self.custom_tools[name]
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": info["description"],
                            "parameters": info["parameters"],
                        },
                    }
                )

        if include_search:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "进行互联网搜索以获取最新信息或解答未知问题。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "搜索关键词或短语",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        return tools


tool_manager = ToolManager()
