import nonebot
from nonebot.log import logger
import ujson as json
import importlib.util
from pathlib import Path
from collections import deque
from .model_selector import model_selector, config_path
from .utils import build_schema_from_func
import inspect
from .mcp_manager import mcp_manager


class ToolManager:
    def __init__(self):
        self.plugin_info = {}
        self.custom_tools = {}  # 存储自定义普通函数: name -> dict

        # 初始化自定义插件说明的配置路径
        self.custom_info_file = Path(config_path / "custom_plugin_info.json")
        # 初始化自定义函数的文件夹路径
        self.custom_tools_dir = Path(config_path / "custom_tools")

        self._init_files()
        self.tool_dependencies = {}
        self.mcp_tool_names = set()
        self.load_custom_tools()

    def _init_files(self):
        """初始化配置文件和文件夹，并生成模板以供用户参考"""
        # 1. 生成自定义插件描述模板
        if not self.custom_info_file.exists():
            default_info = {
                "_comment": "键名必须是你想修改的 nonebot 插件的真实包名（比如 nonebot_plugin_tarot）",
                "nonebot_plugin_example": {
                    "name": "示例插件名称",
                    "description": "详细描述该插件的功能，告诉大模型在什么场景下应该调用它。",
                    "usage": "严格写明该插件的触发指令格式。例如：发送'塔罗牌'或'抽牌'",
                    "dependencies": [
                        "可选：需要一并注入的工具标识，例如 mcp__danbooru_searcher__search_tags"
                    ],
                },
            }
            with open(self.custom_info_file, "w", encoding="utf-8") as f:
                json.dump(default_info, f, ensure_ascii=False, indent=4)

        # 2. 生成自定义函数文件夹及代码模板
        is_first_time_dir = not self.custom_tools_dir.exists()
        self.custom_tools_dir.mkdir(parents=True, exist_ok=True)

        template_file = self.custom_tools_dir / "example.py"
        # 如果是首次创建文件夹，则生成模板
        if is_first_time_dir:
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
        error_count = 0
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
                            logger.debug(
                                f"文件 {file.name} 注入了依赖: {trigger_tool} -> {deps}"
                            )
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
                    error_count += 1
        self._merge_dependencies_from_custom_plugin_info()
        logger.debug(f"最终的工具依赖拓扑: {self.tool_dependencies}")
        logger.debug(f"最终加载的自定义工具: {list(self.custom_tools.keys())}")
        return error_count

    def expand_dependencies(self, plugins: set) -> set:
        """
        展开工具依赖关系，确保多步任务所需的伴生工具被一并注入。
        同时过滤黑名单，避免依赖工具绕过黑名单。
        """
        expanded = {p for p in plugins if not self.is_tool_blacklisted(p)}
        queue = deque(expanded)

        while queue:
            current = queue.popleft()

            if current in self.tool_dependencies:
                for dep in self.tool_dependencies[current]:
                    if self.is_tool_blacklisted(dep):
                        logger.debug(f"依赖工具 [{dep}] 已被黑名单禁用，跳过注入")
                        continue

                    if dep not in expanded:
                        logger.debug(
                            f"尝试注入依赖 [{dep}]。存在性检查 custom_tools: "
                            f"{dep in getattr(self, 'custom_tools', {})}, "
                            f"plugin_info: {dep in getattr(self, 'plugin_info', {})}"
                        )

                        if dep in getattr(self, "custom_tools", {}) or dep in getattr(
                            self, "plugin_info", {}
                        ):
                            expanded.add(dep)
                            queue.append(dep)

        logger.debug(f"收到初始插件集合: {plugins}，依赖展开后: {expanded}")
        return expanded

    def refresh_plugins(self):
        self.plugin_info.clear()

        # 读取自定义插件描述
        custom_info = self._load_custom_plugin_info()

        for plugin in nonebot.plugin.get_loaded_plugins():
            if "saa" in plugin.name:
                continue

            if self.is_tool_blacklisted(plugin.name):
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
        """
        给分类模型看的简版工具目录。
        注意：这里不要再调用 load_custom_tools()，否则会把已加载的 MCP 工具清掉。
        工具刷新统一交给 启动流程 / 刷新工具 命令 / 黑名单变更命令。
        """
        if not self.plugin_info:
            self.refresh_plugins()

        catalog = []

        if model_selector.get_use_tools():
            # 1. NoneBot 原生插件
            for name, info in self.plugin_info.items():
                if self.is_tool_blacklisted(name):
                    continue

                plugin_name = info.get("name") or name
                description = info.get("description") or "无描述"
                usage = info.get("usage") or "无用法说明"

                catalog.append(
                    f"- 插件标识: {name} | 插件名称: {plugin_name} | "
                    f"描述: {description} | 用法: {usage}"
                )

            # 2. 自定义函数 + MCP 工具
            for name, info in self.custom_tools.items():
                if self.is_tool_blacklisted(name):
                    continue

                tool_type = (
                    "MCP工具"
                    if name in getattr(self, "mcp_tool_names", set())
                    else "自定义函数"
                )

                description = info.get("description") or "无描述"

                catalog.append(
                    f"- 插件标识: {name} | 插件名称: {tool_type} | 描述: {description}"
                )

        # 3. 联网搜索
        if model_selector.get_web_search() and not self.is_tool_blacklisted(
            "web_search"
        ):
            catalog.append(
                "- 插件标识: web_search | 插件名称: 联网搜索 | "
                "描述: 回答实时问题，如时间、新闻、天气、近期信息等"
            )

        return (
            "\n".join(catalog)
            if catalog
            else "当前工具调用与联网功能均已关闭，无需返回任何插件。"
        )

    def is_tool_blacklisted(self, tool_name: str) -> bool:
        """统一判断普通插件、自定义函数、MCP 工具是否被黑名单禁用。"""
        blacklist = model_selector.get_tool_blacklist() or []

        for item in blacklist:
            item = str(item).strip()
            if not item:
                continue

            # 精确禁用：extract_webpage / nonebot_plugin_xxx / mcp__filesystem__read_file
            if item == tool_name:
                return True

            # 通配禁用：mcp__filesystem__*
            if item.endswith("*") and tool_name.startswith(item[:-1]):
                return True

            # 服务级禁用：mcp__filesystem 禁用 mcp__filesystem__read_file 等
            if tool_name.startswith(item + "__"):
                return True

        return False

    async def load_mcp_tools(self) -> int:
        """
        从 mcp_servers.toml 发现 MCP 工具，并合并进 custom_tools。
        黑名单在这里过滤一次，get_brief_catalog/get_tool_schema 里也会再兜底过滤。
        """
        # 清理旧 MCP tools
        for name in list(getattr(self, "mcp_tool_names", set())):
            self.custom_tools.pop(name, None)

        self.mcp_tool_names = set()

        mcp_tools = await mcp_manager.discover_tools()

        for name, schema in mcp_tools.items():
            if self.is_tool_blacklisted(name):
                continue

            self.custom_tools[name] = schema
            self.mcp_tool_names.add(name)

        logger.info(f"已加载 MCP 工具: {list(self.mcp_tool_names)}")
        return len(self.mcp_tool_names)

    def _load_custom_plugin_info(self) -> dict:
        """读取 custom_plugin_info.json。"""
        try:
            with open(self.custom_info_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.error(f"读取自定义插件描述文件失败: {e}")
            return {}

    def _merge_dependencies_from_custom_plugin_info(self):
        """
        从 custom_plugin_info.json 读取 dependencies 字段，并合并进 tool_dependencies。

        示例：
        {
          "nonebot_plugin_xxx": {
            "name": "随机图",
            "description": "...",
            "usage": "...",
            "dependencies": ["mcp__danbooru_searcher__search_tags"]
          }
        }
        """
        custom_info = self._load_custom_plugin_info()

        for plugin_name, info in custom_info.items():
            if plugin_name.startswith("_"):
                continue

            if not isinstance(info, dict):
                continue

            deps = info.get("dependencies") or info.get("tool_dependencies")
            if not deps:
                continue

            if isinstance(deps, str):
                deps = [deps]

            if not isinstance(deps, list):
                logger.warning(
                    f"custom_plugin_info.json 中 {plugin_name}.dependencies 格式错误，应为字符串列表"
                )
                continue

            clean_deps = {
                str(dep).strip() for dep in deps if isinstance(dep, str) and dep.strip()
            }

            if not clean_deps:
                continue

            self.tool_dependencies.setdefault(plugin_name, set()).update(clean_deps)

            logger.debug(
                f"custom_plugin_info.json 注入依赖: {plugin_name} -> {list(clean_deps)}"
            )

    def get_tool_schema(self, plugin_names: list, include_search: bool = False) -> list:
        tools = []

        for name in plugin_names:
            if self.is_tool_blacklisted(name):
                continue

            if name in self.plugin_info:
                info = self.plugin_info[name]
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": (
                                f"插件名称：{info.get('name') or name}。"
                                f"功能描述：{info.get('description') or '无描述'}。"
                                f"原始用法说明：{info.get('usage') or '无用法说明'}"
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": (
                                            "严格根据该插件的'原始用法说明'，"
                                            "生成可以直接触发该插件的机器人指令字符串。"
                                        ),
                                    }
                                },
                                "required": ["command"],
                            },
                        },
                    }
                )

            elif name in self.custom_tools:
                info = self.custom_tools[name]
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": info.get("description") or name,
                            "parameters": info.get("parameters")
                            or {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    }
                )

        if include_search and not self.is_tool_blacklisted("web_search"):
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
