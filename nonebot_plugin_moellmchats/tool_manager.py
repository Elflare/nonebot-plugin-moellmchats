import nonebot
from nonebot.log import logger
import ujson as json
import importlib.util
from pathlib import Path
from .model_selector import model_selector, config_path

class ToolManager:
    def __init__(self):
        self.plugin_info = {}
        self.custom_tools = {}  # 存储自定义普通函数: name -> dict

        # 初始化自定义插件说明的配置路径
        self.custom_info_file = Path(config_path / "custom_plugin_info.json")
        # 初始化自定义函数的文件夹路径
        self.custom_tools_dir = Path(config_path / "custom_tools")
        
        self._init_files()
        self._load_custom_tools()

    def _init_files(self):
        """初始化配置文件和文件夹，并生成模板以供用户参考"""
        # 1. 生成自定义插件描述模板
        if not self.custom_info_file.exists():
            default_info = {
                "_comment": "键名必须是你想修改的 nonebot 插件的真实包名（比如 nonebot_plugin_tarot）",
                "nonebot_plugin_example": {
                    "name": "示例插件名称",
                    "description": "详细描述该插件的功能，告诉大模型在什么场景下应该调用它。例如：用于抽取今天的塔罗牌运势。",
                    "usage": "严格写明该插件的触发指令格式。例如：发送'塔罗牌'或'抽牌'"
                }
            }
            with open(self.custom_info_file, "w", encoding="utf-8") as f:
                json.dump(default_info, f, ensure_ascii=False, indent=4)
        
        # 2. 生成自定义函数文件夹及代码模板
        is_first_time_dir = not self.custom_tools_dir.exists()
        self.custom_tools_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = self.custom_tools_dir / "example_tool.py"
        # 如果是首次创建文件夹，且模板文件不存在，则生成一个详细的 Python 模板
        if is_first_time_dir or not list(self.custom_tools_dir.glob("*.py")):
            template_content = '''# 这是一个自定义函数的模板文件
# 你可以在这个文件夹 (custom_tools) 下创建任意多个 .py 文件，机器人每次获取插件目录时都会自动加载它们。

import asyncio

# 1. 编写你的函数（支持普通函数 def 和 异步函数 async def）
# 注意：函数的返回值建议为字符串，它将直接作为工具调用结果反馈给大模型。
async def get_weather(city: str) -> str:
    """这是一个获取天气的示例函数"""
    # 这里可以编写真实的业务逻辑，比如请求外部API、查询数据库等
    await asyncio.sleep(1) # 模拟网络请求
    if city in ["北京", "beijing"]:
        return f"{city}目前天气晴朗，气温24度，适合外出。"
    else:
        return f"暂时无法获取 {city} 的天气信息，请尝试其他城市。"

# 2. 将函数注册到 TOOLS_REGISTRY 中
# 必须定义这个列表，插件管理器会遍历它并将其提供给大模型。
# 参数结构请严格遵守 OpenAI 的 Function Calling Schema 格式。
TOOLS_REGISTRY = [
    {
        "name": "get_weather",  # 工具的唯一英文标识符 (只能包含 a-z, A-Z, 0-9, _ )
        "description": "获取指定城市的天气状态，当用户询问天气时调用此工具。",  # 给大模型看的工具功能描述，写得越清楚大模型调用越准确
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "需要查询天气的城市名称，例如：北京、上海"
                }
            },
            "required": ["city"]  # 必填参数列表
        },
        "func": get_weather  # 绑定上面定义的真实函数对象（不要加括号）
    }
]
'''
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template_content)

    def _load_custom_tools(self):
        """遍历并加载 custom_tools 文件夹下的所有自定义函数"""
        self.custom_tools.clear()
        for file in self.custom_tools_dir.glob("*.py"):
            if file.name.startswith("__"):
                continue
            module_name = f"custom_tools_{file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                try:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # 规范：要求用户在自定义脚本中定义一个 TOOLS_REGISTRY 列表
                    if hasattr(module, "TOOLS_REGISTRY"):
                        for tool in module.TOOLS_REGISTRY:
                            self.custom_tools[tool["name"]] = tool
                except Exception as e:
                    logger.error(f"加载自定义工具文件 {file.name} 失败: {e}")

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
            self._load_custom_tools()  # 确保每次获取目录时自定义工具是最新的

        catalog = []
        if model_selector.get_use_tools():
            # 原有的 Nonebot 插件
            for name, info in self.plugin_info.items():
                display_name = info.get("name") or name
                catalog.append(f"- 插件标识: {name} | 插件名称: {display_name} | 描述: {info['description']}")
            
            # 追加自定义普通函数
            for name, info in self.custom_tools.items():
                catalog.append(f"- 插件标识: {name} | 插件名称: 自定义函数 | 描述: {info['description']} (此为原生函数调用)")

        if model_selector.get_web_search():
            catalog.append("- 插件标识: web_search | 插件名称: 联网搜索 | 描述: 回答实时问题（如时间、新闻、天气等可以使用，进行联网搜索）")
        
        return "\n".join(catalog) if catalog else "当前工具调用与联网功能均已关闭，无需返回任何插件。"

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
                            "parameters": info["parameters"]
                        }
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
