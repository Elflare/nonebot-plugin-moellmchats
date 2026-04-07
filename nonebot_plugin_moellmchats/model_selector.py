from pathlib import Path
import ujson as json

try:
    import tomllib
except ImportError:
    import tomli as tomllib
import nonebot_plugin_localstore as store
from nonebot.log import logger
import aiohttp
from collections import defaultdict

config_path: Path = store.get_plugin_config_dir()


# 模型选择类
class ModelSelector:
    def __init__(self):
        # 配置文件路径
        self.models_file = Path(config_path / "models.json")
        self.providers_file = Path(config_path / "providers.toml")
        self.cache_file = Path(config_path / "model_cache.json")
        self.model_config_file = Path(config_path / "model_config.json")

        self.models = {}
        self.providers = {}

        # 加载配置
        self.load_providers()
        self._load_all_models()
        self.model_config = self._load_model_config()

    def _normalize_url(self, base_url: str, endpoint: str = "/chat/completions") -> str:
        """自动处理末尾的反斜杠和补全路径"""
        base_url = base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            # 如果配置的是完整路径，且需要请求 /models，则进行替换
            return (
                base_url
                if endpoint == "/chat/completions"
                else base_url.replace("/chat/completions", endpoint)
            )
        return base_url + endpoint

    def _normalize_key(self, key: str) -> str:
        """自动补齐 Bearer"""
        key = key.strip()
        return key if key.lower().startswith("bearer ") else f"Bearer {key}"

    def load_providers(self):
        """初始化并读取 providers.toml，如果不存在则自动生成模板"""
        if not self.providers_file.exists():
            template = """# AI服务商配置文件
# base_url: 基础API地址（直接写Base URL即可，程序会自动补全 /chat/completions 及 /models）
# api_key: 你的API密钥（无需手动写 Bearer ，程序会自动补全）
# proxy: [可选] 该服务商的全局代理
# models: [可选] 手动补充的模型列表。若API不支持 /models 自动获取，或获取不全时可在这里手动指定作为补充。

[providers.deepseek]
base_url = "https://api.deepseek.com"
api_key = "sk-xxxxxx"
models = ["deepseek-chat", "deepseek-reasoner"]

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-xxxxxx"
proxy = "http://127.0.0.1:7890"

# 【可选】模型高级配置：对特定模型（无论是自动获取的还是上面手动填写的）进行精细化参数覆写
[providers.openai.model_configs.gpt-4o]
temperature = 1.2
stream = true
is_segment = true
max_segments = 5

[providers.openai.model_configs.o1-preview]
stream = false  # 不支持流式的模型可单独关闭
"""
            self.providers_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.providers_file, "w", encoding="utf-8") as f:
                f.write(template)

        try:
            with open(self.providers_file, "rb") as f:
                config = tomllib.load(f)
                self.providers = config.get("providers", {})
        except Exception as e:
            logger.error(f"解析 providers.toml 失败: {e}")

    def _load_all_models(self):
        """合并加载：旧 models.json、缓存自动获取、TOML手动补充及模型独立配置"""
        self.models.clear()
        # 1. 兼容加载老版 models.json
        if self.models_file.exists():
            try:
                with open(self.models_file, "r", encoding="utf-8") as f:
                    old_models = json.load(f)
                    for mid, info in old_models.items():
                        info["provider"] = "旧版配置(models.json)"
                        self.models[mid] = info
            except Exception as e:
                logger.error(f"读取 models.json 失败: {e}")

        # 预读取缓存文件
        cached_models = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached_models = json.load(f)
            except Exception as e:
                logger.error(f"读取 model_cache.json 失败: {e}")

        # 2. 合并并加载 TOML 中的模型及配置
        for provider, p_info in self.providers.items():
            url = self._normalize_url(p_info.get("base_url", ""))
            key = self._normalize_key(p_info.get("api_key", ""))
            proxy = p_info.get("proxy")
            model_configs = p_info.get("model_configs", {})

            # 收集该供应商下的所有模型ID（去重）
            m_ids = set()
            if provider in cached_models:
                m_ids.update(cached_models[provider])
            m_ids.update(p_info.get("models", []))
            m_ids.update(model_configs.keys())  # 只要写了独立配置的，也隐式作为模型加入

            for mid in m_ids:
                # 基础信息
                model_data = {
                    "url": url,
                    "key": key,
                    "model": mid,
                    "provider": provider,
                }
                # 服务商级别的配置
                if proxy:
                    model_data["proxy"] = proxy

                # 模型级别的覆写配置（会覆盖同名键）
                if mid in model_configs:
                    model_data.update(model_configs[mid])

                self.models[mid] = model_data

    async def fetch_models_from_providers(self):
        """异步拉取各服务商的 /models 列表并落盘，然后重载内存模型字典"""
        new_cache = {}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            for provider, info in self.providers.items():
                base_url = info.get("base_url")
                api_key = info.get("api_key")
                proxy = info.get("proxy")  # 获取服务商配置的代理

                if not base_url or not api_key:
                    continue

                url = self._normalize_url(base_url, "/models")
                headers = {"Authorization": self._normalize_key(api_key)}

                try:
                    # 将 proxy 传入 get 请求
                    async with session.get(url, headers=headers, proxy=proxy) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            models = [
                                m["id"] for m in data.get("data", []) if "id" in m
                            ]
                            new_cache[provider] = models
                            logger.info(
                                f"成功从服务商 {provider} 获取 {len(models)} 个可用模型。"
                            )
                        else:
                            logger.warning(
                                f"从服务商 {provider} 获取模型失败，HTTP状态码: {resp.status}"
                            )
                except Exception as e:
                    logger.warning(f"请求服务商 {provider} 的 /models 接口异常: {e}")

        # 持久化到缓存
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(new_cache, f, ensure_ascii=False, indent=4)

        # 刷新内存
        self._load_all_models()
        logger.info(f"模型列表更新完毕，当前可用模型总数：{len(self.models)}")

    def _load_models(self):
        # 读取models.json文件，获取多个模型的配置
        if self.models_file.exists():
            with open(self.models_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 创建默认
            default_models = {
                "dpsk-chat": {
                    "url": "https://api.deepseek.com/chat/completions",
                    "key": "Bearer xxx",
                    "model": "deepseek-chat",
                    "temperature": 1.0,
                    "max_tokens": 1024,
                    "proxy": "http://127.0.0.1:7890",
                    "stream": True,
                    "is_segment": True,
                    "max_segments": 5,
                },
                "dpsk-r1": {
                    "url": "https://api.deepseek.com/chat/completions",
                    "key": "Bearer xxxx",
                    "model": "deepseek-reasoner",
                    "top_k": 5,
                    "top_p": 1.0,
                },
            }
            self.models_file.parent.mkdir(parents=True, exist_ok=True)
            self.models_file.touch()
            self._write_config(self.models_file, default_models)

    def get_model_config(self):
        return json.dumps(self.model_config, indent=4, ensure_ascii=False)

    def _load_model_config(self):
        # 读取model_config.json文件，获取是否使用MOE及MOE难度模型等配置
        if self.model_config_file.exists():
            with open(self.model_config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 如果model_config.json文件不存在，使用默认配置
            default_config = {
                "use_moe": False,
                "moe_models": {"0": "glm", "1": "glm", "2": "glm"},
                "selected_model": "dpsk-chat",
                "category_model": "glm",
                "vision_model": "",  # 专门处理视觉任务的模型，默认不使用
                "use_web_search": False,
                "use_tools": True,
                "tool_blacklist": [
                    "nonebot_plugin_alconna",
                    "nonebot_plugin_session_orm",
                    "nonebot_plugin_orm",
                    "nonebot_plugin_htmlrender",
                    "nonebot_plugin_apscheduler",
                    "nonebot_plugin_userinfo",
                    "nonebot_plugin_saa",
                    "nonebot_plugin_cesaa",
                    "nonebot_plugin_waiter",
                    "nonebot_plugin_chatrecorder",
                    "nonebot_plugin_session",
                    "nonebot_plugin_localstore",
                    "uniseg",
                    "nonebot_plugin_moellmchats",
                ],
            }
            self._write_config(self.model_config_file, default_config)
            return default_config

    def _write_config(self, file_path, config_data):
        # 将配置写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)

    def get_moe(self):
        # 获取当前是否使用MOE
        return self.model_config["use_moe"]

    def get_web_search(self):
        # 获取当前是否使用联网
        return self.model_config["use_web_search"]

    def get_use_tools(self):
        return self.model_config.get("use_tools", False)

    def get_model(self, key: str) -> dict:
        # 获取单个模型的配置
        selected_model = self.model_config.get(key)
        if selected_model and selected_model in self.models:
            return self.models[selected_model]

    def get_moe_current_model(self, difficulty: str) -> dict:
        # 获取当前MOE模型的配置
        moe_models = self.model_config["moe_models"]
        model_name = moe_models.get(difficulty)
        if model_name and model_name in self.models:
            # 返回模型的完整配置
            return self.models[model_name]

    def get_tool_blacklist(self) -> list:
        return self.model_config.get("tool_blacklist", [])

    def set_moe_model(self, model_query: str, difficulty: str) -> str:
        model_name = self.resolve_model_name(model_query)
        if not model_name:
            return f"找不到编号或名称为 '{model_query}' 的模型！\n" + self.get_formatted_model_list()

        if difficulty not in ["0", "1", "2"]:
            return "difficulty只能是 0、1、2 中的一个"

        self.model_config["moe_models"][difficulty] = model_name
        self._write_config(self.model_config_file, self.model_config)
        return f"已将难度 {difficulty} 的模型切换为 {model_name}"

    def set_web_search(self, is_web_search: bool = True) -> str:
        # 切换联网配置配置
        self.model_config["use_web_search"] = is_web_search
        self._write_config(self.model_config_file, self.model_config)
        return "已开启联网搜索" if is_web_search else "已关闭联网搜索"

    def set_category_model(self, model_name: str) -> str:
        # 设置分类模型，model_name为models.json中的键
        model_name = self.resolve_model_name(model_name)
        if not model_name:
            return f"模型 '{model_name}' 不存在！" + self.get_formatted_model_list()

        # 设置 category_model
        self.model_config["category_model"] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换分类模型为{model_name}的{self.models[model_name]['model']}"

    def set_use_tools(self, is_use_tools: bool = True) -> str:
        self.model_config["use_tools"] = is_use_tools
        self._write_config(self.model_config_file, self.model_config)
        return "已开启函数调用(Tools)" if is_use_tools else "已关闭函数调用(Tools)"

    def manage_tool_blacklist(self, action: str, plugin_name: str) -> str:
        blacklist = self.model_config.setdefault("tool_blacklist", [])
        if action == "add":
            if plugin_name not in blacklist:
                blacklist.append(plugin_name)
                self._write_config(self.model_config_file, self.model_config)
                return f"已将 {plugin_name} 加入工具黑名单"
            return "该插件已在黑名单中"
        elif action == "remove":
            if plugin_name in blacklist:
                blacklist.remove(plugin_name)
                self._write_config(self.model_config_file, self.model_config)
                return f"已将 {plugin_name} 从工具黑名单移除"
            return "该插件不在黑名单中"
        return "无效操作"

    def set_summary_model(self, model_name: str) -> str:
        # 设置单个模型，model_name为models.json中的键
        model_name = self.resolve_model_name(model_name)
        if not model_name:
            return f"模型 '{model_name}' 不存在！" + self.get_formatted_model_list()

        # 设置selected_model
        self.model_config["summary_model"] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换总结模型为{model_name}的{self.models[model_name]['model']}"

    def set_chat_model(self, model_name: str) -> str:
        # 使用解析器处理编号或名称
        model_name = self.resolve_model_name(model_name)
        if not model_name:
            return f"找不到编号或名称为 '{model_name}' 的模型！\n" + self.get_formatted_model_list()

        self.model_config["selected_model"] = model_name
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换聊天模型为 {model_name}"

    def set_moe_model(self, model_name: str, difficulty: str) -> str:
        # 设置MOE模型，model_name为models.json中的键，difficulty为0、1或2
        model_name = self.resolve_model_name(model_name)
        if not model_name:
            return f"模型 '{model_name}' 不存在！" + self.get_formatted_model_list()

        if difficulty not in ["0", "1", "2"]:
            return "difficulty只能是0、1、2中的一个"

        # 更新MOE模型配置
        self.model_config["moe_models"][difficulty] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已将{difficulty}的模型切换为{model_name}的{self.models[model_name]['model']}"

    def set_vision_model(self, model_name: str) -> str:
        # 设置视觉专用模型，model_name为models.json中的键
        model_name = self.resolve_model_name(model_name)
        if not model_name:
            return f"模型 '{model_name}' 不存在！" + self.get_formatted_model_list()

        # 设置 vision_model
        self.model_config["vision_model"] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换视觉模型为{model_name}的{self.models[model_name]['model']}"

    def get_formatted_model_list(self, provider_filter: str = None) -> str:
        """获取美化后的可用模型列表，带有序号编号"""
        sorted_names = self.get_sorted_model_names()
        grouped_models = defaultdict(list)
        
        # 记录每个模型及其全局唯一序号 (编号从1开始)
        for index, mid in enumerate(sorted_names, 1):
            info = self.models[mid]
            grouped_models[info.get("provider", "未知")].append((index, mid))

        if provider_filter:
            if provider_filter not in grouped_models:
                return f"未找到名为 '{provider_filter}' 的供应商。\n当前存在的供应商有：{', '.join(grouped_models.keys())}"
            grouped_models = {provider_filter: grouped_models[provider_filter]}

        lines = ["✨ 当前可用模型列表 ✨"]
        for provider, m_list in grouped_models.items():
            lines.append(f"🔸 【{provider}】")
            # 格式化为 [1] model-a, [2] model-b
            formatted_items = [f"[{idx}] {m}" for idx, m in m_list]
            lines.append("  " + ", ".join(formatted_items))

        return "\n".join(lines)

    def get_sorted_model_names(self) -> list:
        """获取稳定排序的模型列表（按供应商和模型名排序），确保全局编号稳定"""
        return sorted(
            self.models.keys(),
            key=lambda k: (self.models[k].get("provider", "未知"), k),
        )

    def resolve_model_name(self, query: str) -> str:
        """将用户输入的编号或名称解析为实际模型名"""
        if not query:
            return None
        # 精确匹配名称
        if query in self.models:
            return query
        # 匹配纯数字编号
        if query.isdigit():
            idx = int(query) - 1
            sorted_names = self.get_sorted_model_names()
            if 0 <= idx < len(sorted_names):
                return sorted_names[idx]
        return None


model_selector = ModelSelector()
