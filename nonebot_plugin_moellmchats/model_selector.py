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
import random

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
        self.global_default = {}
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

    def _get_random_key(self, model_data: dict) -> str:
        """Key 提取与随机方法，保证只返回字符串"""
        pool = []
        # 1. 提取新版规范的 keys 列表
        if "keys" in model_data and isinstance(model_data["keys"], list):
            pool.extend(model_data["keys"])
        # 2. 提取旧版的 key 字段（兼容老 JSON 中误写的 list 或 单个字符串）
        if "key" in model_data:
            raw_key = model_data["key"]
            if isinstance(raw_key, list):
                pool.extend(raw_key)
            elif isinstance(raw_key, str) and raw_key.strip():
                pool.append(raw_key)
        # 3. 过滤空值并去重
        valid_keys = list(set([k for k in pool if k]))
        if not valid_keys:
            return ""
        # 随机抽取并严格确保拥有 Bearer 前缀
        chosen_key = random.choice(valid_keys)
        return self._normalize_key(chosen_key)

    def load_providers(self):
        """初始化并读取 providers.toml，如果不存在则自动生成模板"""
        if not self.providers_file.exists():
            template = """# AI服务商配置文件
# base_url: 基础API地址（直接写Base URL即可，程序会自动补全 /chat/completions 及 /models）
# api_key: 你的API密钥（无需手动写 Bearer ，程序会自动补全）。支持填入单个字符串，或字符串列表实现随机轮询，如 ["sk-key1", "sk-key2"]
# proxy: [可选] 该服务商的全局代理
# models: [可选] 手动补充的模型列表。若API不支持 /models 自动获取，或获取不全时可在这里手动指定作为补充。

# 【全局默认配置】（所有供应商的所有模型均默认继承此设置，垫底优先级）
[global_default]
stream = true
is_segment = true
max_segments = 5

[providers.deepseek]
base_url = "https://api.deepseek.com"
api_key = "sk-xxxxxx"
models = ["deepseek-chat", "deepseek-reasoner"]

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-xxxxxx"
proxy = "http://127.0.0.1:7890"

# ====================================================
# 【高级参数配置】支持四级继承：全局默认 < 供应商默认 < 分组配置 < 独立配置
# 以下参数设置将自动合并到模型最终的配置字典中
# ====================================================

# 【用法一：供应商默认配置】（覆盖 global_default）
# 该供应商下*所有*拉取到或手动填写的模型，均默认应用此配置
[providers.openai.default_config]
temperature = 1.0

# 【用法二：批量分组配置】（覆盖前两项）
# 将需要相同配置的模型名称放入 models 数组，统一应用参数
[[providers.openai.config_groups]]
models = ["gpt-4o", "gpt-4-turbo"]
temperature = 1.2
max_segments = 5

# 【用法三：单模型独立配置】（最高优先级，覆盖一切）
[providers.openai.model_configs."o1-preview"]
stream = false  # 不支持流式的模型单独关闭
json_mode = true  # <-- 可在此自定义json结构化输出配置，以方便分类模型使用。聊天模型不影响
"""
            self.providers_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.providers_file, "w", encoding="utf-8") as f:
                f.write(template)

        try:
            with open(self.providers_file, "rb") as f:
                config = tomllib.load(f)
                self.providers = config.get("providers", {})
                self.global_default = config.get("global_default", {})
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
                        if "key" in info:
                            raw_key = info["key"]
                            info["keys"] = (
                                [raw_key] if not isinstance(raw_key, list) else raw_key
                            )
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
            raw_key = p_info.get("api_key", "")
            if isinstance(raw_key, list):
                keys = [self._normalize_key(k) for k in raw_key if k]
            else:
                keys = [self._normalize_key(raw_key)] if raw_key else []
            proxy = p_info.get("proxy")
            # 获取三种不同级别的配置
            default_config = p_info.get("default_config", {})  # 1. 供应商全局默认配置
            config_groups = p_info.get("config_groups", [])    # 2. 分组批量配置
            model_configs = p_info.get("model_configs", {})    # 3. 单模型独立配置

            # 收集该供应商下的所有模型ID（去重）
            m_ids = set()
            if provider in cached_models:
                m_ids.update(cached_models[provider])
            m_ids.update(p_info.get("models", []))
            m_ids.update(model_configs.keys())  # 只要写了独立配置的，也隐式作为模型加入
            # 将分组配置里的模型名也加入集合
            for group in config_groups:
                m_ids.update(group.get("models", []))
            for mid in m_ids:
                # 基础信息
                model_data = {
                    "url": url,
                    "key": keys,
                    "model": mid,
                    "provider": provider,
                }
                # 服务商级别的配置
                if proxy:
                    model_data["proxy"] = proxy

                # 优先级 0：应用【全局】默认配置（垫底，所有供应商的底座）
                if getattr(self, "global_default", None):
                    model_data.update(self.global_default)
                
                # 优先级 1：应用【当前供应商】默认配置（覆盖全局）
                if default_config:
                    model_data.update(default_config)
                    
                # 优先级 2：应用【分组】配置（覆盖供应商和全局）
                for group in config_groups:
                    if mid in group.get("models", []):
                        g_conf = {k: v for k, v in group.items() if k != "models"}
                        model_data.update(g_conf)
                
                # 优先级 3：应用【单模型】独立配置（最高优先级，绝对统治）
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
                raw_key = info.get("api_key")
                proxy = info.get("proxy")  # 获取服务商配置的代理

                if not base_url or not raw_key:
                    continue

                # 如果是列表则随机抽选一个，否则直接使用
                api_key = (
                    random.choice(raw_key) if isinstance(raw_key, list) else raw_key
                )

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
            model_data = self.models[selected_model].copy()
            model_data["key"] = self._get_random_key(model_data)
            return model_data
        return None

    def get_moe_current_model(self, difficulty: str) -> dict:
        # 获取当前MOE模型的配置
        moe_models = self.model_config["moe_models"]
        model_name = moe_models.get(difficulty)
        if model_name and model_name in self.models:
            model_data = self.models[model_name].copy()
            model_data["key"] = self._get_random_key(model_data)
            return model_data
        return None

    def get_tool_blacklist(self) -> list:
        return self.model_config.get("tool_blacklist", [])

    def set_moe_model(self, model_query: str, difficulty: str) -> str:
        model_name = self.resolve_model_name(model_query)
        if not model_name:
            return (
                f"找不到编号或名称为 '{model_query}' 的模型！\n"
                + self.get_formatted_model_list()
            )

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

    def set_chat_model(self, model_name: str) -> str:
        # 使用解析器处理编号或名称
        model_name = self.resolve_model_name(model_name)
        if not model_name:
            return (
                f"找不到编号或名称为 '{model_name}' 的模型！\n"
                + self.get_formatted_model_list()
            )

        self.model_config["selected_model"] = model_name
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换聊天模型为 {model_name}"

    def set_moe(self, is_moe: bool = True) -> str:
        """控制是否启用混合专家模型调度 (MoE)"""
        self.model_config["use_moe"] = is_moe
        self._write_config(self.model_config_file, self.model_config)
        return "✅ 已开启 MoE 混合调度" if is_moe else "❌ 已关闭 MoE 混合调度"

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

    def set_summary_model(self, model_query: str) -> str:
        # 使用解析器处理编号或名称
        model_name = self.resolve_model_name(model_query)
        if not model_name:
            return (
                f"找不到编号或名称为 '{model_query}' 的模型！\n"
                + self.get_formatted_model_list()
            )

        # 设置 summary_model
        self.model_config["summary_model"] = model_name
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换总结模型为 {model_name}"

    def get_formatted_model_list(self, search_query: str = None) -> str:
        """获取美化后的可用模型列表，支持多关键词模糊搜索"""
        sorted_names = self.get_sorted_model_names()
        grouped_models = defaultdict(list)

        # 解析搜索词：转小写并按空格拆分为多个关键词
        keywords = search_query.lower().split() if search_query else []

        for index, mid in enumerate(sorted_names, 1):
            info = self.models[mid]
            provider = info.get("provider", "未知")
            
            # 如果存在关键词，则必须【所有】关键词都出现在供应商名或模型名中才保留
            if keywords:
                search_target = f"{provider} {mid}".lower()
                if not all(kw in search_target for kw in keywords):
                    continue

            grouped_models[provider].append((index, mid))

        if not grouped_models:
            return f"没有找到包含关键词 '{search_query}' 的模型或供应商哦~"

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
