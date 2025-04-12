from pathlib import Path
import ujson as json

import nonebot_plugin_localstore as store

config_path: Path = store.get_plugin_config_dir()


# 模型选择类
class ModelSelector:
    def __init__(self):
        # 配置文件路径
        self.models_file = Path(config_path / "models.json")
        self.model_config_file = Path(config_path / "model_config.json")

        # 加载配置文件
        self.models = self._load_models()
        self.model_config = self._load_model_config()

        # 初始化缓存

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
                    "temperature": 1.5,
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
                "use_web_search": False,
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

    def set_moe(self, is_moe: bool = True) -> str:
        # 切换MOE配置
        self.model_config["use_moe"] = is_moe
        self._write_config(self.model_config_file, self.model_config)
        return "已切换为moe" if is_moe else "取消moe"

    def set_web_search(self, is_web_search: bool = True) -> str:
        # 切换联网配置配置
        self.model_config["use_web_search"] = is_web_search
        self._write_config(self.model_config_file, self.model_config)
        return "已开启联网搜索" if is_web_search else "已关闭联网搜索"

    def set_summary_model(self, model_name: str) -> str:
        # 设置单个模型，model_name为models.json中的键
        if model_name not in self.models:
            return f"只能是{list(self.models.keys())}中的模型"

        # 设置selected_model
        self.model_config["summary_model"] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换总结模型为{model_name}的{self.models[model_name]['model']}"

    def set_chat_model(self, model_name: str) -> str:
        # 设置单个模型，model_name为models.json中的键
        if model_name not in self.models:
            return f"只能是{list(self.models.keys())}中的模型"

        # 设置selected_model
        self.model_config["selected_model"] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已切换聊天模型为{model_name}的{self.models[model_name]['model']}"

    def set_moe_model(self, model_name: str, difficulty: str) -> str:
        # 设置MOE模型，model_name为models.json中的键，difficulty为0、1或2
        if model_name not in self.models:
            return f"只能是{list(self.models.keys())}中的模型"

        if difficulty not in ["0", "1", "2"]:
            return "difficulty只能是0、1、2中的一个"

        # 更新MOE模型配置
        self.model_config["moe_models"][difficulty] = model_name

        # 更新配置文件
        self._write_config(self.model_config_file, self.model_config)
        return f"已将{difficulty}的模型切换为{model_name}的{self.models[model_name]['model']}"


model_selector = ModelSelector()
