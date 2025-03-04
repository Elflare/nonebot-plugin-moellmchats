import ujson as json
from traceback import format_exc
from nonebot.log import logger
from pathlib import Path

import nonebot_plugin_localstore as store

config_path: Path = store.get_plugin_config_dir()


# 性格设置类
class TemperamentManager:
    def __init__(self):
        self.temperament_config = Path(config_path / "temperament_config.json")
        self.temperaments_path = Path(config_path / "temperaments.json")
        self.temperaments = self.read_temperaments()
        self.temperament_dict = self.read_temperament()

    def get_temperament(self, qq=None) -> str:
        """根据qq获取每个群友的性格配置"""
        if qq:
            qq = str(qq)
            return self.temperament_dict.get(qq, "默认")
        return "默认"

    def get_temperaments_keys(self) -> list:
        return self.temperaments.keys()

    def get_all_temperaments(self) -> str:
        return json.dumps(self.temperaments, indent=4, ensure_ascii=False)

    def get_temperament_prompt(self, temperament: str) -> str:
        """根据性格获取提示语"""
        return self.temperaments.get(temperament, "你是ai助手。回答像真人且尽量简短，回复格式为@id content")

    def set_temperament_dict(self, qq, temperament) -> bool:
        """设置配置项的值"""
        qq = str(qq)
        self.temperament_dict[qq] = temperament
        return self.write_temperament(qq, temperament)

    # 读取文件
    def read_temperament(self) -> str:
        if not self.temperament_config.exists():
            self.temperament_config.parent.mkdir(parents=True, exist_ok=True)
            self.temperament_config.touch()
            with open(self.temperament_config, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            return {}
        try:
            with open(self.temperament_config, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.error(format_exc())
            return {}

    # 读取文件
    def read_temperaments(self) -> str:
        if not self.temperaments_path.exists():
            self.temperaments_path.parent.mkdir(parents=True, exist_ok=True)
            self.temperaments_path.touch()
            prompt = "你是ai助手。回答像真人且简短"
            with open(self.temperaments_path, "w", encoding="utf-8") as f:
                json.dump({"默认": prompt}, f, ensure_ascii=False, indent=4)
            return prompt
        try:
            with open(self.temperaments_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.error(format_exc())
            return prompt

    # 性格写入文件
    def write_temperament(self, qq: int, temperament: str) -> bool:
        if not self.temperament_config.exists():
            self.temperament_config.parent.mkdir(parents=True, exist_ok=True)

            self.temperament_config.touch()
        try:
            with open(self.temperament_config, "r+", encoding="utf-8") as f:
                if data := f.read():
                    dict_ = json.loads(data)
                    dict_[qq] = temperament
                else:
                    dict_ = {qq: temperament}
                f.seek(0)
                json.dump(dict_, f, ensure_ascii=False, indent=4)
                f.truncate()
                return True
        except Exception:
            logger.error(format_exc())
            return False


temperament_manager = TemperamentManager()
