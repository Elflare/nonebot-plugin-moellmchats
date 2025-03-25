import ujson as json
from pathlib import Path

import nonebot_plugin_localstore as store

config_path: Path = store.get_plugin_config_dir()


class ConfigParser:
    def __init__(self):
        self.filepath = Path(config_path / "config.json")
        self.config = self.parse_config()

    def parse_config(self):
        """Parse the JSON configuration file and return the configuration dictionary."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as file:
                config = json.load(file)
            return config
        except FileNotFoundError:
            # Create a new configuration file with default values
            config = {
                "max_group_history": 10,
                "max_user_history": 8,
                "max_retry_times": 3,
                "user_history_expire_seconds": 600,
                "cd_seconds": 120,
                "search_api": "your api",
                "fastai_enabled": False,
            }
            with open(self.filepath, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=4, ensure_ascii=False)
            return config

    def get_config(self, key):
        """Get the value of a configuration item by key."""
        return self.config.get(key)


config_parser = ConfigParser()
