import aiohttp
import traceback
from nonebot.log import logger
from .Config import config_parser

class Search:
    def __init__(self, plain):
        self.plain = plain

    async def get_search(self) -> str:
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": config_parser.get_config('search_api'),
        }
        data = {
            "query": self.plain,
            "include_answer": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=data, ssl=False
                ) as response:
                    response_data = await response.json()
                    return response_data.get("answer")
        except Exception:
            logger.warning(traceback.format_exc())
            return None
