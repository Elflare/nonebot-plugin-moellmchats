import aiohttp
import traceback
from nonebot.log import logger
from .config import config_parser
from .tool_manager import tool_manager
from .utils import get_session

class Search:
    def __init__(self, plain):
        self.plain = plain

    async def get_search(self) -> str:
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": config_parser.get_config("search_api"),
            "Accept-Encoding": "identity",
        }
        data = {
            "query": self.plain,
            "include_answer": True,
        }

        try:
            async with get_session().post(
                url, headers=headers, json=data, ssl=False
            ) as response:
                response_data = await response.json()

                answer = response_data.get("answer", "")
                results = response_data.get("results", [])

                if answer or results:
                    final_res = answer

                    # 动态检测：判断提取网页的自定义函数是否被用户加载
                    has_extractor = "extract_webpage" in tool_manager.custom_tools

                    # 只有在有搜索结果且用户安装了提取工具的情况下，才暴露 URL 和诱导提示词
                    if results and has_extractor:
                        source_list = [f"- {r.get('title', '未知')}: {r.get('url', '')}" for r in results[:3] if r.get('url')]
                        if source_list:
                            final_res += "\n\n参考来源URL(若需获取更详尽的内容，可调用 extract_webpage 工具读取以下链接)：\n" + "\n".join(source_list)
                    elif results and not has_extractor:
                        # 如果没有安装提取工具，但你想让大模型知道有这些来源，可以仅暴露标题不诱导调用工具
                        source_titles = [f"- {r.get('title', '未知')}" for r in results[:3] if r.get('title')]
                        if source_titles:
                            final_res += "\n\n参考来源：\n" + "\n".join(source_titles)

                    return final_res if final_res else "搜索成功，但未返回直接摘要。"
                else:
                    return False  # 没有相关内容
        except Exception:
            logger.warning(traceback.format_exc())
            return None  # 错误
