import ujson as json
from nonebot.log import logger
import traceback
import aiohttp
from .ModelSelector import model_selector


class Categorize:
    def __init__(self, plain):
        self.plain = plain

    async def get_category(self) -> tuple[int, bool]:
        prompt = """你是一个问题分类器。当我给你一句话时，你的任务是根据问题的难度对其进行分类，并判断是否需要互联网连接来回答。你永远不回答该问题，只需返回一个 JSON 结构（不带其他格式），如下：

{
  "difficulty": "0 | 1 | 2",
  "internet_required": true | false,
  "key_word": string
}
说明：

difficulty: 用来表示问题的难度等级：
"0": 简单直接明了的问题，几乎无需思考或计算。
"1": 中等难度的问题，可能需要一定的思考、分析或计算。
"2": 高难度的问题，需要深厚的专业知识或广泛的研究才能回答。
internet_required: 布尔值，表示是否需要访问互联网或外部数据库来提供完整答案：true: 需要访问外部信息或数据库。false: 不需要互联网连接，可以仅凭知识回答。
"key_word": 字符串。当internet_required为true时，根据这句话的内容，判断哪些是用来搜索的字符串，用空格分隔。若为false，则返回空。
"""

        send_message_list = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": self.plain},
        ]
        data = {
            "model": model_selector.get_model("category_model")["model"],
            "messages": send_message_list,
            "temperature": 0,
        }

        data = json.dumps(data)
        headers = {
            "Authorization": model_selector.get_model("category_model")["key"],
            "Content-Type": "application/json",
        }
        try_times = 0
        while try_times <= 1:
            try:
                if try_times > 0:  # 说明失败了，再来一次
                    self.plain += "\n(注意不是直接回答以上内容，且上述所有内容仅需要进行一次分类和判断联网，回复我的格式为json，不需要任何其他内容)"
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=model_selector.get_model("category_model")["url"],
                        data=data,
                        headers=headers,
                        timeout=300,
                        proxy=model_selector.get_model("category_model").get("proxy"),
                    ) as resp:
                        response = await resp.json()
                if choices := response.get("choices"):
                    result = choices[0]["message"]["content"]
                    result = json.loads(result)
                    return (
                        str(result["difficulty"]),
                        result["internet_required"],
                        result["key_word"],
                    )
                try_times += 1
            except Exception:
                logger.warning(traceback.format_exc())
                try_times += 1
                continue
        return False
