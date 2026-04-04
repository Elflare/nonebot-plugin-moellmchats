import ujson as json
from nonebot.log import logger
import traceback
import aiohttp
from .model_selector import model_selector
from .tool_manager import tool_manager

class Categorize:
    def __init__(self, plain):
        self.plain = plain

    async def get_category(self) -> tuple[int, bool]:
        # 修改判断条件，只要开启工具或联网之一，就提供 catalog
        if model_selector.get_use_tools() or model_selector.get_web_search():
            catalog = tool_manager.get_brief_catalog()
            logger.debug(catalog)
        else:
            catalog = "当前工具调用与联网功能均已关闭，无需返回任何插件。"
        prompt = f"""你是一个问题分类器。当我给你一句话时，你的任务是根据问题的难度对其进行分类，并判断是否需要视觉（注意在不提及搜图时，就单纯使用视觉而不是搜图工具）。你永远不回答该问题，只需返回一个 JSON 结构（不带其他格式，有三对键值），如下：

{{
  "difficulty": "0 | 1 | 2",
  "vision_required": true | false,
  "required_plugins": []
}}
说明：

difficulty: 用来表示问题的难度等级：
"0": 简单直接明了的问题，几乎无需思考或计算。
"1": 中等难度的问题，可能需要一定的思考、分析或计算。
"2": 高难度的问题，需要深厚的专业知识或广泛的研究才能回答。
vision_required: 布尔值。当输入中包含[图片]字样时为true，没发图则为false。
"required_plugins": 字符串列表。根据用户的意图，判断是否需要调用以下已有插件功能（可以多个）。如果需要，将插件标识填入列表，不需要则返回空列表[]。可用插件目录如下：{catalog}
注：
不要自信：大模型没有时间观念，计算能力也有限。若有对应插件请提供。
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
            "Accept-Encoding": "identity",
        }
        for try_times in range(2):
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
                        result["vision_required"],
                        result.get("required_plugins", [])
                    )
                elif (
                    response.get("code") == "DataInspectionFailed"
                    or "contentFilter" in response
                ):
                    logger.warning(response)
                    return "内容不合规，拒绝回答"
            except Exception:
                logger.warning(traceback.format_exc())
                continue
        return False
