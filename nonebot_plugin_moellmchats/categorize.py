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
        prompt = f"""你是一个问题分类器。当我给你一句话时，你的任务是根据问题的难度对其进行分类，并判断是否需要视觉（注意在不提及搜图时，就单纯使用视觉而不是搜图工具）。你永远不回答该问题，只需返回一个 JSON 结构（不是md格式，有三对键值），如下：

{{
  "difficulty": "0 | 1 | 2",
  "vision_required": true | false,
  "required_plugins": []
}}
【参数说明】：
- difficulty: "0"(简单常识/闲聊)、"1"(中等逻辑/计算)、"2"(高难度专业/深度分析)。
- vision_required: 当用户输入中包含"[图片]"字样时必须为 true，否则为 false。
- required_plugins: 字符串列表。根据下方插件目录，判断是否必须调用插件（可以多个）。
【可用插件目录】：
{catalog}
注：不要自信：大模型没有时间观念，计算能力也有限。若有对应插件请提供。若没有，则保持[]。
【示例】：
用户输入："今天北京天气怎么样？"
输出：{{"difficulty": "0", "vision_required": false, "required_plugins": ["web_search"]}}

用户输入："你觉得人工智能会取代人类吗？"
输出：{{"difficulty": "2", "vision_required": false, "required_plugins": []}}

用户输入："[图片]帮我看看这个图里是什么"
输出：{{"difficulty": "1", "vision_required": true, "required_plugins": []}}
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
                raw_result = ""  # 专门用一个变量存原始文本，防止被覆盖
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
                    raw_result = choices[0]["message"]["content"]
                    
                    # 增加容错：去除大模型可能包裹的 markdown json 格式
                    clean_result = raw_result.strip()
                    if clean_result.startswith("```json"):
                        clean_result = clean_result[7:]
                    elif clean_result.startswith("```"):
                        clean_result = clean_result[3:]
                    if clean_result.endswith("```"):
                        clean_result = clean_result[:-3]
                    
                    result_dict = json.loads(clean_result.strip())
                    return (
                        str(result_dict["difficulty"]),
                        result_dict["vision_required"],
                        result_dict.get("required_plugins", []),
                    )
                elif (
                    response.get("code") == "DataInspectionFailed"
                    or "contentFilter" in response
                ):
                    logger.warning(response)
                    return "内容不合规，拒绝回答"
                    
            except Exception:
                logger.warning(traceback.format_exc())
                # 使用 f-string 来正确打印变量，并换行显示更清晰
                logger.warning(f"看看模型返回的啥：\n{raw_result}")
                continue
                
        return False
