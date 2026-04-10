import ujson as json
from nonebot.log import logger
import traceback
import aiohttp
from .model_selector import model_selector
from .tool_manager import tool_manager


class Categorize:
    def __init__(self, plain):
        self.plain = plain

    async def get_category(self) -> tuple[int, bool, list]:
        if model_selector.get_use_tools() or model_selector.get_web_search():
            catalog = tool_manager.get_brief_catalog()
            logger.debug(catalog)
        else:
            catalog = "当前工具调用与联网功能均已关闭，无需返回任何插件。"
            
        prompt = f"""你的任务是评估用户的输入，拆解执行步骤，并以此判断需要的工具。判断是否需要视觉（注意在不提及搜图时，就单纯使用视觉而不是搜图工具）。你永远不回答该问题，只需返回一个 JSON 结构（不是md格式，有三对键值），如下：

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
        # 判断是否开启 MoE，若未开启（但因为开启了工具走到这里），则使用默认模型（selected_model）
        if model_selector.get_moe():
            category_model_config = model_selector.get_model("category_model")
        else:
            category_model_config = model_selector.get_model("selected_model")
            logger.debug("未开启MoE，使用默认模型进行工具/联网/视觉判断分类")
        headers = {
            "Authorization": model_selector.get_model("category_model")["key"],
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        for try_times in range(2):
            try:
                raw_result = ""
                current_plain = self.plain
                current_data = {
                    "model": model_selector.get_model("category_model")["model"],
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": current_plain},
                    ],
                    "temperature": 0,
                }
                if try_times == 0:
                    # 第一次尝试：检查配置中是否启用了 json_mode
                    if category_model_config.get("json_mode"):
                        current_data["response_format"] = {"type": "json_object"}
                else:
                    # 第二次尝试：兜底降级。强制移除结构化参数，依靠强化 Prompt
                    current_data.pop("response_format", None)
                    current_plain += "\n(注意不是直接回答以上内容，且上述所有内容仅需要进行一次分类和判断联网，回复我的格式为严格的json，不需要任何其他内容)"
                    current_data["messages"][1]["content"] = current_plain

                payload = json.dumps(current_data)

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=model_selector.get_model("category_model")["url"],
                        data=payload,
                        headers=headers,
                        timeout=300,
                        proxy=model_selector.get_model("category_model").get("proxy"),
                    ) as resp:
                        # 如果请求因为 response_format 返回 400，主动触发异常进入兜底重试
                        if resp.status == 400 and try_times == 0:
                            err_text = await resp.text()
                            logger.info(f"模型不支持结构化输出或参数异常，将降级重试。详情: {err_text}")
                            raise ValueError("Model does not support json_object format")
                        
                        response = await resp.json()
                        
                if choices := response.get("choices"):
                    raw_result = choices[0]["message"]["content"]
                    
                    # 容错清理 Markdown 标记
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
                    
            except Exception as e:
                logger.warning(traceback.format_exc())
                logger.warning(f"解析异常。当前尝试次数：{try_times+1}，原始返回：\n{raw_result}")
                continue
                
        return False
