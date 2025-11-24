import ujson as json
import aiohttp
import asyncio
import traceback
from asyncio import TimeoutError
from nonebot.log import logger
from collections import defaultdict, deque
from .Categorize import Categorize
from .Search import Search
from .ModelSelector import model_selector
from .MessagesHandler import MessagesHandler
from .Config import config_parser
from .TemperamentManager import temperament_manager
from .utils import get_emotions_names, get_emotion, parse_emotion
import random

context_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_group_history"))
)


class MoeLlm:
    def __init__(
        self,
        bot,
        event,
        format_message_dict: dict,
        is_objective: bool = False,
        temperament="默认",
    ):
        self.bot = bot
        self.event = event
        self.format_message_dict = format_message_dict
        self.user_id = event.user_id
        self.is_objective = is_objective
        self.temperament = temperament
        self.model_info = {}
        self.emotion_flag = False  # 判断本次对话是否发送表情包
        self.prompt = f"{temperament_manager.get_temperament_prompt(temperament)}。我的id是{ event.sender.card or event.sender.nickname}"

    async def send_emotion_message(self, content: str) -> str:
        """处理和发送表情包
        Returns: str: 替换表情之后的内容
        """
        if self.emotion_flag:  # 本次对话发送表情包
            content, emotion_names_list = parse_emotion(content)
            if content:
                await self.bot.send(self.event, content)
            for emotion_name in emotion_names_list:
                # 发送
                if emotion := get_emotion(emotion_name):
                    await self.bot.send(self.event, emotion)
        else:  # 默认直接发送
            await self.bot.send(self.event, content)
        return content

    async def _check_400_error(self, response) -> str:
        """检查是否为400错误及敏感内容拦截，返回错误提示或None"""
        if response.status == 400:
            error_content = await response.text()
            logger.warning(f"API请求400错误: {error_content}")

            sensitive_keywords = [
                "DataInspectionFailed",  # 阿里
                "content_filter",  # OpenAI/Azure
                "sensitive",
                "safety",
                "violation",
                "audit",
                "prohibited",
            ]

            if any(k.lower() in error_content.lower() for k in sensitive_keywords):
                return "图片或内容可能包含敏感信息，被AI审核拦截了喵 >_<"
            return "API请求被拒绝 (400)，请检查后台日志。"
        return None

    async def stream_llm_chat(
        self, session, url, headers, data, proxy, is_segment=False
    ) -> bool:
        # 流式响应内容
        buffer = []
        assistant_result = []  # 后处理助手回复
        punctuation_buffer = ""  # 存标点
        is_second_send = False  # 不是第一次发送
        async with session.post(
            url, headers=headers, json=data, proxy=proxy
        ) as response:
            if error_msg := await self._check_400_error(response):
                return error_msg
            # 确保响应是成功的
            if response.status == 200:
                # 异步迭代响应内容
                MAX_SEGMENTS = self.model_info.get("max_segments", 5)
                current_segment = 0
                jump_out = False  # 判断是否跳出循环
                async for line in response.content:
                    if (
                        not line
                        or line.startswith(b"data: [DONE]")
                        or line.startswith(b"[DONE]")
                        or jump_out
                    ):
                        break  # 结束标记，退出循环e.content:
                    if line.startswith(b"data:"):
                        decoded = line[5:].decode("utf-8")
                    elif line.startswith(b""):
                        decoded = line.decode("utf-8")
                    if not decoded.strip() or decoded.startswith(":"):
                        continue
                    json_data = json.loads(decoded)
                    content = ""
                    # 以此尝试获取完整消息或流式增量
                    choices = json_data.get("choices", [{}])
                    if not choices:  # 防止choices为空列表的情况
                        continue
                    if message := json_data.get("choices", [{}])[0].get("message", {}):
                        content = message.get("content", "")
                    elif message := json_data.get("choices", [{}])[0].get("delta", {}):
                        content = message.get("content", "")
                    if content:
                        if is_segment and self.temperament != "ai助手":  # 分段
                            for char in content:
                                if char in ["。", "？", "！", "—", "\n"]:
                                    punctuation_buffer += char
                                else:
                                    if punctuation_buffer:
                                        # 发送累积的标点内容
                                        current_content = (
                                            "".join(buffer) + punctuation_buffer
                                        ).strip()
                                        if current_content.strip():
                                            if current_segment >= MAX_SEGMENTS:
                                                TOO_LANG = "太长了，不发了"
                                                buffer = [TOO_LANG]
                                                jump_out = True
                                                break
                                            if (
                                                is_second_send
                                            ):  # 第二次开始，会等几秒再发送
                                                await asyncio.sleep(
                                                    2 + len(current_content) / 3
                                                )
                                            else:
                                                is_second_send = True
                                            # 处理表情包和发送
                                            current_content = (
                                                await self.send_emotion_message(
                                                    current_content
                                                )
                                            )
                                            current_segment += 1
                                            assistant_result.append(current_content)
                                        buffer = []
                                        punctuation_buffer = ""
                                    buffer.append(char)
                        else:
                            buffer.append(content)
                # 最后的的句子或者没分段
                if jump_out:
                    result = "".join(buffer)
                else:
                    result = "".join(buffer) + punctuation_buffer
                if is_second_send:
                    await asyncio.sleep(2 + len(current_content) / 3)
                else:
                    is_second_send = True
                if result := result.strip():
                    result = await self.send_emotion_message(result)
                    if not self.is_objective:
                        self.messages_handler.post_process(
                            "".join(assistant_result) + result
                        )
                    return True
                elif is_second_send:
                    if not self.is_objective:
                        self.messages_handler.post_process("".join(assistant_result))
                    return True  # 前面有，最后一句没回复，也当回完了
            else:
                logger.warning(f"Warning: {response}")
        return False

    async def none_stream_llm_chat(self, session, url, headers, data, proxy) -> bool:
        async with session.post(
            url=url,
            data=data,
            headers=headers,
            ssl=False,
            proxy=proxy,
        ) as resp:
            if error_msg := await self._check_400_error(resp):
                return error_msg
            # 获取整个响应文本
            response = await resp.json()
            # 返回200
            if resp.status != 200 or not response:
                logger.warning(response)
                return False
        if choices := response.get("choices"):
            content = choices[0]["message"]["content"]
            start_tag = "<think>"
            end_tag = "</think>"
            start = content.find(start_tag)
            end = content.find(end_tag)
            if start == -1 and end != -1:
                end += len(end_tag)
                start = 0
                result = content[:start] + content[end:]
            elif start != -1 and end != -1:
                end += len(end_tag)
                result = content[:start] + content[end:]
            else:
                result = content
        else:
            logger.warning(response)
            return False
        if not self.is_objective:
            self.messages_handler.post_process(result.strip())
        await self.bot.send(self.event, result.strip())
        return True

    def prompt_handler(self):
        """处理system prompt，表情包和上下文相关"""
        if self.temperament != "ai助手":  # 不为ai助手才加上下文
            # 表情包
            if (
                config_parser.get_config("emotions_enabled")
                and self.model_info.get("is_segment")
                and self.model_info.get("stream")
                and random.random() < config_parser.get_config("emotion_rate")
            ):
                self.emotion_flag = True
                emotion_prompt = f"。回复时根据回答内容，发送表情包，每次回复最多发一个表情包，格式为中括号+表情包名字，如：[表情包名字]。可选表情有{get_emotions_names()}"
            else:
                emotion_prompt = ""
            self.prompt += f"。现在你在一个qq群中,你只需回复我{emotion_prompt}。群里近期聊天内容，冒号前面是id，后面是内容：\n"
            # 去除群聊最新的对话，因为在用户的上下文中
            context_dict_ = list(context_dict[self.event.group_id])[:-1]
            self.prompt += "\n".join(context_dict_)

    async def get_llm_chat(self) -> str:
        self.messages_handler = MessagesHandler(self.user_id)
        plain = self.messages_handler.pre_process(self.format_message_dict)
        # 获取难度和是否联网
        if model_selector.get_moe() or model_selector.get_web_search():
            category = Categorize(plain)
            category_result = await category.get_category()
            if isinstance(category_result, str):  # 如果是str，则拒绝回答
                return category_result
            if isinstance(category_result, tuple):  # 如果是tuple，则说明没有问题
                difficulty, internet_required, key_word, vision_required = (
                    category_result
                )
                logger.info(
                    f"难度：{difficulty}, 联网：{internet_required}, 关键词：{key_word}, 视觉：{vision_required}"
                )
                # 判断是否联网
                if internet_required and model_selector.get_web_search():
                    search = Search(key_word)
                    await self.bot.send(self.event, "検索中...検索中...=￣ω￣=")
                    if search_result := await search.get_search():
                        self.messages_handler.search_message_handler(search_result)
                    elif isinstance(search_result, bool):
                        await self.bot.send(self.event, "没搜到，可能没有相关内容")
                    else:
                        await self.bot.send(
                            self.event, "搜索失败，请检查日志输出"
                        )  # 搜索失败
                # 根据难度改key和url
                if model_selector.get_moe():  # moe
                    if vision_required and self.messages_handler.current_images:
                        # 强制路由到 vision_model (需要在 model_config.json 里配好 "vision_model": "xxx")
                        # 1. 尝试读取 vision_model 配置
                        # 使用 .get() 避免报错，如果不存在则为 None
                        vision_model_key = model_selector.model_config.get(
                            "vision_model"
                        )

                        # 2. 核心检查：如果用户没配这个字段
                        if vision_model_key:
                            # 3. 获取具体模型配置
                            self.model_info = model_selector.get_model("vision_model")
                            logger.info(
                                f"触发视觉任务，切换至视觉模型: {self.model_info['model']}"
                            )
                        else:
                            logger.info(
                                "触发视觉任务，但配置文件 model_config.json 缺少 vision_model 字段，退回普通模型"
                            )

                    else:
                        # 否则按原来的难度分级走
                        self.model_info = model_selector.get_moe_current_model(
                            difficulty
                        )
        if not self.model_info:  # 分类失败或者不是用的moe
            self.model_info = model_selector.get_model("selected_model")
        logger.info(f"模型选择为：{self.model_info['model']}")
        # 处理system prompt，表情包和上下文相关
        self.prompt_handler()
        send_message_list = self.messages_handler.get_send_message_list()
        send_message_list.insert(0, {"role": "system", "content": self.prompt})
        # === 多模态 Payload 构建逻辑 ===
        # 1. 检查模型是否配置了 is_vision
        # 2. 检查本次对话是否有图片
        if self.model_info.get("is_vision") and self.messages_handler.current_images:
            logger.info(
                f"检测到多模态模型 {self.model_info['model']} 且存在图片，正在构建多模态请求..."
            )

            # 获取最后一条消息（即当前用户的纯文本消息）
            current_msg = send_message_list[-1]
            # 构建符合 OpenAI Vision 格式的 content 列表
            vision_content = [{"type": "text", "text": current_msg["content"]}]
            for img_url in self.messages_handler.current_images:
                vision_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url
                            # "detail": "high" # 可选：部分模型需要指定精度
                        },
                    }
                )
            # 替换 Payload 中的 content
            send_message_list[-1]["content"] = vision_content
        data = {
            "model": self.model_info["model"],
            # "reasoning_effort": "none",
            "messages": send_message_list,
            "max_tokens": self.model_info.get("max_tokens"),
            "temperature": self.model_info.get("temperature"),
            "top_p": self.model_info.get("top_p"),
            "stream": self.model_info.get("stream", False),
            # "tools": [
            #     {
            #         "type": "web_search",
            #         "web_search": {"enable": True},
            #     }
            # ],
        }
        # 有的模型没有top_k
        if self.model_info.get("top_k"):
            data["top_k"] = self.model_info.get("top_k")

        headers = {
            "Authorization": self.model_info["key"],
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            max_retry_times = (
                config_parser.get_config("max_retry_times")
                if config_parser.get_config("max_retry_times")
                else 3
            )
            result = ""
            for retry_times in range(max_retry_times):
                if retry_times > 0:
                    await self.bot.send(
                        self.event,
                        f"api又卡了呐！第 {retry_times+1} 次尝试，请勿多次发送~",
                    )
                    await asyncio.sleep(2 ** (retry_times + 1))
                try:
                    if self.model_info.get("stream"):
                        result = await self.stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            data,
                            self.model_info.get("proxy"),
                            self.model_info.get("is_segment"),
                        )
                    else:
                        data = json.dumps(data)
                        result = await self.none_stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            data,
                            self.model_info.get("proxy"),
                        )
                    if result:
                        return result  # 正常返回从这里
                    else:  # 出错
                        continue
                except TimeoutError:
                    return "网络超时呐，多半是api反应太慢（"
                except Exception:
                    logger.warning(str(send_message_list))
                    logger.error(traceback.format_exc())
                    continue
            return "api寄！"
