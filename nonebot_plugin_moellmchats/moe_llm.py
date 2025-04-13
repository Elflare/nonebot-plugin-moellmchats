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
        self.model_info = None
        self.emotion_flag = False  # 判断本次对话是否发送表情包
        self.prompt = f"{temperament_manager.get_temperament_prompt(temperament)}。我的id是{ event.sender.card or event.sender.nickname}"
        if temperament != "ai助手":  # 不为ai助手才加上下文
            # 表情包
            if config_parser.get_config(
                "emotions_enabled"
            ) and random.random() < config_parser.get_config("emotion_rate"):
                self.emotion_flag = True
                emotion_prompt = f"。回复时根据回答内容，发送表情包，每次回复最多发一个表情包，格式为中括号+表情包名字，如：[表情包名字]。可选表情有{get_emotions_names()}"
            else:
                emotion_prompt = ""
            self.prompt += f"。现在你在一个qq群中,你只需回复我{emotion_prompt}。群里近期聊天内容，冒号前面是id，后面是内容：\n"
            # 去除群聊最新的对话，因为在用户的上下文中
            context_dict_ = list(context_dict[event.group_id])[:-1]
            self.prompt += "\n".join(context_dict_)

    # 处理和发送表情包
    async def send_emotion_message(self, content: str) -> str:
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
                    if not decoded.strip():
                        continue
                    json_data = json.loads(decoded)
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
                if not self.is_objective:
                    self.messages_handler.post_process(
                        "".join(assistant_result) + result
                    )
                if is_second_send:
                    await asyncio.sleep(2 + len(current_content) / 3)
                else:
                    is_second_send = True
                if result := result.strip():
                    await self.send_emotion_message(result)
                    return True
                elif is_second_send:
                    return None  # 前面有，最后一句没回复
            else:
                logger.error(f"Error: {response}")
        return "bug了呐，赶快叫机器人主人来修下吧~"

    async def none_stream_llm_chat(self, session, url, headers, data, proxy) -> bool:
        async with session.post(
            url=url,
            data=data,
            headers=headers,
            ssl=False,
            proxy=proxy,
        ) as resp:
            # 获取整个响应文本
            response = await resp.json()
            # 返回200
            if resp.status != 200 or not response:
                logger.error(response)
                return None
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
            if response.get("code") == "DataInspectionFailed":
                self.messages_handler.clrear_messages()
                return "消息合法检查未通过，少说血腥、暴力、色情的话呐~"
            elif response.get("code") == 50501:
                return None
            else:
                logger.error(response)
                return "bug了呐，赶快喊机器人主人来修一下吧~"
        if not self.is_objective:
            self.messages_handler.post_process(result.strip())
        self.bot.send(self.event, result.strip())
        return True

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
                difficulty, internet_required, key_word = category_result
                logger.info(
                    f"难度：{difficulty}, 是否联网：{internet_required}，搜索关键词：{key_word}"
                )
                # 判断是否联网
                if internet_required and model_selector.get_web_search():
                    search = Search(key_word)
                    await self.bot.send(self.event, "正在搜索，请稍等...")
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
                    self.model_info = model_selector.get_moe_current_model(difficulty)
        if not self.model_info:  # 分类失败或者不是用的moe
            self.model_info = model_selector.get_model("selected_model")
        logger.info(f"模型选择为：{self.model_info['model']}")
        send_message_list = self.messages_handler.get_send_message_list()
        send_message_list.insert(0, {"role": "system", "content": self.prompt})
        data = {
            "model": self.model_info["model"],
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
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            ) as session:
                result = "api寄！"
                max_retry_times = (
                    config_parser.get_config("max_retry_times")
                    if config_parser.get_config("max_retry_times")
                    else 3
                )
                for retry_times in range(max_retry_times):
                    if retry_times > 0:
                        await self.bot.send(
                            self.event,
                            f"api又卡了呐！第 {retry_times+1} 次尝试，请勿多次发送~",
                        )
                        await asyncio.sleep(2**retry_times)
                    if self.model_info.get("stream"):
                        result = await self.stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            data,
                            self.model_info.get("proxy"),
                            self.model_info.get("is_segment"),
                        )
                        return result
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
                        return result
                    else:
                        continue
        except TimeoutError:
            return "网络超时呐，多半是api反应太慢（"
        except Exception:
            logger.error(str(send_message_list))
            traceback.print_exc()
            return "日常抽风呐！当然也有可能是请求太快了，慢点吧~"
