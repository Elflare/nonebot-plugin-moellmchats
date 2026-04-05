import ujson as json
import aiohttp
import asyncio
import traceback
from asyncio import TimeoutError
from nonebot.log import logger
from collections import defaultdict, deque
from .categorize import Categorize
from .search import Search
from .model_selector import model_selector
from .messages_handler import MessagesHandler
from .config import config_parser
from .temperament_manager import temperament_manager
from .utils import get_emotions_names, get_emotion, parse_emotion
import random
from .tool_manager import tool_manager
from .event_simulator import event_simulator
import inspect

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
                return "图片或内容可能包含敏感信息"
            logger.warning(f'看看之前的对话记录：{self.format_message_dict}')
            return "API请求被拒绝 (400)，请检查后台日志。"
        return None

    async def stream_llm_chat(
        self, session, url, headers, data, proxy, is_segment=False
    ) -> tuple[bool, str, list]:
        # 流式响应内容
        buffer = []
        assistant_result = []  # 后处理助手回复
        punctuation_buffer = ""  # 存标点
        is_second_send = False  # 不是第一次发送
        async with session.post(
            url, headers=headers, json=data, proxy=proxy
        ) as response:
            if error_msg := await self._check_400_error(response):
                return False, error_msg, None
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
                    return True, "".join(assistant_result) + result, None
                elif is_second_send:
                    if not self.is_objective:
                        self.messages_handler.post_process("".join(assistant_result))
                    return True, "".join(assistant_result), None
            else:
                logger.warning(f"Warning: {response}")
        return False, "API请求异常", None

    async def none_stream_llm_chat(
        self, session, url, headers, data, proxy
    ) -> tuple[bool, str, list]:
        async with session.post(
            url=url, data=data, headers=headers, ssl=False, proxy=proxy
        ) as resp:
            if error_msg := await self._check_400_error(resp):
                return False, error_msg, None
            response = await resp.json()
            if resp.status != 200 or not response:
                logger.warning(response)
                return False, "API返回异常", None

        if choices := response.get("choices"):
            message = choices[0]["message"]
            if tool_calls := message.get("tool_calls"):
                return True, message.get("content", ""), tool_calls

            content = message.get("content", "")
            start_tag, end_tag = "<think>", "</think>"
            start, end = content.find(start_tag), content.find(end_tag)
            if start == -1 and end != -1:
                result = content[:start] + content[end + len(end_tag) :]
            elif start != -1 and end != -1:
                result = content[:start] + content[end + len(end_tag) :]
            else:
                result = content

            # 只返回解析好的文本，【不在这里发送，也不存上下文】
            return True, result.strip(), None
        else:
            logger.warning(response)
            return False, "API解析异常", None

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
        if (
            model_selector.get_moe()
            or model_selector.get_web_search()
            or model_selector.get_use_tools()
        ):
            category = Categorize(plain)
            category_result = await category.get_category()
            if isinstance(category_result, str):  # 如果是str，则拒绝回答
                return category_result
            if isinstance(category_result, tuple):  # 如果是tuple，则说明没有问题
                difficulty, vision_required, required_plugins = category_result
                logger.info(
                    f"难度：{difficulty}, 视觉：{vision_required}, 需要插件：{required_plugins}"
                )
                # 根据难度改key和url
                if model_selector.get_moe():  # moe
                    # 如果识别到需要调用插件，强行剥夺视觉判定。
                    # 因为大模型只需要生成指令文本，目标插件自己会去处理图片，避免视觉模型不支持tools
                    if vision_required and self.messages_handler.current_images:
                        vision_model_key = model_selector.model_config.get(
                            "vision_model"
                        )
                        if vision_model_key:
                            self.model_info = model_selector.get_model("vision_model")
                            logger.info(
                                f"触发视觉任务，切换至视觉模型: {self.model_info['model']}"
                            )
                        else:
                            logger.info(
                                "触发视觉任务，但未配置 vision_model 字段，退回普通模型"
                            )
                            self.model_info = model_selector.get_moe_current_model(
                                difficulty
                            )
                    else:
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
            # 替换最后的内容
            send_message_list[-1] = {
                "role": current_msg["role"],
                "content": vision_content
            }
        current_stream_flag = self.model_info.get("stream", False)
        data = {
            "model": self.model_info["model"],
            "messages": send_message_list,
            "max_tokens": self.model_info.get("max_tokens"),
            "temperature": self.model_info.get("temperature"),
            "top_p": self.model_info.get("top_p"),
        }
        tools_schema = []
        # 注入 Tool Schema 并动态修改流式开关
        current_required = (
            required_plugins
            if "required_plugins" in locals() and required_plugins
            else []
        )
        all_plugins_set = (
            set(current_required) | self.messages_handler.get_all_used_plugins()
        )
        # 如果本次即将调用或使用了联网搜索，且存在网页提取工具，则自动捆绑挂载 Schema，以备不时之需
        if (
            "web_search" in all_plugins_set
            and "extract_webpage" in tool_manager.custom_tools
        ):
            all_plugins_set.add("extract_webpage")
        all_plugins = list(all_plugins_set)
        if all_plugins:
            normal_plugins = [p for p in all_plugins if p != "web_search"]
            if model_selector.get_use_tools() and normal_plugins:
                tools_schema.extend(
                    tool_manager.get_tool_schema(normal_plugins, include_search=False)
                )
            if model_selector.get_web_search() and "web_search" in all_plugins:
                tools_schema.extend(
                    tool_manager.get_tool_schema([], include_search=True)
                )
        if tools_schema:
            data["tools"] = tools_schema
            # 在前面插入, 让llm发一段消息说自己去调用工具了
            send_message_list[0]["content"] += (
                "。特别注意：1. 同步执行：如果你需要调用工具，必须在本次回复的文本(content)中用简短的一句话说明你要做什么，并**在同一次回复中立刻发起工具调用(tool_calls)**！严禁只发提示文字而把工具调用留到下一次回复！2. 如果用户的请求包含多个步骤逻辑，你必须在获取到前置工具的结果后，**自动且连续地调用下一个工具**，直至彻底完成用户的所有要求。"
            )
            current_stream_flag = False
            logger.debug("检测到需要调用工具，已自动将本次请求切换为非流式")
            logger.debug("调用的插件详情：")
            logger.debug(json.dumps(tools_schema, indent=4, ensure_ascii=False))
        # 有的模型没有top_k
        if self.model_info.get("top_k"):
            data["top_k"] = self.model_info.get("top_k")
        # 最终写入决定的流式开关
        data["stream"] = current_stream_flag

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

            # 最多允许大模型连续调用轮次
            max_tool_rounds = config_parser.get_config("max_tool_rounds") or 3
            for tool_round in range(max_tool_rounds):
                result_text = ""
                success = False
                tool_calls = None

                for retry_times in range(max_retry_times):
                    if retry_times > 0:
                        await self.bot.send(
                            self.event,
                            f"api又卡了呐！第 {retry_times+1} 次尝试，请勿多次发送~",
                        )
                        await asyncio.sleep(2 ** (retry_times + 1))
                    try:
                        if current_stream_flag:
                            (
                                success,
                                result_text,
                                tool_calls,
                            ) = await self.stream_llm_chat(
                                session,
                                self.model_info["url"],
                                headers,
                                data,
                                self.model_info.get("proxy"),
                                self.model_info.get("is_segment"),
                            )
                        else:
                            (
                                success,
                                result_text,
                                tool_calls,
                            ) = await self.none_stream_llm_chat(
                                session,
                                self.model_info["url"],
                                headers,
                                json.dumps(data),
                                self.model_info.get("proxy"),
                            )
                        if success:
                            break
                    except TimeoutError:
                        result_text = "网络超时呐，多半是api反应太慢（"
                    except Exception:
                        logger.warning(str(send_message_list))
                        logger.error(traceback.format_exc())
                        continue

                if not success:
                    return result_text or "api寄！"
                # 有工具调用，则继续处理
                if tool_calls:
                    send_message_list.append(
                        {
                            "role": "assistant",
                            "content": result_text,
                            "tool_calls": tool_calls,
                        }
                    )
                    tool_memory_list = []  # 用于长期记忆的文字记录
                    for call in tool_calls:
                        func_name = call["function"]["name"]
                        # 记录该用户在当前轮次解锁了此工具，过期自动销毁
                        self.messages_handler.messages_entity.add_used_plugins(
                            {func_name}
                        )
                        try:
                            args = json.loads(call["function"]["arguments"])
                        except Exception:
                            args = {}
                        logger.info(f"准备执行函数: {func_name}，传入参数: {args}")
                        tool_result = "执行成功"
                        if func_name == "web_search":
                            query = args.get("query", "")
                            if not result_text:
                                await self.bot.send(self.event, f"正在搜索: {query}...")
                            else:
                                # 如果模型说话了，将其发送给用户
                                processed_text = await self.send_emotion_message(
                                    result_text
                                )

                            search_res = await Search(query).get_search()
                            tool_result = search_res if search_res else "未找到相关结果"

                        elif func_name in tool_manager.custom_tools:
                            if not result_text:
                                await self.bot.send(
                                    self.event, f"正在调用函数: {func_name}..."
                                )
                            else:
                                # 同理，模型说话了就发送
                                processed_text = await self.send_emotion_message(
                                    result_text
                                )
                            try:
                                func = tool_manager.custom_tools[func_name]["func"]
                                if inspect.iscoroutinefunction(func):
                                    res = await func(**args)
                                else:
                                    res = func(**args)
                                tool_result = str(res)
                            except Exception as e:
                                logger.error(traceback.format_exc())
                                tool_result = f"函数执行出错: {str(e)}"

                        else:
                            command = args.get("command", "")
                            # 等待真实执行并获取结果
                            tool_result = await event_simulator.dispatch_event(
                                self.bot, self.event, command
                            )

                        # 把工具返回结果塞入消息队列
                        send_message_list.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": tool_result,
                            }
                        )
                        tool_memory_list.append(
                            f"[系统隐藏记录：你刚才调用了工具 {func_name}，参数是 {args}，结果是：{tool_result[:200]}]"
                        )

                    self.current_tool_memory = "\n".join(tool_memory_list)
                    data["messages"] = send_message_list
                    continue  # 继续下一轮循环请求大模型，让它根据工具结果做总结

                # ===== 没有工具调用，正常结束 =====
                # 只有非流式且有结果时，才在这里进行发送
                if not current_stream_flag and result_text:
                    result_text = await self.send_emotion_message(result_text)

                # 统一保存上下文
                if not self.is_objective:
                    # 如果有工具调用记录，将其作为隐藏提示悄悄追加到大模型的记忆里（不会发给用户）
                    final_memory = result_text
                    if (
                        hasattr(self, "current_tool_memory")
                        and self.current_tool_memory
                    ):
                        final_memory = f"{self.current_tool_memory}\n{result_text}"
                    self.messages_handler.post_process(final_memory)

                # 成功必须返回 True，不能返回字符串，否则会被外层当做报错信息再次发出
                return True
