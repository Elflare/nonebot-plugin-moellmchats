import ujson as json
import re
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
import datetime

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

    async def _check_400_error(self, response, request_data=None) -> str:
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
            # 打印发往大模型的完整 JSON Payload
            if request_data is not None:
                if isinstance(request_data, str):
                    try:
                        request_data = json.loads(request_data)
                    except Exception:
                        pass
                logger.warning(
                    f"【导致400错误的完整Payload】:\n{json.dumps(request_data, ensure_ascii=False, indent=2)}"
                )
            else:
                logger.warning(f"看看之前的对话记录：{self.format_message_dict}")

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

        is_thinking = False  # 状态机：是否在思考中
        tag_buffer = ""  # 用于精准匹配标签切片

        async with session.post(url, headers=headers, json=data, proxy=proxy) as resp:
            if error_msg := await self._check_400_error(resp, request_data=data):
                return False, error_msg, None
            # 确保响应是成功的
            if resp.status == 200:
                # 异步迭代响应内容
                MAX_SEGMENTS = self.model_info.get("max_segments", 5)
                current_segment = 0
                jump_out = False  # 判断是否跳出循环
                async for line in resp.content:
                    if (
                        not line
                        or line.startswith(b"data: [DONE]")
                        or line.startswith(b"[DONE]")
                        or jump_out
                    ):
                        break  # 结束标记，退出循环

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
                        for char in content:
                            tag_buffer += char
                            if len(tag_buffer) > 15:
                                tag_buffer = tag_buffer[-15:]

                            # 拦截思考开始
                            if not is_thinking and (
                                tag_buffer.endswith("<thought>")
                                or tag_buffer.endswith("<think>")
                            ):
                                is_thinking = True
                                # 思考标签的字符已经漏进 buffer 了，给它揪出来删掉
                                tag_len = 9 if tag_buffer.endswith("<thought>") else 7
                                for _ in range(
                                    tag_len - 1
                                ):  # -1是因为当前char还没加进去
                                    if punctuation_buffer:
                                        punctuation_buffer = punctuation_buffer[:-1]
                                    elif buffer:
                                        buffer.pop()
                                continue

                            # 拦截思考结束
                            if is_thinking and (
                                tag_buffer.endswith("</thought>")
                                or tag_buffer.endswith("</think>")
                            ):
                                is_thinking = False
                                continue

                            # 如果处于思考状态，直接跳过当前字符，不加入 buffer
                            if is_thinking:
                                continue

                            # 如果是换行且前面刚好结束思考，忽略这个孤儿换行
                            if char == "\n" and tag_buffer.endswith(">"):
                                continue
                            if is_segment and self.temperament != "ai助手":  # 分段
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
                                # 注意这里是 append(char) 不是 append(content)
                                buffer.append(char)

                # 最后的的句子或者没分段
                if jump_out:
                    result = "".join(buffer)
                else:
                    result = "".join(buffer) + punctuation_buffer

                if is_second_send:
                    # 避免最后发空消息计算报错
                    current_len = (
                        len(current_content) if "current_content" in locals() else 0
                    )
                    await asyncio.sleep(2 + current_len / 3)
                else:
                    is_second_send = True

                if result := result.strip():
                    result = await self.send_emotion_message(result)
                    return True, "".join(assistant_result) + result, None
                elif is_second_send:
                    return True, "".join(assistant_result), None
            else:
                logger.warning(f"Warning: {resp}")
        return False, "API请求异常", None

    async def none_stream_llm_chat(
        self, session, url, headers, data, proxy
    ) -> tuple[bool, str, list]:
        async with session.post(
            url=url, data=data, headers=headers, ssl=False, proxy=proxy
        ) as resp:
            if error_msg := await self._check_400_error(resp, request_data=data):
                return False, error_msg, None
            response = await resp.json()
            if resp.status != 200 or not response:
                logger.warning(response)
                return False, "API返回异常", None

        if choices := response.get("choices"):
            message = choices[0]["message"]
            content = message.get("content", "")
            # 清理思考内容
            if content:
                content = re.sub(
                    r"<(think|thought)>.*?</\1>", "", content, flags=re.DOTALL
                )
                content = content.strip()

            # 清洗完再判断并返回 tool_calls
            if tool_calls := message.get("tool_calls"):
                return True, content, tool_calls

            return True, content, None
        else:
            logger.warning(response)
            return False, "API解析异常", None

    def prompt_handler(self):
        """处理system prompt，表情包和上下文相关"""
        # 注入时间
        if config_parser.get_config("show_datetime"):
            now = datetime.datetime.now()
            time_str = now.strftime('%Y年%m月%d日 %H:%M:%S')
            self.prompt = f"当前系统时间: {time_str}。" + self.prompt
        # 仅当不是“ai助手”时，才注入性格设定、表情包和群聊/私聊环境上下文
        if self.temperament != "ai助手":
            emotion_prompt = ""
            # 表情包逻辑
            if (
                config_parser.get_config("emotions_enabled")
                and random.random() < config_parser.get_config("emotion_rate")
            ):
                self.emotion_flag = True
                emotion_prompt = f"。回复时根据回答内容，发送表情包，每次回复最多发一个表情包，格式为中括号+表情包名字，如：[表情包名字]。可选表情有{get_emotions_names()}"

            # 环境与上下文逻辑
            if hasattr(self.event, "group_id"):
                self.prompt += f"。现在你在一个qq群中,你只需回复我{emotion_prompt}。群里近期聊天内容，冒号前面是id，后面是内容：\n"
                context_dict_ = list(context_dict[self.event.group_id])[:-1]
                self.prompt += "\n".join(context_dict_)
            else:
                self.prompt += emotion_prompt
        tool_memory_context = []
        for entity in self.messages_handler.messages_entity_list:
            if entity.tool_memory:
                tool_memory_context.append(entity.tool_memory)

        # 将所有历史工具记录按照时间顺序拼接，作为系统提示注入
        if tool_memory_context:
            self.prompt += "\n\n【系统提示：历史工具执行记录】\n" + "\n".join(
                tool_memory_context
            )

    async def _prepare_model_info(self, plain: str):
        """预处理：获取模型信息、处理难度分类与视觉判断"""
        self.required_plugins = []
        if (
            model_selector.get_moe()
            or model_selector.get_web_search()
            or model_selector.get_use_tools()
        ):
            category = Categorize(plain)
            category_result = await category.get_category()
            if isinstance(category_result, str):
                return category_result
            if isinstance(category_result, tuple):
                difficulty, vision_required, required_plugins = category_result
                logger.info(
                    f"难度：{difficulty}, 视觉：{vision_required}, 需要插件：{required_plugins}"
                )
                self.required_plugins = required_plugins
                # 无论是否开启MoE，只要触发视觉且有图片，优先走视觉模型
                if vision_required and self.messages_handler.current_images:
                    vision_model_key = model_selector.model_config.get("vision_model")
                    if vision_model_key:
                        self.model_info = model_selector.get_model("vision_model")
                        logger.info(
                            f"触发视觉任务，切换至视觉模型: {self.model_info['model']}"
                        )
                    else:
                        logger.info(
                            "触发视觉任务，但未配置 vision_model 字段，退回普通模型/MoE模型"
                        )
                        if model_selector.get_moe():
                            self.model_info = model_selector.get_moe_current_model(
                                difficulty
                            )
                # 纯文本任务，若开启了MoE，则分配对应难度的模型
                elif model_selector.get_moe():
                    self.model_info = model_selector.get_moe_current_model(difficulty)

        # 兜底：既没触发视觉，也没开启MoE，或者啥都没开启（原逻辑），使用默认模型
        if not hasattr(self, "model_info") or not self.model_info:
            self.model_info = model_selector.get_model("selected_model")
        logger.info(f"模型选择为：{self.model_info['model']}")
        return None

    def _build_payload(self, send_message_list: list) -> tuple[dict, bool]:
        """构建发给大模型的 payload 与工具 schema"""
        if self.model_info.get("is_vision") and self.messages_handler.current_images:
            logger.info(
                f"检测到多模态模型 {self.model_info['model']} 且存在图片，正在构建多模态请求..."
            )
            current_msg = send_message_list[-1]
            vision_content = [{"type": "text", "text": current_msg["content"]}]
            for img_url in self.messages_handler.current_images:
                vision_content.append(
                    {"type": "image_url", "image_url": {"url": img_url}}
                )
            send_message_list[-1] = {
                "role": current_msg["role"],
                "content": vision_content,
            }

        current_stream_flag = self.model_info.get("stream", False)
        data = {
            "model": self.model_info["model"],
            "messages": send_message_list,
            "max_tokens": self.model_info.get("max_tokens"),
            "temperature": self.model_info.get("temperature"),
            "top_p": self.model_info.get("top_p"),
        }
        if self.model_info.get("top_k"):
            data["top_k"] = self.model_info.get("top_k")

        tools_schema = []
        all_plugins_set = (
            set(getattr(self, "required_plugins", []))
            | self.messages_handler.get_all_used_plugins()
        )
        all_plugins_set = tool_manager.expand_dependencies(all_plugins_set)
        logger.debug(f"LLM 最终将要注入的插件集合: {all_plugins_set}")
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
            send_message_list[0]["content"] += (
                "。特别注意：1. 同步执行：如果你需要调用工具，必须在本次回复的文本(content)中用简短的一句话说明你要做什么，并**在同一次回复中立刻发起工具调用(tool_calls)**！2. 如果用户的请求包含多个步骤逻辑，你必须在获取到前置工具的结果后，**自动且连续地调用下一个工具**，直至彻底完成要求。"
            )
            current_stream_flag = False
            logger.debug("检测到需要调用工具，已自动将本次请求切换为非流式")
            logger.debug(f"实际发送给大模型的 tools_schema: {tools_schema}")
        data["stream"] = current_stream_flag
        return data, current_stream_flag

    async def _execute_tools(
        self, tool_calls: list, result_text: str, send_message_list: list
    ) -> list:
        """执行工具调用，并更新消息列表和本地隐藏记忆"""
        for call in tool_calls:
            if (
                not call.get("function", {}).get("arguments")
                or not str(call["function"]["arguments"]).strip()
            ):
                call["function"]["arguments"] = "{}"

        assistant_msg = {
            "role": "assistant",
            "content": str(result_text)
            if result_text and str(result_text).strip()
            else "（正在调用工具）",
            "tool_calls": tool_calls,
        }
        send_message_list.append(assistant_msg)
        tool_memory_list = []
        text_to_send = result_text  # 暂存大模型回复文本，防止多个插件时被重复发送
        for call in tool_calls:
            func_name = call["function"]["name"]
            self.messages_handler.messages_entity.add_used_plugins({func_name})
            try:
                args = json.loads(call["function"]["arguments"])
            except Exception:
                args = {}
            logger.info(f"准备执行函数: {func_name}，传入参数: {args}")

            tool_result = "执行成功"
            if func_name == "web_search":
                query = args.get("query", "")
                if text_to_send:
                    await self.send_emotion_message(text_to_send)
                    text_to_send = ""  # 消耗掉，防止下一个插件重发
                else:
                    await self.bot.send(self.event, f"正在搜索: {query}...")
                search_res = await Search(query).get_search()
                tool_result = search_res if search_res else "未找到相关结果"

            elif func_name in tool_manager.custom_tools:
                if text_to_send:
                    await self.send_emotion_message(text_to_send)
                    text_to_send = ""
                else:
                    await self.bot.send(self.event, f"正在调用函数: {func_name}...")
                try:
                    func = tool_manager.custom_tools[func_name]["func"]
                    res = (
                        await func(**args)
                        if inspect.iscoroutinefunction(func)
                        else func(**args)
                    )
                    tool_result = str(res)
                except Exception as e:
                    logger.error(traceback.format_exc())
                    tool_result = f"函数执行出错: {str(e)}"
            else:
                if text_to_send:
                    await self.send_emotion_message(text_to_send)
                    text_to_send = ""
                else:
                    await self.bot.send(self.event, f"正在执行指令: {func_name}...")
                command = args.get("command", "")
                tool_result = await event_simulator.dispatch_event(
                    self.bot, self.event, command
                )

            send_message_list.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result,
                }
            )
            tool_memory_list.append(
                f"【系统环境背景：工具 {func_name} 已执行，参数 {args}，结果：{tool_result[:200]}】"
            )

        if getattr(self, "current_tool_memory", ""):
            self.current_tool_memory += "\n"
        self.current_tool_memory += "\n".join(tool_memory_list)
        return send_message_list

    async def get_llm_chat(self) -> str:
        self.messages_handler = MessagesHandler(self.user_id)
        plain = self.messages_handler.pre_process(self.format_message_dict)

        # 1. 预处理模型信息
        prep_result = await self._prepare_model_info(plain)
        if isinstance(prep_result, str):
            return prep_result

        self.prompt_handler()
        send_message_list = self.messages_handler.get_send_message_list()
        send_message_list.insert(0, {"role": "system", "content": self.prompt})

        # 2. 构建 Payload
        data, current_stream_flag = self._build_payload(send_message_list)

        headers = {
            "Authorization": self.model_info["key"],
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

        self.current_tool_memory = ""
        max_tool_rounds = config_parser.get_config("max_tool_rounds") or 3
        max_retry_times = config_parser.get_config("max_retry_times") or 3

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            # 修改：增加上限至 max_tool_rounds + 1，以容纳最后一次强制总结
            for tool_round in range(max_tool_rounds + 1):
                result_text = ""
                success = False
                tool_calls = None

                # 达到最大轮次时，移除工具强制总结
                if tool_round == max_tool_rounds:
                    data.pop("tools", None)
                    current_stream_flag = data["stream"]
                    send_message_list.append(
                        {
                            "role": "user",
                            "content": "系统提示：工具自动调用次数已达当前轮次上限。请根据前序步骤收集到的隐藏记录信息，得出初步结论或阶段性总结。如果任务未彻底完成，请直接在回复末尾主动询问用户是否需要继续执行。",
                        }
                    )

                # 网络请求重试逻辑
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

                # 3. 执行工具调用（非总结轮次才执行）
                if tool_calls and tool_round < max_tool_rounds:
                    send_message_list = await self._execute_tools(
                        tool_calls, result_text, send_message_list
                    )
                    data["messages"] = send_message_list
                    continue

                # ===== 循环结束分支（无工具调用或已达到总结轮次） =====
                if not current_stream_flag and result_text:
                    result_text = await self.send_emotion_message(result_text)

                # 修改：统一并完整保存上下文，用户说“继续”时大模型能够回想起 current_tool_memory 里的内容
                if not self.is_objective:
                    self.messages_handler.post_process(
                        assistant_msg=result_text,
                        tool_memory=getattr(self, "current_tool_memory", ""),
                    )

                return True

        return "请求处理异常结束"
