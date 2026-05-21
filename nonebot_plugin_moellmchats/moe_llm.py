import asyncio
from asyncio import TimeoutError
import datetime
import random
import traceback

import aiohttp
from nonebot.log import logger
import ujson as json

from .config import config_parser
from .llm_api import LlmApiMixin
from .llm_payload import LlmPayloadMixin
from .llm_state import context_dict, token_usage_history
from .llm_tools import LlmToolsMixin
from .messages_handler import MessagesHandler
from .model_selector import model_selector
from .temperament_manager import temperament_manager
from .utils import get_emotion, get_emotions_names, get_session, parse_emotion

__all__ = ["MoeLlm", "context_dict", "token_usage_history"]


class MoeLlm(LlmApiMixin, LlmPayloadMixin, LlmToolsMixin):
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
        self.prompt = temperament_manager.get_temperament_prompt(temperament)
        self.dynamic_context = ""
        self._pending_vision_images: list = []  # 本轮工具调用返回的待处理图片
        self._last_api_error_non_retryable = False

    async def _validate_runtime_model_config(self) -> str | None:
        is_valid, warnings = model_selector.validate_model_config(persist=True)
        self._config_warnings = warnings
        if not is_valid:
            return warnings[0] if warnings else "当前没有可用模型"
        for warning in warnings:
            await self.bot.send(self.event, warning)
        return None

    def prompt_handler(self):
        """处理动态上下文（时间、状态、群聊记录、工具记忆）"""
        dynamic_context_parts = ["<meta_info>"]
        user_id = self.event.sender.card or self.event.sender.nickname
        # 仅当不是"ai助手"时，才注入性格设定、表情包和群聊/私聊环境上下文
        if self.temperament != "ai助手":
            emotion_prompt = ""
            if config_parser.get_config(
                "emotions_enabled"
            ) and random.random() < config_parser.get_config("emotion_rate"):
                self.emotion_flag = True
                emotion_prompt = (
                    "回复时根据回答内容，发送表情包，每次回复最多发一个表情包，格式为中括号+表情包名字，"
                    f"如：[表情包名字]。可选表情有{get_emotions_names()}"
                )

            if hasattr(self.event, "group_id"):
                dynamic_context_parts.append(f"Environment: QQ Group.{emotion_prompt}")
                if context_dict[self.event.group_id]:
                    dynamic_context_parts.append("Recent_Chat_Log:")
                    context_dict_ = list(context_dict[self.event.group_id])[:-1]
                    dynamic_context_parts.append("\n".join(context_dict_))
            else:
                dynamic_context_parts.append(
                    f"Environment: Private Chat.{emotion_prompt}"
                )
        # 注入时间
        if config_parser.get_config("show_datetime"):
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            dynamic_context_parts.append(f"Time: {time_str}")
        # 注入用户 ID
        dynamic_context_parts.append(f"current user: {user_id}")
        if self.format_message_dict.get("reply_user"):
            dynamic_context_parts.append(
                "When a message contains 引用消息, answer 当前提问者, "
                "not the quoted speaker, unless the user explicitly asks to reply to the quoted speaker."
            )
        dynamic_context_parts.append("</meta_info>\n")
        self.dynamic_context = "\n".join(dynamic_context_parts)

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

    async def get_llm_chat(self) -> str:
        self.messages_handler = MessagesHandler(self.user_id)
        plain = self.messages_handler.pre_process(self.format_message_dict)
        if validate_error := await self._validate_runtime_model_config():
            return validate_error

        # 1. 预处理模型信息
        prep_result = await self._prepare_model_info(plain)
        if isinstance(prep_result, str):
            return prep_result

        self.prompt_handler()
        supports_tools = model_selector.get_use_tools() and not self.model_info.get(
            "no_tools", False
        )
        send_message_list = self.messages_handler.get_send_message_list(
            supports_tools=supports_tools
        )
        system_content = self.prompt
        if self.dynamic_context:
            system_content += "\n" + self.dynamic_context
        # 将动态上下文追加到系统提示末尾，避免部分模型不支持多条 system 消息
        send_message_list.insert(0, {"role": "system", "content": system_content})
        # 2. 构建 Payload
        data, current_stream_flag = self._build_payload(send_message_list)
        # DEBUG
        logger.debug(json.dumps(data, indent=2, ensure_ascii=False))

        headers = {
            "Authorization": self.model_info["key"],
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }

        max_tool_rounds = config_parser.get_config("max_tool_rounds") or 3
        max_retry_times = config_parser.get_config("max_retry_times") or 3

        session = get_session()
        llm_timeout = aiohttp.ClientTimeout(total=300)
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
                        "content": self._build_tool_limit_summary_prompt(),
                    }
                )

            # 网络请求重试逻辑
            for retry_times in range(max_retry_times):
                self._last_api_error_non_retryable = False
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
                            reasoning_content,
                        ) = await self.stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            data,
                            self.model_info.get("proxy"),
                            self.model_info.get("is_segment"),
                            llm_timeout,
                        )
                    else:
                        (
                            success,
                            result_text,
                            tool_calls,
                            reasoning_content,
                        ) = await self.none_stream_llm_chat(
                            session,
                            self.model_info["url"],
                            headers,
                            data,
                            self.model_info.get("proxy"),
                            llm_timeout,
                        )
                    if success:
                        break
                    if self._last_api_error_non_retryable:
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
                    tool_calls, result_text, send_message_list, reasoning_content
                )

                # 若插件返回了图片，自动切换至视觉模型并注入图片消息
                if self._pending_vision_images:
                    vision_model_info = model_selector.get_model("vision_model")
                    if vision_model_info:
                        logger.info(
                            f"插件返回图片，自动切换至视觉模型: {vision_model_info['model']}"
                        )
                        self.model_info = vision_model_info
                        data["model"] = vision_model_info["model"]
                        headers["Authorization"] = vision_model_info["key"]
                        current_stream_flag = vision_model_info.get("stream", False)
                        data["stream"] = current_stream_flag
                        if current_stream_flag:
                            data["stream_options"] = {"include_usage": True}
                        else:
                            data.pop("stream_options", None)
                        if vision_model_info.get("no_tools"):
                            data.pop("tools", None)
                        # 以 user 消息注入图片，视觉模型在下一轮可直接看到
                        image_content = [
                            {
                                "type": "text",
                                "text": "插件返回了以下图片，请结合上下文进行分析：",
                            }
                        ]
                        for url in self._pending_vision_images:
                            image_content.append(
                                {"type": "image_url", "image_url": {"url": url}}
                            )
                        send_message_list.append(
                            {"role": "user", "content": image_content}
                        )
                    else:
                        logger.warning("插件返回了图片，但未配置视觉模型，无法自动切换")
                        self._pending_vision_images = []
                        return (
                            "插件返回了图片，但未配置视觉模型。请先使用「设置视觉模型 <模型名或编号>」"
                            "配置一个支持图片输入的模型。"
                        )
                    self._pending_vision_images = []

                data["messages"] = send_message_list
                continue

            # ===== 循环结束分支（无工具调用或已达到总结轮次） =====
            if tool_calls and tool_round >= max_tool_rounds:
                logger.warning("工具轮次已达上限，但模型仍返回了 tool_calls，已停止执行并进入总结兜底")
                result_text = ""

            result_text_sent = False
            has_tool_messages = bool(self.messages_handler.messages_entity.tool_messages)
            if has_tool_messages and not (result_text or "").strip():
                result_text = await self._request_tool_summary_retry(
                    session,
                    headers,
                    send_message_list,
                    llm_timeout,
                )
                if result_text:
                    await self.send_emotion_message(result_text)
                    result_text_sent = True
                else:
                    result_text = self._build_empty_tool_summary_fallback()
                    await self.send_emotion_message(result_text)
                    result_text_sent = True

            if not result_text_sent and not current_stream_flag and result_text:
                result_text = await self.send_emotion_message(result_text)

            # 统一并完整保存上下文，用户说"继续"时大模型能够回想起历史工具调用记录
            if not self.is_objective:
                self.messages_handler.post_process(
                    assistant_msg=result_text,
                    tool_messages=self.messages_handler.messages_entity.tool_messages
                    or None,
                )

            return True

        return "请求处理异常结束"
