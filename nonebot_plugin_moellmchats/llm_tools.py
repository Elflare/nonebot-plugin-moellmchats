from collections import Counter
import inspect
import traceback

from nonebot.log import logger
import ujson as json

from .event_simulator import event_simulator
from .search import Search
from .tool_manager import tool_manager
from .utils import parse_emotion


class LlmToolsMixin:
    async def _execute_tools(
        self,
        tool_calls: list,
        result_text: str,
        send_message_list: list,
        reasoning_content: str,
    ) -> list:
        """执行工具调用，并更新消息列表"""
        for call in tool_calls:
            if (
                not call.get("function", {}).get("arguments")
                or not str(call["function"]["arguments"]).strip()
            ):
                call["function"]["arguments"] = "{}"

        content_for_history = str(result_text) if result_text else ""
        if self.emotion_flag and content_for_history:
            content_for_history, _ = parse_emotion(content_for_history)
        # 提取本次调用的所有工具名称
        called_func_names = [
            call.get("function", {}).get("name", "未知插件") for call in tool_calls
        ]
        func_names_str = ", ".join(called_func_names)

        assistant_msg = {
            "role": "assistant",
            "content": content_for_history.strip()
            or f"（正在调用工具: {func_names_str}）",
            "tool_calls": tool_calls,
        }
        # 仅在有思维链且非空时附加
        if reasoning_content:
            assistant_msg["reasoning_content"] = reasoning_content
        send_message_list.append(assistant_msg)
        text_to_send = result_text  # 暂存大模型回复文本，防止多个插件时被重复发送
        for call in tool_calls:
            func_name = call["function"]["name"]
            if not hasattr(self, "_current_tool_usage"):
                self._current_tool_usage = Counter()
            self._current_tool_usage[func_name] += 1
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
                    # 依赖注入核心逻辑
                    sig = inspect.signature(func)
                    if "_tool_manager" in sig.parameters:
                        args["_tool_manager"] = tool_manager
                    # 注入 bot 和 event
                    if "_bot" in sig.parameters:
                        args["_bot"] = self.bot
                    if "_event" in sig.parameters:
                        args["_event"] = self.event
                    res = (
                        await func(**args)
                        if inspect.iscoroutinefunction(func)
                        else func(**args)
                    )

                    if isinstance(res, dict):
                        result_text = (
                            res.get("text")
                            or res.get("content")
                            or res.get("message")
                            or ""
                        )
                        result_images = (
                            res.get("images")
                            or res.get("image_urls")
                            or []
                        )

                        if isinstance(result_images, str):
                            result_images = [result_images]

                        result_images = [
                            img for img in result_images
                            if isinstance(img, str) and img.strip()
                        ]

                        if result_images:
                            self._pending_vision_images.extend(result_images)

                            if result_text:
                                tool_result = (
                                    f"函数执行返回结果：\n{result_text}\n\n"
                                    f"[系统提示]：该函数还返回了 {len(result_images)} 张图片。"
                                )
                            else:
                                tool_result = (
                                    f"函数执行完毕并返回了 {len(result_images)} 张图片。"
                                )
                        else:
                            tool_result = str(result_text) if result_text else str(res)
                    else:
                        tool_result = str(res)
                except Exception as e:
                    logger.error(traceback.format_exc())
                    tool_result = f"函数执行出错: {e!s}"
            else:
                if text_to_send:
                    await self.send_emotion_message(text_to_send)
                    text_to_send = ""
                else:
                    await self.bot.send(self.event, f"正在执行指令: {func_name}...")
                command = args.get("command", "")
                plugin_text, plugin_images = await event_simulator.dispatch_event(
                    self.bot,
                    self.event,
                    command,
                    self.format_message_dict,
                )
                _PLUGIN_SYSTEM_HINT = (
                    "\n\n[系统提示]：上述结果已对用户可见。注意：若执行不正确或者用户的原始请求需要多个步骤，"
                    "请务重试或者继续调用下一个工具！如果所有任务均已完成，请直接做一两句话的简短总结，"
                    "严禁重复上述已发送的结果。"
                )
                if plugin_images:
                    self._pending_vision_images.extend(plugin_images)
                    text_part = (
                        f"插件执行返回结果：\n{plugin_text}"
                        if plugin_text
                        else "插件执行完毕并返回了图片（见下方图片消息）"
                    )
                    tool_result = text_part + _PLUGIN_SYSTEM_HINT
                elif plugin_text:
                    tool_result = (
                        f"插件执行返回结果：\n{plugin_text}{_PLUGIN_SYSTEM_HINT}"
                    )
                else:
                    tool_result = "插件已执行，但未返回有效文本。[系统提示]：如果有后续操作，请继续调用下一个工具。"

            send_message_list.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result,
                }
            )

        # 将本 round 的工具消息（截断结果）追加到历史记录 entity，供下轮对话使用
        HISTORY_TOOL_RESULT_LIMIT = 300
        history_tool_calls = self._sanitize_tool_calls_for_history(tool_calls)

        history_msgs = [
            {
                "role": "assistant",
                "content": assistant_msg["content"],
                "tool_calls": history_tool_calls,
            }
        ]
        for call in tool_calls:
            tool_result_content = next(
                (
                    m["content"]
                    for m in reversed(send_message_list)
                    if m.get("role") == "tool" and m.get("tool_call_id") == call["id"]
                ),
                "",
            )
            history_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result_content[:HISTORY_TOOL_RESULT_LIMIT],
                }
            )
        self.messages_handler.messages_entity.tool_messages.extend(history_msgs)
        return send_message_list

    def _build_tool_limit_summary_prompt(self) -> str:
        return (
            "系统提示：工具自动调用轮次已达当前上限。请根据前序步骤收集到的工具结果，"
            "给出初步结论或阶段性总结。不要继续调用工具；如果任务未彻底完成，请直接在回复末尾主动询问用户是否需要继续执行。"
        )

    def _build_empty_tool_summary_fallback(self) -> str:
        tool_messages = self.messages_handler.messages_entity.tool_messages
        last_tool_result = next(
            (
                message.get("content", "")
                for message in reversed(tool_messages)
                if message.get("role") == "tool"
            ),
            "",
        )
        if last_tool_result:
            summary = last_tool_result[:200]
            suffix = "..." if len(last_tool_result) > 200 else ""
            return f"工具已经执行完毕，但模型没有返回总结。最后一次工具结果摘要：{summary}{suffix}"
        return "工具已经执行完毕，但模型没有返回总结。"

    async def _request_tool_summary_retry(
        self,
        session,
        headers: dict,
        send_message_list: list,
        timeout,
    ) -> str:
        retry_messages = list(send_message_list)
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    "系统提示：上一轮工具执行后你没有给出可见总结。请只基于已有工具结果，"
                    "用简短中文回复用户当前结论；不要继续调用工具。"
                ),
            }
        )
        retry_data = {
            "model": self.model_info["model"],
            "messages": retry_messages,
            "stream": False,
        }
        for key in ["max_tokens", "temperature", "top_p", "top_k"]:
            if self.model_info.get(key) is not None:
                retry_data[key] = self.model_info[key]
        if extra_payload := self.model_info.get("extra_payload"):
            if isinstance(extra_payload, dict):
                retry_data.update(extra_payload)

        success, summary_text, retry_tool_calls, _ = await self.none_stream_llm_chat(
            session,
            self.model_info["url"],
            headers,
            retry_data,
            self.model_info.get("proxy"),
            timeout,
        )
        if not success:
            return ""
        if retry_tool_calls:
            logger.warning("工具总结补救请求仍返回了 tool_calls，已忽略并使用兜底总结")
            return ""
        return (summary_text or "").strip()

