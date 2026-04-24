import base64
import asyncio
import traceback
import re
from urllib.parse import urlparse, unquote
from contextvars import ContextVar
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    Message,
    MessageSegment,
    GroupMessageEvent,
    PrivateMessageEvent,
)
from nonebot.message import handle_event
import time
import random

# key -> {"messages": [], "original_id": int/str, "fake_id": int/str}
_intercepted_messages: dict[str, dict] = {}

_current_capture_key: ContextVar[str | None] = ContextVar("_current_capture_key", default=None)

# 防止热重载时重复 patch
if not hasattr(Bot, "_event_simulator_original_call_api"):
    Bot._event_simulator_original_call_api = Bot.call_api

_original_call_api = Bot._event_simulator_original_call_api

_SEND_ACTIONS = {"send_msg", "send_group_msg", "send_private_msg"}

_AT_PATTERN = re.compile(r"\[(at:(\d+)|at_all)\]")


def _rewrite_reply_id(message, original_id: int | str, fake_id: int | str) -> Message:
    """真正修正即将发出去的消息里的 reply id。"""
    msg = Message(message)
    for seg in msg:
        if seg.type == "reply" and str(seg.data.get("id")) == str(fake_id):
            seg.data["id"] = str(original_id)
    return msg


def _extract_send_data(message) -> dict:
    """从最终发送消息中提取文本和图片。"""
    norm_msg = Message(message)
    content_segments = []
    image_urls = []

    for seg in norm_msg:
        if seg.type == "image":
            raw = seg.data.get("url") or seg.data.get("file") or ""
            if not raw:
                continue

            if raw.startswith("base64://"):
                url = "data:image/jpeg;base64," + raw[9:]
            elif raw.startswith("file://"):
                try:
                    parsed = urlparse(raw)
                    local_path = unquote(parsed.path)
                    with open(local_path, "rb") as f:
                        url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
                except Exception:
                    logger.warning(f"读取本地图片失败，跳过: {raw}")
                    continue
            else:
                url = raw

            image_urls.append(url)
            content_segments.append("[图片]")

        elif seg.type == "text":
            content_segments.append(seg.data.get("text", ""))

        elif seg.type == "at":
            qq = seg.data.get("qq")
            content_segments.append(f"[提及:{qq}]")

        elif seg.type == "reply":
            # 不回显 reply CQ 码，避免污染模型上下文
            continue

        else:
            content_segments.append(str(seg))

    full_text = "".join(content_segments).strip()
    if len(full_text) > 2000:
        full_text = full_text[:2000] + "\n...[内容过长已截断]"

    return {"text": full_text, "images": image_urls}


def _build_fake_message(command_str: str, format_message_dict: dict | None = None) -> Message:
    """将占位符指令字符串转为真正的 Message。"""
    format_message_dict = format_message_dict or {}
    mentions = format_message_dict.get("mentions") or []
    reply_user = format_message_dict.get("reply_user") or {}

    fake_message = Message()
    last = 0

    for m in _AT_PATTERN.finditer(command_str):
        if m.start() > last:
            fake_message.append(MessageSegment.text(command_str[last : m.start()]))

        token = m.group(1)

        if token.startswith("at:"):
            idx = int(m.group(2))
            if idx == 0:
                qq = reply_user.get("qq")
                if qq:
                    fake_message.append(MessageSegment.at(int(qq)))
            else:
                mention_idx = idx - 1
                if 0 <= mention_idx < len(mentions):
                    qq = mentions[mention_idx].get("qq")
                    if qq:
                        fake_message.append(MessageSegment.at(int(qq)))

        elif token == "at_all":
            for item in mentions:
                qq = item.get("qq")
                if qq:
                    fake_message.append(MessageSegment.at(int(qq)))

        last = m.end()

    if last < len(command_str):
        fake_message.append(MessageSegment.text(command_str[last:]))

    return fake_message


async def _safe_call_api(self: Bot, action: str, **params):
    """拦截发送类 API：先修正真正要发出的消息，再捕获。"""
    capture_key = _current_capture_key.get()

    if capture_key is not None and action in _SEND_ACTIONS:
        context = _intercepted_messages.get(capture_key)
        message = params.get("message")

        if message and context:
            fixed_message = _rewrite_reply_id(
                message,
                original_id=context["original_id"],
                fake_id=context["fake_id"],
            )
            params["message"] = fixed_message

            data = _extract_send_data(fixed_message)
            context["messages"].append(data)

            logger.debug(
                f'拦截插件发送 [{action}] '
                f'文本: {data["text"][:50] or "(空)"} '
                f'图片数: {len(data["images"])} '
                f'原始ID: {context["original_id"]} fakeID: {context["fake_id"]}'
            )

    return await _original_call_api(self, action, **params)


if getattr(Bot.call_api, "__name__", "") != "_safe_call_api":
    Bot.call_api = _safe_call_api


class EventSimulator:
    @staticmethod
    async def dispatch_event(
        bot,
        original_event,
        command_str: str,
        format_message_dict: dict | None = None,
    ) -> tuple[str, list[str]]:
        """派发伪造事件并捕获插件回复，返回 (文本结果, 图片URL列表)。"""
        if not command_str:
            logger.warning("大模型生成的指令文本为空，不投递事件")
            return "执行失败：指令参数为空", []

        try:
            fake_message = _build_fake_message(command_str, format_message_dict)

            # 保留原消息图片透传
            if hasattr(original_event, "message"):
                for seg in original_event.message:
                    if seg.type == "image":
                        fake_message.append(seg)

            # 当前 fake_event 的消息 id，必须是 int，且尽量避免与原消息重复
            fake_id = int(time.time_ns() % 10**15) + random.randint(1000, 9999)
            if fake_id == int(original_event.message_id):
                fake_id += 1

            kwargs = {
                "time": original_event.time,
                "self_id": original_event.self_id,
                "post_type": original_event.post_type,
                "sub_type": original_event.sub_type,
                "user_id": original_event.user_id,
                "message_type": original_event.message_type,
                "message_id": fake_id,
                "message": fake_message,
                "raw_message": str(fake_message),
                "font": getattr(original_event, "font", 0),
                "sender": original_event.sender,
            }

            original_reply = getattr(original_event, "reply", None)
            if original_reply is not None:
                kwargs["reply"] = original_reply

            if isinstance(original_event, GroupMessageEvent):
                kwargs["group_id"] = original_event.group_id
                fake_event = GroupMessageEvent(**kwargs)
            elif isinstance(original_event, PrivateMessageEvent):
                fake_event = PrivateMessageEvent(**kwargs)
            else:
                return f"不支持的事件类型模拟: {type(original_event)}", []

            logger.info(f"LLM 事件模拟投递 -> 指令: [{command_str}]")

            capture_key = str(id(fake_event))
            _intercepted_messages[capture_key] = {
                "messages": [],
                "original_id": original_event.message_id,
                "fake_id": fake_id,
            }

            token = _current_capture_key.set(capture_key)

            try:
                await asyncio.wait_for(handle_event(bot, fake_event), timeout=60.0)
            except asyncio.TimeoutError:
                return "执行失败：调用的插件处理超时", []
            except Exception as plugin_error:
                return f"插件执行出错: {type(plugin_error).__name__} - {str(plugin_error)}", []
            finally:
                _current_capture_key.reset(token)
                context = _intercepted_messages.pop(capture_key, {"messages": []})
                captured_results = context["messages"]

            all_texts: list[str] = []
            all_images: list[str] = []

            for item in captured_results:
                if item["text"]:
                    all_texts.append(item["text"])
                all_images.extend(item["images"])

            return "\n".join(all_texts), all_images

        except Exception as e:
            logger.error(traceback.format_exc())
            return f"系统级投递事件失败: {str(e)}", []


event_simulator = EventSimulator()
