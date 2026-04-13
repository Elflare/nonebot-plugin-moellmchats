import base64
import asyncio
import traceback
from contextvars import ContextVar
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    Message,
    GroupMessageEvent,
    PrivateMessageEvent,
)
from nonebot.message import handle_event

# ================= 核心拦截器：基于 ContextVar + Bot.call_api patch =================
# 使用字典存储捕获上下文（key 为每次 dispatch_event 调用的唯一标识）
_intercepted_messages: dict[str, list] = {}

# ContextVar：标记当前 asyncio task 是否处于 dispatch_event 捕获上下文
# 值为 capture_key 字符串，默认 None 表示非捕获状态
_current_capture_key: ContextVar[str | None] = ContextVar("_current_capture_key", default=None)

# 保存原始 Bot.call_api（模块加载时只保存一次）
_original_call_api = Bot.call_api

# 需要拦截的发送类 API action 集合
_SEND_ACTIONS = {"send_msg", "send_group_msg", "send_private_msg"}


def _extract_send_data(message) -> dict:
    """从消息对象中提取文本和图片 URL，统一转换为视觉 API 可用格式。"""
    norm_msg = Message(message)
    text = norm_msg.extract_plain_text().strip()
    if len(text) > 1000:
        text = text[:1000] + "\n...[由于插件返回结果过长，为防止超出模型限制已自动截断]"

    image_urls = []
    for seg in norm_msg:
        if seg.type == "image":
            raw = seg.data.get("url") or seg.data.get("file") or ""
            if not raw:
                continue
            if raw.startswith("base64://"):
                # MessageSegment.image(bytes/Path) 产生此格式，转为 data URI
                url = "data:image/jpeg;base64," + raw[9:]
            elif raw.startswith("file:///"):
                # 本地文件路径，读取后转 base64（Windows: "file:///C:/..."）
                try:
                    with open(raw[8:], "rb") as f:
                        url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
                except Exception:
                    logger.warning(f"读取本地图片失败，跳过: {raw}")
                    continue
            else:
                url = raw  # http(s):// 直接使用
            image_urls.append(url)

    return {"text": text, "images": image_urls}


async def _safe_call_api(self: Bot, action: str, **params):
    """拦截所有发送类 API，在 dispatch_event 上下文中捕获消息内容。"""
    capture_key = _current_capture_key.get()
    if capture_key is not None and action in _SEND_ACTIONS:
        message = params.get("message")
        if message:
            data = _extract_send_data(message)
            # setdefault 以防并发场景下 key 被提前 pop
            _intercepted_messages.setdefault(capture_key, []).append(data)
            logger.debug(
                f'拦截到插件发送 [{action}] '
                f'文本: {data["text"][:50] or "(空)"}, '
                f'图片数: {len(data["images"])}'
            )
    return await _original_call_api(self, action, **params)


# 全局替换 Bot.call_api（覆盖 bot.send / bot.send_group_msg / bot.call_api 所有路径）
Bot.call_api = _safe_call_api
# ====================================================================================


class EventSimulator:
    @staticmethod
    async def dispatch_event(bot, original_event, command_str: str) -> tuple[str, list[str]]:
        """派发伪造事件并捕获插件回复，返回 (文本结果, 图片URL列表)。"""
        if not command_str:
            logger.warning("大模型生成的指令文本为空，不投递事件")
            return "执行失败：指令参数为空", []

        try:
            fake_message = Message(command_str)
            if hasattr(original_event, "message"):
                for seg in original_event.message:
                    if seg.type == "image":
                        fake_message.append(seg)

            kwargs = {
                "time": original_event.time,
                "self_id": original_event.self_id,
                "post_type": original_event.post_type,
                "sub_type": original_event.sub_type,
                "user_id": original_event.user_id,
                "message_type": original_event.message_type,
                "message_id": original_event.message_id,
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

            # 用对象 id 作为唯一 capture_key，避免并发冲突
            capture_key = str(id(fake_event))
            _intercepted_messages[capture_key] = []
            # 设置 ContextVar，当前 task 及其 create_task 子任务均会继承此值
            token = _current_capture_key.set(capture_key)

            try:
                # 使用 asyncio.wait_for 强制增加超时时间
                await asyncio.wait_for(handle_event(bot, fake_event), timeout=60.0)
            except asyncio.TimeoutError:
                return "执行失败：调用的插件处理超时"
            except Exception as plugin_error:
                return f"插件执行出错: {type(plugin_error).__name__} - {str(plugin_error)}"
            finally:
                # 重置 ContextVar，弹出捕获结果
                _current_capture_key.reset(token)
                captured_results = _intercepted_messages.pop(capture_key, [])

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
