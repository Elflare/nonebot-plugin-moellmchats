import asyncio
import traceback
from nonebot.log import logger
from nonebot.adapters.onebot.v11 import (
    Bot,
    Message,
    GroupMessageEvent,
    PrivateMessageEvent,
)
from nonebot.message import handle_event

# ================= 核心修复：并发安全的全局拦截器 =================
# 使用字典存储需要拦截的 message_id 及其对应的截获结果
_intercepted_messages = {}

# 仅在模块加载时保存一次原始的 Bot.send 方法
_original_bot_send = Bot.send


async def _safe_mock_send(self: Bot, event, message, **kwargs):
    msg_id = getattr(event, "message_id", None)
    # 检查是否命中当前正在模拟的拦截任务
    if msg_id in _intercepted_messages:
        # 无论 message 是 str 还是 MessageSegment，统一转换为 Message 对象
        norm_msg = Message(message)
        text = norm_msg.extract_plain_text()
        result = text.strip() if text.strip() else "（执行完毕，返回了图片或无文本数据）"
        
        # 安全起见，保留长度截断防止极端长文本撑爆上下文
        if len(result) > 1000:
            result = result[:1000] + "\n...[由于插件返回结果过长，为防止超出模型限制已自动截断]"
        _intercepted_messages[msg_id].append(result)
        logger.debug(f'看看插件返回的内容：{result}')
    # 无论是否被拦截记录，最后都执行原始的发送逻辑，让用户能真实看到插件的输出
    return await _original_bot_send(self, event, message, **kwargs)


# 替换 Bot 类的 send 方法（全局生效，无需在执行时反复替换）
Bot.send = _safe_mock_send
# ==================================================================


class EventSimulator:
    @staticmethod
    async def dispatch_event(bot, original_event, command_str: str) -> str:
        if not command_str:
            logger.warning("大模型生成的指令文本为空，不投递事件")
            return "执行失败：指令参数为空"

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
                return f"不支持的事件类型模拟: {type(original_event)}"

            logger.info(f"LLM 事件模拟投递 -> 指令: [{command_str}]")

            msg_id = fake_event.message_id

            #  执行前：在字典中注册当前 message_id
            _intercepted_messages[msg_id] = []

            try:
                # 阻塞等待框架处理伪造的事件
                await handle_event(bot, fake_event)
            except Exception as plugin_error:
                return (
                    f"插件执行出错: {type(plugin_error).__name__} - {str(plugin_error)}"
                )
            finally:
                #  执行后：无论成功失败，提取结果并从字典中注销，防止内存泄漏
                captured_texts = _intercepted_messages.pop(msg_id, [])

            if captured_texts:
                plugin_result = "\n".join(captured_texts)
                # 修改提示词：允许其继续调用剩余工具
                return f"插件执行返回结果：\n{plugin_result}\n\n[系统提示]：上述结果已对用户可见。注意：若执行不正确或者用户的原始请求需要多个步骤，请务重试或者继续调用下一个工具！如果所有任务均已完成，请直接做一两句话的简短总结，严禁重复上述已发送的结果。"
            return "插件已执行，但未返回有效文本。[系统提示]：如果有后续操作，请继续调用下一个工具。"
        except Exception as e:
            logger.error(traceback.format_exc())
            return f"系统级投递事件失败: {str(e)}"


event_simulator = EventSimulator()
