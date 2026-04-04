import asyncio
import traceback
from nonebot.log import logger  
from nonebot.adapters.onebot.v11 import Message, GroupMessageEvent, PrivateMessageEvent
from nonebot.message import handle_event

class EventSimulator:
    @staticmethod
    async def dispatch_event(bot, original_event, command_str: str) -> bool:
        """
        伪造事件并投递给 Nonebot 框架处理，同时继承原事件的图片与引用属性
        """
        if not command_str:
            logger.warning("大模型生成的指令文本为空，不投递事件")
            return False

        try:
            # 构建基础的文本消息
            fake_message = Message(command_str)
            
            # 继承原始消息中的图片段
            if hasattr(original_event, "message"):
                for seg in original_event.message:
                    if seg.type == "image":
                        fake_message.append(seg)
                        
            # 动态构造 kwargs，避免 Pydantic 因 None 字段报错
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
            
            # 只有当 reply 真实存在时才继承，防止 Pydantic 校验抛错
            original_reply = getattr(original_event, "reply", None)
            if original_reply is not None:
                kwargs["reply"] = original_reply

            if isinstance(original_event, GroupMessageEvent):
                kwargs["group_id"] = original_event.group_id
                fake_event = GroupMessageEvent(**kwargs)
            elif isinstance(original_event, PrivateMessageEvent):
                fake_event = PrivateMessageEvent(**kwargs)
            else:
                logger.warning(f"不支持的事件类型模拟: {type(original_event)}")
                return False

            logger.info(f"LLM 事件模拟投递 -> 指令: [{command_str}] (已继承原始图片/引用)")
            asyncio.create_task(handle_event(bot, fake_event))
            return True
            
        except Exception as e:
            logger.error(f"模拟事件投递失败: {e}")
            logger.error(traceback.format_exc())  # 强制打印完整堆栈以备查
            return False

event_simulator = EventSimulator()
