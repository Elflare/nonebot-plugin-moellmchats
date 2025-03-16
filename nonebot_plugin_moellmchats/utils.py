import nonebot
from nonebot.log import logger

Bot_NICKNAME: str = list(nonebot.get_driver().config.nickname)[0]  # bot的nickname


# hello之类的回复
hello__reply = [
    "你好喵~",
    "呜喵..？！",
    "你好OvO",
    f"喵呜 ~ ，叫{Bot_NICKNAME}做什么呢☆",
    "怎么啦qwq",
    "呜喵 ~ ，干嘛喵？",
    "呼喵 ~ 叫可爱的咱有什么事嘛OvO",
]

# 戳一戳消息
poke__reply = [
    "嗯？",
    "戳我干嘛qwq",
    "呜喵？",
    "喵！",
    "呜...不要用力戳咱...好疼>_<",
    f"请不要戳{Bot_NICKNAME} >_<",
    "放手啦，不给戳QAQ",
    f"喵 ~ ！ 戳{Bot_NICKNAME}干嘛喵！",
    "戳坏了，你赔！",
    "呜......戳坏了",
    "呜呜......不要乱戳",
    "喵喵喵？OvO",
    "(。´・ω・)ん?",
    "怎么了喵？",
    "呜喵！......不许戳 (,,• ₃ •,,)",
    "有什么吩咐喵？",
    "啊呜 ~ ",
    "呼喵 ~ 叫可爱的咱有什么事嘛OvO",
]


# 消息格式转换
async def format_message(event) -> dict[list, str]:
    text_message = []
    reply_text = ""
    if event.reply:
        reply_text = event.reply.message.extract_plain_text().strip()
        reply = f"[回复 {event.reply.sender.card or event.reply.sender.nickname} 的消息 [{reply_text}]]"
        text_message.append(reply)
    for msgseg in event.get_message():
        if msgseg.type == "at":
            qq = msgseg.data.get("qq")
            if qq != nonebot.get_bot().self_id:  # 排除at机器人
                name = await get_member_name(event.group_id, qq)
                text_message.append(name)
        elif msgseg.type == "image":
            text_message.append("[图片]")
        elif msgseg.type == "face":
            pass
        elif msgseg.type == "text":
            if plain := msgseg.data.get("text", ""):
                if plain.startswith("ai"):  # 判断ai开头
                    text_message.append(plain[2:])
                else:
                    text_message.append(plain)
    return {"text": text_message, "reply": reply_text}


async def get_member_name(group: int, sender_id: int) -> str:  # 将QQ号转换成昵称
    for bot in nonebot.get_bots().values():
        try:
            member_info = await bot.get_group_member_info(
                group_id=group, user_id=sender_id, no_cache=False
            )
            name = member_info.get("card") or member_info.get("nickname")
            break
        except Exception:
            logger.warning("该机器人获取成员info失败，尝试下一个")
    else:
        name = sender_id
    return str(name)
