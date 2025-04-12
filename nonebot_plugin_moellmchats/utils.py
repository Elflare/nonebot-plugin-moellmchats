import nonebot
from nonebot.log import logger
from .Config import config_parser
from random import choice
from pathlib import Path
from os import listdir
from nonebot.adapters.onebot.v11 import MessageSegment
from traceback import format_exc
import re

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


# 表情包解析
def parse_emotion(text: str) -> tuple:
    # 使用正则表达式匹配方括号内的内容
    pattern = r"\[(.*?)\]"
    # 提取所有表情包名字
    names = re.findall(pattern, text)
    # 替换所有匹配项为[表情包名字]
    replaced_text = re.sub(pattern, "", text)
    return replaced_text, names


# 获取表情包名字列表
def get_emotions_names() -> list:
    emotions_names = listdir(config_parser.get_config("emotions_dir"))
    return emotions_names


# 获取具体表情包
def get_emotion(emoji_name: str) -> MessageSegment:
    path = Path(config_parser.get_config("emotions_dir")) / emoji_name
    emotion_image_list = list(path.glob("*"))
    if not emotion_image_list:
        return None
    image = path / choice(emotion_image_list)
    try:
        with open(image, "rb") as f:
            img = f.read()
            return MessageSegment.image(img)
    except OSError:
        logger.warning(format_exc())
        return None


# 消息格式转换
async def format_message(event, bot) -> dict[list, str]:
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
                name = await get_member_name(event.group_id, qq, bot)
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


async def get_member_name(group: int, sender_id: int, bot) -> str:  # 将QQ号转换成昵称
    try:
        member_info = await bot.get_group_member_info(
            group_id=group, user_id=sender_id, no_cache=False
        )
        name = member_info.get("card") or member_info.get("nickname")
    except Exception:
        logger.warning("获取成员info失败")
    return str(name)
