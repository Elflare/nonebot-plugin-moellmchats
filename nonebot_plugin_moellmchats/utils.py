import nonebot
from nonebot.log import logger
from .Config import config_parser
from random import choice
from pathlib import Path
from os import listdir
from nonebot.adapters.onebot.v11 import MessageSegment, Message
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
async def format_message(event, bot) -> dict:
    text_message = []
    reply_text = ""
    image_urls = []  # 新增：用于存储提取到的图片URL

    # 1. 处理回复消息中的图片 (使用你提供的逻辑)
    if reply := event.reply:
        # 手动遍历回复内容，保留 [图片] 占位符
        reply_segments = []
        for seg in event.reply.message:
            if seg.type == "text":
                reply_segments.append(seg.data.get("text", ""))
            elif seg.type == "image":
                reply_segments.append("[图片]")
            # 可以在这里加 elif seg.type == "face": 处理表情等其他类型
        reply_text = "".join(reply_segments).strip()
        text_message.append(
            f"[回复 {event.reply.sender.card or event.reply.sender.nickname} 的消息 [{reply_text}]]"
        )

        try:
            # 获取原消息详情以提取图片
            quoted_message = await bot.get_msg(message_id=reply.message_id)
            message_list = quoted_message["message"]
            if isinstance(message_list, str):  # gocq是str
                message_image = Message(message_list)
                # 查找是否有图片段
                for seg in message_image:
                    if seg.type == "image":
                        if url := seg.data.get("url"):
                            image_urls.append(url)
            else:  # shamrock是list
                for message in message_list:
                    if message.get("type") == "image":
                        if url := message.get("data").get("url"):
                            image_urls.append(url)
        except Exception:
            logger.warning("获取回复消息图片失败")

    # 2. 处理当前消息
    for msgseg in event.get_message():
        if msgseg.type == "at":
            qq = msgseg.data.get("qq")
            if qq != nonebot.get_bot().self_id:
                name = await get_member_name(event.group_id, qq, bot)
                text_message.append(name)
        elif msgseg.type == "image":
            text_message.append("[图片]")
            # 新增：提取当前消息的图片URL
            if url := msgseg.data.get("url"):
                image_urls.append(url)
        elif msgseg.type == "face":
            pass
        elif msgseg.type == "text":
            if plain := msgseg.data.get("text", ""):
                if plain.startswith("ai"):
                    text_message.append(plain[2:])
                else:
                    text_message.append(plain)

    # 返回字典中增加 images 字段
    return {"text": text_message, "reply": reply_text, "images": image_urls}


async def get_member_name(group: int, sender_id: int, bot) -> str:  # 将QQ号转换成昵称
    try:
        member_info = await bot.get_group_member_info(
            group_id=group, user_id=sender_id, no_cache=False
        )
        name = member_info.get("card") or member_info.get("nickname")
    except Exception:
        name = sender_id
        logger.warning("获取成员info失败")
    return str(name)
