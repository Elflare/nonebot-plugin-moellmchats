import asyncio
from collections import defaultdict

from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageEvent, PrivateMessageEvent

from . import moe_llm as llm
from .config import config_parser
from .request_manager import register_request, unregister_request
from .temperament_manager import temperament_manager

cd = defaultdict(int)
is_repeat_ask_dict = defaultdict(bool)


async def chat_rule(bot: Bot, event: MessageEvent) -> bool:
    if isinstance(event, GroupMessageEvent):
        return True
    if isinstance(event, PrivateMessageEvent):
        return bool(
            config_parser.get_config("private_chat_enabled")
            and str(event.user_id) in bot.config.superusers
        )
    return False


def reset_user_runtime_state(user_id: int) -> None:
    cd[user_id] = 0
    is_repeat_ask_dict[user_id] = False


def reset_all_runtime_state() -> None:
    cd.clear()
    is_repeat_ask_dict.clear()


async def handle_llm(
    bot: Bot, event: MessageEvent, matcher, format_message_dict: dict, is_ai=False
):
    user_id = event.sender.user_id
    if event.time - cd[user_id] < config_parser.get_config("cd_seconds"):
        sender_name = getattr(event.sender, "card", None) or event.sender.nickname
        wait_seconds = config_parser.get_config("cd_seconds") - (event.time - cd[user_id])
        if is_repeat_ask_dict[user_id]:
            await matcher.finish(
                f"{sender_name}的llm对话cd中, 将会在{wait_seconds}秒后自动回答，请不要重复提问~"
            )
        await matcher.send(
            f"{sender_name}的llm对话cd中, 将会在{wait_seconds}秒后自动回答，请不要重复提问~"
        )
        is_repeat_ask_dict[user_id] = True
        await asyncio.sleep(max(0, wait_seconds))

    cd[user_id] = event.time
    if is_ai:
        temp = "ai助手"
    else:
        temp = temperament_manager.get_temperament(user_id)
        if not temp:
            await matcher.finish("出错了，赶快喊机器人主人来修复一下吧~")

    llm_chat = llm.MoeLlm(bot, event, format_message_dict, temperament=temp)
    request_id = register_request(event, format_message_dict, is_ai)
    try:
        is_finished = await llm_chat.get_llm_chat()
    except asyncio.CancelledError:
        reset_user_runtime_state(user_id)
        await matcher.finish("当前 LLM 请求已被超级管理员终止。")
    finally:
        unregister_request(request_id)

    is_repeat_ask_dict[user_id] = False
    if isinstance(is_finished, str):
        cd[user_id] = 0
        await matcher.finish(is_finished)
    elif not is_finished:
        cd[user_id] = 0
