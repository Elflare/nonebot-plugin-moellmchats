from nonebot.plugin.on import on_message, on_notice, on_command, on_fullmatch
from nonebot.plugin import PluginMetadata, require
from nonebot.rule import to_me
from nonebot.permission import SUPERUSER
from nonebot.params import CommandArg
import asyncio
from nonebot.adapters.onebot.v11 import (
    GROUP,
    Message,
    MessageEvent,
    GroupMessageEvent,
    PokeNotifyEvent,
    Bot,
)
import random
from .utils import (
    hello__reply,
    poke__reply,
    format_message,
)
from collections import defaultdict

require("nonebot_plugin_localstore")
from . import moe_llm as llm
from .ModelSelector import model_selector
from .TemperamentManager import temperament_manager
from .Config import config_parser


__plugin_meta__ = PluginMetadata(
    name="MoEllm聊天",
    description="感谢llm，机器人变聪明了",
    usage="""1.艾特或以bot的名字开头进行对话\n2.用"性格切换xx"来切换性格（每个性格设定绑定每个人账号，不共享）\n3.用"ai xx"来快速调用纯ai助手\n4.超级管理员限定：用切换模型、切换moe、设置moe、设置联网来设置""",
    type="application",
    homepage="https://github.com/Elflare/nonebot-plugin-moellmchats",
    supported_adapters={"~onebot.v11"},
)

cd = defaultdict(int)
is_repeat_ask_dict = defaultdict(bool)  # 记录是否重复提问

message_matcher = on_message(permission=GROUP, priority=1, block=False)


@message_matcher.handle()
async def context_dict_func(bot: Bot, event: MessageEvent):
    if event.message.extract_plain_text().strip():  # 有文字才记录
        if message_dict := await format_message(event):
            sender_name = event.sender.card or event.sender.nickname
            llm.context_dict[event.group_id].append(f"{sender_name}:{''.join(message_dict['text'])}")


# 性格切换
temperament_switch_matcher = on_command("性格切换", aliases={"切换性格", "人格切换", "切换人格"}, priority=10, block=True)


@temperament_switch_matcher.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    if temp := args.extract_plain_text().strip():
        if temp in temperament_manager.get_temperaments_keys():
            # 写入文件
            if temperament_manager.set_temperament_dict(event.user_id, temp):
                await temperament_switch_matcher.finish(f"已切换性格为{temp}")
            else:
                await temperament_switch_matcher.finish("出错了，搞快喊机器人主人来修复一下吧~")
    await temperament_switch_matcher.finish(f"只有{temperament_manager.get_temperaments_keys()}中的性格可以切换")


# 查看性格
temperament_check_matcher = on_fullmatch(("查看性格", "查看人格"), priority=10, block=True)


@temperament_check_matcher.handle()
async def _(event: GroupMessageEvent):
    await temperament_check_matcher.finish(temperament_manager.get_all_temperaments())


check_model_matcher = on_fullmatch("查看模型", priority=10, block=True)


@check_model_matcher.handle()
async def _(event: GroupMessageEvent):
    await check_model_matcher.finish(model_selector.get_model_config())


model_matcher = on_command("切换模型", permission=SUPERUSER, priority=10, block=True)


@model_matcher.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    model_name = args.extract_plain_text().strip()
    result = model_selector.set_chat_model(model_name)
    await model_matcher.finish(result)


set_moe_matcher = on_command("设置moe", permission=SUPERUSER, priority=10, block=True)


@set_moe_matcher.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    is_moe = args.extract_plain_text().strip()
    if is_moe not in ["开", "关", "0", "1"]:
        await model_matcher.finish("参数错误，格式为：设置moe 开、关、1、0")
    if is_moe == "开" or is_moe == "1":
        is_moe = True
    else:
        is_moe = False
    result = model_selector.set_moe(is_moe)
    await model_matcher.finish(result)


set_web_search_matcher = on_command("设置联网", aliases={"切换联网"}, permission=SUPERUSER, priority=10, block=True)


@set_web_search_matcher.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    is_web_search = args.extract_plain_text().strip()
    if is_web_search not in ["开", "关", "0", "1"]:
        await model_matcher.finish("参数错误，格式为：设置联网 开、关、1、0")
    if is_web_search == "开" or is_web_search == "1":
        is_web_search = True
    else:
        is_web_search = False
    result = model_selector.set_web_search(is_web_search)
    await model_matcher.finish(result)


moe_matcher = on_command("切换moe", permission=SUPERUSER, priority=10, block=True)


@moe_matcher.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    try:
        difficulty, model_name = args.extract_plain_text().split()
        result = model_selector.set_moe_model(model_name, difficulty)
    except Exception:
        await model_matcher.finish("参数错误，格式为：切换moe 难度 模型名")
    await model_matcher.finish(result)


async def handle_llm(bot: Bot, event: GroupMessageEvent, matcher, format_message_dict: dict, is_ai=False):
    # 获取消息文本
    user_id = event.sender.user_id
    if (
        # not model_selector.get_moe()  # 单模型加cd
        # model_selector.get_current_model()["model"] == "deepseek-reasoner"
        event.time - cd[user_id] < config_parser.get_config("cd_seconds")
    ):
        sender_name = event.sender.card or event.sender.nickname
        if is_repeat_ask_dict[user_id]:
            await matcher.finish(
                f"{sender_name}的llm对话cd中, 将会在{config_parser.get_config('cd_seconds') - (event.time-cd[user_id])}秒后自动回答，请不要重复提问~"
            )
        await matcher.send(
            f"{sender_name}的llm对话cd中, 将会在{config_parser.get_config('cd_seconds') - (event.time-cd[user_id])}秒后自动回答，请不要重复提问~"
        )
        is_repeat_ask_dict[user_id] = True
        await asyncio.sleep(max(0, config_parser.get_config("cd_seconds") - (event.time - cd[user_id])))
    cd[user_id] = event.time
    if is_ai:
        temp = "ai助手"
    else:
        temp = temperament_manager.get_temperament(user_id)
        if not temp:
            await matcher.finish("出错了，赶快喊机器人主人来修复一下吧~")
    llm_chat = llm.MoeLlm(bot, event, format_message_dict, temperament=temp)
    reply = await llm_chat.get_llm_chat()
    is_repeat_ask_dict[user_id] = False
    await matcher.finish(reply)


llm_matcher = on_message(
    rule=to_me(),
    permission=GROUP,
    priority=15,
    block=True,
)


@llm_matcher.handle()
async def _(bot: Bot, event: MessageEvent):
    if event.message.extract_plain_text().strip():
        format_message_dict = await format_message(event)
    else:
        await llm_matcher.finish(Message(random.choice(hello__reply)))  # 没有就选一个卖萌回复
    await handle_llm(bot, event, llm_matcher, format_message_dict, is_ai=False)


if config_parser.get_config("fastai_enabled"):
    ai_matcher = on_command(
        "ai",
        permission=GROUP,
        priority=17,
        block=True,
    )

    @ai_matcher.handle()
    async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
        if args.extract_plain_text().strip():
            format_message_dict = await format_message(event)
            await handle_llm(bot, event, ai_matcher, format_message_dict, is_ai=True)
        else:
            await ai_matcher.finish(Message(random.choice(hello__reply)))  # 没有就选一个卖萌回复


# 优先级10，不会向下阻断，条件：戳一戳bot触发

poke_ = on_notice(rule=to_me(), priority=11, block=False)


@poke_.handle()
async def _poke_event(event: PokeNotifyEvent):
    if event.is_tome:
        await poke_.send(Message(random.choice(poke__reply)))
        # try:
        #     await poke_.send(Message(f"[CQ:group_poke,qq={event.user_id}]"))
        # except ActionFailed:
        #     await poke_.send(Message(f"[CQ:touch,id={event.user_id}]"))
        # except Exception:
        #     return
