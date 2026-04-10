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
    PrivateMessageEvent,
    Bot,
)
import random
from nonebot import get_driver

require("nonebot_plugin_localstore")
from .utils import (
    hello__reply,
    poke__reply,
    format_message,
)
from collections import defaultdict

from . import moe_llm as llm
from .model_selector import model_selector
from .temperament_manager import temperament_manager
from .config import config_parser
from .tool_manager import tool_manager
from .messages_handler import messages_dict
from .moe_llm import token_usage_history
__plugin_meta__ = PluginMetadata(
    name="MoEllm聊天",
    description="感谢llm，机器人变聪明了\n✨ 混合专家模型调度LLM插件 | 混合调度·联网搜索·上下文优化·个性定制·Token节约·更加拟人 ✨",
    usage="""1.艾特或以bot的名字开头进行对话
2.用"性格切换xx"来切换性格（每个性格设定绑定每个人账号，不共享）
3.用"ai xx"来快速调用纯ai助手
4.超级管理员限定：用"查看配置"、"查看模型 [搜索关键词]"(支持多关键词模糊搜索)、"刷新模型"、"切换模型"、"切换moe"、"设置moe"、"设置联网"、"设置视觉模型"、"设置分类模型"、"设置工具调用"进行系统管理
5.超级管理员限定：用"添加/移除插件黑名单"来禁用bot的特定工具调用
6.超级管理员限定：用"刷新工具/重载工具"来热重载新增的函数
7.超级管理员限定：用"查看插件黑名单/插件黑名单"来查看插件的黑名单列表
8.超级管理员限定：用"设置私聊 开/关"来开启/关闭超级管理员私聊对话模式
9.对bot发送"重置我的/重置对话/清空上下文"来清空自己的上下文对话记忆
10.超级管理员限定：对bot发送"重置全部对话"来清空所有用户的上下文及群组环境记忆
11.超级管理员限定：用"添加常驻插件/移除常驻插件"、"查看常驻插件"来管理无视分类模型强制注入的工具
12.超级管理员限定：用"查看消耗 [数量或范围]"来查询API Token消耗记录（如：查看消耗 5、查看消耗 10-15、查看消耗 -50）
""",
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
        if message_dict := await format_message(event, bot):
            sender_name = event.sender.card or event.sender.nickname
            llm.context_dict[event.group_id].append(
                f"{sender_name}:{''.join(message_dict['text'])}"
            )
        # 概率主动发
        # if random.randint(1, 100) == 1:
        #     llm = llm.MoeLlm(
        # bot, event, message_dict,is_objective=True, temperament='默认')
        #     reply = await llm.handle_llm()


async def chat_rule(bot: Bot, event: MessageEvent) -> bool:
    if isinstance(event, GroupMessageEvent):
        return True
    if isinstance(event, PrivateMessageEvent):
        # 严格判断：开关打开 且 为超级管理员
        return bool(
            config_parser.get_config("private_chat_enabled")
            and str(event.user_id) in bot.config.superusers
        )
    return False


# 性格切换
temperament_switch_matcher = on_command(
    "性格切换", aliases={"切换性格", "人格切换", "切换人格"}, priority=10, block=True
)


@temperament_switch_matcher.handle()
async def _(bot: Bot, event: GroupMessageEvent, args: Message = CommandArg()):
    if temp := args.extract_plain_text().strip():
        if temp in temperament_manager.get_temperaments_keys():
            # 写入文件
            if temperament_manager.set_temperament_dict(event.user_id, temp):
                await temperament_switch_matcher.finish(f"已切换性格为{temp}")
            else:
                await temperament_switch_matcher.finish(
                    "出错了，赶快喊机器人主人来修复一下吧~"
                )
    await temperament_switch_matcher.finish(
        f"只有{temperament_manager.get_temperaments_keys()}中的性格可以切换"
    )


# 查看性格
temperament_check_matcher = on_fullmatch(
    ("查看性格", "查看人格"), priority=10, block=True
)


@temperament_check_matcher.handle()
async def _(event: GroupMessageEvent):
    await temperament_check_matcher.finish(temperament_manager.get_all_temperaments())


# 1. 查看看看库里有什么模型可以切
check_model_matcher = on_command(
    "查看可用模型", aliases={"查看模型"}, permission=SUPERUSER, priority=10, block=True
)


@check_model_matcher.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    # 允许带多参数模糊搜索，例如：查看模型 deepseek coder
    query = args.extract_plain_text().strip()
    result = model_selector.get_formatted_model_list(query if query else None)
    await check_model_matcher.finish(result)


# 2. 查看当前机器人身上挂着哪些配置
check_config_matcher = on_fullmatch(
    ("查看当前配置", "查看配置"), permission=SUPERUSER, priority=10, block=True
)


@check_config_matcher.handle()
async def _(event: MessageEvent):
    cfg = model_selector.model_config

    # 构建美观的配置面板
    msg = (
        "✨ 当前大模型运行配置 ✨\n"
        f"▪ 基础聊天模型: {cfg.get('selected_model')}\n"
        f"▪ 视觉专用模型: {cfg.get('vision_model') or '未设置 (默认走基础模型)'}\n"
        f"▪ 意图分类模型: {cfg.get('category_model')}\n"
        f"▪ 启用MoE调度: {'✅开启' if cfg.get('use_moe') else '❌关闭'}\n"
        f"  - 难度0: {cfg.get('moe_models', {}).get('0')}\n"
        f"  - 难度1: {cfg.get('moe_models', {}).get('1')}\n"
        f"  - 难度2: {cfg.get('moe_models', {}).get('2')}\n"
        f"▪ 启用联网搜索: {'✅开启' if cfg.get('use_web_search') else '❌关闭'}\n"
        f"▪ 启用函数调用: {'✅开启' if cfg.get('use_tools', False) else '❌关闭'}"
    )
    await check_config_matcher.finish(msg)


model_matcher = on_command("切换模型", permission=SUPERUSER, priority=10, block=True)


@model_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    model_name = args.extract_plain_text().strip()
    result = model_selector.set_chat_model(model_name)
    await model_matcher.finish(result)


set_moe_matcher = on_command("设置moe", permission=SUPERUSER, priority=10, block=True)


@set_moe_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    is_moe = args.extract_plain_text().strip()
    if is_moe not in ["开", "关", "0", "1"]:
        await model_matcher.finish("参数错误，格式为：设置moe 开、关、1、0")
    if is_moe == "开" or is_moe == "1":
        is_moe = True
    else:
        is_moe = False
    result = model_selector.set_moe(is_moe)
    await model_matcher.finish(result)


set_web_search_matcher = on_command(
    "设置联网", aliases={"切换联网"}, permission=SUPERUSER, priority=10, block=True
)


@set_web_search_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
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
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    try:
        difficulty, model_name = args.extract_plain_text().split()
        result = model_selector.set_moe_model(model_name, difficulty)
    except Exception:
        await model_matcher.finish("参数错误，格式为：切换moe 难度 模型名")
    await model_matcher.finish(result)


vision_model_matcher = on_command(
    "切换视觉模型",
    aliases={"设置视觉模型"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@vision_model_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    model_name = args.extract_plain_text().strip()
    result = model_selector.set_vision_model(model_name)
    await vision_model_matcher.finish(result)


async def handle_llm(
    bot: Bot, event: MessageEvent, matcher, format_message_dict: dict, is_ai=False
):
    # 获取消息文本
    user_id = event.sender.user_id
    if event.time - cd[user_id] < config_parser.get_config("cd_seconds"):
        sender_name = getattr(event.sender, "card", None) or event.sender.nickname
        if is_repeat_ask_dict[user_id]:
            await matcher.finish(
                f"{sender_name}的llm对话cd中, 将会在{config_parser.get_config('cd_seconds') - (event.time-cd[user_id])}秒后自动回答，请不要重复提问~"
            )
        await matcher.send(
            f"{sender_name}的llm对话cd中, 将会在{config_parser.get_config('cd_seconds') - (event.time-cd[user_id])}秒后自动回答，请不要重复提问~"
        )
        is_repeat_ask_dict[user_id] = True
        await asyncio.sleep(
            max(0, config_parser.get_config("cd_seconds") - (event.time - cd[user_id]))
        )
    cd[user_id] = event.time
    if is_ai:
        temp = "ai助手"
    else:
        temp = temperament_manager.get_temperament(user_id)
        if not temp:
            await matcher.finish("出错了，赶快喊机器人主人来修复一下吧~")
    llm_chat = llm.MoeLlm(bot, event, format_message_dict, temperament=temp)
    is_finished = await llm_chat.get_llm_chat()
    is_repeat_ask_dict[user_id] = False  # 重复提问判定就不用了
    if isinstance(is_finished, str):  # 表示失败，失败描述文字
        cd[user_id] = 0
        await matcher.finish(is_finished)
    elif not is_finished:  # 失败后cd回0
        cd[user_id] = 0


llm_matcher = on_message(
    rule=to_me() & chat_rule,
    priority=99,
    block=True,
)


@llm_matcher.handle()
async def _(bot: Bot, event: MessageEvent):
    if event.message.extract_plain_text().strip():
        format_message_dict = await format_message(event, bot)
    else:
        await llm_matcher.finish(
            Message(random.choice(hello__reply))
        )  # 没有就选一个卖萌回复
    await handle_llm(bot, event, llm_matcher, format_message_dict, is_ai=False)


if config_parser.get_config("fastai_enabled"):
    ai_matcher = on_command(
        "ai",
        rule=chat_rule,
        priority=17,
        block=True,
    )

    @ai_matcher.handle()
    async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
        if args.extract_plain_text().strip():
            format_message_dict = await format_message(event, bot)
            await handle_llm(bot, event, ai_matcher, format_message_dict, is_ai=True)
        else:
            await ai_matcher.finish(
                Message(random.choice(hello__reply))
            )  # 没有就选一个卖萌回复


set_use_tools_matcher = on_command(
    "设置工具调用",
    aliases={"设置函数调用"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@set_use_tools_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    is_use_tools = args.extract_plain_text().strip()
    if is_use_tools not in ["开", "关", "0", "1"]:
        await set_use_tools_matcher.finish(
            "参数错误，格式为：设置工具调用 开、关、1、0"
        )
    result = model_selector.set_use_tools(is_use_tools in ["开", "1"])
    await set_use_tools_matcher.finish(result)


manage_blacklist_matcher = on_command(
    "添加插件黑名单",
    aliases={"移除插件黑名单"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@manage_blacklist_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    plugin_name = args.extract_plain_text().strip()
    if not plugin_name:
        await manage_blacklist_matcher.finish("请提供插件名，如：添加黑名单 xxx")
    command_name = event.message.extract_plain_text().split()[0].strip()
    action = "add" if "添加" in command_name else "remove"
    result = model_selector.manage_tool_blacklist(action, plugin_name)
    tool_manager.refresh_plugins()
    await manage_blacklist_matcher.finish(result)


check_blacklist_matcher = on_command(
    "插件黑名单",
    aliases={"查看插件黑名单"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@check_blacklist_matcher.handle()
async def _(event: MessageEvent):
    blacklist = model_selector.get_tool_blacklist()
    if not blacklist:
        await check_blacklist_matcher.finish(
            "当前插件黑名单为空，大模型可调用所有已加载且未被过滤的工具。"
        )

    lines = ["🚫 当前插件调用黑名单："]
    for plugin in blacklist:
        lines.append(f"  - {plugin}")

    await check_blacklist_matcher.finish("\n".join(lines))


refresh_tools_matcher = on_command(
    "刷新工具",
    aliases={"重载工具", "刷新插件"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@refresh_tools_matcher.handle()
async def _():
    tool_manager.refresh_plugins()
    error_count = tool_manager.load_custom_tools()  # 接收错误文件数
    msg = "✨ 工具重载完成！\n"
    msg += f"✅ 已加载 {len(tool_manager.plugin_info)} 个原生插件\n"
    msg += f"✅ 已加载 {len(tool_manager.custom_tools)} 个自定义函数"

    if error_count > 0:
        msg += f"\n❌ 有 {error_count} 个自定义文件加载报错，详情请查看后台日志！"

    await refresh_tools_matcher.finish(msg)


category_model_matcher = on_command(
    "切换分类模型",
    aliases={"设置分类模型"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@category_model_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    model_name = args.extract_plain_text().strip()
    result = model_selector.set_category_model(model_name)
    await category_model_matcher.finish(result)


# 机器人启动时自动获取并刷新模型缓存
@get_driver().on_startup
async def _auto_fetch_models():
    await model_selector.fetch_models_from_providers()


# 超级管理员可手动触发模型刷新
refresh_models_matcher = on_command(
    "刷新模型", aliases={"刷新模型列表"}, permission=SUPERUSER, priority=10, block=True
)


@refresh_models_matcher.handle()
async def _():
    await refresh_models_matcher.send(
        "正在重新读取本地配置并拉取各服务商模型列表，请稍候..."
    )
    model_selector.load_providers()  # 重新读取 TOML 配置
    await model_selector.fetch_models_from_providers()  # 重新请求 API 并重载
    await refresh_models_matcher.finish(
        f"更新完毕！当前系统共加载了 {len(model_selector.models)} 个模型。"
    )


# 供其他插件总结用
summary_model_matcher = on_command(
    "切换总结模型",
    aliases={"设置总结模型"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@summary_model_matcher.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    model_query = args.extract_plain_text().strip()
    result = model_selector.set_summary_model(model_query)
    await summary_model_matcher.finish(result)


set_private_chat_matcher = on_command(
    "设置私聊", permission=SUPERUSER, priority=10, block=True
)


@set_private_chat_matcher.handle()
async def _(bot: Bot, event: MessageEvent, args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if arg in ["开", "1"]:
        config_parser.set_config("private_chat_enabled", True)
        await set_private_chat_matcher.finish("已开启超级管理员私聊对话模式")
    elif arg in ["关", "0"]:
        config_parser.set_config("private_chat_enabled", False)
        await set_private_chat_matcher.finish("已关闭超级管理员私聊对话模式")
    else:
        await set_private_chat_matcher.finish("参数错误，格式为：设置私聊 开、关、1、0")


# 重置个人对话（需要 @ 机器人触发）
reset_mine_matcher = on_command(
    "重置我的",
    aliases={"重置对话", "清空上下文", "清空对话"},
    rule=to_me(),
    priority=10,
    block=True,
)


@reset_mine_matcher.handle()
async def _(event: MessageEvent):
    user_id = event.user_id
    if user_id in messages_dict:
        messages_dict[user_id].clear()  # 清空个人记忆

    # 清理该用户的调用CD和状态
    cd[user_id] = 0
    is_repeat_ask_dict[user_id] = False

    await reset_mine_matcher.finish("已清空你的专属上下文对话记忆~")


# 重置全部对话（需要超级管理员 + @ 机器人触发）
reset_all_matcher = on_fullmatch(
    {"重置全部对话", "重置所有对话", "清空所有上下文", "清空全部上下文"},
    rule=to_me(),
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@reset_all_matcher.handle()
async def _():
    messages_dict.clear()  # 清空所有人的个人记忆
    llm.context_dict.clear()  # 清空所有群聊的群聊环境记忆
    cd.clear()
    is_repeat_ask_dict.clear()

    await reset_all_matcher.finish("已清空所有用户的上下文及群聊环境记忆！")


manage_resident_matcher = on_command(
    "添加常驻插件",
    aliases={"移除常驻插件", "添加常驻函数", "移除常驻函数"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@manage_resident_matcher.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    plugin_name = args.extract_plain_text().strip()
    if not plugin_name:
        await manage_resident_matcher.finish(
            "请提供插件/函数名，如：添加常驻插件 web_search"
        )

    command_name = event.message.extract_plain_text().split()[0].strip()
    action = "add" if "添加" in command_name else "remove"
    result = model_selector.manage_resident_plugins(action, plugin_name)
    await manage_resident_matcher.finish(result)


check_resident_matcher = on_command(
    "常驻插件",
    aliases={"查看常驻插件", "查看常驻函数"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@check_resident_matcher.handle()
async def _(event: MessageEvent):
    resident = model_selector.get_resident_plugins()
    if not resident:
        await check_resident_matcher.finish(
            "当前常驻插件列表为空。大模型将完全依赖分类模型进行插件调度。"
        )

    lines = ["📌 当前常驻插件/函数列表 (无视分类强制注入)："]
    for plugin in resident:
        lines.append(f"  - {plugin}")

    await check_resident_matcher.finish("\n".join(lines))


# --- 查询 Token 消耗的指令 ---
check_token_matcher = on_command(
    "查看消耗",
    aliases={"查询token", "token消耗"},
    permission=SUPERUSER,
    priority=10,
    block=True,
)


@check_token_matcher.handle()
async def _(event: MessageEvent, args: Message = CommandArg()):
    arg = args.extract_plain_text().strip()
    if not token_usage_history:
        await check_token_matcher.finish("当前暂无 Token 消耗记录。")
    history_list = list(token_usage_history)
    total_records = len(history_list)
    
    # 默认查询参数
    start_idx = 0
    end_idx = min(5, total_records)
    
    if arg:
        if arg.startswith('-') and arg[1:].isdigit():
            # 逻辑 1：处理 "-N"（最远的 N 条记录）
            n = int(arg[1:])
            if n <= 0:
                 await check_token_matcher.finish("范围错误，负数后面需要跟大于 0 的数字哦。")
            n = min(n, total_records)
            start_idx = total_records - n
            end_idx = total_records
            
        elif '-' in arg:
            # 逻辑 2：处理 "X-Y" 范围
            parts = arg.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                x, y = int(parts[0]), int(parts[1])
                if x > y or x <= 0:
                    await check_token_matcher.finish("范围格式错误，请确保 X <= Y 且 X > 0，例如: 10-15")
                start_idx = x - 1
                end_idx = min(y, total_records)
            else:
                await check_token_matcher.finish("参数格式错误！支持的格式: 5、10-15、-10")
                
        elif arg.isdigit():
            # 逻辑 3：处理 "N"（最近的 N 条记录）
            n = int(arg)
            if n <= 0:
                await check_token_matcher.finish("查询次数必须大于 0 哦~")
            start_idx = 0
            end_idx = min(n, total_records)
            
        else:
            await check_token_matcher.finish("无法识别的参数！支持的格式: 5、10-15、-10")

    # 边界检查
    if start_idx >= total_records:
        await check_token_matcher.finish(f"超出范围啦！当前总共只有 {total_records} 条记录哦。")

    # 切片提取所需数据
    display_list = history_list[start_idx:end_idx]
    
    # 动态生成标题
    if arg.startswith('-'):
        title = f"📊 最远的 {len(display_list)} 次 API 调用消耗："
    elif '-' in arg:
        title = f"📊 第 {start_idx + 1} 到 {end_idx} 次 API 调用消耗："
    else:
        title = f"📊 最近 {len(display_list)} 次 API 调用消耗："

    msg = title + "\n"
    
    # enumerate 传入 start_idx + 1，保证序号和实际位置一致
    for idx, record in enumerate(display_list, start_idx + 1):
        msg += f"[{idx}] {record['time']} | {record['model']}\n"
        msg += f" ├ 提示词: {record['prompt']} (其中命中缓存: {record.get('cache', 0)})\n"
        msg += f" ├ 生成:   {record['completion']}\n"
        msg += f" └ 总计:   {record['total']}\n"
        
    await check_token_matcher.finish(msg.strip())

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
