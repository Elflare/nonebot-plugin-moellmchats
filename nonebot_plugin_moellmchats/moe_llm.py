import ujson as json
import aiohttp
import asyncio
import traceback
from asyncio import TimeoutError
from nonebot.log import logger
from collections import defaultdict, deque
from .Categorize import Categorize
from .Search import Search
from .ModelSelector import model_selector
from .MessagesHandler import MessagesHandler
from .Config import config_parser
from .TemperamentManager import temperament_manager

context_dict = defaultdict(lambda: deque(maxlen=config_parser.get_config("max_group_history")))


class MoeLlm:
    def __init__(
        self,
        bot,
        event,
        format_message_dict: dict,
        is_objective: bool = False,
        temperament="默认",
    ):
        self.bot = bot
        self.event = event
        self.format_message_dict = format_message_dict
        self.user_id = event.user_id
        self.is_objective = is_objective
        self.prompt = f"{temperament_manager.get_temperament_prompt(temperament)}。现在你在一个qq群中。我的id是{ event.sender.card or event.sender.nickname},你只需回复我。群里近期聊天内容，冒号前面是id，后面是内容：\n"
        # 去除群聊最新的对话，因为在用户的上下文中
        context_dict_ = list(context_dict[event.group_id])[:-1]
        self.prompt += "\n".join(context_dict_)

    async def stream_llm_chat(self, session, url, headers, data, proxy) -> str:
        # 流式响应内容
        result = []
        async with session.post(url, headers=headers, json=data, proxy=proxy) as response:
            # 确保响应是成功的
            if response.status == 200:
                # 异步迭代响应内容
                async for line in response.content:
                    if line.startswith(b"data: [DONE]"):
                        break  # 结束标记，退出循环e.content:
                    if line and line.startswith(b"data:"):
                        json_data = json.loads(line[5:].decode("utf-8"))
                        if content := json_data.get("choices", [{}])[0].get("delta", {}).get("content", ""):
                            result.append(content)
                result = "".join(result)
                if not self.is_objective:
                    self.messages_handler.post_process(result)
                return result
            else:
                logger.error(f"Error: {response}")
                return None

    async def none_stream_llm_chat(self, session, url, headers, data, proxy) -> str:
        async with session.post(
            url=url,
            data=data,
            headers=headers,
            ssl=False,
            proxy=proxy,
        ) as resp:
            # 获取整个响应文本
            response = await resp.json()
            # 返回200
            if resp.status != 200 or not response:
                logger.error(response)
                return None
        if choices := response.get("choices"):
            content = choices[0]["message"]["content"]
            start_tag = "<think>"
            end_tag = "</think>"
            start = content.find(start_tag)
            end = content.find(end_tag)
            if start == -1 and end != -1:
                end += len(end_tag)
                start = 0
                result = content[:start] + content[end:]
            elif start != -1 and end != -1:
                end += len(end_tag)
                result = content[:start] + content[end:]
            else:
                result = content
        else:
            if response.get("code") == "DataInspectionFailed":
                self.messages_handler.clrear_messages()
                return "消息合法检查未通过，少说血腥、暴力、色情的话呐~"
            elif response.get("code") == 50501:
                return None
            else:
                logger.error(response)
                return "bug了呐，赶快喊机器人主人来修一下吧~"
        if not self.is_objective:
            self.messages_handler.post_process(result.strip())
        return result.strip()

    async def get_llm_chat(self) -> str:
        self.messages_handler = MessagesHandler(self.user_id)
        plain = self.messages_handler.pre_process(self.format_message_dict)
        model_info = None
        # 获取难度和是否联网
        if model_selector.get_moe() or model_selector.get_web_search():
            category = Categorize(plain)
            category_result = await category.get_category()
            if isinstance(category_result, tuple):  # 如果是tuple，则说明没有问题
                difficulty, internet_required, key_word = category_result
                logger.info(f"难度：{difficulty}, 是否联网：{internet_required}，搜索关键词：{key_word}")
                # 判断是否联网
                if internet_required and model_selector.get_web_search():
                    search = Search(key_word)
                    await self.bot.send(self.event, "正在搜索，请稍等...")
                    if search_result := await search.get_search():
                        self.messages_handler.search_message_handler(search_result)
                    else:
                        await self.bot.send(self.event, "搜索失败，请检查日志输出")
                # 根据难度改key和url
                if model_selector.get_moe():  # moe
                    model_info = model_selector.get_moe_current_model(difficulty)
        if not model_info:  # 分类失败或者不是用的moe
            model_info = model_selector.get_model("selected_model")
        logger.info(f"模型选择为：{model_info['model']}")
        send_message_list = self.messages_handler.get_send_message_list()
        send_message_list.insert(0, {"role": "system", "content": self.prompt})
        data = {
            "model": model_info["model"],
            "messages": send_message_list,
            "max_tokens": model_info.get("max_tokens"),
            "temperature": model_info.get("temperature"),
            "top_p": model_info.get("top_p"),
            "top_k": model_info.get("top_k"),
            "stream": model_info.get("stream"),
            # "tools": [
            #     {
            #         "type": "web_search",
            #         "web_search": {"enable": True},
            #     }
            # ],
        }

        headers = {
            "Authorization": model_info["key"],
            "Content-Type": "application/json",
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                retry_times = 0
                result = "api寄！"
                max_retry_times = (
                    config_parser.get_config("max_retry_times")
                    if config_parser.get_config("max_retry_times")
                    else 3
                )
                for retry_times in range(max_retry_times):
                    if retry_times > 0:
                        await self.bot.send(
                            self.event,
                            f"api又卡了呐！第 {retry_times+1} 次尝试，请勿多次发送~",
                        )
                        await asyncio.sleep(2**retry_times)
                    if model_info.get("stream"):
                        result = await self.stream_llm_chat(
                            session,
                            model_info["url"],
                            headers,
                            data,
                            model_info.get("proxy"),
                        )
                        return result
                    else:
                        data = json.dumps(data)
                        result = await self.none_stream_llm_chat(
                            session,
                            model_info["url"],
                            headers,
                            data,
                            model_info.get("proxy"),
                        )
                    if result:
                        return result
                    else:
                        continue
        except TimeoutError:
            return "网络超时呐，多半是api反应太慢（"
        except Exception:
            logger.error(str(send_message_list))
            traceback.print_exc()
            return "日常抽风呐！当然也有可能是请求太快了，慢点吧~"
