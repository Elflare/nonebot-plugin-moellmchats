from collections import defaultdict, deque
import time
from .config import config_parser

messages_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_user_history"))
)  # {'qq':messages_entity_list}
# messages_entity_list = [messages_entity1, messages_entity2]


def _flatten_tool_messages(tool_messages: list) -> dict:
    """把 role:tool / role:assistant+tool_calls 消息拍平为一条 assistant 文本，用于不支持工具格式的模型。"""
    parts = []
    for m in tool_messages:
        if m["role"] == "assistant" and "tool_calls" in m:
            names = [tc["function"]["name"] for tc in m["tool_calls"]]
            parts.append(f"[调用工具: {', '.join(names)}]")
        elif m["role"] == "tool":
            parts.append(f"[结果: {m['content']}]")
    return {"role": "assistant", "content": "\n".join(parts)} if parts else None


class MessagesEntity:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.user_msg = None
        self.assistant_msg = None
        # 调用的工具
        self.used_plugins = set()
        self.tool_messages: list = []  # 本轮工具交换消息（结果已截断，供下轮历史使用）

    def add_user_msg(self, user_msg: dict):
        self.user_msg = user_msg

    def add_used_plugins(self, plugins: set):
        self.used_plugins.update(plugins)

    def add_assistant_msg(self, assistant_msg: dict):
        self.assistant_msg = assistant_msg

    def get_user_msg(self) -> dict:
        return self.user_msg

    def get_assistant_msg(self) -> dict:
        return self.assistant_msg


class MessagesHandler:
    def __init__(self, user_id):
        self.user_id = user_id
        self.timestamp = time.time()
        self.messages_entity = MessagesEntity(self.timestamp)
        self.messages_entity_list = messages_dict[self.user_id]
        self.current_images = []  # 暂存当前轮次的图片

    def get_all_used_plugins(self) -> set:
        """获取整个上下文中所有用过的工具集合"""
        plugins = set()
        for entity in self.messages_entity_list:
            plugins.update(entity.used_plugins)
        plugins.update(self.messages_entity.used_plugins)
        return plugins

    # 预处理用户问题
    def pre_process(self, format_message_dict: dict) -> str:
        # 提取图片列表
        self.current_images = format_message_dict.get("images", [])
        if self.messages_entity_list:  # 之前有对话
            # 超过时间一对对话的删了
            expire_seconds = config_parser.get_config("user_history_expire_seconds")
            while self.messages_entity_list and (
                time.time() - self.messages_entity_list[0].timestamp > expire_seconds
            ):
                self.messages_entity_list.popleft()
            if (
                self.messages_entity_list  # 还有对话
                and format_message_dict["reply"]  # 有回复
                and format_message_dict["reply"].strip()
                == self.messages_entity_list[-1]
                .get_assistant_msg()["content"]
                .strip()  # 如果引用的就是上一条回复
            ):
                format_message_dict["text"].pop(0)

        plain = "".join(format_message_dict["text"])
        self.new_user_msg = {"role": "user", "content": plain}  # 最新的问题
        self.messages_entity.add_user_msg(
            self.new_user_msg
        )  # 添加用户问题，之后再处理回答
        return plain

    def append_message_list(self, messages_entity):
        messages_dict[self.user_id].append(self.messages_entity)

    def get_send_message_list(self, supports_tools: bool = True) -> list:
        result = []
        for messages_entity in self.messages_entity_list:
            user_msg = messages_entity.get_user_msg()
            result.append(user_msg)
            # 插入该轮的工具交换消息（如有）
            if messages_entity.tool_messages:
                if supports_tools:
                    result.extend(messages_entity.tool_messages)
                elif flattened := _flatten_tool_messages(messages_entity.tool_messages):
                    result.append(flattened)
            ast_msg = messages_entity.get_assistant_msg()
            # 拦截历史记录中的空 content，替换为占位符，防止触发 400
            if "content" not in ast_msg or not ast_msg["content"]:
                ast_msg["content"] = "（执行完毕）"
            result.append(ast_msg)

        result.append(self.messages_entity.get_user_msg())
        return result

    # 后处理
    def post_process(self, assistant_msg: str = None, tool_messages: list = None):
        # 如果大模型最终没有输出文本，我们可以用它刚生成的隐藏记忆作为替代品存入历史
        if not assistant_msg or not assistant_msg.strip():
            if tool_messages:
                summary = tool_messages[-1].get("content", "") if tool_messages[-1].get("role") == "tool" else ""
                assistant_msg = f"（已在后台静默执行工具完毕，获得了相关数据：{summary[:100]}...）"
            else:
                assistant_msg = "（已完成操作）"

        self.messages_entity.add_assistant_msg(
            {"role": "assistant", "content": assistant_msg}
        )
        if tool_messages:
            self.messages_entity.tool_messages = tool_messages

        # 避免在流式输出与结尾总结时被重复 append
        if self.messages_entity not in messages_dict[self.user_id]:
            messages_dict[self.user_id].append(self.messages_entity)
