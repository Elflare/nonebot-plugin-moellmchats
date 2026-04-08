from collections import defaultdict, deque
import time
from .config import config_parser

messages_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_user_history"))
)  # {'qq':messages_entity_list}
# messages_entity_list = [messages_entity1, messages_entity2]


class MessagesEntity:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.user_msg = None
        self.assistant_msg = None
        # 调用的工具
        self.used_plugins = set()
        self.tool_memory = ""

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

    def clrear_messages(self):
        self.messages_entity_list = []

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
            for i in range(len(self.messages_entity_list) - 1, -1, -1):
                messages_entity = self.messages_entity_list[i]
                if time.time() - messages_entity.timestamp > config_parser.get_config(
                    "user_history_expire_seconds"
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

    def get_send_message_list(self) -> list:
        result = []
        for messages_entity in self.messages_entity_list:
            user_msg = messages_entity.get_user_msg()
            result.append(user_msg)
            ast_msg = messages_entity.get_assistant_msg()
            # 拦截历史记录中的空 content，替换为占位符，防止触发 400
            if "content" not in ast_msg or not ast_msg["content"]:
                ast_msg["content"] = "（执行完毕）"
            result.append(ast_msg)

        result.append(self.messages_entity.get_user_msg())
        return result

    # 后处理
    def post_process(self, assistant_msg: str = None, tool_memory: str = ""):
        if not assistant_msg or not assistant_msg.strip():
            assistant_msg = "（已完成操作）"

        self.messages_entity.add_assistant_msg(
            {"role": "assistant", "content": assistant_msg}
        )
        self.messages_entity.tool_memory = tool_memory

        # 避免在流式输出与结尾总结时被重复 append
        if self.messages_entity not in messages_dict[self.user_id]:
            messages_dict[self.user_id].append(self.messages_entity)
