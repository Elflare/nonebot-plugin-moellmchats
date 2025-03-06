from collections import defaultdict, deque
import time
from .Config import config_parser

messages_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_user_history"))
)  # {'qq':messages_entity_list}
# messages_entity_list = [messages_entity1, messages_entity2]


class MessagesEntity:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.user_msg = None
        self.assistant_msg = None

    def add_user_msg(self, user_msg: dict):
        self.user_msg = user_msg

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

    def clrear_messages(self):
        self.messages_entity_list = []

    # 预处理用户问题
    def pre_process(self, format_message_dict: dict) -> str:
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

    def search_message_handler(self, search_info: str):
        # 添加搜索信息
        self.new_user_msg["content"] += (
            f"\n(这是联网搜索结果，供你进行参考： {search_info})"
        )

    def append_message_list(self, messages_entity):
        messages_dict[self.user_id].append(self.messages_entity)

    def get_send_message_list(self) -> list:
        result = []
        for messages_entity in self.messages_entity_list:
            # 成对添加
            result.append(messages_entity.get_user_msg())
            result.append(messages_entity.get_assistant_msg())
        result.append(self.messages_entity.get_user_msg())
        return result

    # 后处理
    def post_process(self, assistant_msg: str = None):
        if assistant_msg:
            self.messages_entity.add_assistant_msg(
                {"role": "assistant", "content": assistant_msg}
            )
            messages_dict[self.user_id].append(self.messages_entity)
