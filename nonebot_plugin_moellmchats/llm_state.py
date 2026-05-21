from collections import defaultdict, deque

from .config import config_parser

context_dict = defaultdict(
    lambda: deque(maxlen=config_parser.get_config("max_group_history"))
)
token_usage_history = deque(maxlen=50)
