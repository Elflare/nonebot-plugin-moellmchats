import asyncio
import itertools
import time
from dataclasses import dataclass

from nonebot.adapters.onebot.v11 import MessageEvent


@dataclass
class ActiveRequest:
    request_id: int
    task: asyncio.Task
    user_id: int
    sender_name: str
    target: str
    preview: str
    is_ai: bool
    started_at: float


_active_requests: dict[int, ActiveRequest] = {}
_request_ids = itertools.count(1)


def _truncate(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _extract_preview(format_message_dict: dict) -> str:
    text = "".join(format_message_dict.get("text") or [])
    reply = (format_message_dict.get("reply") or "").strip()
    if reply:
        text = f"[引用: {reply}] {text}"
    return _truncate(text or "<空消息>")


def _extract_target(event: MessageEvent) -> str:
    group_id = getattr(event, "group_id", None)
    if group_id is not None:
        return f"群 {group_id}"
    return "私聊"


def register_request(
    event: MessageEvent,
    format_message_dict: dict,
    is_ai: bool,
    task: asyncio.Task | None = None,
) -> int:
    task = task or asyncio.current_task()
    if task is None:
        raise RuntimeError("无法获取当前请求任务")

    request_id = next(_request_ids)
    sender_name = getattr(event.sender, "card", None) or event.sender.nickname
    _active_requests[request_id] = ActiveRequest(
        request_id=request_id,
        task=task,
        user_id=event.user_id,
        sender_name=sender_name,
        target=_extract_target(event),
        preview=_extract_preview(format_message_dict),
        is_ai=is_ai,
        started_at=time.time(),
    )
    return request_id


def unregister_request(request_id: int) -> None:
    _active_requests.pop(request_id, None)


def get_active_request_count() -> int:
    return len(_active_requests)


def get_current_request_elapsed() -> float | None:
    task = asyncio.current_task()
    if task is None:
        return None

    for request in _active_requests.values():
        if request.task is task:
            return time.time() - request.started_at
    return None


def format_active_requests() -> str:
    if not _active_requests:
        return "当前没有正在处理的 LLM 请求。"

    now = time.time()
    lines = [f"当前正在处理 {len(_active_requests)} 个 LLM 请求："]
    for request_id, request in sorted(_active_requests.items()):
        elapsed = int(now - request.started_at)
        mode = "ai" if request.is_ai else "chat"
        lines.append(
            f"[{request_id}] {request.target} | 用户 {request.sender_name}({request.user_id}) | "
            f"模式 {mode} | 已运行 {elapsed}s"
        )
        lines.append(f"    内容：{request.preview}")
    return "\n".join(lines)


def cancel_request(request_id: int) -> str:
    request = _active_requests.get(request_id)
    if request is None:
        return f"未找到编号为 {request_id} 的正在处理请求。"

    if request.task.done():
        unregister_request(request_id)
        return f"请求 [{request_id}] 已结束，无需停止。"

    request.task.cancel()
    return f"已发送停止信号：[{request_id}] {request.target} | 用户 {request.sender_name}。"


def cancel_all_requests() -> str:
    if not _active_requests:
        return "当前没有正在处理的 LLM 请求。"

    count = 0
    for request in list(_active_requests.values()):
        if not request.task.done():
            request.task.cancel()
            count += 1
    return f"已向 {count} 个正在处理的 LLM 请求发送停止信号。"


def cancel_request_by_arg(arg: str) -> str:
    arg = arg.strip().lower()
    if arg in {"all", "全部", "所有"}:
        return cancel_all_requests()

    if not arg:
        count = len(_active_requests)
        if count == 0:
            return "当前没有正在处理的 LLM 请求。"
        if count == 1:
            request_id = next(iter(_active_requests))
            return cancel_request(request_id)
        return "当前有多个正在处理的请求，请先使用「查看请求」确认编号，再使用「停止请求 编号」。"

    if not arg.isdigit():
        return "参数错误，格式为：停止请求 编号 或 停止请求 all"

    return cancel_request(int(arg))
