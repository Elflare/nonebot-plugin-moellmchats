from collections.abc import Iterable


def _format_elapsed(elapsed) -> str:
    if elapsed is None:
        return ""
    try:
        elapsed = float(elapsed)
    except (TypeError, ValueError):
        return ""
    return f" | 耗时 {elapsed:.1f}s"


def format_token_usage_history(arg: str, token_usage_history: Iterable[dict]) -> str:
    history_list = list(token_usage_history)
    if not history_list:
        return "当前暂无 Token 消耗记录。"

    total_records = len(history_list)
    start_idx = 0
    end_idx = min(5, total_records)

    if arg:
        if arg.startswith("-") and arg[1:].isdigit():
            n = int(arg[1:])
            if n <= 0:
                return "范围错误，负数后面需要跟大于 0 的数字哦。"
            n = min(n, total_records)
            start_idx = total_records - n
            end_idx = total_records
        elif "-" in arg:
            parts = arg.split("-")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                return "参数格式错误！支持的格式: 5、10-15、-10"
            x, y = int(parts[0]), int(parts[1])
            if x > y or x <= 0:
                return "范围格式错误，请确保 X <= Y 且 X > 0，例如: 10-15"
            start_idx = x - 1
            end_idx = min(y, total_records)
        elif arg.isdigit():
            n = int(arg)
            if n <= 0:
                return "查询次数必须大于 0 哦~"
            end_idx = min(n, total_records)
        else:
            return "无法识别的参数！支持的格式: 5、10-15、-10"

    if start_idx >= total_records:
        return f"超出范围啦！当前总共只有 {total_records} 条记录哦。"

    display_list = history_list[start_idx:end_idx]
    if arg.startswith("-"):
        title = f"📊 最远的 {len(display_list)} 次 API 调用消耗："
    elif "-" in arg:
        title = f"📊 第 {start_idx + 1} 到 {end_idx} 次 API 调用消耗："
    else:
        title = f"📊 最近 {len(display_list)} 次 API 调用消耗："

    lines = [title]
    for idx, record in enumerate(display_list, start_idx + 1):
        elapsed_text = _format_elapsed(record.get("elapsed"))
        lines.append(f"[{idx}] {record['time']} | {record['model']}{elapsed_text}")
        lines.append(
            f" ├ 提示词: {record['prompt']} (其中命中缓存: {record.get('cache', 0)})"
        )
        lines.append(f" ├ 生成:   {record['completion']}")
        lines.append(f" └ 总计:   {record['total']}")

    return "\n".join(lines).strip()
