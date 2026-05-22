from nonebot.log import logger

from .tool_manager import tool_manager


async def reload_tools_for_commands() -> tuple[int, int]:
    tool_manager.refresh_plugins()
    error_count = tool_manager.load_custom_tools()

    load_mcp_tools = getattr(tool_manager, "load_mcp_tools", None)
    if not callable(load_mcp_tools):
        logger.warning("当前 ToolManager 缺少 load_mcp_tools 方法，已跳过 MCP 工具刷新")
        return error_count, 0

    mcp_count = await load_mcp_tools()
    return error_count, mcp_count
