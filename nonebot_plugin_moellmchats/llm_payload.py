import re

from nonebot.log import logger
import ujson as json

from .categorize import Categorize
from .model_selector import model_selector
from .tool_manager import tool_manager


class LlmPayloadMixin:
    async def _prepare_model_info(self, plain: str):
        """预处理：获取模型信息、处理难度分类与视觉判断"""
        self.required_plugins = []
        difficulty = "1"
        # 1. 绝对的客观事实：只要真实提取到了图片URL，就必定触发视觉处理，不再完全依赖大模型分类判断
        has_image = bool(self.messages_handler.current_images)
        if (
            model_selector.get_moe()
            or model_selector.get_web_search()
            or model_selector.get_use_tools()
        ):
            category = Categorize(plain)
            category_result = await category.get_category()
            if isinstance(category_result, str):
                return category_result
            if isinstance(category_result, tuple):
                difficulty, vision_required, required_plugins = category_result
                logger.info(
                    f"难度：{difficulty}, 视觉：{vision_required}, 需要插件：{required_plugins}"
                )
                self.required_plugins = required_plugins
                has_image = has_image or vision_required

        # 视觉任务高于 MoE 与 selected_model：只要本轮有图片，就必须使用 vision_model。
        if has_image:
            if not model_selector.model_config.get("vision_model"):
                return "检测到图片消息，但未配置视觉模型。请先使用「设置视觉模型 <模型名或编号>」配置一个支持图片输入的模型。"

            self.model_info = model_selector.get_model("vision_model")
            if not self.model_info:
                return (
                    "检测到图片消息，但视觉模型不可用。请使用「查看模型」确认模型可用后，"
                    "重新执行「设置视觉模型 <模型名或编号>」。"
                )

            logger.info(f"触发视觉任务，切换至视觉模型: {self.model_info['model']}")

        # 纯文本任务，若开启了MoE，则分配对应难度的模型
        elif model_selector.get_moe():
            self.model_info = model_selector.get_moe_current_model(difficulty)

        # 兜底：既没触发视觉，也没开启MoE，或者啥都没开启（原逻辑），使用默认模型
        if not self.model_info:
            self.model_info = model_selector.get_model("selected_model")
        logger.info(f"模型选择为：{self.model_info['model']}")
        return None

    def _build_tool_mention_hint(self) -> str:
        """仅在本轮工具调用时，为模型补充 @ 占位符规则。"""
        mentions = self.format_message_dict.get("mentions") or []
        reply_user = self.format_message_dict.get("reply_user") or {}

        parts = []

        if mentions:
            mention_desc = "，".join(
                f"#{i+1} {m.get('name', '未知用户')}" for i, m in enumerate(mentions)
            )
            parts.append(f"当前消息额外提到的人：{mention_desc}。")

        if reply_user and reply_user.get("name"):
            parts.append(f"当前用户引用的群友：{reply_user['name']}，占位符为[at:0]。")

        if not parts:
            return ""

        if reply_user and reply_user.get("name"):
            parts.append(
                "工具指令里若需提及当前消息中的人，勿写QQ号或者id，用中括号和数字占位。"
                "用户引用的群友为[at:0]，当前消息里有 @ 的人时从 [at:1]、[at:2] ... 开始。"
                "仅当用户明确要求回复、@或转告被引用群友时才使用[at:0]；不要把[at:0]当作当前提问者。"
            )
        else:
            parts.append(
                "工具指令里若需提及当前消息中的人，勿写QQ号或者id，用中括号和数字占位。"
                "当前消息里有 @ 的人时从 [at:1]、[at:2] ... 开始。"
            )

        return "".join(parts)

    def _build_payload(self, send_message_list: list) -> tuple[dict, bool]:
        """构建发给大模型的 payload 与工具 schema"""
        if self.messages_handler.current_images:
            logger.info(
                f"检测到图片且当前模型为 {self.model_info['model']}，正在构建多模态请求..."
            )
            current_msg = send_message_list[-1]
            vision_content = [{"type": "text", "text": current_msg["content"]}]
            for img_url in self.messages_handler.current_images:
                vision_content.append(
                    {"type": "image_url", "image_url": {"url": img_url}}
                )
            send_message_list[-1] = {
                "role": current_msg["role"],
                "content": vision_content,
            }

        current_stream_flag = self.model_info.get("stream", False)
        data = {
            "model": self.model_info["model"],
            "messages": send_message_list,
        }
        # 处理可选的通用参数（避免把 None 传给 API 导致报错）
        for key in ["max_tokens", "temperature", "top_p", "top_k"]:
            if self.model_info.get(key) is not None:
                data[key] = self.model_info[key]

        # 无缝注入用户在 TOML 中配置的额外自定义参数
        if extra_payload := self.model_info.get("extra_payload"):
            if isinstance(extra_payload, dict):
                data.update(extra_payload)
        tools_schema = []
        # 获取常驻插件并转为集合
        resident_plugins = set(model_selector.get_resident_plugins())
        # 通过并集操作 (|) 自动合并并去重：分类模型返回的 + 历史使用的 + 常驻的
        all_plugins_set = (
            set(getattr(self, "required_plugins", []))
            | self.messages_handler.get_all_used_plugins()
            | resident_plugins
        )
        all_plugins_set = tool_manager.expand_dependencies(all_plugins_set)
        logger.debug(f"LLM 最终将要注入的插件集合: {all_plugins_set}")
        all_plugins = list(all_plugins_set)

        model_supports_tools = (
            model_selector.get_use_tools()
            and not self.model_info.get("no_tools", False)
        )
        if all_plugins:
            normal_plugins = [p for p in all_plugins if p != "web_search"]
            if model_supports_tools and normal_plugins:
                tools_schema.extend(
                    tool_manager.get_tool_schema(normal_plugins, include_search=False)
                )
            if (
                model_supports_tools
                and model_selector.get_web_search()
                and "web_search" in all_plugins
            ):
                tools_schema.extend(
                    tool_manager.get_tool_schema([], include_search=True)
                )

        if tools_schema:
            data["tools"] = tools_schema

            mention_hint = self._build_tool_mention_hint()

            send_message_list[0]["content"] += (
                "。特别注意：1. 同步执行：如果你需要调用工具，必须在本次回复的文本(content)中用简短的一句话"
                "说明你要做什么，并**在同一次回复中立刻发起工具调用(tool_calls)**！"
                "2. 如果用户的请求包含多个步骤逻辑，你必须在获取到前置工具的结果后，"
                "**自动且连续地调用下一个工具**，直至彻底完成要求。"
                "3. 工具执行结束后，原始数据将被清理。因此你最终呈现给用户的回复content中，"
                "**必须完整包含查询到的核心数据和关键结论**，这将作为你下一轮对话的记忆依据！"
            )

            if mention_hint:
                send_message_list[0]["content"] += " 4. " + mention_hint

            current_stream_flag = False
            logger.debug("检测到需要调用工具，已自动将本次请求切换为非流式")
            logger.debug(f"实际发送给大模型的 tools_schema: {tools_schema}")
        data["stream"] = current_stream_flag
        if current_stream_flag:
            data["stream_options"] = {"include_usage": True}
        return data, current_stream_flag

    def _render_history_placeholders(self, text: str) -> str:
        """把本轮临时 @ 占位符转成可读文本，避免污染后续上下文。"""
        if not text:
            return text

        mentions = self.format_message_dict.get("mentions") or []
        reply_user = self.format_message_dict.get("reply_user") or {}

        def repl(match):
            token = match.group(0)

            if token.startswith("[at:"):
                try:
                    idx = int(match.group(1))
                    if idx == 0:
                        name = reply_user.get("name") or reply_user.get("qq")
                        return f"@{name}" if name else "@回复对象"

                    mention_idx = idx - 1
                    if 0 <= mention_idx < len(mentions):
                        name = (
                            mentions[mention_idx].get("name")
                            or mentions[mention_idx].get("qq")
                            or f"目标{idx}"
                        )
                        return f"@{name}"
                except Exception:
                    pass
                return "@提及对象"

            if token == "[at_all]":
                return "@当前消息中提到的所有人"

            return token

        return re.sub(r"\[at:(\d+)\]|\[at_all\]", repl, text)

    def _sanitize_tool_calls_for_history(self, tool_calls: list) -> list:
        """仅清洗 tool_calls 里的临时占位符，保留其余有意义参数。"""
        sanitized = []

        for call in tool_calls:
            new_call = {
                "id": call.get("id"),
                "type": call.get("type", "function"),
                "function": {
                    "name": call.get("function", {}).get("name", ""),
                    "arguments": call.get("function", {}).get("arguments", "{}"),
                },
            }

            raw_args_str = new_call["function"]["arguments"]

            try:
                raw_args = json.loads(raw_args_str)
            except Exception:
                # 不是合法 JSON，就直接在原字符串层面替换占位符
                new_call["function"]["arguments"] = self._render_history_placeholders(
                    raw_args_str
                )
                sanitized.append(new_call)
                continue

            def walk(v):
                if isinstance(v, str):
                    return self._render_history_placeholders(v)
                if isinstance(v, list):
                    return [walk(x) for x in v]
                if isinstance(v, dict):
                    return {k: walk(val) for k, val in v.items()}
                return v

            cleaned_args = walk(raw_args)
            new_call["function"]["arguments"] = json.dumps(
                cleaned_args,
                ensure_ascii=False,
            )
            sanitized.append(new_call)

        return sanitized

