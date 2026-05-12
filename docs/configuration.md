# 配置参考

所有配置文件均位于 `nonebot_plugin_localstore.get_plugin_config_dir()` 目录，具体路径参照 [NoneBot Plugin LocalStore](https://github.com/nonebot/plugin-localstore)。

**首次运行时自动生成**，建议先启动一次再停止，然后根据本文档手动修改。

> **注意**：JSON 文件不支持注释，手动复制示例后记得删除 `//` 注释及末尾多余逗号。

---

## 配置文件一览

| 文件 | 维护方式 | 说明 | 修改后是否需重启 |
|------|----------|------|------------------|
| `config.json` | 手动 | 基础行为配置（历史长度、冷却、表情包等） | 是 |
| `providers.toml` | 手动 | 服务商/API 密钥/模型参数，核心配置 | 否（用指令`刷新模型`） |
| `model_config.json` | 指令/手动 | MoE 调度、视觉/工具开关 | 指令实时生效，手动需重启 |
| `model_cache.json` | 系统自动 | 模型列表缓存，无需手动修改 | — |
| `models.json` | 手动（遗留） | 旧版手动模型配置，不推荐新用户使用 | 是 |
| `temperaments.json` | 手动 | 性格预设 system prompt | 是 |
| `temperament_config.json` | 指令自动 | 用户↔性格绑定关系 | 指令实时生效 |
| `mcp_servers.toml` | 手动 | MCP Server 配置，启用后会作为 Function Calling 工具注入 | 否（用指令`刷新工具`） |
| `custom_plugin_info.json` | 手动 | 覆写插件描述，并可声明插件依赖工具 | 否（用指令`刷新工具`） |
| `custom_tools/` | 手动 | 原生 Python 工具函数 | 否（用指令`刷新工具`） |

---

## `config.json` — 基础配置

📌 修改后需要重启。Tavily 搜索 API Key：[获取地址](https://tavily.com/)。

```json5
{
  "max_group_history": 10,        // 群组上下文最大保留条数
  "max_user_history": 8,          // 每个用户上下文最大保留条数
  "max_retry_times": 3,           // LLM 请求失败时的最大重试次数
  "max_tool_rounds": 3,           // 单轮对话中最大工具调用轮次
  "user_history_expire_seconds": 600, // 用户上下文 TTL 过期时间（秒）
  "cd_seconds": 0,                // 每个用户的对话冷却时间（秒）
  "search_api": "Bearer your_tavily_key", // 联网搜索 Tavily API Key（开启搜索必填）
  "fastai_enabled": false,        // 快速 AI 助手开关（无角色扮演、无分段、无表情包）
  "emotions_enabled": false,      // 是否开启表情包功能
  "emotion_rate": 0.1,            // 触发表情包的概率（0~1）
  "emotions_dir": "/absolute/path/to/emotions", // 表情包根目录（绝对路径）
  "private_chat_enabled": false,  // 是否允许超级管理员私聊 Bot
  "show_datetime": false          // 是否在 System Prompt 中注入当前时间
}
```

### 表情包目录结构

`emotions_dir` 下每个子文件夹的名称即为表情包名（中英皆可），LLM 会自动识别文件夹名，无需手动在 prompt 中添加说明。

```plaintext
your_absolute_path/
├── smile/
│   ├── smile1.jpg
│   └── smile2.png
├── 滑稽/
│   ├── huaji001.png
│   └── huaji002.jpg
└── 阴险/
    ├── yinxian_a.jpg
    └── yinxian_b.png
```

---

## `providers.toml` — 服务商配置（核心）

📌 首次运行后自动生成模板。程序会自动补全 API 路径（`/chat/completions`、`/models`）和 Bearer 鉴权头，并在启动时**自动抓取**可用模型列表。支持全局代理与四级参数继承。

### 基本结构

```toml
# base_url: 基础API地址（直接写Base URL即可，程序会自动补全 /chat/completions 及 /models）
# api_key: 你的API密钥（无需手动写 Bearer ，程序会自动补全）。支持填入单个字符串，或字符串列表实现随机轮询，如 ["sk-key1", "sk-key2"]
# proxy: [可选] 该服务商的全局代理
# models: [可选] 手动补充的模型列表。若API不支持 /models 自动获取，或获取不全时可在这里手动指定作为补充。
# extra_payload: [可选] 字典格式。用于透传厂商特有参数（如 Gemini 的 thinking_config ）。
#                该字典下的内容会直接合并到发送给 API 的请求根 JSON 中。
# 【全局默认配置】（所有供应商的所有模型均默认继承此设置，垫底优先级）
[global_default]
stream = true
is_segment = true
max_segments = 5

[providers.deepseek]
base_url = "https://api.deepseek.com"
api_key = "sk-xxxxxx"
models = ["deepseek-chat", "deepseek-reasoner"]

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-xxxxxx"
proxy = "http://127.0.0.1:7890"
```

### 四级参数继承

优先级：**全局默认 < 供应商默认 < 分组配置 < 单模型独立配置**

```toml
# 【一】供应商默认配置（覆盖 global_default，对该供应商所有模型生效）
[providers.openai.default_config]
temperature = 1.0

# 【二】批量分组配置（覆盖前两项，对 models 数组中的模型生效）
[[providers.openai.config_groups]]
models = ["gpt-4o", "gpt-4-turbo"]
temperature = 1.2

# 【三】单模型独立配置（最高优先级，覆盖一切）
[providers.openai.model_configs."o1-preview"]
stream = false      # 该模型不支持流式，单独关闭
json_mode = true    # 开启 JSON 结构化输出（适用于分类模型）

# 【no_tools】标记该模型不支持工具调用格式
# 设置后：本次请求不注入 tool schema，历史中的工具消息也自动转为普通文本
[providers.some-provider.model_configs."some-cheap-model"]
no_tools = true
```

### 多 API Key 轮询

```toml
[providers.deepseek]
base_url = "https://api.deepseek.com"
api_key = ["sk-key1", "sk-key2", "sk-key3"]  # 随机轮询
```

---

## `model_config.json` — 智能调度配置

📌 支持 QQ 指令实时切换；手动修改需重启。**模型名称必须是 `providers.toml` 中可用的模型 ID**（可用`查看模型`指令查看）。

```json5
{
  "use_moe": false,           // 是否启用混合专家调度（开启联网搜索也需要此项为 true）
  "moe_models": {
    "0": "deepseek-chat",     // 简单问题对应的模型
    "1": "deepseek-chat",     // 中等问题对应的模型
    "2": "deepseek-reasoner"  // 复杂问题对应的模型
  },
  "vision_model": "gpt-4o",  // 视觉任务专用模型（有图片时强制使用；未配置则提示用户设置）
  "selected_model": "deepseek-reasoner", // 不启用 MoE 时使用的模型（难度分级失败时也回滚至此）
  "category_model": "glm-4-flash",       // 分类模型（建议用免费或小型模型）
  "use_web_search": false,    // 是否启用网络搜索（需 use_moe 为 true 才生效）
  "use_tools": true,          // 是否启用函数调用（允许 LLM 触发其他 Bot 插件）
  "tool_blacklist": [         // 禁止 LLM 调用的插件黑名单
    "nonebot_plugin_orm",
    "nonebot_plugin_some_dangerous_plugin",
    "mcp__filesystem",
    "mcp__filesystem__read_file",
    "mcp__filesystem__*"
  ],
  "resident_plugins": []      // 常驻插件：无视分类模型，强制每次注入给 LLM
}
```
`tool_blacklist` 现在同时支持：

- NoneBot 插件包名
- `custom_tools/` 自定义函数名
- MCP 工具名
- MCP 服务级禁用，如 `mcp__filesystem`
- MCP 通配禁用，如 `mcp__filesystem__*`

`resident_plugins` 也可以填写以上任意工具标识。

---

## `model_cache.json` — 系统动态缓存

📌 **系统自动维护，无需手动修改。**

存储从 `providers.toml` 中各 API 提供商自动拉取到的最新可用模型列表，避免每次启动或对话时重复请求。可通过`刷新模型`指令实时更新此缓存。

---

## `models.json` — 遗留手动配置（不推荐新用户）

📌 用于兼容旧版手动编写的复杂模型配置。系统启动时会将此文件中的模型与 `providers.toml` 获取的模型合并。**新用户建议直接使用 `providers.toml`。**

```json5
{
  "dpsk-chat": {
    "url": "https://api.deepseek.com/chat/completions",
    "key": "Bearer sk-xxx",
    "model": "deepseek-chat",
    "temperature": 1.5,
    "max_tokens": 1024,
    "stream": true,           // 是否流式响应
    "is_segment": true,       // 是否开启分段发送（仅 stream=true 时生效）
    "max_segments": 5         // 分段发送最大段数（超出后停止发送）
  },
  "dpsk-r1": {
    "url": "https://api.deepseek.com/chat/completions",
    "key": "Bearer sk-xxx",
    "model": "deepseek-reasoner",
    "stream": false,
    "top_k": 5,
    "top_p": 1.0
  },
  "gpt-4o": {
    "url": "https://api.openai.com/v1/chat/completions",
    "key": "Bearer sk-xxx",
    "model": "gpt-4o",
    "proxy": "http://127.0.0.1:7890",
    "stream": true
  }
}
```

## `replies.toml` — 互动回复配置

📌 首次触发“戳一戳”或“空白艾特”时自动生成模板。此文件用于自定义机器人的基础互动文案，修改后实时生效，无需重启。

支持在文案中使用 `{bot_name}` 占位符，程序会自动将其替换为机器人当前的真实昵称。

```toml
# 机器人回复文案配置
# 可在文案中使用 {bot_name} 作为机器人昵称的占位符

# 收到只有艾特，没有具体文字内容的回复文案
hello = [
    "你好喵~",
    "你好OvO",
    "喵呜 ~ ，叫{bot_name}做什么呢☆"
]

# 收到戳一戳时的回复文案
poke = [
    "嗯？",
    "戳我干嘛qwq",
    "请不要戳{bot_name} >_<",
    "喵 ~ ！ 戳{bot_name}干嘛喵！"
]
```

## `mcp_servers.toml` — MCP Server 配置

📌 首次运行后自动生成模板。修改后使用 `刷新工具` 生效，无需重启。

MCP 工具会被并入现有函数调用系统，统一受 `use_tools`、`tool_blacklist`、`resident_plugins` 控制。

```toml
[mcp.filesystem]
enabled = false
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "/tmp"]
description = "本地文件系统 MCP"
cwd = "/path/to/workdir"
timeout = 30
discover_timeout = 30
tool_timeout = 60
result_limit = 6000

[mcp.filesystem.env]
SOME_TOKEN = "xxx"

[mcp.myapi]
enabled = false
transport = "streamable_http"
url = "http://127.0.0.1:8000/mcp"
description = "HTTP MCP 示例"
# timeout = 30
# discover_timeout = 30
# tool_timeout = 60
# result_limit = 6000
```

---

## 相关 Wiki 页面

- [自定义工具开发](./custom-tools.md)
- [NoneBot 插件集成](./plugin-integration.md)
- [性格系统](./personality.md)
- [完整指令表](./commands.md)
