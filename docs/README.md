# nonebot-plugin-moellmchats Wiki

欢迎使用 **nonebot-plugin-moellmchats** 文档中心。这里汇集了插件所有配置项、高级功能与使用技巧的完整说明。

---

## 目录

| 页面 | 内容 |
|------|------|
| [配置参考](./configuration.md) | 所有配置文件的完整字段说明（`config.json`、`providers.toml`、`model_config.json` 等） |
| [自定义工具开发](./custom-tools.md) | 使用 `custom_tools/` 编写原生 Python 工具供 LLM 调用 |
| [NoneBot 插件集成](./plugin-integration.md) | 通过 `custom_plugin_info.json` 覆写插件描述，提升 LLM 调用精准度 |
| [性格系统](./personality.md) | `temperaments.json` 性格预设配置与用户切换管理 |
| [完整指令表](./commands.md) | 所有 Bot 指令的参数与权限说明，包括请求查看/停止与 Token 消耗查询 |

---

## 管理员常用功能

- `查看请求`：查看当前正在处理的 LLM 请求、来源、用户、运行时长与消息预览。
- `停止请求 [编号|all]`：终止卡住或不想继续的 LLM 请求，停止后会清理对应用户 CD。
- `查看消耗 [数量或范围]`：查看 API Token 消耗记录；新记录会显示本轮请求累计耗时。

---

## 快速跳转

- [安装 → 回到 README](../README.md#-安装)
- [处理流程 → 回到 README](../README.md#-处理流程)
- [更新日志](../CHANGELOG.md)
