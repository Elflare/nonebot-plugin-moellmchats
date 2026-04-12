# NoneBot 插件集成

本插件支持让 LLM 调用 NoneBot 系统内已有的其他插件，实现"AI 驱动的多插件协同"。

通过 `custom_plugin_info.json` 可以覆写插件对 LLM 的描述，大幅提升模型调用的准确率。

---

## `custom_plugin_info.json` — 插件描述覆写

📌 首次运行后自动生成模板。修改后需重启生效。

### 格式说明

```json5
{
  // 键名必须是目标 NoneBot 插件的真实包名（如 nonebot_plugin_whatis）
  "nonebot_plugin_whatis": {
    "name": "百科搜索与群词条记忆",
    "description": "提供百度百科检索、自定义词条的记忆与遗忘功能。当用户要求查询客观名词解释、要求记住特定设定时调用。",
    "usage": "必须生成以下指令格式：\n1. 记录词条: `记住 [A] 是 [B]`\n2. 问答查询: `[词汇]是什么?`"
  }
}
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `name` | 展示给 LLM 的工具名称 |
| `description` | 告诉 LLM 何时应该调用此插件（越具体越好） |
| `usage` | 调用此插件时 LLM 应生成的指令格式（务必精确，否则插件无法识别） |

### 如何找到插件包名

插件包名通常就是安装时的 Python 包名（即 `pip install` 后的名字，带下划线）。可通过以下方式确认：

```bash
# 查看已安装插件
nb plugin list
# 或直接查看 src/plugins/ 或 site-packages/ 下的目录名
```

---

## 常驻插件（resident_plugins）

默认情况下，分类模型会在每次对话时决定是否注入某个插件的 schema。如果希望某个插件**每次都强制注入**（不经过分类模型判断），可将其添加到常驻列表。

通过指令管理：

```
查看常驻插件          # 查看当前常驻列表
添加常驻插件 <包名>   # 将插件加入常驻
移除常驻插件 <包名>   # 从常驻列表移除
```

也可直接在 `model_config.json` 的 `resident_plugins` 数组中手动添加包名。

---

## 插件黑名单（tool_blacklist）

某些系统级或危险插件不应被 LLM 调用，可加入黑名单：

```
插件黑名单            # 查看当前黑名单
添加黑名单 <包名>     # 加入黑名单
移除黑名单 <包名>     # 从黑名单移除
```

或在 `model_config.json` 中手动编辑 `tool_blacklist` 数组。

---

## 相关页面

- [自定义工具开发](./custom-tools.md) — 编写原生 Python 函数供 LLM 调用
- [配置参考 → model_config.json](./configuration.md#model_configjson--智能调度配置) — 完整字段说明
- [完整指令表](./commands.md) — 黑名单与常驻插件相关指令
