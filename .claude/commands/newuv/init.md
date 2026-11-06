---
name: newuv:init
description: Use when user types /newuv:init. Interactive wizard that init the uv project.
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, Agent, WebSearch, WebFetch
---

EXECUTE IMMEDIATELY — do not deliberate, do not ask clarifying questions before reading the protocol.


## Execution

1. 仓库里已经有 uv.lock，第一次在本机拉齐环境，优先：

uv sync
含义大致是：按锁文件创建/使用虚拟环境，并把 pyproject.toml 里声明的依赖装到一致版本。比单独 pip install . 更可复现。

若本机没有合适 Python，uv 也可以先装解释器再 sync，例如：

uv python install 3.12
uv sync --python 3.12