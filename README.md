# LangGraph Enterprise Demos

本仓库包含两个互补的示例：

1. **BriefOps 企业洞察工作流**：基于 LangGraph 打造的多代理流程，演示 Planner → Searcher → Summarizer → Evaluator → Publisher 的闭环。
2. **Draw.io MCP MVP Demo**：使用纯标准库实现的 Draw.io Model Context Protocol 客户端 + Mock 服务器，帮助在受限环境下验证端到端交互。

## 快速上手

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r briefops/requirements.txt
```

- 运行 BriefOps Demo：`python briefops/briefops_demo.py`
- 运行 Draw.io MCP Demo：`python -m briefops.drawio_mcp_demo`
- 运行测试：`pytest -q`

## 设计文档

- [Draw.io MCP MVP 设计与运行指南](docs/drawio_mcp_demo.md)：涵盖架构图、流程图、运行步骤与测试说明。
- [BriefOps 工作流说明](briefops/README.md)：详细介绍企业洞察代理系统的设计理念与扩展建议。
