# BriefOps（企业洞察与简报 Agent）

## 产品定位
- **定位**：自托管/私有化的多代理企业洞察平台，聚合检索、知识库、CRM 等信号，生成结构化可审计的业务简报。
- **目标用户**：需要构建“OpenAI GPTs + Actions”体验但受限于数据合规/私有化要求的企业团队。
- **核心价值**：用 LangGraph + 开源模型 + 工具调用，完成 Planner → Searcher → Summarizer → Evaluator → Publisher 闭环，输出带引用的 JSON/Markdown/HTML 简报。

## 架构与创新点
1. **上下文钱包（Context Wallet）**：Planner 根据 `budget_tokens` 动态调节检索条数与片段截断长度，兼顾成本/延迟。【详见 `briefops_demo.py` 中 `node_planner`】
2. **可核查引用（Citable Summary）**：Summarizer 只输出带引用编号的要点，Evaluator 二次校验引用编号范围，避免“幻觉引用”。
3. **类 MCP 工具市场**：工具通过统一 schema (`ToolSpec`) 描述，新增工具只需注册函数，支持热插拔装载。
4. **审计追踪（Audit Trail）**：Publisher 将摘要 JSON 与 Markdown 报告落盘 `audit/` 目录，方便合规审计与再训练。
5. **双引擎策略**：默认 TinyLlama 兼容 CPU；设置 `HF_MODEL` 环境变量即可切换至更强的 Llama-3-8B-Instruct。

## 工作流（LangGraph）
```mermaid
flowchart LR
    U[用户任务] --> P[Planner 规划/预算]
    P --> S[Searcher 检索]
    S --> R[Retriever 片段整形]
    R --> M[Summarizer 结构化要点]
    M --> E[Evaluator 引用核查]
    E --> O[Publisher 输出 + 审计]
```

## 快速开始（CPU 亦可运行）
```bash
cd briefops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python briefops_demo.py
```

## 切换更强模型（需 GPU）
```bash
export HF_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
python briefops_demo.py
```

## 审计产物
- `audit/summary_<hash>.json`：结构化简报（含引用与 CRM 洞察）。
- `audit/report_<hash>.md`：Markdown 版简报，可直接对接发布渠道。

## 下一步建议
1. 接入企业认证与密钥管理，确保工具调用权限隔离。
2. 扩展工具描述文件为 YAML/JSON，可从配置目录自动加载。
3. 引入向量数据库与缓存，提高检索精度和吞吐量。
4. 嵌入实时语音/多模态接口，扩展至会议纪要、运维监控场景。
