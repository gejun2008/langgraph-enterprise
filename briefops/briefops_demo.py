# -----------------------------------------------
# BriefOps: LangGraph + HF 开源LLM + 工具调用（最小可运行）
# 目标：输入任务 -> 检索 -> 摘要(含引用) -> 评估校验 -> 输出结构化简报
# -----------------------------------------------
"""BriefOps demo pipeline.

This module implements a minimal enterprise briefing agent using LangGraph,
Hugging Face open models, and tool calling semantics inspired by OpenAI
function calling and MCP (Model Context Protocol).

The pipeline contains the following stages:
- Planner: create search plan and context budget.
- Searcher: retrieve data via web search, URL fetch, optional CRM mock.
- Summarizer: synthesize structured brief with inline citations.
- Evaluator: validate citations and references.
- Publisher: persist markdown/json artifacts for audit.

The code defaults to TinyLlama for CPU compatibility, but can be switched to
Llama-3-8B-Instruct when GPU is available by exporting ``HF_MODEL``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from duckduckgo_search import DDGS
from langgraph import StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------
HF_MODEL = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
"""Default Hugging Face model ID. Override by setting ``HF_MODEL`` env var."""

print(f"🤖 Using HF model: {HF_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL,
    trust_remote_code=True,
    device_map="auto",
)
_generation = pipeline("text-generation", model=model, tokenizer=tokenizer)


def llm(prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """Simple wrapper around the HF text-generation pipeline."""

    output = _generation(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
    )
    return output[0]["generated_text"][len(prompt) :].strip()


# ---------------------------------------------------------------------------
# Tool registry (MCP-style schema description)
# ---------------------------------------------------------------------------
@dataclass
class ToolSpec:
    """Describe a tool with schema metadata."""

    description: str
    schema: Dict[str, str]
    handler: Any


TOOLS: Dict[str, ToolSpec] = {}


def register_tool(name: str, description: str, schema: Dict[str, str]):
    """Decorator to register a tool handler."""

    def decorator(func):
        TOOLS[name] = ToolSpec(description=description, schema=schema, handler=func)
        return func

    return decorator


@register_tool(
    "web_search",
    description="DuckDuckGo 文本搜索，返回前N条：[{title, body, href}]",
    schema={"q": "string", "max_results": "int"},
)
def tool_web_search(q: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return structured results."""

    rows: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(q, max_results=max_results):
            rows.append(
                {
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "href": item.get("href", ""),
                }
            )
    return rows


@register_tool(
    "fetch_url",
    description="抓取URL文本内容（简化版）",
    schema={"url": "string"},
)
def tool_fetch_url(url: str) -> str:
    """Fetch content of a web page with basic normalization."""

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text = re.sub(r"\s+", " ", response.text)
        return text[:5000]
    except Exception as exc:  # pragma: no cover - network dependent
        return f"[fetch_error] {exc}"


@register_tool(
    "crm_query",
    description="模拟企业CRM检索，返回客户最近动态",
    schema={"customer": "string"},
)
def tool_crm_query(customer: str) -> Dict[str, Any]:
    """Mock CRM lookup returning static sample data."""

    return {
        "customer": customer,
        "last_meetings": ["2025-10-10 QBR", "2025-10-15 技术支持回访"],
        "open_tickets": [
            {"id": "T-1021", "sev": "P2", "topic": "数据同步延迟"},
        ],
        "recent_orders": [
            {"sku": "ENT-PLUS", "amount": 120000, "date": "2025-10-12"},
        ],
    }


# ---------------------------------------------------------------------------
# LangGraph state and nodes
# ---------------------------------------------------------------------------
@dataclass
class BriefOpsState:
    """Graph state shared across nodes."""

    task: str
    budget_tokens: int = 3000
    plan: Dict[str, Any] = field(default_factory=dict)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    crm: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    eval: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[str] = None


def node_planner(state: BriefOpsState) -> BriefOpsState:
    """Plan retrieval strategy and context budget."""

    need_crm = "客户" in state.task or "CRM" in state.task
    state.plan = {
        "search_queries": [
            f"{state.task} 关键趋势",
            f"{state.task} 企业落地 案例",
        ],
        "max_results": 3 if state.budget_tokens < 4000 else 5,
        "snippet_limit": 800 if state.budget_tokens < 4000 else 1400,
        "need_crm": need_crm,
    }
    return state


def node_searcher(state: BriefOpsState) -> BriefOpsState:
    """Execute plan via registered tools and normalize passages."""

    passages: List[Dict[str, Any]] = []
    for query in state.plan["search_queries"]:
        hits = tool_web_search(query, state.plan["max_results"])
        for hit in hits:
            text = hit.get("body", "")
            if hit.get("href"):
                text = tool_fetch_url(hit["href"])
            passages.append(
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("href", ""),
                    "text": text[: state.plan["snippet_limit"]],
                }
            )

    state.retrieved = passages
    state.crm = tool_crm_query("示例客户A") if state.plan["need_crm"] else None
    return state


def _format_citation_blocks(passages: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for idx, passage in enumerate(passages[:6], start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] {passage.get('title', '')}",
                    f"URL: {passage.get('url', '')}",
                    f"TEXT: {passage.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def node_summarizer(state: BriefOpsState) -> BriefOpsState:
    """Generate structured JSON brief with explicit citations."""

    citations = _format_citation_blocks(state.retrieved)
    crm_payload = json.dumps(state.crm, ensure_ascii=False) if state.crm else "null"
    prompt = f"""
你是企业简报助手。请基于引用文本输出“结构化要点(JSON)”，所有论点务必附带引用编号。
- 输出字段：title, bullets[], risks[], actions[], references[]
- bullets/actions 每条后加 (ref: [编号, ...])
- 不要编造引用；引用编号取自下面材料的 [n]
- CRM 如非空，可整合为客户洞察

[引用材料]
{citations}

[CRM]
{crm_payload}

请直接输出 JSON：
"""
    response = llm(prompt, max_new_tokens=700)
    match = re.search(r"\{.*\}", response, re.S)
    state.summary = json.loads(match.group(0)) if match else {"raw": response}
    return state


def node_evaluator(state: BriefOpsState) -> BriefOpsState:
    """Validate citation references stay within retrieved window."""

    refs: set[int] = set()
    if isinstance(state.summary, dict):
        for key in ("bullets", "actions"):
            for item in state.summary.get(key, []):
                numbers = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", json.dumps(item, ensure_ascii=False))
                for group in numbers:
                    for value in group.split(","):
                        value = value.strip()
                        if value.isdigit():
                            refs.add(int(value))

    max_ref = min(6, len(state.retrieved))
    invalid = [ref for ref in sorted(refs) if not 1 <= ref <= max_ref]
    state.eval = {"invalid_refs": invalid, "max_ref": max_ref}
    return state


def node_publisher(state: BriefOpsState) -> BriefOpsState:
    """Persist markdown report and JSON summary for audit trail."""

    title = state.summary.get("title", "企业洞察简报") if isinstance(state.summary, dict) else "企业洞察简报"
    bullets = state.summary.get("bullets", []) if isinstance(state.summary, dict) else []
    risks = state.summary.get("risks", []) if isinstance(state.summary, dict) else []
    actions = state.summary.get("actions", []) if isinstance(state.summary, dict) else []
    references = state.summary.get("references", []) if isinstance(state.summary, dict) else []

    markdown_lines = [f"# {title}", "## 关键要点"] + [f"- {item}" for item in bullets]
    if risks:
        markdown_lines += ["", "## 风险"] + [f"- {item}" for item in risks]
    if actions:
        markdown_lines += ["", "## 行动建议"] + [f"- {item}" for item in actions]
    if references:
        markdown_lines += ["", "## 参考链接"] + [f"- {item}" for item in references]

    os.makedirs("audit", exist_ok=True)
    tag = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]

    summary_path = os.path.join("audit", f"summary_{tag}.json")
    report_path = os.path.join("audit", f"report_{tag}.md")

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(state.summary, handle, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines))

    state.report_path = report_path
    return state


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
graph = StateGraph()
graph.add_node("planner", node_planner)
graph.add_node("searcher", node_searcher)
graph.add_node("summarizer", node_summarizer)
graph.add_node("evaluator", node_evaluator)
graph.add_node("publisher", node_publisher)

graph.add_edge("planner", "searcher")
graph.add_edge("searcher", "summarizer")
graph.add_edge("summarizer", "evaluator")
graph.add_edge("evaluator", "publisher")


def run_briefops(task: str, budget_tokens: int = 3000) -> BriefOpsState:
    """Execute the BriefOps LangGraph pipeline."""

    state = BriefOpsState(task=task, budget_tokens=budget_tokens)
    result: BriefOpsState = graph.run(state)  # type: ignore[assignment]
    return result


if __name__ == "__main__":
    TASK = "Agentic 框架在 CRM 与知识助手的企业应用趋势（含数据治理与合规考量）"
    outcome = run_briefops(TASK)
    print("✅ 报告位置:", outcome.report_path)
    print("🔎 评估信息:", outcome.eval)
