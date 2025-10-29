"""Executable demo showing how to call the Draw.io MCP client."""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - CLI convenience
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from briefops.drawio_mcp_client import DrawioMCPClient, MockDrawioMCPServer


async def main() -> None:
    async with AsyncExitStack() as stack:
        server_uri = await stack.enter_async_context(MockDrawioMCPServer())
        client = await stack.enter_async_context(DrawioMCPClient(server_uri))

        prompt = "MVP architecture for a Draw.io MCP integration"
        diagram = await client.generate_diagram(prompt, diagram_format="svg")

        print("Draw.io MCP demo")
        print("----------------")
        print(f"Prompt       : {prompt}")
        print(f"Format       : {diagram.format}")
        print(f"Preview URL  : {diagram.preview_url}")
        print(f"Diagram XML  : {diagram.diagram_xml}")


if __name__ == "__main__":
    asyncio.run(main())
