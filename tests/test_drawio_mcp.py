import asyncio

import pytest

from briefops.drawio_mcp_client import DrawioMCPClient, DrawioMCPError, MockDrawioMCPServer


def test_generate_diagram_roundtrip():
    async def _run():
        async with MockDrawioMCPServer() as uri:
            async with DrawioMCPClient(uri) as client:
                return await client.generate_diagram("Sales workflow", diagram_format="svg")

    diagram = asyncio.run(_run())

    assert diagram.format == "svg"
    assert "Sales workflow" in diagram.diagram_xml
    assert diagram.preview_url.endswith("Sales_workflow")


def test_unknown_method_raises():
    async def _run():
        async with MockDrawioMCPServer() as uri:
            async with DrawioMCPClient(uri) as client:
                with pytest.raises(DrawioMCPError):
                    await client._call("nonExistent", {})

    asyncio.run(_run())


def test_timeout_handling():
    async def _run():
        async with MockDrawioMCPServer(response_delay=0.5) as uri:
            async with DrawioMCPClient(uri, timeout=0.1) as client:
                with pytest.raises(asyncio.TimeoutError):
                    await client.generate_diagram("Anything")

    asyncio.run(_run())
