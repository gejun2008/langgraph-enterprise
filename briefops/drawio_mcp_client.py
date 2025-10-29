"""Minimal client helpers for interacting with a Draw.io MCP server.

This module contains a small asyncio friendly client that can talk to a
Draw.io Model Context Protocol (MCP) server.  The actual Draw.io MCP uses
WebSockets, however in order to keep the example self-contained and runnable in
restricted execution environments we model the transport as plain TCP sockets
speaking newline delimited JSON-RPC messages.  The high-level protocol remains
compatible with the real server (``initialize`` handshake followed by
``generateDiagram`` calls), which means swapping in a production ready
transport layer is straightforward if needed.

In addition to the client we provide a :class:`MockDrawioMCPServer` that mimics
the behaviour of the Draw.io MCP server.  It is used by the demo script and the
unit tests to exercise the full request/response loop without relying on
external connectivity.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
        "DrawioMCPClient",
        "DrawioMCPError",
        "DrawioDiagram",
        "MockDrawioMCPServer",
]


class DrawioMCPError(RuntimeError):
    """Raised when the Draw.io MCP server replies with an error message."""


@dataclass(slots=True)
class DrawioDiagram:
    """Structured information returned by the ``generateDiagram`` method."""

    format: str
    diagram_xml: str
    preview_url: Optional[str] = None

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "DrawioDiagram":
        return cls(
            format=payload.get("format", "png"),
            diagram_xml=payload.get("diagram"),
            preview_url=payload.get("preview"),
        )


class DrawioMCPClient:
    """Very small JSON-RPC client that speaks to a Draw.io MCP server.

    Parameters
    ----------
    endpoint:
        Socket endpoint of the MCP server (e.g. ``mcp://localhost:8765`` or
        ``127.0.0.1:8765``).
    client_name:
        Identifier sent during the ``initialize`` handshake.
    version:
        Version string advertised during the handshake.
    timeout:
        Optional timeout passed to :func:`asyncio.wait_for` for RPC calls.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client_name: str = "briefops-demo",
        version: str = "0.1.0",
        timeout: Optional[float] = 10.0,
    ) -> None:
        self.host, self.port = self._parse_endpoint(endpoint)
        self.client_name = client_name
        self.version = version
        self.timeout = timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._next_id = 1

    async def __aenter__(self) -> "DrawioMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    async def connect(self) -> None:
        """Open the TCP connection and complete the handshake."""

        if self._writer is not None:
            return

        self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
        await self._call(
            "initialize",
            {
                "client": self.client_name,
                "version": self.version,
            },
        )

    async def close(self) -> None:
        """Gracefully close the TCP connection."""

        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except AttributeError:  # pragma: no cover - Python <3.7 fallback
                pass
        self._reader = None
        self._writer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_diagram(
        self,
        description: str,
        *,
        diagram_format: str = "png",
    ) -> DrawioDiagram:
        """Request the server to generate a diagram for ``description``."""

        result = await self._call(
            "generateDiagram",
            {
                "prompt": description,
                "format": diagram_format,
            },
        )
        if not isinstance(result, dict):  # pragma: no cover - defensive
            raise DrawioMCPError("Malformed result payload from server")
        return DrawioDiagram.from_json(result)

    # ------------------------------------------------------------------
    # Internal RPC helpers
    # ------------------------------------------------------------------
    async def _call(self, method: str, params: Dict[str, Any]) -> Any:
        if self._reader is None or self._writer is None:
            raise DrawioMCPError("Client is not connected. Call connect() first.")

        request_id = self._next_id
        self._next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        data = json.dumps(message).encode("utf-8") + b"\n"
        self._writer.write(data)
        await self._writer.drain()

        async def wait_for_reply() -> Any:
            while True:
                assert self._reader is not None
                raw = await self._reader.readline()
                if not raw:
                    raise DrawioMCPError("Connection closed by server")
                payload = json.loads(raw.decode("utf-8"))

                # Notifications are delivered without an ``id`` field – simply
                # ignore them for this minimal client.
                if "id" not in payload:
                    continue

                if payload.get("id") != request_id:
                    # Unexpected response id; continue waiting.
                    continue

                if "error" in payload:
                    err = payload["error"]
                    raise DrawioMCPError(err.get("message", "Unknown MCP error"))

                return payload.get("result")

        if self.timeout is not None:
            return await asyncio.wait_for(wait_for_reply(), timeout=self.timeout)
        return await wait_for_reply()

    @staticmethod
    def _parse_endpoint(endpoint: str) -> Tuple[str, int]:
        parsed = urlparse(endpoint)
        if not parsed.scheme:
            # ``host:port`` without scheme.
            host, _, port_str = endpoint.partition(":")
            if not host or not port_str:
                raise ValueError("Endpoint must be formatted as host:port or scheme://host:port")
            return host, int(port_str)

        if parsed.hostname is None or parsed.port is None:
            raise ValueError("Endpoint must contain both hostname and port")

        return parsed.hostname, parsed.port


class MockDrawioMCPServer:
    """A tiny mock MCP server used for demos and unit tests.

    The server recognises two methods:

    ``initialize``
        Acknowledge the client.
    ``generateDiagram``
        Echo the provided prompt inside a fake ``<diagram>`` XML payload.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        response_delay: float = 0.0,
    ) -> None:
        self.host = host
        self.port = port
        self._server: Optional[asyncio.AbstractServer] = None
        self.response_delay = response_delay

    async def __aenter__(self) -> str:
        self._server = await asyncio.start_server(self._handler, self.host, self.port)
        sockets = self._server.sockets or []
        if not sockets:  # pragma: no cover - runtime safeguard
            raise RuntimeError("Failed to start mock Draw.io MCP server")
        bound_port = sockets[0].getsockname()[1]
        self.port = bound_port
        return f"mcp://{self.host}:{bound_port}"

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    # ------------------------------------------------------------------
    async def _handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while not reader.at_eof():
            raw_message = await reader.readline()
            if not raw_message:
                break
            payload = json.loads(raw_message.decode("utf-8"))
            message_id = payload.get("id")
            method = payload.get("method")

            delay = 0.0
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "server": "mock-drawio",
                        "version": "0.1",
                    },
                }
            elif method == "generateDiagram":
                prompt = payload.get("params", {}).get("prompt", "")
                diagram_format = payload.get("params", {}).get("format", "png")
                delay = self.response_delay
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "format": diagram_format,
                        "diagram": f"<diagram>{prompt}</diagram>",
                        "preview": f"https://mock.local/{diagram_format}/{prompt.replace(' ', '_')}",
                    },
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}",
                    },
                }

            if delay:
                await asyncio.sleep(delay)

            data = json.dumps(response).encode("utf-8") + b"\n"
            writer.write(data)
            await writer.drain()

        writer.close()
        await writer.wait_closed()
