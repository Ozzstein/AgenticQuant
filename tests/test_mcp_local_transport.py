"""Exercise the actual pinned MCP SDK over localhost; no vendor claims."""
import asyncio
import socket
from datetime import datetime, timezone

import pytest

from src.data.evidence.mcp import MCPManager, ServerManifest, ToolPin
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import RunContext


def test_actual_sdk_http_evidence_roundtrip(tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    async def run():
        server = MCPServer(name="local-read-fixture", version="1.0")

        @server.tool()
        def get_orders(status: str = "open") -> str:
            """Test observation tool, not a broker."""
            return '{"orders":[],"status":"' + status + '"}'

        catalog = await server.list_tools()
        schema = catalog[0].input_schema
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        app = server.streamable_http_app(json_response=True, stateless_http=True)
        runner = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        serving = asyncio.create_task(runner.serve(sockets=[listener]))
        try:
            for _ in range(200):
                if runner.started:
                    break
                await asyncio.sleep(0.01)
            assert runner.started
            pin = ToolPin(name="get_orders", capability="broker_orders", input_schema=schema,
                          schema_hash=digest(canonical_bytes(schema)), max_cost_usd="0")
            manifest = ServerManifest(provider="alpaca", server_name="local-read-fixture", server_version="1.0",
                                      source_revision="test-only", url=f"http://127.0.0.1:{port}/mcp", tools=(pin,), budget_usd="0")
            store = EvidenceStore(tmp_path / "db")
            context = RunContext(run_id="local-protocol", mode="paper", as_of=datetime.now(timezone.utc), config_hash="0"*64, code_version="test")
            async with MCPManager(store, (manifest,)) as manager:
                await manager.connect("alpaca")
                result = await manager.call("alpaca", "get_orders", {"status": "open"}, context, ingested_at=context.as_of)
                assert "orders" in store.document(result.record.evidence_id, as_of=context.as_of).text
                assert manager.health()["alpaca"]["requests"] == 1
        finally:
            runner.should_exit = True
            await asyncio.wait_for(serving, 5)
            listener.close()
    asyncio.run(run())
