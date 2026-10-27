import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.data.adapters.base import ProviderError
from src.data.evidence.mcp import MCPManager, ServerManifest, ToolPin
from src.data.evidence.store import EvidenceStore, canonical_bytes, digest
from src.domain.common import RunContext

SCHEMA = {"type": "object", "properties": {"status": {"type": "string", "enum": ["open", "closed"]}}, "additionalProperties": False}


def manifest():
    tool = ToolPin(name="get_orders", capability="broker_orders", input_schema=SCHEMA,
                   schema_hash=digest(canonical_bytes(SCHEMA)), max_cost_usd="0")
    return ServerManifest(provider="alpaca", server_name="test-observer", server_version="1.0", source_revision="test-fixture",
                          url="http://127.0.0.1:9999/mcp", tools=(tool,), budget_usd="0", max_requests=1)


@pytest.mark.parametrize("name", ["submit_order", "cancel_order_by_id", "close_position", "update_account_config",
                                  "get_account_info_and_submit", "call_tool", "execute", "request", "get_orders"])
def test_mutations_and_generic_bypasses_denied(tmp_path, name):
    manager = MCPManager(EvidenceStore(tmp_path / "db"), (manifest(),))
    args = {"name": name, "arguments": {"method": "DELETE"}}
    with pytest.raises(ProviderError):
        manager.authorize("alpaca", "call_tool", args)


def test_manifest_cannot_authorize_mutation():
    tool = ToolPin(name="cancel_all_orders", capability="orders", input_schema=SCHEMA,
                   schema_hash=digest(canonical_bytes(SCHEMA)), max_cost_usd="0")
    with pytest.raises(ValueError, match="allowlist"):
        ServerManifest(**{**manifest().model_dump(exclude={"tools"}), "tools": (tool,)})


def test_schema_verification_cache_and_budget(tmp_path):
    async def run():
        manager = MCPManager(EvidenceStore(tmp_path / "db"), (manifest(),))
        session = SimpleNamespace(initialize=AsyncMock(return_value=SimpleNamespace(server_info=SimpleNamespace(name="test-observer", version="1.0"))),
                                  list_tools=AsyncMock(return_value=SimpleNamespace(next_cursor=None, tools=[SimpleNamespace(name="get_orders", input_schema=SCHEMA)])),
                                  call_tool=AsyncMock(return_value=SimpleNamespace(is_error=False, model_dump=lambda **_: {"content": [{"type": "text", "text": "[]"}], "isError": False})))
        await manager.verify_session("alpaca", session)
        context = RunContext(run_id="test", mode="paper", as_of=datetime(2020, 1, 1, tzinfo=timezone.utc), config_hash="0"*64, code_version="test")
        first = await manager.call("alpaca", "call_tool", {"name": "get_orders", "arguments": {"status": "open"}}, context, ingested_at=context.as_of)
        second = await manager.call("alpaca", "get_orders", {"status": "open"}, context, ingested_at=context.as_of)
        assert first == second
        session.call_tool.assert_awaited_once_with("get_orders", {"status": "open"})
        with pytest.raises(ProviderError, match="budget"):
            await manager.call("alpaca", "get_orders", {"status": "closed"}, context, ingested_at=context.as_of)
        session.list_tools.return_value.tools[0].input_schema = {"type": "object"}
        with pytest.raises(ProviderError, match="schema_mismatch"):
            await manager.verify_session("alpaca", session)
    asyncio.run(run())
