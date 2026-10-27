"""Deployment settings must preserve explicit isolation and research chronology."""

from pathlib import Path

import pytest

from src.utils import config as config_module
from src.utils.config import AppConfig, QlibConfig, get_config, reset_config
from src.utils.config_loader import get_full_config


@pytest.fixture
def config_root(tmp_path, monkeypatch):
    monkeypatch.delenv("AIQUANT_PAPER_LEDGER_PATH", raising=False)
    folder = tmp_path / "config"
    folder.mkdir()
    (folder / "settings.yaml").write_text(
        "log_level: WARNING\n"
        "output_dir: yaml-output\n"
        "models:\n  topk: 17\n  walk_forward_months: 6\n"
        "ccxt:\n  max_retries: 5\n"
        "rd_agent:\n  budget: 21\n  min_ic: 0.04\n"
    )
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    return folder


def test_environment_overrides_yaml_without_losing_nested_fields(config_root, monkeypatch):
    monkeypatch.setenv("AIQUANT_OUTPUT_DIR", "env-output")
    monkeypatch.setenv("AIQUANT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AIQUANT_MODELS__TOPK", "11")
    monkeypatch.setenv("AIQUANT_CCXT__EXCHANGE", "kraken")
    cfg = get_config()
    assert cfg.output_dir == "env-output"
    assert cfg.paper_ledger_path == str(Path("env-output") / "paper-ledger.sqlite3")
    assert cfg.log_level == "DEBUG"
    assert cfg.models.topk == 11
    assert cfg.models.walk_forward_months == 6
    assert cfg.ccxt.exchange == "kraken"
    assert cfg.ccxt.max_retries == 5


def test_full_config_uses_same_precedence_and_cache_reset(config_root, monkeypatch):
    monkeypatch.setenv("AIQUANT_RD_AGENT__MIN_IC", "0.07")
    first = get_full_config()
    assert first.rd_agent.min_ic == 0.07
    assert first.rd_agent.budget == 21
    assert first.models.topk == get_config().models.topk
    monkeypatch.setenv("AIQUANT_RD_AGENT__MIN_IC", "0.08")
    assert get_full_config() is first
    reset_config()
    assert get_full_config().rd_agent.min_ic == 0.08


def test_explicit_ledger_override_survives_output_change(config_root, monkeypatch):
    monkeypatch.setenv("AIQUANT_OUTPUT_DIR", "new-output")
    monkeypatch.setenv("AIQUANT_PAPER_LEDGER_PATH", "state/accounting.sqlite3")
    monkeypatch.setenv("AIQUANT_PAPER_ACCOUNT_ID", "paper:desk-a")
    cfg = get_config()
    assert cfg.paper_ledger_path == "state/accounting.sqlite3"
    assert cfg.paper_account_id == "paper:desk-a"
    assert cfg.paper_base_currency == "USD"


@pytest.mark.parametrize("overrides", [
    {"paper_ledger_path": ":memory:"}, {"paper_ledger_path": "  "},
    {"paper_account_id": " "}, {"paper_base_currency": "usd"},
    {"models": {"walk_forward_months": 0}},
])
def test_invalid_operational_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        AppConfig(**overrides)


@pytest.mark.parametrize("overrides", [
    {"train_end": "2022-01-01"}, {"valid_end": "2023-01-01"},
    {"test_end": "2022-01-01"}, {"train_start": "not-a-date"},
])
def test_research_segments_cannot_overlap_or_reverse(overrides):
    with pytest.raises(ValueError):
        QlibConfig(**overrides)


def test_yaml_mapping_required(config_root):
    (config_root / "settings.yaml").write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        get_config()
