"""Build/check the operational LEAN execution project; no synthetic runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import model_validator

from src.domain.common import VersionedContract
from src.domain.portfolio import AccountSpec, FillPolicy, RiskPolicy
from src.execution.lean_venue import LeanDeployment
from src.execution.signal_translator import signals_to_orders, target_weights_to_orders
from src.execution.spot_venue import NativeSpotMarket
from src.execution.venue import VenueRules
from src.utils.schemas import Signal

# Re-export for convenience
__all__ = [
    "signals_to_orders",
    "target_weights_to_orders",
    "write_signals_json",
    "LEAN_ALGORITHM_TEMPLATE",
    "LeanLaunchConfig",
    "prepare_project",
    "verify_project",
]

LEAN_ALGORITHM_TEMPLATE: str = """# QuantAgentLab canonical execution outbox.
# Mount the approved repository and ledger at the configured paths in LEAN.
from src.execution.lean_algorithm import AIQuantStackAlgorithm
"""


class LeanLaunchConfig(VersionedContract):
    ledger_path: str
    account: AccountSpec
    deployment: LeanDeployment
    rules: dict[str, VenueRules]
    fill_policy: FillPolicy = FillPolicy()
    risk_policy: RiskPolicy = RiskPolicy()
    api_key_environment: str | None = None
    api_secret_environment: str | None = None
    native_markets: tuple[NativeSpotMarket, ...] = ()
    evidence_store_path: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        import re

        if (
            self.evidence_store_path is not None
            and not Path(self.evidence_store_path).is_absolute()
        ):
            raise ValueError("LEAN EvidenceStore must have an explicit absolute path")
        if self.deployment.brokerage == "BinanceBrokerage" and self.evidence_store_path is None:
            raise ValueError("LEAN Binance requires the frozen-reference EvidenceStore path")

        if self.account.mode == "live" and any(
            value is None or not re.fullmatch(r"[A-Z][A-Z0-9_]*", value)
            for value in (self.api_key_environment, self.api_secret_environment)
        ):
            raise ValueError(
                "live LEAN requires private native read-credential environment variable names"
            )
        if self.deployment.brokerage in ("BinanceBrokerage", "KrakenBrokerage") and {
            item.instrument.instrument_id for item in self.native_markets
        } != {item.instrument.instrument_id for item in self.deployment.instruments}:
            raise ValueError("live crypto LEAN requires complete actual native market mappings")
        if (
            not Path(self.ledger_path).is_absolute()
            or self.account.account_id != self.deployment.account_id
            or self.account.mode != self.deployment.mode
        ):
            raise ValueError("absolute ledger path and exact LEAN account/deployment required")
        instruments = {
            item.instrument.instrument_id: item.instrument for item in self.deployment.instruments
        }
        if set(self.rules) != set(instruments) or any(
            rule.venue != "lean"
            or rule.instrument != instruments[key]
            or rule.instrument.quote_currency != self.account.base_currency
            for key, rule in self.rules.items()
        ):
            raise ValueError("exact LEAN rule/instrument/base-currency coverage required")
        return self


def prepare_project(config: LeanLaunchConfig, output: Path) -> dict:
    """Produce a concrete runtime project without launching or claiming execution."""
    config = LeanLaunchConfig.model_validate(config.model_dump())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime_config = output / "qal-execution.json"
    runtime_config.write_text(config.model_dump_json(indent=2))
    (output / "main.py").write_text(LEAN_ALGORITHM_TEMPLATE)
    (output / "config.json").write_text(
        json.dumps(
            {
                "algorithm-language": "Python",
                "parameters": {"qal-config": str(runtime_config)},
                "description": "Canonical admitted intents and actual LEAN OrderEvent accounting",
            },
            indent=2,
        )
    )
    manifest = {
        "schema_version": "1.0",
        "status": "prepared",
        "deployment_hash": config.deployment.content_hash(),
        "configuration_hash": config.content_hash(),
        "project": str(output),
        "runtime_verified": False,
        "implementation_hashes": _implementation_hashes(),
        "requirements": [
            "actual LEAN runtime with repository dependencies/import path",
            "configured actual live-mode data/brokerage environment",
            "repository and ledger mounted at declared paths",
            "independent authorization before live capital dispatch",
        ],
    }
    (output / "execution-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _implementation_hashes():
    root = Path(__file__).resolve().parents[2]
    files = [
        *sorted((root / "src/execution").glob("*.py")),
        root / "src/domain/common.py",
        root / "src/domain/portfolio.py",
    ]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
    }


def verify_project(config_path: Path) -> LeanLaunchConfig:
    """Bind the attached runtime to the exact reviewed execution implementation."""
    config = LeanLaunchConfig.model_validate_json(config_path.read_text())
    proof = json.loads((config_path.parent / "execution-manifest.json").read_text())
    if (
        proof.get("configuration_hash") != config.content_hash()
        or proof.get("deployment_hash") != config.deployment.content_hash()
        or proof.get("implementation_hashes") != _implementation_hashes()
    ):
        raise ValueError(
            "LEAN prepared configuration/implementation changed; prepare and review again"
        )
    return config


def write_signals_json(signals: list[Signal], path: Path) -> None:
    """Write legacy advisory signals for inspection; never an execution input.

    The output format is::

        {ticker: {direction: str, strength: float, source: str, timestamp: str}}

    Args:
        signals: List of Signal objects to export.
        path: File path to write JSON to.  Parent directories are created if
            they do not already exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        s.ticker: {
            "direction": s.direction.value,
            "strength": s.strength,
            "source": s.source,
            "timestamp": s.timestamp.isoformat(),
        }
        for s in signals
    }
    path.write_text(json.dumps(data, indent=2))
    logger.debug("lean_bridge: wrote {} signal(s) to {}", len(signals), path)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    commands.add_parser("check-runtime")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            answer = prepare_project(
                LeanLaunchConfig.model_validate_json(args.config.read_text()), args.output
            )
        else:
            from QuantConnect.Algorithm import QCAlgorithm
            from System import Decimal

            answer = {
                "status": "bindings_available",
                "algorithm": str(QCAlgorithm),
                "decimal": str(Decimal),
                "runtime_verified": False,
            }
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "runtime_verified": False,
                }
            )
        )
        return 2
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
