"""Reviewable evaluation manifests; deliberately no policy publication command."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from src.decision.experiments import EvaluationManifest, ExperimentService
from src.decision.store import DecisionStore


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen source-bound Jev evaluation experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="print the closed manifest JSON schema; no benchmark data")
    for name in ("freeze", "run", "report", "inspect"):
        command = commands.add_parser(name)
        command.add_argument(
            "identity", help="manifest JSON path for freeze; immutable artifact hash otherwise"
        )
        command.add_argument("--store", type=Path, required=True)
        command.add_argument("--evidence", type=Path, required=True)
        if name == "inspect":
            command.add_argument("--kind", choices=("experiment", "report"), default="experiment")
    args = parser.parse_args(argv)
    if args.command == "schema":
        print(json.dumps(EvaluationManifest.model_json_schema(), indent=2))
        return 0
    try:
        from src.data.evidence.store import EvidenceStore

        if not args.evidence.is_file() or (args.command != "freeze" and not args.store.is_file()):
            raise ValueError("existing source/observation authorities are required")
        manifest = None
        if args.command == "freeze":
            path = Path(args.identity)
            # Read with a bound, not stat followed by an unbounded read.
            with path.open("rb") as handle:
                payload = handle.read(32_000_001)
            if len(payload) > 32_000_000:
                raise ValueError("manifest exceeds 32 MB")
            manifest = EvaluationManifest.model_validate_json(payload)
        service = ExperimentService(
            DecisionStore(args.store), evidence_store=EvidenceStore(args.evidence)
        )
        if args.command == "freeze":
            value = service.freeze(manifest)
        elif args.command == "run":
            value = service.run(args.identity)
        elif args.command == "report":
            value = service.report(args.identity)
        else:
            value = (
                service.verify_report(args.identity)
                if args.kind == "report"
                else service.inspect(args.identity)
            )
        output = {"artifact_id": value.content_hash(), "artifact": value.model_dump(mode="json")}
        if hasattr(value, "report_json"):
            output["report"] = json.loads(output["artifact"].pop("report_json"))
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
        return 0 if "report" not in output or output["report"]["status"] == "complete" else 2
    except (ValueError, KeyError, OSError, sqlite3.Error):
        print(
            "Evaluation rejected: invalid manifest, unavailable/changed authority, or conflicting admission",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
