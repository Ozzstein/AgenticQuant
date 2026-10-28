"""Run: python -m src.decision request.json --store PATH [--mode replay]."""

import argparse
import sys
from pathlib import Path

from src.decision.service import DecisionService
from src.decision.store import DecisionStore
from src.domain.decisions import DecisionRequest, DecisionStatus


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "evaluate":
        from src.decision.evaluation_cli import main as evaluation_main

        return evaluation_main(argv[1:])
    parser = argparse.ArgumentParser(description="Typed Jev judgment, with immutable replay")
    parser.add_argument("request", type=Path)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--evidence", "--evidence-store", dest="evidence", type=Path)
    parser.add_argument("--mode", choices=["cache", "refresh", "replay"], default="cache")
    parser.add_argument("--observation-id")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        request = DecisionRequest.model_validate_json(args.request.read_text())
        evidence = None
        if args.evidence:
            from src.data.evidence.store import EvidenceStore

            if not args.evidence.is_file():
                raise ValueError("evidence authority does not exist")
            evidence = EvidenceStore(args.evidence)
        result = DecisionService(DecisionStore(args.store), evidence_store=evidence).evaluate(
            request,
            mode=args.mode,
            observation_id=args.observation_id,
            timeout_seconds=args.timeout,
        )
        print(result.model_dump_json(indent=2))
        return 0 if result.status == DecisionStatus.OK else 2
    except (ValueError, OSError):
        print(
            "Invalid decision request or inaccessible observation/evidence authority",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
