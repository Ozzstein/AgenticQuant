"""Run: python -m src.decision request.json --store PATH [--mode replay]."""

import argparse
import sys
from pathlib import Path

from src.decision.service import DecisionService
from src.decision.store import DecisionStore
from src.domain.decisions import DecisionRequest, DecisionStatus


def main(argv=None):
    parser = argparse.ArgumentParser(description="Typed Jev judgment, with immutable replay")
    parser.add_argument("request", type=Path)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--mode", choices=["cache", "refresh", "replay"], default="cache")
    parser.add_argument("--observation-id")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        request = DecisionRequest.model_validate_json(args.request.read_text())
        result = DecisionService(DecisionStore(args.store)).evaluate(
            request,
            mode=args.mode,
            observation_id=args.observation_id,
            timeout_seconds=args.timeout,
        )
        print(result.model_dump_json(indent=2))
        return 0 if result.status == DecisionStatus.OK else 2
    except (ValueError, OSError):
        print("Invalid decision request or inaccessible observation store", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
