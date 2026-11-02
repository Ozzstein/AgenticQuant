"""Print only a code-owned structural summary of a private IBKR Flex capture."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data.ibkr_flex import FlexIntakeError, read_flex_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = read_flex_capture(args.capture, as_of=datetime.now(timezone.utc))
    except FlexIntakeError as exc:
        print(json.dumps({"status": "unavailable", "reason": str(exc)}))
        return 1
    except Exception:
        print(json.dumps({"status": "unavailable", "reason": "capture_inspection_failed"}))
        return 1
    print(json.dumps(report.safe_summary(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
