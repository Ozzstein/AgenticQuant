"""Prepare a fresh lifecycle fixture using the unchanged canonical smoke preparer."""

import runpy
from pathlib import Path


def main():
    # The shared CLI performs prepare_project and writes the source hash manifest.
    shared = Path(__file__).with_name("prepare.py")
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runpy.run_path(str(shared), run_name="__main__")
    (args.output / "smoke_baseline.py").write_bytes(
        shared.with_name("smoke_algorithm.py").read_bytes()
    )
    (args.output / "main.py").write_bytes(
        shared.with_name("order_lifecycle_algorithm.py").read_bytes()
    )


if __name__ == "__main__":
    main()
