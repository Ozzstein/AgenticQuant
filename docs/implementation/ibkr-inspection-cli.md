# Read-only IBKR Flex inspection CLI

The canonical command inspects an explicitly supplied local capture through
`src.data.ibkr_flex.read_flex_capture` and prints only its `safe_summary()`:

```sh
aiquant data inspect-ibkr-flex --capture /absolute/private/capture-directory
python -m src.data inspect-ibkr-flex --capture /absolute/private/capture-directory
```

`--capture` identifies a capture directory, including `capture.json`, original XML,
and receipt chain; it is not a loose XML report. Existing reader rules for absolute
paths, private permissions, receipt consistency, and acquisition cutoff apply
unchanged. No path resolution or permission repair occurs in the CLI.

An optional aware ISO 8601 cutoff supports deterministic inspection:

```sh
aiquant data inspect-ibkr-flex \
  --capture /absolute/private/capture-directory \
  --as-of 2026-10-31T00:00:00+00:00
```

The default cutoff is the actual invocation time in UTC. Invalid or naive cutoffs
return exit code 1 with `{"status":"unavailable","reason":"aware_cutoff_required"}`.
Missing or corrupt captures also return exit code 1, with the existing reader's
code-owned reason. Unexpected inspection failures use `capture_inspection_failed`;
arbitrary exception messages are not printed. A successfully inspected capture
returns exit code 0, which does **not** authorize income admission.

The summary contains structural counts, code-owned categories/blockers, capture
consistency, and acquisition availability. It excludes account identifiers,
financial values, native report dates, descriptions, and arbitrary report text.
It retains `income_admission: blocked` and `canonical_income_records_created: 0`.
Local consistency is not independent source authentication. Source, ledger,
operating, and activation gates remain authoritative; no positive IBKR income
admission is added.

`src.utils.cli.data` already forwards arguments to the data CLI. Inspection dispatch
now occurs before evidence imports or `EvidenceStore` construction. No database is
created or opened, even if the existing global `--store` option is supplied before
`inspect-ibkr-flex`; that option is unused for inspection. Every existing data
command still requires `--store` and retains its store-backed execution and failure
behavior. The standalone `python -m scripts.inspect_ibkr_flex_report --capture ...`
remains compatible and unchanged. Neither inspection route fetches reports or
reads credentials.

## Validation

All capture XML and receipts in `tests/test_ibkr_inspection_cli.py` are explicitly
synthetic. Subprocess tests run direct and canonical CLI entry points with this
worktree on `PYTHONPATH`, including runs from an unrelated working directory.
They verify summary parity, private failure output, unchanged capture bytes, and
absence of a supplied database path. Additional checks cover a fresh direct
process without evidence imports, invocation-time/explicit aware cutoffs, each
legacy command's store requirement, actual store-backed query behavior, and
standalone compatibility.

Run from this worktree, using the existing root environment without installing:

```sh
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest \
  tests/test_ibkr_inspection_cli.py tests/test_ibkr_flex_intake.py \
  tests/test_ibkr_flex_capture.py tests/test_local_quant_flow.py --disable-socket -q
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m ruff check \
  src/data/__main__.py tests/test_ibkr_inspection_cli.py
```

This focused offline validation does not resolve the known full-suite failures
and setup errors dependent on missing positive income authority. No gates are
bypassed, skipped, or fabricated to address that dependency.

Checkpoint result: 97 focused offline tests passed; Ruff and `git diff --check`
passed. Validation covers the base reader implementation. The separately owned
security fix for private-file TOCTOU handling and same-second ACK generation
ordering remains an integration dependency; the coordinator must rerun combined
validation after that change. This CLI makes no new filesystem-safety guarantees
beyond the reader's implementation.
