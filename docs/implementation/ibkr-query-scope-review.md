# IBKR query scope review (diagnostic only)

Implemented on integration base `ab068b03e4c6b440de3b76b106f93f8d3ba784d2`
in the existing app-owned `build/ibkr-query-scope-review` worktree. Read repository
`CLAUDE.md`; no applicable `AGENTS.md` was found in checkout/ancestor searches.
No additional agents or threads were created.

The existing `read_flex_capture` already verified the original local receipt chain,
raw report hash and acquisition cutoff. The concrete missing capability was an
immutable configuration association and requested-versus-returned section review.
`read_flex_query_scope_review` adds that bounded capability in `src/data/ibkr_flex.py`.
It reuses the capture reader and `_private_read`/descriptor-based private-file helper.
There are no CLI, shared schema, source allowlist or income-ledger changes.

## Evidence boundary

The configuration snapshot is **opaque bytes**, such as an already saved screenshot
or operator note. This API neither assumes nor invents a native IBKR query export
format. The JSON schema below is an application-owned **operator declaration**.
Its exact bytes must have a separately retained SHA-256 pin passed as
`expected_declaration_hash`. Store snapshots and declarations privately, alongside
whatever original configuration evidence already exists; no new query, acquisition
or account activity is required for development.

The declaration pins the exact configuration bytes, original report bytes, original
capture binding and query-ID hash. Query ID alone is not a configuration version.
The capture binding includes the original manifest/receipt chain (including resume
ancestry when applicable). A rewritten report with consistently rewritten local
receipts still fails an old declaration binding. Replacing the declaration itself
fails its separately retained byte pin. These are **local consistency checks**, not
source authentication: an operator able to replace every artifact and every pin can
create another locally consistent review. Keep pins with the review evidence; do
not claim they are signatures or provider certifications.

The frozen result retains exact declaration bytes and hashes without exposing the
private declaration in its repr. Its deterministic review hash binds declaration,
configuration and capture/report hashes. No current-time call, timestamp restamp,
file write, provider request, canonical record or authorization is performed.

## Closed declaration schema

Every object rejects unknown or missing keys. This is the complete shape (hash
placeholders must be replaced with the exact lowercase SHA-256 values):

```json
{
  "schema_version": "ibkr-flex-operator-scope-1",
  "scope_authority": "operator_declared_not_provider_certified",
  "configuration_sha256": "<exact snapshot hash>",
  "report_sha256": "<exact original XML hash>",
  "capture_binding_sha256": "<read_flex_capture.source_binding_hash>",
  "query_id_sha256": "<read_flex_capture.query_id_hash>",
  "declared_at": "2026-10-30T16:00:00+02:00",
  "configuration_recorded_at": "2026-10-01T10:00:00+02:00",
  "query_version": null,
  "requested_sections": {
    "CashTransactions": ["amount", "currency", "transactionID"],
    "OpenDividendAccruals": null
  },
  "scope": {
    "accounts": {"mode": "selected", "values": ["synthetic-account"]},
    "models": {"mode": "none", "values": []},
    "asset_classes": {"mode": "all", "values": []},
    "currencies": {"mode": "all", "values": []},
    "security_ids": {"mode": "unknown", "values": []},
    "period": {"from_date": "20261001", "to_date": "20261030"},
    "timezone": "Europe/Rome",
    "date_format": "yyyyMMdd",
    "time_format": null,
    "detail_level": "detail",
    "currency_mode": "native",
    "output_format": "XML"
  }
}
```

- Selection modes are `all`, `selected`, `none`, `unknown`. Only `selected` has
  nonempty values; values must be unique. `none` is an explicit operator assertion;
  `unknown` records missing knowledge. Account/model selection, asset, currency and
  security filters remain private and are never inferred from the returned rows.
- Requested-section names come from the reader's existing nine-section `SECTIONS`
  vocabulary. A nonempty field list records explicit operator-selected names;
  `null` means field selection was not recorded. An empty list is rejected because
  it cannot unambiguously mean “all fields.” Unknown section names fail closed.
  Field names remain private; they are not a certified provider field vocabulary.
- `query_version` is a bounded private string or `null` for unknown. The immutable
  version association comes from the byte hashes, not the operator version label.
- Date-only periods use the reader's existing calendar-date parser and must be
  ordered. Timezone/date/time-format strings are private declarations or `null`;
  the API does not interpret them as timestamp-conversion contracts.
- Detail is `detail`, `summary`, `both`, or `unknown`; currency mode is `base`,
  `native`, `both`, or `unknown`. Output must be `XML` for this XML capture reader.
- Both declaration clocks are aware original instants. Configuration-recorded time
  must be at or before declared time, which must be at or before `as_of`. Original
  capture availability separately passes the existing cutoff checks. Retrospective
  declarations are allowed and preserve both clocks; they **never** establish that
  a saved configuration was effective at the original query time. Do not fabricate
  missing clocks to use the API.
- Private strings are bounded to 256 characters, lists to 256 entries, declaration
  bytes to 64 KiB and snapshot bytes to the existing 8 MiB limit. Empty snapshot
  bytes, unsafe permissions, links and duplicate JSON keys fail closed.

The API takes `directory`, keyword-only `configuration_path`, `declaration_path`,
`expected_declaration_hash`, and aware `as_of`. It returns `FlexQueryScopeReview`.
All reads require explicit caller-selected private paths. No discovery scans or
actual private-capture reads were used during this implementation.

## Safe review semantics

`safe_summary()` includes only fixed labels, counts, booleans, hashes and retained
clocks. It excludes account IDs, amounts, paths, query/version text, credentials,
selected field names and arbitrary configuration content. Do not serialize the
private declaration or capture as a public summary.

Each supported section reports whether requested, declared field count (`null`
when unrecorded or unrequested), returned status (`absent`, `empty`, `populated`),
rows, statements with a section, statements missing it, and statements with an
explicitly empty section. Thus an empty section in one statement cannot hide its
absence from another. The aggregate status describes returned data only and is
never a complete-window claim. Returned but unrequested sections remain visible;
unsupported returned sections retain the reader's count without exposing names.
No field-selection mismatch is inferred from a section without rows.

Account-set matching is an exact comparison only for `selected` accounts; other
modes report `null`, never a fabricated positive match. Period matching compares
every delivered statement's interval with the declared interval. Mismatches are
review observations, not silently widened scopes or intake authorizations. Model,
filter, field and format declarations are retained evidence, not certified checks
against broker configuration. A matching period is not an eligible income cutoff.

Every result retains all `BASE_GAPS`, including configuration verification,
source authentication/account mode, payment/deduction/revision linkage,
complete-window/late-amendment semantics and pre-ex terms/entitlement. No caller
flag can clear them: unknown authority/completeness flags are rejected even inside
the scope object. `scope_authority` is always
`operator_declared_not_provider_certified`, `income_admission` stays `blocked`, and
`canonical_income_records_created` stays zero. Fully matching declarations and
empty reports cannot authorize absence, completeness or payment claims.

## Validation

Synthetic files only, including fixed original clocks and a multi-statement case.
Import assertions resolve `src.data.ibkr_flex` under this worktree. Exact commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python -m ruff format src/data/ibkr_flex.py tests/test_ibkr_query_scope_review.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python -m ruff check src/data/ibkr_flex.py tests/test_ibkr_query_scope_review.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest tests/test_ibkr_query_scope_review.py tests/test_ibkr_flex_intake.py tests/test_ibkr_flex_capture.py tests/test_ibkr_private_files.py tests/test_ibkr_resume_query_history.py tests/test_ibkr_inspection_cli.py tests/test_operations_income_proof.py tests/test_action_coverage_accounts.py tests/test_research_income.py --disable-socket -q
git diff --check
```

Results: Ruff formatting completed (two files); Ruff check passed; **192 passed
in 14.39s**, with no skips. The new review suite contributes 33 tests. Diff
whitespace check passed. This is a focused regression result, not a full-suite
clean claim or actual RD-Agent/LEAN acceptance. Sufficient income source authority
remains blocked. Existing Docker WSL setup/build/import/CLI smoke is acknowledged;
none of it needs repeating for this increment.

No secrets, private reports, trades, provider calls, paid services, installations,
Docker operations, push, merge or PR were used. The only edited files are the
additive reader API, its owned synthetic test file and this note.
