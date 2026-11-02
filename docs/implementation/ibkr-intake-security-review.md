# IBKR intake security/correctness review

Reviewed baseline: `4cd1be3894875ab75fa68b383d832b11e9b41f99` (`build/full-system`). Review only; no runtime edits, source registration, positive income acceptance, or gate exceptions. All fixtures are SYNTHETIC. No actual reports, credentials, authenticated services, paid providers, or TCP connections were accessed.

Scope: `src/data/ibkr_flex.py`, the fetch/inspect/credential scripts, both pointer-only IBKR env configs, all three existing IBKR test files, and `src/operations/income.py`. Line references below refer to baseline source, not future fixes.

## Findings

### R1 — Low, conditional local race: reader checks ancestors before opening pathname

`src/data/ibkr_flex.py:296–318`: ancestor symlink checks and parent ownership/mode checks precede `os.open(path, O_NOFOLLOW | O_NONBLOCK)`. O_NOFOLLOW protects the final component only. Replacing the checked parent with a symlink between these operations causes the original reader to accept bytes through a linked ancestor. The final file can still satisfy all fstat checks.

Concrete repro: `test_review_reader_parent_replacement_bypasses_no_link_check` in the historical probe file below schedules a directory rename/symlink immediately before the real os.open. The reader returns `synthetic-replacement` instead of rejecting the link. No reader validation or returned bytes are mocked.

Impact: the advertised no-link property is not atomic; a capture may be read from a different directory than the checked one. This requires a concurrent filesystem mutator with permission to replace a path component (same-user process, or a suitably writable ancestor). It is not a remote XML exploit, broker authentication forgery, or demonstrated privilege escalation. Ordinary private files under stable trusted ancestry pass safely. No economic authority is acquired.

Narrow remedy: walk from a trusted directory descriptor using directory-relative opens with O_DIRECTORY/O_NOFOLLOW, validate the directory descriptors, and open files relative to the pinned parent. Keep bounded reads and final descriptor ownership/mode/link/type checks. Define a stable-directory policy rather than claiming protection against arbitrary same-owner content writes.

**Passing behavior after fix:** the scheduled parent substitution must either be rejected with a code-owned error or read the original file through a pinned directory descriptor. It must never return replacement bytes by following the injected ancestor symlink. A same-owner attacker can still alter files they own; descriptor pinning does not create broker authentication.

### R2 — Low, conditional local race: resume reopens files after lstat

`scripts/fetch_ibkr_flex_report.py:135–147`: resume validates source/ACK/receipt metadata, then uses Path.read_bytes on each pathname. Unlike the intake reader, this read has no final-component O_NOFOLLOW, descriptor checks, or bounded read.

Concrete repro: `test_review_resume_leaf_replacement_bypasses_no_link_check` replaces the checked ACK with a symlink to identical synthetic bytes at the first read boundary. Original `latest_generation` accepts reference `111` through that symlink. Matching content and hash do not repair the failed no-link property.

Impact: conditional on concurrent local mutation, symlinks and replacement permissions/types can bypass the prior checks. A FIFO replacement could block, or a grown file exceed the pre-read size bound; those consequences are inferred from unbounded read_bytes, not exercised with hangs/large allocations. This is defense in depth within a private local namespace, not an independent economic admission path.

Narrow remedy: share a descriptor-based private, bounded read primitive with intake, including protected parent traversal. Validate the actual opened file, not a previous pathname stat. Apply it to ACK and receipt, preserve closed diagnostics and original bytes.

**Passing behavior after fix:** the injected ACK leaf symlink is rejected before any GetStatement call, or an already pinned original file is used without following it. Returned reference through the newly linked file must not be accepted. Add separate desired-behavior tests for bounded growth, nonregular files and metadata checks as appropriate; never make a blocking FIFO probe hang the suite.

### R3 — Low correctness: same-second UUID ordering is not generation chronology

`scripts/fetch_ibkr_flex_report.py:83,124–132`: directories contain second-resolution timestamp plus random UUID; latest_generation sorts path names and selects the last. Two generations in one second are therefore ordered by random UUID, not acquisition time.

Concrete repro: `test_review_latest_generation_uses_random_uuid_order_within_second` creates two original pending captures through fetch_report/MockTransport, preserving receipt clocks. It assigns same-second names with an older `f…` UUID and a newer `0…` UUID. Selection incorrectly returns older reference `111` rather than newer `222`. Renaming is a deterministic fixture for valid names a UUID generator can naturally produce.

Impact: an operator may resume the wrong pending generation or receive a stale completed-source refusal. Concurrent CLI invocations can create same-second generations; no source authentication or payment status is granted.

Narrow remedy: choose by a validated acquisition clock or another explicit monotonic generation ordering, with a deterministic fail-closed policy for genuinely ambiguous ordering. Preserve query-hash checking and original-completed-source refusal; do not silently switch accounts or queries.

**Passing behavior after fix:** this fixture selects `222`, or explicitly refuses an ambiguity if chronology cannot be established. It must not select `111` merely because its UUID sorts later. Fix tests must assert corrected behavior, not the historical defect assertion.

## Verified positives and explicit no-findings

- Economic authority: no IBKR branch exists in income.py's closed `_parsed_capture` allowlist (lines 27–55). Positive authority documents still unconditionally fail (lines 156–169). Reader summaries always report blocked income and zero canonical records (ibkr_flex.py:159–166). No path was found that establishes payment, complete coverage, or mutates ledger/activation authority. Reversals/open accruals/summary rows remain observations; matching amounts/dates never create joins.
- Local binding: reader hashes complete manifest/receipts/raw bodies and resumed original ACK/receipt/resume metadata (lines 403–460,496–515), checks query hashes, sequence, final filename, status/version, statement counts and acquisition cutoff. Existing tamper tests pass, including an inherited receipt-only change altering binding. No full-byte binding omission found within this declared chain. The receipts do not independently attest the HTTP GetStatement reference parameter or authenticate broker origin. Hash consistency is deliberately not a broker signature; rewriting a coherent local chain is not reported as an authentication bypass.
- Identity/cutoff: optional expected account comparison and row account checks work; summary dash account is explicitly supported. read_flex_capture is a diagnostic multi-account reader, not an externally selected-account admission gate. Statement dates are native report scope, not acquisition eligibility; future native dates do not become authoritative by parsing. No finding requiring unsupported query configuration, live/paper authentication, pre-ex terms, or complete coverage to pass.
- XML: DTD/entity declarations, NULs, invalid UTF-8, invalid numeric fields and nonfinite/exponent amounts are refused. Added bounded resource probes verify rejection above 8 MiB, depth 8 and 100,000 nodes. Node/depth checks happen **after** ElementTree allocation; these are acceptance bounds, not strict peak-memory bounds. The fetch parser has byte-bounded HTTP input but no matching structural limits. No XML external-resource access or entity-expansion bypass found. Streaming early structural enforcement is optional hardening; no measured unbounded resource exploit is claimed.
- Privacy: static pointer-only configs contain no secret values. Credential launcher captures provider output, uses fixed diagnostics, isolates Service Account token from the check/fetch child and drops unrelated ambient secret references. HTTP errors are classified without rendering their messages; inspect CLI exposes fixed structural summaries and closed errors. No demonstrated credential/provider-text leak through these supported CLI paths. Observation dataclasses intentionally retain native data: arbitrary repr/attributes are not public summary APIs.
- Pending/resume: existing tests verify bounded polling, same-reference retrieval, no SendRequest on continuation, private output and corrupt source refusal. Repeated continuation using the same original generation, even after a separate resumed capture, is documented in operations.md; **not a finding**. Token-wide coordination across independent processes is also explicitly outside the per-call polling bound.
- Static private checks reject symlinks, hardlinks and shared permissions in existing tests. R1/R2 concern atomicity under concurrent mutation only; they do not invalidate the tested stable-path checks.

## Reproduction and validation

Historical probes: `docs/implementation/review-probes/ibkr_intake_baseline_probes.py`.
They are deliberately outside default test discovery. Three tests assert baseline faults; three parameterized resource tests assert desired rejection. They are audit evidence, not regression acceptance tests. Do not integrate fault assertions into the default suite, mark them skipped, or invert their meaning to claim a fix.

Against a checkout of baseline `4cd1be3` containing this historical probe artifact, run from that checkout:

```sh
PYTHONPATH="$PWD" /home/ozzstein/AgenticQuant/.venv/bin/python -m pytest --disable-socket docs/implementation/review-probes/ibkr_intake_baseline_probes.py tests/test_ibkr_flex_capture.py tests/test_ibkr_flex_intake.py tests/test_ibkr_flex_credentials.py -q
/home/ozzstein/AgenticQuant/.venv/bin/ruff check docs/implementation/review-probes/ibkr_intake_baseline_probes.py
/home/ozzstein/AgenticQuant/.venv/bin/ruff format --check docs/implementation/review-probes/ibkr_intake_baseline_probes.py
git diff --check
```

Validation: 100 passed with sockets disabled; Ruff check/format and diff check passed. Actual parser, fetch worker and income imports were asserted to resolve beneath this T3 worktree, not the shared interpreter's editable root checkout. All filesystem probes use temporary synthetic private files; HTTP uses MockTransport. The full suite was not rerun: the known 3 failures/18 setup errors requiring positive income authority remain unresolved, unmodified and unwaived.

Unresolved dependencies: separate implementation must close R1–R3 and add desired-behavior tests. No runtime fix is supplied here. Positive-source scope/authentication/economic proof remains a separate unresolved requirement, not requested by this review. No model/provider, credential, deployment, PR or integration action is needed to reproduce these findings.
