# IBKR private-file hardening

Scope: bounded fixes for R1–R3 in review commit
`85f0a192eb0a9d09c9a84bcea08134fde0ee0f4a`, reviewed base `4cd1be3`;
implementation starts from integrated `e9d36b8`. The historical fault-expecting
probes remain review evidence, outside default test discovery. Desired-behavior
regressions are in `tests/test_ibkr_private_files.py`.

## Private filesystem operations

`src/data/private_files.py` walks absolute paths from an opened `/` descriptor,
using directory-relative `O_DIRECTORY | O_NOFOLLOW` opens for every component.
It rejects relative paths, parent traversal, NULs, and unsupported POSIX
capabilities. Public ancestors (including `/tmp`) remain compatible; the actual
opened immediate parent must be a directory owned by the current uid with no
group/other permission bits. Parent substitution after opening pins the original
directory; substitution before opening fails instead of following a symlink.

Reads open a single leaf relative to that parent with `O_NOFOLLOW | O_NONBLOCK`.
Before and after reading, fstat checks that exact file for current ownership,
private permissions, regular type, one link, and size. Reads consume at most
limit + 1 bytes and reject growth past the bound. Report/ACK bounds stay 8 MiB;
receipt bounds are 64 KiB. Every acquired descriptor closes on normal and error
paths, including validation/read/write/fsync failures.

Capture creation uses descriptor-relative mkdir and validates the opened new
directory. Exclusive private writes open relative to a verified parent, check
the opened file, handle partial writes, and fsync. Parent swaps cannot redirect
report bytes into a symlink target. Existing leaves, including links/FIFOs, are
never overwritten. Original private captures need no migration. Public parse,
read and summary signatures, retry/resume behavior, and closed CLI error labels
are preserved. The directly executed worker explicitly imports this checkout's
helper rather than an unrelated editable installation.

## Resume ordering

Resume scans the opened base descriptor and reads each original SendRequest ACK
and receipt through an opened private generation directory. It verifies raw hash,
query binding, operation/version/status, successful ACK and reference, duplicate-
free JSON, and aware started/available clocks. Only original SendRequest receipts
participate; empty and resume-only generations do not become new generations.
Selection derives solely from receipt intervals, never UUID, directory timestamp,
or mtime. Equal start clocks, overlapping/touching intervals, or backwards clocks
fail closed with the existing `resume_source_unavailable_or_invalid` label.
The latest completed original still refuses resume rather than choosing an older
pending request. Corrupt retained candidates fail closed rather than being skipped.
There is no hidden recapture or SendRequest on resume.

## Validation and limits

All tests use temporary synthetic bytes, HTTPX MockTransport and disabled sockets.
Deterministic hooks run at the actual descriptor-relative open boundary and assert
that the hook fired: reader parent swap returns original bytes; resume leaf swap
fails; same-second inverted UUIDs select newer reference 222 from retained original
receipt clocks. Further tests cover modes, ownership, hardlinks, FIFOs, parent
links, growth, invalid paths/capabilities, write/create swaps and descriptor leaks.
Existing tests cover original retries, resume chains and acquisition cutoffs.

Validation: 133 focused IBKR tests pass with `--disable-socket`; Ruff check/format
and `git diff --check` pass. Imports were checked against this worktree using the
root interpreter and explicit worktree PYTHONPATH. CLI integration tests are
owned and run by the coordinator; no CLI runtime/test files are changed here.

This protects pathname traversal and bounds each file read, not arbitrary hostile
same-uid processes that can rewrite owned files, change permissions, or alter an
already opened inode. It does not provide an atomic snapshot across multiple files
or serialize concurrent capture processes. Receipt clocks are local consistency
evidence, not authenticated broker time. Full-suite positive-income failures are
outside this bounded work; source authentication, positive income admission and
completed release remain blocked. No broker calls, secrets, real reports,
provisioning, push, merge or PR are involved.
