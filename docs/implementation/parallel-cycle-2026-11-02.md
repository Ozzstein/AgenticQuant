# Parallel development checkpoint — 2026-11-02

The operator requested parallel threads with explicit cost/complexity-based model
and reasoning choices. Live orchestrator catalog was checked before dispatch;
launch/delegation results confirmed the requested provider/model/options. Every
worker used an isolated T3-bound worktree. Standard tier (`serviceTier=default`)
was explicitly selected; priority was not requested. No model parameter is merely
asserted in the task prompt.

- **Runtime inventory:** Codex/gpt-6.1-terra, low effort, Standard. Straightforward
  metadata/PATH/Docker-readiness checks and documentation suited the economical
  worker. Thread `mcp:eecfdd86-4b03-4691-ad44-2103c85e5a80`, source commit `0671850`,
  integrated as `e9d36b8`. Docker WSL integration remains unavailable; lean/dotnet
  absent from PATH. No provisioning or runtime acceptance inferred.
- **Canonical IBKR inspection CLI:** Codex/gpt-6.1-sol, high effort, Standard.
  Coding and compatibility/privacy boundaries justified the stronger coding
  model and deliberate reasoning. Thread `mcp:6c0f5892-7665-4e5f-866d-2ba232f4d057`,
  source `d2c225c`, integrated as `309754d`. Inspection dispatches before store
  imports/creation; nine legacy commands retain required store selection.
- **Initial independent security review:** Codex/gpt-6-astra, high effort,
  Standard. Security/concurrency analysis justified the frontier reviewer.
  Thread `mcp:694fc8b8-8e38-492d-85ce-a7c58735abc4`, source `85f0a19`, integrated
  as `5ace22b`. Two conditional local filesystem races and same-second UUID
  ordering reproduced. Historical fault assertions remain outside default
  regression discovery, with pinned original source and explicit commands.
- **Private-file implementation:** Codex/gpt-6.1-sol, high effort, Standard.
  Descriptor lifetimes, read/write/create races and receipt chronology required
  stronger coding/concurrency reasoning. Thread
  `mcp:cd74630b-1c7a-4757-acfd-b6bc2f741d46`, source `43f514a`, integrated as
  `447876d`. Shared descriptor-relative helper and desired-behavior tests close
  the initial findings; public intake/summary signatures stay unchanged.
- **Post-fix review round 1:** new app-owned delegated task, Codex/gpt-6-astra,
  high effort, Standard; reviewed `447876d`. Found one valid-previous-query
  compatibility regression, no additional filesystem/authority finding.
  Coordinator Codex/gpt-6.1-sol/medium/Standard applied the small prescribed
  correction and desired-behavior tests as `45a10a2`, rather than spending a
  separate high-effort implementation turn on that mechanical change.
- **Post-fix review round 2:** distinct new delegated task, Codex/gpt-6-astra,
  high effort, Standard, full prior brief/findings/responses supplied. Reviewed
  `45a10a2`, no findings. Confirmed original race fixes, current-query/latest
  behavior, no corrupt-history skipping or fallback, and safe CLI boundaries.

Root validation at `45a10a2`: **186 passed in 14.73s**, sockets disabled, across
query-history, private-files, canonical CLI, original capture/intake/credentials,
operating-income proof, action coverage and actual local Qlib flow. Changed Python
Ruff checks/format and diff check pass. Final reviewer separately reported 143
focused tests/15.06s plus ten temporary synthetic probes, all passing; these
counts overlap. The existing actual private paper capture also passed canonical
CLI inspection with all original file hashes unchanged; no financial values were
published. No broker request, paid research-provider call or ledger write was
performed in this cycle. Worker model execution itself is not represented as free.

No full-suite rerun or release acceptance is claimed. Known missing-positive-income
failures remain enabled. Local capture hash consistency is not signed broker
provenance, account-mode authentication, pre-ex evidence or complete coverage.
The reviewed query-editor screenshots establish selected fields only, not complete
provider delivery. Actual payment/deduction linkage and complete income source
scope remain unresolved. Hostile same-UID content mutation, atomic multifile
snapshots and process coordination remain outside the filesystem guarantee.

Completed changes are integrated on `build/full-system`; master is unchanged.
The operator's separate `docs/TASK-09-REGIME.md` edit is preserved, not committed.
No credentials, screenshot files or original financial report bodies are in this
checkpoint. This evidence document and final review notation change no runtime.
