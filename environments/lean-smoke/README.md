# Actual LEAN paper-engine diagnostic

This is a **synthetic offline runtime diagnostic**, not a trading strategy, broker
connection, or release-acceptance certificate. The current pinned runtime fills
and reconciles the paper order but exits 139 during Python shutdown. A minimal
algorithm without project imports reproduces that failure.

Pinned upstream source: QuantConnect/Lean
`3b2bf19e597805cc917dd199ed9ee7d10d5ceb84`.

The Dockerfile is built with that public source checkout as its context. It uses
our existing `quantagentlab:smoke-a233bb5` image for Python/project dependencies
and the recorded .NET SDK digest. Preserve the base image ID in run evidence;
the local tag is not a portable dependency lock. Copy `QalOfflineSmokeDataQueue.cs`
into the checkout's `Engine/DataFeeds/Queues/` before building. It subclasses the
upstream **test-only** FakeDataQueue at two data points per second per symbol.
Prices use the upstream wall-clock synthetic generator; this is not a historical
market-data replay. Keep `.git`, upstream tests and unnecessary market data out
of the build context; retain `Data/market-hours` and `Data/symbol-properties`.

Prepare a new private fixture from the repository root:

```bash
PYTHONPATH="$PWD" .venv/bin/python environments/lean-smoke/prepare.py \
  --source /path/to/pinned/Lean --output /path/to/new/private/run
```

The fixture uses existing `json5` to read upstream's commented configuration.
It calls the production `prepare_project`, preserving the complete execution
implementation manifest; any subsequent source edit requires a fresh preparation.

Run the pinned image with `/app/src` mounted read-only from this checkout and the
fixture mounted at `/smoke`. Set `--network none --cap-drop ALL
--security-opt no-new-privileges --memory 4g --memory-swap 4g --cpus 2
--pids-limit 256`, BLAS/OMP threads to one and HOME to `/smoke`.
Use a unique container name, put creation inside cleanup protection, retain its
exact ID/inspection, impose a 150-second outer timeout, and remove only that
container afterward. The fixture must be writable by the image's UID 10001;
keep its parent private. Do not mount credentials or a real ledger.

The engine command is `--config /smoke/engine.json`. No live authorization is
set. The actual live-mode engine uses **PaperBrokerage**, synthetic cash and the
local fake data queue. Its real order event is passed through the canonical
execution bridge, ledger reconciliation and compliance reader.

`result.json` means the order and ledger checks completed, **not** that the
process exited successfully. Require separate exit/cleanup inspection. Also
retain `exact-decimal.json`, `exact-clock.json`, ledger and console output.
The decimal check preserves `100000.000000000000000000001` through CLR cash
storage despite direct Python access returning float. The clock check preserves
original 100 ns ticks and rounds forward only at the Python microsecond boundary.

See `docs/implementation/lean-runtime-smoke-2026-11-03.md` for actual results and
remaining acceptance gaps. No DAY/trailing/restart/native broker acceptance is
inferred from this one GTC market order.
