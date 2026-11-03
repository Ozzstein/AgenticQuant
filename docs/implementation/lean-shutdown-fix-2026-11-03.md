# LEAN shutdown fix and repeated paper-engine test

## Outcome

The same canonical synthetic paper-order scenario now completes with **process
exit 0**, actual Submitted/Filled OrderEvents, one reconciled fill and execution
compliance PASS. A fresh read-only ledger connection after process shutdown also
returns PASS with one fill. Exact decimal and original-tick assertions remain
active. The container was removed successfully; inspection showed network=none,
no OOM and exit0. This resolves the shutdown blocker in this locally patched,
pinned runtime; it is not a claim of upstream acceptance or broad broker coverage.

Final image: `quantagentlab-lean:source-3b2bf19-gilfix`, ID
`sha256:94ee09ba94c575561b11bc19cc6f5406f507d1f2fa436449b13ea3532612b2d2`.
Paper run container:
`ed1d1366a3a05172763db96eabfb0358d9d3c781d466e3b68467f124ab7ad651`.
The original unpatched image and failing logs remain available for comparison.

## Source-backed diagnosis

The pinned [LEAN initializer](https://github.com/QuantConnect/Lean/blob/3b2bf19e597805cc917dd199ed9ee7d10d5ceb84/Common/Python/PythonInitializer.cs)
creates an outer `Py.GIL()` disposable in Shutdown and never disposes it. The
actual NuGet dependency is QuantConnect.pythonnet 2.0.68; its package repository
metadata identifies commit `300a6924b99600af9c0f30b629be07f8da11975f`, available in
the QuantConnect fork (the package repository URL itself names pythonnet/pythonnet).

At that exact revision:

- [Py.GILState](https://github.com/QuantConnect/pythonnet/blob/300a6924b99600af9c0f30b629be07f8da11975f/src/runtime/Py.cs)
  deliberately throws from its finalizer when not disposed.
- [Runtime.Shutdown](https://github.com/QuantConnect/pythonnet/blob/300a6924b99600af9c0f30b629be07f8da11975f/src/runtime/Runtime.cs)
  acquires the native GIL, performs collection/teardown, releases its own handle
  and releases any remaining current thread state. Forced collection exposes the
  abandoned outer managed handle.
- [PythonEngine.Shutdown](https://github.com/QuantConnect/pythonnet/blob/300a6924b99600af9c0f30b629be07f8da11975f/src/runtime/PythonEngine.cs)
  delegates runtime teardown to that routine. LEAN has no additional registered
  PythonEngine shutdown handlers in the inspected source.

The local patch removes the redundant outer acquisition. It does not dispose a
handle after interpreter teardown, disable finalization, suppress exceptions,
skip PythonEngine.Shutdown, or override the process exit status. The engine's
normal shutdown now reaches `PythonInitializer.Shutdown(): ended` and
`Program.Main(): Exiting Lean...`.

## Patch/build boundary

Committed patch: `environments/lean-smoke/patches/shutdown-gil.patch`.
The Dockerfile requires the original initializer SHA256:
`ca4b5c7374c096309fb8b37f4ae3b29895f00b88b1e56e3db7bb95d489ca896c`,
then checks/applies the patch before compiling. Upstream source stays pinned to
`3b2bf19e597805cc917dd199ed9ee7d10d5ceb84`; .NET SDK digest, project dependency
base and test-only synthetic feed remain as documented in the original smoke
report. No application Python/accounting/risk code changed in this checkpoint.
The original source checkout itself was not patched: application happens inside
the compiler build stage. Changed upstream code requires deliberate re-review.

Actual final build command (public dependency restore, no credentials):

```bash
docker --host unix:///var/run/docker.sock --config /tmp/qal-lean-docker-config \
 build --file /tmp/qal-lean-build-source/Dockerfile.qal-shutdown \
 --tag quantagentlab-lean:source-3b2bf19-gilfix /tmp/qal-lean-build-source
```

The recipe above is copied from the committed smoke Dockerfile; context includes
`qal-shutdown.patch` and the existing test-only feed subclass. Dependency/compiler
warnings remain in the build log; no warning-free claim is made.

## Verification

1. Prior unpatched no-project control and paper fixture both exited139 at the
   same GILState finalizer (original evidence retained).
2. First patched no-project control exited0 and cleanup0.
3. Final hash-guarded image ran the unchanged production paper fixture:
   original real event path, exact monetary/tick checks, canonical reconciliation,
   read-only reopen, compliance PASS, exit0 and cleanup0.
4. Final no-project/no-order control also exercises normal Quit/shutdown on that
   exact final image, returning exit0 and cleanup0. Its container was
   `8d56d2f6f61004d07bfd433f11cef42dcff1551b4ce35ab0f5f8bad7c95b52db`.
   The fixture is committed as `shutdown_control.py`.

All runtime tests use synthetic local data and PaperBrokerage, network disabled,
4 GiB memory/2 CPUs/256 PIDs, no added capabilities and no-new-privileges. Each
run has its own unique container identity and bounded outer timeout. No real
broker calls, credentials or live authorization were used. The application
Python code is unchanged from the prior 121-passing-test checkpoint, so those
unit suites were not repeated as a substitute for actual runtime verification.

Private evidence: `~/.local/share/quantagentlab/runtime-smoke/lean-shutdown-20261103/`:
both clean control runs, actual paper result/ledger, inspections/exit/cleanup,
exact-decimal/tick outputs, source references, final build log and SHA256 inventory.

Remaining scope: this is one GTC market buy with synthetic data and zero simulated
fee. DAY/trailing runtime overloads, nonzero fees, partial fills, cancellation,
restart/recovery and authenticated broker acceptance still need distinct tests.
No sufficient-income, trading efficacy or full production-release claim follows.
