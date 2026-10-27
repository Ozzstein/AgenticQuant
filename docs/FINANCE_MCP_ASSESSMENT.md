# Finance MCP assessment for the complete QuantAgentLab system

Assessment date: 2026-10-27. Scope: the complete technical brief and [full system build plan](FULL_SYSTEM_BUILD_PLAN.md). Starting catalog: [BlockRunAI/awesome-finance-mcp](https://github.com/BlockRunAI/awesome-finance-mcp).

Selection status: the [accepted MCP integration decision](MCP_INTEGRATION_DECISION.md) records the chosen integrations and conditional alternatives. This assessment supplies the supporting evidence.

## Verdict

Use selected MCP integrations for agent evidence retrieval, research orchestration, and operational inspection. They can reduce connector work and improve the agents' access to real information. They do not supply the complete quantitative, portfolio, execution, or research-validation system.

The selected integration set is **OpenBB's data MCP extension, a narrowly configured FMP MCP server, restricted Alpaca observation tools, and Valyu for cited research**. Source macro data from FRED/ALFRED through OpenBB or a direct adapter that preserves vintages. **DexPaprika** is conditional on a defined need for DEX/token context. **Massive** remains a data-provider alternative; **QuantConnect MCP** is conditional on choosing its paid cloud research environment alongside the required working LEAN integration.

These are provider and interface choices inside the full build, not reduced delivery phases. Both asset classes, all strategies/models/optimizers, RD-Agent, research promotion, production execution, dashboards, and acceptance gates remain required.

## What was verified

Reviewed the catalog, upstream READMEs and current provider documentation, GitHub repository metadata, and selected implementation files for OpenBB, FMP, FRED, Alpaca, Financial Datasets, and CCXT. Compared their interfaces with this repository's data tools and agent nodes. Thirteen relevant entries received individual assessment below; unrelated categories were screened for scope fit.

This is a documentation and targeted source review. No candidate was installed, no authenticated data or brokerage calls were made, and no vendor test suites were run. Coverage, latency, entitlement, and production reliability remain to be demonstrated with actual integrations. Repository activity and the presence of tests are useful signals, not evidence that a server is production-ready.

## Candidate decisions

| Candidate | Fit for the brief | Recommendation and constraints |
|---|---|---|
| [OpenBB MCP](https://github.com/openbq-org/OpenBB/tree/develop/openbb_platform/extensions/mcp_server) | Shared access to equity, macro, news, regulatory, and other installed data extensions. Python-facing platform fits the existing stack. | **Selected shared data interface.** Run separately with pinned extensions and selected data tools. It is an integration platform, not a bundled entitlement to every dataset. Check the exact release's provider coverage. |
| [FMP MCP](https://github.com/imbenrabi/Financial-Modeling-Prep-MCP-Server) | Financial statements, analyst data, SEC filings, insider/institutional data, earnings calendars and transcripts. Strong match for fundamental, sentiment, and transcript agents. | **Selected targeted integration.** Community-maintained wrapper, not an official FMP server. Self-host; expose relevant toolsets only. Retain a direct FMP adapter for scheduled ingestion. |
| [Alpaca MCP](https://github.com/alpacahq/alpaca-mcp-server) | Account visibility, orders/positions, market data, corporate actions, news and calendars. Provider-maintained. | **Selected for observation.** Exact read-tool allowlist enforced outside the LLM. Orders still flow through QuantAgentLab's risk service and execution owner. Toolset-level filtering alone is insufficient. |
| [FRED MCP](https://github.com/stefanoamorelli/fred-mcp-server) | Economic series discovery and retrieval for macro analysis. Community implementation. | **Standalone wrapper not selected unchanged.** Use OpenBB or a direct FRED/ALFRED adapter that preserves vintage metadata. The wrapper remains an adaptation candidate if needed; it is an AGPL-3.0 component. |
| [Valyu](https://mcp.valyu.ai/) | Cited web/specialist-source retrieval for research hypotheses, company evidence, and sentiment investigation. | **Selected for research.** Use the current provider-hosted service rather than the old repository linked in the catalog. Verify dataset access; budget search separately from charged background research. No automatic candidate promotion. |
| [DexPaprika](https://github.com/coinpaprika/dexpaprika-mcp) | Token identity, pools, DEX liquidity and OHLCV; useful additional crypto context. Provider-maintained. | **Conditional addition.** Useful when DEX information matters to the chosen crypto universe. Does not establish Binance/Kraken execution liquidity or provide the complete funding/OI/BTC-dominance feature set. |
| [Massive MCP](https://github.com/massive-com/mcp_massive) | Broad market/reference data, corporate actions and historical prices. Provider-maintained. | **Data-provider alternative; MCP not selected.** Explicitly experimental. Uses Python 3.12+, so isolate from the repo's 3.11 runtime if adopted. For training data, ingest through an audited batch adapter and persist artifacts. |
| [QuantConnect MCP](https://github.com/QuantConnect/mcp-server) | Actual cloud projects and backtests; relevant to external research comparison and LEAN workflows. | **Conditional cloud integration.** Current preferred interface is hosted and requires a paid agent node. It does not complete the repo's LEAN bridge, reproduce the entire local portfolio ledger, or replace RD-Agent. Restrict live deployment and deletion tools. |
| [Financial Datasets MCP](https://github.com/financial-datasets/mcp-server) | Compact Python tools for statements, prices, news, crypto and SEC filings. Provider-owned repository. | **Alternative or cross-check.** Simpler surface than FMP, but the inspected statement tools have no as-of/publication cutoff. Useful for current evidence; insufficient as the complete historical fundamentals contract. |
| [Alpha Vantage MCP](https://github.com/alphavantage/alpha_vantage_mcp) | Official access to market data and financial functions. | **Alternative provider.** Avoid redundant subscriptions and differing indicator implementations. Ordinary free access is 25 requests/day; provider-approved open-source/educational exceptions exist. Full-system throughput needs verified entitlement. |
| [CCXT MCP](https://github.com/Nayshins/mcp-server-ccxt) | Seven basic crypto price, volume, exchange and OHLCV tools. | **Do not adopt as a core dependency.** The repo already uses CCXT directly. This wrapper does not expose funding/OI tools, and its OHLCV path makes a single fetch rather than a complete historical paging loop. |
| [Finnhub MCP](https://github.com/sverze/stock-market-mcp-server) | Basic symbols, metrics, news, quotes and candles. Community wrapper. | **Low incremental value.** Preserve and strengthen the existing Finnhub integration; use shared tools rather than a second connector for the same source. |
| [Yahoo Finance MCP](https://github.com/maxscheijen/mcp-yahoo-finance) | Company information and market data through Yahoo. Community wrapper. | **Low incremental value.** Existing yfinance support remains useful. Another wrapper does not establish point-in-time fundamentals, survivorship-safe coverage, or data-service guarantees. Upstream describes its interface as early development. |

### Concrete findings affecting adoption

**OpenBB needs a deliberately restricted deployment.** The inspected extension supports category restrictions, discovery and multiple transports, but also exposes an `install_skill` operation that writes files, optional CLI dispatch, and generic tool/pipeline dispatch. Disable or block irrelevant mutation/dispatch paths, enforce the underlying tool allowlist, and use a read-only filesystem for bundled resources. Merely hiding a tool from the initial list is not an authorization boundary. Its current repository redirects from `OpenBB-finance/OpenBB` to `openbq-org/OpenBB`; the inspected development-branch LICENSE and extension metadata say Apache-2.0. Pin and inspect the release actually deployed. [Extension documentation](https://github.com/openbq-org/OpenBB/blob/develop/openbb_platform/extensions/mcp_server/README.md), [license](https://github.com/openbq-org/OpenBB/blob/develop/LICENSE).

**FMP can fill real agent gaps, with paid entitlements.** The wrapper has actual earnings-transcript tools with symbol/year/quarter inputs, plus relevant statement, filing, analyst, insider and institutional toolsets. Our existing `src/data/transcript_data.py` already calls FMP, but truncates the transcript before downstream analysis. Preserve the full document in the evidence store and return bounded excerpts with retrieval references. FMP currently states that transcripts require at least Ultimate. Its latest-transcript ordering follows call date rather than availability date, so it cannot by itself establish historical publication timing. [Tool catalog](https://github.com/imbenrabi/Financial-Modeling-Prep-MCP-Server/blob/main/docs/API_REFERENCE.md), [implementation](https://github.com/imbenrabi/Financial-Modeling-Prep-MCP-Server/blob/main/src/tools/earnings-transcript.ts), [provider terms of access and timing](https://site.financialmodelingprep.com/developer/docs/stable/search-transcripts).

**Alpaca's account category is not read-only.** Its `account` toolset contains `patchAccountConfig`; `trading` mixes reads with order modifications, cancellation, liquidation and option actions. Allow specific inspection tools rather than entire categories. Current V2 also documents breaking changes from V1, making pinned contracts necessary. [Toolset implementation](https://github.com/alpacahq/alpaca-mcp-server/blob/main/src/alpaca_mcp_server/toolsets.py), [V2 documentation](https://github.com/alpacahq/alpaca-mcp-server).

**FRED supports vintage inputs but loses evidence on output.** The actual schema includes `vintage_dates` and `output_type`, beyond the README's parameter list. However, `getSeriesData` maps observations to only `date` and `value`, dropping per-observation revision fields. Preserve the original response and requested vintage; verify each output format. Macro features must use information available at the simulated decision time. [Schema](https://github.com/stefanoamorelli/fred-mcp-server/blob/main/src/fred/tools.ts), [transformation](https://github.com/stefanoamorelli/fred-mcp-server/blob/main/src/fred/series.ts), [official FRED fields](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).

**The catalog's “free” labels do not describe a complete operating budget.** Open-source wrappers do not pay for data entitlements, historical depth, exchange feeds, research jobs or LLM tokens. Alpha Vantage's standard free limit is unsuitable for an uncached multi-agent daily universe; QuantConnect remote MCP requires a paid node. DexPaprika allows keyless access to some data, but token-level aggregated OHLCV requires a Dev/Pro plan. [Alpha Vantage limits](https://www.alphavantage.co/support/), [QuantConnect requirements](https://www.quantconnect.com/docs/v2/ai-assistance/mcp-server/key-concepts), [DexPaprika access distinctions](https://github.com/coinpaprika/dexpaprika-mcp).

**Current research services can differ substantially from directory links.** Valyu's linked Python repository last shows a push in April 2025. Its current hosted service documents search, source discovery and background research with separate charges. Date-filtered search is useful evidence retrieval; it does not prove that results, document revisions, or the LLM itself were available at a historical date. [Current Valyu MCP](https://mcp.valyu.ai/), [listed repository](https://github.com/valyuAI/valyu-mcp).

## Maintenance evidence

GitHub `pushed_at` values observed during this review are below. These can include branch or maintenance activity; they are not release dates or support commitments. Test files/workflows were inspected for existence, not executed.

| Repository | Last observed push | Evidence affecting deployment |
|---|---|---|
| OpenBB | 2026-10-24 | Dedicated MCP tests/workflow; development and released versions require separate pinning. |
| FMP wrapper | 2026-10-14 | Unit tests and CI; README requires Node 25.3+ for v2.6+. Separate runtime needed. |
| Alpaca | 2026-10-26 | CI and release workflows; V2 interface migration. |
| FRED wrapper | 2026-08-22 | Tests, conformance workflow, retry/cache/rate-limit configuration. Output provenance still needs repair. |
| Massive | 2026-10-08 | Provider-owned, explicitly experimental, Python 3.12+; dynamic endpoint discovery can change the tool surface. |
| DexPaprika | 2026-10-21 | Local and hosted tool sets differ; inspect the actual deployment's tools. |
| QuantConnect | 2026-10-26 | Current repository contains README and LICENSE; hosted implementation cannot be audited from that tree. |
| Financial Datasets | 2025-06-05 | Small readable wrapper; age and missing historical-cutoff contract warrant validation. |
| Alpha Vantage | 2026-10-03 | Official local/hosted options; entitlement and throughput remain separate concerns. |
| CCXT wrapper | 2025-02-17 | README says tests are to be implemented; limited interface duplicates existing CCXT use. |
| Listed Valyu wrapper | 2025-04-04 | Evaluate the current hosted service independently of this old wrapper. |

The provider and repository links in the candidate table identify the metadata sources. No claim of a comprehensive dependency/security audit is made.

## Integration into this repository

### Shared evidence contract

Add one application-owned provider/evidence layer behind the existing `src/data` tools. Both direct API adapters and MCP adapters implement the same typed capabilities. Use a shared MCP connection manager with timeouts, bounded retries, quotas and explicit server/tool identities. The application, not individual agents, owns credentials and provider selection.

Every response needs provider/tool/version, canonical instrument identity, event time, publication/availability time where known, ingestion time, requested as-of cutoff, units/currency, adjustment policy, source URL or document ID, raw-payload hash, and quality status. Unknown timestamps remain unknown; do not invent point-in-time certainty. Preserve full documents and tabular artifacts outside the LLM context. Bounded excerpts link back to that evidence.

Historical runs must read immutable snapshots and reject evidence unavailable at the cutoff. Live evidence must enforce freshness. On a provider failure, return an explicit unavailable/stale state; never synthesize a price or silently change a dataset's provenance. Compatible fallbacks must be declared and recorded.

### Agent wiring

The current fundamental node manually invokes imported LangChain tools and then sends collected text to a structured-output LLM. Adding MCP configuration to the coding assistant will not connect these application agents. The same application-level integration must serve the CLI, scheduler and dashboard.

- **Fundamental:** statements, filings, financial ratios, analyst/insider/institutional evidence; FMP/OpenBB and existing direct sources.
- **Sentiment and transcript:** news, full-document retrieval and cited excerpts; FMP and Valyu as appropriate to the evidence requested, with publication timing and deduplication.
- **Technical:** canonical market snapshots and indicators computed by our versioned factor engine. External indicators are comparisons, not silent replacements for model features.
- **Macro:** versioned macro series plus our market-derived breadth, volatility and momentum signals. FRED alone does not provide every specified regime input or make a substitute USD index equivalent to DXY.
- **Risk and strategist:** internal portfolio, constraints, exposures, evidence and binding veto state. Broker observations may inform reconciliation; no direct order tools are available to these agents.
- **Research:** source discovery, experiment submission and results inspection under explicit budgets. External backtests must carry data/configuration/code/cost assumptions into independent validation.

Use OpenBB and FMP for complementary capabilities. If the pinned OpenBB provider already satisfies a particular FMP capability, expose that route once instead of adding a duplicate standalone tool. Prefer one primary source per dataset with explicit secondary checks.

### Data, research and execution boundaries

Bulk market history, funding/OI ingestion, universe reconstruction and training datasets should use deterministic scheduled adapters and durable storage. MCP can expose queries over those results; it need not move every row through an LLM conversation.

Execution remains `portfolio target → deterministic risk → durable order intent → Alpaca/CCXT/LEAN adapter → reconciliation`. Only the execution owner holds order authority. MCP reads are not the sole portfolio source of truth. Any future MCP execution transport would need the same idempotency, order-state, timeout and reconciliation contracts as the direct adapters.

The mandatory local LEAN work remains in scope even if QuantConnect cloud MCP is selected. Likewise, vendor research reports do not replace actual RD-Agent scenarios, the custom crypto research loop, validation, promotion, or rollback.

### Deployment and operating costs

Use isolated, version-pinned services or environments. This avoids introducing OpenBB's dependencies, Massive's Python 3.12 requirement, or FMP's Node runtime into the Qlib/RD-Agent environment. Local stdio or authenticated private HTTP are appropriate deployment options; choose per server and operating topology.

Enforce exact tool and argument permissions at dispatch, including targets nested inside generic `call_tool`/pipeline operations. Reject unexpected tool-schema changes until reviewed. Treat external news, filings and tool responses as evidence, not executable instructions. Keep provider credentials out of prompts and redact request logs.

Calculate the budget from universe size × required datasets × refresh rate × pagination × research frequency, then account for cache reuse, provider quotas, data subscriptions, compute and LLM spend. Fetch shared facts once per run, not once per analyst or debate round. No defensible total monthly price can be inferred from this catalog alone.

## Required acceptance evidence

For each selected integration, completion means:

1. A pinned server/runtime/provider manifest and captured actual tool schemas, with only intended operations reachable.
2. Real authenticated calls proving the required symbols, historical depth, data fields, paging and entitlements. Distinguish unavailable, empty and permission-denied responses.
3. Direct-source comparisons for representative equities and crypto, including splits, renamed/delisted instruments, timezones, missing bars and macro revisions as applicable.
4. A historical replay proving publication/vintage cutoffs, and repeatable reads of recorded artifacts without external calls.
5. Verified behavior for quota exhaustion, partial responses, malformed payloads, stale data, server restarts and ambiguous timeouts.
6. Tests showing agents cannot call broker mutation tools, modify settings, escape a generic dispatcher allowlist, or promote research through retrieved instructions.
7. Actual use by the full agent desk and unattended runner, with evidence visible in reports, measured latency/cost, and persistence across restart.

This work belongs in full-plan workstreams A (runtime/configuration), B (data), E (agents), G (research), H (operations), and I (acceptance). It does not reduce workstream C's quantitative correctness or F's execution requirements.

## Exclusions and remaining gaps

Do not add wallet/payment/x402, DeFi swapping, perpetual leverage, personal banking, or unrelated regional-market servers merely because they are listed. Those categories do not implement the specified long-only equity/crypto system. This is a scope decision, not a claim that each excluded server was audited.

MCP adoption leaves the substantial verified repository repairs intact: chronological portfolio accounting and backtests, point-in-time datasets, real walk-forward validation, funding/OI features, all strategies and optimizers, risk enforcement, durable execution, actual RD-Agent/LEAN workflows, and operational acceptance. No reviewed server establishes the complete historical dataset or guarantees investable alpha.

**Decision:** incorporate the recommended MCP interfaces into the full architecture; select paid data services against measured coverage and entitlement requirements; retain ownership of QuantAgentLab's data contracts, research validation, risk rules and trading ledger.
