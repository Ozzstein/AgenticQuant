# QuantAgentLab MCP integration decision

Date: 2026-10-27. Status: accepted architecture direction; implementation pending.

This decision selects the MCP integrations for the complete QuantAgentLab system. It governs provider selection in the [full system build plan](FULL_SYSTEM_BUILD_PLAN.md); the [finance MCP assessment](FINANCE_MCP_ASSESSMENT.md) contains the supporting source review and candidate comparisons.

## Selected integrations

Integrate **OpenBB, FMP, Alpaca, and Valyu** into the application through a shared evidence layer. Use **FRED/ALFRED** as a macro data source through OpenBB or a direct adapter that preserves revision history.

| Integration | Role | Implementation requirements |
|---|---|---|
| [OpenBB MCP](https://github.com/openbq-org/OpenBB/tree/develop/openbb_platform/extensions/mcp_server) | Shared financial-data access across selected providers for the agent desk. | Pin the server and provider extensions. Expose approved data capabilities only; block filesystem writes and unrestricted dispatch. Verify the exact release's provider coverage and entitlements. |
| [FMP MCP](https://github.com/imbenrabi/Financial-Modeling-Prep-MCP-Server) | Statements, analyst estimates, insider/institutional information, filings, earnings calendars and transcripts. | Self-host the community wrapper with selected toolsets. Retain direct ingestion for scheduled datasets. Store complete transcripts and retrieve bounded, cited passages for agents. |
| [Alpaca MCP](https://github.com/alpacahq/alpaca-mcp-server) | Account, position and order inspection, plus useful market information. | Enforce an exact read-operation allowlist outside the LLM. Account configuration changes, order submission/modification/cancellation and liquidation are excluded from agent access. The execution service retains order authority. |
| [Valyu MCP](https://mcp.valyu.ai/) | Cited source discovery and research for company investigations, sentiment and research hypotheses. | Use the current hosted service. Verify source access, retain citations and publication metadata, and enforce search/research spending limits. Research findings enter the existing experiment and validation process. |
| [FRED/ALFRED data](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) | Economic history and macro evidence for regime analysis. | Use OpenBB or our direct adapter with preserved vintages, raw responses and availability cutoffs. Do not adopt the listed standalone FRED wrapper unchanged as the historical authority. |

OpenBB and FMP should expose complementary capabilities. Where an OpenBB provider satisfies the same FMP capability and evidence contract, expose it once rather than operating duplicate agent tools.

## Conditional and unselected integrations

- **DexPaprika:** add when DEX liquidity, pool or token context has a defined use in the crypto universe. Exchange funding, open interest and execution liquidity still require their own data paths.
- **QuantConnect MCP:** add if the paid QuantConnect cloud research environment is chosen. The required operational LEAN integration remains part of the full build regardless of this choice.
- **Massive, Financial Datasets and Alpha Vantage:** retain as data-provider alternatives or cross-check candidates. They are not additional selected MCP dependencies.
- **CCXT, Finnhub and Yahoo MCP wrappers:** do not add redundant wrappers. Preserve and strengthen the direct CCXT, Finnhub and yfinance integrations required by the brief.

## Application integration contract

The shared layer belongs inside QuantAgentLab and serves its LangGraph agents, CLI, scheduler and dashboard. Connecting these servers to a development assistant does not implement the application integration.

Direct API and MCP adapters must return the same typed evidence contract: provider/tool/version, canonical instrument identity, event and publication/availability times, ingestion time, requested as-of cutoff, units/currency, adjustment policy, source reference, raw-payload hash and quality status. Preserve unknown fields as unknown. Historical runs use immutable snapshots and reject evidence unavailable at the decision time.

The application owns credentials, connections, timeouts, retries, caching, quotas and tool permissions. Store full documents and bulk datasets outside the LLM context; retrieve bounded evidence with references. Fetch shared facts once per run and reuse them across analysts and debate rounds.

Execution follows the existing contract:

`portfolio target → deterministic risk → durable order intent → execution adapter → reconciliation`

Agents receive evidence and broker observations. The execution owner submits orders and reconciles the durable trading ledger. Retrieved content cannot modify operational instructions, tool permissions or research promotion rules.

## Dependencies and completion criteria

Provider subscriptions and credentials are external dependencies. As documented during the assessment, FMP transcript access requires at least the Ultimate plan; verify the entitlement provisioned for deployment. Valyu source access and research charges must also fit the operating budget. [FMP transcript documentation](https://site.financialmodelingprep.com/developer/docs/stable/search-transcripts), [Valyu service documentation](https://mcp.valyu.ai/).

An integration is complete when it has a pinned deployment manifest, verified actual tool schemas, real provider calls for required coverage, preserved provenance, repeatable historical replay where applicable, tested failure behavior, enforced permissions, and observable use in the unattended application. The [assessment's acceptance requirements](FINANCE_MCP_ASSESSMENT.md#required-acceptance-evidence) apply to all selected integrations.

This decision preserves the full brief: both asset classes, every specified model, strategy and optimizer, the complete agent desk, RD-Agent and crypto research workflows, all execution paths, dashboards, operating cycle and release gates. Optional provider choices do not make any required system capability optional.
