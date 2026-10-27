# QuantAgentLab Jev integration decision

Date: 2026-10-27. Status: accepted architecture and evaluation direction; implementation pending. Replacements and trading influence require the acceptance evidence below.

This decision adds TypeSafe's Jev to the [full system build plan](FULL_SYSTEM_BUILD_PLAN.md) as a shared service for narrow, typed judgments. It records the agreed integration map and replacement boundaries. It does not reduce the brief's required capabilities or change the [selected MCP integrations](MCP_INTEGRATION_DECISION.md).

## Decision and scope

Implement one application-owned `DecisionService` with a Jev adapter, separate from the generative chat-model factory in [llm.py](../src/agents/llm.py). Use the official TypeSafe API/SDK for bounded classification and scoring. Jev is a model dependency, not a financial-data source, an MCP replacement, or the owner of the trading pipeline.

The strongest intended applications are news/transcript classification, evidence verification, and text-derived research features. Integrate retrieval and debate support into the same service. Qualitative macro labeling remains conditional on demonstrating incremental value. Replacing an existing subtask requires measured accuracy and operating benefits; a working adapter alone does not justify activating it in trading.

All five specialist roles, their reports and reasoning, the required debate process, both asset classes, all strategies/models/optimizers, research workflows, and execution paths remain in scope. Jev changes how selected subtasks are performed. It does not justify delivering fewer system capabilities.

## Integration and replacement map

The locations below are current modules or explicitly proposed integration points. None currently implements Jev.

| ID / workstream | Application and location | Jev responsibility | Replacement boundary |
|---|---|---|---|
| J1 / B, E | News sentiment: [news_data.py](../src/data/news_data.py), [sentiment.py](../src/agents/sentiment.py) | Classify each item's tone, company relevance, and event type; preserve unknown/insufficient evidence. | Replace `_sentiment_hint`'s keyword counter after evaluation; move selected LLM classification work into the service. Keep the sentiment analyst's synthesis, reasoning, and investment context. |
| J2 / B, E | Transcript analysis: [transcript_nlp.py](../src/agents/transcript_nlp.py), [earnings_handler.py](../src/agents/earnings_handler.py) | Classify management tone, stated guidance direction, hedging, and whether a response addresses the question. | Replace fixed-label subtasks inside the existing LLM call. Keep narrative explanations and exact quote extraction separate; code owns financial/date comparisons. The transcript node still needs graph integration. |
| J3 / E | New evidence-verification node around analyst/debate outputs in [graph.py](../src/agents/graph.py) | Compare an analyst claim with a source passage: supported, contradicted, or insufficient. | Adds a missing capability; can replace routine LLM verification calls where introduced. Code first verifies source identity and quote existence. Ambiguous evidence returns for analysis; a Jev result cannot clear a hard veto. |
| J4 / B, E | Shared evidence retrieval and [memory.py](../src/agents/memory.py) | Score retrieved passages for relevance and conflict before analysts consume them. | Extend the current recency-based context selection and replace suitable LLM reranking subtasks. Keep storage, search, publication cutoffs, and chronology in code. Preserve material contradictory evidence. |
| J5 / C, G | Proposed text-feature adapter beside [factor_engine.py](../src/core/factor_engine.py), connected to [factor_proposer.py](../src/core/factor_proposer.py) and [rd_agent_runner.py](../src/core/rd_agent_runner.py) | Convert source passages into versioned numeric features, such as demand weakness or management uncertainty. | Adds a research feature family and can replace manual/LLM text labeling. It does not replace numeric factors, model training, Qlib, RD-Agent experiments, or portfolio validation. |
| J6 / E | Debate and qualitative risk support: [debate.py](../src/agents/debate.py), [risk_manager.py](../src/agents/risk_manager.py), graph routing | Flag conflicting claims, unsupported assertions, and missing context for the required challenge rounds and risk review. | Augment or replace the confidence-spread-only disagreement heuristic after evaluation. Preserve configured debate rounds, full risk reasoning, binding veto semantics, and deterministic portfolio controls. |
| J7 / D, E | Conditional qualitative macro labeling: [agents/macro_regime.py](../src/agents/macro_regime.py) | Classify macro narratives as auxiliary evidence or a disagreement flag. | May replace the agent's quick LLM classification call. The HMM in [core/macro_regime.py](../src/core/macro_regime.py) remains the planned authoritative regime model; there is no silent override. |

## Shared decision service contract

The service belongs inside the application and is shared by agents, research, CLI, scheduler, and dashboard. A separate deployed microservice is not required. The adapter must not masquerade as a text-generating `BaseChatModel` or fabricate narrative reasoning to satisfy an existing report schema.

1. **Typed questions and answers.** Define versioned `DecisionQuestion`, `DecisionRequest`, and `DecisionResult` schemas. Support Choice, Score, and Noul with bounded answer spaces and explicit meanings. Store raw probabilities, the selected answer or score, and provider confidence when supplied. Keep policy decisions separate from model answers.
2. **Source and time contract.** Carry question/rubric version, instrument, source IDs and hashes, publication/availability timestamps, requested as-of time, state hash, run identity, and actual response/model version. Retain source passages and their references outside the bounded request context. Use full stored transcripts rather than the current truncated tool output as the source of evidence.
3. **Model and policy versions.** Pin the evaluated model version, SDK, question definitions, and threshold policy. Record the actual returned model ID. Reject a version mismatch for an activated policy; re-evaluate before switching versions. A moving alias must not silently change production judgments.
4. **Cache and replay.** Key cached results by canonical state/source hashes, exact question/rubric definitions, and model version. Preserve raw responses and request timing. Recorded replay does not call the provider again; new provider calls create new observations rather than rewriting existing evidence. Prevent reuse across incompatible as-of cutoffs or source revisions.
5. **Bounded requests.** Send compact relevant evidence, ask one judgment per question, and batch independent questions over the same state. Dependent judgments require a subsequent step. Arithmetic, date ordering, freshness, and exact financial thresholds remain in code.
6. **Operating controls.** Own private credentials, request/token/spend budgets, timeouts, bounded retries, rate limits, and cancellation. Retry transient failures only within the caller's deadline; malformed questions require correction. Record usage, pricing version, latency, cache hits, failures, and unavailable cost estimates.
7. **Failure semantics.** Represent unavailable, timed-out, malformed, stale, and insufficient-evidence outcomes explicitly. None means neutral, approved, or liquidate. Block affected new exposure when required evidence is unavailable; continue deterministic protection, monitoring, and reconciliation. Any approved alternate provider/policy must be separately identified, evaluated, and logged.
8. **Authority.** The service returns judgments only. It has no broker credentials, order-submission capability, risk-limit mutation, or artifact-promotion authority. Treat input documents as untrusted data. A model-based injection detector may supplement filtering but cannot become the security boundary.

These are proposed contracts to implement, not existing APIs or modules.

## Confidence schema migration

Current `AgentReport` and `AnalysisResult` fields use 0–100 confidence, while `RegimeDetectionResult` uses 0–1. Their meanings also differ: an analyst may describe evidence quality or conviction, while the HMM describes a state probability. See [schemas.py](../src/utils/schemas.py) and [agent state](../src/agents/state.py).

Jev's Choice/Score `confidence` summarizes the answer distribution; Noul returns a yes-probability without a separate confidence field. Neither is automatically a probability of profitable trading. Simply multiplying Jev confidence by 100 would preserve a numeric range while losing its meaning. [TypeSafe confidence definitions](https://docs.typesafe.ai/confidence).

Before integrating Jev into routing or allocation:

- Separate answer probabilities, provider confidence and its definition, evidence completeness, analyst conviction, and calibrated task-performance estimates. Include provider/model/question/version provenance; use nullable values when a concept does not apply.
- Version persisted schemas and explicitly migrate legacy records as legacy/unvalidated confidence. Preserve original values; do not manufacture calibrated probabilities for old reports.
- Rework confidence-spread routing and risk aggregation so unlike scores are not averaged or compared as interchangeable evidence. UI percentages must name the quantity being displayed.
- Select thresholds per labeled task and error cost. Calibrate on eligible training/validation observations and evaluate on held-out data. Do not convert confidence directly into position size or claim a predictive edge from calibration alone.

## Retained system responsibilities

Keep market-data APIs/MCP adapters and immutable evidence storage; LangGraph orchestration; full analyst synthesis, explanations, and reports; quantitative factors and Qlib; LightGBM/CatBoost/XGBoost/ridge and ensembles; HMM regimes; portfolio optimizers; chronological replay and walk-forward validation; deterministic risk controls; durable accounting; and paper/Alpaca/CCXT/LEAN execution.

The research loop can propose semantic questions, use Jev to compute text features, and train existing models on those features. The common quarantine, experiment, validation, and promotion process still decides eligibility. Jev does not generate strategy code, replace research memos, or authorize its own outputs for trading.

## Acceptance and activation evidence

1. **Task accuracy:** build labeled financial-text evaluations for J1–J4/J6 and any proposed J7 use. Compare existing heuristics/LLM calls and simple alternatives on identical examples. Measure per-class precision/recall, abstention, missed material evidence, and false blocks; freeze acceptance thresholds before evaluating the held-out set.
2. **Confidence and robustness:** evaluate calibration of task probabilities with reliability curves and proper scoring rules where applicable. Test reordered options, missing/conflicting evidence, misleading wording, prompt injection, and numeric/date boundaries. A valid output schema is not evidence of semantic correctness.
3. **Integration:** exercise actual provider calls, schema validation, version mismatch, caching/replay, deadline expiry, retries, rate limits, and outage behavior. Show observable use in real agent and research runs; mocks alone cannot establish completion.
4. **Evidence quality:** every judgment traces to immutable sources, actual availability times, and exact questions/model versions. Citation checking distinguishes whether a source supports a claim from whether the underlying source is true.
5. **Research value:** evaluate J5 features with the rebuilt six-month walk-forward, costs, baselines, trial registry, and holdout controls in workstream C. Fit question selection, thresholds, transformations, and feature selection inside eligible development windows. Generated text features cannot bypass the common promotion gate.
6. **Prospective attribution:** compare the complete system with and without Jev under matched data, costs, and risk policies. Keep the existing quant-only, quant-plus-veto, and full-agent controls. Historical source cutoffs do not remove knowledge in a pretrained model; recorded replay proves reproducibility, not historical out-of-sample knowledge.
7. **Operating benefit:** measure actual end-to-end latency, reliability, token/spend use, and generative-model calls saved. Retain a replacement only when its quality and operating tradeoffs satisfy the declared criteria; do not promise a fixed cost reduction from vendor pricing.
8. **Safety and rollback:** verify that no judgment can clear a binding risk veto, place an order, promote an artifact, or turn a provider outage into liquidation. Roll back to an evaluated policy/configuration without discarding audit records or required functionality.

Implementing and evaluating the agreed integrations is part of the build. Activating a replacement or a new trading feature remains conditional on its evidence. The optional macro extension does not replace any mandatory HMM work. The full brief's three-month paper observation and other release gates remain mandatory.

## Research basis and limits

- The [Kiyoro article](https://x.com/0xKiyoro/status/2107135100911563017) supplies the useful division between computed state, narrow judgments, policy, and deterministic execution. Its leveraged grids, wallet copying, one-position restriction, and automatic flattening on model outage are not adopted into our long-only portfolio specification.
- The referenced [jev-trader README](https://github.com/jarrodwatts/jev-trader/blob/main/README.md) identifies its public deployment as a mock-model dry run and its measured loop benchmark as using simulated inference. This supports an integration example, not a verified Jev latency guarantee or trading edge.
- TypeSafe documents [citation checking](https://docs.typesafe.ai/cookbooks/citation_check), [retrieved-passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages), and [text-feature discovery for CatBoost](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery). The last example uses wine reviews; financial value remains to be tested.
- The reviewed [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) include arithmetic/date errors, sensitivity to irrelevant/adversarial context, and option order. Treat these as regression cases for the selected version, not proof that every issue is unchanged in later releases. Consult the [model documentation](https://docs.typesafe.ai/models) and [official API/SDK quick start](https://docs.typesafe.ai/introduction/quickstart) when implementing.

This decision is based on public documentation and repository inspection. No authenticated Jev benchmark, financial accuracy evaluation, or provider integration has yet been performed for QuantAgentLab. Account access, data-use permissions, credentials, and measured operating cost remain deployment prerequisites.
