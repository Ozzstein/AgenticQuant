"""Synthetic protocol tests; entire prompts, not one filtered field, are checked."""

# ruff: noqa: F811
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from src.agents.graph import analyze_instrument
from src.agents.history import verify_model_projections
from src.agents.receipts import GenerationReceipt
from src.decision.service import DecisionService, OperatingLimits
from src.decision.store import DecisionStore
from src.decision.transport import JevHTTPTransport
from src.decision.workflows import PROJECTION_VERSION, exact_citation, semantic_projection
from src.domain.decisions import CitedClaim, DecisionSource, digest
from tests.test_agent_desk_invariants import (  # noqa: F401
    GenerativeFixture,
    decision_context,
    decision_instrument,
    decision_source,
    desk_config,
    desk_hmm,
    desk_portfolio,
    desk_sources,
)
from tests.test_agent_history import history_env  # noqa: F401
from tests.test_decision_contracts import decision_question, decision_request  # noqa: F401
from tests.test_evaluated_policy import evaluation_env, publish  # noqa: F401


def test_shadow_label_flip_changes_no_model_visible_prompt(
    tmp_path,
    desk_config,
    decision_context,
    decision_instrument,
    desk_sources,
    desk_hmm,
    desk_portfolio,
):
    transcripts = []
    for flip in (False, True):
        desk_config.agents.budget_store_path = str(
            tmp_path / ("flip-chat.sqlite" if flip else "original-chat.sqlite")
        )

        def handler(req):
            payload = json.loads(req.content)
            answers = {}
            for name, question in payload["questions"].items():
                labels = list(question["criteria"])
                selected = labels[-1] if flip else labels[0]
                answers[name] = {
                    "type": "choice",
                    "choice": selected,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == selected) for key in labels},
                }
            return httpx.Response(
                200,
                json={
                    "model": payload["model"],
                    "answers": answers,
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                },
            )

        service = DecisionService(
            DecisionStore(tmp_path / ("flipped.sqlite" if flip else "original.sqlite")),
            transport=JevHTTPTransport("test", transport=httpx.MockTransport(handler)),
            limits=OperatingLimits(min_interval_seconds=0, run_spend_usd=1, run_tokens=10_000_000),
        )
        calls = []
        result = analyze_instrument(
            decision_instrument,
            decision_context,
            sources=desk_sources,
            hmm_context=desk_hmm,
            portfolio_context=desk_portfolio,
            config=desk_config,
            decision_service=service,
            llm_factory=lambda role: GenerativeFixture(role, calls),
        )
        assert len(calls) == 17 and result.action.value == "BUY"
        transcripts.append(
            [(role, [(m.type, m.content) for m in messages]) for role, messages in calls]
        )
    assert sorted(transcripts[0]) == sorted(transcripts[1])
    for _, messages in transcripts[0]:
        payload = json.loads(messages[-1][1])
        assert payload["judgment_projection_version"] == PROJECTION_VERSION
        assert not payload["accepted_task_judgments"]
        for check in (payload["additional_context"] or {}).get("citation_checks", []):
            assert check["status"] == "quote_verified"


def test_deterministic_reference_and_quote_failures_stay_distinct(
    decision_context, decision_source
):
    claim = CitedClaim(
        claim="test",
        quote="nonexistent",
        evidence_id=decision_source.record.evidence_id,
        passage_hash=decision_source.passage_hash,
    )
    assert exact_citation(claim, (decision_source,), decision_context).status == "quote_missing"
    assert exact_citation(claim, (), decision_context).status == "invalid_reference"


def test_evaluated_projection_preserves_multiple_source_identities(evaluation_env):
    env = evaluation_env
    _, publication, request = publish(env)
    source = request.sources[0]
    text = "Independent second document with different facts."
    record = source.record.model_copy(
        update={"evidence_id": "second-document", "raw_hash": digest(text)}
    )
    env.evidence.put(
        record, text.encode(), kind="transcript", logical_key="second-document", document=text
    )
    other = DecisionSource.from_document(record, text, start=0, end=len(text), kind="transcript")
    projections = []
    for chosen in (source, other):
        actual_request = request.model_copy(update={"sources": (chosen,)})
        result = env.service.evaluate(actual_request, mode="refresh")
        value = semantic_projection(
            env.store,
            env.evidence,
            result,
            context=request.context,
            instrument=request.instrument,
            sources=(source, other),
            as_of=datetime.now(timezone.utc),
        )
        assert value and value[0]["sources"][0]["evidence_id"] == chosen.record.evidence_id
        assert value[0]["evaluation_id"] == publication.content_hash()
        assert value[0]["request_hash"] == actual_request.content_hash()
        assert value[0]["model"] == request.policy.model
        projections.append(value[0])
    assert projections[0]["observation_id"] != projections[1]["observation_id"]
    assert projections[0]["sources"] != projections[1]["sources"]


@pytest.mark.parametrize("tamper", ["result", "source", "clock", "policy_clock", "raw_attempt"])
def test_projection_rejects_unproved_or_future_semantics(evaluation_env, tamper):
    env = evaluation_env
    _, publication, request = publish(env)
    result = env.service.evaluate(request, mode="refresh")
    sources, context, cutoff = request.sources, request.context, datetime.now(timezone.utc)
    if tamper == "result":
        result = result.model_copy(
            update={
                "judgments": (result.judgments[0].model_copy(update={"disposition": "observed"}),)
            }
        )
    elif tamper == "source":
        sources = ()
    elif tamper == "clock":
        cutoff = result.started_at
    elif tamper == "policy_clock":
        context = context.model_copy(
            update={"as_of": publication.published_at - timedelta(microseconds=1)}
        )
    else:
        with env.store.connect() as db:
            db.execute("DROP TRIGGER attempt_immutable_update")
            db.execute(
                "UPDATE attempts SET body='{}' WHERE observation_id=?", (result.observation_id,)
            )
    with pytest.raises(ValueError):
        semantic_projection(
            env.store,
            env.evidence,
            result,
            context=context,
            instrument=request.instrument,
            sources=sources,
            as_of=cutoff,
        )


def test_legacy_and_forged_semantic_prompt_cannot_be_certified_for_memory(
    decision_context, decision_instrument, decision_source
):
    base = {"accepted_task_judgments": [], "additional_context": None}

    def receipt(payload):
        return SimpleNamespace(
            calls=(),
            decision=SimpleNamespace(
                judgment_observation_ids=(),
                context=decision_context,
                instrument=decision_instrument,
            ),
            sources=(decision_source,),
            generations=(
                GenerationReceipt(
                    call_id="test",
                    role="test",
                    request_json=json.dumps([{"content": json.dumps(payload)}]),
                    source_hashes=(),
                    report={
                        "role": "test",
                        "action": "ABSTAIN",
                        "status": "unavailable",
                        "reasoning": "fixture",
                    },
                    available_at=datetime.now(timezone.utc),
                ),
            ),
        )

    # Historical export remains an audit record; lack of isolation cannot become memory.
    verify_model_projections(receipt(base), decision_store=None, evidence_store=None)
    with pytest.raises(ValueError, match="legacy prompt"):
        verify_model_projections(
            receipt(base), decision_store=None, evidence_store=None, require_current=True
        )
    claim = CitedClaim(
        claim="Demand fell",
        quote=decision_source.text,
        evidence_id=decision_source.record.evidence_id,
        passage_hash=decision_source.passage_hash,
    )
    forged = {
        **base,
        "judgment_projection_version": PROJECTION_VERSION,
        "additional_context": {
            "citation_checks": [
                {
                    "claim": claim.model_dump(mode="json"),
                    "status": "supported",
                    "reason": "shadow result",
                }
            ]
        },
    }
    with pytest.raises(ValueError, match="unapproved semantic"):
        verify_model_projections(
            receipt(forged), decision_store=None, evidence_store=None, require_current=True
        )


def test_shadow_flip_preserves_full_prompts_with_frozen_certified_memory(
    history_env, desk_config, desk_hmm, desk_portfolio
):
    decision, history, authorities = history_env
    original = history.export(
        decision.content_hash(), as_of=datetime.now(timezone.utc), **authorities
    )
    cutoff = datetime.now(timezone.utc)
    desk_config.memory.enabled = True
    transcripts = []
    for flip in (False, True):

        def handler(req):
            payload = json.loads(req.content)
            answers = {}
            for name, q in payload["questions"].items():
                options = list(q["criteria"])
                selected = options[-1] if flip else options[0]
                answers[name] = {
                    "type": "choice",
                    "choice": selected,
                    "confidence": 1.0,
                    "probabilities": {k: float(k == selected) for k in options},
                }
            return httpx.Response(
                200,
                json={
                    "model": payload["model"],
                    "answers": answers,
                    "usage": {"input_tokens": 10, "output_tokens": 10},
                },
            )

        calls = []
        service = DecisionService(
            authorities["decision_store"],
            evidence_store=authorities["evidence_store"],
            transport=JevHTTPTransport("test", transport=httpx.MockTransport(handler)),
            limits=OperatingLimits(min_interval_seconds=0, run_spend_usd=1, run_tokens=10_000_000),
        )
        context = decision.context.model_copy(
            update={"as_of": cutoff, "run_id": "frozen-memory-flip-" + str(flip)}
        )
        result = analyze_instrument(
            decision.instrument,
            context,
            sources=original.sources,
            hmm_context=desk_hmm.model_copy(update={"as_of": cutoff}),
            portfolio_context=desk_portfolio.model_copy(update={"as_of": cutoff}),
            config=desk_config,
            decision_service=service,
            evidence_store=authorities["evidence_store"],
            observation_mode="refresh",
            llm_factory=lambda role: GenerativeFixture(role, calls),
        )
        assert result.action.value == "BUY" and len(calls) == 17
        for _, messages in calls:
            assert json.loads(messages[-1].content)["memory"]
        transcripts.append(
            sorted((role, [(m.type, m.content) for m in messages]) for role, messages in calls)
        )
    assert transcripts[0] == transcripts[1]
