"""Current canonical execution authorities for immutable research evaluations.

Locators come from the evaluator's operator-selected stores, never candidate
payloads. A frozen audit is an identity receipt, not current permission.
"""

from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field, model_validator

from src.domain.common import NonEmptyString, Sha256, UtcTimestamp, VersionedContract
from src.research.registry import RegistryError


class ExecutionLedgerLocator(VersionedContract):
    role: NonEmptyString
    account_id: NonEmptyString
    path: NonEmptyString
    device: int = Field(ge=0, strict=True)
    inode: int = Field(gt=0, strict=True)
    account_spec_hash: Sha256
    execution_binding_hash: Sha256
    original_audit_hash: Sha256
    source_hashes: tuple[Sha256, ...]


class EvaluationExecutionBindings(VersionedContract):
    evaluation_hash: Sha256
    trial_id: NonEmptyString
    recorded_at: UtcTimestamp
    ledgers: tuple[ExecutionLedgerLocator, ...]

    @model_validator(mode="after")
    def unique_roles(self):
        roles = tuple(v.role for v in self.ledgers)
        if not roles or roles != tuple(sorted(set(roles))):
            raise ValueError("sorted unique execution authority roles required")
        return self


def requirements(registry, evaluation):
    """Derive the entire original account scope from archived typed receipts."""
    from src.core.backtester import ReplayResult
    from src.domain.portfolio import AccountSpec

    result = ReplayResult.model_validate_json(registry.blob(evaluation.replay_hash))
    manifest = json.loads(registry.blob(evaluation.manifest_hash))
    if result.manifest != manifest:
        raise RegistryError("evaluation replay/manifest authority differs")
    account = AccountSpec.model_validate(manifest["account"])
    if result.account_id != account.account_id:
        raise RegistryError("evaluation replay account differs")
    needed = {
        "replay": (
            account.account_id,
            account.content_hash(),
            (evaluation.replay_hash, evaluation.manifest_hash),
        )
    }
    if evaluation.actual_decision_hashes:
        from src.agents.history import DeskHistoryReceipt
        from src.domain.decisions import AgentDecision
        from src.research.evaluator import load_experiment_plan

        plan = load_experiment_plan(registry, evaluation.trial_id)
        references = {r.decision_hash: r for r in plan.agent_decisions}
        if set(evaluation.actual_decision_hashes) != set(references):
            raise RegistryError("evaluation original decision scope differs")
        for key, reference in sorted(references.items()):
            receipt = DeskHistoryReceipt.model_validate_json(registry.blob(reference.receipt_hash))
            if (
                receipt.decision.content_hash() != key
                or receipt.decision != AgentDecision.model_validate_json(registry.blob(key))
            ):
                raise RegistryError("evaluation original decision receipt differs")
            account_id = receipt.decision.context.account_id
            if not account_id:
                raise RegistryError("original decision lacks an account authority")
            needed["advice:" + key] = (account_id, None, (key, reference.receipt_hash))
    if evaluation.paired_evaluation_hash:
        proof = json.loads(registry.blob(evaluation.paired_evaluation_hash))
        paired = ReplayResult.model_validate_json(registry.blob(proof["baseline_result_hash"]))
        spec = AccountSpec.model_validate(paired.manifest["account"])
        if (
            proof["trial_id"] != evaluation.trial_id
            or paired.account_id != spec.account_id
            or proof["candidate_result_hash"] != evaluation.replay_hash
        ):
            raise RegistryError("paired evaluation account authority differs")
        needed["paired"] = (
            spec.account_id,
            spec.content_hash(),
            (evaluation.paired_evaluation_hash, proof["baseline_result_hash"]),
        )
    return needed


def _file(path):
    from src.research.native_mounts import canonical_path

    path = canonical_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RegistryError("execution authority must be an ordinary unaliased file")
    return path, info


def freeze(registry, evaluation, *, ledger_path, source_ledgers=None, paired_ledger_path=None):
    from src.execution.ledger import PortfolioLedger
    from src.research.native_mounts import NativeMountAuthority

    needed = requirements(registry, evaluation)
    supplied = source_ledgers or {}
    ledgers = []
    for role, (account_id, spec_hash, sources) in sorted(needed.items()):
        selected = (
            ledger_path
            if role == "replay"
            else paired_ledger_path
            if role == "paired"
            else supplied.get(account_id)
        )
        if selected is None:
            raise RegistryError("original execution authority is missing")
        path, before = _file(selected)
        NativeMountAuthority().protect(path, Path(str(path) + "-wal"), Path(str(path) + "-shm"))
        ledger = PortfolioLedger(path, read_only=True)
        try:
            audit = ledger.execution_compliance(account_id)
            if spec_hash is not None and audit.account_spec_hash != spec_hash:
                raise RegistryError("original execution account specification differs")
            _, after = _file(path)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise RegistryError("execution authority changed during binding")
            ledgers.append(
                ExecutionLedgerLocator(
                    role=role,
                    account_id=account_id,
                    path=str(path),
                    device=before.st_dev,
                    inode=before.st_ino,
                    account_spec_hash=audit.account_spec_hash,
                    execution_binding_hash=audit.execution_binding_hash,
                    original_audit_hash=registry.archive_contract(audit),
                    source_hashes=sources,
                )
            )
        finally:
            ledger.close()
    return EvaluationExecutionBindings(
        evaluation_hash=evaluation.content_hash(),
        trial_id=evaluation.trial_id,
        recorded_at=datetime.now(timezone.utc),
        ledgers=tuple(ledgers),
    )


def require_current(registry, evaluation, binding):
    from src.execution.ledger import PortfolioLedger

    if (
        binding.evaluation_hash != evaluation.content_hash()
        or binding.trial_id != evaluation.trial_id
        or binding.recorded_at < evaluation.completed_at
    ):
        raise RegistryError("current execution binding belongs to another evaluation")
    needed = requirements(registry, evaluation)
    if set(needed) != {v.role for v in binding.ledgers}:
        raise RegistryError("current execution authority scope differs")
    for item in binding.ledgers:
        account_id, spec_hash, sources = needed[item.role]
        if item.account_id != account_id or item.source_hashes != sources:
            raise RegistryError("current execution source bindings differ")
        from src.execution.discrepancies import ExecutionComplianceAudit

        original = ExecutionComplianceAudit.model_validate_json(
            registry.blob(item.original_audit_hash)
        )
        if (
            original.account_id != account_id
            or original.account_spec_hash != item.account_spec_hash
            or original.execution_binding_hash != item.execution_binding_hash
            or spec_hash is not None
            and spec_hash != item.account_spec_hash
        ):
            raise RegistryError("original execution identity receipt differs")
        path, before = _file(item.path)
        if str(path) != item.path or (before.st_dev, before.st_ino) != (item.device, item.inode):
            raise RegistryError("current execution authority file identity changed")
        ledger = PortfolioLedger(path, read_only=True)
        try:
            audit = ledger.execution_compliance(account_id)
            _, after = _file(path)
            if (
                (after.st_dev, after.st_ino) != (item.device, item.inode)
                or audit.account_spec_hash != item.account_spec_hash
                or audit.execution_binding_hash != item.execution_binding_hash
            ):
                raise RegistryError("current execution ownership authority differs")
            if audit.status != "PASS":
                raise RegistryError("current execution compliance failed: " + audit.content_hash())
        finally:
            ledger.close()
