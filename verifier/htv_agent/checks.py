from __future__ import annotations

import math
from typing import Any

from jsonschema import ValidationError, validate

from .json_utils import canonicalize_json
from .schemas import (
    CandidateLabel,
    CheckItem,
    DeterministicChecks,
    VerificationResult,
)
from .settings import Settings


def _extract_primary_answer(label: Any) -> Any:
    if isinstance(label, (str, int, float, bool)):
        return label
    if isinstance(label, dict):
        for key in ("answer", "label", "choice", "final_answer"):
            if key in label:
                return label[key]
    return None


def _normalize_choice(value: Any) -> str:
    return str(value).strip().lower()


def candidate_answer_key(candidate: CandidateLabel) -> str | None:
    if candidate.status != "answered":
        return None

    primary_answer = _extract_primary_answer(candidate.label)
    if primary_answer is not None:
        if isinstance(primary_answer, (str, int, float, bool)):
            return f"primary:{_normalize_choice(primary_answer)}"
        return f"primary_json:{canonicalize_json(primary_answer)}"

    if candidate.label is None:
        return None
    return f"label:{canonicalize_json(candidate.label)}"


def _safe_eval_assertion(expression: str, label: Any, metadata: dict[str, Any], choices: list[str] | None) -> bool:
    scope = {
        "__builtins__": {},
        "label": label,
        "metadata": metadata,
        "choices": choices,
        "math": math,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }
    return bool(eval(expression, scope, {}))


def deep_equal(left: Any, right: Any) -> bool:
    return canonicalize_json(left) == canonicalize_json(right)


def labels_consistent(left: Any, right: Any) -> bool:
    left_answer = _extract_primary_answer(left)
    right_answer = _extract_primary_answer(right)
    if left_answer is not None and right_answer is not None:
        return _normalize_choice(left_answer) == _normalize_choice(right_answer)
    return deep_equal(left, right)


def candidate_is_eligible(
    *,
    settings: Settings,
    label_schema: dict[str, Any],
    candidate: CandidateLabel,
    task_choices: list[str] | None,
    task_metadata: dict[str, Any],
    task_constraints: dict[str, Any],
) -> bool:
    if candidate.status != "answered":
        return False

    try:
        validate(instance=candidate.label, schema=label_schema)
    except ValidationError:
        return False

    if candidate.confidence < settings.min_solver_confidence:
        return False
    if settings.require_evidence and not candidate.evidence:
        return False
    if settings.require_tool_use and not candidate.tools_used:
        return False

    if task_choices:
        primary_answer = _extract_primary_answer(candidate.label)
        if primary_answer is None:
            return False
        if _normalize_choice(primary_answer) not in {_normalize_choice(choice) for choice in task_choices}:
            return False

    for assertion in task_constraints.get("python_assertions", []):
        try:
            if not _safe_eval_assertion(assertion, candidate.label, task_metadata, task_choices):
                return False
        except Exception:  # noqa: BLE001
            return False

    return True


def verifier_is_eligible(settings: Settings, verifier: VerificationResult) -> bool:
    return verifier.pass_verification and verifier.confidence >= settings.min_verifier_confidence


def run_deterministic_checks(
    *,
    settings: Settings,
    label_schema: dict[str, Any],
    candidate_primary: CandidateLabel,
    candidate_secondary: CandidateLabel,
    verifier: VerificationResult,
    task_choices: list[str] | None,
    task_metadata: dict[str, Any],
    task_constraints: dict[str, Any],
    primary_name: str = "solver_primary",
    secondary_name: str = "solver_secondary",
    verifier_name: str = "verifier",
) -> DeterministicChecks:
    items: list[CheckItem] = []

    def add(name: str, passed: bool, details: str) -> None:
        items.append(CheckItem(name=name, passed=passed, details=details))

    candidates = {
        primary_name: candidate_primary,
        secondary_name: candidate_secondary,
    }

    for name, candidate in candidates.items():
        answered = candidate.status == "answered"
        add(f"{name}_answered", answered, f"status={candidate.status}")
        if not answered:
            continue

        try:
            validate(instance=candidate.label, schema=label_schema)
            add(f"{name}_schema", True, "Label matches the target JSON schema.")
        except ValidationError as exc:
            add(f"{name}_schema", False, f"Schema validation failed: {exc.message}")

        add(
            f"{name}_confidence",
            candidate.confidence >= settings.min_solver_confidence,
            f"confidence={candidate.confidence:.2f}, threshold={settings.min_solver_confidence:.2f}",
        )

        add(
            f"{name}_evidence",
            (not settings.require_evidence) or bool(candidate.evidence),
            f"evidence_count={len(candidate.evidence)}",
        )

        add(
            f"{name}_tool_use",
            (not settings.require_tool_use) or bool(candidate.tools_used),
            f"tools_used={candidate.tools_used}",
        )

        if task_choices:
            primary_answer = _extract_primary_answer(candidate.label)
            in_choices = primary_answer is not None and _normalize_choice(primary_answer) in {
                _normalize_choice(choice) for choice in task_choices
            }
            add(
                f"{name}_choice_membership",
                in_choices,
                f"primary_answer={primary_answer!r}, choices={task_choices}",
            )

        for index, assertion in enumerate(task_constraints.get("python_assertions", []), start=1):
            try:
                assertion_passed = _safe_eval_assertion(
                    assertion,
                    candidate.label,
                    task_metadata,
                    task_choices,
                )
            except Exception as exc:  # noqa: BLE001
                add(
                    f"{name}_assertion_{index}",
                    False,
                    f"Assertion raised {type(exc).__name__}: {exc}",
                )
            else:
                add(
                    f"{name}_assertion_{index}",
                    assertion_passed,
                    f"assertion={assertion}",
                )

    consensus = (
        candidate_primary.status == "answered"
        and candidate_secondary.status == "answered"
        and labels_consistent(candidate_primary.label, candidate_secondary.label)
    )
    add("solver_consensus", (not settings.require_consensus) or consensus, f"consensus={consensus}")
    add(
        f"{verifier_name}_pass",
        verifier.pass_verification,
        f"verifier.pass_verification={verifier.pass_verification}",
    )
    add(
        f"{verifier_name}_confidence",
        verifier.confidence >= settings.min_verifier_confidence,
        f"confidence={verifier.confidence:.2f}, threshold={settings.min_verifier_confidence:.2f}",
    )

    return DeterministicChecks(
        passed=all(item.passed for item in items),
        items=items,
    )
