from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskSample(BaseModel):
    sample_id: str
    question: str
    images: list[str] = Field(default_factory=list)
    context: str | None = None
    choices: list[str] | None = None
    task_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    kind: Literal["text", "region", "numeric", "artifact", "tool_output"] = "tool_output"
    source: str
    content: str
    bbox: list[int] | None = None
    artifact_path: str | None = None


class CandidateLabel(BaseModel):
    status: Literal["answered", "abstain"]
    label: Any | None = None
    answer_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    concise_reasoning: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    abstain_reason: str | None = None


class VerificationResult(BaseModel):
    pass_verification: bool
    confidence: float = Field(ge=0.0, le=1.0)
    supported_claims: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    summary: str


class CheckItem(BaseModel):
    name: str
    passed: bool
    details: str


class DeterministicChecks(BaseModel):
    passed: bool
    items: list[CheckItem] = Field(default_factory=list)


class ToolResult(BaseModel):
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentEnvelope(BaseModel):
    thought: str
    action: Literal["tool", "final"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final: dict[str, Any] | None = None


class AgentRun(BaseModel):
    role: Literal["solver", "verifier"]
    variant: str
    output: dict[str, Any]
    tools_used: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    attached_images: list[str] = Field(default_factory=list)


class LabelingResult(BaseModel):
    sample_id: str
    status: Literal["accepted", "abstain", "needs_review"]
    accepted_label: Any | None = None
    solver_primary: CandidateLabel
    solver_secondary: CandidateLabel
    solver_tiebreaker: CandidateLabel | None = None
    verifier: VerificationResult
    candidate_verifiers: dict[str, VerificationResult] = Field(default_factory=dict)
    deterministic_checks: DeterministicChecks
    consensus_pass: bool
    selected_solver_variant: str | None = None
    trace_path: str
    failure_reasons: list[str] = Field(default_factory=list)
