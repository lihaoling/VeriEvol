from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .agent import LLMToolAgent
from .checks import (
    candidate_answer_key,
    candidate_is_eligible,
    labels_consistent,
    run_deterministic_checks,
    verifier_is_eligible,
)
from .io_utils import write_trace
from .llm import create_model_client
from .schemas import (
    AgentRun,
    CandidateLabel,
    DeterministicChecks,
    LabelingResult,
    TaskSample,
    VerificationResult,
)
from .settings import Settings
from .tools import ToolRuntime, build_default_registry


class GraphState(TypedDict, total=False):
    task: dict[str, Any]
    label_schema: dict[str, Any]
    run_dir: str
    solver_primary: dict[str, Any]
    solver_secondary: dict[str, Any]
    solver_tiebreaker: dict[str, Any]
    candidate_verifiers: dict[str, dict[str, Any]]
    deterministic_checks: dict[str, Any]
    accepted: bool
    consensus_pass: bool
    selected_solver_variant: str
    selected_support_variants: list[str]
    selected_verifier_variant: str
    final_result: dict[str, Any]


def _forced_candidate(reason: str) -> CandidateLabel:
    return CandidateLabel(
        status="abstain",
        confidence=0.0,
        concise_reasoning=reason,
        abstain_reason=reason,
    )


def _forced_verifier(reason: str) -> VerificationResult:
    return VerificationResult(
        pass_verification=False,
        confidence=0.0,
        supported_claims=[],
        issues=[reason],
        missing_evidence=[reason],
        summary=reason,
    )


class HTVAgentPipeline:
    VARIANT_ORDER = {
        "solver_primary": 0,
        "solver_secondary": 1,
        "solver_tiebreaker": 2,
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = create_model_client(settings)
        self.registry = build_default_registry(settings)

    def _variant_sort_key(self, variant: str) -> int:
        return self.VARIANT_ORDER.get(variant, len(self.VARIANT_ORDER) + 1)

    def _collect_solver_runs(self, state: GraphState) -> dict[str, AgentRun]:
        runs: dict[str, AgentRun] = {}
        for key in ("solver_primary", "solver_secondary", "solver_tiebreaker"):
            raw_run = state.get(key)
            if raw_run:
                runs[key] = AgentRun.model_validate(raw_run)
        return runs

    def _candidate_rank(self, variant: str, candidate: CandidateLabel) -> tuple[float, int, int, int]:
        return (
            candidate.confidence,
            len(candidate.evidence),
            len(candidate.tools_used),
            -self._variant_sort_key(variant),
        )

    def _build_candidate_clusters(
        self,
        *,
        solver_runs: dict[str, AgentRun],
        task: TaskSample,
        label_schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        clusters: dict[str, dict[str, Any]] = {}
        for variant in sorted(solver_runs, key=self._variant_sort_key):
            candidate = CandidateLabel.model_validate(solver_runs[variant].output)
            if not candidate_is_eligible(
                settings=self.settings,
                label_schema=label_schema,
                candidate=candidate,
                task_choices=task.choices,
                task_metadata=task.metadata,
                task_constraints=task.constraints,
            ):
                continue

            answer_key = candidate_answer_key(candidate)
            if answer_key is None:
                continue

            cluster = clusters.get(answer_key)
            if cluster is None:
                clusters[answer_key] = {
                    "answer_key": answer_key,
                    "support_variants": [variant],
                    "representative_variant": variant,
                    "representative_candidate": candidate,
                }
                continue

            cluster["support_variants"].append(variant)
            representative_variant = str(cluster["representative_variant"])
            representative_candidate = CandidateLabel.model_validate(cluster["representative_candidate"])
            if self._candidate_rank(variant, candidate) > self._candidate_rank(
                representative_variant,
                representative_candidate,
            ):
                cluster["representative_variant"] = variant
                cluster["representative_candidate"] = candidate

        ranked_clusters = list(clusters.values())
        ranked_clusters.sort(
            key=lambda cluster: (
                -len(cluster["support_variants"]),
                -CandidateLabel.model_validate(cluster["representative_candidate"]).confidence,
                -len(CandidateLabel.model_validate(cluster["representative_candidate"]).evidence),
                self._variant_sort_key(str(cluster["representative_variant"])),
            )
        )
        return ranked_clusters

    def _pick_selected_cluster(
        self,
        *,
        ranked_clusters: list[dict[str, Any]],
        candidate_verifier_runs: dict[str, AgentRun],
    ) -> dict[str, Any] | None:
        verified_clusters: list[dict[str, Any]] = []
        for cluster in ranked_clusters:
            representative_variant = str(cluster["representative_variant"])
            verifier_run = candidate_verifier_runs.get(representative_variant)
            if verifier_run is None:
                continue
            verifier_output = VerificationResult.model_validate(verifier_run.output)
            if verifier_is_eligible(self.settings, verifier_output):
                verified_clusters.append(cluster)

        if not verified_clusters:
            return None

        if not self.settings.require_consensus:
            return verified_clusters[0]

        for cluster in verified_clusters:
            if len(cluster["support_variants"]) >= 2:
                return cluster
        return None

    def _pick_selected_verifier_variant(
        self,
        *,
        selected_cluster: dict[str, Any] | None,
        ranked_clusters: list[dict[str, Any]],
        candidate_verifier_runs: dict[str, AgentRun],
    ) -> str | None:
        if selected_cluster is not None:
            variant = str(selected_cluster["representative_variant"])
            if variant in candidate_verifier_runs:
                return variant

        for cluster in ranked_clusters:
            variant = str(cluster["representative_variant"])
            if variant in candidate_verifier_runs:
                return variant
        return None

    def _pick_check_variants(
        self,
        *,
        selected_cluster: dict[str, Any] | None,
        ranked_clusters: list[dict[str, Any]],
        solver_runs: dict[str, AgentRun],
    ) -> tuple[str | None, str | None]:
        if selected_cluster is not None and len(selected_cluster["support_variants"]) >= 2:
            support_variants = sorted(selected_cluster["support_variants"], key=self._variant_sort_key)
            return support_variants[0], support_variants[1]

        preferred_cluster = selected_cluster or (ranked_clusters[0] if ranked_clusters else None)
        if preferred_cluster is not None:
            primary_variant = str(preferred_cluster["representative_variant"])
            for cluster in ranked_clusters:
                for variant in sorted(cluster["support_variants"], key=self._variant_sort_key):
                    if variant != primary_variant:
                        return primary_variant, variant
            for variant in sorted(solver_runs, key=self._variant_sort_key):
                if variant != primary_variant:
                    return primary_variant, variant
            return primary_variant, None

        ordered_variants = sorted(solver_runs, key=self._variant_sort_key)
        if not ordered_variants:
            return None, None
        if len(ordered_variants) == 1:
            return ordered_variants[0], None
        return ordered_variants[0], ordered_variants[1]

    def _build_graph(self, runtime: ToolRuntime) -> Any:
        agent = LLMToolAgent(
            settings=self.settings,
            client=self.client,
            registry=self.registry,
            runtime=runtime,
        )
        graph = StateGraph(GraphState)

        def solve_primary(state: GraphState) -> GraphState:
            task = TaskSample.model_validate(state["task"])
            run = agent.run_solver(
                sample_id=task.sample_id,
                question=task.question,
                image_paths=task.images,
                context=task.context,
                choices=task.choices,
                label_schema=state["label_schema"],
                variant="solver_primary",
                temperature=self.settings.solver_primary_temperature,
                initial_note="Primary solve. Be precise and conservative.",
            )
            return {"solver_primary": run.model_dump(mode="json")}

        def solve_secondary(state: GraphState) -> GraphState:
            task = TaskSample.model_validate(state["task"])
            if self.settings.benchmark_mode:
                run = AgentRun(
                    role="solver",
                    variant="solver_secondary",
                    output=_forced_candidate("benchmark_mode_secondary_disabled").model_dump(mode="json"),
                    tools_used=[],
                    steps=[],
                    attached_images=task.images,
                )
            else:
                run = agent.run_solver(
                    sample_id=task.sample_id,
                    question=task.question,
                    image_paths=task.images,
                    context=task.context,
                    choices=task.choices,
                    label_schema=state["label_schema"],
                    variant="solver_secondary",
                    temperature=self.settings.solver_secondary_temperature,
                    initial_note=(
                        "Secondary solve. Solve independently from scratch. "
                        "Prefer a different inspection order before answering."
                    ),
                )
            return {"solver_secondary": run.model_dump(mode="json")}

        def solve_tiebreaker(state: GraphState) -> GraphState:
            task = TaskSample.model_validate(state["task"])
            if self.settings.benchmark_mode:
                run = AgentRun(
                    role="solver",
                    variant="solver_tiebreaker",
                    output=_forced_candidate("benchmark_mode_tiebreaker_disabled").model_dump(mode="json"),
                    tools_used=[],
                    steps=[],
                    attached_images=task.images,
                )
            else:
                run = agent.run_solver(
                    sample_id=task.sample_id,
                    question=task.question,
                    image_paths=task.images,
                    context=task.context,
                    choices=task.choices,
                    label_schema=state["label_schema"],
                    variant="solver_tiebreaker",
                    temperature=min(self.settings.solver_primary_temperature, 0.15),
                    initial_note=(
                        "Audit solve. Focus on the exact evidence-bearing regions, rerun targeted tools, "
                        "and only answer if you can justify the answer more strongly than a default read."
                    ),
                )
            return {"solver_tiebreaker": run.model_dump(mode="json")}

        def verify_candidates(state: GraphState) -> GraphState:
            task = TaskSample.model_validate(state["task"])
            solver_runs = self._collect_solver_runs(state)
            ranked_clusters = self._build_candidate_clusters(
                solver_runs=solver_runs,
                task=task,
                label_schema=state["label_schema"],
            )
            verifier_runs: dict[str, dict[str, Any]] = {}
            clusters_to_verify = ranked_clusters[:1] if self.settings.benchmark_mode else ranked_clusters
            for cluster in clusters_to_verify:
                representative_variant = str(cluster["representative_variant"])
                representative_candidate = CandidateLabel.model_validate(cluster["representative_candidate"])
                run = agent.run_verifier(
                    question=task.question,
                    image_paths=task.images,
                    context=task.context,
                    choices=task.choices,
                    label_schema=state["label_schema"],
                    candidate_label=representative_candidate.model_dump(mode="json"),
                    variant=f"verifier_for_{representative_variant}",
                )
                verifier_runs[representative_variant] = run.model_dump(mode="json")
            return {"candidate_verifiers": verifier_runs}

        def run_checks(state: GraphState) -> GraphState:
            task = TaskSample.model_validate(state["task"])
            solver_runs = self._collect_solver_runs(state)
            ranked_clusters = self._build_candidate_clusters(
                solver_runs=solver_runs,
                task=task,
                label_schema=state["label_schema"],
            )
            candidate_verifier_runs = {
                variant: AgentRun.model_validate(raw_run)
                for variant, raw_run in state.get("candidate_verifiers", {}).items()
            }

            selected_cluster = self._pick_selected_cluster(
                ranked_clusters=ranked_clusters,
                candidate_verifier_runs=candidate_verifier_runs,
            )
            top_cluster = ranked_clusters[0] if ranked_clusters else None
            selected_solver_variant = (
                str(selected_cluster["representative_variant"])
                if selected_cluster is not None
                else str(top_cluster["representative_variant"])
                if top_cluster is not None
                else ""
            )
            selected_support_variants = (
                list(selected_cluster["support_variants"])
                if selected_cluster is not None
                else list(top_cluster["support_variants"])
                if top_cluster is not None
                else []
            )

            check_primary_variant, check_secondary_variant = self._pick_check_variants(
                selected_cluster=selected_cluster,
                ranked_clusters=ranked_clusters,
                solver_runs=solver_runs,
            )
            if check_primary_variant and check_primary_variant in solver_runs:
                check_primary = CandidateLabel.model_validate(solver_runs[check_primary_variant].output)
            else:
                check_primary = _forced_candidate("missing_check_primary")

            if check_secondary_variant and check_secondary_variant in solver_runs:
                check_secondary = CandidateLabel.model_validate(solver_runs[check_secondary_variant].output)
            else:
                check_secondary = _forced_candidate("no_consensus_partner")

            selected_verifier_variant = self._pick_selected_verifier_variant(
                selected_cluster=selected_cluster,
                ranked_clusters=ranked_clusters,
                candidate_verifier_runs=candidate_verifier_runs,
            )
            if selected_verifier_variant and selected_verifier_variant in candidate_verifier_runs:
                selected_verifier = VerificationResult.model_validate(
                    candidate_verifier_runs[selected_verifier_variant].output
                )
            else:
                selected_verifier = _forced_verifier("no_candidate_verifier")

            checks = run_deterministic_checks(
                settings=self.settings,
                label_schema=state["label_schema"],
                candidate_primary=check_primary,
                candidate_secondary=check_secondary,
                verifier=selected_verifier,
                task_choices=task.choices,
                task_metadata=task.metadata,
                task_constraints=task.constraints,
                primary_name=check_primary_variant or "selected_primary",
                secondary_name=check_secondary_variant or "selected_secondary",
                verifier_name=(
                    f"verifier_for_{selected_verifier_variant}"
                    if selected_verifier_variant
                    else "verifier"
                ),
            )

            consensus_pass = (
                check_primary.status == "answered"
                and check_secondary.status == "answered"
                and labels_consistent(check_primary.label, check_secondary.label)
            )
            if self.settings.benchmark_mode:
                accepted = selected_cluster is not None and selected_verifier.pass_verification
            else:
                accepted = (
                    selected_cluster is not None
                    and checks.passed
                    and (len(selected_cluster["support_variants"]) >= 2 or not self.settings.require_consensus)
                )
            return {
                "deterministic_checks": checks.model_dump(mode="json"),
                "accepted": accepted,
                "consensus_pass": consensus_pass,
                "selected_solver_variant": selected_solver_variant,
                "selected_support_variants": selected_support_variants,
                "selected_verifier_variant": selected_verifier_variant or "",
            }

        def finalize(state: GraphState) -> GraphState:
            task = TaskSample.model_validate(state["task"])
            solver_runs = self._collect_solver_runs(state)
            primary_run = solver_runs["solver_primary"]
            secondary_run = solver_runs["solver_secondary"]
            tiebreaker_run = solver_runs.get("solver_tiebreaker")

            primary = CandidateLabel.model_validate(primary_run.output)
            secondary = CandidateLabel.model_validate(secondary_run.output)
            tiebreaker = (
                CandidateLabel.model_validate(tiebreaker_run.output)
                if tiebreaker_run is not None
                else None
            )

            candidate_verifier_runs = {
                variant: AgentRun.model_validate(raw_run)
                for variant, raw_run in state.get("candidate_verifiers", {}).items()
            }
            candidate_verifiers = {
                variant: VerificationResult.model_validate(run.output)
                for variant, run in candidate_verifier_runs.items()
            }

            selected_verifier_variant = state.get("selected_verifier_variant") or ""
            verifier_output = (
                candidate_verifiers[selected_verifier_variant]
                if selected_verifier_variant in candidate_verifiers
                else _forced_verifier("no_candidate_verifier_selected")
            )

            selected_solver_variant = state.get("selected_solver_variant") or None
            selected_candidate = None
            if selected_solver_variant and selected_solver_variant in solver_runs:
                selected_candidate = CandidateLabel.model_validate(solver_runs[selected_solver_variant].output)

            checks = DeterministicChecks.model_validate(state["deterministic_checks"])
            accepted = bool(state.get("accepted", False) and selected_candidate is not None)
            consensus_pass = bool(state.get("consensus_pass", False))

            failure_reasons = [item.name for item in checks.items if not item.passed]
            if not accepted:
                if not selected_solver_variant and "no_eligible_solver_candidate" not in failure_reasons:
                    failure_reasons.append("no_eligible_solver_candidate")
                if not consensus_pass and "solver_consensus" not in failure_reasons:
                    failure_reasons.append("solver_consensus")
                if not selected_verifier_variant and "no_candidate_verifier" not in failure_reasons:
                    failure_reasons.append("no_candidate_verifier")
                if (
                    self.settings.require_consensus
                    and len(state.get("selected_support_variants", [])) < 2
                    and "insufficient_consensus_support" not in failure_reasons
                ):
                    failure_reasons.append("insufficient_consensus_support")

            trace_path = Path(state["run_dir"]) / "trace.json"
            trace_payload = {
                "task": task.model_dump(mode="json"),
                "solver_primary": primary_run.model_dump(mode="json"),
                "solver_secondary": secondary_run.model_dump(mode="json"),
                "solver_tiebreaker": tiebreaker_run.model_dump(mode="json") if tiebreaker_run else None,
                "candidate_verifiers": {
                    variant: run.model_dump(mode="json")
                    for variant, run in candidate_verifier_runs.items()
                },
                "selection": {
                    "accepted": accepted,
                    "selected_solver_variant": selected_solver_variant,
                    "selected_support_variants": state.get("selected_support_variants", []),
                    "selected_verifier_variant": selected_verifier_variant or None,
                    "consensus_pass": consensus_pass,
                },
                "deterministic_checks": checks.model_dump(mode="json"),
            }
            write_trace(trace_path, trace_payload)

            result = LabelingResult(
                sample_id=task.sample_id,
                status="accepted" if accepted else "abstain",
                accepted_label=selected_candidate.label if accepted and selected_candidate is not None else None,
                solver_primary=primary,
                solver_secondary=secondary,
                solver_tiebreaker=tiebreaker,
                verifier=verifier_output,
                candidate_verifiers=candidate_verifiers,
                deterministic_checks=checks,
                consensus_pass=consensus_pass,
                selected_solver_variant=selected_solver_variant,
                trace_path=str(trace_path),
                failure_reasons=failure_reasons,
            )
            return {"final_result": result.model_dump(mode="json")}

        graph.add_node("solve_primary", solve_primary)
        graph.add_node("solve_secondary", solve_secondary)
        graph.add_node("solve_tiebreaker", solve_tiebreaker)
        graph.add_node("verify_candidates", verify_candidates)
        graph.add_node("run_checks", run_checks)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "solve_primary")
        graph.add_edge("solve_primary", "solve_secondary")
        graph.add_edge("solve_secondary", "solve_tiebreaker")
        graph.add_edge("solve_tiebreaker", "verify_candidates")
        graph.add_edge("verify_candidates", "run_checks")
        graph.add_edge("run_checks", "finalize")
        graph.add_edge("finalize", END)
        return graph

    def run_task(self, task: TaskSample, label_schema: dict[str, Any]) -> LabelingResult:
        run_dir = self.settings.output_dir / task.sample_id
        run_dir.mkdir(parents=True, exist_ok=True)
        runtime = ToolRuntime(run_dir=run_dir, settings=self.settings)
        initial_state = {
            "task": task.model_dump(mode="json"),
            "label_schema": label_schema,
            "run_dir": str(run_dir),
        }

        try:
            graph = self._build_graph(runtime)
            if self.settings.use_checkpoints:
                with SqliteSaver.from_conn_string(str(self.settings.checkpoint_db)) as checkpointer:
                    compiled = graph.compile(checkpointer=checkpointer)
                    final_state = compiled.invoke(
                        initial_state,
                        config={
                            "configurable": {
                                "thread_id": task.sample_id,
                                "checkpoint_ns": "htv_agent",
                            }
                        },
                    )
            else:
                compiled = graph.compile()
                final_state = compiled.invoke(initial_state)
            return LabelingResult.model_validate(final_state["final_result"])
        except Exception as exc:  # noqa: BLE001
            trace_path = run_dir / "trace.json"
            write_trace(
                trace_path,
                {
                    "task": task.model_dump(mode="json"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return LabelingResult(
                sample_id=task.sample_id,
                status="needs_review",
                accepted_label=None,
                solver_primary=_forced_candidate(f"pipeline_error: {exc}"),
                solver_secondary=_forced_candidate(f"pipeline_error: {exc}"),
                solver_tiebreaker=_forced_candidate(f"pipeline_error: {exc}"),
                verifier=_forced_verifier(f"pipeline_error: {exc}"),
                candidate_verifiers={},
                deterministic_checks=DeterministicChecks(
                    passed=False,
                    items=[],
                ),
                consensus_pass=False,
                selected_solver_variant=None,
                trace_path=str(trace_path),
                failure_reasons=[f"pipeline_error:{type(exc).__name__}"],
            )
