# HTV-Agent (Hypothesis–Test Answer Verification)

HTV-Agent is VeriEvol's **answer-reliability** axis. It accepts a candidate
answer only after multi-source counter-evidence has **failed to refute** it —
offline hypothesis–test falsification. Each first answer is treated as a
falsifiable hypothesis, never as ground truth.

This implements the four-stage pipeline described in the paper
(*HTV-Agent: Hypothesis–Test Answer Verification*):

1. **Solvers** — three independent solver branches at temperatures
   `{0.2, 0.6, 0.15}`: a low-temperature primary, a higher-temperature
   secondary for independent trajectories, and a low-temperature tiebreaker
   invoked only on disagreement. Each returns a candidate answer, reasoning,
   claimed evidence, and a self-reported confidence.
2. **Refutation-seeking verifier** — prompted to look for *counter-evidence*
   rather than to confirm. Two complementary evidence channels back it:
   a **programmatic** channel (arithmetic/algebraic/combinatorial constraints
   checked via a restricted interpreter / AST-safe evaluator) and a **visual**
   channel (OCR, local crops, and pixel-level structural probes such as
   projection profiles, connected components, and region measurements).
3. **Conflict-aware decider** — a constrained LLM call (no tools) that judges
   whether the verifier's objection is logically valid and evidence-grounded;
   it revises, retains, or rejects rather than blindly trusting either side.
4. **Deterministic acceptance gate** — a conjunctive gate over schema,
   visual-support, verifier-approval, programmatic, and solver-consensus
   channels. Any single failing channel rejects the sample.

The pipeline is orchestrated with LangGraph and prefers `abstain` over a
low-confidence guess.

## Layout

```text
verifier/
  htv_agent/
    agent.py       # LLM + local-tool agent loop (solver / verifier roles)
    tools.py       # local evidence-channel tools (OCR, crop, measure, math, python_exec)
    pipeline.py    # LangGraph state machine: solvers -> verify -> gate -> finalize
    checks.py      # deterministic acceptance gate
    schemas.py     # task / candidate / verification / result schemas
    settings.py    # env-driven configuration (HTV_AGENT_* prefix)
    llm.py         # OpenAI-compatible model client
    cli.py         # batch runner
  examples/
    label_schema.json
    sample_tasks.jsonl
    assets/simple_chart.png
```

## Install

```bash
cd verifier
python3 -m pip install -e .
cp .env.example .env          # then edit with your endpoint + key
```

Only the model call leaves the machine — every tool runs locally. Model access
uses a standard OpenAI-compatible chat-completions endpoint; credentials and the
endpoint are read from the environment (`HTV_AGENT_OPENAI_API_KEY`,
`HTV_AGENT_OPENAI_BASE_URL`, `HTV_AGENT_MODEL`). No credentials are stored in
this repo.

`ocr_image` requires the `tesseract` binary on `PATH`; if it is missing, the OCR
channel degrades gracefully and the remaining channels still run.

## Input format

Each line of the dataset is one JSON object:

```json
{
  "sample_id": "sample_chart_001",
  "question": "Which month has the highest sales?",
  "images": ["examples/assets/simple_chart.png"],
  "context": "Use local tools before finalizing.",
  "choices": ["January", "February", "March"],
  "task_type": "chart_reasoning",
  "metadata": {"source": "example"},
  "constraints": {"python_assertions": ["label['answer'] in choices"]}
}
```

The target label contract is a JSON schema file (see
`examples/label_schema.json`).

## Run

```bash
htv-agent run \
  --dataset examples/sample_tasks.jsonl \
  --schema examples/label_schema.json \
  --output outputs/results.jsonl

# inspect the effective configuration (secrets are masked)
htv-agent print-config
```

Each accepted sample is written with its full verifier trace (`trace.json` per
sample under `outputs/`), so downstream SFT/RL data construction can audit the
decision rather than only inspect the final label. Feed only accepted samples
forward.
