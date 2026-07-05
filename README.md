# VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct

Official code release for **VeriEvol**. VeriEvol treats data scaling for visual
mathematical reasoning as a *verifiable data-construction* problem and decouples
two axes before any policy update:

- **Prompt difficulty** — expanded by route-specific *evolution operators* that
  rewrite low-difficulty image–question seeds into harder, image-grounded prompts.
- **Answer reliability** — enforced by an offline hypothesis–test verifier
  (**HTV-Agent**) that accepts an answer only after multi-source counter-evidence
  has failed to refute it.

The resulting verified `(prompt, answer)` pairs plug directly into standard
SFT and GRPO-style RL recipes.

## Repository layout

```
VeriEvol/
├── evolution/        # Type-aware prompt evolution + answer-quality verification pipeline
├── verifier/         # HTV-Agent: hypothesis–test answer verification
├── third_party/      # Vendored training frameworks (LLaMA-Factory for SFT, verl for RL)
├── requirements.txt  # Core Python deps for evolution + verifier
├── CITATION.bib
└── LICENSE
```

## Pipeline overview

```
seed (image, question)
      │  evolution/classify_topic.py        route the seed to a topic operator
      ▼
      │  evolution/evolve_question.py        route-specific evol operators (harder, image-grounded)
      ▼
evolved prompt
      │  evolution/answer_evol_question.py   generate candidate answers
      │  evolution/vote_consistency.py       self-consistency voting
      │  evolution/answer_quality_check.py   LLM-as-judge dual verification
      │  verifier/ (HTV-Agent)               hypothesis–test falsification + acceptance gate
      ▼
verified (prompt, answer)  ──►  SFT (third_party/LLaMA-Factory)  ──►  RL / GRPO (third_party/verl)
```

## Components

### `evolution/` — prompt evolution & answer construction

A set of standalone, resumable, multi-threaded scripts driven by an
OpenAI-compatible vision-language endpoint (`vision_api.py`):

| Script | Purpose |
| --- | --- |
| `classify_topic.py` | Classify each seed into one of 12 topic categories and label it objective vs. subjective. |
| `evolve_question.py` | Rewrite a simple question into a significantly harder, image-grounded, objective one using the topic-specific operator from `prompt.py`. |
| `answer_evol_question.py` | Generate answers (reasoning + content) for evolved questions; `--output-prefix sft_` for SFT answer generation. |
| `vote_consistency.py` | Self-consistency voting across multiple candidate answers to elect a pseudo ground-truth. |
| `answer_quality_check.py` | LLM-as-a-judge dual verification of answer correctness. |
| `prompt.py` | The 12 route-specific difficulty-escalation operators (OCR, detection, analysis, logical reasoning, scientific, medical, scene understanding, …). |

Each script is `python <script>.py input.jsonl -o output.jsonl --api-endpoints <url> --api-key <key> --model-name <name>`; run with `-h` for the full option set. Failed records are skipped and re-processed on re-run (resume by output ID).

### `verifier/` — HTV-Agent (Hypothesis–Test Answer Verification)

An installable package (`pip install -e verifier`) implementing the four-stage
falsification pipeline: independent **solvers** → refutation-seeking **verifier**
(programmatic + visual evidence channels) → conflict-aware **decider** →
deterministic **acceptance gate**. Every tool runs locally; only the model call
leaves the machine. See [`verifier/README.md`](verifier/README.md) for install,
input format, and CLI usage (`htv-agent run ...`).

### `third_party/` — vendored training frameworks

- `third_party/LLaMA-Factory` — used for the SFT stage.
- `third_party/verl` — used for the GRPO/GSPO RL stage.

These are embedded upstream sources with their own dependencies and docs; install
and run them per their respective READMEs, pointing their configs at the verified
data produced above.

## Quick start

```bash
git clone https://github.com/lihaoling/VeriEvol.git
cd VeriEvol

# core deps for the evolution + verifier pipeline
pip install -r requirements.txt

# HTV-Agent as an installable package (adds the `htv-agent` CLI)
pip install -e verifier
```

Set your OpenAI-compatible endpoint via environment / CLI flags — **no
credentials or endpoints are stored in this repo**. The evolution scripts take
`--api-endpoints` / `--api-key` / `--model-name`; HTV-Agent reads
`HTV_AGENT_OPENAI_API_KEY`, `HTV_AGENT_OPENAI_BASE_URL`, and `HTV_AGENT_MODEL`
(copy `verifier/.env.example` to `verifier/.env`).

## Notes on data & credentials

- **No datasets are bundled.** All scripts take JSONL in/out; point them at your
  own image–question data.
- **No API keys or endpoints are stored.** All model access uses the standard
  OpenAI-compatible interface and reads credentials/endpoints from CLI flags or
  environment variables.

## Citation

```bibtex
@article{li2026verievol,
  title  = {VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct},
  author = {Li, Haoling and Zheng, Kai and Wu, Jie and Xu, Can and Sun, Qingfeng and Hu, Han and Yang, Yujiu},
  year   = {2026}
}
```
