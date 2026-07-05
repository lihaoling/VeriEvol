# VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct

Official code release for **VeriEvol**. VeriEvol treats data scaling for visual
mathematical reasoning as a *verifiable data-construction* problem and decouples
two axes before any policy update:

- **Prompt difficulty** — expanded by route-specific *evolution operators* that
  rewrite low-difficulty image–question seeds into harder, image-grounded prompts.
- **Answer reliability** — enforced by an offline hypothesis–test verifier
  (**HTV-Agent**) that accepts an answer only after multi-source counter-evidence
  has failed to refute it.

The resulting verified data plugs directly into standard GRPO-style RL recipes.

## Repository layout

```
VeriEvol/
├── evolution/         # Type-aware prompt evolution + answer-quality verification pipeline
├── verifier/          # HTV-Agent: hypothesis–test answer verification (solvers → refutation verifier → decider → gate)
├── sft_training/      # SFT stage (LLaMA-Factory): configs, data scripts, dataset registry
├── rl_training/       # RL stage (verl): GRPO/GSPO experiment configs + launcher
├── paper/             # Paper PDF, figure scripts, evolution case studies
├── docs/              # Per-stage guides
└── vendor/            # Upstream framework submodules (LLaMA-Factory, verl)
```

## Pipeline overview

```
seed (image, question)
      │  evolution/          route-specific evol operators
      ▼
evolved prompt (harder, image-grounded)
      │  evolution/ + verifier/   answer generation + HTV-Agent falsification
      ▼
verified (prompt, answer)  ──►  sft_training/   (LLaMA-Factory)   ──►  SFT-init model
      │
      ▼
                                rl_training/     (verl, GRPO)      ──►  VeriEvol model
```

## Quick start

1. **Clone with submodules** (the two training frameworks are vendored):
   ```bash
   git clone --recurse-submodules <this-repo>
   # or, after a plain clone:
   git submodule update --init --recursive
   ```

2. **Set required environment variables** (no credentials are stored in this repo —
   see [docs/api_setup.md](docs/api_setup.md)):
   ```bash
   export OPENAI_API_KEY=...      # standard OpenAI-format key for evolution/verifier
   export OPENAI_BASE_URL=...     # your inference endpoint
   ```

3. Follow the per-stage guides:
   - [docs/evolution_guide.md](docs/evolution_guide.md) — run prompt evolution + answer quality checks
   - [docs/verifier_guide.md](docs/verifier_guide.md) — run the HTV-Agent objective judge
   - [docs/sft_guide.md](docs/sft_guide.md) — SFT training with LLaMA-Factory
   - [docs/rl_guide.md](docs/rl_guide.md) — GRPO RL training with verl

## Notes on data & credentials

- **No datasets are bundled.** `sft_training/dataset_registry.json` and the RL configs
  reference dataset paths from the original training environment; point them at your
  own data. See the per-stage guides for the expected format.
- **No API keys or endpoints are stored.** All API access uses the standard
  OpenAI-compatible interface and reads credentials/endpoints from environment
  variables. Placeholder tokens like `<VLLM_ENDPOINT>` / `<HCF_ENDPOINT>` must be
  filled in by the user.

## Citation

```bibtex
@article{li2026verievol,
  title  = {VeriEvol: Scaling Multimodal Mathematical Reasoning via Verifiable Evol-Instruct},
  author = {Li, Haoling and Zheng, Kai and Wu, Jie and Xu, Can and Sun, Qingfeng and Hu, Han and Yang, Yujiu},
  year   = {2026}
}
```
