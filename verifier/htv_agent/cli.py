from __future__ import annotations

import argparse
import json
from typing import Sequence

from .io_utils import load_json, load_tasks, write_results
from .pipeline import HTVAgentPipeline
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HTV-Agent: hypothesis--test answer verification (VeriEvol)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Verify answers for a JSONL dataset via the HTV-Agent pipeline."
    )
    run_parser.add_argument("--dataset", required=True, help="Path to the input JSONL dataset.")
    run_parser.add_argument("--schema", required=True, help="Path to the target label JSON schema.")
    run_parser.add_argument(
        "--output",
        default="outputs/results.jsonl",
        help="Path to the output JSONL file.",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional task limit for smoke testing.",
    )

    subparsers.add_parser("print-config", help="Print effective configuration as JSON.")
    return parser


def run_command(dataset: str, schema: str, output: str, limit: int) -> int:
    settings = load_settings()
    tasks = load_tasks(dataset)
    if limit > 0:
        tasks = tasks[:limit]
    label_schema = load_json(schema)
    pipeline = HTVAgentPipeline(settings)

    results = []
    for index, task in enumerate(tasks, start=1):
        result = pipeline.run_task(task, label_schema)
        print(
            f"[{index}/{len(tasks)}] {task.sample_id}: "
            f"status={result.status} consensus={result.consensus_pass} trace={result.trace_path}"
        )
        results.append(result)
    write_results(output, results)
    print(f"Wrote {len(results)} results to {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args.dataset, args.schema, args.output, args.limit)

    if args.command == "print-config":
        settings = load_settings()
        config = settings.model_dump(mode="json")
        # Never echo the secret back to the console.
        if config.get("openai_api_key"):
            config["openai_api_key"] = "***"
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
