from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import LabelingResult, TaskSample


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tasks(dataset_path: str | Path) -> list[TaskSample]:
    dataset_path = Path(dataset_path)
    tasks: list[TaskSample] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tasks.append(TaskSample.model_validate(json.loads(line)))
    return tasks


def write_results(output_path: str | Path, results: Iterable[LabelingResult]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def write_trace(trace_path: str | Path, payload: dict) -> None:
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
