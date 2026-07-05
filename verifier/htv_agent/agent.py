from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import ValidationError, validate
from pydantic import BaseModel
from PIL import Image

from .json_utils import extract_json_objects, truncate_text
from .llm import ModelClientProtocol
from .schemas import (
    AgentEnvelope,
    AgentRun,
    CandidateLabel,
    EvidenceItem,
    ToolResult,
    VerificationResult,
)
from .settings import Settings
from .tools import ToolRegistry, ToolRuntime


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MIN_MODEL_IMAGE_DIMENSION = 14


@dataclass
class PreflightExecution:
    steps: list[dict[str, Any]]
    observation_text: str
    tool_names: list[str]
    attached_images: list[str]
    successful_summaries: list[str]


@dataclass
class InteractiveExecution:
    steps: list[dict[str, Any]]
    tool_names: list[str]
    base_images: list[str]
    attached_images: list[str]
    successful_summaries: list[str]
    final_output: BaseModel | None
    last_response: dict[str, Any] | None


class LLMToolAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        client: ModelClientProtocol,
        registry: ToolRegistry,
        runtime: ToolRuntime,
    ):
        self.settings = settings
        self.client = client
        self.registry = registry
        self.runtime = runtime

    def _parse_structured_json(self, raw_text: str, model_cls: type[BaseModel]) -> BaseModel:
        candidates = extract_json_objects(raw_text)
        if not candidates:
            raise ValueError("No JSON object found in model output.")
        candidate_errors: list[str] = []
        for candidate in candidates:
            try:
                return model_cls.model_validate(candidate)
            except Exception as exc:  # noqa: BLE001
                candidate_errors.append(f"{type(exc).__name__}: {exc}")
        raise ValueError(
            f"No JSON candidate matched {model_cls.__name__}. "
            + " | ".join(candidate_errors[:3])
        )

    def _repair_structured_output(
        self,
        *,
        model_cls: type[BaseModel],
        system_prompt: str,
        malformed_text: str,
    ) -> dict[str, Any]:
        repair_system = (
            "You are a strict JSON formatter. "
            f"Convert the provided draft into exactly one valid JSON object matching the {model_cls.__name__} schema. "
            "Return JSON only, with no prose, no markdown, and no explanations."
        )
        repair_user = (
            f"Target JSON schema:\n{json.dumps(model_cls.model_json_schema(), ensure_ascii=False, indent=2)}\n\n"
            f"Original instruction summary:\n{truncate_text(system_prompt, 3000)}\n\n"
            f"Draft to repair:\n{truncate_text(malformed_text, 3000)}\n\n"
            "Return exactly one valid JSON object."
        )
        return self.client.complete(
            system_prompt=repair_system,
            user_prompt=repair_user,
            image_paths=[],
            temperature=0.0,
        )

    def _call_structured_model(
        self,
        *,
        model_cls: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float,
    ) -> tuple[BaseModel, dict[str, Any]]:
        last_error = "unknown_error"
        last_raw_text = ""
        for attempt in range(self.settings.max_format_retries + 1):
            suffix = ""
            if attempt:
                suffix = (
                    "\n\nYour previous answer was not valid JSON. "
                    "Return exactly one JSON object matching the required schema, with no markdown or extra prose.\n"
                    f"Previous raw answer:\n{truncate_text(last_raw_text, 1200)}"
                )
            response = self.client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt + suffix,
                image_paths=image_paths,
                temperature=temperature,
            )
            last_raw_text = response["text"]
            try:
                parsed = self._parse_structured_json(response["text"], model_cls)
                return parsed, response
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"

        if last_raw_text:
            repair_response = self._repair_structured_output(
                model_cls=model_cls,
                system_prompt=system_prompt,
                malformed_text=last_raw_text,
            )
            try:
                parsed = self._parse_structured_json(repair_response["text"], model_cls)
                return parsed, repair_response
            except Exception as exc:  # noqa: BLE001
                salvaged = self._salvage_model_output(repair_response["text"], model_cls)
                if salvaged is not None:
                    return salvaged, repair_response
                last_error = (
                    f"{type(exc).__name__}: {exc}. "
                    f"Repair raw output: {truncate_text(repair_response['text'], 1200)}"
                )

        salvaged = self._salvage_model_output(last_raw_text, model_cls)
        if salvaged is not None:
            return salvaged, {"text": last_raw_text, "raw": {"salvaged": True}}

        raise ValueError(
            f"Could not parse model output into {model_cls.__name__}: {last_error}. "
            f"Last raw output: {truncate_text(last_raw_text, 1200)}"
        )

    def _parse_agent_envelope(self, raw_text: str, final_model_cls: type[BaseModel]) -> AgentEnvelope:
        candidates = extract_json_objects(raw_text)
        if not candidates:
            raise ValueError("No JSON object found in model output.")

        candidate_errors: list[str] = []
        for candidate in candidates:
            envelope_error = None
            try:
                envelope = AgentEnvelope.model_validate(candidate)
                if envelope.action == "tool":
                    if not envelope.tool_name:
                        raise ValueError("AgentEnvelope action='tool' requires tool_name.")
                    return envelope
                if envelope.final is None:
                    raise ValueError("AgentEnvelope action='final' requires final.")
                parsed_final = final_model_cls.model_validate(envelope.final)
                return envelope.model_copy(update={"final": parsed_final.model_dump(mode="json")})
            except Exception as exc:  # noqa: BLE001
                envelope_error = f"{type(exc).__name__}: {exc}"

            try:
                parsed_final = final_model_cls.model_validate(candidate)
                return AgentEnvelope(
                    thought="Recovered direct final output.",
                    action="final",
                    final=parsed_final.model_dump(mode="json"),
                )
            except Exception as exc:  # noqa: BLE001
                candidate_errors.append(
                    f"AgentEnvelope -> {envelope_error} | {final_model_cls.__name__} -> {type(exc).__name__}: {exc}"
                )

        raise ValueError(
            f"No JSON candidate matched AgentEnvelope/{final_model_cls.__name__}. "
            + " | ".join(candidate_errors[:3])
        )

    def _repair_agent_envelope(
        self,
        *,
        system_prompt: str,
        malformed_text: str,
        final_model_cls: type[BaseModel],
    ) -> dict[str, Any]:
        repair_system = (
            "You are a strict JSON formatter for an agent controller. "
            "Return exactly one AgentEnvelope JSON object and nothing else. "
            "If the agent is using a tool, set action='tool'. If the agent is done, set action='final' and provide a valid final object."
        )
        repair_user = (
            f"AgentEnvelope schema:\n{json.dumps(AgentEnvelope.model_json_schema(), ensure_ascii=False, indent=2)}\n\n"
            f"Final object schema ({final_model_cls.__name__}):\n"
            f"{json.dumps(final_model_cls.model_json_schema(), ensure_ascii=False, indent=2)}\n\n"
            f"Original instruction summary:\n{truncate_text(system_prompt, 3000)}\n\n"
            f"Draft to repair:\n{truncate_text(malformed_text, 3000)}\n\n"
            "Return exactly one AgentEnvelope JSON object."
        )
        return self.client.complete(
            system_prompt=repair_system,
            user_prompt=repair_user,
            image_paths=[],
            temperature=0.0,
        )

    def _call_agent_model(
        self,
        *,
        final_model_cls: type[BaseModel],
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float,
    ) -> tuple[AgentEnvelope, dict[str, Any]]:
        last_error = "unknown_error"
        last_raw_text = ""
        for attempt in range(self.settings.max_format_retries + 1):
            suffix = ""
            if attempt:
                suffix = (
                    "\n\nYour previous answer was not valid JSON. "
                    "Return exactly one AgentEnvelope JSON object, with no markdown or extra prose.\n"
                    f"Previous raw answer:\n{truncate_text(last_raw_text, 1200)}"
                )
            response = self.client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt + suffix,
                image_paths=image_paths,
                temperature=temperature,
            )
            last_raw_text = response["text"]
            try:
                envelope = self._parse_agent_envelope(response["text"], final_model_cls)
                return envelope, response
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                salvaged_envelope = self._salvage_agent_envelope(
                    response["text"],
                    final_model_cls,
                    image_paths,
                )
                if salvaged_envelope is not None and salvaged_envelope.action == "tool":
                    return salvaged_envelope, response

        salvaged_envelope = self._salvage_agent_envelope(last_raw_text, final_model_cls, image_paths)
        if salvaged_envelope is not None:
            return salvaged_envelope, {"text": last_raw_text, "raw": {"salvaged": True}}

        if last_raw_text:
            repair_response = self._repair_agent_envelope(
                system_prompt=system_prompt,
                malformed_text=last_raw_text,
                final_model_cls=final_model_cls,
            )
            try:
                envelope = self._parse_agent_envelope(repair_response["text"], final_model_cls)
                return envelope, repair_response
            except Exception as exc:  # noqa: BLE001
                salvaged_envelope = self._salvage_agent_envelope(
                    repair_response["text"],
                    final_model_cls,
                    image_paths,
                )
                if salvaged_envelope is not None:
                    return salvaged_envelope, repair_response
                salvaged = self._salvage_model_output(repair_response["text"], final_model_cls)
                if salvaged is not None:
                    return (
                        AgentEnvelope(
                            thought="Recovered final output from repaired response.",
                            action="final",
                            final=salvaged.model_dump(mode="json"),
                        ),
                        repair_response,
                    )
                last_error = (
                    f"{type(exc).__name__}: {exc}. "
                    f"Repair raw output: {truncate_text(repair_response['text'], 1200)}"
                )

        salvaged = self._salvage_model_output(last_raw_text, final_model_cls)
        if salvaged is not None:
            return (
                AgentEnvelope(
                    thought="Recovered final output from malformed response.",
                    action="final",
                    final=salvaged.model_dump(mode="json"),
                ),
                {"text": last_raw_text, "raw": {"salvaged": True}},
            )

        raise ValueError(
            f"Could not parse model output into AgentEnvelope/{final_model_cls.__name__}: {last_error}. "
            f"Last raw output: {truncate_text(last_raw_text, 1200)}"
        )

    def _extract_thought(self, raw_text: str) -> str | None:
        patterns = [
            r'thought(?: is|:)?\s*"([^"]+)"',
            r"thought(?: is|:)?\s*'([^']+)'",
            r'thought(?: is|:)?\s*(.+?)(?:(?:\.\s+Then action)|(?:\n+\s*Then action)|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            thought = match.group(1).strip().strip('"').strip("'")
            if thought:
                return truncate_text(thought, 500)
        return None

    def _extract_json_argument_object(self, raw_text: str) -> dict[str, Any] | None:
        for candidate in extract_json_objects(raw_text):
            if isinstance(candidate.get("arguments"), dict):
                return candidate["arguments"]
            if any(
                key in candidate
                for key in (
                    "image_path",
                    "bbox",
                    "relative_bbox",
                    "psm",
                    "lang",
                    "resize_factor",
                    "grayscale",
                    "invert",
                    "binarize_threshold",
                    "label",
                    "axis",
                    "top_k",
                    "threshold",
                    "min_area",
                    "connectivity",
                    "rows",
                    "cols",
                    "expression",
                    "code",
                )
            ):
                return candidate

        alias_pattern = re.compile(r"(?:arguments|args)\s*(?:is|are|=|:)?", flags=re.IGNORECASE)
        decoder = json.JSONDecoder()
        for match in alias_pattern.finditer(raw_text):
            start = raw_text.find("{", match.end())
            if start < 0:
                continue
            try:
                parsed, _ = decoder.raw_decode(raw_text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _extract_number_list(self, raw_text: str, aliases: list[str]) -> list[int] | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*\[([^\]]+)\]",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if not match:
                continue
            raw_items = [item.strip() for item in match.group(1).split(",")]
            try:
                return [int(float(item)) for item in raw_items if item]
            except ValueError:
                continue
        return None

    def _extract_float_list(self, raw_text: str, aliases: list[str]) -> list[float] | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*\[([^\]]+)\]",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if not match:
                continue
            raw_items = [item.strip() for item in match.group(1).split(",")]
            try:
                return [float(item) for item in raw_items if item]
            except ValueError:
                continue
        return None

    def _extract_optional_int(self, raw_text: str, aliases: list[str]) -> int | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*([0-9]+)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if match:
                return int(match.group(1))
        return None

    def _extract_optional_float(self, raw_text: str, aliases: list[str]) -> float | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if match:
                return float(match.group(1))
        return None

    def _extract_optional_bool(self, raw_text: str, aliases: list[str]) -> bool | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*(true|false)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if match:
                return match.group(1).lower() == "true"
        return None

    def _extract_expression(self, raw_text: str) -> str | None:
        expression = self._extract_quoted(raw_text, ["expression"])
        if expression:
            return expression
        pattern = re.compile(r"expression\s*(?:is|=|:)?\s*([^\n.]+)", flags=re.IGNORECASE)
        match = pattern.search(raw_text)
        if match:
            return match.group(1).strip().strip('"')
        return None

    def _extract_code(self, raw_text: str) -> str | None:
        code = self._extract_quoted(raw_text, ["code"])
        if code:
            return code
        fence_match = re.search(r"```(?:python)?\n(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            return fence_match.group(1).strip()
        return None

    def _coerce_tool_arguments(
        self,
        *,
        tool_name: str,
        raw_text: str,
        available_image_paths: list[str],
    ) -> dict[str, Any] | None:
        parsed_args = self._extract_json_argument_object(raw_text) or {}
        latest_image_path = available_image_paths[-1] if available_image_paths else None
        base_image_path = parsed_args.get("image_path") if isinstance(parsed_args.get("image_path"), str) else None
        image_path = (
            base_image_path
            or self._extract_quoted(raw_text, ["image_path", "image path"])
            or latest_image_path
        )

        if tool_name == "inspect_image":
            if not image_path:
                return None
            return {"image_path": image_path}

        if tool_name == "crop_image":
            bbox = parsed_args.get("bbox") if isinstance(parsed_args.get("bbox"), list) else None
            if bbox is None:
                bbox = self._extract_number_list(raw_text, ["bbox"])
            if not image_path or not isinstance(bbox, list):
                return None
            arguments = {
                "image_path": image_path,
                "bbox": [int(value) for value in bbox],
            }
            label = parsed_args.get("label") if isinstance(parsed_args.get("label"), str) else None
            label = label or self._extract_quoted(raw_text, ["label"])
            if label:
                arguments["label"] = label
            return arguments

        if tool_name == "relative_crop_image":
            relative_bbox = (
                parsed_args.get("relative_bbox") if isinstance(parsed_args.get("relative_bbox"), list) else None
            )
            if relative_bbox is None:
                relative_bbox = self._extract_float_list(raw_text, ["relative_bbox", "relative bbox"])
            if not image_path or not isinstance(relative_bbox, list):
                return None
            arguments = {
                "image_path": image_path,
                "relative_bbox": [float(value) for value in relative_bbox],
            }
            label = parsed_args.get("label") if isinstance(parsed_args.get("label"), str) else None
            label = label or self._extract_quoted(raw_text, ["label"])
            if label:
                arguments["label"] = label
            return arguments

        if tool_name == "ocr_image":
            if not image_path:
                return None
            arguments: dict[str, Any] = {"image_path": image_path}
            bbox = parsed_args.get("bbox") if isinstance(parsed_args.get("bbox"), list) else None
            if bbox is None:
                bbox = self._extract_number_list(raw_text, ["bbox"])
            if isinstance(bbox, list):
                arguments["bbox"] = [int(value) for value in bbox]

            label = parsed_args.get("label") if isinstance(parsed_args.get("label"), str) else None
            label = label or self._extract_quoted(raw_text, ["label"])
            if label:
                arguments["label"] = label

            psm = parsed_args.get("psm") if isinstance(parsed_args.get("psm"), int) else None
            psm = psm if psm is not None else self._extract_optional_int(raw_text, ["psm"])
            if psm is not None:
                arguments["psm"] = psm

            lang = parsed_args.get("lang") if isinstance(parsed_args.get("lang"), str) else None
            lang = lang or self._extract_quoted(raw_text, ["lang"])
            if lang:
                arguments["lang"] = lang

            resize_factor = (
                parsed_args.get("resize_factor")
                if isinstance(parsed_args.get("resize_factor"), (int, float))
                else None
            )
            resize_factor = (
                float(resize_factor)
                if resize_factor is not None
                else self._extract_optional_float(raw_text, ["resize_factor", "resize factor"])
            )
            if resize_factor is not None:
                arguments["resize_factor"] = resize_factor

            grayscale = (
                parsed_args.get("grayscale")
                if isinstance(parsed_args.get("grayscale"), bool)
                else None
            )
            grayscale = grayscale if grayscale is not None else self._extract_optional_bool(raw_text, ["grayscale"])
            if grayscale is not None:
                arguments["grayscale"] = grayscale

            invert = parsed_args.get("invert") if isinstance(parsed_args.get("invert"), bool) else None
            invert = invert if invert is not None else self._extract_optional_bool(raw_text, ["invert"])
            if invert is not None:
                arguments["invert"] = invert

            threshold = (
                parsed_args.get("binarize_threshold")
                if isinstance(parsed_args.get("binarize_threshold"), int)
                else None
            )
            threshold = (
                threshold
                if threshold is not None
                else self._extract_optional_int(raw_text, ["binarize_threshold", "binarize threshold"])
            )
            if threshold is not None:
                arguments["binarize_threshold"] = threshold
            return arguments

        if tool_name == "math_eval":
            expression = parsed_args.get("expression") if isinstance(parsed_args.get("expression"), str) else None
            expression = expression or self._extract_expression(raw_text)
            if not expression:
                return None
            return {"expression": expression}

        if tool_name in {
            "measure_region",
            "projection_profile",
            "connected_components",
            "foreground_bounds",
            "grid_region_scan",
        }:
            if not image_path:
                return None
            arguments: dict[str, Any] = {"image_path": image_path}

            bbox = parsed_args.get("bbox") if isinstance(parsed_args.get("bbox"), list) else None
            if bbox is None:
                bbox = self._extract_number_list(raw_text, ["bbox"])
            if isinstance(bbox, list):
                arguments["bbox"] = [int(value) for value in bbox]

            threshold = parsed_args.get("threshold") if isinstance(parsed_args.get("threshold"), int) else None
            threshold = threshold if threshold is not None else self._extract_optional_int(raw_text, ["threshold"])
            if threshold is not None:
                arguments["threshold"] = threshold

            if tool_name == "projection_profile":
                axis = parsed_args.get("axis") if isinstance(parsed_args.get("axis"), str) else None
                axis = axis or self._extract_word(raw_text, ["axis"], {"x", "y"})
                if axis is not None:
                    arguments["axis"] = axis
                top_k = parsed_args.get("top_k") if isinstance(parsed_args.get("top_k"), int) else None
                top_k = top_k if top_k is not None else self._extract_optional_int(raw_text, ["top_k", "top k"])
                if top_k is not None:
                    arguments["top_k"] = top_k

            if tool_name == "connected_components":
                min_area = parsed_args.get("min_area") if isinstance(parsed_args.get("min_area"), int) else None
                min_area = (
                    min_area if min_area is not None else self._extract_optional_int(raw_text, ["min_area", "min area"])
                )
                if min_area is not None:
                    arguments["min_area"] = min_area

                connectivity = (
                    parsed_args.get("connectivity") if isinstance(parsed_args.get("connectivity"), int) else None
                )
                connectivity = (
                    connectivity
                    if connectivity is not None
                    else self._extract_optional_int(raw_text, ["connectivity"])
                )
                if connectivity is not None:
                    arguments["connectivity"] = connectivity

            if tool_name == "grid_region_scan":
                rows = parsed_args.get("rows") if isinstance(parsed_args.get("rows"), int) else None
                rows = rows if rows is not None else self._extract_optional_int(raw_text, ["rows", "row count"])
                if rows is not None:
                    arguments["rows"] = rows

                cols = parsed_args.get("cols") if isinstance(parsed_args.get("cols"), int) else None
                cols = cols if cols is not None else self._extract_optional_int(raw_text, ["cols", "columns", "col count"])
                if cols is not None:
                    arguments["cols"] = cols

                top_k = parsed_args.get("top_k") if isinstance(parsed_args.get("top_k"), int) else None
                top_k = top_k if top_k is not None else self._extract_optional_int(raw_text, ["top_k", "top k"])
                if top_k is not None:
                    arguments["top_k"] = top_k
            return arguments

        if tool_name == "python_exec":
            code = parsed_args.get("code") if isinstance(parsed_args.get("code"), str) else None
            code = code or self._extract_code(raw_text)
            if not code:
                return None
            return {"code": code}

        return None

    def _infer_tool_name_from_text(self, raw_text: str, tool_names: set[str]) -> str | None:
        verb_prefix = r"(?:use|using|run|running|call|invoke|try|execute)"
        for tool_name in sorted(tool_names, key=len, reverse=True):
            escaped = re.escape(tool_name)
            patterns = [
                rf"\btool(?:\s+call|\s+name)?\b[^\n]*\b{escaped}\b",
                rf"\b{verb_prefix}\s+{escaped}\b",
            ]
            for pattern in patterns:
                if re.search(pattern, raw_text, flags=re.IGNORECASE):
                    return tool_name

        alias_patterns = [
            (r"\brelative crop(?: image)?\b", "relative_crop_image"),
            (r"\bcrop(?: image)?\b", "crop_image"),
            (r"\bocr\b", "ocr_image"),
            (r"\bprojection profile\b", "projection_profile"),
            (r"\bconnected components?\b", "connected_components"),
            (r"\bforeground bounds?\b", "foreground_bounds"),
            (r"\bgrid scan\b", "grid_region_scan"),
            (r"\bmeasure region\b", "measure_region"),
            (r"\bmath eval\b", "math_eval"),
            (r"\bpython exec\b", "python_exec"),
        ]
        for pattern, tool_name in alias_patterns:
            if tool_name not in tool_names:
                continue
            if re.search(pattern, raw_text, flags=re.IGNORECASE):
                return tool_name
        return None

    def _salvage_agent_envelope(
        self,
        raw_text: str,
        final_model_cls: type[BaseModel],
        available_image_paths: list[str],
    ) -> AgentEnvelope | None:
        if not raw_text.strip():
            return None

        thought = self._extract_thought(raw_text) or truncate_text(raw_text.strip(), 500)
        tool_names = set(self.registry.names())
        action = self._extract_word(raw_text, ["action"], {"tool", "final"})
        tool_name = self._extract_word(raw_text, ["tool_name", "tool name"], tool_names)
        inferred_tool_name = None
        if action != "final" and tool_name is None:
            inferred_tool_name = self._infer_tool_name_from_text(raw_text, tool_names)

        if action == "tool" or tool_name is not None or inferred_tool_name is not None:
            resolved_tool_name = tool_name or inferred_tool_name
            if resolved_tool_name is None:
                return None
            arguments = self._coerce_tool_arguments(
                tool_name=resolved_tool_name,
                raw_text=raw_text,
                available_image_paths=available_image_paths,
            )
            if arguments is None:
                return None
            return AgentEnvelope(
                thought=thought,
                action="tool",
                tool_name=resolved_tool_name,
                arguments=arguments,
            )

        if action == "final" or action is None:
            parsed_final = self._salvage_model_output(raw_text, final_model_cls)
            if parsed_final is not None:
                return AgentEnvelope(
                    thought=thought,
                    action="final",
                    final=parsed_final.model_dump(mode="json"),
                )
        return None

    def _salvage_model_output(self, raw_text: str, model_cls: type[BaseModel]) -> BaseModel | None:
        if model_cls is CandidateLabel:
            return self._salvage_candidate_label(raw_text)
        if model_cls is VerificationResult:
            return self._salvage_verification_result(raw_text)
        return None


    def _extract_quoted(self, raw_text: str, aliases: list[str]) -> str | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf'(?:label(?:\s+object)?(?:\s+has)?\s+)?{alias_pattern}\s*(?:is|=|:)?\s*"([^"]+)"',
                flags=re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(raw_text)
            if match:
                return match.group(1).strip()
        return None

    def _clean_salvaged_answer(self, value: str) -> str | None:
        text = value.strip()
        if not text:
            return None

        boxed_match = re.fullmatch(r"\\boxed\{\s*([^{}]+)\s*\}", text)
        if boxed_match:
            text = boxed_match.group(1).strip()

        text_wrapper_match = re.fullmatch(r"\\(?:text|mathrm)\{\s*([^{}]+)\s*\}", text)
        if text_wrapper_match:
            text = text_wrapper_match.group(1).strip()

        text = text.strip("`*_\"'")
        text = re.sub(r"^(?:option|choice|letter)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^(?:选项|答案)\s*", "", text)
        text = text.strip("()[]{}<>")
        text = re.split(r'"\s*,\s*"[A-Za-z_][A-Za-z0-9_ ]*"\s*:', text, maxsplit=1)[0]
        text = re.split(
            r"[?!]\s+|\s+(?:wait|because|since|as|for|where|when|which|that|but|however|actually|let me|no,)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        text = re.split(r"\n|。|！|？|;|；|,(?=\s)|\.(?=\s|$)", text, maxsplit=1)[0]
        text = text.strip("`*_\"'()[]{}<> ，,.:;!?")
        if not text or len(text) > 80:
            return None
        if text.casefold() in {"is", "are", "be", "not", "unknown", "unclear", "none", "null", "not reliable", "n/a"}:
            return None
        if re.fullmatch(r"[A-Za-z]", text):
            return text.upper()
        return text

    def _extract_unquoted_value(self, raw_text: str, aliases: list[str]) -> str | None:
        tail = raw_text[-2000:]
        candidate = None
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"(?:label(?:\s+object)?(?:\s+has)?\s+)?\b{alias_pattern}\b\s*(?:is|=|:)\s*([^\n]+)",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(tail):
                value = self._clean_salvaged_answer(match.group(1))
                if value:
                    candidate = value
        return candidate

    def _extract_answer_from_prose(self, raw_text: str) -> str | None:
        tail = raw_text[-2000:]

        direct = self._extract_unquoted_value(
            tail,
            ["answer_text", "answer text", "final_answer", "final answer", "choice", "answer", "equation", "formula", "expression"],
        )
        if direct:
            return direct

        candidate = None
        for match in re.finditer(r"\\boxed\{\s*([^{}]{1,80})\s*\}", tail):
            value = self._clean_salvaged_answer(match.group(1))
            if value:
                candidate = value
        if candidate:
            return candidate

        letter_patterns = [
            r"(?:final\s+answer|correct\s+answer|correct\s+option|correct\s+choice|answer|option|choice)\s*(?:is|=|:)\s*\(?([A-Za-z])\)?(?=[^A-Za-z]|$)",
            r"(?:choose|pick|select)\s*\(?([A-Za-z])\)?(?=[^A-Za-z]|$)",
            r"(?:最终答案|正确答案|答案|选项)\s*(?:是|为|:)?\s*([A-Za-z])(?=[^A-Za-z]|$)",
            r"(?:选择|选)\s*([A-Za-z])(?=[^A-Za-z]|$)",
        ]
        for pattern in letter_patterns:
            for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
                value = self._clean_salvaged_answer(match.group(1))
                if value:
                    candidate = value
        if candidate:
            return candidate

        text_patterns = [
            r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*([^\n]+)",
            r"(?:final\s+answer|correct\s+answer|answer_text|answer\s+text|final_answer|answer)\s*(?:is|=|:)\s*([^\n]+)",
            r"(?:equation|formula|expression)\s*(?:is|=|:)\s*([^\n]+)",
            r"(?:choose|pick|select)\s+([^\n]+)",
            r"(?:最终答案|正确答案|答案|选项)\s*(?:是|为|:)?\s*([^\n]+)",
            r"(?:方程|公式|表达式)\s*(?:是|为|:)?\s*([^\n]+)",
            r"(?:选择|选)\s+([^\n]+)",
        ]
        for pattern in text_patterns:
            for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
                value = self._clean_salvaged_answer(match.group(1))
                if value:
                    candidate = value
        if candidate:
            return candidate

        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        if not lines:
            return None
        last_line = lines[-1]
        if len(last_line) > 24:
            return None
        value = self._clean_salvaged_answer(last_line)
        if not value:
            return None
        if re.fullmatch(r"[A-Z]", value):
            return value
        if re.fullmatch(r"-?[0-9]+(?:/[0-9]+)?(?:\.[0-9]+)?%?", value):
            return value
        if len(value.split()) <= 4:
            return value
        return None

    def _extract_word(
        self,
        raw_text: str,
        aliases: list[str],
        allowed: set[str] | None = None,
    ) -> str | None:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf'{alias_pattern}\s*(?:is|=|:)?\s*"?([A-Za-z_][A-Za-z0-9_.-]*)"?',
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if not match:
                continue
            value = match.group(1).strip().lower().rstrip(".,;:")
            if allowed and value not in allowed:
                continue
            return value
        return None

    def _extract_float(self, raw_text: str, aliases: list[str], default: float) -> float:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if not match:
                continue
            try:
                return float(match.group(1))
            except ValueError:
                continue
        return default

    def _extract_bool(self, raw_text: str, aliases: list[str], default: bool) -> bool:
        for alias in aliases:
            alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
            pattern = re.compile(
                rf"{alias_pattern}\s*(?:is|=|:)?\s*(true|false)",
                flags=re.IGNORECASE,
            )
            match = pattern.search(raw_text)
            if match:
                return match.group(1).lower() == "true"
        return default

    def _extract_label_fields(self, raw_text: str) -> dict[str, Any]:
        preferred = {
            "answer": self._extract_quoted(raw_text, ["answer"]) or self._extract_unquoted_value(raw_text, ["answer"]),
            "reasoning_type": self._extract_quoted(raw_text, ["reasoning_type", "reasoning type"])
            or self._extract_unquoted_value(raw_text, ["reasoning_type", "reasoning type"]),
            "support": self._extract_quoted(raw_text, ["support"]) or self._extract_unquoted_value(raw_text, ["support"]),
            "choice": self._extract_quoted(raw_text, ["choice"]) or self._extract_unquoted_value(raw_text, ["choice"]),
            "final_answer": self._extract_quoted(raw_text, ["final_answer", "final answer"])
            or self._extract_unquoted_value(raw_text, ["final_answer", "final answer"]),
        }
        return {key: value for key, value in preferred.items() if value is not None}

    def _salvage_candidate_label(self, raw_text: str) -> CandidateLabel | None:
        if not raw_text:
            return None

        label = self._extract_label_fields(raw_text)
        recovered_answer = label.get("answer") or label.get("choice") or label.get("final_answer")
        if recovered_answer is None:
            recovered_answer = self._extract_answer_from_prose(raw_text)
        if recovered_answer is not None and "answer" not in label:
            label["answer"] = recovered_answer
        if "answer" in label and set(label).issubset({"answer", "choice", "final_answer"}):
            label = {"answer": str(label["answer"])}

        answer_text = (
            self._extract_quoted(raw_text, ["answer_text", "answer text"])
            or self._extract_unquoted_value(raw_text, ["answer_text", "answer text"])
            or label.get("answer")
            or label.get("final_answer")
            or label.get("choice")
        )
        status = self._extract_word(raw_text, ["status"], {"answered", "abstain"})
        if status is None:
            status = "answered" if label or answer_text else "abstain"

        confidence_default = 0.8 if status == "answered" else 0.0
        confidence = self._extract_float(raw_text, ["confidence"], default=confidence_default)
        concise_reasoning = (
            self._extract_quoted(raw_text, ["concise_reasoning", "concise reasoning"])
            or self._extract_unquoted_value(raw_text, ["concise_reasoning", "concise reasoning"])
            or self._extract_quoted(raw_text, ["support"])
            or self._extract_unquoted_value(raw_text, ["support"])
            or truncate_text(raw_text.strip(), 300)
        )

        evidence: list[EvidenceItem] = []
        support = label.get("support")
        if support:
            evidence.append(EvidenceItem(kind="text", source="model_support", content=support))
        content = self._extract_quoted(raw_text, ["content"]) or self._extract_unquoted_value(raw_text, ["content"])
        if content:
            evidence.append(
                EvidenceItem(
                    kind="tool_output",
                    source=self._extract_word(raw_text, ["source"], None) or "recovered_output",
                    content=content,
                )
            )

        abstain_reason = None
        if status == "abstain":
            abstain_reason = (
                self._extract_quoted(raw_text, ["abstain_reason", "abstain reason"])
                or self._extract_unquoted_value(raw_text, ["abstain_reason", "abstain reason"])
                or truncate_text(raw_text.strip(), 200)
            )

        try:
            return CandidateLabel(
                status=status,
                label=label or None,
                answer_text=answer_text,
                confidence=confidence,
                concise_reasoning=concise_reasoning,
                evidence=evidence,
                abstain_reason=abstain_reason,
            )
        except Exception:  # noqa: BLE001
            return None

    def _recover_benchmark_answer_from_raw_text(
        self,
        *,
        candidate: CandidateLabel,
        raw_response: dict[str, Any],
        label_schema: dict[str, Any],
        execution: InteractiveExecution,
    ) -> CandidateLabel:
        if not self.settings.benchmark_mode or candidate.status == "answered":
            return candidate

        raw_text = str(raw_response.get("text", "")).strip()
        if not raw_text:
            return candidate

        recovered_answer = self._extract_answer_from_prose(raw_text)
        if recovered_answer is None:
            return candidate

        recovered_candidate = CandidateLabel(
            status="answered",
            label={"answer": recovered_answer},
            answer_text=str(recovered_answer),
            confidence=max(candidate.confidence, 0.8),
            concise_reasoning=candidate.concise_reasoning or truncate_text(raw_text, 300),
            evidence=list(candidate.evidence),
            tools_used=list(candidate.tools_used),
        )
        recovered_candidate = self._finalize_candidate_label(recovered_candidate, execution)
        if self._validate_candidate_against_label_schema(recovered_candidate, label_schema) is None:
            return recovered_candidate
        return candidate

    def _salvage_verification_result(self, raw_text: str) -> VerificationResult | None:
        if not raw_text:
            return None

        pass_verification = self._extract_bool(raw_text, ["pass_verification", "pass verification"], default=False)
        if not pass_verification:
            lowered = raw_text.casefold()
            positive_markers = [
                "pass verification",
                "verification passes",
                "verification passed",
                "passes verification",
                "pass the verification",
                "finalize the verification as passing",
                "the evidence fully supports",
                "fully supports the candidate",
                "well-supported",
                "well supported",
                "no contradictions",
                "candidate is correct",
                "answer is correct",
                "verified",
            ]
            negative_markers = [
                "fail verification",
                "verification fails",
                "verification failed",
                "fails verification",
                "cannot verify",
                "not enough evidence",
                "insufficient evidence",
                "contradiction",
                "contradictions remain",
                "still contradictory",
                "not supported",
                "unsupported",
            ]
            has_positive = any(marker in lowered for marker in positive_markers)
            has_negative = any(marker in lowered for marker in negative_markers)
            if "no contradictions" in lowered or "without contradiction" in lowered:
                has_negative = False
            if has_positive and not has_negative:
                pass_verification = True

        confidence_default = 0.85 if pass_verification else 0.0
        confidence = self._extract_float(raw_text, ["confidence"], default=confidence_default)
        supported = (
            self._extract_quoted(raw_text, ["supported_claims", "supported claims"])
            or self._extract_unquoted_value(raw_text, ["supported_claims", "supported claims"])
        )
        summary = (
            self._extract_quoted(raw_text, ["summary"])
            or self._extract_unquoted_value(raw_text, ["summary"])
            or truncate_text(raw_text.strip(), 300)
        )
        issue = self._extract_quoted(raw_text, ["issues", "issue"]) or self._extract_unquoted_value(raw_text, ["issues", "issue"])
        missing = (
            self._extract_quoted(raw_text, ["missing_evidence", "missing evidence"])
            or self._extract_unquoted_value(raw_text, ["missing_evidence", "missing evidence"])
        )

        try:
            return VerificationResult(
                pass_verification=pass_verification,
                confidence=confidence,
                supported_claims=[supported] if supported else [],
                issues=[issue] if issue else [],
                missing_evidence=[missing] if missing else [],
                summary=summary,
            )
        except Exception:  # noqa: BLE001
            return None

    def _tool_signature(self, tool_name: str, arguments: dict[str, Any]) -> str:
        return json.dumps(
            {"tool_name": tool_name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
        )

    def _is_attachable_image_path(self, image_path: str) -> bool:
        if image_path.startswith(("data:", "http://", "https://")):
            return True
        path = Path(image_path)
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.exists():
            return False
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:  # noqa: BLE001
            return False
        return width >= MIN_MODEL_IMAGE_DIMENSION and height >= MIN_MODEL_IMAGE_DIMENSION

    def _merge_attached_images(
        self,
        base_images: list[str],
        current_attached: list[str],
        new_artifacts: list[str],
    ) -> list[str]:
        deduped_base: list[str] = []
        for image_path in base_images[: self.settings.max_attached_images]:
            if image_path in deduped_base:
                continue
            if not self._is_attachable_image_path(image_path):
                continue
            deduped_base.append(image_path)

        extras: list[str] = []
        for image_path in current_attached:
            if image_path in deduped_base or image_path in extras:
                continue
            if not self._is_attachable_image_path(image_path):
                continue
            extras.append(image_path)

        for artifact in new_artifacts:
            if artifact in deduped_base or artifact in extras:
                continue
            if not self._is_attachable_image_path(artifact):
                continue
            extras.append(artifact)

        extra_limit = max(0, self.settings.max_attached_images - len(deduped_base))
        if extra_limit == 0:
            return deduped_base[: self.settings.max_attached_images]
        return deduped_base + extras[-extra_limit:]

    def _model_image_paths(
        self,
        *,
        base_images: list[str],
        current_attached: list[str],
    ) -> list[str]:
        deduped_base: list[str] = []
        for image_path in base_images[: self.settings.max_attached_images]:
            if image_path in deduped_base:
                continue
            if not self._is_attachable_image_path(image_path):
                continue
            deduped_base.append(image_path)

        if self.settings.benchmark_mode:
            return deduped_base[: self.settings.max_attached_images]

        extras: list[str] = []
        for image_path in current_attached:
            if image_path in deduped_base or image_path in extras:
                continue
            if not self._is_attachable_image_path(image_path):
                continue
            extras.append(image_path)

        extra_limit = max(0, self.settings.max_attached_images - len(deduped_base))
        if extra_limit == 0:
            return deduped_base[: self.settings.max_attached_images]
        return deduped_base + extras[-extra_limit:]

    def _render_step_history(self, steps: list[dict[str, Any]]) -> str:
        if not steps:
            return "(no tool history yet)"

        lines: list[str] = []
        for step in steps:
            action = step.get("action")
            if action == "tool":
                result = step.get("tool_result", {})
                arguments = step.get("arguments", {})
                summary = truncate_text(str(result.get("summary", "")), 500)
                artifacts = result.get("artifacts") or []
                artifact_text = ""
                if artifacts:
                    artifact_text = f"; artifacts={json.dumps(artifacts, ensure_ascii=False)}"
                lines.append(
                    f"{step.get('step')}. tool {step.get('tool_name')} "
                    f"args={json.dumps(arguments, ensure_ascii=False, sort_keys=True)} "
                    f"-> ok={result.get('ok')} summary={summary}{artifact_text}"
                )
                continue

            final_payload = truncate_text(
                json.dumps(step.get("final", {}), ensure_ascii=False),
                500,
            )
            lines.append(f"{step.get('step')}. final {final_payload}")
        return "\n".join(lines)

    def _render_available_images(self, image_paths: list[str]) -> str:
        if not image_paths:
            return "(no attached images)"
        return "\n".join(f"- {path}" for path in image_paths)

    def _upsert_final_step(
        self,
        steps: list[dict[str, Any]],
        *,
        thought: str,
        raw_model_text: str,
        final_payload: dict[str, Any],
    ) -> None:
        final_step = {
            "step": len(steps) + 1,
            "thought": thought,
            "action": "final",
            "raw_model_text": raw_model_text,
            "final": final_payload,
        }
        if steps and steps[-1].get("action") == "final":
            final_step["step"] = steps[-1].get("step", len(steps))
            steps[-1] = final_step
            return
        steps.append(final_step)

    def _is_low_signal_ocr_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        tokens = re.findall(r"[A-Za-z0-9]+", stripped)
        alnum_count = sum(char.isalnum() for char in stripped)
        if alnum_count < 12:
            return True
        if len(tokens) < 2 and len(stripped) < 24:
            return True
        return False

    def _build_preflight_ocr_retry_plan(self, image_path: str) -> list[tuple[dict[str, Any], str]]:
        retry_variants = [
            (
                {
                    "image_path": image_path,
                    "label": "preflight_dense_text",
                    "psm": 11,
                    "resize_factor": 2.0,
                    "grayscale": True,
                },
                "Framework-managed preflight OCR retry for dense or sparse text.",
            ),
            (
                {
                    "image_path": image_path,
                    "label": "preflight_binarized_text",
                    "psm": 6,
                    "resize_factor": 2.5,
                    "grayscale": True,
                    "binarize_threshold": 180,
                },
                "Framework-managed preflight OCR retry with aggressive binarization.",
            ),
            (
                {
                    "image_path": image_path,
                    "label": "preflight_inverted_text",
                    "psm": 7,
                    "resize_factor": 3.0,
                    "grayscale": True,
                    "invert": True,
                },
                "Framework-managed preflight OCR retry for inverted or low-contrast text.",
            ),
        ]
        return retry_variants[: self.settings.max_ocr_retries_per_region]

    def _should_create_preflight_focus_crop(self, full_bbox: list[int], focus_bbox: list[int]) -> bool:
        if len(full_bbox) != 4 or len(focus_bbox) != 4:
            return False
        if focus_bbox == full_bbox:
            return False
        full_area = max(0, full_bbox[2] - full_bbox[0]) * max(0, full_bbox[3] - full_bbox[1])
        focus_area = max(0, focus_bbox[2] - focus_bbox[0]) * max(0, focus_bbox[3] - focus_bbox[1])
        if full_area <= 0 or focus_area <= 0:
            return False
        edge_delta = max(abs(focus_bbox[index] - full_bbox[index]) for index in range(4))
        return focus_area <= full_area * 0.98 or edge_delta >= 4


    def _execute_preflight_tools(self, image_paths: list[str]) -> PreflightExecution:
        steps: list[dict[str, Any]] = []
        observation_lines: list[str] = []
        tool_names: list[str] = []
        base_images: list[str] = []
        for image_path in image_paths[: self.settings.max_attached_images]:
            if image_path not in base_images:
                base_images.append(image_path)
        attached_images = list(base_images)
        successful_summaries: list[str] = []
        available_tools = set(self.registry.names())
        profile = (self.settings.preflight_profile or "full").strip().lower()
        if profile not in {"full", "focused", "none"}:
            profile = "full"

        step_number = 0

        def record_tool(tool_name: str, arguments: dict[str, Any], thought: str) -> ToolResult:
            nonlocal attached_images, step_number
            step_number += 1
            result = self.registry.invoke(tool_name, arguments, self.runtime)
            result_dict = result.model_dump(mode="json")
            steps.append(
                {
                    "step": step_number,
                    "thought": thought,
                    "action": "tool",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "tool_result": result_dict,
                }
            )
            if result.ok:
                if tool_name not in tool_names:
                    tool_names.append(tool_name)
                successful_summaries.append(result.summary)
                observation_lines.append(
                    f"- {tool_name} on {Path(arguments.get('image_path', '')).name or 'artifact'}: "
                    f"{truncate_text(result.summary, 500)}"
                )
                attached_images = self._merge_attached_images(base_images, attached_images, result.artifacts)
            else:
                observation_lines.append(
                    f"- {tool_name} on {Path(arguments.get('image_path', '')).name or 'artifact'} failed: "
                    f"{truncate_text(result.summary, 500)}"
                )
            return result

        if profile == "none":
            return PreflightExecution(
                steps=steps,
                observation_text="(preflight disabled for this run profile)",
                tool_names=tool_names,
                attached_images=attached_images,
                successful_summaries=successful_summaries,
            )

        for image_path in base_images:
            full_bbox: list[int] | None = None
            if "inspect_image" in available_tools:
                inspect_result = record_tool(
                    "inspect_image",
                    {"image_path": image_path},
                    "Framework-managed preflight inspection.",
                )
                if inspect_result.ok:
                    width = int(inspect_result.data.get("width", 0))
                    height = int(inspect_result.data.get("height", 0))
                    if width > 0 and height > 0:
                        full_bbox = [0, 0, width, height]

            if profile == "focused":
                continue

            if "ocr_image" in available_tools:
                default_result = record_tool(
                    "ocr_image",
                    {"image_path": image_path},
                    "Framework-managed preflight OCR pass.",
                )
                default_text = ""
                if default_result.ok:
                    default_text = str(default_result.data.get("text", "")).strip()

                if self._is_low_signal_ocr_text(default_text):
                    for retry_arguments, retry_thought in self._build_preflight_ocr_retry_plan(image_path):
                        record_tool("ocr_image", retry_arguments, retry_thought)

            if "projection_profile" in available_tools:
                record_tool(
                    "projection_profile",
                    {
                        "image_path": image_path,
                        "axis": "x",
                        "threshold": 220,
                        "top_k": 8,
                    },
                    "Framework-managed preflight horizontal density scan for bars or text columns.",
                )
                record_tool(
                    "projection_profile",
                    {
                        "image_path": image_path,
                        "axis": "y",
                        "threshold": 220,
                        "top_k": 8,
                    },
                    "Framework-managed preflight vertical density scan for baselines or text bands.",
                )

            if "connected_components" in available_tools:
                record_tool(
                    "connected_components",
                    {
                        "image_path": image_path,
                        "threshold": 220,
                        "min_area": 25,
                        "connectivity": 8,
                    },
                    "Framework-managed preflight connected-component scan to estimate object density.",
                )

            focus_bbox: list[int] | None = None
            if "foreground_bounds" in available_tools:
                bounds_result = record_tool(
                    "foreground_bounds",
                    {
                        "image_path": image_path,
                        "threshold": 220,
                    },
                    "Framework-managed preflight foreground-bounds scan to localize the main evidence region.",
                )
                if bounds_result.ok and bool(bounds_result.data.get("has_foreground")):
                    raw_focus_bbox = bounds_result.data.get("foreground_bbox_in_image")
                    if isinstance(raw_focus_bbox, list):
                        focus_bbox = [int(value) for value in raw_focus_bbox]
                    if full_bbox is None:
                        raw_full_bbox = bounds_result.data.get("bbox")
                        if isinstance(raw_full_bbox, list):
                            full_bbox = [int(value) for value in raw_full_bbox]

            if (
                focus_bbox is not None
                and full_bbox is not None
                and "crop_image" in available_tools
                and self._should_create_preflight_focus_crop(full_bbox, focus_bbox)
            ):
                record_tool(
                    "crop_image",
                    {
                        "image_path": image_path,
                        "bbox": focus_bbox,
                        "label": "preflight_focus",
                    },
                    "Framework-managed preflight crop around the main foreground evidence.",
                )

            if "grid_region_scan" in available_tools:
                record_tool(
                    "grid_region_scan",
                    {
                        "image_path": image_path,
                        "threshold": 220,
                        "rows": 3,
                        "cols": 3,
                        "top_k": 4,
                    },
                    "Framework-managed preflight grid scan to compare regional density across the image.",
                )

        observation_text = "\n".join(observation_lines) if observation_lines else "(no preflight tool observations)"
        return PreflightExecution(
            steps=steps,
            observation_text=observation_text,
            tool_names=tool_names,
            attached_images=attached_images,
            successful_summaries=successful_summaries,
        )

    def _build_solver_system_prompt(self, label_schema: dict[str, Any], variant: str) -> str:
        candidate_schema = CandidateLabel.model_json_schema()
        envelope_schema = AgentEnvelope.model_json_schema()
        tools_text = self.registry.render_for_prompt() or "(no tools available)"
        if self.settings.benchmark_mode:
            strategy = (
                "You are a high-precision math and reasoning solver.\n"
                "Goal: maximize answer correctness. Think step by step before deciding whether to use tools.\n"
                "Strategy:\n"
                "1. READ the question carefully. Think through the math/logic FIRST before using any tool.\n"
                "2. If you can solve the problem by reasoning alone, finalize immediately — do NOT use tools unnecessarily.\n"
                "3. Use python_exec or math_eval for non-trivial calculations (arithmetic, algebra, geometry, combinatorics). "
                "ALWAYS prefer python_exec for multi-step math: define variables, compute symbolically, and print the final answer.\n"
                "4. Use image tools (OCR, crop, measure) ONLY when the image contains data you cannot read from the question text alone (e.g., a chart, diagram values, hidden numbers).\n"
                "5. Do NOT spend turns on broad image exploration. If the image just illustrates a geometry problem you can solve from the text, skip image tools.\n"
                "6. When one answer is clearly best supported, return it. NEVER default to abstain when you have a computed answer.\n"
                "7. For math problems: solve step-by-step, use python_exec to verify, then finalize with the computed answer. "
                "Set status='answered', confidence >= 0.8, and put the final numeric or symbolic answer in label.answer.\n"
            )
        else:
            strategy = (
                "You are a high-precision multimodal labeling agent.\n"
                "Goal: maximize answer correctness on complex image reasoning tasks.\n"
                "Use local tools aggressively before finalizing whenever uncertainty can be reduced.\n"
                "Recommended strategy:\n"
                "1. Inspect image layout first.\n"
                "2. Use projection_profile, grid_region_scan, or foreground_bounds to localize evidence-bearing regions early.\n"
                "3. Crop relevant regions before OCR when text is small or localized.\n"
                "4. Use relative_crop_image to target regions by proportional layout when absolute coordinates are awkward.\n"
                "5. Retry OCR with different bbox/psm/resize/grayscale/invert/threshold settings when text is noisy.\n"
                "6. Use measure_region to compare darkness or filled area across candidate regions.\n"
                "7. Use connected_components to count blobs or isolate marks, digits, and symbols.\n"
                "8. Use foreground_bounds to estimate the true extent of bars, marks, or text blocks after coarse localization.\n"
                "9. Use grid_region_scan to quickly compare candidate areas before fine-grained measurement.\n"
                "10. Use math_eval or python_exec instead of mental arithmetic when calculations matter.\n"
                "11. If evidence remains ambiguous, abstain.\n"
            )
        return (
            strategy
            + "Output contract:\n"
            + "- Return exactly one AgentEnvelope JSON object and nothing else.\n"
            + "- For another tool call, set action='tool' and provide tool_name + arguments.\n"
            + "- For the final answer, set action='final' and put a complete CandidateLabel object in final.\n"
            + "- Never invent tool outputs.\n"
            + f"Independent solve variant: {variant}\n"
            + f"Available tools:\n{tools_text}\n"
            + f"Target label JSON schema (for CandidateLabel.label):\n{json.dumps(label_schema, ensure_ascii=False, indent=2)}\n"
            + f"CandidateLabel schema:\n{json.dumps(candidate_schema, ensure_ascii=False, indent=2)}\n"
            + f"AgentEnvelope schema:\n{json.dumps(envelope_schema, ensure_ascii=False, indent=2)}"
        )

    def _build_verifier_system_prompt(self, label_schema: dict[str, Any]) -> str:
        verifier_schema = VerificationResult.model_json_schema()
        envelope_schema = AgentEnvelope.model_json_schema()
        tools_text = self.registry.render_for_prompt() or "(no tools available)"
        strategy = (
            "You are a skeptical independent verifier for multimodal labels.\n"
            "Goal: reject weak answers and only pass labels that are directly supported by image evidence.\n"
            "Try to falsify the candidate before accepting it. Use tools whenever ambiguity remains.\n"
            "Recommended strategy:\n"
            "1. Inspect layout and the regions that determine the answer.\n"
            "2. Use projection_profile, grid_region_scan, or foreground_bounds to localize the decisive evidence region.\n"
            "3. Crop and OCR the specific evidence-bearing regions instead of trusting a full-image read.\n"
            "4. Use relative crops when the region should be selected proportionally from the page or chart.\n"
            "5. Use measure_region to compare bar heights, fill density, or shaded options.\n"
            "6. Use connected_components when counting marks or checking whether distinct objects exist.\n"
            "7. Re-run OCR with different settings if text is noisy or partial.\n"
            "8. Use foreground_bounds to verify extents after you have isolated a candidate region.\n"
            "9. Use math tools when numeric consistency matters.\n"
            "10. If evidence is incomplete or contradictory, fail verification.\n"
        )
        if self.settings.benchmark_mode:
            strategy += (
                "11. These are benchmark evaluation tasks; judge the best-supported answer, not impossible certainty.\n"
                "12. Use math_eval whenever arithmetic, geometry, ratios, or symbolic consistency determine correctness.\n"
                "13. Fail verification when there is a concrete contradiction or a clearly better competing answer, not merely because confidence is imperfect.\n"
            )
        return (
            strategy
            + "Output contract:\n"
            + "- Return exactly one AgentEnvelope JSON object and nothing else.\n"
            + "- For another tool call, set action='tool' and provide tool_name + arguments.\n"
            + "- For the final decision, set action='final' and put a complete VerificationResult object in final.\n"
            + "- Never invent tool outputs.\n"
            + f"Available tools:\n{tools_text}\n"
            + f"Target label schema:\n{json.dumps(label_schema, ensure_ascii=False, indent=2)}\n"
            + f"VerificationResult schema:\n{json.dumps(verifier_schema, ensure_ascii=False, indent=2)}\n"
            + f"AgentEnvelope schema:\n{json.dumps(envelope_schema, ensure_ascii=False, indent=2)}"
        )

    def _build_solver_user_prompt(
        self,
        *,
        question: str,
        context: str | None,
        choices: list[str] | None,
        preflight: PreflightExecution,
        initial_note: str,
    ) -> str:
        sections = [f"Question:\n{question}"]
        if context:
            sections.append(f"Context:\n{context}")
        if choices:
            sections.append(f"Choices:\n{json.dumps(choices, ensure_ascii=False)}")
            sections.append(
                "IMPORTANT: For multiple-choice questions, your label.answer MUST be exactly one uppercase letter "
                f"from {choices}. Do NOT put calculations, explanations, formulas, or 'Option X' — just the single letter."
            )
        if initial_note:
            sections.append(f"Solver note:\n{initial_note}")
        sections.append(f"Framework preflight observations:\n{preflight.observation_text}")
        sections.append(
            "Use tools until the answer is directly supported. "
            "If the answer depends on relative size, count, rank, position, or filled area, rely on at least one structural observation from projection_profile, connected_components, foreground_bounds, grid_region_scan, or measure_region. "
            "If you cannot make the answer reliable, return a final CandidateLabel with status='abstain'."
        )
        if self.settings.benchmark_mode:
            sections.append(
                "Benchmark mode: maximize final-answer correctness. "
                "Use math_eval early for arithmetic or symbolic reasoning, and if one option or number is clearly best supported after tool use, return it instead of defaulting to abstain."
            )
        return "\n\n".join(sections)

    def _build_verifier_user_prompt(
        self,
        *,
        question: str,
        context: str | None,
        choices: list[str] | None,
        preflight: PreflightExecution,
        candidate_label: dict[str, Any],
        variant: str,
    ) -> str:
        sections = [f"Question:\n{question}"]
        if context:
            sections.append(f"Context:\n{context}")
        if choices:
            sections.append(f"Choices:\n{json.dumps(choices, ensure_ascii=False)}")
        sections.append(f"Verifier target variant:\n{variant}")
        sections.append(f"Candidate label to verify:\n{json.dumps(candidate_label, ensure_ascii=False, indent=2)}")
        sections.append(f"Framework preflight observations:\n{preflight.observation_text}")
        sections.append(
            "Actively look for contradictions. "
            "For comparison, counting, or ranking claims, use structural tools rather than relying on OCR alone. "
            "Only pass if the candidate label is well supported by direct evidence from the image and tools."
        )
        if self.settings.benchmark_mode:
            sections.append(
                "Benchmark mode: verify whether this is the best-supported available answer. "
                "Use math_eval for consistency checks and fail only when you find a real contradiction or unresolved competing answer."
            )
        return "\n\n".join(sections)

    def _compose_loop_user_prompt(
        self,
        *,
        base_user_prompt: str,
        steps: list[dict[str, Any]],
        attached_images: list[str],
        iteration: int,
        max_steps: int,
    ) -> str:
        if iteration == max_steps:
            if self.settings.benchmark_mode:
                decision_note = (
                    "This is the last tool-decision turn. "
                    "Take one more tool call only if it can materially change the answer; otherwise finalize the best-supported answer now."
                )
            else:
                decision_note = (
                    "This is the last tool-decision turn. "
                    "If one more tool call would materially reduce uncertainty, take it; otherwise finalize. "
                    "Do not guess."
                )
        else:
            if self.settings.benchmark_mode:
                decision_note = (
                    "Use another tool when it can resolve uncertainty. "
                    "If one option or numeric answer is already clearly best supported, finalize it instead of drifting toward abstention."
                )
            else:
                decision_note = (
                    "If any uncertainty remains, prefer another tool call over an unsupported final answer. "
                    "Finalize only when the evidence is strong."
                )
        sections = [
            base_user_prompt,
            f"Available image paths for tool arguments:\n{self._render_available_images(attached_images)}",
            f"Tool history so far:\n{self._render_step_history(steps)}",
            f"Decision turn {iteration} of {max_steps}.\n{decision_note}",
            "Return AgentEnvelope JSON only.",
        ]
        return "\n\n".join(sections)

    def _build_forced_final_user_prompt(
        self,
        *,
        base_user_prompt: str,
        steps: list[dict[str, Any]],
        attached_images: list[str],
        instruction: str,
    ) -> str:
        sections = [
            base_user_prompt,
            f"Available image paths:\n{self._render_available_images(attached_images)}",
            f"Collected tool history:\n{self._render_step_history(steps)}",
            instruction,
        ]
        return "\n\n".join(sections)

    def _run_agent_loop(
        self,
        *,
        final_model_cls: type[BaseModel],
        system_prompt: str,
        base_user_prompt: str,
        preflight: PreflightExecution,
        original_image_paths: list[str],
        temperature: float,
        max_steps: int,
    ) -> InteractiveExecution:
        steps = list(preflight.steps)
        tool_names = list(preflight.tool_names)
        attached_images = list(preflight.attached_images)
        successful_summaries = list(preflight.successful_summaries)
        base_images: list[str] = []
        for image_path in original_image_paths[: self.settings.max_attached_images]:
            if image_path not in base_images:
                base_images.append(image_path)

        tool_signatures = {
            self._tool_signature(step.get("tool_name", ""), step.get("arguments", {}))
            for step in steps
            if step.get("action") == "tool" and step.get("tool_name")
        }

        for iteration in range(1, max_steps + 1):
            user_prompt = self._compose_loop_user_prompt(
                base_user_prompt=base_user_prompt,
                steps=steps,
                attached_images=attached_images,
                iteration=iteration,
                max_steps=max_steps,
            )
            model_image_paths = self._model_image_paths(
                base_images=base_images,
                current_attached=attached_images,
            )
            envelope, raw_response = self._call_agent_model(
                final_model_cls=final_model_cls,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=model_image_paths,
                temperature=temperature,
            )

            if envelope.action == "final":
                final_output = final_model_cls.model_validate(envelope.final)
                self._upsert_final_step(
                    steps,
                    thought=str(envelope.thought),
                    raw_model_text=raw_response["text"],
                    final_payload=final_output.model_dump(mode="json"),
                )
                return InteractiveExecution(
                    steps=steps,
                    tool_names=tool_names,
                    base_images=base_images,
                    attached_images=attached_images,
                    successful_summaries=successful_summaries,
                    final_output=final_output,
                    last_response=raw_response,
                )

            tool_name = str(envelope.tool_name)
            arguments = dict(envelope.arguments)
            signature = self._tool_signature(tool_name, arguments)
            if signature in tool_signatures:
                result = ToolResult(
                    ok=False,
                    summary=(
                        "Duplicate tool invocation blocked. Choose a different tool or change the arguments "
                        "to gather new evidence."
                    ),
                    error="duplicate_tool_invocation",
                )
            else:
                tool_signatures.add(signature)
                result = self.registry.invoke(tool_name, arguments, self.runtime)

            steps.append(
                {
                    "step": len(steps) + 1,
                    "thought": envelope.thought,
                    "action": "tool",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "raw_model_text": raw_response["text"],
                    "tool_result": result.model_dump(mode="json"),
                }
            )
            if result.ok:
                if tool_name not in tool_names:
                    tool_names.append(tool_name)
                successful_summaries.append(result.summary)
                attached_images = self._merge_attached_images(base_images, attached_images, result.artifacts)

        return InteractiveExecution(
            steps=steps,
            tool_names=tool_names,
            base_images=base_images,
            attached_images=attached_images,
            successful_summaries=successful_summaries,
            final_output=None,
            last_response=None,
        )

    @staticmethod
    def _normalize_choice_answer(answer: str | None, choices: list[str] | None) -> str | None:
        """If the task has choices, force the answer to a valid option letter."""
        if not answer or not choices:
            return answer
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        valid = {letter.upper() for letter in letters}
        stripped = answer.strip().strip("()[]{}<> \"'`*_").upper()
        if stripped in valid:
            return stripped
        # Try regex extraction of a single option letter from noisy answer text.
        text = answer.strip()
        for pattern in [
            r"\(([A-Za-z])\)",
            r"\boption\s+([A-Za-z])\b",
            r"\bchoice\s+([A-Za-z])\b",
            r"\b([A-Za-z])\b",
        ]:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                candidate = match.upper()
                if candidate in valid:
                    return candidate
        return answer

    def _finalize_candidate_label(
        self,
        candidate: CandidateLabel,
        execution: InteractiveExecution,
        choices: list[str] | None = None,
    ) -> CandidateLabel:
        tools_used = list(dict.fromkeys(execution.tool_names))
        evidence = list(candidate.evidence)
        if candidate.status == "answered" and not evidence:
            if isinstance(candidate.label, dict) and isinstance(candidate.label.get("support"), str):
                evidence.append(
                    EvidenceItem(
                        kind="text",
                        source="model_support",
                        content=candidate.label["support"],
                    )
                )
            elif execution.successful_summaries:
                evidence.append(
                    EvidenceItem(
                        kind="tool_output",
                        source=tools_used[0] if tools_used else "tool_summary",
                        content=execution.successful_summaries[0],
                    )
                )

        answer_text = candidate.answer_text
        if answer_text is None and isinstance(candidate.label, dict):
            for key in ("answer", "choice", "final_answer", "label"):
                value = candidate.label.get(key)
                if isinstance(value, str):
                    answer_text = value
                    break

        # Normalize MC answer to valid option letter
        label = candidate.label
        if choices and isinstance(label, dict) and "answer" in label:
            normalized = self._normalize_choice_answer(str(label["answer"]), choices)
            if normalized != label["answer"]:
                label = {**label, "answer": normalized}
                answer_text = normalized

        return candidate.model_copy(
            update={
                "label": label,
                "tools_used": tools_used,
                "evidence": evidence,
                "answer_text": answer_text,
            }
        )

    def _validate_candidate_against_label_schema(
        self,
        candidate: CandidateLabel,
        label_schema: dict[str, Any],
    ) -> str | None:
        if candidate.status != "answered":
            return None
        try:
            validate(instance=candidate.label, schema=label_schema)
        except ValidationError as exc:
            return exc.message
        return None

    def _repair_candidate_schema_if_needed(
        self,
        *,
        candidate: CandidateLabel,
        raw_response: dict[str, Any],
        label_schema: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        execution: InteractiveExecution,
    ) -> tuple[CandidateLabel, dict[str, Any]]:
        schema_error = self._validate_candidate_against_label_schema(candidate, label_schema)
        if schema_error is None:
            return candidate, raw_response

        last_candidate = candidate
        last_response = raw_response
        last_error = schema_error

        for _ in range(2):
            repair_system = (
                "You are a strict JSON repairer for CandidateLabel.\n"
                "Return exactly one CandidateLabel JSON object and nothing else.\n"
                "When status is 'answered', CandidateLabel.label must satisfy the target label schema.\n"
                "Do not omit required label fields.\n"
                "Keep the answer conservative and grounded in the provided observations."
            )
            repair_user = (
                f"Original solver instructions:\n{truncate_text(system_prompt, 2500)}\n\n"
                f"Original task context:\n{truncate_text(user_prompt, 3500)}\n\n"
                f"Collected tool history:\n{truncate_text(self._render_step_history(execution.steps), 3500)}\n\n"
                f"Current invalid CandidateLabel JSON:\n{json.dumps(last_candidate.model_dump(mode='json'), ensure_ascii=False, indent=2)}\n\n"
                f"Target label schema:\n{json.dumps(label_schema, ensure_ascii=False, indent=2)}\n\n"
                f"Schema validation error:\n{last_error}\n\n"
                f"Previous raw model output:\n{truncate_text(last_response.get('text', ''), 2500)}\n\n"
                "Rewrite the full CandidateLabel JSON so that label satisfies the target schema."
            )
            try:
                repaired, repaired_response = self._call_structured_model(
                    model_cls=CandidateLabel,
                    system_prompt=repair_system,
                    user_prompt=repair_user,
                    image_paths=self._model_image_paths(
                        base_images=execution.base_images,
                        current_attached=execution.attached_images,
                    ),
                    temperature=0.0,
                )
            except Exception:  # noqa: BLE001
                break

            repaired_candidate = self._finalize_candidate_label(repaired, execution)  # type: ignore[arg-type]
            repaired_error = self._validate_candidate_against_label_schema(repaired_candidate, label_schema)
            last_candidate = repaired_candidate
            last_response = repaired_response
            last_error = repaired_error or ""
            if repaired_error is None:
                return repaired_candidate, repaired_response

        return last_candidate, last_response

    def _pre_think(
        self,
        *,
        question: str,
        context: str | None,
        choices: list[str] | None,
        preflight: PreflightExecution,
        label_schema: dict[str, Any],
        image_paths: list[str],
    ) -> tuple[CandidateLabel | None, dict[str, Any]]:
        """Single-shot raw LLM call without tools to get a baseline answer."""
        candidate_schema = CandidateLabel.model_json_schema()
        system_prompt = (
            "You are an expert at solving math and reasoning problems from images.\n"
            "Do NOT call tools, functions, notebooks, plugins, Python, or code interpreters.\n"
            "Analyze the question and any provided image observations, then return your answer.\n"
            "Return exactly one CandidateLabel JSON object.\n"
            f"Target label schema:\n{json.dumps(label_schema, ensure_ascii=False, indent=2)}\n"
            f"CandidateLabel schema:\n{json.dumps(candidate_schema, ensure_ascii=False, indent=2)}"
        )
        sections = [f"Question:\n{question}"]
        if context:
            sections.append(f"Context:\n{context}")
        if choices:
            sections.append(f"Choices:\n{json.dumps(choices, ensure_ascii=False)}")
            sections.append(
                "IMPORTANT: Your label.answer MUST be exactly one uppercase letter "
                f"from {choices}. Do NOT put calculations or explanations in the answer field."
            )
        sections.append(f"Image observations:\n{preflight.observation_text}")
        sections.append(
            "Think step by step, then return the CandidateLabel JSON with your best answer. "
            "Set confidence to reflect how sure you are."
        )
        user_prompt = "\n\n".join(sections)
        try:
            parsed, response = self._call_structured_model(
                model_cls=CandidateLabel,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths[: self.settings.max_attached_images],
                temperature=0.0,
            )
            return parsed, response  # type: ignore[return-value]
        except Exception:  # noqa: BLE001
            return None, {"text": ""}

    @staticmethod
    def _classify_needs_tools(
        pre_think: CandidateLabel | None,
        choices: list[str] | None,
    ) -> bool:
        """Decide whether the agent tool loop is needed after pre-think."""
        if pre_think is None:
            return True
        if pre_think.status != "answered":
            return True
        if pre_think.confidence < 0.7:
            return True
        # For MC questions: check if answer is a valid choice letter
        if choices:
            answer = None
            if isinstance(pre_think.label, dict):
                answer = pre_think.label.get("answer")
            if answer is None:
                return True
            valid = {chr(ord("A") + i) for i in range(len(choices))}
            if str(answer).strip().upper() not in valid:
                return True
        return False

    def run_solver(
        self,
        *,
        sample_id: str,
        question: str,
        image_paths: list[str],
        context: str | None,
        choices: list[str] | None,
        label_schema: dict[str, Any],
        variant: str,
        temperature: float,
        initial_note: str = "",
    ) -> AgentRun:
        preflight = self._execute_preflight_tools(image_paths)

        # --- Pre-think + Router (benchmark_mode only) ---
        pre_think_candidate: CandidateLabel | None = None
        pre_think_response: dict[str, Any] = {"text": ""}
        skip_tools = False

        if self.settings.enable_pre_think and self.settings.benchmark_mode:
            pre_think_candidate, pre_think_response = self._pre_think(
                question=question,
                context=context,
                choices=choices,
                preflight=preflight,
                label_schema=label_schema,
                image_paths=image_paths,
            )
            skip_tools = not self._classify_needs_tools(pre_think_candidate, choices)

        if skip_tools and pre_think_candidate is not None:
            # Router says no tools needed - use pre_think directly
            system_prompt = ""
            base_user_prompt = ""
            execution = InteractiveExecution(
                steps=list(preflight.steps),
                tool_names=list(preflight.tool_names),
                base_images=image_paths[: self.settings.max_attached_images],
                attached_images=list(preflight.attached_images),
                successful_summaries=list(preflight.successful_summaries),
                final_output=pre_think_candidate,
                last_response=pre_think_response,
            )
        else:
            # Build note with pre_think hint if available
            solver_note = initial_note or f"Sample id: {sample_id}"
            if pre_think_candidate is not None and pre_think_candidate.status == "answered":
                pt_answer = None
                if isinstance(pre_think_candidate.label, dict):
                    pt_answer = pre_think_candidate.label.get("answer")
                if pt_answer is not None:
                    solver_note += (
                        f"\nPre-analysis suggests the answer may be: {pt_answer}. "
                        "Use tools to verify or correct this if the image provides additional evidence."
                    )

            system_prompt = self._build_solver_system_prompt(label_schema, variant)
            base_user_prompt = self._build_solver_user_prompt(
                question=question,
                context=context,
                choices=choices,
                preflight=preflight,
                initial_note=solver_note,
            )
            execution = self._run_agent_loop(
                final_model_cls=CandidateLabel,
                system_prompt=system_prompt,
                base_user_prompt=base_user_prompt,
                preflight=preflight,
                original_image_paths=image_paths,
                temperature=temperature,
                max_steps=self.settings.max_solver_steps,
            )

        raw_response = execution.last_response or {"text": ""}
        if execution.final_output is None:
            if choices:
                choice_constraint = (
                    f"\nCRITICAL: This is a multiple-choice question. "
                    f"label.answer MUST be exactly one letter from {choices}. "
                    f"Do NOT return 'null', 'not reliable', or any text other than a single option letter. "
                    f"Pick the best-supported option even if uncertain."
                )
            else:
                choice_constraint = ""
            forced_system = (
                "You have exhausted the tool budget. Return exactly one CandidateLabel JSON object and nothing else.\n"
                "Use only the collected evidence."
                + choice_constraint + "\n"
                + f"Target label JSON schema:\n{json.dumps(label_schema, ensure_ascii=False, indent=2)}\n"
                + f"CandidateLabel schema:\n{json.dumps(CandidateLabel.model_json_schema(), ensure_ascii=False, indent=2)}"
            )
            forced_instruction = (
                "No more tool calls are allowed. Produce the final CandidateLabel now. "
                "If one answer is clearly best supported by the collected evidence, return it."
            )
            if self.settings.benchmark_mode:
                forced_instruction += (
                    " Maximize correctness: pick the best-supported answer rather than abstaining."
                    " If your reasoning produced a numeric or symbolic answer, use it."
                    " Set status='answered' and confidence >= 0.8."
                    " NEVER abstain when you have any computed answer — even a partial one is better than nothing."
                )
                if choices:
                    forced_instruction += (
                        f" Your label.answer MUST be one of {choices} \u2014 a single letter, nothing else."
                    )
            else:
                forced_instruction += " Only set status='abstain' when no defensible final answer remains."
            forced_user = self._build_forced_final_user_prompt(
                base_user_prompt=base_user_prompt,
                steps=execution.steps,
                attached_images=execution.attached_images,
                instruction=forced_instruction,
            )
            parsed, raw_response = self._call_structured_model(
                model_cls=CandidateLabel,
                system_prompt=forced_system,
                user_prompt=forced_user,
                image_paths=self._model_image_paths(
                    base_images=execution.base_images,
                    current_attached=execution.attached_images,
                ),
                temperature=0.0,
            )
            execution.final_output = parsed

        candidate = self._finalize_candidate_label(execution.final_output, execution, choices=choices)  # type: ignore[arg-type]
        candidate, raw_response = self._repair_candidate_schema_if_needed(
            candidate=candidate,
            raw_response=raw_response,
            label_schema=label_schema,
            system_prompt=system_prompt,
            user_prompt=base_user_prompt,
            execution=execution,
        )
        candidate = self._recover_benchmark_answer_from_raw_text(
            candidate=candidate,
            raw_response=raw_response,
            label_schema=label_schema,
            execution=execution,
        )
        self._upsert_final_step(
            execution.steps,
            thought=f"Finalized {variant} output.",
            raw_model_text=raw_response["text"],
            final_payload=candidate.model_dump(mode="json"),
        )
        return AgentRun(
            role="solver",
            variant=variant,
            output=candidate.model_dump(mode="json"),
            tools_used=candidate.tools_used,
            steps=execution.steps,
            attached_images=execution.attached_images,
        )

    def run_verifier(
        self,
        *,
        question: str,
        image_paths: list[str],
        context: str | None,
        choices: list[str] | None,
        label_schema: dict[str, Any],
        candidate_label: dict[str, Any],
        variant: str = "independent_verifier",
    ) -> AgentRun:
        preflight = self._execute_preflight_tools(image_paths)
        system_prompt = self._build_verifier_system_prompt(label_schema)
        base_user_prompt = self._build_verifier_user_prompt(
            question=question,
            context=context,
            choices=choices,
            preflight=preflight,
            candidate_label=candidate_label,
            variant=variant,
        )
        execution = self._run_agent_loop(
            final_model_cls=VerificationResult,
            system_prompt=system_prompt,
            base_user_prompt=base_user_prompt,
            preflight=preflight,
            original_image_paths=image_paths,
            temperature=self.settings.verifier_temperature,
            max_steps=self.settings.max_verifier_steps,
        )

        raw_response = execution.last_response or {"text": ""}
        if execution.final_output is None:
            forced_system = (
                "You have exhausted the tool budget. Return exactly one VerificationResult JSON object and nothing else.\n"
                "Use only the collected evidence. If evidence is weak or incomplete, fail verification.\n"
                f"Target label schema:\n{json.dumps(label_schema, ensure_ascii=False, indent=2)}\n"
                f"VerificationResult schema:\n{json.dumps(VerificationResult.model_json_schema(), ensure_ascii=False, indent=2)}"
            )
            forced_user = self._build_forced_final_user_prompt(
                base_user_prompt=base_user_prompt,
                steps=execution.steps,
                attached_images=execution.attached_images,
                instruction=(
                    "No more tool calls are allowed. Produce the final VerificationResult now. "
                    "Fail verification only if support is materially incomplete or contradicted by the collected evidence."
                ),
            )
            parsed, raw_response = self._call_structured_model(
                model_cls=VerificationResult,
                system_prompt=forced_system,
                user_prompt=forced_user,
                image_paths=self._model_image_paths(
                    base_images=execution.base_images,
                    current_attached=execution.attached_images,
                ),
                temperature=0.0,
            )
            execution.final_output = parsed

        verifier = execution.final_output  # type: ignore[assignment]
        self._upsert_final_step(
            execution.steps,
            thought=f"Finalized verifier output for {variant}.",
            raw_model_text=raw_response["text"],
            final_payload=verifier.model_dump(mode="json"),
        )
        return AgentRun(
            role="verifier",
            variant=variant,
            output=verifier.model_dump(mode="json"),
            tools_used=execution.tool_names,
            steps=execution.steps,
            attached_images=execution.attached_images,
        )
