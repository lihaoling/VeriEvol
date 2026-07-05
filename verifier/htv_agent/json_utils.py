from __future__ import annotations

import json
from typing import Any


_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')


def _fix_malformed_json_strings(text: str) -> str:
    """Fix common JSON string issues produced by LLMs.

    Handles two problems:
    1. Literal newlines/tabs inside JSON string values (not escaped).
    2. Invalid escape sequences like \\{ \\} \\t(imes) \\p(mod) from LaTeX —
       these need the backslash doubled to become valid JSON.
    """
    out: list[str] = []
    i = 0
    in_string = False
    while i < len(text):
        ch = text[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
        else:
            if ch == '\\':
                next_i = i + 1
                if next_i < len(text):
                    next_ch = text[next_i]
                    if next_ch in _VALID_JSON_ESCAPES:
                        # Valid JSON escape — keep as-is
                        out.append(ch)
                        out.append(next_ch)
                        i = next_i
                    else:
                        # Invalid escape (e.g., \{ \} \t when it's \times not \t tab)
                        # Double the backslash to make it a literal backslash in JSON
                        out.append('\\\\')
                        # Don't consume next_ch — it will be processed in next iteration
                else:
                    out.append(ch)
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch == '\n':
                out.append('\\n')
            elif ch == '\r':
                out.append('\\r')
            elif ch == '\t':
                out.append('\\t')
            else:
                out.append(ch)
        i += 1
    return ''.join(out)


def extract_json_objects(raw_text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(raw_text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)

    # Fallback: if strict parse only found inner/partial objects, try fixing
    # malformed JSON strings (unescaped newlines, invalid LaTeX escapes) and
    # look for larger top-level objects.
    if '{' in raw_text:
        fixed_text = _fix_malformed_json_strings(raw_text)
        if fixed_text != raw_text:
            fixed_objects: list[dict[str, Any]] = []
            for index, char in enumerate(fixed_text):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(fixed_text[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    fixed_objects.append(parsed)
            # Prefer the fixed parse if it found a larger/more complete object
            if fixed_objects:
                max_fixed = max(len(obj) for obj in fixed_objects)
                max_orig = max(len(obj) for obj in objects) if objects else 0
                if max_fixed > max_orig:
                    objects = fixed_objects

    return objects


def extract_first_json_object(raw_text: str) -> dict[str, Any]:
    objects = extract_json_objects(raw_text)
    if objects:
        return objects[0]
    raise ValueError("No JSON object found in model output.")


def canonicalize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."
