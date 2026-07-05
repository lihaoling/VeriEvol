from __future__ import annotations

import ast
import math
import operator
import re
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from jsonschema import validate
from PIL import Image, ImageOps

from .json_utils import truncate_text
from .schemas import ToolResult
from .settings import Settings


SAFE_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

SAFE_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_safe_eval(key): _safe_eval(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_BINARY_OPERATORS:
        return SAFE_BINARY_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_UNARY_OPERATORS:
        return SAFE_UNARY_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
        return SAFE_FUNCTIONS[node.func.id](*[_safe_eval(arg) for arg in node.args])
    raise ValueError("Expression contains unsupported Python syntax.")


@dataclass
class ToolRuntime:
    run_dir: Path
    settings: Settings

    @property
    def artifact_dir(self) -> Path:
        path = self.run_dir / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class LocalTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolRuntime], ToolResult]

    def invoke(self, arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        validate(instance=arguments, schema=self.input_schema)
        return self.handler(arguments, runtime)


class ToolRegistry:
    def __init__(self, tools: list[LocalTool]):
        self._tools = {tool.name: tool for tool in tools}

    def render_for_prompt(self) -> str:
        chunks = []
        for tool in self._tools.values():
            chunks.append(
                f"- {tool.name}: {tool.description}\n"
                f"  args_json_schema={tool.input_schema}"
            )
        return "\n".join(chunks)

    def invoke(self, tool_name: str, arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                ok=False,
                summary=f"Unknown tool: {tool_name}",
                error=f"Unknown tool: {tool_name}",
            )
        try:
            return self._tools[tool_name].invoke(arguments, runtime)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                ok=False,
                summary=truncate_text(
                    f"Tool {tool_name} failed with {type(exc).__name__}: {exc}",
                    runtime.settings.max_tool_output_chars,
                ),
                error=f"{type(exc).__name__}: {exc}",
            )

    def names(self) -> list[str]:
        return list(self._tools)


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return value or "artifact"


def _normalize_bbox(raw_bbox: list[int], image_size: tuple[int, int]) -> list[int]:
    if len(raw_bbox) != 4:
        raise ValueError("bbox must contain exactly four integers: [left, top, right, bottom].")

    width, height = image_size
    left, top, right, bottom = [int(value) for value in raw_bbox]
    left = max(0, min(left, width))
    right = max(0, min(right, width))
    top = max(0, min(top, height))
    bottom = max(0, min(bottom, height))
    if right <= left or bottom <= top:
        raise ValueError(
            f"bbox must define a positive-area region within the image. Received {[left, top, right, bottom]}."
        )
    return [left, top, right, bottom]


def _resolve_bbox(
    image_size: tuple[int, int],
    *,
    bbox: list[int] | None = None,
    relative_bbox: list[float] | None = None,
) -> list[int]:
    if bbox is not None:
        return _normalize_bbox([int(value) for value in bbox], image_size)
    if relative_bbox is None:
        width, height = image_size
        return [0, 0, width, height]
    if len(relative_bbox) != 4:
        raise ValueError("relative_bbox must contain exactly four numbers: [left, top, right, bottom].")
    width, height = image_size
    left = int(round(float(relative_bbox[0]) * width))
    top = int(round(float(relative_bbox[1]) * height))
    right = int(round(float(relative_bbox[2]) * width))
    bottom = int(round(float(relative_bbox[3]) * height))
    return _normalize_bbox([left, top, right, bottom], image_size)


def _load_region(image_path: str | Path, *, bbox: list[int] | None = None) -> tuple[Image.Image, list[int], str]:
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        resolved_bbox = _resolve_bbox(image.size, bbox=bbox)
        region = image.crop(tuple(resolved_bbox)) if resolved_bbox != [0, 0, *image.size] else image.copy()
    return region, resolved_bbox, image_path.name


def _rgb_stats(image: Image.Image) -> dict[str, float]:
    raw = image.convert("RGB").tobytes()
    if not raw:
        return {
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "brightness_mean": 0.0,
        }
    pixel_count = len(raw) // 3
    red_total = sum(raw[0::3])
    green_total = sum(raw[1::3])
    blue_total = sum(raw[2::3])
    brightness_mean = (red_total + green_total + blue_total) / (3 * pixel_count)
    return {
        "mean_r": round(red_total / pixel_count, 2),
        "mean_g": round(green_total / pixel_count, 2),
        "mean_b": round(blue_total / pixel_count, 2),
        "brightness_mean": round(brightness_mean, 2),
    }


def inspect_image(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = Path(arguments["image_path"])
    with Image.open(image_path) as image:
        width, height = image.size
        return ToolResult(
            ok=True,
            summary=f"Image {image_path.name}: size={width}x{height}, mode={image.mode}",
            data={"width": width, "height": height, "mode": image.mode},
        )


def crop_image(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = Path(arguments["image_path"])
    with Image.open(image_path) as image:
        bbox = _normalize_bbox(arguments["bbox"], image.size)
        label = _slugify(arguments.get("label", "crop"))
        crop_name = f"{image_path.stem}_{label}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"
        destination = runtime.artifact_dir / crop_name
        crop = image.crop(tuple(bbox))
        crop.save(destination)
    return ToolResult(
        ok=True,
        summary=f"Cropped region saved to {destination}",
        data={"bbox": bbox},
        artifacts=[str(destination)],
    )


def relative_crop_image(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = Path(arguments["image_path"])
    relative_bbox = [float(value) for value in arguments["relative_bbox"]]
    with Image.open(image_path) as image:
        bbox = _resolve_bbox(image.size, relative_bbox=relative_bbox)
        label = _slugify(arguments.get("label", "relative_crop"))
        crop_name = f"{image_path.stem}_{label}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.png"
        destination = runtime.artifact_dir / crop_name
        crop = image.crop(tuple(bbox))
        crop.save(destination)
    return ToolResult(
        ok=True,
        summary=(
            f"Relative crop saved to {destination} using relative_bbox={relative_bbox} "
            f"resolved_bbox={bbox}"
        ),
        data={"relative_bbox": relative_bbox, "bbox": bbox},
        artifacts=[str(destination)],
    )


def ocr_image(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return ToolResult(
            ok=False,
            summary="tesseract not found in PATH; OCR unavailable.",
            error="missing_tesseract",
        )

    image_path = Path(arguments["image_path"])
    bbox = arguments.get("bbox")
    resize_factor = float(arguments.get("resize_factor", 1.0))
    grayscale = bool(arguments.get("grayscale", False))
    invert = bool(arguments.get("invert", False))
    binarize_threshold = arguments.get("binarize_threshold")
    processed_image_path: Path | None = None
    source_for_ocr = image_path

    if resize_factor <= 0:
        raise ValueError("resize_factor must be > 0.")
    if binarize_threshold is not None:
        binarize_threshold = int(binarize_threshold)
        if not 0 <= binarize_threshold <= 255:
            raise ValueError("binarize_threshold must be between 0 and 255.")

    if bbox is not None or resize_factor != 1.0 or grayscale or invert or binarize_threshold is not None:
        label = _slugify(arguments.get("label", "ocr"))
        with Image.open(image_path) as image:
            working_image = image.convert("RGB")
            normalized_bbox = None
            if bbox is not None:
                normalized_bbox = _normalize_bbox([int(value) for value in bbox], working_image.size)
                working_image = working_image.crop(tuple(normalized_bbox))
            if grayscale or invert or binarize_threshold is not None:
                working_image = working_image.convert("L")
            if resize_factor != 1.0:
                new_width = max(1, int(round(working_image.width * resize_factor)))
                new_height = max(1, int(round(working_image.height * resize_factor)))
                working_image = working_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if binarize_threshold is not None:
                working_image = working_image.point(
                    lambda pixel: 255 if pixel >= binarize_threshold else 0,
                    mode="1",
                ).convert("L")
            if invert:
                if working_image.mode != "L":
                    working_image = working_image.convert("L")
                working_image = ImageOps.invert(working_image)

            suffix_bits = [label]
            if normalized_bbox is not None:
                suffix_bits.append("bbox_" + "_".join(str(value) for value in normalized_bbox))
            if resize_factor != 1.0:
                suffix_bits.append(f"scale_{str(resize_factor).replace('.', '_')}")
            if binarize_threshold is not None:
                suffix_bits.append(f"thr_{binarize_threshold}")
            if grayscale:
                suffix_bits.append("gray")
            if invert:
                suffix_bits.append("invert")
            processed_name = f"{image_path.stem}_{'_'.join(suffix_bits)}.png"
            processed_image_path = runtime.artifact_dir / processed_name
            working_image.save(processed_image_path)
            source_for_ocr = processed_image_path

    command = [tesseract, str(source_for_ocr), "stdout", "--psm", str(arguments.get("psm", 6))]
    lang = arguments.get("lang")
    if lang:
        command.extend(["-l", str(lang)])
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return ToolResult(
            ok=False,
            summary=truncate_text(process.stderr.strip() or "OCR failed.", runtime.settings.max_tool_output_chars),
            error=process.stderr.strip() or "ocr_failed",
        )
    text = process.stdout.strip()
    artifacts = [str(processed_image_path)] if processed_image_path else []
    return ToolResult(
        ok=True,
        summary=truncate_text(text or "(empty OCR result)", runtime.settings.max_tool_output_chars),
        data={
            "text": text,
            "source_image": str(source_for_ocr),
            "bbox": bbox,
            "resize_factor": resize_factor,
            "grayscale": grayscale,
            "invert": invert,
            "binarize_threshold": binarize_threshold,
        },
        artifacts=artifacts,
    )


def measure_region(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = arguments["image_path"]
    bbox = arguments.get("bbox")
    region, resolved_bbox, image_name = _load_region(image_path, bbox=bbox)
    gray = region.convert("L")
    pixels = gray.tobytes()
    if not pixels:
        return ToolResult(ok=False, summary="Selected region is empty.", error="empty_region")

    threshold = int(arguments.get("threshold", 200))
    dark_pixels = sum(1 for pixel in pixels if pixel <= threshold)
    pixel_count = len(pixels)
    dark_ratio = dark_pixels / pixel_count if pixel_count else 0.0
    mean_gray = sum(pixels) / pixel_count if pixel_count else 0.0
    rgb_stats = _rgb_stats(region)
    summary = (
        f"Region on {image_name} bbox={resolved_bbox}: size={region.width}x{region.height}, "
        f"dark_ratio={dark_ratio:.4f}, mean_gray={mean_gray:.2f}, brightness_mean={rgb_stats['brightness_mean']:.2f}"
    )
    return ToolResult(
        ok=True,
        summary=summary,
        data={
            "bbox": resolved_bbox,
            "width": region.width,
            "height": region.height,
            "pixel_count": pixel_count,
            "dark_pixels": dark_pixels,
            "dark_ratio": dark_ratio,
            "mean_gray": mean_gray,
            **rgb_stats,
        },
    )


def foreground_bounds(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = arguments["image_path"]
    bbox = arguments.get("bbox")
    threshold = int(arguments.get("threshold", 220))

    region, resolved_bbox, image_name = _load_region(image_path, bbox=bbox)
    gray = region.convert("L")
    width, height = gray.size
    pixel_count = width * height

    dark_pixel_count = 0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y in range(height):
        for x in range(width):
            if gray.getpixel((x, y)) > threshold:
                continue
            dark_pixel_count += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if dark_pixel_count == 0:
        return ToolResult(
            ok=True,
            summary=(
                f"Foreground bounds on {image_name} bbox={resolved_bbox}: "
                f"no pixels found below threshold={threshold}"
            ),
            data={
                "bbox": resolved_bbox,
                "threshold": threshold,
                "has_foreground": False,
                "dark_pixel_count": 0,
                "dark_ratio": 0.0,
            },
        )

    foreground_bbox_in_region = [min_x, min_y, max_x + 1, max_y + 1]
    foreground_bbox_in_image = [
        resolved_bbox[0] + min_x,
        resolved_bbox[1] + min_y,
        resolved_bbox[0] + max_x + 1,
        resolved_bbox[1] + max_y + 1,
    ]
    foreground_width = foreground_bbox_in_region[2] - foreground_bbox_in_region[0]
    foreground_height = foreground_bbox_in_region[3] - foreground_bbox_in_region[1]
    dark_ratio = dark_pixel_count / pixel_count if pixel_count else 0.0
    summary = (
        f"Foreground bounds on {image_name} bbox={resolved_bbox}: "
        f"foreground_bbox={foreground_bbox_in_image}, dark_ratio={dark_ratio:.4f}, "
        f"foreground_size={foreground_width}x{foreground_height}"
    )
    return ToolResult(
        ok=True,
        summary=summary,
        data={
            "bbox": resolved_bbox,
            "threshold": threshold,
            "has_foreground": True,
            "dark_pixel_count": dark_pixel_count,
            "dark_ratio": dark_ratio,
            "foreground_bbox_in_region": foreground_bbox_in_region,
            "foreground_bbox_in_image": foreground_bbox_in_image,
            "foreground_width": foreground_width,
            "foreground_height": foreground_height,
            "foreground_width_ratio": foreground_width / width if width else 0.0,
            "foreground_height_ratio": foreground_height / height if height else 0.0,
        },
    )


def projection_profile(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = arguments["image_path"]
    bbox = arguments.get("bbox")
    axis = str(arguments.get("axis", "x")).lower()
    threshold = int(arguments.get("threshold", 200))
    top_k = int(arguments.get("top_k", 5))

    region, resolved_bbox, image_name = _load_region(image_path, bbox=bbox)
    gray = region.convert("L")
    width, height = gray.size
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'.")

    if axis == "x":
        values = []
        for x in range(width):
            dark_count = 0
            for y in range(height):
                if gray.getpixel((x, y)) <= threshold:
                    dark_count += 1
            values.append(dark_count)
    else:
        values = []
        for y in range(height):
            dark_count = 0
            for x in range(width):
                if gray.getpixel((x, y)) <= threshold:
                    dark_count += 1
            values.append(dark_count)

    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=True)
    peaks = [
        {"index": index, "dark_pixels": dark_pixels}
        for index, dark_pixels in indexed[: max(1, top_k)]
    ]
    summary = (
        f"Projection profile on {image_name} bbox={resolved_bbox} axis={axis}: "
        f"max_dark_pixels={max(values) if values else 0}, peaks={peaks}"
    )
    return ToolResult(
        ok=True,
        summary=truncate_text(summary, runtime.settings.max_tool_output_chars),
        data={
            "bbox": resolved_bbox,
            "axis": axis,
            "threshold": threshold,
            "values": values,
            "peaks": peaks,
        },
    )


def connected_components(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = arguments["image_path"]
    bbox = arguments.get("bbox")
    threshold = int(arguments.get("threshold", 200))
    min_area = int(arguments.get("min_area", 1))
    connectivity = int(arguments.get("connectivity", 8))
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8.")

    region, resolved_bbox, image_name = _load_region(image_path, bbox=bbox)
    gray = region.convert("L")
    width, height = gray.size
    visited = [[False for _ in range(width)] for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    components: list[dict[str, Any]] = []

    for y in range(height):
        for x in range(width):
            if visited[y][x] or gray.getpixel((x, y)) > threshold:
                continue
            queue = deque([(x, y)])
            visited[y][x] = True
            area = 0
            min_x = max_x = x
            min_y = max_y = y

            while queue:
                current_x, current_y = queue.popleft()
                area += 1
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                for dx, dy in directions:
                    next_x = current_x + dx
                    next_y = current_y + dy
                    if not (0 <= next_x < width and 0 <= next_y < height):
                        continue
                    if visited[next_y][next_x] or gray.getpixel((next_x, next_y)) > threshold:
                        continue
                    visited[next_y][next_x] = True
                    queue.append((next_x, next_y))

            if area < min_area:
                continue
            component_bbox = [min_x, min_y, max_x + 1, max_y + 1]
            components.append(
                {
                    "bbox": component_bbox,
                    "area": area,
                    "width": component_bbox[2] - component_bbox[0],
                    "height": component_bbox[3] - component_bbox[1],
                }
            )

    components.sort(key=lambda item: item["area"], reverse=True)
    summary = (
        f"Connected components on {image_name} bbox={resolved_bbox}: count={len(components)}, "
        f"largest={components[0] if components else None}"
    )
    return ToolResult(
        ok=True,
        summary=truncate_text(summary, runtime.settings.max_tool_output_chars),
        data={
            "bbox": resolved_bbox,
            "threshold": threshold,
            "min_area": min_area,
            "connectivity": connectivity,
            "component_count": len(components),
            "components": components,
        },
    )


def grid_region_scan(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    image_path = arguments["image_path"]
    bbox = arguments.get("bbox")
    threshold = int(arguments.get("threshold", 220))
    rows = int(arguments.get("rows", 3))
    cols = int(arguments.get("cols", 3))
    top_k = int(arguments.get("top_k", min(rows * cols, 5)))
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers.")

    region, resolved_bbox, image_name = _load_region(image_path, bbox=bbox)
    gray = region.convert("L")
    width, height = gray.size
    if rows > height or cols > width:
        raise ValueError("rows and cols must not exceed the region dimensions in pixels.")

    cells: list[dict[str, Any]] = []
    for row in range(rows):
        top = (row * height) // rows
        bottom = ((row + 1) * height) // rows
        if bottom <= top:
            bottom = min(height, top + 1)
        for col in range(cols):
            left = (col * width) // cols
            right = ((col + 1) * width) // cols
            if right <= left:
                right = min(width, left + 1)

            pixel_count = (right - left) * (bottom - top)
            dark_pixels = 0
            gray_sum = 0
            for y in range(top, bottom):
                for x in range(left, right):
                    value = gray.getpixel((x, y))
                    gray_sum += value
                    if value <= threshold:
                        dark_pixels += 1

            mean_gray = gray_sum / pixel_count if pixel_count else 0.0
            dark_ratio = dark_pixels / pixel_count if pixel_count else 0.0
            cell_bbox = [
                resolved_bbox[0] + left,
                resolved_bbox[1] + top,
                resolved_bbox[0] + right,
                resolved_bbox[1] + bottom,
            ]
            cells.append(
                {
                    "row": row,
                    "col": col,
                    "bbox": cell_bbox,
                    "pixel_count": pixel_count,
                    "dark_pixels": dark_pixels,
                    "dark_ratio": dark_ratio,
                    "mean_gray": mean_gray,
                }
            )

    ranked_cells = sorted(
        cells,
        key=lambda item: (item["dark_ratio"], item["dark_pixels"], -item["mean_gray"]),
        reverse=True,
    )
    darkest_cells = [
        {
            "row": cell["row"],
            "col": cell["col"],
            "bbox": cell["bbox"],
            "dark_ratio": round(cell["dark_ratio"], 4),
            "dark_pixels": cell["dark_pixels"],
        }
        for cell in ranked_cells[: max(1, min(top_k, len(ranked_cells)))]
    ]
    summary = (
        f"Grid scan on {image_name} bbox={resolved_bbox} rows={rows} cols={cols}: "
        f"darkest_cells={darkest_cells}"
    )
    return ToolResult(
        ok=True,
        summary=truncate_text(summary, runtime.settings.max_tool_output_chars),
        data={
            "bbox": resolved_bbox,
            "threshold": threshold,
            "rows": rows,
            "cols": cols,
            "cells": cells,
            "darkest_cells": darkest_cells,
        },
    )


def math_eval(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    expression = arguments["expression"]
    parsed = ast.parse(expression, mode="eval")
    result = _safe_eval(parsed)
    return ToolResult(
        ok=True,
        summary=f"{expression} = {result}",
        data={"result": result},
    )


def python_exec(arguments: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
    code = arguments["code"]
    global_scope = {
        "__builtins__": {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "round": round,
            "range": range,
        },
        "math": math,
    }
    local_scope: dict[str, Any] = {}
    exec(code, global_scope, local_scope)
    result = local_scope.get("result")
    return ToolResult(
        ok=True,
        summary=truncate_text(f"python result={result!r}", runtime.settings.max_tool_output_chars),
        data={"result": result},
    )


def build_default_registry(settings: Settings) -> ToolRegistry:
    tools = [
        LocalTool(
            name="inspect_image",
            description="Read image size and metadata for planning crops and coordinate checks.",
            input_schema={
                "type": "object",
                "properties": {"image_path": {"type": "string"}},
                "required": ["image_path"],
                "additionalProperties": False,
            },
            handler=inspect_image,
        ),
    ]
    if settings.enable_crop_tool:
        tools.append(
            LocalTool(
                name="crop_image",
                description="Crop a rectangular region from an image and return a new image artifact path.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "label": {"type": "string"},
                    },
                    "required": ["image_path", "bbox"],
                    "additionalProperties": False,
                },
                handler=crop_image,
            )
        )
    if settings.enable_relative_crop_tool:
        tools.append(
            LocalTool(
                name="relative_crop_image",
                description="Crop a region using normalized coordinates [left, top, right, bottom] in the range [0, 1].",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "relative_bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "label": {"type": "string"},
                    },
                    "required": ["image_path", "relative_bbox"],
                    "additionalProperties": False,
                },
                handler=relative_crop_image,
            )
        )
    if settings.enable_ocr_tool:
        tools.append(
            LocalTool(
                name="ocr_image",
                description="Run local OCR with tesseract on the image or cropped region.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "label": {"type": "string"},
                        "psm": {"type": "integer"},
                        "lang": {"type": "string"},
                        "resize_factor": {"type": "number", "exclusiveMinimum": 0},
                        "grayscale": {"type": "boolean"},
                        "invert": {"type": "boolean"},
                        "binarize_threshold": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 255,
                        },
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
                handler=ocr_image,
            )
        )
    if settings.enable_measure_region_tool:
        tools.append(
            LocalTool(
                name="measure_region",
                description="Measure dark-pixel density, brightness, and mean RGB values inside a region. Useful for comparing bars, filled areas, or marked options.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "threshold": {"type": "integer", "minimum": 0, "maximum": 255},
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
                handler=measure_region,
            )
        )
    if settings.enable_projection_profile_tool:
        tools.append(
            LocalTool(
                name="projection_profile",
                description="Compute 1D dark-pixel projections across x or y axis to localize bars, lines, or dense text bands.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "axis": {"type": "string", "enum": ["x", "y"]},
                        "threshold": {"type": "integer", "minimum": 0, "maximum": 255},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
                handler=projection_profile,
            )
        )
    if settings.enable_connected_components_tool:
        tools.append(
            LocalTool(
                name="connected_components",
                description="Detect dark connected components in a region. Useful for counting blobs, locating check marks, digits, or chart bars.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "threshold": {"type": "integer", "minimum": 0, "maximum": 255},
                        "min_area": {"type": "integer", "minimum": 1},
                        "connectivity": {"type": "integer", "enum": [4, 8]},
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
                handler=connected_components,
            )
        )
    if settings.enable_foreground_bounds_tool:
        tools.append(
            LocalTool(
                name="foreground_bounds",
                description="Find the tight bounding box of dark foreground pixels in a region. Useful for estimating bar heights, mark extents, or text bounds.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "threshold": {"type": "integer", "minimum": 0, "maximum": 255},
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
                handler=foreground_bounds,
            )
        )
    if settings.enable_grid_region_scan_tool:
        tools.append(
            LocalTool(
                name="grid_region_scan",
                description="Split a region into a grid and measure dark-pixel density per cell. Useful for locating dense answer areas, dominant bars, or marked options.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "threshold": {"type": "integer", "minimum": 0, "maximum": 255},
                        "rows": {"type": "integer", "minimum": 1, "maximum": 20},
                        "cols": {"type": "integer", "minimum": 1, "maximum": 20},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                },
                handler=grid_region_scan,
            )
        )
    if settings.enable_math_tool:
        tools.append(
            LocalTool(
                name="math_eval",
                description="Evaluate a pure arithmetic expression with math helpers like sqrt, log, sin, cos, pi.",
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
                handler=math_eval,
            )
        )
    if settings.enable_python_exec_tool:
        tools.append(
            LocalTool(
                name="python_exec",
                description="Run trusted local Python code. Use only in trusted environments.",
                input_schema={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
                handler=python_exec,
            )
        )
    return ToolRegistry(tools)
