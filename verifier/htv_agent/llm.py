from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from .settings import Settings


def _path_to_data_url(path: str | Path) -> str:
    file_path = Path(path)
    mime_type, _ = mimetypes.guess_type(file_path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _image_url(image_path: str) -> str:
    if image_path.startswith(("data:", "http://", "https://")):
        return image_path
    return _path_to_data_url(image_path)


class ModelClientProtocol(Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        ...


class OpenAICompatibleClient:
    """Thin wrapper over a standard OpenAI-compatible chat-completions endpoint.

    The HTV-Agent solvers, verifiers, and decider all talk to the model through
    this single interface. Only the model call leaves the machine; every tool
    runs locally.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.openai_api_key:
            raise ValueError(
                "HTV-Agent requires an OpenAI-compatible key. "
                "Set HTV_AGENT_OPENAI_API_KEY (and HTV_AGENT_OPENAI_BASE_URL)."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
            "timeout": settings.model_timeout_seconds,
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs)
        self.model = settings.model

    def _extract_text(self, completion: Any) -> str | None:
        if not getattr(completion, "choices", None):
            return None
        message = completion.choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            # Some reasoning endpoints only populate reasoning_content.
            content = getattr(message, "reasoning_content", None)
        if isinstance(content, list):
            texts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("text")
            ]
            content = "\n".join(text for text in texts if text)
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float,
    ) -> dict[str, Any]:
        prompt_text = f"{system_prompt}\n\n{user_prompt}".strip()
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        for image_path in image_paths[-self.settings.max_attached_images :]:
            content.append(
                {"type": "image_url", "image_url": {"url": _image_url(image_path)}}
            )

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if self.settings.reasoning_effort:
            request["reasoning_effort"] = self.settings.reasoning_effort

        completion = self.client.chat.completions.create(**request)
        text = self._extract_text(completion)
        if not text:
            raise ValueError("Could not extract text answer from model response.")
        return {
            "text": text,
            "raw": completion.model_dump() if hasattr(completion, "model_dump") else {},
        }


def create_model_client(settings: Settings) -> ModelClientProtocol:
    return OpenAICompatibleClient(settings)
