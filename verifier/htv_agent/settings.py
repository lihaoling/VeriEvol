from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """HTV-Agent configuration.

    All model access goes through a standard OpenAI-compatible chat-completions
    endpoint. Credentials and the endpoint are read from the environment (see
    ``.env.example``); no secrets are stored in the repository.
    """

    model_config = SettingsConfigDict(
        env_prefix="HTV_AGENT_",
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model backend (OpenAI-compatible) ---
    openai_api_key: str = Field(default="", description="OpenAI-compatible API key.")
    openai_base_url: str = Field(
        default="",
        description="OpenAI-compatible base URL, e.g. http://<VLLM_ENDPOINT>/v1.",
    )
    model: str = Field(
        default="gemini-3.5-flash",
        description="Model name passed to the chat-completions endpoint.",
    )
    max_tokens: int = Field(default=8192, ge=1, description="Max output tokens per call.")
    reasoning_effort: str = Field(
        default="",
        description="Optional reasoning_effort forwarded to the endpoint when non-empty.",
    )
    model_timeout_seconds: int = Field(default=600, ge=5)

    # --- Solver / verifier sampling temperatures (paper Section: Solvers) ---
    solver_primary_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    solver_secondary_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    verifier_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # --- Agent loop budgets ---
    max_solver_steps: int = Field(default=16, ge=1, le=40)
    max_verifier_steps: int = Field(default=10, ge=1, le=40)
    max_format_retries: int = Field(default=2, ge=0, le=5)
    max_attached_images: int = Field(default=4, ge=1, le=16)
    max_tool_output_chars: int = Field(default=2500, ge=256)
    use_checkpoints: bool = True
    preflight_profile: str = Field(
        default="full",
        description="Automatic preflight tool policy: full, focused, or none.",
    )
    benchmark_mode: bool = False
    enable_pre_think: bool = Field(
        default=False,
        description="When True, run a raw LLM pre-think before the tool loop and route simple questions to skip tools.",
    )

    # --- Deterministic acceptance gate thresholds (paper: deterministic gate) ---
    min_solver_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_verifier_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    require_consensus: bool = True
    require_tool_use: bool = True
    require_evidence: bool = True

    # --- Evidence-channel tools (programmatic + visual) ---
    enable_ocr_tool: bool = True
    enable_math_tool: bool = True
    enable_crop_tool: bool = True
    enable_relative_crop_tool: bool = True
    enable_measure_region_tool: bool = True
    enable_projection_profile_tool: bool = True
    enable_connected_components_tool: bool = True
    enable_foreground_bounds_tool: bool = True
    enable_grid_region_scan_tool: bool = True
    enable_python_exec_tool: bool = False

    max_ocr_retries_per_region: int = Field(default=2, ge=1, le=10)

    checkpoint_db: Path = Field(default=Path(".state/checkpoints.sqlite"))
    output_dir: Path = Field(default=Path("outputs"))

    def ensure_directories(self) -> None:
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
