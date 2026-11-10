"""Central configuration: model names, thresholds and limits.

Values can be overridden with environment variables prefixed ``CVA_``
(for example ``CVA_LLM_MODEL=gemma3:1b``). Model names live here, not in
code, so the eval bake-off (PLAN.md §7) can swap them freely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"CVA_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class IngestionConfig:
    # A page with fewer extracted characters than this is treated as image-only.
    min_chars_per_page: int = 40
    # Render resolution for OCR of scanned pages.
    ocr_dpi: int = 300
    # Fraction of page height treated as header / footer band.
    margin_band: float = 0.08
    # Minimum horizontal whitespace (pt) between two text columns.
    min_gutter_width: float = 12.0
    max_pages_warning: int = 3


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))
    host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:1.7b"))
    # Small context keeps the KV cache inside the 2 GB VRAM budget (PLAN.md §4).
    num_ctx: int = field(default_factory=lambda: _env_int("LLM_NUM_CTX", 2048))
    temperature: float = 0.0
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))
    timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 60.0))
    # Layers offloaded to the GPU (None = Ollama decides). Tune to stay < 2 GB VRAM.
    num_gpu: int | None = field(default_factory=lambda: int(v) if (v := _env("LLM_NUM_GPU", "")) else None)
    # Qwen3-style models "think" by default; that wastes tokens on tiny extraction tasks.
    think: bool | None = False
    keep_alive: str = field(default_factory=lambda: _env("LLM_KEEP_ALIVE", "5m"))
    # The CV must not leave the machine: refuse non-localhost hosts unless explicitly allowed.
    allow_remote: bool = field(default_factory=lambda: _env_bool("ALLOW_REMOTE_LLM", False))
    # LLM-proposed values must fuzzy-match the source block at least this well.
    min_grounding_score: float = 90.0


@dataclass(frozen=True)
class NERConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("NER_ENABLED", False))
    model: str = field(default_factory=lambda: _env("NER_MODEL", "urchade/gliner_small-v2.1"))
    threshold: float = field(default_factory=lambda: _env_float("NER_THRESHOLD", 0.5))
    labels: tuple[str, ...] = (
        "skill",
        "programming language",
        "software tool",
        "degree",
        "certification",
        "organization",
    )


@dataclass(frozen=True)
class PrivacyConfig:
    # Optional Presidio + spaCy pass (CPU) on top of the rule-based detectors.
    presidio_enabled: bool = field(default_factory=lambda: _env_bool("PRESIDIO_ENABLED", False))
    spacy_model: str = field(default_factory=lambda: _env("SPACY_MODEL", "en_core_web_lg"))
    presidio_threshold: float = field(default_factory=lambda: _env_float("PRESIDIO_THRESHOLD", 0.6))
    # An image at least this big (pt) with a portrait/square shape on page 1 is treated as a photo.
    photo_min_side: float = 40.0
    photo_max_side: float = 300.0
    # Jobs that started this many years ago or earlier are reported as an age signal.
    old_experience_years: int = 20


@dataclass(frozen=True)
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def get_config() -> Config:
    return Config()
