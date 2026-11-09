"""Ollama client with JSON-schema structured outputs and validation retries.

Small (1–2B) local models produce invalid JSON now and then. Every reply is
validated with Pydantic; on failure the validation error is sent back and
the model gets another try (``LLMConfig.max_retries``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from app.config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class LLMError(RuntimeError):
    """The model is unreachable or kept returning invalid output."""


class LLMClient(Protocol):
    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T: ...


@dataclass
class LLMStats:
    calls: int = 0
    retries: int = 0
    failures: int = 0

    @property
    def retry_rate(self) -> float:
        return self.retries / self.calls if self.calls else 0.0


def ensure_local(host: str, allow_remote: bool = False) -> None:
    name = urlparse(host if "://" in host else f"http://{host}").hostname or ""
    if name not in _LOCAL_HOSTS and not allow_remote:
        raise LLMError(
            f"refusing to send CV text to non-local LLM host {host!r}; "
            "set CVA_ALLOW_REMOTE_LLM=1 if this is really intended"
        )


class OllamaClient:
    def __init__(self, cfg: LLMConfig | None = None, client=None) -> None:
        self.cfg = cfg or LLMConfig()
        ensure_local(self.cfg.host, self.cfg.allow_remote)
        if client is None:
            import ollama

            client = ollama.Client(host=self.cfg.host, timeout=self.cfg.timeout_s)
        self._client = client
        self.stats = LLMStats()
        self._think = self.cfg.think

    def _options(self) -> dict:
        opts = {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx}
        if self.cfg.num_gpu is not None:
            opts["num_gpu"] = self.cfg.num_gpu
        return opts

    def available(self) -> bool:
        """True if the server answers and the configured model is pulled."""
        try:
            listed = self._client.list()
        except Exception as exc:
            log.info("Ollama not reachable at %s: %s", self.cfg.host, exc)
            return False
        models = getattr(listed, "models", None)
        if models is None and isinstance(listed, dict):
            models = listed.get("models", [])
        names = set()
        for m in models or []:
            name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
            if name:
                names.add(name)
                names.add(name.removesuffix(":latest"))
        return self.cfg.model in names

    def _chat(self, messages: list[dict], schema: dict):
        kwargs = dict(model=self.cfg.model, messages=messages, format=schema, options=self._options(),
                      keep_alive=self.cfg.keep_alive)
        if self._think is not None:
            kwargs["think"] = self._think
        try:
            return self._client.chat(**kwargs)
        except Exception as exc:
            if "think" in kwargs and "think" in str(exc).lower():
                # Model without a thinking switch: drop the flag for good.
                self._think = None
                kwargs.pop("think")
                return self._client.chat(**kwargs)
            raise

    def generate_json(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        json_schema = schema.model_json_schema()
        self.stats.calls += 1
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            if attempt:
                self.stats.retries += 1
            try:
                resp = self._chat(messages, json_schema)
            except Exception as exc:
                self.stats.failures += 1
                raise LLMError(f"Ollama request failed: {exc}") from exc
            message = resp["message"] if isinstance(resp, dict) else resp.message
            content = message["content"] if isinstance(message, dict) else message.content
            try:
                return schema.model_validate_json(content or "")
            except ValidationError as exc:
                last_error = exc
                log.debug("invalid JSON from model (attempt %s): %s", attempt + 1, exc)
                messages += [
                    {"role": "assistant", "content": content or ""},
                    {"role": "user", "content": f"That reply was not valid for the schema: {exc.errors()[:3]}. "
                                                "Reply again with JSON only."},
                ]
        self.stats.failures += 1
        raise LLMError(f"model returned invalid JSON {self.cfg.max_retries + 1} times: {last_error}")
