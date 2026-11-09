"""Test doubles for the Ollama client."""

from __future__ import annotations

import json


class FakeOllama:
    """Mimics ``ollama.Client`` with canned replies (dicts are JSON-encoded)."""

    def __init__(self, replies=(), models=("qwen3:1.7b",), fail_with: Exception | None = None):
        self.replies = list(replies)
        self.models = models
        self.fail_with = fail_with
        self.requests: list[dict] = []

    def list(self):
        if self.fail_with:
            raise self.fail_with
        return {"models": [{"model": m} for m in self.models]}

    def chat(self, **kwargs):
        if self.fail_with:
            raise self.fail_with
        self.requests.append(kwargs)
        reply = self.replies.pop(0) if self.replies else "{}"
        if callable(reply):
            reply = reply(kwargs)
        if not isinstance(reply, str):
            reply = json.dumps(reply)
        return {"message": {"role": "assistant", "content": reply}}
