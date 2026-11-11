"""Step 5 of the matcher: the small local LLM judges one requirement per call (PLAN.md §3 [5]).

The model sees the requirement, up to three numbered CV excerpts (redacted
text) and the facts computed in code, and returns a label plus the numbers
of the excerpts that support it. Picking numbers is much easier for a 1–2B
model than quoting, and it cannot produce a quote that is not in the CV:
the text shown to the user comes from the chosen chunks.

Labels map to the 0–4 rating in code (``LABEL_RATING``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.extraction.schemas import Requirement
from app.llm.client import LLMClient
from app.llm.prompts import JUDGE, JUDGE_SYSTEM
from app.matching.schemas import Chunk, Label


class JudgeReply(BaseModel):
    label: Label
    evidence: list[int] = Field(default_factory=list)
    reason: str = Field(default="", max_length=400)


@dataclass
class Verdict:
    label: Label
    chunk_ids: list[int]  # chosen excerpts, mapped back to chunk IDs
    reason: str
    dropped: list[int] = field(default_factory=list)  # excerpt numbers the model made up


def _excerpt(i: int, chunk: Chunk) -> str:
    where = f" (from: {chunk.context})" if chunk.context and chunk.kind in ("bullet", "project", "education") else ""
    return f'[{i}] "{chunk.text}"{where}'


def build_prompt(req: Requirement, chunks: list[Chunk], facts: list[str]) -> str:
    excerpts = "\n".join(_excerpt(i, c) for i, c in enumerate(chunks, 1)) or "(no excerpts found)"
    fact_text = "".join(f"Fact: {f}\n" for f in facts)
    return JUDGE.format(requirement=req.text, excerpts=excerpts, facts=fact_text)


def judge(req: Requirement, chunks: list[Chunk], llm: LLMClient, facts: list[str] | None = None) -> Verdict:
    """One LLM call. Raises ``LLMError`` if the model is unreachable or keeps returning invalid JSON."""
    reply = llm.generate_json(build_prompt(req, chunks, facts or []), JudgeReply, system=JUDGE_SYSTEM)
    ids, dropped = [], []
    for n in reply.evidence:
        if 1 <= n <= len(chunks):
            if chunks[n - 1].id not in ids:
                ids.append(chunks[n - 1].id)
        else:
            dropped.append(n)
    if reply.label == "none":
        ids = []
    return Verdict(label=reply.label, chunk_ids=ids, reason=" ".join(reply.reason.split()), dropped=dropped)
