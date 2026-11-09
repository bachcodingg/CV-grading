"""Reading-order reconstruction for multi-column PDF pages.

PDF text comes as positioned blocks. Sorting them purely top-to-bottom
interleaves the columns of a two-column CV ("Skills" sidebar lines end up
between job bullets). We look for a vertical gutter, then read each column
top-to-bottom. Blocks that span the gutter (a centred name header, a
full-width section heading) split the page into horizontal bands.

A right-aligned date next to a job title must *not* count as a column, so a
split is only accepted when both sides carry real text and the right side
is not mostly dates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.document import BBox

_DATEISH = re.compile(
    r"^\s*(?:(?:\d{1,2}[./])?\d{4}|[A-Za-z]{3,9}\.?\s+\d{4}|present|current|now|today|heute)"
    r"(?:\s*(?:-|–|—|to|bis)\s*(?:(?:\d{1,2}[./])?\d{4}|[A-Za-z]{3,9}\.?\s+\d{4}|present|current|now|today|heute))?\s*$",
    re.IGNORECASE,
)


@dataclass
class LayoutBlock:
    bbox: BBox
    text: str

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]


def _chars(blocks: list[LayoutBlock]) -> int:
    return sum(len(b.text.strip()) for b in blocks)


def _is_dateish(block: LayoutBlock) -> bool:
    """A short block like "Jan 2020 – Present" or "Berlin, DE" next to an entry."""
    lines = [ln for ln in block.text.splitlines() if ln.strip()]
    if not lines or len(lines) > 2:
        return False
    return any(_DATEISH.match(ln) for ln in lines) and all(
        _DATEISH.match(ln) or len(ln.split()) <= 3 for ln in lines
    )


def _find_gutter(
    blocks: list[LayoutBlock], min_gutter: float
) -> tuple[list[LayoutBlock], list[LayoutBlock], list[LayoutBlock]] | None:
    """Return (left, right, spanning) for the best gutter, or None."""
    if len(blocks) < 4:
        return None
    lo = min(b.x0 for b in blocks)
    hi = max(b.x1 for b in blocks)
    width = hi - lo
    if width <= 0:
        return None
    total = _chars(blocks)
    step = 2.0

    # Contiguous runs of x positions that induce the same left/right partition.
    runs: list[tuple[float, float, frozenset[int]]] = []
    x = lo + 0.12 * width
    while x <= lo + 0.88 * width:
        crossing = frozenset(i for i, b in enumerate(blocks) if b.x0 < x < b.x1)
        if _chars([blocks[i] for i in crossing]) <= 0.25 * total:
            if runs and runs[-1][2] == crossing and abs(runs[-1][1] - (x - step)) < 1e-6:
                runs[-1] = (runs[-1][0], x, crossing)
            else:
                runs.append((x, x, crossing))
        x += step

    best: tuple[float, tuple[list, list, list]] | None = None
    for start, end, crossing in runs:
        gap = end - start + step
        if gap < min_gutter:
            continue
        mid = (start + end) / 2
        spanning = [b for i, b in enumerate(blocks) if i in crossing]
        left = [b for i, b in enumerate(blocks) if i not in crossing and b.x1 <= mid]
        right = [b for i, b in enumerate(blocks) if i not in crossing and b.x0 >= mid]
        if len(left) < 2 or len(right) < 2:
            continue
        lc, rc = _chars(left), _chars(right)
        if min(lc, rc) < 0.12 * total:
            continue
        # Right-aligned dates / locations next to entries are rows, not a column.
        if sum(_is_dateish(b) for b in right) >= 0.6 * len(right):
            continue
        if sum(_is_dateish(b) for b in left) >= 0.6 * len(left):
            continue
        # Columns must sit side by side, not one above the other.
        overlap = min(max(b.y1 for b in left), max(b.y1 for b in right)) - max(
            min(b.y0 for b in left), min(b.y0 for b in right)
        )
        if overlap <= 0:
            continue
        score = gap
        if best is None or score > best[0]:
            best = (score, (left, right, spanning))
    return best[1] if best else None


def order_blocks(
    blocks: list[LayoutBlock], min_gutter: float = 12.0, _depth: int = 0
) -> list[tuple[LayoutBlock, int]]:
    """Return blocks in reading order, each paired with a column index."""
    simple = sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))
    if _depth >= 2:
        return [(b, 0) for b in simple]
    split = _find_gutter(blocks, min_gutter)
    if split is None:
        return [(b, 0) for b in simple]

    left, right, spanning = split
    left_ids = {id(b) for b in left}
    breakers = sorted(spanning, key=lambda b: b.y0)
    result: list[tuple[LayoutBlock, int]] = []

    def flush(band: list[LayoutBlock]) -> None:
        lband = [b for b in band if id(b) in left_ids]
        rband = [b for b in band if id(b) not in left_ids]
        lres = order_blocks(lband, min_gutter, _depth + 1)
        rres = order_blocks(rband, min_gutter, _depth + 1)
        ncols_left = 1 + max((c for _, c in lres), default=0)
        result.extend(lres)
        result.extend((b, c + ncols_left) for b, c in rres)

    columned = sorted(left + right, key=lambda b: b.y0)
    i = 0
    for brk in breakers:
        band = []
        while i < len(columned) and columned[i].y0 < brk.y0:
            band.append(columned[i])
            i += 1
        flush(band)
        result.append((brk, 0))
    flush(columned[i:])
    return result


def column_count(ordered: list[tuple[LayoutBlock, int]]) -> int:
    return 1 + max((c for _, c in ordered), default=0)
