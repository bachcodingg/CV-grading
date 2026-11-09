"""Regexes shared by parse-check and the CV structurer."""

from __future__ import annotations

import re

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# International or national numbers with at least 7 digits, e.g. +49 151 2345 6789,
# (089) 123-4567, 0151/23456789. Years like "2016 – 2018" must not match.
PHONE = re.compile(
    r"(?<![\w/.-])(?:(?:\+|00)\d{1,4}[\s./-]?)?(?:\(\d{1,5}\)[\s./-]?)?\d{2,5}(?:[\s./-]?\d{2,5}){1,4}(?![\w/-]|\.\d)"
)

URL = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"(?:linkedin\.com/in/[\w-]+|github\.com/[\w-]+|gitlab\.com/[\w-]+|[\w-]+\.(?:dev|io|me|com|de|org|net)(?:/[\w./-]*)?)",
    re.IGNORECASE,
)

# Private Use Area code points: icon fonts (Font Awesome, Material Icons...).
PRIVATE_USE = re.compile(r"[-]")

# Signs that a font had no usable Unicode mapping.
GARBLED = re.compile(r"�|\(cid:\d+\)|[\x00-\x08\x0b\x0c\x0e-\x1f]")


def find_phones(text: str) -> list[re.Match]:
    """PHONE matches that contain at least 7 digits and are not a year range."""
    out = []
    for m in PHONE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) < 7 or len(digits) > 15:
            continue
        groups = re.findall(r"\d+", m.group())
        if all(re.fullmatch(r"(19|20)\d{2}", g) for g in groups):  # "2016 2018", years only
            continue
        out.append(m)
    return out
