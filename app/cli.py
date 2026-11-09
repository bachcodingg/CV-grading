"""Command line entry point.

    cv-analyzer check CV.pdf            parse-check report + parser view
    cv-analyzer structure CV.docx       structured CV (add --json for machine output)
    cv-analyzer analyze CV.pdf          both

Everything runs locally. CV text is printed to your terminal only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from app.extraction import structure_cv
from app.extraction.schemas import Candidate
from app.ingestion import load_document
from app.ingestion.parse_check import check_document


def _fmt_range(e) -> str:
    start = str(e.start) if e.start else "?"
    end = "present" if e.current else (str(e.end) if e.end else "?")
    return f"{start} – {end}"


def format_candidate(c: Candidate) -> str:
    out = []
    ct = c.contact
    out.append(f"Name: {ct.name or '—'}   Email: {ct.email or '—'}   Phone: {ct.phone or '—'}")
    if ct.location or ct.links:
        out.append(f"Location: {ct.location or '—'}   Links: {', '.join(ct.links) or '—'}")
    if c.summary:
        out.append(f"\nSummary: {c.summary}")
    if c.years_of_experience is not None:
        out.append(f"\nYears of experience (overlaps merged): {c.years_of_experience}")

    def mark(entry, name):
        src = entry.provenance.get(name)
        flag = "?" if name in entry.uncertain else ""
        return f"{flag}[{src}]" if src and src != "rules" else flag

    if c.experience:
        out.append("\nExperience:")
        for e in c.experience:
            out.append(f"  • {e.title or '?'}{mark(e, 'title')} @ {e.org or '?'}{mark(e, 'org')}"
                       f"  ({_fmt_range(e)}{', ' + e.location if e.location else ''})")
            for b in e.bullets[:4]:
                out.append(f"      - {b.text}")
            if len(e.bullets) > 4:
                out.append(f"      … {len(e.bullets) - 4} more")
    if c.education:
        out.append("\nEducation:")
        for e in c.education:
            grade = f", grade {e.grade}" if e.grade else ""
            field = f" in {e.field}" if e.field else ""
            out.append(f"  • {e.degree or '?'}{field} — {e.institution or '?'}{mark(e, 'institution')}"
                       f"  ({_fmt_range(e)}{grade})")
    if c.projects:
        out.append("\nProjects:")
        for p in c.projects:
            tech = f" [{', '.join(p.technologies)}]" if p.technologies else ""
            out.append(f"  • {p.name or '?'}{tech}")
    if c.skills:
        listed = [s.name for s in c.skills if s.provenance.get("name") != "ner"]
        mentioned = [s.name for s in c.skills if s.provenance.get("name") == "ner"]
        out.append(f"\nSkills: {', '.join(listed) or '—'}")
        if mentioned:
            out.append(f"Mentioned elsewhere (NER): {', '.join(mentioned)}")
    if c.languages:
        out.append("Languages: " + ", ".join(f"{l.name}" + (f" ({l.level})" if l.level else "") for l in c.languages))
    if c.certifications:
        out.append("Certifications: " + "; ".join(
            x.name + (f" — {x.issuer}" if x.issuer else "") + (f" ({x.date})" if x.date else "")
            for x in c.certifications))
    if c.publications:
        out.append(f"Publications: {len(c.publications)}")
    for o in c.other_sections:
        out.append(f"Other section: {o.heading}")
    if c.warnings:
        out.append("\nStructurer warnings:")
        out += [f"  ! {w}" for w in c.warnings]
    out.append("\n(? = guessed by rules, [ner]/[llm] = filled by a model)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cv-analyzer", description="Local CV parse-check and structurer.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("check", "parse-check report"), ("structure", "structured CV"),
                        ("analyze", "parse-check + structured CV")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("cv", type=Path, help="CV file (PDF or DOCX)")
        p.add_argument("--json", action="store_true", help="print JSON instead of text")
        if name != "structure":
            p.add_argument("--no-text", action="store_true", help="hide the extracted parser view")
        if name != "check":
            p.add_argument("--ner", action="store_true", help="enable GLiNER enrichment (CPU)")
            p.add_argument("--no-llm", action="store_true", help="skip local LLM gap-filling")
            p.add_argument("--today", type=date.fromisoformat, help="reference date for 'present' (YYYY-MM-DD)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING)

    try:
        doc = load_document(args.cv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result: dict = {}
    text_out: list[str] = []
    if args.command in ("check", "analyze"):
        report = check_document(doc)
        result["parse_check"] = report.to_dict()
        text_out.append(report.format(show_text=not args.no_text))
    if args.command in ("structure", "analyze"):
        cand = structure_cv(doc, use_ner=args.ner or None, use_llm=False if args.no_llm else None,
                            today=args.today)
        result["candidate"] = cand.model_dump(mode="json")
        text_out.append(format_candidate(cand))

    if args.json:
        if args.command in ("check", "structure"):
            result = next(iter(result.values()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n\n".join(text_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
