"""Command line entry point.

    cv-analyzer check CV.pdf            parse-check report + parser view
    cv-analyzer structure CV.docx       structured CV (add --json for machine output)
    cv-analyzer privacy CV.pdf          bias-exposure scan + the redacted copy the grader sees
    cv-analyzer analyze CV.pdf          all of the above
    cv-analyzer jd posting.txt          requirements from a job posting (use - for stdin)
    cv-analyzer match CV.pdf post.txt   requirement-by-requirement evidence for that posting

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
from app.extraction.jd_parser import format_posting, parse_job_posting
from app.extraction.schemas import Candidate
from app.ingestion import load_document
from app.ingestion.parse_check import check_document
from app.matching import format_matches, match_requirements
from app.privacy import analyze_privacy


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
    parser = argparse.ArgumentParser(prog="cv-analyzer", description="Local, privacy-first CV analyzer.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (("check", "parse-check report"), ("structure", "structured CV"),
                        ("privacy", "bias-exposure scan + redacted copy"),
                        ("analyze", "parse-check + structured CV + bias-exposure scan")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("cv", type=Path, help="CV file (PDF or DOCX)")
        p.add_argument("--json", action="store_true", help="print JSON instead of text")
        if name != "structure":
            p.add_argument("--no-text", action="store_true", help="hide the extracted / redacted text")
        if name in ("privacy", "analyze"):
            p.add_argument("--presidio", action="store_true", help="also run Presidio + spaCy (CPU)")
        if name != "check":
            p.add_argument("--ner", action="store_true", help="enable GLiNER enrichment (CPU)")
            p.add_argument("--no-llm", action="store_true", help="skip local LLM gap-filling")
            p.add_argument("--today", type=date.fromisoformat, help="reference date for 'present' (YYYY-MM-DD)")
        p.add_argument("-v", "--verbose", action="store_true")
    for name, help_ in (("jd", "requirements from a job posting"),
                        ("match", "match a CV against a job posting, requirement by requirement")):
        p = sub.add_parser(name, help=help_)
        if name == "match":
            p.add_argument("cv", type=Path, help="CV file (PDF or DOCX)")
        p.add_argument("posting", help="text file with the pasted posting, or - for stdin")
        p.add_argument("--json", action="store_true", help="print JSON instead of text")
        p.add_argument("--priority", action="append", default=[], metavar="ID=PRIORITY",
                       help="override a priority, e.g. --priority R3=nice (repeatable)")
        p.add_argument("--ner", action="store_true", help="enable GLiNER enrichment (CPU)")
        p.add_argument("--no-llm", action="store_true", help="skip the local LLM (refinement and judge)")
        if name == "match":
            p.add_argument("--models", action="store_true",
                           help="retrieve evidence with CPU embeddings + reranker ('matching' extra)")
            p.add_argument("--presidio", action="store_true", help="also run Presidio + spaCy during redaction")
            p.add_argument("--today", type=date.fromisoformat, help="reference date for 'present' (YYYY-MM-DD)")
        p.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING)

    if args.command == "jd":
        return _jd(args)
    if args.command == "match":
        return _match(args)

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
    if args.command != "check":
        cand = structure_cv(doc, use_ner=args.ner or None, use_llm=False if args.no_llm else None,
                            today=args.today)
    if args.command in ("structure", "analyze"):
        result["candidate"] = cand.model_dump(mode="json")
        text_out.append(format_candidate(cand))
    if args.command in ("privacy", "analyze"):
        priv = analyze_privacy(doc, cand, use_presidio=args.presidio or None, today=args.today)
        result["bias_exposure"] = priv.exposure.to_dict()
        result["redacted_text"] = priv.redacted.text
        text_out.append(priv.exposure.format())
        if priv.redacted.warnings:
            text_out.append("\n".join(f"  ! {w}" for w in priv.redacted.warnings))
        if not args.no_text:
            counts = ", ".join(f"{k} ×{v}" for k, v in sorted(priv.redacted.counts().items())) or "nothing"
            photo = "; photo dropped" if priv.redacted.photo_removed else ""
            text_out.append(f"─── What the grader sees (redacted: {counts}{photo}) ───\n{priv.redacted.text}")

    if args.json:
        if args.command in ("check", "structure"):
            result = next(iter(result.values()))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n\n".join(text_out))
    return 0


def _load_posting(args):
    """The parsed posting with --priority overrides applied, or an exit code."""
    try:
        if args.posting == "-":
            text = sys.stdin.read()
        else:
            text = Path(args.posting).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not text.strip():
        print("error: the job posting is empty", file=sys.stderr)
        return 2
    posting = parse_job_posting(text, use_ner=args.ner or None, use_llm=False if args.no_llm else None)
    for override in args.priority:
        req_id, _, prio = override.partition("=")
        if prio not in ("must", "should", "nice"):
            print(f"error: --priority {override!r}: use ID=must|should|nice", file=sys.stderr)
            return 2
        try:
            posting.set_priority(req_id.strip(), prio)
        except KeyError:
            print(f"error: --priority {override!r}: no requirement {req_id!r}", file=sys.stderr)
            return 2
    return posting


def _jd(args) -> int:
    posting = _load_posting(args)
    if isinstance(posting, int):
        return posting
    if args.json:
        print(json.dumps(posting.model_dump(mode="json", exclude={"text"}), indent=2, ensure_ascii=False))
    else:
        print(format_posting(posting))
    return 0


def _match(args) -> int:
    try:
        doc = load_document(args.cv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    posting = _load_posting(args)
    if isinstance(posting, int):
        return posting
    use_llm = False if args.no_llm else None
    cand = structure_cv(doc, use_ner=args.ner or None, use_llm=use_llm, today=args.today)
    # The matcher (and the LLM judge) only ever see the redacted copy.
    priv = analyze_privacy(doc, cand, use_presidio=args.presidio or None, today=args.today)
    result = match_requirements(posting, priv.redacted.candidate, use_llm=use_llm,
                                use_models=args.models or None, today=args.today)
    if args.json:
        data = result.model_dump(mode="json")
        quotes = {c.id: c.quote(doc.text) for c in result.chunks}
        for c in data["chunks"]:
            c["original_text"] = quotes[c["id"]]
        data["requirements"] = posting.model_dump(mode="json", exclude={"text"})["requirements"]
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"Job posting: {posting.title or '(no title found)'}  ·  {len(posting.requirements)} requirements")
        for w in posting.warnings + cand.warnings:
            print(f"  ! {w}")
        print(format_matches(result, posting, doc.text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
