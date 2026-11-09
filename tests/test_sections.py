import pytest

from app.extraction.sections import classify_heading, normalize_heading, split_sections
from app.ingestion import load_document
from app.ingestion.document import Line


@pytest.mark.parametrize("raw,norm", [
    ("E X P E R I E N C E", "experience"),
    ("W O R K  E X P E R I E N C E", "work experience"),
    ("1. Work Experience", "work experience"),
    ("— SKILLS —", "skills"),
    ("Skills:", "skills"),
    ("Awards & Honors", "awards and honors"),
])
def test_normalize_heading(raw, norm):
    assert normalize_heading(raw) == norm


@pytest.mark.parametrize("text,kind", [
    ("EXPERIENCE", "experience"),
    ("Professional Experience", "experience"),
    ("Experiance", "experience"),  # typo
    ("Education & Training", "education"),
    ("TECHNICAL SKILLS", "skills"),
    ("Licenses and Certifications", "certifications"),
    ("Hobbies", "interests"),
    ("VOLUNTEER WORK", "volunteering"),
])
def test_known_headings(text, kind):
    assert classify_heading(Line(text)) == kind


@pytest.mark.parametrize("text", [
    "Languages: Python, Java, SQL",  # inline label, not a heading
    "Jan 2021 – Present",
    "Built Python ETL pipelines processing 2 TB per day.",
    "Software Engineer, Beta Corp",
    "Tools",  # weak alias without a layout cue
])
def test_not_headings(text):
    assert classify_heading(Line(text)) is None


def test_weak_alias_needs_layout_cue():
    assert classify_heading(Line("Tools", bold=True)) == "skills"
    assert classify_heading(Line("Research", style="Heading 2")) == "publications"


def test_unknown_heading_by_layout():
    assert classify_heading(Line("MISCELLANEOUS")) == "other"
    assert classify_heading(Line("Things I Like", bold=True, font_size=14), body_size=10) == "other"
    assert classify_heading(Line("Senior Software Engineer", bold=True, font_size=10.5), body_size=10) is None


@pytest.mark.parametrize("fixture", ["single_column_pdf", "two_column_pdf", "plain_docx", "tricky_docx"])
def test_fixture_cvs_split_into_expected_sections(fixture, request):
    sections = split_sections(load_document(request.getfixturevalue(fixture)))
    kinds = [s.kind for s in sections]
    assert kinds[0] == "header"
    assert {"experience", "education", "skills", "languages"} <= set(kinds)
    exp = next(s for s in sections if s.kind == "experience")
    assert any("Acme" in ln.text for ln in exp.lines)
    assert not any("University" in ln.text for ln in exp.lines)
