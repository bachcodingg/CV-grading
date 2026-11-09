import pytest

from app.extraction.cv_extractor import extract_candidate, split_entries
from app.extraction.schemas import PartialDate as D
from app.ingestion import load_document
from samples import VARIED_CV, doc_from_text


def assert_spans_valid(cand, doc):
    """Every recorded field span must slice exactly to its value in Document.text."""
    groups = [cand.contact, *cand.experience, *cand.education, *cand.certifications]
    for item in groups:
        for name, span in item.field_spans.items():
            if name in ("dates", "grade", "date"):
                continue  # spans cover the matched expression, not the parsed value
            value = getattr(item, name)
            assert span.of(doc.text).lower() == str(value).lower(), (name, value, span.of(doc.text))
    for s in cand.skills:
        assert s.name in s.span.of(doc.text)
    for e in cand.experience:
        for b in e.bullets:
            assert b.span.of(doc.text) == b.text


@pytest.mark.parametrize("fixture", ["single_column_pdf", "two_column_pdf", "plain_docx", "tricky_docx"])
def test_fixture_cvs(fixture, request):
    doc = load_document(request.getfixturevalue(fixture))
    cand = extract_candidate(doc)
    assert cand.contact.name == "Jane Doe"
    assert cand.contact.email == "jane.doe@example.com"
    assert cand.contact.phone == "+49 151 2345 6789"
    first, second = cand.experience[:2]
    assert (first.title, first.org, first.start, first.current) == (
        "Senior Software Engineer", "Acme GmbH", D(year=2021, month=1), True)
    assert (second.title, second.org, second.end) == ("Software Engineer", "Beta Corp", D(year=2020, month=12))
    assert first.bullets[0].text.startswith("Led migration of 40 services")
    msc = cand.education[0]
    assert (msc.degree, msc.field, msc.institution) == ("M.Sc.", "Computer Science", "Technical University of Munich")
    assert {"Python", "Java", "Docker", "Kubernetes", "Terraform"} <= set(cand.skill_names())
    assert [(l.name, l.level) for l in cand.languages] == [("English", "C2"), ("German", "native")]
    assert_spans_valid(cand, doc)


@pytest.fixture(scope="module")
def varied():
    doc = doc_from_text(VARIED_CV)
    return doc, extract_candidate(doc)


def test_varied_contact_and_summary(varied):
    doc, c = varied
    assert c.contact.name == "John Smith"
    assert (c.contact.email, c.contact.phone, c.contact.location) == ("john.smith@mail.com", "(512) 555-0199", "Austin, TX")
    assert c.contact.links == ["github.com/jsmith"]
    assert c.summary.startswith("Data scientist focused on forecasting")
    assert_spans_valid(c, doc)


def test_varied_experience_layouts(varied):
    _, c = varied
    rows = [(e.title, e.org, e.location, e.start, e.end, e.current) for e in c.experience]
    assert rows == [
        # "Title at Org" + "Location | Dates" line
        ("Data Scientist", "Globex Corporation", "Austin, TX", D(year=2022, month=3), None, True),
        # Org on its own line, then title, then dates
        ("Machine Learning Intern", "Initech", None, D(year=2021, month=6), D(year=2021, month=8), False),
        # Dates first; "at" inside a university name is not a separator
        ("Research Assistant", "University of Texas at Austin", None, D(year=2019), D(year=2021), False),
    ]
    assert c.experience[1].uncertain == ["org"]  # "Initech" has no org keyword
    assert c.experience[2].bullets[0].text.startswith("Analysed survey data")


def test_varied_education(varied):
    _, c = varied
    ms, ba = c.education
    assert (ms.degree, ms.field, ms.institution, ms.grade) == (
        "Master of Science", "Statistics", "University of Texas at Austin", "3.9/4.0")
    assert (ms.start, ms.end) == (D(year=2019, month=9), D(year=2021, month=5))
    assert (ba.degree, ba.field, ba.institution, ba.grade) == (
        "Bachelor of Arts", "Economics", "Rice University", "magna cum laude")


def test_varied_projects_skills_languages(varied):
    _, c = varied
    (proj,) = c.projects
    assert proj.name == "Bike Demand Forecaster" and proj.end == D(year=2023)
    assert proj.technologies == ["Python", "Flask", "Docker", "PostgreSQL", "Grafana"]
    assert [b.text for b in proj.bullets] == ["Web app predicting hourly bike rentals"]
    skills = {s.name: s for s in c.skills}
    assert skills["PyTorch"].level == "intermediate" and skills["PyTorch"].category == "ML"
    assert "English" not in skills  # "Languages: English (native)…" goes to languages
    assert [(l.name, l.level) for l in c.languages] == [("English", "native"), ("Spanish", "B2")]


def test_varied_certs_publications_other(varied):
    _, c = varied
    assert [(x.name, x.issuer, x.date) for x in c.certifications] == [
        ("Certified Kubernetes Administrator (CKA)", "CNCF", D(year=2023)),
        ("Google Data Analytics Certificate", "Coursera", D(year=2021, month=1)),
    ]
    assert c.publications[0].year == 2021
    assert [(o.kind, o.heading) for o in c.other_sections] == [("volunteering", "VOLUNTEERING")]
    assert c.warnings == []


def test_split_entries_dates_below_title():
    doc = doc_from_text("""
M.Sc. Computer Science, TUM
2016 – 2018
Grade: 1.3
B.Sc. Informatics, University of Hamburg
2013 – 2016
""")
    groups = split_entries(doc.lines)
    assert [[l.text for l in g] for g in groups] == [
        ["M.Sc. Computer Science, TUM", "2016 – 2018", "Grade: 1.3"],
        ["B.Sc. Informatics, University of Hamburg", "2013 – 2016"],
    ]


def test_missing_sections_produce_warnings():
    c = extract_candidate(doc_from_text("Jane Doe\nSome text without any structure at all."))
    assert any("work-experience" in w for w in c.warnings)
    assert any("email" in w for w in c.warnings)


def test_certification_level_suffix_is_not_an_issuer():
    c = extract_candidate(doc_from_text("""
CERTIFICATIONS
AWS Certified Solutions Architect – Associate (2022)
Professional Scrum Master I – Scrum.org
Deep Learning Specialization from DeepLearning.AI
"""))
    assert [(x.name, x.issuer) for x in c.certifications] == [
        ("AWS Certified Solutions Architect – Associate", "AWS"),
        ("Professional Scrum Master I", "Scrum.org"),
        ("Deep Learning Specialization", "DeepLearning.AI"),
    ]
