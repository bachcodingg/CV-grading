from dataclasses import dataclass
from datetime import date

import pytest

from app.config import Config, LLMConfig, NERConfig, PrivacyConfig
from app.extraction import structure_cv
from app.extraction.schemas import Span
from app.ingestion import load_document
from app.privacy import analyze_privacy, find_personal_data
from samples import VARIED_CV, doc_from_text

TODAY = date(2026, 11, 10)
OFFLINE = Config(llm=LLMConfig(enabled=False), ner=NERConfig(enabled=False),
                 privacy=PrivacyConfig(presidio_enabled=False))

GERMAN_CV = """
Ms. Anna Maria Schmidt (she/her)
Hauptstraße 12, 80331 München | anna.schmidt@web.de | +49 170 1234567 | linkedin.com/in/annaschmidt

PERSONAL DETAILS
Date of birth: 12.03.1995 | Place of birth: Hanoi, Vietnam
Nationality: German | Marital status: married, 2 children
Religion: Catholic

SUMMARY
Anna is a backend engineer. She has 5 years of experience; her focus is APIs.

EXPERIENCE
Software Engineer — Acme GmbH, Jan 2021 – Present
- Built a single-page app in ASP.NET for 3 women-led teams
- Treasurer of the Vietnamese Students' Association

LANGUAGES
German (native), Vietnamese (C1), English (C2)

INTERESTS
Church choir, chess
"""


def _privacy(text: str, **kw):
    doc = doc_from_text(text)
    cand = structure_cv(doc, cfg=OFFLINE, today=TODAY)
    return doc, cand, analyze_privacy(doc, cand, cfg=OFFLINE, today=TODAY, **kw)


# ------------------------------------------------------------------ detection


def test_detects_german_style_personal_details():
    doc, _, res = _privacy(GERMAN_CV)
    by_kind = {}
    for it in res.personal.items:
        by_kind.setdefault(it.kind, []).append(it.text)
    assert by_kind["name"] == ["Anna Maria Schmidt", "Anna"]
    assert by_kind["date_of_birth"] == ["Date of birth: 12.03.1995"]
    assert by_kind["place_of_birth"] == ["Place of birth: Hanoi, Vietnam"]
    assert by_kind["nationality"] == ["Nationality: German", "Vietnamese"]
    assert by_kind["marital_status"] == ["Marital status: married, 2 children"]
    assert by_kind["children"] == ["2 children"]
    assert by_kind["religion"] == ["Religion: Catholic", "Church"]
    assert by_kind["address"] == ["Hauptstraße 12, 80331 München"]
    assert by_kind["gender"] == ["Ms."]
    assert by_kind["pronouns"] == ["(she/her)"]
    assert by_kind["pronoun"] == ["She", "her"]
    assert by_kind["gender_term"] == ["women"]
    assert "phone" in by_kind and by_kind["phone"] == ["+49 170 1234567"]  # the DOB is not a phone number
    # Every item's span points at its text.
    assert all(it.span.of(doc.text) == it.text for it in res.personal.items if it.span)


def test_skills_and_job_text_are_not_personal_data():
    _, _, res = _privacy(GERMAN_CV)
    texts = [it.text for it in res.personal.items]
    assert "ASP.NET" not in texts  # not a link
    assert not any(it.kind == "marital_status" and "single" in it.text.lower() for it in res.personal.items)
    # Language skills stay intact: "German (native)" is a skill, not a nationality.
    assert "German (native), Vietnamese (C1), English (C2)" in res.redacted.text


def test_marital_status_only_counts_in_personal_context():
    _, _, res = _privacy("""
Jane Doe
jane@example.com
Single | Munich, Germany

EXPERIENCE
Engineer at Acme GmbH, 2020 – 2023
- Engaged 40 stakeholders and built a single sign-on service
""")
    assert [i.text for i in res.personal.of_kind("marital_status")] == ["Single"]


def test_free_text_birth_age_citizenship_and_pronouns():
    _, _, res = _privacy("""
John Smith
john@example.com

SUMMARY
Born on 4 July 1990 in Springfield. 36 years old, US citizen, father of two.
""")
    kinds = {i.kind: i.text for i in res.personal.items}
    assert kinds["date_of_birth"] == "Born on 4 July 1990"
    assert kinds["place_of_birth"] == "Springfield"
    assert kinds["age"] == "36 years old"
    assert kinds["nationality"] == "US citizen"
    assert kinds["children"] == "father of two"


def test_name_is_redacted_everywhere_including_publications():
    _, _, res = _privacy(VARIED_CV)
    assert "Smith" not in res.redacted.text
    assert "[NAME], J. and Lee, K. (2021)" in res.redacted.text
    assert "github.com/jsmith" not in res.redacted.text


def test_missing_name_is_a_warning():
    _, _, res = _privacy("""
EXPERIENCE
Engineer at Acme GmbH, 2020 – 2023
- Built things
""")
    assert any("No candidate name" in w for w in res.redacted.warnings)


# ------------------------------------------------------------------ redaction


def test_redacted_text_has_no_personal_values():
    _, _, res = _privacy(GERMAN_CV)
    red = res.redacted.text
    for value in ("Anna", "Schmidt", "anna.schmidt", "1234567", "12.03.1995", "Hanoi", "Nationality: German",
                  "married", "Catholic", "Hauptstraße", "she/her", " She ", "Ms."):
        assert value not in red, value
    assert red.splitlines()[0] == "[GENDER] [NAME] [PRONOUNS]"
    assert "[DATE OF BIRTH] | [PLACE OF BIRTH]" in red
    assert "Treasurer of the [NATIONALITY] Students' Association" in red
    assert res.redacted.counts()["name"] == 2


def _variant(name, pronouns, pron, poss, nationality, dob, religion, group, mf):
    return f"""
{name} ({pronouns})
{name.split()[0].lower()}@example.com | +49 170 1234567

PERSONAL DETAILS
Date of birth: {dob} | Nationality: {nationality}
Religion: {religion}

SUMMARY
{name.split()[0]} is a data engineer. {pron} has 6 years of experience and {poss} focus is streaming.

EXPERIENCE
Data Engineer — Contoso GmbH, Mar 2019 – Present
- Built Kafka pipelines processing 1 TB per day
- Co-founded the {group} engineers network for {mf} in tech
"""


def test_counterfactual_variants_redact_to_identical_text():
    a = _variant("Jane Doe", "she/her", "She", "her", "German", "12.03.1995", "Catholic", "Vietnamese", "women")
    b = _variant("Mohammed Al Rashid", "he/him", "He", "his", "Syrian", "01/07/1979", "Muslim", "Nigerian", "men")
    texts = [_privacy(v)[2].redacted.text for v in (a, b)]
    assert texts[0] == texts[1]
    assert "Kafka pipelines processing 1 TB per day" in texts[0]


def test_offsets_map_back_to_the_original():
    doc, _, res = _privacy(GERMAN_CV)
    red = res.redacted
    i = red.text.index("[PRONOUN] has 5 years")
    assert red.original_of(Span(start=i, end=i + len("[PRONOUN] has"))) == "She has"
    j = red.text.index("Built a single-page app")
    k = j + len("Built a single-page app in ASP.NET for 3 [GENDER TERM]-led teams")
    assert red.original_of(Span(start=j, end=k)) == "Built a single-page app in ASP.NET for 3 women-led teams"
    # Text before the first placeholder and after the last one maps 1:1.
    tail = red.text.index("chess")
    assert doc.text[red.to_original(tail):].startswith("chess")


def test_redacted_candidate_keeps_original_spans():
    doc, cand, res = _privacy(GERMAN_CV)
    rc = res.redacted.candidate
    assert rc.contact.name == "[NAME]" and rc.contact.email == "[EMAIL]" and rc.contact.links == ["[LINK]"]
    assert rc.summary.startswith("[NAME] is a backend engineer. [PRONOUN] has")
    bullets = [b.text for b in rc.experience[0].bullets]
    assert bullets == ["Built a single-page app in ASP.NET for 3 [GENDER TERM]-led teams",
                       "Treasurer of the [NATIONALITY] Students' Association"]
    # Spans still point at the user's own CV text.
    assert rc.experience[0].bullets[0].span.of(doc.text).endswith("women-led teams")
    # The input candidate is not modified.
    assert cand.contact.email == "anna.schmidt@web.de"
    assert rc.skill_names() == cand.skill_names()


# ------------------------------------------------------------------ photos & reports


def test_photo_detected_in_docx(tricky_docx, plain_docx):
    photo = find_personal_data(load_document(tricky_docx), cfg=OFFLINE).of_kind("photo")
    assert len(photo) == 1
    assert find_personal_data(load_document(plain_docx), cfg=OFFLINE).of_kind("photo") == []


def test_icons_are_not_photos(icon_pdf):
    assert find_personal_data(load_document(icon_pdf), cfg=OFFLINE).of_kind("photo") == []


def test_bias_exposure_codes_and_severities():
    _, _, res = _privacy(GERMAN_CV)
    sev = {f.code: f.severity for f in res.exposure.findings}
    assert sev == {
        "DATE_OF_BIRTH": "warning", "PLACE_OF_BIRTH": "warning", "NATIONALITY": "warning",
        "MARITAL_STATUS": "warning", "CHILDREN": "warning", "RELIGION": "warning",
        "ORIGIN_GROUP": "info", "RELIGIOUS_AFFILIATION": "info", "SALUTATION": "info", "PRONOUNS": "info",
        "THIRD_PERSON": "info", "GENDER_CODED_WORDS": "info", "FULL_ADDRESS": "info",
    }
    group = next(f for f in res.exposure.findings if f.code == "ORIGIN_GROUP")
    assert group.examples == ["Vietnamese Students' Association"]
    text = res.exposure.format()
    assert "information only" in text and "redacted copy" in text


def test_clean_cv_has_no_exposure_findings(single_column_pdf):
    doc = load_document(single_column_pdf)
    cand = structure_cv(doc, cfg=OFFLINE, today=TODAY)
    res = analyze_privacy(doc, cand, cfg=OFFLINE, today=TODAY)
    assert res.exposure.findings == []
    assert "Nothing on the CV is a common trigger" in res.exposure.format()
    assert set(res.redacted.counts()) == {"name", "email", "phone", "link", "location"}


def test_age_signals_from_dates():
    _, _, res = _privacy("""
Jane Doe
jane@example.com

EXPERIENCE
Engineer at Acme GmbH, Jan 2015 – Present
- Built things
Junior Developer at Beta Corp, 2001 – 2014
- Built other things

EDUCATION
Abitur, Goethe-Gymnasium Frankfurt, 2000
""")
    codes = res.exposure.codes()
    assert {"OLD_EXPERIENCE", "SCHOOL_DATES"} <= codes
    old = next(f for f in res.exposure.findings if f.code == "OLD_EXPERIENCE")
    assert old.examples == ["Junior Developer @ Beta Corp (2001)"]


# ------------------------------------------------------------------ Presidio (optional)


@dataclass
class _Result:
    entity_type: str
    start: int
    end: int
    score: float


class FakePresidio:
    """Tags a fixed vocabulary the way Presidio's AnalyzerEngine reports results."""

    VOCAB = {"Max Mustermann": "PERSON", "Spring Boot": "PERSON", "Kurdish": "NRP", "German": "NRP"}

    def __init__(self):
        self.calls = []

    def analyze(self, text, language, entities=None, **kw):
        self.calls.append((text, tuple(entities or ())))
        out = []
        for word, ent in self.VOCAB.items():
            if entities and ent not in entities:
                continue
            start = text.find(word)
            while start >= 0:
                out.append(_Result(ent, start, start + len(word), 0.85))
                start = text.find(word, start + 1)
        return out


PRESIDIO_CV = """
Jane Doe
jane@example.com

EXPERIENCE
Engineer at Acme GmbH, 2020 – 2023
- Built REST APIs with Spring Boot

LANGUAGES
German (native), Kurdish (B1)

INTERESTS
Kurdish cooking

REFERENCES
Max Mustermann, Acme GmbH
"""


def test_presidio_adds_names_and_groups_only_where_safe():
    fake = FakePresidio()
    _, _, res = _privacy(PRESIDIO_CV, presidio=fake)
    model_items = {(i.kind, i.text, i.section) for i in res.personal.items if i.source == "presidio"}
    assert model_items == {("name", "Max Mustermann", "references"), ("nationality", "Kurdish", "interests")}
    assert "Spring Boot" in res.redacted.text
    assert "German (native), Kurdish (B1)" in res.redacted.text
    # PERSON is never even requested for experience text.
    assert all("PERSON" not in ents for text, ents in fake.calls if "Spring Boot" in text)


def test_missing_presidio_is_a_warning(monkeypatch):
    import app.privacy.presidio as presidio_mod

    def boom(cfg):
        raise ImportError("no presidio")

    monkeypatch.setattr(presidio_mod, "load_presidio", boom)
    found = find_personal_data(doc_from_text(PRESIDIO_CV), use_presidio=True, cfg=OFFLINE)
    assert any("install the 'privacy' extra" in w for w in found.warnings)


@pytest.mark.slow
def test_real_presidio_finds_a_referee_name():
    pytest.importorskip("presidio_analyzer")
    spacy = pytest.importorskip("spacy")
    model = "en_core_web_lg" if spacy.util.is_package("en_core_web_lg") else "en_core_web_sm"
    if not spacy.util.is_package(model):
        pytest.skip("no spaCy English model installed")
    cfg = Config(llm=LLMConfig(enabled=False), ner=NERConfig(enabled=False),
                 privacy=PrivacyConfig(presidio_enabled=True, spacy_model=model, presidio_threshold=0.5))
    doc = doc_from_text(PRESIDIO_CV)
    found = find_personal_data(doc, use_presidio=True, cfg=cfg)
    assert found.warnings == []
    assert any(i.source == "presidio" and "Mustermann" in i.text for i in found.items)
