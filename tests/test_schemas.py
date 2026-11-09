from app.extraction.schemas import Candidate, ExperienceEntry, PartialDate, Span


def test_partial_date_ordinal_uses_mid_year_for_missing_month():
    assert PartialDate(year=2020, month=3).ordinal(is_end=False) == 2020 * 12 + 2
    assert PartialDate(year=2020).ordinal(is_end=False) == 2020 * 12 + 6
    assert PartialDate(year=2020).ordinal(is_end=True) == 2020 * 12 + 5
    assert str(PartialDate(year=2020, month=3)) == "2020-03" and str(PartialDate(year=2020)) == "2020"


def test_candidate_round_trips_through_json():
    c = Candidate(experience=[ExperienceEntry(title="Engineer", start=PartialDate(year=2020), span=Span(start=0, end=8))])
    again = Candidate.model_validate_json(c.model_dump_json())
    assert again == c
    assert "experience" in Candidate.model_json_schema()["properties"]


def test_span_slices_text():
    assert Span(start=5, end=8).of("Jane Doe Engineer") == "Doe"
