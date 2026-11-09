from datetime import date

import pytest

from app.extraction.dates import find_date_range, find_single_date, months_between, parse_date, total_years
from app.extraction.schemas import PartialDate as D

TODAY = date(2026, 11, 9)


@pytest.mark.parametrize("token,expected", [
    ("Jan 2020", D(year=2020, month=1)),
    ("Sept. 2019", D(year=2019, month=9)),
    ("January, 2020", D(year=2020, month=1)),
    ("März 2018", D(year=2018, month=3)),
    ("Jan '20", D(year=2020, month=1)),
    ("01/2020", D(year=2020, month=1)),
    ("3.2021", D(year=2021, month=3)),
    ("2020-04", D(year=2020, month=4)),
    ("2019", D(year=2019)),
    ("Summer 2019", D(year=2019, month=7)),
])
def test_parse_date(token, expected):
    assert parse_date(token) == expected


def test_parse_date_dateparser_fallback():
    assert parse_date("15 March 2020") == D(year=2020, month=3)
    assert parse_date("no date here") is None


@pytest.mark.parametrize("text,start,end,current", [
    ("Senior Engineer — Acme GmbH  Jan 2021 – Present", D(year=2021, month=1), None, True),
    ("06/2018 – 12/2020", D(year=2018, month=6), D(year=2020, month=12), False),
    ("2016–2018", D(year=2016), D(year=2018), False),
    ("Sept. 2019 to June 2020", D(year=2019, month=9), D(year=2020, month=6), False),
    ("Acme | since 2021", D(year=2021), None, True),
    ("Oct 2023 -", D(year=2023, month=10), None, True),
    ("2020-03 – now", D(year=2020, month=3), None, True),
    ("10/2022 – heute", D(year=2022, month=10), None, True),
    ("M.Sc. (2024 – expected 2026)", D(year=2024), D(year=2026), False),
])
def test_find_date_range(text, start, end, current):
    r = find_date_range(text)
    assert r is not None
    assert (r.start, r.end, r.current) == (start, end, current)
    assert text[r.span.start:r.span.end] == r.text


@pytest.mark.parametrize("text", [
    "Built Python ETL pipelines processing 2 TB per day",
    "+49 151 2345 6789",
    "Grade: 1.3",
    "Reduced latency by 2019 ms",  # a number, but no range
    "2021 – 2015",  # end before start
])
def test_no_date_range(text):
    assert find_date_range(text) is None


def test_find_date_range_offset():
    r = find_date_range("Engineer 2019 - 2020", offset=100)
    assert (r.span.start, r.span.end) == (109, 120)


def test_find_single_date():
    d, span = find_single_date("AWS Certified Solutions Architect (2022)")
    assert d == D(year=2022)
    assert find_single_date("Python 3.11, 40 services") is None


def test_months_between():
    assert months_between(D(year=2021, month=1), D(year=2021, month=12), False, TODAY) == 12
    assert months_between(D(year=2021, month=1), None, True, TODAY) == 71
    assert months_between(D(year=2018), D(year=2020), False, TODAY) == 24  # mid-year to mid-year
    assert months_between(D(year=2020), D(year=2020), False, TODAY) == 6


def test_total_years_merges_overlaps():
    periods = [
        (D(year=2018, month=6), D(year=2020, month=12), False),  # 31 months
        (D(year=2020, month=1), D(year=2020, month=6), False),  # inside the first one
        (D(year=2021, month=1), None, True),  # Jan 2021 – Nov 2026 = 71 months
    ]
    assert total_years(periods, TODAY) == round((31 + 71) / 12, 1)
    assert total_years([], TODAY) == 0.0
