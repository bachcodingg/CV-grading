import pytest

from app.patterns import EMAIL, URL, find_phones


@pytest.mark.parametrize("text", ["+49 151 2345 6789", "(089) 123-4567", "0151/23456789",
                                  "+1 (555) 010-9999", "0049 89 1234567", "+49 (0) 89 1234 567"])
def test_phone_numbers_match(text):
    assert [m.group() for m in find_phones(text)] == [text]


@pytest.mark.parametrize("text", ["2016 – 2018", "06/2018 – 12/2020", "2013 2016 2018", "GPA 3.8/4.0",
                                  "80333 München", "10.1145/3292500.3330701", "processed 2 TB per day"])
def test_not_phone_numbers(text):
    assert find_phones(text) == []


def test_email_and_urls():
    assert EMAIL.search("mail: jane.doe+cv@example.co.uk").group() == "jane.doe+cv@example.co.uk"
    assert URL.findall("linkedin.com/in/janedoe | github.com/jd | https://jane.dev/blog") == [
        "linkedin.com/in/janedoe", "github.com/jd", "https://jane.dev/blog"]
