from app.ingestion.document import Document, Line
from app.ingestion.layout import LayoutBlock, column_count, order_blocks


def _texts(ordered):
    return [b.text for b, _ in ordered]


def test_two_columns_are_read_one_after_the_other():
    blocks = [
        LayoutBlock((40, 40, 500, 60), "Jane Doe"),  # spans both columns
        LayoutBlock((40, 100, 180, 160), "SKILLS\nPython\nJava\nDocker"),
        LayoutBlock((220, 100, 550, 160), "EXPERIENCE\nSenior Engineer at Acme\nBuilt pipelines"),
        LayoutBlock((40, 180, 180, 240), "LANGUAGES\nEnglish\nGerman"),
        LayoutBlock((220, 180, 550, 240), "EDUCATION\nM.Sc. Computer Science\nTUM"),
    ]
    ordered = order_blocks(blocks)
    assert _texts(ordered) == [
        "Jane Doe",
        "SKILLS\nPython\nJava\nDocker",
        "LANGUAGES\nEnglish\nGerman",
        "EXPERIENCE\nSenior Engineer at Acme\nBuilt pipelines",
        "EDUCATION\nM.Sc. Computer Science\nTUM",
    ]
    assert column_count(ordered) == 2


def test_right_aligned_dates_are_rows_not_a_column():
    blocks = [
        LayoutBlock((50, 100, 300, 112), "Senior Engineer, Acme GmbH"),
        LayoutBlock((460, 100, 545, 112), "Jan 2021 – Present"),
        LayoutBlock((50, 115, 400, 127), "Built things that mattered a lot"),
        LayoutBlock((50, 140, 300, 152), "Engineer, Beta Corp"),
        LayoutBlock((460, 140, 545, 152), "06/2018 – 12/2020"),
        LayoutBlock((50, 155, 400, 167), "Built other things as well"),
    ]
    ordered = order_blocks(blocks)
    assert _texts(ordered)[:2] == ["Senior Engineer, Acme GmbH", "Jan 2021 – Present"]
    assert column_count(ordered) == 1


def test_single_column_sorted_top_to_bottom():
    blocks = [LayoutBlock((50, y, 500, y + 10), f"line {y}") for y in (300, 100, 200)]
    assert _texts(order_blocks(blocks)) == ["line 100", "line 200", "line 300"]


def test_document_offsets_point_into_text():
    doc = Document(source="x", file_type="pdf", extractor="test")
    doc.lines = [Line("Jane Doe", block=0), Line("   ", block=0), Line("EXPERIENCE", block=1),
                 Line("Engineer  ", block=1)]
    doc.finalize()
    assert doc.text == "Jane Doe\n\nEXPERIENCE\nEngineer"
    for ln in doc.lines:
        assert doc.text[ln.start:ln.end] == ln.text
