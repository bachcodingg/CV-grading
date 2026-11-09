"""Synthetic CV content and file builders shared by the tests.

The CVs are generated at test time so the repo holds no real personal data.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

# (text, style) where style is one of: name, contact, heading, entry, bullet, body
SAMPLE_CV: list[tuple[str, str]] = [
    ("Jane Doe", "name"),
    ("jane.doe@example.com | +49 151 2345 6789 | linkedin.com/in/janedoe | Munich, Germany", "contact"),
    ("SUMMARY", "heading"),
    ("Backend engineer with 5 years of experience building data platforms.", "body"),
    ("EXPERIENCE", "heading"),
    ("Senior Software Engineer — Acme GmbH", "entry", "Jan 2021 – Present"),
    ("• Led migration of 40 services to Kubernetes, cutting infra cost by 30%", "bullet"),
    ("• Built Python ETL pipelines processing 2 TB per day", "bullet"),
    ("Software Engineer, Beta Corp", "entry", "06/2018 – 12/2020"),
    ("• Developed REST APIs in Java and Spring Boot", "bullet"),
    ("• Mentored 3 junior developers", "bullet"),
    ("EDUCATION", "heading"),
    ("M.Sc. in Computer Science, Technical University of Munich", "entry", "2016 – 2018"),
    ("Grade: 1.3", "body"),
    ("B.Sc. Informatics, University of Hamburg", "entry", "2013 – 2016"),
    ("SKILLS", "heading"),
    ("Languages: Python, Java, SQL", "body"),
    ("Tools: Docker, Kubernetes, Terraform, Git", "body"),
    ("LANGUAGES", "heading"),
    ("English (C2), German – native", "body"),
    ("CERTIFICATIONS", "heading"),
    ("AWS Certified Solutions Architect – Associate (2022)", "body"),
]

_SIZES = {"name": 20, "heading": 12, "entry": 10.5}
_BOLD = {"name", "heading", "entry"}


def put(page: pymupdf.Page, x: float, y: float, text: str, size: float = 10, bold: bool = False,
        right: float | None = None) -> None:
    """Write one line with an embedded Unicode font. ``y`` is the baseline.

    With ``right`` set, the text is right-aligned so it ends at that x.
    """
    weight = "bold" if bold else "normal"
    align = "right" if right is not None else "left"
    x1 = right if right is not None else page.rect.width - 20
    rect = pymupdf.Rect(x if right is None else 20, y - size - 2, x1, y + size)
    css = f"* {{font-family: sans-serif; font-size: {size}pt; font-weight: {weight}; text-align: {align}}}"
    page.insert_htmlbox(rect, text.replace("&", "&amp;").replace("<", "&lt;"), css=css)


def build_single_column_pdf(path: Path) -> Path:
    """One column, dates right-aligned on the same baseline as the entry title."""
    pdf = pymupdf.open()
    page = pdf.new_page()  # A4-ish 595x842
    y = 60.0
    for item in SAMPLE_CV:
        text, style = item[0], item[1]
        size = _SIZES.get(style, 10)
        if style == "heading":
            y += 8
        put(page, 50, y, text, size, style in _BOLD)
        if len(item) > 2:
            put(page, 50, y, item[2], 10, right=545)
        y += size + 6
    pdf.save(path)
    return path


def build_two_column_pdf(path: Path) -> Path:
    """Sidebar (contact, skills, languages) on the left, main column on the right."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    put(page, 40, 60, "Jane Doe", 22, bold=True)
    sidebar = [
        ("CONTACT", "heading"),
        ("jane.doe@example.com", "body"),
        ("+49 151 2345 6789", "body"),
        ("Munich, Germany", "body"),
        ("SKILLS", "heading"),
        ("Python", "body"),
        ("Java", "body"),
        ("Docker", "body"),
        ("Kubernetes", "body"),
        ("Terraform", "body"),
        ("LANGUAGES", "heading"),
        ("English (C2)", "body"),
        ("German (native)", "body"),
    ]
    main = [
        ("EXPERIENCE", "heading"),
        ("Senior Software Engineer", "entry"),
        ("Acme GmbH | Jan 2021 – Present", "body"),
        ("• Led migration of 40 services to Kubernetes", "bullet"),
        ("• Built Python ETL pipelines processing 2 TB per day", "bullet"),
        ("Software Engineer", "entry"),
        ("Beta Corp | 06/2018 – 12/2020", "body"),
        ("• Developed REST APIs in Java and Spring Boot", "bullet"),
        ("EDUCATION", "heading"),
        ("M.Sc. Computer Science", "entry"),
        ("Technical University of Munich | 2016 – 2018", "body"),
    ]
    for items, x in ((sidebar, 40), (main, 220)):
        y = 110.0
        for text, style in items:
            size = _SIZES.get(style, 10)
            if style == "heading" and y > 110:
                y += 14
            put(page, x, y, text, size, style in _BOLD)
            y += size + 6
    pdf.save(path)
    return path


def build_table_pdf(path: Path) -> Path:
    """Experience laid out as a ruled table (a classic ATS failure)."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    put(page, 50, 60, "Jane Doe", 20, bold=True)
    put(page, 50, 85, "jane.doe@example.com", 10)
    rows = [("2021 – Present", "Senior Engineer, Acme GmbH"), ("2018 – 2020", "Engineer, Beta Corp"),
            ("2016 – 2018", "M.Sc. Computer Science, TUM")]
    x0, x1, x2, y = 50, 170, 545, 110
    row_h = 24
    for i in range(len(rows) + 1):
        page.draw_line((x0, y + i * row_h), (x2, y + i * row_h))
    for x in (x0, x1, x2):
        page.draw_line((x, y), (x, y + len(rows) * row_h))
    for i, (a, b) in enumerate(rows):
        put(page, x0 + 4, y + i * row_h + 16, a, 10)
        put(page, x1 + 4, y + i * row_h + 16, b, 10)
    pdf.save(path)
    return path


def build_scanned_pdf(path: Path) -> Path:
    """A page that holds only an image of text (no text layer)."""
    src = pymupdf.open()
    sp = src.new_page(width=300, height=120)
    sp.insert_text((20, 50), "Jane Doe - Software Engineer", fontsize=14)
    pix = sp.get_pixmap(dpi=150)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_image(pymupdf.Rect(40, 40, 555, 250), pixmap=pix)
    pdf.save(path)
    return path


def build_multipage_header_pdf(path: Path) -> Path:
    """Two pages with the contact line repeated as a running header."""
    pdf = pymupdf.open()
    for n in range(2):
        page = pdf.new_page()
        put(page, 50, 30, "Jane Doe · jane.doe@example.com · +49 151 2345 6789", 8)
        put(page, 50, 820, f"Page {n + 1} of 2", 8)
        put(page, 50, 120, "EXPERIENCE" if n == 0 else "EDUCATION", 12, bold=True)
        put(page, 50, 140, "Some content on this page that is long enough to count.", 10)
    pdf.save(path)
    return path


def build_icon_pdf(path: Path) -> Path:
    """Contact details prefixed by small pictogram images instead of words."""
    pdf = pymupdf.open()
    page = pdf.new_page()
    put(page, 50, 60, "Jane Doe", 20, bold=True)
    icon = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16), False)
    icon.clear_with(80)
    for y, text in ((90, "jane.doe@example.com"), (105, "+49 151 2345 6789")):
        page.insert_image(pymupdf.Rect(50, y - 9, 60, y + 1), pixmap=icon)
        put(page, 64, y, text, 10)
    put(page, 50, 140, "EXPERIENCE", 12, bold=True)
    put(page, 50, 160, "Software Engineer at Acme GmbH, 2020 - 2023", 10)
    pdf.save(path)
    return path


# ------------------------------------------------------------------- DOCX


def _png_bytes(w: int = 120, h: int = 150) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    pix.clear_with(180)
    return pix.tobytes("png")


def _append_raw_run(paragraph, xml: str) -> None:
    from docx.oxml import parse_xml

    ns = (
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:v="urn:schemas-microsoft-com:vml"'
    )
    paragraph._p.append(parse_xml(f"<w:r {ns}>{xml}</w:r>"))


def build_docx(
    path: Path,
    *,
    contact_in_header: bool = False,
    experience_as_table: bool = False,
    textbox: bool = False,
    symbols: bool = False,
    photo: bool = False,
) -> Path:
    import io

    from docx import Document as new_docx
    from docx.shared import Pt

    d = new_docx()
    contact = "jane.doe@example.com | +49 151 2345 6789 | linkedin.com/in/janedoe | Munich, Germany"
    if contact_in_header:
        d.sections[0].header.paragraphs[0].text = "Jane Doe"
        d.sections[0].header.add_paragraph(contact)
    else:
        d.add_heading("Jane Doe", level=0)
        p = d.add_paragraph()
        if symbols:
            _append_raw_run(p, '<w:sym w:font="Wingdings" w:char="F02A"/>')
        p.add_run(contact)
    if photo:
        d.add_picture(io.BytesIO(_png_bytes()))
    if textbox:
        p = d.add_paragraph()
        _append_raw_run(
            p,
            '<w:pict><v:shape style="width:200pt;height:40pt"><v:textbox><w:txbxContent>'
            "<w:p><w:r><w:t>Open to relocation</w:t></w:r></w:p>"
            "</w:txbxContent></v:textbox></v:shape></w:pict>",
        )

    d.add_heading("Summary", level=1)
    d.add_paragraph("Backend engineer with 5 years of experience building data platforms.")

    d.add_heading("Experience", level=1)
    jobs = [
        ("Senior Software Engineer — Acme GmbH", "Jan 2021 – Present",
         ["Led migration of 40 services to Kubernetes, cutting infra cost by 30%",
          "Built Python ETL pipelines processing 2 TB per day"]),
        ("Software Engineer, Beta Corp", "06/2018 – 12/2020",
         ["Developed REST APIs in Java and Spring Boot", "Mentored 3 junior developers"]),
    ]
    if experience_as_table:
        t = d.add_table(rows=len(jobs), cols=2)
        for row, (title, dates, bullets) in zip(t.rows, jobs):
            row.cells[0].text = dates
            row.cells[1].text = title
            for b in bullets:
                row.cells[1].add_paragraph(b, style="List Bullet")
    else:
        for title, dates, bullets in jobs:
            p = d.add_paragraph()
            run = p.add_run(f"{title}\t{dates}")
            run.bold = True
            run.font.size = Pt(11)
            for b in bullets:
                d.add_paragraph(b, style="List Bullet")

    d.add_heading("Education", level=1)
    d.add_paragraph("M.Sc. in Computer Science, Technical University of Munich\t2016 – 2018").runs[0].bold = True
    d.add_paragraph("Grade: 1.3")
    d.add_paragraph("B.Sc. Informatics, University of Hamburg\t2013 – 2016").runs[0].bold = True

    d.add_heading("Skills", level=1)
    d.add_paragraph("Languages: Python, Java, SQL")
    d.add_paragraph("Tools: Docker, Kubernetes, Terraform, Git")
    d.add_heading("Languages", level=1)
    d.add_paragraph("English (C2), German – native")
    d.add_heading("Certifications", level=1)
    d.add_paragraph("AWS Certified Solutions Architect – Associate (2022)")
    d.save(path)
    return path
