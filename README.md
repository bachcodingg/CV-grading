# CV Analyzer

A local, privacy-first tool that grades a CV against a job posting. The CV never leaves your machine.
See [PLAN.md](PLAN.md) for the full design.

**Status:** components **[1] Ingestion + parse-check**, **[2] CV Structurer** and
**[3] Redaction + bias-exposure scan** are implemented.
JD parsing, matching, scoring and the coach (components [4]–[9]) are next.

## What works today

### [1] Ingestion + parse-check (`app/ingestion/`)
- **PDF** via PyMuPDF with **column-aware reading order** (two-column CVs are read column by column;
  right-aligned dates stay next to their job), pdfplumber fallback, optional Tesseract OCR for scans.
- **DOCX** via python-docx, walking the body in order and also extracting what python-docx skips:
  text boxes, header/footer text, symbol-font glyphs.
- **Parse-check**: "this is what a parser extracts from your CV", with concrete warnings for the usual ATS
  failure causes:

  | Code | Meaning |
  |---|---|
  | `NO_TEXT`, `IMAGE_ONLY_PAGE`, `OCR_USED` | scanned / image-only content |
  | `TABLES`, `TEXT_BOXES` | text in table cells or text boxes |
  | `CONTACT_IN_HEADER`, `HEADER_FOOTER_TEXT` | contact details in page headers/footers |
  | `ICON_GLYPHS`, `ICON_IMAGES` | icon fonts / pictograms instead of words |
  | `GARBLED_TEXT`, `ODD_FONTS` | fonts without a text mapping, Type3 / symbol fonts |
  | `LARGE_IMAGE`, `MULTI_COLUMN` | infographics, multi-column layouts |
  | `NO_EMAIL`, `NO_PHONE`, `LONG_CV` | content sanity checks |

### [2] CV Structurer (`app/extraction/`)
Rules first, models only for what is left:

1. **Section splitter**: ~150 heading aliases, letter-spaced / numbered / misspelled headings, layout cues.
2. **Rule-based entries**: contact, experience (title, org, location, dates, bullets), education (degree,
   field, institution, dates, grade), projects, skills (with categories/levels), languages (CEFR levels),
   certifications (issuer, date), publications.
3. **GLiNER** (optional, CPU only): skills mentioned in bullets, missing organisations and degrees.
4. **Local LLM** via Ollama (optional): fills remaining gaps **one entry at a time** with JSON-schema
   structured outputs, Pydantic validation and retries. Every LLM value must be found in the entry's
   own text (rapidfuzz ≥ 90), so the model cannot add facts.
5. **Years of experience** computed in code, overlapping jobs merged.

Every extracted value keeps its **character span** in the extracted text and its **provenance**
(`rules` / `ner` / `llm`); fallback guesses are marked `uncertain`.

### [3] Redaction + bias-exposure scan (`app/privacy/`)
One detection pass, two outputs:

- **Redacted copy for the grader.** Name (everywhere it appears, including publication author lists), email,
  phone, links, address/location, date of birth, age, place of birth, gender markers and pronouns,
  nationality and origin groups, marital status, children and religion are replaced by **fixed placeholders**
  (`[NAME]`, `[DATE OF BIRTH]`, …). The placeholders do not depend on the value, so two CVs that differ only
  in name, gender, age or origin redact to the **same text** (tested). Offsets are mapped back, so evidence
  found in the redacted text can be quoted from the user's original CV. A redacted `Candidate` is built too.
- **Bias-exposure scan for the user** (information only, the user decides):

  | Code | Severity | Found |
  |---|---|---|
  | `PHOTO` | warning | portrait-shaped image on page 1 |
  | `DATE_OF_BIRTH`, `AGE`, `PLACE_OF_BIRTH` | warning | "Date of birth: …", "Born 1990 in …", "31 years old" |
  | `NATIONALITY`, `ETHNICITY` | warning | "Nationality: …", "US citizen", "dual citizenship" |
  | `MARITAL_STATUS`, `CHILDREN`, `RELIGION`, `GENDER` | warning | labelled personal fields, "mother of two" |
  | `ORIGIN_GROUP`, `RELIGIOUS_AFFILIATION` | info | "Vietnamese Students' Association", "church choir" |
  | `SALUTATION`, `PRONOUNS`, `THIRD_PERSON`, `GENDER_CODED_WORDS` | info | "Ms.", "(she/her)", "She has…", "women's" |
  | `FULL_ADDRESS`, `SCHOOL_DATES`, `OLD_EXPERIENCE` | info | street address, Abitur/high-school dates, 20+ year old roles |

Rules are context-aware: "German" in the languages section stays a skill, "single" only counts as a marital
status in a personal-details block (not in "single-page app"), and ".NET" in job text is not a link.
**Presidio + spaCy** (optional, CPU) adds other people's names (referees) and origin/religious groups in
free text; it is never run for person names on experience or skills text.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"           # core + tests
pip install -e ".[ner]"           # optional: GLiNER (pulls PyTorch; install the CPU wheel first:
                                  #   pip install torch --index-url https://download.pytorch.org/whl/cpu)
pip install -e ".[llm]"           # optional: Ollama client
pip install -e ".[privacy]"       # optional: Presidio for redaction, then:
                                  #   python -m spacy download en_core_web_lg
pip install -e ".[ocr]"           # optional: OCR (also needs the Tesseract binary on PATH)
```

For LLM gap-filling, install [Ollama](https://ollama.com), keep it on `127.0.0.1`, and pull a small model,
e.g. `ollama pull qwen3:1.7b`. Recommended server settings for the 2 GB VRAM budget (PLAN.md §4):
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`.

## Usage

```bash
cv-analyzer check my_cv.pdf               # parse-check + parser view
cv-analyzer structure my_cv.docx          # structured CV summary
cv-analyzer privacy my_cv.pdf             # bias-exposure scan + redacted copy the grader sees
cv-analyzer analyze my_cv.pdf --ner       # all three, with GLiNER enrichment
cv-analyzer structure my_cv.pdf --json    # machine-readable Candidate JSON
```

```python
from app.ingestion import load_document
from app.ingestion.parse_check import check_document
from app.extraction import structure_cv

doc = load_document("my_cv.pdf")          # path or uploaded bytes
print(check_document(doc).format())
candidate = structure_cv(doc, use_llm=False)
print(candidate.years_of_experience, candidate.skill_names())

from app.privacy import analyze_privacy
priv = analyze_privacy(doc, candidate)    # one detection pass
print(priv.exposure.format())             # what a screener could react to
grader_text = priv.redacted.text          # placeholders instead of personal data
priv.redacted.original_of(span)           # map a redacted span back to the user's own text
```

## Configuration

Model names and thresholds live in `app/config.py` and can be overridden with `CVA_*` environment variables:

| Variable | Default | |
|---|---|---|
| `CVA_LLM_ENABLED` | `1` | use Ollama gap-filling when reachable |
| `CVA_LLM_MODEL` | `qwen3:1.7b` | any pulled Ollama model |
| `CVA_OLLAMA_HOST` | `http://127.0.0.1:11434` | non-local hosts are refused… |
| `CVA_ALLOW_REMOTE_LLM` | `0` | …unless this is set |
| `CVA_LLM_NUM_CTX` / `CVA_LLM_NUM_GPU` | `2048` / auto | VRAM tuning |
| `CVA_NER_ENABLED` | `0` | run GLiNER by default |
| `CVA_NER_MODEL` | `urchade/gliner_small-v2.1` | |
| `CVA_PRESIDIO_ENABLED` | `0` | run Presidio + spaCy during redaction |
| `CVA_SPACY_MODEL` | `en_core_web_lg` | spaCy model used by Presidio |
| `CVA_PRESIDIO_THRESHOLD` | `0.6` | minimum Presidio score |

## Tests

```bash
pytest -m "not slow"     # fast suite; sample CVs are generated at test time (no real personal data)
pytest -m slow           # also runs the real GLiNER model (downloads it once) and Presidio, if installed
```

## Layout

```
app/
  config.py              model names, thresholds (env-overridable)
  patterns.py            shared email / phone / URL / icon regexes
  cli.py                 cv-analyzer command
  ingestion/             document.py, layout.py, pdf.py, docx.py, parse_check.py
  extraction/            schemas.py, sections.py, dates.py, cv_extractor.py, ner.py, gap_fill.py
  privacy/               detectors.py, redactor.py, bias_exposure.py, presidio.py
  llm/                   client.py (Ollama), prompts.py
tests/                   pytest suite + synthetic CV builders (samples.py)
```
