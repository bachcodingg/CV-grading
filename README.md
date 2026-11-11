# CV Analyzer

A local, privacy-first tool that grades a CV against a job posting. The CV never leaves your machine.
See [PLAN.md](PLAN.md) for the full design.

**Status:** components **[1] Ingestion + parse-check**, **[2] CV Structurer**,
**[3] Redaction + bias-exposure scan**, **[4] JD Parser** and **[5] Matcher** are implemented.
Evidence verifier, scoring and the coach (components [6]–[9]) are next.

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

### [4] JD Parser (`app/extraction/jd_parser.py`)
A pasted job posting becomes numbered, **atomic requirements**:

```
Requirement { id, text, category: skill|experience|education|certification|soft_skill|language|other,
              priority: must|should|nice, priority_source: cue|section|default|user, priority_cue,
              min_years, degree_level, languages, language_level, keywords[], any_of, span }
```

1. **Sections** by heading ("What you'll bring", "Your profile", "Nice to have", "Preferred qualifications",
   "Responsibilities", "Tech stack", "Benefits", "About us", ...). Requirement sections set a default priority
   (must / should / nice); benefits and company blurbs are ignored. Postings without headings are read sentence by
   sentence, keeping sentences that read like requirements.
2. **Atomic splitting**: a bullet is cut where its clauses ask for different things or carry different priorities:
   "Excellent communication skills and fluency in English (C1); German is a plus" → soft skill (must),
   English C1 (must), German (nice). "Knowledge of PostgreSQL and Redis" stays one requirement.
3. **In code, per requirement**: category; **priority from wording** ("required", "is a plus", "preferred",
   "a strong plus", "not required"), which overrides the section default; minimum years ("3–5 years",
   "at least two years", "6 months"); lowest accepted degree; spoken languages and CEFR level; keywords and
   whether they are alternatives ("AWS, GCP or Azure", "e.g.").
4. **Keywords** come from a curated vocabulary (`app/matching/skills_aliases.yaml`, ~180 skills with aliases such
   as K8s → Kubernetes, case-aware so "react quickly" or "the rest of the team" do not count) plus tech-looking
   terms (gRPC, S3, OAuth2). The posting's keyword list (verbatim, in order) feeds the ATS coverage check.
5. **Optional models**: GLiNER adds unknown skills; the local LLM classifies requirements the rules could not and
   splits very long ones. Every LLM part must be verbatim posting text or it is discarded.

The user can change any priority (`JobPosting.set_priority`, or `--priority R3=nice` on the command line).

### [5] Matcher (`app/matching/`)
Each requirement is matched against the **redacted** CV, cheapest step first:

1. **Chunks**: the CV is cut into numbered evidence chunks (one bullet, job header, education entry, skills line,
   certificate). Chunk text is the redacted copy; the span points into the original CV for quoting.
2. **Deterministic checks** in code: years of experience (in total, or only in jobs that mention the keywords),
   degree level and field of study (small synonym list: Informatics = Computer Science), spoken languages and
   CEFR levels, certificate names (fuzzy). A check either decides the requirement or sets a **cap** the judge
   cannot exceed (2 of 5 years can never "meet").
3. **Keyword + alias**: posting keywords looked up by canonical name (CV "K8s" satisfies "Kubernetes"), with
   `verbatim` recorded for the ATS check. Plain skill requirements with clear evidence are decided here;
   "strong", "expert", years etc. go on to the judge.
4. **Retrieval**: lexical (idf-weighted, alias-aware, no models) or, with `--models`, `bge-small-en-v1.5`
   embeddings + `bge-reranker-base` on the CPU. Up to 3 candidate chunks per requirement.
5. **LLM judge** (Ollama, one requirement per call): returns a label (`none|tangential|partial|meets|exceeds`,
   rating 0–4 in code) and excerpt **numbers**, not quotes. Unknown numbers are dropped; a positive label
   with no valid excerpt is not used ("no evidence, no points").

Without a judge, undecided requirements get a cautious `low`-confidence label; retrieval alone never earns
more than "partial". Not yet included: the optional NLI pre-check and careful mode (PLAN.md M5).

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
pip install -e ".[matching]"      # optional: CPU embeddings + reranker (sentence-transformers)
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
cv-analyzer jd posting.txt                # requirements from a pasted job posting
cv-analyzer jd - --priority R3=nice < posting.txt --json   # from stdin, with a priority override
cv-analyzer match my_cv.pdf posting.txt   # requirement-by-requirement evidence (add --models, --json)
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

from app.extraction.jd_parser import parse_job_posting
job = parse_job_posting(open("posting.txt").read(), use_llm=False)
for r in job.requirements:
    print(r.id, r.priority, r.category, r.text, [k.canonical or k.text for k in r.keywords])
job.set_priority("R3", "nice")            # the user knows better
```

## Configuration

Model names and thresholds live in `app/config.py` and can be overridden with `CVA_*` environment variables:

| Variable | Default | |
|---|---|---|
| `CVA_LLM_ENABLED` | `1` | use Ollama gap-filling / JD refinement when reachable |
| `CVA_LLM_MODEL` | `qwen3:1.7b` | any pulled Ollama model |
| `CVA_OLLAMA_HOST` | `http://127.0.0.1:11434` | non-local hosts are refused… |
| `CVA_ALLOW_REMOTE_LLM` | `0` | …unless this is set |
| `CVA_LLM_NUM_CTX` / `CVA_LLM_NUM_GPU` | `2048` / auto | VRAM tuning |
| `CVA_NER_ENABLED` | `0` | run GLiNER by default |
| `CVA_NER_MODEL` | `urchade/gliner_small-v2.1` | |
| `CVA_PRESIDIO_ENABLED` | `0` | run Presidio + spaCy during redaction |
| `CVA_SPACY_MODEL` | `en_core_web_lg` | spaCy model used by Presidio |
| `CVA_PRESIDIO_THRESHOLD` | `0.6` | minimum Presidio score |
| `CVA_RETRIEVAL_MODELS` | `0` | use CPU embeddings + reranker instead of lexical retrieval |
| `CVA_EMBEDDING_MODEL` / `CVA_RERANKER_MODEL` | `BAAI/bge-small-en-v1.5` / `BAAI/bge-reranker-base` | |
| `CVA_RERANKER_ENABLED` | `1` | rerank embedding results |
| `CVA_RETRIEVAL_TOP_K` / `CVA_JUDGE_CHUNKS` | `10` / `3` | chunks retrieved / shown to the judge |

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
  extraction/            schemas.py, sections.py, dates.py, cv_extractor.py, ner.py, gap_fill.py, jd_parser.py
  matching/              skills_aliases.yaml, skills.py, schemas.py, chunks.py, deterministic.py, keywords.py,
                         retrieval.py, judge.py, matcher.py
  privacy/               detectors.py, redactor.py, bias_exposure.py, presidio.py
  llm/                   client.py (Ollama), prompts.py
tests/                   pytest suite + synthetic CV builders (samples.py)
```
