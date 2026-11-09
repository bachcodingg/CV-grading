# CV Analyzer: System Plan

Goal: a **job seeker** gives the tool their **CV** (PDF/DOCX) and a **job posting**. The tool returns a **grade for that position**, shows which requirements are met and which are missing, and suggests honest improvements. **Everything runs locally**, so the CV never leaves the machine.

## 0. Decisions
| Decision | Choice | Consequence |
|---|---|---|
| Primary user | **Job seeker** (self-check) | No batch ranking, recruiter dashboard, or override workflow. Focus on gaps, improvement suggestions, and a "how a screener sees you" view |
| LLM hosting | **Local model** | Ollama on the local machine. No cloud APIs and no telemetry. Hardware limits shape the design (see §4) |
| VRAM budget | **1–2 GB max** (hard cap 2 GB) | Default LLM is 1–2B parameters; embeddings and helper models run on CPU; the LLM gets only small, narrow tasks (see §4) |
| Language | **English only** | One spaCy model, English rubric/prompts, no multilingual taxonomy needed |

**Target machine:** i7-12700H, 16 GB RAM, RTX 3050 Ti Laptop (4 GB physical, **of which the app may use at most 2 GB**), Python 3.13. Ollama is not installed yet.

---

## 1. Research summary

### 1.1 The reference project (devpost.com/software/resume-bias)
I couldn't read the page directly because Devpost returned HTTP 403 to automated fetching. Hackathon projects on the same theme follow a common pattern:

| Project | Approach | Takeaway |
|---|---|---|
| **EquiHire** (HackHer413) | spaCy NER finds and neutralises identity markers (race, age, gender, religion) | Sanitisation is hard because deciding what to remove is the real problem |
| **Nameless Resume** (DeltaHacks) | Hides the name before review | Simple, and backed by callback-gap studies, but names are only one proxy |
| **Shellhacks / Resume Lens** | Bias-free scanner that starts by removing names | Same idea, early-stage |
| **Automated resume screening** | Retrieves CV chunks per criterion and requires the LLM to **quote exact CV text** for each score | Best design idea found: evidence-cited scoring |
| Most "match scorer" projects | TF-IDF + cosine similarity between CV and JD | Their own authors note: *text similarity ≠ qualified* |

### 1.2 Related open-source tools
- **Resume Matcher** (srbhr/Resume-Matcher): job-seeker focused. Match score, keyword gaps, suggestions, and **local models through Ollama**. This is the closest existing tool to this plan; study it, and look for what it does better and worse.
- **Hiring Agent** (HackerRank): rubric with separate dimensions plus bonuses and deductions. Easy to explain to users.
- **strata-match**: cheap vector similarity first, LLM scoring second. This matters a lot on a 4 GB GPU.

### 1.3 Bias research (why it still matters for a self-check tool)
- **Wilson & Caliskan, AIES 2024**: embedding models favoured White-associated names in 85.1% of tests and female-associated names in only 11.1%. Shorter CVs made the bias worse. Code and data: `kyrawilson/Resume-Screening-Bias`.
- **"Can LLMs Hire Fairly?" (2026 preprint)**: the direction of bias changes between model generations, so it has to be measured on the exact local model you run.
- **FAIRE**: a benchmark for LLM resume-evaluation bias.

For a job seeker this means two things:
1. **The tool's own grade must not depend on the user's name, gender, age, etc.** Grade a redacted CV, and test this with counterfactual variants (§3 [9]).
2. **Real screeners (people and ATS) can be biased.** A useful feature is to point out what on the CV could *trigger* bias, such as a photo, date of birth, marital status, nationality, or full address. All of these are common on German-style CVs. The user then decides whether to keep them.

### 1.4 Legal context
- The EU AI Act's high-risk category (Annex III §4) targets systems that **employers or recruiters** use to screen or evaluate candidates. A self-assessment tool that a person runs on their own CV is very likely outside that use case. If the tool is ever offered to employers, check again.
- With data processed only locally for the user's own purposes, GDPR obligations are minimal. Privacy-by-design is still a core feature of the tool (§5).

---

## 2. Design principles
1. **The LLM judges evidence; code computes the score.** The LLM rates each requirement on a 0–4 rubric, and a deterministic formula produces the grade.
2. **No evidence, no points.** Each rating has to cite verbatim CV text, and code checks that the text exists. Small local models hallucinate more often, so this check matters even more here.
3. **Do as much as possible without the LLM.** Dates, years of experience, keyword coverage, and parsing checks are all done in code, and small CPU models handle entity extraction, retrieval, and entailment. The LLM is used only where judgment is needed, with short prompts and simple outputs, because a 1–2B model is reliable only on narrow tasks.
4. **Honest suggestions.** The tool never invents experience. Any rewrite that adds facts not in the CV is flagged.
5. **Offline by default.** After one-time model downloads, the app works with networking disabled.

---

## 3. Architecture

```
 CV (PDF/DOCX)                 Job posting (pasted text)
      │                                 │
 [1] Ingestion + parse-check      [4] JD Parser ──► requirements (user can edit priority)
      │                                 │
 [2] CV Structurer                      │
      │                                 │
 [3] Redaction (for grading)            │
   + Bias-exposure scan (for user)      │
      │                                 │
      └──────────► [5] Matcher ◄────────┘
             deterministic → keyword/alias → embeddings → small LLM judge
                          │
                   [6] Evidence verifier
                          │
                   [7] Scoring engine ──► grade A–E + breakdown
                          │
                   [8] Coach: gaps, ATS keywords, bullet rewrites (fact-checked)
                          │
                   [9] Bias audit harness (dev/CI)
```

### [1] Ingestion + parse-check
- PDF: `PyMuPDF` (sorts text blocks by column for multi-column layouts), with `pdfplumber` as a fallback. DOCX: `python-docx`.
- Scanned PDF: OCR with Tesseract (optional install).
- **Parse-check for the user:** "This is what a parser extracts from your CV." It warns about text in tables, icons used instead of words, headers/footers holding contact info, image-only sections, and odd fonts. These are real ATS failure causes, and this is a cheap, very useful feature.

### [2] CV Structurer
- A **rules-first hybrid**:
  1. A rule-based section splitter (headings like *Experience*, *Education*, *Skills*).
  2. Regex + `dateparser` for date ranges; layout cues (bold/first line of a block) for title/org.
  3. **GLiNER** (zero-shot NER, ~200–500M params, **CPU**) for skills, tools, degrees, certifications, and organisations.
  4. The local LLM fills only what's still missing, **one entry at a time** (e.g. one job block → `{title, org, start, end}`). Tiny prompts suit 1–2B models and keep the KV cache small.
- Uses Ollama **structured outputs** (`format=<JSON schema>`) and validates the result with Pydantic, retrying on failure.
```
Candidate { contact{…}, summary, experience[{title, org, start, end, bullets[]}],
            education[{degree, field, institution, start, end, grade}],
            skills[], projects[], certifications[], languages[], publications[] }
```
- Date parsing uses `dateparser`. **Years of experience are computed in code**, merging overlapping periods.
- Every field keeps its source text span.

### [3] Redaction + Bias-exposure scan
- **Redaction (internal):** the copy of the CV sent to the grader has name, photo, DOB/age, gender markers, nationality, address, marital status, and religion removed. Tools: **Presidio** + spaCy `en_core_web_lg`, plus regex rules (DOB, "married", "nationality:", etc.).
- **Bias-exposure scan (shown to the user):** lists each detected item with a short explanation, e.g. *"Photo detected. Many English-speaking employers advise against photos, and they can trigger appearance bias."* Information only; the user decides.

### [4] JD Parser
- The user pastes the posting, and the LLM turns it into atomic requirements:
```
Requirement { id, text, category: skill|experience|education|certification|soft_skill|language,
              priority: must|should|nice, min_years?, keywords[] }
```
- Priority comes from wording cues ("required", "must" → must; "plus", "nice to have" → nice). The user can adjust it in the UI.
- The extracted **keywords** feed the ATS coverage check in [8].

### [5] Matcher (cheapest step first)
1. **Deterministic:** min years, degree level, required certifications.
2. **Keyword + alias:** exact and alias matching (a curated `skills_aliases.yaml`: "K8s" → "Kubernetes", "JS" → "JavaScript", …). Matched requirements with clear evidence can skip the LLM.
3. **Embeddings + reranker (CPU):** `bge-small-en-v1.5` retrieves the top-10 CV chunks for each requirement, and a cross-encoder reranker (`bge-reranker-base` or `ms-marco-MiniLM`) narrows them to the top-3. This also picks the **evidence candidates in code**, so the LLM doesn't have to find quotes itself.
4. **Entailment pre-check (CPU, optional):** a small NLI model (e.g. DeBERTa-v3 MNLI) scores "this chunk supports this requirement". High confidence either way decides the requirement without the LLM.
5. **LLM judge:** only for requirements still undecided, **one requirement per call**. Given the requirement and 3 numbered chunks, it returns:
   ```
   { requirement_id, label: none|tangential|partial|meets|exceeds, evidence_chunk_ids: [1..3], reason (1 sentence) }
   ```
   The label maps to a 0–4 rating in code. Choosing chunk IDs is far easier for a 1–2B model than writing verbatim quotes, and it makes hallucinated evidence impossible: the quotes shown to the user come from the chosen chunks.
   - Temperature 0, structured output, and a short rubric with 1–2 examples per label in the prompt.
   - Optional "careful mode": 3 samples with the majority label taken, or the larger CPU-offloaded model (§4).

### [6] Evidence verifier
- Checks that the chosen chunk IDs are valid and that each chunk's text exists in the original CV. For any free-text quote (e.g. in rewrites), fuzzy-matches against the CV (`rapidfuzz`, ≥ 90).
- Sanity rule: if the judge says *meets/exceeds* but the NLI/reranker score of every chosen chunk is very low, the rating is downgraded and flagged for "low confidence".

### [7] Scoring engine
```
overall = Σ w_i · (rating_i / 4) / Σ w_i  × 100      w: must=3, should=2, nice=1
must-have gate: any must-have with rating ≤ 1 → overall capped at 49
```
- Grades: **A** ≥ 85 · **B** 70–84 · **C** 55–69 · **D** 40–54 · **E** < 40.
- Shows a confidence level: low if parsing quality is poor, many requirements went to the LLM, or samples disagreed.
- The fit grade is kept separate from **ATS keyword coverage** (matched JD keywords ÷ total) and **CV quality** (quantified achievements, action verbs, length, sections).

### [8] Coach (what the job seeker actually needs)
- **Gap list**, split into:
  - *"Probably have it, but it isn't visible"* (rating 1–2, related evidence found) → suggests making it explicit.
  - *"True gap"* (rating 0) → states it plainly, optionally with a learning or project idea.
- **ATS keywords:** JD terms missing word-for-word from the CV, even where the skill is there under another name (e.g. CV says "K8s", JD says "Kubernetes").
- **Bullet rewrites:** the LLM rewrites selected bullets to target the job. A **fact-checker** flags any new number, tool, or organisation that is not in the original bullet. Rewrite quality is the weakest point of a 1–2B model, so rewrites are shown as *suggestions to edit*, and "careful mode" can switch to the larger model for this step only.
- **Multi-job comparison:** grades one CV against several saved postings and ranks where the user fits best.
- **Version history:** stores past CV versions and their grades for the same job in local SQLite, to show the effect of each revision.

### [9] Bias audit harness (developer-side)
- **Counterfactual tests:** the same CV with only the name, pronouns, age signals, or nationality changed. Run both with redaction on and off, and measure mean |Δscore| and band changes. Pass criterion: mean |Δ| < 2 points with redaction on.
- Test data: synthetic CVs, the Wilson & Caliskan dataset, and FAIRE.
- Re-run whenever the model, prompt, or rubric changes.

---

## 4. Local model strategy (VRAM budget 1–2 GB, 16 GB RAM)

Only the LLM uses the GPU. Every other model runs on the CPU (the i7-12700H handles these small models comfortably).

| Role | Device | Size tier | Candidates to benchmark* | Approx. memory |
|---|---|---|---|---|
| Extraction gaps + judge (default) | GPU | **1–2B, Q4_K_M** | Qwen3-1.7B, Gemma 3 1B, Llama 3.2 1B, Granite (2B), SmolLM2-1.7B | ~0.8–1.3 GB weights + KV cache → **≤ 1.6 GB** |
| "Careful mode" (judge/rewrites) | GPU + CPU split | 3–4B, Q4 | Qwen3-4B, Phi-4-mini, Gemma 3 4B | Only as many layers on GPU as fit in 2 GB (`num_gpu`), the rest in RAM; ~3–5× slower |
| Embeddings | CPU | ~33M | `bge-small-en-v1.5` | ~150 MB RAM |
| Reranker | CPU | ~100–280M | `bge-reranker-base`, `ms-marco-MiniLM-L-6` | < 1 GB RAM |
| NER | CPU | ~200–500M | GLiNER (small/medium) | < 1 GB RAM |
| NLI (optional) | CPU | ~180M | DeBERTa-v3-base MNLI | < 1 GB RAM |

\* Newer small models may be out by the time this is built. Keep model names in config and choose by running the eval set (§7), not by reputation.

**How the 2 GB cap is enforced** (Ollama has no direct VRAM limit, so the cap is enforced through settings):
- `num_ctx` = 2048–4096. Prompts are small by design (one requirement + 3 chunks ≈ 600–900 tokens).
- `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` halve the KV cache.
- `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NUM_PARALLEL=1`, so only one model and one request use VRAM at a time.
- `num_gpu` (layers on GPU) is set per model in config. For the careful-mode 4B model, it's tuned so usage stays under 2 GB.
- Sentence-transformers / GLiNER / NLI are forced to `device="cpu"`, so PyTorch never allocates CUDA memory.
- A **VRAM guard test** runs the full pipeline while polling `nvidia-smi` and fails if the app's peak usage is > 2 GB.

**Latency budget:** < 60 s per CV×JD in default mode (a 1–2B model on GPU generates fast; most of the time goes to the CPU models). If it's slower, add more deterministic, keyword, and NLI coverage before changing the LLM. Careful mode may take a few minutes.

**Accuracy risk:** 1–2B models are noticeably weaker at nuanced judgment. Mitigations: narrow tasks, labels instead of numbers, chunk IDs instead of free quotes, CPU models deciding the easy cases, and the eval bake-off. If the gold-set agreement is too low, the fallback is careful mode as the default for the judge only.

LLM client interface: Ollama natively. Any OpenAI-compatible local server (LM Studio, llama.cpp `server`) also works, so models can be swapped.

---

## 5. Privacy (offline by default)
- Ollama listens only on `127.0.0.1`. Don't expose it on the network.
- Streamlit sends usage stats by default. Set `browser.gatherUsageStats = false` and `server.address = "localhost"` in `.streamlit/config.toml`.
- After setup, set `HF_HUB_OFFLINE=1`. No other outbound calls; add a test that fails if the app opens any non-localhost socket.
- CVs and results live in a local `data/` folder (SQLite). The UI has a "delete everything" button. No logs contain CV text unless debug mode is on.

---

## 6. Tech stack and layout
| Layer | Choice |
|---|---|
| Runtime | Python 3.13 (fall back to a 3.12 venv if a dependency lacks wheels), `uv` for env management |
| LLM | Ollama + `ollama` Python client, JSON-schema structured outputs |
| NLP (all CPU) | spaCy `en_core_web_lg`, Presidio, GLiNER, sentence-transformers (embeddings + cross-encoder reranker), optional NLI model, rapidfuzz, dateparser |
| Parsing | PyMuPDF, pdfplumber, python-docx, (Tesseract optional) |
| Storage | SQLite (CVs, JDs, results, versions) |
| UI | Streamlit (localhost) |
| Tests/eval | pytest + `eval/` harness |

```
cv-analyzer/
  app/
    config.py          model names, weights, thresholds
    llm/               client.py (Ollama / OpenAI-compatible), prompts/
    ingestion/         pdf.py, docx.py, parse_check.py
    extraction/        sections.py, cv_extractor.py, jd_parser.py, schemas.py
    privacy/           redactor.py, bias_exposure.py
    matching/          deterministic.py, keywords.py, skills_aliases.yaml, retrieval.py, judge.py
    scoring/           verifier.py, scorer.py
    coach/             gaps.py, ats_keywords.py, rewrite.py, fact_check.py
    storage/           db.py
  ui/streamlit_app.py
  eval/
    datasets/          sample CVs, JDs, hand-graded labels
    run_eval.py        accuracy, stability, faithfulness, latency per model
    bias_audit.py      counterfactual tests
  tests/
```

---

## 7. Evaluation
1. **Gold set:** 20–30 CV×JD pairs graded by hand with the same rubric. Use your own CV against several jobs, plus friends' (anonymised) or synthetic CVs.
2. **Accuracy:** Spearman correlation between the system score and the human score, and agreement on grade bands.
3. **Faithfulness:** > 98% of quotes verify, and 0 rewrites with unflagged new facts.
4. **Stability:** score std-dev < 3 points across 5 reruns.
5. **Fairness:** counterfactual mean |Δ| < 2 points.
6. **Robustness:** the same content in a 1-column vs 2-column layout, PDF vs DOCX → within ±3.
7. **Latency:** < 60 s default mode on the target laptop.
8. **VRAM:** peak ≤ 2 GB in both default and careful mode (VRAM guard test).
9. **Model bake-off:** run 1–8 for each candidate model in §4 and pick the default from the results.

---

## 8. Milestones
| Phase | Scope | Done when |
|---|---|---|
| **M0 – Setup** | Install Ollama with VRAM-limiting env vars, pull 2 candidate 1–2B models, repo + uv env, ingestion, parse-check, section splitter, VRAM guard script | 10 sample CVs parse; parse-check warnings are shown; idle model ≤ 2 GB VRAM |
| **M1 – Extraction** | CV + JD schemas, rules + GLiNER extraction, LLM gap-filling, years-of-experience calc | Valid JSON for 10 CVs / 10 JDs, retries < 10% |
| **M2 – Grader MVP** | Deterministic + keyword matching, CPU embeddings + reranker, LLM judge (labels + chunk IDs), evidence verifier, scorer, basic Streamlit report | End-to-end grade with verified evidence for each requirement, < 60 s, ≤ 2 GB VRAM |
| **M3 – Fair + offline** | Redaction, bias-exposure scan, counterfactual audit, offline hardening | Bias Δ target met; app works with Wi-Fi off |
| **M4 – Coach** | Gap split, ATS keywords, fact-checked bullet rewrites | Suggestions pass the fact-checker on the gold set |
| **M5 – Polish** | NLI pre-check, multi-job comparison, version history, model bake-off, careful mode (4B split GPU/CPU) | Eval targets in §7 met |

---

## Sources
- Devpost: [Shellhacks](https://devpost.com/software/shellhacks) · [Resume Lens](https://devpost.com/software/resume-lens) · [EquiHire](https://devpost.com/software/equihire-v65luf) · [Nameless Resume](https://deltahacks.devpost.com/submissions/33010-nameless-resume) · [Automated resume screening](https://devpost.com/software/automated-resume-screening) · [AI Resume & Job Match Scorer](https://devpost.com/software/ai-powered-resume-and-job-match-scorer-7wnlck)
- OSS: [Resume Matcher](https://awesome.ecosyste.ms/projects/github.com%2Fsrbhr%2FResume-Matcher) · [Hiring Agent](https://github.com/NotArsal/hiring-agent) · [strata-match](https://pypi.org/project/strata-match/0.2.6/) · [EquiHire repo](https://github.com/cadecuddy/equihire)
- Bias research: [Wilson & Caliskan, AIES 2024](https://arxiv.org/abs/2407.20371v2) · [Brookings summary](https://www.brookings.edu/articles/gender-race-and-intersectional-bias-in-ai-resume-screening-via-language-model-retrieval) · [Can LLMs Hire Fairly? (2026)](https://www.alphaxiv.org/abs/2606.28978) · [FAIRE benchmark](https://arxiv.org/pdf/2504.01420v1)
- Legal: [EU AI Act & recruitment](https://www.crowe.com/pl/en-us/insights/ai-in-hr-under-regulatory-scrutiny) · [AI hiring laws in the EU](https://skima.ai/blog/recruitment-compliance/ai-hiring-laws-in-the-eu)
