# CV Analyzer

A local, privacy-first tool that grades a CV against a job posting. The CV never leaves your machine.
See [PLAN.md](PLAN.md) for the full design.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
pytest
```
