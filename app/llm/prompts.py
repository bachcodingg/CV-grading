"""Prompts for the local LLM. Kept tiny: one CV entry per call (PLAN.md §3 [2])."""

EXTRACT_SYSTEM = (
    "You extract fields from one entry of a CV. Copy text exactly as written in the entry. "
    "Never invent, translate or reformat. Use null when a field is not present. Reply with JSON only."
)

EXPERIENCE_FILL = """Entry from the work-experience section of a CV:
<<<
{block}
>>>
Return the job title, the employer (organisation), and the start and end dates exactly as written.
If the job is ongoing, set "end" to the word used in the entry (e.g. "Present").

Example entry: "Acme Robotics | Backend Developer | 03/2019 – 07/2021"
Example answer: {{"title": "Backend Developer", "org": "Acme Robotics", "start": "03/2019", "end": "07/2021"}}"""

EDUCATION_FILL = """Entry from the education section of a CV:
<<<
{block}
>>>
Return the degree, the field of study, the institution, and the start and end dates exactly as written.

Example entry: "Bachelor in Physics — Uni Wien (2015-2018)"
Example answer: {{"degree": "Bachelor", "field": "Physics", "institution": "Uni Wien", "start": "2015", "end": "2018"}}"""
