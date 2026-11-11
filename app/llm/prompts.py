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

JD_SYSTEM = (
    "You analyse one requirement from a job posting. Copy text exactly as written. "
    "Never invent, translate or reformat. Reply with JSON only."
)

JD_REFINE = """Requirement from a job posting:
<<<
{requirement}
>>>
1. "category": what the requirement asks for. One of: skill (a tool, technology or method), experience
   (work experience, often with years), education (a degree or field of study), certification,
   soft_skill (communication, teamwork, ...), language (a spoken language), other.
2. "parts": if it asks for several different things, copy each separate requirement exactly as written;
   otherwise return an empty list.

Example requirement: "Degree in Computer Science and strong communication skills"
Example answer: {{"category": "education", "parts": ["Degree in Computer Science", "strong communication skills"]}}"""

JUDGE_SYSTEM = (
    "You check whether a CV shows evidence for one job requirement. Judge only from the numbered CV excerpts. "
    "Never assume anything that is not written in them. Reply with JSON only."
)

JUDGE = """Job requirement:
<<<
{requirement}
>>>
CV excerpts:
{excerpts}
{facts}
Choose the label that fits best:
- "exceeds": the excerpts clearly show more than the requirement asks (more years, more senior, wider scope).
- "meets": the excerpts clearly show the requirement.
- "partial": the excerpts show part of it, or show it only weakly (for example only listed, never used).
- "tangential": the excerpts show only something related, not the requirement itself.
- "none": nothing in the excerpts relates to the requirement.
In "evidence", list the numbers of the excerpts that support your label (empty for "none").
In "reason", give one short sentence.

Example requirement: "Experience with Docker"
Example excerpts: [1] "Containerised 12 services with Docker and Compose" [2] "Organised the team offsite"
Example answer: {{"label": "meets", "evidence": [1], "reason": "Excerpt 1 shows hands-on Docker use."}}"""
