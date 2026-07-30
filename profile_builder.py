"""
Profile Builder — the LangChain pipeline that drafted my portfolio page.

Takes structured facts from my CV plus the target job description, and
generates the positioning copy for portfolio.html: intro, fit mapping,
and section drafts. The point: I don't just talk about AI-native GTM,
I use LangChain to market myself the way I'd use it to prospect.

Usage:
  python profile_builder.py            # needs OPENAI_API_KEY or ANTHROPIC_API_KEY
  pip install langchain langchain-openai   # or langchain-anthropic

Pipeline:
  facts + job description
    -> position (structured: hooks, fit_points, differentiators)
    -> draft copy (intro paragraph, fit cards, closing pitch)
"""

import os
import sys

from pydantic import BaseModel, Field

# ---------- My facts (the only manual input) ----------

CV_FACTS = """
- 4 years at Bloomberg selling data/analytics platforms to quants, data
  scientists, engineers, and C-level, across the UK and France
- 150% of quota; $3.5M+ cumulative pipeline; $1M+ cross-sell incl. new logos
- High-volume outbound + MEDDPICC qualification, partnering with enterprise AEs
- Champions Bloomberg's AI coding agent with clients; relays field feedback
  to product; Python automations; prompt engineering
- 9+ innovation sessions with senior stakeholders; webinars with 250+ attendees
- Fluent French (Baccalauréat, supported French institutional clients natively)
- MSc Finance & Investment (Distinction); CFA Level III candidate
- First saw LangGraph via Bridgewater's "pocket analyst" — the hook that led here
"""

JOB_DESCRIPTION = """
LangChain SDR: turn inbound/outbound demand into qualified opportunities.
Use LangChain- and LangGraph-powered agents to automate list building,
enrichment, and outreach. Requires C2 French or German. Values: outreach as
a craft, experimentation, engaging technical and business audiences.
"""


# ---------- Structured positioning ----------

class Positioning(BaseModel):
    hook: str = Field(description="One-sentence opening hook tying candidate to company")
    fit_points: list[str] = Field(description="4 requirement->evidence mappings, one line each")
    differentiator: str = Field(description="The single strongest reason to interview this candidate")


def main():
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import ChatPromptTemplate

    if os.environ.get("ANTHROPIC_API_KEY"):
        model = init_chat_model("anthropic:claude-haiku-4-5")
    elif os.environ.get("OPENAI_API_KEY"):
        model = init_chat_model("openai:gpt-4o-mini")
    else:
        sys.exit("Set OPENAI_API_KEY or ANTHROPIC_API_KEY first.")

    # Step 1: structured positioning
    position_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a career positioning strategist. Map candidate facts "
                   "to job requirements. Be specific and evidence-led; no fluff."),
        ("human", "Candidate facts:\n{facts}\n\nJob:\n{job}\n\nPosition this candidate."),
    ])
    positioning = (position_prompt | model.with_structured_output(Positioning)).invoke(
        {"facts": CV_FACTS, "job": JOB_DESCRIPTION}
    )

    print("=== Positioning ===")
    print(f"Hook: {positioning.hook}\n")
    for p in positioning.fit_points:
        print(f"  - {p}")
    print(f"\nDifferentiator: {positioning.differentiator}")

    # Step 2: draft the page copy from the positioning
    copy_prompt = ChatPromptTemplate.from_messages([
        ("system", "You write tight portfolio copy. Rules: first person, concrete "
                   "numbers, no buzzwords, no exclamation marks, UK English."),
        ("human", "Using this positioning:\nHook: {hook}\nFit: {fit}\n"
                  "Differentiator: {diff}\n\nWrite: 1) a 60-word intro paragraph, "
                  "2) four 30-word 'why I fit' cards, 3) a 40-word closing pitch."),
    ])
    draft = (copy_prompt | model).invoke({
        "hook": positioning.hook,
        "fit": "\n".join(positioning.fit_points),
        "diff": positioning.differentiator,
    })

    print("\n=== Draft page copy ===")
    print(draft.content)


if __name__ == "__main__":
    main()
