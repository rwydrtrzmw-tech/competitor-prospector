"""
Competitor Prospector — a LangChain-powered SDR agent.

The play: Bridgewater built its "pocket analyst" on LangGraph, compressing
days of expert research into minutes. Every fund competing with Bridgewater
now has an agentic-AI gap. This agent runs that play end-to-end:

  1. FIND    — identifies Bridgewater competitors and their AI-adoption signals
  2. MAP     — maps each fund's likely pain to a specific LangChain value-add
  3. SEQUENCE — drafts a personalized 3-touch outreach sequence per fund,
               anchored on the "your competitor already did this" hook

Built by Martin Tomlinson as a working demo of AI-native pipeline generation.

Usage:
  python competitor_prospector.py --demo         # no API key needed
  python competitor_prospector.py                # live, top 3 competitors
  python competitor_prospector.py --n 5

Requires (live mode):
  pip install langchain langchain-openai   # or langchain-anthropic
  export OPENAI_API_KEY=...                # or ANTHROPIC_API_KEY=...
"""

import argparse
import os
import sys

from pydantic import BaseModel, Field


# ---------- Structured schemas ----------

class Competitor(BaseModel):
    name: str = Field(description="Fund / asset manager name")
    segment: str = Field(description="e.g. global macro, multi-strat, quant")
    why_competitor: str = Field(description="One line on overlap with Bridgewater")
    ai_signal: str = Field(description="Public signal of AI investment or ambition")
    target_persona: str = Field(description="Best first contact, e.g. Head of Research Technology")


class CompetitorList(BaseModel):
    competitors: list[Competitor]


class ValueMap(BaseModel):
    pain: str = Field(description="The fund's likely pain in agentic-AI adoption")
    langchain_hook: str = Field(description="The specific LangChain/LangGraph/LangSmith capability that answers it")
    bridgewater_angle: str = Field(description="How to use the pocket-analyst story with THIS fund without overclaiming")


# ---------- Prompts ----------

FIND_SYSTEM = (
    "You are a sales research analyst for LangChain. Identify direct and "
    "adjacent competitors of Bridgewater Associates (global macro, multi-strat, "
    "systematic/quant managers). Prefer funds with public AI signals: hiring "
    "ML engineers, AI labs, published research, exec commentary. Be factual; "
    "if unsure of a signal, say 'no public signal found'."
)

MAP_SYSTEM = (
    "You are a value-engineering analyst for LangChain (LangGraph agents, "
    "LangSmith observability/evals/deployment). Bridgewater built a 'pocket "
    "analyst' research agent on LangGraph. Map one fund's pain to one specific "
    "LangChain capability. Never overclaim what competitors have built."
)

SEQUENCE_SYSTEM = (
    "You write cold outreach for LangChain SDRs. Rules: under 120 words per "
    "email, no buzzwords, no 'hope you're well', one idea per touch, one CTA "
    "(20-min call). Touch 1: the Bridgewater/pocket-analyst hook, tailored. "
    "Touch 2: one concrete proof point or asset. Touch 3: a believable breakup. "
    "Reference the fund's own AI signal, not invented facts."
)


def build_model():
    from langchain.chat_models import init_chat_model
    if os.environ.get("ANTHROPIC_API_KEY"):
        return init_chat_model("anthropic:claude-haiku-4-5")
    if os.environ.get("OPENAI_API_KEY"):
        return init_chat_model("openai:gpt-4o-mini")
    sys.exit("No OPENAI_API_KEY or ANTHROPIC_API_KEY set. Try --demo mode.")


def run_live(n: int):
    from langchain_core.prompts import ChatPromptTemplate

    model = build_model()

    # 1. FIND
    find_prompt = ChatPromptTemplate.from_messages([
        ("system", FIND_SYSTEM),
        ("human", "List the top {n} Bridgewater competitors most likely to buy "
                  "agent tooling now, with AI signals and target personas."),
    ])
    funds = (find_prompt | model.with_structured_output(CompetitorList)).invoke({"n": n})

    for fund in funds.competitors:
        print(f"\n{'=' * 60}\n{fund.name}  ({fund.segment})")
        print(f"  Overlap:   {fund.why_competitor}")
        print(f"  AI signal: {fund.ai_signal}")
        print(f"  Persona:   {fund.target_persona}")

        # 2. MAP
        map_prompt = ChatPromptTemplate.from_messages([
            ("system", MAP_SYSTEM),
            ("human", "Fund: {name} ({segment}). AI signal: {signal}. "
                      "Map pain -> LangChain value-add -> Bridgewater angle."),
        ])
        vm = (map_prompt | model.with_structured_output(ValueMap)).invoke({
            "name": fund.name, "segment": fund.segment, "signal": fund.ai_signal,
        })
        print(f"  Pain:      {vm.pain}")
        print(f"  Hook:      {vm.langchain_hook}")

        # 3. SEQUENCE
        seq_prompt = ChatPromptTemplate.from_messages([
            ("system", SEQUENCE_SYSTEM),
            ("human", "Fund: {name}. Persona: {persona}. AI signal: {signal}.\n"
                      "Pain: {pain}\nLangChain hook: {hook}\n"
                      "Bridgewater angle: {angle}\n\n"
                      "Write the 3-touch email sequence."),
        ])
        seq = (seq_prompt | model).invoke({
            "name": fund.name, "persona": fund.target_persona,
            "signal": fund.ai_signal, "pain": vm.pain,
            "hook": vm.langchain_hook, "angle": vm.bridgewater_angle,
        })
        print(f"\n--- 3-touch sequence for {fund.name} ---")
        print(seq.content)


# ---------- Demo mode (no API key) ----------

DEMO_OUTPUT = """
============================================================
Man Group / Man AHL  (systematic & quant)
  Overlap:   Systematic macro strategies compete directly with Bridgewater's Pure Alpha
  AI signal: Public ML research group; hiring LLM engineers in London
  Persona:   Head of Research Technology
  Pain:      Research agents stuck in notebooks — no path to supervised production use
  Hook:      LangGraph for controllable multi-step research agents + LangSmith evals on live runs

--- 3-touch sequence for Man Group ---

Touch 1 (Day 1) — Subject: AHL's research agents, past the notebook stage

Hi Alex,

Bridgewater built what they call a pocket analyst on LangGraph — expert-level
research runs that used to take analysts days, now minutes, with human
checkpoints where judgment matters.

AHL's ML group is publishing and hiring in exactly this direction. The hard
part isn't the model — it's control and evaluation once agents touch real
research workflows. That's the layer we build.

Worth 20 minutes to compare how funds are structuring this?

Martin

Touch 2 (Day 4) — reply in thread

Hi Alex — one concrete thing rather than a nudge: a 3-minute walkthrough of
an agent evaluation loop on a research-style workflow [link]. It shows the
difference between an agent that demos well and one a PM will actually trust.
If agent reliability is on the roadmap this quarter, this is the shortcut.

Touch 3 (Day 9) — breakup

Hi Alex — closing the loop. If agentic research tooling isn't a priority this
quarter, I'll stop here. When it becomes one — and the hiring suggests it
will — this thread will be easy to find. Good luck with the ML build-out.

============================================================
Two Sigma  (quant / data science)
  Overlap:   Systematic strategies and heavy data-science org, like Bridgewater's tech ambitions
  AI signal: Long-standing AI research arm; public engineering blog on ML platforms
  Persona:   ML Platform Lead
  Pain:      Custom in-house agent scaffolding = maintenance burden with no evals standard
  Hook:      LangSmith observability/evals as the missing measurement layer over existing stacks

--- 3-touch sequence for Two Sigma ---

Touch 1 (Day 1) — Subject: the measurement layer under your agent stack

Hi Jordan,

Teams with platforms as mature as Two Sigma's usually don't need help building
agents — they need help proving which ones work. Bridgewater's pocket-analyst
team paired LangGraph with tracing and evals for exactly that reason.

LangSmith drops onto existing stacks (LangChain or not) and gives you run
traces, eval suites, and regression checks per prompt change.

20 minutes to see it against one of your internal workflows?

Martin

[Touches 2-3 follow the same pattern — proof point, then breakup.]
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridgewater competitor prospector")
    parser.add_argument("--n", type=int, default=3, help="Number of competitors")
    parser.add_argument("--demo", action="store_true", help="Run without an API key")
    args = parser.parse_args()

    if args.demo:
        print(DEMO_OUTPUT)
    else:
        run_live(args.n)
