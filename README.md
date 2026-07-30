# Competitor Prospector — a LangChain SDR agent

A prospecting agent I built with [LangChain](https://langchain.com) that runs a real sales play end-to-end: Bridgewater built its "pocket analyst" on LangGraph — so every fund competing with Bridgewater now has an agentic-AI gap. This agent turns that insight into pipeline.

## What it does

Three chained stages, each with structured output:

1. **Find** — identifies Bridgewater competitors (global macro, multi-strat, quant) and their public AI-adoption signals: ML hiring, research labs, exec commentary
2. **Map** — for each fund, maps its likely pain to one specific LangChain capability (LangGraph agent control, LangSmith evals/observability) and frames the Bridgewater story without overclaiming
3. **Sequence** — drafts a personalized 3-touch outreach sequence per fund: tailored hook, concrete proof point, believable breakup. Under 120 words per touch, one CTA

## Why I built it

I sold Bloomberg's quant platform to these exact buyers for four years. The "your competitor already built this" play is the strongest opener in enterprise sales — but doing the research and first drafts manually caps you at a handful of accounts a day. An agent removes the cap; the SDR keeps the judgment and the conversations.

## Run it

No API key — see the full output instantly:

```bash
python competitor_prospector.py --demo
```

Live mode:

```bash
pip install langchain langchain-openai   # or langchain-anthropic
export OPENAI_API_KEY=sk-...             # or ANTHROPIC_API_KEY
python competitor_prospector.py --n 5
```

Works with OpenAI or Anthropic — it auto-detects whichever key is set via `init_chat_model`.

## Bonus: `profile_builder.py`

The copy on my portfolio page was drafted by a second LangChain pipeline in this repo: structured CV facts + the job description → positioning (structured output) → page copy. Same principle — automate the draft, keep the judgment.

## What I'd build next

- A web-search tool node so the FIND stage cites live signals (job posts, engineering blogs) instead of model knowledge
- A LangGraph state machine with a human-approval checkpoint before any sequence is sent
- LangSmith tracing on every run, logging which sequences get replies and feeding winners back as few-shot examples
