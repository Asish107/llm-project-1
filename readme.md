# Project 1 — unified LLM client

A clean Python wrapper that puts Anthropic and OpenAI behind the same interface,
with multi-turn history and token cost tracking built in from the start.

## Setup

```bash
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

python demo.py
```

## Files

| File | Purpose |
|------|---------|
| `llm_client.py` | The wrapper — `LLMClient`, `ConversationSession`, `TokenUsage`, `LLMResponse` |
| `demo.py` | Runnable walkthrough of every feature |
| `requirements.txt` | `anthropic` and `openai` SDKs |

## Key design decisions

**Why a single `LLMClient` class?**
Anthropic and OpenAI have subtly different APIs. Anthropic takes `system` as a
top-level param; OpenAI folds it into the messages array. By handling that in
one place, the rest of your code never needs to care which provider it's using.

**Why `ConversationSession` as a separate object?**
Keeping history outside the client means you can run multiple independent
conversations from one client, reset a session without recreating the client,
and inspect/serialize the history easily.

**Why track cost from day one?**
Production AI pipelines fail financially before they fail technically. Seeing
`$0.000012` per call in development builds the habit of cost-awareness early.

ode for Project 2: adding streaming"*