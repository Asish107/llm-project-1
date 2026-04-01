"""
demo_streaming.py — demonstrates Project 2 streaming features.

Run with:
    python demo_streaming.py

Sections:
    1. Basic streaming — watch tokens arrive live
    2. stream() vs chat() — same prompt, feel the difference
    3. Streaming with session — multi-turn with live output
    4. TTFT comparison — Haiku vs Sonnet side by side
    5. Graceful interruption — Ctrl+C handling
"""

from dotenv import load_dotenv
load_dotenv()

import time
from main import LLMClient


def separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ---------------------------------------------------------------------------
# 1. Basic streaming — the simplest possible example
# ---------------------------------------------------------------------------

separator("1. Basic streaming")

client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")

print("Streaming response:\n")
response = client.stream("Explain what a data pipeline is in 3 sentences.")

# stream() returns the same LLMResponse as chat() — use it normally after
print(f"\nFull response captured: {len(response.content)} characters")


# ---------------------------------------------------------------------------
# 2. stream() vs chat() — identical API, very different feel
# ---------------------------------------------------------------------------

separator("2. stream() vs chat() — same prompt, different experience")

prompt = "What are the three most important skills for a data engineer in 2025?"

print("── chat() — waits for full response before printing:")
t0 = time.time()
r_batch = client.chat(prompt)
print(r_batch.content)
print(f"  Wait time before first character: {time.time() - t0:.2f}s\n")

print("── stream() — prints as tokens arrive:")
response = client.stream(prompt)

# Key insight: total time is similar, but perceived speed is completely different.
# TTFT for stream() is typically 0.3–0.8s vs 2–4s wait for chat().


# ---------------------------------------------------------------------------
# 3. Streaming with a session — history works exactly the same
# ---------------------------------------------------------------------------

separator("3. Multi-turn streaming session")

session_client = LLMClient(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    system_prompt="You are a concise data engineering mentor. Keep answers to 2-3 sentences.",
)
session = session_client.new_session()

turns = [
    "What is Apache Airflow used for?",
    "How does it compare to Prefect?",
    "Which one should I learn first?",
]

for question in turns:
    print(f"\nUser: {question}")
    print("Assistant: ", end="")
    session_client.stream(question, session=session, show_metrics=False)

print(f"\n{session.summary()}")


# ---------------------------------------------------------------------------
# 4. TTFT comparison — Haiku vs Sonnet
#    This is the metric that matters most for real-time UX
# ---------------------------------------------------------------------------

separator("4. TTFT comparison — Haiku vs Sonnet")

prompt = (
    "List five best practices for designing a scalable data warehouse schema. "
    "Be specific and practical."
)

haiku  = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
sonnet = LLMClient(provider="anthropic", model="claude-sonnet-4-5")

print("── Haiku:")
r_haiku = haiku.stream(prompt)

print("\n── Sonnet:")
r_sonnet = sonnet.stream(prompt)

# After running this, look at the TTFT values in the metrics line.
# Haiku is typically faster to first token AND cheaper — the right default
# for streaming interfaces. Sonnet is worth the cost when output quality matters.


# ---------------------------------------------------------------------------
# 5. Graceful interruption — wrap in try/except for production use
# ---------------------------------------------------------------------------

separator("5. Graceful interruption (press Ctrl+C to test)")

print("Streaming a long response — press Ctrl+C to interrupt:\n")

try:
    client.stream(
        "Explain the full history of SQL from its invention in the 1970s "
        "to modern analytical databases. Be detailed.",
        show_metrics=True,
    )
except KeyboardInterrupt:
    print("\n\n  [Stream interrupted by user — this is handled gracefully]")
    print("  In production: save partial response, log the interruption,")
    print("  and return whatever was collected so far.")