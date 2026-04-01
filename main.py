"""
main.py — unified LLM client with streaming support.

Project 2 additions over Project 1:
  - import time
  - LLMClient.stream()         — public streaming method, same API as chat()
  - LLMClient._stream_anthropic()  — Anthropic SSE streaming
  - LLMClient._stream_openai()     — OpenAI streaming
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from openai import OpenAI


# ---------------------------------------------------------------------------
# Cost tables (USD per 1M tokens, as of mid-2025)
# ---------------------------------------------------------------------------

COST_PER_MILLION = {
    "claude-opus-4-5":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-5":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.00},
    "gpt-4o":                    {"input":  2.50, "output": 10.00},
    "gpt-4o-mini":               {"input":  0.15, "output":  0.60},
    "o1-mini":                   {"input":  3.00, "output": 12.00},
}

Provider = Literal["anthropic", "openai"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        rates = COST_PER_MILLION.get(self.model)
        if not rates:
            return 0.0
        return (
            self.input_tokens  / 1_000_000 * rates["input"] +
            self.output_tokens / 1_000_000 * rates["output"]
        )

    def __str__(self) -> str:
        cost = f"${self.cost_usd:.6f}" if self.cost_usd else "cost unknown"
        return (
            f"Tokens — input: {self.input_tokens:,}  "
            f"output: {self.output_tokens:,}  "
            f"total: {self.total_tokens:,}  |  {cost}"
        )


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: Provider
    usage: TokenUsage

    def __str__(self) -> str:
        return self.content


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class ConversationSession:
    history: list[Message] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_user(self, content: str) -> None:
        self.history.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.history.append(Message(role="assistant", content=content))

    def update_totals(self, usage: TokenUsage) -> None:
        self.total_input_tokens  += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost_usd      += usage.cost_usd

    def to_api_messages(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.history]

    def summary(self) -> str:
        turns = sum(1 for m in self.history if m.role == "user")
        return (
            f"Session: {turns} turn(s) | "
            f"input: {self.total_input_tokens:,} | "
            f"output: {self.total_output_tokens:,} | "
            f"total cost: ${self.total_cost_usd:.6f}"
        )

    def reset(self) -> None:
        self.history.clear()
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.total_cost_usd      = 0.0


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    One client, two providers. Supports both batch (chat) and streaming (stream).
    The public API is identical — swap chat() for stream() anywhere.
    """

    DEFAULT_MODELS: dict[Provider, str] = {
        "anthropic": "claude-sonnet-4-5",
        "openai":    "gpt-4o-mini",
    }

    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
    ) -> None:
        self.provider      = provider
        self.model         = model or self.DEFAULT_MODELS[provider]
        self.system_prompt = system_prompt
        self.max_tokens    = max_tokens

        if provider == "anthropic":
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        else:
            self._client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )

    # ── Batch (original Project 1) ──────────────────────────────────────────

    def new_session(self) -> ConversationSession:
        return ConversationSession()

    def chat(
        self,
        user_message: str,
        session: ConversationSession | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        sys_prompt = system_prompt or self.system_prompt

        if session:
            session.add_user(user_message)
            messages = session.to_api_messages()
        else:
            messages = [{"role": "user", "content": user_message}]

        if self.provider == "anthropic":
            response = self._call_anthropic(messages, sys_prompt)
        else:
            response = self._call_openai(messages, sys_prompt)

        if session:
            session.add_assistant(response.content)
            session.update_totals(response.usage)

        return response

    def compare(
        self,
        user_message: str,
        other_client: "LLMClient",
        system_prompt: str | None = None,
    ) -> tuple[LLMResponse, LLMResponse]:
        r1 = self.chat(user_message, system_prompt=system_prompt)
        r2 = other_client.chat(user_message, system_prompt=system_prompt)
        return r1, r2

    def _call_anthropic(self, messages, system_prompt) -> LLMResponse:
        raw = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        content = raw.content[0].text
        usage   = TokenUsage(
            input_tokens=raw.usage.input_tokens,
            output_tokens=raw.usage.output_tokens,
            model=self.model,
        )
        return LLMResponse(content=content, model=self.model,
                           provider="anthropic", usage=usage)

    def _call_openai(self, messages, system_prompt) -> LLMResponse:
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        raw = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
        )
        content = raw.choices[0].message.content
        usage   = TokenUsage(
            input_tokens=raw.usage.prompt_tokens,
            output_tokens=raw.usage.completion_tokens,
            model=self.model,
        )
        return LLMResponse(content=content, model=self.model,
                           provider="openai", usage=usage)

    # ── Streaming (Project 2) ───────────────────────────────────────────────

    def stream(
        self,
        user_message: str,
        session: ConversationSession | None = None,
        system_prompt: str | None = None,
        show_metrics: bool = True,
    ) -> LLMResponse:
        """
        Identical signature to chat() — swap them freely.
        Prints tokens as they arrive, then returns the full LLMResponse.
        """
        sys_prompt = system_prompt or self.system_prompt

        if session:
            session.add_user(user_message)
            messages = session.to_api_messages()
        else:
            messages = [{"role": "user", "content": user_message}]

        if self.provider == "anthropic":
            response = self._stream_anthropic(messages, sys_prompt, show_metrics)
        else:
            response = self._stream_openai(messages, sys_prompt, show_metrics)

        if session:
            session.add_assistant(response.content)
            session.update_totals(response.usage)

        return response

    def _stream_anthropic(self, messages, system_prompt, show_metrics) -> LLMResponse:
        """
        Anthropic streaming uses a context manager.
        .text_stream yields one text chunk per SSE event.
        Token counts come from .get_final_message() after the stream closes.

        TTFT = time to first token — the most important UX latency metric.
        It measures how long the user waits before seeing ANY output.
        """
        t_start = time.time()
        ttft: float | None = None
        full_text = ""

        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                if chunk:
                    if ttft is None:
                        ttft = time.time() - t_start
                    print(chunk, end="", flush=True)  # flush= forces immediate print
                    full_text += chunk

        print()  # move to next line when stream ends

        final = stream.get_final_message()
        usage = TokenUsage(
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            model=self.model,
        )

        if show_metrics:
            total = time.time() - t_start
            tps   = usage.output_tokens / total if total > 0 else 0
            print(
                f"\n  TTFT: {ttft:.2f}s | "
                f"total: {total:.2f}s | "
                f"{tps:.1f} tok/s | "
                f"{usage}"
            )

        return LLMResponse(content=full_text, model=self.model,
                           provider="anthropic", usage=usage)

    def _stream_openai(self, messages, system_prompt, show_metrics) -> LLMResponse:
        """
        OpenAI streaming uses stream=True.
        Each chunk is a ChatCompletionChunk — text lives at
        chunk.choices[0].delta.content (can be None, so guard with `or ""`).

        OpenAI does NOT return token counts during streaming.
        We use a word-count heuristic (words * 1.33) as an estimate.
        For exact counts, use the tiktoken library.
        """
        full_messages = [{"role": "system", "content": system_prompt}, *messages]

        t_start = time.time()
        ttft: float | None = None
        full_text = ""

        with self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=full_messages,
            stream=True,
        ) as stream:
            for chunk in stream:
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t_start
                    print(piece, end="", flush=True)
                    full_text += piece

        print()

        estimated_output = int(len(full_text.split()) * 1.33)
        usage = TokenUsage(
            input_tokens=0,
            output_tokens=estimated_output,
            model=self.model,
        )

        if show_metrics:
            total = time.time() - t_start
            tps   = estimated_output / total if total > 0 else 0
            print(
                f"\n  TTFT: {ttft:.2f}s | "
                f"total: {total:.2f}s | "
                f"{tps:.1f} tok/s (estimated) | "
                f"output ~{estimated_output} tokens"
            )

        return LLMResponse(content=full_text, model=self.model,
                           provider="openai", usage=usage)