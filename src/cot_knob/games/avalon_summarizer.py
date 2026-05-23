"""Sonnet-based game-state summarizer for Avalon.

After each quest resolves, ``AvalonSummarizer.summarize()`` calls
``claude-sonnet-4-6`` (via the Anthropic SDK) to produce a compact
factual record of the current game state.

Design constraints:
  - PUBLIC KNOWLEDGE ONLY. The summary contains only facts observable by all
    players: quest outcomes, team compositions, vote tallies. No inferences,
    no suspicion analysis. Deductive reasoning is the player model's job and
    must stay inside its CoT budget. Pre-cooked conclusions in the summary
    would contaminate the budget treatment by giving low-B players high-quality
    deductions for free.
  - ROLE SEPARATION. llm_role is passed only to label which player is the LLM.
    For Merlin cells, private evil-player knowledge is NOT included in the
    summary — Merlin's system prompt already supplies that at decision time via
    _render_knowledge(). Keeping it out of the summary ensures Servant and
    Merlin cells produce structurally comparable summaries.
  - STRUCTURED OUTPUT. A fixed template avoids length variance across
    summarizer models (Sonnet vs future Llama ablation).
  - FACTS ONLY guard. The prompt explicitly prohibits inference and
    hallucination, especially important for the Llama-as-summarizer ablation.

The summary is stored in:
  - The SQLite ``summaries`` table (kind="public") via ``Store.insert_summary``.
  - The JSONL trace as a ``"summary"`` event.
  - The model call in ``model_calls`` with role="summarize".

This is a logging/analysis aid only. Decision prompts use the full history
(via ``_render_history``), not the summary. The summaries table enables
post-hoc summary-stability analysis (docs/tracking_schema.md § summaries).
"""

from __future__ import annotations

import time
from typing import Any

# ── Summarizer prompt (facts-only, structured template) ────────────────────
#
# Design rationale for each constraint:
# - "Use ONLY facts present in the event log": prevents hallucination of
#   vote outcomes, especially critical for the Llama-summarizer ablation.
# - "Do NOT infer, deduce, or speculate": keeps suspicion analysis inside
#   the player model's CoT budget where it belongs.
# - Structured sections: length-stable across summarizer models, making the
#   summarizer ablation (L-S-llamasumm) a controlled comparison.
# - llm_player_idx label: lets analysts identify the LLM player's specific
#   decisions without any reasoning about role.

_SUMMARIZER_SYSTEM = """\
You are an Avalon game recorder. Your job is to produce a structured factual
log of a game-in-progress.

CRITICAL CONSTRAINTS:
1. Use ONLY facts present in the event log. Do NOT infer, deduce, or speculate.
2. Do NOT write "Player X is suspicious", "Player X is likely evil", or any
   judgement about player alignment. Record observable facts only.
3. Do NOT use Merlin's private knowledge even if the role is provided. Write
   only what ALL players can observe: quest outcomes, vote tallies, team lists.
4. Fill every section. Write "none" if a section is empty.
5. Do not add prose outside the template sections below."""

_SUMMARIZER_USER_TEMPLATE = """\
The tracked player is Player {llm_player_idx} ({llm_role}).
Event log (pre-formatted natural language — one event per line):
{event_log}

Fill in this template exactly. Replace [...] with the facts from the event log.

== QUEST LOG ==
[One line per completed quest:
  QN: [SUCCESS|FAIL] | Team: [player list] | Fail-vote count: N]

== VOTE RECORD ==
[One line per team proposal:
  QN attempt R: Leader=PX proposed [team] | approve: [player list] | reject: [player list] | [ACCEPTED|REJECTED]]

== TEAM STATS ==
Players on 1+ failed quests: [list or "none"]
Players never on any quest team yet: [list or "none"]
Tracked player (P{llm_player_idx}) quest appearances: [comma-separated QN SUCCESS/FAIL or "none"]

== SCORE ==
Good wins: N | Evil wins: N | Quests remaining: N"""


def _format_event_log(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for ev in history:
        if ev["type"] == "team_vote_result":
            q = ev["quest_turn"] + 1
            r = ev["round"] + 1
            votes = ", ".join(
                f"P{i}={'approve' if v else 'reject'}"
                for i, v in enumerate(ev["votes"])
            )
            result = "ACCEPTED" if ev["accepted"] else "REJECTED"
            lines.append(
                f"Quest {q} attempt {r}: P{ev['leader']} proposed {sorted(ev['team'])}. "
                f"Votes [{votes}] → {result}."
            )
        elif ev["type"] == "quest_result":
            q = ev["quest_turn"] + 1
            result = "SUCCESS" if ev["succeeded"] else "FAILED"
            lines.append(
                f"Quest {q}: team {ev['team']} → {result} ({ev['num_fails']} fail votes)."
            )
        elif ev["type"] == "discussion":
            q = ev["quest_turn"] + 1
            r = ev["round"] + 1
            stmts = "  ".join(
                f"P{p}: '{s[:80]}'" for p, s in ev["statements"].items() if s
            )
            lines.append(f"Quest {q} attempt {r} discussion: {stmts}")
    return "\n".join(lines) if lines else "(no events yet)"


class AvalonSummarizerSGLang:
    """SGLang-backend summarizer for Llama-as-summarizer ablation.

    Identical prompt template and logging as AvalonSummarizer; only the
    transport layer differs (OpenAI-compatible SGLang instead of Anthropic SDK).
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:30000",
    ) -> None:
        import httpx  # imported lazily
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=120.0)

    async def summarize(
        self,
        *,
        trial_id: str,
        after_quest_idx: int,
        history: list[dict],
        quest_results: list[bool],
        llm_player_idx: int,
        llm_role: str,
        store: Any,
        jsonl: Any,
    ) -> str:
        event_log = _format_event_log(history)
        user_msg = _SUMMARIZER_USER_TEMPLATE.format(
            llm_player_idx=llm_player_idx,
            llm_role=llm_role,
            event_log=event_log or "(no events yet)",
        )

        t0 = time.monotonic()
        resp = await self._http.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SUMMARIZER_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        latency_ms = (time.monotonic() - t0) * 1000
        data = resp.json()

        summary_text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        n_tokens = int(usage.get("completion_tokens", len(summary_text.split())))
        n_input_tokens = int(usage.get("prompt_tokens", 0))

        store.insert_summary(
            trial_id=trial_id,
            after_turn_idx=after_quest_idx,
            summary_text=summary_text,
            n_tokens_est=n_tokens,
            kind="public",
        )
        store.insert_model_call(
            trial_id=trial_id,
            turn_id=None,
            role="summarize",
            prompt_text=user_msg,
            response_text=summary_text,
            n_input_tokens=n_input_tokens,
            n_output_tokens=n_tokens,
            finish_reason=data["choices"][0].get("finish_reason") or "stop",
            temperature=0.0,
            seed=None,
            latency_ms=latency_ms,
            backend="sglang",
            model=self._model,
        )
        jsonl.write(trial_id, "summary", {
            "after_quest_idx": after_quest_idx,
            "summary_text": summary_text,
            "n_tokens": n_tokens,
            "latency_ms": latency_ms,
            "model": self._model,
        })
        return summary_text

    async def aclose(self) -> None:
        await self._http.aclose()


class AvalonSummarizer:
    """Calls Sonnet to summarize Avalon game state after each quest."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        import anthropic  # imported lazily; anthropic is optional at module level
        self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def summarize(
        self,
        *,
        trial_id: str,
        after_quest_idx: int,  # 0-based quest index just completed
        history: list[dict[str, Any]],
        quest_results: list[bool],
        llm_player_idx: int,
        llm_role: str,  # used only as a label, NOT for private knowledge
        store: Any,   # Store — avoid circular import
        jsonl: Any,   # JSONLWriter
    ) -> str:
        """Generate and persist a public-knowledge game-state summary.

        The summary contains only facts observable by ALL players. Merlin's
        private evil-player knowledge is intentionally excluded even when
        llm_role=="Merlin" — the system prompt at decision time already
        supplies that via _render_knowledge(). This keeps Servant and Merlin
        summaries structurally comparable.
        """
        event_log = _format_event_log(history)
        user_msg = _SUMMARIZER_USER_TEMPLATE.format(
            llm_player_idx=llm_player_idx,
            llm_role=llm_role,
            event_log=event_log or "(no events yet)",
        )

        t0 = time.monotonic()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,  # generous for structured template; prose < 300 words
            system=_SUMMARIZER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        latency_ms = (time.monotonic() - t0) * 1000

        summary_text = response.content[0].text if response.content else ""
        n_tokens = response.usage.output_tokens if response.usage else len(summary_text.split())

        # Log to SQLite summaries table. kind="public" signals facts-only design.
        store.insert_summary(
            trial_id=trial_id,
            after_turn_idx=after_quest_idx,
            summary_text=summary_text,
            n_tokens_est=n_tokens,
            kind="public",
        )

        # Log the model call for completeness.
        store.insert_model_call(
            trial_id=trial_id,
            turn_id=None,
            role="summarize",
            prompt_text=user_msg,
            response_text=summary_text,
            n_input_tokens=response.usage.input_tokens if response.usage else 0,
            n_output_tokens=n_tokens,
            finish_reason=response.stop_reason or "stop",
            temperature=0.0,
            seed=None,
            latency_ms=latency_ms,
            backend="anthropic",
            model=self._model,
        )

        # Write to JSONL trace.
        jsonl.write(trial_id, "summary", {
            "after_quest_idx": after_quest_idx,
            "summary_text": summary_text,
            "n_tokens": n_tokens,
            "latency_ms": latency_ms,
            "model": self._model,
        })

        return summary_text

    async def aclose(self) -> None:
        await self._client.close()
