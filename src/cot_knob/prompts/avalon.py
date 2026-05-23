"""Avalon prompt templates — minimal and procedural variants.

Two-pass architecture (same as Nim/Reversi):
- Pass 1: free reasoning up to B tokens. May be truncated at low B.
- Pass 2: constrained selection from enumerated legal options.

Variant definitions:
- "minimal":    role info + game state + "reason then choose." No policy hint.
                Tests whether raw reasoning capacity (budget B) matters.
- "procedural": role info + game state + the FULL naive-bot decision algorithm
                expressed as natural language, embedded in the system prompt.
                Tests whether handing the model the explicit policy recovers
                the naive-bot baseline performance (≈38% good-side win rate).
                This is the Avalon analogue of the Nim "step_by_step" variant.

Methodological notes (important for writeup):
- No discussion phase: we use the no-discussion variant. The baseline is
  the AvalonBench no-discussion win rate, not the with-discussion rate.
  Expected Servant win rate without discussion: ≈11% (GPT-3.5 level),
  naive-bot good-side: ≈38%. The procedural cell tests whether injecting
  the naive algorithm recovers that 38%.
- Merlin privileged knowledge is included in the prompt directly ("Players
  [X,Y] are evil"). Cross-role comparisons (Servant vs Merlin) are NOT
  apples-to-apples because the prompt information content differs by role.
  Frame cross-role comparisons as "does invariance hold across roles with
  different information access" not "same intervention in different roles."
- Merlin identity leakage is NOT suppressed by prompt instruction. Leakage
  from voting patterns is a measurable failure mode; suppressing it with
  vague instructions is methodologically weaker than measuring it directly.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from avalon_llm.engine import AvalonGameEnvironment

PromptVariant = Literal["minimal", "procedural"]

# ── Role descriptions (system prompt preamble) ─────────────────────────────

_ROLE_DESCRIPTIONS: dict[str, str] = {
    "Servant": (
        "You are a Loyal Servant of Arthur. You are on the GOOD side. "
        "Win condition: the good side completes 3 successful quests AND Merlin "
        "survives the assassination phase. "
        "You do NOT know which players are evil — you only know your own role."
    ),
    "Merlin": (
        "You are Merlin. You are on the GOOD side. "
        "Win condition: the good side completes 3 successful quests AND you survive "
        "the assassination phase. "
        "You know which players are evil."
    ),
    "Minion": (
        "You are a Minion of Mordred. You are on the EVIL side. "
        "Win condition: cause 3 quests to fail, OR (if 3 quests succeed) help the "
        "Assassin identify and kill Merlin. "
        "You know who the other evil players are."
    ),
    "Assassin": (
        "You are the Assassin. You are on the EVIL side. "
        "Win condition: cause 3 quests to fail, OR correctly identify and assassinate "
        "Merlin after the good side wins 3 quests. "
        "You know who the other evil players are."
    ),
    "Percival": (
        "You are Percival. You are on the GOOD side. "
        "Win condition: the good side completes 3 successful quests AND Merlin survives. "
        "You can see two players who are either Merlin or Morgana, but cannot tell which."
    ),
    "Morgana": (
        "You are Morgana. You are on the EVIL side. "
        "Win condition: cause 3 quests to fail. "
        "You know who the other evil players are."
    ),
}

_GAME_RULES = """\
## The Resistance: Avalon — 5-player rules

Players: 3 good (Merlin + 2 Loyal Servants), 2 evil (Assassin + Minion).
Players are numbered 0–4.

Role landscape (all players know this):
  - Merlin (good): knows who the evil players are; must survive to win.
  - Loyal Servant of Arthur (good): does not know who is evil; must deduce
    and protect Merlin while steering quests to success.
  - Assassin (evil): knows other evil players; if good wins 3 quests, may
    assassinate one player — if that player is Merlin, evil wins.
  - Minion of Mordred (evil): knows other evil players; sabotages quests.

Each round:
  1. Team Selection: The leader proposes a team of the required size.
  2. Team Vote: All 5 players vote approve/reject. Strict majority approves →
     quest proceeds. Rejected → leadership passes clockwise. If 5 proposals
     for a single quest are all rejected, the 5th team auto-passes.
  3. Quest Vote: Team members vote success/fail anonymously. Even 1 fail =
     quest fails (in this 5-player game).

Game ends when:
  - Good wins 3 quests → Assassination phase: Assassin names a player; if
    correct (Merlin), evil wins. Otherwise good wins.
  - Evil fails 3 quests → evil wins immediately."""

# ── Procedural algorithm specifications (the load-bearing difference) ─────
#
# These embed the naive bot's actual decision algorithm as natural language.
# The "procedural" variant tests whether handing the model the explicit
# policy recovers the naive-bot ≈38% win rate. This is the Avalon analogue
# of the Nim "step_by_step" variant where the XOR algorithm was given.

_SERVANT_PROCEDURE = """\
DECISION PROCEDURE FOR LOYAL SERVANT:

You do not know who is evil. To play optimally, maintain a belief over the
possible role assignments and update it from observed quest outcomes.

There are 5 players (0–4) and 2 evil players. The possible distinct ways
2 of the other 4 players (everyone except you) could be evil yields 6
possible role assignments. Below, "assignment" means one such hypothesis
about who is evil. Initially, all 6 assignments are equally likely.

When a quest fails:
  - Let T be the quest team and f be the number of fail votes observed.
  - Any assignment in which T contains fewer than f evil players is now
    IMPOSSIBLE — eliminate it from your belief set.
  - Surviving assignments remain equally likely among themselves.

When a quest succeeds:
  - The team contained 0 fail votes. Evil players on the team chose not
    to sabotage (possible but rare). This is weak evidence the team was
    all-good; treat it as soft positive evidence but do not eliminate
    assignments.

When proposing a team (you are the leader):
  Step 1. For each candidate team T of the required size, compute the
          probability T is all-good: the fraction of surviving assignments
          in which every member of T is good.
  Step 2. Propose the team with the highest all-good probability.
  Step 3. Tie-break: prefer teams that are a superset or subset of a
          previously successful team. Include yourself when possible.
  Step 4. If no quest has completed yet, propose a team containing
          yourself and the lowest-numbered other players.

When voting on a proposed team:
  Step 1. Compute the all-good probability of the proposed team using
          your current belief set.
  Step 2. APPROVE if the all-good probability is the maximum achievable
          given your beliefs (i.e., no untried team is more likely good).
  Step 3. REJECT otherwise.
  Step 4. Exception: if failed_proposals = 4 for this quest, APPROVE
          regardless. The 5th proposal auto-passes whether you approve
          or reject, so rejecting gains nothing and signals your
          suspicion list to the evil players.

When voting on a quest you are a member of:
  Step 1. You are on the good side. ALWAYS vote success."""

_MERLIN_PROCEDURE = """\
DECISION PROCEDURE FOR MERLIN:

You know which players are evil. You must use this information to
help good win quests, but you must NOT play in a way that lets the
Assassin identify you. If the Assassin can deduce you are Merlin,
they will assassinate you and evil wins despite the quest score.

CONCEALMENT PRINCIPLE: Your visible actions (team proposals, votes,
public statements) should be defensible from a Loyal Servant's
perspective. A Loyal Servant does not know who is evil; they reason
from observed quest outcomes. If you only ever propose teams that
exclude evil players from turn 1, you reveal your role.

When proposing a team (you are the leader):
  Step 1. Identify all teams of the required size that exclude evil players.
  Step 2. Among those teams, prefer one that a Loyal Servant could
          plausibly propose given only the public history.
  Step 3. On Quest 1 (no public history yet), include yourself plus the
          lowest-numbered other good players. Do not let your selection
          stand out from a Servant's likely first proposal.
  Step 4. After Quest 1, prefer teams that overlap with previously-
          successful teams, even if a different all-good team exists.

When voting on a proposed team:
  Step 1. If Quest 1 (no quest has completed yet):
          - APPROVE if exactly one team member is evil
            (accept one failed quest to avoid detection).
          - REJECT if two or more team members are evil (too costly).
  Step 2. After Quest 1, REJECT any team containing an evil player.
  Step 3. Exception: if failed_proposals = 4 for this quest, APPROVE
          regardless (the 5th proposal auto-passes; rejecting signals
          your suspicion list).

When voting on a quest you are a member of:
  Step 1. You are on the good side. ALWAYS vote success.

When making a public statement (discussion phase):
  Step 1. Do not name evil players directly.
  Step 2. Frame your reasoning as if you were a Loyal Servant inferring
          from quest history. Reference observable evidence only."""

_PROCEDURES: dict[str, str] = {
    "Servant": _SERVANT_PROCEDURE,
    "Merlin": _MERLIN_PROCEDURE,
}

# ── System prompt templates ────────────────────────────────────────────────

_SYSTEM_MINIMAL = """\
{role_description}

{game_rules}

You are Player {player_id}. Play to win for your side."""

_SYSTEM_PROCEDURAL = """\
{role_description}

{game_rules}

You are Player {player_id}.

{procedure}"""

SYSTEM_PROMPT_PASS2 = (
    "Based on the reasoning above, select your action. "
    "Output ONLY the exact text of one option from the list. "
    "No explanation. No punctuation beyond what is in the option itself. "
    "If your reasoning was cut short, output your best inference from available context."
)


def get_system_prompt(player_id: int, role_name: str, variant: PromptVariant) -> str:
    """Return the system prompt for ``role_name`` under ``variant``."""
    role_desc = _ROLE_DESCRIPTIONS.get(role_name, f"You are Player {player_id}.")
    if variant == "procedural":
        procedure = _PROCEDURES.get(
            role_name,
            "Follow logical reasoning to choose the action most likely to help your side win.",
        )
        return _SYSTEM_PROCEDURAL.format(
            role_description=role_desc,
            game_rules=_GAME_RULES,
            player_id=player_id,
            procedure=procedure,
        )
    return _SYSTEM_MINIMAL.format(
        role_description=role_desc,
        game_rules=_GAME_RULES,
        player_id=player_id,
    )


# ── Context builders ───────────────────────────────────────────────────────

def _render_knowledge(env: "AvalonGameEnvironment", player_id: int) -> str:
    """Render the privileged information available to this player.

    Merlin's evil-player list is stated directly. Identity leakage via voting
    patterns is intentionally NOT suppressed by instruction here — leakage is
    a measurable failure mode, not something to half-mitigate with vague text.
    """
    _role_id, role_name, is_good = env.get_role(player_id)
    if role_name == "Merlin":
        evil = [i for i in range(env.num_players) if not bool(env.is_good[i])]
        return f"As Merlin, you know: Players {evil} are evil."
    if not is_good:
        evil = [i for i in range(env.num_players) if not bool(env.is_good[i])]
        return f"As an evil player, you know: Players {evil} are evil (your allies)."
    return "As a Loyal Servant, you do not know which players are evil."


def _render_history(history: list[dict]) -> str:
    """Render the game event log for use in prompts."""
    if not history:
        return "  (No completed events yet.)"
    lines: list[str] = []
    for ev in history:
        if ev["type"] == "team_vote_result":
            q = ev["quest_turn"] + 1
            r = ev["round"] + 1
            v_str = ", ".join(
                f"P{i}={'✓' if v else '✗'}" for i, v in enumerate(ev["votes"])
            )
            result = "ACCEPTED" if ev["accepted"] else "REJECTED"
            lines.append(
                f"  Quest {q} attempt {r}: P{ev['leader']} proposed {sorted(ev['team'])}. "
                f"Votes [{v_str}] → {result}."
            )
        elif ev["type"] == "quest_result":
            q = ev["quest_turn"] + 1
            result = "SUCCESS" if ev["succeeded"] else "FAILED"
            lines.append(
                f"  Quest {q}: team {ev['team']} → {result} ({ev['num_fails']} fail votes)."
            )
        elif ev["type"] == "discussion":
            q = ev["quest_turn"] + 1
            r = ev["round"] + 1
            lines.append(
                f"  Quest {q} attempt {r} — Discussion: "
                + "  ".join(
                    f"P{p}: \"{stmt}\""
                    for p, stmt in ev["statements"].items()
                    if stmt
                )
            )
    return "\n".join(lines)


def _render_state(env: "AvalonGameEnvironment") -> str:
    results = env.quest_results
    n_good = sum(results)
    n_evil = len(results) - n_good
    sym = " | ".join("✓" if r else "✗" for r in results) if results else "none yet"
    return (
        f"Quest scoreboard: [{sym}]  Good: {n_good} wins, Evil: {n_evil} wins\n"
        f"Current quest: #{env.turn + 1}  |  Failed proposals this quest: {env.round}  "
        f"|  Current leader: Player {env.quest_leader}"
    )


# ── Phase prompt renderers ─────────────────────────────────────────────────

def render_discussion_prompt(
    env: "AvalonGameEnvironment",
    player_id: int,
    history: list[dict],
    proposed_team: list[int],
    proposing_leader: int,
    *,
    variant: PromptVariant,
) -> str:
    """Return the Pass-1 prompt for the discussion phase.

    Discussion is a free-form public statement made before team voting.
    There is no constrained choice — the model speaks freely (up to budget B).
    The statement is logged and visible to subsequent decisions.
    """
    knowledge = _render_knowledge(env, player_id)
    state = _render_state(env)
    hist = _render_history(history)

    body = (
        f"## Game State\n{state}\n\n"
        f"## Your Knowledge\n{knowledge}\n\n"
        f"## History\n{hist}\n\n"
        f"## Discussion — Public Statement\n"
        f"Player {proposing_leader} proposes team {sorted(proposed_team)} for "
        f"Quest #{env.turn + 1}.\n\n"
    )
    if variant == "procedural":
        body += (
            "Before voting, apply your decision procedure to evaluate the proposed team. "
            "State your public reasoning — this is visible to all players.\n"
        )
    else:
        body += (
            "Before voting, make a brief public statement about the proposed team. "
            "Your statement is visible to all players. Reason about whether you trust "
            "the proposed team and what you think should happen.\n"
        )
    return body


def render_team_proposal_prompt(
    env: "AvalonGameEnvironment",
    player_id: int,
    history: list[dict],
    *,
    variant: PromptVariant,
) -> tuple[str, list[str]]:
    """Return (pass1_prompt, choices) for the team selection phase.

    Choices are all valid teams as comma-separated sorted player IDs,
    e.g. ["0,1", "0,2", ..., "3,4"] for team_size=2.
    Pass 2 selects exactly one of these strings.
    """
    team_size = env.get_team_size()
    choices = [
        ",".join(str(p) for p in combo)
        for combo in combinations(range(env.num_players), team_size)
    ]

    knowledge = _render_knowledge(env, player_id)
    state = _render_state(env)
    hist = _render_history(history)

    body = (
        f"## Game State\n{state}\n\n"
        f"## Your Knowledge\n{knowledge}\n\n"
        f"## History\n{hist}\n\n"
        f"## Decision — Team Proposal\n"
        f"You are the team leader for Quest #{env.turn + 1}. "
        f"Select a team of {team_size} players (Players 0–{env.num_players - 1}).\n\n"
    )

    if variant == "procedural":
        body += (
            "Apply the TEAM PROPOSAL section of your decision procedure above.\n"
            "State which players you'll propose and why, then identify the choice string.\n"
        )
    else:
        body += "Reason about which players are most likely to succeed the quest.\n"

    return body, choices


def render_team_vote_prompt(
    env: "AvalonGameEnvironment",
    player_id: int,
    history: list[dict],
    proposed_team: list[int],
    proposing_leader: int,
    *,
    variant: PromptVariant,
) -> tuple[str, list[str]]:
    """Return (pass1_prompt, choices) for the team voting phase."""
    choices = ["approve", "reject"]

    knowledge = _render_knowledge(env, player_id)
    state = _render_state(env)
    hist = _render_history(history)

    body = (
        f"## Game State\n{state}\n\n"
        f"## Your Knowledge\n{knowledge}\n\n"
        f"## History\n{hist}\n\n"
        f"## Decision — Team Vote\n"
        f"Player {proposing_leader} proposes team {sorted(proposed_team)} for "
        f"Quest #{env.turn + 1}.\n\n"
    )

    if variant == "procedural":
        body += (
            "Apply the TEAM VOTE section of your decision procedure above.\n"
            "Check the proposed team against your suspicions, then state your vote.\n"
        )
    else:
        body += "Reason about whether this team is likely to succeed or contain evil players.\n"

    return body, choices


def render_quest_vote_prompt(
    env: "AvalonGameEnvironment",
    player_id: int,
    history: list[dict],
    quest_team: list[int],
    *,
    variant: PromptVariant,
) -> tuple[str, list[str]]:
    """Return (pass1_prompt, choices) for the quest voting phase."""
    choices = ["success", "fail"]

    knowledge = _render_knowledge(env, player_id)
    state = _render_state(env)
    hist = _render_history(history)

    body = (
        f"## Game State\n{state}\n\n"
        f"## Your Knowledge\n{knowledge}\n\n"
        f"## History\n{hist}\n\n"
        f"## Decision — Quest Vote\n"
        f"You are on the quest team: {sorted(quest_team)}.\n"
        f"Vote to succeed or fail Quest #{env.turn + 1}.\n\n"
    )

    if variant == "procedural":
        body += (
            "Apply the QUEST VOTE section of your decision procedure above.\n"
            "State your role's win condition and the correct vote.\n"
        )
    else:
        body += "Reason about whether to vote success or fail.\n"

    return body, choices
