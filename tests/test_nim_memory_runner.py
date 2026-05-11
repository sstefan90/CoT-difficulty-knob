"""Regression tests for the Nim memory system + runner integration.

These tests guard against the bug where memory.update() was called with the
pre-move state instead of the post-move state, causing the LLM to see stale
pile values in "Piles after:" despite the board section showing the correct
current state.

Isolation: all tests run without LLM calls (NimState + LastMoveMemory only).
"""
from __future__ import annotations

import asyncio

import pytest

from cot_knob.games.nim import NimState, _encode
from cot_knob.memory.base import TurnRecord
from cot_knob.memory.last_move import LastMoveMemory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simulate_runner_turn(
    state: NimState,
    move_id: int,
    memory: LastMoveMemory,
    player: int | None = None,
) -> NimState:
    """Mirror the fixed runner.py order: apply_move → memory.update.

    This is the CORRECT order after the 2026-05-09 bug fix.
    """
    player_this_turn = player if player is not None else state.current_player
    move_str = state.move_to_str(move_id)
    new_state = state.apply_move(move_id)
    rec = TurnRecord(
        turn_idx=state.turn_idx,
        player=player_this_turn,
        move_str=move_str,
        state_after_serialized=new_state.to_serializable(),  # post-move ✓
    )
    memory.update(rec)
    return new_state


def _memory_piles(memory: LastMoveMemory) -> list[int]:
    """Extract the pile list stored in the last TurnRecord."""
    assert memory._last is not None
    return memory._last.state_after_serialized["piles"]


# ── Core fix regression ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memory_shows_post_move_piles():
    """Memory must reflect the pile sizes AFTER the move, not before.

    Scenario: piles=[3,5,7], opponent takes 6 from C.
    Expected post-move: C=1.  Pre-move (old bug): C=7.
    """
    state = NimState((3, 5, 7))
    memory = LastMoveMemory()

    # take 6 from C (pile index 2, stones=6)
    move_id = _encode(2, 6)
    state = _simulate_runner_turn(state, move_id, memory)

    snap = await memory.render()
    # post-move piles must match
    assert _memory_piles(memory) == [3, 5, 1], (
        "Memory recorded pre-move state — the bug is back!"
    )
    assert "C=1" in snap.text, f"Expected C=1 in memory text, got: {snap.text!r}"
    assert "C=7" not in snap.text, f"Stale C=7 found in memory text: {snap.text!r}"


@pytest.mark.asyncio
async def test_memory_board_and_memory_consistent_after_each_turn():
    """The board the model sees and memory_text must always agree.

    Walk a full 3-turn sequence and check that for every LLM turn the
    current board state matches what memory records as 'Piles after'.
    """
    state = NimState((3, 5, 7))
    memory = LastMoveMemory()

    moves = [
        _encode(2, 6),  # take 6 from C: [3,5,7] → [3,5,1]
        _encode(0, 2),  # take 2 from A: [3,5,1] → [1,5,1]
        _encode(1, 4),  # take 4 from B: [1,5,1] → [1,1,1]
    ]

    for move_id in moves:
        state = _simulate_runner_turn(state, move_id, memory)
        snap = await memory.render()
        recorded_piles = _memory_piles(memory)
        actual_piles = list(state.piles)
        assert recorded_piles == actual_piles, (
            f"Memory has {recorded_piles} but board is {actual_piles}: {snap.text!r}"
        )


@pytest.mark.asyncio
async def test_memory_empty_before_first_move():
    memory = LastMoveMemory()
    snap = await memory.render()
    assert "no moves yet" in snap.text.lower()
    assert snap.kind == "last_move"


@pytest.mark.asyncio
async def test_memory_reset_clears_state():
    """LastMoveMemory.reset() must return to empty-game state."""
    state = NimState((3, 5, 7))
    memory = LastMoveMemory()
    state = _simulate_runner_turn(state, _encode(2, 6), memory)

    memory.reset()
    snap = await memory.render()
    assert "no moves yet" in snap.text.lower()
    assert memory._last is None


@pytest.mark.asyncio
async def test_memory_records_correct_player_label():
    """Player label in memory text must match who actually moved."""
    state = NimState((3, 5, 7))  # Player 1 moves first
    memory = LastMoveMemory()

    # Player 1 takes 1 from A
    assert state.current_player == 1
    state = _simulate_runner_turn(state, _encode(0, 1), memory)
    snap = await memory.render()
    assert "Player 1" in snap.text, f"Expected 'Player 1' in: {snap.text!r}"

    # Player 2 takes 1 from B
    assert state.current_player == -1
    state = _simulate_runner_turn(state, _encode(1, 1), memory)
    snap = await memory.render()
    assert "Player 2" in snap.text, f"Expected 'Player 2' in: {snap.text!r}"


@pytest.mark.asyncio
async def test_memory_pile_labels_correct():
    """Pile labels A, B, C must map to indices 0, 1, 2."""
    state = NimState((3, 5, 7))
    memory = LastMoveMemory()
    # Take all 3 from A
    state = _simulate_runner_turn(state, _encode(0, 3), memory)
    snap = await memory.render()
    assert "A=0" in snap.text, f"Expected A=0 in: {snap.text!r}"
    assert "B=5" in snap.text, f"Expected B=5 in: {snap.text!r}"
    assert "C=7" in snap.text, f"Expected C=7 in: {snap.text!r}"


# ── Parser / extraction edge cases ───────────────────────────────────────────

class TestNimMoveTagExtraction:
    """Test _extract_nim_move_tag and str_to_move edge cases."""

    def setup_method(self):
        self.state = NimState((3, 5, 7))

    def test_formal_tag_last_occurrence_wins(self):
        """When multiple MOVE tags appear, the LAST one is used."""
        from cot_knob.agents.llm_agent import _extract_nim_move_tag
        text = "I'll take 2 from A.\nMOVE: pile=A take=2\nWait, better move:\nMOVE: pile=C take=4"
        tag = _extract_nim_move_tag(text)
        assert tag == ("C", 4), f"Last MOVE tag should win, got {tag}"

    def test_formal_tag_case_insensitive(self):
        from cot_knob.agents.llm_agent import _extract_nim_move_tag
        text = "move: pile=b take=3"
        tag = _extract_nim_move_tag(text)
        assert tag is not None
        assert tag[0] == "B" and tag[1] == 3

    def test_tag_with_nonexistent_pile_falls_through(self):
        """Pile 'D' doesn't exist in [3,5,7]; str_to_move returns None."""
        result = self.state.str_to_move("take 1 from D")
        assert result is None

    def test_tag_with_overcommit_falls_through(self):
        """Taking more stones than available returns None."""
        result = self.state.str_to_move("take 8 from A")  # pile A=3
        assert result is None

    def test_tag_with_zero_take_is_invalid(self):
        result = self.state.str_to_move("take 0 from A")
        assert result is None

    def test_no_move_tag_returns_none(self):
        from cot_knob.agents.llm_agent import _extract_nim_move_tag
        text = "The nim sum is 3 XOR 5 XOR 7 = 1."
        assert _extract_nim_move_tag(text) is None

    def test_str_to_move_various_formats(self):
        """str_to_move should handle multiple phrasing formats."""
        legal = set(self.state.legal_moves())
        cases = [
            "take 2 from A",
            "2 from A",
            "from A take 2",
        ]
        for s in cases:
            m = self.state.str_to_move(s)
            assert m is not None and m in legal, f"Failed for {s!r}: got {m}"


# ── NimState correctness ───────────────────────────────────────────────────────

class TestNimStateCorrectness:

    def test_terminal_state_winner(self):
        """The player who takes the last stone wins; loser is current player."""
        state = NimState((0, 0, 1))  # only 1 stone left
        move = _encode(2, 1)  # take it
        new_state = state.apply_move(move)
        assert new_state.is_terminal()
        # current_player of new_state is player 2 (who can't move) → loses
        # → winner is player 1 (who just moved)
        assert new_state.winner() == 1

    def test_apply_move_does_not_mutate(self):
        """NimState is immutable; apply_move returns a new object."""
        state = NimState((3, 5, 7))
        new_state = state.apply_move(_encode(0, 1))
        assert state.piles == (3, 5, 7)
        assert new_state.piles == (2, 5, 7)

    def test_legal_moves_empty_at_terminal(self):
        state = NimState((0, 0, 0))
        assert state.legal_moves() == []

    def test_nim_sum_in_serializable(self):
        state = NimState((3, 5, 7))
        s = state.to_serializable()
        assert s["nim_sum"] == (3 ^ 5 ^ 7)

    def test_player_alternates(self):
        state = NimState((3, 5, 7))
        assert state.current_player == 1
        state = state.apply_move(_encode(0, 1))
        assert state.current_player == -1
        state = state.apply_move(_encode(0, 1))
        assert state.current_player == 1


# ── NimOptimal oracle correctness ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_nim_optimal_winning_position():
    """From [3,5,7] (nim_sum=1≠0), optimal agent must pick a winning move."""
    from cot_knob.agents.nim_optimal import NimOptimalAgent
    agent = NimOptimalAgent(seed=0)
    state = NimState((3, 5, 7))  # nim_sum = 3^5^7 = 1 → winning for current player
    tel = await agent.choose(state)
    # After the winning move nim_sum must be 0
    new_state = state.apply_move(tel.chosen_move)
    ns = 0
    for p in new_state.piles:
        ns ^= p
    assert ns == 0, f"Nim-optimal didn't leave nim_sum=0; chose {state.move_to_str(tel.chosen_move)}"


@pytest.mark.asyncio
async def test_nim_optimal_losing_position_plays_legal():
    """From [1,2,3] (nim_sum=0), any move is losing; agent picks a legal one."""
    from cot_knob.agents.nim_optimal import NimOptimalAgent
    agent = NimOptimalAgent(seed=0)
    state = NimState((1, 2, 3))  # nim_sum = 0 → losing position
    tel = await agent.choose(state)
    assert tel.chosen_move in state.legal_moves()
    # uct_top3 should flag these all as 0.5 (uniform, losing)
    for entry in tel.uct_top3:
        assert entry["win_rate"] == 0.5


@pytest.mark.asyncio
async def test_nim_optimal_oracle_ranks_winning_moves_first():
    """Oracle must rank nim-winning moves with win_rate=1.0 at top of list."""
    from cot_knob.agents.nim_optimal import NimOptimalAgent
    agent = NimOptimalAgent(top_k=None, seed=0)
    state = NimState((3, 5, 7))
    tel = await agent.choose(state)
    assert tel.uct_top3[0]["win_rate"] == 1.0
    # All winning moves should come before losing ones
    rates = [e["win_rate"] for e in tel.uct_top3]
    assert rates == sorted(rates, reverse=True)
