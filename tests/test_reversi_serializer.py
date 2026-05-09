"""Tests for the pure-Python Reversi state."""

from __future__ import annotations

from cot_knob.games.reversi import BOARD_SIZE, ReversiState, initial_state


def test_initial_state_has_4_stones():
    s = initial_state()
    flat = [v for row in s.board for v in row]
    assert sum(1 for v in flat if v == 1) == 2
    assert sum(1 for v in flat if v == -1) == 2
    assert s.current_player == 1
    assert s.turn_idx == 0
    assert not s.is_terminal()


def test_initial_legal_moves_for_black():
    s = initial_state()
    moves = s.legal_moves()
    expected = {s.str_to_move(x) for x in ["d3", "c4", "f5", "e6"]}
    assert set(moves) == expected
    assert all(0 <= m < BOARD_SIZE * BOARD_SIZE for m in moves)


def test_apply_legal_move_flips():
    s = initial_state()
    m = s.str_to_move("d3")
    assert m is not None
    s2 = s.apply_move(m)
    # After Black plays d3, square d3 is Black, e4 stays Black, d4 (was White) is now Black.
    r, c = divmod(m, BOARD_SIZE)
    assert s2.board[r][c] == 1
    d4 = s.str_to_move("d4")
    rr, cc = divmod(d4, BOARD_SIZE)  # type: ignore[arg-type]
    assert s2.board[rr][cc] == 1, "d4 should have been flipped to Black"
    assert s2.current_player == -1
    assert s2.turn_idx == 1


def test_illegal_move_raises():
    s = initial_state()
    a1 = s.str_to_move("a1")
    assert a1 is not None
    try:
        s.apply_move(a1)
    except ValueError:
        pass
    else:
        raise AssertionError("a1 should be illegal in the opening position")


def test_move_str_roundtrip():
    s = initial_state()
    for label in ["a1", "h8", "d3", "e6"]:
        m = s.str_to_move(label)
        assert m is not None
        assert s.move_to_str(m) == label


def test_to_serializable_shape():
    s = initial_state()
    blob = s.to_serializable()
    assert blob["current_player"] == 1
    assert blob["score_black"] == 2
    assert blob["score_white"] == 2
    assert blob["turn_idx"] == 0
    assert sorted(blob["legal_moves"]) == sorted(["d3", "c4", "f5", "e6"])


def test_render_text_contains_board():
    s = initial_state()
    r = s.render_text()
    # Header
    assert "a b c d e f g h" in r
    # Pieces rendered as B / W
    assert "B" in r and "W" in r
    # Rows are numbered 8 → 1 top to bottom: row 8 appears before row 1
    assert r.index("8 ") < r.index("1 ")
    # "You play" label present
    assert "You play" in r
    # Legal moves, piece lists, and orientation header present
    assert "Legal moves:" in r
    assert "Black pieces:" in r
    assert "White pieces:" in r
    assert "row 8 is TOP" in r


def test_full_game_terminates_for_random_play():
    import random

    rng = random.Random(0)
    s: ReversiState = initial_state()
    safety = 200
    pass_streak = 0
    while not s.is_terminal() and safety > 0 and pass_streak < 2:
        moves = s.legal_moves()
        if not moves:
            s = s.apply_move(-1)
            pass_streak += 1
        else:
            pass_streak = 0
            s = s.apply_move(rng.choice(moves))
        safety -= 1
    assert s.is_terminal() or pass_streak >= 2
    assert s.winner() in (1, -1, 0)
