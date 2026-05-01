"""Pure-Python Reversi (Othello) with an 8x8 board.

This is the **Mac-friendly** game backend. It conforms to the same
``GameState`` protocol as the (forthcoming) Ludii-backed implementation,
so the runner doesn't care which one is in use.

Move ids are flat indices ``0..63`` with row-major ordering:

    move_id = row * 8 + col       (0 <= row, col < 8)
    move_str = "<col_letter><row_digit>"   e.g. (3,2) -> "c4"   (a..h, 1..8)

Players: ``+1`` is Black (moves first), ``-1`` is White.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

BOARD_SIZE = 8

# Eight directions for flipping.
_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _initial_board() -> list[list[int]]:
    b = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    b[3][3] = -1
    b[3][4] = 1
    b[4][3] = 1
    b[4][4] = -1
    return b


def _flips_for_move(board: list[list[int]], r: int, c: int, player: int) -> list[tuple[int, int]]:
    """Return all squares that would flip if `player` plays at (r, c). Empty list = illegal."""
    if board[r][c] != 0:
        return []
    flips: list[tuple[int, int]] = []
    opp = -player
    for dr, dc in _DIRS:
        line: list[tuple[int, int]] = []
        rr, cc = r + dr, c + dc
        while _in_bounds(rr, cc) and board[rr][cc] == opp:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and _in_bounds(rr, cc) and board[rr][cc] == player:
            flips.extend(line)
    return flips


@dataclass
class ReversiState:
    """Immutable-from-the-caller's-POV Reversi state."""

    board: list[list[int]] = field(default_factory=_initial_board)
    _current_player: int = 1
    _turn_idx: int = 0
    _consecutive_passes: int = 0

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def turn_idx(self) -> int:
        return self._turn_idx

    def legal_moves(self) -> list[int]:
        moves: list[int] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if _flips_for_move(self.board, r, c, self._current_player):
                    moves.append(r * BOARD_SIZE + c)
        return moves

    def is_terminal(self) -> bool:
        if self._consecutive_passes >= 2:
            return True
        # Or: board full.
        if all(self.board[r][c] != 0 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)):
            return True
        return False

    def winner(self) -> int | None:
        if not self.is_terminal():
            return None
        score_b = sum(1 for row in self.board for v in row if v == 1)
        score_w = sum(1 for row in self.board for v in row if v == -1)
        if score_b > score_w:
            return 1
        if score_w > score_b:
            return -1
        return 0

    def apply_move(self, move: int) -> "ReversiState":
        if move < 0:
            # Sentinel for "pass". Caller must pass only when no legal moves exist.
            if self.legal_moves():
                raise ValueError("apply_move: cannot pass when legal moves exist")
            return ReversiState(
                board=deepcopy(self.board),
                _current_player=-self._current_player,
                _turn_idx=self._turn_idx + 1,
                _consecutive_passes=self._consecutive_passes + 1,
            )

        r, c = divmod(move, BOARD_SIZE)
        if not _in_bounds(r, c):
            raise ValueError(f"apply_move: out of bounds: {move}")
        flips = _flips_for_move(self.board, r, c, self._current_player)
        if not flips:
            raise ValueError(f"apply_move: illegal move {self.move_to_str(move)} for player {self._current_player}")

        new_board = deepcopy(self.board)
        new_board[r][c] = self._current_player
        for rr, cc in flips:
            new_board[rr][cc] = self._current_player
        return ReversiState(
            board=new_board,
            _current_player=-self._current_player,
            _turn_idx=self._turn_idx + 1,
            _consecutive_passes=0,
        )

    def to_serializable(self) -> dict[str, Any]:
        return {
            "board": deepcopy(self.board),
            "current_player": self._current_player,
            "turn_idx": self._turn_idx,
            "score_black": sum(1 for row in self.board for v in row if v == 1),
            "score_white": sum(1 for row in self.board for v in row if v == -1),
            "legal_moves": [self.move_to_str(m) for m in self.legal_moves()],
        }

    def render_text(self) -> str:
        col_header = "  a b c d e f g h"
        rows = [col_header]
        for r in range(BOARD_SIZE):
            cells = []
            for c in range(BOARD_SIZE):
                v = self.board[r][c]
                cells.append("." if v == 0 else ("X" if v == 1 else "O"))
            rows.append(f"{r + 1} " + " ".join(cells))
        score_b = sum(1 for row in self.board for v in row if v == 1)
        score_w = sum(1 for row in self.board for v in row if v == -1)
        rows.append(
            f"turn={self._turn_idx} "
            f"player={'X (Black)' if self._current_player == 1 else 'O (White)'} "
            f"score: X={score_b} O={score_w}"
        )
        return "\n".join(rows)

    def move_to_str(self, move: int) -> str:
        if move < 0:
            return "pass"
        r, c = divmod(move, BOARD_SIZE)
        return f"{chr(ord('a') + c)}{r + 1}"

    def str_to_move(self, s: str) -> int | None:
        s = s.strip().lower()
        if s in ("pass", "-", ""):
            return -1
        if len(s) != 2:
            return None
        col_ch, row_ch = s[0], s[1]
        if not ("a" <= col_ch <= "h") or not ("1" <= row_ch <= "8"):
            return None
        c = ord(col_ch) - ord("a")
        r = int(row_ch) - 1
        return r * BOARD_SIZE + c


def initial_state() -> ReversiState:
    return ReversiState()
