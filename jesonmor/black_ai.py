"""The black player of the finished engine: the opening plan of data/plan_min3, then an engine.

The plan is plan.minimal_reply2 with the support of an occupied ring square (plan.SUPPORT_OCCUPIED, as in
plan.expand_minimal2(support_occupied=True)). Expanded against every white move (routes.expand) it ends every line in
a proven black win (133,736 positions, data/plan_min3/black_wins_ply*.bin) or in one of the 945 canonical positions
(canonical_white.bin, canonical_black.bin); 0 white wins, 0 unfinished lines. The player switches to the engine
exactly where those lines end: at a canonical (or other full) position, or after the plan's proven win (the
engine finds it as well). `switch` records where it happened, so a game can be checked against the expansion.

  PlanEnginePlayer(engine.MeanPlayer(h, 5, top=8, ext=4))     # the black AI of black_test --plan"""
from typing import Dict, Optional, Tuple

from . import plan as PL
from .board import Move, Position
from .routes import _classify
from .search import WIN

ENGINE_AT = ("canonical", "full, non-canonical")


class PlanEnginePlayer:
    def __init__(self, engine, plan_depth: int = 2, name: str = None):
        self.engine, self.plan_depth = engine, plan_depth
        self.name = name or "opening plan + %s" % engine.name
        self.reset()

    @property
    def depth(self) -> int:
        return self.engine.depth

    @depth.setter
    def depth(self, d: int) -> None:                 # the window of play.py changes the depth during a game
        self.engine.depth = d

    def reset(self):
        self.engine.reset()
        self.tt: Dict = {}
        self.phase = "plan"
        self.switch: Optional[Tuple[str, int, Position]] = None    # (reason, ply, position)

    def sync(self, positions) -> None:
        """Phase for a game line after a take-back: the engine if the line already passed a position where the plan's
        lines end (canonical or full position), the plan otherwise (it finds a proven win again by itself)."""
        self.reset()
        for i, p in enumerate(positions):
            if _classify(p) in ENGINE_AT:
                self.phase, self.switch = "engine", ("canonical", i, p)
                break

    def plan_move(self, pos: Position):
        old = PL.SUPPORT_OCCUPIED, PL.ADVANCE_ON_RETREAT
        PL.SUPPORT_OCCUPIED, PL.ADVANCE_ON_RETREAT = True, False
        try:
            return PL.minimal_reply2(pos, None, self.plan_depth, self.tt)
        finally:
            PL.SUPPORT_OCCUPIED, PL.ADVANCE_ON_RETREAT = old

    def move(self, pos: Position, history: Optional[Dict] = None, rng=None) -> Tuple[Move, str]:
        history = history or {}
        ply = max(0, sum(history.values()) - 1)          # plies played (history counts the positions of the game)
        if self.phase == "plan" and _classify(pos) in ENGINE_AT:
            self.phase, self.switch = "engine", ("canonical", ply, pos)
        if self.phase == "engine":
            return self.engine.move(pos, history, rng)
        m, kind, s = self.plan_move(pos)
        if s >= WIN - 1000:                              # the expansion stopped here: proven black win
            self.phase, self.switch = "engine", ("win", ply, pos)
        elif _classify(pos.play(m)) in ENGINE_AT:
            self.phase, self.switch = "engine", ("canonical", ply + 1, pos.play(m))
        return m, "plan: %s" % kind
