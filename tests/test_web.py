import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jesonmor import engine as E
from jesonmor.black_ai import PlanEnginePlayer
from jesonmor.board import BLACK, CENTER, WHITE, Position, exchange_winner, sq, winner
from jesonmor.web import Game


def fast_ai():
    return PlanEnginePlayer(E.HeuristicPlayer(E.Heuristic(), 1))


def test_moves_and_undo():
    g = Game(E.Heuristic(), ai=fast_ai(), out=None)
    try:
        g.human(sq(0, 0), sq(0, 1))                                   # a1-a2 is not a knight jump
        assert False
    except ValueError:
        pass
    g.human(sq(2, 0), sq(3, 2))                                       # c1-d3
    assert g.pos.turn == BLACK and g.reply() is not None and g.pos.turn == WHITE
    st = g.state("x")
    assert st["moves"][0] == "c1-d3" and st["turn"] == "w" and st["canUndo"] and st["over"] is None
    assert all(st["board"][s] == 1 for s in st["legal"])
    g.undo()
    assert g.positions == [Position.start()] and g.undos == 1 and g.ai.phase == "plan"


def test_game_is_played_out_to_the_rules_end():
    # a game does not stop when the exchange on the center is decided (exchange_winner): it goes on until a knight
    # survives on e5 or a side has no move (board.winner)
    for seed in (1, 2, 3):
        g, white, rng = Game(E.Heuristic(), ai=fast_ai(), out=None), E.BaselinePlayer(2), random.Random(seed)
        decided_before_end = False
        while not g.over:
            if g.pos.turn == WHITE:
                m, _ = white.move(g.pos, g.history(), rng)
                g.human(*m)
            else:
                g.reply()
            decided_before_end |= not g.over and exchange_winner(g.pos) is not None
        wd, reason = g.over
        assert reason in ("center", "no-move", "repetition", "limit")
        if reason == "center":
            assert winner(g.pos) == wd and g.pos.own()[0] >> CENTER & 1
        assert decided_before_end or reason != "center" or len(g.moves) < 4


def test_background_reply_and_resign():
    import time
    g = Game(E.Heuristic(), ai=fast_ai(), out=None)
    g.human(sq(2, 0), sq(3, 2))
    g.start_reply()
    t = time.time()
    while g.thinking and time.time() - t < 60:
        time.sleep(0.05)
    st = g.state("x")
    assert not st["thinking"] and st["turn"] == "w" and len(st["moves"]) == 2 and st["error"] is None
    g.resign()
    assert g.over == (BLACK, "resign") and g.state("x")["canUndo"] is False


def test_server_keeps_few_games():
    from jesonmor import web as W
    srv = W.Server()
    ids = [srv.new() for _ in range(W.MAX_GAMES + 3)]
    assert len(srv.games) == W.MAX_GAMES and ids[-1] in srv.games and ids[0] not in srv.games


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
