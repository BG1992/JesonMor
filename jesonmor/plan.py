"""Fixed opening plan for black (no safety search):
1. (2,8)->(3,6), then (6,8)->(5,6); a white knight on (3,6)/(5,6) is recaptured from (4,8).
2. (1,8)->(0,6)->(2,5) and (7,8)->(8,6)->(6,5).
3. If (2,5)/(6,5) holds a white knight (no jump without a capture), support: (3,8)->(1,7) or (5,8)->(3,7)
   for (2,5), (3,8)->(5,7) or (5,8)->(7,7) for (6,5).
A proven black win is always taken; when the plan is done or blocked black uses best_black_move."""
from .board import sq
from .search import WIN, best_black_move

S = lambda r, c: sq(r, c)
RECAPTURE = [(S(4, 8), S(3, 6)), (S(4, 8), S(5, 6))]
STEPS = [(S(2, 8), S(3, 6)), (S(6, 8), S(5, 6)),
         (S(1, 8), S(0, 6)), (S(0, 6), S(2, 5)),
         (S(7, 8), S(8, 6)), (S(8, 6), S(6, 5))]
SUPPORT = {S(2, 5): [(S(3, 8), S(1, 7)), (S(5, 8), S(3, 7))],
           S(6, 5): [(S(3, 8), S(5, 7)), (S(5, 8), S(7, 7))]}


CAPTURE_B1 = False      # True: a white knight on (2,5)/(6,5) is captured when possible, support only otherwise
GUARD = False           # True: a plan move leaving an attacked black first-order knight undefended is replaced
                        # by a support move ((3,8)->(1,7)/(5,7), (5,8)->(3,7)/(7,7), else any defending move)


SAFE = False            # True: like GUARD but with the full safety test (guard rule + 4-ply alpha-beta)
_TT = {}


def _guarded(pos, mv, kind):
    """Apply the guard rule (or the full safety test) to the chosen move."""
    from .routes import defended, safe
    ok = (lambda m: safe(pos.play(m), _TT)) if SAFE else (lambda m: defended(pos.play(m)))
    if not (GUARD or SAFE) or ok(mv):
        return mv, kind
    legal = pos.moves()
    supports = [m for lst in SUPPORT.values() for m in lst if m in legal]
    for m in supports + legal:
        if ok(m):
            return m, "podparcie" if m in supports else "bezpieczny ruch spoza planu"
    return mv, kind


def plan_reply(pos, white_move, depth: int = 2, tt=None):
    """(move, kind, score) of the fixed plan."""
    m, s = best_black_move(pos, depth, tt)
    if s >= WIN - 1000:
        return m, "wygrana", s
    legal = set(pos.moves())
    for mv in RECAPTURE:
        if mv in legal and pos.w >> mv[1] & 1:
            return mv, "odbicie z (4,8)", 0
    for f, t in STEPS:
        if not pos.b >> f & 1 or pos.b >> t & 1:
            continue                                   # step done or knight not there
        if pos.w >> t & 1 and CAPTURE_B1:              # take the intruder on our first-order square
            caps = sorted(mv for mv in legal if mv[1] == t)
            if caps:
                return caps[0], "bicie intruza", 0
        if pos.w >> t & 1:                             # target held by white: support instead of capturing
            if t in SUPPORT:
                for mv in SUPPORT[t]:
                    if mv in legal:
                        return mv, "podparcie", 0
            continue
        if (f, t) in legal:
            return _guarded(pos, (f, t), "plan") + (0,)
    return m, "zwykła polityka", s


def expand(max_plies: int = 16, log=print, capture: bool = False, guard: bool = False, safe: bool = False):
    import os
    from . import routes
    global CAPTURE_B1, GUARD, SAFE
    CAPTURE_B1, GUARD, SAFE = capture, guard, safe
    name = "plan" + ("_capture" if capture else "") + ("_guard" if guard else "") + ("_safe" if safe else "")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", name)
    return routes.expand(max_plies, 2, out, log, policy=plan_reply)


if __name__ == "__main__":
    import sys
    expand(int(sys.argv[1]) if len(sys.argv) > 1 else 16, log=lambda s: print(s, flush=True),
           capture="--capture" in sys.argv, guard="--guard" in sys.argv, safe="--safe" in sys.argv)


MIN_STEPS = [(S(2, 8), S(3, 6)), (S(6, 8), S(5, 6)),
             (S(1, 8), S(0, 6)), (S(0, 6), S(2, 5)), (S(3, 8), S(1, 7)),
             (S(7, 8), S(8, 6)), (S(8, 6), S(6, 5)), (S(5, 8), S(7, 7))]
RECAPTURERS = {S(3, 6): [S(4, 8)], S(5, 6): [S(4, 8)], S(2, 5): [S(1, 7), S(0, 6)], S(6, 5): [S(7, 7), S(8, 6)]}


def minimal_reply(pos, white_move, depth: int = 2, tt=None):
    """Minimal opening plan: win > recapture on a black first-order square > urgent ring entry > next step."""
    from .board import FIRST_BLACK, KNIGHT
    m, s = best_black_move(pos, depth, tt)
    if s >= WIN - 1000:
        return m, "wygrana", s
    legal = set(pos.moves())
    for t in FIRST_BLACK:                                       # recapture an intruder on our square
        if pos.w >> t & 1:
            pref = [(f, t) for f in RECAPTURERS[t] if (f, t) in legal]
            other = sorted(mv for mv in legal if mv[1] == t)
            if pref or other:
                return (pref or other)[0], "odbicie", 0
    aw, ab = pos.attackers()
    todo = [(f, t) for f, t in MIN_STEPS if pos.b >> f & 1 and not pos.b >> t & 1 and (f, t) in legal]
    if aw > ab:                                                 # urgent: enter the ring
        urgent = [mv for mv in todo if mv[1] in FIRST_BLACK]
        if urgent:
            return urgent[0], "plan (pilne wejście)", 0
    if todo:
        return todo[0], "plan", 0
    return m, "zwykła polityka", s


def expand_minimal(max_plies: int = 20, log=print):
    import os
    from . import routes
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "plan_min")
    return routes.expand(max_plies, 2, out, log, policy=minimal_reply)


MIN2_STEPS = [(S(2, 8), S(3, 6)), (S(6, 8), S(5, 6)),
              (S(1, 8), S(0, 6)), (S(0, 6), S(2, 5)),
              (S(7, 8), S(8, 6)), (S(8, 6), S(6, 5))]
COND_SUPPORT = {(S(0, 6), S(2, 5)): (S(3, 8), S(1, 7)), (S(8, 6), S(6, 5)): (S(5, 8), S(7, 7))}


def minimal_reply2(pos, white_move, depth: int = 2, tt=None):
    """Minimal plan with conditional support: support (3,8)->(1,7) / (5,8)->(7,7) only when the jump
    (0,6)->(2,5) / (8,6)->(6,5) is possible but the target is attacked by a white knight and not yet supported."""
    from .board import FIRST_BLACK, KNIGHT
    m, s = best_black_move(pos, depth, tt)
    if s >= WIN - 1000:
        return m, "wygrana", s
    legal = set(pos.moves())
    for t in FIRST_BLACK:                                       # recapture an intruder on our square
        if pos.w >> t & 1:
            pref = [(f, t) for f in RECAPTURERS[t] if (f, t) in legal]
            other = sorted(mv for mv in legal if mv[1] == t)
            if pref or other:
                return (pref or other)[0], "odbicie", 0
    aw, ab = pos.attackers()
    todo = [(f, t) for f, t in MIN2_STEPS if pos.b >> f & 1 and not pos.b >> t & 1 and (f, t) in legal]
    if aw > ab:
        urgent = [mv for mv in todo if mv[1] in FIRST_BLACK]
        if urgent:
            return urgent[0], "plan (pilne wejście)", 0
    if SUPPORT_OCCUPIED:                                        # our knight already on (2,5)/(6,5) and attacked
        for (src, tgt), sup in COND_SUPPORT.items():
            if pos.b >> tgt & 1 and not pos.b >> sup[1] & 1 and sup in legal \
                    and any(pos.w >> k & 1 for k in KNIGHT[tgt]):
                return sup, "podparcie (stojący atakowany)", 0
    if ADVANCE_ON_RETREAT and white_move is not None:   # white retreated: take the space instead of mirroring
        adv = advance_on_retreat(pos, white_move, tt)
        if adv is not None:
            return adv, "do przodu po cofnięciu białych", 0
    for mv in todo:
        sup = COND_SUPPORT.get(mv)
        if sup:
            attacked = any(pos.w >> k & 1 for k in KNIGHT[mv[1]])
            supported = pos.b >> sup[1] & 1
            if attacked and not supported:
                if sup in legal:
                    return sup, "podparcie (pole atakowane)", 0
                continue                                        # cannot support: do not jump into a capture
        return mv, "plan", 0
    return m, "zwykła polityka", s


SUPPORT_OCCUPIED = False   # True: also support a black knight already standing on (2,5)/(6,5) once it is attacked
ADVANCE_ON_RETREAT = False  # True: after a white free-knight retreat the most symmetric black knight advances


def advance_on_retreat(pos, white_move, tt=None):
    """Forward move (lower c) of the black knight closest to the mirror of the square white retreated from,
    as close as possible to that square; must keep the ring defended and not lose within 3 plies."""
    from .board import FIRST_BB, mirror, rc
    from .mirror_adv import is_retreat, most_symmetric
    from .advance import DIST
    from .routes import defended
    from .search import minimax
    if not is_retreat(white_move):
        return None
    f = mirror(white_move[0])
    k = most_symmetric(pos, f)
    if k is None:
        return None
    tt = {} if tt is None else tt
    cands = []
    for x in pos.moves():
        if x[0] == k and rc(x[1])[1] < rc(k)[1]:
            child = pos.play(x)
            v = minimax(child, 3, tt=tt, ply=1)
            if defended(child) and v > -(WIN - 1000):
                cands.append((DIST[x[1]][f], -v, x))
    return min(cands)[2] if cands else None


def expand_minimal2(max_plies: int = 20, log=print, support_occupied: bool = False, advance: bool = False):
    import os
    from . import routes
    global SUPPORT_OCCUPIED, ADVANCE_ON_RETREAT
    SUPPORT_OCCUPIED, ADVANCE_ON_RETREAT = support_occupied, advance
    name = "plan_min4" if advance else ("plan_min3" if support_occupied else "plan_min2")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", name)
    return routes.expand(max_plies, 2, out, log, policy=minimal_reply2)
