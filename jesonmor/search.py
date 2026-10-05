"""Search for the full game. Scores are from BLACK's point of view: black maximises (only its best
move matters), white minimises (every white move is considered)."""
from typing import Dict, List, Optional, Tuple

from . import canonical as cn
from .board import (BLACK, CENTER, FIRST, FIRST_BB, KNIGHT_BB, N, RING, WHITE, Move, Position, bits, exchange_winner,
                    mask, sq)

WIN = 100000
THEORY = 50000          # value of a full position decided by the free-knight theory (not a proof in plies)
AXIS_BB = mask(sq(r, N // 2) for r in range(N))


def _side_features(own: int, opp: int, empty_first: int) -> int:
    free = own & ~FIRST_BB
    reinforce = 0
    for f in bits(free):
        reinforce |= KNIGHT_BB[f] & empty_first
    threats = sum(bin(KNIGHT_BB[f] & opp & FIRST_BB).count("1") > 0 for f in bits(free))
    hunts = sum(bin(KNIGHT_BB[f] & opp & ~FIRST_BB).count("1") for f in bits(free))
    near = sum(max(0, 4 - RING[f]) for f in bits(free))
    return (120 * bin(own & FIRST_BB).count("1") + 45 * bin(reinforce).count("1") + 30 * threats
            + 8 * hunts + 3 * near + 60 * bin(own).count("1"))


def _exact(st) -> Optional[int]:
    """Exact value of a canonical state from the retrograde tables ((2,2), (1,3)/(3,1), (2,3)/(3,2))."""
    return cn.retro_value(st)


def rank(pos: Position, theory_depth: int = 1) -> int:
    """Static ranking of a position for black (higher = better for black)."""
    wd = exchange_winner(pos)
    if wd is not None:
        return WIN if wd == BLACK else -WIN
    st = cn.to_state(pos)
    v = _exact(st) if st is not None else None
    if v is None and st is not None and len(st[1]) <= 2 and len(st[2]) <= 2:
        v = cn.predict(st, theory_depth)
    if v is not None:
        if v != cn.DRAW:
            return THEORY if (v == cn.WIN) == (pos.turn == BLACK) else -THEORY
    occupied = pos.w | pos.b
    empty_first = FIRST_BB & ~occupied
    aw, ab = pos.attackers()
    score = _side_features(pos.b, pos.w, empty_first) - _side_features(pos.w, pos.b, empty_first)
    me, opp = (ab, aw) if pos.turn == BLACK else (aw, ab)
    if me < opp:                       # side to move must fix the count at once
        score += -200 if pos.turn == BLACK else 200
    return score


def ordered_moves(pos: Position) -> List[Move]:
    _, opp = pos.own()
    key = lambda m: -(2 * (opp >> m[1] & 1) + (m[1] in FIRST))
    return sorted(pos.moves(), key=key)


QUIESCENCE = True
QDEPTH = 4                 # max extra plies of noisy moves at the horizon


def noisy_moves(pos: Position) -> List[Move]:
    """Moves that change the attacker balance: jumps onto first-order squares or the center, and captures.
    A first-order knight leaving the ring is excluded (it only lowers its own count)."""
    _, opp = pos.own()
    return [m for m in ordered_moves(pos)
            if m[1] in FIRST or m[1] == CENTER or (opp >> m[1] & 1 and not FIRST_BB >> m[0] & 1)]


def qsearch(pos: Position, alpha: int, beta: int, ply: int, qd: int) -> int:
    """Quiescence search: at the horizon keep playing noisy moves until the attacker balance is quiet.
    A side with fewer attackers than the opponent may not 'stand pat' (the opponent would enter the center)."""
    wd = exchange_winner(pos)
    if wd is not None:
        return (WIN - ply) if wd == BLACK else -(WIN - ply)
    maximize = pos.turn == BLACK
    aw, ab = pos.attackers()
    forced = (ab < aw) if maximize else (aw < ab)
    if qd == 0:
        return rank(pos)
    if forced:
        best = -WIN - 1 if maximize else WIN + 1
    else:
        best = rank(pos)
        if (maximize and best >= beta) or (not maximize and best <= alpha):
            return best
        alpha, beta = (max(alpha, best), beta) if maximize else (alpha, min(beta, best))
    for m in noisy_moves(pos):
        v = qsearch(pos.play(m), alpha, beta, ply + 1, qd - 1)
        if maximize:
            best, alpha = max(best, v), max(alpha, v)
        else:
            best, beta = min(best, v), min(beta, v)
        if alpha >= beta:
            break
    if forced and abs(best) == WIN + 1:          # no move restores the balance: the opponent enters next
        return -(WIN - ply - 1) if maximize else (WIN - ply - 1)
    return best


def minimax(pos: Position, depth: int, alpha: int = -WIN - 1, beta: int = WIN + 1,
            tt: Optional[Dict] = None, ply: int = 0) -> int:
    """Depth-first alpha-beta search to `depth` plies. Proven wins are WIN - ply (faster is better)."""
    wd = exchange_winner(pos)
    if wd is not None:
        return (WIN - ply) if wd == BLACK else -(WIN - ply)
    if depth == 0:
        return qsearch(pos, alpha, beta, ply, QDEPTH) if QUIESCENCE else rank(pos)
    tt = {} if tt is None else tt
    key = (pos, depth)
    if key in tt:
        lo, hi = tt[key]
        if lo >= beta or lo == hi:
            return lo
        if hi <= alpha:
            return hi
    a0, b0 = alpha, beta
    maximize = pos.turn == BLACK
    best = -WIN - 1 if maximize else WIN + 1
    for m in ordered_moves(pos):
        v = minimax(pos.play(m), depth - 1, alpha, beta, tt, ply + 1)
        if maximize:
            best, alpha = max(best, v), max(alpha, v)
        else:
            best, beta = min(best, v), min(beta, v)
        if alpha >= beta:
            break
    tt[key] = (best if best > a0 else -WIN - 1, best if best < b0 else WIN + 1)
    return best


def evaluate_dfs(pos: Position, n: int) -> int:
    """Evaluate a position by depth-first search to the n-th ply."""
    return minimax(pos, n)


def is_symmetric(pos: Position) -> bool:
    """Mirror-safe: black = white mirrored (c -> 8-c) off the middle column, no white knight on the middle
    column (extra black knights there are allowed - they appear after black's mirror captures)."""
    m = pos.mirrored()
    return pos.w & AXIS_BB == 0 and m.b == pos.b & ~AXIS_BB


def rank_black_moves(pos: Position, depth: int = 3, tt: Optional[Dict] = None) -> List[Tuple[int, Move]]:
    """All black moves with their search score, best first (mirror moves win ties)."""
    assert pos.turn == BLACK
    tt = {} if tt is None else tt
    scored = []
    for m in ordered_moves(pos):
        child = pos.play(m)
        scored.append((minimax(child, depth - 1, tt=tt, ply=1), is_symmetric(child), m))
    scored.sort(key=lambda x: (-x[0], not x[1]))
    return [(s, m) for s, _, m in scored]


def mirror_move(pos: Position) -> Optional[Move]:
    """Black move restoring colour-mirror symmetry (never loses, see docs), if one exists."""
    for m in pos.moves():
        if is_symmetric(pos.play(m)):
            return m
    return None


def best_black_move(pos: Position, depth: int = 3, tt: Optional[Dict] = None,
                    use_mirror: bool = True) -> Tuple[Optional[Move], int]:
    """Best black move and its score. Policy: a move with a (theory) win score if found, otherwise the
    mirror move when it exists (guaranteed non-losing), otherwise the best-scoring move (alpha-beta)."""
    assert pos.turn == BLACK
    tt = {} if tt is None else tt
    mm = mirror_move(pos) if use_mirror else None
    floor = v_mm = -WIN - 2
    if mm is not None:
        v_mm = minimax(pos.play(mm), depth - 1, -WIN - 1, WIN + 1, tt, 1)
        floor = max(v_mm, THEORY - 1)
    best, best_m = floor, None
    for m in ordered_moves(pos):
        v = minimax(pos.play(m), depth - 1, best, WIN + 1, tt, 1)
        if v > best:
            best, best_m = v, m
    if best_m is None and mm is not None:
        return mm, v_mm
    return best_m, best


def choose(pos: Position, ranked: List[Tuple[int, Move]]) -> Tuple[int, Move]:
    """Same policy as best_black_move applied to a full ranking."""
    mm = mirror_move(pos)
    if ranked[0][0] >= THEORY or mm is None:
        return ranked[0]
    return next(x for x in ranked if x[1] == mm)

def expand_bfs(pos: Position, n: int, depth: int = 2) -> List[set]:
    """Breadth-first expansion to the n-th ply: all white moves, only the best black move.
    Returns the list of levels (sets of positions); decided positions are not expanded."""
    levels, tt = [{pos}], {}
    for _ in range(n):
        nxt = set()
        for p in levels[-1]:
            if exchange_winner(p) is not None:
                continue
            if p.turn == WHITE:
                nxt.update(p.play(m) for m in p.moves())
            else:
                m, _ = best_black_move(p, depth, tt)
                if m is not None:
                    nxt.add(p.play(m))
        levels.append(nxt)
    return levels
