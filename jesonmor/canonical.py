"""Exact theory of 'full' positions: all 8 first-order squares occupied (4 white + 4 black), center empty.

Proven reductions (see docs):
  * the side to move wins when it has more attackers (forced exchange on the center);
  * in a full 4:4 position moving a first-order knight (out or into the center) loses at once,
    so only free knights move;
  * capturing an enemy first-order knight forces the opponent to recapture any of our first-order
    knights with a free knight (else we win); after that it is 4:4 again, capturer to move,
    both sides with one free knight less;
  * the side to move without a free knight loses.
So a full position reduces to a game of free knights: State = (owner, white_free, black_free, turn),
owner = 8-bit mask, bit i set when FIRST[i] holds a white knight.
"""
import os
import random
import zlib
from collections import deque
from itertools import combinations
from typing import Dict, Optional, Tuple

from .board import BLACK, CENTER, FIRST, FIRST_BB, KNIGHT, WHITE, Position, bits, mask

CANON = 0b00001111                 # FIRST[0:4] = white first-order squares
FREE = [s for s in range(81) if s not in FIRST and s != CENTER]
FIRST_IDX = {s: i for i, s in enumerate(FIRST)}
WIN, DRAW, LOSS = 1, 0, -1

State = Tuple[int, tuple, tuple, int]


def to_state(pos: Position) -> Optional[State]:
    """Map a full 4:4 position to its reduced state (None if the position is not full)."""
    if (pos.w | pos.b) >> CENTER & 1 or (pos.w | pos.b) & FIRST_BB != FIRST_BB or pos.attackers() != (4, 4):
        return None
    owner = sum(1 << i for i, s in enumerate(FIRST) if pos.w >> s & 1)
    return owner, tuple(bits(pos.w & ~FIRST_BB)), tuple(bits(pos.b & ~FIRST_BB)), pos.turn


def to_position(state: State) -> Position:
    owner, wf, bf, turn = state
    w = mask(FIRST[i] for i in range(8) if owner >> i & 1) | mask(wf)
    return Position(w, mask(FIRST) & ~w | mask(bf), turn)


def _side_squares(owner: int, side: int):
    return [FIRST[i] for i in range(8) if (owner >> i & 1) == (side == WHITE)]


def outcomes(state: State):
    """Yield ('quiet', child) or ('ext', value_for_mover) for every non-losing option of the mover."""
    owner, wf, bf, turn = state
    mine, theirs = (wf, bf) if turn == WHITE else (bf, wf)
    for f in mine:
        rest = tuple(x for x in mine if x != f)
        for t in KNIGHT[f]:
            if t in mine:
                continue
            if t in FIRST_IDX:
                i = FIRST_IDX[t]
                if (owner >> i & 1) == (turn == WHITE):
                    continue
                yield "ext", _exchange(owner ^ 1 << i, rest, theirs, turn, t)
            elif t in theirs:
                child = tuple(x for x in theirs if x != t)
                new = tuple(sorted(rest + (t,)))
                yield "ext", -value(_make(owner, new, child, turn, 1 - turn))
            else:
                yield "quiet", _make(owner, tuple(sorted(rest + (t,))), theirs, turn, 1 - turn)


def _make(owner, mine, theirs, side, turn) -> State:
    return (owner, mine, theirs, turn) if side == WHITE else (owner, theirs, mine, turn)


def _exchange(owner, mine_rest, theirs, side, s) -> int:
    """We just captured on first-order square s (owner already updated). Opponent must recapture."""
    targets = set(_side_squares(owner, side))
    best = None
    for g in theirs:
        for t in KNIGHT[g]:
            if t in targets:
                o2 = owner ^ 1 << FIRST_IDX[t]
                v = value(_make(o2, mine_rest, tuple(x for x in theirs if x != g), side, side))
                best = v if best is None else min(best, v)
    return WIN if best is None else best


_TABLES: Dict[Tuple[int, int, int], Dict[State, Tuple[int, int]]] = {}


SOLVE_LIMIT = 1_000_000        # families larger than this (and without a numpy table) are estimated, not solved
USE_TABLES = True              # False: ignore big tables (used to test how estimates behave for many knights)


def family_size(nw: int, nb: int) -> int:
    from math import comb
    return 2 * comb(len(FREE), nw) * comb(len(FREE) - nw, nb)


def value(state: State) -> int:
    """Exact value when a table exists or the family is small enough to solve, otherwise the rule estimate."""
    v = retro_value(state) if USE_TABLES else None
    if v is not None:
        return v
    if family_size(len(state[1]), len(state[2])) > SOLVE_LIMIT or not USE_TABLES and sum(map(len, state[1:3])) > 3:
        return estimate(state, 1)             # children of big families: 1-ply tactics, then the rule
    return lookup(state)[0]


def rule_value(state: State, tempo: bool = False) -> int:
    """Rule distilled from the exact tables: immediate tactic -> win; unequal free knights -> the majority wins
    (tempo=True: with the tempo exception of majority_rule); equal -> parity law (odd number of free knights on
    even squares -> side to move wins)."""
    if quick_rules(state) is not None:
        return quick_rules(state)
    return majority_rule(state, tempo)


def estimate(state: State, depth: int = 2) -> int:
    """Value for any number of free knights: exact tactics to `depth` plies, children outside the exact
    tables valued by rule_value (never draws). Verified against the exact (2,2)-(3,2) families."""
    v = search(state, depth)
    return v if v != DRAW else rule_value(state, TEMPO_RULE and depth == 1)


def retro_value(state: State) -> Optional[int]:
    """Exact value from the numpy retrograde tables ((2,2), (1,3)/(3,1), (2,3)/(3,2)), if computed."""
    if state[0] != CANON:
        return None
    from . import retro13, retro22, retro23
    counts = tuple(sorted((len(state[1]), len(state[2]))))
    for c, mod, fn in (((2, 2), retro22, "value22"), ((1, 3), retro13, "value13"), ((2, 3), retro23, "value23")):
        if counts == c and os.path.exists(mod.PATH):
            return getattr(mod, fn)(state)
    return None


def lookup(state: State) -> Tuple[int, int]:
    """(value for side to move, depth in plies of free-knight moves) - solves the family on demand."""
    owner, wf, bf, turn = state
    key = (owner, len(wf), len(bf))
    if key not in _TABLES:
        if MAX_BIG is not None and sum(key[1:]) >= 3:   # optional cap on big dict families (parallel runs)
            big = [k for k in _TABLES if sum(k[1:]) >= 3]
            for k in big[:max(0, len(big) - MAX_BIG + 1)]:
                del _TABLES[k]
        _TABLES[key] = _load(key) or solve_family(*key)
        if sum(key[1:]) >= 3:
            _save(key, _TABLES[key])
    return _TABLES[key][state]


MAX_BIG = None    # e.g. 2 for several parallel processes (slower: tables are reloaded)


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tables")


def _path(key) -> str:
    return os.path.join(CACHE_DIR, "%02x_%d%d.bin" % key)


def _save(key, table) -> None:
    """One byte per state (value + depth), states in family_states order, zlib-compressed."""
    if os.path.exists(_path(key)):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw = bytes((v + 1) | min(d, 63) << 2 for v, d in (table[s] for s in family_states(*key)))
    with open(_path(key), "wb") as fh:
        fh.write(zlib.compress(raw, 9))


def _load(key):
    if not os.path.exists(_path(key)):
        return None
    with open(_path(key), "rb") as fh:
        raw = zlib.decompress(fh.read())
    return {s: ((x & 3) - 1, x >> 2) for s, x in zip(family_states(*key), raw)}


def family_states(owner: int, nw: int, nb: int):
    for wf in combinations(FREE, nw):
        rest = [s for s in FREE if s not in wf]
        for bf in combinations(rest, nb):
            yield owner, wf, bf, WHITE
            yield owner, wf, bf, BLACK


def solve_family(owner: int, nw: int, nb: int) -> Dict[State, Tuple[int, int]]:
    """Retrograde analysis of all states with the given owner mask and free-knight counts."""
    res: Dict[State, Tuple[int, int]] = {}
    parents: Dict[State, list] = {}
    count: Dict[State, int] = {}
    ext_best: Dict[State, int] = {}
    queue = deque()
    for st in family_states(owner, nw, nb):
        if (nw, nb)[st[3]] == 0:
            res[st] = (LOSS, 0)
            queue.append(st)
            continue
        n, best = 0, None
        for kind, x in outcomes(st):
            if kind == "quiet":
                n += 1
                parents.setdefault(x, []).append(st)
            else:
                best = x if best is None else max(best, x)
        ext_best[st], count[st] = best, n
        if best == WIN or n == 0:
            res[st] = (LOSS if best is None else best, 1)
            queue.append(st)
    while queue:
        st = queue.popleft()
        v, d = res[st]
        if v == DRAW:
            continue
        for p in parents.get(st, ()):
            if p in res:
                continue
            if v == LOSS:
                res[p] = (WIN, d + 1)
                queue.append(p)
            else:
                count[p] -= 1
                if count[p] == 0:
                    eb = ext_best[p]
                    res[p] = (LOSS if eb in (None, LOSS) else eb, d + 1)
                    queue.append(p)
    for st in count:
        res.setdefault(st, (DRAW, 0))
    return res


def quick_rules(state: State) -> Optional[int]:
    """Closed-form one-move rules (value for the side to move) valid for any free-knight counts:
    R0 mover has no free knight -> LOSS;
    R1 mover attacks the enemy's only free knight -> WIN (enemy is left without free knights);
    R2 capture of an enemy first-order knight that no enemy free knight can answer -> WIN."""
    owner, wf, bf, turn = state
    mine, theirs = (wf, bf) if turn == WHITE else (bf, wf)
    if not mine:
        return LOSS
    their_att = set(t for g in theirs for t in KNIGHT[g])
    my_first = set(_side_squares(owner, turn))
    for f in mine:
        for t in KNIGHT[f]:
            if len(theirs) == 1 and t == theirs[0]:
                return WIN
            if t in FIRST_IDX and t not in my_first and not (their_att & (my_first | {t})):
                return WIN
    return None


def search(state: State, depth: int, alpha: int = LOSS, beta: int = WIN) -> int:
    """Depth-limited negamax over quiet moves; exchanges/captures resolved by exact tables.
    Returns WIN/LOSS if proven within depth, DRAW when undecided."""
    q = quick_rules(state)
    if q is not None:
        return q
    best, quiet = LOSS, []
    for kind, x in outcomes(state):
        if kind == "ext":
            best = max(best, x)
        else:
            quiet.append(x)
    if best == WIN or not quiet:
        return best
    if depth == 0:
        return max(best, DRAW)
    for child in quiet:
        best = max(best, -search(child, depth - 1, -beta, -max(alpha, best)))
        if best >= beta:
            break
    return best


def random_state(nw: int, nb: int, turn: int, owner: int = CANON, rng=random) -> State:
    sq = rng.sample(FREE, nw + nb)
    return owner, tuple(sorted(sq[:nw])), tuple(sorted(sq[nw:])), turn


def active(s: int) -> bool:
    """A free knight on an 'even' square ((r+c) even) attacks first-order squares (all of them are odd)."""
    return (s // 9 + s % 9) % 2 == 0


def parity_rule(state: State) -> int:
    """Parity law: the mover wins iff an odd number of free knights (both colours) stand on even squares."""
    return WIN if sum(active(s) for s in state[1] + state[2]) % 2 else LOSS


def _immediate(owner: int, turn: int, x: int, y: int) -> bool:
    """(1,1): mover's knight x wins at once (captures y, or an unanswerable first-order capture)."""
    my_first = set(_side_squares(owner, turn))
    their_first = set(FIRST) - my_first
    return y in KNIGHT[x] or any(not (set(KNIGHT[y]) & (my_first | {s}))
                                 for s in KNIGHT[x] if s in their_first)


def rule_11(state: State) -> int:
    """Closed-form exact rule for one free knight per side (verified against retrograde tables)."""
    owner, wf, bf, turn = state
    x, y = (wf[0], bf[0]) if turn == WHITE else (bf[0], wf[0])
    my_first = set(_side_squares(owner, turn))
    their_first = set(FIRST) - my_first
    if _immediate(owner, turn, x, y):
        return WIN
    free_moves = [t for t in KNIGHT[x] if t not in FIRST_IDX]
    if active(x) != active(y):
        threats = set(KNIGHT[y]) & my_first
        if active(x) or not threats:
            return WIN
        guard = lambda t: set(KNIGHT[t]) & their_first or all(s in KNIGHT[t] for s in threats)
        return WIN if any(guard(t) for t in free_moves) else LOSS
    if active(x):
        return LOSS
    replies = lambda t: [u for u in KNIGHT[y] if u not in FIRST_IDX and u != t]
    ok = lambda t: t not in KNIGHT[y] and all(_immediate(owner, turn, t, u) for u in replies(t))
    return WIN if any(ok(t) for t in free_moves) else LOSS


def predict(state: State, depth: int = 2) -> int:
    """Practical rule: short exact search, then the majority rule (more free knights wins)
    for unequal counts or the parity law for equal counts; after a 1-ply search with the tempo exception."""
    v = search(state, depth)
    if v != DRAW:
        return v
    return majority_rule(state, TEMPO_RULE and depth == 1)


TEMPO_RULE = True      # False: no tempo exception (the numbers in README sections 4-5 before 2026-10-03)


def majority_rule(state: State, tempo: bool = False) -> int:
    """Equal free-knight counts -> parity law; otherwise the majority wins.
    tempo=True adds the exception mined from the exact tables (1,2), (1,3), (2,3): the side to move with exactly
    one free knight less wins when more of its free knights than of the opponent's stand on odd squares (ring 3,
    one jump from attacking the first-order ring). It predicts what the next ply of search would find, so it is
    right only for a state left undecided by a 1-ply search: after 0 plies (open tactics) or 2 plies (a real
    tempo win would already be found) it is wrong more often than right (README_EN 4, "Tempo rule")."""
    mine, theirs = state[1 + state[3]], state[2 - state[3]]
    d = len(mine) - len(theirs)
    if d == 0:
        return parity_rule(state)
    if tempo and d == -1 and odd_free(mine) > odd_free(theirs):
        return WIN
    return WIN if d > 0 else LOSS


def odd_free(squares) -> int:
    """Free knights on odd squares (all of them in ring 3)."""
    return sum(not active(s) for s in squares)


RING2 = [s for s in FREE if any(t in FIRST_IDX for t in KNIGHT[s])]   # 32 even squares, = Z_W | Z_B


def capture_win(state: State, memo: Optional[dict] = None) -> bool:
    """Forced win of the side to move using only ring-2 -> ring-1 captures (the opponent's replies are
    forced recaptures anyway, L4). Exact recursion; a side left without free knights loses (L5)."""
    memo = {} if memo is None else memo
    if state in memo:
        return memo[state]
    owner, wf, bf, turn = state
    mine, theirs = (wf, bf) if turn == WHITE else (bf, wf)
    res = bool(mine) and not theirs
    for f in mine if not res else ():
        rest = tuple(x for x in mine if x != f)
        for s in KNIGHT[f]:
            if s not in FIRST_IDX or (owner >> FIRST_IDX[s] & 1) == (turn == WHITE):
                continue
            o1 = owner ^ 1 << FIRST_IDX[s]
            targets = set(_side_squares(o1, turn))
            replies = [(g, t) for g in theirs for t in KNIGHT[g] if t in targets]
            if all(capture_win(_make(o1 ^ 1 << FIRST_IDX[t], rest, tuple(x for x in theirs if x != g), turn, turn),
                               memo) for g, t in replies):
                res = True
                break
        if res:
            break
    memo[state] = res
    return res
