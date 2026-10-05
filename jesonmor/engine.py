"""Game engine for the search of the best black heuristic (driven by `python -m jesonmor.tuner`).

* `features(pos)` - 19 numbers describing a position from black's point of view (black minus white);
* `Heuristic` - feature weights + root bonus for the mirror move; evaluation = exact knowledge (exchange oracle,
  free-knight theory with the endgame tables) or the weighted feature sum;
* `HeuristicPlayer` - alpha-beta (negamax, transposition table, quiescence, lemma L3 pruning) with a heuristic;
* `MeanPlayer` - the root move by the mean of the scores at depth - 1 and depth (against the odd-depth optimism);
* `BaselinePlayer` - the reference opponent of all earlier experiments (search.minimax, random among equal best);
* `PolicyPlayer` - adapter for the hand-written black policies (middle.middle_reply, h750.reply, ...);
* `play_game` - one game with a ply limit and threefold repetition (both = draw);
* `low_memory_tables` - endgame tables at 1 byte per state, the (2,3) table memory-mapped (shared by processes).
Scores are from black's point of view (higher = better for black) unless stated otherwise."""
import os
import random
import zlib
from collections.abc import Mapping
from math import comb, floor
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import canonical as cn
from .board import BLACK, CENTER, FIRST_BB, KNIGHT_BB, RING, WHITE, Move, Position, bits, exchange_winner, mask
from .search import THEORY, WIN, is_symmetric

CENTER_BB = 1 << CENTER
EVEN_BB = mask(s for s in range(81) if (s // 9 + s % 9) % 2 == 0)
ACTIVE_BB = mask(s for s in range(81) if KNIGHT_BB[s] & FIRST_BB and not FIRST_BB >> s & 1 and s != CENTER)
NEAR = [max(0, 4 - RING[s]) for s in range(81)]
ENTRY = {r: KNIGHT_BB[r] & ~FIRST_BB & ~(1 << CENTER) for r in bits(FIRST_BB)}   # squares attacking ring square r
INF = 10 ** 9

FEATURES = ("attackers", "free", "majority", "threats", "active", "hunts", "mobility", "near", "reinforce",
            "hanging", "exposed", "unanswered", "entries", "parity", "tempo", "overloaded", "loose", "outnumbered",
            "bluff")
FEATURE_INFO = {
    "attackers": "skoczki na polach pierwszego rzędu",
    "free": "wolne skoczki (poza pierwszym rzędem)",
    "majority": "reguła przewagi: znak różnicy wolnych skoczków w pozycji pełnej 4:4",
    "threats": "wolne skoczki atakujące skoczka przeciwnika w pierwszym rzędzie",
    "active": "wolne skoczki na polach atakujących pierwszy rząd",
    "hunts": "ataki wolnych skoczków na wolne skoczki przeciwnika",
    "mobility": "ruchy wolnych skoczków",
    "near": "bliskość wolnych skoczków do pierścienia (3 - nr pierścienia)",
    "reinforce": "puste pola pierwszego rzędu osiągalne jednym skokiem",
    "hanging": "wolne skoczki atakowane przez wolnego skoczka przeciwnika i niebronione",
    "exposed": "własne skoczki pierwszego rzędu atakowane przez wolne skoczki przeciwnika",
    "unanswered": "atakowane skoczki pierwszego rzędu bez odpowiedzi (reguła strażnika, L4)",
    "entries": "puste, niepilnowane pola atakujące własny pierścień, na które może skoczyć wolny skoczek przeciwnika",
    "parity": "prawo parzystości przy równej liczbie wolnych skoczków w pozycji pełnej (+1 = dobre dla czarnych)",
    "tempo": "+1 gdy ruch czarnych, -1 gdy ruch białych",
    "overloaded": "przeciążona obrona strony na posunięciu bez własnej groźby: jej skoczki pierwszego rzędu atakowane "
                  "przez wolne skoczki przeciwnika, których wszyscy wolni obrońcy wiszą (atakowani i niebronieni)",
    "loose": "luźne skoczki pierwszego rzędu: niebronione, żaden własny wolny skoczek nie dojdzie jednym skokiem na "
             "pole obrony, a wolny skoczek przeciwnika już je atakuje albo może je zaatakować jednym skokiem",
    "outnumbered": "skoczki pierwszego rzędu atakowane przez więcej wolnych skoczków przeciwnika, niż bronią ich własne "
                   "wolne skoczki (co najmniej jeden obrońca; bez obrońcy: unanswered, loose)",
    "bluff": "pozorna inicjatywa: atakowane niebronione skoczki pierwszego rzędu, które unanswered pomija z powodu "
             "własnej groźby, choć każda własna groźba jest pokryta (atakujących nie więcej niż obrońców)",
}
# start point of the search: the weights of search.rank (attacker 120 + material 60, ...) plus the newer features
DEFAULT_WEIGHTS = {"attackers": 180, "free": 60, "majority": 100, "threats": 30, "active": 10, "hunts": 8,
                   "mobility": 2, "near": 3, "reinforce": 45, "hanging": -40, "exposed": -10, "unanswered": -60,
                   "entries": -15, "parity": 50, "tempo": 10, "overloaded": -60, "loose": -60, "outnumbered": -60,
                   "bluff": -60}


def _side(free: int, own: int, enemy_ring: int, enemy_free: int):
    """Attack map and per-knight counts of one side's free knights."""
    amap = threats = active = hunts = mob = near = 0
    notown = ~own
    while free:
        low = free & -free
        f = low.bit_length() - 1
        free ^= low
        a = KNIGHT_BB[f]
        amap |= a
        if a & enemy_ring:
            threats += 1
        if a & FIRST_BB:
            active += 1
        if a & enemy_free:
            hunts += bin(a & enemy_free).count("1")
        mob += bin(a & notown).count("1")
        near += NEAR[f]
    return amap, threats, active, hunts, mob, near


def features(pos: Position) -> Tuple[int, ...]:
    """Feature vector (black minus white, see FEATURES / FEATURE_INFO)."""
    w, b = pos.w, pos.b
    wr, br = w & FIRST_BB, b & FIRST_BB
    wf, bf = w & ~FIRST_BB, b & ~FIRST_BB
    empty_ring = FIRST_BB & ~(w | b)
    aw, tw, cw, hw, mw, nw = _side(wf, w, br, bf)
    ab, tb, cb, hb, mb, nb = _side(bf, b, wr, wf)
    n_wr, n_br = bin(wr).count("1"), bin(br).count("1")
    n_wf, n_bf = bin(wf).count("1"), bin(bf).count("1")
    un_b = 0 if ab & wr else bin(br & aw & ~ab).count("1")
    un_w = 0 if aw & br else bin(wr & ab & ~aw).count("1")
    hang_b, hang_w = bf & aw & ~ab, wf & ab & ~aw
    # only the side to move (the other side's are captures, seen by the quiescence) and, as for un_b/un_w, only
    # without an own threat against the enemy ring (L4 initiative)
    if pos.turn == BLACK:
        over = 0 if ab & wr else _overloaded(br & aw & ab, bf, hang_b)
    else:
        over = 0 if aw & br else -_overloaded(wr & ab & aw, wf, hang_w)
    entry_b = entry_w = 0
    for r in bits(br):
        entry_b |= ENTRY[r]
    for r in bits(wr):
        entry_w |= ENTRY[r]
    free_sq = ~(w | b)
    parity = majority = 0
    if n_wr == n_br == 4 and not empty_ring:
        majority = (n_bf > n_wf) - (n_bf < n_wf)
        if n_wf == n_bf:
            mover_wins = bin((wf | bf) & EVEN_BB).count("1") & 1
            parity = 1 if bool(mover_wins) == (pos.turn == BLACK) else -1
    return (n_br - n_wr, n_bf - n_wf, majority, tb - tw, cb - cw, hb - hw, mb - mw, nb - nw,
            bin(ab & empty_ring).count("1") - bin(aw & empty_ring).count("1"),
            bin(hang_b).count("1") - bin(hang_w).count("1"),
            bin(br & aw).count("1") - bin(wr & ab).count("1"),
            un_b - un_w,
            bin(entry_b & free_sq & aw & ~ab).count("1") - bin(entry_w & free_sq & ab & ~aw).count("1"),
            parity, 1 if pos.turn == BLACK else -1, over,
            _loose(br, ab, b, wf, aw, w) - _loose(wr, aw, w, bf, ab, b),
            _outnumbered(br & aw, bf, wf) - _outnumbered(wr & ab, wf, bf),
            _bluff(br & aw & ~ab, wr & ab, bf, wf) - _bluff(wr & ab & ~aw, br & aw, wf, bf))


def _bluff(unanswered: int, threatened: int, free: int, enemy_free: int) -> int:
    """Attacked undefended own ring knights (`unanswered`) that the L4-initiative exception of `unanswered` skips
    because of own threats (`threatened`: enemy ring knights attacked by own free knights), although every threat
    is covered (no more own attackers than enemy free defenders): one quiet move answers them and the own knight
    still falls (945/205: a6-b4 e1-c2, b4 attacked d3 and b6 attacked c4, both covered, c6 lost after d4xc6)."""
    if not threatened:
        return 0
    for r in bits(threatened):
        if bin(KNIGHT_BB[r] & free).count("1") > bin(KNIGHT_BB[r] & enemy_free).count("1"):
            return 0
    return bin(unanswered).count("1")


def _overloaded(guarded: int, free: int, hanging: int) -> int:
    """Attacked ring knights (`guarded`: attacked and defended by own free knights) all of whose free defenders
    hang: the defender may neither leave nor stay, the opponent takes it and then the ring knight cannot be
    recaptured (L4). Quiescence only plays captures, so this threat against the side to move is invisible there."""
    n = 0
    for r in bits(guarded):
        if not KNIGHT_BB[r] & free & ~hanging:
            n += 1
    return n


def _loose(ring: int, amap: int, own: int, enemy_free: int, enemy_amap: int, enemy: int) -> int:
    """Own ring knights (`ring`) that no own free knight defends or can defend after one jump (`amap`: attack map of
    the own free knights), while an enemy free knight attacks them or can attack them after one jump. The attacked
    knight may not recapture (it would leave the ring), so the opponent wins it or forces a counter-exchange whose
    result the quiescence often does not see (945/184: c8-b6 lost the defence of c6 against e2-d4)."""
    n = 0
    for r in bits(ring):
        e = ENTRY[r]
        if amap >> r & 1 or amap & e & ~own:
            continue
        if enemy_free & e or enemy_amap & e & ~enemy:
            n += 1
    return n


def _outnumbered(attacked: int, free: int, enemy_free: int) -> int:
    """Own ring knights (`attacked`: attacked by enemy free knights) with at least one free defender but more free
    attackers than free defenders: after the exchange an enemy knight stays on the ring square. Own threats do not
    help when one quiet move covers them (945/935 and 945/944: g8-f6 attacked g4 instead of defending g6, attacked
    by f4 and i5 and defended only by h8; f1-e3 covered c4 and g4)."""
    n = 0
    for r in bits(attacked):
        d = bin(KNIGHT_BB[r] & free).count("1")
        if d and bin(KNIGHT_BB[r] & enemy_free).count("1") > d:
            n += 1
    return n


def theory_value(pos: Position) -> Optional[int]:
    """+-THEORY for full 4:4 positions decided by the free-knight theory: side to move without free knights loses
    (L5); exact tables (2,2), (1,3), (2,3); rules for at most 2 free knights per side (as in search.rank)."""
    w, b = pos.w, pos.b
    occ = w | b
    if occ & FIRST_BB != FIRST_BB or occ & CENTER_BB or bin(w & FIRST_BB).count("1") != 4:
        return None
    nw, nb = bin(w).count("1") - 4, bin(b).count("1") - 4
    if (nw if pos.turn == WHITE else nb) == 0:
        v = cn.LOSS
    elif nw + nb <= 5 and nw <= 3 and nb <= 3:
        st = cn.to_state(pos)
        v = cn.retro_value(st)
        if v is None and nw <= 2 and nb <= 2:
            v = cn.predict(st, 1)
    else:
        return None
    if v is None or v == cn.DRAW:
        return None
    return THEORY if (v == cn.WIN) == (pos.turn == BLACK) else -THEORY


class Heuristic:
    """Weighted feature sum + exact knowledge. `mirror_bonus` is added at the root to a black move that restores
    the colour-mirror symmetry (the move that never loses, README 6.1)."""

    def __init__(self, weights=None, mirror_bonus: float = 0.0, theory: bool = True, name: str = "heurystyka"):
        weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.weights = {k: float(weights.get(k, 0.0)) for k in FEATURES}
        self.w = [self.weights[k] for k in FEATURES]
        self.mirror_bonus, self.theory, self.name = float(mirror_bonus), theory, name

    def evaluate(self, pos: Position) -> int:
        """Static value for black (proven results are handled by the search)."""
        h = int(round(sum(a * x for a, x in zip(self.w, features(pos)))))
        if self.theory:
            t = theory_value(pos)
            if t is not None:                       # decided by theory: keep the features as a progress measure
                return t + max(-2000, min(2000, h))
        return h

    def to_dict(self) -> dict:
        return {"name": self.name, "weights": self.weights, "mirror_bonus": self.mirror_bonus, "theory": self.theory}

    @classmethod
    def from_dict(cls, d: dict) -> "Heuristic":
        return cls(d["weights"], d.get("mirror_bonus", 0.0), d.get("theory", True), d.get("name", "heurystyka"))

    def __repr__(self):
        return "Heuristic(%s, mirror_bonus=%g)" % (", ".join("%s=%g" % kv for kv in self.weights.items()),
                                                   self.mirror_bonus)


def full44(pos: Position) -> bool:
    """Full 4:4 position: only free knights may move (lemma L3)."""
    occ = pos.w | pos.b
    return occ & FIRST_BB == FIRST_BB and not occ & CENTER_BB and bin(pos.w & FIRST_BB).count("1") == 4


def ordered_moves(pos: Position, first: Optional[Move] = None, noisy: bool = False) -> List[Move]:
    """Legal moves, best candidates first (center/ring captures, free-knight captures, ring entries, active squares).
    In full 4:4 positions first-order knights do not move (L3: every such move loses at once).
    noisy=True: only moves changing the attacker balance (quiescence), as search.noisy_moves."""
    own, opp = pos.own()
    src = own & ~FIRST_BB if full44(pos) else own
    if not src:
        src = own
    top, caps, entries, active, quiet = [], [], [], [], []
    seen_first = False
    for f in bits(src):
        from_ring = FIRST_BB >> f & 1
        for t in bits(KNIGHT_BB[f] & ~own):
            m = (f, t)
            if m == first:
                seen_first = True
                continue
            if t == CENTER or (FIRST_BB >> t & 1 and opp >> t & 1):
                top.append(m)
            elif opp >> t & 1:
                if not (noisy and from_ring):
                    caps.append(m)
            elif FIRST_BB >> t & 1:
                entries.append(m)
            elif not noisy:
                (active if ACTIVE_BB >> t & 1 else quiet).append(m)
    out = top + caps + entries + active + quiet
    if seen_first:                                 # transposition-table move (legal: it was generated) first
        out.insert(0, first)
    return out


class HeuristicPlayer:
    """Alpha-beta player with a Heuristic. depth = full plies, qdepth = extra plies of noisy moves at the horizon.
    ext > 0 (endgame extension, off by default): a horizon node that is a full 4:4 position with at most ext_free
    free knights is searched ext plies deeper instead of going to the quiescence, once per line. The static value
    of such endings is often wrong while the tables decide them a few plies later (945/211 at depth 4: the 3:3 ending
    after b7-d6 b3-d4 h4xf3 d4xf3 looks equal and is lost; with ext=4 black plays h4-g2 and wins). Costs about three
    times the time per game at depth 4."""

    EXACT, LOWER, UPPER = 0, 1, 2

    def __init__(self, heuristic: Heuristic, depth: int = 2, qdepth: int = 4, name: str = None,
                 tt_limit: int = 300000, ext: int = 0, ext_free: int = 6):
        self.h, self.depth, self.qdepth = heuristic, depth, qdepth
        self.ext, self.ext_free = ext, ext_free
        self.name = name or "%s (głębokość %d%s)" % (heuristic.name, depth,
                                                      ", końcówka +%d" % ext if ext else "")
        self.tt_limit = tt_limit
        self.reset()

    def reset(self):
        self.tt: Dict[Position, tuple] = {}
        self.cache: Dict[Position, int] = {}
        self.nodes = 0

    def static(self, pos: Position) -> int:
        """Heuristic value for the side to move."""
        v = self.cache.get(pos)
        if v is None:
            if len(self.cache) > self.tt_limit:
                self.cache.clear()
            v = self.cache[pos] = self.h.evaluate(pos)
        return v if pos.turn == BLACK else -v

    def qsearch(self, pos: Position, alpha: int, beta: int, ply: int, qd: int) -> int:
        self.nodes += 1
        wd = exchange_winner(pos)
        if wd is not None:
            return WIN - ply if wd == pos.turn else -(WIN - ply)
        if qd == 0:
            return self.static(pos)
        aw, ab = pos.attackers()
        forced = (aw < ab) if pos.turn == WHITE else (ab < aw)    # may not stand pat: the opponent would enter
        if forced:
            best = -INF
        else:
            best = self.static(pos)
            if best >= beta:
                return best
            alpha = max(alpha, best)
        for m in ordered_moves(pos, noisy=True):
            v = -self.qsearch(pos.play(m), -beta, -alpha, ply + 1, qd - 1)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        if best == -INF:                                          # no move restores the balance
            return -(WIN - ply - 1)
        return best

    def endgame(self, pos: Position) -> bool:
        """Horizon node of the endgame extension: full 4:4 with at most ext_free free knights."""
        return full44(pos) and bin((pos.w | pos.b) & ~FIRST_BB).count("1") <= self.ext_free

    def negamax(self, pos: Position, depth: int, alpha: int, beta: int, ply: int, can_ext: bool = True) -> int:
        """Value for the side to move (proven wins: WIN - ply). can_ext: the endgame extension is still unused on
        this line (with ext > 0 it is part of the transposition key)."""
        wd = exchange_winner(pos)
        if wd is not None:
            self.nodes += 1
            return WIN - ply if wd == pos.turn else -(WIN - ply)
        if depth <= 0:
            if not (can_ext and self.ext and self.endgame(pos)):
                return self.qsearch(pos, alpha, beta, ply, self.qdepth)
            depth, can_ext = self.ext, False
        self.nodes += 1
        key = (pos, can_ext) if self.ext else pos
        entry = self.tt.get(key)
        first = None
        if entry is not None:
            d, flag, v, first = entry
            if d >= depth and (flag == self.EXACT or (flag == self.LOWER and v >= beta)
                               or (flag == self.UPPER and v <= alpha)):
                return v
        a0, best, best_m = alpha, -INF, None
        for m in ordered_moves(pos, first):
            v = -self.negamax(pos.play(m), depth - 1, -beta, -alpha, ply + 1, can_ext)
            if v > best:
                best, best_m = v, m
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        if len(self.tt) > self.tt_limit:
            self.tt.clear()
        self.tt[key] = (depth, self.LOWER if best >= beta else (self.UPPER if best <= a0 else self.EXACT),
                        best, best_m)
        return best

    def root(self, pos: Position, history: Optional[Dict] = None) -> List[Tuple[int, Move]]:
        """(value for the side to move, move) for every move, best first. A move repeating an earlier position of
        the game is worth 0 (draw); black's mirror move gets the heuristic's mirror_bonus."""
        history = history or {}
        bonus = self.h.mirror_bonus if pos.turn == BLACK else 0.0
        scored, alpha = [], -INF
        for m in ordered_moves(pos):
            child = pos.play(m)
            if child in history:
                v = 0
            else:
                b = bonus if bonus and is_symmetric(child) else 0
                v = -self.negamax(child, self.depth - 1, -INF, -(alpha - b) if alpha > -INF else INF, 1) + b
            scored.append((v, m))
            alpha = max(alpha, v)
        scored.sort(key=lambda x: -x[0])
        return scored

    def move(self, pos: Position, history: Optional[Dict] = None, rng=None) -> Tuple[Move, str]:
        scored = self.root(pos, history)
        v, m = scored[0]
        return m, "%d" % v

    def root_bonus(self, pos: Position, child: Position) -> float:
        return self.h.mirror_bonus if pos.turn == BLACK and self.h.mirror_bonus and is_symmetric(child) else 0

    def root_value(self, pos: Position, history: Dict, m: Move) -> float:
        """Exact (full-window) value of the root move m, as root() counts it: a repetition is 0, black's mirror move
        gets the mirror bonus. root() gives only an upper bound for the moves after the best."""
        child = pos.play(m)
        if child in history:
            return 0
        return -self.negamax(child, self.depth - 1, -INF, INF, 1) + self.root_bonus(pos, child)

    def reaches(self, pos: Position, history: Dict, m: Move, t: float) -> bool:
        """root_value(pos, history, m) >= t, decided by one null-window search (values below the root are integers)."""
        child = pos.play(m)
        if child in history:
            return 0 >= t
        u = floor(self.root_bonus(pos, child) - t)     # -w + bonus >= t  <=>  w <= u for the integer value w
        return self.negamax(child, self.depth - 1, u, u + 1, 1) <= u


class MeanPlayer:
    """Chooses the root move by the mean of the exact scores of two HeuristicPlayers at depth - 1 and depth. An odd
    depth alone is too optimistic for the side to move (it makes the last move of the tree): 945/701 at depth 5
    rated f9-e7 +100 and depths 4 and 6 about 0; depth 5 played it and lost, the mean plays i9-h7 and wins, and it
    keeps depth 5's wins where depth 4 lost (black_test: b1-c3 against heur3, los4/033 against heur4).
    Candidates: the `top` best moves by the shallow score (a move gets an exact score only if a null-window test
    shows it can enter them); the deep score of a candidate only if a null-window test shows it can beat the best
    mean so far. Best = maximum of (mean, deep score, shallow score), the earlier candidate on a full tie."""

    def __init__(self, heuristic: Heuristic, depth: int = 5, top: int = 8, ext: int = 0, ext_free: int = 6,
                 tt_limit: int = 300000, name: str = None):
        self.lo = HeuristicPlayer(heuristic, max(1, depth - 1), ext=ext, ext_free=ext_free, tt_limit=tt_limit)
        self.hi = HeuristicPlayer(heuristic, depth, ext=ext, ext_free=ext_free, tt_limit=tt_limit)
        self.h, self.top, self.ext = heuristic, top, ext
        self.name = name or "%s (średnia gł. %d i %d%s)" % (heuristic.name, depth - 1, depth,
                                                            ", końcówka +%d" % ext if ext else "")

    @property
    def depth(self) -> int:
        return self.hi.depth

    @depth.setter
    def depth(self, d: int) -> None:                 # the window of play.py changes the depth during a game
        self.lo.depth, self.hi.depth = max(1, d - 1), d

    def reset(self):
        self.lo.reset()
        self.hi.reset()

    def candidates(self, pos: Position, history: Dict) -> Tuple[List[Move], Dict[Move, float]]:
        """The `top` best moves by the exact shallow score, in the order of a stable sort (best first)."""
        lo, top = {}, []                               # top: (index in ordered_moves, move)
        for i, m in enumerate(ordered_moves(pos)):
            if len(top) == self.top and not self.lo.reaches(pos, history, m, lo[top[-1][1]] + 1e-6):
                continue
            lo[m] = self.lo.root_value(pos, history, m)
            top.append((i, m))
            top.sort(key=lambda x: (-lo[x[1]], x[0]))
            del top[self.top:]
        return [m for _, m in top], lo

    def move(self, pos: Position, history: Optional[Dict] = None, rng=None) -> Tuple[Move, str]:
        history = history or {}
        cands, lo = self.candidates(pos, history)
        hi, best = {}, None
        key = lambda c: ((lo[c] + hi[c]) / 2, hi[c], lo[c])
        for c in cands:
            if best is not None and not self.hi.reaches(pos, history, c, lo[best] + hi[best] - lo[c]):
                continue
            hi[c] = self.hi.root_value(pos, history, c)
            if best is None or key(c) > key(best):
                best = c
        return best, "%d" % ((lo[best] + hi[best]) / 2)


class BaselinePlayer:
    """Reference opponent (middle.selfplay): search.minimax with search.rank, random among the equally best moves."""

    def __init__(self, depth: int = 3, name: str = None, tt_limit: int = 300000):
        self.depth, self.tt_limit = depth, tt_limit
        self.name = name or "bazowy alfa-beta (głębokość %d)" % depth
        self.reset()

    def reset(self):
        self.tt = {}

    def move(self, pos: Position, history=None, rng=None) -> Tuple[Move, str]:
        from .search import minimax
        rng = rng or random
        if len(self.tt) > self.tt_limit:
            self.tt.clear()
        sc = [(minimax(pos.play(m), self.depth - 1, tt=self.tt, ply=1), m) for m in pos.moves()]
        best = (max if pos.turn == BLACK else min)(v for v, _ in sc)
        return rng.choice([m for v, m in sc if v == best]), "%d" % best


class PolicyPlayer:
    """Adapter for the hand-written black policies with the signature policy(pos, last, depth, tt) -> (move, kind, s)."""

    def __init__(self, policy, name: str, depth: int = 2):
        self.policy, self.name, self.depth = policy, name, depth
        self.reset()

    def reset(self):
        self.tt = {}

    def move(self, pos: Position, history=None, rng=None) -> Tuple[Move, str]:
        from . import middle
        middle._HISTORY.clear()
        middle._HISTORY.update(history or ())
        m, kind, _ = self.policy(pos, None, self.depth, self.tt)
        return m, kind


class GameResult(NamedTuple):
    root: Position
    winner: Optional[int]        # WHITE, BLACK or None (draw: repetition or ply limit)
    plies: int
    moves: List[Move]
    reason: str                  # 'koniec' (decided), 'powtórzenie', 'limit'

    def score(self, max_plies: int, loss: float = -2.0) -> float:
        """Black's score: win 1 + up to 0.25 for speed, draw 0, loss `loss`."""
        if self.winner == BLACK:
            return 1.0 + 0.25 * max(0.0, 1.0 - self.plies / max_plies)
        return loss if self.winner == WHITE else 0.0

    def positions(self) -> List[Position]:
        out = [self.root]
        for m in self.moves:
            out.append(out[-1].play(m))
        return out


def play_game(root: Position, white, black, max_plies: int = 100, seed: int = 0) -> GameResult:
    """One game. The game ends when the exchange oracle decides it (lemma L1: the result is forced), after the
    third occurrence of a position or after max_plies (both draws)."""
    rng = random.Random(seed)
    white.reset()
    black.reset()
    p, history, moves = root, {root: 1}, []
    while True:
        wd = exchange_winner(p)
        if wd is not None:
            return GameResult(root, wd, len(moves), moves, "koniec")
        if len(moves) >= max_plies:
            return GameResult(root, None, len(moves), moves, "limit")
        m, _ = (white if p.turn == WHITE else black).move(p, history, rng)
        p = p.play(m)
        moves.append(m)
        history[p] = history.get(p, 0) + 1
        if history[p] >= 3:
            return GameResult(root, None, len(moves), moves, "powtórzenie")


def save_gif(game: GameResult, path: str, title: str = "") -> str:
    from .gif import line_gif
    pos = game.positions()
    caps = ["%s start" % title] + ["%s %d. %s %s" % (title, i + 1, "białe" if p.turn == BLACK else "czarne",
                                                     "%s-%s" % (_nm(m[0]), _nm(m[1])))
                                   for i, (p, m) in enumerate(zip(pos[1:], game.moves))]
    res = {WHITE: "wygrywają białe", BLACK: "wygrywają czarne", None: "remis (%s)" % game.reason}[game.winner]
    caps[-1] += " - " + res
    return line_gif(pos, [None] + list(game.moves), caps, path)


def _nm(s: int) -> str:
    from .board import name
    return name(s)


# ---------------------------------------------------------------------------------------------------------------
# Memory-light endgame tables (the laptop has little free RAM and the tuner runs several processes)

_CNK = [[comb(n, k) for k in range(6)] for n in range(len(cn.FREE) + 1)]
_FI = {s: i for i, s in enumerate(cn.FREE)}


def _comb_rank(idx: Sequence[int], n: int) -> int:
    """Lexicographic rank of a sorted tuple in itertools.combinations(range(n), len(idx))."""
    k, r, prev = len(idx), 0, -1
    for i, c in enumerate(idx):
        for j in range(prev + 1, c):
            r += _CNK[n - j - 1][k - i - 1]
        prev = c
    return r


class CompactTable(Mapping):
    """Solved family (owner, nw, nb): one byte per state (value + depth) in canonical.family_states order,
    the same bytes as the files in data/tables (canonical._save). Replaces the Python dicts (about 70 MB for one
    (1,2) family) by 0.36 MB."""

    def __init__(self, key, raw: bytes):
        self.key, self.raw = key, raw
        self.nb_count = _CNK[len(cn.FREE) - key[1]][key[2]]

    def index(self, state) -> int:
        owner, wf, bf, turn = state
        wi = sorted(_FI[s] for s in wf)
        bi = sorted(_FI[s] - sum(1 for x in wi if x < _FI[s]) for s in bf)
        n = len(cn.FREE)
        return 2 * (_comb_rank(wi, n) * self.nb_count + _comb_rank(bi, n - len(wi))) + turn

    def __getitem__(self, state):
        x = self.raw[self.index(state)]
        return (x & 3) - 1, x >> 2

    def __iter__(self):
        return cn.family_states(*self.key)

    def __len__(self):
        return len(self.raw)


_COMPACT: Dict[tuple, CompactTable] = {}


def _read_family(key) -> Optional[bytes]:
    try:
        with open(cn._path(key), "rb") as fh:
            return zlib.decompress(fh.read())
    except (OSError, zlib.error):
        return None


def compact_lookup(state) -> Tuple[int, int]:
    """Drop-in replacement of canonical.lookup using CompactTable. Families without a file are solved and saved
    (also the small ones, so that parallel processes do not solve them again; atomic write)."""
    owner, wf, bf, turn = state
    key = (owner, len(wf), len(bf))
    tab = _COMPACT.get(key)
    if tab is None:
        raw = _read_family(key)
        if raw is None:
            table = cn.solve_family(*key)
            raw = bytes((v + 1) | min(d, 63) << 2 for v, d in (table[s] for s in cn.family_states(*key)))
            os.makedirs(cn.CACHE_DIR, exist_ok=True)
            tmp = cn._path(key) + ".%d.tmp" % os.getpid()
            with open(tmp, "wb") as fh:
                fh.write(zlib.compress(raw, 9))
            os.replace(tmp, cn._path(key))
        tab = _COMPACT[key] = CompactTable(key, raw)
    return tab[state]


def cache_baseline_rank(limit: int = 300000) -> None:
    """Memoise search.rank (a pure function of the position) for the BaselinePlayer's search in this process."""
    from . import search
    if getattr(search.rank, "_cached", False):
        return
    orig, cache = search.rank, {}

    def rank(pos, theory_depth: int = 1):
        key = (pos, theory_depth)
        v = cache.get(key)
        if v is None:
            if len(cache) > limit:
                cache.clear()
            v = cache[key] = orig(pos, theory_depth)
        return v
    rank._cached = True
    search.rank = rank


def low_memory_tables() -> None:
    """Install the memory-light tables in this process: compact family tables, only arrays A/B of (1,3),
    the (2,3) table memory-mapped from uncompressed .npy files (created once next to 0f_23.npz, 2 x 152 MB on disk;
    all processes share the same pages)."""
    import numpy as np
    from . import retro13, retro23
    cn.lookup = compact_lookup
    if not retro13._CACHE and os.path.exists(retro13.PATH):
        z = np.load(retro13.PATH)
        retro13._CACHE.update(A=z["A"], B=z["B"])
    if not retro23._CACHE and os.path.exists(retro23.PATH):
        npy = os.path.join(cn.CACHE_DIR, "0f_23_%s.npy")
        for k in ("A", "B"):
            if not os.path.exists(npy % k):
                arr = np.load(retro23.PATH)[k]
                np.save(npy % k, arr)
                del arr
        retro23._CACHE.update(A=np.load(npy % "A", mmap_mode="r"), B=np.load(npy % "B", mmap_mode="r"))
