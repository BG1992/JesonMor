"""Exact retrograde solution of the canonical (2,2) family (numpy).
State = (white free pair, black free pair), canonical ownership. Only white-to-move values are stored:
black to move is the colour mirror (c -> 8-c, colours swapped). Values for the side to move: 1 win, -1 loss, 0 draw.
Degree 0 seeds: captures of a free knight -> exact (2,1)/(1,2) tables, first-order captures -> exchange -> (1,1).
Then values are propagated backwards along quiet free-knight moves, one ply per degree."""
import os
import time
from itertools import combinations

import numpy as np

from . import canonical as cn
from .board import BLACK, FIRST_BLACK, FIRST_WHITE, KNIGHT, WHITE, mirror

FREE = cn.FREE
FI = {s: i for i, s in enumerate(FREE)}
PAIRS = list(combinations(range(len(FREE)), 2))
PI = {p: k for k, p in enumerate(PAIRS)}
P = len(PAIRS)
PA = np.array([a for a, b in PAIRS])
PB = np.array([b for a, b in PAIRS])
MP = np.array([PI[tuple(sorted((FI[mirror(FREE[a])], FI[mirror(FREE[b])])))] for a, b in PAIRS])
CONTAINS = np.zeros((len(FREE), P), bool)
OTHER = np.full((len(FREE), P), -1)
for k, (a, b) in enumerate(PAIRS):
    CONTAINS[a, k] = CONTAINS[b, k] = True
    OTHER[a, k], OTHER[b, k] = b, a
PATH = os.path.join(cn.CACHE_DIR, "0f_22.npz")


def _moves(k):
    """Quiet-or-capture targets (new pair, target index) and exchanges (rest index, first-order square)."""
    quiet, exch = [], []
    for f, rest in ((PAIRS[k][0], PAIRS[k][1]), (PAIRS[k][1], PAIRS[k][0])):
        for t in KNIGHT[FREE[f]]:
            if t in FIRST_BLACK:
                exch.append((rest, t))
            elif t in FI and FI[t] != rest:
                quiet.append((PI[tuple(sorted((rest, FI[t])))], FI[t]))
    return quiet, exch


def _exchange_vectors(log):
    """E[(rest, S)] = value for white of capturing S (black first-order) with the other knight, over black pairs."""
    tabs, out = {}, {}
    white_first = set(FIRST_WHITE)
    for S in FIRST_BLACK:
        o1 = cn.CANON ^ 1 << cn.FIRST_IDX[S]
        targets = sorted(white_first | {S})
        att = {T: np.array([T in KNIGHT[g] for g in FREE]) for T in targets}
        for T in targets:
            mask = o1 ^ 1 << cn.FIRST_IDX[T]
            if mask not in tabs:
                tabs[mask] = cn.solve_family(mask, 1, 1)
        for r in range(len(FREE)):
            best = np.full(P, 2, np.int8)
            for T in targets:
                tab = tabs[o1 ^ 1 << cn.FIRST_IDX[T]]
                m = np.array([tab[(o1 ^ 1 << cn.FIRST_IDX[T], (FREE[r],), (FREE[u],), WHITE)][0] if u != r else 2
                              for u in range(len(FREE))], np.int8)
                for g, u in ((PA, PB), (PB, PA)):
                    best = np.where(att[T][g], np.minimum(best, m[u]), best)
            out[(r, S)] = np.where(best == 2, 1, best).astype(np.int8)
    log("degree 0: ring exchanges -> (1,1): %d mask tables, %d vectors" % (len(tabs), len(out)))
    return out


def _capture_matrix(log):
    """CAP[k2, u] = value for white after capturing a black free knight: (2,1) state, black to move."""
    cap = np.zeros((P, len(FREE)), np.int8)
    for k2, (a, b) in enumerate(PAIRS):
        for u in range(len(FREE)):
            if u != a and u != b:
                cap[k2, u] = -cn.value((cn.CANON, (FREE[a], FREE[b]), (FREE[u],), BLACK))
    log("degree 0: captures of free knights -> (2,1)/(1,2): %d states read" % int((cap != 0).sum()))
    return cap


def solve(log=print):
    t0 = time.time()
    lg = lambda msg: log("[%5.0fs] %s" % (time.time() - t0, msg))
    moves = [_moves(k) for k in range(P)]
    E, CAP = _exchange_vectors(lg), _capture_matrix(lg)
    member = CONTAINS.T.astype(np.int16)                    # P x 72
    valid = (member @ member.T) == 0                        # disjoint pairs
    ext = np.full((P, P), -2, np.int8)
    nquiet = np.zeros((P, P), np.int16)
    for k in range(P):
        quiet, exch = moves[k]
        row = ext[k]
        for rest, S in exch:
            np.maximum(row, E[(rest, S)], out=row)
        for k2, ti in quiet:
            c = CONTAINS[ti]
            row[c] = np.maximum(row[c], CAP[k2, OTHER[ti, c]])
            nquiet[k] += ~c
    W = np.zeros((P, P), np.int8)
    W[valid & (ext == 1)] = 1
    W[valid & (ext != 1) & (nquiet == 0)] = -1
    deg = np.zeros((P, P), np.int16)
    deg[W != 0] = 1
    n = int(valid.sum())
    lg("degree 1: states %d, wins %d, losses %d, unknown %d"
       % (n, int((W == 1).sum()), int((W == -1).sum()), int((valid & (W == 0)).sum())))
    level = 1
    while True:
        level += 1
        WT = W.T.copy()
        anyL = np.zeros((P, P), bool)
        allW = np.ones((P, P), bool)
        for k in range(P):
            for k2, ti in moves[k][0]:
                vals = WT[MP[k2]][MP]                        # black-to-move child values (mirror)
                c = CONTAINS[ti]
                anyL[k] |= (vals == -1) & ~c
                allW[k] &= (vals == 1) | c
        unknown = valid & (W == 0)
        new_w = unknown & anyL
        new_l = unknown & ~anyL & allW & (ext <= -1)
        W[new_w], W[new_l] = 1, -1
        deg[new_w | new_l] = level
        lg("degree %d: +%d wins, +%d losses | total W %d, L %d, unknown %d"
           % (level, int(new_w.sum()), int(new_l.sum()), int((W == 1).sum()), int((W == -1).sum()),
              int((valid & (W == 0)).sum())))
        if not new_w.any() and not new_l.any():
            break
    os.makedirs(cn.CACHE_DIR, exist_ok=True)
    np.savez_compressed(PATH, W=W, deg=deg, valid=valid)
    lg("done: draws (endless game) %d; saved %s (%.1f MB)"
       % (int((valid & (W == 0)).sum()), PATH, os.path.getsize(PATH) / 1e6))
    return W, deg, valid


_CACHE = {}


def value22(state) -> int:
    """Exact value for the side to move of a canonical (2,2) state (owner must be CANON)."""
    if "W" not in _CACHE:
        _CACHE["W"] = np.load(PATH)["W"]
    owner, wf, bf, turn = state
    assert owner == cn.CANON and len(wf) == 2 and len(bf) == 2
    if turn == BLACK:
        wf, bf = tuple(mirror(s) for s in bf), tuple(mirror(s) for s in wf)
    pw, pb = PI[tuple(sorted(FI[s] for s in wf))], PI[tuple(sorted(FI[s] for s in bf))]
    return int(_CACHE["W"][pw, pb])


if __name__ == "__main__":
    solve()
