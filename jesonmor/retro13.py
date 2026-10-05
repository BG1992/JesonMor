"""Exact retrograde solution of the canonical (1,3) family (and (3,1) = its colour mirror), numpy.
A[x, t]: white to move, white free knight x, black free triple t; B[x, t]: black to move.
(3,1) with white to move = mirror of B. Values for the side to move: 1 win, -1 loss, 0 draw."""
import os
import time
from itertools import combinations

import numpy as np

from . import canonical as cn
from .board import BLACK, FIRST_BLACK, FIRST_WHITE, KNIGHT, WHITE, mirror

FREE = cn.FREE
F = len(FREE)
FI = {s: i for i, s in enumerate(FREE)}
TRI = list(combinations(range(F), 3))
TI = {t: k for k, t in enumerate(TRI)}
T = len(TRI)
TR = np.array(TRI)
CONT = np.zeros((F, T), bool)
for j in range(3):
    CONT[TR[:, j], np.arange(T)] = True
ATT = [[FI[t] for t in KNIGHT[s] if t in FI] for s in FREE]          # free-square neighbours
PATH = os.path.join(cn.CACHE_DIR, "0f_13.npz")


def _black_successors():
    """succ[t, j] = triple after the j-th quiet move of black, dest[t, j] = target square index (or -1)."""
    rows = []
    for tri in TRI:
        out = []
        for g in tri:
            rest = [x for x in tri if x != g]
            for d in ATT[g]:
                if d not in tri:
                    out.append((TI[tuple(sorted(rest + [d]))], d))
        rows.append(out)
    m = max(map(len, rows))
    succ, dest = np.full((T, m), -1), np.full((T, m), -1)
    for k, out in enumerate(rows):
        for j, (a, b) in enumerate(out):
            succ[k, j], dest[k, j] = a, b
    return succ, dest


def _externals(lg):
    W1 = set(FIRST_WHITE)
    # white to move: capture of a black free knight -> (1,2) black to move; ring capture -> exchange
    extA = np.full((F, T), -2, np.int8)
    for x in range(F):
        for S in (t for t in KNIGHT[FREE[x]] if t in FIRST_BLACK):
            resp = np.zeros(T, bool)
            for T_ in W1 | {S}:
                for g in (FI[s] for s in KNIGHT[T_] if s in FI):
                    resp |= CONT[g]
            extA[x] = np.maximum(extA[x], np.where(resp, -1, 1))        # answered -> capturer left with 0 free
        for u in ATT[x]:
            for k in np.nonzero(CONT[u])[0]:
                rest = tuple(FREE[i] for i in TRI[k] if i != u)
                v = -cn.value((cn.CANON, (FREE[u],), rest, BLACK))
                extA[x, k] = max(extA[x, k], v)
    lg("stopień 0: białe na ruchu - bicia wolnych skoczków -> (1,2), wymiany -> (0,2)")
    # black to move: capture of white's only free knight wins; ring capture -> white must answer with x
    extB = np.full((F, T), -2, np.int8)
    for x in range(F):
        hit = np.zeros(T, bool)
        for g in ATT[x]:
            hit |= CONT[g]
        extB[x][hit] = 1
    v02, can = {}, {s: np.array([s in KNIGHT[q] for q in FREE]) for s in FIRST_BLACK + FIRST_WHITE}
    for k, tri in enumerate(TRI):
        for g in tri:
            rest = tuple(FREE[i] for i in tri if i != g)
            for S in (t for t in KNIGHT[FREE[g]] if t in FIRST_WHITE):
                o1 = cn.CANON ^ 1 << cn.FIRST_IDX[S]
                opts = sorted(set(FIRST_BLACK) | {S})
                vals = []
                for T_ in opts:
                    key = (o1 ^ 1 << cn.FIRST_IDX[T_], rest)
                    if key not in v02:
                        v02[key] = cn.value((key[0], (), rest, BLACK))
                    vals.append(v02[key])
                col = np.full(F, 1, np.int8)                         # no white answer -> black wins
                for T_, v in zip(opts, vals):
                    col = np.where(can[T_], np.minimum(col, v), col)
                extB[:, k] = np.maximum(extB[:, k], col)
    lg("stopień 0: czarne na ruchu - bicie jedynego wolnego białego, wymiany -> (0,2) (%d stanów (0,2))" % len(v02))
    return extA, extB


def solve(log=print):
    t0 = time.time()
    lg = lambda msg: log("[%5.0fs] %s" % (time.time() - t0, msg))
    valid = ~CONT                                                        # x not in the triple
    succ, dest = _black_successors()
    lg("następniki ruchów czarnych: %d trójek x %d ruchów" % succ.shape)
    extA, extB = _externals(lg)
    nqA = np.array([len(ATT[x]) - sum((CONT[d].astype(int) for d in ATT[x]), np.zeros(T, int)) for x in range(F)])
    nqB = ((succ >= 0)[None, :, :] & (dest[None, :, :] != np.arange(F)[:, None, None])).sum(2)
    A, B = np.zeros((F, T), np.int8), np.zeros((F, T), np.int8)
    for X, ext, nq in ((A, extA, nqA), (B, extB, nqB)):
        X[valid & (ext == 1)] = 1
        X[valid & (ext != 1) & (nq == 0)] = -1
    degA, degB = np.where(A != 0, 1, 0).astype(np.int16), np.where(B != 0, 1, 0).astype(np.int16)
    report = lambda lvl, a, b: lg(
        "stopień %d: (1,3) białe na ruchu +W %d +L %d | (3,1) białe na ruchu +W %d +L %d | nieznane %d / %d"
        % (lvl, a[0], a[1], b[0], b[1], int((valid & (A == 0)).sum()), int((valid & (B == 0)).sum())))
    report(1, ((A == 1).sum(), (A == -1).sum()), ((B == 1).sum(), (B == -1).sum()))
    xs = np.arange(F)[:, None]
    level = 1
    while True:
        level += 1
        anyA, allA = np.zeros((F, T), bool), np.ones((F, T), bool)
        for x in range(F):
            for d in ATT[x]:
                c = CONT[d]
                anyA[x] |= (B[d] == -1) & ~c
                allA[x] &= (B[d] == 1) | c
        anyB, allB = np.zeros((F, T), bool), np.ones((F, T), bool)
        for j in range(succ.shape[1]):
            ok = (succ[:, j] >= 0)[None, :] & (dest[:, j][None, :] != xs)
            child = A[:, np.maximum(succ[:, j], 0)]
            anyB |= ok & (child == -1)
            allB &= ~ok | (child == 1)
        newA = [valid & (A == 0) & anyA, valid & (A == 0) & ~anyA & allA & (extA <= -1)]
        newB = [valid & (B == 0) & anyB, valid & (B == 0) & ~anyB & allB & (extB <= -1)]
        for X, deg, (nw, nl) in ((A, degA, newA), (B, degB, newB)):
            X[nw], X[nl] = 1, -1
            deg[nw | nl] = level
        report(level, (int(newA[0].sum()), int(newA[1].sum())), (int(newB[0].sum()), int(newB[1].sum())))
        if not any(m.any() for m in newA + newB):
            break
    os.makedirs(cn.CACHE_DIR, exist_ok=True)
    np.savez_compressed(PATH, A=A, B=B, degA=degA, degB=degB, valid=valid)
    n = int(valid.sum())
    for name, X in (("(1,3) białe na ruchu", A), ("(3,1) białe na ruchu", B)):
        lg("%s: stanów %d, wygrane %d (%.1f%%), przegrane %d (%.1f%%), remisy %d"
           % (name, n, (X == 1).sum(), 100 * (X == 1).sum() / n, (X == -1).sum(),
              100 * (X == -1).sum() / n, int((valid & (X == 0)).sum())))
    lg("zapisano %s (%.1f MB)" % (PATH, os.path.getsize(PATH) / 1e6))


_CACHE = {}


def value13(state) -> int:
    """Exact value for the side to move of a canonical (1,3) or (3,1) state."""
    if not _CACHE:
        _CACHE.update(np.load(PATH))
    owner, wf, bf, turn = state
    assert owner == cn.CANON and sorted((len(wf), len(bf))) == [1, 3]
    if len(wf) == 3:                                   # (3,1): mirror to (1,3), swapping colours and turn
        wf, bf, turn = tuple(mirror(s) for s in bf), tuple(mirror(s) for s in wf), 1 - turn
    x, t = FI[wf[0]], TI[tuple(sorted(FI[s] for s in bf))]
    return int((_CACHE["A"] if turn == WHITE else _CACHE["B"])[x, t])


if __name__ == "__main__":
    solve()
