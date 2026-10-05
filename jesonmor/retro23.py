"""Exact retrograde solution of the canonical (2,3) family (and (3,2) = its colour mirror), numpy, low memory.
A[k, t]: white to move, white free pair k, black free triple t; B[k, t]: black to move.
(3,2) with white to move = mirror of B. Values for the side to move: 1 win, -1 loss, 0 draw.
Degree 0 seeds: free-knight captures -> exact (2,2) / (1,3) tables; ring exchanges -> (1,2) with 17 masks."""
import os
import time
from itertools import combinations

import numpy as np

from . import canonical as cn
from . import retro13, retro22
from .board import BLACK, FIRST_BLACK, FIRST_WHITE, KNIGHT, WHITE, mirror

FREE, F = cn.FREE, len(cn.FREE)
FI = {s: i for i, s in enumerate(FREE)}
PAIRS, PI, P = retro22.PAIRS, retro22.PI, retro22.P
PA, PB, MP2 = retro22.PA, retro22.PB, retro22.MP
C2, OTHER2 = retro22.CONTAINS, retro22.OTHER                    # (F, P)
TRI, TI, T = retro13.TRI, retro13.TI, retro13.T
TR, C3 = retro13.TR, retro13.CONT                               # (T, 3), (F, T)
ATT = retro13.ATT
RP = np.array([[PI[tuple(x for x in tri if x != tri[j])] for j in range(3)] for tri in TRI])   # (T, 3)
ATTACKS = {s: np.array([s in KNIGHT[q] for q in FREE]) for s in FIRST_WHITE + FIRST_BLACK}
PATH = os.path.join(cn.CACHE_DIR, "0f_23.npz")
CHUNK = 128


def _v12(masks, lg):
    """(1,2) values per mask as arrays: Vw[m][r, p] (white to move), Vb[m][w, p] (black to move)."""
    Vw, Vb = {}, {}
    for m in sorted(masks):
        key = (m, 1, 2)
        tab = cn._load(key)
        if tab is None:
            tab = cn.solve_family(*key)
            cn._save(key, tab)
        vw, vb = np.zeros((F, P), np.int8), np.zeros((F, P), np.int8)
        for (o, wf, bf, turn), (v, d) in tab.items():
            (vw if turn == WHITE else vb)[FI[wf[0]], PI[(FI[bf[0]], FI[bf[1]])]] = v
        Vw[m], Vb[m] = vw, vb
        del tab
        lg("stopień 0: tablica (1,2) dla maski %02x gotowa (%d/%d)" % (m, len(Vw), len(masks)))
    return Vw, Vb


def _masks():
    out = {cn.CANON}
    for S in FIRST_BLACK:
        for Tq in FIRST_WHITE:
            out.add(cn.CANON ^ 1 << cn.FIRST_IDX[S] ^ 1 << cn.FIRST_IDX[Tq])
    return out


def _mask(S, Tq):
    return cn.CANON ^ 1 << cn.FIRST_IDX[S] ^ 1 << cn.FIRST_IDX[Tq]


def _white_moves(k):
    quiet, exch = [], []
    for f, rest in ((PAIRS[k][0], PAIRS[k][1]), (PAIRS[k][1], PAIRS[k][0])):
        for t in KNIGHT[FREE[f]]:
            if t in FIRST_BLACK:
                exch.append((rest, t))
            elif t in FI and FI[t] != rest:
                quiet.append((PI[tuple(sorted((rest, FI[t])))], FI[t], rest))
    return quiet, exch


def _ext_white(moves, Vw, W22, lg):
    ext = np.full((P, T), -2, np.int8)
    E = {}
    for r in range(F):                                       # ring exchange vectors over triples
        for S in FIRST_BLACK:
            best = np.full(T, 2, np.int8)
            for Tq in set(FIRST_WHITE) | {S}:
                vw = Vw[_mask(S, Tq)][r]
                for j in range(3):
                    best = np.where(ATTACKS[Tq][TR[:, j]], np.minimum(best, vw[RP[:, j]]), best)
            E[(r, S)] = np.where(best == 2, 1, best).astype(np.int8)
    lg("stopień 0: białe na ruchu - %d wektorów wymian -> (1,2)" % len(E))
    idx_u = [np.nonzero(C3[u])[0] for u in range(F)]
    rest_u = [np.array([PI[tuple(x for x in TRI[t] if x != u)] for t in idx_u[u]]) for u in range(F)]
    for k in range(P):
        row = ext[k]
        for rest, S in moves[k][1]:
            np.maximum(row, E[(rest, S)], out=row)
        for k2, u, rest in moves[k][0]:
            vals = -W22[MP2[rest_u[u]], MP2[k2]]            # (2,2) black to move, via mirror
            row[idx_u[u]] = np.maximum(row[idx_u[u]], vals)
    lg("stopień 0: białe na ruchu - bicia wolnych czarnych -> (2,2)")
    return ext


def _ext_black(succ, dest, Vb, A13, lg):
    ext = np.full((P, T), -2, np.int8)
    for d in range(F):                                       # capture of a white free knight -> (1,3)
        tt, jj = np.nonzero(dest == d)
        tp = succ[tt, jj]
        for k in np.nonzero(C2[d])[0]:
            np.maximum.at(ext[k], tt, -A13[OTHER2[d, k], tp])
    lg("stopień 0: czarne na ruchu - bicia wolnych białych -> (1,3)")
    for j in range(3):                                       # ring exchanges -> (1,2)
        for S in FIRST_WHITE:
            sel = np.nonzero(ATTACKS[S][TR[:, j]])[0]
            rp = RP[sel, j]
            best = np.full((P, len(sel)), 2, np.int8)
            for Tq in set(FIRST_BLACK) | {S}:
                vb = Vb[_mask(Tq, S) if Tq != S else cn.CANON]
                for f, o in ((PA, PB), (PB, PA)):
                    can = ATTACKS[Tq][f]
                    val = vb[o[can][:, None], rp[None, :]]
                    best[can] = np.minimum(best[can], val)
            ext[:, sel] = np.maximum(ext[:, sel], np.where(best == 2, 1, best))
    lg("stopień 0: czarne na ruchu - wymiany w pierścieniu -> (1,2)")
    return ext


def solve(log=print):
    t0 = time.time()
    lg = lambda msg: log("[%5.0fs] %s" % (time.time() - t0, msg))
    W22 = np.load(retro22.PATH)["W"]
    A13 = np.load(retro13.PATH)["A"]
    Vw, Vb = _v12(_masks(), lg)
    moves = [_white_moves(k) for k in range(P)]
    succ, dest = retro13._black_successors()
    valid = lambda k0, k1: ~((C2[:, k0:k1].T.astype(np.uint8) @ C3.astype(np.uint8)) > 0)
    extA = _ext_white(moves, Vw, W22, lg)
    extB = _ext_black(succ, dest, Vb, A13, lg)
    del Vw, Vb
    A, B = np.zeros((P, T), np.int8), np.zeros((P, T), np.int8)
    DA, DB = np.zeros((P, T), np.uint8), np.zeros((P, T), np.uint8)   # degree (bits 0-6), ext draw (bit 7)
    nvalid = 0
    for k0 in range(0, P, CHUNK):
        k1 = min(P, k0 + CHUNK)
        v = valid(k0, k1)
        nvalid += int(v.sum())
        for X, D, ext in ((A, DA, extA), (B, DB, extB)):
            X[k0:k1][v & (ext[k0:k1] == 1)] = 1
            D[k0:k1][v & (ext[k0:k1] == 1)] = 1
            D[k0:k1][v & (ext[k0:k1] == 0)] |= 128
            D[k0:k1][~v] = 127                                            # invalid (overlapping) states
    del extA, extB
    unknown = lambda X, D: int(((X == 0) & (D != 127)).sum())
    lg("stopień 1: stanów %d | (2,3) białe na ruchu +W %d | (3,2) białe na ruchu +W %d | nieznane %d / %d"
       % (nvalid, int((A == 1).sum()), int((B == 1).sum()), unknown(A, DA), unknown(B, DB)))
    level = 1
    while True:
        level += 1
        newA = np.zeros((P, T), np.int8)
        for k in range(P):                                   # white to move: children are B rows
            anyL, allW = np.zeros(T, bool), np.ones(T, bool)
            for k2, u, rest in moves[k][0]:
                vals, c = B[k2], C3[u]
                anyL |= (vals == -1) & ~c
                allW &= (vals == 1) | c
            open_ = (A[k] == 0) & (DA[k] != 127)
            newA[k][open_ & anyL] = 1
            newA[k][open_ & ~anyL & allW & (DA[k] < 128)] = -1
        nB = [0, 0]
        for k0 in range(0, P, CHUNK):                        # black to move: children are A columns
            k1 = min(P, k0 + CHUNK)
            anyL, allW = np.zeros((k1 - k0, T), bool), np.ones((k1 - k0, T), bool)
            for j in range(succ.shape[1]):
                ok = (succ[:, j] >= 0)[None, :] & ~C2[np.maximum(dest[:, j], 0), k0:k1].T
                child = A[k0:k1][:, np.maximum(succ[:, j], 0)]
                anyL |= ok & (child == -1)
                allW &= ~ok | (child == 1)
            Bc, Dc = B[k0:k1], DB[k0:k1]
            open_ = (Bc == 0) & (Dc != 127)
            w, l = open_ & anyL, open_ & ~anyL & allW & (Dc < 128)
            Bc[w], Bc[l] = 1, -1
            Dc[w | l] = level
            nB[0] += int(w.sum())
            nB[1] += int(l.sum())
        wA, lA = newA == 1, newA == -1
        A[wA], A[lA] = 1, -1
        DA[wA | lA] = level
        nA = [int(wA.sum()), int(lA.sum())]
        del newA, wA, lA
        lg("stopień %d: (2,3) białe na ruchu +W %d +L %d | (3,2) białe na ruchu +W %d +L %d | nieznane %d / %d"
           % (level, nA[0], nA[1], nB[0], nB[1], unknown(A, DA), unknown(B, DB)))
        if not any(nA + nB):
            break
    os.makedirs(cn.CACHE_DIR, exist_ok=True)
    np.savez_compressed(PATH, A=A, B=B)
    for name, X, D in (("(2,3) białe na ruchu", A, DA), ("(3,2) białe na ruchu", B, DB)):
        ok = D != 127
        lg("%s: stanów %d, wygrane %d (%.1f%%), przegrane %d (%.1f%%), remisy %d, najdłuższa partia %d"
           % (name, nvalid, (X == 1).sum(), 100 * (X == 1).sum() / nvalid, (X == -1).sum(),
              100 * (X == -1).sum() / nvalid, int(((X == 0) & ok).sum()), int((D[ok] & 127).max())))
    lg("zapisano %s (%.1f MB)" % (PATH, os.path.getsize(PATH) / 1e6))


_CACHE = {}


def value23(state) -> int:
    """Exact value for the side to move of a canonical (2,3) or (3,2) state."""
    if not _CACHE:
        _CACHE.update(np.load(PATH))
    owner, wf, bf, turn = state
    assert owner == cn.CANON and sorted((len(wf), len(bf))) == [2, 3]
    if len(wf) == 3:
        wf, bf, turn = tuple(mirror(s) for s in bf), tuple(mirror(s) for s in wf), 1 - turn
    k, t = PI[tuple(sorted(FI[s] for s in wf))], TI[tuple(sorted(FI[s] for s in bf))]
    return int((_CACHE["A"] if turn == WHITE else _CACHE["B"])[k, t])


if __name__ == "__main__":
    solve()
