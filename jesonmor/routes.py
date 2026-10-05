"""Route policy for black in the opening.
Route move = a black move lowering by one the minimal number of jumps needed to fill the empty black
first-order squares (e.g. (2,8)->(3,6), (1,8)->(0,6)->(2,5), (5,8)->(4,6)->(6,5)).
Black: if white's move progressed towards its own first-order squares and the mirror move is a route move,
play the mirror; otherwise play the best route move (short alpha-beta, proven losses rejected);
fallback best_black_move. Expansion stops at black wins and at canonical (or other full) positions."""
from collections import Counter
from functools import lru_cache

from .board import BLACK, FIRST_BB, FIRST_BLACK, KNIGHT, KNIGHT_BB, WHITE, Position, bits, exchange_winner, mirror
from .search import WIN, best_black_move, minimax
from .to_canonical import DIST, assign_cost, fill_cost, is_canonical

DIST_B = {t: [DIST[mirror(t)][mirror(s)] for s in range(81)] for t in FIRST_BLACK}


@lru_cache(maxsize=1 << 20)
def black_fill_cost(b: int) -> int:
    empty = [t for t in FIRST_BLACK if not b >> t & 1]
    return assign_cost([s for s in bits(b) if s not in FIRST_BLACK], empty, DIST_B) if empty else 0


def route_moves(pos: Position):
    cost = black_fill_cost(pos.b)
    return [m for m in pos.moves() if cost and black_fill_cost(pos.play(m).b) == cost - 1]


SAFE_DEPTH = 4      # detects all 68 losing route/mirror moves found in the first 16-ply run (depth 3 already does)


def defended(child: Position) -> bool:
    """Guard rule: every black first-order knight attacked by a white free knight has an answer - a black free
    knight attacking that square or any white first-order knight (the recapture required by lemma L4)."""
    free_b = [k for k in bits(child.b) if not FIRST_BB >> k & 1]
    free_w = [k for k in bits(child.w) if not FIRST_BB >> k & 1]
    white_ring = child.w & FIRST_BB
    for s in bits(child.b & FIRST_BB):
        if any(s in KNIGHT[k] for k in free_w):
            if not any(KNIGHT_BB[k] & (white_ring | 1 << s) for k in free_b):
                return False
    return True


def safe(child: Position, tt) -> bool:
    return defended(child) and minimax(child, SAFE_DEPTH, tt=tt, ply=1) > -(WIN - 1000)


def black_reply(pos: Position, white_move, depth: int = 2, tt=None):
    """(move, kind, score): mirror of a progressing white move or the best route move, provided it passes the
    guard rule and a SAFE_DEPTH-ply safety search; otherwise any safe move; otherwise best_black_move."""
    routes = route_moves(pos)
    progressed = fill_cost(pos.w) < fill_cost(pos.w ^ (1 << white_move[1]) | (1 << white_move[0]))
    mir = (mirror(white_move[0]), mirror(white_move[1]))
    groups = [([mir], "mirror")] if progressed and mir in routes else []
    groups += [(routes, "route"), (pos.moves(), "safe move off the route")]
    for cands, kind in groups:
        scored = sorted(((minimax(pos.play(m), depth - 1, tt=tt, ply=1), m) for m in cands), reverse=True)
        for s, m in scored:
            if s > -(WIN - 1000) and (s >= WIN - 1000 or safe(pos.play(m), tt)):
                return m, kind, s
    m, s = best_black_move(pos, depth, tt)
    return m, "default policy (no safe move)", s


def _classify(p: Position):
    wd = exchange_winner(p)
    if wd is not None:
        return ("white" if wd == WHITE else "black") + " wins"
    if is_canonical(p):
        return "canonical"
    if (p.w | p.b) & FIRST_BB == FIRST_BB:
        return "full, non-canonical"
    return None


def expand(max_plies: int = 16, depth: int = 2, out_dir: str = None, log=print, parents: dict = None, policy=None):
    """Expansion with saving: out_dir/black_wins_plyNN.bin, canonical_white.bin / canonical_black.bin (by side to
    move), full_noncanonical.bin, white_wins.bin, unfinished.bin (positions left open at the ply limit)."""
    import os
    from .storage import save
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "routes")
    os.makedirs(out_dir, exist_ok=True)
    level, tt, out = {(Position.start(), None)}, {}, Counter()
    keep = {"canonical": [], "full, non-canonical": [], "white wins": []}
    for ply in range(1, max_plies + 1):
        nxt, kinds, wins = set(), Counter(), []
        for p, last in level:
            if p.turn == WHITE:
                children = [(p.play(m), m) for m in p.moves()]
            else:
                m, kind, s = (policy or black_reply)(p, last, depth, tt)
                if s >= WIN - 1000:                       # black has a proven win: stop this line here
                    out["black wins (search)"] += 1
                    wins.append(p)
                    continue
                kinds[kind] += 1
                children = [(p.play(m), None)]
            for c, mv in children:
                k = _classify(c)
                if parents is not None and (k is None or k == "white wins") and c not in parents:
                    parents[c] = (p, mv if mv else m, None if mv else kind)
                if k is None:
                    nxt.add((c, mv))
                elif k == "black wins":
                    wins.append(c)
                    out[k] += 1
                else:
                    keep[k].append(c)
                    out[k] += 1
        save(os.path.join(out_dir, "black_wins_ply%02d.bin" % ply), set(wins))
        log("ply %d done: open %d | black moves %s | total %s"
            % (ply, len(nxt), dict(kinds), dict(out)))
        level = nxt
        if not level:
            break
    canon = set(keep["canonical"])
    save(os.path.join(out_dir, "canonical_white.bin"), [p for p in canon if p.turn == WHITE])
    save(os.path.join(out_dir, "canonical_black.bin"), [p for p in canon if p.turn == BLACK])
    save(os.path.join(out_dir, "full_noncanonical.bin"), set(keep["full, non-canonical"]))
    save(os.path.join(out_dir, "white_wins.bin"), set(keep["white wins"]))
    save(os.path.join(out_dir, "unfinished.bin"), {p for p, _ in level})
    log("END: distinct canonical positions %d (white to move %d, black to move %d) | open at the limit %d | saved in %s"
        % (len(canon), sum(p.turn == WHITE for p in canon), sum(p.turn == BLACK for p in canon), len(level), out_dir))
    return canon, out


if __name__ == "__main__":
    import sys
    expand(int(sys.argv[1]) if len(sys.argv) > 1 else 16, log=lambda s: print(s, flush=True))

def lost_lines(max_plies: int = 16, depth: int = 2, log=print):
    """Re-run the expansion remembering parents and return the move lists leading to white wins."""
    import os
    from .storage import load
    parents = {}
    tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "routes_rerun")
    expand(max_plies, depth, tmp, log, parents)
    lines = []
    for end in load(os.path.join(tmp, "white_wins.bin")):
        line, q = [], end
        while q in parents:
            prev, mv, kind = parents[q]
            line.append((prev, mv, kind))
            q = prev
        lines.append((end, line[::-1]))
    return lines
