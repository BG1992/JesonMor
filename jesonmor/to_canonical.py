"""Shortest-path expansion from the start to canonical positions.
White: every move that lowers by one the minimal number of moves needed to fill the empty white first-order
squares (a shortest path). Black: only its best move (best_black_move). Lines that black wins earlier are
not expanded but recorded with their kind."""
from collections import Counter, deque
from functools import lru_cache
from typing import Dict, List

from .board import BLACK, FIRST_BLACK, FIRST_WHITE, KNIGHT, SIZE, WHITE, Position, bits, exchange_winner, mask
from .search import WIN, best_black_move, is_symmetric

W1_BB, B1_BB = mask(FIRST_WHITE), mask(FIRST_BLACK)


def _distances(target: int) -> List[int]:
    dist, queue = [-1] * SIZE, deque([target])
    dist[target] = 0
    while queue:
        s = queue.popleft()
        for t in KNIGHT[s]:
            if dist[t] < 0:
                dist[t] = dist[s] + 1
                queue.append(t)
    return dist


DIST = {t: _distances(t) for t in FIRST_WHITE}


def assign_cost(knights, targets, dist) -> int:
    """Minimal total distance assigning distinct knights to all targets (DP over subsets of targets)."""
    full, inf = (1 << len(targets)) - 1, 10 ** 6
    best = [0] + [inf] * full
    for k in knights:
        for m in range(full, 0, -1):
            for i, t in enumerate(targets):
                if m >> i & 1 and best[m ^ 1 << i] + dist[t][k] < best[m]:
                    best[m] = best[m ^ 1 << i] + dist[t][k]
    return best[full]


@lru_cache(maxsize=1 << 20)
def fill_cost(w: int) -> int:
    """Minimal total knight moves (empty-board distances) to put white knights on all empty W1 squares."""
    empty = [t for t in FIRST_WHITE if not w >> t & 1]
    return assign_cost(list(bits(w & ~W1_BB)), empty, DIST) if empty else 0


def is_canonical(pos: Position) -> bool:
    return pos.w & W1_BB == W1_BB and pos.b & B1_BB == B1_BB


def classify_white_move(pos: Position, m, depth: int = 3, tt=None) -> str:
    """'shortest' (on a shortest path to the canonical positions), 'black_wins' (black wins at once or
    by force within `depth` plies) or 'other' (a deviation black cannot punish)."""
    child = pos.play(m)
    if fill_cost(child.w) == fill_cost(pos.w) - 1:
        return "shortest"
    wd = exchange_winner(child)
    if wd is not None:
        return "black_wins" if wd == BLACK else "other"
    return "black_wins" if best_black_move(child, depth, tt)[1] >= WIN - 1000 else "other"


def allowed_white_moves(pos: Position, depth: int = 3, tt=None) -> Dict:
    """Restricted opening mode: only moves leading to the 18 canonical positions or to a black win."""
    kinds = {m: classify_white_move(pos, m, depth, tt) for m in pos.moves()}
    return {m: k for m, k in kinds.items() if k != "other"}


def expand(max_plies: int = 16, depth: int = 2, verbose: bool = False, punish: bool = False):
    """BFS to canonical positions. Returns (canonical {position: ply}, ended {position: (ply, kind)}, stats).
    With punish=True also white deviations that black wins are recorded (kind 'black_wins_after_deviation')."""
    level, canonical, ended, stats, tt = {Position.start()}, {}, {}, Counter(), {}
    for ply in range(max_plies):
        nxt = set()
        for p in level:
            wd = exchange_winner(p)
            if wd is not None:
                ended[p] = (ply, "black_wins" if wd == BLACK else "white_wins")
            elif is_canonical(p):
                canonical[p] = ply
            elif p.turn == WHITE:
                cost = fill_cost(p.w)
                moves = [m for m in p.moves() if fill_cost(p.play(m).w) == cost - 1]
                stats["white_moves"] += len(moves)
                if not moves:
                    ended[p] = (ply, "white_stuck")
                nxt.update(p.play(m) for m in moves)
                if punish:
                    for m in p.moves():
                        if m not in moves and classify_white_move(p, m, 3, tt) == "black_wins":
                            ended[p.play(m)] = (ply + 1, "black_wins_after_deviation")
                            stats["punished_deviations"] += 1
            else:
                m, score = best_black_move(p, depth, tt)
                stats["black_nodes"] += 1
                if score >= WIN - 1000:
                    ended[p] = (ply, "black_wins_by_search")
                    continue
                child = p.play(m)
                stats["black_mirror"] += is_symmetric(child)
                nxt.add(child)
        stats["ply%02d" % (ply + 1)] = len(nxt)
        if verbose:
            print("ply", ply + 1, "positions", len(nxt), "canonical", len(canonical), "ended", len(ended), flush=True)
        if not nxt:
            break
        level = nxt
    for p in level:
        if is_canonical(p):
            canonical.setdefault(p, max_plies)
    return canonical, ended, stats


if __name__ == "__main__":
    import sys
    from .storage import save
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    can, ended, stats = expand(int(args[0]) if args else 16, verbose=True, punish="--punish" in sys.argv)
    save("canonical_from_start.bin", can)
    save("black_wins_before_canonical.bin", [p for p, (_, k) in ended.items() if k.startswith("black")])
    print("canonical", len(can), Counter(can.values()))
    print("ended", Counter(k for _, k in ended.values()), Counter(pl for pl, _ in ended.values()))
    for k in sorted(stats):
        print(k, stats[k])
