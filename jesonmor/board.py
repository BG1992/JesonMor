"""Jeson Mor 9x9 rules: geometry, positions, move generation, terminal detection."""
from typing import Iterator, List, NamedTuple, Optional, Tuple

N = 9
SIZE = N * N
CENTER = 4 * N + 4
WHITE, BLACK = 0, 1
JUMPS = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))

Move = Tuple[int, int]


def sq(r: int, c: int) -> int:
    return r * N + c


def rc(s: int) -> Tuple[int, int]:
    return divmod(s, N)


def name(s: int) -> str:
    """Chess-like name: row letter a..i + column number 1..9, e.g. (4,4) -> 'e5'."""
    r, c = rc(s)
    return "abcdefghi"[r] + str(c + 1)


def bits(bb: int) -> Iterator[int]:
    while bb:
        low = bb & -bb
        yield low.bit_length() - 1
        bb ^= low


def mask(squares) -> int:
    m = 0
    for s in squares:
        m |= 1 << s
    return m


KNIGHT = [[sq(r + dr, c + dc) for dr, dc in JUMPS if 0 <= r + dr < N and 0 <= c + dc < N]
          for r in range(N) for c in range(N)]
KNIGHT_BB = [mask(t) for t in KNIGHT]

FIRST_WHITE = [sq(2, 3), sq(3, 2), sq(5, 2), sq(6, 3)]
FIRST_BLACK = [sq(2, 5), sq(3, 6), sq(5, 6), sq(6, 5)]
FIRST = FIRST_WHITE + FIRST_BLACK
FIRST_BB = mask(FIRST)


def _rings() -> List[int]:
    """Ring (order) of every square: 0 = center, 1 = first order, 2 = second, ..."""
    ring, frontier, k = [-1] * SIZE, [CENTER], 0
    ring[CENTER] = 0
    while frontier:
        k += 1
        nxt = [t for s in frontier for t in KNIGHT[s] if ring[t] < 0]
        for t in nxt:
            ring[t] = k
        frontier = list(set(nxt))
    return ring


RING = _rings()


def mirror(s: int) -> int:
    """Colour-swapping mirror c -> 8 - c (maps white first-order squares to black ones)."""
    r, c = rc(s)
    return sq(r, N - 1 - c)


class Position(NamedTuple):
    w: int      # white bitboard (81 bits)
    b: int      # black bitboard
    turn: int   # WHITE or BLACK to move

    @classmethod
    def start(cls) -> "Position":
        return cls(mask(sq(r, 0) for r in range(N)), mask(sq(r, N - 1) for r in range(N)), WHITE)

    @classmethod
    def from_squares(cls, white, black, turn: int = WHITE) -> "Position":
        """Build from iterables of (r, c) pairs or square indices."""
        conv = lambda it: mask(s if isinstance(s, int) else sq(*s) for s in it)
        return cls(conv(white), conv(black), turn)

    def own(self) -> Tuple[int, int]:
        return (self.w, self.b) if self.turn == WHITE else (self.b, self.w)

    def moves(self) -> List[Move]:
        own, _ = self.own()
        return [(f, t) for f in bits(own) for t in bits(KNIGHT_BB[f] & ~own)]

    def play(self, m: Move) -> "Position":
        f, t = m
        own, opp = self.own()
        own = own ^ (1 << f) | (1 << t)
        opp &= ~(1 << t)
        return Position(own, opp, BLACK) if self.turn == WHITE else Position(opp, own, WHITE)

    def mirrored(self) -> "Position":
        """Colour-swapped mirror image (white <-> black, c -> 8 - c, turn swapped)."""
        flip = lambda bb: mask(mirror(s) for s in bits(bb))
        return Position(flip(self.b), flip(self.w), 1 - self.turn)

    def attackers(self) -> Tuple[int, int]:
        """Number of white and black knights standing on first-order squares."""
        return bin(self.w & FIRST_BB).count("1"), bin(self.b & FIRST_BB).count("1")


def winner(pos: Position) -> Optional[int]:
    """Exact rule check: WHITE/BLACK if decided, None otherwise.
    A knight of the side to move on the center survived the opponent's move -> that side wins.
    A side with no legal move (e.g. no knights) loses."""
    own, _ = pos.own()
    if own >> CENTER & 1:
        return pos.turn
    if not any(KNIGHT_BB[f] & ~own for f in bits(own)):
        return 1 - pos.turn
    return None


def exchange_winner(pos: Position) -> Optional[int]:
    """Forced-exchange oracle (exact). Moves into / captures on the center are forced sequences,
    so a position is decided when the center is occupied or the side to move has more attackers."""
    wd = winner(pos)
    if wd is not None:
        return wd
    aw, ab = pos.attackers()
    me, opp = (aw, ab) if pos.turn == WHITE else (ab, aw)
    own, other = pos.own()
    if other >> CENTER & 1:                       # must capture on center: wins iff me > opp
        return pos.turn if me > opp else 1 - pos.turn
    if me > opp:                                  # enter the center and win the exchange
        return pos.turn
    return None
