"""Compact position storage: 21-byte records (81+81+1 bits), optional zlib files."""
import zlib
from typing import Iterable, List

from .board import Position, bits, mask, name, sq

RECORD = 21  # ceil(163 / 8)


def encode(pos: Position) -> bytes:
    return (pos.w | pos.b << 81 | pos.turn << 162).to_bytes(RECORD, "little")


def decode(data: bytes) -> Position:
    v = int.from_bytes(data, "little")
    full = (1 << 81) - 1
    return Position(v & full, v >> 81 & full, v >> 162 & 1)


def save(path: str, positions: Iterable[Position], compress: bool = True) -> int:
    raw = b"".join(encode(p) for p in positions)
    with open(path, "wb") as fh:
        fh.write(b"Z" + zlib.compress(raw, 9) if compress else b"R" + raw)
    return len(raw) // RECORD


def load(path: str) -> List[Position]:
    with open(path, "rb") as fh:
        head, raw = fh.read(1), fh.read()
    if head == b"Z":
        raw = zlib.decompress(raw)
    return [decode(raw[i:i + RECORD]) for i in range(0, len(raw), RECORD)]


def to_text(pos: Position) -> str:
    """Human readable form, e.g. 'W:a1,b1 B:a9 w'."""
    side = lambda bb: ",".join(name(s) for s in bits(bb))
    return "W:%s B:%s %s" % (side(pos.w), side(pos.b), "wb"[pos.turn])


def from_text(text: str) -> Position:
    w, b, t = text.split()
    parse = lambda part: mask(sq("abcdefghi".index(x[0]), int(x[1:]) - 1) for x in part[2:].split(",") if x)
    return Position(parse(w), parse(b), "wb".index(t))
