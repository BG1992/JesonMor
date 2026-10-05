"""Small position images: SVG (text, ~2-4 KB) or palette PNG (~0.5 KB, needs Pillow).
Orientation: white at the bottom, black at the top; row letters a..i (r) along x, columns 1..9 (c) upwards."""
import io
import time

from .board import CENTER, FIRST_BLACK, FIRST_WHITE, N, Position, bits, rc

COLORS = {"light": "#eed9b6", "dark": "#b58863", "center": "#d9534f", "fw": "#9fc5e8", "fb": "#b6d7a8"}


def _square_color(s: int) -> str:
    if s == CENTER:
        return COLORS["center"]
    if s in FIRST_WHITE:
        return COLORS["fw"]
    if s in FIRST_BLACK:
        return COLORS["fb"]
    r, c = rc(s)
    return COLORS["dark" if (r + c) % 2 else "light"]


def _xy(s: int, cell: int, left: int = 0):
    """Top-left screen corner of square s (left = label margin in cells)."""
    r, c = rc(s)
    return (r + left) * cell, (N - 1 - c) * cell


def to_svg(pos: Position, cell: int = 24, path: str = None) -> str:
    size = cell * (N + 1)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" '
           'font-size="%d" text-anchor="middle">' % (size, size, cell // 2)]
    out.append('<defs><pattern id="k" width="%d" height="%d" patternUnits="userSpaceOnUse" x="%d">'
               '<rect width="%d" height="%d" fill="%s"/><rect width="%d" height="%d" fill="%s"/>'
               '<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/></pattern></defs>'
               % (2 * cell, 2 * cell, cell, 2 * cell, 2 * cell, COLORS["dark"], cell, cell, COLORS["light"],
                  cell, cell, cell, cell, COLORS["light"]))
    out.append('<rect x="%d" width="%d" height="%d" fill="url(#k)"/>' % (cell, N * cell, N * cell))
    for s in [CENTER] + FIRST_WHITE + FIRST_BLACK:
        x, y = _xy(s, cell, 1)
        out.append('<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>' % (x, y, cell, cell, _square_color(s)))
    for i in range(N):
        out.append('<text x="%d" y="%d">%d</text>' % (cell // 2, (N - 1 - i) * cell + cell * 2 // 3, i + 1))
        out.append('<text x="%d" y="%d">%s</text>' % ((i + 1.5) * cell, N * cell + cell * 2 // 3, "abcdefghi"[i]))
    for bb, fill in ((pos.w, "#fff"), (pos.b, "#000")):
        for s in bits(bb):
            x, y = _xy(s, cell, 1)
            out.append('<circle cx="%d" cy="%d" r="%d" fill="%s" stroke="#444"/>'
                       % (x + cell // 2, y + cell // 2, cell * 0.36, fill))
    out.append('<circle cx="%d" cy="%d" r="%d" fill="%s" stroke="#444"/></svg>'   # side to move marker
               % (cell // 2, N * cell + cell // 2, cell // 4, "#fff" if pos.turn == 0 else "#000"))
    svg = "".join(out)
    if path:
        with open(path, "w") as fh:
            fh.write(svg)
    return svg


def to_png(pos: Position, path: str, cell: int = 20) -> None:
    """Palette PNG without labels; side to move shown by a thin bar (white/black) under the board."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (cell * N, cell * N + 4), "#808080")
    d = ImageDraw.Draw(img)
    for s in range(N * N):
        x, y = _xy(s, cell)
        d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=_square_color(s))
    m = cell // 6
    for bb, fill in ((pos.w, "#ffffff"), (pos.b, "#000000")):
        for s in bits(bb):
            x, y = _xy(s, cell)
            d.ellipse([x + m, y + m, x + cell - m, y + cell - m], fill=fill, outline="#444444")
    d.rectangle([0, cell * N, cell * N, cell * N + 4], fill="#ffffff" if pos.turn == 0 else "#000000")
    buf = io.BytesIO()
    img.quantize(colors=16).save(buf, format="PNG", optimize=True)
    for attempt in range(10):             # Windows may briefly lock a freshly written file (indexer/AV)
        try:
            with open(path, "wb") as fh:
                fh.write(buf.getvalue())
            return
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.3)
