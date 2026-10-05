"""Jeson Mor in the browser: the human plays white, the black AI of black_ai plays black (the opening plan of
data/plan_min3, then the mean of depths 4 and 5 with the endgame extension, as in black_test --plan; no depth to
choose, no evaluations shown). A game ends only by the rules (board.winner): a knight that survived the opponent's
move on e5, or a side without a legal move; a threefold repetition or MAX_PLIES are draws, the human may resign. The
black AI plays every game out to that end.

A server of the standard library (no tkinter, no extra packages besides numpy). The black move is searched in a
background thread (POST /api/reply starts it, the page polls GET /api/state), so a slow move does not hit a proxy
timeout of a hosting service. Idle games expire; at most MAX_GAMES are kept. --save appends finished games to
data/heuristic/human/web/games.csv (+ GIFs, needs Pillow) like the games of play.py.

  python -m jesonmor.web [--host 127.0.0.1] [--port 8765 | env PORT] [--no-browser] [--save] [--heuristic PATH]"""
import argparse
import json
import os
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from . import engine as E
from .black_ai import PlanEnginePlayer
from .board import BLACK, CENTER, WHITE, Move, Position, name, winner
from .render import _square_color

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEUR = os.path.join(ROOT, "data", "heuristic")
OUT = os.path.join(HEUR, "human", "web")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
MAX_PLIES = 400                 # the rules have no limit; a safety net against an endless game (draw)
MAX_GAMES, IDLE = 16, 30 * 60   # kept games, seconds after which an idle game is dropped
TT_LIMIT = 60000                # transposition table size of the tested black AI (black_test, evaluate945_best)
AI_LOCK = threading.Lock()      # one black search at a time (memory, CPU)
SIDE = {WHITE: "w", BLACK: "b"}


def load_heuristic(path: Optional[str] = None):
    """(heuristic, source): `path`, else data/heuristic/best.json, else the default weights."""
    path = path or os.path.join(HEUR, "best.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return E.Heuristic.from_dict(json.load(fh)), os.path.basename(path)
    return E.Heuristic(name="default"), "default weights"


def black_ai(h: E.Heuristic):
    return PlanEnginePlayer(E.MeanPlayer(h, 5, top=8, ext=4, tt_limit=TT_LIMIT))


class Game:
    """One game: the human's (white) moves, the black AI's replies, the end by the rules."""

    def __init__(self, h: E.Heuristic, source: str = "", ai=None, root: Optional[Position] = None,
                 out: Optional[str] = None):
        self.h, self.source, self.out = h, source, out
        self.ai = ai or black_ai(h)
        self.positions: List[Position] = [root or Position.start()]
        self.moves: List[Move] = []
        self.over = None                # (winner or None, reason)
        self.undos, self.t0, self.seen = 0, time.time(), time.time()
        self.lock = threading.RLock()
        self.thinking, self.version, self.error = False, 0, None

    @property
    def pos(self) -> Position:
        return self.positions[-1]

    def history(self) -> Dict[Position, int]:
        out: Dict[Position, int] = {}
        for p in self.positions:
            out[p] = out.get(p, 0) + 1
        return out

    def _play(self, m: Move) -> None:
        self.moves.append(m)
        self.positions.append(self.pos.play(m))
        p = self.pos
        wd = winner(p)
        if wd is not None:
            self.finish(wd, "center" if p.own()[0] >> CENTER & 1 else "no-move")
        elif self.positions.count(p) >= 3:
            self.finish(None, "repetition")
        elif len(self.moves) >= MAX_PLIES:
            self.finish(None, "limit")

    def finish(self, wd: Optional[int], reason: str) -> None:
        self.over = (wd, reason)
        self.version += 1
        if self.out:
            try:
                from .play import save_game         # only when saving (play.py needs tkinter, gif.py Pillow)
                game = E.GameResult(self.positions[0], wd, len(self.moves), list(self.moves), reason)
                save_game(game, self.out, self.h, self.source, [5], self.undos, time.time() - self.t0)
            except Exception:
                pass                    # a lost record must not stop the game
        if not self.thinking:
            self.ai.reset()             # frees the transposition tables

    def human(self, f: int, t: int) -> None:
        with self.lock:
            if self.over or self.thinking or self.pos.turn != WHITE:
                raise ValueError("it is not White's move")
            if (f, t) not in self.pos.moves():
                raise ValueError("illegal move")
            self._play((f, t))

    def reply(self) -> Optional[Move]:
        """The black AI's move, searched in this thread (None when the game is over or white is to move)."""
        with self.lock:
            if self.over or self.pos.turn != BLACK:
                return None
            pos, hist = self.pos, self.history()
        with AI_LOCK:
            m, _ = self.ai.move(pos, hist)
        with self.lock:
            self._play(m)
        return m

    def start_reply(self) -> None:
        """Start the black AI's search in a background thread; the move is played when it ends, unless the game
        changed meanwhile (resignation)."""
        with self.lock:
            if self.thinking or self.over or self.pos.turn != BLACK:
                return
            self.thinking, self.error = True, None
            pos, hist, ver = self.pos, self.history(), self.version

        def work():
            try:
                with AI_LOCK:
                    m, _ = self.ai.move(pos, hist)
                with self.lock:
                    if self.version == ver and not self.over:
                        self._play(m)
            except Exception as e:      # shown on the page instead of a silent thread death
                self.error = "engine error: %r" % e
            finally:
                with self.lock:
                    self.thinking = False
                    if self.over:
                        self.ai.reset()
        threading.Thread(target=work, daemon=True).start()

    def undo(self) -> None:
        """Take back the human's last move together with the black reply (not after the end, not while thinking)."""
        with self.lock:
            if self.over or self.thinking or not self.moves:
                raise ValueError("nothing to undo")
            while self.moves:
                self.moves.pop()
                self.positions.pop()
                if self.pos.turn == WHITE:
                    break
            self.undos += 1
            self.version += 1
            if hasattr(self.ai, "sync"):
                self.ai.sync(self.positions)
            else:
                self.ai.reset()

    def resign(self) -> None:
        with self.lock:
            if not self.over:
                self.finish(BLACK, "resign")

    def state(self, gid: str) -> dict:
        with self.lock:
            self.seen = time.time()
            p = self.pos
            board = [1 if p.w >> s & 1 else 2 if p.b >> s & 1 else 0 for s in range(81)]
            legal: Dict[int, List[int]] = {}
            if not self.over and not self.thinking and p.turn == WHITE:
                for f, t in p.moves():
                    legal.setdefault(f, []).append(t)
            return {"id": gid, "board": board, "turn": SIDE[p.turn], "legal": legal,
                    "last": list(self.moves[-1]) if self.moves else None,
                    "moves": ["%s-%s" % (name(f), name(t)) for f, t in self.moves],
                    "over": None if not self.over else {"winner": SIDE.get(self.over[0]), "reason": self.over[1]},
                    "thinking": self.thinking, "error": self.error,
                    "canUndo": bool(self.moves) and not self.over and not self.thinking}


class Server:
    def __init__(self, heuristic: Optional[str] = None, save: bool = False):
        self.heuristic, self.save = heuristic, save
        self.games: Dict[str, Game] = {}
        self.lock = threading.Lock()

    def new(self) -> str:
        h, source = load_heuristic(self.heuristic)
        gid = uuid.uuid4().hex[:12]
        with self.lock:
            now = time.time()
            for k in [k for k, g in self.games.items() if now - g.seen > IDLE and not g.thinking]:
                del self.games[k]
            while len(self.games) >= MAX_GAMES:
                del self.games[min(self.games, key=lambda k: self.games[k].seen)]
            self.games[gid] = Game(h, source, out=OUT if self.save else None)
        return gid


def handler(server: Server):
    colors = [_square_color(s) for s in range(81)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet console
            pass

        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def json(self, data, code: int = 200) -> None:
            self.send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    self.send(200, fh.read(), "text/html; charset=utf-8")
            elif url.path == "/healthz":
                self.send(200, b"ok", "text/plain")
            elif url.path == "/api/state":
                gid = parse_qs(url.query).get("id", [""])[0]
                g = server.games.get(gid)
                self.json(dict(g.state(gid), colors=colors) if g else {"error": "unknown game"}, 200 if g else 404)
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self.json({"error": "bad request"}, 400)
            if url.path == "/api/new":
                gid = server.new()
                return self.json(dict(server.games[gid].state(gid), colors=colors))
            gid = data.get("id", "")
            g = server.games.get(gid)
            if g is None:
                return self.json({"error": "unknown game"}, 404)
            try:
                if url.path == "/api/move":
                    g.human(int(data["from"]), int(data["to"]))
                    g.start_reply()
                elif url.path == "/api/reply":
                    g.start_reply()
                elif url.path == "/api/undo":
                    g.undo()
                elif url.path == "/api/resign":
                    g.resign()
                else:
                    return self.json({"error": "unknown command"}, 404)
                return self.json(g.state(gid))
            except (ValueError, KeyError) as e:
                return self.json({"error": str(e)}, 400)

    return Handler


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Jeson Mor in the browser: you play White against the black AI")
    ap.add_argument("--host", default="127.0.0.1", help="server address (0.0.0.0 on a hosting service)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--save", action="store_true", help="save finished games in data/heuristic/human/web (needs Pillow and tkinter)")
    ap.add_argument("--heuristic", help="heuristic JSON file (default data/heuristic/best.json)")
    a = ap.parse_args(argv)
    E.low_memory_tables()
    httpd = ThreadingHTTPServer((a.host, a.port), handler(Server(a.heuristic, a.save)))
    url = "http://%s:%d/" % ("127.0.0.1" if a.host in ("0.0.0.0", "") else a.host, a.port)
    print("Jeson Mor: %s (Ctrl+C to stop)" % url, flush=True)
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
