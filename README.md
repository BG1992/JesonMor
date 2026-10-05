---
title: Jeson Mor 9x9
emoji: ♞
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Jeson Mor 9×9 – play White against the black AI

A browser game. You play White, the computer plays Black: first a fixed opening plan, checked against every
White move (it ends in a Black win or in one of 945 canonical positions), then an alpha-beta engine that chooses
its move by the mean of the scores at depths 4 and 5, with an endgame extension. In tests it won 210 of 210 full
games from the start (three White engines, every first move and 40 random openings) and all 945 canonical
positions against the reference alpha-beta.

**Rules.** A 9×9 board, 9 knights a side. You win by jumping onto the centre square e5 and keeping your knight
there through your opponent's move. A side with no legal move loses. A threefold repetition is a draw.
Every game is played out to such an end.

## Run locally

Python 3.9 or newer:

```bash
pip install -r requirements.txt
python -m jesonmor.web
```

The page opens at `http://127.0.0.1:8765/`. On the first start the server unpacks an endgame table (2 × 152 MB in
`data/tables/`, a few seconds). A Black move usually takes a few seconds, rarely up to a minute or two.

Options: `--host`, `--port` (or the `PORT` variable), `--no-browser`, `--save` (saves games, needs Pillow and
tkinter).

## Tests

```bash
python tests/test_web.py
```

## Contents

* `jesonmor/web.py` – server (standard library) and game logic; `jesonmor/web/index.html` – the page.
* `jesonmor/black_ai.py` – the black AI: the opening plan (`plan.py`), then `engine.MeanPlayer`.
* `jesonmor/engine.py`, `search.py`, `canonical.py`, `retro*.py` – engine, heuristic features and endgame tables.
* `data/heuristic/best.json` – heuristic weights; `data/tables/` – endgame tables.

This folder is generated from the analysis project with `python -m jesonmor.build_webapp`.
