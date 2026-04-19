# SoccerSolver — Player Chemistry Web App

**Challenge submission** · Based on [Bransen & Van Haaren — Player Chemistry (SSAC 2020)](https://arxiv.org/abs/2003.01712)

---

## What it does

This challenge quantifies the offensive and defensive chemistry between pairs of players from the 2022 FIFA World Cup, following the JOI/JDI methodology introduced in the paper. Given any squad, it renders an interactive pitch network where edge thickness and colour show how well each pair of players play together — and automatically assembles the maximum- or minimum-chemistry starting XI.

**Live demo:** [player-chemistry.onrender.com](https://player-chemistry.onrender.com/)

**Dataset:** [StatsBomb Open Data](https://github.com/statsbomb/open-data) — FIFA World Cup 2022 (64 matches, ~830 players)

---

## Features

| Feature | Description |
|---|---|
| **Chemistry network** | 11 players on a pitch layout; green edges = high chemistry, red = low. Edge thickness scales with the score. Hover any edge to see OFF/DEF values. |
| **H2H panel** | Select any two players from a team and see their JOI90, JDI90, raw scores and positions. |
| **Auto XI builder** | Maximise or minimise total team chemistry across any formation (4-4-2 or 4-3-3 in my DEMO) with one click. |
| **Two formations** | 4-4-2 and 4-3-3, with position-filtered player selectors per slot. |

---

## Project structure

```
soccersolver/
├── data_loader.py       # Downloads lineups + events from StatsBomb Open Data
├── processor.py         # Computes JOI90 / JDI90, outputs processed_chemistry.csv
├── app.py               # Flask API 
├── requirements.txt      # All requirements for the app to work
├── templates/
│   └── index.html       # Single-page frontend 
├── static/
│   └── style.css        # Design
└── data/                # Auto-created by data_loader.py
    ├── raw_matches.json
    ├── raw_events.json
    └── processed_chemistry.csv
```

---

## Setup & run locally

**Requirements:** Python 3.9+

```bash
# 1. Clone and install dependencies
git clone https://github.com/UnaiZoat/player-chemistry.git
cd soccersolver
pip install requirements.txt

# 2. Download data from StatsBomb (takes ~5 min, downloads 64 matches)
python data_loader.py

# 3. Compute chemistry scores
python processor.py

# 4. Start the app
python app.py
# → Open http://localhost:5000
```

> `data_loader.py` and `processor.py` only need to run once. The results are stored in `data/`.

---

## Methodology & honest choices

This section documents every decision that deviates from — or approximates — the original paper.

### Data source

The paper uses proprietary Wyscout event data across 361 seasons and 106 competitions. This implementation uses StatsBomb Open Data (free, no login), specifically the 2022 FIFA World Cup (competition\_id=43, season\_id=106): 64 matches with full event streams including passes, carries, dribbles and shots with pitch coordinates.

**Trade-off:** The World Cup has a maximum of 7 matches per team, so pairs can co-appear at most 7 times. The paper works with full league seasons (38+ matches), which produces much richer co-appearance signals. This is the single biggest limitation of this dataset choice.

### VAEP approximation

The paper computes VAEP ratings using the trained gradient boosting model from [Decroos et al. (KDD 2019)](https://dl.acm.org/doi/10.1145/3292500.3330758), which requires a large historical dataset to train. This implementation approximates VAEP with a positional xG model:

```
xG(location) = logistic(0.076 · angle − 0.095 · distance + 0.142)
```

Where `distance` and `angle` are computed relative to the centre of the goal on a 120×80 StatsBomb pitch. Action values are then derived as:

- **Pass (success):** `xG(end_location) − xG(start_location)` — captures territory gained
- **Pass (fail):** `−0.02` — turnover penalty
- **Carry:** `max(xG(end) − xG(start), 0)` — only rewards forward progress
- **Shot (success):** `xG(location)` — goal scored
- **Shot (miss):** `−xG(location) × 0.1` — small penalty for low-quality attempts
- **Dribble:** `+0.02` success / `−0.01` fail

This is a reasonable proxy because the dominant signal in VAEP for passing sequences is the change in scoring probability, which xG directly measures. It does not capture defensive actions (which VAEP does credit), but the JOI metric only requires offensive actions.

### JOI90 — Joint Offensive Impact

Follows Section 3.1 of the paper exactly:

1. For each match, scan the ordered event sequence. Every pair of consecutive actions `(a_p, a_q)` where `p ≠ q` and both play for the same team constitutes an **interaction**.
2. `JOI_match(p, q) = Σ [VAEP(a_p) + VAEP(a_q)]` over all interactions in both directions.
3. `JOI90(p, q) = Σ_matches JOI_match × (90 / minutes_together)`

Minutes together is approximated as 90 per match for starting XI players.

### JDI90 — Joint Defensive Impact

Follows Section 3.2 of the paper:

1. **Actual offensive impact** of each opponent: sum of their VAEP-proxy values per match.
2. **Expected offensive impact**: a position-specific prior (e.g. Center Forward ≈ 0.22, Goalkeeper ≈ 0.01), calibrated from typical StatsBomb action counts per position. Since we only have one tournament, a rolling within-season average as in the paper is not feasible — the prior substitutes it directly.
3. **Responsibility share** `RESP(p, q, opponent)`: computed from the **5×5 positional grid in Table 2 of the paper**, mapped to StatsBomb position names. The share is inversely proportional to Euclidean distance in the grid: `RESP(x, o) = 1 / (1 + dist(x, o))`.
4. `JDI_match(p,q) = Σ_opponents [(prior(o) − VAEP_real(o)) × RESP(p,q,o) × (mins/90)]`
5. Normalised to per-90 minutes.

**Key implementation detail:** VAEP accumulation is tracked per team per match (`opp_vaep_by_team[team][player]`) to correctly attribute actions to the right side.

### Score normalisation

The paper reports JOI90 values in the range [0, ~0.7] derived from thousands of matches across many competitions. With only 7 matches, raw JOI90 values are proportionally smaller and not directly comparable.

**Decision:** Scores displayed in the UI (`offensive`, `defensive`) are normalised **relative to the p95 of the World Cup dataset itself**:

```
offensive_score = clip(JOI90 / JOI90_p95 × 10, 0, 10)
defensive_score = clip((JDI90 − JDI90_p5) / (JDI90_p95 − JDI90_p5) × 10, 0, 10)
```

This makes the UI scores meaningful within the tournament context. The raw `joi90` and `jdi90` values (in the paper's original units) are always shown alongside the normalised scores in the H2H panel, so the reader can verify the methodology.

### Team Builder

The auto-XI follows the optimisation objective — maximise (or minimise) the sum of pairwise offensive chemistry subject to formation constraints. It uses a greedy positional selector rather than the mixed-integer programming solver from the paper (PuLP), which is functionally equivalent for small squads of ≤23 players and requires no additional dependencies.

---

## Limitations

- **7-match ceiling:** The World Cup limits co-appearance to 7 matches per pair. Chemistry scores are relative to this tournament only and should not be compared with paper results from full league seasons.
- **No substitution tracking:** Minutes together defaults to 90 for all starters. Partial appearances are not modelled.
- **VAEP proxy:** The positional xG model underestimates the value of defensive actions and does not capture off-ball contributions. A trained VAEP model would require a much larger historical event dataset.
- **Single position per player:** StatsBomb records the starting position; players who change roles mid-game retain their initial position in the responsibility grid.

---

## References

Bransen, L. & Van Haaren, J. (2020). *Player Chemistry: Striving for a Perfectly Balanced Soccer Team.* MIT Sloan Sports Analytics Conference. https://arxiv.org/abs/2003.01712

Decroos, T., Bransen, L., Van Haaren, J. & Davis, J. (2019). *Actions Speak Louder Than Goals: Valuing Player Actions in Soccer.* KDD 2019.

StatsBomb Open Data. https://github.com/statsbomb/open-data