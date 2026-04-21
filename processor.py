"""
Implementación de Player Chemistry (Bransen & Van Haaren, SSAC 2020)
usando StatsBomb Open Data del Mundial 2022.

JOI90(p,q): suma de VAEP-proxy de interacciones consecutivas entre p y q,
            normalizada por minutos juntos en cancha, por 90'.
JDI90(p,q): para cada oponente o, (VAEP_esperado(o) - VAEP_real(o))
            * responsabilidad_del_par(p,q,o), normalizado por minutos juntos.

Sin socceraction: el VAEP se aproxima con expected goals posicionales
basados en la ubicación de las acciones (zona del campo → xG proxy).
La responsabilidad defensiva usa el grid 5×5 de la Tabla 2 del paper.
"""

import json
import math
import os
from collections import defaultdict
from itertools import combinations

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Grid 5x5 del paper (Tabla 2)
#    Fila 0 = zona mas ofensiva, fila 4/5 = mas defensiva
#    Col 0 = izquierda, col 4 = derecha (perspectiva del equipo)
# ---------------------------------------------------------------------------
POSITION_GRID = {
    "Left Wing Forward":           (0, 0),
    "Striker":                     (0, 2),
    "Center Forward":              (0, 2),
    "Right Wing Forward":          (0, 4),
    "Left Attacking Midfield":     (1, 1),
    "Center Attacking Midfield":   (1, 2),
    "Second Striker":              (1, 2),
    "Right Attacking Midfield":    (1, 3),
    "Left Wing":                   (2, 0),
    "Left Center Midfield":        (2, 1),
    "Center Midfield":             (2, 2),
    "Right Center Midfield":       (2, 3),
    "Right Wing":                  (2, 4),
    "Left Wing Back":              (3, 0),
    "Left Defensive Midfield":     (3, 1),
    "Center Defensive Midfield":   (3, 2),
    "Right Defensive Midfield":    (3, 3),
    "Right Wing Back":             (3, 4),
    "Left Back":                   (4, 0),
    "Left Center Back":            (4, 1),
    "Center Back":                 (4, 2),
    "Right Center Back":           (4, 3),
    "Right Back":                  (4, 4),
    "Goalkeeper":                  (5, 2),
    "Substitute":                  (2, 2),
}

def get_grid_pos(pos_name):
    if pos_name in POSITION_GRID:
        return POSITION_GRID[pos_name]
    p = str(pos_name).lower()
    if "goalkeeper" in p:                              return (5, 2)
    if "left" in p and "wing" in p and "back" in p:   return (3, 0)
    if "right" in p and "wing" in p and "back" in p:  return (3, 4)
    if "left" in p and "center" in p and "back" in p: return (4, 1)
    if "right" in p and "center" in p and "back" in p:return (4, 3)
    if "left" in p and "back" in p:                   return (4, 0)
    if "right" in p and "back" in p:                  return (4, 4)
    if "center" in p and "back" in p:                 return (4, 2)
    if "left" in p and "defensive" in p:              return (3, 1)
    if "right" in p and "defensive" in p:             return (3, 3)
    if "defensive" in p and "mid" in p:               return (3, 2)
    if "left" in p and "attacking" in p:              return (1, 1)
    if "right" in p and "attacking" in p:             return (1, 3)
    if "attacking" in p and "mid" in p:               return (1, 2)
    if "left" in p and "wing" in p:                   return (2, 0)
    if "right" in p and "wing" in p:                  return (2, 4)
    if "left" in p and "mid" in p:                    return (2, 1)
    if "right" in p and "mid" in p:                   return (2, 3)
    if "center" in p and "mid" in p:                  return (2, 2)
    if "forward" in p or "striker" in p:              return (0, 2)
    return (2, 2)

def grid_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def responsibility(grid_p, grid_q, grid_opp):
    """RESP(p,q,o) = (1/(1+d(p,o)) + 1/(1+d(q,o))) / 2"""
    rp = 1.0 / (1.0 + grid_distance(grid_p, grid_opp))
    rq = 1.0 / (1.0 + grid_distance(grid_q, grid_opp))
    return (rp + rq) / 2.0

# ---------------------------------------------------------------------------
# 2. VAEP proxy (xG posicional)
# ---------------------------------------------------------------------------
GOAL_X = 120.0
GOAL_Y = 40.0

def xg_from_location(loc):
    if not loc or not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return 0.02
    x, y = float(loc[0]), float(loc[1])
    dist = math.sqrt((GOAL_X - x)**2 + (GOAL_Y - y)**2)
    denom = (GOAL_X - x)**2 + (GOAL_Y - y)**2 - (7.32/2)**2
    angle = math.atan2(7.32 * (GOAL_X - x), denom) if denom != 0 else 0.0
    angle = max(angle, 0.0)
    log_odds = 0.076 * angle - 0.095 * dist + 0.142
    return min(max(1.0 / (1.0 + math.exp(-log_odds)), 0.001), 0.99)

def vaep_proxy(event):
    ev_type = event["type"]
    success = event["success"]
    loc     = event.get("loc")
    end_loc = event.get("end_loc")

    if ev_type == "Shot":
        xg = xg_from_location(loc)
        return xg if success else -xg * 0.1

    if ev_type == "Pass":
        if not success:
            return -0.02
        xg_d = xg_from_location(end_loc or loc)
        xg_o = xg_from_location(loc)
        return max(xg_d - xg_o, 0.001)

    if ev_type == "Carry":
        xg_d = xg_from_location(end_loc or loc)
        xg_o = xg_from_location(loc)
        return max(xg_d - xg_o, 0.0)

    if ev_type == "Dribble":
        return 0.02 if success else -0.01

    return 0.0

# ---------------------------------------------------------------------------
# 3. Prior posicional para JDI (VAEP esperado por posicion)
# ---------------------------------------------------------------------------
POSITION_PRIOR = {
    "Goalkeeper": 0.01, "Center Back": 0.04, "Left Back": 0.06,
    "Right Back": 0.06, "Left Center Back": 0.04, "Right Center Back": 0.04,
    "Left Wing Back": 0.09, "Right Wing Back": 0.09,
    "Center Defensive Midfield": 0.08, "Left Defensive Midfield": 0.07,
    "Right Defensive Midfield": 0.07, "Center Midfield": 0.12,
    "Left Center Midfield": 0.11, "Right Center Midfield": 0.11,
    "Left Wing": 0.16, "Right Wing": 0.16,
    "Center Attacking Midfield": 0.18, "Left Attacking Midfield": 0.15,
    "Right Attacking Midfield": 0.15, "Center Forward": 0.22,
    "Left Wing Forward": 0.20, "Right Wing Forward": 0.20,
    "Striker": 0.22, "Second Striker": 0.18, "Substitute": 0.10,
}

def get_prior(pos_name):
    if pos_name in POSITION_PRIOR:
        return POSITION_PRIOR[pos_name]
    p = pos_name.lower()
    if "goalkeeper" in p: return 0.01
    if "back" in p:       return 0.05
    if "defensive" in p:  return 0.08
    if "mid" in p:        return 0.12
    if "wing" in p:       return 0.16
    if "forward" in p or "striker" in p: return 0.21
    return 0.10

# ---------------------------------------------------------------------------
# 4. Proceso principal
# ---------------------------------------------------------------------------
def process_chemistry():
    matches_file = "data/raw_matches.json"
    events_file  = "data/raw_events.json"

    if not os.path.exists(matches_file):
        print("raw_matches.json no encontrado. Ejecuta data_loader.py primero.")
        return

    with open(matches_file, "r", encoding="utf-8") as f:
        matches = json.load(f)

    has_events = os.path.exists(events_file)
    if has_events:
        with open(events_file, "r", encoding="utf-8") as f:
            all_events = json.load(f)
        print(f"Eventos cargados: {len(all_events)} partidos.")
    else:
        all_events = {}
        print("Sin raw_events.json — usando co-aparicion.")

    chemistry_data = defaultdict(lambda: defaultdict(lambda: {
        "co_games": 0, "joi_sum": 0.0, "jdi_sum": 0.0, "mins": 0.0
    }))
    player_info = {}
    MINS = 90.0

    for match in matches:
        match_id = str(match["fixture_id"])
        events   = all_events.get(match_id, [])

        team_players = {}
        player_pos   = {}

        for lineup in match.get("lineups", []):
            team_name = lineup["team"]["name"]
            team_players[team_name] = []
            for p in lineup.get("startXI", []):
                name = p["player"]["name"]
                pos  = p["player"]["pos"]
                team_players[team_name].append(name)
                player_pos[name] = pos
                player_info[name] = {"pos": pos, "team": team_name}

        # JOI: interacciones consecutivas entre jugadores del mismo equipo
        if events:
            player_to_team = {}
            for tn, pls in team_players.items():
                for nm in pls:
                    player_to_team[nm] = tn

            for idx in range(len(events) - 1):
                ev_a = events[idx]
                ev_b = events[idx + 1]
                pa, pb = ev_a["player"], ev_b["player"]
                if pa == pb:
                    continue
                ta = player_to_team.get(pa)
                tb = player_to_team.get(pb)
                if not ta or ta != tb:
                    continue
                joi_val = vaep_proxy(ev_a) + vaep_proxy(ev_b)
                pair = tuple(sorted([pa, pb]))
                chemistry_data[ta][pair]["joi_sum"] += joi_val

        # JDI: rendimiento de oponentes vs. prior posicional
        if events:
            opp_vaep_by_team = defaultdict(lambda: defaultdict(float))
            for ev in events:
                opp_vaep_by_team[ev["team"]][ev["player"]] += vaep_proxy(ev)

            for team_name, players in team_players.items():
                opp_teams = [t for t in team_players if t != team_name]
                for opp_team in opp_teams:
                    for opp in team_players[opp_team]:
                        opp_pos   = player_pos.get(opp, "Substitute")
                        g_opp     = get_grid_pos(opp_pos)
                        vaep_real = opp_vaep_by_team[opp_team].get(opp, 0.0)
                        prior     = get_prior(opp_pos)
                        delta     = prior - vaep_real   # positivo → oponente bajo expectativa
                        for pair in combinations(sorted(players), 2):
                            gp   = get_grid_pos(player_pos.get(pair[0], "Substitute"))
                            gq   = get_grid_pos(player_pos.get(pair[1], "Substitute"))
                            resp = responsibility(gp, gq, g_opp)
                            chemistry_data[team_name][pair]["jdi_sum"] += delta * resp * (MINS / 90.0)

        # Co-aparicion
        for team_name, players in team_players.items():
            for pair in combinations(sorted(players), 2):
                chemistry_data[team_name][pair]["co_games"] += 1
                chemistry_data[team_name][pair]["mins"]     += MINS

    # ── Construir CSV con normalización adaptativa ────────────────────────
    # En lugar de usar los máximos del paper (calibrados con ligas enteras),
    # normalizamos contra el percentil 95 del propio dataset del Mundial.
    rows_raw = []
    for team_name, pairs in chemistry_data.items():
        for (p1, p2), stats in pairs.items():
            mins  = max(stats["mins"], 1.0)
            joi90 = stats["joi_sum"] * (90.0 / mins) if has_events else 0.0
            jdi90 = stats["jdi_sum"] * (90.0 / mins) if has_events else 0.0
            rows_raw.append({
                "team": team_name, "player1": p1, "player2": p2,
                "p1_pos":   player_info.get(p1, {}).get("pos", "Substitute"),
                "p2_pos":   player_info.get(p2, {}).get("pos", "Substitute"),
                "co_games": stats["co_games"],
                "joi90":    joi90, "jdi90": jdi90,
            })

    if has_events and rows_raw:
        import statistics
        joi_vals = [r["joi90"] for r in rows_raw]
        jdi_vals = [r["jdi90"] for r in rows_raw]

        def percentile(data, p):
            s = sorted(data)
            idx = int(len(s) * p / 100)
            return s[min(idx, len(s)-1)]

        joi_p95 = percentile(joi_vals, 95) or 0.01
        jdi_p5  = percentile(jdi_vals, 5)
        jdi_p95 = percentile(jdi_vals, 95)
        jdi_range = (jdi_p95 - jdi_p5) or 0.01

        print(f"  JOI90 p95={joi_p95:.4f} | JDI90 p5={jdi_p5:.4f} p95={jdi_p95:.4f}")
    else:
        joi_p95 = 0.7; jdi_p5 = -1.0; jdi_range = 2.0

    rows = []
    for r in rows_raw:
        if has_events:
            offensive = round(min(max(r["joi90"] / joi_p95 * 10.0, 0.0), 10.0), 4)
            defensive = round(min(max((r["jdi90"] - jdi_p5) / jdi_range * 10.0, 0.0), 10.0), 4)
        else:
            co_norm   = r["co_games"] / 7.0
            offensive = round(co_norm * 10, 2)
            defensive = round(co_norm * 10, 2)

        rows.append({
            **r,
            "offensive": offensive,
            "defensive": defensive,
            "joi90":     round(r["joi90"], 6),
            "jdi90":     round(r["jdi90"], 6),
        })

    pd.DataFrame(rows).to_csv("data/processed_chemistry.csv", index=False)
    print(f"CSV generado: {len(rows)} pares | {'JOI90/JDI90' if has_events else 'co-aparicion'}.")

if __name__ == "__main__":
    process_chemistry()