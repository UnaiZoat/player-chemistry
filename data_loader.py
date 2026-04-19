import os
import json
from statsbombpy import sb

# Configuración Mundial 2022 según statsbombpy
COMP_ID = 43
SEASON_ID = 106

def download_world_cup_data():
    if not os.path.exists("data"):
        os.makedirs("data")

    print("--- Descargando datos del Mundial 2022 ---")
    matches = sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)

    raw_data = []
    raw_events = {}   # {match_id: [eventos filtrados]}
    total_matches = len(matches)

    for i, (_, match) in enumerate(matches.iterrows()):
        match_id = match['match_id']
        print(f"[{i+1}/{total_matches}] {match['home_team']} vs {match['away_team']}")

        try:
            # ── Alineaciones ──────────────────────────────────────────────
            lineups = sb.lineups(match_id=match_id)
            formatted_lineups = []
            player_team_map = {}   # player_name → team_name (para este partido)

            for team_name, lineup_df in lineups.items():
                players = []
                for _, row in lineup_df.iterrows():
                    pos_list = row['positions']
                    pos_name = pos_list[0]['position'] if pos_list else "Substitute"
                    # Minutos jugados: usar from/to del primer periodo si están disponibles
                    minutes = 90  
                    players.append({
                        "player": {
                            "name": row['player_name'],
                            "pos": pos_name
                        }
                    })
                    player_team_map[row['player_name']] = team_name

                formatted_lineups.append({
                    "team": {"name": team_name},
                    "startXI": players
                })

            raw_data.append({
                "fixture_id": match_id,
                "teams": {
                    "home": {"name": match['home_team']},
                    "away": {"name": match['away_team']}
                },
                "lineups": formatted_lineups,
                "player_team_map": player_team_map
            })

            # ── Eventos ───────────────────────────────────────────────────
            # Solo guardamos los campos necesarios para JOI y JDI:
            # tipo, jugador, equipo, location, resultado, minuto, índice
            events_df = sb.events(match_id=match_id)

            # Tipos ofensivos según el paper: pass, cross, dribble, carry, shot
            # StatsBomb usa: Pass, Carry, Dribble, Shot
            OFFENSIVE_TYPES = {"Pass", "Carry", "Dribble", "Shot"}

            match_events = []
            for _, ev in events_df.iterrows():
                ev_type = str(ev.get('type', ''))
                if ev_type not in OFFENSIVE_TYPES:
                    continue

                player = ev.get('player', None)
                team   = ev.get('team', None)
                if not player or not team:
                    continue

                loc = ev.get('location', None)
                end_loc = ev.get('pass_end_location',
                          ev.get('carry_end_location',
                          ev.get('shot_end_location', None)))

                outcome_raw = ev.get('pass_outcome', ev.get('dribble_outcome',
                              ev.get('shot_outcome', None)))
                outcome = str(outcome_raw) if outcome_raw else "Complete"
                success = outcome in {"Complete", "Goal", "Won", "Success",
                                      "Success In Play", "Success Out", "nan", "None"}

                match_events.append({
                    "index":   int(ev.get('index', 0)),
                    "minute":  int(ev.get('minute', 0)),
                    "second":  int(ev.get('second', 0)),
                    "type":    ev_type,
                    "player":  str(player),
                    "team":    str(team),
                    "loc":     loc,
                    "end_loc": end_loc,
                    "success": success
                })

            raw_events[str(match_id)] = match_events

        except Exception as e:
            print(f"  ⚠ Error en match {match_id}: {e}")

    with open("data/raw_matches.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)

    with open("data/raw_events.json", "w", encoding="utf-8") as f:
        json.dump(raw_events, f, ensure_ascii=False)

    print(f"✅ {len(raw_data)} partidos y eventos guardados.")

if __name__ == "__main__":
    download_world_cup_data()