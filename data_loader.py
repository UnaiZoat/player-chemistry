import os
import json
from statsbombpy import sb

# ---------------------------------------------------------------------------
# Tournaments to download.
# Both are FIFA World Cups available in StatsBomb Open Data.
# Adding more tournaments here is the only change needed to expand the dataset.
# competition_id=43 is the FIFA World Cup across all StatsBomb seasons.
# ---------------------------------------------------------------------------
TOURNAMENTS = [
    {"comp_id": 43, "season_id": 3,   "label": "WC 2018"},
    {"comp_id": 43, "season_id": 106, "label": "WC 2022"},
]

# Action types used for JOI and JDI (Section 3 of the paper).
# StatsBomb event type names: Pass, Carry, Dribble, Shot.
OFFENSIVE_TYPES = {"Pass", "Carry", "Dribble", "Shot"}


def _process_match(match_id, home_team, away_team, tournament_label):
    """
    Download lineups and events for a single match.
    Returns (match_record, match_events) or raises on failure.
    """
    # Lineups
    lineups = sb.lineups(match_id=match_id)
    formatted_lineups = []
    player_team_map = {}

    for team_name, lineup_df in lineups.items():
        players = []
        for _, row in lineup_df.iterrows():
            pos_list = row["positions"]
            pos_name = pos_list[0]["position"] if pos_list else "Substitute"
            players.append({"player": {"name": row["player_name"], "pos": pos_name}})
            player_team_map[row["player_name"]] = team_name

        formatted_lineups.append({
            "team": {"name": team_name},
            "startXI": players,
        })

    match_record = {
        "fixture_id":      match_id,
        "tournament":      tournament_label,
        "teams": {
            "home": {"name": home_team},
            "away": {"name": away_team},
        },
        "lineups":         formatted_lineups,
        "player_team_map": player_team_map,
    }

    # Events
    events_df = sb.events(match_id=match_id)
    match_events = []

    for _, ev in events_df.iterrows():
        ev_type = str(ev.get("type", ""))
        if ev_type not in OFFENSIVE_TYPES:
            continue

        player = ev.get("player", None)
        team   = ev.get("team",   None)
        if not player or not team:
            continue

        loc     = ev.get("location", None)
        end_loc = ev.get("pass_end_location",
                  ev.get("carry_end_location",
                  ev.get("shot_end_location", None)))

        outcome_raw = ev.get("pass_outcome",
                      ev.get("dribble_outcome",
                      ev.get("shot_outcome", None)))
        outcome = str(outcome_raw) if outcome_raw else "Complete"
        success = outcome in {
            "Complete", "Goal", "Won", "Success",
            "Success In Play", "Success Out", "nan", "None",
        }

        match_events.append({
            "index":   int(ev.get("index",  0)),
            "minute":  int(ev.get("minute", 0)),
            "second":  int(ev.get("second", 0)),
            "type":    ev_type,
            "player":  str(player),
            "team":    str(team),
            "loc":     loc,
            "end_loc": end_loc,
            "success": success,
        })

    return match_record, match_events


def download_data():
    if not os.path.exists("data"):
        os.makedirs("data")

    all_matches = []
    all_events  = {}

    for tournament in TOURNAMENTS:
        comp_id   = tournament["comp_id"]
        season_id = tournament["season_id"]
        label     = tournament["label"]

        print(f"\n--- Downloading {label} ---")
        matches_df = sb.matches(competition_id=comp_id, season_id=season_id)
        total = len(matches_df)

        for i, (_, match) in enumerate(matches_df.iterrows()):
            match_id  = match["match_id"]
            home_team = match["home_team"]
            away_team = match["away_team"]
            print(f"  [{i+1}/{total}] {home_team} vs {away_team}")

            try:
                record, events = _process_match(
                    match_id, home_team, away_team, label
                )
                all_matches.append(record)
                all_events[str(match_id)] = events
            except Exception as e:
                print(f"    Warning: match {match_id} failed: {e}")

    with open("data/raw_matches.json", "w", encoding="utf-8") as f:
        json.dump(all_matches, f, indent=2, ensure_ascii=False)

    with open("data/raw_events.json", "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False)

    total_matches = len(all_matches)
    total_events  = sum(len(v) for v in all_events.values())
    print(f"\nDone. {total_matches} matches saved "
          f"({total_events:,} offensive actions across all tournaments).")


if __name__ == "__main__":
    download_data()