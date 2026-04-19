from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

def get_db():
    path = "data/processed_chemistry.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def categorize_pos(pos_name):
    pos = str(pos_name).lower()
    if "goalkeeper" in pos: return "PO"
    if any(x in pos for x in ["back", "defender"]): return "DF"
    if "midfield" in pos: return "MC"
    if any(x in pos for x in ["forward", "wing", "striker", "center"]): return "DL"
    return "MC"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/teams")
def get_teams():
    df = get_db()
    if df.empty: return jsonify([])
    return jsonify(sorted(df["team"].unique().tolist()))

@app.route("/players_by_team", methods=["POST"])
def players_by_team():
    team = request.json.get("team")
    df = get_db()
    sub = df[df["team"] == team]

    p1 = sub[['player1', 'p1_pos']].rename(columns={'player1': 'name', 'p1_pos': 'pos'})
    p2 = sub[['player2', 'p2_pos']].rename(columns={'player2': 'name', 'p2_pos': 'pos'})
    all_p = pd.concat([p1, p2]).drop_duplicates('name')

    players = []
    for name in all_p['name'].unique():
        p_rows = all_p[all_p['name'] == name]
        real_pos = next((p for p in p_rows['pos'] if p != "Substitute"), "Substitute")
        players.append({
            "name": name,
            "pos": real_pos,
            "cat": categorize_pos(real_pos)
        })
    return jsonify(sorted(players, key=lambda x: x['name']))

@app.route("/team_chemistry", methods=["POST"])
def team_chemistry():
    data = request.json
    players, team = data.get("players", []), data.get("team", "")
    df = get_db()
    sub = df[df["team"] == team]
    edges = []

    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            row = sub[((sub.player1 == p1) & (sub.player2 == p2)) |
                      ((sub.player1 == p2) & (sub.player2 == p1))]
            if not row.empty:
                edges.append({
                    "p1": p1, "p2": p2,
                    "value": float(row.iloc[0]["offensive"]),
                    "defensive": float(row.iloc[0]["defensive"]) if "defensive" in row.columns else None
                })
    return jsonify(edges)

@app.route("/pair_chemistry", methods=["POST"])
def pair_chemistry():
    """Devuelve química ofensiva y defensiva entre dos jugadores de un equipo."""
    data = request.json
    team = data.get("team", "")
    p1_name = data.get("player1", "")
    p2_name = data.get("player2", "")

    df = get_db()
    sub = df[df["team"] == team]

    row = sub[
        ((sub.player1 == p1_name) & (sub.player2 == p2_name)) |
        ((sub.player1 == p2_name) & (sub.player2 == p1_name))
    ]

    if row.empty:
        return jsonify({"found": False})

    r = row.iloc[0]
    return jsonify({
        "found":     True,
        "player1":   p1_name,
        "player2":   p2_name,
        "co_games":  int(r["co_games"]),
        "offensive": float(r["offensive"]),
        "defensive": float(r["defensive"]) if "defensive" in r.index else None,
        "joi90":     float(r["joi90"])     if "joi90"     in r.index else None,
        "jdi90":     float(r["jdi90"])     if "jdi90"     in r.index else None,
        "p1_pos":    str(r["p1_pos"])      if "p1_pos"    in r.index else None,
        "p2_pos":    str(r["p2_pos"])      if "p2_pos"    in r.index else None,
    })

@app.route("/team_total", methods=["POST"])
def team_total():
    """Devuelve la química total del XI: suma de todos los edges ofensivos."""
    data = request.json
    players, team = data.get("players", []), data.get("team", "")
    df = get_db()
    sub = df[df["team"] == team]

    total_off = 0.0
    total_def = 0.0
    count = 0

    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            row = sub[
                ((sub.player1 == p1) & (sub.player2 == p2)) |
                ((sub.player1 == p2) & (sub.player2 == p1))
            ]
            if not row.empty:
                total_off += float(row.iloc[0]["offensive"])
                total_def += float(row.iloc[0]["defensive"]) if "defensive" in row.columns else 0.0
                count += 1

    avg_off = round(total_off / count, 2) if count else 0
    avg_def = round(total_def / count, 2) if count else 0

    return jsonify({
        "total_offensive": round(total_off, 2),
        "total_defensive": round(total_def, 2),
        "avg_offensive": avg_off,
        "avg_defensive": avg_def,
        "pairs": count
    })

@app.route("/best_xi", methods=["POST"])
def best_xi():
    data = request.json
    team, slots, mode = data.get("team"), data.get("slots"), data.get("mode", "max")
    df = get_db()
    sub = df[df["team"] == team]

    p1 = sub[['player1', 'p1_pos', 'offensive']].rename(columns={'player1': 'name', 'p1_pos': 'pos'})
    p2 = sub[['player2', 'p2_pos', 'offensive']].rename(columns={'player2': 'name', 'p2_pos': 'pos'})
    all_p = pd.concat([p1, p2])
    stats = all_p.groupby(['name', 'pos'])['offensive'].mean().reset_index()

    selection, used = [], set()
    for cat in slots:
        candidates = [r for _, r in stats.iterrows() if r['name'] not in used and categorize_pos(r['pos']) == cat]
        if not candidates:
            candidates = [r for _, r in stats.iterrows() if r['name'] not in used]

        if candidates:
            candidates.sort(key=lambda x: x['offensive'], reverse=(mode == "max"))
            selection.append(candidates[0]['name'])
            used.add(candidates[0]['name'])
        else:
            selection.append("")
    return jsonify(selection)

if __name__ == "__main__":
    app.run(debug=True)