from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from scipy.stats import poisson
import pickle

app = FastAPI(title="FIFA 2026 Predictor")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load Poisson model ─────────────────────────────────────
with open("model/poisson_model.pkl", "rb") as f:
    model_data = pickle.load(f)

team_stats     = model_data["team_stats"]
avg_goals      = model_data["avg_goals"]
avg_home_goals = model_data["avg_home_goals"]
avg_away_goals = model_data["avg_away_goals"]
home_advantage = model_data["home_advantage"]
fifa_rankings  = model_data["fifa_rankings"]
wc_groups      = model_data["wc_groups"]

# ── Flag codes (ISO) ───────────────────────────────────────
FLAG_CODES = {
    "France":"fr","Spain":"es","Argentina":"ar","England":"gb-eng",
    "Portugal":"pt","Brazil":"br","Netherlands":"nl","Belgium":"be",
    "Germany":"de","Morocco":"ma","Croatia":"hr","Colombia":"co",
    "Senegal":"sn","Mexico":"mx","United States":"us","Uruguay":"uy",
    "Japan":"jp","Switzerland":"ch","Iran":"ir","Turkey":"tr",
    "Ecuador":"ec","Austria":"at","South Korea":"kr","Australia":"au",
    "Algeria":"dz","Egypt":"eg","Canada":"ca","Norway":"no",
    "Panama":"pa","Ivory Coast":"ci","Sweden":"se","Scotland":"gb-sct",
    "Paraguay":"py","Czech Republic":"cz","Tunisia":"tn","DR Congo":"cd",
    "Uzbekistan":"uz","Qatar":"qa","Iraq":"iq","Saudi Arabia":"sa",
    "South Africa":"za","Jordan":"jo","Bosnia and Herzegovina":"ba",
    "Cape Verde":"cv","Ghana":"gh","Curaçao":"cw","Haiti":"ht",
    "New Zealand":"nz",
}

# ── Confederations ─────────────────────────────────────────
CONFEDERATION = {
    "France":"UEFA","Spain":"UEFA","England":"UEFA","Portugal":"UEFA",
    "Germany":"UEFA","Netherlands":"UEFA","Belgium":"UEFA","Croatia":"UEFA",
    "Switzerland":"UEFA","Austria":"UEFA","Norway":"UEFA","Scotland":"UEFA",
    "Sweden":"UEFA","Turkey":"UEFA","Bosnia and Herzegovina":"UEFA",
    "Czech Republic":"UEFA",
    "Argentina":"CONMEBOL","Brazil":"CONMEBOL","Colombia":"CONMEBOL",
    "Uruguay":"CONMEBOL","Ecuador":"CONMEBOL","Paraguay":"CONMEBOL",
    "United States":"CONCACAF","Mexico":"CONCACAF","Canada":"CONCACAF",
    "Panama":"CONCACAF","Haiti":"CONCACAF","Curaçao":"CONCACAF",
    "Morocco":"CAF","Senegal":"CAF","Ivory Coast":"CAF","Algeria":"CAF",
    "Egypt":"CAF","Tunisia":"CAF","Ghana":"CAF","South Africa":"CAF",
    "DR Congo":"CAF","Cape Verde":"CAF",
    "Japan":"AFC","South Korea":"AFC","Iran":"AFC","Saudi Arabia":"AFC",
    "Australia":"AFC","Uzbekistan":"AFC","Jordan":"AFC","Iraq":"AFC",
    "Qatar":"AFC","New Zealand":"OFC",
}

# ── Confederation strength multiplier ─────────────────────
CONF_MULTIPLIER = {
    "UEFA":1.15,"CONMEBOL":1.12,
    "CAF":0.95,"AFC":0.93,
    "CONCACAF":0.97,"OFC":0.88,
}

# ── Enhanced squad scores from player data ─────────────────
SQUAD_DATA = {
    "France":      {"squad_score":1.0843,"attack_boost":1.30,"defence_boost":1.20},
    "Spain":       {"squad_score":1.0730,"attack_boost":1.30,"defence_boost":1.20},
    "Germany":     {"squad_score":1.0682,"attack_boost":1.30,"defence_boost":1.20},
    "United States":{"squad_score":1.0639,"attack_boost":1.30,"defence_boost":1.20},
    "England":     {"squad_score":1.0620,"attack_boost":1.30,"defence_boost":1.20},
    "Portugal":    {"squad_score":1.0595,"attack_boost":1.30,"defence_boost":1.20},
    "Argentina":   {"squad_score":1.0545,"attack_boost":1.30,"defence_boost":1.20},
    "Belgium":     {"squad_score":1.0501,"attack_boost":1.30,"defence_boost":1.20},
    "Algeria":     {"squad_score":1.0498,"attack_boost":1.30,"defence_boost":1.20},
    "Switzerland": {"squad_score":1.0460,"attack_boost":1.30,"defence_boost":1.20},
    "Senegal":     {"squad_score":1.0452,"attack_boost":1.30,"defence_boost":1.20},
    "Sweden":      {"squad_score":1.0432,"attack_boost":1.30,"defence_boost":1.20},
    "Australia":   {"squad_score":1.0429,"attack_boost":1.30,"defence_boost":1.20},
    "Netherlands": {"squad_score":1.0390,"attack_boost":1.30,"defence_boost":1.20},
    "Japan":       {"squad_score":1.0385,"attack_boost":1.30,"defence_boost":1.20},
    "Morocco":     {"squad_score":1.0350,"attack_boost":1.25,"defence_boost":1.15},
    "Croatia":     {"squad_score":1.0320,"attack_boost":1.25,"defence_boost":1.15},
    "Colombia":    {"squad_score":1.0300,"attack_boost":1.25,"defence_boost":1.15},
    "Norway":      {"squad_score":1.0280,"attack_boost":1.25,"defence_boost":1.15},
    "Uruguay":     {"squad_score":1.0250,"attack_boost":1.20,"defence_boost":1.10},
    "Austria":     {"squad_score":1.0220,"attack_boost":1.20,"defence_boost":1.10},
    "South Korea": {"squad_score":1.0180,"attack_boost":1.20,"defence_boost":1.10},
    "Mexico":      {"squad_score":1.0150,"attack_boost":1.15,"defence_boost":1.05},
    "Turkey":      {"squad_score":1.0100,"attack_boost":1.15,"defence_boost":1.05},
    "Iran":        {"squad_score":1.0050,"attack_boost":1.10,"defence_boost":1.05},
    "Tunisia":     {"squad_score":1.0020,"attack_boost":1.10,"defence_boost":1.00},
    "Ivory Coast": {"squad_score":1.0010,"attack_boost":1.10,"defence_boost":1.00},
    "Ghana":       {"squad_score":0.9983,"attack_boost":1.05,"defence_boost":0.98},
    "Brazil":      {"squad_score":0.9927,"attack_boost":1.20,"defence_boost":1.10},
    "Scotland":    {"squad_score":0.9868,"attack_boost":1.05,"defence_boost":0.98},
    "New Zealand": {"squad_score":0.9838,"attack_boost":0.95,"defence_boost":0.90},
    "Ecuador":     {"squad_score":0.9710,"attack_boost":1.00,"defence_boost":0.95},
    "Paraguay":    {"squad_score":0.9603,"attack_boost":1.00,"defence_boost":0.95},
    "Canada":      {"squad_score":0.9542,"attack_boost":1.00,"defence_boost":0.95},
    "Egypt":       {"squad_score":0.8681,"attack_boost":0.95,"defence_boost":0.90},
    "Qatar":       {"squad_score":0.6900,"attack_boost":0.80,"defence_boost":0.80},
    "Iraq":        {"squad_score":0.6012,"attack_boost":0.75,"defence_boost":0.75},
    "Saudi Arabia":{"squad_score":1.0220,"attack_boost":1.00,"defence_boost":0.95},
    "Jordan":      {"squad_score":1.0100,"attack_boost":0.90,"defence_boost":0.90},
    "Bosnia and Herzegovina":{"squad_score":1.0000,"attack_boost":1.05,"defence_boost":1.00},
    "Cape Verde":  {"squad_score":1.0000,"attack_boost":1.00,"defence_boost":0.95},
    "Curaçao":     {"squad_score":1.0000,"attack_boost":0.95,"defence_boost":0.90},
    "Haiti":       {"squad_score":0.8500,"attack_boost":0.85,"defence_boost":0.85},
    "Czech Republic":{"squad_score":1.0000,"attack_boost":1.05,"defence_boost":1.00},
    "DR Congo":    {"squad_score":0.9180,"attack_boost":1.00,"defence_boost":0.95},
    "Uzbekistan":  {"squad_score":0.9150,"attack_boost":0.95,"defence_boost":0.90},
    "Panama":      {"squad_score":0.9300,"attack_boost":0.90,"defence_boost":0.90},
    "South Africa":{"squad_score":0.9120,"attack_boost":0.90,"defence_boost":0.90},
}

# ── Core prediction function ───────────────────────────────
def predict_match(home_team, away_team, neutral=False):
    if home_team not in team_stats or away_team not in team_stats:
        return None

    home = team_stats[home_team]
    away = team_stats[away_team]

    home_sd = SQUAD_DATA.get(home_team, {"squad_score":1.0,"attack_boost":1.0,"defence_boost":1.0})
    away_sd = SQUAD_DATA.get(away_team, {"squad_score":1.0,"attack_boost":1.0,"defence_boost":1.0})

    home_conf = CONF_MULTIPLIER.get(CONFEDERATION.get(home_team, "UEFA"), 1.0)
    away_conf = CONF_MULTIPLIER.get(CONFEDERATION.get(away_team, "UEFA"), 1.0)

    host_bonus = 1.0
    if not neutral and home_team in ["United States","Mexico","Canada"]:
        host_bonus = home_advantage

    if neutral:
        home_exp = (avg_goals
                    * home["attack_strength"] * home_sd["attack_boost"]
                    * away["defence_weakness"] / away_sd["defence_boost"]
                    * home_sd["squad_score"] * home_conf)
        away_exp = (avg_goals
                    * away["attack_strength"] * away_sd["attack_boost"]
                    * home["defence_weakness"] / home_sd["defence_boost"]
                    * away_sd["squad_score"] * away_conf)
    else:
        home_exp = (avg_goals
                    * home["attack_strength"] * home_sd["attack_boost"]
                    * away["defence_weakness"] / away_sd["defence_boost"]
                    * home_sd["squad_score"] * home_conf * host_bonus)
        away_exp = (avg_goals
                    * away["attack_strength"] * away_sd["attack_boost"]
                    * home["defence_weakness"] / home_sd["defence_boost"]
                    * away_sd["squad_score"] * away_conf)

    home_exp = min(max(home_exp, 0.3), 5.0)
    away_exp = min(max(away_exp, 0.3), 5.0)

    max_goals = 8
    home_win = draw = away_win = 0
    score_matrix = {}

    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = poisson.pmf(hg, home_exp) * poisson.pmf(ag, away_exp)
            score_matrix[(hg, ag)] = p
            if hg > ag:    home_win += p
            elif hg == ag: draw     += p
            else:          away_win += p

    best = max(score_matrix, key=score_matrix.get)
    hw = round(home_win * 100, 1)
    dr = round(draw * 100, 1)
    aw = round(away_win * 100, 1)

    if hw > aw:   winner = home_team
    elif aw > hw: winner = away_team
    else:         winner = "Draw"

    return {
        "home_team":       home_team,
        "away_team":       away_team,
        "home_code":       FLAG_CODES.get(home_team, "un"),
        "away_code":       FLAG_CODES.get(away_team, "un"),
        "home_ranking":    fifa_rankings.get(home_team, "N/A"),
        "away_ranking":    fifa_rankings.get(away_team, "N/A"),
        "home_conf":       CONFEDERATION.get(home_team, ""),
        "away_conf":       CONFEDERATION.get(away_team, ""),
        "home_exp":        round(home_exp, 1),
        "away_exp":        round(away_exp, 1),
        "home_win":        hw,
        "draw":            dr,
        "away_win":        aw,
        "best_score":      f"{best[0]}-{best[1]}",
        "best_score_prob": round(score_matrix[best] * 100, 1),
        "home_attack":     round(home["attack_strength"] * home_sd["attack_boost"], 3),
        "away_attack":     round(away["attack_strength"] * away_sd["attack_boost"], 3),
        "home_defence":    round(home["defence_weakness"] / home_sd["defence_boost"], 3),
        "away_defence":    round(away["defence_weakness"] / away_sd["defence_boost"], 3),
        "winner":          winner,
    }

# ── Knockout — no draws ────────────────────────────────────
def predict_knockout(home_team, away_team):
    result = predict_match(home_team, away_team, neutral=True)
    if not result:
        return None

    hw    = result["home_win"]
    aw    = result["away_win"]
    total = hw + aw

    if total > 0:
        result["home_win_ko"] = round(hw / total * 100, 1)
        result["away_win_ko"] = round(aw / total * 100, 1)
    else:
        result["home_win_ko"] = 50.0
        result["away_win_ko"] = 50.0

    if result["home_win_ko"] > result["away_win_ko"]:
        result["winner"]  = home_team
        result["penalty"] = False
    elif result["away_win_ko"] > result["home_win_ko"]:
        result["winner"]  = away_team
        result["penalty"] = False
    else:
        home_rank = fifa_rankings.get(home_team, 50)
        away_rank = fifa_rankings.get(away_team, 50)
        result["winner"]  = home_team if home_rank < away_rank else away_team
        result["penalty"] = True

    score_parts = result["best_score"].split("-")
    if score_parts[0] == score_parts[1]:
        result["penalty"] = True

    return result

# ── Group simulation ───────────────────────────────────────
def simulate_group(group_teams):
    standings = {
        t: {"team":t,"p":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"gd":0}
        for t in group_teams
    }
    matches = []

    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            home, away = group_teams[i], group_teams[j]
            pred = predict_match(home, away, neutral=True)
            if not pred:
                continue
            hg = round(pred["home_exp"])
            ag = round(pred["away_exp"])
            matches.append({
                "home":          home,
                "away":          away,
                "home_team":     home,
                "away_team":     away,
                "home_code":     FLAG_CODES.get(home, "un"),
                "away_code":     FLAG_CODES.get(away, "un"),
                "score":         f"{hg}-{ag}",
                "home_win_prob": pred["home_win"],
                "draw_prob":     pred["draw"],
                "away_win_prob": pred["away_win"],
            })
            standings[home]["gf"] += hg
            standings[home]["ga"] += ag
            standings[away]["gf"] += ag
            standings[away]["ga"] += hg
            standings[home]["gd"] = standings[home]["gf"] - standings[home]["ga"]
            standings[away]["gd"] = standings[away]["gf"] - standings[away]["ga"]
            if hg > ag:
                standings[home]["w"] += 1
                standings[home]["p"] += 3
                standings[away]["l"] += 1
            elif hg < ag:
                standings[away]["w"] += 1
                standings[away]["p"] += 3
                standings[home]["l"] += 1
            else:
                standings[home]["d"] += 1
                standings[home]["p"] += 1
                standings[away]["d"] += 1
                standings[away]["p"] += 1

    sorted_teams = sorted(
        standings.values(),
        key=lambda x: (-x["p"], -x["gd"], -x["gf"])
    )
    return sorted_teams, matches

# ── Shared simulation helper ───────────────────────────────
def run_simulation():
    all_groups   = {}
    third_placed = []

    for group_name, group_teams in wc_groups.items():
        standings, matches = simulate_group(group_teams)
        all_groups[group_name] = {
            "standings": standings,
            "matches":   matches,
            "qualified": [standings[0]["team"], standings[1]["team"]],
        }
        third_placed.append({
            "team": standings[2]["team"],
            "pts":  standings[2]["p"],
            "gd":   standings[2]["gd"],
            "gf":   standings[2]["gf"],
            "rank": fifa_rankings.get(standings[2]["team"], 999)
        })

    best_third = [
        t["team"] for t in
        sorted(third_placed, key=lambda x: (-x["pts"], -x["gd"], -x["gf"], x["rank"]))[:8]
    ]

    pairs = [
        ("Group A","Group B"),("Group C","Group D"),
        ("Group E","Group F"),("Group G","Group H"),
        ("Group I","Group J"),("Group K","Group L"),
    ]
    r32_matches = []
    r32_winners = []

    for g1, g2 in pairs:
        t1w = all_groups[g1]["standings"][0]["team"]
        t1r = all_groups[g1]["standings"][1]["team"]
        t2w = all_groups[g2]["standings"][0]["team"]
        t2r = all_groups[g2]["standings"][1]["team"]
        for h, a in [(t1w, t2r), (t2w, t1r)]:
            m = predict_knockout(h, a)
            if m:
                r32_matches.append(m)
                r32_winners.append(m["winner"])

    for i in range(0, len(best_third), 2):
        if i+1 < len(best_third):
            m = predict_knockout(best_third[i], best_third[i+1])
            if m:
                r32_matches.append(m)
                r32_winners.append(m["winner"])

    r16_matches = []
    r16_winners = []
    for i in range(0, len(r32_winners), 2):
        if i+1 < len(r32_winners):
            m = predict_knockout(r32_winners[i], r32_winners[i+1])
            if m:
                r16_matches.append(m)
                r16_winners.append(m["winner"])

    qf_matches = []
    qf_winners = []
    for i in range(0, len(r16_winners), 2):
        if i+1 < len(r16_winners):
            m = predict_knockout(r16_winners[i], r16_winners[i+1])
            if m:
                qf_matches.append(m)
                qf_winners.append(m["winner"])

    sf_matches = []
    sf_winners = []
    sf_losers  = []
    for i in range(0, len(qf_winners), 2):
        if i+1 < len(qf_winners):
            m = predict_knockout(qf_winners[i], qf_winners[i+1])
            if m:
                sf_matches.append(m)
                sf_winners.append(m["winner"])
                loser = m["away_team"] if m["winner"] == m["home_team"] else m["home_team"]
                sf_losers.append(loser)

    third_match = None
    third_place = None
    if len(sf_losers) >= 2:
        third_match = predict_knockout(sf_losers[0], sf_losers[1])
        if third_match:
            third_place = third_match["winner"]

    final_match = None
    champion    = None
    if len(sf_winners) >= 2:
        final_match = predict_knockout(sf_winners[0], sf_winners[1])
        if final_match:
            champion = final_match["winner"]

    return {
        "all_groups":  all_groups,
        "r32_matches": r32_matches,
        "r16_matches": r16_matches,
        "qf_matches":  qf_matches,
        "sf_matches":  sf_matches,
        "final_match": final_match,
        "third_match": third_match,
        "champion":    champion,
        "third_place": third_place,
        "flag_codes":  FLAG_CODES,
        "best_third":  best_third,
    }

# ── Routes ─────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    teams = sorted(FLAG_CODES.keys())
    return templates.TemplateResponse("index.html", {
        "request":    request,
        "teams":      teams,
        "flag_codes": FLAG_CODES,
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request:   Request,
    home_team: str = Form(...),
    away_team: str = Form(...),
    neutral:   str = Form("off")
):
    is_neutral = neutral == "on"
    result = predict_match(home_team, away_team, is_neutral)
    if not result:
        return templates.TemplateResponse("index.html", {
            "request":    request,
            "error":      "Teams not found in model.",
            "teams":      sorted(FLAG_CODES.keys()),
            "flag_codes": FLAG_CODES,
        })
    return templates.TemplateResponse("predictor.html", {
        "request":    request,
        "r":          result,
        "is_neutral": is_neutral,
        "flag_codes": FLAG_CODES,
    })

@app.get("/groups", response_class=HTMLResponse)
async def groups_page(request: Request):
    return templates.TemplateResponse("groups.html", {
        "request":       request,
        "groups":        wc_groups,
        "rankings":      fifa_rankings,
        "confederation": CONFEDERATION,
        "flag_codes":    FLAG_CODES,
    })

@app.get("/simulator", response_class=HTMLResponse)
async def simulator(request: Request):
    ctx = run_simulation()
    ctx["request"] = request
    return templates.TemplateResponse("simulator.html", ctx)

@app.get("/matches", response_class=HTMLResponse)
async def matches_page(request: Request):
    ctx = run_simulation()
    ctx["request"] = request
    return templates.TemplateResponse("matches.html", ctx)

@app.get("/knockout", response_class=HTMLResponse)
async def knockout_page(request: Request):
    ctx = run_simulation()
    ctx["request"] = request
    return templates.TemplateResponse("knockout.html", ctx)

@app.get("/health")
async def health():
    return {"status":"ok","teams":len(team_stats),"model":"Poisson + Squad Data"}
