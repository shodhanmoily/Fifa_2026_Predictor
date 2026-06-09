import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import (
    predict_match,
    predict_knockout,
    simulate_group,
    FLAG_CODES,
    CONFEDERATION,
    SQUAD_DATA,
)


def test_predict_match_returns_result():
    result = predict_match("Brazil", "Argentina", neutral=True)
    assert result is not None


def test_predict_match_probabilities_sum_to_100():
    result = predict_match("France", "Spain", neutral=True)
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert abs(total - 100.0) < 1.0


def test_predict_match_has_all_keys():
    result = predict_match("Germany", "England", neutral=True)

    required_keys = [
        "home_team",
        "away_team",
        "home_win",
        "draw",
        "away_win",
        "best_score",
        "home_exp",
        "away_exp",
        "home_code",
        "away_code",
    ]

    for key in required_keys:
        assert key in result


def test_predict_match_same_team_returns_none():
    result = predict_match("Brazil", "Brazil", neutral=True)
    assert result is None


def test_strong_team_beats_weak_team():
    result = predict_match("France", "Haiti", neutral=True)
    assert result["home_win"] > result["away_win"]


def test_predict_match_neutral_vs_home():
    neutral = predict_match("Mexico", "France", neutral=True)
    home = predict_match("Mexico", "France", neutral=False)

    assert home["home_win"] >= neutral["home_win"]


def test_knockout_no_draw():
    result = predict_knockout("Brazil", "Germany")
    assert result["home_win_ko"] + result["away_win_ko"] == 100.0


def test_knockout_has_winner():
    result = predict_knockout("Spain", "Portugal")
    assert result["winner"] in ["Spain", "Portugal"]


def test_knockout_winner_is_higher_probability():
    result = predict_knockout("France", "Haiti")
    assert result["winner"] == "France"


def test_knockout_has_penalty_flag():
    result = predict_knockout("Brazil", "Argentina")
    assert "penalty" in result
    assert isinstance(result["penalty"], bool)


def test_simulate_group_returns_4_teams():
    teams = ["France", "Morocco", "Haiti", "Scotland"]
    standings, matches = simulate_group(teams)
    assert len(standings) == 4


def test_simulate_group_returns_6_matches():
    teams = ["Brazil", "Morocco", "Haiti", "Scotland"]
    standings, matches = simulate_group(teams)
    assert len(matches) == 6


def test_simulate_group_standings_sorted_by_points():
    teams = ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"]
    standings, _ = simulate_group(teams)

    for i in range(len(standings) - 1):
        assert standings[i]["p"] >= standings[i + 1]["p"]


def test_all_48_teams_have_flag_codes():
    assert len(FLAG_CODES) == 48


def test_all_48_teams_have_confederation():
    assert len(CONFEDERATION) == 48


def test_all_48_teams_have_squad_data():
    assert len(SQUAD_DATA) == 48


def test_no_team_predicts_against_itself():
    teams = list(FLAG_CODES.keys())

    for team in teams[:10]:
        result = predict_match(team, team)
        assert result is None