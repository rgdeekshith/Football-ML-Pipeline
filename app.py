# app.py

# -----------------------------
# Imports
# -----------------------------
import os
import sqlite3

import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from build import build_database  # build_database() creates football.db


# -----------------------------
# Basic page config
# -----------------------------
st.set_page_config(
    page_title="Football Match Insights",
    page_icon="⚽",
    layout="wide",
)


# -----------------------------
# Database helpers
# -----------------------------
DB_PATH = "football.db"


def ensure_db():
    """Create football.db once if it does not exist."""
    if not os.path.exists(DB_PATH):
        build_database()


@st.cache_resource(show_spinner=False)
def get_connection(db_path=DB_PATH):
    """
    Open a connection to the SQLite database.
    Cached as a resource so we keep a single connection per session.
    """
    ensure_db()
    conn = sqlite3.connect(db_path)
    return conn


# -----------------------------
# Load filtered matches + win rates
# -----------------------------
@st.cache_data(show_spinner=False)
def load_matches_df():
    """
    Load a filtered subset of matches (only selected clubs/national teams)
    with team and league info, plus simple win rates.
    """
    conn = get_connection()

    query = """
    SELECT
        m.match_id,
        th.team_name AS home_team,
        ta.team_name AS away_team,
        l.league_name,
        l.level,
        m.full_time,
        m.home_score,
        m.away_score,
        m.result_class
    FROM matches m
    JOIN teams th ON m.home_team_id = th.team_id
    JOIN teams ta ON m.away_team_id = ta.team_id
    JOIN leagues l ON m.league_id = l.league_id
    WHERE th.team_name IN ("Arsenal", "Aston Villa", "Bournemouth", "Brentford",
                           "Brighton & Hove Albion", "Burnley", "Chelsea",
                           "Crystal Palace", "Everton", "Fulham", "Leeds United",
                           "Liverpool", "Manchester City", "Manchester United",
                           "Newcastle United", "Nottingham Forest", "Sunderland",
                           "Tottenham Hotspur", "West Ham United",
                           "Wolverhampton Wanderers", "Alavés", "Athletic Bilbao",
                           "Atlético Madrid", "Barcelona", "Celta Vigo", "Elche",
                           "Espanyol", "Getafe", "Girona", "Levante", "Mallorca",
                           "Osasuna", "Oviedo", "Rayo Vallecano", "Real Betis",
                           "Real Madrid", "Real Sociedad", "Sevilla", "Valencia",
                           "Villarreal", "FC Augsburg", "Union Berlin",
                           "Werder Bremen", "Borussia Dortmund",
                           "Eintracht Frankfurt", "SC Freiburg", "Hamburger SV",
                           "1. FC Heidenheim", "TSG Hoffenheim", "1. FC Köln",
                           "RB Leipzig", "Bayer Leverkusen", "Mainz 05",
                           "Borussia Mönchengladbach", "Bayern Munich",
                           "FC St. Pauli", "VfB Stuttgart", "VfL Wolfsburg",
                           "Angers", "Auxerre", "Brest", "Le Havre", "Lens",
                           "Lille", "Lorient", "Lyon", "Marseille", "Metz",
                           "Monaco", "Nantes", "Nice", "Paris FC",
                           "Paris Saint-Germain", "Rennes", "Strasbourg",
                           "Toulouse", "Ajax", "AZ", "Excelsior", "Feyenoord",
                           "Go Ahead Eagles", "Groningen", "Heerenveen",
                           "Heracles Almelo", "NAC Breda", "NEC", "PEC Zwolle",
                           "PSV Eindhoven", "Sparta Rotterdam", "Telstar",
                           "Twente", "Utrecht", "Volendam", "Willem II",
                           "France", "Spain", "Argentina", "England", "Portugal",
                           "Brazil", "Netherlands", "Morocco", "Belgium",
                           "Germany", "Croatia", "Italy", "Colombia", "Senegal",
                           "Mexico", "USA", "Uruguay", "Japan", "Switzerland",
                           "Denmark", "Iran", "Turkiye", "Ecuador", "Austria",
                           "South Korea")
       OR ta.team_name IN ("Arsenal", "Aston Villa", "Bournemouth", "Brentford",
                           "Brighton & Hove Albion", "Burnley", "Chelsea",
                           "Crystal Palace", "Everton", "Fulham", "Leeds United",
                           "Liverpool", "Manchester City", "Manchester United",
                           "Newcastle United", "Nottingham Forest", "Sunderland",
                           "Tottenham Hotspur", "West Ham United",
                           "Wolverhampton Wanderers", "Alavés", "Athletic Bilbao",
                           "Atlético Madrid", "Barcelona", "Celta Vigo", "Elche",
                           "Espanyol", "Getafe", "Girona", "Levante", "Mallorca",
                           "Osasuna", "Oviedo", "Rayo Vallecano", "Real Betis",
                           "Real Madrid", "Real Sociedad", "Sevilla", "Valencia",
                           "Villarreal", "FC Augsburg", "Union Berlin",
                           "Werder Bremen", "Borussia Dortmund",
                           "Eintracht Frankfurt", "SC Freiburg", "Hamburger SV",
                           "1. FC Heidenheim", "TSG Hoffenheim", "1. FC Köln",
                           "RB Leipzig", "Bayer Leverkusen", "Mainz 05",
                           "Borussia Mönchengladbach", "Bayern Munich",
                           "FC St. Pauli", "VfB Stuttgart", "VfL Wolfsburg",
                           "Angers", "Auxerre", "Brest", "Le Havre", "Lens",
                           "Lille", "Lorient", "Lyon", "Marseille", "Metz",
                           "Monaco", "Nantes", "Nice", "Paris FC",
                           "Paris Saint-Germain", "Rennes", "Strasbourg",
                           "Toulouse", "Ajax", "AZ", "Excelsior", "Feyenoord",
                           "Go Ahead Eagles", "Groningen", "Heerenveen",
                           "Heracles Almelo", "NAC Breda", "NEC", "PEC Zwolle",
                           "PSV Eindhoven", "Sparta Rotterdam", "Telstar",
                           "Twente", "Utrecht", "Volendam", "Willem II",
                           "France", "Spain", "Argentina", "England", "Portugal",
                           "Brazil", "Netherlands", "Morocco", "Belgium",
                           "Germany", "Croatia", "Italy", "Colombia", "Senegal",
                           "Mexico", "USA", "Uruguay", "Japan", "Switzerland",
                           "Denmark", "Iran", "Turkiye", "Ecuador", "Austria",
                           "South Korea");
    """

    matches_df = pd.read_sql(query, conn)

    # Compute simple win stats per team
    team_stats = {}
    for row in matches_df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        result = row.result_class

        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {"games": 0, "wins": 0}

        team_stats[home]["games"] += 1
        team_stats[away]["games"] += 1

        if result == "home":
            team_stats[home]["wins"] += 1
        elif result == "away":
            team_stats[away]["wins"] += 1

    team_stats_df = (
        pd.DataFrame.from_dict(team_stats, orient="index")
        .reset_index()
        .rename(columns={"index": "team_name"})
    )
    team_stats_df["win_rate"] = team_stats_df["wins"] / team_stats_df["games"]

    # Merge home win_rate
    matches_df = (
        matches_df.merge(
            team_stats_df[["team_name", "win_rate"]],
            left_on="home_team",
            right_on="team_name",
            how="left",
        )
        .rename(columns={"win_rate": "home_win_rate"})
        .drop(columns=["team_name"])
    )

    # Merge away win_rate
    matches_df = (
        matches_df.merge(
            team_stats_df[["team_name", "win_rate"]],
            left_on="away_team",
            right_on="team_name",
            how="left",
        )
        .rename(columns={"win_rate": "away_win_rate"})
        .drop(columns=["team_name"])
    )

    return matches_df


# -----------------------------
# Train Gradient Boosting pipeline
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_gb_pipeline():
    matches_df = load_matches_df()

    target_col = "result_class"
    feature_cols = [
        "home_team",
        "away_team",
        "league_name",
        "level",
        "home_win_rate",
        "away_win_rate",
    ]

    X = matches_df[feature_cols]
    y = matches_df[target_col]

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = ["home_win_rate", "away_win_rate"]
    categorical_features = ["home_team", "away_team", "league_name", "level"]

    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    gb_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("gb", gb_clf),
        ]
    )

    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    return model, val_acc, matches_df


# -----------------------------
# Simple xG heuristic
# -----------------------------
def estimate_expected_goals(row_dict):
    """
    row_dict should have 'home_win_rate' and 'away_win_rate'.
    """
    home_wr = max(0.0, min(1.0, row_dict["home_win_rate"]))
    away_wr = max(0.0, min(1.0, row_dict["away_win_rate"]))
    home_xg = 0.5 + 2.0 * home_wr
    away_xg = 0.5 + 2.0 * away_wr
    return home_xg, away_xg


# -----------------------------
# Load model and data once
# -----------------------------
model, val_acc, matches_df = train_gb_pipeline()
all_teams = sorted(set(matches_df["home_team"]).union(set(matches_df["away_team"])))
all_leagues = sorted(matches_df["league_name"].unique())
all_levels = sorted(matches_df["level"].unique())


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    # ⚽ Football Match Insights

    Explore historical matches, team performance, and predict outcomes with a Gradient Boosting model.
    """
)


# -----------------------------
# Tabs for the dashboard
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Welcome", "Predict Match", "Team EDA"])

# ==========================
# Tab 1: Welcome / Favorites
# ==========================
with tab1:
    st.subheader("Welcome, football fan!")

    st.write(
        "Tell us your favorite club and country. "
        "This is just for fun and will not affect predictions."
    )

    fav_club = st.selectbox("Your favorite club", all_teams)

    favorite_countries_list = [
        "France", "Spain", "Argentina", "England", "Portugal", "Brazil",
        "Netherlands", "Morocco", "Belgium", "Germany", "Croatia",
        "Italy", "Colombia", "Senegal", "Mexico", "USA", "Uruguay",
        "Japan", "Switzerland", "Denmark", "Iran", "Turkiye",
        "Ecuador", "Austria", "South Korea"
    ]
    fav_country = st.selectbox("Your favorite country", favorite_countries_list)

    if "fav_club" not in st.session_state:
        st.session_state["fav_club"] = fav_club
        st.session_state["fav_country"] = fav_country

    if st.button("Save my favorites"):
        st.session_state["fav_club"] = fav_club
        st.session_state["fav_country"] = fav_country
        st.success(
            f"Saved! Club: {fav_club}, Country: {fav_country}. "
            "Head to the other tabs to explore matches and predictions."
        )

# ==========================
# Tab 2: Prediction
# ==========================
with tab2:
    st.subheader("Predict a match outcome")

    st.write(f"Validation accuracy of the model: {val_acc:.3f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        home_team = st.selectbox("Home team", all_teams, key="home_team_pred")
    with col2:
        away_team = st.selectbox("Away team", all_teams, key="away_team_pred")
    with col3:
        league = st.selectbox("League", all_leagues, key="league_pred")

    level = st.selectbox("Level", all_levels, key="level_pred")

    if home_team == away_team:
        st.warning("Home and away team must be different.")
    else:
        if st.button("Predict outcome", key="predict_btn"):
            home_wr = matches_df.loc[
                matches_df["home_team"] == home_team, "home_win_rate"
            ].mean()
            away_wr = matches_df.loc[
                matches_df["away_team"] == away_team, "away_win_rate"
            ].mean()

            input_df = pd.DataFrame(
                [{
                    "home_team": home_team,
                    "away_team": away_team,
                    "league_name": league,
                    "level": level,
                    "home_win_rate": home_wr,
                    "away_win_rate": away_wr,
                }]
            )

            pred = model.predict(input_df)[0]
            st.success(f"Predicted result: {pred}")

            home_xg, away_xg = estimate_expected_goals(
                {
                    "home_win_rate": home_wr,
                    "away_win_rate": away_wr,
                }
            )
            st.info(
                f"Estimated expected goals — {home_team}: {home_xg:.2f}, "
                f"{away_team}: {away_xg:.2f}"
            )

# ==========================
# Tab 3: Team EDA (initial)
# ==========================
with tab3:
    st.subheader("Team and country analysis")

    mode = st.radio("Analyze:", ["Clubs", "Countries"], horizontal=True)

    if mode == "Clubs":
        club = st.selectbox("Club", all_teams)
        club_df = matches_df[
            (matches_df["home_team"] == club) | (matches_df["away_team"] == club)
        ]
        st.write(f"Total matches found for {club}: {len(club_df)}")
        st.dataframe(club_df.head(20))
    else:
        countries = [
            "France", "Spain", "Argentina", "England", "Portugal", "Brazil",
            "Netherlands", "Morocco", "Belgium", "Germany", "Croatia",
            "Italy", "Colombia", "Senegal", "Mexico", "USA", "Uruguay",
            "Japan", "Switzerland", "Denmark", "Iran", "Turkiye",
            "Ecuador", "Austria", "South Korea"
        ]
        country = st.selectbox("Country", countries)
        country_df = matches_df[
            (matches_df["home_team"] == country) | (matches_df["away_team"] == country)
        ]
        st.write(f"Total matches found for {country}: {len(country_df)}")
        st.dataframe(country_df.head(20))
