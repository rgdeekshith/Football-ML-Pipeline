# app.py

# -----------------------------
# Imports
# -----------------------------
import os
import sqlite3  # For connecting to football.db

import pandas as pd  # For data manipulation
import streamlit as st  # Streamlit web framework

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from build import build_database  # make sure this import is present


# -----------------------------
# Basic page config (title, icon, layout)
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

    Cached as a resource so you keep a single connection per session.
    """
    ensure_db()
    conn = sqlite3.connect(db_path)
    return conn


@st.cache_data(show_spinner=False)
def load_matches_df():
    """
    Load matches with teams and leagues, plus win rates, similar
    to your training query in football_ml_pipeline.py.
    """
    conn = get_connection()

    # Core matches + team + league info
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
    """
    matches_df = pd.read_sql(query, conn)

    # Compute simple win stats per team
    team_stats = {}
    for row in matches_df.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        result = row.result_class

        # Ensure dict entries
        for team in [home, away]:
            if team not in team_stats:
                team_stats[team] = {"games": 0, "wins": 0}

        # Count games
        team_stats[home]["games"] += 1
        team_stats[away]["games"] += 1

        # Count wins
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
# Build / train the Gradient Boosting pipeline
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_gb_pipeline():
    """
    Train the Gradient Boosting pipeline on matches_df.

    Cached as a resource so it's trained only once per session.
    """
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

    # Split (simple; in app we don't need validation metrics every time)
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

    # Simple validation accuracy for info
    from sklearn.metrics import accuracy_score

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    return model, val_acc, matches_df


# -----------------------------
# Utility: expected goals (simple approximation)
# -----------------------------
def estimate_expected_goals(row):
    """
    Simple heuristic: convert win rates into approximate expected goals.
    This is NOT a true xG model, just a placeholder.
    """
    # Clip win rates to avoid extremes
    home_wr = max(0.0, min(1.0, row["home_win_rate"]))
    away_wr = max(0.0, min(1.0, row["away_win_rate"]))

    # Scale into 0.5–2.5 goals range
    home_xg = 0.5 + 2.0 * home_wr
    away_xg = 0.5 + 2.0 * away_wr
    return home_xg, away_xg


# -----------------------------
# Load model and data once
# -----------------------------
model, val_acc, matches_df = train_gb_pipeline()

# Get unique lists for selectors
all_teams = sorted(set(matches_df["home_team"]).union(set(matches_df["away_team"])))
all_leagues = sorted(matches_df["league_name"].unique())
all_levels = sorted(matches_df["level"].unique())

# -----------------------------
# UI: main title and tabs
# -----------------------------
st.markdown(
    """
    # ⚽ Football Match Insights

    Explore historical matches, team performance, and predict outcomes with a Gradient Boosting model.
    """
)

# (…rest of your UI code: tabs, filters, prediction form, charts, etc…)
