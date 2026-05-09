# app.py

# -----------------------------
# Imports
# -----------------------------
import os
import sqlite3                       # For connecting to football.db
import pandas as pd                  # For data manipulation
import streamlit as st               # Streamlit web framework

from build import build_database   # Function that builds football.db from raw data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

DB_PATH = "football.db"

# -----------------------------
# Basic page config (title, icon, layout)
# -----------------------------
st.set_page_config(
    page_title="Football Match Insights",
    page_icon="⚽",
    layout="wide"
)

# -----------------------------
# Database helper
# -----------------------------
def ensure_db():
    if not os.path.exists(DB_PATH):
        build_database()

@st.cache_data(show_spinner=False)
def get_connection(db_path=DB_PATH):
    ensure_db()
    conn = sqlite3.connect(db_path)
    return conn

@st.cache_data(show_spinner=False)
def load_matches_df():
    """
    Load matches with teams and leagues, plus win rates.
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
    matches_df = matches_df.merge(
        team_stats_df[["team_name", "win_rate"]],
        left_on="home_team",
        right_on="team_name",
        how="left"
    ).rename(columns={"win_rate": "home_win_rate"}).drop(columns=["team_name"])

    # Merge away win_rate
    matches_df = matches_df.merge(
        team_stats_df[["team_name", "win_rate"]],
        left_on="away_team",
        right_on="team_name",
        how="left"
    ).rename(columns={"win_rate": "away_win_rate"}).drop(columns=["team_name"])

    return matches_df

# -----------------------------
# Build / train the Gradient Boosting pipeline
# -----------------------------
@st.cache_resource(show_spinner=True)
def train_gb_pipeline():
    """
    Train the Gradient Boosting pipeline on matches_df.
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

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
        random_state=42
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
# Utility: expected goals (simple approximation)
# -----------------------------
def estimate_expected_goals(row):
    home_wr = max(0.0, min(1.0, row["home_win_rate"]))
    away_wr = max(0.0, min(1.0, row["away_win_rate"]))
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
# UI: main title and tabs
# -----------------------------
st.markdown(
    "<h1 style='text-align: center; font-family: sans-serif;'>⚽ Football Match Insights Dashboard</h1>",
    unsafe_allow_html=True,
)
st.write(f"Current Gradient Boosting (win-rate) validation accuracy: **{val_acc:.3f}**")

tab1, tab2, tab3 = st.tabs(["🏠 Welcome", "📊 Match Prediction", "📈 Club & Country EDA"])

# -----------------------------
# Tab 1: Welcome / Fan profile
# -----------------------------
with tab1:
    st.subheader("Welcome, football fan!")

    st.write(
        "Tell us a bit about your football preferences. "
        "This is just for fun and does not affect any predictions."
    )

    col1, col2 = st.columns(2)

    with col1:
        fav_club = st.selectbox(
            "Your favourite club",
            options=sorted([t for t in all_teams if " " in t or t not in all_leagues]),
            help="Choose any club from the dataset."
        )

    with col2:
        fav_country = st.selectbox(
            "Your favourite country",
            options=sorted(
                [t for t in all_teams if t in [
                    "France", "Spain", "Argentina", "England",
                    "Portugal", "Brazil", "Netherlands",
                    "Germany", "Italy", "Belgium",
                    "Croatia", "USA", "Mexico"
                ]]
            ),
            help="Pick a national team you support."
        )

    st.success(
        f"Nice! A fan of **{fav_club}** and **{fav_country}** – let's see how teams perform in the other tabs."
    )

# -----------------------------
# Tab 2: Match Prediction
# -----------------------------
with tab2:
    st.subheader("Match Outcome & Expected Goals")

    st.write(
        "Choose teams and league context to predict match outcome probabilities "
        "and estimated goals using the trained Gradient Boosting model."
    )

    with st.container():
        col_left, col_right = st.columns(2)

        with col_left:
            league = st.selectbox(
                "League",
                options=all_leagues,
                index=0
            )

            level_options = sorted(
                matches_df.loc[matches_df["league_name"] == league, "level"].unique()
            )
            level = st.selectbox(
                "Level",
                options=level_options
            )

        with col_right:
            home_team = st.selectbox(
                "Home team",
                options=sorted(
                    matches_df.loc[matches_df["league_name"] == league, "home_team"].unique()
                ),
                key="home_team_select"
            )

            away_team = st.selectbox(
                "Away team",
                options=sorted(
                    matches_df.loc[matches_df["league_name"] == league, "away_team"].unique()
                ),
                key="away_team_select"
            )

        if st.button("Swap home/away teams"):
            home_team, away_team = away_team, home_team

    st.markdown("---")
    st.write("Using historical data from the database to compute win rates for the selected teams.")

    home_hist = matches_df[
        (matches_df["home_team"] == home_team) | (matches_df["away_team"] == home_team)
    ]
    away_hist = matches_df[
        (matches_df["home_team"] == away_team) | (matches_df["away_team"] == away_team)
    ]

    def compute_team_win_rate(df_team, team_name):
        if df_team.empty:
            return 0.5
        wins = 0
        games = len(df_team)
        for row in df_team.itertuples(index=False):
            if row.home_team == team_name and row.result_class == "home":
                wins += 1
            elif row.away_team == team_name and row.result_class == "away":
                wins += 1
        return wins / games if games > 0 else 0.5

    home_wr = compute_team_win_rate(home_hist, home_team)
    away_wr = compute_team_win_rate(away_hist, away_team)

    st.write(f"Estimated historical win rate for **{home_team}**: `{home_wr:.3f}`")
    st.write(f"Estimated historical win rate for **{away_team}**: `{away_wr:.3f}`")

    input_df = pd.DataFrame(
        {
            "home_team": [home_team],
            "away_team": [away_team],
            "league_name": [league],
            "level": [level],
            "home_win_rate": [home_wr],
            "away_win_rate": [away_wr],
        }
    )

    if st.button("Predict match outcome"):
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_

        prob_df = pd.DataFrame(
            {
                "Outcome": classes,
                "Probability": probs
            }
        ).sort_values("Probability", ascending=False)

        st.subheader("Predicted outcome probabilities")
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}), use_container_width=True)

        st.bar_chart(
            data=prob_df.set_index("Outcome"),
            height=300
        )

        home_xg, away_xg = estimate_expected_goals(
            {
                "home_win_rate": home_wr,
                "away_win_rate": away_wr
            }
        )

        st.subheader("Estimated expected goals (heuristic)")
        xg_df = pd.DataFrame(
            {
                "Team": [home_team, away_team],
                "Expected Goals": [home_xg, away_xg],
            }
        )
        st.dataframe(xg_df.style.format({"Expected Goals": "{:.2f}"}), use_container_width=True)
        st.bar_chart(
            data=xg_df.set_index("Team"),
            height=300
        )

# -----------------------------
# Tab 3: Club & Country EDA
# -----------------------------
with tab3:
    st.subheader("Club and Country Statistics")

    st.write(
        "Explore descriptive statistics for clubs and national teams using data from the database. "
        "This uses only football data (matches, teams, leagues) and not the ML model outputs."
    )

    all_countries = [
        "France", "Spain", "Argentina", "England", "Portugal", "Brazil",
        "Netherlands", "Germany", "Italy", "Belgium", "Croatia",
        "USA", "Mexico", "Uruguay", "Japan", "Switzerland", "Denmark",
        "Iran", "Turkey", "Ecuador", "Austria", "South Korea"
    ]

    st.markdown("### Clubs section")

    club_team = st.selectbox(
        "Choose a club",
        options=sorted([t for t in all_teams if t not in all_countries]),
        key="club_eda_select"
    )

    club_matches = matches_df[
        (matches_df["home_team"] == club_team) | (matches_df["away_team"] == club_team)
    ]

    if club_matches.empty:
        st.warning("No data available for this club.")
    else:
        st.write(f"Total matches found for **{club_team}**: {len(club_matches)}")

        wins = draws = losses = 0
        goals_for = goals_against = 0

        for row in club_matches.itertuples(index=False):
            if row.home_team == club_team:
                goals_for += row.home_score
                goals_against += row.away_score
                if row.result_class == "home":
                    wins += 1
                elif row.result_class == "away":
                    losses += 1
                else:
                    draws += 1
            else:
                goals_for += row.away_score
                goals_against += row.home_score
                if row.result_class == "away":
                    wins += 1
                elif row.result_class == "home":
                    losses += 1
                else:
                    draws += 1

        club_summary = pd.DataFrame(
            {
                "Metric": ["Games", "Wins", "Draws", "Losses", "Goals For", "Goals Against"],
                "Value": [len(club_matches), wins, draws, losses, goals_for, goals_against],
            }
        )
        st.dataframe(club_summary, use_container_width=True)

    st.markdown("---")
    st.markdown("### Countries section")

    country_team = st.selectbox(
        "Choose a country",
        options=sorted([t for t in all_teams if t in all_countries]),
        key="country_eda_select"
    )

    country_matches = matches_df[
        (matches_df["home_team"] == country_team) | (matches_df["away_team"] == country_team)
    ]

    if country_matches.empty:
        st.warning("No data available for this country.")
    else:
        st.write(f"Total matches found for **{country_team}**: {len(country_matches)}")

        wins = draws = losses = 0
        goals_for = goals_against = 0

        for row in country_matches.itertuples(index=False):
            if row.home_team == country_team:
                goals_for += row.home_score
                goals_against += row.away_score
                if row.result_class == "home":
                    wins += 1
                elif row.result_class == "away":
                    losses += 1
                else:
                    draws += 1
            else:
                goals_for += row.away_score
                goals_against += row.home_score
                if row.result_class == "away":
                    wins += 1
                elif row.result_class == "home":
                    losses += 1
                else:
                    draws += 1

        country_summary = pd.DataFrame(
            {
                "Metric": ["Games", "Wins", "Draws", "Losses", "Goals For", "Goals Against"],
                "Value": [len(country_matches), wins, draws, losses, goals_for, goals_against],
            }
        )
        st.dataframe(country_summary, use_container_width=True)
