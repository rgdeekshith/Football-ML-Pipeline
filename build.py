# build.py
import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = "football.db"
RAW_PATH = "raw_games.parquet"  # or "raw_games.csv"

def build_database():
    print("Building football.db from raw_games...")
    # Load raw data
    if RAW_PATH.endswith(".parquet"):
        df = pd.read_parquet(RAW_PATH)
    else:
        df = pd.read_csv(RAW_PATH)

    # Open SQLite connection
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop tables if they exist (to rebuild cleanly)
    cur.execute("DROP TABLE IF EXISTS matches;")
    cur.execute("DROP TABLE IF EXISTS teams;")
    cur.execute("DROP TABLE IF EXISTS leagues;")

    # Create tables (simplified schema similar to your pipeline)
    cur.execute("""
    CREATE TABLE teams (
        team_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        team_name TEXT UNIQUE NOT NULL,
        country   TEXT,
        code      TEXT,
        continent TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE leagues (
        league_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        league_name TEXT UNIQUE NOT NULL,
        country     TEXT,
        continent   TEXT,
        level       TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE matches (
        match_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        league_id    INTEGER NOT NULL,
        match_date   TEXT NOT NULL,
        full_time    TEXT,
        home_score   INTEGER NOT NULL,
        away_score   INTEGER NOT NULL,
        result_class TEXT NOT NULL
    );
    """)

    conn.commit()

    # ---------- Build teams ----------
    home_teams = df[["home", "home_country", "home_code", "home_continent"]].copy()
    home_teams.columns = ["team_name", "country", "code", "continent"]

    away_teams = df[["away", "away_country", "away_code", "away_continent"]].copy()
    away_teams.columns = ["team_name", "country", "code", "continent"]

    teams_df = pd.concat([home_teams, away_teams], ignore_index=True)
    teams_df = teams_df.dropna(subset=["team_name"])
    teams_df["team_name"] = teams_df["team_name"].astype(str).str.strip()
    teams_df = teams_df.drop_duplicates(subset=["team_name"]).reset_index(drop=True)

    teams_df.to_sql("teams", conn, if_exists="append", index=False)

    # Create mapping team_name -> team_id
    teams_db = pd.read_sql("SELECT team_id, team_name FROM teams;", conn)
    team_name_to_id = dict(zip(teams_db["team_name"], teams_db["team_id"]))

    # ---------- Build leagues ----------
    leagues_df = pd.DataFrame({
        "league_name": df["competition"].astype(str).str.strip(),
        "country": df["competition"].astype(str).str.strip(),
        "continent": df["continent"].astype(str).str.strip(),
        "level": df["level"].astype(str).str.strip(),
    }).drop_duplicates(subset=["league_name"]).reset_index(drop=True)

    leagues_df.to_sql("leagues", conn, if_exists="append", index=False)
    leagues_db = pd.read_sql("SELECT league_id, league_name FROM leagues;", conn)
    league_name_to_id = dict(zip(leagues_db["league_name"], leagues_db["league_id"]))

    # ---------- Build matches ----------
    matches_to_insert = []

    for row in df.itertuples(index=False):
        home_team_name = getattr(row, "home")
        away_team_name = getattr(row, "away")
        league_name = getattr(row, "competition")
        date_value = getattr(row, "date")

        # Handle missing scores as 0
        home_score = int(getattr(row, "gh") if pd.notna(getattr(row, "gh")) else 0)
        away_score = int(getattr(row, "ga") if pd.notna(getattr(row, "ga")) else 0)
        full_time = getattr(row, "full_time")

        # Filter if mapping missing
        if home_team_name not in team_name_to_id or away_team_name not in team_name_to_id:
            continue
        if league_name not in league_name_to_id:
            continue

        home_id = team_name_to_id[home_team_name]
        away_id = team_name_to_id[away_team_name]
        league_id = league_name_to_id[league_name]

        # Handle date
        try:
            match_date = pd.to_datetime(date_value).date().isoformat()
        except Exception:
            continue

        # Result class
        if home_score > away_score:
            result_class = "home"
        elif away_score > home_score:
            result_class = "away"
        else:
            result_class = "draw"

        matches_to_insert.append(
            (
                home_id,
                away_id,
                league_id,
                match_date,
                full_time,
                home_score,
                away_score,
                result_class,
            )
        )

    cur.executemany(
        """
        INSERT INTO matches (
            home_team_id,
            away_team_id,
            league_id,
            match_date,
            full_time,
            home_score,
            away_score,
            result_class
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        matches_to_insert,
    )
    conn.commit()
    conn.close()
    print(f"Inserted {len(matches_to_insert)} matches.")
