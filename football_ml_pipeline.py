!pip install pandas pyarrow sqlalchemy

# Importing the dataset

import pandas as pd
url = "https://raw.githubusercontent.com/schochastics/football-data/master/data/results/games.parquet"
df = pd.read_parquet(url)
df.head()

# code to create these tables in SQLite using SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# 1. Create a database engine for SQLite
engine = create_engine("sqlite:///football.db", echo=False)

# 2. Base class for our ORM models (tables)
Base = declarative_base()

# 3. Create all tables in the database
Base.metadata.create_all(engine)

# 4. Create a Session factory to interact with the DB later
SessionLocal = sessionmaker(bind=engine)


# Tables Definition
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship

from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    DateTime,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from datetime import datetime

class Team(Base):
    __tablename__ = "teams"
    team_id = Column(Integer, primary_key=True, index=True)
    team_name = Column(String, unique=True, nullable=False)
    country = Column(String, nullable=True)
    code = Column(String, nullable=True)
    continent = Column(String, nullable=True)

class League(Base):
    __tablename__ = "leagues"
    league_id = Column(Integer, primary_key=True, index=True)
    league_name = Column(String, unique=True, nullable=False)
    country = Column(String, nullable=True)
    continent = Column(String, nullable=True)
    level = Column(String, nullable=True)

class Match(Base):
    __tablename__ = "matches"
    match_id = Column(Integer, primary_key=True, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    league_id = Column(Integer, ForeignKey("leagues.league_id"), nullable=False)
    match_date = Column(Date, nullable=False)
    full_time = Column(String, nullable=True)
    home_score = Column(Integer, nullable=False)
    away_score = Column(Integer, nullable=False)
    result_class = Column(String, nullable=False)
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    league = relationship("League", backref="matches")

class ModelInfo(Base):
    __tablename__ = "models"

    model_id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, unique=True, nullable=False)
    algorithm = Column(String, nullable=False)
    params_json = Column(String, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.model_id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    predicted_class = Column(String, nullable=False)
    prob_home = Column(Float, nullable=True)
    prob_away = Column(Float, nullable=True)
    prob_draw = Column(Float, nullable=True)

    model = relationship("ModelInfo")
    match = relationship("Match")

Base.metadata.create_all(bind=engine)


from sqlalchemy.exc import IntegrityError
import pandas as pd

# ---------- Build teams_df from games.parquet ----------

home_teams = df[["home", "home_country", "home_code", "home_continent"]].copy()
home_teams.columns = ["team_name", "country", "code", "continent"]

away_teams = df[["away", "away_country", "away_code", "away_continent"]].copy()
away_teams.columns = ["team_name", "country", "code", "continent"]

teams_df = pd.concat([home_teams, away_teams], axis=0, ignore_index=True)

teams_df = teams_df.dropna(subset=["team_name"])
teams_df["team_name"] = teams_df["team_name"].astype(str).str.strip()
teams_df = teams_df.drop_duplicates(subset=["team_name"]).reset_index(drop=True)

session = SessionLocal()

# Clear existing teams so reruns are clean
session.query(Team).delete()
session.commit()

# ---------- Insert teams ----------

teams_to_add = []
for row in teams_df.itertuples(index=False):
    team = Team(
        team_name=row.team_name,
        country=row.country,
        code=row.code,
        continent=row.continent
    )
    teams_to_add.append(team)

session.add_all(teams_to_add)

try:
    session.commit()
    print(f"Inserted {len(teams_to_add)} teams into the database.")
except IntegrityError as e:
    session.rollback()
    print("Error inserting teams:", e)

# ---------- team_name -> team_id mapping ----------

team_name_to_id = {
    team.team_name: team.team_id
    for team in session.query(Team).all()
}

print("Total teams:", len(team_name_to_id))
list(team_name_to_id.items())[:5]


from sqlalchemy.exc import IntegrityError
import pandas as pd

# ---------- Build leagues_df from games.parquet ----------

leagues_df = pd.DataFrame({
    "league_name": df["competition"].astype(str).str.strip(),
    "country": df["competition"].astype(str).str.strip(),
    "continent": df["continent"].astype(str).str.strip(),
    "level": df["level"].astype(str).str.strip()
})

leagues_df = leagues_df.dropna(subset=["league_name"])
leagues_df = leagues_df.drop_duplicates(subset=["league_name"]).reset_index(drop=True)

session = SessionLocal()

# Clear existing leagues so reruns are clean
session.query(League).delete()
session.commit()

# ---------- Insert leagues ----------

leagues_to_add = []
for row in leagues_df.itertuples(index=False):
    league = League(
        league_name=row.league_name,
        country=row.country,
        continent=row.continent,
        level=row.level
    )
    leagues_to_add.append(league)

session.add_all(leagues_to_add)

try:
    session.commit()
    print(f"Inserted {len(leagues_to_add)} leagues into the database.")
except IntegrityError as e:
    session.rollback()
    print("Error inserting leagues:", e)

# ---------- league_name -> league_id mapping ----------

league_name_to_id = {
    league.league_name: league.league_id
    for league in session.query(League).all()
}

print("Total leagues:", len(league_name_to_id))
list(league_name_to_id.items())[:5]


from sqlalchemy.exc import IntegrityError
from datetime import datetime
import pandas as pd

# ---------- Helper ----------

def get_result_class(home_score, away_score):
    if home_score > away_score:
        return "home"
    elif away_score > home_score:
        return "away"
    else:
        return "draw"

session = SessionLocal()

# Assumes team_name_to_id and league_name_to_id exist from previous cells

# Clear existing matches so reruns are clean
session.query(Match).delete()
session.commit()

matches_to_add = []

for row in df.itertuples(index=False):
    home_team_name = getattr(row, "home")
    away_team_name = getattr(row, "away")
    league_name = getattr(row, "competition")
    date_value = getattr(row, "date")
    
    # Handle NaN values for scores by filling with 0 before converting to int
    home_score = int(getattr(row, "gh") if pd.notna(getattr(row, "gh")) else 0)
    away_score = int(getattr(row, "ga") if pd.notna(getattr(row, "ga")) else 0)
    
    full_time = getattr(row, "full_time")

    if home_team_name not in team_name_to_id or away_team_name not in team_name_to_id:
        continue
    if league_name not in league_name_to_id:
        continue

    home_id = team_name_to_id[home_team_name]
    away_id = team_name_to_id[away_team_name]
    league_id = league_name_to_id[league_name]

    if isinstance(date_value, datetime):
        match_date = date_value.date()
    else:
        try:
            match_date = pd.to_datetime(date_value).date()
        except Exception:
            continue

    result_class = get_result_class(home_score, away_score)

    match = Match(
        home_team_id=home_id,
        away_team_id=away_id,
        league_id=league_id,
        match_date=match_date,
        full_time=full_time,
        home_score=home_score,
        away_score=away_score,
        result_class=result_class
    )

    matches_to_add.append(match)

session.add_all(matches_to_add)

try:
    session.commit()
    print(f"Inserted {len(matches_to_add)} matches into the database.")
except IntegrityError as e:
    session.rollback()
    print("Error inserting matches:", e)
    
    

import pandas as pd

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
WHERE th.team_name IN ("Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion", "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds United", "Liverpool", "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest", "Sunderland", "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers", "Alavés", "Athletic Bilbao", "Atlético Madrid", "Barcelona", "Celta Vigo", "Elche", "Espanyol", "Getafe", "Girona", "Levante", "Mallorca", "Osasuna", "Oviedo", "Rayo Vallecano", "Real Betis", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal", "FC Augsburg", "Union Berlin", "Werder Bremen", "Borussia Dortmund", "Eintracht Frankfurt", "SC Freiburg", "Hamburger SV", "1. FC Heidenheim", "TSG Hoffenheim", "1. FC Köln", "RB Leipzig", "Bayer Leverkusen", "Mainz 05", "Borussia Mönchengladbach", "Bayern Munich", "FC St. Pauli", "VfB Stuttgart", "VfL Wolfsburg", "Angers", "Auxerre", "Brest", "Le Havre", "Lens", "Lille", "Lorient", "Lyon", "Marseille", "Metz", "Monaco", "Nantes", "Nice", "Paris FC", "Paris Saint-Germain", "Rennes", "Strasbourg", "Toulouse", "Ajax", "AZ", "Excelsior", "Feyenoord", "Go Ahead Eagles", "Groningen", "Heerenveen", "Heracles Almelo", "NAC Breda", "NEC", "PEC Zwolle", "PSV Eindhoven", "Sparta Rotterdam", "Telstar", "Twente", "Utrecht", "Volendam", "Willem II", "France", "Spain", "Argentina", "England", "Portugal", "Brazil", "Netherlands", "Morocco", "Belgium", "Germany", "Croatia", "Italy", "Colombia", "Senegal", "Mexico", "USA", "Uruguay", "Japan", "Switzerland", "Denmark", "Iran", "Turkiye", "Ecuador", "Austria", "South Korea")
OR ta.team_name IN ("Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion", "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds United", "Liverpool", "Manchester City", "Manchester United", "Newcastle United", "Nottingham Forest", "Sunderland", "Tottenham Hotspur", "West Ham United", "Wolverhampton Wanderers", "Alavés", "Athletic Bilbao", "Atlético Madrid", "Barcelona", "Celta Vigo", "Elche", "Espanyol", "Getafe", "Girona", "Levante", "Mallorca", "Osasuna", "Oviedo", "Rayo Vallecano", "Real Betis", "Real Madrid", "Real Sociedad", "Sevilla", "Valencia", "Villarreal", "FC Augsburg", "Union Berlin", "Werder Bremen", "Borussia Dortmund", "Eintracht Frankfurt", "SC Freiburg", "Hamburger SV", "1. FC Heidenheim", "TSG Hoffenheim", "1. FC Köln", "RB Leipzig", "Bayer Leverkusen", "Mainz 05", "Borussia Mönchengladbach", "Bayern Munich", "FC St. Pauli", "VfB Stuttgart", "VfL Wolfsburg", "Angers", "Auxerre", "Brest", "Le Havre", "Lens", "Lille", "Lorient", "Lyon", "Marseille", "Metz", "Monaco", "Nantes", "Nice", "Paris FC", "Paris Saint-Germain", "Rennes", "Strasbourg", "Toulouse", "Ajax", "AZ", "Excelsior", "Feyenoord", "Go Ahead Eagles", "Groningen", "Heerenveen", "Heracles Almelo", "NAC Breda", "NEC", "PEC Zwolle", "PSV Eindhoven", "Sparta Rotterdam", "Telstar", "Twente", "Utrecht", "Volendam", "Willem II", "France", "Spain", "Argentina", "England", "Portugal", "Brazil", "Netherlands", "Morocco", "Belgium", "Germany", "Croatia", "Italy", "Colombia", "Senegal", "Mexico", "USA", "Uruguay", "Japan", "Switzerland", "Denmark", "Iran", "Turkiye", "Ecuador", "Austria", "South Korea");
"""
matches_df = pd.read_sql(query, con=engine)

matches_df.head()
len(matches_df)
matches_df["result_class"].value_counts(normalize=True)


# Compute simple win stats per team from matches_df
team_stats = {}

for row in matches_df.itertuples(index=False):
    home = row.home_team
    away = row.away_team
    result = row.result_class

    # Ensure both teams exist in dict
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

# Convert to DataFrame
team_stats_df = (
    pd.DataFrame.from_dict(team_stats, orient="index")
    .reset_index()
    .rename(columns={"index": "team_name"})
)

team_stats_df["win_rate"] = team_stats_df["wins"] / team_stats_df["games"]
team_stats_df.head()


# Merge home team win_rate
matches_df = matches_df.merge(
    team_stats_df[["team_name", "win_rate"]],
    left_on="home_team",
    right_on="team_name",
    how="left"
).rename(columns={"win_rate": "home_win_rate"}).drop(columns=["team_name"])

# Merge away team win_rate
matches_df = matches_df.merge(
    team_stats_df[["team_name", "win_rate"]],
    left_on="away_team",
    right_on="team_name",
    how="left"
).rename(columns={"win_rate": "away_win_rate"}).drop(columns=["team_name"])

matches_df[["home_team", "away_team", "home_win_rate", "away_win_rate"]].head()


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# 1. Define target and feature columns
# -------------------------------------------------
target_col = "result_class"  # "home" / "draw" / "away"

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

# -------------------------------------------------
# 2. Split into train and validation sets
# -------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 3. Preprocessing: numeric vs categorical
# -------------------------------------------------
numeric_features = [
    "home_win_rate",
    "away_win_rate",
]

categorical_features = [
    "home_team",
    "away_team",
    "league_name",
    "level",
]

numeric_transformer = "passthrough"

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -------------------------------------------------
# 4. Define Gradient Boosting model
# -------------------------------------------------
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# -------------------------------------------------
# 5. Build Pipeline: preprocessing + model
# -------------------------------------------------
model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("gb", gb_clf),
    ]
)

# -------------------------------------------------
# 6. Fit the pipeline on training data
# -------------------------------------------------
model.fit(X_train, y_train)

# -------------------------------------------------
# 7. Evaluate on validation set
# -------------------------------------------------
y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
print("Gradient Boosting (win rate) validation accuracy:", val_acc)

# -------------------------------------------------
# 8. Get class probabilities on validation set
# -------------------------------------------------
proba_val = model.predict_proba(X_val)

print("Classes (model.classes_):", model.classes_)
print("First validation row predicted class:", y_val_pred[0])
print("First validation row probabilities:", proba_val[0])