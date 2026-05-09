# Football ML Pipeline Dashboard ⚽

This project is a small football analytics dashboard built with Python and Streamlit.  
It builds a local database from raw match data, trains a machine learning model, and lets you explore matches and predict results for popular clubs and countries.

---

## Live app

You can open the app here:

https://football-ml-pipeline-dashboard.streamlit.app/

---

## What this project does

- Reads raw match data (from a parquet file).
- Builds a local SQLite database called `football.db`.
- Calculates simple team statistics like games played and win rate.
- Trains a Gradient Boosting model to predict match results (home win, away win, draw).
- Shows a Streamlit dashboard with three main tabs:
  - **Welcome** – choose your favourite club and country (just for fun).
  - **Predict Match** – pick home and away teams and get a predicted result and expected goals.
  - **Team EDA** – basic data view for clubs and countries.

The main goal is to practice an end‑to‑end ML workflow: data → database → model → dashboard.

---

## How to run the app locally

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-user>/football-ml-pipeline.git
   cd football-ml-pipeline
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure the data is available**

   - Put the raw data file as `data/games.parquet`.  
   - The `build.py` script will read this file and create `football.db` the first time the app runs.

5. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   Streamlit will open a browser window (or give you a local URL).  
   You should see the “Football Match Insights” dashboard with the three tabs.

---

## Main files in simple words

- **`app.py`**  
  - Main Streamlit script.  
  - Connects to the database, loads data into pandas, trains the model, and builds the UI.

- **`build.py`**  
  - Reads the raw matches file and creates the `football.db` SQLite database with tables for teams, leagues, and matches.

- **`requirements.txt`**  
  - List of Python packages needed (Streamlit, pandas, scikit‑learn, etc.).

---

## Notes

- This is a learning project, so the model is simple and the “expected goals” calculation is only a rough heuristic, not a full xG model.
- Feel free to fork the repo and improve the EDA, charts, or model.

---

## Future ideas

Some ideas to improve this project later:

- Improve the accuracy of the model for predictions, this is a baseline model with minimal tweakings on hyperparameters
- Add more animations, font style and polish to the UI.

## Acknowledgements/Credits
- I was able to complete this baseline project with the help of Perplexity Pro, right from database creation, building ML model & streamlit app creation. 
