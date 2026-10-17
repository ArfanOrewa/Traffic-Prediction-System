# 🚦 Intelligent Traffic Management System

Predict road congestion with machine learning and compute the fastest route
with a hand-written Dijkstra implementation — wrapped in an interactive
Streamlit app with live simulation, alternative routes, a 24-hour time-lapse,
an incident simulator, and an analytics page.

> **Dataset is fully synthetic** (fixed seed 42) — no real traffic data involved.

## What it does

1. **Congestion prediction** — a `RandomForestClassifier` predicts Low / Medium /
   High congestion per road segment from historical traffic features
   (vehicle count, speed, hour, weekday, weather, holiday, road id).
2. **Optimal routing** — Dijkstra's algorithm (implemented manually with `heapq`)
   finds the fastest path, where each road's travel time is scaled by its
   predicted congestion (Low ×1.0, Medium ×1.5, High ×2.2). Yen's algorithm
   adds Google-Maps-style alternative routes.
3. **Interactive dashboard** — pick source, destination, day, hour and weather;
   get the optimal route, ETA, per-segment congestion, and a color-coded
   network map. Extras: 🔴 live simulation mode, ▶ 24-hour time-lapse,
   🚧 incident simulator (close roads, watch it re-route), and a 📊 analytics
   page with a Random Forest vs XGBoost vs Logistic Regression comparison.

## Project structure

```
intelligent-traffic-management/
├── generate_data.py        # synthetic traffic dataset -> data/traffic_data.csv
├── road_network.py         # 10-intersection graph (single source of truth)
├── preprocess.py           # feature engineering + ColumnTransformer
├── model.py                # train Random Forest (+XGBoost/LogReg comparison)
├── predict.py              # inference helper (router + dashboard share it)
├── routing.py              # manual Dijkstra + Yen's k-shortest paths
├── simulate_live.py        # live-mode tick logic
├── dashboard.py            # Streamlit app
├── pages/
│   └── 1_📊_Analytics.py   # peak hours, busiest segments, model comparison
├── test_routing.py         # unit tests (Dijkstra, k-routes, weights)
├── requirements.txt
├── README.md
├── HOW_TO_RUN.md
├── .gitignore
├── data/                   # generated, git-ignored
└── models/                 # trained artifacts (pkl git-ignored)
```

## The road network

10 intersections (A–J), 15 bidirectional road segments with base travel times
in minutes. `road_network.py` is the single source of truth imported by the
generator, router, and dashboard.

## Quickstart

```bash
pip install -r requirements.txt
python generate_data.py     # build the ~8,000-row dataset
python model.py             # train + evaluate (saves models/)
python test_routing.py      # unit tests
streamlit run dashboard.py  # launch the app
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for Windows / macOS / Linux specifics.

## Model results

Reported on the 20% hold-out set by `python model.py` (also saved to
`models/model_comparison.json` and shown in the app):

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Random Forest | 0.9962 | 0.9956 | 0.9963 | 0.9960 |
| XGBoost | 0.9981 | 0.9977 | 0.9987 | 0.9982 |
| Logistic Regression | 0.9869 | 0.9853 | 0.9866 | 0.9859 |

*(Reproduced by `python model.py` — also saved to `models/model_comparison.json`.)*

## Tech stack

Python · pandas · scikit-learn · XGBoost · Streamlit · networkx · matplotlib
