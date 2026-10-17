# How to Run — Intelligent Traffic Management System

Works on **Windows**, **macOS**, and **Linux**. You need Python 3.9+ and Git.

## 1. Get the code

```bash
git clone <repo-url>
cd intelligent-traffic-management
```

## 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```
> Tip: if `pip` isn't found, use `python -m pip install -r requirements.txt`
> (or `python3 -m pip ...` on macOS/Linux).

## 4. Generate the dataset

```bash
python generate_data.py        # Windows
python3 generate_data.py       # macOS / Linux
```
Creates `data/traffic_data.csv` (~8,000 synthetic rows, seed 42).
Git-ignored on purpose — regenerates byte-identically.

## 5. Train the models

```bash
python model.py                # Windows
python3 model.py               # macOS / Linux
```
Trains Random Forest (+ XGBoost / Logistic Regression comparison),
prints accuracy and the classification report, and saves artifacts to `models/`.
Expect test accuracy ≥ 0.85.

## 6. Run the unit tests (optional but recommended)

```bash
python test_routing.py
```
Tests the hand-written Dijkstra, Yen's k-shortest paths, and blocked-edge handling.

## 7. Launch the dashboard

```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`. Pick source/destination/day/hour/weather,
then explore: optimal + alternative routes, the color-coded network map,
🔴 live mode, ▶ 24h time-lapse, 🚧 incident simulator, and the 📊 Analytics
page (sidebar navigation).

> First launch: Streamlit may ask for an email — press **Enter** to skip.
> Stop the app with `Ctrl+C`.
