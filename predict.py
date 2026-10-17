"""
predict.py — Reusable congestion inference helper.

Single entry point used by the router AND the Streamlit dashboard, so the
prediction logic lives in exactly one place:

    predict_congestion(road_id, hour, day_of_week, weather,
                       is_holiday, vehicle_count=None, avg_speed_kmph=None) -> str

When live sensor readings are unavailable, typical historical conditions for
the (road, hour, weekday/weekend) slot are used instead.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from preprocess import PEAK_HOURS, WEEKEND

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


@lru_cache(maxsize=1)
def _load_pipeline():
    return joblib.load(MODELS_DIR / "traffic_rf_model.pkl")


@lru_cache(maxsize=1)
def _load_typical() -> dict:
    """
    Flattened lookup: (road_id, hour, is_weekend) -> {vehicle_count, avg_speed_kmph}.
    JSON object keys are strings, so they are converted back to ints here.
    """
    with open(MODELS_DIR / "typical_conditions.json") as f:
        raw = json.load(f)
    flat = {}
    for road_id, by_hour in raw.items():
        for hour_s, by_we in by_hour.items():
            for we_s, vals in by_we.items():
                flat[(road_id, int(hour_s), int(we_s))] = vals
    return flat


def typical_conditions(road_id: str, hour: int, day_of_week: str) -> dict:
    """Median vehicle_count / avg_speed for a (road, hour, weekday/weekend) slot."""
    flat = _load_typical()
    key = (road_id, hour, int(day_of_week in WEEKEND))
    if key in flat:
        return flat[key]
    # Fallback: any slot for this road (should not normally happen).
    for (r, _h, _w), vals in flat.items():
        if r == road_id:
            return vals
    raise KeyError(f"No typical conditions for road {road_id}")


def predict_congestion(road_id: str, hour: int, day_of_week: str,
                       weather: str, is_holiday: int,
                       vehicle_count=None, avg_speed_kmph=None) -> str:
    """
    Predict congestion ('Low' / 'Medium' / 'High') for one road segment.

    Args:
        road_id: edge id like 'A-B'.
        hour: 0-23.  day_of_week: 'Monday'..'Sunday'.
        weather: 'Clear' or 'Rain'.  is_holiday: 0/1.
        vehicle_count / avg_speed_kmph: live readings; when omitted,
            typical historical conditions for the slot are used.
    """
    if vehicle_count is None or avg_speed_kmph is None:
        typical = typical_conditions(road_id, hour, day_of_week)
        vehicle_count = typical["vehicle_count"]
        avg_speed_kmph = typical["avg_speed_kmph"]
        # Rain slows traffic relative to the typical (usually clear) reading.
        if weather == "Rain":
            avg_speed_kmph = max(8.0, avg_speed_kmph * 0.85)

    row = pd.DataFrame([{
        "hour": int(hour),
        "is_peak_hour": int(hour in PEAK_HOURS),
        "is_weekend": int(day_of_week in WEEKEND),
        "is_holiday": int(is_holiday),
        "vehicle_count": float(vehicle_count),
        "avg_speed_kmph": float(avg_speed_kmph),
        "day_of_week": day_of_week,
        "weather": weather,
        "road_id": road_id,
    }])
    return str(_load_pipeline().predict(row)[0])
