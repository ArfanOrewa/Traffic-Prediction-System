"""
generate_data.py — Synthetic traffic dataset generator.

Creates ~8,000 realistic traffic records and saves them to
data/traffic_data.csv. A fixed random seed (42) makes every run produce a
byte-identical file. No real traffic data is involved.

Columns:
    timestamp, road_id, day_of_week, hour, vehicle_count,
    avg_speed_kmph, weather, is_holiday, congestion_level

The congestion label follows a deterministic rule (see label_congestion),
so an ML model trained on this data can genuinely learn the pattern.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from road_network import all_edge_ids

# ---------------------------------------------------------------- constants
RANDOM_SEED = 42
N_ROWS = 8000
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "traffic_data.csv"

START_DATE = datetime(2023, 9, 1)
N_DAYS = 90  # 2023-09-01 .. 2023-11-29

PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}
NIGHT_HOURS = {0, 1, 2, 3, 4, 5}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


# ------------------------------------------------------------------ helpers
def label_congestion(vehicle_count: int, hour: int,
                     weather: str, is_holiday: int) -> str:
    """Deterministic congestion label from the spec's scoring rule."""
    score = float(vehicle_count)
    if hour in PEAK_HOURS:
        score += 25
    if weather == "Rain":
        score += 15
    if is_holiday:
        score -= 20
    if score >= 110:
        return "High"
    if score >= 70:
        return "Medium"
    return "Low"


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    edge_ids = all_edge_ids()
    n_edges = len(edge_ids)

    rows = []
    for _ in range(N_ROWS):
        road_id = edge_ids[int(rng.integers(0, n_edges))]

        day_offset = int(rng.integers(0, N_DAYS))
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        ts = START_DATE + timedelta(days=day_offset, hours=hour, minutes=minute)
        day_of_week = DAY_NAMES[ts.weekday()]

        is_holiday = int(rng.random() < 0.08)

        # Vehicle count: base volume shaped by time of day and holidays.
        count = float(rng.uniform(20, 120))
        if hour in PEAK_HOURS:
            count *= 1.8
        elif hour in NIGHT_HOURS:
            count *= 0.6
        if is_holiday:
            count *= 0.6
        count += float(rng.normal(0, 8))
        vehicle_count = max(5, int(round(count)))

        # Average speed falls as volume rises; rain slows traffic further.
        weather = "Rain" if rng.random() < 0.20 else "Clear"
        speed = 50.0 - vehicle_count * 0.25 + float(rng.normal(0, 3))
        if weather == "Rain":
            speed *= 0.85
        avg_speed = round(float(np.clip(speed, 8, 60)), 1)

        congestion = label_congestion(vehicle_count, hour, weather, is_holiday)

        rows.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "road_id": road_id,
            "day_of_week": day_of_week,
            "hour": hour,
            "vehicle_count": vehicle_count,
            "avg_speed_kmph": avg_speed,
            "weather": weather,
            "is_holiday": is_holiday,
            "congestion_level": congestion,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
