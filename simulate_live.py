"""
simulate_live.py — Simulated real-time traffic feed.

Live mode has no real sensors, so each tick perturbs every edge's vehicle
count with Gaussian noise and re-runs congestion prediction — the dashboard
then refreshes the map as if live data arrived.

Determinism: the RNG for tick N is seeded with (base_seed + N), so the same
tick always produces the same state. Pure functions; no global state.
"""

import numpy as np

from predict import predict_congestion, typical_conditions
from road_network import CONGESTION_LEVELS, all_edge_ids

NOISE_STD = 10.0  # vehicles, per tick


def init_state(hour: int, day_of_week: str) -> dict:
    """Starting live state: typical vehicle counts for the slot."""
    return {
        eid: typical_conditions(eid, hour, day_of_week)["vehicle_count"]
        for eid in all_edge_ids()
    }


def _speed_for_count(vehicle_count: float, weather: str = "Clear") -> float:
    speed = 50.0 - vehicle_count * 0.25
    if weather == "Rain":
        speed *= 0.85
    return round(float(np.clip(speed, 8, 60)), 1)


def tick(state: dict, tick_index: int, hour: int, day_of_week: str,
         weather: str = "Clear", is_holiday: int = 0,
         base_seed: int = 42) -> tuple:
    """
    Advance the simulation one tick.

    Returns (new_state, congestion) where new_state maps edge_id ->
    vehicle_count and congestion maps edge_id -> 'Low'/'Medium'/'High'.
    """
    rng = np.random.default_rng(base_seed + tick_index)
    edge_ids = all_edge_ids()  # stable order -> stable noise assignment
    noise = rng.normal(0, NOISE_STD, size=len(edge_ids))

    new_state = {}
    congestion = {}
    for eid, n in zip(edge_ids, noise):
        count = max(5.0, float(state[eid]) + float(n))
        new_state[eid] = round(count, 1)
        speed = _speed_for_count(count, weather)
        label = predict_congestion(eid, hour, day_of_week, weather,
                                   is_holiday,
                                   vehicle_count=count, avg_speed_kmph=speed)
        assert label in CONGESTION_LEVELS, f"Bad label: {label}"
        congestion[eid] = label
    return new_state, congestion
