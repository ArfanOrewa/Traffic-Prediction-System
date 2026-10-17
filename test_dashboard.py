"""test_dashboard.py — Functional tests for the Streamlit dashboard (AppTest).

Runs dashboard.py headlessly via streamlit.testing.v1.AppTest:
drives widgets, clicks buttons, and asserts on rendered outputs.
"""
from streamlit.testing.v1 import AppTest


def _run(**overrides):
    at = AppTest.from_file("dashboard.py", default_timeout=120).run()
    assert not at.exception, f"Script raised: {at.exception}"
    return at


def test_app_loads_with_title_and_route():
    at = _run()
    titles = [t.value for t in at.title]
    assert any("Intelligent Traffic Management" in t for t in titles), titles
    # Default A->J route metrics rendered
    metrics = [m.label for m in at.metric]
    assert "Optimal route" in metrics, metrics
    assert "Estimated travel time" in metrics, metrics
    # Congestion table rendered (map figure verified separately in headless render)
    assert len(at.dataframe) >= 1, "expected congestion dataframe"
    print("PASS: app loads, title + route metrics + congestion table")


def test_rush_hour_changes_congestion():
    at = AppTest.from_file("dashboard.py", default_timeout=120).run()
    at.sidebar.slider[0].set_value(3).run()   # 3 AM
    eta_night = float(
        [m for m in at.metric if m.label == "Estimated travel time"][0].value.split()[0]
    )
    at.sidebar.slider[0].set_value(18).run()  # 6 PM rush hour
    eta_rush = float(
        [m for m in at.metric if m.label == "Estimated travel time"][0].value.split()[0]
    )
    assert eta_rush != eta_night, "rush hour should change travel time"
    print(f"PASS: time-lapse changes ETA (night={eta_night} vs rush={eta_rush})")


def test_incident_reroutes():
    at = AppTest.from_file("dashboard.py", default_timeout=120).run()
    route_before = [m for m in at.metric if m.label == "Optimal route"][0].value
    # Extract an edge actually ON the optimal route, then block it
    nodes = [n.strip() for n in route_before.replace("→", " ").split() if n.strip() and n.strip() != "→"]
    assert len(nodes) >= 2, f"unexpected path format: {route_before}"
    edge_to_block = f"{nodes[0]}-{nodes[1]}"
    at.sidebar.multiselect[0].set_value([edge_to_block]).run()
    warnings = [w.value for w in at.warning]
    assert any("closed" in w for w in warnings), warnings
    route_after = [m for m in at.metric if m.label == "Optimal route"][0].value
    assert route_after != route_before, "blocking a road must change the route"
    print(f"PASS: incident reroutes ({route_before} -> {route_after})")


def test_same_source_destination_warns():
    at = AppTest.from_file("dashboard.py", default_timeout=120).run()
    at.sidebar.selectbox[1].set_value("A").run()  # dst = A, same as src
    warnings = [w.value for w in at.warning]
    assert any("same" in w for w in warnings), warnings
    print("PASS: same src/dst shows warning")


def test_live_mode_tick():
    # Live UI loop is time.sleep(5) + st.rerun() (infinite by design in AppTest),
    # so verify the underlying tick logic directly — same thing the UI calls.
    from simulate_live import init_state, tick
    from road_network import all_edge_ids, CONGESTION_LEVELS
    state = init_state(18, "Monday")
    assert set(state) == set(all_edge_ids())
    for i in range(3):
        state, congestion = tick(state, i, 18, "Monday", "Clear")
        assert set(congestion) == set(all_edge_ids())
        assert all(c in CONGESTION_LEVELS for c in congestion.values())
    # Determinism: same tick index -> same result
    s2, c2 = tick(init_state(18, "Monday"), 0, 18, "Monday", "Clear")
    s3, c3 = tick(init_state(18, "Monday"), 0, 18, "Monday", "Clear")
    assert c2 == c3, "ticks must be deterministic"
    print("PASS: live tick logic (3 ticks, valid labels, deterministic)")


def test_k_routes_displayed():
    at = AppTest.from_file("dashboard.py", default_timeout=120).run()
    md = [m.value for m in at.markdown]
    assert any("Alternative routes" in m for m in md), "expected alt-routes section"
    print("PASS: alternative routes section rendered")


if __name__ == "__main__":
    test_app_loads_with_title_and_route()
    test_rush_hour_changes_congestion()
    test_incident_reroutes()
    test_same_source_destination_warns()
    test_live_mode_tick()
    test_k_routes_displayed()
    print("\nALL DASHBOARD TESTS PASSED")
