"""
dashboard.py — Intelligent Traffic Management (Streamlit).

- Sidebar: source / destination / day / hour / weather / live-mode toggle /
  incident (road closure) simulator / 24-hour time-lapse slider.
- Main: congestion overview, optimal route card, top-3 alternative routes,
  color-coded network map, model info.
- The map-drawing helper `draw_network_map` is importable so it can be
  smoke-tested headlessly (no Streamlit runtime needed).

Run:
    streamlit run dashboard.py
"""

import time
from pathlib import Path

import matplotlib
import pandas as pd
import streamlit as st

from road_network import (CONGESTION_LEVELS, NODES, POS, all_edge_ids,
                          edge_id, edge_nodes, node_label)
from routing import edge_congestion_for_query, find_k_routes, find_optimal_route

BASE_DIR = Path(__file__).resolve().parent

EDGE_COLORS = {"Low": "#2ca02c", "Medium": "#e6a817", "High": "#d62728"}
EDGE_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]


# ------------------------------------------------------- map drawing (pure)
def draw_network_map(edge_congestion: dict, highlight_paths: list = None,
                     blocked=()) -> "matplotlib.figure.Figure":
    """
    Draw the 10-node road network with edges colored by congestion.
    highlight_paths: [optimal_path, alt1, alt2, ...] node lists; the first is
    drawn bold blue, the rest grey. Blocked edges are dashed grey.
    Returns the matplotlib Figure (headless-safe with the Agg backend).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    highlight_paths = highlight_paths or []
    blocked = set(blocked)

    G = nx.Graph()
    G.add_nodes_from(NODES)
    for eid in all_edge_ids():
        u, v = edge_nodes(eid)
        G.add_edge(u, v)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Road network — edge color = predicted congestion", fontsize=13)

    # Base edges colored by congestion.
    for eid in all_edge_ids():
        u, v = edge_nodes(eid)
        x = [POS[u][0], POS[v][0]]
        y = [POS[u][1], POS[v][1]]
        if eid in blocked:
            ax.plot(x, y, color="#9e9e9e", linewidth=2.5,
                    linestyle=(0, (4, 3)), zorder=1)
        else:
            ax.plot(x, y, color=EDGE_COLORS[edge_congestion.get(eid, "Low")],
                    linewidth=3.5, solid_capstyle="round", zorder=1)

    # Alternative routes (grey), then optimal route (bold blue) on top.
    for i, path in enumerate(highlight_paths):
        xs = [POS[n][0] for n in path]
        ys = [POS[n][1] for n in path]
        if i == 0:
            ax.plot(xs, ys, color="#1f77b4", linewidth=6,
                    solid_capstyle="round", zorder=3, label="Optimal route")
        else:
            ax.plot(xs, ys, color="#7f7f7f", linewidth=3,
                    linestyle="--", zorder=2, label=f"Alternate {i}")

    # Nodes.
    nx.draw_networkx_nodes(G, POS, node_size=900, node_color="white",
                           edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, POS,
                            labels={n: n for n in NODES},
                            font_size=11, font_weight="bold", ax=ax)
    for n, (x, y) in POS.items():
        ax.text(x, y - 0.22, NODES[n], ha="center", fontsize=8,
                color="#333333", zorder=5)

    # Legend.
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color=EDGE_COLORS["Low"], lw=4, label="Low congestion"),
        Line2D([0], [0], color=EDGE_COLORS["Medium"], lw=4, label="Medium congestion"),
        Line2D([0], [0], color=EDGE_COLORS["High"], lw=4, label="High congestion"),
        Line2D([0], [0], color="#1f77b4", lw=6, label="Optimal route"),
        Line2D([0], [0], color="#9e9e9e", lw=2.5, linestyle=(0, (4, 3)),
               label="Road closed"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.5, 4.6)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ app
def _route_inputs():
    st.sidebar.header("🧭 Route planner")
    node_ids = list(NODES)
    src = st.sidebar.selectbox("Source", node_ids,
                               format_func=node_label, index=0)
    dst = st.sidebar.selectbox("Destination", node_ids,
                               format_func=node_label, index=9)
    day = st.sidebar.selectbox("Day of week", DAYS, index=0)
    hour = st.sidebar.slider("Hour of day (time-lapse)", 0, 23, 9,
                             help="Drag to watch congestion change through the day")
    weather = st.sidebar.radio("Weather", ["Clear", "Rain"], horizontal=True)
    return src, dst, day, hour, weather


def _extensions():
    st.sidebar.header("⚙️ Extensions")
    live = st.sidebar.toggle("🔴 Live mode (5s refresh)",
                             help="Simulate a real-time traffic feed")
    animate = st.sidebar.button("▶ Animate 24h")
    edge_options = all_edge_ids()
    blocked = st.sidebar.multiselect(
        "🚧 Incident simulator — close roads",
        options=edge_options,
        format_func=lambda e: f"{e} ({NODES[edge_nodes(e)[0]]} ↔ {NODES[edge_nodes(e)[1]]})",
        help="Closed roads are excluded; the router re-routes around them.")
    return live, animate, blocked


def main() -> None:
    st.set_page_config(page_title="Intelligent Traffic Management",
                       page_icon="🚦", layout="wide")
    st.title("🚦 Intelligent Traffic Management System")
    st.caption("Random Forest congestion prediction + Dijkstra optimal routing "
               "· synthetic data · 10-intersection city network")

    src, dst, day, hour, weather = _route_inputs()
    live, animate, blocked = _extensions()

    # --- 24h animation: step the hour slider via reruns ---------------------
    if animate:
        st.session_state["anim_hour"] = (hour + 1) % 24
        time.sleep(0.6)
        st.rerun()
    if "anim_hour" in st.session_state:
        hour = st.session_state.pop("anim_hour")

    # --- live mode -----------------------------------------------------------
    congestion = None
    if live:
        from simulate_live import init_state, tick
        if "live_tick" not in st.session_state:
            st.session_state["live_tick"] = 0
            st.session_state["live_state"] = init_state(hour, day)
        st.session_state["live_state"], congestion = tick(
            st.session_state["live_state"], st.session_state["live_tick"],
            hour, day, weather)
        st.session_state["live_tick"] += 1
        st.sidebar.info(f"Live tick #{st.session_state['live_tick']}")

    if src == dst:
        st.warning("Source and destination are the same — pick two different intersections.")
        return

    # --- routing ---------------------------------------------------------------
    if congestion is None:
        congestion = edge_congestion_for_query(day, hour, weather)
    route = find_optimal_route(src, dst, day, hour, weather, blocked=blocked)
    k_routes = find_k_routes(src, dst, day, hour, weather, k=3, blocked=blocked)

    if blocked:
        st.warning(f"🚧 {len(blocked)} road(s) closed: "
                   + ", ".join(blocked) + " — routes avoid them.")

    # --- route cards -------------------------------------------------------------
    st.subheader(f"Route: {node_label(src)} → {node_label(dst)}")
    if route is None:
        st.error("⛔ No route available — incidents have cut every connection. "
                 "Reopen a road to restore routing.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Optimal route", route["path_str"])
        c2.metric("Estimated travel time", f"{route['total_minutes']:.1f} min")
        worst = max((e["congestion"] for e in route["edges"]),
                    key=CONGESTION_LEVELS.index)
        c3.metric("Worst segment", f"{EDGE_EMOJI[worst]} {worst}")

        if len(k_routes) > 1:
            st.markdown("**Alternative routes** (Google-Maps style)")
            for i, r in enumerate(k_routes[1:], start=2):
                st.write(f"Route {i} (grey): `{r['path_str']}` — "
                         f"{r['total_minutes']:.1f} min")

    # --- congestion overview -------------------------------------------------------
    st.subheader("Predicted congestion per road segment")
    seg_rows = []
    for eid in all_edge_ids():
        u, v = edge_nodes(eid)
        seg_rows.append({
            "Road": eid,
            "From": NODES[u],
            "To": NODES[v],
            "Congestion": f"{EDGE_EMOJI[congestion[eid]]} {congestion[eid]}",
            "Status": "🚧 Closed" if eid in set(blocked) else "Open",
        })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)

    # --- network map ---------------------------------------------------------------
    st.subheader("Network map")
    paths = [r["path"] for r in k_routes] if k_routes else []
    fig = draw_network_map(congestion, highlight_paths=paths, blocked=blocked)
    st.pyplot(fig, use_container_width=True)

    # --- model info ------------------------------------------------------------------
    with st.expander("🧠 Model info"):
        comp_path = BASE_DIR / "models" / "model_comparison.json"
        if comp_path.exists():
            import json
            comp = json.load(open(comp_path))
            st.dataframe(pd.DataFrame(comp), use_container_width=True,
                         hide_index=True)
            st.caption("Random Forest is the deployed model (predict.py loads "
                       "models/traffic_rf_model.pkl). Metrics on the 20% hold-out set.")
        else:
            st.info("Train the model first: `python model.py`")

    # --- live-mode auto refresh ---------------------------------------------------------
    if live:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
