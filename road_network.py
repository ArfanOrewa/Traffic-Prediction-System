"""
road_network.py — Road network definition (SINGLE SOURCE OF TRUTH).

The 10-intersection road graph used by the dataset generator, the router, and
the dashboard. Every module imports the graph from here so the map, the data,
and the routing logic can never disagree.

Nodes are intersections; edges are bidirectional road segments with a base
travel time in minutes (free-flow, no congestion).
"""

# Node id -> human-readable name
NODES = {
    "A": "North Gate",
    "B": "Market Square",
    "C": "Railway Station",
    "D": "University",
    "E": "Hospital Junction",
    "F": "Airport Road",
    "G": "Tech Park",
    "H": "Old Town",
    "I": "River Bridge",
    "J": "South End",
}

# (node_u, node_v, base travel time in minutes) — bidirectional
EDGES = [
    ("A", "B", 8),
    ("A", "D", 10),
    ("B", "C", 6),
    ("B", "E", 7),
    ("C", "F", 12),
    ("D", "E", 5),
    ("D", "G", 9),
    ("E", "F", 8),
    ("E", "H", 6),
    ("F", "I", 10),
    ("G", "H", 7),
    ("G", "J", 11),
    ("H", "I", 5),
    ("H", "J", 9),
    ("I", "J", 6),
]

# Approximate map positions (x, y) for drawing the network.
# North is up (+y); laid out to roughly match the node names.
POS = {
    "A": (1.0, 4.0),   # North Gate
    "B": (2.2, 3.4),   # Market Square
    "C": (3.6, 3.8),   # Railway Station
    "D": (1.2, 2.6),   # University
    "E": (2.4, 2.4),   # Hospital Junction
    "F": (3.8, 2.6),   # Airport Road
    "G": (1.6, 1.4),   # Tech Park
    "H": (2.8, 1.4),   # Old Town
    "I": (3.8, 1.0),   # River Bridge
    "J": (2.2, 0.2),   # South End
}

# Congestion multipliers applied to base travel time when routing.
CONGESTION_MULTIPLIER = {
    "Low": 1.0,
    "Medium": 1.5,
    "High": 2.2,
}

CONGESTION_LEVELS = ["Low", "Medium", "High"]


def edge_id(u: str, v: str) -> str:
    """Canonical edge id, e.g. edge_id('B', 'A') -> 'A-B'."""
    return "-".join(sorted((u, v)))


def edge_nodes(eid: str) -> tuple:
    """Inverse of edge_id: 'A-B' -> ('A', 'B')."""
    u, v = eid.split("-")
    return u, v


def all_edge_ids() -> list:
    """Every edge id in the network, in a stable order."""
    return [edge_id(u, v) for u, v, _ in EDGES]


def base_minutes(eid: str) -> float:
    """Free-flow travel time for an edge id."""
    u, v = edge_nodes(eid)
    for a, b, mins in EDGES:
        if {a, b} == {u, v}:
            return float(mins)
    raise KeyError(f"Unknown edge: {eid}")


def node_label(node: str) -> str:
    """Display label for a node, e.g. 'A – North Gate'."""
    return f"{node} – {NODES[node]}"


def adjacency() -> dict:
    """Unweighted adjacency: node -> {neighbor: base_minutes}."""
    adj = {n: {} for n in NODES}
    for u, v, mins in EDGES:
        adj[u][v] = float(mins)
        adj[v][u] = float(mins)
    return adj
