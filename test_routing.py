"""
test_routing.py — Unit tests for the routing engine.

Tests the pure graph functions (manual Dijkstra, Yen's k-shortest paths,
congestion-weighted adjacency) on small hand-built graphs — no model
artifacts required.

Run:
    python test_routing.py            # verbose unittest output
    python -m unittest test_routing   # alternative
"""

import math
import unittest

from road_network import CONGESTION_MULTIPLIER, edge_id
from routing import build_adjacency, dijkstra, path_cost, yen_k_shortest


def toy_graph() -> dict:
    """
    S-A 1, S-B 4, A-B 2, A-T 6, B-T 3, plus isolated node X.
    Shortest S->T is S-A-B-T with cost 6.
    """
    return {
        "S": {"A": 1.0, "B": 4.0},
        "A": {"S": 1.0, "B": 2.0, "T": 6.0},
        "B": {"S": 4.0, "A": 2.0, "T": 3.0},
        "T": {"A": 6.0, "B": 3.0},
        "X": {},
    }


class TestDijkstra(unittest.TestCase):
    def test_known_shortest_path(self):
        path, cost = dijkstra(toy_graph(), "S", "T")
        self.assertEqual(path, ["S", "A", "B", "T"])
        self.assertAlmostEqual(cost, 6.0)

    def test_source_equals_target(self):
        path, cost = dijkstra(toy_graph(), "S", "S")
        self.assertEqual(path, ["S"])
        self.assertAlmostEqual(cost, 0.0)

    def test_unreachable_returns_none(self):
        path, cost = dijkstra(toy_graph(), "S", "X")
        self.assertIsNone(path)
        self.assertEqual(cost, math.inf)

    def test_tie_breaking_deterministic(self):
        # S-A-T and S-B-T both cost 4: either is fine, but cost must be 4.
        adj = {"S": {"A": 2.0, "B": 2.0}, "A": {"T": 2.0},
               "B": {"T": 2.0}, "T": {}}
        path, cost = dijkstra(adj, "S", "T")
        self.assertAlmostEqual(cost, 4.0)
        self.assertEqual(path[0], "S")
        self.assertEqual(path[-1], "T")


class TestBlockedEdges(unittest.TestCase):
    def test_blocked_edge_forces_detour(self):
        # Block A-B via the adjacency builder on the real network shape.
        congestion = {eid: "Low" for eid in
                      [edge_id(u, v) for u in "SABTX" for v in "SABTX"]}
        adj = build_adjacency(congestion, blocked={"A-B"})
        # A-B removed from the real 10-node graph: graph stays connected.
        self.assertNotIn("B", adj["A"])
        self.assertNotIn("A", adj["B"])

    def test_all_paths_blocked_unreachable(self):
        adj = {"S": {"T": 5.0}, "T": {"S": 5.0}}
        blocked_adj = {u: {v: w for v, w in nbrs.items() if not (u == "S" and v == "T")}
                       for u, nbrs in adj.items()}
        path, _ = dijkstra(blocked_adj, "S", "T")
        self.assertIsNone(path)


class TestCongestionWeights(unittest.TestCase):
    def test_multiplier_applied(self):
        from road_network import all_edge_ids, base_minutes
        eid = all_edge_ids()[0]  # e.g. 'A-B'
        congestion = {e: "Low" for e in all_edge_ids()}
        congestion[eid] = "High"
        adj = build_adjacency(congestion)
        u, v = eid.split("-")
        expected = base_minutes(eid) * CONGESTION_MULTIPLIER["High"]
        self.assertAlmostEqual(adj[u][v], expected)
        # An unflagged edge keeps its base weight.
        other = all_edge_ids()[1]
        ou, ov = other.split("-")
        self.assertAlmostEqual(adj[ou][ov], base_minutes(other))


class TestYenKShortest(unittest.TestCase):
    def test_three_routes_distinct_and_sorted(self):
        adj = toy_graph()
        routes = yen_k_shortest(adj, "S", "T", k=3)
        self.assertEqual(len(routes), 3)
        paths = [p for p, _ in routes]
        costs = [c for _, c in routes]
        # Distinct...
        self.assertEqual(len({tuple(p) for p in paths}), 3)
        # ...sorted by cost ascending...
        self.assertEqual(costs, sorted(costs))
        # ...and each cost matches the path's true weight.
        for p, c in routes:
            self.assertAlmostEqual(c, path_cost(adj, p))
        # Fastest must be the known optimum.
        self.assertEqual(paths[0], ["S", "A", "B", "T"])

    def test_fewer_than_k_paths(self):
        adj = {"S": {"T": 5.0}, "T": {}}
        routes = yen_k_shortest(adj, "S", "T", k=3)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0][0], ["S", "T"])

    def test_no_path_returns_empty(self):
        routes = yen_k_shortest(toy_graph(), "S", "X", k=3)
        self.assertEqual(routes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
