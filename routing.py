"""
routing.py — Congestion-aware routing.

Design choice: the road graph is a hand-rolled adjacency dict and Dijkstra's
algorithm is implemented manually with a heapq priority queue (this is a
portfolio piece — the manual implementation is the point). networkx is used
only for drawing the map, never for shortest paths.

Public API:
    find_optimal_route(source, destination, day_of_week, hour, weather,
                       is_holiday=0, blocked=())
        -> dict(path, path_labels, total_minutes, edges) | None

    find_k_routes(source, destination, day_of_week, hour, weather,
                  k=3, is_holiday=0, blocked=())
        -> [route_dict, ...] sorted by total_minutes (may be fewer than k)

Edge weight = base_minutes * CONGESTION_MULTIPLIER[predicted congestion].
Blocked (incident) edges are excluded from the graph entirely.
"""

import heapq
import math

from road_network import (CONGESTION_MULTIPLIER, NODES, adjacency, base_minutes,
                          edge_id)

from predict import predict_congestion

INF = math.inf


# ------------------------------------------------------------- core graph ops
def build_adjacency(congestion_by_edge: dict, blocked=frozenset()) -> dict:
    """
    Weighted adjacency from per-edge congestion labels.
    blocked: iterable of edge ids excluded from the graph (incidents).
    """
    blocked = {edge_id(*e.split("-")) if "-" in e else e for e in blocked}
    adj = {n: {} for n in NODES}
    base = adjacency()
    for u, neighbors in base.items():
        for v in neighbors:
            eid = edge_id(u, v)
            if eid in blocked:
                continue
            label = congestion_by_edge.get(eid, "Low")
            adj[u][v] = base[u][v] * CONGESTION_MULTIPLIER[label]
    return adj


def dijkstra(adj: dict, source: str, target: str):
    """
    Manual Dijkstra with a binary heap.
    Returns (path, cost); (None, inf) when target is unreachable.
    """
    dist = {n: INF for n in adj}
    prev = {}
    dist[source] = 0.0
    heap = [(0.0, source)]
    visited = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in adj[u].items():
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v] - 1e-9:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[target] == INF:
        return None, INF
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[target]


def path_cost(adj: dict, path: list) -> float:
    """Total weight of a node path in the given adjacency."""
    return sum(adj[path[i]][path[i + 1]] for i in range(len(path) - 1))


def yen_k_shortest(adj: dict, source: str, target: str, k: int):
    """
    Yen's k-shortest loopless paths, built on the manual dijkstra above.
    Returns [(path, cost), ...] sorted by cost; may be shorter than k.
    """
    first_path, first_cost = dijkstra(adj, source, target)
    if first_path is None:
        return []
    A = [(first_path, first_cost)]          # confirmed shortest paths
    B = []                                  # candidate heap: (cost, path)
    seen = {tuple(first_path)}

    for _ in range(1, k):
        last_path, _ = A[-1]
        for i in range(len(last_path) - 1):
            spur = last_path[i]
            root = last_path[:i + 1]

            # Copy adjacency and remove edges/nodes that would repeat A paths.
            import copy
            mod = copy.deepcopy(adj)
            for p, _ in A:
                if len(p) > i and p[:i + 1] == root:
                    u, v = p[i], p[i + 1]
                    mod[u].pop(v, None)
            for node in root[:-1]:
                mod.pop(node, None)
                for nbrs in mod.values():
                    nbrs.pop(node, None)

            spur_path, _ = dijkstra(mod, spur, target)
            if spur_path is None:
                continue
            total_path = root[:-1] + spur_path
            key = tuple(total_path)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(B, (path_cost(adj, total_path), total_path))

        if not B:
            break
        cost, path = heapq.heappop(B)
        A.append((path, cost))

    return A


# ------------------------------------------------------------ query wrappers
def edge_congestion_for_query(day_of_week: str, hour: int,
                              weather: str, is_holiday: int = 0) -> dict:
    """Predicted congestion label for every edge under the query conditions."""
    from road_network import all_edge_ids
    return {
        eid: predict_congestion(eid, hour, day_of_week, weather, is_holiday)
        for eid in all_edge_ids()
    }


def _describe_route(path, adj, congestion_by_edge) -> dict:
    from road_network import node_label
    edges = []
    for i in range(len(path) - 1):
        eid = edge_id(path[i], path[i + 1])
        edges.append({
            "edge_id": eid,
            "from": path[i],
            "to": path[i + 1],
            "congestion": congestion_by_edge.get(eid, "Low"),
            "minutes": round(adj[path[i]][path[i + 1]], 1),
        })
    return {
        "path": path,
        "path_labels": [node_label(n) for n in path],
        "path_str": " → ".join(path),
        "total_minutes": round(sum(e["minutes"] for e in edges), 1),
        "edges": edges,
    }


def find_optimal_route(source: str, destination: str, day_of_week: str,
                       hour: int, weather: str, is_holiday: int = 0,
                       blocked=()) -> dict | None:
    """
    Fastest route under predicted congestion. Returns a route dict, or None
    when no path exists (e.g. incidents cut every connection).
    """
    congestion = edge_congestion_for_query(day_of_week, hour, weather, is_holiday)
    adj = build_adjacency(congestion, blocked)
    if source not in adj or destination not in adj:
        return None
    path, _ = dijkstra(adj, source, destination)
    if path is None:
        return None
    return _describe_route(path, adj, congestion)


def find_k_routes(source: str, destination: str, day_of_week: str, hour: int,
                  weather: str, k: int = 3, is_holiday: int = 0,
                  blocked=()) -> list:
    """Top-k distinct routes, fastest first (Google-Maps-style alternates)."""
    congestion = edge_congestion_for_query(day_of_week, hour, weather, is_holiday)
    adj = build_adjacency(congestion, blocked)
    if source not in adj or destination not in adj:
        return []
    routes = []
    for path, _ in yen_k_shortest(adj, source, destination, k):
        routes.append(_describe_route(path, adj, congestion))
    return routes
